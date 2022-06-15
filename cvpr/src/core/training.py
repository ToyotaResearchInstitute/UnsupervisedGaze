"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import argparse
from collections import OrderedDict
import functools
import gc
import hashlib
import logging
import os
import sys
import time
import shutil
import random

import coloredlogs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import DefaultConfig, CheckpointManager, Tensorboard
sys.path.append("../utils")
from utils.torch_utils import container_send_to_device, container_detach, container_to_numpy
from utils.data_types import TypedOrderedDict

config = DefaultConfig()

# Setup logger
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _convert_cli_arg_type(key, value):
    attr = getattr(config, key)
    config_type = type(attr)
    if config_type == bool:
        if value.lower() in ('true', 'yes', 'y') or value == '1':
            return True
        elif value.lower() in ('false', 'no', 'n') or value == '0':
            return False
        else:
            raise ValueError('Invalid input for bool config "%s": %s' % (key, value))
    elif config_type == TypedOrderedDict:
        new_dict = TypedOrderedDict(attr.key_type, attr.value_type)
        new_dict.insert_list_of_strings(value)
        return new_dict
    else:
        return config_type(value)


def script_init_common():
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='Desired logging level.', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('config_json', type=str, nargs='*',
                        help=('Path to config in JSON format. '
                              'Multiple configs will be parsed in the specified order.'))
    for key in dir(config):
        if key.startswith('_DefaultConfig') or key.startswith('__'):
            continue
        if key in vars(DefaultConfig) and isinstance(vars(DefaultConfig)[key], property):
            continue
        value = getattr(config, key)
        value_type = type(value)
        arg_type = value_type
        nargs = '?'
        if value_type == list:
            arg_type = type(value[0])
            nargs = '+'
        elif value_type == TypedOrderedDict:
            arg_type = str
            nargs = '+'
        if value_type == bool:
            # Handle booleans separately, otherwise arbitrary values become `True`
            arg_type = str
        if callable(value):
            continue
        parser.add_argument('--' + key.replace('_', '-'), type=arg_type, nargs=nargs, metavar=value,
                            help='Expected type is `%s`.' % value_type.__name__)
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set logger format and verbosity level
    coloredlogs.install(
        datefmt='%d/%m %H:%M:%S',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Parse configs in order specified by user
    for json_path in args.config_json:
        config.import_json(json_path)

    # Apply configs passed through command line
    config.import_dict({
        key.replace('-', '_'): _convert_cli_arg_type(key, value)
        for key, value in vars(args).items()
        if value is not None and hasattr(config, key)
    })

    # Improve reproducibility
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    if config.fully_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    return config, device


def setup_common(model, optimizers):
    identifier = ""
    if config.group_name != "":
        identifier += config.group_name + "/"
    identifier += config.exp_name
    output_dir = './outputs/checkpoints/' + identifier

    # Initialize tensorboard and wandb
    if os.path.isdir(output_dir):
        if not config.overwrite:
            raise Exception("ERROR: Output directory already exists. Set --overwrite 1 if you want to overwrite the previous results.")
        else:
            shutil.rmtree(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tensorboard = Tensorboard(output_dir)

    # Write source code to output dir
    # NOTE: do not over-write if resuming from an output directory
    if len(config.resume_from) == 0:
        config.write_file_contents(output_dir)

    # Log messages to file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(output_dir + '/messages.log')
    file_handler.setFormatter(root_logger.handlers[0].formatter)
    for handler in root_logger.handlers[1:]:  # all except stdout
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    # Print model details
    num_params = sum([
        np.prod(p.size())
        for p in filter(lambda p: p.requires_grad, model.parameters())
    ])
    logger.info('\nThere are %d trainable parameters.\n' % num_params)

    # Cache base and target learning rate for each optimizer
    for optimizer in optimizers:
        optimizer.target_lr = optimizer.param_groups[0]['lr']
        optimizer.base_lr = optimizer.target_lr / config.batch_size

    # Sneak in some extra information into the model class instance
    model.identifier = identifier
    model.output_dir = output_dir
    model.checkpoint_manager = CheckpointManager(model, optimizers)
    model.last_epoch = 0.0
    model.last_step = 0

    # Load pre-trained model weights if available
    if len(config.resume_from) > 0:
        model.last_step = model.checkpoint_manager.load_last_checkpoint()

    return model, optimizers, tensorboard


def salvage_memory():
    """Try to free whatever memory that can be freed."""
    torch.cuda.empty_cache()
    gc.collect()


def get_training_batches(train_data_dicts):
    """Get training batches of data from all training data sources."""
    out = {}
    # for tag, data_dict in train_data_dicts.items():
    data_dict = train_data_dicts
    if 'data_iterator' not in data_dict:
        data_dict['data_iterator'] = iter(data_dict['dataloader'])
    # Try to get data
    while True:
        try:
            out = next(data_dict['data_iterator'])
            break
        except StopIteration:
            del data_dict['data_iterator']
            salvage_memory()
            data_dict['data_iterator'] = iter(data_dict['dataloader'])
    return out


def learning_rate_schedule(optimizer, epoch_len, tensorboard_log_func, step):
    num_warmup_steps = int(epoch_len * config.num_warmup_epochs)
    selected_lr = None
    if step < num_warmup_steps:
        b = optimizer.base_lr
        a = (optimizer.target_lr - b) / float(num_warmup_steps)
        selected_lr = a * step + b
    else:
        # Decay learning rate with step function and exponential decrease?
        new_step = step - num_warmup_steps
        epoch = new_step / float(epoch_len)
        current_interval = int(epoch / config.lr_decay_epoch_interval)
        if config.lr_decay_strategy == 'exponential':
            # Step function decay
            selected_lr = optimizer.target_lr * np.power(config.lr_decay_factor, current_interval)
        elif config.lr_decay_strategy == 'cyclic':
            # Note, we start from the up state (due to previous warmup stage)
            # so each period consists of down-up (not up-down)
            peak_a = optimizer.target_lr * np.power(config.lr_decay_factor, current_interval)
            peak_b = peak_a * config.lr_decay_factor
            half_interval = 0.5 * config.lr_decay_epoch_interval
            current_interval_start = current_interval * config.lr_decay_epoch_interval
            current_interval_half = current_interval_start + half_interval
            if epoch < current_interval_half:
                # negative slope (down from peak_a)
                slope = -(peak_a - optimizer.base_lr) / half_interval
            else:
                # positive slope (up to peak_b)
                slope = (peak_b - optimizer.base_lr) / half_interval
            selected_lr = slope * (epoch - current_interval_half) + optimizer.base_lr
        else:
            selected_lr = optimizer.target_lr

    # Log to Tensorboard and return
    if step_modulo(step, config.tensorboard_learning_rate_every_n_steps):
        tensorboard_log_func(selected_lr)
    return selected_lr


def step_modulo(current, interval_size):
    if interval_size == 0:
        return False
    return current % interval_size == 0 # (interval_size - 1)


def test_model_on_all(model, test_data_dicts, current_step, tensorboard=None):
    """Get training batches of data from all training data sources."""
    model.eval()
    salvage_memory()
    metrics = OrderedDict()

    with torch.no_grad():
        num_entries = len(test_data_dicts['dataset'])
        for i, input_data in enumerate(test_data_dicts['dataloader']):
            batch_size = len(input_data['sample_number'])
            input_data['tags'] = test_data_dicts['dataset'].original_full_dataset.tags

            # Move tensors to GPU
            input_data = container_send_to_device(input_data, device)

            # Get model outputs
            outputs = model.train_batch(input_data, temperature=0.)
            outputs = container_detach(outputs)

            # Log images
            if i == 0:
                tensorboard.add_grids(outputs['frames'], 'val')

            # Calculate loss
            outputs = container_to_numpy(outputs)
            for loss_name, loss_amount in outputs['losses'].items():
                metric_name = 'loss_'+loss_name
                if metric_name not in metrics:
                    metrics[metric_name] = 0
                metrics[metric_name] = loss_amount * batch_size / num_entries

    # Log the validation losses to command line
    log = ('Validation at Step %d: ' % current_step  + ', '.join(['%s: %.4g' % (k, v) for k, v in metrics.items()]))
    logger.info(log)

    # Write to tensorboard
    if tensorboard:
        tensorboard.update_current_step(current_step)
        for k, v in metrics.items():
            tensorboard.add_scalar('%s/%s' % ('val-losses', k), v)   

    # Free up memory
    salvage_memory()

    return metrics


def main_loop(model, optimizers, train_val_data, tensorboard=None):
    # Skip this entirely if requested
    if config.skip_training:
        return

    assert tensorboard is not None  # We assume this exists in LR schedule logging
    initial_step = model.last_step  # Allow resuming
    train_data = train_val_data['train']
    max_dataset_len = len(train_data['dataset'])
    num_steps_per_epoch = int(max_dataset_len / config.batch_size)
    num_training_steps = int(config.num_epochs * num_steps_per_epoch)
    lr_schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            functools.partial(learning_rate_schedule, optimizer, num_steps_per_epoch,
                              functools.partial(tensorboard.add_scalar, 'lr/optim_%d' % i)),
        ) for i, optimizer in enumerate(optimizers)
    ]
    model.train()
    current_step = 0
    best_val = None
    for current_step in range(initial_step, num_training_steps):
        current_epoch = (current_step * config.batch_size) / max_dataset_len  # fractional value
        tensorboard.update_current_step(current_step + 1)
        input_data = get_training_batches(train_data)
        input_data['tags'] = train_data['dataset'].original_full_dataset.tags

        # Set correct states before training iteration
        model.train()
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Move tensors to GPU
        input_data = container_send_to_device(input_data, device)

        # Get model outputs
        outputs = model.train_batch(input_data, temperature=config.train_final_temperature*((current_step+1)/num_training_steps))

        # NOTE: If we fill in index i, it would pick the optimizer at index i.
        #       In this case, there is only one optimizer, and one variant of the full loss.
        loss_terms = []
        loss_terms.append(outputs['losses']['total'])

        # There should be as many loss terms as there are optimizers!
        assert len(loss_terms) == len(optimizers)

        # Prune out None values
        valid_loss_terms = []
        valid_optimizers = []
        for loss_term, optimizer in zip(loss_terms, optimizers):
            if loss_term is not None:
                valid_loss_terms.append(loss_term)
                valid_optimizers.append(optimizer)

        # Perform gradient calculations for each loss term
        for i, (loss, optimizer) in enumerate(zip(valid_loss_terms, valid_optimizers)):
            not_last = i < (len(optimizers) - 1)
            if not isinstance(loss, torch.Tensor):
                continue
            loss.backward(retain_graph=not_last)

        # Maybe clip gradients
        if config.do_gradient_clipping:
            if config.gradient_clip_by == 'norm':
                clip_func = nn.utils.clip_grad_norm_
            elif config.gradient_clip_by == 'value':
                clip_func = nn.utils.clip_grad_value_
            clip_amount = config.gradient_clip_amount
            clip_func(model.parameters(), clip_amount)

        # Apply gradients
        for optimizer in valid_optimizers:
            optimizer.step()

        # Output tensorboard images
        outputs = container_detach(outputs)
        if step_modulo(current_step, config.tensorboard_images_every_n_steps):
            tensorboard.add_grids(outputs['frames'], 'train')
        outputs = container_to_numpy(outputs)

        # Print outputs
        if step_modulo(current_step, config.log_every_n_steps):
            metrics = OrderedDict()
            for loss_name, loss_amount in outputs['losses'].items():
                metrics['loss_'+loss_name] = loss_amount
            for valid_name, valid_amount in outputs['valid'].items():
                metrics['valid_'+valid_name] = valid_amount

            log = ('Step %d, Epoch %.2f> ' % (current_step, current_epoch)
                   + ', '.join(['%s: %.4g' % (k, v) for k, v in metrics.items()]))
            logger.info(log)

            # Log to Tensorboard
            if step_modulo(current_step, config.tensorboard_scalars_every_n_steps):
                for key, metric in metrics.items():
                    if key.startswith('loss_'):
                        key = key[len('loss_'):]
                        tensorboard.add_scalar('train-losses/%s' % key, metric)
                    elif key.startswith('valid_'):
                        key = key[len('valid_'):]
                        tensorboard.add_scalar('train-valids/%s' % key, metric)
                    elif key.startswith('metric_'):
                        key = key[len('metric_'):]
                        tensorboard.add_scalar('train-metrics/%s' % key, metric)
                    else:
                        tensorboard.add_scalar('train-other/%s' % key, metric)
                tensorboard.add_scalar('lr/epoch', current_epoch)

            # Quit if NaNs
            there_are_NaNs = False
            for k, v in metrics.items():
                if np.any(np.isnan(v)):
                    logger.error('NaN encountered during training at value: %s' % k)
                    there_are_NaNs = True
            if there_are_NaNs:
                exit(1)

        # We're done with the previous outputs
        del input_data, outputs, loss_terms

        # Save checkpoint
        if step_modulo(current_step, config.checkpoints_save_every_n_steps):
            model.checkpoint_manager.save_at_step(current_step + 1)

        # test over subset of evaluation datasets
        if 'val' in train_val_data and step_modulo(current_step, config.test_every_n_steps):

            # Do test on subset of validation datasets
            test_metrics = test_model_on_all(model, train_val_data['val'], current_step + 1, tensorboard=tensorboard)

            # Check for best validation so far
            if best_val is None or test_metrics['loss_total'] < best_val:
                best_val = test_metrics['loss_total']
                model.checkpoint_manager.save_special('best')
                print('New best model! Total validation loss of {:.4f}'.format(best_val))

            # Free memory
            salvage_memory()

        # Remember what the last step/epoch were
        model.last_epoch = current_epoch
        model.last_step = current_step

        # Update learning rate
        # NOTE: should be last
        tensorboard.update_current_step(current_step + 2)
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

    # We're out of the training loop now, make a checkpoint
    current_step += 1
    model.checkpoint_manager.save_special('last')

    # Final eval
    if 'val' in train_val_data:
        test_metrics = test_model_on_all(model, train_val_data['val'], current_step + 1, tensorboard=tensorboard)

        # Check for best validation so far
        if best_val is None or test_metrics['loss_total'] < best_val:
            best_val = test_metrics['loss_total']
            model.checkpoint_manager.save_special('best')
            print('New best model! Total validation loss of {:.4f}'.format(best_val))

    # Clear memory where possible
    salvage_memory()
