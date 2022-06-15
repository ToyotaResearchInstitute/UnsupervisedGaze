"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

from collections import OrderedDict
import logging
from operator import is_
import os
import sys
import pickle
import shutil
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from core.config_default import DefaultConfig
from core.training import salvage_memory, step_modulo, learning_rate_schedule
sys.path.append("../utils")
from utils.torch_utils import container_send_to_device, container_detach, container_to_numpy
from utils.angles import angular_loss

config = DefaultConfig()

# Setup logger
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_loss(preds, labels, valids, tags, is_eval):
    # Perform final appearance mean, if used
    if config.gaze_estimation_pred_mean:
        sample_combs = torch.stack((tags['head'], tags['gaze']), dim=-1)
        unique_combs = torch.unique(sample_combs, dim=0)
        nun_unique = unique_combs.shape[0]
        for i in range(nun_unique):
            u_idx = torch.all(sample_combs == unique_combs[i], dim=-1)
            u_valids = valids[:,u_idx]
            u_preds = preds[:,u_idx]
            V = u_preds.shape[1]
            valid_total = torch.unsqueeze(torch.sum(u_valids, dim=1, keepdim=True), dim=-1)
            valid_total[valid_total==0.] = 1.
            u_preds = torch.sum(u_preds * torch.unsqueeze(u_valids, dim=-1), dim=1, keepdim=True) / valid_total
            preds[:,u_idx] = u_preds.repeat((1,V,1))

    # Only predict the sample-specific values when testing
    if is_eval:
        sample_idx = torch.logical_and(tags['head'] == 0, tags['gaze'] == 0)
        preds = preds[:,sample_idx]
        labels = labels[:,sample_idx]
        valids = valids[:,sample_idx]

    # Calculate loss, depending on eval target
    if config.eval_binary:
        preds = preds.squeeze(-1)
        if is_eval:
            loss = -(((preds >= 0.) == labels).float())
        else:
            loss = F.binary_cross_entropy_with_logits(preds, labels.float(), reduction='none') * valids
    else:
        loss = angular_loss(preds, labels) * valids
    return torch.sum(loss) / torch.sum(valids)


def test_model_on_all_gaze_estimation(model, fold_data, fold_name, current_step, tensorboard=None):
    """Get training batches of data from all training data sources."""
    model.eval()
    salvage_memory()
    metrics = {}
    metrics['loss'] = 0
    with torch.no_grad():
        num_entries = len(fold_data['dataset'])
        for input_data in fold_data['dataloader']:
            batch_size = len(input_data['sample_number'])
            
            # Move tensors to GPU
            input_data = container_send_to_device(input_data, device)
            input_data['tags'] = fold_data['dataset'].original_full_dataset.tags

            # Forward pass
            preds = model(input_data)
            labels = input_data['label']
            loss = calculate_loss(preds, labels, input_data['valids'], input_data['tags'], is_eval=True)
            outputs = {}
            outputs['output_direction'] = preds
            outputs['loss'] = loss

            # Calculate loss
            outputs = container_detach(outputs)
            outputs = container_to_numpy(outputs)
            metrics['loss'] += outputs['loss'] * batch_size / num_entries

        # Log to command line
        log = ('{:s} at Step {:d}: '.format(fold_name, current_step)  + 
                ', '.join(['%s: %.4g' % (k, metrics[k]) for k in sorted(metrics.keys())]))
        logger.info(log)

        # Write to tensorboard
        if tensorboard:
            tensorboard.update_current_step(current_step)
            for k, v in metrics.items():
                tensorboard.add_scalar(fold_name + '_eval/%s' % (k), v)

    salvage_memory()
    return metrics


def main_loop(model, optimizers, data, tensorboard=None):
    # Skip this entirely if requested
    if config.skip_training:
        return

    assert tensorboard is not None  # We assume this exists in LR schedule logging
    initial_step = model.last_step  # Allow resuming
    max_dataset_len = len(data['train']['dataset'])
    num_steps_per_epoch = int(max_dataset_len / config.gaze_estimation_batch_size)
    if num_steps_per_epoch  == 0:
        num_steps_per_epoch = 1
    num_training_steps = int(config.num_epochs_gaze_estimation * num_steps_per_epoch)
    if 'mean' in config.eval_features:
        num_training_steps = 0

    model.train()
    current_step = 0

    logger.info('Training set size: %d' % max_dataset_len)
    logger.info('Validation set size: %d' % len(data['val']['dataset']))
    logger.info('Testing set size: %d' % len(data['test']['dataset']))
    logger.info('Eval features: %s' % ', '.join(config.eval_features))
    logger.info('Eval target: %s' % config.eval_target)

    eval_metrics = {'min_val': None, 'min_test': None, 'final_test': None}
    for current_step in range(initial_step, num_training_steps):
        
        current_epoch = (current_step * config.gaze_estimation_batch_size) / max_dataset_len  # fractional value
        tensorboard.update_current_step(current_step + 1)
        input_data = next(iter(data['train']['dataloader']))
        input_data['tags'] = data['train']['dataset'].original_full_dataset.tags

        input_data = container_send_to_device(input_data, device)

        # Set correct states before training iteration
        model.train()
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Forward pass
        loss_terms = []
        preds = model(input_data)
        labels = input_data['label']
        loss = calculate_loss(preds, labels, input_data['valids'], input_data['tags'], is_eval=False)
        outputs = {}
        outputs['output_direction'] = preds
        outputs['loss'] = loss
        loss_terms.append(loss)

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

        # Apply gradients
        for optimizer in valid_optimizers:
            optimizer.step()

        # Print outputs
        outputs = container_detach(outputs)
        outputs = container_to_numpy(outputs)
        metrics = {}
        metrics['loss'] = outputs['loss']

        log = ('Step %d, Epoch %.2f> ' % (current_step, current_epoch)
                + ', '.join(['%s: %.4g' % (k, metrics[k]) for k in sorted(metrics.keys())]))
        logger.info(log)

        # Log to Tensorboard
        for key, metric in metrics.items():
            if key.startswith('loss_'):
                key = key[len('loss_'):]
                tensorboard.add_scalar('train_losses/%s' % key, metric)
            elif key.startswith('metric_'):
                key = key[len('metric_'):]
                tensorboard.add_scalar('train_metrics/%s' % key, metric)
            else:
                tensorboard.add_scalar('train/%s' % key, metric)

        tensorboard.add_scalar('lr/epoch', current_epoch)
        tensorboard.add_scalar('lr/lr', optimizers[0].param_groups[0]['lr'])

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
        if step_modulo(current_step, config.test_every_n_steps_gaze_estimation):
            is_best = {}
            metrics = {}
            for fold_name in ['val', 'test']:
                metrics[fold_name] = test_model_on_all_gaze_estimation(model, data[fold_name], fold_name, current_step + 1, tensorboard)
                is_best[fold_name] = eval_metrics['min_'+fold_name] is None or metrics[fold_name]['loss'] <= eval_metrics['min_'+fold_name]
                if is_best[fold_name]:
                    eval_metrics['min_'+fold_name] = metrics[fold_name]['loss']
        
            # Update final test if best val
            if is_best['val']:
                eval_metrics['final_test'] = metrics['test']['loss']

            # Log to command line
            log = ('Eval at Step {:d}: '.format(current_step + 1)  + 
                    ', '.join(['%s: %.4g' % (k, eval_metrics[k]) for k in sorted(eval_metrics.keys())]))
            logger.info(log)

            # Write to tensorboard
            if tensorboard:
                for k, v in eval_metrics.items():
                    tensorboard.add_scalar(k, v)

        # Free memory
        salvage_memory()

        # Remember what the last step/epoch were
        model.last_epoch = current_epoch
        model.last_step = current_step

        # Update learning rate
        # NOTE: should be last
        tensorboard.update_current_step(current_step + 2)

    # We're out of the training loop now, make a checkpoint
    current_step += 1
    model.checkpoint_manager.save_at_step(current_step + 1)

    # Final test
    is_best = {}
    metrics = {}
    for fold_name in ['val', 'test']:
        metrics[fold_name] = test_model_on_all_gaze_estimation(model, data[fold_name], fold_name, current_step + 1, tensorboard)
        is_best[fold_name] = eval_metrics['min_'+fold_name] is None or metrics[fold_name]['loss'] <= eval_metrics['min_'+fold_name]
        if is_best[fold_name]:
            eval_metrics['min_'+fold_name] = metrics[fold_name]['loss']

    # Update final test if best val
    if is_best['val']:
        eval_metrics['final_test'] = metrics['test']['loss']

    # Log to command line
    log = ('Eval at Step {:d}: '.format(current_step + 1)  + 
            ', '.join(['%s: %.4g' % (k, eval_metrics[k]) for k in sorted(eval_metrics.keys())]))
    logger.info(log)

    # Write to tensorboard
    if tensorboard:
        for k, v in eval_metrics.items():
            tensorboard.add_scalar(k, v)

    # Clear memory where possible
    salvage_memory()
