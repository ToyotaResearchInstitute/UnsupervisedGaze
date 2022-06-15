"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import glob
import json
import os
import sys
import zipfile
import numpy as np
sys.path.append("../utils")
from utils.data_types import TypedOrderedDict

import logging
logger = logging.getLogger(__name__)


class DefaultConfig(object):

    # Tag to separate Emre and my experiments
    identifier_suffix = ''

    # Misc. notes
    note = ''

    # Dataset used
    dataset_name = 'eve'
    eve_raw_path = '/home/ubuntu/data/eve_dataset/'
    eve_preprocessed_path = '/home/ubuntu/data/eve_preprocessed/'

    # Data loading
    @property
    def raw_input_size(self):
        return 128 if self.dataset_name == 'eve' else 192
    final_input_size = 128
    use_cam_gaze = False

    # Training
    skip_training = False
    fully_reproducible = False  # enable with possible penalty of performance
    train_final_temperature = 1.0
    @property
    def train_augment_images(self):
        return False
    train_denoise_images = True

    batch_size = 64 #12*4
    num_epochs = 10.
    num_epochs_gaze_estimation = 2000

    train_data_workers = 8

    log_every_n_steps = 1  # NOTE: Every other interval has to be a multiple of this!!!
    tensorboard_scalars_every_n_steps = 1
    tensorboard_images_every_n_steps = 100
    tensorboard_learning_rate_every_n_steps = 100
    visualization_max_number = 8*8
    grid_images_per_row = 8

    # Learning rate
    base_learning_rate = 0.0001

    @property
    def learning_rate(self):
        return self.batch_size * self.base_learning_rate
        
    # Available strategies:
    #     'exponential': step function with exponential decay
    #     'cyclic':      spiky down-up-downs (with exponential decay of peaks)
    num_warmup_epochs = 0.0  # No. of epochs to warmup LR from base to target
    lr_decay_strategy = 'none'
    lr_decay_factor = 0.5
    lr_decay_epoch_interval = 0.5

    # Gradient Clipping
    do_gradient_clipping = False
    gradient_clip_by = 'norm'  # 'norm' or 'value'
    gradient_clip_amount = 5.0

    # WANDB
    wandb_project_name = 'eyesontheroad'
    exp_name = 'temp-exp-name'
    group_name = ''
    exp_tags = ['default',]
    overwrite = False
    num_repeats = 0

    # cross encoder configuration
    patches_used = ['left', 'right']
    data_views = TypedOrderedDict(str, int)
    data_views['app'] = 2
    data_views['gaze'] = 2
    data_views['head'] = 2
    data_grayscale = True
    train_view_duplicates = TypedOrderedDict(str, int)
    train_view_duplicates['app'] = 1
    train_view_duplicates['gaze'] = 1
    train_view_duplicates['head'] = 1
    feature_sizes = TypedOrderedDict(str, int)
    feature_sizes['app'] = 32
    feature_sizes['gaze'] = 12
    feature_sizes['head'] = 12
    loss_weights = TypedOrderedDict(str, float)
    loss_weights['recon_gaze'] = 1.0
    loss_weights['recon_app'] = 1.0
    loss_weights['recon_head'] = 1.0
    loss_weights['recon_ref'] = 1.0
    cross_encoder_load_pretrained = False
    cross_encoder_checkpoint_folder = ""

    # Reference feature parameters
    reference_dropout = False
    reference_comb_func = 'mean'  # first / mean / max / median / vec_median
    reference_confidence = False
    reference_confidence_softmax_beta = 1000.
    gaze_estimation_use_reference_features = False
    @property
    def gaze_estimation_pred_mean(self):
        return False

    # gaze estimation
    random_seed = 0
    gaze_feature_path = ""
    gaze_estimation_pretrained = False
    gaze_estimation_checkpoint = ''
    eval_features = ['gaze', 'head']
    eval_target = 'cam_gaze_dir'
    @property
    def eval_binary(self):
        return self.eval_target == 'eye_side' or self.eval_target == 'glasses_on'

    # Evaluation
    test_batch_size = batch_size
    test_data_workers = 8
    test_every_n_steps = 100
    test_every_n_steps_gaze_estimation = 100

    # Subsampling
    subsample_fold = TypedOrderedDict(str, int)
    @property
    def gaze_estimation_batch_size(self):
        max_batch = 2000
        if 'train' in self.subsample_fold and self.subsample_fold['train'] > 0 and self.subsample_fold['train'] < max_batch:
            return self.subsample_fold['train']
        else:
            return max_batch

    # Checkpoints management
    checkpoints_save_every_n_steps = 100
    checkpoints_keep_n = 3
    resume_from = ''

    # Below lie necessary methods for working configuration tracking

    __instance = None

    # Make this a singleton class
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__filecontents = cls.__get_config_file_contents()
            cls.__pycontents = cls.__get_python_file_contents()
            cls.__immutable = True
        return cls.__instance

    def import_json(self, json_path, strict=True):
        """Import JSON config to over-write existing config entries."""
        assert os.path.isfile(json_path)
        assert not hasattr(self.__class__, '__imported_json_path')
        logger.info('Loading ' + json_path)
        with open(json_path, 'r') as f:
            json_string = f.read()
        self.import_dict(json.loads(json_string), strict=strict)
        self.__class__.__imported_json_path = json_path
        self.__class__.__filecontents[os.path.basename(json_path)] = json_string

    def override(self, key, value):
        self.__class__.__immutable = False
        setattr(self, key, value)
        self.__class__.__immutable = True

    def import_dict(self, dictionary, strict=True):
        """Import a set of key-value pairs from a dict to over-write existing config entries."""
        self.__class__.__immutable = False
        for key, value in dictionary.items():
            if strict is True:
                if not hasattr(self, key):
                    raise ValueError('Unknown configuration key: ' + key)
                if type(getattr(self, key)) is float and type(value) is int:
                    value = float(value)
                else:
                    assert type(getattr(self, key)) is type(value)
                if not isinstance(getattr(DefaultConfig, key), property):
                    setattr(self, key, value)
            else:
                if hasattr(DefaultConfig, key):
                    if not isinstance(getattr(DefaultConfig, key), property):
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        self.__class__.__immutable = True

    def __get_config_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        for relpath in ['config_default.py']:
            path = os.path.relpath(os.path.dirname(__file__) + '/' + relpath)
            assert os.path.isfile(path)
            with open(path, 'r') as f:
                out[os.path.basename(path)] = f.read()
        return out

    def __get_python_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        base_path = os.path.relpath(os.path.dirname(__file__) + '/../')
        source_fpaths = [
            p for p in glob.glob(base_path + '/**/*.py')
            if not p.startswith('./3rdparty/')
        ]
        source_fpaths += [os.path.relpath(sys.argv[0])]
        for fpath in source_fpaths:
            assert os.path.isfile(fpath)
            with open(fpath, 'r') as f:
                out[fpath[2:]] = f.read()
        return out

    def get_all_key_values(self):
        return dict([
            (key, getattr(self, key))
            for key in dir(self)
            if not key.startswith('_DefaultConfig')
            and not key.startswith('__')
            and not callable(getattr(self, key))
        ])

    def get_full_json(self):
        return json.dumps(self.get_all_key_values(), indent=4)

    def write_file_contents(self, target_base_dir):
        """Write cached config file contents to target directory."""
        assert os.path.isdir(target_base_dir)

        # Write config file contents
        target_dir = target_base_dir + '/configs'
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        outputs = {  # Also output flattened config
            'combined.json': self.get_full_json(),
        }
        outputs.update(self.__class__.__filecontents)
        for fname, content in outputs.items():
            fpath = os.path.relpath(target_dir + '/' + fname)
            with open(fpath, 'w') as f:
                f.write(content)
                logger.info('Written %s' % fpath)

        # Copy source folder contents over
        target_path = os.path.relpath(target_base_dir + '/src.zip')
        source_path = os.path.relpath(os.path.dirname(__file__) + '/../')
        filter_ = lambda x: x.endswith('.py') or x.endswith('.json')  # noqa
        with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(source_path):
                for file_or_dir in files + dirs:
                    full_path = os.path.join(root, file_or_dir)
                    if os.path.isfile(full_path) and filter_(full_path):
                        zip_file.write(
                            os.path.join(root, file_or_dir),
                            os.path.relpath(os.path.join(root, file_or_dir),
                                            os.path.join(source_path, os.path.pardir)))
        logger.info('Written source folder to %s' % os.path.relpath(target_path))

    def __setattr__(self, name, value):
        """Initial configs should not be overwritten!"""
        if self.__class__.__immutable:
            raise AttributeError('DefaultConfig instance attributes are immutable.')
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        """Initial configs should not be removed!"""
        if self.__class__.__immutable:
            raise AttributeError('DefaultConfig instance attributes are immutable.')
        else:
            super().__delattr__(name)
