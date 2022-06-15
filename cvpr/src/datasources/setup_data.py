"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import logging
import numpy as np
from collections import OrderedDict

import core.training as training
from datasources.preprocessed_dset import PreprocessedDataset
from datasources.gaze_estimation_dset import GazeEstimationDset

from torch.utils.data import DataLoader, Subset


logger = logging.getLogger(__name__)
config, device = training.script_init_common()


def setup_data(mode):
    # Specify folds used
    if config.test_every_n_steps == 0 and mode == 'training':
        folds_used = ['train',]
    elif mode == 'extraction':
        folds_used = ['train', 'val', 'test']
    else:
        folds_used = ['train', 'val']

    # Gather data for each fold
    data = {}
    for fold_name in folds_used:
        # Determine tag combos to use
        tag_combos = []
        for head in range(config.data_views['head']):
            for gaze in range(config.data_views['gaze']):
                for app in range(config.data_views['app']):
                    tag_combos.append({'head': head, 'gaze': gaze, 'app': app})
        
        # Get dataset
        is_eval = (mode == 'extraction') if (fold_name == 'train') else True
        if config.dataset_name == 'eve':
            split_sample_level = 'head' if (mode == 'training') else 'head'
            dataset = PreprocessedDataset(config.eve_preprocessed_path, fold_name, split_sample_level=split_sample_level, 
                                          patches_used=config.patches_used, tag_combos=tag_combos, is_eval=is_eval)
        dataset.original_full_dataset = dataset

        # Subsample, if desired
        if mode != 'extraction' and fold_name in config.subsample_fold and config.subsample_fold[fold_name] != 0:
            if len(dataset) > config.subsample_fold[fold_name]:
                subset = Subset(dataset, sorted(np.random.permutation(len(dataset))[:config.subsample_fold[fold_name]]))
                subset.original_full_dataset = dataset
                dataset = subset

        # Get dataloader
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=not is_eval,
                                drop_last=not is_eval,
                                num_workers=config.train_data_workers,
                                pin_memory=True,
                                )
        
        # Log dataset
        logger.info('> Ready to use {:s} dataset: {:s}'.format(fold_name, config.dataset_name))
        logger.info('          with number of entries: %d' % len(dataset.original_full_dataset))
        if dataset.original_full_dataset != dataset:
            logger.info('          of which we use a subset: %d' % len(dataset))

        # Add to dict
        fold_data = {
            'dataset': dataset,
            'dataloader': dataloader,
        }
        data[fold_name] = fold_data

    return data


def init_datasets_gaze_estimation(dataset_path):
    # Gather data for each fold
    folds_used = ['train', 'val', 'test']
    data = {}
    for fold_name in folds_used:
        # Initialize dataset
        dataset = GazeEstimationDset(dataset_path+'/{:s}.pkl'.format(fold_name))
        dataset.original_full_dataset = dataset

        # Subsample, if desired
        if fold_name in config.subsample_fold and config.subsample_fold[fold_name] != 0:
            if len(dataset) > config.subsample_fold[fold_name]:
                subset = Subset(dataset, sorted(np.random.permutation(len(dataset))[:config.subsample_fold[fold_name]]))
                subset.original_full_dataset = dataset
                dataset = subset

        # Setup dataloader
        dataloader = DataLoader(dataset,
                                batch_size=config.gaze_estimation_batch_size,
                                shuffle=fold_name=='train',
                                drop_last=False,
                                num_workers=config.train_data_workers,
                                pin_memory=True,
                                )

        # Log dataset
        logger.info('> Ready to use training dataset: Gaze estimation')
        logger.info('          with number of entries: %d' % len(dataset.original_full_dataset))
        if dataset.original_full_dataset != dataset:
            logger.info('          of which we use a subset: %d' % len(dataset))

        # Add to dict
        fold_data = {
            'dataset': dataset,
            'dataloader': dataloader,
        }
        data[fold_name] = fold_data

    return data
