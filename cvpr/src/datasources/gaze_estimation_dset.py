"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import os
import random
from collections import OrderedDict
from turtle import xcor
import numpy as np
import pickle
from torch.utils.data import Dataset
import sys

import torch

from core.config_default import DefaultConfig
config = DefaultConfig()
sys.path.append("../utils")
from utils.torch_utils import container_to_tensors
from utils.angles import pitch_yaw_to_vector, vector_to_pitch_yaw

class GazeEstimationDset(Dataset):
    def __init__(self, dataset_path):
        f = open(dataset_path,'rb')
        data = pickle.load(f)
        f.close()
        self.samples = data['samples']
        self.tags = data['tags']

        # Calculate mean ground truth for baseline oracle
        
        if 'mean' in config.eval_features:
            self.just_label = True
            self.mean_gts = None
            self.total_valid = 0
            for sample in self:
                label = sample['label']
                if self.mean_gts is None:
                    if len(label.shape) == 1:
                        self.mean_gts = np.zeros(1,)
                    else:
                        self.mean_gts = np.zeros(label.shape[-1])
                if config.eval_target == 'eye_side':
                    label = label.astype(float) - 0.5
                self.mean_gts += np.sum(label[sample['valids']], axis=0)
                self.total_valid += np.sum(sample['valids'])
            self.mean_gts = self.mean_gts / self.total_valid
        self.just_label = False
            

        self.num_views = len(next(iter(data['tags'].values())))
        self.num_features = np.sum([x.shape[-1] for x in self[0]['features'].values()])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = {}
        data['sample_number'] = idx
        data['valids'] = sample['valids']
        data['paths'] = sample['paths']

        # Make abs version of gaze label
        sample['cam_gaze_dirs'] = pitch_yaw_to_vector(vector_to_pitch_yaw(sample['gaze_dirs']) + vector_to_pitch_yaw(sample['head_dirs']))

        # Select anchor label
        data['label'] = sample[config.eval_target + 's']
        if self.just_label:
            return data

        # Assemble features
        feature_dict = OrderedDict()
        conf_dict = OrderedDict()
        for eval_feature in config.eval_features:
            if eval_feature in sample['features']:
                feature_dict[eval_feature] = sample['features'][eval_feature]
                conf_dict[eval_feature] = sample['confidences'][eval_feature]
            elif eval_feature == 'mean':
                # Used for mean oracle baseline
                feature_dict[eval_feature] = np.repeat(np.expand_dims(self.mean_gts, axis=0), self.num_views, axis=0)
            else:
                raise Exception('ERROR: Feature type "{:s}" not found in dataset'.format(eval_feature))
        data['features'] = feature_dict
        data['confidences'] = conf_dict
        data = container_to_tensors(data)

        return data
    