"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import os
import random
from collections import OrderedDict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from frozendict import frozendict
import random
import _pickle as cPickle
import bz2

from core import DefaultConfig
config = DefaultConfig()
from utils.angles import pitch_yaw_to_vector, vector_to_pitch_yaw
from datasources.patch_dset import PatchDataset


class PreprocessedDataset(PatchDataset):
    def __init__(self, dataset_path, fold_name, patches_used, split_sample_level, tag_combos, is_eval):
        super(PreprocessedDataset, self).__init__(split_sample_level, tag_combos, is_eval)
        actually_use_fold = 'val' if fold_name == 'test' else 'train'
        self.dataset_path = os.path.join(dataset_path, actually_use_fold)
        data = bz2.BZ2File(os.path.join(self.dataset_path, 'index.pbz2'), 'rb')
        self.patches = cPickle.load(data)

        self.patches.filter_level('app', set(patches_used))

        self.finalize_patches()

        if fold_name == 'test':
            return

        participant_list = [x['sub']['participant'] for x in self.sample_key_list]
        unique_participants = np.sort(np.unique(participant_list))
        nTrain = 30
        if fold_name == 'train':
            unique_participants = set(unique_participants[:nTrain])
        else:
            unique_participants = set(unique_participants[nTrain:])
        self.sample_key_list = [x for x in self.sample_key_list if x['sub']['participant'] in unique_participants]

    def load_patch(self, access_info, sample_tags):
        entry = bz2.BZ2File(os.path.join(self.dataset_path, access_info), 'rb')
        entry = cPickle.load(entry)
        entry['frame'] = np.transpose(entry['frame'], [1, 2, 0]) 
        return entry, os.path.join(self.dataset_path, access_info)
