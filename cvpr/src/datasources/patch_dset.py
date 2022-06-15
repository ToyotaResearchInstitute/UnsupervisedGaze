"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

from frozendict import frozendict
import numpy as np
import random
import copy
import cv2

import torch
from PIL import ImageOps
from torchvision import transforms
from torch.utils.data import Dataset

from core import DefaultConfig
config = DefaultConfig()
from utils.torch_utils import container_to_tensors
from utils.data_types import MultiDict


eve_has_glasses = set(['test03', 'test08', 'val04', 'train06', 'train30', 'train39'])

class PatchDataset(Dataset):
    def __init__(self, split_sample_level, tag_combos, is_eval):
        super(PatchDataset, self).__init__()
        self.patch_level_order = ['sub', 'head', 'gaze', 'app']
        self.split_sample_level = split_sample_level
        self.is_eval = is_eval  # If evaluation, sampling is deterministic
        self.patches = MultiDict(self.patch_level_order) # {tags:{sub, head, app, gaze}} --> Info needed to load
        self.sub_unique_tags = {}  # sub --> Dict of tag --> set(unique tag values)
        self.set_tag_combos(tag_combos)

    def add_patch(self, tags, access_info):
        if tags in self.patches:
            raise Exception('Patch already exists')
        self.patches[tags] = access_info

    def finalize_patches(self):
        self.sample_key_list = list(self.patches.keys(stop_level=self.split_sample_level))

        # Add to sub unique tags
        for sub, sub_dict in self.patches.items():
            cur_sub_unique_tags = {}
            cur_sub_unique_tags['sub'] = set([sub,])
            for keys in sub_dict.keys():
                for tag, tag_value in keys.items():
                    if tag not in cur_sub_unique_tags:
                        cur_sub_unique_tags[tag] = set()
                    cur_sub_unique_tags[tag].add(tag_value)
            self.sub_unique_tags[sub] = cur_sub_unique_tags

        # Pre-determine random values when in evaluation
        if self.is_eval:
            self.eval_sample_anchors = []
            self.eval_tag_perm_to_values = []
            for sample_key in self.sample_key_list:
                sample = self.patches[sample_key]
                if isinstance(sample, MultiDict):
                    anchor = random.choice(list(sample.keys()))
                else:
                    anchor = {}
                self.eval_sample_anchors.append(anchor)
                self.eval_tag_perm_to_values.append(self.anchor_get_tag_values(sample_key, anchor))


    def set_tag_combos(self, tag_combos):
        for tag_combo in tag_combos:
            tag_combo['sub'] = 0
        self.tag_combos = tag_combos
        self.tags = {}
        for combo_i, tag_combo in enumerate(self.tag_combos):
            for tag_name, tag_value in tag_combo.items():
                if tag_name not in self.tags:
                    self.tags[tag_name] = np.empty((len(self.tag_combos),), dtype=np.int)
                self.tags[tag_name][combo_i] = tag_value
        self.tags = container_to_tensors(self.tags)
        self.unique_tag_perms = {}
        for tag_combo in self.tag_combos:
            for tag, tag_perm in tag_combo.items():
                if tag not in self.unique_tag_perms:
                    self.unique_tag_perms[tag] = set()
                self.unique_tag_perms[tag].add(tag_perm)

    def __len__(self):
        return len(self.sample_key_list)

    def load_patch(self, access_info, sample_tags):
        raise NotImplementedError()

    def anchor_get_tag_values(self, sample_key, anchor, attempts=100):
        tag_perm_to_value = {}
        sub = sample_key['sub']
        tags = sample_key
        tags.update(anchor)
        unique_tags = self.sub_unique_tags[sub]

        # Add anchor
        for tag, tag_value in tags.items():
            tag_perm_to_value[tag] = {}
            tag_perm_to_value[tag][0] = tag_value

        # Try to find the most valid permutations
        best_tag_perm_to_value = None
        best_valid = -1
        for _ in range(attempts):
            # Add other permutations
            for tag, unique_perms in self.unique_tag_perms.items():
                available_tags = unique_tags[tag] - set([tag_perm_to_value[tag][0],])
                for tag_perm in unique_perms:
                    if tag_perm != 0:
                        # Assign a random value to the permutation (not already in use)
                        selected_tag = random.choice(list(available_tags))
                        tag_perm_to_value[tag][tag_perm] = selected_tag
                        available_tags = available_tags - set([selected_tag,])

            # Count how many valid
            valid = 0
            for tag_combo in self.tag_combos:
                combo_tags = {}
                combo_tags['sub'] = sample_key['sub']
                for tag, tag_perm in tag_combo.items():
                    tag_value = tag_perm_to_value[tag][tag_perm]
                    combo_tags[tag] = tag_value
                if combo_tags in self.patches:
                    valid += 1

            # Save if best so far
            if valid > best_valid:
                best_valid = valid
                best_tag_perm_to_value = copy.deepcopy(tag_perm_to_value)
                if valid == len(self.tag_combos):
                    break

        return best_tag_perm_to_value

    def prepare_images(self, images):
        gt = []
        for im in images:
            im_proc = transforms.ToPILImage()(im)
            if config.data_grayscale:
                im_proc = transforms.functional.to_grayscale(im_proc)
            im_proc = ImageOps.equalize(im_proc)
            im_proc = transforms.ToTensor()(im_proc)
            gt.append(im_proc.detach())
        gt = torch.stack(gt, axis=0)
        return gt

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (config.raw_input_size, config.raw_input_size))
        frame = np.transpose(frame, [2, 0, 1])
        return frame

    def __getitem__(self, idx):
        # Get the list of all patches in the sample
        sample_key = self.sample_key_list[idx]
        sample = self.patches[sample_key]
        if self.is_eval:
            anchor = self.eval_sample_anchors[idx]
            tag_perm_to_value = self.eval_tag_perm_to_values[idx]
        else:
            if isinstance(sample, MultiDict):
                anchor = random.choice(list(sample.keys()))
            else:
                anchor = {}
            tag_perm_to_value = self.anchor_get_tag_values(sample_key, anchor)

        # Create the samples
        # Frame data will be in format: V x C x H x W
        V = len(self.tag_combos)
        data = {}
        data['sample_number'] = idx
        data['sub'] = sample_key['sub']
        data['paths'] = []
        data['frames'] = np.empty((V, 3, config.raw_input_size, config.raw_input_size), dtype=np.uint8)
        data['valids'] = np.empty((V,), dtype=np.bool)
        # Gaze and head data will be in format: V x 3
        data['gaze_dirs'] = np.empty((V, 3), dtype=np.float32)
        data['head_dirs'] = np.empty((V, 3), dtype=np.float32)
        data['eye_sides'] = np.empty((V,), dtype=np.bool)
        data['glasses_ons'] = np.empty((V,), dtype=np.bool)
        for combo_i, tag_combo in enumerate(self.tag_combos):
            combo_tags = {}
            combo_tags['sub'] = sample_key['sub']
            for tag, tag_perm in tag_combo.items():
                tag_value = tag_perm_to_value[tag][tag_perm]
                combo_tags[tag] = tag_value
            if combo_tags in self.patches:
                patch, patch_path = self.load_patch(self.patches[combo_tags], combo_tags)
                data['paths'].append(patch_path)
                data['frames'][combo_i] = self.preprocess_frame(patch['frame'])
                data['valids'][combo_i] = combo_tags['app'] != 'face' or not self.is_eval
                data['gaze_dirs'][combo_i] = patch['gaze_dir']
                data['head_dirs'][combo_i] = patch['head_dir']
                data['eye_sides'][combo_i] = combo_tags['app'] == 'right'
                if config.dataset_name == 'eve':
                    data['glasses_ons'][combo_i] = combo_tags['sub']['participant'] in eve_has_glasses
                else:
                    data['glasses_ons'][combo_i] = 0
            else:
                random_noise = np.random.randn(3, config.raw_input_size, config.raw_input_size)
                random_min = np.min(random_noise)
                random_max = np.max(random_noise)
                random_range = random_max - random_min
                random_noise = ((random_noise - random_min) * 255.) / random_range
                data['paths'].append('NONE')
                data['frames'][combo_i] = random_noise
                data['valids'][combo_i] = 0
                data['gaze_dirs'][combo_i] = 0.
                data['head_dirs'][combo_i] = 0.
                data['eye_sides'][combo_i] = 0
                data['glasses_ons'][combo_i] = 0
        data = container_to_tensors(data)
        data['frames'] = self.prepare_images(data['frames'])
        return data
