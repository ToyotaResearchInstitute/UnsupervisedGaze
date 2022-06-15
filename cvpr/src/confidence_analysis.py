"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import logging
import numpy as np
import os
import shutil
import _pickle as cPickle
import bz2
import cv2

import torch

from core import training, training_gaze_estimation
from models.gaze_estimator import GazeEstimation
from datasources.setup_data import init_datasets_gaze_estimation
from models.cross_encoder import CrossEncoder
from utils.torch_utils import container_detach, container_to_numpy, container_to_tensors

config, device = training.script_init_common()

# Delete directory if already exists
output_dir = "outputs/confidence/val"
if os.path.exists(output_dir):
    if config.overwrite: 
        shutil.rmtree(output_dir)
    else:
        raise Exception("Output directory already exists and not overwriting")

def output_patches(patches, out_dir, sub_name=None):
    all_frames = []
    for i, patch in enumerate(patches):
        entry = bz2.BZ2File(patch['path'], 'rb')
        entry = cPickle.load(entry)
        frame = entry['frame']
        frame = np.transpose(frame, [1, 2, 0])
        frame = cv2.resize(frame, (config.raw_input_size, config.raw_input_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        all_frames.append(frame)
    all_frames = np.concatenate(all_frames, axis=1)

    # Create generated images
    concat_features = []
    for patch in patches: 
        patch_features = []
        for tag in config.feature_sizes.keys():
            patch_features.append(patch['features'][tag])
        patch_features = np.concatenate(patch_features, axis=-1)
        concat_features.append(patch_features)
    concat_features = np.stack(concat_features, axis=0)
    concat_features = container_to_tensors(concat_features)
    concat_features = concat_features.unsqueeze(-1).unsqueeze(-1)
    frame_preds = model.generate_images(concat_features)
    frame_preds = container_to_numpy(container_detach(frame_preds))
    
    # Output predicted frames
    all_pred = []
    for i in range(len(patches)):
        if sub_name is not None:
            out_pred_path = os.path.join(out_dir, "{:s}_pred_{:03d}.png".format(sub_name, i))
        else:
            out_pred_path = os.path.join(out_dir, "pred_{:03d}.png".format(i))
        frame_pred = frame_preds[i]
        min_val = np.min(frame_pred)
        max_val = np.max(frame_pred)
        frame_pred = ((frame_pred - min_val) * 255) / (max_val - min_val)
        frame_pred = frame_pred.astype(np.uint8)
        frame_pred = np.transpose(frame_pred, [1, 2, 0])
        frame_pred = np.repeat(frame_pred, 3, axis=-1)
        all_pred.append(frame_pred)
    all_pred = np.concatenate(all_pred, axis=1)
    if sub_name is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "{:s}.png".format(sub_name, i))
    else:
        out_path = out_dir + ".png"
    final_image = np.concatenate([all_frames, all_pred], axis=0)
    cv2.imwrite(out_path, final_image)

# Specify datasets used
dataset_path = config.gaze_feature_path
data = init_datasets_gaze_estimation(dataset_path)

# Load model
model = CrossEncoder()
model.eval()

# Get all patches
all_sub_patches = {}
use_folds = ['val',]
for fold_name in use_folds:
    for sample in data[fold_name]['dataset'].original_full_dataset.samples:
        V = sample['valids'].shape[0]
        for i in range(V):
            if os.path.isfile(sample['paths'][i]):
                new_patch = {}
                new_patch['path'] = sample['paths'][i]
                new_patch['features'] = {}
                for conf_key in sample['features'].keys():
                    new_patch['features'][conf_key] = sample['features'][conf_key][i]
                new_patch['confidences'] = {}
                for conf_key in sample['confidences'].keys():
                    new_patch['confidences'][conf_key] = sample['confidences'][conf_key][i]
                sub = new_patch['path'].split('/')[6]
                if sub not in all_sub_patches:
                    all_sub_patches[sub] = []
                all_sub_patches[sub].append(new_patch)

# Normalize confidences by subject
sub_stats = {}
tags = next(iter(all_sub_patches.values()))[0]['confidences'].keys()
for tag in tags:
    sub_stats[tag] = []
    for sub, sub_patches in all_sub_patches.items():
        confs = np.array([patch['confidences'][tag] for patch in sub_patches])
        mean_conf = np.mean(confs)
        std_conf = np.std(confs)
        sub_stats[tag].append({'subject': sub, 'mean': mean_conf, 'std': std_conf})
        for patch in sub_patches:
            patch['confidences'][tag] = (patch['confidences'][tag] - mean_conf) / std_conf
    sub_stats[tag].sort(key=lambda x: x['mean'])

# Sort by total confidence and output best and worst per subject
N = 1
for sub, sub_patches in all_sub_patches.items():
    sub_patches.sort(key=lambda patch: np.sum([patch['confidences'][tag] for tag in tags]))
    output_patches(sub_patches[:N], os.path.join(output_dir, 'total_low'), sub)
    output_patches(sub_patches[-N:], os.path.join(output_dir, 'total_high'), sub)

# Output lowest to most confident subjects
for tag in tags:
    output_patches([all_sub_patches[x['subject']][-1] for x in sub_stats[tag]], os.path.join(output_dir, '{:s}_sub'.format(tag)))

# Sort by each confidence (low)
for tag in tags:
    low_conf = []
    high_conf = []
    for sub, sub_patches in all_sub_patches.items():
        sub_patches.sort(key=lambda patch: patch['confidences'][tag])
        low_conf.extend(sub_patches[:N])
        high_conf.extend(sub_patches[-N:])
    output_patches(low_conf, os.path.join(output_dir, '{:s}_low'.format(tag)))
    output_patches(high_conf, os.path.join(output_dir, '{:s}_high'.format(tag)))
