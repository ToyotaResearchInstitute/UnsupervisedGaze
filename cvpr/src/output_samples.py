"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import logging
import numpy as np
import os
import shutil
import _pickle as cPickle
import bz2
import cv2
import random

import torch

from core import training, training_gaze_estimation
from models.gaze_estimator import GazeEstimation
from datasources.setup_data import init_datasets_gaze_estimation
from models.cross_encoder import CrossEncoder
from utils.torch_utils import container_detach, container_to_numpy

config, device = training.script_init_common()

# Delete directory if already exists
output_dir = "outputs/samples/example_samples"
if os.path.exists(output_dir):
    if config.overwrite: 
        shutil.rmtree(output_dir)
    else:
        raise Exception("Output directory already exists and not overwriting")

def output_sample(sample, tags, out_dir, sub_name, model):
    # Output ground truth frames
    os.makedirs(out_dir, exist_ok=True)
    for v, in_path in enumerate(sample['paths']):
        out_gt_path = os.path.join(out_dir, "{:s}-gt-H{:d}_G{:d}_A{:d}.png".format(sub_name, tags['head'][v], tags['gaze'][v], tags['app'][v]))
        out_bw_path = os.path.join(out_dir, "{:s}-bw-H{:d}_G{:d}_A{:d}.png".format(sub_name, tags['head'][v], tags['gaze'][v], tags['app'][v]))
        entry = bz2.BZ2File(in_path, 'rb')
        entry = cPickle.load(entry)
        frame_gt = entry['frame']
        frame_gt = np.transpose(frame_gt, [1, 2, 0])
        frame_gt = cv2.resize(frame_gt, (config.raw_input_size, config.raw_input_size))
        frame_gt = cv2.cvtColor(frame_gt, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_gt_path, frame_gt)
        frame_bw = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out_bw_path, frame_bw)

    # Create generated images
    concat_features = []
    for tag in config.feature_sizes.keys():
        concat_features.append(sample['features'][tag])
    concat_features = torch.cat(concat_features, dim=-1)
    concat_features = concat_features.unsqueeze(-1).unsqueeze(-1)
    frame_preds = model.generate_images(concat_features)
    frame_preds = container_to_numpy(container_detach(frame_preds))
    
    # Output predicted frames
    for v in range(len(sample['paths'])):
        out_pred_path = os.path.join(out_dir, "{:s}-recon-H{:d}_G{:d}_A{:d}.png".format(sub_name, tags['head'][v], tags['gaze'][v], tags['app'][v]))
        frame_pred = frame_preds[v]
        min_val = np.min(frame_pred)
        max_val = np.max(frame_pred)
        frame_pred = ((frame_pred - min_val) * 255) / (max_val - min_val)
        frame_pred = frame_pred.astype(np.uint8)
        frame_pred = np.transpose(frame_pred, [1, 2, 0])
        cv2.imwrite(out_pred_path, frame_pred)
        

# Specify datasets used
dataset_path = config.gaze_feature_path
data = init_datasets_gaze_estimation(dataset_path)

# Load model
model = CrossEncoder()
model.eval()

# Get some samples
fold_name = 'train'
tags = data[fold_name]['dataset'].tags

for _ in range(1):
    while True:
        idx = 31234 #random.randrange(len(data[fold_name]['dataset']))
        sample = data[fold_name]['dataset'][idx]
        if torch.all(sample['valids'] == 1):
            break
    output_sample(sample, tags, output_dir, str(idx), model)
