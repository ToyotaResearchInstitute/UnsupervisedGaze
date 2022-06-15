"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import logging
import os
import shutil
import pickle
import sys
import numpy as np

import torch

from core import training
from datasources.setup_data import setup_data
from models.cross_encoder import CrossEncoder

sys.path.append("../utils")
from utils.torch_utils import container_send_to_device, container_detach, container_to_numpy
config, device = training.script_init_common()

# Check for existing features
if os.path.exists(config.gaze_feature_path):
    if not config.overwrite:
        raise Exception("ERROR: Output directory already exists. Set --overwrite 1 if you want to overwrite the previous results.")
    else:
        shutil.rmtree(config.gaze_feature_path)
os.makedirs(config.gaze_feature_path)

# Get data
data_dicts = setup_data(mode='extraction')

# Load model
model = CrossEncoder()
model = model.to(device)
    
# Inference
model.eval()
training.salvage_memory()
for fold_name, data_dict in data_dicts.items():
    # Extract the features
    output_dict = {}
    output_dict['tags'] = data_dict['dataset'].original_full_dataset.tags
    output_dict['samples'] = []
    with torch.no_grad():
        for batch_it, input_data in enumerate(data_dict['dataloader']):
            # Status update
            print('Extracting sample batch {:d}/{:d}'.format(batch_it+1,len(data_dict['dataloader'])))

            # Move tensors to GPU
            input_data['tags'] = data_dict['dataset'].original_full_dataset.tags
            input_data = container_send_to_device(input_data, device)

            # preprocess images for input & ground truth
            model_outputs = model.get_features(input_data)
            model_outputs = container_detach(model_outputs)
            model_outputs = container_to_numpy(model_outputs)
            input_data = container_detach(input_data)
            input_data = container_to_numpy(input_data)

            # Loop through samples in the batch and save
            for i, sample_number in enumerate(input_data['sample_number']):
                new_sample = {}
                new_sample['paths'] = []
                for view_paths in input_data['paths']:
                    new_sample['paths'].append(view_paths[i])
                new_sample['gaze_dirs'] = input_data['gaze_dirs'][i]
                new_sample['head_dirs'] = input_data['head_dirs'][i]
                new_sample['valids'] = input_data['valids'][i]
                new_sample['eye_sides'] = input_data['eye_sides'][i]
                new_sample['glasses_ons'] = input_data['glasses_ons'][i]
                new_sample['features'] = {}
                new_sample['confidences'] = {}
                for feature_name in config.feature_sizes.keys():
                    new_sample['features'][feature_name] = model_outputs['features'][feature_name][i]
                    new_sample['confidences'][feature_name] = model_outputs['confidences'][feature_name][i]
                output_dict['samples'].append(new_sample)
            
        # Save the features
        save_path = os.path.join(config.gaze_feature_path, fold_name+'.pkl')
        f = open(save_path,'wb')
        pickle.dump(output_dict, f)
        f.close()

# Free up memory
training.salvage_memory()
