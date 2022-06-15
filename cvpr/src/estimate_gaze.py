"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import logging

import torch

from core import training, training_gaze_estimation
from models.gaze_estimator import GazeEstimation
from datasources.setup_data import init_datasets_gaze_estimation

config, device = training.script_init_common()

# Specify datasets used
dataset_path = config.gaze_feature_path
data = init_datasets_gaze_estimation(dataset_path)

# Define model
model = GazeEstimation(data['train']['dataset'].original_full_dataset.num_features) 
model = model.to(device)

# Optimizer
if 'mean' in config.eval_features:
    optimizers = []
else:
    optimizers = [
        torch.optim.Adam(
            model.parameters(),
            lr=0.0001
        ),
    ]

# Setup
model, optimizers, tensorboard = training.setup_common(model, optimizers)

# Training
training_gaze_estimation.main_loop(model, optimizers, data, tensorboard)
