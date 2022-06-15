"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import torch
from torch.nn import functional as F

import core.training as training
from datasources.setup_data import setup_data
from models.cross_encoder import CrossEncoder

config, device = training.script_init_common()

# Setup data
train_val_data = setup_data(mode='training')

# Define model
model = CrossEncoder()
model = model.to(device)

# Optimizer
optimizers = [
    torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    ),
]

# Setup
model, optimizers, tensorboard = training.setup_common(model, optimizers)

# Training
training.main_loop(model, optimizers, train_val_data, tensorboard)
