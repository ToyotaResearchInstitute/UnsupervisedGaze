"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

from tensorboardX import SummaryWriter
import torchvision
import torch
import numpy as np
from PIL import Image

from core import DefaultConfig
config = DefaultConfig()

import logging
logger = logging.getLogger(__name__)


class Tensorboard(object):

    __instance = None
    __writer = None
    __current_step = 0
    __output_dir = None

    # Make this a singleton class
    def __new__(cls, *args):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    exp_name = 'temp-exp-name'
    group_name = 'temp-group-name'
    exp_tags = ['default',]

    def __init__(self, output_dir, model=None, input_to_model=None):
        self.__output_dir = output_dir
        self.__writer = SummaryWriter(output_dir)
        if model is not None:
            self.add_graph(model, input_to_model)
        self.use_wandb = config.wandb_project_name is not None
        if self.use_wandb:
            import wandb
            wandb.init(project=config.wandb_project_name,
                    group=config.group_name,
                    name=config.exp_name,
                    tags=config.exp_tags)

    def __del__(self):
        self.__writer.close()

    @property
    def output_dir(self):
        return self.__output_dir

    def update_current_step(self, step):
        self.__current_step = step

    def add_graph(self, model, input_to_model=None):
        self.__writer.add_graph(model, input_to_model)

    def prepare_image(self, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
            value = np.transpose(value, (1,2,0))
        min_val = np.min(value)
        max_val = np.max(value)
        value = ((value - min_val) * 255) / (max_val - min_val)
        value = value.astype(np.uint8)
        value = Image.fromarray(value)
        return value

    def add_grids(self, frames, prefix):
        for loss_type, loss_frames in frames.items():
            num_samples = next(iter(loss_frames.values())).shape[0]
            if num_samples > config.visualization_max_number:
                ss_idx = torch.randperm(num_samples)[:config.visualization_max_number]
            else:
                ss_idx = torch.arange(num_samples)
            for name, frames in loss_frames.items():
                self.add_grid(prefix+'-frames_diff_'+loss_type+'/'+name, frames[ss_idx], normalize=True, scale_each=True)

    def add_grid(self, tag, values, normalize=True, scale_each=True):
        grid = torchvision.utils.make_grid(values, normalize=normalize, scale_each=scale_each, nrow=config.grid_images_per_row)
        self.__writer.add_image(tag, grid, self.__current_step)
        if self.use_wandb:
            import wandb
            wandb.log({tag.replace('/','-'): wandb.Image(self.prepare_image(grid))}, step = self.__current_step)

    def add_scalar(self, tag, value):
        self.__writer.add_scalar(tag, value, self.__current_step)
        if self.use_wandb:
            import wandb
            wandb.log({tag.replace('/','-'): value}, step = self.__current_step)

    def add_image(self, tag, value):
        self.__writer.add_image(tag, value, self.__current_step)
        if self.use_wandb:
            import wandb
            wandb.log({tag.replace('/','-'): wandb.Image(self.prepare_image(value))}, step = self.__current_step)
