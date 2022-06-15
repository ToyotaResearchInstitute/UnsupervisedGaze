"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

from core.config_default import DefaultConfig

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.total_features = np.sum(list(config.feature_sizes.values()))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


class BaselineEncoder(Encoder):
    def __init__(self):
        super(BaselineEncoder, self).__init__()

        self.cnn_layers = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                                 num_classes=self.total_features,
                                 norm_layer=nn.InstanceNorm2d)
        
        self.fc_features = nn.ModuleDict()
        for feature_name, num_feature in config.feature_sizes.items():
            self.fc_features[feature_name] = nn.Sequential(
                nn.Linear(self.total_features, num_feature),
                nn.SELU(inplace=True),
                nn.Linear(num_feature, num_feature),
            )

        self.fc_conf = nn.ModuleDict()
        for feature_name, num_feature in config.feature_sizes.items():
            self.fc_conf[feature_name] = nn.Sequential(
                nn.Linear(self.total_features, num_feature),
                nn.SELU(inplace=True),
                nn.Linear(num_feature, 1),
            )

    def forward(self, eye_image):
        x = self.cnn_layers(eye_image)
        out_features = OrderedDict()
        out_confs = OrderedDict()
        for feature_name in config.feature_sizes.keys():
            out_features[feature_name] = self.fc_features[feature_name](x)
            out_confs[feature_name] = torch.squeeze(self.fc_conf[feature_name](x), dim=-1)
        return out_features, out_confs


class BaselineGenerator(Generator):
    def __init__(self, input_num_feature, generator_num_feature=64):
        super(BaselineGenerator, self).__init__()

        self.input_num_feature = input_num_feature
        self.generator_num_feature = generator_num_feature

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_num_feature, generator_num_feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_num_feature * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(generator_num_feature * 8, generator_num_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_num_feature * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(generator_num_feature * 4, generator_num_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_num_feature * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(generator_num_feature * 2, generator_num_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_num_feature),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(generator_num_feature , int(generator_num_feature/2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(generator_num_feature/2)),
            nn.ReLU(True),
            # state size. (ngf/2) x 64 x 64
            nn.ConvTranspose2d(int(generator_num_feature/2), 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 128 x 128
        )

        # initialization
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        return self.main(input)
