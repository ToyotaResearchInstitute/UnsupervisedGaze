"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import numpy as np
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config_default import DefaultConfig
sys.path.append("../utils")
from utils.angles import pitch_yaw_to_vector
from models.reference_features import ReferenceFeatureCreator, form_features_from_references

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GazeEstimationNet(nn.Module):
    def __init__(self, num_feature_in):
        super(GazeEstimationNet, self).__init__()
    
        self.num_feature_in = num_feature_in
        self.mid_feat_num_1 = 128
        self.mid_feat_num_2 = 64
        self.mid_feat_num_3 = 16

        # network for gaze estimation
        self.out_size = 1 if config.eval_binary else 2
        self.fc_gaze = nn.Sequential(
            nn.Linear(self.num_feature_in, self.mid_feat_num_1),
            nn.ReLU(),
            # nn.Dropout(0.0001),
            nn.Linear(self.mid_feat_num_1, self.mid_feat_num_2),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.mid_feat_num_2, self.mid_feat_num_3),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.mid_feat_num_3, self.out_size),
        )

    def forward(self, feats):
        B = feats.shape[0]
        V = feats.shape[1]
        N = feats.shape[2]
        flat_feats = feats.view(B*V,N)
        flat_preds = self.fc_gaze(flat_feats) 
        preds = flat_preds.view(B, V, self.out_size)
        if not config.eval_binary:
            preds = F.hardtanh(preds)
            preds = pitch_yaw_to_vector(preds * np.pi)
        return preds


class GazeEstimation(nn.Module):
    def __init__(self, num_feature_in):
        super(GazeEstimation, self).__init__()

        # Mean oracle baseline
        if 'mean' in config.eval_features:
            return

        self.net = GazeEstimationNet(num_feature_in)
        if config.gaze_estimation_pretrained:
            net_state_dict = torch.load(config.gaze_estimation_checkpoint, map_location=device)
            state_dict = {}
            for k, v in net_state_dict.items():
                dict_name = k[k.index('.')+1:]
                state_dict[dict_name] = v
            self.fc_gaze.load_state_dict(state_dict)
    
        # Reference feature creators
        self.reference_feature_creators = nn.ModuleDict()
        for feature_name in config.eval_features:
            self.reference_feature_creators[feature_name] = ReferenceFeatureCreator(config.feature_sizes[feature_name])


    def forward(self, input_data):
        # Mean oracle baseline
        if 'mean' in config.eval_features:
            return input_data['features']['mean']

        # Calculate reconstruction loss versus reference features
        if config.gaze_estimation_use_reference_features and self.training:
            # Determine unique tags
            unique_tags = OrderedDict()
            for feature_name in config.eval_features:
                unique_tags[feature_name] = list(torch.unique(input_data['tags'][feature_name], sorted=True).detach().cpu().numpy())

            # Calculate reference features
            reference_features = OrderedDict()
            for feature_name in config.eval_features:
                reference_features[feature_name] = {}
                for tag in unique_tags[feature_name]:
                    # Gather the features and confidences associated with a certain tag
                    tag_idx = input_data['tags'][feature_name] == tag
                    feats = input_data['features'][feature_name][:,tag_idx]                    
                    confs = input_data['confidences'][feature_name][:,tag_idx]
                    
                    reference_features[feature_name][tag] = self.reference_feature_creators[feature_name](feats, confs) 

            # Construct input features
            feats = form_features_from_references(reference_features, input_data['tags'], unique_tags)
        else:
            feats = [input_data['features'][feat_name] for feat_name in config.eval_features]
            feats = torch.cat(feats, dim=-1)

        # Pass through network
        preds = self.net(feats)

        return preds
