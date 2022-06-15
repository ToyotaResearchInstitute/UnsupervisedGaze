"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

from collections import OrderedDict
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomAffine, RandomErasing
import logging

from core.config_default import DefaultConfig
from models.cross_encoder_nets import Encoder, Generator, BaselineEncoder, BaselineGenerator
from models.reconstruction_list import ReconstructionList
from models.reference_features import ReferenceFeatureCreator

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def load_weights_from_local_training(model_instance):
    if isinstance(model_instance, Encoder):
        model_fname = config.cross_encoder_checkpoint_folder + '/encoder.pt'
    elif isinstance(model_instance, Generator):
        model_fname = config.cross_encoder_checkpoint_folder + '/generator.pt'
    else:
        raise ValueError('Cannot load weights for given model instance: %s' %
                         model_instance.__class__)

    net_state_dict = torch.load(model_fname, map_location=device)
    state_dict = {}
    for k, v in net_state_dict.items():
        dict_name = k[k.index('.')+1:]
        state_dict[dict_name] = v
    model_instance.load_state_dict(state_dict)
    logger.info('Loaded model weights from: %s' % model_fname)


class CrossEncoder(nn.Module):
    def __init__(self, generator_num_feature=64):
        super(CrossEncoder, self).__init__()
    
        # network for cross encoder
        self.total_features = np.sum(list(config.feature_sizes.values()))

        self.encoder = BaselineEncoder()
        self.generator = BaselineGenerator(self.total_features, generator_num_feature)

        if config.cross_encoder_load_pretrained:
            load_weights_from_local_training(self.encoder)
            load_weights_from_local_training(self.generator)

        # Reference feature creators
        self.reference_feature_creators = nn.ModuleDict()
        for feature_name, feature_size in config.feature_sizes.items():
            self.reference_feature_creators[feature_name] = ReferenceFeatureCreator(feature_size)


    def randomly_augment_frames(self, all_frames, all_tags, unique_tags, temperature=1.): 
        # Get input dimensions
        B = all_frames.shape[0]
        C = all_frames.shape[2]
        H = all_frames.shape[3]
        W = all_frames.shape[4]
        tar_H_offset = (H - config.final_input_size) // 2
        tar_W_offset = (W - config.final_input_size) // 2

        # Only augment if temperature high enough
        if config.train_augment_images and temperature > 0.01:
            # Setup affine transforms
            max_rotate_amount = 5. * temperature
            random_rotation = RandomAffine(degrees=[-max_rotate_amount, max_rotate_amount])
            max_translate = 0.10 * temperature
            min_scale = 1. - (0.25 * temperature)
            max_scale = 1. + (0.33 * temperature)
            random_affine = RandomAffine(degrees=0, translate=[max_translate, max_translate], scale=[min_scale, max_scale])

            # Randomly alter gamma
            gamma_range = 0.4 * temperature
            gamma = (1. - (gamma_range / 2.)) + (torch.rand((1,), device=device) * gamma_range)
            all_frames = TF.adjust_gamma(all_frames.view(-1,C,H,W), gamma).view(B,-1,C,H,W)

            # Horizontal flip for all views
            if torch.randint(2, (1,)) == 1:
                all_frames = TF.hflip(all_frames.view(-1,C,H,W)).view(B,-1,C,H,W)

            # Random yaw rotation for each camera view (head feature)
            for tag in unique_tags['head']:
                tag_idx = all_tags['head'] == tag
                all_frames[:,tag_idx] = random_rotation(all_frames[:,tag_idx].view(-1,C,H,W)).view(B,-1,C,H,W)

            # Random scale and translation for each appearance
            for tag in unique_tags['app']:
                tag_idx = all_tags['app'] == tag
                all_frames[:,tag_idx] = random_affine(all_frames[:,tag_idx].view(-1,C,H,W)).view(B,-1,C,H,W)

        # Return center pixels
        return all_frames[...,tar_H_offset:tar_H_offset+config.final_input_size,tar_W_offset:tar_W_offset+config.final_input_size].detach()


    def add_random_noise(self, all_frames, temperature=0.):
        # Only add noise if temperature high enough
        if config.train_denoise_images and temperature > 0.01:
            # Setup random erase
            erase_scale = [0.02, 0.02 + (0.31 * temperature)]
            random_erase = RandomErasing(p=0.5, scale=erase_scale, value=0.5, inplace=True)

            # Loop through all views
            V = all_frames.shape[1]
            for i in range(V):
                # Black out parts of image
                all_frames[:,i] = random_erase(all_frames[:,i])                

                # Add Guassian noise
                noise_factor = 0.1 * temperature * torch.rand(size=(1,), device=device)
                all_frames[:,i] = all_frames[:,i] + (noise_factor * torch.randn(size=all_frames[:,i].shape, device=device))

        return all_frames.detach()


    def get_features(self, input_data, temperature=0.):
        # Determine unique tags
        unique_tags = OrderedDict()
        for feature_name in input_data['tags'].keys():
            unique_tags[feature_name] = list(torch.unique(input_data['tags'][feature_name], sorted=True).detach().cpu().numpy())

        # Randomly augment the inputs
        input_data['frames'] = self.randomly_augment_frames(input_data['frames'], input_data['tags'], unique_tags, temperature)

        # Add random noise and patch removal to input
        input_frames = self.add_random_noise(input_data['frames'].clone(), temperature)

        # Frame data will be in format: N x V x C x H x W
        batch_size = input_frames.shape[0]
        num_views = input_frames.shape[1]
        C = input_frames.shape[2]
        H = input_frames.shape[3]
        W = input_frames.shape[4]

        # Flatten the frames and add color dimentions
        flat_input_frames = input_frames.view(-1,C,H,W).repeat(1,3,1,1)

        # Encode the frames
        flat_features, flat_confs = self.encoder(flat_input_frames)

        # Unflatten outputs
        dict_features = OrderedDict()
        dict_conf = OrderedDict()
        # Feature data will be in format: N x V x Features
        for feat_name, flat_feature in flat_features.items():
            dict_features[feat_name] = flat_feature.view(batch_size,num_views,flat_feature.shape[-1])
        for feat_name, flat_conf in flat_confs.items():
            dict_conf[feat_name] = flat_conf.view(batch_size,num_views)
        gt_frames = input_data['frames']

        # Pack up outputs
        return {'features': dict_features, 'confidences': dict_conf, 'gt_frames': gt_frames, 'input_frames': input_frames}


    def duplicate_inputs(self, input_data):
        # If no duplicates, just return input
        total_duplicates = np.prod(np.array(list(config.train_view_duplicates.values())))
        if not self.training or np.all(total_duplicates == 1):
            return input_data

        # Get the new tags
        new_tags = copy.deepcopy(input_data['tags'])
        for duplicate_feature in ['app', 'gaze', 'head']:
            # Add duplicates for duplicate_feature
            num_duplicates = config.train_view_duplicates[duplicate_feature]
            for feature_name, feature_tags in new_tags.items():
                if feature_name == duplicate_feature:
                    tag_factor = torch.max(feature_tags) + 1
                    tag_copies = [feature_tags + (tag_factor * rep) for rep in range(num_duplicates)]
                    new_tags[feature_name] = torch.cat(tag_copies, dim=0).detach()
                else:
                    new_tags[feature_name] = feature_tags.repeat(num_duplicates).detach()

        # Need to duplicate frames and valids
        duplicated_data = {}
        duplicated_data['tags'] = new_tags
        duplicated_data['frames'] = input_data['frames'].repeat(1,total_duplicates,1,1,1).detach()
        duplicated_data['valids'] = input_data['valids'].repeat(1,total_duplicates).detach()
        return duplicated_data


    def train_batch(self, input_data, temperature=0.):
        # Duplicate inputs when requested
        input_data = self.duplicate_inputs(input_data)

        # Encode inputs
        # Frame data will be in format: N x V x C x H x W
        model_outputs = self.get_features(input_data, temperature)
        dict_features = model_outputs['features']
        dict_conf = model_outputs['confidences']
        gt_frames = model_outputs['gt_frames']
        input_frames = model_outputs['input_frames']

        # Determine unique tags
        unique_tags = OrderedDict()
        for feature_name in input_data['tags'].keys():
            unique_tags[feature_name] = list(torch.unique(input_data['tags'][feature_name], sorted=True).detach().cpu().numpy())

        # Mix up features for certain reconstruction losses
        recon_list = ReconstructionList(dict_features, gt_frames, input_frames, input_data['valids'], input_data['tags'], unique_tags)

        # Only changing one feature - paired losses
        # WARNING: Using this loss requires correctly ordered tag permutations (e.g. 0-0, 0-1, 1-0, 1-1)
        for feature_name in config.feature_sizes.keys():
            if 'recon_' + feature_name in list(config.loss_weights.keys()):
                recon_list.add_diff_pair(feature_name)

        # Calculate reconstruction loss versus reference features
        if 'recon_ref' in config.loss_weights:
            # Calculate reference features
            reference_features = OrderedDict()
            for feature_name in config.feature_sizes.keys():
                reference_features[feature_name] = {}
                for tag in unique_tags[feature_name]:
                    # Gather the features and confidences associated with a certain tag
                    tag_idx = input_data['tags'][feature_name] == tag
                    feats = dict_features[feature_name][:,tag_idx]
                    confs = dict_conf[feature_name][:,tag_idx]

                    reference_features[feature_name][tag] = self.reference_feature_creators[feature_name](feats, confs)
            recon_list.add_reference_features(reference_features)           

        # Get final features and ground truth for generator from the ReconstructionList
        recon_features, recon_gt, recon_ref, recon_valids, recon_types = recon_list.join()

        # Pass through generator
        recon_estimates = self.generator(recon_features) * recon_valids.view(-1,1,1,1)

        # Determine loss for each type
        output_dict = {'losses': OrderedDict(), 'frames': OrderedDict()}
        recon_losses = torch.abs(recon_estimates - recon_gt)
        recon_losses = torch.mean(recon_losses, dim=(1,2,3)) * recon_valids
        unique_loss_types = np.unique(recon_types)
        output_dict['valid'] = {}
        output_dict['valid']['input'] = torch.sum(input_data['valids'].type(torch.IntTensor))
        for loss_type in unique_loss_types:
            is_loss = (recon_types == loss_type)
            output_dict['losses']['recon_'+loss_type] = torch.sum(recon_losses[is_loss]) / torch.sum(recon_valids[is_loss])
            output_dict['valid']['recon_'+loss_type] = torch.sum(recon_valids[is_loss])
            output_dict['frames'][loss_type] = {}
            output_dict['frames'][loss_type]["in"] = recon_gt[is_loss]
            output_dict['frames'][loss_type]["ref"] = recon_ref[is_loss]
            output_dict['frames'][loss_type]["est"] = recon_estimates[is_loss]

        # Calculate total loss
        total_loss = 0
        for loss_name, loss_weight in config.loss_weights.items():
            if loss_name in output_dict['losses']:
                total_loss += loss_weight * output_dict['losses'][loss_name]
        output_dict['losses']['total'] = total_loss

        return output_dict


    def generate_images(self, features):
        return self.generator(features)
