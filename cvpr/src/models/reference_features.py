"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from core.config_default import DefaultConfig

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReferenceFeatureCreator(nn.Module):
    def __init__(self, num_feature_in):
        super(ReferenceFeatureCreator, self).__init__()


    def forward(self, feats, confs):
        # Take random subset of references
        if self.training and config.reference_dropout:
            total_ref = feats.shape[1]
            num_ref_ss = torch.randint(low=1, high=total_ref+1, size=(1,))
            ref_idx = torch.randperm(total_ref)[:num_ref_ss]
            feats = feats[:,ref_idx]
            confs = confs[:,ref_idx]
        V = feats.shape[1]

        # Weight by confidence
        if config.reference_confidence:
            if config.reference_confidence_softmax_beta > 0.:
                confs = F.softmax(confs * config.reference_confidence_softmax_beta, dim=1) * V
            else:
                confs = (confs * V) / torch.sum(confs, dim=-1, keepdim=True)
            feats = feats * torch.unsqueeze(confs, dim=-1)

        # Combine into one feature
        if V == 1:
            # If only one feature provided - use it
            feats = feats[:,0]
        elif config.reference_comb_func == 'first':
            feats = feats[:,0]
        elif config.reference_comb_func == 'mean':
            feats = torch.mean(feats, dim=1)
        elif config.reference_comb_func == 'max':
            feats = torch.max(feats, dim=1)[0]
        elif config.reference_comb_func == 'median':
            feats = torch.median(feats, dim=1)[0]
        elif config.reference_comb_func == 'vec_median':
            # Calculate pairwise L2 distances
            feats = feats.contiguous()
            dist = torch.cdist(feats, feats)

            # Convert to similarity, making diagonal nan
            dist = dist.clone()
            diag_mask = torch.eye(V, dtype=torch.bool, device=device)
            dist.masked_fill_(diag_mask.view(1,V,V), np.nan)
            sim = -dist

            # Calculate mean similarity
            sim = torch.nanmean(sim, dim=-1)

            # Find the max similarity
            W = torch.unsqueeze(F.softmax(sim * 1000., dim=-1), dim=-1)

            # Get the median features (by weighting with W)
            feats = W * feats
            feats = torch.sum(feats, dim=1)
        else:
            raise Exception('Unknown reference feature combination function "{:s}"'.format(config.reference_comb_func))

        feats = torch.unsqueeze(feats, dim=1)
        return feats


def form_features_from_references(reference_features, all_tags, unique_tags):
    batch_size = next(iter(reference_features.values()))[0].shape[0]
    num_views = next(iter(all_tags.values())).shape[0]
    feature_sizes = {k: v[0].shape[-1] for (k, v) in reference_features.items()}
    total_features = np.sum(list(feature_sizes.values()))

    # Assemble features
    concat_features = torch.empty((batch_size, num_views, total_features), dtype=torch.float32, device=device)
    feature_offset = 0
    for feature_name, feature_size in feature_sizes.items():
        feature_unique_tags = unique_tags[feature_name]
        for tag_i in feature_unique_tags:
            idx_i = tag_i==all_tags[feature_name]
            concat_features[:,idx_i,feature_offset:feature_offset+feature_size] = reference_features[feature_name][tag_i]
        feature_offset += feature_size

    return concat_features
