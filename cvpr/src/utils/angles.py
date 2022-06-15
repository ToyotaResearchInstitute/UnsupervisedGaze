"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import numpy as np 

import torch
import torch.nn.functional as F


def angular_loss(pred, gt):
    sim = F.cosine_similarity(pred, gt, dim=-1, eps=1e-8)
    sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
    return torch.acos(sim) * 180.0 / np.pi

def pitch_yaw_to_vector(pitch_yaw):
    pitch = pitch_yaw[...,0]
    yaw = pitch_yaw[...,1]
    if isinstance(pitch_yaw, torch.Tensor):
        vector = torch.stack([torch.cos(pitch)*torch.cos(yaw), torch.sin(pitch), torch.cos(pitch)*torch.sin(-yaw)], axis=-1)
    else:
        vector = np.stack([np.cos(pitch)*np.cos(yaw), np.sin(pitch), np.cos(pitch)*np.sin(-yaw)], axis=-1)
    return vector

def vector_to_pitch_yaw(vector):
    x = vector[...,0]
    y = vector[...,1]
    z = vector[...,2]
    if isinstance(vector, torch.Tensor):
        pitch = torch.asin(y)
        yaw = -torch.atan2(z,x)
        pitch_yaw = torch.stack([pitch, yaw], axis=-1)
    else:
        pitch = np.arcsin(y)
        yaw = -np.arctan2(z,x)
        pitch_yaw = np.stack([pitch, yaw], axis=-1)
    return pitch_yaw
