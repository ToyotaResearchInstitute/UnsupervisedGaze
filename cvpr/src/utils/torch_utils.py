"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import numpy as np

import torch
import torch.nn.functional as F


def iterate_containers(d, func, p_k=None):
    if isinstance(d, dict):
        new_dict = {}
        for k,v in d.items():        
            new_dict[k] = iterate_containers(v, func, k)
        return new_dict
    elif isinstance(d, list):
        new_list = []
        for k, v in enumerate(d):        
            new_list.append(iterate_containers(v, func, k))
        return new_list
    else:
        return func(d, p_k)

def convert_to_tensor(d, p_k):
    if isinstance(d, np.ndarray):
        return torch.from_numpy(d)
    else:
        return d

def container_to_tensors(d):
    return iterate_containers(d, convert_to_tensor)

def send_to_device(v, k, device):
    if isinstance(v, torch.Tensor):
        return v.detach().to(device, non_blocking=True)
    else:
        return v

def container_send_to_device(d, device):
    return iterate_containers(d, lambda v, k: send_to_device(v, k, device))

def detach_tensor(v, p_k):
    if isinstance(v, torch.Tensor):
        return v.detach()
    else:
        return v

def container_detach(d):
    return iterate_containers(d, detach_tensor)

def tensor_to_numpy(v, p_k):
    if isinstance(v, torch.Tensor):
        return v.cpu().numpy()
    else:
        return v

def container_to_numpy(d):
    return iterate_containers(d, tensor_to_numpy)

def nansoftmax(d, dim=-1):
    invalids = torch.isnan(d)
    d = d.clone()
    d[invalids] = float("-Inf")
    d = F.softmax(d, dim=dim)
    d = d.clone()
    d[invalids] = np.nan
    return d

def nanmax(d, dim=-1):
    invalids = torch.isnan(d)
    all_invalid = torch.all(invalids, dim=dim)
    d = d.clone()
    d[invalids] = float("-Inf")
    d = torch.max(d, dim=dim)[0]
    d = d.clone()
    d[all_invalid] = np.nan
    return d
