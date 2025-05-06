import numpy as np

import torch
import torch.nn.functional as F

def normalize(batch, mean, std):
    batch['image'] = (batch['image'] - mean['image'].view(1, -1, 1, 1).numpy()) / std['image'].view(1, -1, 1, 1).numpy()
    
    return batch

def reverse_normalize(batch, mean=None, std=None, H=None, W=None, mode="bicubic"):
    if std is not None:
        std = torch.Tensor(std).to(batch.device)
        batch = batch * std
    if mean is not None:
        mean = torch.Tensor(mean).to(batch.device)
        batch = batch + mean
    if H is not None and W is not None:
        if batch.dim() == 2:
            batch = batch.unsqueeze(0).unsqueeze(0)
        elif batch.dim() == 3:
            batch = batch.unsqueeze(0)
        
        batch = F.interpolate(batch, size=(H, W), mode=mode)
    
    return batch
