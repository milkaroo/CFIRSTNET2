import torch
from torch import nn
import torch.nn.functional as F

from torcheval.metrics.functional import binary_f1_score, binary_recall, binary_precision
from src.transforms import reverse_normalize as reverse_transform

def get_exceeding_indices(pred: torch.Tensor, threshold: float = 0.9):
    """
    pred: (B, 1, H, W) or (1, H, W) tensor
    threshold: 기준 비율 (보통 0.9)
    return: 예측값이 threshold 이상인 위치 인덱스 리스트
    """
    if pred.ndim == 4:  # (B, 1, H, W)
        pred = pred.squeeze(1)  # -> (B, H, W)
    
    results = []
    for p in pred:
        t = p.max() * threshold
        indices = torch.nonzero(p > t, as_tuple=False)  # (N, 2)
        results.append(indices)
    return results
    
def map_to_physical_coordinates(indices, pred_shape, layout_size_um):
    """
    indices: torch.Tensor of shape (N, 2), each [y, x]
    pred_shape: (H, W) shape of prediction tensor
    layout_size_um: (layout_height_um, layout_width_um)
    """
    H, W = pred_shape
    layout_H, layout_W = layout_size_um

    scale_y = layout_H / H
    scale_x = layout_W / W

    mapped_coords = []
    for y, x in indices:
        Y_um = y.item() * scale_y
        X_um = x.item() * scale_x
        mapped_coords.append((Y_um, X_um))

    return mapped_coords


def single_f1_score(input, target):
    # CHW or HW
    expandedinput = torch.as_tensor(input)
    expandedtarget = torch.as_tensor(target)
    
    target_thresold = expandedtarget.max() * 0.9
    
    boolinput = (expandedinput > target_thresold)
    booltarget = (expandedtarget > target_thresold) 
    f1 = binary_f1_score(boolinput.flatten(), booltarget.flatten())
    
    return f1.numpy()

def f1_score(input, target, threshold):
    # CHW or HW
    target_thresold = target.max() * threshold
    
    boolinput = (input > target_thresold)
    booltarget = (target > target_thresold)
    
    return binary_f1_score(boolinput.flatten(), booltarget.flatten())

class F1Score(nn.Module):
    def __init__(self, threshold = 0.9, reduction="mean", mean=None, std=None):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
        self.mean = mean
        self.std = std
    
    def forward(self, input, target):
        values = torch.zeros(0).to(input.device)
        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_transform(input.unsqueeze(0), self.mean, self.std, H, W)
            target = reverse_transform(target.unsqueeze(0), self.mean, self.std)
            
            value = f1_score(input, target, self.threshold).view(1)
            values = torch.cat((values, value), dim=0)
            
        if self.reduction == "mean":
            return values.mean()
        elif self.reduction == "sum":
            return values.sum()
        elif self.reduction == "none":
            return values
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

def precision(input, target, threshold):
    # CHW or HW
    target_thresold = target.max() * threshold
    
    boolinput = (input > target_thresold)
    booltarget = (target > target_thresold)
    
    return binary_precision(boolinput.flatten(), booltarget.flatten())

class Precision(nn.Module):
    def __init__(self, threshold = 0.9, reduction="mean", mean=None, std=None):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
        self.mean = mean
        self.std = std
    
    def forward(self, input, target):
        values = torch.zeros(0).to(input.device)
        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_transform(input.unsqueeze(0), self.mean, self.std, H, W)
            target = reverse_transform(target.unsqueeze(0), self.mean, self.std)
            
            value = precision(input, target, self.threshold).view(1)
            values = torch.cat((values, value), dim=0)
            
        if self.reduction == "mean":
            return values.mean()
        elif self.reduction == "sum":
            return values.sum()
        elif self.reduction == "none":
            return values
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

def recall(input, target, threshold):
    # CHW or HW
    target_thresold = target.max() * threshold
    
    boolinput = (input > target_thresold)
    booltarget = (target > target_thresold)
    
    return binary_recall(boolinput.flatten(), booltarget.flatten())

class Recall(nn.Module):
    def __init__(self, threshold = 0.9, reduction="mean", mean=None, std=None):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
        self.mean = mean
        self.std = std
    
    def forward(self, input, target):
        values = torch.zeros(0).to(input.device)
        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_transform(input.unsqueeze(0), self.mean, self.std, H, W)
            target = reverse_transform(target.unsqueeze(0), self.mean, self.std)
            
            value = recall(input, target, self.threshold).view(1)
            values = torch.cat((values, value), dim=0)
            
        if self.reduction == "mean":
            return values.mean()
        elif self.reduction == "sum":
            return values.sum()
        elif self.reduction == "none":
            return values
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

class MaxLoss(nn.Module):
    def __init__(self, reduction="mean", mean = None, std = None):
        super().__init__()
        self.reduction = reduction
        self.mean = mean
        self.std = std
        
    def forward(self, input, target):
        loss = torch.zeros(0).to(input.device)
        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_transform(input.unsqueeze(0), self.mean, self.std, H=H, W=W)
            target = reverse_transform(target.unsqueeze(0), self.mean, self.std)
            
            max = (input - target).abs().max()
            loss = torch.cat((loss, max.view(1)), dim=0)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
