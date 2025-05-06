import torch
from torch import nn
import torch.nn.functional as F

from src.transforms import reverse_normalize

def smooth_f1_loss(input, target, beta, epsilon):
    TP = (input * target).sum()
    FP = ((1 - input) * target).sum()
    FN = (input * (1 - target)).sum()
    fbeta = (1 + beta**2) * TP / ((1 + beta**2) * TP + (beta**2) * FN + FP + epsilon)
    fbeta = fbeta.clamp(min=epsilon, max=1 - epsilon)
    
    return 1 - fbeta

class SmoothF1Loss(nn.Module):
    def __init__(self, beta=1.0, epsilon=1e-2, reduction="mean", mean = None, std = None):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.reduction = reduction
        self.std = std
        
    def forward(self, input, target):
        loss = torch.zeros(0).to(input.device)
        self.std = self.std.to(input.device)

        if target.dim() == 3:
          target = target.unsqueeze(1)
          
        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_normalize(input, H=H, W=W)
            
            target_threshold = target.max() * 0.9
            target = (target > target_threshold).float()
            
            input = (input - target_threshold) / self.std
            input = F.sigmoid(input)
            
            f1_loss = smooth_f1_loss(input, target, self.beta, self.epsilon).mean()
            loss = torch.cat((loss, f1_loss.view(1)), dim=0)
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

def single_dice_loss(input, target, squared_pred, jaccard, smooth_nr, smooth_dr):
    intersection = (input * target).sum()
    
    if squared_pred:
        truth = (target ** 2).sum()
        pred = (input ** 2).sum()
    else:
        truth = target.sum()
        pred = input.sum()

    denominator = truth + pred
    
    if jaccard:
        denominator = 2.0 * (denominator - intersection)
    
    return 1.0 - (2.0 * intersection + smooth_nr) / (denominator + smooth_dr)

class DiceLoss(nn.Module):
    def __init__(
            self,
            squared_pred = False,
            jaccard = False,
            smooth_nr = 1e-5,
            smooth_dr = 1e-5,
            reduction = "mean",
            mean = None,
            std = None,
        ):
        super().__init__()
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.reduction = reduction
        self.std = std
    
    def forward(self, input, target):
        loss = torch.zeros(0).to(input.device)
        self.std = self.std.to(input.device)

        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_normalize(input, H=H, W=W)
            
            target_threshold = target.max() * 0.9
            target = (target > target_threshold).float()
            
            input = (input - target_threshold) / self.std
            input = F.sigmoid(input)

            input = torch.where((input > 0.5) == (target > 0.5), target, input)

            diceloss = single_dice_loss(input, target, self.squared_pred, self.jaccard, self.smooth_nr, self.smooth_dr)
            loss = torch.cat((loss, diceloss.view(1)), dim=0)
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

class MSELoss(nn.Module):
    def __init__(self, reduction="mean", mean = None, std = None):
        super().__init__()
        self.MSE = nn.MSELoss()
        self.reduction = reduction
    
    def forward(self, input, target):
        loss = torch.zeros(0).to(input.device)

        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_normalize(input, H=H, W=W)
            
            mse = self.MSE(input, target)
            loss = torch.cat((loss, mse.view(1)), dim=0)
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        
class RMSELoss(nn.Module):
    def __init__(self, reduction="mean", mean = None, std = None):
        super().__init__()
        self.MSE = nn.MSELoss()
        self.reduction = reduction
    
    def forward(self, input, target):
        loss = torch.zeros(0).to(input.device)

        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_normalize(input, H=H, W=W)
            
            mse = self.MSE(input, target)
            rmse = torch.sqrt(mse)
            loss = torch.cat((loss, rmse.view(1)), dim=0)
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
class MAELoss(nn.Module):
    def __init__(self, reduction="mean", mean = None, std = None):
        super().__init__()
        self.MAE = nn.L1Loss()
        self.reduction = reduction
        
    def forward(self, input, target):
        loss = torch.zeros(0).to(input.device)

        for input, target in zip(input, target):
            H, W = target.shape[-2:]
            input = reverse_normalize(input, H=H, W=W)
            
            mae = self.MAE(input, target)
            loss = torch.cat((loss, mae.view(1)), dim=0)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
