from typing import List

import math
import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import default_collate

from src.loss import *
from src.metrics import *

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def data_mapping(data):
    data["ir_drop"] = data["ir_drop"].reshape(1, data["H"][0], data["W"][0], 1).transpose((0, 3, 1, 2))
    
    return data

def collate_fn(batch):
    elem = batch[0]
    return {key: ([d[key] for d in batch] if "ir_drop" in key else default_collate([d[key] for d in batch])) for key in elem}

def get_min_max_mean_std(
    dataset: Dataset,
    in_chs: List[int],
    data_keys: List[str],
):
    d = dict()
    min = dict()
    max = dict()
    mean = dict()
    std = dict()
    
    for data_key, in_ch in zip(data_keys, in_chs):
        if "H" in data_key or "W" in data_key:
            min[data_key] = torch.full((1, ), float("inf"))
            max[data_key] = torch.zeros(1)
            d[data_key]   = []
        else:
            min[data_key]  = torch.full((in_ch, ), float("inf"))
            max[data_key]  = torch.zeros(in_ch)
            mean[data_key] = torch.zeros(in_ch)
            std[data_key]  = torch.zeros(in_ch)
    
    dataset = dataset.with_format("torch", columns=data_keys)

    for data in tqdm(dataset):
        for data_key in data_keys:
            data[data_key] = data[data_key].float()
            if "H" in data_key or "W" in data_key:
                min[data_key] = torch.min(min[data_key], data[data_key])
                max[data_key] = torch.max(max[data_key], data[data_key])
                d[data_key].append(data[data_key])
            else:
                min[data_key] = torch.min(min[data_key], data[data_key].flatten(-2).min(-1)[0].view(-1))
                max[data_key] = torch.max(max[data_key], data[data_key].flatten(-2).max(-1)[0].view(-1))
                mean[data_key] += data[data_key].flatten(-2).mean(-1).view(-1) / len(dataset)
                std[data_key]  += data[data_key].flatten(-2).std(-1).view(-1) / len(dataset)
                
    for data_key in data_keys:
        if "H" in data_key or "W" in data_key:
            d[data_key] = torch.stack(d[data_key])
            mean[data_key] = d[data_key].mean()
            std[data_key] = d[data_key].std()
    
    return min, max, mean, std

def display_dataset(dataset, start_idx = 0, end_idx = 10, key = ["image"]):
    for idx in range(start_idx, end_idx):
        data = dataset[idx]

        in_chs = 0
        for k in key:
            in_chs += len(data[k])
        
        fig, ax = plt.subplots(math.ceil(in_chs / 5.0), 5, figsize=(20, math.ceil(in_chs / 5.0) * 4))
        fig.suptitle(data["data_idx"])
        
        c = 0
        for k in key:
            for i in range(len(data[k])):
                _ax = ax[c // 5][c % 5].imshow(data[k][i])
                fig.colorbar(_ax, ax=ax[c // 5][c % 5])
                c = c + 1
        
        # delete blank plots
        for _ax in ax.flatten():
            if not _ax.has_data():
                fig.delaxes(_ax)
        
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        
        _ax = ax[0].imshow(np.array(data["ir_drop"]).squeeze())
        fig.colorbar(_ax, ax=ax[0])
        _ax = ax[1].imshow(np.array(data["ir_drop"]).squeeze() > np.array(data["ir_drop"]).squeeze().max() * 0.9)
        fig.colorbar(_ax, ax=ax[1])
        
        plt.show()

def display_prediction(index, input, target):
    fig, ax = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle(index, fontsize=20)
    
    ax[0][0].set_title("Prediction")
    _ax = ax[0][0].imshow(input.squeeze(), vmin=target.min(), vmax=target.max())
    fig.colorbar(_ax, ax=ax[0][0])

    ax[0][1].set_title("Truth")
    _ax = ax[0][1].imshow(target.squeeze(), vmin=target.min(), vmax=target.max())
    fig.colorbar(_ax, ax=ax[0][1])

    ax[1][0].set_title("Prediction")
    _ax = ax[1][0].imshow(input.squeeze() > target.max() * 0.9, vmin=0, vmax=1)
    fig.colorbar(_ax, ax=ax[1][0])

    ax[1][1].set_title("Truth")
    _ax = ax[1][1].imshow(target.squeeze() > target.max() * 0.9, vmin=0, vmax=1)
    fig.colorbar(_ax, ax=ax[1][1])

    error = input.squeeze() - target.squeeze()
    ax[2][0].set_title("Positive Error")
    _ax = ax[2][0].imshow(torch.where(error > 0, error, 0))
    fig.colorbar(_ax, ax=ax[2][0])

    ax[2][1].set_title("Negative Error")
    _ax = ax[2][1].imshow(torch.where(error < 0, -error, 0))
    fig.colorbar(_ax, ax=ax[2][1])
    
    # delete blank plots
    for _ax in ax.flatten():
        if not _ax.has_data():
            fig.delaxes(_ax)
    
    plt.show()

class Result():
    def __init__(self,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.mean = mean
        self.std  = std
        self.reduction = reduction
        
        self.MAE       = MAELoss  (reduction="none", mean=self.mean, std=self.std)
        self.MSE       = MSELoss  (reduction="none", mean=self.mean, std=self.std)
        self.F1        = F1Score  (reduction="none", mean=self.mean, std=self.std)
        self.Recall    = Recall   (reduction="none", mean=self.mean, std=self.std)
        self.Precision = Precision(reduction="none", mean=self.mean, std=self.std)
        self.Max       = MaxLoss  (reduction="none", mean=self.mean, std=self.std)
        self.RMSE      = RMSELoss (reduction="none", mean=self.mean, std=self.std)
        
        self.data_len = 0
        
        self.loss = 0.0
        self.mae = 0.0
        self.mae_max = 0.0
        self.mae_min = float("inf")
        self.mse = 0.0
        self.mse_max = 0.0
        self.mse_min = float("inf")
        self.f1_score = 0.0
        self.f1_score_max = 0.0
        self.f1_score_min = 1.0
        self.recall = 0.0
        self.recall_max = 0.0
        self.recall_min = 1.0
        self.precision = 0.0
        self.precision_max = 0.0
        self.precision_min = 1.0

        self.max = 0.0
        self.max_max = 0.0
        self.max_min = float("inf")
        self.rmse = 0.0
        self.rmse_max = 0.0
        self.rmse_min = float("inf")
        
    def update(self, 
        input: torch.Tensor,
        target: List[torch.Tensor],
        loss: torch.Tensor,
    ):
        data_len  = len(target)
        mae       = self.MAE      (input, target).data
        mse       = self.MSE      (input, target).data
        f1_score  = self.F1       (input, target).data
        recall    = self.Recall   (input, target).data
        precision = self.Precision(input, target).data
        max       = self.Max      (input, target).data
        rmse      = self.RMSE     (input, target).data
        
        if self.reduction == "mean":
            self.data_len += 1
        elif self.reduction == "sum":
            self.data_len += data_len
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        
        self.loss += loss

        if self.reduction == "mean":
            self.mae += mae.mean()
        elif self.reduction == "sum":
            self.mae += mae.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        if mae.max() > self.mae_max:
            self.mae_max = mae.max()
        if mae.min() < self.mae_min:
            self.mae_min = mae.min()
        
        if self.reduction == "mean":
            self.mse += mse.mean()
        elif self.reduction == "sum":
            self.mse += mse.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        if mse.max() > self.mse_max:
            self.mse_max = mse.max()
        if mse.min() < self.mse_min:
            self.mse_min = mse.min()
        
        if self.reduction == "mean":
            self.f1_score += f1_score.mean()
        elif self.reduction == "sum":
            self.f1_score += f1_score.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        if f1_score.max() > self.f1_score_max:
            self.f1_score_max = f1_score.max()
        if f1_score.min() < self.f1_score_min:
            self.f1_score_min = f1_score.min()
            
        if self.reduction == "mean":
            self.recall += recall.mean()
        elif self.reduction == "sum":
            self.recall += recall.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        if recall.max() > self.recall_max:
            self.recall_max = recall.max()
        if recall.min() < self.recall_min:
            self.recall_min = recall.min()
            
        if self.reduction == "mean":
            self.precision += precision.mean()
        elif self.reduction == "sum":
            self.precision += precision.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        if precision.max() > self.precision_max:
            self.precision_max = precision.max()
        if precision.min() < self.precision_min:
            self.precision_min = precision.min()

        if self.reduction == "mean":
            self.max += max.mean()
        elif self.reduction == "sum":
            self.max += max.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        if max.max() > self.max_max:
            self.max_max = max.max()
        if mae.min() < self.max_min:
            self.max_min = max.min()

        if self.reduction == "mean":
            self.rmse += rmse.mean()
        elif self.reduction == "sum":
            self.rmse += rmse.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        if rmse.max() > self.rmse_max:
            self.rmse_max = rmse.max()
        if rmse.min() < self.rmse_min:
            self.rmse_min = rmse.min()
            
        return {
            "loss": loss,
            "mae": mae.mean(),
            "mse": mse.mean(),
            "f1_score": f1_score.mean(),
            "recall": recall.mean(),
            "precision": precision.mean(),
            "max": max.mean(),
            "rmse": rmse.mean(),
        }

    def reset(self):
        self.data_len = 0
        self.loss = 0.0
        self.mae = 0.0
        self.mae_max = 0.0
        self.mae_min = float("inf")
        self.mse = 0.0
        self.mse_max = 0.0
        self.mse_min = float("inf")
        self.f1_score = 0.0
        self.f1_score_max = 0.0
        self.f1_score_min = 1.0
        self.recall = 0.0
        self.recall_max = 0.0
        self.recall_min = 1.0
        self.precision = 0.0
        self.precision_max = 0.0
        self.precision_min = 1.0
        self.max = 0.0
        self.max_max = 0.0
        self.max_min = float("inf")
        self.rmse = 0.0
        self.rmse_max = 0.0
        self.rmse_min = float("inf")
    
    def average(self):
        return {
            "loss": self.loss / self.data_len,
            "mae": self.mae / self.data_len,
            "mae_max": self.mae_max,
            "mae_min": self.mae_min,
            "mse": self.mse / self.data_len,
            "mse_max": self.mse_max,
            "mse_min": self.mse_min,
            "f1_score": self.f1_score / self.data_len,
            "f1_score_max": self.f1_score_max,
            "f1_score_min": self.f1_score_min,
            "recall": self.recall / self.data_len,
            "recall_max": self.recall_max,
            "recall_min": self.recall_min,
            "precision": self.precision / self.data_len,
            "precision_max": self.precision_max,
            "precision_min": self.precision_min,
            "max": self.max / self.data_len,
            "max_max": self.max_max,
            "max_min": self.max_min,
            "rmse": self.rmse / self.data_len,
            "rmse_max": self.rmse_max,
            "rmse_min": self.rmse_min,
        }
        
    def print(self):
        result = self.average()
        print(f"Loss: {result['loss']:.4f}, \
MAE: {result['mae']:.4f} ({result['mae_max']:.4f} ~ {result['mae_min']:.4f}), \
Max: {result['max']:.4f} ({result['max_max']:.4f} ~ {result['max_min']:.4f}), \n\
RMSE: {result['rmse']:.4f} ({result['rmse_max']:.4f} ~ {result['rmse_min']:.4f}), \
MSE: {result['mse']:.4f} ({result['mse_max']:.4f} ~ {result['mse_min']:.4f}), \n\
F1 Score: {result['f1_score']:.3f} ({result['f1_score_max']:.3f} ~ {result['f1_score_min']:.3f}), \
Recall: {result['recall']:.3f} ({result['recall_max']:.3f} ~ {result['recall_min']:.3f}), \
Precision: {result['precision']:.3f} ({result['precision_max']:.3f} ~ {result['precision_min']:.3f})")
