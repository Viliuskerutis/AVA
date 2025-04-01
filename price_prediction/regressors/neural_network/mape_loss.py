import torch
import torch.nn as nn


class MAPELoss(nn.Module):
    def forward(self, y_pred, y_true):
        epsilon = 1e-8
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
