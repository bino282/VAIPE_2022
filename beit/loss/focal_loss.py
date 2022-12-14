"""
https://arxiv.org/abs/1708.02002
"""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None,ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, input, target):
        logpt = -self.ce_fn(input, target)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss
