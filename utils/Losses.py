"""
Custom Losses 
"""
import torch
import torch.nn as nn

class diceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sigmoid(x):
        return torch.round(1/(1+torch.exp(-x)))

    def forward(self, pred, target):
        pred = self.sigmoid(pred.contiguous())
        target = self.sigmoid(target.contiguous())

        dice_coef = (2. * (pred * target).double().sum() + 1) / \
            (pred.double().sum() + target.double().sum() + 1)
        loss = 1-dice_coef

        return loss.mean()
