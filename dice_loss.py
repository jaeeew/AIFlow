import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1.0
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2. * intersection + smooth) / (union + smooth)
        loss = 1 - dice.mean()

        return loss