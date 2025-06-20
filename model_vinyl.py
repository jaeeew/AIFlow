# model_vinyl.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

# Focal Loss (Binary용)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        BCE_loss = self.bce(preds, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss

# 혼합 손실함수: BCE + Dice + Focal
class HybridLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.4, focal_weight=0.3):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.w_bce = bce_weight
        self.w_dice = dice_weight
        self.w_focal = focal_weight

    def forward(self, preds, targets):
        loss = (
            self.w_bce * self.bce(preds, targets) +
            self.w_dice * self.dice(preds, targets) +
            self.w_focal * self.focal(preds, targets)
        )
        return loss

def get_model(device):
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model.to(device)