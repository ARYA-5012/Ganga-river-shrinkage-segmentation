"""
Combined BCE + Dice Loss for binary segmentation.
Based on the paper: "Satellite Image Segmentation Using DeepLabV3+"
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    
    This is the loss function specified in the research paper for training
    the DeepLabV3+ model on water body segmentation.
    """
    
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        """
        Args:
            bce_weight: Weight for BCE loss component
            dice_weight: Weight for Dice loss component
        """
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (before sigmoid) (B, 1, H, W)
            target: Ground truth binary masks (B, 1, H, W)
        
        Returns:
            loss: Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


def get_loss_function(loss_type='bce_dice'):
    """
    Factory function to get loss function based on type.
    
    Args:
        loss_type: Type of loss ('bce', 'dice', 'bce_dice', 'focal')
    
    Returns:
        loss_fn: Loss function
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return smp.losses.DiceLoss(mode='binary', from_logits=True)
    elif loss_type == 'bce_dice':
        return CombinedLoss(bce_weight=1.0, dice_weight=1.0)
    elif loss_type == 'focal':
        return smp.losses.FocalLoss(mode='binary')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
