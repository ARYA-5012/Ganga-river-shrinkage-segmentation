"""
Metrics calculation utilities for semantic segmentation.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def compute_iou(pred, target, threshold=0.5):
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    
    Args:
        pred: Predicted probabilities (B, H, W) or (B, 1, H, W)
        target: Ground truth binary masks (B, H, W) or (B, 1, H, W)
        threshold: Threshold for binarizing predictions
    
    Returns:
        iou: Mean IoU score
    """
    pred = (pred > threshold).float()
    
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def compute_dice(pred, target, threshold=0.5):
    """
    Compute Dice coefficient for binary segmentation.
    
    Args:
        pred: Predicted probabilities (B, H, W) or (B, 1, H, W)
        target: Ground truth binary masks (B, H, W) or (B, 1, H, W)
        threshold: Threshold for binarizing predictions
    
    Returns:
        dice: Mean Dice coefficient
    """
    pred = (pred > threshold).float()
    
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    intersection = (pred * target).sum(dim=(1, 2))
    dice = (2 * intersection + 1e-6) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1e-6)
    
    return dice.mean().item()


def compute_metrics(pred, target, threshold=0.5):
    """
    Compute all evaluation metrics.
    
    Args:
        pred: Predicted probabilities (numpy array or tensor)
        target: Ground truth binary masks (numpy array or tensor)
        threshold: Threshold for binarizing predictions
    
    Returns:
        dict: Dictionary containing all metrics
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.uint8).flatten()
    target_binary = target.astype(np.uint8).flatten()
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(target_binary, pred_binary).ravel()
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(target_binary, pred_binary, zero_division=0)
    recall = recall_score(target_binary, pred_binary, zero_division=0)
    f1 = f1_score(target_binary, pred_binary, zero_division=0)
    
    # IoU
    iou = tp / (tp + fp + fn + 1e-6)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def print_metrics(metrics, title="Evaluation Metrics"):
    """
    Pretty print metrics table.
    
    Args:
        metrics: Dictionary of metrics from compute_metrics()
        title: Table title
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1']:.4f}")
    print(f"IoU:        {metrics['iou']:.4f}")
    print(f"{'='*50}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"{'':>15} {'Pred 0':>10} {'Pred 1':>10}")
    print(f"{'True 0':<15} {cm['tn']:>10} {cm['fp']:>10}")
    print(f"{'True 1':<15} {cm['fn']:>10} {cm['tp']:>10}")
    print(f"{'='*50}\n")
