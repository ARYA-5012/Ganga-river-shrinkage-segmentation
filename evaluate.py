"""
Evaluation script for trained DeepLabV3+ model.

Usage:
    python src/evaluation/evaluate.py --checkpoint model_checkpoints/best_model.pth --val_dir data/val
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.deeplabv3plus import create_model, load_checkpoint
from model.train import WaterSegmentationDataset
from utils.metrics import compute_metrics, print_metrics


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        threshold: Binary classification threshold
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions.numpy(), all_targets.numpy(), threshold=threshold)
    
    return metrics


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = create_model(encoder_name=args.encoder_name)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    
    # Create dataset
    print(f"Loading validation data from: {args.val_dir}")
    val_dataset = WaterSegmentationDataset(args.val_dir)
    print(f"Validation samples: {len(val_dataset)}\n")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    metrics = evaluate_model(model, val_loader, device, threshold=args.threshold)
    
    # Print results
    print_metrics(metrics, title="Validation Metrics")
    
    # Save metrics to file if specified
    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DeepLabV3+ model')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    
    # Optional
    parser.add_argument('--encoder_name', type=str, default='resnet50', help='Encoder name')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary classification threshold')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for metrics')
    
    args = parser.parse_args()
    main(args)
