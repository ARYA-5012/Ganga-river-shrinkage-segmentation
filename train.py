"""
Training script for DeepLabV3+ water segmentation model.

Usage:
    python src/model/train.py --train_dir data/train --val_dir data/val --epochs 15
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.deeplabv3plus import create_model, save_checkpoint
from model.loss import get_loss_function
from utils.metrics import compute_iou, compute_metrics
from utils.config import *


class WaterSegmentationDataset(Dataset):
    """Dataset for water body segmentation."""
    
    def __init__(self, data_dir, cities=['prayagraj', 'varanasi', 'patna']):
        """
        Args:
            data_dir: Root directory containing city subdirectories
            cities: List of city names to include
        """
        self.image_paths = []
        self.mask_paths = []
        
        for city in cities:
            city_img_dir = os.path.join(data_dir, city, 'images')
            city_mask_dir = os.path.join(data_dir, city, 'masks')
            
            if not os.path.exists(city_img_dir):
                continue
            
            for img_name in os.listdir(city_img_dir):
                img_path = os.path.join(city_img_dir, img_name)
                mask_path = os.path.join(city_mask_dir, img_name)
                
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        
        # Convert to tensors (C, H, W) for image, (H, W) for mask
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            iou = compute_iou(torch.sigmoid(outputs), masks)
        
        total_loss += loss.item()
        total_iou += iou
        
        pbar.set_postfix({'loss': loss.item(), 'iou': iou})
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            iou = compute_iou(torch.sigmoid(outputs), masks)
            
            total_loss += loss.item()
            total_iou += iou
            
            pbar.set_postfix({'loss': loss.item(), 'iou': iou})
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = WaterSegmentationDataset(args.train_dir)
    val_dataset = WaterSegmentationDataset(args.val_dir)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = get_loss_function(args.loss_type)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_iou = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_iou, save_path)
            print(f"✓ New best model saved! IoU: {val_iou:.4f}\n")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    print(f"\nTraining completed! Best Val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ for water segmentation')
    
    # Data
    parser.add_argument('--train_dir', type=str, default='data/train', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='data/val', help='Validation data directory')
    
    # Model
    parser.add_argument('--encoder_name', type=str, default='resnet50', help='Encoder name')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='Encoder weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    
    # Loss
    parser.add_argument('--loss_type', type=str, default='bce_dice', choices=['bce', 'dice', 'bce_dice', 'focal'])
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='model_checkpoints', help='Checkpoint directory')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)
