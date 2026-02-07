"""
DeepLabV3+ model definition using segmentation_models_pytorch.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_model(encoder_name="resnet50", 
                encoder_weights="imagenet",
                in_channels=3,
                classes=1):
    """
    Create DeepLabV3+ model for binary water segmentation.
    
    Args:
        encoder_name: Encoder backbone (e.g., 'resnet50', 'resnet101')
        encoder_weights: Pretrained weights ('imagenet', None)
        in_channels: Number of input channels (3 for RGB)
        classes: Number of output classes (1 for binary segmentation)
    
    Returns:
        model: DeepLabV3Plus model
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    
    return model


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file (.pth)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    return model


def save_checkpoint(model, optimizer, epoch, val_loss, val_iou, save_path):
    """
    Save model checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch number
        val_loss: Validation loss
        val_iou: Validation IoU
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_iou': val_iou
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
