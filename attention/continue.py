"""
Resume training from a saved checkpoint
Compatible with V2 model (Attention + Huber Loss + Dual Tracking)
Useful for continuing training with a lower learning rate or more epochs
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

from config import Config
from data import get_dataloaders
from model import InceptionV3WithAttention, Trainer, get_device


def resume_training(
    checkpoint_path='checkpoints/best_model_v2_attention.pth',
    num_additional_epochs=25,
    new_learning_rate=None,  # If None, uses the LR from checkpoint
    reset_scheduler=False,   # If True, resets the learning rate scheduler
    reset_early_stopping=True  # If True, resets early stopping counter
):
    """
    Resume training from a checkpoint (V2 Model)
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_additional_epochs: Number of additional epochs to train
        new_learning_rate: Optional new learning rate (if None, uses checkpoint LR)
        reset_scheduler: Whether to reset the learning rate scheduler
        reset_early_stopping: Whether to reset early stopping patience counter
    """
    
    # Setup
    Config.validate_paths()
    Config.create_directories()
    
    # Get device
    device = get_device()
    
    # Load data (with enhanced augmentation)
    print("\n📦 Loading data with enhanced augmentation...")
    train_loader, val_loader = get_dataloaders()
    
    # Print config
    Config.print_config()
    
    print(f"\n✓ Training samples:   {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    
    # Create model (V2 with attention)
    print("\n🏗️  Creating InceptionV3 + Attention model...")
    model = InceptionV3WithAttention(
        num_channels=5,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n📂 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    
    # Print previous performance (V2 has both Huber and MSE)
    if 'val_huber_loss' in checkpoint:
        # V2 checkpoint
        print(f"✓ Previous best Val Loss (Huber): {checkpoint['val_huber_loss']:.4f}")
        print(f"✓ Previous best Val MSE:          {checkpoint['val_mse']:.4f}")
        print(f"✓ Previous best Val RMSE:         {checkpoint['val_rmse']:.4f}")
    else:
        # Fallback for V1 checkpoint (shouldn't happen, but just in case)
        print(f"✓ Previous best Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"✓ Previous best Val RMSE: {checkpoint.get('val_rmse', 'N/A')}")
    
    # Create trainer (with Huber loss and dual tracking)
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Load optimizer state
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Optionally set new learning rate
    if new_learning_rate is not None:
        print(f"\n🔧 Setting new learning rate: {new_learning_rate}")
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_learning_rate
    else:
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"📊 Continuing with learning rate: {current_lr}")
    
    # Load training history (V2 has both Huber and MSE tracking)
    if 'history' in checkpoint:
        trainer.history = checkpoint['history']
        print(f"✓ Training history loaded ({len(trainer.history['train_loss'])} previous epochs)")
        
        # Print current metrics
        if len(trainer.history['train_loss']) > 0:
            last_train = trainer.history['train_loss'][-1]
            last_val = trainer.history['val_loss'][-1]
            last_rmse = trainer.history['val_rmse'][-1]
            print(f"  Last Train Loss (Huber): {last_train:.4f}")
            print(f"  Last Val Loss (Huber):   {last_val:.4f}")
            print(f"  Last Val RMSE:           {last_rmse:.4f}")
    
    # Set best validation loss (Huber loss for V2)
    if 'val_huber_loss' in checkpoint:
        trainer.best_val_loss = checkpoint['val_huber_loss']
    else:
        trainer.best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    # Reset or preserve early stopping counter
    if reset_early_stopping:
        trainer.patience_counter = 0
        print("✓ Early stopping counter reset")
    else:
        # Try to load patience counter if available
        if 'patience_counter' in checkpoint:
            trainer.patience_counter = checkpoint['patience_counter']
            print(f"✓ Early stopping counter: {trainer.patience_counter}/{Config.EARLY_STOP_PATIENCE}")
        else:
            trainer.patience_counter = 0
            print("✓ Early stopping counter not found in checkpoint, reset to 0")
    
    # Optionally reset scheduler (useful if you want fresh scheduler behavior)
    if reset_scheduler:
        print("✓ Learning rate scheduler reset")
        # Scheduler will start fresh with current learning rate
    
    # Print training info
    print(f"\n🚀 Resuming training for {num_additional_epochs} more epochs...")
    print(f"   Starting from epoch {checkpoint['epoch'] + 1}")
    print(f"   Target total epochs: {checkpoint['epoch'] + num_additional_epochs + 1}")
    
    # Train for additional epochs
    trainer.train(num_epochs=num_additional_epochs)
    
    print("\n✅ Resume training complete!")


def main():
    """Main function with default settings for V2 model"""
    
    # Configuration for resume training
    CHECKPOINT_PATH = 'checkpoints/best_model_v2_attention.pth'
    NUM_ADDITIONAL_EPOCHS = 40
    NEW_LEARNING_RATE = None  # Keep current LR, or set to e.g., 0.0001 for fine-tuning
    RESET_SCHEDULER = False   # Keep scheduler state
    RESET_EARLY_STOPPING = True  # Reset early stopping (give it fresh chances)
    
    print("=" * 60)
    print("🔄 RESUME TRAINING (V2 - Attention + Huber)")
    print("=" * 60)
    print(f"Checkpoint:        {CHECKPOINT_PATH}")
    print(f"Additional Epochs: {NUM_ADDITIONAL_EPOCHS}")
    print(f"New Learning Rate: {NEW_LEARNING_RATE if NEW_LEARNING_RATE else 'Keep current'}")
    print(f"Reset Scheduler:   {RESET_SCHEDULER}")
    print(f"Reset Early Stop:  {RESET_EARLY_STOPPING}")
    print("=" * 60)
    
    resume_training(
        checkpoint_path=CHECKPOINT_PATH,
        num_additional_epochs=NUM_ADDITIONAL_EPOCHS,
        new_learning_rate=NEW_LEARNING_RATE,
        reset_scheduler=RESET_SCHEDULER,
        reset_early_stopping=RESET_EARLY_STOPPING
    )


if __name__ == '__main__':
    main()