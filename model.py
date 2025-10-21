"""
Model definition and training for Nutrition5k calorie estimation with surface normals
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from data import get_dataloaders


class CalorieEstimatorCNN(nn.Module):
    """Triple-stream CNN: processes RGB, Depth, and Surface Normals separately, then fuses"""
    
    def __init__(self):
        super(CalorieEstimatorCNN, self).__init__()
        
        # RGB stream - 3 convolutional layers
        self.rgb_stream = nn.Sequential(
            # Conv1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            
            # Conv2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            
            # Conv3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Depth stream - 3 convolutional layers
        self.depth_stream = nn.Sequential(
            # Conv1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Surface normals stream - 3 convolutional layers
        self.normals_stream = nn.Sequential(
            # Conv1: 3 -> 32 (normals have 3 channels: nx, ny, nz)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fusion layer + regression head
        self.fusion = nn.Sequential(
            nn.Linear(384, 256),  # 128 (RGB) + 128 (Depth) + 128 (Normals) = 384
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, 1)
        )
    
    def forward(self, rgb, depth, normals):
        """
        Args:
            rgb: RGB images [batch, 3, 224, 224]
            depth: Depth images [batch, 1, 224, 224]
            normals: Surface normals [batch, 3, 224, 224]
        
        Returns:
            calories: Predicted calories [batch]
        """
        # Extract features
        rgb_feat = self.rgb_stream(rgb).flatten(1)          # [batch, 128]
        depth_feat = self.depth_stream(depth).flatten(1)    # [batch, 128]
        normals_feat = self.normals_stream(normals).flatten(1)  # [batch, 128]
        
        # Fuse features
        fused = torch.cat([rgb_feat, depth_feat, normals_feat], dim=1)  # [batch, 384]
        
        # Regress to calories
        calories = self.fusion(fused).squeeze(1)  # [batch]
        
        return calories


class AttentionFusionCNN(nn.Module):
    """Triple-stream CNN with attention-based fusion"""
    
    def __init__(self):
        super().__init__()
        
        # Same RGB, depth, and normals streams as original
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.normals_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Attention mechanism for 3 streams
        self.attention = nn.Sequential(
            nn.Linear(384, 128),  # 128 * 3 = 384
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, 1)
        )
    
    def forward(self, rgb, depth, normals):
        rgb_feat = self.rgb_stream(rgb).flatten(1)
        depth_feat = self.depth_stream(depth).flatten(1)
        normals_feat = self.normals_stream(normals).flatten(1)
        
        # Learn attention weights for 3 streams
        concat = torch.cat([rgb_feat, depth_feat, normals_feat], dim=1)
        weights = self.attention(concat)  # [batch, 3]
        
        # Weighted fusion
        fused = (weights[:, 0:1] * rgb_feat + 
                 weights[:, 1:2] * depth_feat + 
                 weights[:, 2:3] * normals_feat)
        
        return self.regressor(fused).squeeze(1)


class Trainer:
    """Trainer class to handle training and validation"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=Config.LR_SCHEDULER_FACTOR,
            patience=Config.LR_SCHEDULER_PATIENCE,
            min_lr=Config.LR_SCHEDULER_MIN_LR
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for rgb, depth, normals, calories in tqdm(self.train_loader, desc='Training'):
            rgb = rgb.to(self.device)
            depth = depth.to(self.device)
            normals = normals.to(self.device)
            calories = calories.to(self.device)
            
            # Forward pass
            pred_calories = self.model(rgb, depth, normals)
            loss = self.criterion(pred_calories, calories)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for rgb, depth, normals, calories in tqdm(self.val_loader, desc='Validation'):
                rgb = rgb.to(self.device)
                depth = depth.to(self.device)
                normals = normals.to(self.device)
                calories = calories.to(self.device)
                
                pred_calories = self.model(rgb, depth, normals)
                loss = self.criterion(pred_calories, calories)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs=None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs (defaults to Config.EPOCHS)
        """
        num_epochs = num_epochs or Config.EPOCHS
        
        print("\n🚀 Starting training...\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            val_rmse = np.sqrt(val_loss)
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            old_lr = current_lr
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_rmse)
            
            # Print metrics
            print(f"Train Loss:    {train_loss:.4f}")
            print(f"Val Loss:      {val_loss:.4f}")
            print(f"Val RMSE:      {val_rmse:.4f}")
            print(f"Learning Rate: {new_lr:.6f}")
            
            # Check if learning rate changed
            if new_lr < old_lr:
                print(f"⚠️ Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_rmse, is_best=True)
                print(f"✓ Best model saved (Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOP_PATIENCE:
                    print(f"\n⚠️ Early stopping triggered after {Config.EARLY_STOP_PATIENCE} epochs without improvement")
                    break
            
            print()
        
        # Save training history
        self.save_history()
        
        # Plot results
        self.plot_training_history()
        
        print("\n✅ Training complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val RMSE: {np.sqrt(self.best_val_loss):.4f}")
    
    def save_checkpoint(self, epoch, val_loss, val_rmse, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'history': self.history
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        filepath = Config.CHECKPOINT_DIR / filename
        torch.save(checkpoint, filepath)
    
    def save_history(self):
        """Save training history"""
        history_path = Config.CHECKPOINT_DIR / 'training_history.npy'
        np.save(history_path, self.history)
        print(f"✓ Training history saved to {history_path}")
    
    def plot_training_history(self):
        """Generate training visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE curve
        axes[0, 1].plot(epochs, self.history['val_rmse'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('RMSE', fontsize=12)
        axes[0, 1].set_title('Validation RMSE', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(self.history['val_loss']) + 1
        best_rmse = self.history['val_rmse'][best_epoch - 1]
        axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', linewidth=2,
                           label=f'Best: Epoch {best_epoch}, RMSE={best_rmse:.2f}')
        axes[0, 1].legend(fontsize=10)
        
        # 3. Train vs Val comparison
        axes[1, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].fill_between(epochs, self.history['train_loss'], self.history['val_loss'],
                                alpha=0.3, color='gray', label='Gap')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Overfitting Check (Train-Val Gap)', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
    Training Summary
    ================
    
    Total Epochs: {len(epochs)}
    
    Best Performance:
    • Epoch: {best_epoch}
    • Val Loss: {self.history['val_loss'][best_epoch-1]:.4f}
    • Val RMSE: {best_rmse:.4f}
    
    Final Performance:
    • Train Loss: {self.history['train_loss'][-1]:.4f}
    • Val Loss: {self.history['val_loss'][-1]:.4f}
    • Val RMSE: {self.history['val_rmse'][-1]:.4f}
    
    Improvement:
    • Initial RMSE: {self.history['val_rmse'][0]:.4f}
    • Best RMSE: {best_rmse:.4f}
    • Reduction: {self.history['val_rmse'][0] - best_rmse:.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                        verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        save_path = Config.CHECKPOINT_DIR / 'training_results.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
        plt.close()


def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    
    return device


def main():
    """Main training function"""
    # Setup
    Config.validate_paths()
    Config.create_directories()
    Config.print_config()
    
    # Get device
    device = get_device()
    
    # Load data
    print("\n📦 Loading data...")
    train_loader, val_loader = get_dataloaders()
    print(f"✓ Training samples:   {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n🏗️ Creating model...")
    model = AttentionFusionCNN().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters:     {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()