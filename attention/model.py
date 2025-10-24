"""
Model definition and training for Nutrition5k calorie estimation
InceptionV3 with 5-channel input (RGB + Depth + Height) + Spatial Attention
Version 2: Attention-enhanced architecture with Huber loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from data import get_dataloaders


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module
    Uses depth+height channels to generate attention map for RGB features
    """
    
    def __init__(self, feature_channels):
        """
        Args:
            feature_channels: Number of channels in the feature maps to attend to
        """
        super().__init__()
        
        # Attention generation pathway
        # Takes depth+height features and generates spatial attention map
        self.attention_conv1 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.attention_conv2 = nn.Sequential(
            nn.Conv2d(feature_channels // 4, feature_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels // 8),
            nn.ReLU(inplace=True)
        )
        
        # Final attention map: 1 channel with sigmoid activation
        self.attention_conv3 = nn.Sequential(
            nn.Conv2d(feature_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, features):
        """
        Args:
            features: Feature maps [batch, channels, H, W]
        
        Returns:
            attended_features: Features weighted by attention [batch, channels, H, W]
            attention_map: Spatial attention map [batch, 1, H, W]
        """
        # Generate attention map
        attention = self.attention_conv1(features)
        attention = self.attention_conv2(attention)
        attention_map = self.attention_conv3(attention)  # [batch, 1, H, W]
        
        # Apply attention: element-wise multiplication
        # Broadcasting: [batch, channels, H, W] * [batch, 1, H, W]
        attended_features = features * attention_map
        
        return attended_features, attention_map


class InceptionBlock(nn.Module):
    """Basic Inception module"""
    
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv -> 5x5 conv branch (using two 3x3 convs)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionV3WithAttention(nn.Module):
    """
    InceptionV3-inspired architecture with Spatial Attention
    Modified to accept 5-channel input (RGB + Depth + Height)
    Attention applied after inception3b using depth+height features
    """
    
    def __init__(self, num_channels=5, dropout_rate=0.3):
        super().__init__()
        
        # ============= Initial Convolution Layers =============
        # Modified first layer to accept 5 channels instead of 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        # ============= Inception Blocks =============
        # Inception 3a, 3b
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        
        # ============= ATTENTION MODULE =============
        # Applied after inception3b (480 channels)
        self.spatial_attention = SpatialAttentionModule(480)
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        # Inception 4a, 4b, 4c, 4d, 4e
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Inception 5a, 5b
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        
        # ============= Regression Head =============
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Final regression layers
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, 5, 299, 299] (RGB + Depth + Height)
        
        Returns:
            calories: Predicted calories [batch]
        """
        # Initial convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)
        
        # Inception modules (before attention)
        x = self.inception3a(x)
        x = self.inception3b(x)
        
        # ============= APPLY SPATIAL ATTENTION =============
        x, attention_map = self.spatial_attention(x)
        # attention_map stored but not returned (could be used for visualization)
        
        x = self.maxpool3(x)
        
        # Inception modules (after attention)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        # Regression head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x.squeeze(1)  # [batch]


class Trainer:
    """Trainer class with Huber loss and dual metric tracking"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.criterion = nn.HuberLoss(delta=50.0)  # Huber for training
        self.mse_criterion = nn.MSELoss()  # MSE for comparison tracking
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler (based on Huber loss)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=Config.LR_SCHEDULER_FACTOR,
            patience=Config.LR_SCHEDULER_PATIENCE,
            min_lr=Config.LR_SCHEDULER_MIN_LR
        )
        
        # Training history - DUAL TRACKING
        self.history = {
            'train_loss': [],       # Huber loss (used for training)
            'val_loss': [],         # Huber loss (used for early stopping)
            'val_mse': [],          # MSE (for comparison with baseline)
            'val_rmse': []          # RMSE from MSE (for comparison)
        }
        
        self.best_val_loss = float('inf')  # Best Huber loss
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch using Huber loss"""
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for inputs, calories in pbar:
                inputs = inputs.to(self.device)
                calories = calories.to(self.device)
                
                # Forward pass
                pred_calories = self.model(inputs)
                loss = self.criterion(pred_calories, calories)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """
        Validate and compute BOTH Huber and MSE losses
        Returns: (huber_loss, mse_loss)
        """
        self.model.eval()
        total_huber_loss = 0
        total_mse_loss = 0
        
        with torch.no_grad():
            for inputs, calories in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs = inputs.to(self.device)
                calories = calories.to(self.device)
                
                pred_calories = self.model(inputs)
                
                # Compute both losses
                huber_loss = self.criterion(pred_calories, calories)
                mse_loss = self.mse_criterion(pred_calories, calories)
                
                total_huber_loss += huber_loss.item()
                total_mse_loss += mse_loss.item()
        
        avg_huber_loss = total_huber_loss / len(self.val_loader)
        avg_mse_loss = total_mse_loss / len(self.val_loader)
        
        return avg_huber_loss, avg_mse_loss
    
    def train(self, num_epochs=None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs (defaults to Config.EPOCHS)
        """
        num_epochs = num_epochs or Config.EPOCHS
        
        print("\n🚀 Starting training with Spatial Attention + Huber Loss...\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train (using Huber loss)
            train_loss = self.train_epoch()
            
            # Validate (compute both Huber and MSE)
            val_huber_loss, val_mse_loss = self.validate()
            val_rmse = np.sqrt(val_mse_loss)
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler (based on Huber loss)
            old_lr = current_lr
            self.scheduler.step(val_huber_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_huber_loss)
            self.history['val_mse'].append(val_mse_loss)
            self.history['val_rmse'].append(val_rmse)
            
            # Print metrics (SHOW BOTH!)
            print(f"Train Loss (Huber): {train_loss:.4f}")
            print(f"Val Loss (Huber):   {val_huber_loss:.4f}")
            print(f"Val MSE:            {val_mse_loss:.4f}  👈 For comparison with baseline")
            print(f"Val RMSE:           {val_rmse:.4f}  👈 Target metric!")
            print(f"Learning Rate:      {new_lr:.6f}")
            
            # Check if learning rate changed
            if new_lr < old_lr:
                print(f"⚠️  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save best model (based on Huber loss)
            if val_huber_loss < self.best_val_loss:
                self.best_val_loss = val_huber_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_huber_loss, val_mse_loss, val_rmse, is_best=True)
                print(f"✅ Best model saved (Huber: {val_huber_loss:.4f}, MSE: {val_mse_loss:.4f}, RMSE: {val_rmse:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOP_PATIENCE:
                    print(f"\n⚠️  Early stopping triggered after {Config.EARLY_STOP_PATIENCE} epochs without improvement")
                    break
            
            print()
        
        # Save training history
        self.save_history()
        
        # Plot results
        self.plot_training_history()
        
        print("\n✅ Training complete!")
        print(f"Best Val Loss (Huber): {self.best_val_loss:.4f}")
        
        # Find best MSE/RMSE at the epoch where Huber was best
        best_epoch_idx = np.argmin(self.history['val_loss'])
        best_mse = self.history['val_mse'][best_epoch_idx]
        best_rmse = self.history['val_rmse'][best_epoch_idx]
        print(f"Best Val MSE:          {best_mse:.4f}")
        print(f"Best Val RMSE:         {best_rmse:.4f}  👈 Compare with baseline 87 RMSE!")
    
    def save_checkpoint(self, epoch, val_huber_loss, val_mse, val_rmse, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_huber_loss': val_huber_loss,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'history': self.history
        }
        
        filename = 'best_model_v2_attention.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        filepath = Config.CHECKPOINT_DIR / filename
        torch.save(checkpoint, filepath)
    
    def save_history(self):
        """Save training history"""
        history_path = Config.CHECKPOINT_DIR / 'training_history_v2_attention.npy'
        np.save(history_path, self.history)
        print(f"✓ Training history saved to {history_path}")
    
    def plot_training_history(self):
        """Generate training visualization with dual metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. Huber Loss curves (used for training)
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss (Huber)', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss (Huber)', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Huber Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss (Huber)', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE curve (from MSE - for comparison!)
        axes[0, 1].plot(epochs, self.history['val_rmse'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=87, color='orange', linestyle='--', linewidth=2, label='Baseline RMSE: 87')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('RMSE (from MSE)', fontsize=12)
        axes[0, 1].set_title('Validation RMSE - Compare with Baseline', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(self.history['val_loss']) + 1
        best_rmse = self.history['val_rmse'][best_epoch - 1]
        axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', linewidth=2,
                           label=f'Best: Epoch {best_epoch}, RMSE={best_rmse:.2f}')
        axes[0, 1].legend(fontsize=10)
        
        # 3. MSE curve (for direct comparison)
        axes[1, 0].plot(epochs, self.history['val_mse'], 'm-', label='Val MSE', linewidth=2)
        axes[1, 0].axhline(y=7600, color='orange', linestyle='--', linewidth=2, label='Baseline MSE: 7600')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('MSE', fontsize=12)
        axes[1, 0].set_title('Validation MSE - Compare with Baseline', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        best_mse = self.history['val_mse'][best_epoch - 1]
        axes[1, 0].axvline(x=best_epoch, color='r', linestyle='--', linewidth=2,
                          label=f'Best: Epoch {best_epoch}, MSE={best_mse:.0f}')
        axes[1, 0].legend(fontsize=10)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        
        improvement_rmse = 87 - best_rmse
        improvement_pct = (improvement_rmse / 87) * 100
        
        summary_text = f"""
    Training Summary (InceptionV3 + Spatial Attention)
    ===================================================
    
    Total Epochs: {len(epochs)}
    
    Best Performance (Epoch {best_epoch}):
    • Val Huber Loss: {self.history['val_loss'][best_epoch-1]:.4f}
    • Val MSE:        {best_mse:.4f}
    • Val RMSE:       {best_rmse:.4f}
    
    Comparison with Baseline:
    • Baseline RMSE:  87.00
    • New RMSE:       {best_rmse:.2f}
    • Improvement:    {improvement_rmse:.2f} ({improvement_pct:.1f}%)
    
    Final Performance:
    • Train Loss:     {self.history['train_loss'][-1]:.4f}
    • Val Huber:      {self.history['val_loss'][-1]:.4f}
    • Val RMSE:       {self.history['val_rmse'][-1]:.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                        verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        save_path = Config.CHECKPOINT_DIR / 'training_results_v2_attention.png'
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
        print("⚠️  Using CPU")
    
    return device


def main():
    """Main training function"""
    # Setup
    Config.validate_paths()
    Config.create_directories()
    
    # Get device
    device = get_device()
    
    # Load data
    print("\n📦 Loading data...")
    train_loader, val_loader = get_dataloaders()
    
    # Print config (now with normalization stats)
    Config.print_config()
    
    print(f"\n✓ Training samples:   {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n🏗️  Creating InceptionV3 + Spatial Attention model...")
    model = InceptionV3WithAttention(
        num_channels=5,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)
    
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