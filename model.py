"""
Model definition and training for Nutrition5k calorie estimation
Uses Inception-ResNet architecture with 5-channel input
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from data import get_dataloaders


class InceptionModule(nn.Module):
    """Inception module with multiple kernel sizes"""
    
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.BatchNorm2d(reduce_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 convolution branch (using two 3x3 for efficiency)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.BatchNorm2d(reduce_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class InceptionResNetV2(nn.Module):
    """
    Inception-ResNet architecture for 5-channel input
    Combines Inception modules with ResNet skip connections
    """
    
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(InceptionResNetV2, self).__init__()
        
        # Stem: Initial convolutions to process 5-channel input
        self.stem = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception-ResNet-A blocks (3 blocks)
        self.inception_resnet_a1 = self._make_inception_resnet_a(64, scale=0.17)
        self.inception_resnet_a2 = self._make_inception_resnet_a(256, scale=0.17)
        self.inception_resnet_a3 = self._make_inception_resnet_a(256, scale=0.17)
        
        # Reduction-A
        self.reduction_a = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        
        # Inception-ResNet-B blocks (3 blocks)
        self.inception_resnet_b1 = self._make_inception_resnet_b(384, scale=0.10)
        self.inception_resnet_b2 = self._make_inception_resnet_b(896, scale=0.10)
        self.inception_resnet_b3 = self._make_inception_resnet_b(896, scale=0.10)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(896, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def _make_inception_resnet_a(self, in_channels, scale=1.0):

    
        # Branch 1: 1x1 conv
        branch1 = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)
        
        # Branch 2: 1x1 -> 3x3
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        
        # Branch 3: 1x1 -> 3x3 -> 3x3
        branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1)
        )
        
        # Concatenate: 32 + 32 + 64 = 128
        conv_out_channels = 128
        
        # 1x1 conv to match input channels
        up = nn.Conv2d(conv_out_channels, in_channels if in_channels != 64 else 256, 
                    kernel_size=1, padding=0)
        
        # Combine into module
        class InceptionResNetA(nn.Module):
            def __init__(self, b1, b2, b3, up_conv, scale, in_ch):
                super().__init__()
                self.branch1 = b1
                self.branch2 = b2
                self.branch3 = b3
                self.up = up_conv
                self.scale = scale
                self.relu = nn.ReLU(inplace=True)
                
                # Adjust input if needed
                self.need_adjust = (in_ch == 64)
                if self.need_adjust:
                    self.adjust = nn.Conv2d(64, 256, kernel_size=1)
            
            def forward(self, x):
                identity = self.adjust(x) if self.need_adjust else x
                
                b1 = self.branch1(x)
                b2 = self.branch2(x)
                b3 = self.branch3(x)
                
                mixed = torch.cat([b1, b2, b3], dim=1)
                up = self.up(mixed)
                
                out = identity + self.scale * up
                out = self.relu(out)
                
                return out
        
        return InceptionResNetA(branch1, branch2, branch3, up, scale, in_channels)
        
    def _make_inception_resnet_b(self, in_channels, scale=1.0):
        """Create Inception-ResNet-B block"""
        
        # Branch 1: 1x1 conv
        branch1 = nn.Conv2d(in_channels, 192, kernel_size=1, padding=0)
        
        # Branch 2: 1x1 -> 1x7 -> 7x1
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 192, kernel_size=(7, 1), padding=(3, 0))
        )
        
        # Concatenate: 192 + 192 = 384
        conv_out_channels = 384
        
        # 1x1 conv to match input channels
        up = nn.Conv2d(conv_out_channels, in_channels if in_channels != 384 else 896,
                    kernel_size=1, padding=0)
        
        class InceptionResNetB(nn.Module):
            def __init__(self, b1, b2, up_conv, scale, in_ch):
                super().__init__()
                self.branch1 = b1
                self.branch2 = b2
                self.up = up_conv
                self.scale = scale
                self.relu = nn.ReLU(inplace=True)
                
                # Adjust input if needed
                self.need_adjust = (in_ch == 384)
                if self.need_adjust:
                    self.adjust = nn.Conv2d(384, 896, kernel_size=1)
            
            def forward(self, x):
                identity = self.adjust(x) if self.need_adjust else x
                
                b1 = self.branch1(x)
                b2 = self.branch2(x)
                
                mixed = torch.cat([b1, b2], dim=1)
                up = self.up(mixed)
                
                out = identity + self.scale * up
                out = self.relu(out)
                
                return out
        
        return InceptionResNetB(branch1, branch2, up, scale, in_channels)
    
    def forward(self, x):
        """
        Args:
            x: 5-channel input [batch, 5, H, W]
        
        Returns:
            calories: Predicted calories [batch]
        """
        # Stem
        x = self.stem(x)
        
        # Inception-ResNet-A blocks
        x = self.inception_resnet_a1(x)
        x = self.inception_resnet_a2(x)
        x = self.inception_resnet_a3(x)
        
        # Reduction-A
        x = self.reduction_a(x)
        
        # Inception-ResNet-B blocks
        x = self.inception_resnet_b1(x)
        x = self.inception_resnet_b2(x)
        x = self.inception_resnet_b3(x)
        
        # Global pooling
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        
        # Regression
        x = self.regressor(x)
        
        return x.squeeze(1)


class SimpleInceptionResNet(nn.Module):
    """
    Simplified Inception-ResNet for faster training
    Good balance between performance and efficiency
    """
    
    def __init__(self, dropout_rate=0.5):
        super(SimpleInceptionResNet, self).__init__()
        
        # Initial convolution for 5 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception module 1
        self.inception1 = InceptionModule(64, 64, 96, 128, 16, 32, 32)  # Output: 256
        
        # Residual blocks 1
        self.res1 = ResidualBlock(256, 256)
        self.res2 = ResidualBlock(256, 256)
        
        # Downsample
        self.downsample1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Inception module 2
        self.inception2 = InceptionModule(512, 128, 128, 192, 32, 96, 64)  # Output: 480
        
        # Residual blocks 2
        self.res3 = ResidualBlock(480, 512, stride=2)
        self.res4 = ResidualBlock(512, 512)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: 5-channel input [batch, 5, H, W]
        
        Returns:
            calories: Predicted calories [batch]
        """
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.downsample1(x)
        x = self.inception2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.regressor(x)
        
        return x.squeeze(1)


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
        
        for combined, calories in tqdm(self.train_loader, desc='Training'):
            combined = combined.to(self.device)  # 5-channel input
            calories = calories.to(self.device)
            
            # Forward pass
            pred_calories = self.model(combined)
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
            for combined, calories in tqdm(self.val_loader, desc='Validation'):
                combined = combined.to(self.device)  # 5-channel input
                calories = calories.to(self.device)
                
                pred_calories = self.model(combined)
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
                print(f"⚠️  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_rmse, is_best=True)
                print(f"✓ Best model saved (Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f})")
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
        print("⚠️  Using CPU")
    
    return device


def main():
    """Main training function"""
    # Setup
    Config.validate_paths()
    Config.create_directories()
    Config.print_config()
    
    # Get device
    device = get_device()
    
    # Load data with 5-channel output
    print("\n📦 Loading data with 5-channel tensors...")
    train_loader, val_loader = get_dataloaders(
        use_relative_depth=True,
        depth_method='percentile',
        plate_percentile=10
    )
    print(f"✓ Training samples:   {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    
    # Test data shape
    sample_batch, sample_calories = next(iter(train_loader))
    print(f"✓ Input shape: {sample_batch.shape}")  # Should be (batch, 5, H, W)
    print(f"  - Channels: RGB(3) + Depth(1) + Height(1) = 5")
    
    # Create model
    print("\n🏗️  Creating Inception-ResNet model...")
    
    # Choose model architecture
    model = InceptionResNetV2(dropout_rate=Config.DROPOUT_RATE).to(device)  # Full version
    # model = SimpleInceptionResNet(dropout_rate=Config.DROPOUT_RATE).to(device)  # Simpler version (recommended)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters:     {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    with torch.no_grad():
        test_output = model(sample_batch[:2].to(device))
        print(f"✓ Output shape: {test_output.shape}")  # Should be (2,)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()