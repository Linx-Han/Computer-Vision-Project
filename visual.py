"""
Visualize learned convolutional filters from the trained model
Shows what features the model has learned to detect
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from model import InceptionResNetV2, SimpleInceptionResNet, get_device


class FilterVisualizer:
    """Visualize convolutional filters from trained model"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_first_layer_filters(self):
        """Extract filters from the first convolutional layer"""
        # Get the first conv layer (processes 5-channel input)
        first_conv = None
        
        # Navigate to stem or conv1 depending on model architecture
        if hasattr(self.model, 'stem'):
            # InceptionResNetV2
            first_conv = self.model.stem[0]  # First conv in stem
        elif hasattr(self.model, 'conv1'):
            # SimpleInceptionResNet
            first_conv = self.model.conv1[0]  # First conv in conv1
        
        if first_conv is None:
            raise ValueError("Could not find first convolutional layer")
        
        # Get weights: shape [out_channels, in_channels, kernel_h, kernel_w]
        filters = first_conv.weight.data.cpu()
        
        return filters
    
    def visualize_first_layer_filters(self, save_dir=None):
        """
        Visualize filters from first layer
        Shows what each filter detects from the 5 input channels
        """
        save_dir = save_dir or Config.CHECKPOINT_DIR
        save_dir = Path(save_dir)
        
        print("\n🔍 Extracting first layer filters...")
        filters = self.get_first_layer_filters()
        
        out_channels, in_channels, kh, kw = filters.shape
        print(f"   Filter shape: {filters.shape}")
        print(f"   {out_channels} filters, {in_channels} input channels, {kh}x{kw} kernel")
        
        # Select a subset of filters to visualize
        num_filters_to_show = min(32, out_channels)
        
        # Create figure
        fig, axes = plt.subplots(num_filters_to_show, in_channels, 
                                figsize=(in_channels * 2, num_filters_to_show * 2))
        
        if num_filters_to_show == 1:
            axes = axes.reshape(1, -1)
        
        channel_names = ['Red', 'Green', 'Blue', 'Depth', 'Height']
        
        for filter_idx in range(num_filters_to_show):
            for channel_idx in range(in_channels):
                ax = axes[filter_idx, channel_idx]
                
                # Get filter for this channel
                filter_slice = filters[filter_idx, channel_idx, :, :].numpy()
                
                # Normalize for visualization
                vmin, vmax = filter_slice.min(), filter_slice.max()
                if vmax - vmin > 0:
                    filter_normalized = (filter_slice - vmin) / (vmax - vmin)
                else:
                    filter_normalized = filter_slice
                
                # Plot
                im = ax.imshow(filter_normalized, cmap='RdBu_r', aspect='auto')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add labels
                if filter_idx == 0:
                    ax.set_title(channel_names[channel_idx], fontsize=10, fontweight='bold')
                if channel_idx == 0:
                    ax.set_ylabel(f'Filter {filter_idx}', fontsize=9)
                
                # Add colorbar for first row
                if filter_idx == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'First Layer Convolutional Filters\n({num_filters_to_show} filters × {in_channels} input channels)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / 'first_layer_filters.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Filter visualization saved to {save_path}")
        plt.close()
    
    def visualize_filter_statistics(self, save_dir=None):
        """Analyze and visualize filter statistics across channels"""
        save_dir = save_dir or Config.CHECKPOINT_DIR
        save_dir = Path(save_dir)
        
        print("\n📊 Analyzing filter statistics...")
        filters = self.get_first_layer_filters()
        
        out_channels, in_channels, kh, kw = filters.shape
        channel_names = ['Red', 'Green', 'Blue', 'Depth', 'Height']
        
        # Calculate statistics per channel
        channel_means = []
        channel_stds = []
        channel_l2_norms = []
        
        for channel_idx in range(in_channels):
            channel_filters = filters[:, channel_idx, :, :]
            channel_means.append(channel_filters.mean().item())
            channel_stds.append(channel_filters.std().item())
            channel_l2_norms.append(torch.norm(channel_filters).item())
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Mean weights per channel
        ax1 = axes[0, 0]
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        bars1 = ax1.bar(channel_names, channel_means, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean Filter Weight', fontsize=12)
        ax1.set_title('Average Filter Weights by Channel', fontsize=13, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, channel_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        
        # 2. Standard deviation per channel
        ax2 = axes[0, 1]
        bars2 = ax2.bar(channel_names, channel_stds, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Standard Deviation', fontsize=12)
        ax2.set_title('Filter Weight Variability by Channel', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, channel_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. L2 norm per channel (magnitude of filters)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(channel_names, channel_l2_norms, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('L2 Norm', fontsize=12)
        ax3.set_title('Filter Magnitude by Channel\n(Higher = Model pays more attention)', 
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars3, channel_l2_norms):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Distribution of all weights by channel
        ax4 = axes[1, 1]
        for channel_idx in range(in_channels):
            channel_filters = filters[:, channel_idx, :, :].flatten().numpy()
            ax4.hist(channel_filters, bins=50, alpha=0.5, label=channel_names[channel_idx],
                    color=colors[channel_idx], edgecolor='none')
        
        ax4.set_xlabel('Filter Weight Value', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Weight Distribution by Channel', fontsize=13, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = save_dir / 'filter_statistics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Filter statistics saved to {save_path}")
        plt.close()
        
        # Print statistics
        print(f"\n📈 Filter Statistics by Channel:")
        print(f"{'Channel':<10} {'Mean':<12} {'Std':<12} {'L2 Norm':<12}")
        print("-" * 50)
        for i, name in enumerate(channel_names):
            print(f"{name:<10} {channel_means[i]:>10.4f}  {channel_stds[i]:>10.4f}  {channel_l2_norms[i]:>10.2f}")
    
    def visualize_channel_filters_detailed(self, num_filters=8, save_dir=None):
        """
        Show detailed view of selected filters across all channels
        Good for understanding what patterns each filter detects
        """
        save_dir = save_dir or Config.CHECKPOINT_DIR
        save_dir = Path(save_dir)
        
        print(f"\n🔬 Creating detailed view of {num_filters} filters...")
        filters = self.get_first_layer_filters()
        
        out_channels, in_channels, kh, kw = filters.shape
        channel_names = ['Red', 'Green', 'Blue', 'Depth', 'Height']
        
        # Select diverse filters (evenly spaced)
        filter_indices = np.linspace(0, out_channels-1, num_filters, dtype=int)
        
        fig = plt.figure(figsize=(16, num_filters * 2.5))
        
        for i, filter_idx in enumerate(filter_indices):
            # Create subplot for this filter
            for channel_idx in range(in_channels):
                ax = plt.subplot(num_filters, in_channels, i * in_channels + channel_idx + 1)
                
                filter_slice = filters[filter_idx, channel_idx, :, :].numpy()
                
                # Use diverging colormap centered at zero
                vmax = max(abs(filter_slice.min()), abs(filter_slice.max()))
                
                im = ax.imshow(filter_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax, 
                              aspect='auto', interpolation='nearest')
                
                # Add grid to show individual weights
                if kh <= 7 and kw <= 7:
                    for x in range(kw + 1):
                        ax.axvline(x - 0.5, color='gray', linewidth=0.5)
                    for y in range(kh + 1):
                        ax.axhline(y - 0.5, color='gray', linewidth=0.5)
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Labels
                if i == 0:
                    ax.set_title(channel_names[channel_idx], fontsize=11, fontweight='bold')
                if channel_idx == 0:
                    ax.set_ylabel(f'Filter {filter_idx}\n{kh}×{kw}', fontsize=10)
                
                # Show colorbar for last column
                if channel_idx == in_channels - 1:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
        
        plt.suptitle(f'Detailed Filter View: {num_filters} Selected Filters Across All Channels', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / 'detailed_filters.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Detailed filter view saved to {save_path}")
        plt.close()


def load_best_model(model_class, device):
    """Load the best saved model"""
    checkpoint_path = Config.CHECKPOINT_DIR / 'best_model.pth'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    print(f"\n📂 Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = model_class(dropout_rate=Config.DROPOUT_RATE).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    
    return model


def main():
    """Run filter visualization"""
    # Setup
    Config.validate_paths()
    Config.create_directories()
    
    device = get_device()
    
    # Load model (change to match your trained model)
    print("\n🏗️ Loading trained model...")
    model = load_best_model(InceptionResNetV2, device)  # or SimpleInceptionResNet
    
    # Create visualizer
    visualizer = FilterVisualizer(model, device)
    
    print("\n" + "="*60)
    print("Visualizing Learned Filters")
    print("="*60)
    
    # 1. Show first layer filters
    visualizer.visualize_first_layer_filters()
    
    # 2. Show filter statistics
    visualizer.visualize_filter_statistics()
    
    # 3. Show detailed view
    visualizer.visualize_channel_filters_detailed(num_filters=8)
    
    print("\n" + "="*60)
    print("✅ Filter Visualization Complete!")
    print("="*60)
    print(f"\nResults saved to: {Config.CHECKPOINT_DIR}")
    print("\nWhat to look for:")
    print("  • Edge detectors: Filters with strong positive/negative transitions")
    print("  • Color detectors: Strong weights in specific RGB channels")
    print("  • Depth/Height usage: Non-zero weights in channels 4 & 5")
    print("  • Balanced learning: Similar L2 norms across channels")


if __name__ == '__main__':
    main()