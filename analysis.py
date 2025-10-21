"""
Model analysis and visualization script
Provides insights into model predictions, errors, and learned features
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from config import Config
from data import get_dataloaders, get_test_loader
from model import InceptionResNetV2, SimpleInceptionResNet, get_device


class ModelAnalyzer:
    """Analyze model predictions and behavior"""
    
    def __init__(self, model, val_loader, device):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.model.eval()
    
    def collect_predictions(self):
        """Collect all predictions and ground truth values"""
        predictions = []
        ground_truth = []
        
        print("\n🔍 Collecting predictions...")
        with torch.no_grad():
            for combined, calories in tqdm(self.val_loader, desc='Predicting'):
                combined = combined.to(self.device)
                pred = self.model(combined)
                
                predictions.extend(pred.cpu().numpy())
                ground_truth.extend(calories.numpy())
        
        return np.array(predictions), np.array(ground_truth)
    
    def analyze_predictions(self, save_dir=None):
        """Comprehensive prediction analysis"""
        save_dir = save_dir or Config.CHECKPOINT_DIR
        save_dir = Path(save_dir)
        
        predictions, ground_truth = self.collect_predictions()
        
        # Calculate metrics
        errors = predictions - ground_truth
        abs_errors = np.abs(errors)
        
        # Handle division by zero for percentage errors
        percent_errors = np.where(ground_truth > 0, 
                                 (abs_errors / ground_truth) * 100, 
                                 0)
        
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(abs_errors)
        mape = np.mean(percent_errors[np.isfinite(percent_errors)])
        
        print(f"\n📊 Prediction Metrics:")
        print(f"   RMSE:  {rmse:.2f} calories")
        print(f"   MAE:   {mae:.2f} calories")
        print(f"   MAPE:  {mape:.2f}%")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Prediction vs Ground Truth Scatter
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.scatter(ground_truth, predictions, alpha=0.5, s=20)
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('Ground Truth (calories)', fontsize=12)
        ax1.set_ylabel('Predicted (calories)', fontsize=12)
        ax1.set_title('Predictions vs Ground Truth', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(ground_truth, predictions)[0, 1]
        ax1.text(0.05, 0.95, f'R² = {corr**2:.3f}', transform=ax1.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Error Distribution
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction Error (calories)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        ax2.text(0.05, 0.95, 
                f'Mean: {errors.mean():.2f}\nStd: {errors.std():.2f}',
                transform=ax2.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 3. Absolute Error vs Calorie Range
        ax3 = fig.add_subplot(gs[1, 0:2])
        scatter = ax3.scatter(ground_truth, abs_errors, c=abs_errors, 
                            cmap='YlOrRd', alpha=0.6, s=20)
        ax3.set_xlabel('Ground Truth (calories)', fontsize=12)
        ax3.set_ylabel('Absolute Error (calories)', fontsize=12)
        ax3.set_title('Error by Calorie Range', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Absolute Error')
        ax3.grid(True, alpha=0.3)
        
        # 4. Percentage Error Distribution
        ax4 = fig.add_subplot(gs[1, 2:4])
        # Filter out infinite and very large values for better visualization
        valid_percent_errors = percent_errors[np.isfinite(percent_errors)]
        valid_percent_errors = valid_percent_errors[valid_percent_errors < 200]  # Cap at 200%
        ax4.hist(valid_percent_errors, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax4.set_xlabel('Percentage Error (%)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Best Predictions (lowest error)
        ax5 = fig.add_subplot(gs[2, 0])
        best_indices = np.argsort(abs_errors)[:10]
        best_gt = ground_truth[best_indices]
        best_pred = predictions[best_indices]
        x = np.arange(len(best_indices))
        width = 0.35
        ax5.bar(x - width/2, best_gt, width, label='Ground Truth', alpha=0.8)
        ax5.bar(x + width/2, best_pred, width, label='Predicted', alpha=0.8)
        ax5.set_xlabel('Sample', fontsize=11)
        ax5.set_ylabel('Calories', fontsize=11)
        ax5.set_title('10 Best Predictions', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Worst Predictions (highest error)
        ax6 = fig.add_subplot(gs[2, 1])
        worst_indices = np.argsort(abs_errors)[-10:]
        worst_gt = ground_truth[worst_indices]
        worst_pred = predictions[worst_indices]
        x = np.arange(len(worst_indices))
        ax6.bar(x - width/2, worst_gt, width, label='Ground Truth', alpha=0.8)
        ax6.bar(x + width/2, worst_pred, width, label='Predicted', alpha=0.8)
        ax6.set_xlabel('Sample', fontsize=11)
        ax6.set_ylabel('Calories', fontsize=11)
        ax6.set_title('10 Worst Predictions', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Calorie Range Performance
        ax7 = fig.add_subplot(gs[2, 2])
        bins = [0, 200, 400, 600, 800, 1000, float('inf')]
        labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000+']
        bin_errors = []
        bin_counts = []
        
        for i in range(len(bins)-1):
            mask = (ground_truth >= bins[i]) & (ground_truth < bins[i+1])
            if mask.sum() > 0:
                bin_errors.append(abs_errors[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_errors.append(0)
                bin_counts.append(0)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(bin_errors)))
        bars = ax7.bar(labels, bin_errors, color=colors, alpha=0.8)
        ax7.set_xlabel('Calorie Range', fontsize=11)
        ax7.set_ylabel('Mean Absolute Error', fontsize=11)
        ax7.set_title('Error by Calorie Range', fontsize=12, fontweight='bold')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add sample counts on bars
        for bar, count in zip(bars, bin_counts):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        # 8. Summary Statistics
        ax8 = fig.add_subplot(gs[2, 3])
        ax8.axis('off')
        
        summary_text = f"""
Performance Summary
==================

Overall Metrics:
• RMSE:  {rmse:.2f} cal
• MAE:   {mae:.2f} cal  
• MAPE:  {mape:.2f}%
• R²:    {corr**2:.3f}

Error Statistics:
• Min:   {errors.min():.2f} cal
• Max:   {errors.max():.2f} cal
• Std:   {errors.std():.2f} cal

Dataset Info:
• Samples: {len(predictions)}
• Cal Range: {ground_truth.min():.0f}-{ground_truth.max():.0f}

Within Tolerance:
• ±50 cal:  {(abs_errors <= 50).sum()/len(abs_errors)*100:.1f}%
• ±100 cal: {(abs_errors <= 100).sum()/len(abs_errors)*100:.1f}%
• ±150 cal: {(abs_errors <= 150).sum()/len(abs_errors)*100:.1f}%
        """
        
        ax8.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('Model Prediction Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        save_path = save_dir / 'prediction_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Analysis saved to {save_path}")
        plt.close()
        
        return predictions, ground_truth, errors
    
    def visualize_channel_importance(self, save_dir=None):
        """Analyze the importance of different input channels"""
        save_dir = save_dir or Config.CHECKPOINT_DIR
        save_dir = Path(save_dir)
        
        print("\n🔬 Analyzing channel importance...")
        
        # Get a batch of data
        combined, calories = next(iter(self.val_loader))
        combined = combined.to(self.device)
        calories = calories.to(self.device)
        
        # Original prediction
        with torch.no_grad():
            original_pred = self.model(combined)
        
        # Test each channel by zeroing it out
        channel_names = ['R', 'G', 'B', 'Depth', 'Height']
        channel_impacts = []
        
        for i in range(5):
            combined_modified = combined.clone()
            combined_modified[:, i, :, :] = 0
            
            with torch.no_grad():
                modified_pred = self.model(combined_modified)
            
            # Calculate impact (change in prediction)
            impact = torch.abs(original_pred - modified_pred).mean().item()
            channel_impacts.append(impact)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        bars = ax1.bar(channel_names, channel_impacts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Average Impact on Prediction (calories)', fontsize=12)
        ax1.set_title('Channel Importance (Ablation Study)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, impact in zip(bars, channel_impacts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{impact:.1f}', ha='center', va='bottom', fontsize=11)
        
        # Pie chart
        ax2.pie(channel_impacts, labels=channel_names, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11})
        ax2.set_title('Relative Channel Contribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = save_dir / 'channel_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Channel analysis saved to {save_path}")
        plt.close()
        
        # Print results
        print(f"\n📊 Channel Importance Results:")
        for name, impact in zip(channel_names, channel_impacts):
            print(f"   {name:8s}: {impact:.2f} calories impact")
        
        return channel_impacts


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
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"   Val RMSE: {checkpoint['val_rmse']:.4f}")
    
    return model


def main():
    """Run model analysis"""
    # Setup
    Config.validate_paths()
    Config.create_directories()
    
    device = get_device()
    
    # Load validation data
    print("\n📦 Loading validation data...")
    _, val_loader = get_dataloaders(
        use_relative_depth=True,
        depth_method='percentile',
        plate_percentile=10
    )
    
    # Load model (change this to match your trained model)
    print("\n🏗️ Loading trained model...")
    # Use the same model architecture you trained with
    model = load_best_model(InceptionResNetV2, device)  # or SimpleInceptionResNet
    
    # Create analyzer
    analyzer = ModelAnalyzer(model, val_loader, device)
    
    # Run analyses
    print("\n" + "="*60)
    print("Running Model Analysis")
    print("="*60)
    
    # 1. Prediction analysis
    predictions, ground_truth, errors = analyzer.analyze_predictions()
    
    # 2. Channel importance analysis
    channel_impacts = analyzer.visualize_channel_importance()
    
    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)
    print(f"\nResults saved to: {Config.CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()