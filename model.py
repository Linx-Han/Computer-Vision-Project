# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from data import get_dataloaders

# ============= 轻量版模型定义 =============
class CalorieEstimatorCNN(nn.Module):
    """双流CNN：分别处理RGB和Depth，然后融合"""
    def __init__(self):
        super(CalorieEstimatorCNN, self).__init__()
        
        # RGB流 - 只用3层卷积
        self.rgb_stream = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            
            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            
            # Conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )
        
        # Depth流 - 只用3层卷积
        self.depth_stream = nn.Sequential(
            # Conv1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 融合层 + 回归头 - 更简单
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),  # 128 + 128 = 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, rgb, depth):
        # 提取特征
        rgb_feat = self.rgb_stream(rgb).flatten(1)      # [batch, 128]
        depth_feat = self.depth_stream(depth).flatten(1) # [batch, 128]
        
        # 融合
        fused = torch.cat([rgb_feat, depth_feat], dim=1) # [batch, 256]
        
        # 回归
        calories = self.fusion(fused).squeeze(1)         # [batch]
        
        return calories


# ============= 训练函数 =============
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for rgb, depth, calories in tqdm(train_loader, desc='Training'):
        rgb = rgb.to(device)
        depth = depth.to(device)
        calories = calories.to(device)
        
        # 前向传播
        pred_calories = model(rgb, depth)
        loss = criterion(pred_calories, calories)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for rgb, depth, calories in tqdm(val_loader, desc='Validation'):
            rgb = rgb.to(device)
            depth = depth.to(device)
            calories = calories.to(device)
            
            pred_calories = model(rgb, depth)
            loss = criterion(pred_calories, calories)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def plot_training_history(history, save_path='training_results.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE曲线
    axes[0, 1].plot(epochs, history['val_rmse'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Validation RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 找到最佳epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_rmse = history['val_rmse'][best_epoch - 1]
    axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', linewidth=2, 
                       label=f'Best: Epoch {best_epoch}, RMSE={best_rmse:.2f}')
    axes[0, 1].legend(fontsize=10)
    
    # 3. Train vs Val Loss对比
    axes[1, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].fill_between(epochs, history['train_loss'], history['val_loss'], 
                            alpha=0.3, color='gray', label='Gap')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Overfitting Check (Train-Val Gap)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计摘要
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    ================
    
    Total Epochs: {len(epochs)}
    
    Best Performance:
    • Epoch: {best_epoch}
    • Val Loss: {history['val_loss'][best_epoch-1]:.4f}
    • Val RMSE: {best_rmse:.4f}
    
    Final Performance:
    • Train Loss: {history['train_loss'][-1]:.4f}
    • Val Loss: {history['val_loss'][-1]:.4f}
    • Val RMSE: {history['val_rmse'][-1]:.4f}
    
    Improvement:
    • Initial RMSE: {history['val_rmse'][0]:.4f}
    • Best RMSE: {best_rmse:.4f}
    • Reduction: {history['val_rmse'][0] - best_rmse:.4f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 训练曲线已保存: {save_path}")
    plt.close()


# ============= 主训练流程 =============
def main():
    # 超参数 - 针对小数据集优化
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001  
    VAL_SPLIT = 0.2
    
    # 路径
    ROOT_DIR = os.getenv('DATA_ROOT_DIR', './data')
    CSV_FILE = os.getenv('TRAIN_CSV_FILE', './data/nutrition5k_train.csv')
    CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')
    print(f"📁 数据根目录: {ROOT_DIR}")
    print(f"📄 训练CSV文件: {CSV_FILE}")
    print(f"💾 检查点目录: {CHECKPOINT_DIR}")
    
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU")
    
    print(f"设备: {device}")
    
    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 数据加载
    print("\n加载数据...")
    train_loader, val_loader = get_dataloaders(
        root_dir=ROOT_DIR,
        csv_file=CSV_FILE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model = CalorieEstimatorCNN().to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': []
    }
    
    # 训练
    print("\n开始训练...\n")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_rmse = np.sqrt(val_loss)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 调整学习率
        old_lr = current_lr
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val RMSE: {val_rmse:.4f}")
        print(f"Learning Rate: {new_lr:.6f}")
        
        # 如果学习率改变了，打印提示
        if new_lr < old_lr:
            print(f"⚠ 学习率降低: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
            }, 'checkpoints/best_model.pth')
            print(f"✓ 保存最佳模型 (Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⚠ 早停触发！{early_stop_patience} 个epoch没有改善")
                break
        
        print()
    
    # 保存训练历史
    np.save('checkpoints/training_history.npy', history)
    
    # 绘制训练曲线
    print("\n生成训练可视化...")
    plot_training_history(history, save_path='checkpoints/training_results.png')
    
    print("\n训练完成！")
    print(f"最佳验证Loss: {best_val_loss:.4f}")
    print(f"最佳验证RMSE: {np.sqrt(best_val_loss):.4f}")
    
    # 打印最后几个epoch的结果
    print("\n最后5个epoch:")
    for i in range(max(0, len(history['val_rmse'])-5), len(history['val_rmse'])):
        print(f"  Epoch {i+1}: Train Loss={history['train_loss'][i]:.4f}, "
              f"Val Loss={history['val_loss'][i]:.4f}, "
              f"Val RMSE={history['val_rmse'][i]:.4f}")


if __name__ == '__main__':
    main()