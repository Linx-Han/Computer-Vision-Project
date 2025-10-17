# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from data import get_dataloaders

# ============= 模型定义 =============
class CalorieEstimatorCNN(nn.Module):
    """双流CNN：分别处理RGB和Depth，然后融合"""
    def __init__(self):
        super(CalorieEstimatorCNN, self).__init__()
        
        # RGB流
        self.rgb_stream = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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
            nn.MaxPool2d(2),
            
            # Conv4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Depth流
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
            nn.MaxPool2d(2),
            
            # Conv4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 融合层 + 回归头
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, rgb, depth):
        # 提取特征
        rgb_feat = self.rgb_stream(rgb).flatten(1)      # [batch, 256]
        depth_feat = self.depth_stream(depth).flatten(1) # [batch, 256]
        
        # 融合
        fused = torch.cat([rgb_feat, depth_feat], dim=1) # [batch, 512]
        
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


# ============= 主训练流程 =============
def main():
    # 超参数
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    
    # 路径
    ROOT_DIR = '/Users/hanlinxuan/Desktop/Learning/Unimelb/2025 S2/CV/Assignment/Project/content'
    CSV_FILE = '/Users/hanlinxuan/Desktop/Learning/Unimelb/2025 S2/CV/Assignment/Project/content/comp-90086-nutrition-5-k/Nutrition5K/Nutrition5K/nutrition5k_train.csv'
    
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
        print("✅ 使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU")
        print(f"使用设备: {device}")
        
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
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器 - 移除verbose参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
            }, 'checkpoints/best_model.pth')
            print(f"✓ 保存最佳模型 (Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f})")
        
        print()
    
    # 保存训练历史
    np.save('checkpoints/training_history.npy', history)
    
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