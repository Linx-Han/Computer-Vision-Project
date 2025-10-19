# train_autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from autoencoder import ImageAutoencoder
from data import validate_dataset, Nutrition5kDataset

def train_autoencoder():
    """步骤2: 训练Autoencoder学习图像特征"""
    
    print("=" * 60)
    print("步骤2: 训练Autoencoder")
    print("=" * 60)
    
    # 超参数
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    EMBEDDING_SIZE = 64
    
    ROOT_DIR = '/Users/hanlinxuan/Desktop/Learning/Unimelb/2025 S2/CV/Assignment/Project/content'
    CSV_FILE = '/Users/hanlinxuan/Desktop/Learning/Unimelb/2025 S2/CV/Assignment/Project/content/comp-90086-nutrition-5-k/Nutrition5K/Nutrition5K/nutrition5k_train.csv'
    
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU")
    
    # 创建数据集
    print("\n加载数据...")
    valid_df = validate_dataset(ROOT_DIR, CSV_FILE)
    
    n_val = int(len(valid_df) * 0.2)
    np.random.seed(42)
    indices = np.random.permutation(len(valid_df))
    
    train_df = valid_df.iloc[indices[n_val:]].reset_index(drop=True)
    train_df.to_csv('train_split.csv', index=False)
    
    train_dataset = Nutrition5kDataset(ROOT_DIR, 'train_split.csv', is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"训练样本数: {len(train_dataset)}")
    
    # 创建RGB和Depth的Autoencoder
    print("\n创建Autoencoder模型...")
    rgb_autoencoder = ImageAutoencoder(in_channels=3, embedding_size=EMBEDDING_SIZE).to(device)
    depth_autoencoder = ImageAutoencoder(in_channels=1, embedding_size=EMBEDDING_SIZE).to(device)
    
    total_params_rgb = sum(p.numel() for p in rgb_autoencoder.parameters())
    total_params_depth = sum(p.numel() for p in depth_autoencoder.parameters())
    print(f"RGB Autoencoder参数量: {total_params_rgb:,}")
    print(f"Depth Autoencoder参数量: {total_params_depth:,}")
    
    # 损失函数和优化器 - 使用AdamW
    criterion = nn.MSELoss()
    rgb_optimizer = optim.AdamW(rgb_autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    depth_optimizer = optim.AdamW(depth_autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练
    print("\n开始训练Autoencoder...\n")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        rgb_autoencoder.train()
        depth_autoencoder.train()
        
        total_rgb_loss = 0
        total_depth_loss = 0
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        for rgb, depth, _ in tqdm(train_loader, desc='Training Autoencoder'):
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            # 训练RGB Autoencoder
            rgb_recon, _ = rgb_autoencoder(rgb)
            rgb_loss = criterion(rgb_recon, rgb)
            
            rgb_optimizer.zero_grad()
            rgb_loss.backward()
            rgb_optimizer.step()
            
            # 训练Depth Autoencoder
            depth_recon, _ = depth_autoencoder(depth)
            depth_loss = criterion(depth_recon, depth)
            
            depth_optimizer.zero_grad()
            depth_loss.backward()
            depth_optimizer.step()
            
            total_rgb_loss += rgb_loss.item()
            total_depth_loss += depth_loss.item()
        
        avg_rgb_loss = total_rgb_loss / len(train_loader)
        avg_depth_loss = total_depth_loss / len(train_loader)
        avg_loss = (avg_rgb_loss + avg_depth_loss) / 2
        
        print(f"RGB Loss: {avg_rgb_loss:.4f}")
        print(f"Depth Loss: {avg_depth_loss:.4f}")
        print(f"Avg Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'rgb_encoder': rgb_autoencoder.encoder.state_dict(),
                'depth_encoder': depth_autoencoder.encoder.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, 'checkpoints/autoencoder_best.pth')
            print(f"✓ 保存最佳Autoencoder (Loss: {avg_loss:.4f})")
        
        print()
    
    print("\n✅ Autoencoder训练完成！")
    print(f"最佳Loss: {best_loss:.4f}")
    print("编码器权重已保存到: checkpoints/autoencoder_best.pth")
    print("\n下一步: 运行 model.py 训练CNN")


if __name__ == '__main__':
    train_autoencoder()