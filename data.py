# data.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class Nutrition5kDataset(Dataset):
    def __init__(self, root_dir, csv_file, is_train=True):
        """
        Args:
            root_dir: content文件夹路径
            csv_file: CSV文件路径
            is_train: 训练模式(True)或验证模式(False)
        """
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_file)
        self.is_train = is_train
        
        # 构建基础路径
        # / 'comp-90086-nutrition-5-k' / 'Nutrition5K' / 'Nutrition5K'
        base_path = self.root_dir 
        
        # RGB和Depth目录
        self.color_dir = base_path / 'train' / 'color'
        self.depth_dir = base_path / 'train' / 'depth_raw'
        
        # 过滤掉损坏的图像
        print("检查数据完整性...")
        valid_indices = []
        for idx in range(len(self.df)):
            dish_id = self.df.iloc[idx, 0]
            rgb_path = self.color_dir / dish_id / 'rgb.png'
            depth_path = self.depth_dir / dish_id / 'depth_raw.png'
            
            try:
                # 尝试打开图像
                if rgb_path.exists() and depth_path.exists():
                    Image.open(rgb_path).convert('RGB')
                    Image.open(depth_path)
                    valid_indices.append(idx)
            except Exception as e:
                print(f"  跳过损坏的样本: {dish_id} - {str(e)}")
        
        # 只保留有效样本
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"✓ 有效样本数: {len(self.df)} / {len(valid_indices) + (len(self.df.index) - len(valid_indices))}")
        
        # 图像变换
        if is_train:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取dish_id和calories
        row = self.df.iloc[idx]
        dish_id = row.iloc[0]
        calories = row.iloc[1]
        
        # 构建图像路径
        rgb_path = self.color_dir / dish_id / 'rgb.png'
        depth_path = self.depth_dir / dish_id / 'depth_raw.png'
        
        try:
            # 读取图像
            rgb_img = Image.open(rgb_path).convert('RGB')
            depth_img = Image.open(depth_path)
            
            # 处理深度图
            if depth_img.mode != 'L':
                depth_img = depth_img.convert('L')
            
            # 应用变换
            rgb_img = self.rgb_transform(rgb_img)
            depth_img = self.depth_transform(depth_img)
            
            # 归一化深度图到[0,1]
            depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
            
            return rgb_img, depth_img, torch.tensor(calories, dtype=torch.float32)
        
        except Exception as e:
            print(f"\n错误: 无法加载 {dish_id}: {str(e)}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self))


# 创建数据加载器
def get_dataloaders(root_dir, csv_file, batch_size=16, val_split=0.2):
    """
    创建训练和验证数据加载器
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 划分训练集和验证集
    n_val = int(len(df) * val_split)
    indices = np.random.permutation(len(df))
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    # 创建训练和验证CSV
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    # 保存临时CSV
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    
    # 创建数据集
    train_dataset = Nutrition5kDataset(root_dir, 'train_split.csv', is_train=True)
    val_dataset = Nutrition5kDataset(root_dir, 'val_split.csv', is_train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader


# 测试数据集类（用于Kaggle提交）
class Nutrition5kTestDataset(Dataset):
    def __init__(self, root_dir):
        """测试集数据加载器"""
        self.root_dir = Path(root_dir)
        base_path = self.root_dir / 'comp-90086-nutrition-5-k' / 'Nutrition5K' / 'Nutrition5K'
        
        self.color_dir = base_path / 'test' / 'color'
        self.depth_dir = base_path / 'test' / 'depth_raw'
        
        # 获取所有dish_id
        self.dish_ids = sorted([d.name for d in self.color_dir.iterdir() if d.is_dir()])
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.dish_ids)
    
    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        
        rgb_path = self.color_dir / dish_id / 'rgb.png'
        depth_path = self.depth_dir / dish_id / 'depth_raw.png'
        
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('L')
        
        rgb_img = self.rgb_transform(rgb_img)
        depth_img = self.depth_transform(depth_img)
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
        
        return rgb_img, depth_img, dish_id


# ============= 使用示例 =============
if __name__ == '__main__':
    # 设置路径
    ROOT_DIR = '/Users/apple/Code/Computer-Vision-Project'
    CSV_FILE = '/Users/apple/Code/Computer-Vision-Project/data/nutrition5k_train.csv'
    
    # 创建数据加载器
    train_loader, val_loader = get_dataloaders(
        root_dir=ROOT_DIR,
        csv_file=CSV_FILE,
        batch_size=16,
        val_split=0.2
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 测试加载一个batch
    rgb, depth, calories = next(iter(train_loader))
    print(f"\nRGB shape: {rgb.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Calories shape: {calories.shape}")
    print(f"Calories 范围: [{calories.min():.2f}, {calories.max():.2f}]")
    
    print("\n✅ 数据预处理完成！")