# data.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pickle
from segmentation import UNet

class Nutrition5kDataset(Dataset):
    def __init__(self, root_dir, csv_file, is_train=True, use_segmentation=True, unet_path=None):
        """
        Args:
            root_dir: content文件夹路径
            csv_file: CSV文件路径
            is_train: 训练模式(True)或验证模式(False)
            use_segmentation: 是否使用U-Net语义分割
            unet_path: U-Net模型路径
        """
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_file)
        self.is_train = is_train
        self.use_segmentation = use_segmentation
        
        # 构建基础路径
        base_path = self.root_dir / 'comp-90086-nutrition-5-k' / 'Nutrition5K' / 'Nutrition5K'
        
        # RGB和Depth目录
        self.color_dir = base_path / 'train' / 'color'
        self.depth_dir = base_path / 'train' / 'depth_raw'
        
        # 加载U-Net模型
        self.unet = None
        if use_segmentation and unet_path and os.path.exists(unet_path):
            print(f"✓ 加载U-Net模型: {unet_path}")
            self.unet = UNet(in_channels=1, out_channels=1)
            checkpoint = torch.load(unet_path, map_location='cpu', weights_only=False)
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            self.unet.eval()
        elif use_segmentation:
            print("⚠️ U-Net模型未找到，使用简单阈值分割")
        
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
    
    def segment_with_unet(self, depth_tensor):
        """使用U-Net生成mask"""
        with torch.no_grad():
            # depth_tensor: [1, 224, 224]
            mask = self.unet(depth_tensor.unsqueeze(0))  # [1, 1, 224, 224]
            mask = mask.squeeze(0)  # [1, 224, 224]
        return mask
    
    def simple_segment(self, depth_img):
        """简单阈值分割（备用）"""
        depth_array = np.array(depth_img, dtype=np.float32)
        
        if depth_array.max() > depth_array.min():
            depth_norm = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        else:
            depth_norm = depth_array
        
        threshold = 0.5
        mask = (depth_norm < threshold).astype(np.float32)
        
        return torch.from_numpy(mask).unsqueeze(0)  # [1, 224, 224]
    
    def apply_mask_tensor(self, img_tensor, mask_tensor):
        """将mask应用到tensor图像"""
        # img_tensor: [C, H, W], mask_tensor: [1, H, W]
        return img_tensor * mask_tensor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row.iloc[0]
        calories = row.iloc[1]
        
        rgb_path = self.color_dir / dish_id / 'rgb.png'
        depth_path = self.depth_dir / dish_id / 'depth_raw.png'
        
        try:
            # 读取图像
            rgb_img = Image.open(rgb_path).convert('RGB')
            depth_img = Image.open(depth_path)
            
            if depth_img.mode != 'L':
                depth_img = depth_img.convert('L')
            
            # 应用变换
            rgb_tensor = self.rgb_transform(rgb_img)
            depth_tensor = self.depth_transform(depth_img)
            
            # 归一化深度图
            depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)
            
            # 生成mask
            if self.use_segmentation:
                if self.unet is not None:
                    # 使用U-Net
                    mask = self.segment_with_unet(depth_tensor)
                else:
                    # 使用简单分割
                    mask = self.simple_segment(depth_img)
                
                # 应用mask
                rgb_tensor = self.apply_mask_tensor(rgb_tensor, mask)
                depth_tensor = self.apply_mask_tensor(depth_tensor, mask)
            
            return rgb_tensor, depth_tensor, torch.tensor(calories, dtype=torch.float32)
        
        except Exception as e:
            print(f"\n错误: 无法加载 {dish_id}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))


def validate_dataset(root_dir, csv_file, cache_file='valid_data_cache.pkl'):
    """验证数据集完整性"""
    if os.path.exists(cache_file):
        print("✓ 发现缓存，直接加载有效数据")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("检查数据完整性...")
    root_dir = Path(root_dir)
    df = pd.read_csv(csv_file)
    
    base_path = root_dir / 'comp-90086-nutrition-5-k' / 'Nutrition5K' / 'Nutrition5K'
    color_dir = base_path / 'train' / 'color'
    depth_dir = base_path / 'train' / 'depth_raw'
    
    valid_rows = []
    
    for idx in range(len(df)):
        dish_id = df.iloc[idx, 0]
        rgb_path = color_dir / dish_id / 'rgb.png'
        depth_path = depth_dir / dish_id / 'depth_raw.png'
        
        try:
            if rgb_path.exists() and depth_path.exists():
                Image.open(rgb_path).convert('RGB')
                Image.open(depth_path)
                valid_rows.append(df.iloc[idx])
        except Exception as e:
            print(f"  跳过损坏的样本: {dish_id} - {str(e)}")
    
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(valid_df, f)
    
    print(f"✓ 有效样本数: {len(valid_df)} / {len(df)}")
    print(f"✓ 缓存已保存到 {cache_file}")
    
    return valid_df


def get_dataloaders(root_dir, csv_file, batch_size=16, val_split=0.2, use_segmentation=True, unet_path='checkpoints/unet_best.pth'):
    """
    创建训练和验证数据加载器
    
    Args:
        use_segmentation: 是否使用U-Net语义分割
        unet_path: U-Net模型路径
    """
    valid_df = validate_dataset(root_dir, csv_file)
    
    n_val = int(len(valid_df) * val_split)
    np.random.seed(42)
    indices = np.random.permutation(len(valid_df))
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_df = valid_df.iloc[train_indices].reset_index(drop=True)
    val_df = valid_df.iloc[val_indices].reset_index(drop=True)
    
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    
    train_dataset = Nutrition5kDataset(root_dir, 'train_split.csv', is_train=True, 
                                       use_segmentation=use_segmentation, unet_path=unet_path)
    val_dataset = Nutrition5kDataset(root_dir, 'val_split.csv', is_train=False, 
                                     use_segmentation=use_segmentation, unet_path=unet_path)
    
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


class Nutrition5kTestDataset(Dataset):
    def __init__(self, root_dir, use_segmentation=True, unet_path='checkpoints/unet_best.pth'):
        """测试集数据加载器"""
        self.root_dir = Path(root_dir)
        self.use_segmentation = use_segmentation
        base_path = self.root_dir / 'comp-90086-nutrition-5-k' / 'Nutrition5K' / 'Nutrition5K'
        
        self.color_dir = base_path / 'test' / 'color'
        self.depth_dir = base_path / 'test' / 'depth_raw'
        
        self.dish_ids = sorted([d.name for d in self.color_dir.iterdir() if d.is_dir()])
        
        # 加载U-Net
        self.unet = None
        if use_segmentation and unet_path and os.path.exists(unet_path):
            self.unet = UNet(in_channels=1, out_channels=1)
            checkpoint = torch.load(unet_path, map_location='cpu', weights_only=False)
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            self.unet.eval()
        
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
    
    def segment_with_unet(self, depth_tensor):
        """使用U-Net生成mask"""
        with torch.no_grad():
            mask = self.unet(depth_tensor.unsqueeze(0))
            mask = mask.squeeze(0)
        return mask
    
    def simple_segment(self, depth_img):
        """简单阈值分割"""
        depth_array = np.array(depth_img, dtype=np.float32)
        
        if depth_array.max() > depth_array.min():
            depth_norm = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        else:
            depth_norm = depth_array
        
        threshold = 0.5
        mask = (depth_norm < threshold).astype(np.float32)
        
        return torch.from_numpy(mask).unsqueeze(0)
    
    def apply_mask_tensor(self, img_tensor, mask_tensor):
        """将mask应用到tensor"""
        return img_tensor * mask_tensor
    
    def __len__(self):
        return len(self.dish_ids)
    
    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        
        rgb_path = self.color_dir / dish_id / 'rgb.png'
        depth_path = self.depth_dir / dish_id / 'depth_raw.png'
        
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('L')
        
        rgb_tensor = self.rgb_transform(rgb_img)
        depth_tensor = self.depth_transform(depth_img)
        depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)
        
        # 生成mask
        if self.use_segmentation:
            if self.unet is not None:
                mask = self.segment_with_unet(depth_tensor)
            else:
                mask = self.simple_segment(depth_img)
            
            rgb_tensor = self.apply_mask_tensor(rgb_tensor, mask)
            depth_tensor = self.apply_mask_tensor(depth_tensor, mask)
        
        return rgb_tensor, depth_tensor, dish_id