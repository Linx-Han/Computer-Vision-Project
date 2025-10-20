"""
Data loading and preprocessing for Nutrition5k dataset
"""
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

from config import Config


class Nutrition5kDataset(Dataset):
    """Dataset class for Nutrition5k training/validation data"""
    
    def __init__(self, csv_file, is_train=True):
        """
        Args:
            csv_file: Path to CSV file containing dish_id and calories
            is_train: If True, apply data augmentation
        """
        self.df = pd.read_csv(csv_file)
        self.is_train = is_train
        
        # Define transforms
        self.rgb_transform = self._get_rgb_transform(is_train)
        self.depth_transform = self._get_depth_transform()
    
    def _get_rgb_transform(self, is_train):
        """Get RGB image transforms"""
        if is_train:
            return transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.RandomRotation(Config.ROTATION_DEGREES),
                transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=Config.CROP_SCALE),
                transforms.ColorJitter(
                    brightness=Config.COLOR_JITTER_BRIGHTNESS,
                    contrast=Config.COLOR_JITTER_CONTRAST
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.RGB_MEAN, std=Config.RGB_STD)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.RGB_MEAN, std=Config.RGB_STD)
            ])
    
    def _get_depth_transform(self):
        """Get depth image transforms"""
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            # Get dish_id and calories
            row = self.df.iloc[idx]
            dish_id = row.iloc[0]
            calories = row.iloc[1]
            
            # Build image paths
            rgb_path = Config.TRAIN_COLOR_DIR / dish_id / 'rgb.png'
            depth_path = Config.TRAIN_DEPTH_DIR / dish_id / 'depth_raw.png'
            
            # Load images
            rgb_img = Image.open(rgb_path).convert('RGB')
            depth_img = Image.open(depth_path).convert('L')
            
            # Apply transforms
            rgb_img = self.rgb_transform(rgb_img)
            depth_img = self.depth_transform(depth_img)
            
            # Normalize depth to [0, 1]
            depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
            
            return rgb_img, depth_img, torch.tensor(calories, dtype=torch.float32)
        
        except Exception as e:
            print(f"\n⚠️  Error loading {dish_id}: {str(e)}")
            # Return next sample on error
            return self.__getitem__((idx + 1) % len(self))


class Nutrition5kTestDataset(Dataset):
    """Dataset class for Nutrition5k test data (for Kaggle submission)"""
    
    def __init__(self):
        """Initialize test dataset"""
        # Get all dish IDs from test directory
        self.dish_ids = sorted([d.name for d in Config.TEST_COLOR_DIR.iterdir() if d.is_dir()])
        
        # Define transforms (no augmentation for test)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.RGB_MEAN, std=Config.RGB_STD)
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.dish_ids)
    
    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        
        rgb_path = Config.TEST_COLOR_DIR / dish_id / 'rgb.png'
        depth_path = Config.TEST_DEPTH_DIR / dish_id / 'depth_raw.png'
        
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('L')
        
        rgb_img = self.rgb_transform(rgb_img)
        depth_img = self.depth_transform(depth_img)
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
        
        return rgb_img, depth_img, dish_id


def validate_dataset():
    """
    Validate dataset integrity and return filtered DataFrame
    Uses cache to avoid re-validation
    """
    # Check cache
    if Config.VALID_DATA_CACHE.exists():
        print("✓ Loading cached valid data")
        with open(Config.VALID_DATA_CACHE, 'rb') as f:
            return pickle.load(f)
    
    print("🔍 Validating dataset integrity...")
    df = pd.read_csv(Config.TRAIN_CSV)
    
    valid_rows = []
    invalid_count = 0
    
    for idx in range(len(df)):
        dish_id = df.iloc[idx, 0]
        rgb_path = Config.TRAIN_COLOR_DIR / dish_id / 'rgb.png'
        depth_path = Config.TRAIN_DEPTH_DIR / dish_id / 'depth_raw.png'
        
        try:
            if rgb_path.exists() and depth_path.exists():
                Image.open(rgb_path).convert('RGB')
                Image.open(depth_path)
                valid_rows.append(df.iloc[idx])
            else:
                invalid_count += 1
        except Exception as e:
            print(f"  ⚠️  Skipping corrupted sample: {dish_id}")
            invalid_count += 1
    
    # Create valid DataFrame
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    
    # Save cache
    with open(Config.VALID_DATA_CACHE, 'wb') as f:
        pickle.dump(valid_df, f)
    
    print(f"✓ Valid samples: {len(valid_df)} / {len(df)}")
    if invalid_count > 0:
        print(f"  ⚠️  Skipped {invalid_count} invalid samples")
    print(f"✓ Cache saved to {Config.VALID_DATA_CACHE}")
    
    return valid_df


def get_dataloaders(batch_size=None, val_split=None):
    """
    Create train and validation data loaders
    
    Args:
        batch_size: Batch size (defaults to Config.BATCH_SIZE)
        val_split: Validation split ratio (defaults to Config.VAL_SPLIT)
    
    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or Config.BATCH_SIZE
    val_split = val_split or Config.VAL_SPLIT
    
    # Validate and get valid data
    valid_df = validate_dataset()
    
    # Split into train/val
    n_val = int(len(valid_df) * val_split)
    
    # Use fixed random seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    indices = np.random.permutation(len(valid_df))
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    # Create split DataFrames
    train_df = valid_df.iloc[train_indices].reset_index(drop=True)
    val_df = valid_df.iloc[val_indices].reset_index(drop=True)
    
    # Save split CSVs
    train_df.to_csv(Config.TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(Config.VAL_SPLIT_CSV, index=False)
    
    # Create datasets
    train_dataset = Nutrition5kDataset(Config.TRAIN_SPLIT_CSV, is_train=True)
    val_dataset = Nutrition5kDataset(Config.VAL_SPLIT_CSV, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    return train_loader, val_loader


def get_test_loader(batch_size=None):
    """
    Create test data loader
    
    Args:
        batch_size: Batch size (defaults to Config.BATCH_SIZE)
    
    Returns:
        test_loader
    """
    batch_size = batch_size or Config.BATCH_SIZE
    
    test_dataset = Nutrition5kTestDataset()
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    return test_loader


# ============= Usage Example =============
if __name__ == '__main__':
    # Validate configuration
    Config.validate_paths()
    Config.create_directories()
    Config.print_config()
    
    # Create data loaders
    print("\n📦 Loading data...")
    train_loader, val_loader = get_dataloaders()
    
    print(f"\n✓ Training samples:   {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Train batches:      {len(train_loader)}")
    print(f"✓ Val batches:        {len(val_loader)}")
    
    # Test loading a batch
    print("\n🧪 Testing batch loading...")
    rgb, depth, calories = next(iter(train_loader))
    print(f"✓ RGB shape:     {rgb.shape}")
    print(f"✓ Depth shape:   {depth.shape}")
    print(f"✓ Calories shape: {calories.shape}")
    print(f"✓ Calorie range: [{calories.min():.1f}, {calories.max():.1f}]")
    
    print("\n✅ Data loading successful!")