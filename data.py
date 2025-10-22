"""
Data loading and preprocessing for Nutrition5k dataset
5-channel input: RGB + Depth + Height
"""
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from config import Config


def calculate_height_from_depth(depth_np):
    """
    Calculate height above plate from depth image
    
    Args:
        depth_np: Depth image as numpy array
    
    Returns:
        height_cm: Height above plate in cm
    """
    # Convert to cm
    depth_cm = depth_np.astype(np.float32) / Config.DEPTH_TO_CM_SCALE
    
    # Find plate depth (reference plane) using percentile
    plate_depth = float(np.percentile(depth_cm, Config.HEIGHT_PERCENTILE))
    
    # Calculate height above plate (clip negative values)
    height_cm = np.clip(plate_depth - depth_cm, a_min=0, a_max=None)
    
    return height_cm


class Nutrition5kDataset(Dataset):
    """Dataset class for Nutrition5k with 5-channel input"""
    
    def __init__(self, csv_file, is_train=True):
        """
        Args:
            csv_file: Path to CSV file containing dish_id and calories
            is_train: If True, apply data augmentation
        """
        self.df = pd.read_csv(csv_file)
        self.is_train = is_train
        
        # Load normalization stats
        if Config.DEPTH_MEAN is None:
            raise ValueError("Normalization stats not set! Call compute_normalization_stats() first.")
        
        # Convert to numpy arrays for easier use
        self.rgb_mean = np.array(Config.RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.rgb_std = np.array(Config.RGB_STD, dtype=np.float32).reshape(3, 1, 1)
        self.depth_mean = Config.DEPTH_MEAN
        self.depth_std = Config.DEPTH_STD
        self.height_mean = Config.HEIGHT_MEAN
        self.height_std = Config.HEIGHT_STD
        
        # Basic transforms (resize only, we'll do augmentation manually)
        self.resize = transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        self.to_tensor = transforms.ToTensor()
    
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
            
            # Resize
            rgb_img = self.resize(rgb_img)
            depth_img = self.resize(depth_img)
            
            # Convert to numpy for processing
            rgb_np = np.array(rgb_img).astype(np.float32) / 255.0  # [H, W, 3] in [0, 1]
            depth_np = np.array(depth_img).astype(np.float32)       # [H, W] raw values
            
            # Calculate height from depth
            height_np = calculate_height_from_depth(depth_np)  # [H, W] in cm
            
            # Convert depth to cm
            depth_cm = depth_np / Config.DEPTH_TO_CM_SCALE  # [H, W] in cm
            
            # Convert to tensors and rearrange to [C, H, W]
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)  # [3, H, W]
            depth_tensor = torch.from_numpy(depth_cm).unsqueeze(0)  # [1, H, W]
            height_tensor = torch.from_numpy(height_np).unsqueeze(0)  # [1, H, W]
            
            # Apply data augmentation if training
            if self.is_train:
                # Stack all channels for synchronized augmentation
                all_channels = torch.cat([rgb_tensor, depth_tensor, height_tensor], dim=0)  # [5, H, W]
                
                # Random rotation
                if np.random.rand() < 0.5:
                    angle = np.random.uniform(-Config.ROTATION_DEGREES, Config.ROTATION_DEGREES)
                    all_channels = transforms.functional.rotate(all_channels, angle)
                
                # Random crop and resize
                if np.random.rand() < 0.5:
                    i, j, h, w = transforms.RandomResizedCrop.get_params(
                        all_channels, scale=Config.CROP_SCALE, ratio=(1.0, 1.0)
                    )
                    all_channels = transforms.functional.resized_crop(
                        all_channels, i, j, h, w, (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                    )
                
                # Split back
                rgb_tensor = all_channels[0:3]
                depth_tensor = all_channels[3:4]
                height_tensor = all_channels[4:5]
                
                # Color jitter (RGB only)
                if np.random.rand() < 0.5:
                    color_jitter = transforms.ColorJitter(
                        brightness=Config.COLOR_JITTER_BRIGHTNESS,
                        contrast=Config.COLOR_JITTER_CONTRAST
                    )
                    rgb_pil = transforms.ToPILImage()(rgb_tensor)
                    rgb_tensor = self.to_tensor(color_jitter(rgb_pil))
            
            # Normalize all channels
            rgb_normalized = (rgb_tensor - torch.from_numpy(self.rgb_mean)) / torch.from_numpy(self.rgb_std)
            depth_normalized = (depth_tensor - self.depth_mean) / self.depth_std
            height_normalized = (height_tensor - self.height_mean) / self.height_std
            
            # Concatenate all 5 channels
            five_channel_input = torch.cat([
                rgb_normalized,      # [3, H, W]
                depth_normalized,    # [1, H, W]
                height_normalized    # [1, H, W]
            ], dim=0)  # [5, H, W]
            
            return five_channel_input, torch.tensor(calories, dtype=torch.float32)
        
        except Exception as e:
            print(f"\n⚠️  Error loading {dish_id}: {str(e)}")
            # Return next sample on error
            return self.__getitem__((idx + 1) % len(self))


class Nutrition5kTestDataset(Dataset):
    """Test dataset with 5-channel input"""
    
    def __init__(self):
        """Initialize test dataset"""
        # Get all dish IDs from test directory
        self.dish_ids = sorted([d.name for d in Config.TEST_COLOR_DIR.iterdir() if d.is_dir()])
        
        # Load normalization stats
        if Config.DEPTH_MEAN is None:
            raise ValueError("Normalization stats not set!")
        
        self.rgb_mean = np.array(Config.RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.rgb_std = np.array(Config.RGB_STD, dtype=np.float32).reshape(3, 1, 1)
        self.depth_mean = Config.DEPTH_MEAN
        self.depth_std = Config.DEPTH_STD
        self.height_mean = Config.HEIGHT_MEAN
        self.height_std = Config.HEIGHT_STD
        
        self.resize = transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    
    def __len__(self):
        return len(self.dish_ids)
    
    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        
        rgb_path = Config.TEST_COLOR_DIR / dish_id / 'rgb.png'
        depth_path = Config.TEST_DEPTH_DIR / dish_id / 'depth_raw.png'
        
        # Load and process (same as training, no augmentation)
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('L')
        
        rgb_img = self.resize(rgb_img)
        depth_img = self.resize(depth_img)
        
        rgb_np = np.array(rgb_img).astype(np.float32) / 255.0
        depth_np = np.array(depth_img).astype(np.float32)
        
        height_np = calculate_height_from_depth(depth_np)
        depth_cm = depth_np / Config.DEPTH_TO_CM_SCALE
        
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        depth_tensor = torch.from_numpy(depth_cm).unsqueeze(0)
        height_tensor = torch.from_numpy(height_np).unsqueeze(0)
        
        # Normalize
        rgb_normalized = (rgb_tensor - torch.from_numpy(self.rgb_mean)) / torch.from_numpy(self.rgb_std)
        depth_normalized = (depth_tensor - self.depth_mean) / self.depth_std
        height_normalized = (height_tensor - self.height_mean) / self.height_std
        
        five_channel_input = torch.cat([rgb_normalized, depth_normalized, height_normalized], dim=0)
        
        return five_channel_input, dish_id


def compute_normalization_stats():
    """
    Compute mean and std for depth and height channels from training data
    """
    print("\n📊 Computing normalization statistics for depth and height...")
    
    # Check if already computed and cached
    if Config.NORMALIZATION_STATS.exists():
        print("✓ Loading cached normalization stats")
        with open(Config.NORMALIZATION_STATS, 'rb') as f:
            stats = pickle.load(f)
        Config.set_normalization_stats(
            stats['depth_mean'], stats['depth_std'],
            stats['height_mean'], stats['height_std']
        )
        print(f"  Depth:  mean={stats['depth_mean']:.3f}, std={stats['depth_std']:.3f}")
        print(f"  Height: mean={stats['height_mean']:.3f}, std={stats['height_std']:.3f}")
        return
    
    # Load valid dataframe
    valid_df = validate_dataset()
    
    depth_values = []
    height_values = []
    
    print("  Processing images...")
    for idx in tqdm(range(min(len(valid_df), 1000)), desc="Computing stats"):  # Sample 1000 images
        dish_id = valid_df.iloc[idx, 0]
        depth_path = Config.TRAIN_DEPTH_DIR / dish_id / 'depth_raw.png'
        
        try:
            depth_img = Image.open(depth_path).convert('L')
            depth_img = transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))(depth_img)
            depth_np = np.array(depth_img).astype(np.float32)
            
            # Convert to cm
            depth_cm = depth_np / Config.DEPTH_TO_CM_SCALE
            
            # Calculate height
            height_cm = calculate_height_from_depth(depth_np)
            
            # Collect values
            depth_values.append(depth_cm.flatten())
            height_values.append(height_cm.flatten())
        except:
            continue
    
    # Calculate statistics
    depth_all = np.concatenate(depth_values)
    height_all = np.concatenate(height_values)
    
    depth_mean = float(np.mean(depth_all))
    depth_std = float(np.std(depth_all))
    height_mean = float(np.mean(height_all))
    height_std = float(np.std(height_all))
    
    # Save stats
    stats = {
        'depth_mean': depth_mean,
        'depth_std': depth_std,
        'height_mean': height_mean,
        'height_std': height_std
    }
    
    with open(Config.NORMALIZATION_STATS, 'wb') as f:
        pickle.dump(stats, f)
    
    # Set in config
    Config.set_normalization_stats(depth_mean, depth_std, height_mean, height_std)
    
    print(f"✓ Statistics computed and cached")
    print(f"  Depth:  mean={depth_mean:.3f}, std={depth_std:.3f}")
    print(f"  Height: mean={height_mean:.3f}, std={height_std:.3f}")


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
    
    for idx in tqdm(range(len(df)), desc="Validating"):
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
    
    # Compute normalization statistics BEFORE creating datasets
    compute_normalization_stats()
    
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
    """Create test data loader"""
    batch_size = batch_size or Config.BATCH_SIZE
    
    # Ensure normalization stats are loaded
    if Config.DEPTH_MEAN is None:
        compute_normalization_stats()
    
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
    
    # Create data loaders
    print("\n📦 Loading data...")
    train_loader, val_loader = get_dataloaders()
    
    Config.print_config()
    
    print(f"\n✓ Training samples:   {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Train batches:      {len(train_loader)}")
    print(f"✓ Val batches:        {len(val_loader)}")
    
    # Test loading a batch
    print("\n🧪 Testing batch loading...")
    five_channel, calories = next(iter(train_loader))
    print(f"✓ Input shape:    {five_channel.shape}  (should be [batch, 5, 299, 299])")
    print(f"✓ Calories shape: {calories.shape}")
    print(f"✓ Calorie range:  [{calories.min():.1f}, {calories.max():.1f}]")
    
    print("\n✅ Data loading successful!")