"""
Data loading and preprocessing for Nutrition5k dataset with relative depth
"""
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
from scipy import ndimage

from config import Config


class RelativeDepthProcessor:
    """Process depth images to calculate relative depth from plate surface"""
    
    @staticmethod
    def detect_plate_region(depth_array, percentile=10):
        """
        Detect plate region (lowest depth values = furthest from camera)
        
        Args:
            depth_array: Raw depth image as numpy array
            percentile: Percentile threshold to identify plate pixels
            
        Returns:
            plate_mask: Binary mask of plate region
            plate_depth: Median depth value of plate
        """
        # Find plate pixels (typically the lowest/furthest depth values)
        threshold = np.percentile(depth_array, percentile)
        plate_mask = depth_array <= threshold
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        plate_mask = cv2.morphologyEx(plate_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, kernel)
        
        # Get plate depth (median of plate region)
        plate_depth = np.median(depth_array[plate_mask > 0])
        
        return plate_mask, plate_depth
    
    @staticmethod
    def calculate_relative_depth(depth_array, method='percentile', plate_percentile=10):
        """
        Calculate relative depth: distance of each pixel from the plate surface
        
        Args:
            depth_array: Raw depth image as numpy array (H, W)
            method: Method for plate detection ('percentile', 'edge_based', 'adaptive')
            plate_percentile: Percentile for plate detection
            
        Returns:
            relative_depth: Depth relative to plate surface (H, W)
            plate_mask: Binary mask of detected plate region
        """
        # Normalize input to [0, 1]
        depth_norm = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
        
        if method == 'percentile':
            # Simple percentile-based method
            plate_mask, plate_depth = RelativeDepthProcessor.detect_plate_region(
                depth_norm, percentile=plate_percentile
            )
            
            # Calculate relative depth (food height above plate)
            # Higher values = closer to camera = taller food
            relative_depth = depth_norm - plate_depth
            relative_depth = np.clip(relative_depth, 0, None)  # Only positive heights
            
        elif method == 'edge_based':
            # More sophisticated: use edge detection to find plate boundary
            edges = cv2.Canny((depth_norm * 255).astype(np.uint8), 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Assume largest contour is the plate
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                plate_mask = np.zeros_like(depth_norm, dtype=np.uint8)
                cv2.drawContours(plate_mask, [largest_contour], -1, 1, -1)
                
                plate_depth = np.median(depth_norm[plate_mask > 0])
                relative_depth = depth_norm - plate_depth
                relative_depth = np.clip(relative_depth, 0, None)
            else:
                # Fallback to percentile method
                return RelativeDepthProcessor.calculate_relative_depth(
                    depth_array, method='percentile', plate_percentile=plate_percentile
                )
        
        elif method == 'adaptive':
            # Adaptive thresholding for varying lighting/depth conditions
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            plate_mask = cv2.adaptiveThreshold(
                depth_uint8, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Clean mask
            kernel = np.ones((5, 5), np.uint8)
            plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, kernel)
            
            if np.sum(plate_mask) > 0:
                plate_depth = np.median(depth_norm[plate_mask > 0])
                relative_depth = depth_norm - plate_depth
                relative_depth = np.clip(relative_depth, 0, None)
            else:
                # Fallback
                return RelativeDepthProcessor.calculate_relative_depth(
                    depth_array, method='percentile', plate_percentile=plate_percentile
                )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize relative depth to [0, 1]
        if relative_depth.max() > 0:
            relative_depth = relative_depth / relative_depth.max()
        
        return relative_depth, plate_mask


class Nutrition5kDataset(Dataset):
    """Dataset class for Nutrition5k training/validation data with relative depth"""
    
    def __init__(self, csv_file, is_train=True, use_relative_depth=True, 
                 depth_method='percentile', plate_percentile=10):
        """
        Args:
            csv_file: Path to CSV file containing dish_id and calories
            is_train: If True, apply data augmentation
            use_relative_depth: If True, use relative depth instead of raw depth
            depth_method: Method for plate detection ('percentile', 'edge_based', 'adaptive')
            plate_percentile: Percentile threshold for plate detection
        """
        self.df = pd.read_csv(csv_file)
        self.is_train = is_train
        self.use_relative_depth = use_relative_depth
        self.depth_method = depth_method
        self.plate_percentile = plate_percentile
        
        # Define transforms
        self.rgb_transform = self._get_rgb_transform(is_train)
        self.depth_transform = self._get_depth_transform()
        
        # Relative depth processor
        self.depth_processor = RelativeDepthProcessor()
    
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
        """Get depth image transforms (resize only, normalization done after relative depth)"""
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
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
            
            # Apply RGB transform
            rgb_tensor = self.rgb_transform(rgb_img)  # (3, H, W)
            
            # Process depth
            depth_array = np.array(depth_img)
            depth_array = cv2.resize(depth_array, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
            # Normalize raw depth to [0, 1]
            depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
            depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).float()  # (1, H, W)
            
            if self.use_relative_depth:
                # Calculate relative depth (height above plate)
                relative_depth, plate_mask = self.depth_processor.calculate_relative_depth(
                    depth_array, 
                    method=self.depth_method,
                    plate_percentile=self.plate_percentile
                )
                height_tensor = torch.from_numpy(relative_depth).unsqueeze(0).float()  # (1, H, W)
                
                # Concatenate: 3 RGB + 1 depth + 1 height = 5 channels
                combined_tensor = torch.cat([rgb_tensor, depth_tensor, height_tensor], dim=0)  # (5, H, W)
            else:
                # Without relative depth: 3 RGB + 1 depth = 4 channels
                combined_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4, H, W)
            
            return combined_tensor, torch.tensor(calories, dtype=torch.float32)
        
        except Exception as e:
            print(f"\n⚠️  Error loading {dish_id}: {str(e)}")
            # Return next sample on error
            return self.__getitem__((idx + 1) % len(self))


class Nutrition5kTestDataset(Dataset):
    """Dataset class for Nutrition5k test data with 5-channel output (for Kaggle submission)"""
    
    def __init__(self, use_relative_depth=True, depth_method='percentile', plate_percentile=10):
        """
        Args:
            use_relative_depth: If True, use relative depth instead of raw depth
            depth_method: Method for plate detection
            plate_percentile: Percentile threshold for plate detection
        """
        # Get all dish IDs from test directory
        self.dish_ids = sorted([d.name for d in Config.TEST_COLOR_DIR.iterdir() if d.is_dir()])
        self.use_relative_depth = use_relative_depth
        self.depth_method = depth_method
        self.plate_percentile = plate_percentile
        
        # Define transforms
        self.rgb_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.RGB_MEAN, std=Config.RGB_STD)
        ])
        
        self.depth_processor = RelativeDepthProcessor()
    
    def __len__(self):
        return len(self.dish_ids)
    
    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        
        rgb_path = Config.TEST_COLOR_DIR / dish_id / 'rgb.png'
        depth_path = Config.TEST_DEPTH_DIR / dish_id / 'depth_raw.png'
        
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('L')
        
        # Apply RGB transform
        rgb_tensor = self.rgb_transform(rgb_img)  # (3, H, W)
        
        # Process depth
        depth_array = np.array(depth_img)
        depth_array = cv2.resize(depth_array, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        
        # Normalize raw depth to [0, 1]
        depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
        depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).float()  # (1, H, W)
        
        if self.use_relative_depth:
            # Calculate relative depth (height above plate)
            relative_depth, plate_mask = self.depth_processor.calculate_relative_depth(
                depth_array,
                method=self.depth_method,
                plate_percentile=self.plate_percentile
            )
            height_tensor = torch.from_numpy(relative_depth).unsqueeze(0).float()  # (1, H, W)
            
            # Concatenate: 3 RGB + 1 depth + 1 height = 5 channels
            combined_tensor = torch.cat([rgb_tensor, depth_tensor, height_tensor], dim=0)  # (5, H, W)
        else:
            # Without relative depth: 3 RGB + 1 depth = 4 channels
            combined_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4, H, W)
        
        return combined_tensor, dish_id


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


def get_dataloaders(batch_size=None, val_split=None, use_relative_depth=True, 
                    depth_method='percentile', plate_percentile=10):
    """
    Create train and validation data loaders
    
    Args:
        batch_size: Batch size (defaults to Config.BATCH_SIZE)
        val_split: Validation split ratio (defaults to Config.VAL_SPLIT)
        use_relative_depth: If True, use relative depth calculation
        depth_method: Method for plate detection ('percentile', 'edge_based', 'adaptive')
        plate_percentile: Percentile threshold for plate detection
    
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
    
    # Create datasets with relative depth
    train_dataset = Nutrition5kDataset(
        Config.TRAIN_SPLIT_CSV, 
        is_train=True,
        use_relative_depth=use_relative_depth,
        depth_method=depth_method,
        plate_percentile=plate_percentile
    )
    val_dataset = Nutrition5kDataset(
        Config.VAL_SPLIT_CSV, 
        is_train=False,
        use_relative_depth=use_relative_depth,
        depth_method=depth_method,
        plate_percentile=plate_percentile
    )
    
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


def get_test_loader(batch_size=None, use_relative_depth=True, 
                   depth_method='percentile', plate_percentile=10):
    """
    Create test data loader
    
    Args:
        batch_size: Batch size (defaults to Config.BATCH_SIZE)
        use_relative_depth: If True, use relative depth calculation
        depth_method: Method for plate detection
        plate_percentile: Percentile threshold for plate detection
    
    Returns:
        test_loader
    """
    batch_size = batch_size or Config.BATCH_SIZE
    
    test_dataset = Nutrition5kTestDataset(
        use_relative_depth=use_relative_depth,
        depth_method=depth_method,
        plate_percentile=plate_percentile
    )
    
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
    
    # Create data loaders with relative depth
    print("\n📦 Loading data with relative depth calculation...")
    train_loader, val_loader = get_dataloaders(
        use_relative_depth=True,
        depth_method='percentile',  # Try 'edge_based' or 'adaptive' for different methods
        plate_percentile=10
    )
    
    print(f"\n✓ Training samples:   {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Train batches:      {len(train_loader)}")
    print(f"✓ Val batches:        {len(val_loader)}")
    
    # Test loading a batch
    print("\n🧪 Testing batch loading with 5-channel tensor...")
    combined, calories = next(iter(train_loader))
    print(f"✓ Combined tensor shape: {combined.shape}")  # Should be (batch_size, 5, H, W)
    print(f"  - RGB channels:        {combined[:, :3, :, :].shape}")
    print(f"  - Depth channel:       {combined[:, 3:4, :, :].shape}")
    print(f"  - Height channel:      {combined[:, 4:5, :, :].shape}")
    print(f"✓ Calories shape:        {calories.shape}")
    print(f"✓ Calorie range:         [{calories.min():.1f}, {calories.max():.1f}]")
    
    # Display channel statistics
    print(f"\n📊 Channel Statistics:")
    print(f"  RGB mean:    [{combined[:, 0, :, :].mean():.3f}, {combined[:, 1, :, :].mean():.3f}, {combined[:, 2, :, :].mean():.3f}]")
    print(f"  Depth range:  [{combined[:, 3, :, :].min():.3f}, {combined[:, 3, :, :].max():.3f}]")
    print(f"  Height range: [{combined[:, 4, :, :].min():.3f}, {combined[:, 4, :, :].max():.3f}]")
    
    print("\n✅ 5-channel data loading successful!")