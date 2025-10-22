"""
Height-based depth preprocessing for Nutrition5k dataset
Converts depth maps to food height above plate with caching
"""
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pickle

from config import Config


def process_depth_to_height(depth_img, height_percentile=None):
    """
    Convert depth map to food height above plate
    
    Args:
        depth_img: PIL Image or numpy array of depth map
        height_percentile: Percentile to use as plate reference (default: from Config)
    
    Returns:
        height_cm: numpy array of food heights in centimeters
        plate_depth: the reference plate depth value
        food_mask: binary mask where food exists (height > threshold)
    """
    if height_percentile is None:
        height_percentile = Config.HEIGHT_PERCENTILE
    
    # Convert to numpy
    if isinstance(depth_img, Image.Image):
        depth_np = np.array(depth_img)
    else:
        depth_np = depth_img
    
    # Convert to cm
    depth_cm = depth_np.astype(np.float32) / Config.DEPTH_TO_CM_SCALE
    
    # Find plate depth (reference plane)
    plate_depth = float(np.percentile(depth_cm, height_percentile))
    
    # Calculate height above plate
    height_cm = np.clip(plate_depth - depth_cm, a_min=0, a_max=None)
    
    # Create food mask
    food_mask = (height_cm > Config.FOOD_HEIGHT_THRESHOLD).astype(np.uint8)
    
    return height_cm, plate_depth, food_mask


def preprocess_single_dish(dish_id, depth_dir, height_cache_dir, mask_cache_dir):
    """
    Preprocess a single dish: compute height map and mask, save to cache
    
    Args:
        dish_id: Dish identifier (e.g., 'dish_0200')
        depth_dir: Directory containing depth images
        height_cache_dir: Directory to save height maps
        mask_cache_dir: Directory to save masks
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load depth image
        depth_path = depth_dir / dish_id / 'depth_raw.png'
        if not depth_path.exists():
            return False
        
        depth_img = Image.open(depth_path)
        
        # Process to height
        height_cm, plate_depth, food_mask = process_depth_to_height(depth_img)
        
        # Save height map as float32 numpy array
        height_save_path = height_cache_dir / f'{dish_id}.npy'
        np.save(height_save_path, height_cm.astype(np.float32))
        
        # Save mask as uint8 PNG (more space efficient)
        mask_save_path = mask_cache_dir / f'{dish_id}.png'
        mask_img = Image.fromarray(food_mask * 255)  # Scale to 0-255 for PNG
        mask_img.save(mask_save_path)
        
        return True
    
    except Exception as e:
        print(f"  ⚠️  Error processing {dish_id}: {str(e)}")
        return False


def preprocess_dataset(depth_dir, height_cache_dir, mask_cache_dir, dataset_name='dataset'):
    """
    Preprocess entire dataset: compute all height maps and masks
    
    Args:
        depth_dir: Directory containing depth images
        height_cache_dir: Directory to save height maps
        mask_cache_dir: Directory to save masks
        dataset_name: Name for progress display
    
    Returns:
        success_count: Number of successfully processed dishes
    """
    # Get all dish IDs
    dish_ids = sorted([d.name for d in depth_dir.iterdir() if d.is_dir()])
    
    print(f"\n🔧 Preprocessing {dataset_name} dataset...")
    print(f"   Total dishes: {len(dish_ids)}")
    print(f"   Height percentile: P{Config.HEIGHT_PERCENTILE}")
    print(f"   Food threshold: {Config.FOOD_HEIGHT_THRESHOLD} cm")
    
    success_count = 0
    
    for dish_id in tqdm(dish_ids, desc=f'Processing {dataset_name}'):
        if preprocess_single_dish(dish_id, depth_dir, height_cache_dir, mask_cache_dir):
            success_count += 1
    
    print(f"✅ Successfully processed {success_count}/{len(dish_ids)} dishes")
    
    # Save metadata
    metadata = {
        'height_percentile': Config.HEIGHT_PERCENTILE,
        'food_threshold': Config.FOOD_HEIGHT_THRESHOLD,
        'depth_scale': Config.DEPTH_TO_CM_SCALE,
        'total_dishes': len(dish_ids),
        'successful': success_count
    }
    
    metadata_path = height_cache_dir.parent / f'{dataset_name}_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return success_count


def preprocess_all():
    """Preprocess both train and test datasets"""
    print("=" * 60)
    print("Height-Based Depth Preprocessing")
    print("=" * 60)
    
    # Create directories
    Config.create_directories()
    
    # Preprocess training data
    if Config.TRAIN_DEPTH_DIR.exists():
        preprocess_dataset(
            Config.TRAIN_DEPTH_DIR,
            Config.TRAIN_HEIGHT_CACHE,
            Config.TRAIN_MASK_CACHE,
            'train'
        )
    else:
        print(f"⚠️  Training depth directory not found: {Config.TRAIN_DEPTH_DIR}")
    
    # Preprocess test data
    if Config.TEST_DEPTH_DIR.exists():
        preprocess_dataset(
            Config.TEST_DEPTH_DIR,
            Config.TEST_HEIGHT_CACHE,
            Config.TEST_MASK_CACHE,
            'test'
        )
    else:
        print(f"⚠️  Test depth directory not found: {Config.TEST_DEPTH_DIR}")
    
    print("\n✅ Preprocessing complete!")
    print(f"   Height maps saved to: {Config.HEIGHT_CACHE_DIR}")
    print(f"   Masks saved to: {Config.MASK_CACHE_DIR}")


def check_cache_exists(dish_id, height_cache_dir, mask_cache_dir):
    """
    Check if cached height map and mask exist for a dish
    
    Args:
        dish_id: Dish identifier
        height_cache_dir: Directory containing height maps
        mask_cache_dir: Directory containing masks
    
    Returns:
        True if both exist, False otherwise
    """
    height_path = height_cache_dir / f'{dish_id}.npy'
    mask_path = mask_cache_dir / f'{dish_id}.png'
    return height_path.exists() and mask_path.exists()


def load_cached_height_and_mask(dish_id, height_cache_dir, mask_cache_dir):
    """
    Load cached height map and mask for a dish
    
    Args:
        dish_id: Dish identifier
        height_cache_dir: Directory containing height maps
        mask_cache_dir: Directory containing masks
    
    Returns:
        height_cm: numpy array of heights
        food_mask: numpy array of binary mask (0 or 1)
    """
    height_path = height_cache_dir / f'{dish_id}.npy'
    mask_path = mask_cache_dir / f'{dish_id}.png'
    
    # Load height map
    height_cm = np.load(height_path)
    
    # Load mask (convert from 0-255 back to 0-1)
    mask_img = Image.open(mask_path)
    food_mask = (np.array(mask_img) > 127).astype(np.uint8)
    
    return height_cm, food_mask


if __name__ == '__main__':
    # Run preprocessing
    preprocess_all()