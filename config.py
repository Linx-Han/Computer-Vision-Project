"""
Configuration file for Nutrition5k project
Centralized path and hyperparameter management
"""
import os
from pathlib import Path

class Config:
    # ============= Directory Configuration =============
    # Base directories
    PROJECT_ROOT = Path(__file__).parent.absolute()
    DATA_ROOT = Path(os.getenv('DATA_ROOT_DIR', PROJECT_ROOT/'data'))
    
    # Dataset paths
    DATASET_BASE = DATA_ROOT
    
    # Training data
    TRAIN_COLOR_DIR = DATASET_BASE / 'train' / 'color'
    TRAIN_DEPTH_DIR = DATASET_BASE / 'train' / 'depth_raw'
    TRAIN_CSV = Path(os.getenv('TRAIN_CSV_FILE', 
                               DATASET_BASE / 'nutrition5k_train.csv'))
    
    # Test data
    TEST_COLOR_DIR = DATASET_BASE / 'test' / 'color'
    TEST_DEPTH_DIR = DATASET_BASE / 'test' / 'depth_raw'
    
    # Output directories
    CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', PROJECT_ROOT / 'checkpoints'))
    CACHE_DIR = PROJECT_ROOT / 'cache'
    
    # Cache files (NEW NAMES - fresh start!)
    VALID_DATA_CACHE = CACHE_DIR / 'valid_data_cache_v2.pkl'
    TRAIN_SPLIT_CSV = CACHE_DIR / 'train_split_v2.csv'
    VAL_SPLIT_CSV = CACHE_DIR / 'val_split_v2.csv'
    NORMALIZATION_STATS = CACHE_DIR / 'normalization_stats.pkl'
    
    # ============= Feature Engineering Parameters =============
    DEPTH_TO_CM_SCALE = 100  # Conversion factor for depth to cm
    HEIGHT_PERCENTILE = 70   # Percentile for plate depth estimation
    
    # ============= Model Hyperparameters =============
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Learning rate scheduler
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_MIN_LR = 1e-6
    
    # Early stopping (10 epochs without improvement)
    EARLY_STOP_PATIENCE = 10
    
    # Model parameters
    DROPOUT_RATE = 0.3
    
    # ============= Image Parameters =============
    IMAGE_SIZE = 299  # InceptionV3 uses 299x299
    
    # RGB normalization (ImageNet stats)
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]
    
    # Depth and Height normalization (to be calculated from training data)
    DEPTH_MEAN = None  # Will be set after calculation
    DEPTH_STD = None
    HEIGHT_MEAN = None
    HEIGHT_STD = None
    
    # Data augmentation
    ROTATION_DEGREES = 15
    CROP_SCALE = (0.9, 1.0)
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    
    # ============= DataLoader Parameters =============
    NUM_WORKERS = 0
    PIN_MEMORY = False
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_paths(cls):
        """Validate that all required paths exist"""
        required_paths = [
            (cls.DATA_ROOT, "Data root directory"),
            (cls.DATASET_BASE, "Dataset base directory"),
            (cls.TRAIN_COLOR_DIR, "Training color images"),
            (cls.TRAIN_DEPTH_DIR, "Training depth images"),
            (cls.TRAIN_CSV, "Training CSV file"),
        ]
        
        missing_paths = []
        for path, description in required_paths:
            if not path.exists():
                missing_paths.append(f"{description}: {path}")
        
        if missing_paths:
            raise FileNotFoundError(
                "Missing required paths:\n" + "\n".join(f"  - {p}" for p in missing_paths)
            )
    
    @classmethod
    def set_normalization_stats(cls, depth_mean, depth_std, height_mean, height_std):
        """Set depth and height normalization statistics"""
        cls.DEPTH_MEAN = depth_mean
        cls.DEPTH_STD = depth_std
        cls.HEIGHT_MEAN = height_mean
        cls.HEIGHT_STD = height_std
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("🚀 Configuration (v2 - InceptionV3 + 5 Channels)")
        print("=" * 60)
        print(f"📁 Data Root:        {cls.DATA_ROOT}")
        print(f"📄 Training CSV:     {cls.TRAIN_CSV}")
        print(f"💾 Checkpoint Dir:   {cls.CHECKPOINT_DIR}")
        print(f"🗄️  Cache Dir:        {cls.CACHE_DIR}")
        print(f"\n🔧 Hyperparameters:")
        print(f"   Batch Size:       {cls.BATCH_SIZE}")
        print(f"   Epochs:           {cls.EPOCHS}")
        print(f"   Learning Rate:    {cls.LEARNING_RATE}")
        print(f"   Val Split:        {cls.VAL_SPLIT}")
        print(f"   Early Stop:       {cls.EARLY_STOP_PATIENCE} epochs")
        print(f"\n🎨 Feature Engineering:")
        print(f"   Image Size:       {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"   Depth Scale:      1/{cls.DEPTH_TO_CM_SCALE}")
        print(f"   Height Percentile: {cls.HEIGHT_PERCENTILE}")
        if cls.DEPTH_MEAN is not None:
            print(f"\n📊 Normalization Stats:")
            print(f"   RGB Mean:         {[f'{x:.3f}' for x in cls.RGB_MEAN]}")
            print(f"   RGB Std:          {[f'{x:.3f}' for x in cls.RGB_STD]}")
            print(f"   Depth Mean:       {cls.DEPTH_MEAN:.3f}")
            print(f"   Depth Std:        {cls.DEPTH_STD:.3f}")
            print(f"   Height Mean:      {cls.HEIGHT_MEAN:.3f}")
            print(f"   Height Std:       {cls.HEIGHT_STD:.3f}")
        print("=" * 60)