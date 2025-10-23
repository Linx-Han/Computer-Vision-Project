"""
Prediction script for Nutrition5k test set
Generates Kaggle submission file using InceptionV3 + 5-channel model
"""
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from config import Config
from data import get_test_loader
from model import InceptionV3Regression, get_device


def predict_test_set(model_path='checkpoints/best_model.pth', output_file='submission.csv'):
    """
    Generate predictions on test set and create Kaggle submission file
    
    Args:
        model_path: Path to best model checkpoint
        output_file: Output CSV filename
    
    Returns:
        submission_df: DataFrame with predictions
    """
    # Setup
    Config.validate_paths()
    Config.create_directories()
    
    # Get device
    device = get_device()
    
    # Load model
    print("\n📂 Loading model...")
    model = InceptionV3Regression(
        num_channels=5,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val RMSE: {checkpoint['val_rmse']:.4f}")
    
    # Load test data
    print("\n📦 Loading test data...")
    test_loader = get_test_loader()
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Predict
    print("\n🔮 Generating predictions...")
    predictions = []
    dish_ids = []
    
    with torch.no_grad():
        for inputs, batch_dish_ids in tqdm(test_loader, desc='Predicting'):
            inputs = inputs.to(device)
            
            # Predict calories
            pred_calories = model(inputs)
            
            # Collect results
            predictions.extend(pred_calories.cpu().numpy().tolist())
            dish_ids.extend(batch_dish_ids)
    
    # Create submission DataFrame
    print("\n📝 Creating submission file...")
    submission_df = pd.DataFrame({
        'ID': dish_ids,
        'Value': predictions
    })
    
    # Sort by ID (ensure correct order)
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    
    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    print(f"✓ Submission saved: {output_file}")
    
    # Display preview
    print("\n" + "="*70)
    print("📊 PREDICTION PREVIEW (First 10 samples)")
    print("="*70)
    print(submission_df.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("📈 PREDICTION STATISTICS")
    print("="*70)
    print(f"  Total samples:  {len(submission_df):,}")
    print(f"  Min calories:   {submission_df['Value'].min():.2f}")
    print(f"  Max calories:   {submission_df['Value'].max():.2f}")
    print(f"  Mean calories:  {submission_df['Value'].mean():.2f}")
    print(f"  Median:         {submission_df['Value'].median():.2f}")
    print(f"  Std deviation:  {submission_df['Value'].std():.2f}")
    
    # Calorie distribution
    print(f"\n📊 CALORIE DISTRIBUTION")
    print(f"  {'Range':<15} {'Count':>8} {'Percentage':>12}")
    print(f"  {'-'*15} {'-'*8} {'-'*12}")
    
    ranges = [
        (0, 100, "0-100 cal"),
        (100, 200, "100-200 cal"),
        (200, 300, "200-300 cal"),
        (300, 400, "300-400 cal"),
        (400, 500, "400-500 cal"),
        (500, float('inf'), "500+ cal")
    ]
    
    for low, high, label in ranges:
        if high == float('inf'):
            count = len(submission_df[submission_df['Value'] >= low])
        else:
            count = len(submission_df[(submission_df['Value'] >= low) & (submission_df['Value'] < high)])
        pct = count / len(submission_df) * 100
        print(f"  {label:<15} {count:>8,} {pct:>11.1f}%")
    
    print("\n" + "="*70)
    print("✅ READY FOR KAGGLE SUBMISSION!")
    print("="*70)
    
    return submission_df


def verify_submission_format(submission_file='submission.csv'):
    """
    Verify submission file format is correct for Kaggle
    
    Args:
        submission_file: Path to submission CSV
    
    Returns:
        bool: True if format is valid
    """
    print("\n🔍 Verifying submission format...")
    
    try:
        df = pd.read_csv(submission_file)
        
        # Check column names
        assert list(df.columns) == ['ID', 'Value'], "❌ Columns must be ['ID', 'Value']"
        print("  ✓ Column names correct: ID, Value")
        
        # Check ID format
        assert all(df['ID'].str.startswith('dish_')), "❌ All IDs must start with 'dish_'"
        print(f"  ✓ ID format correct: {len(df):,} samples")
        
        # Check Value is numeric
        assert df['Value'].dtype in [np.float64, np.float32, np.int64, np.int32], \
            "❌ Value must be numeric type"
        print("  ✓ Value type correct: numeric")
        
        # Check for missing values
        assert df['ID'].notna().all(), "❌ Missing values in ID column"
        assert df['Value'].notna().all(), "❌ Missing values in Value column"
        print("  ✓ No missing values")
        
        # Check for duplicate IDs
        assert len(df['ID'].unique()) == len(df), "❌ Duplicate IDs found"
        print("  ✓ No duplicate IDs")
        
        # Check for negative predictions (calories should be positive)
        if (df['Value'] < 0).any():
            neg_count = (df['Value'] < 0).sum()
            print(f"  ⚠️  Warning: {neg_count} negative predictions found")
        
        print("\n✅ Submission format validation PASSED!")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Validation FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        return False


def compare_with_previous(current_file='submission.csv', previous_file=None):
    """
    Compare current predictions with previous submission (if available)
    
    Args:
        current_file: Current submission file
        previous_file: Previous submission file (optional)
    """
    if previous_file is None or not Path(previous_file).exists():
        print("\n📌 No previous submission to compare with")
        return
    
    print("\n📊 Comparing with previous submission...")
    
    current_df = pd.read_csv(current_file)
    previous_df = pd.read_csv(previous_file)
    
    # Merge on ID
    merged = current_df.merge(previous_df, on='ID', suffixes=('_current', '_previous'))
    
    # Calculate differences
    merged['diff'] = merged['Value_current'] - merged['Value_previous']
    merged['abs_diff'] = merged['diff'].abs()
    merged['pct_diff'] = (merged['diff'] / merged['Value_previous']) * 100
    
    print(f"\n  Total samples: {len(merged):,}")
    print(f"  Mean difference: {merged['diff'].mean():.2f} calories")
    print(f"  Mean absolute difference: {merged['abs_diff'].mean():.2f} calories")
    print(f"  Mean percentage change: {merged['pct_diff'].mean():.2f}%")
    print(f"  Max increase: {merged['diff'].max():.2f} calories")
    print(f"  Max decrease: {merged['diff'].min():.2f} calories")
    
    # Samples with biggest changes
    print(f"\n  Top 5 biggest increases:")
    top_increases = merged.nlargest(5, 'diff')[['ID', 'Value_previous', 'Value_current', 'diff']]
    print(top_increases.to_string(index=False))
    
    print(f"\n  Top 5 biggest decreases:")
    top_decreases = merged.nsmallest(5, 'diff')[['ID', 'Value_previous', 'Value_current', 'diff']]
    print(top_decreases.to_string(index=False))


def main():
    """Main prediction function"""
    
    # Configuration
    MODEL_PATH = 'checkpoints/best_model.pth'
    OUTPUT_FILE = 'submission.csv'
    PREVIOUS_SUBMISSION = None  # Set to previous submission file for comparison
    
    print("="*70)
    print("🔮 NUTRITION5K TEST SET PREDICTION")
    print("="*70)
    print(f"Model:  {MODEL_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    print("="*70)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Error: Model file not found: {MODEL_PATH}")
        print("Please train the model first by running: python model.py")
        return
    
    try:
        # Generate predictions
        submission_df = predict_test_set(
            model_path=MODEL_PATH,
            output_file=OUTPUT_FILE
        )
        
        # Verify format
        verify_submission_format(OUTPUT_FILE)
        
        # Compare with previous submission if available
        if PREVIOUS_SUBMISSION:
            compare_with_previous(OUTPUT_FILE, PREVIOUS_SUBMISSION)
        
        # Next steps
        print("\n" + "="*70)
        print("📤 NEXT STEPS: SUBMIT TO KAGGLE")
        print("="*70)
        print("1. Go to the Kaggle competition page")
        print("2. Click 'Submit Predictions' button")
        print(f"3. Upload {OUTPUT_FILE}")
        print("4. Check your score on the leaderboard!")
        print("="*70)
        
        print("\n✅ Prediction complete!")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()