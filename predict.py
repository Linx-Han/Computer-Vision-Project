# predict.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import CalorieEstimatorCNN
from data import Nutrition5kTestDataset

def predict_test_set(model_path, root_dir, output_file='submission.csv'):
    """步骤4: 生成Kaggle提交文件"""
    
    print("=" * 60)
    print("步骤4: 预测测试集")
    print("=" * 60)
    
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU")
    
    # 加载模型
    print("\n加载模型...")
    model = CalorieEstimatorCNN(use_pretrained=False).to(device)  # 不需要预训练，直接加载完整模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 已加载模型")
    print(f"  训练轮数: Epoch {checkpoint['epoch']+1}")
    print(f"  验证RMSE: {checkpoint['val_rmse']:.4f}")
    
    # 创建测试数据集
    print("\n加载测试数据...")
    test_dataset = Nutrition5kTestDataset(root_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✓ 测试集样本数: {len(test_dataset)}")
    
    # 预测
    print("\n开始预测...")
    predictions = []
    dish_ids = []
    
    with torch.no_grad():
        for rgb, depth, batch_dish_ids in tqdm(test_loader, desc='Predicting'):
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            pred_calories = model(rgb, depth)
            
            predictions.extend(pred_calories.cpu().numpy().tolist())
            dish_ids.extend(batch_dish_ids)
    
    # 创建提交文件
    print("\n生成提交文件...")
    submission_df = pd.DataFrame({
        'ID': dish_ids,
        'Value': predictions
    })
    
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    submission_df.to_csv(output_file, index=False)
    
    print(f"✓ 提交文件已保存: {output_file}")
    print(f"\n预测统计:")
    print(f"  样本数量: {len(submission_df)}")
    print(f"  最小值: {submission_df['Value'].min():.2f} 卡路里")
    print(f"  最大值: {submission_df['Value'].max():.2f} 卡路里")
    print(f"  平均值: {submission_df['Value'].mean():.2f} 卡路里")
    
    print("\n✅ 完成！现在可以提交 submission.csv 到Kaggle")
    
    return submission_df


def main():
    ROOT_DIR = '/Users/hanlinxuan/Desktop/Learning/Unimelb/2025 S2/CV/Assignment/Project/content'
    MODEL_PATH = 'checkpoints/best_model.pth'
    
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        print("请先运行 model.py 训练模型！")
        return
    
    predict_test_set(MODEL_PATH, ROOT_DIR, 'submission.csv')


if __name__ == '__main__':
    main()