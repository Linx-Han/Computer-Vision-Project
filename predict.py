# predict.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import CalorieEstimatorCNN
from data import Nutrition5kTestDataset

def predict_test_set(model_path, root_dir, output_file='submission.csv'):
    """
    在测试集上预测并生成Kaggle提交文件
    
    Args:
        model_path: 最佳模型路径
        root_dir: 数据根目录
        output_file: 输出CSV文件名
    """
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用 Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ 使用 NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU")
    
    print(f"设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    model = CalorieEstimatorCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 已加载模型")
    print(f"  训练轮数: Epoch {checkpoint['epoch']+1}")
    print(f"  验证Loss: {checkpoint['val_loss']:.4f}")
    print(f"  验证RMSE: {checkpoint['val_rmse']:.4f}")
    
    # 创建测试数据集
    print("\n加载测试数据...")
    test_dataset = Nutrition5kTestDataset(root_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"✓ 测试集样本数: {len(test_dataset)}")
    
    # 预测
    print("\n开始预测...")
    predictions = []
    dish_ids = []
    
    with torch.no_grad():
        for rgb, depth, batch_dish_ids in tqdm(test_loader, desc='Predicting'):
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            # 预测
            pred_calories = model(rgb, depth)
            
            # 收集结果
            predictions.extend(pred_calories.cpu().numpy().tolist())
            dish_ids.extend(batch_dish_ids)
    
    # 创建提交文件
    print("\n生成提交文件...")
    submission_df = pd.DataFrame({
        'ID': dish_ids,
        'Value': predictions
    })
    
    # 排序（确保顺序正确）
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    
    # 保存
    submission_df.to_csv(output_file, index=False)
    print(f"✓ 提交文件已保存: {output_file}")
    
    # 显示预览
    print("\n" + "="*60)
    print("预测结果预览 (前10条):")
    print("="*60)
    print(submission_df.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("预测统计:")
    print("="*60)
    print(f"  样本数量: {len(submission_df)}")
    print(f"  最小值: {submission_df['Value'].min():.2f} 卡路里")
    print(f"  最大值: {submission_df['Value'].max():.2f} 卡路里")
    print(f"  平均值: {submission_df['Value'].mean():.2f} 卡路里")
    print(f"  中位数: {submission_df['Value'].median():.2f} 卡路里")
    print(f"  标准差: {submission_df['Value'].std():.2f}")
    
    # 卡路里分布
    print(f"\n卡路里分布:")
    print(f"  0-100:   {len(submission_df[submission_df['Value'] < 100]):4d} 样本 ({len(submission_df[submission_df['Value'] < 100])/len(submission_df)*100:.1f}%)")
    print(f"  100-200: {len(submission_df[(submission_df['Value'] >= 100) & (submission_df['Value'] < 200)]):4d} 样本 ({len(submission_df[(submission_df['Value'] >= 100) & (submission_df['Value'] < 200)])/len(submission_df)*100:.1f}%)")
    print(f"  200-300: {len(submission_df[(submission_df['Value'] >= 200) & (submission_df['Value'] < 300)]):4d} 样本 ({len(submission_df[(submission_df['Value'] >= 200) & (submission_df['Value'] < 300)])/len(submission_df)*100:.1f}%)")
    print(f"  300-400: {len(submission_df[(submission_df['Value'] >= 300) & (submission_df['Value'] < 400)]):4d} 样本 ({len(submission_df[(submission_df['Value'] >= 300) & (submission_df['Value'] < 400)])/len(submission_df)*100:.1f}%)")
    print(f"  400+:    {len(submission_df[submission_df['Value'] >= 400]):4d} 样本 ({len(submission_df[submission_df['Value'] >= 400])/len(submission_df)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ 完成！现在可以提交 submission.csv 到Kaggle")
    print("="*60)
    
    return submission_df


def verify_submission_format(submission_file='submission.csv'):
    """验证提交文件格式是否正确"""
    print("\n验证提交文件格式...")
    
    try:
        df = pd.read_csv(submission_file)
        
        # 检查列名
        assert list(df.columns) == ['ID', 'Value'], "列名必须是 ['ID', 'Value']"
        print("✓ 列名正确: ID, Value")
        
        # 检查ID格式
        assert all(df['ID'].str.startswith('dish_')), "所有ID必须以 'dish_' 开头"
        print(f"✓ ID格式正确: {len(df)} 个样本")
        
        # 检查Value是数值
        assert df['Value'].dtype in [np.float64, np.float32, np.int64, np.int32], "Value必须是数值类型"
        print("✓ Value类型正确: 数值")
        
        # 检查是否有缺失值
        assert df['ID'].notna().all(), "ID列有缺失值"
        assert df['Value'].notna().all(), "Value列有缺失值"
        print("✓ 无缺失值")
        
        # 检查是否有重复ID
        assert len(df['ID'].unique()) == len(df), "有重复的ID"
        print("✓ 无重复ID")
        
        print("\n✅ 提交文件格式验证通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        return False


def main():
    # 设置路径
    ROOT_DIR = '/Users/hanlinxuan/Desktop/Learning/Unimelb/2025 S2/CV/Assignment/Project/content'
    MODEL_PATH = 'checkpoints/best_model.pth'
    OUTPUT_FILE = 'submission.csv'
    
    # 检查模型文件是否存在
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        print("请先运行 model.py 训练模型！")
        return
    
    # 生成预测
    submission_df = predict_test_set(
        model_path=MODEL_PATH,
        root_dir=ROOT_DIR,
        output_file=OUTPUT_FILE
    )
    
    # 验证格式
    verify_submission_format(OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("下一步: 提交到Kaggle")
    print("="*60)
    print("1. 登录 Kaggle 竞赛页面")
    print("2. 点击 'Submit Predictions' 按钮")
    print("3. 上传 submission.csv 文件")
    print("4. 查看你的分数！")
    print("="*60)


if __name__ == '__main__':
    main()