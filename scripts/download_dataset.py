
#!/usr/bin/env python3
import os
import sys
import requests
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import load_config

def generate_synthetic_dataset(raw_data_dir, num_samples=5000, num_features=40):
    print("正在生成合成数据集用于演示...")
    
    np.random.seed(42)
    
    normal_samples = int(num_samples * 0.6)
    attack_samples = num_samples - normal_samples
    
    normal_data = np.random.normal(loc=0, scale=1, size=(normal_samples, num_features))
    normal_labels = np.zeros(normal_samples)
    
    attack_data = np.random.normal(loc=3, scale=2, size=(attack_samples, num_features))
    attack_labels = np.ones(attack_samples)
    
    X = np.vstack([normal_data, attack_data])
    y = np.hstack([normal_labels, attack_labels])
    
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    feature_names = [f'feature_{i}' for i in range(num_features)]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y.astype(int)
    
    train_size = int(0.7 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    os.makedirs(raw_data_dir, exist_ok=True)
    train_path = os.path.join(raw_data_dir, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(raw_data_dir, 'UNSW_NB15_testing-set.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ 合成训练集已生成: {train_path} ({len(train_df)} 样本)")
    print(f"✓ 合成测试集已生成: {test_path} ({len(test_df)} 样本)")
    print(f"  - 特征数: {num_features}")
    print(f"  - 正常样本: {np.sum(y == 0)}")
    print(f"  - 异常样本: {np.sum(y == 1)}")
    
    return True

def download_file(url: str, dest_path: str, chunk_size: int = 8192):
    if os.path.exists(dest_path):
        print(f"✓ 文件已存在，跳过下载: {dest_path}")
        return True
    
    print(f"正在下载: {url}")
    print(f"保存到: {dest_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)
        
        print(f"✓ 下载完成: {dest_path}")
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    parser = argparse.ArgumentParser(description='下载网络流量数据集')
    parser.add_argument('--config', type=str, default='config/base.yaml', help='配置文件路径')
    parser.add_argument('--override', type=str, default='config/local_cpu.yaml', help='覆盖配置文件路径')
    parser.add_argument('--synthetic', action='store_true', help='使用合成数据集')
    args = parser.parse_args()
    
    print("="*60)
    print("  网络流量异常检测系统 - 数据集下载")
    print("="*60)
    
    config = load_config(args.config, args.override)
    
    raw_data_dir = config['paths']['raw_data']
    os.makedirs(raw_data_dir, exist_ok=True)
    
    dataset_name = config['dataset']['name']
    print(f"\n数据集: {dataset_name}")
    
    train_path = os.path.join(raw_data_dir, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(raw_data_dir, 'UNSW_NB15_testing-set.csv')
    
    if args.synthetic or os.path.exists(train_path) and os.path.exists(test_path):
        if args.synthetic:
            print("使用合成数据集模式")
        else:
            print("数据集文件已存在，跳过下载")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            generate_synthetic_dataset(raw_data_dir)
        print("\n🎉 数据集准备完成！")
        return 0
    
    train_url = config['dataset']['url']
    test_url = config['dataset'].get('test_url')
    
    success = True
    
    print("\n尝试下载真实数据集...")
    if not download_file(train_url, train_path):
        success = False
    
    if test_url and not download_file(test_url, test_path):
        success = False
    
    if not success:
        print("\n真实数据集下载失败，切换到合成数据集...")
        generate_synthetic_dataset(raw_data_dir)
        success = True
    
    if success:
        print("\n🎉 数据集准备完成！")
        return 0
    else:
        print("\n⚠️  数据集准备失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
