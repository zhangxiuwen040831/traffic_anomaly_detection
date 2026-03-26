
#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import load_config

def load_and_merge_data(raw_data_dir):
    train_path = os.path.join(raw_data_dir, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(raw_data_dir, 'UNSW_NB15_testing-set.csv')
    
    print("正在加载训练数据...")
    train_df = pd.read_csv(train_path)
    print(f"训练数据形状: {train_df.shape}")
    
    print("正在加载测试数据...")
    test_df = pd.read_csv(test_path)
    print(f"测试数据形状: {test_df.shape}")
    
    print("合并训练和测试数据...")
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    print(f"合并后数据形状: {df.shape}")
    
    return df

def preprocess_data(df, config):
    print("\n开始数据预处理...")
    
    sample_size = config['preprocessing']['sample_size']
    normal_label = config['preprocessing']['normal_label']
    random_state = config['preprocessing']['random_state']
    
    if sample_size and len(df) > sample_size:
        print(f"采样 {sample_size} 条数据进行快速验证...")
        df = df.sample(n=sample_size, random_state=random_state)
    
    print(f"\n原始列名: {df.columns.tolist()}")
    
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        print("已删除 'id' 列")
    
    if 'attack_cat' in df.columns:
        print(f"\n攻击类别分布:")
        print(df['attack_cat'].value_counts())
        df = df.drop('attack_cat', axis=1)
        print("已删除 'attack_cat' 列")
    
    if 'label' not in df.columns:
        raise ValueError("数据中缺少 'label' 列！")
    
    print(f"\n标签分布 (0=正常, 1=异常):")
    print(df['label'].value_counts())
    
    categorical_cols = []
    numerical_cols = []
    
    for col in df.columns:
        if col == 'label':
            continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    print(f"\n分类特征: {categorical_cols}")
    print(f"数值特征: {numerical_cols}")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"标签数组形状: {y.shape}")
    
    test_size = config['preprocessing']['test_size']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n数据标准化完成")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': df.drop('label', axis=1).columns.tolist()
    }

def save_processed_data(data, processed_data_dir):
    os.makedirs(processed_data_dir, exist_ok=True)
    
    np.save(os.path.join(processed_data_dir, 'X_train.npy'), data['X_train'])
    np.save(os.path.join(processed_data_dir, 'X_test.npy'), data['X_test'])
    np.save(os.path.join(processed_data_dir, 'y_train.npy'), data['y_train'])
    np.save(os.path.join(processed_data_dir, 'y_test.npy'), data['y_test'])
    
    with open(os.path.join(processed_data_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(data['scaler'], f)
    
    with open(os.path.join(processed_data_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(data['label_encoders'], f)
    
    with open(os.path.join(processed_data_dir, 'feature_names.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(data['feature_names']))
    
    print(f"\n预处理数据已保存至: {processed_data_dir}")

def main():
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument('--config', type=str, default='config/base.yaml', help='配置文件路径')
    parser.add_argument('--override', type=str, default='config/local_cpu.yaml', help='覆盖配置文件路径')
    args = parser.parse_args()
    
    print("="*60)
    print("  网络流量异常检测系统 - 数据预处理")
    print("="*60)
    
    config = load_config(args.config, args.override)
    
    raw_data_dir = config['paths']['raw_data']
    processed_data_dir = config['paths']['processed_data']
    
    df = load_and_merge_data(raw_data_dir)
    processed_data = preprocess_data(df, config)
    save_processed_data(processed_data, processed_data_dir)
    
    print("\n🎉 数据预处理完成！")
    return 0

if __name__ == "__main__":
    sys.exit(main())
