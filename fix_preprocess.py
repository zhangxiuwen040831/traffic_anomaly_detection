#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# 配置参数
raw_data_dir = "/root/lanyun-tmp/data/raw"
processed_data_dir = "/root/lanyun-tmp/data/processed"
sample_size = 0
normal_label = 0
test_size = 0.2
random_state = 42

os.makedirs(processed_data_dir, exist_ok=True)

print("加载数据...")
train_path = os.path.join(raw_data_dir, 'UNSW_NB15_training-set.csv')

train_df = pd.read_csv(train_path)
print("训练数据形状:", train_df.shape)

df = train_df.copy()
print("数据形状:", df.shape)

if sample_size and len(df) > sample_size:
    print("采样", sample_size, "条数据...")
    df = df.sample(n=sample_size, random_state=random_state)

if 'id' in df.columns:
    df = df.drop('id', axis=1)
if 'attack_cat' in df.columns:
    df = df.drop('attack_cat', axis=1)

# 处理NaN值
print("处理前NaN值数量:", df.isna().sum().sum())
df = df.dropna()
print("处理后数据形状:", df.shape)

print("标签分布:")
print(df['label'].value_counts())

# 强制将所有非数值列转换为分类并编码
print("\n检查所有列的数据类型:")
for col in df.columns:
    if col == 'label':
        continue
    print(f"  {col}: {df[col].dtype}")

print("\n编码所有非数值列:")
for col in df.columns:
    if col == 'label':
        continue
    # 检查是否为字符串类型
    if df[col].dtype == 'object':
        print(f"  编码特征: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    # 检查是否包含字符串值
    else:
        try:
            # 尝试转换为数值
            pd.to_numeric(df[col], errors='raise')
        except:
            print(f"  编码特征 (包含字符串值): {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

print("\n编码后的数据类型:")
for col in df.columns:
    if col == 'label':
        continue
    print(f"  {col}: {df[col].dtype}")

X = df.drop('label', axis=1).values
y = df['label'].values

print("特征矩阵形状:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

print("训练集大小:", X_train.shape[0])
print("测试集大小:", X_test.shape[0])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.save(os.path.join(processed_data_dir, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(processed_data_dir, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(processed_data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(processed_data_dir, 'y_test.npy'), y_test)

with open(os.path.join(processed_data_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(processed_data_dir, 'feature_names.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(df.drop('label', axis=1).columns.tolist()))

print("✓ 预处理完成，数据已保存至:", processed_data_dir)
