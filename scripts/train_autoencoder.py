
#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import load_config
from src.models.autoencoder import MLPAutoencoder, count_parameters

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ 使用 CPU 设备")
    return device

def load_processed_data(processed_data_dir):
    print("正在加载预处理数据...")
    X_train = np.load(os.path.join(processed_data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(processed_data_dir, 'y_train.npy'))
    
    normal_idx = y_train == 0
    X_train_normal = X_train[normal_idx]
    
    print(f"正常样本数量: {len(X_train_normal)}")
    print(f"特征维度: {X_train_normal.shape[1]}")
    
    return X_train_normal

def train_model(model, train_loader, device, config):
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    optimizer_name = config['training']['optimizer']
    loss_name = config['training']['loss']
    
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.MSELoss()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n训练配置:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  学习率: {learning_rate}")
    print(f"  优化器: {optimizer_name}")
    print(f"  损失函数: {loss_name}")
    
    model.to(device)
    model.train()
    
    train_losses = []
    start_time = time.time()
    
    print("\n开始训练...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], 平均损失: {avg_epoch_loss:.6f}")
    
    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time:.2f} 秒")
    
    return model, train_losses, total_time

def save_model(model, checkpoints_dir):
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_path = os.path.join(checkpoints_dir, 'autoencoder.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存至: {model_path}")

def main():
    parser = argparse.ArgumentParser(description='训练 Autoencoder 模型')
    parser.add_argument('--config', type=str, default='config/base.yaml', help='配置文件路径')
    parser.add_argument('--override', type=str, default='config/local_cpu.yaml', help='覆盖配置文件路径')
    args = parser.parse_args()
    
    print("="*60)
    print("  网络流量异常检测系统 - 模型训练")
    print("="*60)
    
    config = load_config(args.config, args.override)
    
    device = get_device()
    
    processed_data_dir = config['paths']['processed_data']
    checkpoints_dir = config['paths']['checkpoints']
    
    X_train_normal = load_processed_data(processed_data_dir)
    
    input_dim = X_train_normal.shape[1]
    hidden_dims = config['model']['hidden_dims']
    activation = config['model']['activation']
    dropout = config['model']['dropout']
    
    print(f"\n创建 Autoencoder 模型...")
    model = MLPAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout
    )
    
    num_params = count_parameters(model)
    print(f"模型参数量: {num_params:,}")
    print(f"模型结构:\n{model}")
    
    X_tensor = torch.FloatTensor(X_train_normal)
    dataset = TensorDataset(X_tensor)
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, train_losses, train_time = train_model(model, train_loader, device, config)
    
    save_model(model, checkpoints_dir)
    
    train_info = {
        'train_losses': train_losses,
        'train_time': train_time,
        'num_params': num_params,
        'input_dim': input_dim
    }
    
    np.save(os.path.join(checkpoints_dir, 'train_info.npy'), train_info)
    
    print("\n🎉 模型训练完成！")
    return 0

if __name__ == "__main__":
    sys.exit(main())
