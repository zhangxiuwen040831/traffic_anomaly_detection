#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 配置参数
processed_data_dir = "/root/lanyun-tmp/data/processed"
checkpoints_dir = "/root/lanyun-tmp/traffic_anomaly_detection/outputs_stage2/checkpoints"
hidden_dims = [128, 64, 32, 64, 128]
activation = "relu"
dropout = 0.3
use_batchnorm = True
batch_size = 256
epochs = 1000
learning_rate = 0.001
use_scheduler = True
early_stopping_patience = 50

os.makedirs(checkpoints_dir, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        try:
            # 测试CUDA是否真的可用
            torch.cuda.current_device()
            return torch.device('cuda')
        except:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

device = get_device()

print("加载数据...")
X_train = np.load(os.path.join(processed_data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(processed_data_dir, 'y_train.npy'))

normal_idx = y_train == 0
X_train_normal = X_train[normal_idx]
print("正常样本数量:", len(X_train_normal))
input_dim = X_train_normal.shape[1]

class EnhancedMLPAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='relu', dropout=0.3, use_batchnorm=True):
        super().__init__()
        self.input_dim = input_dim
        
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU()
        else:
            act = nn.ReLU()
        
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims[:len(hidden_dims)//2 + 1]:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(act)
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for dim in hidden_dims[len(hidden_dims)//2 + 1:]:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(act)
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

X_tensor = torch.FloatTensor(X_train_normal)
dataset = TensorDataset(X_tensor)

def build_model():
    model = EnhancedMLPAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        use_batchnorm=use_batchnorm
    )
    return model

def train_once(device, current_batch_size, early_stopping_patience):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_loader = DataLoader(
        dataset,
        batch_size=current_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )

    model.train()
    train_losses = []
    start_time = time.time()
    best_loss = float('inf')
    patience = 0

    print("\n开始训练 (" + str(epochs) + " epochs), device=" + str(device) + ", batch_size=" + str(current_batch_size) + " ...")
    print("早停设置: patience=" + str(early_stopping_patience))
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch[0].to(device, non_blocking=(device.type == 'cuda'))

            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_epoch_loss)
        
        # 早停逻辑
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("\n早停触发: " + str(patience) + "个epoch没有改进，最佳损失: " + str(best_loss)[:8])
                break
        
        if scheduler is not None:
            scheduler.step(avg_epoch_loss)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        print("Epoch [" + str(epoch+1) + "/" + str(epochs) + "], Loss: " + str(avg_epoch_loss)[:8] + ", Patience: " + str(patience))

    total_time = time.time() - start_time
    peak_mem_gb = None
    if device.type == 'cuda':
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    return model, train_losses, total_time, num_params, peak_mem_gb

def train_with_auto_batch(device, init_batch_size, early_stopping_patience):
    current_bs = int(init_batch_size)
    min_bs = 16
    last_err = None
    while current_bs >= min_bs:
        try:
            return train_once(device, current_bs, early_stopping_patience) + (current_bs,)
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            if device.type == 'cuda' and ('out of memory' in msg or 'cuda error' in msg):
                print("⚠️ 训练失败 (batch_size=" + str(current_bs) + "): " + str(e))
                current_bs = current_bs // 2
                print("➡️ 尝试降低 batch_size 到 " + str(current_bs))
                continue
            raise
    raise RuntimeError("batch_size 降到 " + str(current_bs) + " 仍失败: " + str(last_err))

gpu_name = None
if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    print("✓ 使用 GPU: " + str(gpu_name))
else:
    print("⚠️ 使用 CPU")

print("创建模型并训练...")
try:
    model, train_losses, total_time, num_params, peak_mem_gb, used_batch_size = train_with_auto_batch(device, batch_size, early_stopping_patience)
except Exception as e:
    if device.type == 'cuda':
        print("⚠️ GPU 训练失败，切换到 CPU: " + str(e))
        device = torch.device('cpu')
        model, train_losses, total_time, num_params, peak_mem_gb, used_batch_size = train_with_auto_batch(device, batch_size, early_stopping_patience)
    else:
        raise

print("模型参数量: " + str(num_params))
print(model)
print("\n训练完成！总耗时: " + str(total_time)[:6] + " 秒")
if peak_mem_gb is not None:
    print("GPU 峰值显存占用: " + str(peak_mem_gb)[:6] + " GB")

model_path = os.path.join(checkpoints_dir, 'autoencoder_enhanced.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'hidden_dims': hidden_dims,
    'train_losses': train_losses,
    'train_time': total_time,
    'num_params': num_params,
    'device': str(device),
    'gpu_name': gpu_name,
    'batch_size': used_batch_size,
    'peak_memory_gb': peak_mem_gb
}, model_path)
print("模型已保存至: " + model_path)
