
#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import argparse
import yaml
import logging
from datetime import datetime

LOGGER = None
LOG_FILE = None

def setup_logging(log_dir: str):
    global LOGGER, LOG_FILE
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = os.path.join(log_dir, f"stage2_{ts}.log")

    logger = logging.getLogger("stage2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    LOGGER = logger
    logger.info(f"日志文件: {LOG_FILE}")
    return logger

def print_header(title):
    if LOGGER is None:
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
        return
    LOGGER.info("="*80)
    LOGGER.info(f"{title}")
    LOGGER.info("="*80)

def run_command(cmd, description):
    logger = LOGGER
    if logger is None:
        print(f"\n{'-'*80}")
        print(f"执行: {description}")
        print(f"命令: {cmd}")
        print(f"{'-'*80}")
    else:
        logger.info("-"*80)
        logger.info(f"执行: {description}")
        logger.info(f"命令: {cmd}")
        logger.info("-"*80)

    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if logger is None:
                print(line)
            else:
                logger.info(line)
        rc = proc.wait()
        if rc == 0:
            if logger is None:
                print(f"\n✓ {description} 完成！")
            else:
                logger.info(f"✓ {description} 完成！")
            return True
        if logger is None:
            print(f"\n✗ {description} 失败！错误码: {rc}")
        else:
            logger.error(f"✗ {description} 失败！错误码: {rc}")
        return False
    except Exception as e:
        if logger is None:
            print(f"\n✗ {description} 执行异常: {e}")
        else:
            logger.exception(f"✗ {description} 执行异常: {e}")
        return False

def check_environment():
    print_header("环境检查")
    
    LOGGER.info("检查 Python 版本...") if LOGGER else print("检查 Python 版本...")
    python_version = sys.version
    (LOGGER.info(f"Python 版本: {python_version}") if LOGGER else print(f"Python 版本: {python_version}"))
    
    (LOGGER.info("检查 PyTorch 和 CUDA...") if LOGGER else print("\n检查 PyTorch 和 CUDA..."))
    try:
        import torch
        (LOGGER.info(f"PyTorch 版本: {torch.__version__}") if LOGGER else print(f"PyTorch 版本: {torch.__version__}"))
        if torch.cuda.is_available():
            (LOGGER.info("✓ CUDA 可用!") if LOGGER else print("✓ CUDA 可用!"))
            (LOGGER.info(f"  CUDA 版本: {torch.version.cuda}") if LOGGER else print(f"  CUDA 版本: {torch.version.cuda}"))
            (LOGGER.info(f"  GPU 数量: {torch.cuda.device_count()}") if LOGGER else print(f"  GPU 数量: {torch.cuda.device_count()}"))
            for i in range(torch.cuda.device_count()):
                (LOGGER.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}") if LOGGER else print(f"  GPU {i}: {torch.cuda.get_device_name(i)}"))
                props = torch.cuda.get_device_properties(i)
                (LOGGER.info(f"    显存: {props.total_memory / 1024**3:.2f} GB") if LOGGER else print(f"    显存: {props.total_memory / 1024**3:.2f} GB"))
        else:
            (LOGGER.warning("✗ CUDA 不可用，将使用 CPU") if LOGGER else print("✗ CUDA 不可用，将使用 CPU"))
    except ImportError:
        (LOGGER.warning("✗ PyTorch 未安装") if LOGGER else print("✗ PyTorch 未安装"))
    run_command("nvidia-smi", "nvidia-smi")
    
    (LOGGER.info("检查磁盘空间...") if LOGGER else print("\n检查磁盘空间..."))
    try:
        import shutil
        disk_usage = shutil.disk_usage('/')
        total_gb = disk_usage.total / (1024**3)
        free_gb = disk_usage.free / (1024**3)
        (LOGGER.info(f"磁盘总空间: {total_gb:.2f} GB") if LOGGER else print(f"磁盘总空间: {total_gb:.2f} GB"))
        (LOGGER.info(f"磁盘可用空间: {free_gb:.2f} GB") if LOGGER else print(f"磁盘可用空间: {free_gb:.2f} GB"))
    except Exception as e:
        (LOGGER.warning(f"无法检查磁盘空间: {e}") if LOGGER else print(f"无法检查磁盘空间: {e}"))
    
    return True

def install_dependencies(skip_install=False):
    if skip_install:
        print_header("跳过依赖安装")
        LOGGER.info("跳过依赖安装")
        return True
    
    print_header("安装依赖")
    
    # 更换为阿里云镜像
    pip_mirror = "https://mirrors.aliyun.com/pypi/simple/"
    
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0"
    ]
    
    cmd = f"pip install {' '.join(requirements)} -i {pip_mirror} --no-cache-dir"
    
    return run_command(cmd, "安装 Python 依赖")

def download_dataset(config):
    print_header("下载数据集")
    
    raw_data_dir = config['paths']['raw_data']
    os.makedirs(raw_data_dir, exist_ok=True)
    
    train_urls = config['dataset'].get('train_urls') or [config['dataset'].get('train_url')]
    train_urls = [u for u in train_urls if u]
    
    train_path = os.path.join(raw_data_dir, 'UNSW_NB15_training-set.csv')
    
    download_script = f"""
import os
import requests
from tqdm import tqdm

def is_valid_csv(file_path, min_lines=10):
    # 检查文件是否为有效的CSV文件
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
        return len(lines) >= min_lines
    except Exception:
        return False

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        if is_valid_csv(dest_path):
            print(f"✓ 文件已存在且有效: {{dest_path}}")
            return True
        else:
            print(f"⚠️  文件存在但无效，重新下载: {{dest_path}}")
            os.remove(dest_path)
    
    print(f"正在下载: {{url}}")
    print(f"保存到: {{dest_path}}")
    
    try:
        response = requests.get(url, stream=True, timeout=300)
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
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        if is_valid_csv(dest_path):
            print(f"✓ 下载完成: {{dest_path}}")
            return True
        else:
            print(f"✗ 下载的文件无效: {{dest_path}}")
            os.remove(dest_path)
            return False
    except Exception as e:
        print(f"✗ 下载失败: {{e}}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

success = True
train_urls = {train_urls}
for u in train_urls:
    if download_file(u, "{train_path}"):
        break
else:
    success = False

import sys
sys.exit(0 if success else 1)
"""
    
    temp_script = "temp_download.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(download_script)
    
    result = run_command(f"python {temp_script}", "下载 UNSW-NB15 数据集")
    
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    return result

def preprocess_data(config):
    print_header("数据预处理")
    
    preprocess_script = f"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

raw_data_dir = "{config['paths']['raw_data']}"
processed_data_dir = "{config['paths']['processed_data']}"
sample_size = {config['preprocessing']['sample_size']}
normal_label = {config['preprocessing']['normal_label']}
test_size = {config['preprocessing']['test_size']}
random_state = {config['preprocessing']['random_state']}

os.makedirs(processed_data_dir, exist_ok=True)

print("加载数据...")
train_path = os.path.join(raw_data_dir, 'UNSW_NB15_training-set.csv')

train_df = pd.read_csv(train_path)
print(f"训练数据形状: {{train_df.shape}}")

df = train_df.copy()
print(f"数据形状: {{df.shape}}")

if sample_size and len(df) > sample_size:
    print(f"采样 {{sample_size}} 条数据...")
    df = df.sample(n=sample_size, random_state=random_state)

if 'id' in df.columns:
    df = df.drop('id', axis=1)
if 'attack_cat' in df.columns:
    df = df.drop('attack_cat', axis=1)

# 处理NaN值
print(f"处理前NaN值数量: {{df.isna().sum().sum()}}")
df = df.dropna()
print(f"处理后数据形状: {{df.shape}}")

print(f"标签分布:")
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

print(f"分类特征: {{categorical_cols}}")
print(f"数值特征: {{len(numerical_cols)}} 个")

# 强制检查所有列，确保没有字符串类型
print('\n检查所有列的数据类型:')
for col in df.columns:
    if col == 'label':
        continue
    print(f'  {{col}}: {{df[col].dtype}}')

# 强制将所有非数值列转换为分类并编码
categorical_cols = []
numerical_cols = []
for col in df.columns:
    if col == 'label':
        continue
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        categorical_cols.append(col)
    else:
        # 检查是否有字符串值
        try:
            # 尝试转换为数值
            pd.to_numeric(df[col], errors='raise')
            numerical_cols.append(col)
        except:
            # 如果转换失败，视为分类特征
            categorical_cols.append(col)

print(f'\n最终识别 - 分类特征: {{categorical_cols}}')
print(f'最终识别 - 数值特征: {{len(numerical_cols)}} 个')

# 编码所有分类特征
label_encoders = {{}}
for col in categorical_cols:
    print(f'  编码特征: {{col}}')
    le = LabelEncoder()
    # 确保所有值都是字符串
    df[col] = df[col].astype(str)
    # 编码
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 再次检查所有列的数据类型
print('\n编码后的数据类型:')
for col in df.columns:
    if col == 'label':
        continue
    print(f'  {{col}}: {{df[col].dtype}}')

# 只对数值类型的列计算平均值
numerical_cols = [col for col in numerical_cols if pd.api.types.is_numeric_dtype(df[col])]
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

X = df.drop('label', axis=1).values
y = df['label'].values

print(f"特征矩阵形状: {{X.shape}}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

print(f"训练集大小: {{X_train.shape[0]}}")
print(f"测试集大小: {{X_test.shape[0]}}")

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
    f.write('\\n'.join(df.drop('label', axis=1).columns.tolist()))

print(f"✓ 预处理完成，数据已保存至: {{processed_data_dir}}")
"""
    
    temp_script = "temp_preprocess.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(preprocess_script)
    
    result = run_command(f"python {temp_script}", "数据预处理")
    
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    return result

def train_model(config):
    print_header("GPU 模型训练")
    
    train_script = f"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

processed_data_dir = "{config['paths']['processed_data']}"
checkpoints_dir = "{config['paths']['checkpoints']}"
hidden_dims = {config['model']['hidden_dims']}
activation = "{config['model']['activation']}"
dropout = {config['model']['dropout']}
use_batchnorm = {config['model']['use_batchnorm']}
batch_size = {config['training']['batch_size']}
epochs = {config['training']['epochs']}
learning_rate = {config['training']['learning_rate']}
use_scheduler = {config['training']['use_learning_rate_scheduler']}
early_stopping_patience = {config.get('training', {}).get('early_stopping_patience', 50)}

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
print(f"正常样本数量: {{len(X_train_normal)}}")
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

def train_once(device, current_batch_size):
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
    early_stopping_patience = early_stopping_patience

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

def train_with_auto_batch(device, init_batch_size):
    current_bs = int(init_batch_size)
    min_bs = 16
    last_err = None
    while current_bs >= min_bs:
        try:
            return train_once(device, current_bs) + (current_bs,)
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            if device.type == 'cuda' and ('out of memory' in msg or 'cuda error' in msg):
                print(f"⚠️ 训练失败 (batch_size={{current_bs}}): {{e}}")
                current_bs = current_bs // 2
                print(f"➡️ 尝试降低 batch_size 到 {{current_bs}}")
                continue
            raise
    raise RuntimeError(f"batch_size 降到 {{current_bs}} 仍失败: {{last_err}}")

gpu_name = None
if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ 使用 GPU: {{gpu_name}}")
else:
    print("⚠️ 使用 CPU")

print("创建模型并训练...")
try:
    model, train_losses, total_time, num_params, peak_mem_gb, used_batch_size = train_with_auto_batch(device, batch_size)
except Exception as e:
    if device.type == 'cuda':
        print(f"⚠️ GPU 训练失败，切换到 CPU: {{e}}")
        device = torch.device('cpu')
        model, train_losses, total_time, num_params, peak_mem_gb, used_batch_size = train_with_auto_batch(device, batch_size)
    else:
        raise

print(f"模型参数量: {{num_params:,}}")
print(model)
print(f"\\n训练完成！总耗时: {{total_time:.2f}} 秒")
if peak_mem_gb is not None:
    print(f"GPU 峰值显存占用: {{peak_mem_gb:.2f}} GB")

model_path = os.path.join(checkpoints_dir, 'autoencoder_enhanced.pth')
torch.save({{
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
}}, model_path)
print(f"模型已保存至: {{model_path}}")
"""
    
    temp_script = "temp_train.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(train_script)
    
    result = run_command(f"python3 {temp_script}", "GPU 训练增强版 Autoencoder")
    
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    return result

def evaluate_model(config):
    print_header("模型评估")
    
    eval_script = f"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

processed_data_dir = "{config['paths']['processed_data']}"
checkpoints_dir = "{config['paths']['checkpoints']}"
figures_dir = "{config['paths']['figures']}"
reports_dir = "{config['paths']['reports']}"
threshold_method = "{config['evaluation'].get('threshold_method', 'percentile')}"
percentile = {config['evaluation']['percentile']}

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("加载测试数据...")
X_test = np.load(os.path.join(processed_data_dir, 'X_test.npy'))
y_test = np.load(os.path.join(processed_data_dir, 'y_test.npy'))
print(f"测试集大小: {{len(X_test)}}")
print(f"正常样本: {{np.sum(y_test == 0)}}")
print(f"异常样本: {{np.sum(y_test == 1)}}")

print("加载训练数据 (用于阈值计算)...")
X_train = np.load(os.path.join(processed_data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(processed_data_dir, 'y_train.npy'))
X_train_normal = X_train[y_train == 0]
print(f"训练集正常样本: {{len(X_train_normal)}}")

print("加载模型...")
checkpoint = torch.load(os.path.join(checkpoints_dir, 'autoencoder_enhanced.pth'), map_location='cpu')

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

model = EnhancedMLPAutoencoder(
    input_dim=checkpoint['input_dim'],
    hidden_dims=checkpoint['hidden_dims']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

from torch.utils.data import DataLoader, TensorDataset

def compute_scores(X):
    X_tensor = torch.from_numpy(X).float()
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=8192 if device.type == 'cuda' else 4096,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    scores = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=(device.type == 'cuda'))
            recon = model(xb)
            mse = torch.mean((recon - xb) ** 2, dim=1)
            scores.append(mse.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)

print("计算训练集正常样本重构误差 (阈值用)...")
train_normal_scores = compute_scores(X_train_normal)
print(f"训练集正常平均重构误差: {{np.mean(train_normal_scores):.6f}}")

print("计算测试集异常分数...")
anomaly_scores = compute_scores(X_test)

normal_scores = anomaly_scores[y_test == 0]
abnormal_scores = anomaly_scores[y_test == 1]
print(f"正常样本平均重构误差: {{np.mean(normal_scores):.6f}}")
print(f"异常样本平均重构误差: {{np.mean(abnormal_scores):.6f}}")

print("计算评估指标...")
roc_auc = roc_auc_score(y_test, anomaly_scores)
precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
pr_auc = auc(recall, precision)

if threshold_method == 'roc_optimal':
    fpr_t, tpr_t, thr = roc_curve(y_test, anomaly_scores)
    best_idx = int(np.argmax(tpr_t - fpr_t))
    threshold = float(thr[best_idx])
    print(f"阈值方法: ROC 最优点 (Youden), threshold={{threshold:.6f}}")
else:
    threshold = float(np.percentile(train_normal_scores, percentile))
    print(f"阈值方法: 训练集正常分位数 P{{percentile}}, threshold={{threshold:.6f}}")
y_pred = (anomaly_scores > threshold).astype(int)
f1 = f1_score(y_test, y_pred)
precision_val = precision_score(y_test, y_pred, zero_division=0)
recall_val = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\\n评估结果:")
print(f"  ROC-AUC: {{roc_auc:.4f}}")
print(f"  PR-AUC: {{pr_auc:.4f}}")
print(f"  F1 Score: {{f1:.4f}}")
print(f"  Precision: {{precision_val:.4f}}")
print(f"  Recall: {{recall_val:.4f}}")
print(f"  混淆矩阵: TN={{cm[0,0]}}, FP={{cm[0,1]}}, FN={{cm[1,0]}}, TP={{cm[1,1]}}")

print("生成可视化...")

plt.figure(figsize=(10, 6))
plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
plt.hist(abnormal_scores, bins=50, alpha=0.5, label='Anomaly', density=True)
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.title('Anomaly Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figures_dir, 'anomaly_score_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 异常分数分布图已保存")

fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC-AUC = {{roc_auc:.4f}}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figures_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ ROC 曲线已保存")

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f'PR-AUC = {{pr_auc:.4f}}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figures_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ PR 曲线已保存")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred Normal', 'Pred Anomaly'],
            yticklabels=['Actual Normal', 'Actual Anomaly'])
plt.title('Confusion Matrix')
plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 混淆矩阵已保存")

report = []
report.append("="*80)
report.append("网络流量异常检测系统 - 第二阶段评估报告")
report.append("="*80)
report.append("")
report.append("1. 数据集信息:")
report.append(f"   总样本数: {{len(X_test) + len(np.load(os.path.join(processed_data_dir, 'X_train.npy')))}}")
report.append(f"   特征数: {{X_test.shape[1]}}")
report.append(f"   测试集正常样本: {{np.sum(y_test == 0)}}")
report.append(f"   测试集异常样本: {{np.sum(y_test == 1)}}")
report.append("")
report.append("2. 模型信息:")
report.append(f"   模型类型: Enhanced MLP Autoencoder")
report.append(f"   隐藏层维度: {{checkpoint['hidden_dims']}}")
report.append(f"   模型参数量: {{checkpoint['num_params']:,}}")
report.append("")
report.append("3. 训练信息:")
report.append(f"   训练耗时: {{checkpoint['train_time']:.2f}} 秒")
report.append(f"   Epochs: {{len(checkpoint['train_losses'])}}")
report.append(f"   最终损失: {{checkpoint['train_losses'][-1]:.6f}}")
report.append("")
report.append("4. 评估结果:")
report.append(f"   ROC-AUC: {{roc_auc:.4f}}")
report.append(f"   PR-AUC: {{pr_auc:.4f}}")
report.append(f"   F1 Score: {{f1:.4f}}")
report.append(f"   Precision: {{precision_val:.4f}}")
report.append(f"   Recall: {{recall_val:.4f}}")
report.append("")
report.append("5. 结论:")
if roc_auc > 0.7:
    report.append("   ✓ 模型表现良好，异常检测能力明显优于随机水平")
elif roc_auc > 0.5:
    report.append("   ⚠️  模型有一定效果，但仍有提升空间")
else:
    report.append("   ✗ 模型效果不佳，需要进一步调试")
report.append("")
report.append("="*80)

report_text = "\\n".join(report)
print(f"\\n{{report_text}}")

with open(os.path.join(reports_dir, 'stage2_evaluation_report.md'), 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"\\n评估报告已保存至: {{os.path.join(reports_dir, 'stage2_evaluation_report.md')}}")

conclusion = "模型效果不佳，需要进一步调试"
if roc_auc > 0.7:
    conclusion = "模型在真实数据上有效，异常检测能力明显优于随机水平"
elif roc_auc > 0.5:
    conclusion = "模型在真实数据上有一定效果，但仍有提升空间"

summary_lines = []
summary_lines.append("# 第二阶段训练总结 (研究级)")
summary_lines.append("")
summary_lines.append("## 数据集信息")
summary_lines.append(f"- 数据集: UNSW-NB15")
summary_lines.append(f"- 总样本数: {{len(X_train) + len(X_test)}}")
summary_lines.append(f"- 特征数: {{X_test.shape[1]}}")
summary_lines.append(f"- 训练集正常样本: {{len(X_train_normal)}}")
summary_lines.append(f"- 测试集正常样本: {{np.sum(y_test == 0)}}, 异常样本: {{np.sum(y_test == 1)}}")
summary_lines.append("")
summary_lines.append("## 模型结构")
summary_lines.append(f"- 类型: Enhanced MLP Autoencoder")
summary_lines.append(f"- 隐藏层: {{checkpoint['hidden_dims']}}")
summary_lines.append(f"- 参数量: {{checkpoint['num_params']:,}}")
summary_lines.append("")
summary_lines.append("## 训练信息")
summary_lines.append(f"- device: {{checkpoint.get('device')}}")
summary_lines.append(f"- GPU: {{checkpoint.get('gpu_name')}}")
summary_lines.append(f"- batch_size: {{checkpoint.get('batch_size')}}")
summary_lines.append(f"- 峰值显存(GB): {{checkpoint.get('peak_memory_gb')}}")
summary_lines.append(f"- epochs: {{len(checkpoint['train_losses'])}}")
summary_lines.append(f"- 训练耗时(s): {{checkpoint['train_time']:.2f}}")
summary_lines.append("")
summary_lines.append("## 评估指标")
summary_lines.append(f"- ROC-AUC: {{roc_auc:.4f}}")
summary_lines.append(f"- PR-AUC: {{pr_auc:.4f}}")
summary_lines.append(f"- F1-score: {{f1:.4f}}")
summary_lines.append(f"- Precision: {{precision_val:.4f}}")
summary_lines.append(f"- Recall: {{recall_val:.4f}}")
summary_lines.append(f"- Confusion Matrix: TN={{cm[0,0]}}, FP={{cm[0,1]}}, FN={{cm[1,0]}}, TP={{cm[1,1]}}")
summary_lines.append("")
summary_lines.append("## 结果文件")
summary_lines.append(f"- 模型: {{os.path.join(checkpoints_dir, 'autoencoder_enhanced.pth')}}")
summary_lines.append(f"- 图表目录: {{figures_dir}}")
summary_lines.append(f"- 报告: {{os.path.join(reports_dir, 'stage2_evaluation_report.md')}}")
summary_lines.append("")
summary_lines.append("## 一句话结论")
summary_lines.append(f"- {{conclusion}}")
summary_lines.append("")

summary_text = "\\n".join(summary_lines)
with open(os.path.join(reports_dir, 'summary.md'), 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"summary.md 已保存至: {{os.path.join(reports_dir, 'summary.md')}}")
"""
    
    temp_script = "temp_eval.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(eval_script)
    
    result = run_command(f"python3 {temp_script}", "模型评估与结果生成")
    
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='第二阶段云 GPU 训练')
    parser.add_argument('--config', type=str, default='stage2_config.yaml', help='配置文件路径')
    parser.add_argument('--skip-install', action='store_true', default=False, help='跳过依赖安装')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    for dir_path in config['paths'].values():
        os.makedirs(dir_path, exist_ok=True)

    setup_logging(config['paths']['logs'])

    print_header("第二阶段：云 GPU 深度训练")
    LOGGER.info("基于深度学习的网络流量异常检测系统")
    LOGGER.info(f"配置文件: {args.config}")
    LOGGER.info(f"数据目录: {config.get('data_dir')}")
    LOGGER.info(f"输出目录: {config.get('output_dir')}")
    LOGGER.info(f"跳过依赖安装: {args.skip_install}")
    
    steps = [
        ("环境检查", check_environment),
        ("安装依赖", lambda: install_dependencies(args.skip_install)),
        ("GPU 模型训练", lambda: train_model(config)),
        ("模型评估", lambda: evaluate_model(config))
    ]
    
    for desc, func in steps:
        try:
            success = func()
            if not success and desc != "环境检查":
                LOGGER.error("="*80)
                LOGGER.error("流程执行失败！")
                LOGGER.error("="*80)
                return 1
        except Exception as e:
            LOGGER.exception(f"✗ 执行失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    LOGGER.info("="*80)
    LOGGER.info("第二阶段云 GPU 训练完成！")
    LOGGER.info("="*80)
    LOGGER.info("结果文件位置:")
    LOGGER.info(f"  - 模型: {config['paths']['checkpoints']}/")
    LOGGER.info(f"  - 图表: {config['paths']['figures']}/")
    LOGGER.info(f"  - 报告: {config['paths']['reports']}/")
    LOGGER.info(f"  - 日志: {LOG_FILE}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
