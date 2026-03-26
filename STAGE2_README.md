
# 第二阶段：云 GPU 训练部署指南

## 快速开始

### 1. 连接到云 GPU 服务器

```bash
ssh -p 44410 root@qhdlink.lanyun.net
# 按提示输入密码
```

### 2. 上传文件到服务器

在本地执行（从 traffic_anomaly_detection 目录）：

```bash
# 使用 scp 上传文件
scp -P 44410 -r ./* root@qhdlink.lanyun.net:/root/lanyun-tmp/traffic_anomaly_detection/
```

或者逐个上传关键文件：

```bash
scp -P 44410 stage2_config.yaml root@qhdlink.lanyun.net:/root/lanyun-tmp/traffic_anomaly_detection/
scp -P 44410 run_stage2.py root@qhdlink.lanyun.net:/root/lanyun-tmp/traffic_anomaly_detection/
scp -P 44410 -r src/ root@qhdlink.lanyun.net:/root/lanyun-tmp/traffic_anomaly_detection/
```

### 3. 在服务器上执行

连接到服务器后：

```bash
cd /root/lanyun-tmp/traffic_anomaly_detection

# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 可选：使用虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir

# 运行第二阶段训练
python run_stage2.py --config stage2_config.yaml
```

## 文件说明

### 核心文件

- **stage2_config.yaml** - 第二阶段配置文件
- **run_stage2.py** - 第二阶段主执行脚本
- **src/models/autoencoder_enhanced.py** - 增强版 Autoencoder 模型

### 配置参数 (stage2_config.yaml)

```yaml
dataset:
  sample_size: 0              # 0=使用全量（UNSW-NB15 规模可控）
  
model:
  hidden_dims: [128, 64, 32, 64, 128]  # 网络结构
  dropout: 0.3                            # Dropout 率
  use_batchnorm: true                     # 是否使用 BatchNorm

training:
  batch_size: 256          # Batch size（可根据显存调整）
  epochs: 30               # 训练轮数
  learning_rate: 0.001     # 学习率
```

## 第二阶段特性

### 1. 增强版 MLP Autoencoder

- 网络结构：[input → 128 → 64 → 32 → 64 → 128 → output]
- 增加 Batch Normalization
- 增加 Dropout (0.3)
- 参数量：约 3-5 万（根据输入维度）

### 2. 真实数据集

- 使用 UNSW-NB15 真实网络流量数据集
- 默认使用全量（如需加速可采样）
- Flow-based 特征输入

### 3. GPU 加速

- 自动检测并使用 RTX 4090
- Batch size 256
- 支持学习率调度

### 4. 完整评估

- ROC-AUC / PR-AUC
- F1-score / Precision / Recall
- Confusion Matrix
- 可视化图表

## 输出目录结构

```
/root/lanyun-tmp/traffic_anomaly_detection/outputs_stage2/
├── logs/
├── checkpoints/
│   └── autoencoder_enhanced.pth
├── figures/
│   ├── anomaly_score_distribution.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── confusion_matrix.png
└── reports/
    └── stage2_evaluation_report.md
```

## 故障排查

### 如果显存不足

修改 stage2_config.yaml：

```yaml
training:
  batch_size: 128    # 减小到 128 或 64
  
preprocessing:
  sample_size: 30000  # 减小样本规模
```

### 如果下载失败

脚本会自动处理，但如果持续失败，可以手动下载：

```bash
cd /root/lanyun-tmp/data/raw
wget https://hf-mirror.com/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv
wget https://hf-mirror.com/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_testing-set.csv
```

### 如果 PyTorch 安装问题

使用国内镜像安装：

```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 性能预期

在 RTX 4090 上：

- 5 万样本，30 epochs：约 5-10 分钟
- ROC-AUC：预期 > 0.85
- 模型大小：约 1-2 MB

## 完成标准

✅ 使用真实数据集（UNSW-NB15）  
✅ 模型在 GPU 上训练完成  
✅ 输出完整指标（ROC-AUC, PR-AUC 等）  
✅ 生成可视化图表  
✅ 输出完整评估报告  

---

**重要提醒：**
- 第一阶段已经完成，不要重复
- 全程使用国内镜像
- 注意磁盘空间管理
- 训练完成后请查看 outputs_stage2/reports/ 目录
