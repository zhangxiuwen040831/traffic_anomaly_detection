
# 基于深度学习的网络流量异常检测系统

## 项目概述

本项目实现了一个基于深度学习的网络流量异常检测系统，采用 Autoencoder 架构进行异常检测。系统分为两个阶段：
1. **第一阶段**：本地 CPU 环境的最小可行验证（MVP）
2. **第二阶段**：云 GPU 环境的完整训练和增强实验（待确认后执行）

## 项目结构

```
traffic_anomaly_detection/
├── README.md
├── requirements.txt
├── environment_check.py
├── run_local_cpu.py
├── config/
│   ├── base.yaml
│   └── local_cpu.yaml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── download_dataset.py
│   ├── preprocess.py
│   ├── train_autoencoder.py
│   └── evaluate.py
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   │   └── autoencoder.py
│   ├── training/
│   └── utils/
│       └── config.py
└── outputs/
    ├── logs/
    ├── checkpoints/
    ├── figures/
    └── reports/
```

## 环境安装

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖（使用国内镜像）

```bash
# 使用清华镜像安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 备用：阿里云镜像
# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 备用：中科大镜像
# pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

### 3. 验证环境

```bash
python environment_check.py
```

## 快速开始（第一阶段）

### 一键运行完整流程

```bash
python run_local_cpu.py
```

### 分步执行

#### 1. 环境检查

```bash
python environment_check.py
```

#### 2. 下载数据集

```bash
python scripts/download_dataset.py
```

数据集说明：
- 使用 UNSW-NB15 数据集
- 从 GitHub 镜像下载
- 包含训练集和测试集

#### 3. 数据预处理

```bash
python scripts/preprocess.py
```

处理内容：
- 合并训练和测试数据
- 采样 3000 条进行快速验证
- 编码分类特征
- 标准化数值特征
- 划分训练集和测试集

#### 4. 训练模型

```bash
python scripts/train_autoencoder.py
```

训练配置（本地 CPU）：
- 模型：MLP Autoencoder
- 隐藏层：[32, 16, 8, 16, 32]
- Epochs：5
- Batch Size：32
- 学习率：0.001

#### 5. 模型评估

```bash
python scripts/evaluate.py
```

评估指标：
- ROC-AUC
- PR-AUC
- F1 Score
- Precision & Recall
- 混淆矩阵

生成的可视化：
- 异常分数分布图
- ROC 曲线
- PR 曲线
- 混淆矩阵图

## 配置说明

### 基础配置（config/base.yaml）

完整配置，包含所有参数。

### 本地 CPU 配置（config/local_cpu.yaml）

轻量级配置，适合 CPU 快速验证：
- 更小的样本量
- 更小的模型
- 更少的训练轮数

## 输出文件

所有输出文件保存在 `outputs/` 目录下：

```
outputs/
├── logs/
│   └── environment_check_report.txt    # 环境检查报告
├── checkpoints/
│   ├── autoencoder.pth                  # 训练好的模型
│   └── train_info.npy                   # 训练信息
├── figures/
│   ├── anomaly_score_distribution.png   # 异常分数分布
│   ├── roc_curve.png                    # ROC 曲线
│   ├── pr_curve.png                     # PR 曲线
│   └── confusion_matrix.png             # 混淆矩阵
└── reports/
    └── evaluation_report.md             # 评估报告
```

## 第二阶段计划（待确认）

第一阶段完成后，建议的第二阶段工作包括：

1. **数据增强**
   - 使用完整数据集
   - 增加数据增强策略

2. **模型升级**
   - 尝试 Transformer Autoencoder
   - 尝试 TabTransformer
   - 增加模型容量

3. **训练优化**
   - 使用云 GPU 加速
   - 增加训练轮数
   - 超参数搜索

4. **评估增强**
   - 更全面的评估指标
   - 交叉验证
   - 消融实验

5. **部署准备**
   - 模型导出
   - API 接口设计
   - 性能优化

## 注意事项

1. **第一阶段限制**
   - 只使用 CPU
   - 小样本验证
   - 轻量模型
   - 快速验证可行性

2. **网络问题**
   - 如 GitHub 下载失败，可手动下载数据集到 `data/raw/`
   - pip 安装失败时尝试其他镜像源

3. **资源限制**
   - 确保有至少 4GB 可用内存
   - 确保有至少 2GB 磁盘空间

## 依赖版本

- torch >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pyyaml >= 6.0
- tqdm >= 4.65.0
- requests >= 2.31.0

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题，请通过项目 Issues 反馈。
