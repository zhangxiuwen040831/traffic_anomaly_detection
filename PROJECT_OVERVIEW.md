# 网络流量异常检测系统 - 完整项目文档

## 项目概述

本项目实现了一个基于深度学习的网络流量异常检测系统，采用 Autoencoder 架构进行异常检测。系统共分为四个阶段：

1. **第一阶段**：本地 CPU 环境的最小可行验证（MVP）
2. **第二阶段**：完整训练和增强实验
3. **第三阶段**：论文级训练和评估
4. **第四阶段**：最终算法优化、模型定型、部署打包和前端系统构建

## 项目结构

```
traffic_anomaly_detection/
├── README.md                  # 项目总览
├── README_DEPLOY.md           # 部署说明
├── README_FRONTEND.md         # 前端使用说明
├── PROJECT_OVERVIEW.md        # 完整项目文档
├── requirements.txt           # 依赖文件
├── run_stage2.py              # 第二阶段执行脚本
├── run_stage4.py              # 第四阶段执行脚本
├── verify_system.py           # 系统验证脚本
├── config/                    # 配置文件
│   ├── base.yaml
│   ├── local_cpu.yaml
│   ├── stage3/                # 第三阶段配置
│   └── stage4/                # 第四阶段配置
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── stage4/processed/      # 第四阶段数据
├── deploy_bundle/             # 部署包
│   ├── model/                 # 模型文件
│   ├── inference/             # 推理模块
│   ├── reports/               # 报告
│   └── figures/               # 图表
├── frontend_app/              # 前端应用
│   ├── app.py                 # Streamlit 应用
│   └── run_frontend.bat       # 启动脚本
├── outputs/                   # 输出目录
│   ├── checkpoints/           # 模型检查点
│   ├── figures/               # 可视化图表
│   └── reports/               # 评估报告
├── outputs_stage4/            # 第四阶段输出
│   ├── final_model/           # 最终模型
│   └── final_optimization/     # 优化结果
├── scripts/                   # 脚本文件
│   ├── cloud/                 # 云环境脚本
│   ├── download_dataset.py    # 数据集下载
│   ├── preprocess.py          # 数据预处理
│   ├── train_autoencoder.py   # 模型训练
│   └── evaluate.py            # 模型评估
└── src/                       # 源代码
    ├── models/                # 模型定义
    ├── stage3/                # 第三阶段代码
    └── utils/                 # 工具函数
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

## 各阶段执行说明

### 第一阶段：本地 CPU 验证

**目标**：快速验证系统可行性

**执行命令**：
```bash
python run_local_cpu.py
```

**特点**：
- 使用 CPU 训练
- 小样本验证（3000条数据）
- 轻量模型
- 快速验证

### 第二阶段：完整训练

**目标**：完整训练和增强实验

**执行命令**：
```bash
python run_stage2.py
```

**特点**：
- 完整数据集训练
- 模型增强
- 更全面的评估

### 第三阶段：论文级训练

**目标**：达到论文级别的训练和评估

**执行命令**：
```bash
# 本地执行
python run_stage3.py

# 云端执行
# 参考 scripts/cloud/ 目录下的脚本
```

**特点**：
- 完整数据集
- 多种模型对比
- 详细的评估报告
- 图表和可视化

### 第四阶段：最终优化和部署

**目标**：完成可交付系统

**执行命令**：
```bash
python run_stage4.py --mode full --dataset cic_ids2017
```

**特点**：
- 算法优化（Hybrid Anomaly Score）
- 模型定型
- 部署打包
- 前端系统构建

## 模型架构

### 基础 Autoencoder
- **输入层**：根据数据集特征维度
- **编码器**：多层神经网络，逐渐降维
- **解码器**：多层神经网络，逐渐升维
- **输出层**：与输入层维度相同

### Enhanced MLP Autoencoder
- **输入维度**：36（CIC-IDS2017）/ 40（UNSW-NB15）
- **编码器**：128 → 64 → 32（latent）
- **解码器**：32 → 64 → 128 → 输入维度
- **激活函数**：ReLU
- **批量归一化**：Yes
- **Dropout**：0.2

### VAE（变分自编码器）
- **输入维度**：40（UNSW-NB15）
- **编码器**：128 → 64 → 16（latent）
- **解码器**：16 → 64 → 128 → 40
- **激活函数**：ReLU
- **Dropout**：0.1
- **Beta**：0.05

## 评分方法

### 1. 重构误差（Reconstruction Error）
- 计算输入和重构输出之间的差异
- 常用 MSE 或 MAE

### 2. 混合评分（Hybrid Score）
- **组件**：
  - 重构误差
  - 潜在空间距离
  - 密度估计
  - 加权特征误差
- **权重**：可配置

### 3. 阈值策略
- **f1_optimal**：最大化 F1 分数
- **youden**：Youden's J 统计量
- **pr_optimal**：最大化 PR 曲线下面积

## 数据集

### 1. UNSW-NB15
- **特征维度**：40
- **类别**：正常流量和多种攻击类型
- **样本数**：约 25 万条

### 2. CIC-IDS2017
- **特征维度**：36
- **类别**：正常流量和多种攻击类型
- **样本数**：约 280 万条

## 性能指标

### 最终模型性能（CIC-IDS2017）

| 指标 | 值 |
|------|-----|
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |
| F1 Score | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |

### 阈值策略

| 策略 | 阈值 | F1 | Precision | Recall |
|------|------|----|-----------|--------|
| f1_optimal | 7.1842 | 1.0 | 1.0 | 1.0 |
| youden | 7.1842 | 1.0 | 1.0 | 1.0 |
| pr_optimal | 7.1842 | 1.0 | 1.0 | 1.0 |

## 部署说明

### 部署包结构

```
deploy_bundle/
├── model/                    # 模型文件
│   ├── best.ckpt            # 最佳模型检查点
│   ├── model_config.yaml    # 模型配置
│   ├── threshold_config.yaml # 阈值配置
│   └── preprocessing.json   # 预处理信息
├── inference/                # 推理模块
│   ├── model_loader.py      # 模型加载器
│   ├── infer.py             # 单样本推理
│   └── batch_infer.py       # 批量推理
├── reports/                  # 报告
│   ├── final_model_summary.md
│   └── metrics.json
└── figures/                  # 图表
    ├── roc.png
    ├── pr.png
    ├── score_dist.png
    └── confusion.png
```

### 推理使用

```python
from deploy_bundle.inference.model_loader import TrafficAnomalyModel

# 加载模型
model = TrafficAnomalyModel("deploy_bundle/model")

# 准备数据
import numpy as np
features = np.random.randn(10, 36)  # 10个样本，36个特征

# 预测
result = model.predict(features, threshold_method="f1_optimal")

# 获取结果
print("Predictions:", result['predictions'])
print("Scores:", result['scores'])
print("Threshold:", result['threshold'])
```

## 前端系统

### 技术栈
- **框架**：Streamlit
- **语言**：Python
- **依赖**：pandas, numpy, matplotlib, seaborn

### 功能特性
- **数据输入**：上传 CSV 或使用演示数据
- **推理控制**：选择阈值策略
- **结果展示**：表格、图表、统计信息
- **可视化**：分数分布、ROC/PR 曲线、混淆矩阵

### 启动方式

```bash
# Windows
cd frontend_app
run_frontend.bat

# Linux/Mac
cd frontend_app
streamlit run app.py
```

## 系统验证

### 验证脚本

```bash
python verify_system.py
```

**验证内容**：
- 部署包目录结构
- 前端应用
- 文档
- 模型加载器导入

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型目录是否存在
   - 确认所有必要文件都在正确位置
   - 检查 PyTorch 版本兼容性

2. **推理结果异常**
   - 确认输入数据格式正确
   - 检查特征维度是否匹配
   - 确保已调用 `fit_scorer()` 方法

3. **前端启动失败**
   - 确保已安装所有依赖
   - 检查端口是否被占用
   - 查看终端错误信息

4. **数据下载失败**
   - 检查网络连接
   - 尝试手动下载数据集到 `data/raw/` 目录

## 未来工作计划

### 模型优化
- 尝试 Transformer-based 模型
- 探索自监督学习方法
- 集成多模型融合

### 部署优化
- 模型量化和压缩
- 容器化部署
- 实时推理优化

### 功能扩展
- 实时流量监控
- 攻击类型识别
- 异常趋势分析
- 多数据源集成

## 依赖版本

- torch >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pyyaml >= 6.0
- tqdm >= 4.65.0
- streamlit >= 1.20.0

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题，请通过项目 Issues 反馈。
