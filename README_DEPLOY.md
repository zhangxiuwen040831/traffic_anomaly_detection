# 部署说明文档

## 项目概述
这是一个网络流量异常检测系统，使用增强的MLP自动编码器（enhanced_mlp_ae）在UNSW-NB15数据集上训练。

## 目录结构
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

## 安装依赖

### 使用国内镜像安装（推荐）
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 主要依赖
- torch >= 2.0.0
- numpy >= 1.24.0
- pandas >= 1.5.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- pyyaml >= 6.0

## 模型性能

### 最终主模型
- **模型**: enhanced_mlp_ae
- **数据集**: UNSW-NB15
- **评分方法**: reconstruction
- **默认阈值策略**: f1_optimal

### 性能指标
| 指标 | 值 |
|------|-----|
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |
| F1 Score | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |

### 阈值策略
系统支持三种阈值策略：

1. **f1_optimal** (默认): 最大化F1分数
2. **youden**: Youden's J统计量
3. **pr_optimal**: 最大化PR曲线下面积

## 使用方法

### 1. 单样本推理
```bash
python deploy_bundle/inference/infer.py --model-dir deploy_bundle/model --input <input_file>
```

### 2. 批量推理
```bash
python deploy_bundle/inference/batch_infer.py --model-dir deploy_bundle/model --input-dir <input_dir> --output-dir <output_dir>
```

### 3. 在Python代码中使用
```python
from deploy_bundle.inference.model_loader import TrafficAnomalyModel

# 加载模型
model = TrafficAnomalyModel("deploy_bundle/model")

# 准备数据
import numpy as np
features = np.random.randn(10, 40)  # 10个样本，40个特征

# 首先需要拟合评分器（使用正常数据）
normal_data = features[:5]  # 前5个作为正常数据
model.fit_scorer(normal_data)

# 预测
result = model.predict(features, threshold_method="f1_optimal")

# 获取结果
print("Predictions:", result['predictions'])
print("Scores:", result['scores'])
print("Threshold:", result['threshold'])
```

## 切换阈值策略

### 在代码中切换
```python
# 使用高Precision模式
result = model.predict(features, threshold_method="pr_optimal")

# 使用Youden方法
result = model.predict(features, threshold_method="youden")
```

## 替换模型

### 1. 准备新模型
将新的模型文件放置在 `deploy_bundle/model/` 目录下，确保包含：
- `best.ckpt`: 模型检查点
- `model_config.yaml`: 模型配置
- `threshold_config.yaml`: 阈值配置
- `preprocessing.json`: 预处理信息

### 2. 更新配置
根据新模型的要求更新配置文件。

## 故障排除

### 模型加载失败
- 检查模型目录是否存在
- 确认所有必要文件都在正确位置
- 检查PyTorch版本兼容性

### 推理结果异常
- 确认输入数据格式正确
- 检查特征维度是否匹配（应为40维）
- 确保已调用 `fit_scorer()` 方法

## 技术支持
如有问题，请查看 `deploy_bundle/reports/final_model_summary.md` 获取更多详细信息。
