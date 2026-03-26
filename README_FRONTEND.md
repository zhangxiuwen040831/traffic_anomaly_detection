# 前端系统使用说明

## 概述
这是一个基于Streamlit的网络流量异常检测前端演示系统，提供直观的用户界面来使用我们的异常检测模型。

## 系统要求
- Python 3.8+
- 所有依赖已在 `requirements.txt` 中列出

## 安装依赖

### 使用国内镜像（推荐）
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 启动前端

### Windows系统
```cmd
cd frontend_app
run_frontend.bat
```

### Linux/Mac系统
```bash
cd frontend_app
streamlit run app.py
```

### 直接使用Python
```bash
streamlit run frontend_app/app.py
```

启动后，系统会自动在默认浏览器中打开，通常地址为：
```
http://localhost:8501
```

## 功能特性

### 1. 数据输入
- **演示数据**: 使用内置的UNSW-NB15测试数据
- **上传CSV**: 上传您自己的CSV格式数据文件

### 2. 推理控制
- 选择阈值策略（f1_optimal / youden / pr_optimal）
- 一键运行推理
- 实时显示推理进度

### 3. 结果展示
- 总体统计（总样本数、正常/异常数量）
- 详细结果表格
- 异常分数分布图
- 预测分布饼图
- 模型性能图表（ROC曲线、PR曲线、混淆矩阵）

### 4. 侧边栏信息
- 模型加载状态
- 当前阈值策略和阈值
- 模型性能指标（ROC-AUC、PR-AUC）

## 使用流程

### 步骤1: 选择数据源
1. 在左侧选择"使用演示数据"或"上传CSV文件"
2. 如果选择上传，点击"Browse files"选择您的CSV文件

### 步骤2: 选择阈值策略
在左侧边栏选择合适的阈值策略：
- **f1_optimal**: 平衡Precision和Recall（默认）
- **youden**: Youden's J统计量
- **pr_optimal**: 高Precision模式

### 步骤3: 运行推理
点击"Run Inference"按钮开始推理

### 步骤4: 查看结果
- 查看总体统计信息
- 浏览详细结果表格
- 分析可视化图表

## CSV文件格式要求

如果您选择上传自己的CSV文件，请确保：

1. 文件格式为CSV（逗号分隔值）
2. 除最后一列外，所有列都是特征
3. 最后一列（可选）是真实标签
4. 特征维度应为40维（与模型训练时一致）

示例格式：
```
feature1,feature2,...,feature40,label
0.1,0.5,...,0.3,0
0.2,0.7,...,0.8,1
...
```

## 阈值策略说明

### f1_optimal（默认）
- 适用于大多数场景
- 平衡Precision和Recall
- 推荐用于一般异常检测任务

### youden
- 最大化敏感性和特异性的平衡
- 适用于需要同时考虑两类错误的场景

### pr_optimal
- 最大化Precision
- 减少误报（False Positives）
- 适用于对误报敏感的场景

## 常见问题

### Q: 前端无法启动？
A: 确保已安装所有依赖，尝试使用 `pip install -r requirements.txt` 重新安装。

### Q: 模型加载失败？
A: 检查 `deploy_bundle/model/` 目录是否存在且包含所有必要文件。

### Q: 推理结果不准确？
A: 确保输入数据格式正确，特征维度匹配。尝试切换不同的阈值策略。

### Q: 如何停止前端服务？
A: 在终端中按 `Ctrl + C` 即可停止服务。

### Q: 可以修改端口吗？
A: 可以，使用以下命令指定端口：
```bash
streamlit run app.py --server.port 8888
```

## 技术支持
如遇到问题，请查看：
- `README_DEPLOY.md` - 部署和模型使用说明
- `deploy_bundle/reports/final_model_summary.md` - 模型详细信息

## 系统架构
前端系统使用Streamlit框架构建，通过 `deploy_bundle/inference/model_loader.py` 加载和使用模型，实现了完整的端到端异常检测流程。
