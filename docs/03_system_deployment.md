# 网络流量异常检测系统 - 系统部署

## 1. 部署包结构

### 1.1 目录结构

```
deploy_bundle/
  ├── model/              # 模型文件
  │   ├── best.ckpt            # 最佳模型检查点
  │   ├── model_config.yaml     # 模型配置
  │   ├── threshold_config.yaml # 阈值配置
  │   └── preprocessing.json    # 预处理信息
  ├── inference/          # 推理模块
  │   ├── model_loader.py      # 模型加载器
  │   ├── infer.py             # 单样本推理脚本
  │   └── batch_infer.py        # 批量推理脚本
  ├── reports/            # 报告
  │   └── metrics.json         # 性能指标
  └── figures/            # 图表
      ├── confusion.png        # 混淆矩阵
      ├── pr.png               # PR曲线
      ├── roc.png              # ROC曲线
      └── score_dist.png       # 分数分布
```

### 1.2 核心文件说明

- **model/best.ckpt**：训练好的模型权重文件
- **model/model_config.yaml**：模型配置信息，包括模型结构、参数等
- **model/threshold_config.yaml**：阈值配置，包括不同阈值策略的阈值值
- **model/preprocessing.json**：预处理信息，包括特征名称、归一化参数等
- **inference/model_loader.py**：模型加载器，负责加载模型和处理推理
- **inference/infer.py**：单样本推理脚本，用于对单个样本进行异常检测
- **inference/batch_infer.py**：批量推理脚本，用于对批量样本进行异常检测

## 2. 模型加载与推理

### 2.1 模型加载

```python
from deploy_bundle.inference.model_loader import TrafficAnomalyModel

# 加载模型
model = TrafficAnomalyModel("deploy_bundle/model")
```

### 2.2 单样本推理

```python
# 准备特征数据
features = np.array([[...]])  # 输入特征

# 推理
result = model.predict(features, threshold_method="f1_optimal")

# 结果
print("预测结果:", result["predictions"])
print("异常分数:", result["scores"])
print("使用的阈值:", result["threshold"])
```

### 2.3 批量推理

```bash
python deploy_bundle/inference/batch_infer.py --model-dir deploy_bundle/model --input data/processed/X_test.npy --output predictions.npy
```

## 3. 前端界面

### 3.1 前端功能

- **数据上传**：支持上传CSV文件进行异常检测
- **实时推理**：对输入数据进行实时异常检测
- **结果可视化**：展示异常分数分布、预测结果饼图等
- **模型性能**：展示模型在训练数据上的性能指标
- **阈值选择**：支持选择不同的阈值策略

### 3.2 启动前端

```bash
# 进入前端目录
cd frontend_app

# 启动前端
run_frontend.bat
```

### 3.3 前端界面说明

- **侧边栏**：显示模型信息、阈值选择、性能指标
- **主界面**：数据上传区域、推理结果展示
- **结果区域**：显示详细的检测结果、可视化图表

### 3.4 前端技术栈

- **Streamlit**：用于构建交互式Web应用
- **Matplotlib**：用于绘制可视化图表
- **Seaborn**：用于绘制统计图表
- **NumPy**：用于数据处理
- **Pandas**：用于数据处理和展示

## 4. 部署步骤

### 4.1 环境准备

1. **创建虚拟环境**

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. **安装依赖**

```bash
# 使用清华镜像安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.2 模型部署

1. **生成部署包**

```bash
# 运行第四阶段脚本生成部署包
python run_stage4.py --mode deploy
```

2. **验证部署包**

```bash
# 验证系统完整性
python verify_system.py
```

### 4.3 系统运行

1. **启动前端**

```bash
# 启动前端应用
cd frontend_app
run_frontend.bat
```

2. **访问前端**

打开浏览器，访问 `http://localhost:8501`

## 5. 部署配置

### 5.1 配置文件

- **config/stage4/base.yaml**：第四阶段配置文件，包含模型配置、训练参数等
- **deploy_bundle/model/model_config.yaml**：部署模型的配置文件

### 5.2 环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| PYTHONPATH | Python模块搜索路径 | 项目根目录 |
| DEPLOY_BUNDLE_PATH | 部署包路径 | deploy_bundle |
| FRONTEND_PORT | 前端端口 | 8501 |

## 6. 系统集成

### 6.1 与其他系统集成

- **API接口**：可以通过推理脚本提供API接口
- **日志集成**：可以将检测结果写入日志系统
- **告警集成**：可以与告警系统集成，当检测到异常时触发告警

### 6.2 监控与维护

- **模型监控**：定期评估模型性能
- **数据监控**：监控输入数据的质量
- **系统监控**：监控系统资源使用情况

## 7. 故障排除

### 7.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 模型加载失败 | 模型文件不存在 | 确保deploy_bundle/model目录存在模型文件 |
| 推理失败 | 输入特征维度不匹配 | 确保输入特征维度与模型期望一致 |
| 前端启动失败 | 端口被占用 | 修改前端端口或关闭占用端口的进程 |
| 性能下降 | 数据分布变化 | 重新训练模型或更新阈值 |

### 7.2 日志与调试

- **前端日志**：Streamlit运行日志
- **推理日志**：推理脚本输出日志
- **系统日志**：系统运行日志

## 8. 性能优化

### 8.1 推理优化

- **批处理**：使用批量推理提高效率
- **模型量化**：对模型进行量化，减少内存使用
- **硬件加速**：使用GPU加速推理

### 8.2 前端优化

- **数据缓存**：缓存常用数据
- **懒加载**：按需加载数据
- **异步处理**：使用异步处理提高响应速度

## 9. 安全考虑

### 9.1 数据安全

- **数据加密**：对敏感数据进行加密
- **访问控制**：限制对系统的访问
- **数据脱敏**：对敏感信息进行脱敏处理

### 9.2 模型安全

- **模型保护**：保护模型权重不被篡改
- **输入验证**：验证输入数据的合法性
- **异常检测**：检测异常输入

## 10. 部署最佳实践

1. **环境隔离**：使用虚拟环境隔离依赖
2. **版本控制**：对部署包进行版本控制
3. **备份策略**：定期备份模型和配置文件
4. **监控告警**：设置监控和告警机制
5. **文档维护**：保持部署文档的更新

## 11. 系统演示

### 11.1 演示步骤

1. **启动前端**：运行 `run_frontend.bat`
2. **上传数据**：在前端界面上传测试数据
3. **选择阈值**：选择合适的阈值策略
4. **运行推理**：点击"Run Inference"按钮
5. **查看结果**：查看检测结果和可视化图表

### 11.2 演示数据

- **示例数据**：可以使用 `data/processed/X_test.npy` 作为演示数据
- **自定义数据**：可以上传自己的CSV文件进行测试

## 12. 总结

本系统已经实现了完整的部署和前端界面，能够方便地进行网络流量异常检测。系统具有良好的可扩展性和可维护性，可以根据实际需求进行定制和优化。

通过部署包和前端界面，用户可以直观地使用系统进行异常检测，无需深入了解底层实现细节。同时，系统的模块化设计使得后续的维护和升级变得更加容易。