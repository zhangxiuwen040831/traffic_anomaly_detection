# 网络流量异常检测系统 - 仓库结构

## 1. 目录结构

```
traffic_anomaly_detection/
├── README.md               # 项目主说明文件
├── README_V2_UPDATE.md     # README更新说明
├── CHANGELOG.md            # 版本变更记录
├── RELEASE_NOTES_V2.0.md   # v2.0版本说明
├── LICENSE                 # 许可证文件
├── requirements.txt        # 依赖包列表
├── .gitignore              # Git忽略文件
├── config/                 # 配置文件目录
│   ├── stage3/             # 第三阶段配置
│   ├── stage4/             # 第四阶段配置
│   ├── base.yaml           # 基础配置
│   └── local_cpu.yaml      # 本地CPU配置
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   ├── stage3/             # 第三阶段数据
│   └── stage4/             # 第四阶段数据
├── src/                    # 源代码目录
│   ├── models/             # 模型定义
│   ├── stage3/             # 第三阶段代码
│   ├── utils/              # 工具函数
│   └── __init__.py         # 包初始化文件
├── scripts/                # 脚本目录
│   ├── cloud/              # 云端相关脚本
│   ├── download_dataset.py # 数据集下载脚本
│   ├── preprocess.py       # 数据预处理脚本
│   ├── train_autoencoder.py # 模型训练脚本
│   └── evaluate.py         # 模型评估脚本
├── deploy_bundle/          # 部署包目录
│   ├── model/              # 模型文件
│   ├── inference/          # 推理模块
│   ├── reports/            # 报告
│   └── figures/            # 图表
├── frontend_app/           # 前端应用目录
│   ├── app.py              # 前端应用代码
│   └── run_frontend.bat    # 前端启动脚本
├── docs/                   # 文档目录
│   ├── 00_project_overview.md          # 项目概览
│   ├── 01_stage_history.md             # 阶段历史
│   ├── 02_experiment_results.md        # 实验结果
│   ├── 03_system_deployment.md         # 系统部署
│   ├── 04_audit_and_trust.md           # 审计与可信度
│   ├── 05_defense_materials.md         # 答辩材料
│   ├── 06_limitations_and_future_work.md # 局限性与未来工作
│   └── 07_repo_structure.md            # 仓库结构
├── outputs/                # 输出目录
│   ├── checkpoints/        # 模型检查点
│   ├── figures/            # 图表
│   └── reports/            # 报告
├── audit/                  # 审计相关文件
├── defense/                # 答辩材料
└── examples/               # 示例代码
```

## 2. 目录说明

### 2.1 根目录文件

- **README.md**：项目主说明文件，包含项目简介、功能、使用方法等
- **README_V2_UPDATE.md**：README更新说明，对比旧版README的变化
- **CHANGELOG.md**：版本变更记录，记录各版本的主要变更
- **RELEASE_NOTES_V2.0.md**：v2.0版本说明，详细介绍v2.0版本的特性和改进
- **LICENSE**：许可证文件，说明项目的开源许可证
- **requirements.txt**：依赖包列表，列出项目所需的Python包
- **.gitignore**：Git忽略文件，指定Git应忽略的文件和目录

### 2.2 config/ 目录

- **stage3/**：第三阶段配置文件，包含各种实验配置
- **stage4/**：第四阶段配置文件，包含部署和前端配置
- **base.yaml**：基础配置文件，包含通用配置参数
- **local_cpu.yaml**：本地CPU配置，适合在CPU环境下运行

### 2.3 data/ 目录

- **raw/**：原始数据目录，存放未处理的原始数据集
- **processed/**：处理后的数据目录，存放预处理后的数据集
- **stage3/**：第三阶段数据目录，存放第三阶段的实验数据
- **stage4/**：第四阶段数据目录，存放第四阶段的实验数据

### 2.4 src/ 目录

- **models/**：模型定义目录，包含各种模型的实现
- **stage3/**：第三阶段代码目录，包含第三阶段的核心代码
- **utils/**：工具函数目录，包含通用工具函数
- **__init__.py**：包初始化文件

### 2.5 scripts/ 目录

- **cloud/**：云端相关脚本，用于在云端环境运行实验
- **download_dataset.py**：数据集下载脚本，用于下载实验数据集
- **preprocess.py**：数据预处理脚本，用于预处理原始数据
- **train_autoencoder.py**：模型训练脚本，用于训练Autoencoder模型
- **evaluate.py**：模型评估脚本，用于评估模型性能

### 2.6 deploy_bundle/ 目录

- **model/**：模型文件目录，包含训练好的模型和配置
- **inference/**：推理模块目录，包含模型加载和推理代码
- **reports/**：报告目录，包含模型性能报告
- **figures/**：图表目录，包含性能可视化图表

### 2.7 frontend_app/ 目录

- **app.py**：前端应用代码，基于Streamlit构建的交互式界面
- **run_frontend.bat**：前端启动脚本，用于启动前端应用

### 2.8 docs/ 目录

- **00_project_overview.md**：项目概览，介绍项目的整体情况
- **01_stage_history.md**：阶段历史，记录项目各阶段的进展
- **02_experiment_results.md**：实验结果，展示实验性能指标
- **03_system_deployment.md**：系统部署，介绍系统的部署方法
- **04_audit_and_trust.md**：审计与可信度，说明结果的可信性
- **05_defense_materials.md**：答辩材料，包含答辩报告和PPT提纲
- **06_limitations_and_future_work.md**：局限性与未来工作，讨论系统的局限性和改进方向
- **07_repo_structure.md**：仓库结构，说明项目的目录结构

### 2.9 outputs/ 目录

- **checkpoints/**：模型检查点目录，存放训练过程中的模型权重
- **figures/**：图表目录，存放实验生成的图表
- **reports/**：报告目录，存放实验生成的报告

### 2.10 audit/ 目录

- 存放审计相关文件，包括审计报告和可信度评估

### 2.11 defense/ 目录

- 存放答辩材料，包括答辩讲稿、PPT和问答库

### 2.12 examples/ 目录

- 存放示例代码和使用示例

## 3. 文件用途

### 3.1 核心文件

| 文件 | 用途 | 位置 |
|------|------|------|
| README.md | 项目主说明 | 根目录 |
| requirements.txt | 依赖包列表 | 根目录 |
| run_stage4.py | 第四阶段执行脚本 | 根目录 |
| verify_system.py | 系统验证脚本 | 根目录 |
| src/stage3/models.py | 模型定义 | src/stage3/ |
| src/stage3/data.py | 数据处理 | src/stage3/ |
| src/stage3/trainer.py | 模型训练 | src/stage3/ |
| src/stage3/evaluator.py | 模型评估 | src/stage3/ |
| deploy_bundle/inference/model_loader.py | 模型加载器 | deploy_bundle/inference/ |
| frontend_app/app.py | 前端应用 | frontend_app/ |

### 3.2 配置文件

| 文件 | 用途 | 位置 |
|------|------|------|
| config/stage4/base.yaml | 第四阶段基础配置 | config/stage4/ |
| config/base.yaml | 通用基础配置 | config/ |
| config/local_cpu.yaml | 本地CPU配置 | config/ |

### 3.3 数据文件

| 文件 | 用途 | 位置 |
|------|------|------|
| data/raw/UNSW_NB15_training-set.csv | 原始训练数据 | data/raw/ |
| data/raw/UNSW_NB15_testing-set.csv | 原始测试数据 | data/raw/ |
| data/processed/X_train.npy | 处理后的训练特征 | data/processed/ |
| data/processed/y_train.npy | 处理后的训练标签 | data/processed/ |
| data/processed/X_test.npy | 处理后的测试特征 | data/processed/ |
| data/processed/y_test.npy | 处理后的测试标签 | data/processed/ |

### 3.4 部署文件

| 文件 | 用途 | 位置 |
|------|------|------|
| deploy_bundle/model/best.ckpt | 最佳模型检查点 | deploy_bundle/model/ |
| deploy_bundle/model/model_config.yaml | 模型配置 | deploy_bundle/model/ |
| deploy_bundle/inference/infer.py | 单样本推理脚本 | deploy_bundle/inference/ |
| deploy_bundle/inference/batch_infer.py | 批量推理脚本 | deploy_bundle/inference/ |

### 3.5 前端文件

| 文件 | 用途 | 位置 |
|------|------|------|
| frontend_app/app.py | 前端应用代码 | frontend_app/ |
| frontend_app/run_frontend.bat | 前端启动脚本 | frontend_app/ |

## 4. 目录使用建议

### 4.1 开发流程

1. **数据准备**：使用 `scripts/download_dataset.py` 下载数据集
2. **数据预处理**：使用 `scripts/preprocess.py` 预处理数据
3. **模型训练**：使用 `scripts/train_autoencoder.py` 训练模型
4. **模型评估**：使用 `scripts/evaluate.py` 评估模型
5. **部署打包**：使用 `run_stage4.py` 生成部署包
6. **前端运行**：使用 `frontend_app/run_frontend.bat` 启动前端

### 4.2 实验流程

1. **配置修改**：修改 `config/` 目录下的配置文件
2. **实验运行**：运行相应的脚本执行实验
3. **结果查看**：在 `outputs/` 目录查看实验结果
4. **模型部署**：使用 `run_stage4.py` 部署模型
5. **系统验证**：使用 `verify_system.py` 验证系统完整性

### 4.3 部署流程

1. **环境准备**：创建虚拟环境并安装依赖
2. **部署包生成**：运行 `run_stage4.py --mode deploy` 生成部署包
3. **前端启动**：运行 `frontend_app/run_frontend.bat` 启动前端
4. **系统验证**：运行 `verify_system.py` 验证系统

## 5. 最佳实践

1. **环境隔离**：使用虚拟环境隔离依赖
2. **版本控制**：使用Git进行版本控制
3. **配置管理**：通过配置文件管理实验参数
4. **结果记录**：及时记录实验结果和配置
5. **代码规范**：遵循Python代码规范
6. **文档更新**：及时更新项目文档

## 6. 总结

本项目采用了清晰的目录结构，便于代码管理和维护。各个目录职责明确，文件组织合理，有助于提高开发效率和代码可读性。通过遵循上述目录结构和使用建议，可以更加高效地开发、测试和部署网络流量异常检测系统。