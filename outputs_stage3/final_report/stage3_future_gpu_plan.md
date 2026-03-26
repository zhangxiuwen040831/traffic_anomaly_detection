# 第三阶段子阶段 B GPU 开启后执行计划

## 1. 目标

子阶段 B 的目标不是继续补框架，而是直接利用已经完成的 Stage 3 研究框架，在云端 GPU 环境中执行论文级正式实验。

执行原则：

1. 先拿可发表的核心结果
2. 再补充消融、稳定性与攻击类型分析
3. 全程使用现有 Stage 3 统一接口，不做结构性返工

## 2. GPU 开启后的第一批动作

GPU 开启后，建议按以下顺序立即执行。

### 2.1 环境确认

在远端项目目录执行：

```bash
cd /root/lanyun-fs/traffic_anomaly_detection_stage3
bash scripts/cloud/stage3_env_probe.sh outputs_stage3/cloud_prepare/reports
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
```

目标：

- 确认 GPU 真正可见
- 确认 CUDA 与 PyTorch 可用
- 记录显存、GPU 型号和驱动

### 2.2 依赖安装或补装

如果 GPU 镜像环境缺包，再执行：

```bash
pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

如阿里源异常，再切换：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：

- 只有 GPU 已开启时才建议进行正式依赖安装
- 如果当前镜像已带 `torch+cuda`，优先保留现成版本，避免无必要重装

## 3. 正式数据准备顺序

### 3.1 优先级

正式数据准备顺序建议如下：

1. UNSW-NB15
2. CIC-IDS2017

原因：

- UNSW 已有基线结果，可最快完成第三阶段第一批核心对比
- CIC-IDS2017 可作为跨数据集泛化验证，增强论文说服力

### 3.2 下载命令

#### UNSW-NB15

```bash
cd /root/lanyun-fs/traffic_anomaly_detection_stage3
bash scripts/cloud/stage3_download_datasets.sh /root/lanyun-fs/traffic_anomaly_detection_stage3 unsw_nb15
```

#### CIC-IDS2017

```bash
cd /root/lanyun-fs/traffic_anomaly_detection_stage3
bash scripts/cloud/stage3_download_datasets.sh /root/lanyun-fs/traffic_anomaly_detection_stage3 cic_ids2017
```

脚本特性：

- 已下载文件自动跳过
- 支持断点续传
- 压缩包解压后自动删除归档文件
- UNSW 优先尝试国内镜像
- CIC 预留镜像入口，必要时可通过环境变量覆盖镜像 URL

### 3.3 正式缓存化预处理

下载完成后，先分别建立正式缓存：

```bash
python3 scripts/stage3_cli.py --override config/stage3/cloud_gpu.yaml prepare-data --dataset unsw_nb15
python3 scripts/stage3_cli.py --override config/stage3/cloud_gpu.yaml prepare-data --dataset cic_ids2017
```

目标：

- 在正式训练前先确认数据接口、缓存目录和特征维度都正常
- 避免第一次正式训练时把“数据问题”和“模型问题”混在一起

## 4. 正式训练顺序

### 4.1 第一轮核心结果

优先跑最重要、最容易形成论文主结论的一组实验：

1. `enhanced_mlp_ae` on `unsw_nb15`
2. `transformer_ae` on `unsw_nb15`
3. `vae` on `unsw_nb15`
4. `hybrid_ae` on `unsw_nb15`

这一轮的目标是尽快回答三个关键问题：

1. Transformer 是否优于 MLP baseline
2. VAE 的 latent regularization 是否提升 recall / F1
3. Hybrid score 是否能显著缓解当前“高 precision、低 recall”问题

### 4.2 第二轮泛化结果

完成 UNSW 第一轮后，再做跨数据集复现：

1. `enhanced_mlp_ae` on `cic_ids2017`
2. `transformer_ae` on `cic_ids2017`
3. `vae` on `cic_ids2017`
4. `hybrid_ae` on `cic_ids2017`

这一轮用于支撑：

- 模型泛化性
- 跨数据集稳定性
- 不是只对单一数据集有效

## 5. 长训启动方式

### 5.1 推荐方式

优先使用 `tmux`：

```bash
cd /root/lanyun-fs/traffic_anomaly_detection_stage3
LAUNCH_MODE=tmux MODEL_NAME=enhanced_mlp_ae DATASET_NAME=unsw_nb15 bash scripts/cloud/stage3_gpu_train.sh /root/lanyun-fs/traffic_anomaly_detection_stage3
```

如需无交互后台运行，可用：

```bash
cd /root/lanyun-fs/traffic_anomaly_detection_stage3
LAUNCH_MODE=nohup MODEL_NAME=hybrid_ae DATASET_NAME=unsw_nb15 bash scripts/cloud/stage3_gpu_train.sh /root/lanyun-fs/traffic_anomaly_detection_stage3
```

### 5.2 已具备的容错能力

当前训练链路已支持：

- `best.ckpt` 自动保存
- `last.ckpt` 自动保存
- `--resume` 断点续训
- 日志落盘
- 历史曲线落盘

因此如果长训中断，恢复方式是：

```bash
cd /root/lanyun-fs/traffic_anomaly_detection_stage3
LAUNCH_MODE=direct MODEL_NAME=hybrid_ae DATASET_NAME=unsw_nb15 RUN_NAME=unsw_nb15_hybrid_ae_gpu bash scripts/cloud/stage3_gpu_train.sh /root/lanyun-fs/traffic_anomaly_detection_stage3
```

只要 `RUN_NAME` 保持一致，训练器就会基于已有 `last.ckpt` 继续。

## 6. GPU 阶段论文级实验编排

### 6.1 多模型对比实验

必须完成：

- MLP AE vs Transformer AE vs VAE vs Hybrid AE
- 数据集：
  - UNSW-NB15
  - CIC-IDS2017

输出重点：

- ROC-AUC
- PR-AUC
- F1
- Precision
- Recall
- Threshold comparison

### 6.2 阈值策略比较

必须比较：

- percentile
- F1-optimal
- Youden
- PR-optimal

核心目标：

- 量化阈值选择对 recall / F1 的影响
- 证明第三阶段不只是“换模型”，也针对低召回问题优化决策机制

### 6.3 消融实验

建议先围绕 `hybrid_ae` 做：

1. `reconstruction_only`
2. `hybrid_full`
3. `hybrid_no_latent_distance`
4. `hybrid_no_density`

如果 GPU 时间充足，再补：

- Transformer 去除版
- 只保留 AE backbone + hybrid score
- 只保留 latent distance

### 6.4 攻击类型分析

对可用攻击标签的数据集，必须输出：

- 各攻击族 recall
- 各攻击族漏检情况
- 哪些攻击最易检测
- 哪些攻击最易漏检

这部分是论文讨论段非常关键的证据。

### 6.5 稳定性实验

建议优先使用最有潜力的两个模型做稳定性分析：

- `transformer_ae`
- `hybrid_ae`

参数维度建议：

1. 随机种子
   - 42 / 52 / 62
2. latent dim
   - 8 / 16 / 32 / 64
3. batch size
   - 128 / 256 / 512
4. threshold method
   - 四种阈值全比较

## 7. 建议的正式执行节奏

### 第一天 GPU 开启后

目标：

- 确认 GPU 环境
- 下载 UNSW
- 完成 UNSW 数据缓存
- 跑通 UNSW baseline 正式训练

### 第二天

目标：

- 完成 UNSW 四模型对比
- 先拿第一批 Recall / F1 提升结果

### 第三天

目标：

- 下载并准备 CIC-IDS2017
- 跑 CIC baseline 与 hybrid

### 第四天

目标：

- 完成 CIC 四模型对比
- 生成跨数据集对比表

### 第五天及之后

目标：

- 消融实验
- 稳定性实验
- 攻击类型分析
- 最终总表与论文图表整理

## 8. 最先要拿到的关键论文结果

GPU 开启后，最优先拿的不是所有结果，而是以下三类“最能支撑论文”的证据：

1. `hybrid_ae` 是否显著提高 recall 和 F1
   - 这是直接回应当前 baseline 痛点的核心结果

2. `transformer_ae` / `vae` 是否在至少一个真实数据集上优于 baseline
   - 这是论文创新性和结构对比的核心结果

3. 阈值策略切换是否能带来稳定的 recall 提升
   - 这是方法设计不只依赖网络结构，而是联合优化评分与决策机制的关键证据

## 9. 当前已经准备好的支撑条件

GPU 阶段开始前，不需要再重构以下内容：

- 数据接口
- 模型接口
- 训练接口
- scorer 接口
- threshold 模块
- evaluator
- 图表导出
- metrics 导出
- 对比表导出
- 消融入口
- 云端启动器

因此子阶段 B 启动时，工作重心可以完全转到“正式实验执行”，而不是再补工程框架。
