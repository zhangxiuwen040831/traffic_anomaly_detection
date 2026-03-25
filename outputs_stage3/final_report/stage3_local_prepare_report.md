# 第三阶段子阶段 A 本地与云端准备报告

## 1. 阶段定位

本阶段严格遵守当前资源约束，目标不是立即产出论文最终结果，而是把项目推进到“GPU 一开启即可直接进入论文级长训”的状态。

本阶段已完成两条并行工作线：

1. 本地 CPU 线
   - 完成统一研究框架开发
   - 完成多模型与多阈值模块实现
   - 用最小规模数据完成代码正确性与输出链路验证

2. 云端无 GPU 线
   - 完成弱资源环境探测
   - 完成目录结构、脚本、断点续训和下载器部署
   - 明确保留“只准备、不重训”的执行边界

## 2. 已实现的论文级研究框架

### 2.1 统一接口

项目已新增 `src/stage3/` 研究框架主干，包含：

- `src/stage3/data.py`
  - 统一 dataset 接口
  - 支持 `unsw_nb15` 与 `cic_ids2017` 两类数据入口
  - 支持 smoke / full 两种模式
  - 支持缓存化预处理结果
  - 支持统一训练/验证/测试拆分
  - 支持只用正常样本作为训练输入

- `src/stage3/models.py`
  - 统一 model 接口
  - 已实现：
    - `enhanced_mlp_ae`
    - `transformer_ae`
    - `vae`
    - `hybrid_ae`

- `src/stage3/trainer.py`
  - 统一 trainer 接口
  - 支持：
    - 自动保存 `best.ckpt`
    - 自动保存 `last.ckpt`
    - 训练日志落盘
    - `history.csv`
    - 断点续训
    - 早停
    - 学习率调度

- `src/stage3/scoring.py`
  - 统一 anomaly scoring 接口
  - 支持：
    - reconstruction error
    - latent distance
    - density-related score
    - hybrid weighted score

- `src/stage3/thresholds.py`
  - 统一阈值模块
  - 已实现：
    - percentile threshold
    - F1-optimal threshold
    - Youden index
    - PR-optimal threshold

- `src/stage3/evaluator.py`
  - 统一 evaluator 接口
  - 支持：
    - ROC-AUC
    - PR-AUC
    - Precision
    - Recall
    - F1
    - confusion matrix
    - score distribution 原始数据导出
    - per-attack recall

- `src/stage3/reporting.py`
  - 统一结果导出
  - 自动生成：
    - `metrics.json`
    - `metrics.csv`
    - `summary.md`
    - 阈值对比表
    - 训练曲线
    - 分数分布图
    - ROC / PR 曲线
    - 混淆矩阵
    - per-attack recall 图

- `src/stage3/pipeline.py`
  - 统一实验流水线
  - 统一训练、评估、汇总、消融调用逻辑

### 2.2 配置与命令行

已新增配置驱动和 CLI：

- `config/stage3/base.yaml`
- `config/stage3/local_smoke.yaml`
- `config/stage3/cloud_gpu.yaml`
- `scripts/stage3_cli.py`

当前已支持命令：

- `train`
- `eval`
- `prepare-data`
- `smoke-test`
- `ablation`
- `compare`
- `summarize`

### 2.3 云端准备脚本

已新增云端部署与长训准备资产：

- `scripts/stage3_cloud_prepare.py`
- `scripts/cloud/stage3_prepare_layout.sh`
- `scripts/cloud/stage3_env_probe.sh`
- `scripts/cloud/stage3_download_datasets.sh`
- `scripts/cloud/stage3_gpu_train.sh`

## 3. 已实现模型与研究路线映射

### 3.1 Baseline

- Enhanced MLP Autoencoder
- 保留了第二阶段的核心思路：正常样本训练、重构误差检测
- 在第三阶段中已纳入统一实验框架，可直接作为正式 baseline

### 3.2 Model A

- Transformer Autoencoder（轻量版）
- 使用特征 token 化 + Transformer encoder + 轻量 decoder
- 适合未来在 GPU 阶段做结构对比

### 3.3 Model B

- Variational Autoencoder
- 支持 reconstruction loss + KL regularization
- 适合未来研究 latent 表示对异常可分性的影响

### 3.4 Model C

- Hybrid Anomaly Scoring 模型
- 当前默认融合：
  - reconstruction error
  - latent distance
  - density-related score
- 已支持消融掉 latent distance / density / hybrid 只保留 reconstruction

## 4. 本地最小规模验证结果

### 4.1 已完成的本地 smoke 验证

已实际跑通：

1. UNSW-NB15 接口 smoke
   - `enhanced_mlp_ae`
   - `transformer_ae`
   - `vae`
   - `hybrid_ae`

2. 阈值策略 smoke
   - 同一实验内自动计算四种阈值方法
   - 自动输出阈值对比表

3. 消融 smoke
   - `reconstruction_only`
   - `hybrid_full`
   - `hybrid_no_latent_distance`
   - `hybrid_no_density`

4. CIC-IDS2017 接口 smoke
   - 验证跨数据集入口
   - 验证 `per-attack recall`
   - 验证攻击类型图表生成

### 4.2 结果解读边界

当前本地 smoke 数据仅用于：

- 验证代码正确性
- 验证 loss 能收敛
- 验证指标计算无误
- 验证图表与报告链路可用
- 验证不同阈值策略的接口逻辑

当前 smoke 结果不用于论文最终结论，因为当前本地数据是轻量化/冒烟验证数据，不代表真实 GPU 正式实验结果。

### 4.3 关键输出示例

本地已产出如下关键实验结果目录：

- `outputs_stage3/local_smoke_test/unsw_nb15_enhanced_mlp_ae`
- `outputs_stage3/local_smoke_test/unsw_nb15_transformer_ae`
- `outputs_stage3/local_smoke_test/unsw_nb15_vae`
- `outputs_stage3/local_smoke_test/unsw_nb15_hybrid_ae`
- `outputs_stage3/local_smoke_test/cic_ids2017_hybrid_smoke_attack`
- `outputs_stage3/local_smoke_test/comparison`
- `outputs_stage3/ablation/comparison`

### 4.4 阈值模块验证结论

以 `hybrid_ae` 的 smoke 结果为例：

- percentile 阈值相对保守，虽然 recall 可以高，但更容易引入额外 FP
- F1-optimal / Youden / PR-optimal 已能通过统一接口自动完成阈值搜索
- 这条链路已经为未来解决“高 precision、低 recall”的正式问题做好了工程准备

### 4.5 per-attack recall 模块验证结论

`cic_ids2017` smoke 已生成：

- `per_attack_recall.png`
- 按攻击族写入 `summary.md`

当前验证结果说明：

- 攻击族级 recall 统计链路已打通
- 后续只需接入真实 CIC 数据，即可直接产出论文需要的攻击类型分析

## 5. 已自动生成的论文级输出

每个实验目录下已自动生成：

- `metrics.json`
- `metrics.csv`
- `reports/summary.md`
- `tables/threshold_comparison.csv`
- `figures/training_curve.png`
- `figures/score_distribution.png`
- `figures/roc_curve.png`
- `figures/pr_curve.png`
- `figures/confusion_matrix.png`
- `figures/per_attack_recall.png`（若攻击标签可用）

这意味着第三阶段的“结果结构”已经稳定下来，后续正式 GPU 长训只需要替换为真实数据与更长训练轮数，不需要再重构输出链路。

## 6. 云端当前无 GPU 阶段准备结果

### 6.1 实际探测结果

已实际连接远端并生成环境报告：

- 远端项目根目录：
  - `/root/lanyun-fs/traffic_anomaly_detection_stage3`

- Python:
  - `Python 3.10.12`

- pip:
  - `pip 26.0.1`

- PyTorch:
  - `torch 2.1.0+cu118`

- GPU:
  - `cuda_available: False`
  - 当前无 GPU 可用

- 容器配额（关键）：
  - `cpu_limit_cores: 0.50`
  - `memory_limit_gb: 2.00`

- 工具可用性：
  - `tmux`
  - `screen`
  - `nohup`
  - `curl`
  - `wget`
  - `unzip`
  - `tar`
  - `git`

### 6.2 关于“宿主资源”和“容器配额”的说明

远端 `nproc` / `free -h` 会显示宿主级资源，但 cgroup 配额已明确显示当前容器仅有：

- 0.5 核 CPU
- 2GB 内存

因此后续执行策略严格以 cgroup 配额为准，而不是以宿主机显示值为准。

### 6.3 已完成的云端准备动作

已完成：

1. 远端目录创建
2. Stage 3 源码轻量同步
3. 云端布局脚本部署
4. 环境探测脚本部署并执行
5. 数据下载脚本部署
6. 长训启动脚本部署
7. `tmux / screen / nohup` 兼容命令准备

### 6.4 已准备好的云端关键脚本

- `scripts/cloud/stage3_prepare_layout.sh`
- `scripts/cloud/stage3_env_probe.sh`
- `scripts/cloud/stage3_download_datasets.sh`
- `scripts/cloud/stage3_gpu_train.sh`

### 6.5 当前云端明确禁止的操作

本阶段没有在远端执行以下任何重操作：

- 没有进行正式训练
- 没有进行完整数据下载
- 没有进行大规模解压
- 没有进行重型 pip 安装
- 没有进行大规模预处理

## 7. 本阶段已经达到的状态

第三阶段子阶段 A 已达到以下可交付状态：

1. 本地 CPU 已具备完整研究框架
2. 四类核心模型已实现并完成最小规模验证
3. 四种阈值策略已实现并完成逻辑验证
4. 评估、图表、报告、比较表输出链路已打通
5. 多数据集接口已建立，CIC 入口已完成 smoke 验证
6. 云端弱主机已完成部署与长训前准备
7. 未来 GPU 开启后，无需重构项目，只需按计划执行正式数据准备与长训

## 8. 当前遗留边界

仍需明确属于子阶段 B 的工作：

- 真实 UNSW-NB15 全量正式训练
- 真实 CIC-IDS2017 正式下载、解压与预处理
- 多模型正式对比
- 跨数据集正式泛化实验
- 论文级消融实验全量结果
- 多随机种子稳定性实验
- 攻击族正式统计结果

这些内容当前都没有强行在弱云端执行，已被明确保留到 GPU 阶段。
