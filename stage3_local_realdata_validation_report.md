# Stage 3 Local Real-Data Validation Report

## 1. Experiment Setup
- Primary dataset: UNSW-NB15 (real CSV, official train/test files).
- Train file shape: (175341, 45)
- Test file shape: (82332, 45)
- Split design: keep official train/test separation; split official training file into train/validation; use official testing file only for final test.
- CPU-friendly subset policy:
  - train normal max = 10,000
  - validation sample size = 4,000
  - test sample size = 8,000
- Training regime: CPU only, small models, 15 epochs max, early stopping enabled.

## 2. Screening Comparison
| Model | Scoring | Params | Train(s) | ROC-AUC | PR-AUC | F1 | Precision | Recall | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| enhanced_mlp_ae | reconstruction | 20514 | 6.81 | 0.8283 | 0.8273 | 0.7967 | 0.7141 | 0.9010 | 0.078139 |
| vae | reconstruction | 19946 | 4.75 | 0.8042 | 0.7909 | 0.7940 | 0.7184 | 0.8874 | 0.110827 |
| hybrid_ae | hybrid | 20514 | 6.82 | 0.7890 | 0.7886 | 0.7810 | 0.6873 | 0.9044 | -0.158456 |
| transformer_ae | reconstruction | 14841 | 41.52 | 0.7445 | 0.7948 | 0.7102 | 0.5506 | 1.0000 | 0.000026 |

## 3. Key Research Questions
- Q1. Is there a promising non-baseline direction? 目前没有看到任何非 baseline 模型在小规模真实数据上形成明显且稳定的优势信号。最接近的是 `vae`，但其 F1 仍比 baseline 低 0.0027。
- Q2. Is low recall mainly a threshold problem? 低召回很大程度上是阈值问题。以 baseline 为例，最佳阈值相对 percentile 带来了 Recall +0.4177、F1 +0.1732。percentile 在 baseline 上只有 Recall=0.4833，而 PR/F1-optimal 已经能把 Recall 拉到 0.9010。
- Q3. Is hybrid score worth continuing? Hybrid score 暂未显示稳定优势。当前最优 hybrid 变体的 F1=0.7846、Recall=0.9264，仍落后于纯 reconstruction 的 F1=0.7967；说明 latent distance / density 设计要么信息不足，要么小样本下噪声偏大。
- Q4. Is Transformer suitable here? Transformer 当前更像阈值退化案例而不是明确优势模型。在 PR-optimal 下它几乎把所有样本都判为异常，Recall=1.0000，但 Precision 只有 0.5506；改用 Youden 后虽更平衡，但 F1 仍只有 0.7041。
- Q5. Is VAE suitable here? VAE 没有超过 baseline，但与 baseline 非常接近且训练更快，适合作为第二优先路线继续保留。

## 4. Threshold Strategy Analysis
- Baseline `enhanced_mlp_ae`: best-vs-percentile recall gain = 0.4177, F1 gain = 0.1732.
- Hybrid `hybrid_ae`: best-vs-percentile recall gain = 0.4690, F1 gain = 0.1996.
- Percentile is consistently the most conservative choice on baseline / VAE / Hybrid, and it largely recreates the earlier low-recall regime.
- PR-optimal and F1-optimal deliver the strongest recall/F1 on baseline-class models, while Youden is the safer middle ground when we want to avoid extreme false positives.
- Transformer warning sign: PR-optimal gives Recall=1.0000 but Precision=0.5506; Youden lowers recall to 0.6193 but avoids total all-anomaly collapse.
- Interpretation: if AUC is already decent while percentile recall is weak, the model likely has ranking ability but the deployment threshold is too conservative.

## 5. Hybrid Score Analysis
| Variant | Scoring | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| unsw_nb15_hybrid_ae_reconstruction_only | reconstruction | 0.8283 | 0.8273 | 0.7967 | 0.7141 | 0.9010 |
| unsw_nb15_hybrid_ae_hybrid_no_latent_distance | hybrid | 0.8090 | 0.8091 | 0.7846 | 0.6804 | 0.9264 |
| unsw_nb15_hybrid_ae_hybrid_full | hybrid | 0.7890 | 0.7886 | 0.7810 | 0.6873 | 0.9044 |
| unsw_nb15_hybrid_ae_hybrid_no_density | hybrid | 0.8018 | 0.8012 | 0.7789 | 0.6707 | 0.9287 |

## 6. Attack-Type Notes
- Baseline hardest attacks: Shellcode=0.600, Fuzzers=0.630, Reconnaissance=0.701.
- Strongest non-baseline hardest attacks: Fuzzers=0.621, Shellcode=0.686, Reconnaissance=0.719.
- Small or sparse attack families such as Fuzzers / Reconnaissance / Shellcode remain the main source of missed detections even when overall recall is high.

## 7. Likely Bottleneck Diagnosis
- 当前结果表明，第二阶段的低召回很大程度上来自阈值过于保守。一旦改用更合适的阈值，baseline 已经能达到 Recall=0.9010。真正的剩余瓶颈不是 backbone 太弱，而是更复杂模型和混合评分并没有带来更好的排序质量。下一步最值得优先改的是 anomaly score 与 one-class / density 机制，而不是盲目放大模型。
- If a model shows good ROC-AUC / PR-AUC but poor percentile recall, the main issue is threshold calibration rather than representation collapse.
- If hybrid score does not help even after threshold optimization, the latent-space statistics are likely not discriminative enough on tabular flow features.
- If Transformer is slower but not better, the current feature granularity may not justify sequence-style inductive bias.

## 8. Recommended Next Moves
- Route 1: continue `enhanced_mlp_ae` as the strongest current candidate.
- Route 2: keep `vae` as the secondary route because it is the closest non-baseline candidate under the current CPU-friendly setup.
- Keep `enhanced_mlp_ae` as the deployment-strength control baseline because it remains the reference point for stability and cost.
- Prioritize threshold calibration and anomaly-score design before scaling model size.
- If more compute becomes available later, the first long-run combination should be baseline + VAE + score/one-class ablation rather than all directions equally.