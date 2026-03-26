# Stage 3 Local Real-Data Summary

- Primary dataset: UNSW-NB15 real train/test CSV.
- Best current model: enhanced_mlp_ae (F1=0.7967, Recall=0.9010).
- Strongest non-baseline candidate: vae (F1=0.7940, Recall=0.8874).
- Baseline threshold sensitivity: Recall +0.4177, F1 +0.1732 vs percentile.
- Hybrid threshold sensitivity: Recall +0.4690, F1 +0.1996 vs percentile.
- Hardest baseline attacks: Shellcode=0.600, Fuzzers=0.630, Reconnaissance=0.701.
- Hybrid score verdict: not yet convincing.
- Transformer verdict: Transformer 当前更像阈值退化案例而不是明确优势模型。在 PR-optimal 下它几乎把所有样本都判为异常，Recall=1.0000，但 Precision 只有 0.5506；改用 Youden 后虽更平衡，但 F1 仍只有 0.7041。
- VAE verdict: VAE 没有超过 baseline，但与 baseline 非常接近且训练更快，适合作为第二优先路线继续保留。
- Main bottleneck judgement: 当前结果表明，第二阶段的低召回很大程度上来自阈值过于保守。一旦改用更合适的阈值，baseline 已经能达到 Recall=0.9010。真正的剩余瓶颈不是 backbone 太弱，而是更复杂模型和混合评分并没有带来更好的排序质量。下一步最值得优先改的是 anomaly score 与 one-class / density 机制，而不是盲目放大模型。