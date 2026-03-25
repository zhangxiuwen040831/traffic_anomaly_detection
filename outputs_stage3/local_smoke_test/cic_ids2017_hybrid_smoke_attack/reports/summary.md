# Stage 3 Experiment Summary

## Context
- Dataset: cic_ids2017
- Model: hybrid_ae
- Run directory: D:\zlx\traffic_anomaly_detection\outputs_stage3\local_smoke_test\cic_ids2017_hybrid_smoke_attack
- Scoring method: hybrid
- Primary threshold method: pr_optimal

## Metrics
- ROC-AUC: 1.0000
- PR-AUC: 1.0000
- F1: 0.9983
- Precision: 0.9965
- Recall: 1.0000

## Threshold Comparison
- percentile: F1=0.9441, Precision=0.8941, Recall=1.0000, Threshold=1.348883
- f1_optimal: F1=0.9983, Precision=0.9965, Recall=1.0000, Threshold=5.720492
- youden: F1=0.9983, Precision=0.9965, Recall=1.0000, Threshold=5.720492
- pr_optimal: F1=0.9983, Precision=0.9965, Recall=1.0000, Threshold=5.720492

## Per-Attack Recall
- DoS: 1.0000
- Exploits: 1.0000
- Fuzzers: 1.0000