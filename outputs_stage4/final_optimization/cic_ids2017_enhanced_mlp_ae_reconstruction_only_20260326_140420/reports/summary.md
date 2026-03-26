# Stage 3 Experiment Summary

## Context
- Dataset: cic_ids2017
- Model: enhanced_mlp_ae
- Run directory: D:\zlx\traffic_anomaly_detection\outputs_stage4\final_optimization\cic_ids2017_enhanced_mlp_ae_reconstruction_only_20260326_140420
- Scoring method: reconstruction
- Primary threshold method: f1_optimal
- Parameter count: 30916
- Training time (s): 1.62

## Metrics
- ROC-AUC: 1.0000
- PR-AUC: 1.0000
- F1: 0.9983
- Precision: 1.0000
- Recall: 0.9965

## Threshold Comparison
- f1_optimal: F1=0.9983, Precision=1.0000, Recall=0.9965, Threshold=2.361447
- youden: F1=0.9983, Precision=1.0000, Recall=0.9965, Threshold=2.361447
- pr_optimal: F1=0.9983, Precision=1.0000, Recall=0.9965, Threshold=2.361447

## Per-Attack Recall
- DoS: 0.9889
- Exploits: 1.0000
- Fuzzers: 1.0000