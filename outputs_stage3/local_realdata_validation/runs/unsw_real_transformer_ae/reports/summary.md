# Stage 3 Experiment Summary

## Context
- Dataset: unsw_nb15
- Model: transformer_ae
- Run directory: D:\zlx\traffic_anomaly_detection\outputs_stage3\local_realdata_validation\runs\unsw_real_transformer_ae
- Scoring method: reconstruction
- Primary threshold method: pr_optimal
- Parameter count: 14841
- Training time (s): 41.52

## Metrics
- ROC-AUC: 0.7445
- PR-AUC: 0.7948
- F1: 0.7102
- Precision: 0.5506
- Recall: 1.0000

## Threshold Comparison
- percentile: F1=0.6233, Precision=0.8702, Recall=0.4856, Threshold=0.000323
- f1_optimal: F1=0.7102, Precision=0.5506, Recall=1.0000, Threshold=0.000026
- youden: F1=0.7041, Precision=0.8158, Recall=0.6193, Threshold=0.000248
- pr_optimal: F1=0.7102, Precision=0.5506, Recall=1.0000, Threshold=0.000026

## Per-Attack Recall
- Analysis: 1.0000
- Backdoor: 1.0000
- DoS: 1.0000
- Exploits: 1.0000
- Fuzzers: 1.0000
- Generic: 1.0000
- Reconnaissance: 1.0000
- Shellcode: 1.0000
- Worms: 1.0000