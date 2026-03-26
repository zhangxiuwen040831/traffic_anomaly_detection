# Stage 3 Experiment Summary

## Context
- Dataset: unsw_nb15
- Model: hybrid_ae
- Run directory: D:\zlx\traffic_anomaly_detection\outputs_stage3\local_realdata_validation\runs\unsw_real_hybrid_ae
- Scoring method: hybrid
- Primary threshold method: pr_optimal
- Parameter count: 20514
- Training time (s): 6.82

## Metrics
- ROC-AUC: 0.7890
- PR-AUC: 0.7886
- F1: 0.7810
- Precision: 0.6873
- Recall: 0.9044

## Threshold Comparison
- percentile: F1=0.5814, Precision=0.8746, Recall=0.4354, Threshold=0.897484
- f1_optimal: F1=0.7810, Precision=0.6873, Recall=0.9044, Threshold=-0.158456
- youden: F1=0.6881, Precision=0.7616, Recall=0.6275, Threshold=0.097694
- pr_optimal: F1=0.7810, Precision=0.6873, Recall=0.9044, Threshold=-0.158456

## Per-Attack Recall
- Analysis: 0.9595
- Backdoor: 1.0000
- DoS: 0.9676
- Exploits: 0.9258
- Fuzzers: 0.6410
- Generic: 0.9951
- Reconnaissance: 0.7407
- Shellcode: 0.6000
- Worms: 1.0000