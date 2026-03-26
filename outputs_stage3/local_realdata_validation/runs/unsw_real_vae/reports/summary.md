# Stage 3 Experiment Summary

## Context
- Dataset: unsw_nb15
- Model: vae
- Run directory: D:\zlx\traffic_anomaly_detection\outputs_stage3\local_realdata_validation\runs\unsw_real_vae
- Scoring method: reconstruction
- Primary threshold method: pr_optimal
- Parameter count: 19946
- Training time (s): 4.75

## Metrics
- ROC-AUC: 0.8042
- PR-AUC: 0.7909
- F1: 0.7940
- Precision: 0.7184
- Recall: 0.8874

## Threshold Comparison
- percentile: F1=0.5525, Precision=0.8471, Recall=0.4100, Threshold=0.353523
- f1_optimal: F1=0.7851, Precision=0.7397, Recall=0.8365, Threshold=0.128382
- youden: F1=0.7295, Precision=0.7734, Recall=0.6904, Threshold=0.181423
- pr_optimal: F1=0.7940, Precision=0.7184, Recall=0.8874, Threshold=0.110827

## Per-Attack Recall
- Analysis: 0.9595
- Backdoor: 1.0000
- DoS: 0.9776
- Exploits: 0.9417
- Fuzzers: 0.6213
- Generic: 0.9516
- Reconnaissance: 0.7191
- Shellcode: 0.6857
- Worms: 1.0000