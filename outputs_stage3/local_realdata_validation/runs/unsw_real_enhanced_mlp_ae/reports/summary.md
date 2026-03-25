# Stage 3 Experiment Summary

## Context
- Dataset: unsw_nb15
- Model: enhanced_mlp_ae
- Run directory: D:\zlx\traffic_anomaly_detection\outputs_stage3\local_realdata_validation\runs\unsw_real_enhanced_mlp_ae
- Scoring method: reconstruction
- Primary threshold method: pr_optimal
- Parameter count: 20514
- Training time (s): 6.81

## Metrics
- ROC-AUC: 0.8283
- PR-AUC: 0.8273
- F1: 0.7967
- Precision: 0.7141
- Recall: 0.9010

## Threshold Comparison
- percentile: F1=0.6235, Precision=0.8783, Recall=0.4833, Threshold=0.323614
- f1_optimal: F1=0.7967, Precision=0.7141, Recall=0.9010, Threshold=0.078139
- youden: F1=0.7513, Precision=0.8078, Recall=0.7022, Threshold=0.151119
- pr_optimal: F1=0.7967, Precision=0.7141, Recall=0.9010, Threshold=0.078139

## Per-Attack Recall
- Analysis: 0.9865
- Backdoor: 1.0000
- DoS: 0.9751
- Exploits: 0.9323
- Fuzzers: 0.6295
- Generic: 0.9913
- Reconnaissance: 0.7006
- Shellcode: 0.6000
- Worms: 1.0000