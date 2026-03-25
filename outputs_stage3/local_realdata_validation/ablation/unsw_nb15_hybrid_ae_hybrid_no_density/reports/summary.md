# Stage 3 Experiment Summary

## Context
- Dataset: unsw_nb15
- Model: hybrid_ae
- Run directory: D:\zlx\traffic_anomaly_detection\outputs_stage3\local_realdata_validation\ablation\unsw_nb15_hybrid_ae_hybrid_no_density
- Scoring method: hybrid
- Primary threshold method: pr_optimal
- Parameter count: 20514
- Training time (s): 6.82

## Metrics
- ROC-AUC: 0.8018
- PR-AUC: 0.8012
- F1: 0.7789
- Precision: 0.6707
- Recall: 0.9287

## Threshold Comparison
- percentile: F1=0.5942, Precision=0.8721, Recall=0.4506, Threshold=0.867629
- f1_optimal: F1=0.7789, Precision=0.6707, Recall=0.9287, Threshold=-0.172568
- youden: F1=0.7029, Precision=0.7749, Recall=0.6431, Threshold=0.133595
- pr_optimal: F1=0.7789, Precision=0.6707, Recall=0.9287, Threshold=-0.172568

## Per-Attack Recall
- Analysis: 1.0000
- Backdoor: 1.0000
- DoS: 0.9825
- Exploits: 0.9474
- Fuzzers: 0.7000
- Generic: 0.9984
- Reconnaissance: 0.8333
- Shellcode: 0.6857
- Worms: 1.0000