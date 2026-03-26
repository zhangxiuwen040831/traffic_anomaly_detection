# Stage 3 Experiment Summary

## Context
- Dataset: unsw_nb15
- Model: hybrid_ae
- Run directory: D:\zlx\traffic_anomaly_detection\outputs_stage3\local_realdata_validation\ablation\unsw_nb15_hybrid_ae_hybrid_no_latent_distance
- Scoring method: hybrid
- Primary threshold method: pr_optimal
- Parameter count: 20514
- Training time (s): 6.89

## Metrics
- ROC-AUC: 0.8090
- PR-AUC: 0.8091
- F1: 0.7846
- Precision: 0.6804
- Recall: 0.9264

## Threshold Comparison
- percentile: F1=0.5931, Precision=0.8740, Recall=0.4488, Threshold=0.941025
- f1_optimal: F1=0.7922, Precision=0.7028, Recall=0.9076, Threshold=-0.131075
- youden: F1=0.7021, Precision=0.7879, Recall=0.6331, Threshold=0.181990
- pr_optimal: F1=0.7846, Precision=0.6804, Recall=0.9264, Threshold=-0.161152

## Per-Attack Recall
- Analysis: 1.0000
- Backdoor: 1.0000
- DoS: 0.9825
- Exploits: 0.9427
- Fuzzers: 0.6984
- Generic: 0.9978
- Reconnaissance: 0.8241
- Shellcode: 0.6857
- Worms: 1.0000