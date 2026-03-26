# Stage 3 Local Real-Data Comparison

| dataset | model | scoring | threshold | roc_auc | pr_auc | f1 | precision | recall |
|---|---|---|---|---:|---:|---:|---:|---:|
| unsw_nb15 | enhanced_mlp_ae | reconstruction | pr_optimal | 0.8283 | 0.8273 | 0.7967 | 0.7141 | 0.9010 |
| unsw_nb15 | vae | reconstruction | pr_optimal | 0.8042 | 0.7909 | 0.7940 | 0.7184 | 0.8874 |
| unsw_nb15 | hybrid_ae | hybrid | pr_optimal | 0.7890 | 0.7886 | 0.7810 | 0.6873 | 0.9044 |
| unsw_nb15 | transformer_ae | reconstruction | pr_optimal | 0.7445 | 0.7948 | 0.7102 | 0.5506 | 1.0000 |