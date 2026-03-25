# Stage 3 Local Smoke Comparison

| dataset | model | scoring | threshold | roc_auc | pr_auc | f1 | precision | recall |
|---|---|---|---|---:|---:|---:|---:|---:|
| cic_ids2017 | hybrid_ae | hybrid | pr_optimal | 1.0000 | 1.0000 | 0.9983 | 0.9965 | 1.0000 |
| cic_ids2017 | hybrid_ae | hybrid | pr_optimal | 1.0000 | 1.0000 | 0.9965 | 1.0000 | 0.9930 |
| unsw_nb15 | enhanced_mlp_ae | reconstruction | pr_optimal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unsw_nb15 | hybrid_ae | hybrid | pr_optimal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unsw_nb15 | transformer_ae | reconstruction | pr_optimal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unsw_nb15 | vae | reconstruction | pr_optimal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unsw_nb15 | enhanced_mlp_ae | reconstruction | pr_optimal | 1.0000 | 1.0000 | 0.9985 | 1.0000 | 0.9969 |