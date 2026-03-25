# Recommendation

- First priority: continue `enhanced_mlp_ae` with tuned thresholding on real UNSW data.
- Second priority: keep `vae` as the nearest non-baseline route, but treat it as a challenger rather than a replacement.
- Always retain `enhanced_mlp_ae` as the control baseline for future comparisons.
- Hybrid scoring recommendation: defer unless score design is improved toward one-class / density-style scoring.
- Threshold recommendation: treat thresholding as a first-order optimization target.
- If compute budget remains limited, do not prioritize larger Transformer variants before score design is settled.