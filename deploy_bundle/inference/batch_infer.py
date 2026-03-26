"""
Batch inference script
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model_loader import TrafficAnomalyModel


def main():
    parser = argparse.ArgumentParser(description="Traffic Anomaly Detection Batch Inference")
    parser.add_argument("--model-dir", type=str, default="../model", help="Model directory")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with input npy files")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--threshold", type=str, default="f1_optimal", 
                       choices=["f1_optimal", "youden", "pr_optimal"],
                       help="Threshold method")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    model = TrafficAnomalyModel(args.model_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all npy files
    input_dir = Path(args.input_dir)
    npy_files = list(input_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {args.input_dir}")
        return
    
    print(f"Found {len(npy_files)} input files")
    
    results = []
    
    for npy_file in npy_files:
        print(f"\nProcessing {npy_file.name}...")
        
        features = np.load(npy_file)
        result = model.predict(features, threshold_method=args.threshold)
        
        normal_count = np.sum(result["predictions"] == 0)
        anomaly_count = np.sum(result["predictions"] == 1)
        
        results.append({
            "file": npy_file.name,
            "total_samples": len(features),
            "normal": normal_count,
            "anomaly": anomaly_count,
            "anomaly_ratio": anomaly_count / len(features),
            "mean_score": float(np.mean(result["scores"])),
            "min_score": float(np.min(result["scores"])),
            "max_score": float(np.max(result["scores"])),
        })
        
        # Save predictions
        output_file = output_dir / f"pred_{npy_file.name}"
        np.save(output_file, result["predictions"])
        print(f"  Predictions saved to {output_file}")
        
        print(f"  Normal: {normal_count}, Anomaly: {anomaly_count} ({anomaly_count/len(features)*100:.1f}%)")
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_file = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")
    
    print("\n=== Batch Inference Complete ===")
    print(summary_df)


if __name__ == "__main__":
    main()
