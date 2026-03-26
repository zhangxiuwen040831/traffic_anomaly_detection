"""
Single sample inference script
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model_loader import TrafficAnomalyModel


def main():
    parser = argparse.ArgumentParser(description="Traffic Anomaly Detection Inference")
    parser.add_argument("--model-dir", type=str, default="../model", help="Model directory")
    parser.add_argument("--input", type=str, required=True, help="Input features (npy file)")
    parser.add_argument("--threshold", type=str, default="f1_optimal", 
                       choices=["f1_optimal", "youden", "pr_optimal"],
                       help="Threshold method")
    parser.add_argument("--output", type=str, help="Output file for predictions")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    model = TrafficAnomalyModel(args.model_dir)
    
    # Load input
    print(f"Loading input from {args.input}...")
    features = np.load(args.input)
    
    # Predict
    print("Running inference...")
    result = model.predict(features, threshold_method=args.threshold)
    
    # Print summary
    normal_count = np.sum(result["predictions"] == 0)
    anomaly_count = np.sum(result["predictions"] == 1)
    print(f"\nResults:")
    print(f"  Total samples: {len(features)}")
    print(f"  Normal: {normal_count}")
    print(f"  Anomaly: {anomaly_count}")
    print(f"  Threshold ({args.threshold}): {result['threshold']:.4f}")
    print(f"  Score range: [{result['scores'].min():.4f}, {result['scores'].max():.4f}]")
    
    # Save output
    if args.output:
        np.save(args.output, result["predictions"])
        print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
