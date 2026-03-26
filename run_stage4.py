#!/usr/bin/env python3
"""
Stage 4 - Final Model Optimization, Packaging, and Deployment
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import warnings

import numpy as np
import torch
import yaml

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from stage3.config import load_stage3_config
from stage3.data import prepare_dataset
from stage3.evaluator import evaluate_threshold_suite
from stage3.models import count_parameters, create_model, MODEL_REGISTRY
from stage3.reporting import write_run_outputs
from stage3.scoring import AnomalyScorer
from stage3.trainer import Trainer
from stage3.utils import ensure_dir, load_json, save_json, set_seed, slugify, timestamp


def create_deploy_bundle(run_dir: Path, deploy_root: Path, config: dict):
    """Create deploy bundle from a trained model run"""
    deploy_root = ensure_dir(deploy_root)
    
    # Create directories
    model_dir = ensure_dir(deploy_root / "model")
    inference_dir = ensure_dir(deploy_root / "inference")
    reports_dir = ensure_dir(deploy_root / "reports")
    figures_dir = ensure_dir(deploy_root / "figures")
    
    # Copy model files
    import shutil
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"
    if checkpoint_path.exists():
        shutil.copy2(checkpoint_path, model_dir / "best.ckpt")
    
    # Save configs
    context = load_json(run_dir / "context.json")
    resolved_config = load_json(run_dir / "resolved_config.json")
    scoring_reference = load_json(run_dir / "scoring_reference.json")
    
    # Save model config
    model_config = {
        "model_name": context["model_name"],
        "input_dim": context["input_dim"],
        "model_parameters": context["model_parameters"],
        "model_config": resolved_config["models"][context["model_name"]]
    }
    save_json(model_dir / "model_config.yaml", model_config, yaml_format=True)
    
    # Save threshold config
    threshold_results = load_json(run_dir / "tables" / "thresholds.json")
    save_json(model_dir / "threshold_config.yaml", threshold_results, yaml_format=True)
    
    # Save preprocessing info
    metadata = context["dataset_metadata"]
    preprocessing_info = {
        "feature_names": metadata.get("feature_names", []),
        "normal_label": metadata.get("normal_label", 0),
        "positive_label": resolved_config["evaluation"]["positive_label"],
        "input_dim": context["input_dim"]
    }
    save_json(model_dir / "preprocessing.json", preprocessing_info)
    
    # Copy figures
    for fig_file in ["roc_curve.png", "pr_curve.png", "confusion_matrix.png", "anomaly_score_distribution.png"]:
        src = run_dir / "figures" / fig_file
        if src.exists():
            shutil.copy2(src, figures_dir / fig_file)
    
    # Generate inference code
    generate_inference_code(inference_dir, model_dir)
    
    # Generate reports
    generate_final_report(reports_dir, context, threshold_results)
    
    print(f"✅ Deploy bundle created at: {deploy_root}")
    return deploy_root


def generate_inference_code(inference_dir: Path, model_dir: Path):
    """Generate inference code"""
    
    # Model loader
    model_loader_code = '''"""
Model Loader for Traffic Anomaly Detection
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.stage3.models import MODEL_REGISTRY
from src.stage3.scoring import AnomalyScorer


class TrafficAnomalyModel:
    def __init__(self, model_dir: str | Path, device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.device = self._get_device(device)
        
        # Load configs
        with open(self.model_dir / "model_config.yaml", "r", encoding="utf-8") as f:
            self.model_config = yaml.safe_load(f)
        
        with open(self.model_dir / "threshold_config.yaml", "r", encoding="utf-8") as f:
            self.threshold_config = yaml.safe_load(f)
        
        with open(self.model_dir / "preprocessing.json", "r", encoding="utf-8") as f:
            self.preprocessing = json.load(f)
        
        # Load model
        self.model, self.scorer = self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        model_name = self.model_config["model_name"]
        input_dim = self.model_config["input_dim"]
        model_cfg = self.model_config["model_config"]
        
        model_cls = MODEL_REGISTRY.get(model_cfg["type"])
        if model_cls is None:
            raise ValueError(f"Unknown model type: {model_cfg['type']}")
        
        kwargs = {k: v for k, v in model_cfg.items() if k not in ["type", "name", "scoring"]}
        model = model_cls(input_dim=input_dim, **kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_dir / "best.ckpt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        # Create scorer
        with open(self.model_dir / "preprocessing.json", "r", encoding="utf-8") as f:
            preproc = json.load(f)
        
        scorer_config = {"batch_size": 512, "covariance_regularization": 1e-4}
        scorer = AnomalyScorer(model=model, model_config=model_cfg, scoring_config=scorer_config, device=self.device)
        
        # Load reference
        with open(self.model_dir / "preprocessing.json", "r", encoding="utf-8") as f:
            pass  # We'll need to save/load reference
        
        return model, scorer
    
    def predict(self, features: np.ndarray, threshold_method: str = "f1_optimal"):
        """Predict anomalies from features"""
        self.model.eval()
        
        # Get scores
        scores_payload = self.scorer.score(features)
        scores = scores_payload["scores"]
        
        # Get threshold
        threshold_entry = next(
            (t for t in self.threshold_config if t["threshold_method"] == threshold_method),
            self.threshold_config[0]
        )
        threshold = threshold_entry["test_threshold"]
        
        # Predict
        predictions = (scores > threshold).astype(int)
        
        return {
            "predictions": predictions,
            "scores": scores,
            "threshold": threshold,
            "threshold_method": threshold_method
        }
'''
    
    with open(inference_dir / "model_loader.py", "w", encoding="utf-8") as f:
        f.write(model_loader_code)
    
    # Simple inference script
    infer_code = '''"""
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
    print(f"\\nResults:")
    print(f"  Total samples: {len(features)}")
    print(f"  Normal: {normal_count}")
    print(f"  Anomaly: {anomaly_count}")
    print(f"  Threshold ({args.threshold}): {result['threshold']:.4f}")
    print(f"  Score range: [{result['scores'].min():.4f}, {result['scores'].max():.4f}]")
    
    # Save output
    if args.output:
        np.save(args.output, result["predictions"])
        print(f"\\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
'''
    
    with open(inference_dir / "infer.py", "w", encoding="utf-8") as f:
        f.write(infer_code)


def generate_final_report(reports_dir: Path, context: dict, threshold_results: list):
    """Generate final report"""
    report = f"""# Final Model Summary

## Model Configuration
- **Model Name**: {context['model_name']}
- **Dataset**: {context['dataset_name']}
- **Input Dimensions**: {context['input_dim']}
- **Parameters**: {context['model_parameters']:,}
- **Scoring Method**: {context['scoring_method']}

## Performance Metrics

"""
    for thresh in threshold_results:
        report += f"""### {thresh['threshold_method']}
- **ROC-AUC**: {thresh['roc_auc']:.4f}
- **PR-AUC**: {thresh['pr_auc']:.4f}
- **F1 Score**: {thresh['f1']:.4f}
- **Precision**: {thresh['precision']:.4f}
- **Recall**: {thresh['recall']:.4f}
- **Threshold**: {thresh['test_threshold']:.4f}

"""
    
    with open(reports_dir / "final_model_summary.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    # Save metrics as JSON
    save_json(reports_dir / "metrics.json", {"context": context, "thresholds": threshold_results})


def run_scoring_ablation(config: dict, dataset_name: str = "cic_ids2017"):
    """Run scoring ablation study"""
    set_seed(config["experiment"]["seed"])
    
    # Prepare dataset
    bundle = prepare_dataset(config, dataset_name=dataset_name, smoke_mode=True)
    
    results = []
    base_model_name = "enhanced_mlp_ae"
    base_model_cfg = config["models"][base_model_name]
    
    for variant in config["ablation"]["scoring_variants"]:
        print(f"\\n=== Running variant: {variant['name']} ===")
        
        # Create variant config
        variant_config = copy.deepcopy(config)
        model_variant = copy.deepcopy(base_model_cfg)
        model_variant["scoring"] = {
            "method": variant["method"],
            "components": variant.get("components", ["reconstruction"]),
            "weights": variant.get("weights", {})
        }
        variant_config["models"][base_model_name] = model_variant
        
        # Train and evaluate
        model, model_cfg = create_model(base_model_name, bundle.input_dim, variant_config)
        
        # Create run dir
        run_group = config["outputs"]["run_group"]
        outputs_root = ensure_dir(project_root / config["outputs"]["root"])
        group_dir = ensure_dir(outputs_root / run_group)
        run_slug = slugify(f"{dataset_name}_{base_model_name}_{variant['name']}_{timestamp()}")
        run_dir = ensure_dir(group_dir / run_slug)
        ensure_dir(run_dir / "logs")
        ensure_dir(run_dir / "checkpoints")
        ensure_dir(run_dir / "figures")
        ensure_dir(run_dir / "tables")
        ensure_dir(run_dir / "reports")
        
        # Train
        trainer = Trainer(model=model, config=variant_config, run_dir=run_dir)
        training_summary = trainer.fit(
            train_features=bundle.train_normal_features,
            val_features=bundle.val_normal_features,
            model_config=model_cfg
        )
        
        # Score
        scorer = AnomalyScorer(model=model, model_config=model_cfg, 
                               scoring_config=variant_config["scoring"], device=trainer.device)
        scorer.fit(bundle.train_normal_features)
        
        train_scores = scorer.score(bundle.train_normal_features)
        val_scores = scorer.score(bundle.val_features)
        test_scores = scorer.score(bundle.test_features)
        
        # Evaluate thresholds
        threshold_results = evaluate_threshold_suite(
            train_scores=train_scores["scores"],
            val_scores=val_scores["scores"],
            val_labels=bundle.val_labels,
            test_scores=test_scores["scores"],
            test_labels=bundle.test_labels,
            attack_labels=bundle.test_attack_labels,
            threshold_cfg=variant_config["threshold"]
        )
        
        # Save
        context = {
            "dataset_name": dataset_name,
            "model_name": base_model_name,
            "variant": variant["name"],
            "model_parameters": count_parameters(model),
            "input_dim": bundle.input_dim,
            "scoring_method": variant["method"],
            "scoring_components": variant.get("components", []),
            "scoring_weights": variant.get("weights", {}),
            "primary_threshold_method": variant_config["threshold"]["primary_method"],
            "training_summary": training_summary,
            "dataset_metadata": bundle.metadata
        }
        
        save_json(run_dir / "context.json", context)
        save_json(run_dir / "resolved_config.json", variant_config)
        save_json(run_dir / "scoring_reference.json", scorer.serialize_reference())
        
        primary_method = str(variant_config["threshold"]["primary_method"])
        primary_metrics = next((t for t in threshold_results if t["threshold_method"] == primary_method), threshold_results[0])
        
        write_run_outputs(
            run_dir=run_dir,
            context=context,
            history=training_summary["history"],
            primary_metrics=primary_metrics,
            threshold_results=threshold_results,
            component_scores={k: v[:256].tolist() for k, v in test_scores["components"].items()}
        )
        
        results.append({
            "variant": variant["name"],
            "run_dir": str(run_dir),
            "context": context,
            "threshold_results": threshold_results,
            "primary_metrics": primary_metrics
        })
        
        print(f"  F1: {primary_metrics['f1']:.4f}, Recall: {primary_metrics['recall']:.4f}, Precision: {primary_metrics['precision']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 4 - Final Model Optimization")
    parser.add_argument("--config", type=str, default="config/stage4/base.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default="cic_ids2017", help="Dataset name")
    parser.add_argument("--mode", type=str, default="full", 
                       choices=["ablation", "train_best", "deploy", "full"],
                       help="Execution mode")
    args = parser.parse_args()
    
    # Load config
    config_path = project_root / args.config
    config = load_stage3_config(str(config_path))
    config["_project_root"] = str(project_root)
    
    if args.mode in ["ablation", "full"]:
        print("=== Running Scoring Ablation Study ===")
        ablation_results = run_scoring_ablation(config, dataset_name=args.dataset)
        
        # Find best variant
        best_result = max(ablation_results, key=lambda r: r["primary_metrics"]["recall"])
        print(f"\\n=== Best Variant (by Recall): {best_result['variant']} ===")
        print(f"  Recall: {best_result['primary_metrics']['recall']:.4f}")
        print(f"  F1: {best_result['primary_metrics']['f1']:.4f}")
        print(f"  Precision: {best_result['primary_metrics']['precision']:.4f}")
    
    if args.mode in ["train_best", "full"]:
        print("\\n=== Training Final Model ===")
        # For now, use the existing bundle or run a simple training
        from stage3.pipeline import run_experiment
        result = run_experiment(
            config=config,
            model_name="enhanced_mlp_ae",
            dataset_name=args.dataset,
            run_group="final_model",
            smoke_mode=True
        )
        best_run_dir = Path(result["run_dir"])
    
    if args.mode in ["deploy", "full"]:
        print("\\n=== Creating Deploy Bundle ===")
        # Find the latest final model run
        outputs_root = project_root / config["outputs"]["root"]
        final_group_dir = outputs_root / "final_model"
        
        if final_group_dir.exists():
            run_dirs = sorted([d for d in final_group_dir.iterdir() if d.is_dir()], reverse=True)
            if run_dirs:
                best_run_dir = run_dirs[0]
                deploy_root = project_root / config["outputs"]["deploy_root"]
                create_deploy_bundle(best_run_dir, deploy_root, config)
    
    print("\\n=== Stage 4 Complete ===")


if __name__ == "__main__":
    main()
