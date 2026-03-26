"""
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
        
        # Load scoring reference if available
        self.scoring_reference = None
        ref_path = self.model_dir / "scoring_reference.json"
        if ref_path.exists():
            with open(ref_path, "r", encoding="utf-8") as f:
                self.scoring_reference = json.load(f)
        
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
        
        model_type = model_cfg.get("type", model_name)
        model_cls = MODEL_REGISTRY.get(model_type)
        if model_cls is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        kwargs = {k: v for k, v in model_cfg.items() if k not in ["type", "name", "scoring"]}
        model = model_cls(input_dim=input_dim, **kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_dir / "best.ckpt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        # Create scorer
        scorer_config = {"batch_size": 512, "covariance_regularization": 1e-4}
        scorer = AnomalyScorer(model=model, model_config=model_cfg, scoring_config=scorer_config, device=self.device)
        
        # Load reference if available
        if self.scoring_reference:
            scorer.reference = self.scoring_reference
        
        return model, scorer
    
    def fit_scorer(self, train_features: np.ndarray):
        """Fit the scorer on training data (if not already fitted)"""
        self.scorer.fit(train_features)
    
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
        threshold = threshold_entry.get("test_threshold", threshold_entry.get("threshold", 0.5))
        
        # Predict
        predictions = (scores > threshold).astype(int)
        
        return {
            "predictions": predictions,
            "scores": scores,
            "threshold": threshold,
            "threshold_method": threshold_method,
            "components": scores_payload.get("components", {})
        }
