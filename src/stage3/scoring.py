from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import BaseAnomalyModel


class AnomalyScorer:
    def __init__(self, model: BaseAnomalyModel, model_config: dict[str, Any], scoring_config: dict[str, Any], device: torch.device) -> None:
        self.model = model
        self.model_config = model_config
        self.scoring_config = scoring_config
        self.device = device
        self.method = model_config.get("scoring", {}).get("method", "reconstruction")
        self.components = list(model_config.get("scoring", {}).get("components", ["reconstruction"]))
        self.weights = dict(model_config.get("scoring", {}).get("weights", {}))
        self.reference: dict[str, Any] = {}

    def _collect_outputs(self, features: np.ndarray) -> dict[str, np.ndarray]:
        loader = DataLoader(
            TensorDataset(torch.tensor(features, dtype=torch.float32)),
            batch_size=int(self.scoring_config.get("batch_size", 512)),
            shuffle=False,
        )
        self.model.eval()
        recon_errors: list[np.ndarray] = []
        latents: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                reconstruction = outputs["reconstruction"]
                recon_error = torch.mean((reconstruction - batch) ** 2, dim=1)
                recon_errors.append(recon_error.detach().cpu().numpy())
                latent = outputs.get("latent")
                if latent is not None:
                    latents.append(latent.detach().cpu().numpy())
        payload = {"reconstruction": np.concatenate(recon_errors, axis=0)}
        if latents:
            payload["latent"] = np.concatenate(latents, axis=0)
        return payload

    def fit(self, train_normal_features: np.ndarray) -> None:
        payload = self._collect_outputs(train_normal_features)
        reconstruction_scores = payload["reconstruction"]
        self.reference["component_stats"] = {
            "reconstruction": {
                "mean": float(np.mean(reconstruction_scores)),
                "std": float(np.std(reconstruction_scores) + 1e-8),
            }
        }

        latent = payload.get("latent")
        if latent is not None and len(latent) > 1:
            mean = np.mean(latent, axis=0)
            cov = np.cov(latent, rowvar=False)
            if cov.ndim == 0:
                cov = np.eye(latent.shape[1], dtype=np.float64)
            reg = float(self.scoring_config.get("covariance_regularization", 1e-4))
            cov = cov + np.eye(cov.shape[0]) * reg
            inv_cov = np.linalg.pinv(cov)
            diag_var = np.var(latent, axis=0) + reg
            latent_distance = self._mahalanobis(latent, mean, inv_cov)
            density = self._gaussian_nll(latent, mean, diag_var)
            self.reference["latent_mean"] = mean
            self.reference["latent_inv_cov"] = inv_cov
            self.reference["latent_diag_var"] = diag_var
            self.reference["component_stats"]["latent_distance"] = {
                "mean": float(np.mean(latent_distance)),
                "std": float(np.std(latent_distance) + 1e-8),
            }
            self.reference["component_stats"]["density"] = {
                "mean": float(np.mean(density)),
                "std": float(np.std(density) + 1e-8),
            }

    def _mahalanobis(self, latent: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
        centered = latent - mean
        return np.einsum("bi,ij,bj->b", centered, inv_cov, centered)

    def _gaussian_nll(self, latent: np.ndarray, mean: np.ndarray, diag_var: np.ndarray) -> np.ndarray:
        centered = latent - mean
        return 0.5 * np.sum(np.log(2 * np.pi * diag_var) + (centered**2) / diag_var, axis=1)

    def _normalize_component(self, name: str, scores: np.ndarray) -> np.ndarray:
        stats = self.reference["component_stats"][name]
        return (scores - stats["mean"]) / stats["std"]

    def score(self, features: np.ndarray) -> dict[str, Any]:
        payload = self._collect_outputs(features)
        components: dict[str, np.ndarray] = {"reconstruction": payload["reconstruction"]}
        latent = payload.get("latent")
        if latent is not None and "latent_mean" in self.reference:
            components["latent_distance"] = self._mahalanobis(latent, self.reference["latent_mean"], self.reference["latent_inv_cov"])
            components["density"] = self._gaussian_nll(latent, self.reference["latent_mean"], self.reference["latent_diag_var"])

        if self.method == "hybrid":
            active_components = [component for component in self.components if component in components]
            if not active_components:
                active_components = ["reconstruction"]
            weighted_scores = []
            weight_total = 0.0
            for component in active_components:
                weight = float(self.weights.get(component, 1.0))
                weighted_scores.append(weight * self._normalize_component(component, components[component]))
                weight_total += weight
            final_scores = np.sum(np.vstack(weighted_scores), axis=0) / max(weight_total, 1e-8)
        else:
            final_scores = components["reconstruction"]

        return {
            "scores": final_scores.astype(np.float64),
            "components": {key: value.astype(np.float64) for key, value in components.items()},
            "method": self.method,
        }
