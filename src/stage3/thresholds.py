from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def select_threshold(
    method: str,
    train_scores: np.ndarray,
    val_scores: np.ndarray | None,
    val_labels: np.ndarray | None,
    percentile: float = 95,
) -> dict[str, Any]:
    method = str(method)
    if method == "percentile":
        threshold = float(np.percentile(train_scores, percentile))
        return {"method": method, "threshold": threshold, "calibration_source": "train_normal", "criterion_value": percentile}

    if val_scores is None or val_labels is None:
        threshold = float(np.percentile(train_scores, percentile))
        return {"method": method, "threshold": threshold, "calibration_source": "fallback_train_normal", "criterion_value": percentile}

    val_scores = np.asarray(val_scores)
    val_labels = np.asarray(val_labels)

    if method == "f1_optimal":
        precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
        mask = np.isfinite(thresholds)
        thresholds = thresholds[mask]
        precision = precision[:-1][mask]
        recall = recall[:-1][mask]
        if thresholds.size == 0:
            threshold = float(np.percentile(train_scores, percentile))
            return {"method": method, "threshold": threshold, "calibration_source": "fallback_train_normal", "criterion_value": percentile}
        f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        best_idx = int(np.nanargmax(f1_scores))
        return {"method": method, "threshold": float(thresholds[best_idx]), "calibration_source": "validation", "criterion_value": float(f1_scores[best_idx])}

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
        mask = np.isfinite(thresholds)
        thresholds = thresholds[mask]
        fpr = fpr[mask]
        tpr = tpr[mask]
        if thresholds.size == 0:
            threshold = float(np.percentile(train_scores, percentile))
            return {"method": method, "threshold": threshold, "calibration_source": "fallback_train_normal", "criterion_value": percentile}
        youden = tpr - fpr
        best_idx = int(np.nanargmax(youden))
        return {"method": method, "threshold": float(thresholds[best_idx]), "calibration_source": "validation", "criterion_value": float(youden[best_idx])}

    if method == "pr_optimal":
        precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
        mask = np.isfinite(thresholds)
        thresholds = thresholds[mask]
        precision = precision[:-1][mask]
        recall = recall[:-1][mask]
        if thresholds.size == 0:
            threshold = float(np.percentile(train_scores, percentile))
            return {"method": method, "threshold": threshold, "calibration_source": "fallback_train_normal", "criterion_value": percentile}
        objective = precision * recall
        best_idx = int(np.nanargmax(objective))
        return {"method": method, "threshold": float(thresholds[best_idx]), "calibration_source": "validation", "criterion_value": float(objective[best_idx])}

    raise ValueError(f"Unsupported threshold method: {method}")
