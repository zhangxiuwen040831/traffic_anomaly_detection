from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .thresholds import select_threshold


def _safe_curve_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, scores)
    pr_auc = float(auc(recall_curve, precision_curve))
    fpr, tpr, _ = roc_curve(y_true, scores)
    return roc_auc, pr_auc, precision_curve, recall_curve, pr_thresholds, fpr, tpr


def _per_attack_recall(y_true: np.ndarray, y_pred: np.ndarray, attack_labels: np.ndarray | None) -> dict[str, float]:
    if attack_labels is None or len(attack_labels) == 0:
        return {}
    mask = y_true == 1
    if np.sum(mask) == 0:
        return {}
    attack_labels = np.asarray(attack_labels, dtype=object)[mask]
    y_pred = np.asarray(y_pred)[mask]
    recalls: dict[str, float] = {}
    for attack_name in sorted({str(label) for label in attack_labels if str(label) not in {"", "normal"}}):
        attack_mask = attack_labels.astype(str) == attack_name
        if np.sum(attack_mask) == 0:
            continue
        recalls[attack_name] = float(np.mean(y_pred[attack_mask] == 1))
    return recalls


def evaluate_scores(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold_info: dict[str, Any],
    attack_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    threshold = float(threshold_info["threshold"])
    y_pred = (scores >= threshold).astype(int)
    roc_auc, pr_auc, precision_curve, recall_curve, pr_thresholds, fpr, tpr = _safe_curve_metrics(y_true, scores)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold_method": threshold_info["method"],
        "threshold": threshold,
        "threshold_calibration_source": threshold_info["calibration_source"],
        "threshold_criterion_value": float(threshold_info["criterion_value"]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "score_summary": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        "per_attack_recall": _per_attack_recall(y_true, y_pred, attack_labels),
        "curves": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
            "pr_thresholds": pr_thresholds.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "labels": y_true.tolist(),
        "predictions": y_pred.tolist(),
        "scores": scores.tolist(),
    }


def evaluate_threshold_suite(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    attack_labels: np.ndarray | None,
    threshold_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for method in threshold_cfg.get("methods", []):
        threshold_info = select_threshold(
            method=method,
            train_scores=train_scores,
            val_scores=val_scores,
            val_labels=val_labels,
            percentile=float(threshold_cfg.get("percentile", 95)),
        )
        metrics = evaluate_scores(scores=test_scores, y_true=test_labels, threshold_info=threshold_info, attack_labels=attack_labels)
        results.append(metrics)
    return results
