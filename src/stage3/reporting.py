from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import as_serializable, ensure_dir, flatten_dict, load_json, save_json


def _save_metrics_csv(path: Path, payload: dict[str, Any]) -> None:
    row = flatten_dict(payload)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def plot_training_curve(history: list[dict[str, Any]], figures_dir: Path) -> None:
    if not history:
        return
    ensure_dir(figures_dir)
    frame = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(frame["epoch"], frame["train_loss"], label="train_loss")
    plt.plot(frame["epoch"], frame["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "training_curve.png", dpi=150)
    plt.close()


def plot_score_distribution(scores: list[float], labels: list[int], threshold: float, figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    score_series = pd.Series(scores)
    label_series = pd.Series(labels)
    plt.figure(figsize=(8, 5))
    plt.hist(score_series[label_series == 0], bins=40, alpha=0.6, label="normal", density=True)
    plt.hist(score_series[label_series == 1], bins=40, alpha=0.6, label="anomaly", density=True)
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4f}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.title("Score Distribution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "score_distribution.png", dpi=150)
    plt.close()


def plot_roc_curve(metrics: dict[str, Any], figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    curve = metrics["curves"]
    plt.figure(figsize=(8, 5))
    plt.plot(curve["fpr"], curve["tpr"], label=f"ROC-AUC={metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=150)
    plt.close()


def plot_pr_curve(metrics: dict[str, Any], figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    curve = metrics["curves"]
    plt.figure(figsize=(8, 5))
    plt.plot(curve["recall"], curve["precision"], label=f"PR-AUC={metrics['pr_auc']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(metrics: dict[str, Any], figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    cm = metrics["confusion_matrix"]
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_per_attack_recall(metrics: dict[str, Any], figures_dir: Path) -> None:
    per_attack = metrics.get("per_attack_recall", {})
    if not per_attack:
        return
    ensure_dir(figures_dir)
    names = list(per_attack.keys())
    values = list(per_attack.values())
    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=values, color="#4c72b0")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.title("Per-Attack Recall")
    plt.tight_layout()
    plt.savefig(figures_dir / "per_attack_recall.png", dpi=150)
    plt.close()


def write_run_outputs(
    run_dir: Path,
    context: dict[str, Any],
    history: list[dict[str, Any]],
    primary_metrics: dict[str, Any],
    threshold_results: list[dict[str, Any]],
    component_scores: dict[str, list[float]] | None = None,
) -> None:
    figures_dir = ensure_dir(run_dir / "figures")
    tables_dir = ensure_dir(run_dir / "tables")
    reports_dir = ensure_dir(run_dir / "reports")

    payload = {
        "context": as_serializable(context),
        "history_tail": as_serializable(history[-5:]),
        "primary_metrics": as_serializable(primary_metrics),
        "threshold_results": as_serializable(threshold_results),
        "component_scores_available": bool(component_scores),
    }
    save_json(run_dir / "metrics.json", payload)
    _save_metrics_csv(run_dir / "metrics.csv", payload)

    threshold_rows = []
    for result in threshold_results:
        threshold_rows.append(
            {
                "threshold_method": result["threshold_method"],
                "threshold": result["threshold"],
                "roc_auc": result["roc_auc"],
                "pr_auc": result["pr_auc"],
                "f1": result["f1"],
                "precision": result["precision"],
                "recall": result["recall"],
                "tn": result["confusion_matrix"]["tn"],
                "fp": result["confusion_matrix"]["fp"],
                "fn": result["confusion_matrix"]["fn"],
                "tp": result["confusion_matrix"]["tp"],
            }
        )
    pd.DataFrame(threshold_rows).to_csv(tables_dir / "threshold_comparison.csv", index=False)

    if component_scores:
        pd.DataFrame(component_scores).to_csv(tables_dir / "component_scores_preview.csv", index=False)

    plot_training_curve(history, figures_dir)
    plot_score_distribution(primary_metrics["scores"], primary_metrics["labels"], primary_metrics["threshold"], figures_dir)
    plot_roc_curve(primary_metrics, figures_dir)
    plot_pr_curve(primary_metrics, figures_dir)
    plot_confusion_matrix(primary_metrics, figures_dir)
    plot_per_attack_recall(primary_metrics, figures_dir)

    lines = [
        "# Stage 3 Experiment Summary",
        "",
        "## Context",
        f"- Dataset: {context['dataset_name']}",
        f"- Model: {context['model_name']}",
        f"- Run directory: {run_dir}",
        f"- Scoring method: {context['scoring_method']}",
        f"- Primary threshold method: {context['primary_threshold_method']}",
        f"- Parameter count: {context.get('model_parameters')}",
        f"- Training time (s): {context.get('training_summary', {}).get('total_train_seconds', 0.0):.2f}",
        "",
        "## Metrics",
        f"- ROC-AUC: {primary_metrics['roc_auc']:.4f}",
        f"- PR-AUC: {primary_metrics['pr_auc']:.4f}",
        f"- F1: {primary_metrics['f1']:.4f}",
        f"- Precision: {primary_metrics['precision']:.4f}",
        f"- Recall: {primary_metrics['recall']:.4f}",
        "",
        "## Threshold Comparison",
    ]
    for row in threshold_rows:
        lines.append(
            f"- {row['threshold_method']}: F1={row['f1']:.4f}, Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, Threshold={row['threshold']:.6f}"
        )
    if primary_metrics.get("per_attack_recall"):
        lines.extend(["", "## Per-Attack Recall"])
        for attack_name, recall in primary_metrics["per_attack_recall"].items():
            lines.append(f"- {attack_name}: {recall:.4f}")
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def collect_run_metrics(root_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root_dir.rglob("metrics.json")):
        payload = load_json(path)
        context = payload["context"]
        metrics = payload["primary_metrics"]
        records.append(
            {
                "run_dir": str(path.parent),
                "dataset_name": context["dataset_name"],
                "model_name": context["model_name"],
                "scoring_method": context["scoring_method"],
                "primary_threshold_method": context["primary_threshold_method"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "threshold": metrics["threshold"],
            }
        )
    return records


def write_comparison_outputs(records: list[dict[str, Any]], output_dir: Path, title: str = "Stage 3 Comparison") -> None:
    ensure_dir(output_dir)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=["run_dir", "dataset_name", "model_name", "roc_auc", "pr_auc", "f1", "precision", "recall"])
    else:
        frame.sort_values(by=["dataset_name", "f1", "recall"], ascending=[True, False, False], inplace=True, ignore_index=True)
    frame.to_csv(output_dir / "comparison_metrics.csv", index=False)

    lines = [f"# {title}", ""]
    if frame.empty:
        lines.append("No experiment metrics were found.")
    else:
        lines.append("| dataset | model | scoring | threshold | roc_auc | pr_auc | f1 | precision | recall |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in frame.iterrows():
            lines.append(
                f"| {row['dataset_name']} | {row['model_name']} | {row['scoring_method']} | {row['primary_threshold_method']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
            )
    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
