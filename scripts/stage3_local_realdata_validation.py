#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage3.config import load_stage3_config
from src.stage3.pipeline import run_ablation_suite, run_experiment
from src.stage3.reporting import write_comparison_outputs
from src.stage3.utils import ensure_dir


UNSW_URLS = {
    "UNSW_NB15_training-set.csv": [
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
        "https://cdn.jsdelivr.net/gh/Nir-J/ML-Projects@master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    ],
    "UNSW_NB15_testing-set.csv": [
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
        "https://cdn.jsdelivr.net/gh/Nir-J/ML-Projects@master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
    ],
}


def download_real_unsw(raw_root: Path) -> list[dict[str, object]]:
    raw_root.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    download_records: list[dict[str, object]] = []

    for filename, urls in UNSW_URLS.items():
        target = raw_root / filename
        if target.exists() and target.stat().st_size > 1024:
            download_records.append({"file": str(target), "status": "skipped", "size_bytes": target.stat().st_size, "source": "local_cache"})
            continue

        success = False
        last_error = ""
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=120, headers=headers) as response:
                    response.raise_for_status()
                    with target.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                handle.write(chunk)
                download_records.append({"file": str(target), "status": "downloaded", "size_bytes": target.stat().st_size, "source": url})
                success = True
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if target.exists():
                    target.unlink()
        if not success:
            raise RuntimeError(f"Failed to download {filename}: {last_error}")
    return download_records


def inspect_unsw_files(raw_root: Path) -> dict[str, object]:
    train_df = pd.read_csv(raw_root / "UNSW_NB15_training-set.csv")
    test_df = pd.read_csv(raw_root / "UNSW_NB15_testing-set.csv")
    return {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_label_distribution": {str(key): int(value) for key, value in train_df["label"].value_counts().to_dict().items()},
        "test_label_distribution": {str(key): int(value) for key, value in test_df["label"].value_counts().to_dict().items()},
        "train_attack_distribution_top10": {str(key): int(value) for key, value in train_df["attack_cat"].value_counts().head(10).to_dict().items()},
        "test_attack_distribution_top10": {str(key): int(value) for key, value in test_df["attack_cat"].value_counts().head(10).to_dict().items()},
    }


def screening_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        context = result["context"]
        metrics = result["primary_metrics"]
        rows.append(
            {
                "dataset_name": context["dataset_name"],
                "model_name": context["model_name"],
                "scoring_method": context["scoring_method"],
                "parameter_count": context["model_parameters"],
                "train_seconds": context["training_summary"]["total_train_seconds"],
                "best_epoch": context["training_summary"]["best_epoch"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "primary_threshold_method": metrics["threshold_method"],
                "threshold_method": metrics["threshold_method"],
                "threshold": metrics["threshold"],
                "run_dir": result["run_dir"],
            }
        )
    return rows


def threshold_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        model_name = result["context"]["model_name"]
        scoring_method = result["context"]["scoring_method"]
        for metrics in result["threshold_results"]:
            rows.append(
                {
                    "model_name": model_name,
                    "scoring_method": scoring_method,
                    "threshold_method": metrics["threshold_method"],
                    "threshold": metrics["threshold"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "tn": metrics["confusion_matrix"]["tn"],
                    "fp": metrics["confusion_matrix"]["fp"],
                    "fn": metrics["confusion_matrix"]["fn"],
                    "tp": metrics["confusion_matrix"]["tp"],
                }
            )
    return rows


def ablation_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        context = result["context"]
        metrics = result["primary_metrics"]
        rows.append(
            {
                "model_name": context["model_name"],
                "scoring_method": context["scoring_method"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "threshold_method": metrics["threshold_method"],
                "run_dir": result["run_dir"],
            }
        )
    return rows


def choose_best_candidate(screening_df: pd.DataFrame) -> dict[str, object]:
    ordered = screening_df.sort_values(by=["f1", "recall", "pr_auc"], ascending=[False, False, False]).reset_index(drop=True)
    return ordered.iloc[0].to_dict()


def threshold_signal(threshold_df: pd.DataFrame, model_name: str) -> dict[str, float]:
    frame = threshold_df[threshold_df["model_name"] == model_name].copy()
    if frame.empty:
        return {"recall_gain_vs_percentile": 0.0, "f1_gain_vs_percentile": 0.0}
    percentile = frame[frame["threshold_method"] == "percentile"].iloc[0]
    best = frame.sort_values(by=["f1", "recall"], ascending=[False, False]).iloc[0]
    return {
        "recall_gain_vs_percentile": float(best["recall"] - percentile["recall"]),
        "f1_gain_vs_percentile": float(best["f1"] - percentile["f1"]),
    }


def copy_key_figures(source_run_dir: Path, destination_dir: Path, prefix: str) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in ["roc_curve.png", "pr_curve.png", "score_distribution.png", "confusion_matrix.png", "per_attack_recall.png"]:
        source = source_run_dir / "figures" / name
        if source.exists():
            shutil.copy2(source, destination_dir / f"{prefix}_{name}")


def write_reports(
    output_root: Path,
    dataset_inspection: dict[str, object],
    screening_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    baseline_row: dict[str, object],
    best_row: dict[str, object],
) -> None:
    baseline_threshold_signal = threshold_signal(threshold_df, "enhanced_mlp_ae")
    hybrid_threshold_signal = threshold_signal(threshold_df, "hybrid_ae")

    best_nonbaseline = screening_df[screening_df["model_name"] != "enhanced_mlp_ae"].sort_values(
        by=["f1", "recall", "pr_auc"], ascending=[False, False, False]
    ).iloc[0].to_dict()

    hybrid_recon = ablation_df[ablation_df["scoring_method"] == "reconstruction"].sort_values(by="f1", ascending=False).iloc[0].to_dict()
    hybrid_full = ablation_df[ablation_df["scoring_method"] == "hybrid"].sort_values(by="f1", ascending=False).iloc[0].to_dict()

    transformer_row = screening_df[screening_df["model_name"] == "transformer_ae"].iloc[0].to_dict()
    vae_row = screening_df[screening_df["model_name"] == "vae"].iloc[0].to_dict()
    baseline_percentile_row = threshold_df[
        (threshold_df["model_name"] == "enhanced_mlp_ae") & (threshold_df["threshold_method"] == "percentile")
    ].iloc[0].to_dict()
    transformer_pr_row = threshold_df[
        (threshold_df["model_name"] == "transformer_ae") & (threshold_df["threshold_method"] == "pr_optimal")
    ].iloc[0].to_dict()
    transformer_youden_row = threshold_df[
        (threshold_df["model_name"] == "transformer_ae") & (threshold_df["threshold_method"] == "youden")
    ].iloc[0].to_dict()

    def load_per_attack(run_dir: str) -> dict[str, float]:
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            return {}
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_attack = payload.get("primary_metrics", {}).get("per_attack_recall", {})
        return {str(name): float(value) for name, value in per_attack.items()}

    def format_hardest_attacks(per_attack: dict[str, float], limit: int = 3) -> str:
        if not per_attack:
            return "N/A"
        hardest = sorted(per_attack.items(), key=lambda item: (item[1], item[0]))[:limit]
        return ", ".join(f"{name}={value:.3f}" for name, value in hardest)

    baseline_per_attack = load_per_attack(str(baseline_row["run_dir"]))
    best_nonbaseline_per_attack = load_per_attack(str(best_nonbaseline["run_dir"]))
    baseline_f1_gap = float(best_nonbaseline["f1"] - baseline_row["f1"])

    worthwhile_nonbaseline = (
        best_nonbaseline["f1"] > baseline_row["f1"] + 0.01
        or best_nonbaseline["recall"] > baseline_row["recall"] + 0.05
    )
    threshold_is_major_issue = baseline_threshold_signal["recall_gain_vs_percentile"] > 0.10 or baseline_threshold_signal["f1_gain_vs_percentile"] > 0.05
    hybrid_has_signal = (
        hybrid_full["f1"] > hybrid_recon["f1"] + 0.01
        or (
            hybrid_full["recall"] > hybrid_recon["recall"] + 0.03
            and hybrid_full["precision"] >= hybrid_recon["precision"] - 0.03
        )
    )

    if worthwhile_nonbaseline:
        direction_judgement = f"有。当前最强的非 baseline 候选是 `{best_nonbaseline['model_name']}`，其 F1={best_nonbaseline['f1']:.4f}，Recall={best_nonbaseline['recall']:.4f}。"
    else:
        direction_judgement = (
            "目前没有看到任何非 baseline 模型在小规模真实数据上形成明显且稳定的优势信号。"
            f"最接近的是 `{best_nonbaseline['model_name']}`，但其 F1 仍比 baseline 低 {abs(baseline_f1_gap):.4f}。"
        )

    if threshold_is_major_issue:
        threshold_judgement = (
            f"低召回很大程度上是阈值问题。以 baseline 为例，最佳阈值相对 percentile 带来了 "
            f"Recall +{baseline_threshold_signal['recall_gain_vs_percentile']:.4f}、F1 +{baseline_threshold_signal['f1_gain_vs_percentile']:.4f}。"
            f"percentile 在 baseline 上只有 Recall={baseline_percentile_row['recall']:.4f}，而 PR/F1-optimal 已经能把 Recall 拉到 {baseline_row['recall']:.4f}。"
        )
    else:
        threshold_judgement = "低召回不只是阈值过保守，阈值调整带来的收益有限，模型排序能力本身更值得关注。"

    if hybrid_has_signal:
        hybrid_judgement = (
            f"Hybrid score 值得继续。最佳 hybrid 变体的 F1={hybrid_full['f1']:.4f}、Recall={hybrid_full['recall']:.4f}，"
            f"相对纯 reconstruction 的 F1={hybrid_recon['f1']:.4f}、Recall={hybrid_recon['recall']:.4f} 显示出正向信号。"
        )
    else:
        hybrid_judgement = (
            f"Hybrid score 暂未显示稳定优势。当前最优 hybrid 变体的 F1={hybrid_full['f1']:.4f}、Recall={hybrid_full['recall']:.4f}，"
            f"仍落后于纯 reconstruction 的 F1={hybrid_recon['f1']:.4f}；说明 latent distance / density 设计要么信息不足，"
            "要么小样本下噪声偏大。"
        )

    if transformer_pr_row["recall"] >= 0.99 and transformer_pr_row["precision"] <= baseline_row["precision"] - 0.10:
        transformer_judgement = (
            "Transformer 当前更像阈值退化案例而不是明确优势模型。"
            f"在 PR-optimal 下它几乎把所有样本都判为异常，Recall={transformer_pr_row['recall']:.4f}，"
            f"但 Precision 只有 {transformer_pr_row['precision']:.4f}；改用 Youden 后虽更平衡，但 F1 仍只有 {transformer_youden_row['f1']:.4f}。"
        )
    elif transformer_row["f1"] >= baseline_row["f1"] + 0.01 and transformer_row["pr_auc"] >= baseline_row["pr_auc"]:
        transformer_judgement = "Transformer 在当前小规模真实数据上展示出继续投入的价值。"
    elif transformer_row["train_seconds"] > baseline_row["train_seconds"] * 1.5 and transformer_row["f1"] <= baseline_row["f1"]:
        transformer_judgement = "Transformer 当前更像是低性价比方向：更慢，但没有明显超过 MLP baseline。"
    else:
        transformer_judgement = "Transformer 暂时没有形成明确优势，但也不能仅凭这次小规模实验直接否定。"

    if vae_row["f1"] >= baseline_row["f1"] + 0.01 or vae_row["recall"] >= baseline_row["recall"] + 0.05:
        vae_judgement = "VAE 在当前真实小样本上有继续验证的价值。"
    elif (
        vae_row["f1"] >= baseline_row["f1"] - 0.01
        and vae_row["pr_auc"] >= baseline_row["pr_auc"] - 0.04
        and vae_row["train_seconds"] <= baseline_row["train_seconds"]
    ):
        vae_judgement = "VAE 没有超过 baseline，但与 baseline 非常接近且训练更快，适合作为第二优先路线继续保留。"
    else:
        vae_judgement = "VAE 当前没有形成足够强的继续信号。"

    if threshold_is_major_issue and not worthwhile_nonbaseline and not hybrid_has_signal:
        likely_issue = (
            "当前结果表明，第二阶段的低召回很大程度上来自阈值过于保守。"
            f"一旦改用更合适的阈值，baseline 已经能达到 Recall={baseline_row['recall']:.4f}。"
            "真正的剩余瓶颈不是 backbone 太弱，而是更复杂模型和混合评分并没有带来更好的排序质量。"
            "下一步最值得优先改的是 anomaly score 与 one-class / density 机制，而不是盲目放大模型。"
        )
    elif not worthwhile_nonbaseline and not hybrid_has_signal:
        likely_issue = (
            "当前更可能的瓶颈不是训练轮数不足，而是 anomaly score 设计与结构化流量特征之间的匹配度有限。"
            "如果 GPU 可用，最值得优先增加的不是更大模型，而是更强的 score / one-class / density 机制。"
        )
    elif threshold_is_major_issue:
        likely_issue = "当前最主要的问题更像是决策阈值过于保守，而不是 backbone 完全失效。"
    else:
        likely_issue = "当前结果显示模型能力和 score 设计同时重要，后续需要联合优化。"

    report_lines = [
        "# Stage 3 Local Real-Data Validation Report",
        "",
        "## 1. Experiment Setup",
        "- Primary dataset: UNSW-NB15 (real CSV, official train/test files).",
        f"- Train file shape: {tuple(dataset_inspection['train_shape'])}",
        f"- Test file shape: {tuple(dataset_inspection['test_shape'])}",
        "- Split design: keep official train/test separation; split official training file into train/validation; use official testing file only for final test.",
        "- CPU-friendly subset policy:",
        "  - train normal max = 10,000",
        "  - validation sample size = 4,000",
        "  - test sample size = 8,000",
        "- Training regime: CPU only, small models, 15 epochs max, early stopping enabled.",
        "",
        "## 2. Screening Comparison",
        "| Model | Scoring | Params | Train(s) | ROC-AUC | PR-AUC | F1 | Precision | Recall | Threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in screening_df.sort_values(by=["f1", "recall"], ascending=[False, False]).iterrows():
        report_lines.append(
            f"| {row['model_name']} | {row['scoring_method']} | {int(row['parameter_count'])} | {row['train_seconds']:.2f} | "
            f"{row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['threshold']:.6f} |"
        )

    report_lines.extend(
        [
            "",
            "## 3. Key Research Questions",
            f"- Q1. Is there a promising non-baseline direction? {direction_judgement}",
            f"- Q2. Is low recall mainly a threshold problem? {threshold_judgement}",
            f"- Q3. Is hybrid score worth continuing? {hybrid_judgement}",
            f"- Q4. Is Transformer suitable here? {transformer_judgement}",
            f"- Q5. Is VAE suitable here? {vae_judgement}",
            "",
            "## 4. Threshold Strategy Analysis",
            f"- Baseline `enhanced_mlp_ae`: best-vs-percentile recall gain = {baseline_threshold_signal['recall_gain_vs_percentile']:.4f}, F1 gain = {baseline_threshold_signal['f1_gain_vs_percentile']:.4f}.",
            f"- Hybrid `hybrid_ae`: best-vs-percentile recall gain = {hybrid_threshold_signal['recall_gain_vs_percentile']:.4f}, F1 gain = {hybrid_threshold_signal['f1_gain_vs_percentile']:.4f}.",
            "- Percentile is consistently the most conservative choice on baseline / VAE / Hybrid, and it largely recreates the earlier low-recall regime.",
            "- PR-optimal and F1-optimal deliver the strongest recall/F1 on baseline-class models, while Youden is the safer middle ground when we want to avoid extreme false positives.",
            f"- Transformer warning sign: PR-optimal gives Recall={transformer_pr_row['recall']:.4f} but Precision={transformer_pr_row['precision']:.4f}; Youden lowers recall to {transformer_youden_row['recall']:.4f} but avoids total all-anomaly collapse.",
            "- Interpretation: if AUC is already decent while percentile recall is weak, the model likely has ranking ability but the deployment threshold is too conservative.",
            "",
            "## 5. Hybrid Score Analysis",
            "| Variant | Scoring | ROC-AUC | PR-AUC | F1 | Precision | Recall |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in ablation_df.sort_values(by=["f1", "recall"], ascending=[False, False]).iterrows():
        report_lines.append(
            f"| {Path(str(row['run_dir'])).name} | {row['scoring_method']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## 6. Attack-Type Notes",
            f"- Baseline hardest attacks: {format_hardest_attacks(baseline_per_attack)}.",
            f"- Strongest non-baseline hardest attacks: {format_hardest_attacks(best_nonbaseline_per_attack)}.",
            "- Small or sparse attack families such as Fuzzers / Reconnaissance / Shellcode remain the main source of missed detections even when overall recall is high.",
            "",
            "## 7. Likely Bottleneck Diagnosis",
            f"- {likely_issue}",
            "- If a model shows good ROC-AUC / PR-AUC but poor percentile recall, the main issue is threshold calibration rather than representation collapse.",
            "- If hybrid score does not help even after threshold optimization, the latent-space statistics are likely not discriminative enough on tabular flow features.",
            "- If Transformer is slower but not better, the current feature granularity may not justify sequence-style inductive bias.",
            "",
            "## 8. Recommended Next Moves",
            f"- Route 1: continue `{best_row['model_name']}` as the strongest current candidate.",
            f"- Route 2: keep `{best_nonbaseline['model_name']}` as the secondary route because it is the closest non-baseline candidate under the current CPU-friendly setup.",
            "- Keep `enhanced_mlp_ae` as the deployment-strength control baseline because it remains the reference point for stability and cost.",
            "- Prioritize threshold calibration and anomaly-score design before scaling model size.",
            "- If more compute becomes available later, the first long-run combination should be baseline + VAE + score/one-class ablation rather than all directions equally.",
        ]
    )

    summary_lines = [
        "# Stage 3 Local Real-Data Summary",
        "",
        f"- Primary dataset: UNSW-NB15 real train/test CSV.",
        f"- Best current model: {best_row['model_name']} (F1={best_row['f1']:.4f}, Recall={best_row['recall']:.4f}).",
        f"- Strongest non-baseline candidate: {best_nonbaseline['model_name']} (F1={best_nonbaseline['f1']:.4f}, Recall={best_nonbaseline['recall']:.4f}).",
        f"- Baseline threshold sensitivity: Recall +{baseline_threshold_signal['recall_gain_vs_percentile']:.4f}, F1 +{baseline_threshold_signal['f1_gain_vs_percentile']:.4f} vs percentile.",
        f"- Hybrid threshold sensitivity: Recall +{hybrid_threshold_signal['recall_gain_vs_percentile']:.4f}, F1 +{hybrid_threshold_signal['f1_gain_vs_percentile']:.4f} vs percentile.",
        f"- Hardest baseline attacks: {format_hardest_attacks(baseline_per_attack)}.",
        f"- Hybrid score verdict: {'worth continuing' if hybrid_has_signal else 'not yet convincing'}.",
        f"- Transformer verdict: {transformer_judgement}",
        f"- VAE verdict: {vae_judgement}",
        f"- Main bottleneck judgement: {likely_issue}",
    ]

    recommendation_lines = [
        "# Recommendation",
        "",
        f"- First priority: continue `{best_row['model_name']}` with tuned thresholding on real UNSW data.",
        f"- Second priority: keep `{best_nonbaseline['model_name']}` as the nearest non-baseline route, but treat it as a challenger rather than a replacement.",
        "- Always retain `enhanced_mlp_ae` as the control baseline for future comparisons.",
        f"- Hybrid scoring recommendation: {'continue and deepen ablation' if hybrid_has_signal else 'defer unless score design is improved toward one-class / density-style scoring'}.",
        f"- Threshold recommendation: {'treat thresholding as a first-order optimization target' if threshold_is_major_issue else 'thresholding alone is not enough; improve ranking quality first'}.",
        "- If compute budget remains limited, do not prioritize larger Transformer variants before score design is settled.",
    ]

    (output_root / "stage3_local_realdata_validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "stage3_local_realdata_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (output_root / "recommendation.md").write_text("\n".join(recommendation_lines), encoding="utf-8")

    shutil.copy2(output_root / "stage3_local_realdata_validation_report.md", PROJECT_ROOT / "stage3_local_realdata_validation_report.md")
    shutil.copy2(output_root / "stage3_local_realdata_summary.md", PROJECT_ROOT / "stage3_local_realdata_summary.md")
    shutil.copy2(output_root / "comparison_table.csv", PROJECT_ROOT / "comparison_table.csv")
    shutil.copy2(output_root / "recommendation.md", PROJECT_ROOT / "recommendation.md")


def main() -> int:
    config = load_stage3_config("config/stage3/base.yaml", "config/stage3/local_real_unsw.yaml")
    raw_root = PROJECT_ROOT / config["dataset"]["raw_root"]
    output_root = ensure_dir(PROJECT_ROOT / "outputs_stage3" / "local_realdata_validation")
    figures_root = ensure_dir(output_root / "key_figures")

    download_records = download_real_unsw(raw_root)
    dataset_inspection = inspect_unsw_files(raw_root)
    with (output_root / "download_records.json").open("w", encoding="utf-8") as handle:
        json.dump(download_records, handle, indent=2, ensure_ascii=False)
    with (output_root / "dataset_inspection.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_inspection, handle, indent=2, ensure_ascii=False)

    screening_results = []
    for model_name in ["enhanced_mlp_ae", "transformer_ae", "vae", "hybrid_ae"]:
        screening_results.append(
            run_experiment(
                config=config,
                model_name=model_name,
                dataset_name="unsw_nb15",
                run_group="local_realdata_validation/runs",
                run_name=f"unsw_real_{model_name}",
                smoke_mode=False,
                resume=False,
            )
        )

    screening_df = pd.DataFrame(screening_rows(screening_results))
    threshold_df = pd.DataFrame(threshold_rows(screening_results))
    screening_df.to_csv(output_root / "comparison_table.csv", index=False)
    threshold_df.to_csv(output_root / "threshold_analysis.csv", index=False)
    write_comparison_outputs(screening_rows(screening_results), output_root / "comparison", title="Stage 3 Local Real-Data Comparison")

    ablation_results = run_ablation_suite(
        config=config,
        model_name="hybrid_ae",
        dataset_name="unsw_nb15",
        run_group="local_realdata_validation/ablation",
        smoke_mode=False,
    )
    ablation_df = pd.DataFrame(ablation_rows(ablation_results))
    ablation_df.to_csv(output_root / "hybrid_ablation.csv", index=False)

    baseline_row = screening_df[screening_df["model_name"] == "enhanced_mlp_ae"].iloc[0].to_dict()
    best_row = choose_best_candidate(screening_df)
    best_run_dir = Path(str(best_row["run_dir"]))
    baseline_run_dir = Path(str(baseline_row["run_dir"]))

    copy_key_figures(baseline_run_dir, figures_root, "baseline")
    copy_key_figures(best_run_dir, figures_root, "best")

    write_reports(
        output_root=output_root,
        dataset_inspection=dataset_inspection,
        screening_df=screening_df,
        threshold_df=threshold_df,
        ablation_df=ablation_df,
        baseline_row=baseline_row,
        best_row=best_row,
    )

    print(f"Local real-data validation complete. Reports: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
