#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage3.config import load_stage3_config
from src.stage3.data import prepare_dataset
from src.stage3.pipeline import evaluate_existing_run, run_ablation_suite, run_experiment, summarize_runs
from src.stage3.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 research framework CLI")
    parser.add_argument("--config", default="config/stage3/base.yaml", help="Base Stage 3 config path")
    parser.add_argument("--override", default=None, help="Optional override config path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate one experiment")
    train_parser.add_argument("--model", default=None)
    train_parser.add_argument("--dataset", default=None)
    train_parser.add_argument("--run-group", default=None)
    train_parser.add_argument("--run-name", default=None)
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--resume", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Re-evaluate an existing run")
    eval_parser.add_argument("--run-dir", required=True)
    eval_parser.add_argument("--smoke", action="store_true")

    smoke_parser = subparsers.add_parser("smoke-test", help="Run local CPU smoke suite for required models")
    smoke_parser.add_argument("--dataset", default=None)
    smoke_parser.add_argument("--run-group", default="local_smoke_test")

    compare_parser = subparsers.add_parser("compare", help="Aggregate experiment metrics")
    compare_parser.add_argument("--root-dir", default=None)
    compare_parser.add_argument("--title", default="Stage 3 Comparison")

    summarize_parser = subparsers.add_parser("summarize", help="Alias of compare for report generation")
    summarize_parser.add_argument("--root-dir", default=None)
    summarize_parser.add_argument("--title", default="Stage 3 Summary")

    ablation_parser = subparsers.add_parser("ablation", help="Run configured ablation suite")
    ablation_parser.add_argument("--model", default="hybrid_ae")
    ablation_parser.add_argument("--dataset", default=None)
    ablation_parser.add_argument("--run-group", default="ablation")
    ablation_parser.add_argument("--smoke", action="store_true")

    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare dataset cache only")
    prepare_parser.add_argument("--dataset", default=None)
    prepare_parser.add_argument("--smoke", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_stage3_config(args.config, args.override)

    if args.command == "train":
        result = run_experiment(
            config=config,
            model_name=args.model,
            dataset_name=args.dataset,
            run_group=args.run_group,
            run_name=args.run_name,
            smoke_mode=args.smoke,
            resume=args.resume,
        )
        print(f"Run completed: {result['run_dir']}")
        print(
            "Primary metrics: "
            f"ROC-AUC={result['primary_metrics']['roc_auc']:.4f}, "
            f"PR-AUC={result['primary_metrics']['pr_auc']:.4f}, "
            f"F1={result['primary_metrics']['f1']:.4f}"
        )
        return 0

    if args.command == "eval":
        result = evaluate_existing_run(run_dir=args.run_dir, config=config, smoke_mode=args.smoke)
        print(f"Evaluation refreshed: {result['run_dir']}")
        return 0

    if args.command == "smoke-test":
        config = load_stage3_config(args.config, "config/stage3/local_smoke.yaml")
        dataset_name = args.dataset or config["dataset"]["name"]
        run_group = args.run_group
        for model_name in ["enhanced_mlp_ae", "transformer_ae", "vae", "hybrid_ae"]:
            run_experiment(
                config=config,
                model_name=model_name,
                dataset_name=dataset_name,
                run_group=run_group,
                run_name=f"{dataset_name}_{model_name}",
                smoke_mode=True,
                resume=False,
            )
        summary_dir = summarize_runs(config=config, root_dir=PROJECT_ROOT / config["outputs"]["root"] / run_group, title="Stage 3 Local Smoke Comparison")
        print(f"Smoke suite completed. Comparison outputs: {summary_dir}")
        return 0

    if args.command in {"compare", "summarize"}:
        summary_dir = summarize_runs(config=config, root_dir=args.root_dir, title=args.title)
        print(f"Comparison outputs written to: {summary_dir}")
        return 0

    if args.command == "ablation":
        results = run_ablation_suite(
            config=config,
            model_name=args.model,
            dataset_name=args.dataset or config["dataset"]["name"],
            run_group=args.run_group,
            smoke_mode=args.smoke,
        )
        summary_root = PROJECT_ROOT / config["outputs"]["root"] / args.run_group
        summary_dir = summarize_runs(config=config, root_dir=summary_root, title="Stage 3 Ablation Comparison")
        print(f"Ablation runs: {len(results)}")
        print(f"Ablation summary: {summary_dir}")
        return 0

    if args.command == "prepare-data":
        bundle = prepare_dataset(config=config, dataset_name=args.dataset, smoke_mode=args.smoke)
        output_dir = ensure_dir(PROJECT_ROOT / config["outputs"]["root"] / "dataset_prepare")
        (output_dir / "dataset_prepare_summary.txt").write_text(
            f"dataset={bundle.dataset_name}\n"
            f"input_dim={bundle.input_dim}\n"
            f"train_normal={len(bundle.train_normal_features)}\n"
            f"val={len(bundle.val_features)}\n"
            f"test={len(bundle.test_features)}\n"
            f"cache_dir={bundle.cache_dir}\n",
            encoding="utf-8",
        )
        print(f"Dataset prepared and cached at: {bundle.cache_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
