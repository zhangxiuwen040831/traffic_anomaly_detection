from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_defaults(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    config.setdefault("experiment", {})
    config.setdefault("dataset", {})
    config.setdefault("model", {})
    config.setdefault("models", {})
    config.setdefault("training", {})
    config.setdefault("scoring", {})
    config.setdefault("threshold", {})
    config.setdefault("evaluation", {})
    config.setdefault("outputs", {})
    config.setdefault("ablation", {})
    config.setdefault("cloud", {})

    config["experiment"].setdefault("seed", 42)
    config["experiment"].setdefault("stage", "stage3")

    config["dataset"].setdefault("name", "unsw_nb15")
    config["dataset"].setdefault("raw_root", "data/raw")
    config["dataset"].setdefault("processed_root", "data/stage3/processed")
    config["dataset"].setdefault("cache_enabled", True)
    config["dataset"].setdefault("force_refresh", False)
    config["dataset"].setdefault("smoke_max_rows", 4096)
    config["dataset"].setdefault("train_size", 0.6)
    config["dataset"].setdefault("val_size", 0.2)
    config["dataset"].setdefault("test_size", 0.2)
    config["dataset"].setdefault("normal_label", 0)
    config["dataset"].setdefault("label_column", "label")
    config["dataset"].setdefault("label_column_candidates", ["label", "Label", " Label"])
    config["dataset"].setdefault("attack_column_candidates", ["attack_cat", "attack", "Attack", " Label"])
    config["dataset"].setdefault("drop_columns", [])
    config["dataset"].setdefault("synthetic_fallback_rows", 4096)
    config["dataset"].setdefault("datasets", {})

    config["model"].setdefault("name", "enhanced_mlp_ae")
    config["model"].setdefault("checkpoint_metric", "val_loss")

    config["training"].setdefault("device", "auto")
    config["training"].setdefault("epochs", 10)
    config["training"].setdefault("batch_size", 128)
    config["training"].setdefault("learning_rate", 1e-3)
    config["training"].setdefault("weight_decay", 1e-5)
    config["training"].setdefault("optimizer", "adamw")
    config["training"].setdefault("scheduler", "plateau")
    config["training"].setdefault("scheduler_patience", 5)
    config["training"].setdefault("scheduler_factor", 0.5)
    config["training"].setdefault("early_stopping_patience", 10)
    config["training"].setdefault("min_delta", 1e-4)
    config["training"].setdefault("grad_clip_norm", 5.0)
    config["training"].setdefault("num_workers", 0)
    config["training"].setdefault("resume", False)

    config["scoring"].setdefault("batch_size", 512)
    config["scoring"].setdefault("covariance_regularization", 1e-4)

    config["threshold"].setdefault("primary_method", "pr_optimal")
    config["threshold"].setdefault("methods", ["percentile", "f1_optimal", "youden", "pr_optimal"])
    config["threshold"].setdefault("percentile", 95)

    config["evaluation"].setdefault("positive_label", 1)
    config["evaluation"].setdefault("save_component_plots", True)

    config["outputs"].setdefault("root", "outputs_stage3")
    config["outputs"].setdefault("run_group", "local_smoke_test")
    config["outputs"].setdefault("comparison_dir", "comparison")
    config["outputs"].setdefault("ablation_dir", "ablation")
    config["outputs"].setdefault("final_report_dir", "final_report")

    config["_config_path"] = str(config_path.resolve())
    config["_project_root"] = str(config_path.resolve().parents[2])
    return config


def load_stage3_config(
    config_path: str | Path,
    override_path: str | Path | None = None,
    inline_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if override_path:
        override_file = Path(override_path)
        if override_file.exists():
            with override_file.open("r", encoding="utf-8") as handle:
                override = yaml.safe_load(handle) or {}
            config = deep_merge(config, override)

    if inline_overrides:
        config = deep_merge(config, inline_overrides)

    return _apply_defaults(config, config_file)
