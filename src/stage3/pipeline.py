from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from .data import prepare_dataset
from .evaluator import evaluate_threshold_suite
from .models import count_parameters, create_model
from .reporting import collect_run_metrics, write_comparison_outputs, write_run_outputs
from .scoring import AnomalyScorer
from .trainer import Trainer
from .utils import ensure_dir, save_json, set_seed, slugify, timestamp


def create_run_directory(config: dict[str, Any], dataset_name: str, model_name: str, run_group: str | None, run_name: str | None = None) -> Path:
    project_root = Path(config["_project_root"])
    outputs_root = ensure_dir(project_root / config["outputs"]["root"])
    group_dir = ensure_dir(outputs_root / (run_group or config["outputs"]["run_group"]))
    run_slug = slugify(run_name or f"{dataset_name}_{model_name}_{timestamp()}")
    run_dir = ensure_dir(group_dir / run_slug)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "tables")
    ensure_dir(run_dir / "reports")
    return run_dir


def run_experiment(
    config: dict[str, Any],
    model_name: str | None = None,
    dataset_name: str | None = None,
    run_group: str | None = None,
    run_name: str | None = None,
    smoke_mode: bool = False,
    resume: bool | None = None,
) -> dict[str, Any]:
    set_seed(int(config["experiment"]["seed"]))
    dataset_name = dataset_name or config["dataset"]["name"]
    model_name = model_name or config["model"]["name"]
    run_dir = create_run_directory(config, dataset_name, model_name, run_group=run_group, run_name=run_name)

    bundle = prepare_dataset(config, dataset_name=dataset_name, smoke_mode=smoke_mode)
    model, model_cfg = create_model(model_name, bundle.input_dim, config)
    trainer = Trainer(model=model, config=config, run_dir=run_dir)
    training_summary = trainer.fit(
        train_features=bundle.train_normal_features,
        val_features=bundle.val_normal_features,
        model_config=model_cfg,
        resume=bool(config["training"].get("resume", False) if resume is None else resume),
    )

    scorer = AnomalyScorer(model=model, model_config=model_cfg, scoring_config=config["scoring"], device=trainer.device)
    scorer.fit(bundle.train_normal_features)
    train_scores_payload = scorer.score(bundle.train_normal_features)
    val_scores_payload = scorer.score(bundle.val_features)
    test_scores_payload = scorer.score(bundle.test_features)

    threshold_results = evaluate_threshold_suite(
        train_scores=train_scores_payload["scores"],
        val_scores=val_scores_payload["scores"],
        val_labels=bundle.val_labels,
        test_scores=test_scores_payload["scores"],
        test_labels=bundle.test_labels,
        attack_labels=bundle.test_attack_labels,
        threshold_cfg=config["threshold"],
    )
    primary_method = str(config["threshold"]["primary_method"])
    primary_metrics = next((item for item in threshold_results if item["threshold_method"] == primary_method), threshold_results[0])

    context = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "model_parameters": count_parameters(model),
        "input_dim": bundle.input_dim,
        "scoring_method": model_cfg.get("scoring", {}).get("method", "reconstruction"),
        "primary_threshold_method": primary_method,
        "training_summary": training_summary,
        "dataset_metadata": bundle.metadata,
    }
    save_json(run_dir / "context.json", context)
    save_json(run_dir / "resolved_config.json", config)
    write_run_outputs(
        run_dir=run_dir,
        context=context,
        history=training_summary["history"],
        primary_metrics=primary_metrics,
        threshold_results=threshold_results,
        component_scores={key: value[: min(len(value), 256)].tolist() for key, value in test_scores_payload["components"].items()},
    )
    return {"run_dir": str(run_dir), "context": context, "primary_metrics": primary_metrics, "threshold_results": threshold_results}


def evaluate_existing_run(run_dir: str | Path, config: dict[str, Any], smoke_mode: bool = False) -> dict[str, Any]:
    import json

    run_path = Path(run_dir)
    with (run_path / "context.json").open("r", encoding="utf-8") as handle:
        stored_context = json.load(handle)

    dataset_name = stored_context["dataset_name"]
    model_name = stored_context["model_name"]
    checkpoint = torch.load(run_path / "checkpoints" / "best.ckpt", map_location="cpu")

    bundle = prepare_dataset(config, dataset_name=dataset_name, smoke_mode=smoke_mode)
    model, model_cfg = create_model(model_name, bundle.input_dim, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] != "cpu" else "cpu")
    model.to(device)

    scorer = AnomalyScorer(model=model, model_config=model_cfg, scoring_config=config["scoring"], device=device)
    scorer.fit(bundle.train_normal_features)
    train_scores_payload = scorer.score(bundle.train_normal_features)
    val_scores_payload = scorer.score(bundle.val_features)
    test_scores_payload = scorer.score(bundle.test_features)
    threshold_results = evaluate_threshold_suite(
        train_scores=train_scores_payload["scores"],
        val_scores=val_scores_payload["scores"],
        val_labels=bundle.val_labels,
        test_scores=test_scores_payload["scores"],
        test_labels=bundle.test_labels,
        attack_labels=bundle.test_attack_labels,
        threshold_cfg=config["threshold"],
    )
    primary_method = str(config["threshold"]["primary_method"])
    primary_metrics = next((item for item in threshold_results if item["threshold_method"] == primary_method), threshold_results[0])
    write_run_outputs(
        run_dir=run_path,
        context=stored_context,
        history=list(checkpoint.get("history", [])),
        primary_metrics=primary_metrics,
        threshold_results=threshold_results,
        component_scores={key: value[: min(len(value), 256)].tolist() for key, value in test_scores_payload["components"].items()},
    )
    return {"run_dir": str(run_path), "primary_metrics": primary_metrics, "threshold_results": threshold_results}


def run_ablation_suite(config: dict[str, Any], model_name: str, dataset_name: str, run_group: str, smoke_mode: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    base_model_cfg = config["models"][model_name]
    for variant in config.get("ablation", {}).get("variants", []):
        variant_cfg = copy.deepcopy(config)
        model_variant = copy.deepcopy(base_model_cfg)
        if variant.get("scoring_method"):
            model_variant.setdefault("scoring", {})["method"] = variant["scoring_method"]
        if variant.get("hybrid_components"):
            model_variant.setdefault("scoring", {})["components"] = variant["hybrid_components"]
        variant_cfg["models"][model_name] = model_variant
        result = run_experiment(
            config=variant_cfg,
            model_name=model_name,
            dataset_name=dataset_name,
            run_group=run_group,
            run_name=f"{dataset_name}_{model_name}_{variant['name']}",
            smoke_mode=smoke_mode,
        )
        results.append(result)
    return results


def summarize_runs(config: dict[str, Any], root_dir: str | Path | None = None, title: str = "Stage 3 Comparison") -> Path:
    project_root = Path(config["_project_root"])
    target_root = Path(root_dir) if root_dir else project_root / config["outputs"]["root"]
    records = collect_run_metrics(target_root)
    output_dir = ensure_dir(target_root / config["outputs"]["comparison_dir"])
    write_comparison_outputs(records, output_dir=output_dir, title=title)
    return output_dir
