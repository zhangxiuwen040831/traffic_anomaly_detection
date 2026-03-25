from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import BaseAnomalyModel
from .utils import ensure_dir


class Trainer:
    def __init__(self, model: BaseAnomalyModel, config: dict[str, Any], run_dir: Path) -> None:
        self.model = model
        self.config = config
        self.run_dir = run_dir
        self.checkpoint_dir = ensure_dir(run_dir / "checkpoints")
        self.logs_dir = ensure_dir(run_dir / "logs")
        self.history_path = self.logs_dir / "training_history.csv"
        self.logger = self._build_logger()
        self.device = self._get_device()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"stage3_trainer_{self.run_dir.name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(self.logs_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _get_device(self) -> torch.device:
        setting = str(self.config["training"]["device"]).lower()
        if setting == "cpu":
            return torch.device("cpu")
        if setting == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if setting == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        learning_rate = float(self.config["training"]["learning_rate"])
        weight_decay = float(self.config["training"]["weight_decay"])
        optimizer_name = str(self.config["training"]["optimizer"]).lower()
        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        if str(self.config["training"].get("scheduler", "none")).lower() != "plateau":
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.config["training"]["scheduler_factor"]),
            patience=int(self.config["training"]["scheduler_patience"]),
        )

    def _make_loader(self, features: np.ndarray, shuffle: bool) -> DataLoader:
        tensor = torch.tensor(features, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        return DataLoader(
            dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            shuffle=shuffle,
            num_workers=int(self.config["training"]["num_workers"]),
            pin_memory=self.device.type == "cuda",
        )

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best.ckpt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last.ckpt"

    def _append_history(self, row: dict[str, Any]) -> None:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "train_reconstruction_loss",
            "val_reconstruction_loss",
            "train_kl_loss",
            "val_kl_loss",
            "learning_rate",
        ]
        is_new_file = not self.history_path.exists()
        with self.history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if is_new_file:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        aggregates: dict[str, float] = {}
        batches = 0

        for batch in loader:
            inputs = batch[0].to(self.device)
            if training:
                optimizer.zero_grad()
            outputs = self.model(inputs)
            losses = self.model.compute_loss(outputs, inputs)
            if training:
                losses["loss"].backward()
                grad_clip = float(self.config["training"].get("grad_clip_norm", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                optimizer.step()
            for key, value in losses.items():
                aggregates[key] = aggregates.get(key, 0.0) + float(value.detach().cpu().item())
            batches += 1

        if batches == 0:
            return {"loss": 0.0}
        return {key: value / batches for key, value in aggregates.items()}

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
        best_metric: float,
        history: list[dict[str, Any]],
        model_config: dict[str, Any],
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_metric": best_metric,
                "history": history,
                "model_config": model_config,
                "training_config": self.config["training"],
            },
            path,
        )

    def fit(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        model_config: dict[str, Any],
        resume: bool = False,
    ) -> dict[str, Any]:
        self.model.to(self.device)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)

        start_epoch = 1
        best_metric = float("inf")
        best_epoch = 0
        history: list[dict[str, Any]] = []
        patience_counter = 0

        if resume and self.last_checkpoint_path.exists():
            checkpoint = torch.load(self.last_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and checkpoint.get("scheduler_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint.get("best_metric", best_metric))
            history = list(checkpoint.get("history", []))
            self.logger.info("Resumed training from epoch %s", start_epoch)

        train_loader = self._make_loader(train_features, shuffle=True)
        val_loader = self._make_loader(val_features if len(val_features) > 0 else train_features, shuffle=False)

        total_epochs = int(self.config["training"]["epochs"])
        early_stopping_patience = int(self.config["training"]["early_stopping_patience"])
        min_delta = float(self.config["training"].get("min_delta", 0.0))

        self.logger.info("Training on device=%s for %s epochs", self.device, total_epochs)
        train_start = time.perf_counter()
        for epoch in range(start_epoch, total_epochs + 1):
            train_stats = self._run_epoch(train_loader, optimizer)
            with torch.no_grad():
                val_stats = self._run_epoch(val_loader, optimizer=None)

            current_lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "train_loss": train_stats.get("loss", 0.0),
                "val_loss": val_stats.get("loss", 0.0),
                "train_reconstruction_loss": train_stats.get("reconstruction_loss", 0.0),
                "val_reconstruction_loss": val_stats.get("reconstruction_loss", 0.0),
                "train_kl_loss": train_stats.get("kl_loss", 0.0),
                "val_kl_loss": val_stats.get("kl_loss", 0.0),
                "learning_rate": current_lr,
            }
            history.append(row)
            self._append_history(row)

            monitored = row["val_loss"]
            if scheduler:
                scheduler.step(monitored)

            self._save_checkpoint(
                self.last_checkpoint_path,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_metric,
                history=history,
                model_config=model_config,
            )

            self.logger.info(
                "Epoch %s/%s | train_loss=%.6f | val_loss=%.6f | lr=%.6f",
                epoch,
                total_epochs,
                row["train_loss"],
                row["val_loss"],
                current_lr,
            )

            if monitored < best_metric - min_delta:
                best_metric = monitored
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(
                    self.best_checkpoint_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    history=history,
                    model_config=model_config,
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info("Early stopping triggered at epoch %s", epoch)
                    break

        if self.best_checkpoint_path.exists():
            best_state = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(best_state["model_state_dict"])

        total_train_seconds = time.perf_counter() - train_start

        return {
            "device": str(self.device),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_metric,
            "best_checkpoint": str(self.best_checkpoint_path),
            "last_checkpoint": str(self.last_checkpoint_path),
            "total_train_seconds": total_train_seconds,
        }
