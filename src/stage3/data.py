from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .utils import ensure_dir, load_json, resolve_path, save_json


@dataclass
class PreparedDataset:
    dataset_name: str
    feature_names: list[str]
    metadata: dict[str, Any]
    cache_dir: Path
    train_features: np.ndarray
    train_labels: np.ndarray
    train_attack_labels: np.ndarray
    train_normal_features: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    val_attack_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    test_attack_labels: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.train_features.shape[1])

    @property
    def val_normal_features(self) -> np.ndarray:
        return self.val_features[self.val_labels == self.metadata["normal_label"]]

    def save(self) -> None:
        ensure_dir(self.cache_dir)
        np.savez_compressed(
            self.cache_dir / "bundle.npz",
            train_features=self.train_features,
            train_labels=self.train_labels,
            train_attack_labels=self.train_attack_labels,
            train_normal_features=self.train_normal_features,
            val_features=self.val_features,
            val_labels=self.val_labels,
            val_attack_labels=self.val_attack_labels,
            test_features=self.test_features,
            test_labels=self.test_labels,
            test_attack_labels=self.test_attack_labels,
            feature_names=np.asarray(self.feature_names, dtype=object),
        )
        save_json(self.cache_dir / "metadata.json", self.metadata)

    @classmethod
    def load(cls, cache_dir: str | Path) -> "PreparedDataset":
        cache_path = Path(cache_dir)
        arrays = np.load(cache_path / "bundle.npz", allow_pickle=True)
        metadata = load_json(cache_path / "metadata.json")
        return cls(
            dataset_name=metadata["dataset_name"],
            feature_names=list(arrays["feature_names"]),
            metadata=metadata,
            cache_dir=cache_path,
            train_features=arrays["train_features"],
            train_labels=arrays["train_labels"],
            train_attack_labels=arrays["train_attack_labels"],
            train_normal_features=arrays["train_normal_features"],
            val_features=arrays["val_features"],
            val_labels=arrays["val_labels"],
            val_attack_labels=arrays["val_attack_labels"],
            test_features=arrays["test_features"],
            test_labels=arrays["test_labels"],
            test_attack_labels=arrays["test_attack_labels"],
        )


class BaseDatasetAdapter:
    name = "base"

    def __init__(self, dataset_cfg: dict[str, Any]):
        self.dataset_cfg = dataset_cfg
        self.dataset_meta = dataset_cfg.get("datasets", {}).get(self.name, {})

    def _label_candidates(self) -> list[str]:
        return list(self.dataset_meta.get("label_column_candidates", self.dataset_cfg["label_column_candidates"]))

    def _attack_candidates(self) -> list[str]:
        return list(self.dataset_meta.get("attack_column_candidates", self.dataset_cfg["attack_column_candidates"]))

    def _read_csv(self, path: Path, nrows: int | None = None) -> pd.DataFrame:
        for encoding in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(path, nrows=nrows, low_memory=False, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, nrows=nrows, low_memory=False)

    def load_raw_frame(self, raw_root: Path, smoke_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
        raise NotImplementedError

    def load_official_split_frames(
        self,
        raw_root: Path,
        train_rows: int | None,
        test_rows: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        raise NotImplementedError(f"Official split is not implemented for dataset '{self.name}'.")

    def normalize_labels(
        self,
        labels: pd.Series,
        attack_labels: pd.Series | None,
        normal_label: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        label_values = pd.Series(labels).copy()
        if pd.api.types.is_numeric_dtype(label_values):
            y = pd.to_numeric(label_values, errors="coerce").fillna(0).astype(int).to_numpy()
        else:
            normal_tokens = {str(token).strip().lower() for token in self.dataset_meta.get("normal_tokens", ["0", "normal", "benign"])}
            y = label_values.astype(str).str.strip().str.lower().apply(lambda value: 0 if value in normal_tokens else 1).astype(int).to_numpy()

        if attack_labels is None:
            attack_array = np.asarray(["" for _ in range(len(y))], dtype=object)
        else:
            attack_series = attack_labels.astype(str).replace({"nan": "", "None": ""}).to_numpy(dtype=object)
            attack_array = np.asarray(attack_series, dtype=object)
            attack_array = np.where(y == normal_label, "normal", attack_array)
            attack_array = np.where((y == 1) & (attack_array == ""), "anomaly", attack_array)
        return y, attack_array


class UNSWNB15Adapter(BaseDatasetAdapter):
    name = "unsw_nb15"

    def load_raw_frame(self, raw_root: Path, smoke_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
        candidate_paths = [
            raw_root / "UNSW_NB15_training-set.csv",
            raw_root / "UNSW_NB15_testing-set.csv",
            raw_root / "UNSW_NB15.csv",
        ]
        frames: list[pd.DataFrame] = []
        used_files: list[str] = []
        for path in candidate_paths:
            if path.exists():
                frames.append(self._read_csv(path))
                used_files.append(path.name)

        if not frames:
            generated = _generate_synthetic_frame(self.name, rows=self.dataset_cfg["synthetic_fallback_rows"])
            return generated, {"raw_source": "synthetic_fallback", "used_files": []}

        frame = pd.concat(frames, axis=0, ignore_index=True)
        if smoke_rows and len(frame) > smoke_rows:
            frame = frame.sample(n=smoke_rows, random_state=int(self.dataset_cfg.get("random_state", 42))).reset_index(drop=True)
        return frame, {"raw_source": "local_raw", "used_files": used_files}

    def load_official_split_frames(
        self,
        raw_root: Path,
        train_rows: int | None,
        test_rows: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        train_path = raw_root / self.dataset_meta.get("train_file", "UNSW_NB15_training-set.csv")
        test_path = raw_root / self.dataset_meta.get("test_file", "UNSW_NB15_testing-set.csv")

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Official UNSW split files are missing under {raw_root}")

        train_df = self._read_csv(train_path)
        test_df = self._read_csv(test_path)

        if train_rows and len(train_df) > train_rows:
            train_df = train_df.sample(n=train_rows, random_state=int(self.dataset_cfg.get("random_state", 42))).reset_index(drop=True)
        if test_rows and len(test_df) > test_rows:
            test_df = test_df.sample(n=test_rows, random_state=int(self.dataset_cfg.get("random_state", 42))).reset_index(drop=True)

        return train_df, test_df, {
            "raw_source": "official_train_test",
            "used_files": [train_path.name, test_path.name],
        }


class CICIDS2017Adapter(BaseDatasetAdapter):
    name = "cic_ids2017"

    def load_raw_frame(self, raw_root: Path, smoke_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
        csv_paths = sorted(path for path in raw_root.rglob("*.csv") if "cic" in str(path).lower() or "monday" in path.name.lower() or "friday" in path.name.lower())
        if not csv_paths:
            generated = _generate_synthetic_frame(self.name, rows=self.dataset_cfg["synthetic_fallback_rows"], include_attack_labels=True)
            return generated, {"raw_source": "synthetic_fallback", "used_files": []}

        frames: list[pd.DataFrame] = []
        used_files: list[str] = []
        rows_remaining = smoke_rows
        for path in csv_paths:
            nrows = None
            if rows_remaining:
                per_file = max(256, int(math.ceil(rows_remaining / max(len(csv_paths) - len(used_files), 1))))
                nrows = per_file
            frame = self._read_csv(path, nrows=nrows)
            frames.append(frame)
            used_files.append(str(path.relative_to(raw_root)))
            if rows_remaining:
                rows_remaining -= len(frame)
                if rows_remaining <= 0:
                    break

        frame = pd.concat(frames, axis=0, ignore_index=True)
        if smoke_rows and len(frame) > smoke_rows:
            frame = frame.sample(n=smoke_rows, random_state=int(self.dataset_cfg.get("random_state", 42))).reset_index(drop=True)
        return frame, {"raw_source": "local_raw", "used_files": used_files}


DATASET_REGISTRY = {
    UNSWNB15Adapter.name: UNSWNB15Adapter,
    CICIDS2017Adapter.name: CICIDS2017Adapter,
}


def _generate_synthetic_frame(dataset_name: str, rows: int = 4096, include_attack_labels: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    feature_count = 42 if dataset_name == "unsw_nb15" else 36
    normal_rows = int(rows * 0.65)
    anomaly_rows = rows - normal_rows

    normal = rng.normal(loc=0.0, scale=1.0, size=(normal_rows, feature_count))
    anomalies: list[np.ndarray] = []
    attack_labels: list[str] = []
    attack_types = ["DoS", "Exploits", "Fuzzers"]
    for index, attack_name in enumerate(attack_types):
        current_rows = anomaly_rows // len(attack_types) + (1 if index < anomaly_rows % len(attack_types) else 0)
        attack_block = rng.normal(loc=2.0 + index, scale=1.1 + 0.3 * index, size=(current_rows, feature_count))
        anomalies.append(attack_block)
        attack_labels.extend([attack_name] * current_rows)

    anomaly = np.vstack(anomalies) if anomalies else np.empty((0, feature_count))
    X = np.vstack([normal, anomaly])
    y = np.concatenate([np.zeros(normal_rows, dtype=int), np.ones(anomaly.shape[0], dtype=int)])

    permutation = rng.permutation(len(X))
    X = X[permutation]
    y = y[permutation]
    feature_names = [f"feature_{idx}" for idx in range(feature_count)]
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    if include_attack_labels:
        attack_series = np.asarray(["normal"] * normal_rows + attack_labels, dtype=object)[permutation]
        df["attack_cat"] = attack_series
    return df


def _infer_column(candidates: list[str], frame: pd.DataFrame) -> str | None:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _stratified_sample_indices(labels: np.ndarray, sample_size: int | None, random_state: int) -> np.ndarray:
    indices = np.arange(len(labels))
    if not sample_size or sample_size >= len(indices):
        return indices
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        rng = np.random.default_rng(random_state)
        return np.sort(rng.choice(indices, size=int(sample_size), replace=False))
    sampled, _ = train_test_split(
        indices,
        train_size=int(sample_size),
        random_state=random_state,
        stratify=labels,
    )
    return np.asarray(sampled)


def _prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    normal_mask_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    categorical_cols = train_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [column for column in train_df.columns if column not in categorical_cols]

    for column in numeric_cols:
        train_df[column] = pd.to_numeric(train_df[column], errors="coerce")
        val_df[column] = pd.to_numeric(val_df[column], errors="coerce")
        test_df[column] = pd.to_numeric(test_df[column], errors="coerce")

    numeric_means = train_df[numeric_cols].mean() if numeric_cols else pd.Series(dtype=float)
    if numeric_cols:
        train_num = train_df[numeric_cols].fillna(numeric_means).to_numpy(dtype=np.float32)
        val_num = val_df[numeric_cols].fillna(numeric_means).to_numpy(dtype=np.float32)
        test_num = test_df[numeric_cols].fillna(numeric_means).to_numpy(dtype=np.float32)
    else:
        train_num = np.empty((len(train_df), 0), dtype=np.float32)
        val_num = np.empty((len(val_df), 0), dtype=np.float32)
        test_num = np.empty((len(test_df), 0), dtype=np.float32)

    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_cat = encoder.fit_transform(train_df[categorical_cols].fillna("__nan__").astype(str)).astype(np.float32)
        val_cat = encoder.transform(val_df[categorical_cols].fillna("__nan__").astype(str)).astype(np.float32)
        test_cat = encoder.transform(test_df[categorical_cols].fillna("__nan__").astype(str)).astype(np.float32)
    else:
        train_cat = np.empty((len(train_df), 0), dtype=np.float32)
        val_cat = np.empty((len(val_df), 0), dtype=np.float32)
        test_cat = np.empty((len(test_df), 0), dtype=np.float32)

    train_features = np.concatenate([train_num, train_cat], axis=1)
    val_features = np.concatenate([val_num, val_cat], axis=1)
    test_features = np.concatenate([test_num, test_cat], axis=1)

    scaler = StandardScaler()
    fit_source = train_features[normal_mask_train]
    if len(fit_source) == 0:
        fit_source = train_features
    scaler.fit(fit_source)

    train_features = scaler.transform(train_features).astype(np.float32)
    val_features = scaler.transform(val_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)

    feature_names = list(numeric_cols) + list(categorical_cols)
    preprocessing_meta = {
        "categorical_columns": categorical_cols,
        "numerical_columns": numeric_cols,
        "feature_count": len(feature_names),
        "scaler_mean_preview": scaler.mean_[: min(5, len(scaler.mean_))].tolist(),
    }
    return train_features, val_features, test_features, train_features[normal_mask_train], feature_names, preprocessing_meta


def prepare_dataset(config: dict[str, Any], dataset_name: str | None = None, smoke_mode: bool = False) -> PreparedDataset:
    project_root = config["_project_root"]
    dataset_cfg = config["dataset"]
    dataset_name = dataset_name or dataset_cfg["name"]
    adapter_cls = DATASET_REGISTRY.get(dataset_name)
    if adapter_cls is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    raw_root = resolve_path(project_root, dataset_cfg["raw_root"])
    processed_root = resolve_path(project_root, dataset_cfg["processed_root"])
    cache_dir = processed_root / dataset_name / ("smoke" if smoke_mode else "full")

    if dataset_cfg.get("cache_enabled", True) and cache_dir.exists() and (cache_dir / "bundle.npz").exists() and not dataset_cfg.get("force_refresh", False):
        return PreparedDataset.load(cache_dir)

    adapter = adapter_cls(dataset_cfg)
    smoke_rows = dataset_cfg.get("smoke_max_rows") if smoke_mode else None
    split_random_state = int(dataset_cfg.get("random_state", config["experiment"]["seed"]))

    use_official_split = bool(dataset_cfg.get("use_official_split", False) and dataset_name == "unsw_nb15")
    if use_official_split:
        train_rows = dataset_cfg.get("official_train_sample_size")
        test_rows = dataset_cfg.get("official_test_sample_size")
        train_frame, test_frame, source_info = adapter.load_official_split_frames(raw_root, train_rows=train_rows, test_rows=test_rows)
        label_col = _infer_column(adapter._label_candidates(), train_frame) or dataset_cfg["label_column"]
        if label_col not in train_frame.columns or label_col not in test_frame.columns:
            raise ValueError(f"Label column '{label_col}' not found for official split dataset '{dataset_name}'.")

        attack_col = _infer_column(adapter._attack_candidates(), train_frame)
        drop_columns = []
        for column in dataset_cfg.get("drop_columns", []):
            if column in train_frame.columns and column not in {label_col, attack_col}:
                drop_columns.append(column)

        train_attack_series = train_frame[attack_col] if attack_col and attack_col in train_frame.columns else None
        test_attack_series = test_frame[attack_col] if attack_col and attack_col in test_frame.columns else None
        train_labels_full, train_attack_full = adapter.normalize_labels(train_frame[label_col], train_attack_series, normal_label=dataset_cfg["normal_label"])
        test_labels, test_attack = adapter.normalize_labels(test_frame[label_col], test_attack_series, normal_label=dataset_cfg["normal_label"])

        train_feature_frame = train_frame.drop(columns=[label_col] + drop_columns, errors="ignore")
        test_feature_frame = test_frame.drop(columns=[label_col] + drop_columns, errors="ignore")
        if attack_col and attack_col in train_feature_frame.columns:
            train_feature_frame = train_feature_frame.drop(columns=[attack_col], errors="ignore")
        if attack_col and attack_col in test_feature_frame.columns:
            test_feature_frame = test_feature_frame.drop(columns=[attack_col], errors="ignore")

        val_fraction = float(dataset_cfg.get("official_val_fraction", 0.2))
        X_train_df, X_val_df, y_train, y_val, attack_train, attack_val = train_test_split(
            train_feature_frame,
            train_labels_full,
            train_attack_full,
            test_size=val_fraction,
            random_state=split_random_state,
            stratify=train_labels_full,
        )

        normal_mask_train = np.asarray(y_train == dataset_cfg["normal_label"])

        if dataset_cfg.get("train_normal_max"):
            normal_indices = np.where(normal_mask_train)[0]
            sampled_normal_indices = _stratified_sample_indices(
                labels=np.ones(len(normal_indices), dtype=int),
                sample_size=int(dataset_cfg["train_normal_max"]),
                random_state=split_random_state,
            )
            keep_indices = normal_indices[sampled_normal_indices]
            X_train_df = X_train_df.iloc[keep_indices].reset_index(drop=True)
            y_train = np.asarray(y_train)[keep_indices]
            attack_train = np.asarray(attack_train, dtype=object)[keep_indices]
            normal_mask_train = np.asarray(y_train == dataset_cfg["normal_label"])

        if dataset_cfg.get("val_sample_size"):
            sampled_val = _stratified_sample_indices(np.asarray(y_val), int(dataset_cfg["val_sample_size"]), split_random_state)
            X_val_df = X_val_df.iloc[sampled_val].reset_index(drop=True)
            y_val = np.asarray(y_val)[sampled_val]
            attack_val = np.asarray(attack_val, dtype=object)[sampled_val]

        if dataset_cfg.get("test_sample_size"):
            sampled_test = _stratified_sample_indices(np.asarray(test_labels), int(dataset_cfg["test_sample_size"]), split_random_state)
            test_feature_frame = test_feature_frame.iloc[sampled_test].reset_index(drop=True)
            test_labels = np.asarray(test_labels)[sampled_test]
            test_attack = np.asarray(test_attack, dtype=object)[sampled_test]

        train_features, val_features, test_features, train_normal_features, feature_names, preprocessing_meta = _prepare_features(
            X_train_df,
            X_val_df,
            test_feature_frame,
            normal_mask_train,
        )

        metadata = {
            "dataset_name": dataset_name,
            "raw_source": source_info["raw_source"],
            "used_files": source_info["used_files"],
            "split_strategy": "official_train_test_with_train_validation_split",
            "label_column": label_col,
            "attack_column": attack_col or "",
            "normal_label": int(dataset_cfg["normal_label"]),
            "input_dim": int(train_features.shape[1]),
            "train_size": int(len(train_features)),
            "train_normal_size": int(len(train_normal_features)),
            "val_size": int(len(val_features)),
            "test_size": int(len(test_features)),
            "train_label_distribution": {
                "normal": int(np.sum(np.asarray(y_train) == dataset_cfg["normal_label"])),
                "anomaly": int(np.sum(np.asarray(y_train) != dataset_cfg["normal_label"])),
            },
            "val_label_distribution": {
                "normal": int(np.sum(np.asarray(y_val) == dataset_cfg["normal_label"])),
                "anomaly": int(np.sum(np.asarray(y_val) != dataset_cfg["normal_label"])),
            },
            "test_label_distribution": {
                "normal": int(np.sum(np.asarray(test_labels) == dataset_cfg["normal_label"])),
                "anomaly": int(np.sum(np.asarray(test_labels) != dataset_cfg["normal_label"])),
            },
            "preprocessing": preprocessing_meta,
        }

        bundle = PreparedDataset(
            dataset_name=dataset_name,
            feature_names=feature_names,
            metadata=metadata,
            cache_dir=cache_dir,
            train_features=train_features,
            train_labels=np.asarray(y_train, dtype=np.int64),
            train_attack_labels=np.asarray(attack_train, dtype=object),
            train_normal_features=np.asarray(train_normal_features, dtype=np.float32),
            val_features=np.asarray(val_features, dtype=np.float32),
            val_labels=np.asarray(y_val, dtype=np.int64),
            val_attack_labels=np.asarray(attack_val, dtype=object),
            test_features=np.asarray(test_features, dtype=np.float32),
            test_labels=np.asarray(test_labels, dtype=np.int64),
            test_attack_labels=np.asarray(test_attack, dtype=object),
        )
        bundle.save()
        return bundle

    raw_frame, source_info = adapter.load_raw_frame(raw_root, smoke_rows)

    label_col = _infer_column(adapter._label_candidates(), raw_frame) or dataset_cfg["label_column"]
    if label_col not in raw_frame.columns:
        raise ValueError(f"Label column '{label_col}' not found for dataset '{dataset_name}'.")

    attack_col = _infer_column(adapter._attack_candidates(), raw_frame)

    drop_columns = []
    for column in dataset_cfg.get("drop_columns", []):
        if column in raw_frame.columns and column not in {label_col, attack_col}:
            drop_columns.append(column)

    attack_series = raw_frame[attack_col] if attack_col and attack_col in raw_frame.columns else None
    labels, attack_labels = adapter.normalize_labels(raw_frame[label_col], attack_series, normal_label=dataset_cfg["normal_label"])

    feature_frame = raw_frame.drop(columns=[label_col] + drop_columns, errors="ignore")
    if attack_col and attack_col in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=[attack_col], errors="ignore")

    train_size = float(dataset_cfg["train_size"])
    val_size = float(dataset_cfg["val_size"])
    test_size = float(dataset_cfg["test_size"])
    total = train_size + val_size + test_size
    train_ratio = train_size / total
    holdout_ratio = 1.0 - train_ratio

    X_train_df, X_hold_df, y_train, y_hold, attack_train, attack_hold = train_test_split(
        feature_frame,
        labels,
        attack_labels,
        test_size=holdout_ratio,
        random_state=split_random_state,
        stratify=labels,
    )

    relative_test_ratio = test_size / (val_size + test_size)
    X_val_df, X_test_df, y_val, y_test, attack_val, attack_test = train_test_split(
        X_hold_df,
        y_hold,
        attack_hold,
        test_size=relative_test_ratio,
        random_state=split_random_state,
        stratify=y_hold,
    )

    normal_mask_train = np.asarray(y_train == dataset_cfg["normal_label"])
    train_features, val_features, test_features, train_normal_features, feature_names, preprocessing_meta = _prepare_features(
        X_train_df,
        X_val_df,
        X_test_df,
        normal_mask_train,
    )

    metadata = {
        "dataset_name": dataset_name,
        "raw_source": source_info["raw_source"],
        "used_files": source_info["used_files"],
        "label_column": label_col,
        "attack_column": attack_col or "",
        "normal_label": int(dataset_cfg["normal_label"]),
        "input_dim": int(train_features.shape[1]),
        "train_size": int(len(train_features)),
        "train_normal_size": int(len(train_normal_features)),
        "val_size": int(len(val_features)),
        "test_size": int(len(test_features)),
        "train_label_distribution": {
            "normal": int(np.sum(y_train == dataset_cfg["normal_label"])),
            "anomaly": int(np.sum(y_train != dataset_cfg["normal_label"])),
        },
        "val_label_distribution": {
            "normal": int(np.sum(y_val == dataset_cfg["normal_label"])),
            "anomaly": int(np.sum(y_val != dataset_cfg["normal_label"])),
        },
        "test_label_distribution": {
            "normal": int(np.sum(y_test == dataset_cfg["normal_label"])),
            "anomaly": int(np.sum(y_test != dataset_cfg["normal_label"])),
        },
        "preprocessing": preprocessing_meta,
    }

    bundle = PreparedDataset(
        dataset_name=dataset_name,
        feature_names=feature_names,
        metadata=metadata,
        cache_dir=cache_dir,
        train_features=train_features,
        train_labels=np.asarray(y_train, dtype=np.int64),
        train_attack_labels=np.asarray(attack_train, dtype=object),
        train_normal_features=np.asarray(train_normal_features, dtype=np.float32),
        val_features=np.asarray(val_features, dtype=np.float32),
        val_labels=np.asarray(y_val, dtype=np.int64),
        val_attack_labels=np.asarray(attack_val, dtype=object),
        test_features=np.asarray(test_features, dtype=np.float32),
        test_labels=np.asarray(y_test, dtype=np.int64),
        test_attack_labels=np.asarray(attack_test, dtype=object),
    )
    bundle.save()
    return bundle
