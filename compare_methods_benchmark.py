#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"

BASE_RESULT_COLUMNS = [
    "dataset_name",
    "dataset_dir",
    "task_type",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "n_classes",
    "accuracy",
    "fit_seconds",
    "predict_seconds",
    "status",
    "error",
]
EXTRA_RESULT_COLUMNS = [
    "method_name",
    "backend",
    "model_version",
    "model_path",
    "categorical_feature_indices",
    "worker_id",
    "gpu_id",
]
RESULT_COLUMNS = BASE_RESULT_COLUMNS + EXTRA_RESULT_COLUMNS


@dataclass
class LoadedDataset:
    dataset_name: str
    dataset_dir: Path
    task_type: Optional[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    categorical_feature_indices: list[int]
    n_val: int = 0

    @property
    def n_classes(self) -> int:
        return int(len(pd.unique(pd.Series(np.concatenate([self.y_train, self.y_test], axis=0)))))


@dataclass
class PredictionResult:
    y_pred: Any
    fit_seconds: float
    predict_seconds: float
    y_proba: Any | None = None
    classes: Any | None = None


@dataclass
class ResultRow:
    dataset_name: str
    dataset_dir: str
    task_type: Optional[str]
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    n_classes: int
    accuracy: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]
    method_name: str
    backend: str
    model_version: str
    model_path: Optional[str]
    categorical_feature_indices: str
    worker_id: int
    gpu_id: int


class SkipDataset(Exception):
    pass


class MethodAdapter:
    method_name = "method"
    backend = "unknown"
    model_version = "unknown"
    model_path: Optional[str] = None

    def __init__(self, args: argparse.Namespace, device: str) -> None:
        self.args = args
        self.device = device

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        categorical_feature_indices: list[int],
        dataset_name: str,
    ) -> PredictionResult:
        raise NotImplementedError


class TabPFNAdapter(MethodAdapter):
    backend = "tabpfn"

    def __init__(self, args: argparse.Namespace, device: str, version_name: str = "v2_5") -> None:
        super().__init__(args, device)
        self.version_name = version_name
        self.model_version = version_name.replace("_", ".")
        self.method_name = {
            "v2": "tabpfnv2",
            "v2_5": "tabpfn25",
            "v2_6": "tabpfn26",
            "v3": "tabpfnv3",
        }.get(version_name, f"tabpfn_{version_name}")
        attr_name = f"{self.method_name}_model_path"
        self.model_path = getattr(args, attr_name, None)
        if self.model_path is None and version_name == "v2_5":
            self.model_path = getattr(args, "tabpfn25_model_path", None)

    def _make_classifier(self, categorical_feature_indices: list[int]):
        try:
            from tabpfn import TabPFNClassifier
        except Exception as exc:
            raise RuntimeError(f"Failed to import tabpfn: {exc}") from exc

        kwargs: dict[str, Any] = {"device": self.device}
        if categorical_feature_indices:
            kwargs["categorical_features_indices"] = list(categorical_feature_indices)
        if self.model_path:
            kwargs["model_path"] = self.model_path
        return TabPFNClassifier(**kwargs)

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        categorical_feature_indices: list[int],
        dataset_name: str,
    ) -> PredictionResult:
        classifier = self._make_classifier(categorical_feature_indices)
        fit_started = time.time()
        classifier.fit(X_train, y_train)
        fit_seconds = time.time() - fit_started
        predict_started = time.time()
        y_pred = classifier.predict(X_test)
        predict_seconds = time.time() - predict_started
        return PredictionResult(y_pred=y_pred, fit_seconds=fit_seconds, predict_seconds=predict_seconds)


def load_dataset_info(dataset_dir: Path) -> dict[str, Any] | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_by_suffix(files: list[Path], suffix: str) -> Path | None:
    lower_suffix = suffix.lower()
    for file_path in files:
        if file_path.name.lower().endswith(lower_suffix):
            return file_path
    return None


def find_split_files(dataset_dir: Path):
    files = [path for path in dataset_dir.iterdir() if path.is_file()]
    n_train = find_by_suffix(files, "n_train.npy")
    c_train = find_by_suffix(files, "c_train.npy")
    y_train = find_by_suffix(files, "y_train.npy")
    n_val = find_by_suffix(files, "n_val.npy")
    c_val = find_by_suffix(files, "c_val.npy")
    y_val = find_by_suffix(files, "y_val.npy")
    n_test = find_by_suffix(files, "n_test.npy")
    c_test = find_by_suffix(files, "c_test.npy")
    y_test = find_by_suffix(files, "y_test.npy")
    if y_train is None or y_test is None:
        raise FileNotFoundError("Missing y_train.npy or y_test.npy")
    if n_train is None and c_train is None:
        raise FileNotFoundError("Missing both N_train.npy and C_train.npy")
    if n_test is None and c_test is None:
        raise FileNotFoundError("Missing both N_test.npy and C_test.npy")
    train_split = (n_train, c_train, y_train)
    val_split = (n_val, c_val, y_val) if y_val is not None and (n_val is not None or c_val is not None) else None
    test_split = (n_test, c_test, y_test)
    return train_split, val_split, test_split


def load_array(file_path: Path) -> np.ndarray:
    arr = np.load(file_path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[list(arr.files)[0]]
    return np.asarray(arr)


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    string_series = series.astype("string").fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def make_feature_frame(values: Any, *, kind: str, prefix: str) -> pd.DataFrame:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    frame = pd.DataFrame(arr)
    if kind == "numeric":
        frame = frame.apply(pd.to_numeric, errors="coerce")
    elif kind == "categorical":
        frame = frame.apply(normalize_categorical_series)
    else:
        raise ValueError(f"Unsupported feature kind: {kind}")
    frame.columns = [f"{prefix}_{i}" for i in range(frame.shape[1])]
    return frame


def make_target_array(values: Any) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_split(num_path: Path | None, cat_path: Path | None, y_path: Path, *, split_name: str):
    pieces: list[pd.DataFrame] = []
    if num_path is not None:
        pieces.append(make_feature_frame(load_array(num_path), kind="numeric", prefix=f"{split_name}_n"))
    if cat_path is not None:
        pieces.append(make_feature_frame(load_array(cat_path), kind="categorical", prefix=f"{split_name}_c"))
    if not pieces:
        raise ValueError("No feature files found for split")
    X = pieces[0] if len(pieces) == 1 else pd.concat(pieces, axis=1)
    y = make_target_array(load_array(y_path))
    if len(X) != len(y):
        raise ValueError(f"Feature/target row mismatch: X has {len(X)} rows while y has {len(y)}")
    return X, y


def categorical_indices_from_split(train_split) -> list[int]:
    num_path, cat_path, _ = train_split
    n_num = 0
    if num_path is not None:
        arr = load_array(Path(num_path))
        n_num = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    if cat_path is None:
        return []
    arr = load_array(Path(cat_path))
    n_cat = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    return list(range(n_num, n_num + n_cat))


def align_columns(reference: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    aligned.columns = list(reference.columns)
    return aligned


def load_classification_dataset(dataset_dir: Path) -> LoadedDataset:
    info = load_dataset_info(dataset_dir)
    task_type = str(info.get("task_type", "")).lower() if info else None
    if task_type not in CLASSIFICATION_TASKS:
        raise SkipDataset(f"Skipped due to task_type={task_type!r}")

    train_split, val_split, test_split = find_split_files(dataset_dir)
    categorical_feature_indices = categorical_indices_from_split(train_split)
    X_train, y_train = load_split(*train_split, split_name="train")
    n_val = 0
    if val_split is not None:
        X_val, y_val = load_split(*val_split, split_name="val")
        X_val = align_columns(X_train, X_val)
        X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_train = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
        n_val = int(len(y_val))
    X_test, y_test = load_split(*test_split, split_name="test")
    X_test = align_columns(X_train, X_test)

    return LoadedDataset(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        task_type=task_type,
        X_train=X_train,
        y_train=np.asarray(y_train),
        X_test=X_test,
        y_test=np.asarray(y_test),
        categorical_feature_indices=categorical_feature_indices,
        n_val=n_val,
    )


def empty_row_for_dataset(
    adapter: MethodAdapter,
    dataset_dir: Path,
    status: str,
    error: str,
    *,
    task_type: Optional[str],
    worker_id: int,
    gpu_id: int,
) -> ResultRow:
    return ResultRow(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir.as_posix(),
        task_type=task_type,
        n_train=0,
        n_val=0,
        n_test=0,
        n_features=0,
        n_classes=0,
        accuracy=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status=status,
        error=error,
        method_name=str(adapter.method_name),
        backend=str(adapter.backend),
        model_version=str(adapter.model_version),
        model_path=adapter.model_path,
        categorical_feature_indices="[]",
        worker_id=worker_id,
        gpu_id=gpu_id,
    )


def evaluate_one_dataset(adapter: MethodAdapter, dataset_dir: Path, worker_id: int, gpu_id: int) -> ResultRow:
    task_type: Optional[str] = None
    try:
        loaded = load_classification_dataset(dataset_dir)
        task_type = loaded.task_type
        result = adapter.fit_predict(
            loaded.X_train,
            loaded.y_train,
            loaded.X_test,
            loaded.categorical_feature_indices,
            loaded.dataset_name,
        )
        y_pred = np.asarray(result.y_pred)
        y_test = np.asarray(loaded.y_test)
        if len(y_pred) != len(y_test):
            raise ValueError(f"Prediction length mismatch: got {len(y_pred)}, expected {len(y_test)}")
        accuracy = float(np.mean(y_pred == y_test))
        return ResultRow(
            dataset_name=loaded.dataset_name,
            dataset_dir=loaded.dataset_dir.as_posix(),
            task_type=loaded.task_type,
            n_train=int(len(loaded.y_train)),
            n_val=loaded.n_val,
            n_test=int(len(y_test)),
            n_features=int(loaded.X_train.shape[1]),
            n_classes=loaded.n_classes,
            accuracy=accuracy,
            fit_seconds=float(result.fit_seconds),
            predict_seconds=float(result.predict_seconds),
            status="ok",
            error=None,
            method_name=str(adapter.method_name),
            backend=str(adapter.backend),
            model_version=str(adapter.model_version),
            model_path=adapter.model_path,
            categorical_feature_indices=json.dumps(loaded.categorical_feature_indices),
            worker_id=worker_id,
            gpu_id=gpu_id,
        )
    except SkipDataset as exc:
        return empty_row_for_dataset(adapter, dataset_dir, "skip", str(exc), task_type=task_type, worker_id=worker_id, gpu_id=gpu_id)
    except Exception as exc:
        return empty_row_for_dataset(
            adapter,
            dataset_dir,
            "fail",
            f"{type(exc).__name__}: {exc}",
            task_type=task_type,
            worker_id=worker_id,
            gpu_id=gpu_id,
        )


def rows_to_frame(rows: Iterable[ResultRow]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[RESULT_COLUMNS]


def detect_gpu_count() -> int:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            return len([line for line in completed.stdout.splitlines() if line.strip()])
    except Exception:
        pass
    return 0


def parse_gpu_id_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_workers_and_gpu_ids(args: argparse.Namespace) -> tuple[int, list[int]]:
    if str(args.gpus).strip().lower() == "auto":
        gpu_ids = list(range(detect_gpu_count()))
        if not gpu_ids:
            raise RuntimeError("No visible GPU detected. Pass --gpus explicitly if needed.")
    else:
        gpu_ids = parse_gpu_id_list(str(args.gpus))
        if not gpu_ids:
            raise ValueError("--gpus must contain at least one GPU id or use 'auto'")
    workers = len(gpu_ids) if args.workers is None else int(args.workers)
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if len(gpu_ids) != workers:
        raise ValueError(f"--gpus must contain exactly --workers ids; got {len(gpu_ids)} ids for {workers} workers")
    return workers, gpu_ids


def write_method_summary(summary_path: Path, frame: pd.DataFrame, dataset_dirs: list[Path], wall_seconds: float) -> None:
    ok_df = frame[frame["status"] == "ok"].copy() if len(frame) else pd.DataFrame()
    failed_df = frame[frame["status"] == "fail"].copy() if len(frame) else pd.DataFrame()
    skipped_df = frame[frame["status"] == "skip"].copy() if len(frame) else pd.DataFrame()
    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(frame)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        f"avg_accuracy_ok: {ok_df['accuracy'].mean():.6f}" if len(ok_df) else "avg_accuracy_ok: (none)",
        f"wall_seconds: {wall_seconds:.3f}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_method_frame(method_name: str, frame: pd.DataFrame, wall_seconds: float) -> dict[str, Any]:
    ok_df = frame[frame["status"] == "ok"].copy() if len(frame) else pd.DataFrame()
    failed_df = frame[frame["status"] == "fail"].copy() if len(frame) else pd.DataFrame()
    skipped_df = frame[frame["status"] == "skip"].copy() if len(frame) else pd.DataFrame()
    return {
        "method_name": method_name,
        "ok_count": int(len(ok_df)),
        "failed_count": int(len(failed_df)),
        "skipped_count": int(len(skipped_df)),
        "avg_accuracy_ok": float(ok_df["accuracy"].mean()) if len(ok_df) else None,
        "wall_seconds": float(wall_seconds),
    }


def format_optional_float(value: Any) -> str:
    if value is None:
        return "(none)"
    try:
        if pd.isna(value):
            return "(none)"
    except Exception:
        pass
    return f"{float(value):.6f}"


def write_comparison_markdown(path: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "| method | ok | fail | skip | avg_accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| {method} | {ok} | {fail} | {skip} | {acc} |".format(
                method=row.get("method_name"),
                ok=int(row.get("ok_count", 0)),
                fail=int(row.get("failed_count", 0)),
                skip=int(row.get("skipped_count", 0)),
                acc=format_optional_float(row.get("avg_accuracy_ok")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
