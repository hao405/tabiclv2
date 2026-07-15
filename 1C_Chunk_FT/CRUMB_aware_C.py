#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_ROOT = Path("data200")
DEFAULT_MODEL_PATH = "tabicl-classifier-v1.1-20250506.ckpt"
DEFAULT_CHECKPOINT_VERSION = "tabicl-classifier-v1.1-20250506.ckpt"
DEFAULT_OUT_DIR_ROOT = Path("1b_result")
CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"

np = None
pd = None


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
    f1: Optional[float]
    balanced_accuracy: Optional[float]
    roc_auc: Optional[float]
    log_loss: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]
    n_train_a: int = 0
    n_train_b: int = 0
    n_holdout_c: int = 0
    n_test_d: int = 0
    ttt_loss: Optional[float] = None
    ttt_wise_ft: bool = False
    ttt_wise_alpha: Optional[float] = None
    ttt_wise_applied: bool = False
    ttt_steps: int = 0
    ttt_lr: Optional[float] = None
    ttt_applied: bool = False
    ttt_update_seconds: float = 0.0
    ttt_split_strategy: Optional[str] = None
    ttt_split_reason: Optional[str] = None
    ttt_epochs: int = 0
    ttt_chunks_per_epoch: int = 0
    ttt_batch_mode: Optional[str] = None
    ttt_val_eval_metric: Optional[str] = None
    ttt_val_baseline_metric: Optional[float] = None
    ttt_val_best_metric: Optional[float] = None
    ttt_val_baseline_accuracy: Optional[float] = None
    ttt_val_best_accuracy: Optional[float] = None
    ttt_best_epoch: int = 0
    ttt_stopped_early: bool = False
    ttt_oom_fallback: bool = False
    ttt_fallback_reason: Optional[str] = None
    ttt_c_selection: Optional[str] = None
    ttt_c_source: Optional[str] = None
    ttt_c_metric: Optional[str] = None
    ttt_c_f_distance_mean: Optional[float] = None
    ttt_c_f_distance_std: Optional[float] = None
    ttt_c_fallback_reason: Optional[str] = None
    ttt_c_label_coverage_ok: bool = True


@dataclass
class TTTConfig:
    enabled: bool = False
    lr: float = 1e-5
    scheduler: str = "cosine_warmup"
    warmup_proportion: float = 0.1
    grad_clip: float = 1.0
    amp: bool | str = "auto"
    dtype: str = "float32"
    micro_batch_size: int = 1
    weight_decay: float = 0.01
    wise_ft: bool = True
    wise_alpha: float = 0.1
    epochs: int = 30
    steps: int = 30
    max_chunk_size: int = 10_000
    min_chunk_size: int = 50
    query_ratio: float = 0.2
    n_estimators_finetune: int = 2
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 1e-4
    eval_metric: str = "roc_auc"
    validation_fraction: float = 0.1
    validation_n_estimators: int = 2
    freeze_col: bool = False
    freeze_row: bool = False
    freeze_icl: bool = False
    train_fraction: float = 0.75
    random_state: int = 42
    data_parallel: bool = False
    gpu_group: Optional[str] = None
    save_ckpt: bool = True
    save_ckpt_every: int = 2
    save_ckpt_start_step: Optional[int] = None
    ckpt_root: Optional[str] = None
    c_selection: str = "crumb_mmd"
    c_source: str = "test"
    c_metric: str = "standardized_l2"
    c_candidate_multiplier: int = 5
    c_class_balance: bool = True
    c_reserve_ratio: float = 0.4
    crumb_bandwidth: str = "median"


@dataclass
class TTTSplit:
    b_indices: Any
    c_indices: Any
    strategy: str
    reason: str


@dataclass
class TTTUpdateResult:
    applied: bool
    loss: Optional[float]
    steps: int
    update_seconds: float
    wise_ft: bool = False
    wise_alpha: Optional[float] = None
    wise_applied: bool = False
    reason: Optional[str] = None
    epochs: int = 0
    chunks_per_epoch: int = 0
    batch_mode: str = "epoch_chunk"
    val_eval_metric: Optional[str] = None
    val_baseline_metric: Optional[float] = None
    val_best_metric: Optional[float] = None
    val_baseline_accuracy: Optional[float] = None
    val_best_accuracy: Optional[float] = None
    best_epoch: int = 0
    stopped_early: bool = False
    c_distance_mean: Optional[float] = None
    c_distance_std: Optional[float] = None
    c_fallback_reason: Optional[str] = None
    c_label_coverage_ok: bool = True


@dataclass
class TTTValidationResult:
    primary: float
    secondary: Dict[str, float]


@dataclass
class MetaBatch:
    X: Any
    y_train: Any
    y_query: Any
    train_size: int
    skip_reason: Optional[str] = None
    c_distance_sum: float = 0.0
    c_distance_sumsq: float = 0.0
    c_distance_count: int = 0
    c_fallback_reason: Optional[str] = None
    c_label_coverage_ok: bool = True


@dataclass
class CSelectionContext:
    reference: Any
    metric: str
    mean: Any = None
    scale: Any = None
    target_mean: Any = None
    target_var: Any = None
    reserved_context_indices: Any = None
    reserved_context_scores: Any = None
    reserved_context_ratio: float = 0.0
    reserved_context_score_sum: float = 0.0
    reserved_context_score_sumsq: float = 0.0
    reserved_context_score_count: int = 0


@dataclass
class CSelectionResult:
    ctx_idx: Any
    qry_idx: Any
    split_strategy: str
    distance_sum: float = 0.0
    distance_sumsq: float = 0.0
    distance_count: int = 0
    fallback_reason: Optional[str] = None
    label_coverage_ok: bool = True


@dataclass
class ModelSummaryRow:
    model_name: str
    model_path: str
    gpu_id: int
    datasets_discovered: int
    ok_count: int
    failed_count: int
    skipped_count: int
    ttt_oom_fallback_count: int
    avg_accuracy_ok: Optional[float]
    avg_f1_ok: Optional[float]
    avg_balanced_accuracy_ok: Optional[float]
    avg_roc_auc_ok: Optional[float]
    avg_log_loss_ok: Optional[float]
    avg_fit_seconds_ok: Optional[float]
    avg_predict_seconds_ok: Optional[float]
    avg_dataset_seconds_ok: Optional[float]
    total_dataset_seconds_ok: float
    model_wall_seconds: float
    status: str
    error: Optional[str]
    failed_datasets: str


OOM_ERROR_MARKERS = (
    "out of memory",
    "oom",
    "cuda out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "outofmemoryerror",
)


def ensure_runtime_deps() -> None:
    global np
    global pd

    if np is None or pd is None:
        import numpy as _np
        import pandas as _pd

        np = _np
        pd = _pd


def format_optional_float(value: Optional[float], precision: int = 6) -> str:
    if value is None:
        return "None"
    try:
        if pd is not None and pd.isna(value):
            return "None"
    except Exception:
        pass
    return f"{float(value):.{precision}f}"


def compute_weighted_f1(y_true: Any, y_pred: Any) -> Optional[float]:
    ensure_runtime_deps()
    try:
        from sklearn.metrics import f1_score

        return float(
            f1_score(
                np.asarray(y_true),
                np.asarray(y_pred),
                average="weighted",
                zero_division=0,
            )
        )
    except Exception:
        return None


def compute_balanced_accuracy(y_true: Any, y_pred: Any) -> Optional[float]:
    ensure_runtime_deps()
    try:
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))
    except Exception:
        return None


def compute_tabpfn_roc_auc(
    y_true: Any,
    y_proba: Any,
    classes: Any | None = None,
) -> Optional[float]:
    ensure_runtime_deps()
    try:
        from sklearn.metrics import roc_auc_score

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != y_true_arr.shape[0] or proba_arr.shape[1] < 2:
            return None
        if len(np.unique(y_true_arr)) < 2:
            return None
        class_arr = None if classes is None else np.asarray(classes)
        if class_arr is not None and (
            class_arr.ndim != 1 or len(class_arr) != proba_arr.shape[1]
        ):
            class_arr = None
        if proba_arr.shape[1] == 2:
            if class_arr is not None:
                y_binary = (y_true_arr == class_arr[1]).astype(int)
                if len(np.unique(y_binary)) < 2:
                    return None
                return float(roc_auc_score(y_binary, proba_arr[:, 1]))
            return float(roc_auc_score(y_true_arr, proba_arr[:, 1]))
        if class_arr is not None:
            return float(
                roc_auc_score(
                    y_true_arr,
                    proba_arr,
                    labels=list(class_arr),
                    multi_class="ovr",
                )
            )
        return float(roc_auc_score(y_true_arr, proba_arr, multi_class="ovr"))
    except (ValueError, RuntimeError, AttributeError, TypeError):
        return None


def compute_log_loss(
    y_true: Any,
    y_proba: Any | None,
    classes: Any | None,
) -> Optional[float]:
    if y_proba is None:
        return None
    ensure_runtime_deps()
    try:
        from sklearn.metrics import log_loss

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != len(y_true_arr):
            return None

        class_arr = None if classes is None else np.asarray(classes)
        if class_arr is not None and (
            class_arr.ndim != 1 or len(class_arr) != proba_arr.shape[1]
        ):
            class_arr = None
        if class_arr is not None:
            return float(log_loss(y_true_arr, proba_arr, labels=list(class_arr)))
        return float(log_loss(y_true_arr, proba_arr))
    except Exception:
        return None


def is_oom_exception(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        oom_types = [
            getattr(torch, "OutOfMemoryError", None),
            getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
        ]
        for oom_type in oom_types:
            if oom_type is not None and isinstance(exc, oom_type):
                return True

    text = f"{type(exc).__name__}: {exc}".strip().lower()
    return any(marker in text for marker in OOM_ERROR_MARKERS)


def format_exception_for_csv(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def append_ttt_reason(existing: Optional[str], addition: str) -> str:
    if not existing:
        return addition
    return f"{existing} | {addition}"


def truthy_column_mask(frame: Any, column: str) -> Any:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)

    values = frame[column].fillna(False)
    if values.dtype == bool:
        return values
    return values.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def format_dataset_result_log(
    worker_label: str,
    row: ResultRow,
    *,
    model_name: str | None = None,
) -> str:
    prefix = f"[{worker_label}]"
    if model_name:
        prefix = f"{prefix} [{model_name}]"

    if row.status == "ok":
        return (
            f"{prefix} [ok] {row.dataset_name} "
            f"accuracy={format_optional_float(row.accuracy)} "
            f"f1={format_optional_float(row.f1)} "
            f"balanced_accuracy={format_optional_float(row.balanced_accuracy)} "
            f"roc_auc={format_optional_float(row.roc_auc)} "
            f"log_loss={format_optional_float(row.log_loss)} "
            f"fit={row.fit_seconds:.3f}s "
            f"predict={row.predict_seconds:.3f}s "
            f"ttt_applied={row.ttt_applied} "
            f"ttt_update={row.ttt_update_seconds:.3f}s "
            f"ttt_loss={format_optional_float(row.ttt_loss)} "
            f"ttt_steps={row.ttt_steps} "
            f"ttt_epochs={row.ttt_epochs} "
            f"ttt_mode={row.ttt_batch_mode} "
            f"ttt_val_metric={row.ttt_val_eval_metric} "
            f"ttt_val_best={format_optional_float(row.ttt_val_best_metric)} "
            f"ttt_best_epoch={row.ttt_best_epoch} "
            f"ttt_stopped_early={row.ttt_stopped_early} "
            f"wise_ft={row.ttt_wise_ft} "
            f"wise_alpha={format_optional_float(row.ttt_wise_alpha)} "
            f"wise_applied={row.ttt_wise_applied} "
            f"ttt_oom_fallback={row.ttt_oom_fallback}"
        )
    if row.status == "skip":
        return f"{prefix} [skip] {row.dataset_name} reason={row.error}"
    return f"{prefix} [fail] {row.dataset_name} error={row.error}"


def parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    return int(value)


def parse_auto_bool(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be one of: auto, true, false")


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be one of: true, false")


def parse_kv_cache(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in {"false", "0", "no", "off"}:
        return False
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"kv", "repr"}:
        return lowered
    raise argparse.ArgumentTypeError("kv_cache must be one of: false, true, kv, repr")


def normalize_gpu_group(value: int | str) -> str:
    if isinstance(value, int):
        return str(value)
    gpu_group = str(value).strip()
    gpu_ids = parse_gpu_id_list(gpu_group)
    if not gpu_ids:
        raise ValueError(f"GPU group must contain at least one GPU id: {value!r}")
    return ",".join(str(gpu_id) for gpu_id in gpu_ids)


def first_gpu_id_from_group(value: int | str) -> int:
    return parse_gpu_id_list(normalize_gpu_group(value))[0]


def use_rocm_gpu_visibility() -> bool:
    backend = (os.environ.get("TFM_GPU_BACKEND") or os.environ.get("TABICL_GPU_BACKEND") or "").strip().lower()
    return backend in {"rocm", "hip", "amd"}


def apply_worker_environment_updates(gpu_id: int | str) -> str:
    gpu_id_str = normalize_gpu_group(gpu_id)
    if use_rocm_gpu_visibility():
        os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
        os.environ["HIP_VISIBLE_DEVICES"] = gpu_id_str
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return "cuda:0"


def parse_gpu_id_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_gpu_group_list(value: str) -> List[str]:
    gpu_groups = []
    for raw_group in value.split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        gpu_groups.append(normalize_gpu_group(raw_group))
    return gpu_groups


def detect_default_gpu_ids() -> List[int]:
    for env_name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            continue
        try:
            gpu_ids = parse_gpu_id_list(raw_value)
        except ValueError:
            gpu_ids = []
        if gpu_ids:
            return gpu_ids

    try:
        import torch

        device_count = int(torch.cuda.device_count())
    except Exception:
        device_count = 0

    if device_count <= 0:
        raise RuntimeError("No visible GPU detected; please pass --gpus explicitly")
    return list(range(device_count))


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def should_skip_ttt_for_dataset(dataset_dir: Path, info: dict | None) -> bool:
    dataset_names = {dataset_dir.name.strip().lower()}
    if info:
        info_name = str(info.get("name", "")).strip().lower()
        if info_name:
            dataset_names.add(info_name)
    return "volkert" in dataset_names


def find_dataset_dirs(data_root: Path) -> List[Path]:
    return [path for path in sorted(data_root.iterdir()) if path.is_dir()]


def collect_torch_diagnostics() -> Dict[str, object]:
    import torch

    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:
        device_count = f"error: {exc}"

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": device_count,
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def stable_feature_prefix(context: str, fallback: str) -> str:
    stem = Path(context or fallback).stem
    for suffix in ("-train", "-test", "-val", "-single"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or fallback


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    string_series = series.astype("string")
    string_series = string_series.fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def make_feature_frame(
    values,
    *,
    kind: str,
    prefix: str,
) -> pd.DataFrame:
    ensure_runtime_deps()

    if isinstance(values, pd.DataFrame):
        df = values.copy()
    else:
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr)

    if kind == "numeric":
        df = df.apply(pd.to_numeric, errors="coerce")
    elif kind == "categorical":
        df = df.apply(normalize_categorical_series)
    else:
        raise ValueError(f"Unsupported feature kind: {kind}")

    df.columns = [f"{prefix}_{i}" for i in range(df.shape[1])]
    return df


def make_target_array(values) -> np.ndarray:
    ensure_runtime_deps()

    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_array(file_path: Path) -> np.ndarray:
    ensure_runtime_deps()

    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)

    raise ValueError(f"Unsupported split file type: {file_path}")


def find_by_suffix(files: List[Path], suffix: str) -> Path | None:
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


def load_split(
    num_path: Path | None,
    cat_path: Path | None,
    y_path: Path,
    *,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    ensure_runtime_deps()

    features: List[pd.DataFrame] = []
    feature_prefix = stable_feature_prefix(context, y_path.stem)

    if num_path is not None:
        x_num = load_array(num_path)
        features.append(make_feature_frame(x_num, kind="numeric", prefix=f"{feature_prefix}_n"))
    if cat_path is not None:
        x_cat = load_array(cat_path)
        features.append(make_feature_frame(x_cat, kind="categorical", prefix=f"{feature_prefix}_c"))

    if not features:
        raise ValueError("No feature files found for split")

    n_samples = features[0].shape[0]
    for idx, feature_df in enumerate(features):
        if feature_df.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent number of rows across feature blocks: block {idx} has "
                f"{feature_df.shape[0]} rows but expected {n_samples}"
            )

    X = features[0] if len(features) == 1 else pd.concat(features, axis=1)
    y = make_target_array(load_array(y_path))

    if len(X) != len(y):
        raise ValueError(f"Feature/target row mismatch: X has {len(X)} rows while y has {len(y)}")

    return X, y


def normalize_model_path(model_path: str | None) -> str | None:
    if model_path is None:
        return None
    if str(model_path).strip().lower() == "none":
        return None

    path = Path(model_path).expanduser()
    try:
        if path.exists():
            path = path.resolve()
    except Exception:
        pass
    return str(path)


def extract_last_int(stem: str) -> int | None:
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    return int(numbers[-1])


def discover_model_paths(models_dir: Path, max_models: int | None = None) -> List[Path]:
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")
    if not models_dir.is_dir():
        raise NotADirectoryError(f"Models directory is not a directory: {models_dir}")

    model_paths = [
        path
        for path in models_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".ckpt", ".pt", ".pth"}
    ]
    model_paths.sort(
        key=lambda path: (
            0 if extract_last_int(path.stem) is not None else 1,
            extract_last_int(path.stem) if extract_last_int(path.stem) is not None else path.stem,
            path.stem,
        )
    )
    if max_models is not None:
        model_paths = model_paths[:max_models]
    return model_paths


def preload_model_once(classifier: Any, worker_label: str, verbose: bool) -> None:
    load_fn = getattr(classifier, "_load_model", None)
    if not callable(load_fn):
        return

    t0 = time.perf_counter()
    load_fn()

    if hasattr(classifier, "_get_model_load_key"):
        try:
            classifier._loaded_model_key = classifier._get_model_load_key()
        except Exception:
            pass

    def _skip_reloading():
        return None

    try:
        classifier._load_model = _skip_reloading
    except Exception:
        pass

    if verbose:
        print(
            f"[{worker_label}] model preloaded once in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )


def release_classifier_resources(classifier: Any) -> None:
    if classifier is None:
        return

    try:
        model = getattr(classifier, "model_", None)
        if model is not None:
            clear_cache = getattr(model, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
    except Exception:
        pass

    for attr_name in ("model_kv_cache_", "ensemble_generator_", "X_encoder_", "y_encoder_", "model_"):
        if hasattr(classifier, attr_name):
            try:
                setattr(classifier, attr_name, None)
            except Exception:
                pass


def force_memory_cleanup(device_str: str) -> None:
    gc.collect()

    try:
        import torch
    except Exception:
        return

    if not device_str.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return

    try:
        device = torch.device(device_str)
        with torch.cuda.device(device):
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        pass


def build_ttt_config(args: argparse.Namespace) -> TTTConfig:
    if int(args.ttt_save_ckpt_every) < 1:
        raise ValueError("--ttt-save-ckpt-every must be >= 1")
    if args.ttt_save_ckpt_start_step is not None and int(args.ttt_save_ckpt_start_step) < 1:
        raise ValueError("--ttt-save-ckpt-start-step must be >= 1 or None")
    if int(args.ttt_epochs) < 1:
        raise ValueError("--ttt-epochs must be >= 1")
    if int(args.ttt_max_chunk_size) < 2:
        raise ValueError("--ttt-max-chunk-size must be >= 2")
    if int(args.ttt_min_chunk_size) < 1:
        raise ValueError("--ttt-min-chunk-size must be >= 1")
    if not 0.0 < float(args.ttt_query_ratio) < 1.0:
        raise ValueError("--ttt-query-ratio must be in (0, 1)")
    if int(args.ttt_n_estimators_finetune) < 1:
        raise ValueError("--ttt-n-estimators-finetune must be >= 1")
    if int(args.ttt_patience) < 1:
        raise ValueError("--ttt-patience must be >= 1")
    if float(args.ttt_min_delta) < 0:
        raise ValueError("--ttt-min-delta must be >= 0")
    if float(args.ttt_warmup_proportion) < 0:
        raise ValueError("--ttt-warmup-proportion must be >= 0")
    if not 0.0 < float(args.ttt_validation_fraction) < 1.0:
        raise ValueError("--ttt-validation-fraction must be in (0, 1)")
    if int(args.ttt_validation_n_estimators) < 1:
        raise ValueError("--ttt-validation-n-estimators must be >= 1")
    if str(args.ttt_eval_metric) not in {"roc_auc", "log_loss", "accuracy"}:
        raise ValueError("--ttt-eval-metric must be one of: roc_auc, log_loss, accuracy")
    if not 0.0 <= float(args.ttt_wise_alpha) <= 1.0:
        raise ValueError("--ttt-wise-alpha must be in [0, 1]")
    if str(args.ttt_c_selection) not in {
        "random",
        "stratified_random",
        "f_nearest",
        "f_mmd",
        "f_density_mixed",
        "crumb_mmd",
    }:
        raise ValueError(
            "--ttt-c-selection must be one of: random, stratified_random, "
            "f_nearest, f_mmd, f_density_mixed, crumb_mmd"
        )
    if str(args.ttt_c_source) not in {"none", "validation", "test"}:
        raise ValueError("--ttt-c-source must be one of: none, validation, test")
    if str(args.ttt_c_metric) not in {"tabicl_encoded_l2", "standardized_l2"}:
        raise ValueError("--ttt-c-metric must be one of: tabicl_encoded_l2, standardized_l2")
    if int(args.ttt_c_candidate_multiplier) < 1:
        raise ValueError("--ttt-c-candidate-multiplier must be >= 1")
    if not 0.0 < float(args.ttt_c_reserve_ratio) < 1.0:
        raise ValueError("--ttt-c-reserve-ratio must be in (0, 1)")
    if str(args.ttt_c_selection) == "f_mmd" and float(args.ttt_c_reserve_ratio) + float(args.ttt_query_ratio) >= 1.0:
        raise ValueError("--ttt-c-reserve-ratio + --ttt-query-ratio must be < 1 for f_mmd")
    if str(args.ttt_c_selection) == "f_mmd" and str(args.ttt_c_source) != "test":
        raise ValueError("F-aware --ttt-c-selection f_mmd now requires --ttt-c-source test")
    if str(args.ttt_c_selection) == "crumb_mmd" and str(args.ttt_c_source) != "test":
        raise ValueError("CRUMB-aware --ttt-c-selection crumb_mmd requires --ttt-c-source test")
    if str(args.ttt_c_selection) in {"f_nearest", "f_density_mixed"} and str(args.ttt_c_source) == "none":
        raise ValueError("F-aware --ttt-c-selection requires --ttt-c-source validation or test")
    if str(args.ttt_crumb_bandwidth) != "median":
        try:
            crumb_bandwidth = float(args.ttt_crumb_bandwidth)
        except ValueError as exc:
            raise ValueError("--ttt-crumb-bandwidth must be 'median' or a positive float") from exc
        if crumb_bandwidth <= 0.0 or not math.isfinite(crumb_bandwidth):
            raise ValueError("--ttt-crumb-bandwidth must be 'median' or a positive finite float")

    return TTTConfig(
        enabled=bool(args.ttt_enabled),
        lr=float(args.ttt_lr),
        scheduler=str(args.ttt_scheduler),
        warmup_proportion=float(args.ttt_warmup_proportion),
        grad_clip=float(args.ttt_grad_clip),
        amp=args.use_amp,
        dtype=str(args.ttt_dtype),
        micro_batch_size=int(args.ttt_micro_batch_size),
        weight_decay=float(args.ttt_weight_decay),
        wise_ft=bool(args.ttt_wise_ft),
        wise_alpha=float(args.ttt_wise_alpha),
        epochs=int(args.ttt_epochs),
        steps=int(args.ttt_epochs),
        max_chunk_size=int(args.ttt_max_chunk_size),
        min_chunk_size=int(args.ttt_min_chunk_size),
        query_ratio=float(args.ttt_query_ratio),
        n_estimators_finetune=int(args.ttt_n_estimators_finetune),
        early_stopping=bool(args.ttt_early_stopping),
        patience=int(args.ttt_patience),
        min_delta=float(args.ttt_min_delta),
        eval_metric=str(args.ttt_eval_metric),
        validation_fraction=float(args.ttt_validation_fraction),
        validation_n_estimators=int(args.ttt_validation_n_estimators),
        freeze_col=bool(args.ttt_freeze_col),
        freeze_row=bool(args.ttt_freeze_row),
        freeze_icl=bool(args.ttt_freeze_icl),
        random_state=int(args.random_state),
        data_parallel=bool(args.ttt_data_parallel),
        save_ckpt=bool(args.ttt_save_ckpt),
        save_ckpt_every=int(args.ttt_save_ckpt_every),
        save_ckpt_start_step=(
            int(args.ttt_save_ckpt_start_step) if args.ttt_save_ckpt_start_step is not None else None
        ),
        ckpt_root=str((Path(args.out_dir).expanduser() / "ttt_ckpts").resolve()),
        c_selection=str(args.ttt_c_selection),
        c_source=str(args.ttt_c_source),
        c_metric=str(args.ttt_c_metric),
        c_candidate_multiplier=int(args.ttt_c_candidate_multiplier),
        c_class_balance=bool(args.ttt_c_class_balance),
        c_reserve_ratio=float(args.ttt_c_reserve_ratio),
        crumb_bandwidth=str(args.ttt_crumb_bandwidth),
    )


def take_rows(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices].reset_index(drop=True)
    return np.asarray(X)[indices]


def count_ttt_chunks(n_samples: int, max_chunk_size: int, min_chunk_size: int) -> int:
    if n_samples <= 0:
        return 0
    if n_samples <= max_chunk_size:
        return 1
    n_full = n_samples // max_chunk_size
    remainder = n_samples - n_full * max_chunk_size
    return n_full + (1 if remainder >= min_chunk_size else 0)


def _chunk_indices(
    n_samples: int,
    *,
    max_chunk_size: int,
    min_chunk_size: int,
    rng,
) -> List[Any]:
    perm = rng.permutation(n_samples)
    if n_samples <= max_chunk_size:
        return [perm]
    n_full = n_samples // max_chunk_size
    chunks = [perm[i * max_chunk_size : (i + 1) * max_chunk_size] for i in range(n_full)]
    remainder = n_samples - n_full * max_chunk_size
    if remainder >= min_chunk_size:
        chunks.append(perm[n_full * max_chunk_size :])
    return chunks


def _split_ctx_query(y_chunk, *, query_size: int, seed: int) -> tuple[Any, Any, str]:
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

    n = len(y_chunk)
    query_size = max(1, min(int(query_size), n - 1))
    dummy_X = np.zeros((n, 1))
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=query_size, random_state=seed)
        ctx_idx, qry_idx = next(splitter.split(dummy_X, y_chunk))
        return ctx_idx, qry_idx, "stratified"
    except Exception as exc:
        splitter = ShuffleSplit(n_splits=1, test_size=query_size, random_state=seed)
        ctx_idx, qry_idx = next(splitter.split(dummy_X, y_chunk))
        return ctx_idx, qry_idx, f"random_fallback:{type(exc).__name__}"


def _row_l2_distances(X_rows, reference_rows) -> Any:
    if reference_rows is None or len(reference_rows) == 0:
        return np.zeros(len(X_rows), dtype=np.float64)

    X_arr = np.asarray(X_rows, dtype=np.float64)
    ref_arr = np.asarray(reference_rows, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if ref_arr.ndim == 1:
        ref_arr = ref_arr.reshape(-1, 1)

    centroid = np.nanmean(ref_arr, axis=0)
    distances = np.linalg.norm(np.nan_to_num(X_arr - centroid, nan=0.0), axis=1)
    return distances.astype(np.float64, copy=False)


def _standardize_for_selection(X_train, X_reference) -> tuple[Any, Any, Any, Any]:
    train_arr = np.asarray(X_train, dtype=np.float64)
    ref_arr = np.asarray(X_reference, dtype=np.float64)
    if train_arr.ndim == 1:
        train_arr = train_arr.reshape(-1, 1)
    if ref_arr.ndim == 1:
        ref_arr = ref_arr.reshape(-1, 1)

    mean = np.nanmean(train_arr, axis=0)
    scale = np.nanstd(train_arr, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (
        np.nan_to_num((train_arr - mean) / scale, nan=0.0),
        np.nan_to_num((ref_arr - mean) / scale, nan=0.0),
        mean,
        scale,
    )


def build_c_selection_context(
    classifier,
    X_train,
    X_reference,
    config: TTTConfig,
) -> CSelectionContext | None:
    if X_reference is None or len(X_reference) == 0:
        return None
    if config.c_source == "none" or config.c_selection in {"random", "stratified_random"}:
        return None

    train_encoded = classifier.X_encoder_.transform(X_train)
    reference_encoded = classifier.X_encoder_.transform(X_reference)
    if config.c_metric == "tabicl_encoded_l2":
        reference = np.asarray(reference_encoded, dtype=np.float64)
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        reference = np.nan_to_num(reference, nan=0.0)
        return CSelectionContext(
            reference=reference,
            metric=config.c_metric,
            target_mean=np.nanmean(reference, axis=0),
            target_var=np.nanvar(reference, axis=0),
        )

    _X_train_std, reference_std, mean, scale = _standardize_for_selection(train_encoded, reference_encoded)
    return CSelectionContext(
        reference=reference_std,
        metric=config.c_metric,
        mean=mean,
        scale=scale,
        target_mean=np.nanmean(reference_std, axis=0),
        target_var=np.nanvar(reference_std, axis=0),
    )


def transform_chunk_for_c_selection(X_chunk, context: CSelectionContext | None) -> Any:
    X_arr = np.asarray(X_chunk, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if context is None:
        return np.nan_to_num(X_arr, nan=0.0)
    if context.metric == "standardized_l2":
        return np.nan_to_num((X_arr - context.mean) / context.scale, nan=0.0)
    return np.nan_to_num(X_arr, nan=0.0)


def _has_query_label_coverage(y_chunk, ctx_idx, qry_idx) -> bool:
    y_arr = np.asarray(y_chunk)
    ctx_labels = set(y_arr[np.asarray(ctx_idx, dtype=int)].astype(int).tolist())
    qry_labels = set(y_arr[np.asarray(qry_idx, dtype=int)].astype(int).tolist())
    return qry_labels.issubset(ctx_labels)


def _class_balanced_order(order, y_chunk, query_size: int) -> list[int]:
    y_arr = np.asarray(y_chunk).astype(int)
    labels, counts = np.unique(y_arr, return_counts=True)
    selected: list[int] = []
    selected_set: set[int] = set()
    per_label_cap = {label: max(0, int(count) - 1) for label, count in zip(labels.astype(int), counts)}
    selected_per_label = {label: 0 for label in per_label_cap}

    for idx in order:
        idx = int(idx)
        if idx in selected_set:
            continue
        label = int(y_arr[idx])
        if selected_per_label.get(label, 0) >= per_label_cap.get(label, 0):
            continue
        selected.append(idx)
        selected_set.add(idx)
        selected_per_label[label] = selected_per_label.get(label, 0) + 1
        if len(selected) >= query_size:
            return selected
    return selected


def _mmd_greedy_order(
    X_chunk_for_selection,
    reference_context: CSelectionContext | None,
    candidate_indices,
    *,
    max_items: int,
    y_chunk=None,
) -> list[int]:
    X_arr = np.asarray(X_chunk_for_selection, dtype=np.float64)
    if reference_context is None or reference_context.target_mean is None:
        return [int(idx) for idx in candidate_indices[:max_items]]

    target_mean = np.asarray(reference_context.target_mean, dtype=np.float64)
    target_var = np.asarray(reference_context.target_var, dtype=np.float64)
    remaining = [int(idx) for idx in candidate_indices]
    selected: list[int] = []
    current_sum = np.zeros(X_arr.shape[1], dtype=np.float64)
    current_sumsq = np.zeros(X_arr.shape[1], dtype=np.float64)
    y_arr = None if y_chunk is None else np.asarray(y_chunk).astype(int)
    selected_per_label: dict[int, int] = {}
    per_label_cap: dict[int, int] = {}
    if y_arr is not None:
        labels, counts = np.unique(y_arr, return_counts=True)
        per_label_cap = {int(label): max(0, int(count) - 1) for label, count in zip(labels, counts)}
        selected_per_label = {label: 0 for label in per_label_cap}

    for _ in range(min(int(max_items), len(remaining))):
        best_pos = None
        best_score = None
        next_n = len(selected) + 1
        for pos, idx in enumerate(remaining):
            if y_arr is not None:
                label = int(y_arr[idx])
                if selected_per_label.get(label, 0) >= per_label_cap.get(label, 0):
                    continue
            row = X_arr[idx]
            cand_mean = (current_sum + row) / next_n
            cand_var = (current_sumsq + row * row) / next_n - cand_mean * cand_mean
            score = float(np.mean((cand_mean - target_mean) ** 2) + 0.25 * np.mean((cand_var - target_var) ** 2))
            if best_score is None or score < best_score:
                best_score = score
                best_pos = pos
        if best_pos is None:
            break
        idx = remaining.pop(best_pos)
        selected.append(idx)
        if y_arr is not None:
            label = int(y_arr[idx])
            selected_per_label[label] = selected_per_label.get(label, 0) + 1
        row = X_arr[idx]
        current_sum += row
        current_sumsq += row * row
    return selected


def _mmd_far_context_scores(X_rows, reference_context: CSelectionContext | None) -> Any:
    ensure_runtime_deps()
    X_arr = np.asarray(X_rows, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if reference_context is None or reference_context.target_mean is None:
        return np.zeros(X_arr.shape[0], dtype=np.float64)

    target_mean = np.asarray(reference_context.target_mean, dtype=np.float64)
    target_var = np.asarray(reference_context.target_var, dtype=np.float64)
    row_var = np.zeros_like(X_arr, dtype=np.float64)
    scores = np.mean((X_arr - target_mean) ** 2, axis=1) + 0.25 * np.mean((row_var - target_var) ** 2, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def _pairwise_squared_distances_chunked(X_left, X_right, *, chunk_size: int = 512) -> Iterator[Any]:
    X_left = np.asarray(X_left, dtype=np.float64)
    X_right = np.asarray(X_right, dtype=np.float64)
    if X_left.ndim == 1:
        X_left = X_left.reshape(-1, 1)
    if X_right.ndim == 1:
        X_right = X_right.reshape(-1, 1)
    right_norm = np.sum(X_right * X_right, axis=1)
    for start in range(0, X_left.shape[0], int(chunk_size)):
        block = X_left[start : start + int(chunk_size)]
        distances = np.sum(block * block, axis=1, keepdims=True) + right_norm[None, :] - 2.0 * block @ X_right.T
        yield np.maximum(distances, 0.0)


def _resolve_crumb_rbf_sigma(X_candidates, X_reference, bandwidth: str, seed: int) -> float:
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    X_reference = np.asarray(X_reference, dtype=np.float64)
    if str(bandwidth) != "median":
        sigma = float(bandwidth)
        return sigma if math.isfinite(sigma) and sigma > 0.0 else 1.0

    if X_candidates.ndim == 1:
        X_candidates = X_candidates.reshape(-1, 1)
    if X_reference.ndim == 1:
        X_reference = X_reference.reshape(-1, 1)
    if len(X_candidates) == 0 or len(X_reference) == 0:
        return 1.0

    max_candidates = min(int(len(X_candidates)), 512)
    max_reference = min(int(len(X_reference)), 512)
    rng = np.random.default_rng(seed)
    cand_idx = rng.choice(len(X_candidates), size=max_candidates, replace=False) if len(X_candidates) > max_candidates else np.arange(len(X_candidates))
    ref_idx = rng.choice(len(X_reference), size=max_reference, replace=False) if len(X_reference) > max_reference else np.arange(len(X_reference))
    cand_sample = X_candidates[np.asarray(cand_idx, dtype=int)]
    ref_sample = X_reference[np.asarray(ref_idx, dtype=int)]
    distances = next(_pairwise_squared_distances_chunked(cand_sample, ref_sample, chunk_size=max(1, len(cand_sample))))
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if positive.size == 0:
        return 1.0
    sigma = math.sqrt(float(np.median(positive)))
    return sigma if math.isfinite(sigma) and sigma > 0.0 else 1.0


def _rbf_mean_affinity(X_candidates, X_reference, sigma: float) -> Any:
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    X_reference = np.asarray(X_reference, dtype=np.float64)
    if X_candidates.ndim == 1:
        X_candidates = X_candidates.reshape(-1, 1)
    if X_reference.ndim == 1:
        X_reference = X_reference.reshape(-1, 1)
    if len(X_candidates) == 0 or len(X_reference) == 0:
        return np.zeros(len(X_candidates), dtype=np.float64)
    denom = max(2.0 * float(sigma) * float(sigma), np.finfo(np.float64).tiny)
    affinities = np.zeros(len(X_candidates), dtype=np.float64)
    row_start = 0
    for distances in _pairwise_squared_distances_chunked(X_candidates, X_reference):
        row_end = row_start + distances.shape[0]
        affinities[row_start:row_end] = np.mean(np.exp(-distances / denom), axis=1)
        row_start = row_end
    return np.nan_to_num(affinities, nan=0.0, posinf=0.0, neginf=0.0)


def _select_crumb_mmd_query_indices(
    y_chunk,
    X_chunk_for_selection,
    *,
    query_size: int,
    seed: int,
    config: TTTConfig,
    selection_context: CSelectionContext | None,
) -> CSelectionResult | None:
    if selection_context is None or selection_context.reference is None or len(selection_context.reference) == 0:
        return None

    X_arr = np.asarray(X_chunk_for_selection, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    reference = np.asarray(selection_context.reference, dtype=np.float64)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)

    n = int(len(y_chunk))
    if n < 2:
        return None
    query_size = max(1, min(int(query_size), n - 1))
    y_arr = np.asarray(y_chunk).astype(int)
    sigma = _resolve_crumb_rbf_sigma(X_arr, reference, config.crumb_bandwidth, seed)
    cross_affinity = _rbf_mean_affinity(X_arr, reference, sigma)

    remaining = list(range(n))
    selected: list[int] = []
    selected_scores: list[float] = []
    selected_per_label: dict[int, int] = {}
    per_label_cap: dict[int, int] = {}
    if config.c_class_balance:
        labels, counts = np.unique(y_arr, return_counts=True)
        per_label_cap = {int(label): max(0, int(count) - 1) for label, count in zip(labels, counts)}
        selected_per_label = {label: 0 for label in per_label_cap}

    denom = max(2.0 * float(sigma) * float(sigma), np.finfo(np.float64).tiny)
    for _ in range(query_size):
        best_pos = None
        best_score = None
        next_n = len(selected) + 1
        selected_arr = X_arr[np.asarray(selected, dtype=int)] if selected else None
        for pos, idx in enumerate(remaining):
            label = int(y_arr[idx])
            if config.c_class_balance and selected_per_label.get(label, 0) >= per_label_cap.get(label, 0):
                continue
            redundancy = 0.0
            if selected_arr is not None and len(selected_arr) > 0:
                distances = next(_pairwise_squared_distances_chunked(X_arr[idx : idx + 1], selected_arr, chunk_size=1))
                redundancy = float(np.sum(np.exp(-distances[0] / denom)))
            score = float(-2.0 * cross_affinity[idx] + (2.0 / float(next_n)) * redundancy)
            if best_score is None or score < best_score:
                best_score = score
                best_pos = pos

        if best_pos is None:
            break
        idx = int(remaining.pop(best_pos))
        selected.append(idx)
        selected_scores.append(float(best_score))
        if config.c_class_balance:
            label = int(y_arr[idx])
            selected_per_label[label] = selected_per_label.get(label, 0) + 1

    if len(selected) < query_size:
        return CSelectionResult(
            ctx_idx=np.array([], dtype=int),
            qry_idx=np.array([], dtype=int),
            split_strategy="crumb_mmd:fallback",
            distance_sum=float(np.sum(selected_scores)) if selected_scores else 0.0,
            distance_sumsq=float(np.sum(np.asarray(selected_scores, dtype=np.float64) ** 2)) if selected_scores else 0.0,
            distance_count=int(len(selected_scores)),
            fallback_reason=f"crumb_mmd produced only {len(selected)}/{query_size} query rows",
            label_coverage_ok=False,
        )

    qry_idx = np.asarray(selected[:query_size], dtype=int)
    qry_set = set(qry_idx.tolist())
    ctx_idx = np.asarray([idx for idx in range(n) if idx not in qry_set], dtype=int)
    coverage_ok = _has_query_label_coverage(y_chunk, ctx_idx, qry_idx)
    selected_scores_arr = np.asarray(selected_scores[:query_size], dtype=np.float64)
    bandwidth_label = config.crumb_bandwidth if str(config.crumb_bandwidth) == "median" else f"{sigma:g}"
    return CSelectionResult(
        ctx_idx=ctx_idx,
        qry_idx=qry_idx,
        split_strategy=f"crumb_mmd:{config.c_source}:{config.c_metric}:rbf:{bandwidth_label}",
        distance_sum=float(np.sum(selected_scores_arr)),
        distance_sumsq=float(np.sum(selected_scores_arr * selected_scores_arr)),
        distance_count=int(len(selected_scores_arr)),
        fallback_reason=None if coverage_ok else "crumb_mmd query labels absent from context",
        label_coverage_ok=coverage_ok,
    )


def build_f_mmd_reserved_context(
    X_train_for_selection,
    selection_context: CSelectionContext | None,
    *,
    reserve_ratio: float = 0.2,
) -> None:
    ensure_runtime_deps()
    if selection_context is None:
        return
    n = int(len(X_train_for_selection))
    if n < 2:
        return

    reserve_count = max(1, int(math.ceil(float(reserve_ratio) * n)))
    reserve_count = min(reserve_count, n - 1)
    scores = _mmd_far_context_scores(X_train_for_selection, selection_context)
    ranked = np.argsort(scores, kind="stable")[::-1]
    reserved = np.asarray(ranked[:reserve_count], dtype=int)
    reserved_scores = scores[reserved]

    selection_context.reserved_context_indices = reserved
    selection_context.reserved_context_scores = scores
    selection_context.reserved_context_ratio = float(reserve_ratio)
    selection_context.reserved_context_score_sum = float(np.sum(reserved_scores))
    selection_context.reserved_context_score_sumsq = float(np.sum(reserved_scores * reserved_scores))
    selection_context.reserved_context_score_count = int(len(reserved_scores))


def _split_ctx_query_from_allowed(
    y_chunk,
    allowed_idx,
    *,
    query_size: int,
    seed: int,
) -> tuple[Any, Any, str]:
    allowed_idx = np.asarray(allowed_idx, dtype=int)
    if len(allowed_idx) < 2:
        raise ValueError("Need at least two non-reserved rows for query selection")
    query_size = max(1, min(int(query_size), len(allowed_idx) - 1))
    local_ctx_idx, local_qry_idx, split_strategy = _split_ctx_query(
        np.asarray(y_chunk)[allowed_idx],
        query_size=query_size,
        seed=seed,
    )
    return allowed_idx[local_ctx_idx], allowed_idx[local_qry_idx], split_strategy


def _select_f_mmd_reserved_context_indices(
    y_chunk,
    *,
    global_indices,
    query_size: int,
    seed: int,
    selection_context: CSelectionContext | None,
) -> CSelectionResult | None:
    if selection_context is None or selection_context.reserved_context_indices is None:
        return None

    global_indices_arr = np.asarray(global_indices, dtype=int)
    reserved_global = set(np.asarray(selection_context.reserved_context_indices, dtype=int).tolist())
    reserved_local = np.asarray(
        [idx for idx, global_idx in enumerate(global_indices_arr) if int(global_idx) in reserved_global],
        dtype=int,
    )
    reserved_local_set = set(reserved_local.tolist())
    allowed_local = np.asarray(
        [idx for idx in range(len(global_indices_arr)) if idx not in reserved_local_set],
        dtype=int,
    )

    if len(allowed_local) < 2:
        return CSelectionResult(
            ctx_idx=np.array([], dtype=int),
            qry_idx=np.array([], dtype=int),
            split_strategy="f_mmd_reserved_context:fallback",
            distance_sum=selection_context.reserved_context_score_sum,
            distance_sumsq=selection_context.reserved_context_score_sumsq,
            distance_count=selection_context.reserved_context_score_count,
            fallback_reason="f_mmd reserved context left fewer than two query candidates",
            label_coverage_ok=False,
        )

    try:
        allowed_ctx_idx, qry_idx, split_strategy = _split_ctx_query_from_allowed(
            y_chunk,
            allowed_local,
            query_size=query_size,
            seed=seed,
        )
    except Exception as exc:
        return CSelectionResult(
            ctx_idx=np.array([], dtype=int),
            qry_idx=np.array([], dtype=int),
            split_strategy="f_mmd_reserved_context:fallback",
            distance_sum=selection_context.reserved_context_score_sum,
            distance_sumsq=selection_context.reserved_context_score_sumsq,
            distance_count=selection_context.reserved_context_score_count,
            fallback_reason=f"f_mmd allowed query split failed: {type(exc).__name__}",
            label_coverage_ok=False,
        )

    ctx_idx = np.unique(np.concatenate([reserved_local, allowed_ctx_idx])).astype(int)
    coverage_ok = _has_query_label_coverage(y_chunk, ctx_idx, qry_idx)
    if not coverage_ok:
        rng = np.random.default_rng(seed)
        safe_order = allowed_local.astype(int).tolist()
        rng.shuffle(safe_order)
        safe_qry = _class_balanced_order(safe_order, y_chunk, query_size)
        if len(safe_qry) >= min(query_size, max(1, len(allowed_local) - 1)):
            safe_qry_idx = np.asarray(safe_qry[: min(query_size, max(1, len(allowed_local) - 1))], dtype=int)
            safe_qry_set = set(safe_qry_idx.tolist())
            safe_ctx_idx = np.asarray([idx for idx in range(len(global_indices_arr)) if idx not in safe_qry_set], dtype=int)
            if _has_query_label_coverage(y_chunk, safe_ctx_idx, safe_qry_idx):
                ctx_idx = safe_ctx_idx
                qry_idx = safe_qry_idx
                split_strategy = f"{split_strategy}:coverage_repaired"
                coverage_ok = True

    return CSelectionResult(
        ctx_idx=ctx_idx,
        qry_idx=np.asarray(qry_idx, dtype=int),
        split_strategy=f"f_mmd_reserved_context:test:{selection_context.metric}:{split_strategy}",
        distance_sum=selection_context.reserved_context_score_sum,
        distance_sumsq=selection_context.reserved_context_score_sumsq,
        distance_count=selection_context.reserved_context_score_count,
        fallback_reason=None if coverage_ok else "f_mmd reserved-context query labels absent from context",
        label_coverage_ok=coverage_ok,
    )


def _select_faware_query_indices(
    y_chunk,
    X_chunk_for_selection,
    *,
    query_size: int,
    seed: int,
    config: TTTConfig,
    selection_context: CSelectionContext | None,
) -> CSelectionResult | None:
    if selection_context is None or selection_context.reference is None or len(selection_context.reference) == 0:
        return None

    n = len(y_chunk)
    query_size = max(1, min(int(query_size), n - 1))
    distances = _row_l2_distances(X_chunk_for_selection, selection_context.reference)
    order = np.argsort(distances, kind="stable").astype(int).tolist()
    rng = np.random.default_rng(seed)

    candidate_limit = min(n, max(query_size, query_size * int(config.c_candidate_multiplier)))
    if config.c_selection == "f_density_mixed":
        nearest_count = int(math.ceil(query_size * 0.5))
        selected_near = order[:nearest_count]
        selected_set = set(selected_near)
        random_pool = [idx for idx in range(n) if idx not in selected_set]
        rng.shuffle(random_pool)
        ranked = selected_near + random_pool
    else:
        ranked = order

    if config.c_class_balance:
        qry_list = _class_balanced_order(ranked, y_chunk, query_size)
    else:
        qry_list = [int(idx) for idx in ranked[:query_size]]

    if len(qry_list) < query_size:
        return CSelectionResult(
            ctx_idx=np.array([], dtype=int),
            qry_idx=np.array([], dtype=int),
            split_strategy=f"{config.c_selection}:fallback",
            fallback_reason=f"{config.c_selection} produced only {len(qry_list)}/{query_size} query rows",
            label_coverage_ok=False,
        )

    qry_idx = np.asarray(qry_list[:query_size], dtype=int)
    qry_set = set(qry_idx.tolist())
    ctx_idx = np.asarray([idx for idx in range(n) if idx not in qry_set], dtype=int)
    coverage_ok = _has_query_label_coverage(y_chunk, ctx_idx, qry_idx)
    selected_distances = distances[qry_idx]
    return CSelectionResult(
        ctx_idx=ctx_idx,
        qry_idx=qry_idx,
        split_strategy=f"{config.c_selection}:{config.c_source}:{config.c_metric}",
        distance_sum=float(np.sum(selected_distances)),
        distance_sumsq=float(np.sum(selected_distances * selected_distances)),
        distance_count=int(len(selected_distances)),
        fallback_reason=None if coverage_ok else f"{config.c_selection} query labels absent from context",
        label_coverage_ok=coverage_ok,
    )


def _select_ctx_query_for_chunk(
    y_chunk,
    X_chunk_for_selection,
    *,
    global_indices,
    query_size: int,
    seed: int,
    config: TTTConfig,
    selection_context: CSelectionContext | None,
) -> CSelectionResult:
    if config.c_selection == "f_mmd":
        selected = _select_f_mmd_reserved_context_indices(
            y_chunk,
            global_indices=global_indices,
            query_size=query_size,
            seed=seed,
            selection_context=selection_context,
        )
        if selected is not None:
            return selected
        return CSelectionResult(
            ctx_idx=np.array([], dtype=int),
            qry_idx=np.array([], dtype=int),
            split_strategy="f_mmd_reserved_context:fallback",
            fallback_reason="missing f_mmd reserved context",
            label_coverage_ok=False,
        )
    elif config.c_selection == "crumb_mmd":
        selected = _select_crumb_mmd_query_indices(
            y_chunk,
            X_chunk_for_selection,
            query_size=query_size,
            seed=seed,
            config=config,
            selection_context=selection_context,
        )
        if selected is not None and selected.label_coverage_ok:
            return selected
        fallback_reason = (
            selected.fallback_reason if selected is not None and selected.fallback_reason else "missing CRUMB-aware selection context"
        )
    elif config.c_selection in {"f_nearest", "f_density_mixed"}:
        selected = _select_faware_query_indices(
            y_chunk,
            X_chunk_for_selection,
            query_size=query_size,
            seed=seed,
            config=config,
            selection_context=selection_context,
        )
        if selected is not None and selected.label_coverage_ok:
            return selected
        fallback_reason = (
            selected.fallback_reason if selected is not None and selected.fallback_reason else "missing F-aware selection context"
        )
    else:
        fallback_reason = None

    ctx_idx, qry_idx, split_strategy = _split_ctx_query(y_chunk, query_size=query_size, seed=seed)
    if not _has_query_label_coverage(y_chunk, ctx_idx, qry_idx):
        distances = np.zeros(len(y_chunk), dtype=np.float64)
        if selection_context is not None and selection_context.reference is not None:
            distances = _row_l2_distances(X_chunk_for_selection, selection_context.reference)
        safe_order = np.argsort(distances, kind="stable").astype(int).tolist()
        safe_qry = _class_balanced_order(safe_order, y_chunk, query_size)
        if len(safe_qry) >= query_size:
            safe_qry_idx = np.asarray(safe_qry[:query_size], dtype=int)
            safe_qry_set = set(safe_qry_idx.tolist())
            safe_ctx_idx = np.asarray([idx for idx in range(len(y_chunk)) if idx not in safe_qry_set], dtype=int)
            if _has_query_label_coverage(y_chunk, safe_ctx_idx, safe_qry_idx):
                ctx_idx = safe_ctx_idx
                qry_idx = safe_qry_idx
                split_strategy = f"{split_strategy}:coverage_repaired"
    return CSelectionResult(
        ctx_idx=np.asarray(ctx_idx, dtype=int),
        qry_idx=np.asarray(qry_idx, dtype=int),
        split_strategy=split_strategy if fallback_reason is None else f"{split_strategy}:fallback_from_{config.c_selection}",
        fallback_reason=fallback_reason,
        label_coverage_ok=_has_query_label_coverage(y_chunk, ctx_idx, qry_idx),
    )


def split_ttt_validation(
    X,
    y,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[Any, Any, Any, Any, str]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    y_array = np.asarray(y)
    if len(indices) < 3:
        return X, y_array, None, None, "none: fewer than three samples"

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
            stratify=y_array,
        )
        strategy = "auto_stratified_validation"
    except Exception as exc:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
        strategy = f"auto_random_validation:{type(exc).__name__}"

    return (
        take_rows(X, np.asarray(train_idx, dtype=int)),
        y_array[np.asarray(train_idx, dtype=int)],
        take_rows(X, np.asarray(val_idx, dtype=int)),
        y_array[np.asarray(val_idx, dtype=int)],
        strategy,
    )


def _torch_dtype(dtype_name: str):
    import torch

    normalized = dtype_name.strip().lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError("--ttt-dtype must be one of: float32, float16, bfloat16")


def _resolve_amp_enabled(amp_value: bool | str, device) -> bool:
    import torch

    device_type = getattr(device, "type", str(device).split(":")[0])
    if amp_value == "auto":
        return device_type == "cuda" and torch.cuda.is_available()
    return bool(amp_value) and device_type == "cuda" and torch.cuda.is_available()


def _make_ttt_amp(config: TTTConfig, device):
    import torch

    use_amp = _resolve_amp_enabled(config.amp, device)
    scaler = torch.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        amp_ctx_factory = lambda: torch.autocast(  # noqa: E731
            device_type="cuda",
            dtype=torch.float16,
        )
    else:
        from contextlib import nullcontext

        amp_ctx_factory = nullcontext
    return use_amp, scaler, amp_ctx_factory


def _set_requires_grad(module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _get_ttt_base_model(classifier):
    model = classifier.model_
    return getattr(model, "module", model)


def _resolve_ttt_data_parallel_device_ids(classifier, config: TTTConfig) -> List[int]:
    if not config.data_parallel or not config.gpu_group:
        return []

    device = getattr(classifier, "device_", None)
    device_type = getattr(device, "type", str(device).split(":")[0])
    if device_type != "cuda":
        return []

    try:
        gpu_ids = parse_gpu_id_list(normalize_gpu_group(config.gpu_group))
    except Exception:
        return []
    if len(gpu_ids) < 2:
        return []

    import torch

    if not torch.cuda.is_available():
        return []

    # Device visibility has already been rewritten per worker, so DataParallel
    # must use process-local logical ids rather than original global ids.
    return list(range(len(gpu_ids)))


def _build_ttt_forward_model(classifier, config: TTTConfig):
    base_model = _get_ttt_base_model(classifier)
    device_ids = _resolve_ttt_data_parallel_device_ids(classifier, config)
    if not device_ids:
        return base_model, 1

    import torch

    try:
        return torch.nn.DataParallel(base_model, device_ids=device_ids), len(device_ids)
    except Exception:
        return base_model, 1


def _configure_ttt_trainable_params(classifier, config: TTTConfig):
    model = _get_ttt_base_model(classifier)
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    if config.freeze_col:
        model.col_embedder.eval()
    else:
        model.col_embedder.train()
        _set_requires_grad(model.col_embedder, True)

    if config.freeze_row:
        model.row_interactor.eval()
    else:
        model.row_interactor.train()
        _set_requires_grad(model.row_interactor, True)

    if config.freeze_icl:
        model.icl_predictor.eval()
    else:
        model.icl_predictor.train()
        _set_requires_grad(model.icl_predictor, True)

    return [param for param in model.parameters() if param.requires_grad]


def _set_ttt_train_mode(classifier) -> None:
    model = _get_ttt_base_model(classifier)
    model.train()
    if not any(param.requires_grad for param in model.col_embedder.parameters()):
        model.col_embedder.eval()
    if not any(param.requires_grad for param in model.row_interactor.parameters()):
        model.row_interactor.eval()
    if not any(param.requires_grad for param in model.icl_predictor.parameters()):
        model.icl_predictor.eval()
    else:
        model.icl_predictor.train()


def _fit_preserving_model_weights(classifier, X, y) -> None:
    original_load_model = getattr(classifier, "_load_model", None)

    def _skip_model_reload():
        return None

    try:
        classifier._load_model = _skip_model_reload
        classifier.fit(X, y)
    finally:
        if original_load_model is not None:
            classifier._load_model = original_load_model


def _optional_metric_to_float(value: Optional[float]) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_is_valid(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        return not bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _ttt_metric_improved(current: Optional[float], best: Optional[float], min_delta: float) -> bool:
    if not _metric_is_valid(current) or not _metric_is_valid(best):
        return False
    return float(current) > float(best) + float(min_delta)


def _evaluate_ttt_validation_metrics(
    classifier,
    X_context,
    y_context,
    X_val,
    y_val,
    config: TTTConfig,
) -> Optional[TTTValidationResult]:
    if X_val is None or y_val is None or len(y_val) == 0:
        return None

    import torch

    base_model = _get_ttt_base_model(classifier)
    original_n_estimators = getattr(classifier, "n_estimators", None)
    try:
        base_model.eval()
        if original_n_estimators is not None:
            classifier.n_estimators = min(int(original_n_estimators), int(config.validation_n_estimators))
        _fit_preserving_model_weights(classifier, X_context, y_context)
        with torch.inference_mode():
            proba = classifier.predict_proba(X_val)
        proba_arr = np.asarray(proba)
        classes = getattr(getattr(classifier, "y_encoder_", None), "classes_", None)
        class_arr = np.asarray(classes) if classes is not None else None
        if class_arr is not None and class_arr.ndim == 1 and len(class_arr) == proba_arr.shape[1]:
            y_pred = class_arr[proba_arr.argmax(axis=1)]
        else:
            y_pred = proba_arr.argmax(axis=1)

        roc_auc = _optional_metric_to_float(compute_tabpfn_roc_auc(y_val, proba_arr, class_arr))
        log_loss_score = _optional_metric_to_float(compute_log_loss(y_val, proba_arr, class_arr))
        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_val)))
        secondary = {
            "roc_auc": roc_auc,
            "log_loss": log_loss_score,
            "accuracy": accuracy,
        }
        if config.eval_metric == "roc_auc":
            primary = roc_auc
        elif config.eval_metric == "log_loss":
            primary = -log_loss_score if _metric_is_valid(log_loss_score) else float("nan")
        elif config.eval_metric == "accuracy":
            primary = accuracy
        else:
            raise ValueError(f"Unsupported TTT eval metric: {config.eval_metric!r}")
        return TTTValidationResult(primary=float(primary), secondary=secondary)
    except (ValueError, RuntimeError):
        return TTTValidationResult(primary=float("nan"), secondary={})
    finally:
        if original_n_estimators is not None:
            classifier.n_estimators = original_n_estimators


def _build_classification_meta_batch(
    classifier,
    X_chunk,
    y_chunk,
    *,
    global_indices,
    config: TTTConfig,
    query_size: int,
    epoch_seed: int,
    chunk_idx: int,
    selection_context: CSelectionContext | None = None,
) -> MetaBatch:
    try:
        from tabicl._sklearn.preprocessing import EnsembleGenerator
    except ImportError:
        from tabicl.sklearn.preprocessing import EnsembleGenerator

    if len(y_chunk) < 2:
        return MetaBatch(None, None, None, 0, "chunk has fewer than two samples")

    split_seed = int(epoch_seed + chunk_idx * 7919)
    n_classes_in_chunk = int(np.max(y_chunk)) + 1
    query_size = max(int(query_size), n_classes_in_chunk)
    X_chunk_for_selection = transform_chunk_for_c_selection(X_chunk, selection_context)
    selection_result = _select_ctx_query_for_chunk(
        y_chunk,
        X_chunk_for_selection,
        global_indices=global_indices,
        query_size=query_size,
        seed=split_seed,
        config=config,
        selection_context=selection_context,
    )
    ctx_idx = selection_result.ctx_idx
    qry_idx = selection_result.qry_idx
    split_strategy = selection_result.split_strategy
    if len(ctx_idx) == 0 or len(qry_idx) == 0:
        return MetaBatch(
            None,
            None,
            None,
            0,
            f"{split_strategy} produced empty context/query split",
            c_distance_sum=selection_result.distance_sum,
            c_distance_sumsq=selection_result.distance_sumsq,
            c_distance_count=selection_result.distance_count,
            c_fallback_reason=selection_result.fallback_reason,
            c_label_coverage_ok=selection_result.label_coverage_ok,
        )

    y_ctx_raw = np.asarray(y_chunk)[ctx_idx].astype(int)
    y_qry_raw = np.asarray(y_chunk)[qry_idx].astype(int)

    missing_query_labels = sorted(set(y_qry_raw.tolist()) - set(y_ctx_raw.tolist()))
    if missing_query_labels:
        return MetaBatch(
            None,
            None,
            None,
            0,
            "query labels absent from context after "
            f"{split_strategy} split: {','.join(str(item) for item in missing_query_labels)}",
            c_distance_sum=selection_result.distance_sum,
            c_distance_sumsq=selection_result.distance_sumsq,
            c_distance_count=selection_result.distance_count,
            c_fallback_reason=selection_result.fallback_reason,
            c_label_coverage_ok=False,
        )

    local_classes = np.asarray(sorted(set(y_ctx_raw.tolist())), dtype=np.int64)
    local_label_map = {int(label): idx for idx, label in enumerate(local_classes.tolist())}
    y_ctx = np.asarray([local_label_map[int(label)] for label in y_ctx_raw], dtype=np.int64)
    y_qry = np.asarray([local_label_map[int(label)] for label in y_qry_raw], dtype=np.int64)
    X_ctx = X_chunk[ctx_idx]
    X_qry = X_chunk[qry_idx]

    gen = EnsembleGenerator(
        classification=True,
        n_estimators=config.n_estimators_finetune,
        norm_methods=getattr(classifier, "norm_methods", None),
        feat_shuffle_method=getattr(classifier, "feat_shuffle_method", "latin"),
        class_shuffle_method=getattr(classifier, "class_shuffle_method", "shift"),
        outlier_threshold=getattr(classifier, "outlier_threshold", 4.0),
        random_state=config.random_state,
    )
    gen.fit(X_ctx, y_ctx)
    variants = gen.transform(X_qry, mode="both")

    X_list = []
    y_train_list = []
    y_query_list = []
    for norm_method, (X_variant, y_variant) in variants.items():
        X_list.append(X_variant)
        y_train_list.append(y_variant)
        shuffle_configs = gen.ensemble_configs_[norm_method]
        if len(shuffle_configs) != X_variant.shape[0]:
            raise RuntimeError(
                f"Ensemble data/class shuffle mismatch for norm_method={norm_method!r}: "
                f"{X_variant.shape[0]} views vs {len(shuffle_configs)} class shuffles"
            )
        for _feat_shuffle, class_shuffle in shuffle_configs:
            if class_shuffle is None:
                y_query_list.append(y_qry)
            else:
                y_query_list.append(np.asarray(class_shuffle, dtype=np.int64)[y_qry.astype(int)])

    import torch

    return MetaBatch(
        X=torch.from_numpy(np.concatenate(X_list, axis=0)).float(),
        y_train=torch.from_numpy(np.concatenate(y_train_list, axis=0)).float(),
        y_query=torch.from_numpy(np.stack(y_query_list, axis=0)).long(),
        train_size=int(len(ctx_idx)),
        skip_reason=None,
        c_distance_sum=selection_result.distance_sum,
        c_distance_sumsq=selection_result.distance_sumsq,
        c_distance_count=selection_result.distance_count,
        c_fallback_reason=selection_result.fallback_reason,
        c_label_coverage_ok=selection_result.label_coverage_ok,
    )


def iter_epoch_meta_batches(
    classifier,
    X_encoded,
    y_encoded,
    *,
    config: TTTConfig,
    epoch_seed: int,
    selection_context: CSelectionContext | None = None,
) -> Iterator[MetaBatch]:
    rng = np.random.default_rng(epoch_seed)
    chunks = _chunk_indices(
        len(y_encoded),
        max_chunk_size=config.max_chunk_size,
        min_chunk_size=config.min_chunk_size,
        rng=rng,
    )
    for chunk_idx, indices in enumerate(chunks):
        X_chunk = X_encoded[indices]
        y_chunk = y_encoded[indices]
        query_size = max(1, int(len(indices) * config.query_ratio))
        yield _build_classification_meta_batch(
            classifier,
            X_chunk,
            y_chunk,
            global_indices=indices,
            config=config,
            query_size=query_size,
            epoch_seed=epoch_seed,
            chunk_idx=chunk_idx,
            selection_context=selection_context,
        )


def move_meta_batch(batch: MetaBatch, device) -> MetaBatch:
    return MetaBatch(
        X=batch.X.to(device, non_blocking=True),
        y_train=batch.y_train.to(device, non_blocking=True),
        y_query=batch.y_query.to(device, non_blocking=True),
        train_size=batch.train_size,
        skip_reason=batch.skip_reason,
        c_distance_sum=batch.c_distance_sum,
        c_distance_sumsq=batch.c_distance_sumsq,
        c_distance_count=batch.c_distance_count,
        c_fallback_reason=batch.c_fallback_reason,
        c_label_coverage_ok=batch.c_label_coverage_ok,
    )


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unknown"


def _build_ttt_ckpt_path(config: TTTConfig, model_name: str, dataset_name: str, step_idx: int) -> Path:
    if not config.ckpt_root:
        raise ValueError("TTT checkpoint root is not configured")
    return (
        Path(config.ckpt_root)
        / _sanitize_path_component(model_name)
        / _sanitize_path_component(dataset_name)
        / f"step_{step_idx}.ckpt"
    )


def _save_ttt_model_ckpt(classifier, config: TTTConfig, model_name: str, dataset_name: str, step_idx: int) -> Path:
    import torch

    model_config = getattr(classifier, "model_config_", None)
    if model_config is None:
        raise RuntimeError("TTT checkpoint save requested before model_config_ is available")

    base_model = _get_ttt_base_model(classifier)
    ckpt_path = _build_ttt_ckpt_path(config, model_name, dataset_name, step_idx)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "config": dict(model_config),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in base_model.state_dict().items()},
        "ttt_metadata": {
            "model_name": str(model_name),
            "dataset_name": str(dataset_name),
            "step": int(step_idx),
        },
    }
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def _should_save_ttt_ckpt_step(config: TTTConfig, step_idx: int) -> bool:
    if not config.save_ckpt:
        return False
    if config.save_ckpt_start_step is None:
        return step_idx % config.save_ckpt_every == 0
    return (
        step_idx >= config.save_ckpt_start_step
        and (step_idx - config.save_ckpt_start_step) % config.save_ckpt_every == 0
    )


def _should_save_ttt_final_ckpt(config: TTTConfig, final_step: int) -> bool:
    if not config.save_ckpt:
        return False
    if config.save_ckpt_start_step is None:
        return True
    return final_step >= config.save_ckpt_start_step


def _derive_model_name(model_path: str | None, checkpoint_version: str | None) -> str:
    candidate = model_path if model_path else checkpoint_version
    if not candidate:
        return "tabicl_model"
    return Path(str(candidate)).stem


def _format_path_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _derive_tabicl_version_label(args: argparse.Namespace) -> str:
    if args.models_dir is not None:
        return f"tabicl-modelpool-{Path(args.models_dir).expanduser().name}"

    source = args.model_path or args.checkpoint_version or DEFAULT_MODEL_PATH
    stem = Path(str(source)).stem
    match = re.search(r"(v\d+(?:\.\d+)?(?:-\d{8})?)", stem)
    if match:
        return f"tabicl-{match.group(1)}"
    return stem or "tabicl_model"


def _derive_dataset_label(data_root: Path, dataset_dirs: List[Path]) -> str:
    if len(dataset_dirs) == 1:
        return dataset_dirs[0].name
    root_name = data_root.name or "datasets"
    return f"{root_name}_{len(dataset_dirs)}datasets"


def build_auto_out_dir(
    args: argparse.Namespace,
    *,
    data_root: Path,
    dataset_dirs: List[Path],
) -> Path:
    version_label = _derive_tabicl_version_label(args)
    dataset_label = _derive_dataset_label(data_root, dataset_dirs)
    model_param_label = "_".join(
        [
            f"inferest{args.n_estimators}",
            f"bs{_format_path_value(args.batch_size)}",
            f"kv{_format_path_value(args.kv_cache)}",
            f"amp{_format_path_value(args.use_amp)}",
            f"fa3{_format_path_value(args.use_fa3)}",
            f"offload{_format_path_value(args.offload_mode)}",
        ]
    )

    if args.ttt_enabled:
        eval_estimator_label = "_".join(
            [
                f"ttt_eval-{args.ttt_eval_metric}",
                f"finetuneest{args.ttt_n_estimators_finetune}",
                f"valest{args.ttt_validation_n_estimators}",
                f"ep{args.ttt_epochs}",
                f"lr{args.ttt_lr}",
                f"q{args.ttt_query_ratio}",
                f"wise{_format_path_value(args.ttt_wise_ft)}",
                f"wisealpha{_format_path_value(args.ttt_wise_alpha)}",
            ]
        )
    else:
        eval_estimator_label = "no_ttt"

    seed_label = f"seed{args.random_state}"
    name_parts = [
        version_label,
        dataset_label,
        model_param_label,
        eval_estimator_label,
        seed_label,
    ]
    auto_name = "__".join(_sanitize_path_component(part) for part in name_parts)
    return DEFAULT_OUT_DIR_ROOT / auto_name


def _build_ttt_lr_scheduler(optimizer, config: TTTConfig, total_steps: int):
    try:
        from tabicl.train._optim import get_scheduler
    except ModuleNotFoundError as exc:
        if exc.name not in {"transformers", "tabicl.train", "tabicl.train._optim"}:
            raise
        return _build_fallback_ttt_lr_scheduler(optimizer, config, total_steps)

    sched_cfg = SimpleNamespace(
        max_steps=max(1, int(total_steps)),
        warmup_proportion=float(config.warmup_proportion),
        warmup_steps=0,
        scheduler=str(config.scheduler),
    )
    return get_scheduler(sched_cfg, optimizer)


def _build_fallback_ttt_lr_scheduler(optimizer, config: TTTConfig, total_steps: int):
    import math
    import torch

    total_steps = max(1, int(total_steps))
    warmup_steps = float(total_steps) * float(config.warmup_proportion)

    if config.scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    if config.scheduler != "cosine_warmup":
        raise ValueError("--ttt-scheduler must be one of: constant, cosine_warmup")

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1.0, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1.0, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _clone_model_state_dict_cpu(model) -> Dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _interpolate_state_dicts(initial_state: Dict[str, Any], tuned_state: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    blended_state: Dict[str, Any] = {}
    alpha = float(alpha)
    for name, tuned_tensor in tuned_state.items():
        initial_tensor = initial_state.get(name)
        if initial_tensor is None:
            blended_state[name] = tuned_tensor.detach().clone()
            continue
        if getattr(tuned_tensor, "is_floating_point", lambda: False)() and initial_tensor.is_floating_point():
            initial_for_tuned = initial_tensor.to(dtype=tuned_tensor.dtype)
            blended_state[name] = initial_for_tuned.mul(1.0 - alpha).add(tuned_tensor, alpha=alpha)
        else:
            blended_state[name] = tuned_tensor.detach().clone()
    return blended_state


def _apply_wise_ft_weights(base_model, initial_state: Dict[str, Any], alpha: float) -> None:
    tuned_state = _clone_model_state_dict_cpu(base_model)
    blended_state = _interpolate_state_dicts(initial_state, tuned_state, alpha)
    base_model.load_state_dict(blended_state)


def run_ttt_epoch_chunk_update(
    classifier,
    X_train,
    y_train,
    X_val,
    y_val,
    config: TTTConfig,
    *,
    model_name: str,
    dataset_name: str,
    X_c_reference=None,
    apply_wise_for_dataset: bool = False,
) -> TTTUpdateResult:
    ensure_runtime_deps()
    wise_ft_for_dataset = bool(config.wise_ft) and bool(apply_wise_for_dataset)
    wise_alpha_for_dataset = float(config.wise_alpha) if bool(config.wise_ft) else None

    if config.scheduler not in {"constant", "cosine_warmup"}:
        raise ValueError("--ttt-scheduler must be one of: constant, cosine_warmup")
    if config.epochs < 1:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=0.0,
            wise_ft=bool(config.wise_ft),
            wise_alpha=wise_alpha_for_dataset,
            reason="--ttt-epochs must be >= 1",
            epochs=0,
            chunks_per_epoch=0,
        )
    if config.micro_batch_size < 1:
        raise ValueError("--ttt-micro-batch-size must be >= 1")

    update_start = time.time()
    classifier.fit(X_train, y_train)
    if classifier.n_classes_ > classifier.model_.max_classes:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=time.time() - update_start,
            wise_ft=bool(config.wise_ft),
            wise_alpha=wise_alpha_for_dataset,
            reason=(
                f"TTT training skipped because n_classes={classifier.n_classes_} "
                f"exceeds model max_classes={classifier.model_.max_classes}"
            ),
            epochs=0,
            chunks_per_epoch=0,
        )

    import torch
    import torch.nn.functional as F

    base_model = _get_ttt_base_model(classifier)
    initial_state = _clone_model_state_dict_cpu(base_model) if wise_ft_for_dataset else None
    trainable_params = _configure_ttt_trainable_params(classifier, config)
    if not trainable_params:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=time.time() - update_start,
            wise_ft=bool(config.wise_ft),
            wise_alpha=wise_alpha_for_dataset,
            reason="No trainable parameters selected for TTT",
            epochs=0,
            chunks_per_epoch=0,
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    forward_model, data_parallel_world_size = _build_ttt_forward_model(classifier, config)
    effective_micro_batch_size = config.micro_batch_size * data_parallel_world_size

    X_encoded = classifier.X_encoder_.transform(X_train)
    y_encoded = classifier.y_encoder_.transform(y_train)
    selection_context = build_c_selection_context(classifier, X_train, X_c_reference, config)
    if config.c_selection == "f_mmd":
        build_f_mmd_reserved_context(
            transform_chunk_for_c_selection(X_encoded, selection_context),
            selection_context,
            reserve_ratio=config.c_reserve_ratio,
        )
    chunks_per_epoch = count_ttt_chunks(
        int(len(y_encoded)),
        max_chunk_size=config.max_chunk_size,
        min_chunk_size=config.min_chunk_size,
    )
    scheduler = _build_ttt_lr_scheduler(
        optimizer,
        config,
        total_steps=max(1, config.epochs * chunks_per_epoch),
    )

    device = classifier.device_
    use_amp, scaler, amp_ctx_factory = _make_ttt_amp(config, device)
    if str(config.dtype).lower() != "float32":
        print(
            f"[ttt-amp] model={model_name} dataset={dataset_name} "
            f"--ttt-dtype={config.dtype} is ignored; TTT AMP follows --use-amp "
            f"and uses float16 autocast on CUDA. use_amp={use_amp}",
            flush=True,
        )

    last_loss = None
    update_steps = 0
    skipped_batches = 0
    skip_reasons: Dict[str, int] = {}
    baseline_metric: Optional[float] = None
    best_metric: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    best_accuracy: Optional[float] = None
    best_epoch = 0
    best_state = None
    patience_counter = 0
    stopped_early = False
    c_distance_sum = 0.0
    c_distance_sumsq = 0.0
    c_distance_count = 0
    c_fallback_reasons: Dict[str, int] = {}
    c_label_coverage_ok = True
    wise_applied = False
    try:
        if X_encoded.shape[0] < 2 or chunks_per_epoch == 0:
            return TTTUpdateResult(
                applied=False,
                loss=None,
                steps=0,
                update_seconds=time.time() - update_start,
                wise_ft=bool(config.wise_ft),
                wise_alpha=wise_alpha_for_dataset,
                reason="Need at least two encoded training samples for chunk TTT",
                epochs=0,
                chunks_per_epoch=chunks_per_epoch,
            )

        if config.early_stopping and X_val is not None and y_val is not None and len(y_val) > 0:
            baseline_result = _evaluate_ttt_validation_metrics(classifier, X_train, y_train, X_val, y_val, config)
            if baseline_result is not None:
                baseline_metric = float(baseline_result.primary)
                best_metric = float(baseline_result.primary)
                baseline_accuracy = baseline_result.secondary.get("accuracy")
                best_accuracy = baseline_accuracy
                best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                print(
                    f"[ttt-val] model={model_name} dataset={dataset_name} "
                    f"baseline_{config.eval_metric}={best_metric:.6f}",
                    flush=True,
                )

        last_saved_step = 0
        for epoch_idx in range(config.epochs):
            _set_ttt_train_mode(classifier)
            epoch_seed = config.random_state + epoch_idx
            epoch_loss_sum = 0.0
            epoch_updates = 0
            for batch in iter_epoch_meta_batches(
                classifier,
                X_encoded,
                y_encoded,
                config=config,
                epoch_seed=epoch_seed,
                selection_context=selection_context,
            ):
                if batch.c_distance_count:
                    c_distance_sum += float(batch.c_distance_sum)
                    c_distance_sumsq += float(batch.c_distance_sumsq)
                    c_distance_count += int(batch.c_distance_count)
                if batch.c_fallback_reason:
                    c_fallback_reasons[batch.c_fallback_reason] = c_fallback_reasons.get(batch.c_fallback_reason, 0) + 1
                c_label_coverage_ok = c_label_coverage_ok and bool(batch.c_label_coverage_ok)
                if batch.skip_reason:
                    skipped_batches += 1
                    skip_reasons[batch.skip_reason] = skip_reasons.get(batch.skip_reason, 0) + 1
                    continue

                batch = move_meta_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                total_views = int(batch.X.shape[0])

                for start_idx in range(0, total_views, effective_micro_batch_size):
                    end_idx = min(start_idx + effective_micro_batch_size, total_views)
                    X_batch = batch.X[start_idx:end_idx]
                    y_train_batch = batch.y_train[start_idx:end_idx]
                    y_query_batch = batch.y_query[start_idx:end_idx]
                    with amp_ctx_factory():
                        logits = forward_model(X_batch, y_train_batch.float())
                        n_classes = int(y_train_batch.max().item()) + 1
                        logits_used = logits[..., :n_classes].reshape(-1, n_classes)
                        loss = F.cross_entropy(logits_used, y_query_batch.long().reshape(-1))
                        scaled_loss = loss * (X_batch.shape[0] / total_views)

                    scaler.scale(scaled_loss).backward()
                    batch_loss += float(scaled_loss.detach().cpu())

                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                update_steps += 1
                epoch_updates += 1
                epoch_loss_sum += batch_loss
                last_loss = batch_loss
                current_lr = scheduler.get_last_lr()[0]

                if update_steps % 3 == 0:
                    print(
                        f"[ttt-loss] model={model_name} dataset={dataset_name} "
                        f"epoch={epoch_idx + 1}/{config.epochs} step={update_steps} "
                        f"loss={batch_loss:.6f} lr={current_lr:.2e}",
                        flush=True,
                    )
                if _should_save_ttt_ckpt_step(config, update_steps):
                    ckpt_path = _save_ttt_model_ckpt(classifier, config, model_name, dataset_name, update_steps)
                    last_saved_step = update_steps
                    print(
                        f"[ttt-ckpt] saved model={model_name} dataset={dataset_name} "
                        f"step={update_steps} path={ckpt_path}",
                        flush=True,
                    )

            if epoch_updates > 0:
                print(
                    f"[ttt-loss] model={model_name} dataset={dataset_name} "
                    f"epoch={epoch_idx + 1}/{config.epochs} "
                    f"mean_loss={epoch_loss_sum / epoch_updates:.6f} "
                    f"updates={epoch_updates} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

            if best_state is not None:
                val_result = _evaluate_ttt_validation_metrics(classifier, X_train, y_train, X_val, y_val, config)
                if val_result is not None:
                    val_metric = float(val_result.primary)
                    improved = _ttt_metric_improved(val_metric, best_metric, config.min_delta)
                    if improved:
                        best_metric = val_metric
                        best_accuracy = val_result.secondary.get("accuracy")
                        best_epoch = epoch_idx + 1
                        patience_counter = 0
                        best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                    elif _metric_is_valid(val_metric):
                        patience_counter += 1
                    print(
                        f"[ttt-val] model={model_name} dataset={dataset_name} "
                        f"epoch={epoch_idx + 1}/{config.epochs} "
                        f"{config.eval_metric}={val_metric:.6f} best={best_metric:.6f} "
                        f"patience={patience_counter}/{config.patience}",
                        flush=True,
                    )
                    if patience_counter >= config.patience:
                        stopped_early = True
                        print(
                            f"[ttt-early-stop] model={model_name} dataset={dataset_name} "
                            f"epoch={epoch_idx + 1} best_epoch={best_epoch} "
                            f"best_{config.eval_metric}={best_metric:.6f}",
                            flush=True,
                        )
                        break

        if update_steps == 0:
            reason = "No valid epoch chunks produced an optimizer update"
            if skip_reasons:
                top_reasons = sorted(skip_reasons.items(), key=lambda item: (-item[1], item[0]))[:3]
                reason += "; skipped=" + "; ".join(f"{count}x {text}" for text, count in top_reasons)
            return TTTUpdateResult(
                applied=False,
                loss=None,
                steps=0,
                update_seconds=time.time() - update_start,
                wise_ft=bool(config.wise_ft),
                wise_alpha=wise_alpha_for_dataset,
                reason=reason,
                epochs=config.epochs,
                chunks_per_epoch=chunks_per_epoch,
                val_eval_metric=config.eval_metric,
                val_baseline_metric=baseline_metric,
                val_best_metric=best_metric,
                val_baseline_accuracy=baseline_accuracy,
                val_best_accuracy=best_accuracy,
                best_epoch=best_epoch,
                stopped_early=stopped_early,
                c_distance_mean=(
                    c_distance_sum / c_distance_count if c_distance_count > 0 else None
                ),
                c_distance_std=(
                    math.sqrt(max(0.0, c_distance_sumsq / c_distance_count - (c_distance_sum / c_distance_count) ** 2))
                    if c_distance_count > 0
                    else None
                ),
                c_fallback_reason=(
                    "; ".join(f"{count}x {text}" for text, count in sorted(c_fallback_reasons.items(), key=lambda item: (-item[1], item[0]))[:3])
                    if c_fallback_reasons
                    else None
                ),
                c_label_coverage_ok=c_label_coverage_ok,
            )

        if best_state is not None:
            base_model.load_state_dict(best_state)

        if wise_ft_for_dataset and initial_state is not None:
            _apply_wise_ft_weights(base_model, initial_state, config.wise_alpha)
            wise_applied = True
            print(
                f"[ttt-wise] model={model_name} dataset={dataset_name} "
                f"alpha={config.wise_alpha:.6f} endpoint="
                f"{'best_epoch' if best_state is not None else 'final_epoch'}",
                flush=True,
            )

        if _should_save_ttt_final_ckpt(config, update_steps) and last_saved_step != update_steps:
            ckpt_path = _save_ttt_model_ckpt(classifier, config, model_name, dataset_name, update_steps)
            print(
                f"[ttt-ckpt] saved model={model_name} dataset={dataset_name} "
                f"step={update_steps} path={ckpt_path}",
                flush=True,
            )
    finally:
        if hasattr(base_model, "clear_cache"):
            base_model.clear_cache()
        classifier.model_kv_cache_ = None
        base_model.eval()

    return TTTUpdateResult(
        applied=True,
        loss=last_loss,
        steps=update_steps,
        update_seconds=time.time() - update_start,
        wise_ft=bool(config.wise_ft),
        wise_alpha=wise_alpha_for_dataset,
        wise_applied=wise_applied,
        reason=None,
        epochs=config.epochs,
        chunks_per_epoch=chunks_per_epoch,
        val_eval_metric=config.eval_metric,
        val_baseline_metric=baseline_metric,
        val_best_metric=best_metric,
        val_baseline_accuracy=baseline_accuracy,
        val_best_accuracy=best_accuracy,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        c_distance_mean=(
            c_distance_sum / c_distance_count if c_distance_count > 0 else None
        ),
        c_distance_std=(
            math.sqrt(max(0.0, c_distance_sumsq / c_distance_count - (c_distance_sum / c_distance_count) ** 2))
            if c_distance_count > 0
            else None
        ),
        c_fallback_reason=(
            "; ".join(f"{count}x {text}" for text, count in sorted(c_fallback_reasons.items(), key=lambda item: (-item[1], item[0]))[:3])
            if c_fallback_reasons
            else None
        ),
        c_label_coverage_ok=c_label_coverage_ok,
    )


def build_model_summary_row(
    model_path: Path,
    gpu_id: int,
    dataset_dirs: List[Path],
    rows: List[ResultRow],
    model_wall_seconds: float,
    error: str | None = None,
) -> ModelSummaryRow:
    ensure_runtime_deps()

    result_df = (
        pd.DataFrame([asdict(row) for row in rows])
        if rows
        else pd.DataFrame(columns=ResultRow.__annotations__.keys())
    )
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    oom_fallback_count = int(truthy_column_mask(ok_df, "ttt_oom_fallback").sum()) if len(ok_df) else 0

    avg_fit_seconds_ok = float(ok_df["fit_seconds"].mean()) if len(ok_df) else None
    avg_predict_seconds_ok = float(ok_df["predict_seconds"].mean()) if len(ok_df) else None
    total_dataset_seconds_ok = (
        float((ok_df["fit_seconds"] + ok_df["predict_seconds"]).sum())
        if len(ok_df)
        else 0.0
    )
    avg_dataset_seconds_ok = (
        float((ok_df["fit_seconds"] + ok_df["predict_seconds"]).mean())
        if len(ok_df)
        else None
    )
    failed_datasets = ",".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else ""

    if error is not None:
        return ModelSummaryRow(
            model_name=model_path.stem,
            model_path=model_path.as_posix(),
            gpu_id=gpu_id,
            datasets_discovered=len(dataset_dirs),
            ok_count=0,
            failed_count=len(dataset_dirs),
            skipped_count=0,
            ttt_oom_fallback_count=0,
            avg_accuracy_ok=None,
            avg_f1_ok=None,
            avg_balanced_accuracy_ok=None,
            avg_roc_auc_ok=None,
            avg_log_loss_ok=None,
            avg_fit_seconds_ok=None,
            avg_predict_seconds_ok=None,
            avg_dataset_seconds_ok=None,
            total_dataset_seconds_ok=0.0,
            model_wall_seconds=float(model_wall_seconds),
            status="fail",
            error=error,
            failed_datasets=",".join(path.name for path in dataset_dirs),
        )

    return ModelSummaryRow(
        model_name=model_path.stem,
        model_path=model_path.as_posix(),
        gpu_id=gpu_id,
        datasets_discovered=len(dataset_dirs),
        ok_count=int(len(ok_df)),
        failed_count=int(len(failed_df)),
        skipped_count=int(len(skipped_df)),
        ttt_oom_fallback_count=oom_fallback_count,
        avg_accuracy_ok=(float(ok_df["accuracy"].mean()) if len(ok_df) else None),
        avg_f1_ok=(float(ok_df["f1"].mean()) if len(ok_df) else None),
        avg_balanced_accuracy_ok=(
            float(ok_df["balanced_accuracy"].mean())
            if len(ok_df) and ok_df["balanced_accuracy"].notna().any()
            else None
        ),
        avg_roc_auc_ok=(
            float(ok_df["roc_auc"].mean())
            if len(ok_df) and ok_df["roc_auc"].notna().any()
            else None
        ),
        avg_log_loss_ok=(
            float(ok_df["log_loss"].mean())
            if len(ok_df) and ok_df["log_loss"].notna().any()
            else None
        ),
        avg_fit_seconds_ok=avg_fit_seconds_ok,
        avg_predict_seconds_ok=avg_predict_seconds_ok,
        avg_dataset_seconds_ok=avg_dataset_seconds_ok,
        total_dataset_seconds_ok=total_dataset_seconds_ok,
        model_wall_seconds=float(model_wall_seconds),
        status="ok" if len(ok_df) else "fail",
        error=None if len(ok_df) else "No successful datasets processed",
        failed_datasets=failed_datasets,
    )


class BackgroundPrefetcher:
    def __init__(self, enabled: bool, verbose: bool) -> None:
        self.enabled = bool(enabled)
        self.verbose = verbose
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._seen: set[str] = set()
        self._thread: threading.Thread | None = None

        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def schedule(self, model_paths: List[Path | str]) -> None:
        if not self.enabled:
            return

        for model_path in model_paths:
            key = str(Path(model_path).resolve())
            if key in self._seen:
                continue
            self._seen.add(key)
            self._queue.put(key)

    def close(self) -> None:
        if not self.enabled:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            model_path = self._queue.get()
            if model_path is None:
                return

            try:
                with open(model_path, "rb") as handle:
                    while handle.read(8 * 1024 * 1024):
                        pass
                if self.verbose:
                    print(f"[prefetch] warmed page cache for {model_path}", flush=True)
            except Exception as exc:
                print(f"[prefetch] warning: failed to warm {model_path}: {exc}", flush=True)


def evaluate_one_dataset(
    classifier,
    dataset_dir: Path,
    ttt_config: TTTConfig | None = None,
    *,
    model_name: str,
) -> ResultRow:
    ensure_runtime_deps()
    if ttt_config is None:
        ttt_config = TTTConfig()

    task_type: Optional[str] = None
    try:
        info = load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else None
        if task_type not in CLASSIFICATION_TASKS:
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
                f1=None,
                balanced_accuracy=None,
                roc_auc=None,
                log_loss=None,
                fit_seconds=0.0,
                predict_seconds=0.0,
                status="skip",
                error=f"Skipped due to task_type={task_type!r}",
            )

        train_split, val_split, test_split = find_split_files(dataset_dir)

        X_train, y_train = load_split(
            train_split[0],
            train_split[1],
            train_split[2],
            context=f"{dataset_dir.name}-train",
        )
        val_count = 0
        X_ttt_train = X_train
        y_ttt_train = np.asarray(y_train)
        X_ttt_val = None
        y_ttt_val = None
        ttt_validation_reason = None
        if val_split is not None:
            X_val, y_val = load_split(
                val_split[0],
                val_split[1],
                val_split[2],
                context=f"{dataset_dir.name}-val",
            )
            val_count = int(len(y_val))
            X_ttt_val = X_val
            y_ttt_val = np.asarray(y_val)
            ttt_validation_reason = "dataset_val_split"
            X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
        elif ttt_config.enabled and ttt_config.early_stopping:
            X_ttt_train, y_ttt_train, X_ttt_val, y_ttt_val, ttt_validation_reason = split_ttt_validation(
                X_train,
                y_train,
                validation_fraction=ttt_config.validation_fraction,
                random_state=ttt_config.random_state,
            )
            val_count = int(len(y_ttt_val)) if y_ttt_val is not None else 0

        X_test, y_test = load_split(
            test_split[0],
            test_split[1],
            test_split[2],
            context=f"{dataset_dir.name}-test",
        )

        classes = pd.unique(pd.Series(np.concatenate([np.asarray(y_train), np.asarray(y_test)], axis=0)))
        dataset_total_rows = int(len(y_train) + len(y_test))
        apply_wise_for_dataset = bool(ttt_config.wise_ft) and dataset_total_rows < 2000

        ttt_loss = None
        ttt_wise_ft = bool(ttt_config.wise_ft) if ttt_config.enabled else False
        ttt_wise_alpha = float(ttt_config.wise_alpha) if ttt_config.enabled and ttt_config.wise_ft else None
        ttt_wise_applied = False
        ttt_steps = 0
        ttt_lr = ttt_config.lr if ttt_config.enabled else None
        ttt_applied = False
        ttt_update_seconds = 0.0
        ttt_split_strategy = None
        ttt_split_reason = None
        ttt_epochs = 0
        ttt_chunks_per_epoch = 0
        ttt_batch_mode = None
        ttt_val_eval_metric = None
        ttt_val_baseline_metric = None
        ttt_val_best_metric = None
        ttt_val_baseline_accuracy = None
        ttt_val_best_accuracy = None
        ttt_best_epoch = 0
        ttt_stopped_early = False
        ttt_oom_fallback = False
        ttt_fallback_reason = None
        ttt_c_selection = ttt_config.c_selection if ttt_config.enabled else None
        ttt_c_source = ttt_config.c_source if ttt_config.enabled else None
        ttt_c_metric = ttt_config.c_metric if ttt_config.enabled else None
        ttt_c_f_distance_mean = None
        ttt_c_f_distance_std = None
        ttt_c_fallback_reason = None
        ttt_c_label_coverage_ok = True
        n_train_b = 0
        n_holdout_c = 0

        t0 = time.time()
        if ttt_config.enabled:
            if should_skip_ttt_for_dataset(dataset_dir, info):
                ttt_split_reason = "TTT skipped for dataset=volkert to avoid OOM"
                classifier.fit(X_train, y_train)
            else:
                ttt_split_strategy = "full_train_epoch_chunks"
                ttt_split_reason = "full train set chunked per epoch"
                if ttt_validation_reason:
                    ttt_split_reason += f" | validation={ttt_validation_reason}"
                if ttt_config.c_selection != "random" or ttt_config.c_source != "none":
                    ttt_split_reason += (
                        f" | c_selection={ttt_config.c_selection}"
                        f" c_source={ttt_config.c_source}"
                        f" c_metric={ttt_config.c_metric}"
                    )
                    if ttt_config.c_selection == "f_mmd":
                        reserve_pct = float(ttt_config.c_reserve_ratio) * 100.0
                        ttt_split_reason += f" | f_mmd_reserved_context={reserve_pct:g}% source=test"
                    elif ttt_config.c_selection == "crumb_mmd":
                        ttt_split_reason += (
                            f" | crumb_mmd_kernel=rbf bandwidth={ttt_config.crumb_bandwidth}"
                            " source=test"
                        )
                if ttt_config.wise_ft:
                    if apply_wise_for_dataset:
                        ttt_split_reason += (
                            f" | wise_ft=alpha{ttt_config.wise_alpha:g}"
                            f" rows={dataset_total_rows}<2000"
                        )
                    else:
                        ttt_split_reason += f" | wise_ft_skipped rows={dataset_total_rows}>=2000"
                n_train_b = int(len(y_ttt_train))
                n_holdout_c = 0
                X_c_reference = None
                if ttt_config.c_source == "validation":
                    X_c_reference = X_ttt_val
                elif ttt_config.c_source == "test":
                    X_c_reference = X_test
                ttt_attempt_start = time.time()
                try:
                    ttt_result = run_ttt_epoch_chunk_update(
                        classifier,
                        X_ttt_train,
                        y_ttt_train,
                        X_ttt_val,
                        y_ttt_val,
                        ttt_config,
                        model_name=model_name,
                        dataset_name=dataset_dir.name,
                        X_c_reference=X_c_reference,
                        apply_wise_for_dataset=apply_wise_for_dataset,
                    )
                except Exception as ttt_exc:
                    if not is_oom_exception(ttt_exc):
                        raise
                    ttt_oom_fallback = True
                    ttt_update_seconds = time.time() - ttt_attempt_start
                    ttt_fallback_reason = (
                        "TTT OOM; used original model parameters for inference: "
                        f"{format_exception_for_csv(ttt_exc)}"
                    )
                    ttt_split_reason = append_ttt_reason(ttt_split_reason, ttt_fallback_reason)
                    force_memory_cleanup(str(getattr(classifier, "device_", "cuda:0")))
                    classifier.fit(X_train, y_train)
                else:
                    ttt_loss = ttt_result.loss
                    ttt_steps = ttt_result.steps
                    ttt_wise_ft = ttt_result.wise_ft
                    ttt_wise_alpha = ttt_result.wise_alpha
                    ttt_wise_applied = ttt_result.wise_applied
                    ttt_applied = ttt_result.applied
                    ttt_update_seconds = float(ttt_result.update_seconds)
                    ttt_epochs = ttt_result.epochs
                    ttt_chunks_per_epoch = ttt_result.chunks_per_epoch
                    ttt_batch_mode = ttt_result.batch_mode
                    ttt_val_eval_metric = ttt_result.val_eval_metric
                    ttt_val_baseline_metric = ttt_result.val_baseline_metric
                    ttt_val_best_metric = ttt_result.val_best_metric
                    ttt_val_baseline_accuracy = ttt_result.val_baseline_accuracy
                    ttt_val_best_accuracy = ttt_result.val_best_accuracy
                    ttt_best_epoch = ttt_result.best_epoch
                    ttt_stopped_early = ttt_result.stopped_early
                    ttt_c_f_distance_mean = ttt_result.c_distance_mean
                    ttt_c_f_distance_std = ttt_result.c_distance_std
                    ttt_c_fallback_reason = ttt_result.c_fallback_reason
                    ttt_c_label_coverage_ok = ttt_result.c_label_coverage_ok
                    if ttt_result.reason:
                        ttt_split_reason = append_ttt_reason(ttt_split_reason, ttt_result.reason)
                    if ttt_c_fallback_reason:
                        ttt_split_reason = append_ttt_reason(
                            ttt_split_reason,
                            f"C-selection fallback: {ttt_c_fallback_reason}",
                        )

                    if ttt_applied:
                        _fit_preserving_model_weights(classifier, X_train, y_train)
                    else:
                        classifier.fit(X_train, y_train)
        else:
            classifier.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        y_proba = classifier.predict_proba(X_test)
        y_pred_encoded = np.argmax(np.asarray(y_proba), axis=1)
        y_pred = classifier.y_encoder_.inverse_transform(y_pred_encoded)
        proba_classes = getattr(getattr(classifier, "y_encoder_", None), "classes_", None)
        roc_auc = compute_tabpfn_roc_auc(y_test, y_proba, proba_classes)
        log_loss_score = compute_log_loss(y_test, y_proba, proba_classes)
        predict_seconds = time.time() - t1

        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
        f1 = compute_weighted_f1(y_test, y_pred)
        balanced_accuracy = compute_balanced_accuracy(y_test, y_pred)

        return ResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            task_type=task_type,
            n_train=int(len(y_train)),
            n_val=val_count,
            n_test=int(len(y_test)),
            n_features=int(X_train.shape[1]),
            n_classes=int(len(classes)),
            accuracy=accuracy,
            f1=f1,
            balanced_accuracy=balanced_accuracy,
            roc_auc=roc_auc,
            log_loss=log_loss_score,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
            n_train_a=int(len(y_train)),
            n_train_b=n_train_b,
            n_holdout_c=n_holdout_c,
            n_test_d=int(len(y_test)),
            ttt_loss=ttt_loss,
            ttt_wise_ft=ttt_wise_ft,
            ttt_wise_alpha=ttt_wise_alpha,
            ttt_wise_applied=ttt_wise_applied,
            ttt_steps=ttt_steps,
            ttt_lr=ttt_lr,
            ttt_applied=ttt_applied,
            ttt_update_seconds=ttt_update_seconds,
            ttt_split_strategy=ttt_split_strategy,
            ttt_split_reason=ttt_split_reason,
            ttt_epochs=ttt_epochs,
            ttt_chunks_per_epoch=ttt_chunks_per_epoch,
            ttt_batch_mode=ttt_batch_mode,
            ttt_val_eval_metric=ttt_val_eval_metric,
            ttt_val_baseline_metric=ttt_val_baseline_metric,
            ttt_val_best_metric=ttt_val_best_metric,
            ttt_val_baseline_accuracy=ttt_val_baseline_accuracy,
            ttt_val_best_accuracy=ttt_val_best_accuracy,
            ttt_best_epoch=ttt_best_epoch,
            ttt_stopped_early=ttt_stopped_early,
            ttt_oom_fallback=ttt_oom_fallback,
            ttt_fallback_reason=ttt_fallback_reason,
            ttt_c_selection=ttt_c_selection,
            ttt_c_source=ttt_c_source,
            ttt_c_metric=ttt_c_metric,
            ttt_c_f_distance_mean=ttt_c_f_distance_mean,
            ttt_c_f_distance_std=ttt_c_f_distance_std,
            ttt_c_fallback_reason=ttt_c_fallback_reason,
            ttt_c_label_coverage_ok=ttt_c_label_coverage_ok,
        )
    except Exception as exc:
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
            f1=None,
            balanced_accuracy=None,
            roc_auc=None,
            log_loss=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    assigned_dataset_dirs: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict,
    ttt_config: TTTConfig,
    verbose: bool,
) -> None:
    def write_result_rows_csv(rows: List[ResultRow]) -> None:
        out_path = Path(worker_out_csv)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
        pd.DataFrame([asdict(row) for row in rows]).to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)

    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_group)
        worker_label = f"worker {worker_id} | gpu {gpu_group}"
        worker_ttt_config = replace(ttt_config, gpu_group=gpu_group)

        import torch
        from tabicl import TabICLClassifier

        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = device_str
        classifier = TabICLClassifier(**worker_kwargs)
        model_name = _derive_model_name(
            str(worker_kwargs.get("model_path")) if worker_kwargs.get("model_path") is not None else None,
            str(worker_kwargs.get("checkpoint_version")) if worker_kwargs.get("checkpoint_version") is not None else None,
        )

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "gpu_group": gpu_group,
                "status": "ready",
                "assigned_count": len(assigned_dataset_dirs),
            }
        )
        start_event.wait()

        rows: List[ResultRow] = []
        for dataset_dir in assigned_dataset_dirs:
            row = evaluate_one_dataset(
                classifier,
                Path(dataset_dir),
                worker_ttt_config,
                model_name=model_name,
            )
            rows.append(row)
            print(
                format_dataset_result_log(
                    worker_label,
                    row,
                ),
                flush=True,
            )
            write_result_rows_csv(rows)

        write_result_rows_csv(rows)
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass

        ensure_runtime_deps()
        crash_row = pd.DataFrame(
            [
                asdict(
                    ResultRow(
                        dataset_name=f"__WORKER_CRASH__{worker_id}",
                        dataset_dir="__worker__",
                        task_type=None,
                        n_train=0,
                        n_val=0,
                        n_test=0,
                        n_features=0,
                        n_classes=0,
                        accuracy=None,
                        f1=None,
                        balanced_accuracy=None,
                        roc_auc=None,
                        log_loss=None,
                        fit_seconds=0.0,
                        predict_seconds=0.0,
                        status="fail",
                        error=traceback.format_exc(),
                    )
                )
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def make_worker_failure_row(
    kind: str,
    worker_id: int | str,
    *,
    assigned_count: int,
    error: str,
) -> ResultRow:
    return ResultRow(
        dataset_name=f"__{kind}__{worker_id}",
        dataset_dir="__worker__",
        task_type=None,
        n_train=0,
        n_val=0,
        n_test=0,
        n_features=0,
        n_classes=0,
        accuracy=None,
        f1=None,
        balanced_accuracy=None,
        roc_auc=None,
        log_loss=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status="fail",
        error=f"{error}; assigned_count={assigned_count}",
    )


def collect_worker_output_frames(
    *,
    worker_csv_paths: List[Path],
    worker_assigned_counts: List[int],
    processes: List[mp.Process],
    dataset_dirs: List[Path],
) -> tuple[List[Any], List[str]]:
    ensure_runtime_deps()

    frames: List[Any] = []
    integrity_errors: List[str] = []
    failure_rows: List[ResultRow] = []

    for worker_id, worker_csv in enumerate(worker_csv_paths):
        assigned_count = worker_assigned_counts[worker_id]
        proc = processes[worker_id]
        worker_frame = None

        if worker_csv.exists():
            try:
                worker_frame = pd.read_csv(worker_csv)
                frames.append(worker_frame)
            except Exception as exc:
                error = f"failed to read {worker_csv}: {type(exc).__name__}: {exc}"
                integrity_errors.append(f"worker_{worker_id}: {error}")
                failure_rows.append(
                    make_worker_failure_row(
                        "WORKER_BAD_CSV",
                        worker_id,
                        assigned_count=assigned_count,
                        error=error,
                    )
                )

        if proc.exitcode not in (0, None):
            error = f"worker exited with exitcode={proc.exitcode}"
            completed_count = 0 if worker_frame is None else len(worker_frame)
            if completed_count < assigned_count:
                integrity_errors.append(
                    f"worker_{worker_id}: {error}; completed_count={completed_count}"
                )
                failure_rows.append(
                    make_worker_failure_row(
                        "WORKER_EXIT",
                        worker_id,
                        assigned_count=assigned_count,
                        error=error,
                    )
                )

        if worker_csv.exists():
            continue

        if assigned_count > 0:
            error = f"missing worker CSV: {worker_csv}"
            integrity_errors.append(f"worker_{worker_id}: {error}")
            failure_rows.append(
                make_worker_failure_row(
                    "WORKER_MISSING_CSV",
                    worker_id,
                    assigned_count=assigned_count,
                    error=error,
                )
            )

    if failure_rows:
        frames.append(pd.DataFrame([asdict(row) for row in failure_rows]))

    if not frames and dataset_dirs:
        error = "no worker results were produced"
        integrity_errors.append(error)
        frames.append(
            pd.DataFrame(
                [
                    asdict(
                        make_worker_failure_row(
                            "EMPTY_RESULT",
                            "all",
                            assigned_count=len(dataset_dirs),
                            error=error,
                        )
                    )
                ]
            )
        )

    return frames, integrity_errors


def model_pool_worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    dataset_dirs: List[str],
    ready_queue,
    task_queue,
    result_queue,
    base_model_kwargs: Dict,
    ttt_config: TTTConfig,
    verbose: bool,
) -> None:
    device_str = "cuda:0"
    worker_label = f"worker {worker_id} | gpu {gpu_group}"

    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_group)
        worker_ttt_config = replace(ttt_config, gpu_group=gpu_group)

        import torch
        from tabicl import TabICLClassifier

        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "gpu_group": gpu_group,
                "status": "ready",
                "assigned_count": len(dataset_dirs),
            }
        )

        resolved_dataset_dirs = [Path(item) for item in dataset_dirs]
        while True:
            task = task_queue.get()
            if task is None:
                return

            model_path = Path(str(task["model_path"]))
            started_at = time.time()
            classifier = None

            try:
                worker_kwargs = dict(base_model_kwargs)
                worker_kwargs["device"] = device_str
                worker_kwargs["model_path"] = normalize_model_path(str(model_path))
                classifier = TabICLClassifier(**worker_kwargs)
                model_name = _derive_model_name(str(model_path), worker_kwargs.get("checkpoint_version"))
                if not worker_ttt_config.enabled:
                    preload_model_once(classifier, worker_label, verbose)

                rows: List[ResultRow] = []
                for dataset_dir in resolved_dataset_dirs:
                    row = evaluate_one_dataset(
                        classifier,
                        dataset_dir,
                        worker_ttt_config,
                        model_name=model_name,
                    )
                    rows.append(row)
                    print(
                        format_dataset_result_log(
                            worker_label,
                            row,
                            model_name=model_path.stem,
                        ),
                        flush=True,
                    )

                summary = build_model_summary_row(
                    model_path=model_path,
                    gpu_id=gpu_id,
                    dataset_dirs=resolved_dataset_dirs,
                    rows=rows,
                    model_wall_seconds=time.time() - started_at,
                )
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "status": "model_done",
                        "summary": asdict(summary),
                    }
                )
            except Exception as exc:
                summary = build_model_summary_row(
                    model_path=model_path,
                    gpu_id=gpu_id,
                    dataset_dirs=resolved_dataset_dirs,
                    rows=[],
                    model_wall_seconds=time.time() - started_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "status": "model_done",
                        "summary": asdict(summary),
                    }
                )
            finally:
                release_classifier_resources(classifier)
                force_memory_cleanup(device_str)
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass

        try:
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "worker_crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass


def write_summary(
    summary_path: Path,
    result_df: pd.DataFrame,
    dataset_dirs: List[Path],
    wall_seconds: float,
) -> None:
    ensure_runtime_deps()

    result_df = result_df.copy()
    for metric_column in ("accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"):
        if metric_column in result_df.columns:
            result_df[metric_column] = pd.to_numeric(result_df[metric_column], errors="coerce")

    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    oom_fallback_df = (
        ok_df[truthy_column_mask(ok_df, "ttt_oom_fallback")].copy()
        if len(ok_df)
        else pd.DataFrame()
    )

    def mean_line(label: str, column: str) -> str:
        if len(ok_df) and column in ok_df.columns and ok_df[column].notna().any():
            return f"{label}: {ok_df[column].mean():.6f}"
        return f"{label}: (none)"

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        f"ttt_oom_fallback_count: {len(oom_fallback_df)}",
        mean_line("avg_accuracy_ok", "accuracy"),
        mean_line("avg_f1_ok", "f1"),
        mean_line("avg_balanced_accuracy_ok", "balanced_accuracy"),
        mean_line("avg_roc_auc_ok", "roc_auc"),
        mean_line("avg_log_loss_ok", "log_loss"),
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        failed_names = ", ".join(failed_df["dataset_name"].astype(str).tolist())
        lines.append(f"failed_datasets: {failed_names}")
    else:
        lines.append("failed_datasets: (none)")

    if len(skipped_df):
        skipped_names = ", ".join(skipped_df["dataset_name"].astype(str).tolist())
        lines.append(f"skipped_datasets: {skipped_names}")
    else:
        lines.append("skipped_datasets: (none)")

    if len(oom_fallback_df):
        oom_fallback_names = ", ".join(oom_fallback_df["dataset_name"].astype(str).tolist())
        lines.append(f"ttt_oom_fallback_datasets: {oom_fallback_names}")
    else:
        lines.append("ttt_oom_fallback_datasets: (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_pool_outputs(
    out_dir: Path,
    model_summaries: List[dict[str, Any]],
    wall_seconds: float,
) -> None:
    ensure_runtime_deps()

    summary_df = (
        pd.DataFrame(model_summaries)
        if model_summaries
        else pd.DataFrame(columns=ModelSummaryRow.__annotations__.keys())
    )
    for column in (
        "avg_accuracy_ok",
        "avg_f1_ok",
        "avg_balanced_accuracy_ok",
        "avg_roc_auc_ok",
        "avg_log_loss_ok",
        "avg_fit_seconds_ok",
        "avg_predict_seconds_ok",
        "avg_dataset_seconds_ok",
        "total_dataset_seconds_ok",
        "model_wall_seconds",
        "ttt_oom_fallback_count",
    ):
        if column in summary_df.columns:
            summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")
    all_csv = out_dir / "all_models_summary.csv"
    summary_txt = out_dir / "summary.txt"
    summary_df.to_csv(all_csv, index=False)

    ok_df = summary_df[summary_df["status"] == "ok"].copy() if len(summary_df) else pd.DataFrame()
    failed_df = summary_df[summary_df["status"] == "fail"].copy() if len(summary_df) else pd.DataFrame()

    def mean_line(label: str, column: str) -> str:
        if len(ok_df) and column in ok_df.columns and ok_df[column].notna().any():
            return f"{label}: {ok_df[column].mean():.6f}"
        return f"{label}: (none)"

    lines = [
        f"total_models: {len(summary_df)}",
        f"successful_models: {len(ok_df)}",
        f"failed_models: {len(failed_df)}",
        mean_line("average_avg_dataset_seconds_ok", "avg_dataset_seconds_ok"),
        mean_line("average_avg_accuracy_ok", "avg_accuracy_ok"),
        mean_line("average_avg_f1_ok", "avg_f1_ok"),
        mean_line("average_avg_balanced_accuracy_ok", "avg_balanced_accuracy_ok"),
        mean_line("average_avg_roc_auc_ok", "avg_roc_auc_ok"),
        mean_line("average_avg_log_loss_ok", "avg_log_loss_ok"),
        f"global_wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        lines.append(
            "failed_models_list: "
            + ", ".join(failed_df["model_name"].astype(str).tolist())
        )
    else:
        lines.append("failed_models_list: (none)")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabICLv2 classification benchmarks on dataset roots with "
            "epoch-shuffled chunk TTT and CRUMB-aware C-selection."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--checkpoint-version", default=DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument(
        "--out-dir",
        default="results/CRUMB_C/Tabiclv1.1_CRUMB_aware_C_standardized_l2_data200",
        help=(
            "Output directory. If omitted, generate one under 1b_result from "
            "TabICL version, dataset label, model parameters, TTT eval metric, "
            "estimator counts, and random seed."
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-groups", default="0;1")
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--batch-size", type=parse_optional_int, default=8)
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--prefetch-models", type=int, default=4)
    parser.add_argument(
        "--ttt-holdout",
        dest="ttt_enabled",
        action="store_true",
        help="Compatibility alias: enable full-train epoch-chunk TTT. No B/C holdout is used.",
    )
    parser.add_argument(
        "--no-ttt",
        dest="ttt_enabled",
        action="store_false",
        help="Disable TTT and run ordinary TabICL inference.",
    )
    parser.set_defaults(ttt_enabled=True)
    parser.add_argument("--ttt-lr", type=float, default=1e-5)
    parser.add_argument("--ttt-scheduler", choices=["constant", "cosine_warmup"], default="cosine_warmup")
    parser.add_argument("--ttt-warmup-proportion", type=float, default=0.1)
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--ttt-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--ttt-micro-batch-size",
        type=int,
        default=1,
        help=(
            "Per-step TTT micro-batch size. When TTT data parallel is active, "
            "this value is interpreted per GPU and the effective batch becomes "
            "micro_batch_size x number_of_gpus_in_gpu_group."
        ),
    )
    parser.add_argument("--ttt-weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--ttt-wise-ft",
        type=parse_bool,
        default=True,
        help=(
            "Apply fixed-alpha WiSE-FT weight interpolation after the FT update path "
            "for datasets with total rows < 2000."
        ),
    )
    parser.add_argument(
        "--ttt-wise-alpha",
        type=float,
        default=0.1,
        help=(
            "WiSE-FT interpolation alpha in [0, 1], where 0 keeps the initial "
            "model weights and 1 keeps the fine-tuned endpoint."
        ),
    )
    parser.add_argument(
        "--ttt-epochs",
        "--ttt-steps",
        dest="ttt_epochs",
        type=int,
        default=30,
        help="Number of epoch-shuffled chunk TTT passes. --ttt-steps is kept as a compatibility alias.",
    )
    parser.add_argument("--ttt-max-chunk-size", type=int, default=10000)
    parser.add_argument("--ttt-min-chunk-size", type=int, default=50)
    parser.add_argument("--ttt-query-ratio", type=float, default=0.2)
    parser.add_argument("--ttt-n-estimators-finetune", type=int, default=2)
    parser.add_argument("--ttt-early-stopping", type=parse_bool, default=True)
    parser.add_argument("--ttt-patience", type=int, default=8)
    parser.add_argument("--ttt-min-delta", type=float, default=1e-4)
    parser.add_argument("--ttt-eval-metric", choices=["roc_auc", "log_loss", "accuracy"], default="accuracy")
    parser.add_argument("--ttt-validation-fraction", type=float, default=0.1)
    parser.add_argument("--ttt-validation-n-estimators", type=int, default=2)
    parser.add_argument("--ttt-freeze-col", type=parse_bool, default=False)
    parser.add_argument("--ttt-freeze-row", type=parse_bool, default=False)
    parser.add_argument("--ttt-freeze-icl", type=parse_bool, default=False)
    parser.add_argument(
        "--ttt-c-selection",
        choices=["random", "stratified_random", "f_nearest", "f_mmd", "f_density_mixed", "crumb_mmd"],
        default="crumb_mmd",
        help=(
            "How to choose the per-chunk query C used for TTT. random preserves "
            "the original 1C_Chunk_TTT.py split behavior; f_nearest/f_density_mixed choose C "
            "by matching an unlabeled validation/test reference distribution; f_mmd reserves "
            "the farthest --ttt-c-reserve-ratio train rows from the test distribution as context-only rows; "
            "crumb_mmd uses CRUMB-style RBF-kernel MMD herding to choose query rows."
        ),
    )
    parser.add_argument(
        "--ttt-c-source",
        choices=["none", "validation", "test"],
        default="test",
        help=(
            "Unlabeled reference split used by f_* C-selection. Use validation for "
            "inductive/proxy experiments and test only for transductive F-aware TTT."
        ),
    )
    parser.add_argument(
        "--ttt-c-metric",
        choices=["tabicl_encoded_l2", "standardized_l2"],
        default="standardized_l2",
        help=(
            "Distance space for F-aware C-selection. Both options start from the "
            "TabICL numerical encoder; standardized_l2 additionally z-scores the encoded features."
        ),
    )
    parser.add_argument("--ttt-c-candidate-multiplier", type=int, default=5)
    parser.add_argument("--ttt-c-class-balance", type=parse_bool, default=True)
    parser.add_argument(
        "--ttt-crumb-bandwidth",
        default="median",
        help="Gaussian RBF bandwidth for --ttt-c-selection=crumb_mmd. Use 'median' or a positive float.",
    )
    parser.add_argument(
        "--ttt-c-reserve-ratio",
        type=float,
        default=0.05,
        help=(
            "Fraction of train rows reserved as test-farthest context-only rows when "
            "--ttt-c-selection=f_mmd. Must be in (0, 1), and reserve_ratio + "
            "--ttt-query-ratio must be < 1."
        ),
    )
    parser.add_argument(
        "--ttt-save-ckpt",
        type=parse_bool,
        default=False,
        help="Whether to save intermediate TabICL checkpoints during the TTT update path.",
    )
    parser.add_argument(
        "--ttt-save-ckpt-every",
        type=int,
        default=30,
        help="Save a TTT checkpoint every N optimizer steps and always save the final step.",
    )
    parser.add_argument(
        "--ttt-save-ckpt-start-step",
        type=parse_optional_int,
        default=None,
        help=(
            "First optimizer step to save a TTT checkpoint. Use None to keep the legacy "
            "multiple-of --ttt-save-ckpt-every schedule."
        ),
    )
    parser.add_argument(
        "--ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_true",
        help="Enable intra-worker multi-GPU data parallelism for the TTT update path.",
    )
    parser.add_argument(
        "--no-ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_false",
        help="Disable intra-worker multi-GPU data parallelism for the TTT update path.",
    )
    parser.set_defaults(ttt_data_parallel=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_gpu_ids(args: argparse.Namespace) -> List[int]:
    gpu_ids = parse_gpu_id_list(args.gpus) if args.gpus else detect_default_gpu_ids()
    if args.workers is None:
        args.workers = len(gpu_ids)
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")
    return gpu_ids


def resolve_gpu_assignments(args: argparse.Namespace) -> tuple[List[int], List[str]]:
    if args.gpu_groups:
        if args.gpus:
            raise ValueError("Use either --gpu-groups or --gpus, not both")
        gpu_groups = parse_gpu_group_list(args.gpu_groups)
        if not gpu_groups:
            raise ValueError("--gpu-groups must contain at least one group")
        if args.workers is None:
            args.workers = len(gpu_groups)
        if len(gpu_groups) != args.workers:
            raise ValueError(f"--gpu-groups must contain exactly {args.workers} groups")
        gpu_ids = [first_gpu_id_from_group(gpu_group) for gpu_group in gpu_groups]
        return gpu_ids, gpu_groups

    gpu_ids = resolve_gpu_ids(args)
    return gpu_ids, [str(gpu_id) for gpu_id in gpu_ids]


def build_common_model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "n_estimators": args.n_estimators,
        "batch_size": args.batch_size,
        "kv_cache": args.kv_cache,
        "allow_auto_download": True,
        "checkpoint_version": args.checkpoint_version,
        "use_amp": args.use_amp,
        "use_fa3": args.use_fa3,
        "offload_mode": args.offload_mode,
        "random_state": args.random_state,
    }


def run_single_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    gpu_groups: List[str],
    out_dir: Path,
) -> None:
    model_kwargs = build_common_model_kwargs(args)
    model_kwargs["model_path"] = normalize_model_path(args.model_path or DEFAULT_MODEL_PATH)
    ttt_config = build_ttt_config(args)

    start_time = time.time()
    ready_queue: mp.Queue = mp.Queue()
    start_event = mp.Event()

    worker_csv_paths: List[Path] = []
    worker_assigned_counts: List[int] = []
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        assigned_dirs = [str(path.resolve()) for path in dataset_dirs[worker_id::args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)
        worker_assigned_counts.append(len(assigned_dirs))

        proc = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                gpu_groups[worker_id],
                assigned_dirs,
                ready_queue,
                start_event,
                str(worker_csv),
                dict(model_kwargs),
                ttt_config,
                args.verbose,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    ready_workers: set[int] = set()
    while len(ready_workers) < args.workers:
        try:
            message = ready_queue.get(timeout=10)
        except Exception:
            dead_workers = [
                str(idx)
                for idx, proc in enumerate(processes)
                if not proc.is_alive() and idx not in ready_workers
            ]
            if dead_workers:
                raise RuntimeError(
                    "Some workers exited before initialization completed: "
                    + ", ".join(dead_workers)
                )
            continue

        if message.get("status") == "ready":
            ready_workers.add(int(message["worker_id"]))
            if args.verbose:
                print(
                    f"[worker {message['worker_id']} | gpu {message.get('gpu_group', message['gpu_id'])}] "
                    f"ready assigned={message.get('assigned_count', '?')}"
                )
            continue

        if message.get("status") == "crash":
            raise RuntimeError(
                f"Worker {message['worker_id']} on gpu {message['gpu_id']} crashed "
                f"during initialization:\n{message.get('error', '(no traceback)')}"
            )

    start_event.set()

    for proc in processes:
        proc.join()

    dfs, integrity_errors = collect_worker_output_frames(
        worker_csv_paths=worker_csv_paths,
        worker_assigned_counts=worker_assigned_counts,
        processes=processes,
        dataset_dirs=dataset_dirs,
    )

    all_df = (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else pd.DataFrame(columns=ResultRow.__annotations__.keys())
    )
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(summary_txt, all_df, dataset_dirs, wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))
    if ttt_config.enabled:
        print("ttt_config:")
        print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))

    system_failure_mask = (
        all_df["dataset_name"].astype(str).str.startswith("__WORKER_")
        | all_df["dataset_name"].astype(str).str.startswith("__EMPTY_RESULT__")
    )
    if integrity_errors or system_failure_mask.any():
        details = "; ".join(integrity_errors) if integrity_errors else "worker failure rows were produced"
        raise RuntimeError(f"TabICL TTT worker output integrity check failed: {details}")


def run_multi_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    gpu_groups: List[str],
    out_dir: Path,
) -> None:
    model_paths = discover_model_paths(Path(args.models_dir), max_models=args.max_models)
    if not model_paths:
        raise FileNotFoundError(f"No checkpoint files found under {args.models_dir}")

    worker_count = min(args.workers, len(model_paths))
    base_model_kwargs = build_common_model_kwargs(args)
    ttt_config = build_ttt_config(args)
    ready_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    task_queues: List[mp.Queue] = []
    processes: List[mp.Process] = []
    prefetcher = BackgroundPrefetcher(enabled=args.prefetch_models > 0, verbose=args.verbose)
    closed_workers: set[int] = set()

    try:
        for worker_id in range(worker_count):
            task_queue: mp.Queue = mp.Queue()
            task_queues.append(task_queue)
            proc = mp.Process(
                target=model_pool_worker_main,
                args=(
                    worker_id,
                    gpu_ids[worker_id],
                    gpu_groups[worker_id],
                    [str(path.resolve()) for path in dataset_dirs],
                    ready_queue,
                    task_queue,
                    result_queue,
                    dict(base_model_kwargs),
                    ttt_config,
                    args.verbose,
                ),
                daemon=False,
            )
            proc.start()
            processes.append(proc)

        ready_workers: set[int] = set()
        while len(ready_workers) < worker_count:
            try:
                message = ready_queue.get(timeout=10)
            except Exception:
                dead_workers = [
                    str(idx)
                    for idx, proc in enumerate(processes)
                    if not proc.is_alive() and idx not in ready_workers
                ]
                if dead_workers:
                    raise RuntimeError(
                        "Some workers exited before initialization completed: "
                        + ", ".join(dead_workers)
                    )
                continue

            if message.get("status") == "ready":
                ready_workers.add(int(message["worker_id"]))
                if args.verbose:
                    print(
                        f"[worker {message['worker_id']} | gpu {message.get('gpu_group', message['gpu_id'])}] "
                        f"model-pool ready datasets={message.get('assigned_count', '?')}",
                        flush=True,
                    )
                continue

            if message.get("status") == "crash":
                raise RuntimeError(
                    f"Worker {message['worker_id']} on gpu {message['gpu_id']} crashed "
                    f"during initialization:\n{message.get('error', '(no traceback)')}"
                )

        next_model_idx = 0
        completed_models = 0
        start_time = time.time()
        collected_summaries: List[dict[str, Any]] = []
        prefetcher.schedule(model_paths[: min(len(model_paths), worker_count + max(0, args.prefetch_models))])

        for worker_id in range(worker_count):
            if next_model_idx >= len(model_paths):
                break
            task_queues[worker_id].put({"model_path": str(model_paths[next_model_idx])})
            next_model_idx += 1
            prefetcher.schedule(model_paths[next_model_idx : next_model_idx + max(0, args.prefetch_models)])

        while completed_models < len(model_paths):
            message = result_queue.get()
            status = message.get("status")

            if status == "worker_crash":
                raise RuntimeError(
                    f"Worker {message.get('worker_id')} on gpu {message.get('gpu_id')} crashed:\n"
                    f"{message.get('error', '(no traceback)')}"
                )

            if status != "model_done":
                continue

            worker_id = int(message["worker_id"])
            summary = dict(message["summary"])
            collected_summaries.append(summary)
            completed_models += 1

            if args.verbose:
                print(
                    f"[worker {worker_id} | gpu {summary['gpu_id']}] "
                    f"finished model={summary['model_name']} status={summary['status']} "
                    f"ok={summary['ok_count']} fail={summary['failed_count']} "
                    f"wall={summary['model_wall_seconds']:.2f}s",
                    flush=True,
                )

            if next_model_idx < len(model_paths):
                task_queues[worker_id].put({"model_path": str(model_paths[next_model_idx])})
                next_model_idx += 1
                prefetcher.schedule(model_paths[next_model_idx : next_model_idx + max(0, args.prefetch_models)])
            elif worker_id not in closed_workers:
                task_queues[worker_id].put(None)
                closed_workers.add(worker_id)

        for worker_id in range(worker_count):
            if worker_id not in closed_workers:
                task_queues[worker_id].put(None)

        for proc in processes:
            proc.join()

        wall_seconds = time.time() - start_time
        write_model_pool_outputs(out_dir, collected_summaries, wall_seconds)

        print(f"saved_all_models_csv: {out_dir / 'all_models_summary.csv'}")
        print(f"saved_summary: {out_dir / 'summary.txt'}")
        print("base_model_kwargs:")
        print(json.dumps(base_model_kwargs, indent=2, ensure_ascii=False))
        if ttt_config.enabled:
            print("ttt_config:")
            print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))
    finally:
        for worker_id in range(len(task_queues)):
            if worker_id not in closed_workers:
                try:
                    task_queues[worker_id].put(None)
                except Exception:
                    pass
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        prefetcher.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_runtime_deps()

    if args.models_dir is not None and args.model_path is not None:
        raise ValueError("--models-dir and --model-path are mutually exclusive")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    dataset_dirs = find_dataset_dirs(data_root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir is not None
        else build_auto_out_dir(args, data_root=data_root, dataset_dirs=dataset_dirs)
    )
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")

    gpu_ids, gpu_groups = resolve_gpu_assignments(args)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.models_dir is not None:
        run_multi_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)
    else:
        run_single_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)


if __name__ == "__main__":
    main()
