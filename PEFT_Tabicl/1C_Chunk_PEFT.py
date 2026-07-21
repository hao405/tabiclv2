#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_ROOT = Path("results/dataset_views/openml_cc18_max10")
DEFAULT_MODEL_PATH = "tabicl-classifier-v2-20260212.ckpt"
DEFAULT_CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
DEFAULT_TABPFN_MODEL_PATH = Path(
    "baseline_compare/TabPFN-main/tabpfn-v2-classifier-v2_default.ckpt"
)
DEFAULT_TABPFN_V3_BINARY_MODEL_PATH = Path(
    "baseline_compare/TabPFN-main/tabpfn-v3-classifier-v3_20260417_binary.ckpt"
)
DEFAULT_TABPFN_V3_MULTICLASS_MODEL_PATH = Path(
    "baseline_compare/TabPFN-main/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
)
DEFAULT_OUT_DIR_ROOT = Path("results/PEFT")
TABPFN_RUNNER_PATH = REPO_ROOT / "baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py"
PEFT_METHODS = ("lora", "last_layers", "ln_head_embedding")
MODEL_FAMILIES = ("tabiclv2", "tabpfnv2", "tabpfnv3")
MATRIX_MODEL_FAMILIES = ("tabiclv2", "tabpfnv3")
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
    peft_method: Optional[str] = None
    peft_targets: Optional[str] = None
    peft_trainable_params: Optional[int] = None
    peft_trainable_ratio: Optional[float] = None
    peft_rank: Optional[int] = None
    model_family: str = "tabiclv2"


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
    peft_method: str = "lora"
    peft_targets: str = "col,row,icl"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    last_n_icl_blocks: int = 1


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
    peft_method: Optional[str] = None
    peft_targets: Optional[str] = None
    peft_trainable_params: Optional[int] = None
    peft_trainable_ratio: Optional[float] = None
    peft_rank: Optional[int] = None


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
            f"ft_applied={row.ttt_applied} "
            f"ft_update={row.ttt_update_seconds:.3f}s "
            f"ft_loss={format_optional_float(row.ttt_loss)} "
            f"ft_steps={row.ttt_steps} "
            f"ft_epochs={row.ttt_epochs} "
            f"ft_mode={row.ttt_batch_mode} "
            f"ft_val_metric={row.ttt_val_eval_metric} "
            f"ft_val_best={format_optional_float(row.ttt_val_best_metric)} "
            f"ft_best_epoch={row.ttt_best_epoch} "
            f"ft_stopped_early={row.ttt_stopped_early} "
            f"ft_oom_fallback={row.ttt_oom_fallback}"
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
    peft_method = str(args.ttt_peft_method).strip().lower()
    if peft_method not in {"lora", "last_layers", "ln_head_embedding"}:
        raise ValueError("--ttt-peft-method must be one of: lora, last_layers, ln_head_embedding")
    peft_targets = _normalize_peft_targets(args.ttt_peft_targets)
    if int(args.ttt_lora_rank) < 1:
        raise ValueError("--ttt-lora-rank must be >= 1")
    if float(args.ttt_lora_alpha) <= 0:
        raise ValueError("--ttt-lora-alpha must be > 0")
    if not 0.0 <= float(args.ttt_lora_dropout) < 1.0:
        raise ValueError("--ttt-lora-dropout must be in [0, 1)")
    if int(args.ttt_last_n_icl_blocks) < 1:
        raise ValueError("--ttt-last-n-icl-blocks must be >= 1")
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
        ckpt_root=str((Path(args.out_dir).expanduser() / "ft_ckpts").resolve()),
        peft_method=peft_method,
        peft_targets=",".join(sorted(peft_targets, key=("col", "row", "icl").index)),
        lora_rank=int(args.ttt_lora_rank),
        lora_alpha=float(args.ttt_lora_alpha),
        lora_dropout=float(args.ttt_lora_dropout),
        last_n_icl_blocks=int(args.ttt_last_n_icl_blocks),
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


def _get_lora_attention_class():
    import torch

    return torch.nn.MultiheadAttention


def _normalize_peft_targets(value: str | None) -> Set[str]:
    raw_targets = [part.strip().lower() for part in str(value or "").split(",") if part.strip()]
    if not raw_targets:
        raise ValueError("--ttt-peft-targets must include at least one of: col,row,icl")
    allowed = {"col", "row", "icl"}
    invalid = sorted(set(raw_targets) - allowed)
    if invalid:
        raise ValueError(
            "--ttt-peft-targets contains unsupported target(s): "
            + ",".join(invalid)
            + ". Supported targets: col,row,icl"
        )
    return set(raw_targets)


def _target_modules_for_peft(model, targets: Set[str]) -> List[Any]:
    modules: List[Any] = []
    if "col" in targets and hasattr(model, "col_embedder"):
        modules.append(model.col_embedder)
    if "row" in targets and hasattr(model, "row_interactor"):
        modules.append(model.row_interactor)
    if "icl" in targets and hasattr(model, "icl_predictor"):
        modules.append(model.icl_predictor)
    return modules


def _make_lora_weight_parametrization(weight, *, rank: int, alpha: float, dropout: float):
    import torch

    class LoRAWeightParametrization(torch.nn.Module):
        def __init__(self):
            super().__init__()
            out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
            self.rank = int(rank)
            self.alpha = float(alpha)
            self.scaling = float(alpha) / float(rank)
            self.dropout = float(dropout)
            self._ttt_lora_A = torch.nn.Parameter(
                torch.empty(self.rank, in_features, device=weight.device, dtype=weight.dtype)
            )
            self._ttt_lora_B = torch.nn.Parameter(
                torch.zeros(out_features, self.rank, device=weight.device, dtype=weight.dtype)
            )
            torch.nn.init.kaiming_uniform_(self._ttt_lora_A, a=5**0.5)

        def forward(self, base_weight):
            lora_a = self._ttt_lora_A
            if self.dropout > 0:
                lora_a = torch.nn.functional.dropout(lora_a, p=self.dropout, training=self.training)
            delta = self._ttt_lora_B @ lora_a
            return base_weight + delta.to(dtype=base_weight.dtype) * self.scaling

    return LoRAWeightParametrization()


def _install_lora_on_parameter(module, param_name: str, config: TTTConfig) -> bool:
    import torch
    from torch.nn.utils import parametrize

    if not hasattr(module, param_name):
        return False
    weight = getattr(module, param_name)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return False
    if parametrize.is_parametrized(module, param_name):
        return False

    lora = _make_lora_weight_parametrization(
        weight,
        rank=int(config.lora_rank),
        alpha=float(config.lora_alpha),
        dropout=float(config.lora_dropout),
    )
    parametrize.register_parametrization(module, param_name, lora)
    original = getattr(module.parametrizations, param_name).original
    original.requires_grad = False
    setattr(module, f"_ttt_lora_{param_name}_installed", True)
    if param_name == "in_proj_weight":
        module._ttt_lora_in_proj_installed = True
    return True


def _install_lora(module, config: TTTConfig) -> int:
    import torch

    installed = 0
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            installed += int(_install_lora_on_parameter(child, "weight", config))
        if hasattr(child, "in_proj_weight"):
            installed += int(_install_lora_on_parameter(child, "in_proj_weight", config))
    return installed


def _is_lora_param_name(name: str) -> bool:
    return "._ttt_lora_" in name or name.endswith("._ttt_lora_A") or name.endswith("._ttt_lora_B")


def _set_lora_trainable_only(model) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = _is_lora_param_name(name)


def _select_last_layer_params(model, config: TTTConfig, targets: Set[str]) -> None:
    if "icl" not in targets or not hasattr(model, "icl_predictor"):
        return
    icl = model.icl_predictor
    blocks = getattr(getattr(icl, "tf_icl", None), "blocks", None)
    if blocks is not None:
        n_blocks = len(blocks)
        start = max(0, n_blocks - int(config.last_n_icl_blocks))
        for block in list(blocks)[start:]:
            _set_requires_grad(block, True)
    if hasattr(icl, "ln"):
        _set_requires_grad(icl.ln, True)
    if hasattr(icl, "decoder"):
        _set_requires_grad(icl.decoder, True)


def _set_normalization_trainable(module) -> None:
    import torch

    normalization_types = [torch.nn.LayerNorm]
    rms_norm = getattr(torch.nn, "RMSNorm", None)
    if rms_norm is not None:
        normalization_types.append(rms_norm)
    normalization_types_tuple = tuple(normalization_types)
    for child in module.modules():
        if isinstance(child, normalization_types_tuple) or child.__class__.__name__.endswith(
            "RMSNorm"
        ):
            _set_requires_grad(child, True)


def _set_layer_norm_trainable(module) -> None:
    """Compatibility alias that now also covers TabPFNv3 RMSNorm modules."""
    _set_normalization_trainable(module)


def _set_named_child_trainable(module, *names: str) -> None:
    for name in names:
        if hasattr(module, name):
            child = getattr(module, name)
            if hasattr(child, "parameters"):
                _set_requires_grad(child, True)


def _select_ln_head_embedding_params(model, targets: Set[str]) -> None:
    for module in _target_modules_for_peft(model, targets):
        _set_layer_norm_trainable(module)
    if "col" in targets and hasattr(model, "col_embedder"):
        _set_named_child_trainable(model.col_embedder, "in_linear", "y_encoder")
    if "icl" in targets and hasattr(model, "icl_predictor"):
        _set_named_child_trainable(model.icl_predictor, "decoder", "y_encoder")


def _peft_trainable_summary(model) -> tuple[int, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    ratio = float(trainable) / float(total) if total else 0.0
    return int(trainable), ratio


def _merged_ttt_state_dict(model) -> Dict[str, Any]:
    from torch.nn.utils import parametrize

    merged: Dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        if ".parametrizations." in name or "_ttt_lora_" in name:
            continue
        merged[name] = tensor.detach().cpu()

    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for param_name in ("weight", "in_proj_weight"):
            if hasattr(module, param_name) and parametrize.is_parametrized(module, param_name):
                merged[prefix + param_name] = getattr(module, param_name).detach().cpu()
    return merged


def _ttt_checkpoint_state_dict(model, config: TTTConfig) -> Dict[str, Any]:
    if str(config.peft_method).lower() == "lora":
        return _merged_ttt_state_dict(model)
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def _configure_ttt_trainable_params(classifier, config: TTTConfig):
    model = _get_ttt_base_model(classifier)
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    targets = _normalize_peft_targets(config.peft_targets)
    method = str(config.peft_method).lower()
    if method == "lora":
        installed = 0
        for module in _target_modules_for_peft(model, targets):
            installed += _install_lora(module, config)
        if installed == 0:
            raise RuntimeError("LoRA PEFT did not find any target Linear or attention projection layers")
        _set_lora_trainable_only(model)
    elif method == "last_layers":
        _select_last_layer_params(model, config, targets)
    elif method == "ln_head_embedding":
        _select_ln_head_embedding_params(model, targets)
    else:
        raise ValueError("--ttt-peft-method must be one of: lora, last_layers, ln_head_embedding")

    return [param for param in model.parameters() if param.requires_grad]


def _configure_tabpfn_trainable_params(model, config: TTTConfig):
    """Apply the shared PEFT policies to a loaded TabPFNv2 architecture."""
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    method = str(config.peft_method).lower()
    if method == "lora":
        installed = _install_lora(model, config)
        if installed == 0:
            raise RuntimeError(
                "LoRA PEFT did not find any TabPFN Linear or attention projection layers"
            )
        _set_lora_trainable_only(model)
    elif method == "last_layers":
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise RuntimeError("TabPFN last_layers PEFT requires model.blocks")
        start = max(0, len(blocks) - int(config.last_n_icl_blocks))
        for block in list(blocks)[start:]:
            _set_requires_grad(block, True)
        _set_named_child_trainable(model, "output_projection")
    elif method == "ln_head_embedding":
        _set_layer_norm_trainable(model)
        _set_named_child_trainable(
            model,
            "feature_group_embedder",
            "target_embedder",
            "feature_positional_embedding_embeddings",
            "output_projection",
        )
    else:
        raise ValueError(
            "--ttt-peft-method must be one of: lora, last_layers, ln_head_embedding"
        )

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError(f"TabPFN PEFT method {method!r} selected no trainable parameters")
    return trainable


def _tabpfnv3_output_head_names(model) -> tuple[str, ...]:
    if hasattr(model, "many_class_decoder"):
        return ("many_class_decoder",)
    if hasattr(model, "output_projection"):
        return ("output_projection",)
    raise RuntimeError(
        "TabPFNv3 PEFT requires many_class_decoder or output_projection"
    )


def _configure_tabpfnv3_trainable_params(model, config: TTTConfig):
    """Apply PEFT policies to a loaded TabPFNv3 binary or multiclass model."""
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    method = str(config.peft_method).lower()
    if method == "lora":
        installed = _install_lora(model, config)
        if installed == 0:
            raise RuntimeError(
                "LoRA PEFT did not find any TabPFNv3 Linear or attention projection layers"
            )
        _set_lora_trainable_only(model)
    elif method == "last_layers":
        blocks = getattr(model, "icl_blocks", None)
        if blocks is None:
            raise RuntimeError("TabPFNv3 last_layers PEFT requires model.icl_blocks")
        start = max(0, len(blocks) - int(config.last_n_icl_blocks))
        for block in list(blocks)[start:]:
            _set_requires_grad(block, True)
        _set_named_child_trainable(model, "output_norm", *_tabpfnv3_output_head_names(model))
    elif method == "ln_head_embedding":
        _set_normalization_trainable(model)
        _set_named_child_trainable(
            model,
            "x_embed",
            "col_y_encoder",
            "icl_y_encoder",
            *_tabpfnv3_output_head_names(model),
        )
    else:
        raise ValueError(
            "--ttt-peft-method must be one of: lora, last_layers, ln_head_embedding"
        )

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError(f"TabPFNv3 PEFT method {method!r} selected no trainable parameters")
    return trainable


def _set_ttt_train_mode(classifier) -> None:
    model = _get_ttt_base_model(classifier)
    model.train()
    # PEFT freezes most backbone parameters but still needs TabICL's training
    # forward path. Eval-mode row/ICL forwards use inference-only tensor updates
    # that cannot participate in gradient-based adaptation.
    model.col_embedder.train()
    model.row_interactor.train()
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
            raise ValueError(f"Unsupported FT eval metric: {config.eval_metric!r}")
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
    config: TTTConfig,
    query_size: int,
    epoch_seed: int,
    chunk_idx: int,
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
    ctx_idx, qry_idx, split_strategy = _split_ctx_query(y_chunk, query_size=query_size, seed=split_seed)
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
    )


def iter_epoch_meta_batches(
    classifier,
    X_encoded,
    y_encoded,
    *,
    config: TTTConfig,
    epoch_seed: int,
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
            config=config,
            query_size=query_size,
            epoch_seed=epoch_seed,
            chunk_idx=chunk_idx,
        )


def move_meta_batch(batch: MetaBatch, device) -> MetaBatch:
    return MetaBatch(
        X=batch.X.to(device, non_blocking=True),
        y_train=batch.y_train.to(device, non_blocking=True),
        y_query=batch.y_query.to(device, non_blocking=True),
        train_size=batch.train_size,
        skip_reason=batch.skip_reason,
    )


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unknown"


def _build_ttt_ckpt_path(config: TTTConfig, model_name: str, dataset_name: str, step_idx: int) -> Path:
    if not config.ckpt_root:
        raise ValueError("FT checkpoint root is not configured")
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
        raise RuntimeError("FT checkpoint save requested before model_config_ is available")

    base_model = _get_ttt_base_model(classifier)
    ckpt_path = _build_ttt_ckpt_path(config, model_name, dataset_name, step_idx)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "config": dict(model_config),
        "state_dict": _ttt_checkpoint_state_dict(base_model, config),
        "ttt_metadata": {
            "model_name": str(model_name),
            "dataset_name": str(dataset_name),
            "step": int(step_idx),
            "peft_method": str(config.peft_method),
            "peft_targets": str(config.peft_targets),
            "peft_trainable_params": int(getattr(config, "_peft_trainable_params", 0)),
            "peft_trainable_ratio": float(getattr(config, "_peft_trainable_ratio", 0.0)),
            "peft_rank": int(config.lora_rank) if str(config.peft_method).lower() == "lora" else None,
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
        peft_label_parts = [
            f"paperpeft-{args.ttt_peft_method}",
            f"targets-{args.ttt_peft_targets}",
        ]
        if args.ttt_peft_method == "lora":
            peft_label_parts.extend(
                [
                    f"r{args.ttt_lora_rank}",
                    f"a{args.ttt_lora_alpha}",
                ]
            )
        elif args.ttt_peft_method == "last_layers":
            peft_label_parts.append(f"last{args.ttt_last_n_icl_blocks}")
        peft_label = "_".join(peft_label_parts)
        eval_estimator_label = "_".join(
            [
                peft_label,
                f"ft_eval-{args.ttt_eval_metric}",
                f"finetuneest{args.ttt_n_estimators_finetune}",
                f"valest{args.ttt_validation_n_estimators}",
                f"ep{args.ttt_epochs}",
                f"lr{args.ttt_lr}",
                f"q{args.ttt_query_ratio}",
            ]
        )
    else:
        eval_estimator_label = "no_ft"

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
) -> TTTUpdateResult:
    ensure_runtime_deps()

    if config.scheduler not in {"constant", "cosine_warmup"}:
        raise ValueError("--ttt-scheduler must be one of: constant, cosine_warmup")
    if config.epochs < 1:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=0.0,
            reason="--ttt-epochs must be >= 1",
            epochs=0,
            chunks_per_epoch=0,
            peft_method=str(config.peft_method),
            peft_targets=str(config.peft_targets),
            peft_trainable_params=0,
            peft_trainable_ratio=0.0,
            peft_rank=int(config.lora_rank) if str(config.peft_method).lower() == "lora" else None,
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
            reason=(
                f"FT training skipped because n_classes={classifier.n_classes_} "
                f"exceeds model max_classes={classifier.model_.max_classes}"
            ),
            epochs=0,
            chunks_per_epoch=0,
            peft_method=str(config.peft_method),
            peft_targets=str(config.peft_targets),
            peft_trainable_params=0,
            peft_trainable_ratio=0.0,
            peft_rank=int(config.lora_rank) if str(config.peft_method).lower() == "lora" else None,
        )

    import torch
    import torch.nn.functional as F

    base_model = _get_ttt_base_model(classifier)
    trainable_params = _configure_ttt_trainable_params(classifier, config)
    peft_trainable_params, peft_trainable_ratio = _peft_trainable_summary(base_model)
    config._peft_trainable_params = peft_trainable_params
    config._peft_trainable_ratio = peft_trainable_ratio
    if not trainable_params:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=time.time() - update_start,
            reason="No trainable parameters selected for FT",
            epochs=0,
            chunks_per_epoch=0,
            peft_method=str(config.peft_method),
            peft_targets=str(config.peft_targets),
            peft_trainable_params=peft_trainable_params,
            peft_trainable_ratio=peft_trainable_ratio,
            peft_rank=int(config.lora_rank) if str(config.peft_method).lower() == "lora" else None,
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    forward_model, data_parallel_world_size = _build_ttt_forward_model(classifier, config)
    effective_micro_batch_size = config.micro_batch_size * data_parallel_world_size

    X_encoded = classifier.X_encoder_.transform(X_train)
    y_encoded = classifier.y_encoder_.transform(y_train)
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
            f"[ft-amp] model={model_name} dataset={dataset_name} "
            f"--ttt-dtype={config.dtype} is ignored; FT AMP follows --use-amp "
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
    try:
        if X_encoded.shape[0] < 2 or chunks_per_epoch == 0:
            return TTTUpdateResult(
                applied=False,
                loss=None,
                steps=0,
                update_seconds=time.time() - update_start,
                reason="Need at least two encoded training samples for chunk FT",
                epochs=0,
                chunks_per_epoch=chunks_per_epoch,
                peft_method=str(config.peft_method),
                peft_targets=str(config.peft_targets),
                peft_trainable_params=peft_trainable_params,
                peft_trainable_ratio=peft_trainable_ratio,
                peft_rank=int(config.lora_rank) if str(config.peft_method).lower() == "lora" else None,
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
                    f"[ft-val] model={model_name} dataset={dataset_name} "
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
            ):
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
                        f"[ft-loss] model={model_name} dataset={dataset_name} "
                        f"epoch={epoch_idx + 1}/{config.epochs} step={update_steps} "
                        f"loss={batch_loss:.6f} lr={current_lr:.2e}",
                        flush=True,
                    )
                if _should_save_ttt_ckpt_step(config, update_steps):
                    ckpt_path = _save_ttt_model_ckpt(classifier, config, model_name, dataset_name, update_steps)
                    last_saved_step = update_steps
                    print(
                        f"[ft-ckpt] saved model={model_name} dataset={dataset_name} "
                        f"step={update_steps} path={ckpt_path}",
                        flush=True,
                    )

            if epoch_updates > 0:
                print(
                    f"[ft-loss] model={model_name} dataset={dataset_name} "
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
                        f"[ft-val] model={model_name} dataset={dataset_name} "
                        f"epoch={epoch_idx + 1}/{config.epochs} "
                        f"{config.eval_metric}={val_metric:.6f} best={best_metric:.6f} "
                        f"patience={patience_counter}/{config.patience}",
                        flush=True,
                    )
                    if patience_counter >= config.patience:
                        stopped_early = True
                        print(
                            f"[ft-early-stop] model={model_name} dataset={dataset_name} "
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
                peft_method=str(config.peft_method),
                peft_targets=str(config.peft_targets),
                peft_trainable_params=peft_trainable_params,
                peft_trainable_ratio=peft_trainable_ratio,
                peft_rank=int(config.lora_rank) if str(config.peft_method).lower() == "lora" else None,
            )

        if best_state is not None:
            base_model.load_state_dict(best_state)

        if _should_save_ttt_final_ckpt(config, update_steps) and last_saved_step != update_steps:
            ckpt_path = _save_ttt_model_ckpt(classifier, config, model_name, dataset_name, update_steps)
            print(
                f"[ft-ckpt] saved model={model_name} dataset={dataset_name} "
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
        peft_method=str(config.peft_method),
        peft_targets=str(config.peft_targets),
        peft_trainable_params=peft_trainable_params,
        peft_trainable_ratio=peft_trainable_ratio,
        peft_rank=int(config.lora_rank) if str(config.peft_method).lower() == "lora" else None,
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
                peft_method=str(ttt_config.peft_method) if ttt_config.enabled else None,
                peft_targets=str(ttt_config.peft_targets) if ttt_config.enabled else None,
                peft_rank=int(ttt_config.lora_rank)
                if ttt_config.enabled and str(ttt_config.peft_method).lower() == "lora"
                else None,
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

        ttt_loss = None
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
        peft_method = str(ttt_config.peft_method) if ttt_config.enabled else None
        peft_targets = str(ttt_config.peft_targets) if ttt_config.enabled else None
        peft_trainable_params = None
        peft_trainable_ratio = None
        peft_rank = (
            int(ttt_config.lora_rank)
            if ttt_config.enabled and str(ttt_config.peft_method).lower() == "lora"
            else None
        )
        n_train_b = 0
        n_holdout_c = 0

        t0 = time.time()
        if ttt_config.enabled:
            if should_skip_ttt_for_dataset(dataset_dir, info):
                ttt_split_reason = "FT skipped for dataset=volkert to avoid OOM"
                classifier.fit(X_train, y_train)
            else:
                ttt_split_strategy = "full_train_epoch_chunks"
                ttt_split_reason = "full train set chunked per epoch"
                if ttt_validation_reason:
                    ttt_split_reason += f" | validation={ttt_validation_reason}"
                n_train_b = int(len(y_ttt_train))
                n_holdout_c = 0
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
                    )
                except Exception as ttt_exc:
                    if not is_oom_exception(ttt_exc):
                        raise
                    ttt_oom_fallback = True
                    ttt_update_seconds = time.time() - ttt_attempt_start
                    ttt_fallback_reason = (
                        "FT OOM; used original model parameters for inference: "
                        f"{format_exception_for_csv(ttt_exc)}"
                    )
                    ttt_split_reason = append_ttt_reason(ttt_split_reason, ttt_fallback_reason)
                    force_memory_cleanup(str(getattr(classifier, "device_", "cuda:0")))
                    classifier.fit(X_train, y_train)
                else:
                    ttt_loss = ttt_result.loss
                    ttt_steps = ttt_result.steps
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
                    peft_method = ttt_result.peft_method
                    peft_targets = ttt_result.peft_targets
                    peft_trainable_params = ttt_result.peft_trainable_params
                    peft_trainable_ratio = ttt_result.peft_trainable_ratio
                    peft_rank = ttt_result.peft_rank
                    if ttt_result.reason:
                        ttt_split_reason = append_ttt_reason(ttt_split_reason, ttt_result.reason)

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
            peft_method=peft_method,
            peft_targets=peft_targets,
            peft_trainable_params=peft_trainable_params,
            peft_trainable_ratio=peft_trainable_ratio,
            peft_rank=peft_rank,
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
            peft_method=str(ttt_config.peft_method) if ttt_config.enabled else None,
            peft_targets=str(ttt_config.peft_targets) if ttt_config.enabled else None,
            peft_rank=int(ttt_config.lora_rank)
            if ttt_config.enabled and str(ttt_config.peft_method).lower() == "lora"
            else None,
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
        f"ft_oom_fallback_count: {len(oom_fallback_df)}",
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
        lines.append(f"ft_oom_fallback_datasets: {oom_fallback_names}")
    else:
        lines.append("ft_oom_fallback_datasets: (none)")

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


def _load_tabpfn_runner_module():
    if not TABPFN_RUNNER_PATH.exists():
        raise FileNotFoundError(f"TabPFN runner does not exist: {TABPFN_RUNNER_PATH}")
    result_naming_source = REPO_ROOT / "baseline_compare/result_naming.py"
    if not result_naming_source.exists() and "result_naming" not in sys.modules:
        # This checkout may intentionally omit the optional auto-naming helper.
        # The PEFT backend never calls TabPFN's run_benchmark/auto-naming path.
        sys.modules["result_naming"] = type(sys)("result_naming")
    module_name = f"_peft_tabpfn_runner_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, TABPFN_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import TabPFN runner from {TABPFN_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class TabICLBackend:
    model_family: str = "tabiclv2"


class TabPFNBackend:
    model_family: str
    model_version: str
    peft_targets: str

    def configure_runner_args(
        self,
        args: argparse.Namespace,
        tabpfn_args: argparse.Namespace,
    ) -> None:
        raise NotImplementedError

    def configure_trainable_params(self, model, config: TTTConfig):
        raise NotImplementedError


class TabPFNv2Backend(TabPFNBackend):
    model_family = "tabpfnv2"
    model_version = "v2"
    peft_targets = "tabpfnv2_all"

    def configure_runner_args(
        self,
        args: argparse.Namespace,
        tabpfn_args: argparse.Namespace,
    ) -> None:
        tabpfn_args.model_version = self.model_version
        tabpfn_args.model_path = str(Path(args.tabpfn_model_path).expanduser().resolve())
        tabpfn_args.v3_binary_model_path = "auto"
        tabpfn_args.v3_multiclass_model_path = "auto"

    def configure_trainable_params(self, model, config: TTTConfig):
        return _configure_tabpfn_trainable_params(model, config)


class TabPFNv3Backend(TabPFNBackend):
    model_family = "tabpfnv3"
    model_version = "v3"
    peft_targets = "tabpfnv3_all"

    def configure_runner_args(
        self,
        args: argparse.Namespace,
        tabpfn_args: argparse.Namespace,
    ) -> None:
        tabpfn_args.model_version = self.model_version
        tabpfn_args.model_path = None
        tabpfn_args.v3_binary_model_path = str(
            Path(args.tabpfn_v3_binary_model_path).expanduser().resolve()
        )
        tabpfn_args.v3_multiclass_model_path = str(
            Path(args.tabpfn_v3_multiclass_model_path).expanduser().resolve()
        )

    def configure_trainable_params(self, model, config: TTTConfig):
        return _configure_tabpfnv3_trainable_params(model, config)


TABPFN_BACKENDS: Dict[str, TabPFNBackend] = {
    backend.model_family: backend
    for backend in (TabPFNv2Backend(), TabPFNv3Backend())
}


def _get_tabpfn_backend(model_family: str) -> TabPFNBackend:
    try:
        return TABPFN_BACKENDS[model_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported TabPFN model family: {model_family}") from exc


def _tabpfn_args_from_common(
    args: argparse.Namespace,
    module,
    backend: Optional[TabPFNBackend] = None,
) -> argparse.Namespace:
    backend = backend or _get_tabpfn_backend(args.model_family)
    tabpfn_args = module.build_arg_parser().parse_args([])
    tabpfn_args.data_root = str(Path(args.data_root).resolve())
    tabpfn_args.out_dir = str(Path(args.out_dir).resolve())
    tabpfn_args.workers = 1
    tabpfn_args.gpus = "0"
    tabpfn_args.max_datasets = args.max_datasets
    tabpfn_args.random_state = int(args.random_state)
    tabpfn_args.n_estimators = int(args.n_estimators)
    tabpfn_args.verbose = bool(args.verbose)
    backend.configure_runner_args(args, tabpfn_args)
    tabpfn_args.many_class = "off"
    tabpfn_args.ttt = bool(args.ttt_enabled)
    tabpfn_args.ttt_epochs = int(args.ttt_epochs)
    tabpfn_args.ttt_max_chunk_size = int(args.ttt_max_chunk_size)
    tabpfn_args.ttt_grad_accumulation_steps = 1
    tabpfn_args.ttt_query_ratio = float(args.ttt_query_ratio)
    tabpfn_args.ttt_lr = float(args.ttt_lr)
    tabpfn_args.ttt_weight_decay = float(args.ttt_weight_decay)
    tabpfn_args.ttt_grad_clip = float(args.ttt_grad_clip)
    tabpfn_args.ttt_patience = int(args.ttt_patience)
    tabpfn_args.ttt_min_delta = float(args.ttt_min_delta)
    tabpfn_args.ttt_eval_metric = {
        "accuracy": "acc",
        "roc_auc": "roc_auc",
        "log_loss": "log_loss",
    }[str(args.ttt_eval_metric)]
    tabpfn_args.ttt_validation_fraction = float(args.ttt_validation_fraction)
    tabpfn_args.ttt_n_estimators_finetune = int(args.ttt_n_estimators_finetune)
    tabpfn_args.ttt_validation_n_estimators = int(args.ttt_validation_n_estimators)
    tabpfn_args.ttt_lr_scheduler = str(args.ttt_scheduler) != "constant"
    tabpfn_args.ttt_lr_warmup_only = False
    # Reentrant activation checkpointing requires at least one grad-requiring
    # input. PEFT freezes the upstream backbone, so keeping it enabled can make
    # checkpointed trainable blocks receive no gradients.
    tabpfn_args.ttt_activation_checkpointing = False
    tabpfn_args.ignore_pretraining_limits = True
    tabpfn_args.retry_failed_datasets_only = False
    tabpfn_args.merge_results_from_csv = None
    return tabpfn_args


def tabpfn_peft_worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    assigned_dataset_dirs: List[str],
    worker_out_csv: str,
    args_dict: Dict[str, Any],
) -> None:
    rows: List[Dict[str, Any]] = []

    def write_rows() -> None:
        ensure_runtime_deps()
        out_path = Path(worker_out_csv)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
        pd.DataFrame(rows).to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)

    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_group)
        if "," in normalize_gpu_group(gpu_group):
            raise ValueError("TabPFN PEFT supports one physical GPU per worker")

        args = argparse.Namespace(**args_dict)
        backend = _get_tabpfn_backend(args.model_family)
        module = _load_tabpfn_runner_module()
        tabpfn_args = _tabpfn_args_from_common(args, module, backend)
        peft_config = build_ttt_config(args)
        peft_telemetry: Dict[str, Any] = {}

        class PEFTTabPFNAdapter(module.TabPFNAdapter):
            def _make_finetuned_classifier(self, loaded):
                classifier = super()._make_finetuned_classifier(loaded)

                def configure_model_for_optimization(instance, model) -> None:
                    del instance
                    backend.configure_trainable_params(model, peft_config)
                    trainable, ratio = _peft_trainable_summary(model)
                    peft_telemetry.update(
                        {
                            "peft_trainable_params": int(trainable),
                            "peft_trainable_ratio": float(ratio),
                        }
                    )

                classifier._configure_model_for_optimization = MethodType(
                    configure_model_for_optimization,
                    classifier,
                )
                return classifier

        adapter = PEFTTabPFNAdapter(tabpfn_args, device=device_str)
        for dataset_dir in assigned_dataset_dirs:
            peft_telemetry.clear()
            row = module.evaluate_one_dataset(adapter, Path(dataset_dir))
            record = asdict(row)
            record.update(
                {
                    "model_family": backend.model_family,
                    "peft_method": str(peft_config.peft_method),
                    "peft_targets": backend.peft_targets,
                    "peft_trainable_params": peft_telemetry.get("peft_trainable_params"),
                    "peft_trainable_ratio": peft_telemetry.get("peft_trainable_ratio"),
                    "peft_rank": (
                        int(peft_config.lora_rank)
                        if str(peft_config.peft_method) == "lora"
                        else None
                    ),
                }
            )
            rows.append(record)
            print(
                f"[worker {worker_id} | gpu {gpu_group}] [{record['status']}] "
                f"{record['dataset_name']} accuracy={format_optional_float(record.get('accuracy'))} "
                f"peft={peft_config.peft_method} trainable={record.get('peft_trainable_params')}",
                flush=True,
            )
            write_rows()
        write_rows()
    except Exception:
        rows.append(
            {
                **asdict(
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
                        model_family=str(args_dict.get("model_family", "tabpfn")),
                    )
                ),
                "peft_method": args_dict.get("ttt_peft_method"),
            }
        )
        write_rows()


def run_tabpfn_single_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    gpu_groups: List[str],
    out_dir: Path,
) -> None:
    started = time.time()
    worker_csv_paths: List[Path] = []
    worker_assigned_counts: List[int] = []
    processes: List[mp.Process] = []
    args_dict = vars(args).copy()

    for worker_id in range(args.workers):
        assigned = [str(path.resolve()) for path in dataset_dirs[worker_id :: args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)
        worker_assigned_counts.append(len(assigned))
        proc = mp.Process(
            target=tabpfn_peft_worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                gpu_groups[worker_id],
                assigned,
                str(worker_csv),
                args_dict,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    frames, integrity_errors = collect_worker_output_frames(
        worker_csv_paths=worker_csv_paths,
        worker_assigned_counts=worker_assigned_counts,
        processes=processes,
        dataset_dirs=dataset_dirs,
    )
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    all_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, all_df, dataset_dirs, time.time() - started)
    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")

    system_failure = all_df["dataset_name"].astype(str).str.startswith("__WORKER_")
    if integrity_errors or system_failure.any():
        details = "; ".join(integrity_errors) if integrity_errors else "worker failure row produced"
        raise RuntimeError(f"TabPFN PEFT worker output integrity check failed: {details}")


def expand_matrix_trials(model_family: str, peft_method: str) -> List[tuple[str, str]]:
    models = list(MATRIX_MODEL_FAMILIES) if model_family == "all" else [model_family]
    methods = list(PEFT_METHODS) if peft_method == "all" else [peft_method]
    return [(model, method) for model in models for method in methods]


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _trial_is_complete(trial_dir: Path) -> bool:
    required = (
        trial_dir / "all_classification_results.csv",
        trial_dir / "summary.txt",
        trial_dir / "run_config.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        config = json.loads((trial_dir / "run_config.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return config.get("status") == "success"


def _resolve_manifest_output_dir(manifest_path: Path, value: Any) -> Path:
    output_dir = Path(str(value)).expanduser()
    if output_dir.is_absolute():
        return output_dir
    repo_relative = REPO_ROOT / output_dir
    if repo_relative.exists():
        return repo_relative
    return manifest_path.parent / output_dir


def _load_reused_matrix_trials(
    source_manifest_value: Optional[str],
    trials: List[tuple[str, str]],
) -> Dict[tuple[str, str], Dict[str, Any]]:
    if not source_manifest_value:
        return {}

    source_manifest = Path(source_manifest_value).expanduser()
    if not source_manifest.is_absolute():
        source_manifest = REPO_ROOT / source_manifest
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Reuse matrix manifest does not exist: {source_manifest}")
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_index = {
        (item.get("model_family"), item.get("peft_method")): item
        for item in source_payload.get("trials", [])
    }

    reused: Dict[tuple[str, str], Dict[str, Any]] = {}
    for key in trials:
        if key[0] != "tabiclv2":
            continue
        source_item = source_index.get(key)
        if source_item is None:
            raise ValueError(
                f"Reuse manifest has no completed trial for {key[0]}/{key[1]}"
            )
        if source_item.get("status") != "success":
            raise ValueError(
                f"Reuse trial {key[0]}/{key[1]} is not success: "
                f"{source_item.get('status')!r}"
            )
        source_output_dir = _resolve_manifest_output_dir(
            source_manifest,
            source_item.get("output_dir"),
        )
        if not _trial_is_complete(source_output_dir):
            raise ValueError(
                f"Reuse trial {key[0]}/{key[1]} has incomplete artifacts: "
                f"{source_output_dir}"
            )
        reused[key] = {
            "model_family": key[0],
            "peft_method": key[1],
            "status": "success",
            "execution": "reused",
            "output_dir": str(source_output_dir),
            "source_manifest": str(source_manifest),
            "source_output_dir": str(source_output_dir),
            **_summarize_trial_output(source_output_dir),
        }
    return reused


def _append_log(log_paths: List[Path], text_value: str) -> None:
    for path in log_paths:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text_value)


def _run_logged_command(command: List[str], log_paths: List[Path]) -> int:
    command_line = "command: " + " ".join(command) + "\n"
    print(command_line, end="", flush=True)
    _append_log(log_paths, command_line)
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        _append_log(log_paths, line)
    return int(proc.wait())


def build_matrix_trial_command(
    args: argparse.Namespace,
    *,
    model_family: str,
    peft_method: str,
    trial_dir: Path,
) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-family",
        model_family,
        "--ttt-peft-method",
        peft_method,
        "--data-root",
        str(args.data_root),
        "--out-dir",
        str(trial_dir),
        "--workers",
        str(args.workers),
        "--gpu-groups",
        str(args.gpu_groups),
        "--n-estimators",
        str(args.n_estimators),
        "--random-state",
        str(args.random_state),
        "--ttt-lr",
        str(args.ttt_lr),
        "--ttt-weight-decay",
        str(args.ttt_weight_decay),
        "--ttt-epochs",
        str(args.ttt_epochs),
        "--ttt-max-chunk-size",
        str(args.ttt_max_chunk_size),
        "--ttt-min-chunk-size",
        str(args.ttt_min_chunk_size),
        "--ttt-query-ratio",
        str(args.ttt_query_ratio),
        "--ttt-n-estimators-finetune",
        str(args.ttt_n_estimators_finetune),
        "--ttt-validation-n-estimators",
        str(args.ttt_validation_n_estimators),
        "--ttt-validation-fraction",
        str(args.ttt_validation_fraction),
        "--ttt-eval-metric",
        str(args.ttt_eval_metric),
        "--ttt-patience",
        str(args.ttt_patience),
        "--ttt-min-delta",
        str(args.ttt_min_delta),
        "--ttt-peft-targets",
        str(args.ttt_peft_targets),
        "--ttt-lora-rank",
        str(args.ttt_lora_rank),
        "--ttt-lora-alpha",
        str(args.ttt_lora_alpha),
        "--ttt-lora-dropout",
        str(args.ttt_lora_dropout),
        "--ttt-last-n-icl-blocks",
        str(args.ttt_last_n_icl_blocks),
        "--tabicl-model-path",
        str(args.tabicl_model_path),
        "--tabpfn-model-path",
        str(args.tabpfn_model_path),
        "--tabpfn-v3-binary-model-path",
        str(args.tabpfn_v3_binary_model_path),
        "--tabpfn-v3-multiclass-model-path",
        str(args.tabpfn_v3_multiclass_model_path),
        "--checkpoint-version",
        str(args.checkpoint_version),
        "--ttt-early-stopping",
        str(bool(args.ttt_early_stopping)),
        "--ttt-save-ckpt",
        str(bool(args.ttt_save_ckpt)),
    ]
    if args.max_datasets is not None:
        command.extend(["--max-datasets", str(args.max_datasets)])
    if args.verbose:
        command.append("--verbose")
    return command


def _summarize_trial_output(trial_dir: Path) -> Dict[str, Any]:
    ensure_runtime_deps()
    csv_path = trial_dir / "all_classification_results.csv"
    if not csv_path.exists():
        return {
            "ok_count": 0,
            "peft_applied_count": 0,
            "nonzero_trainable_count": 0,
            "mean_accuracy": None,
        }
    frame = pd.read_csv(csv_path)
    ok = frame[frame["status"].astype(str) == "ok"] if "status" in frame else frame.iloc[0:0]
    applied = ok[truthy_column_mask(ok, "ttt_applied")] if len(ok) else ok
    accuracy = pd.to_numeric(applied.get("accuracy"), errors="coerce") if len(applied) else None
    trainable = (
        pd.to_numeric(applied.get("peft_trainable_params"), errors="coerce")
        if len(applied) and "peft_trainable_params" in applied
        else None
    )
    return {
        "row_count": int(len(frame)),
        "ok_count": int(len(ok)),
        "peft_applied_count": int(len(applied)),
        "nonzero_trainable_count": (
            int((trainable > 0).sum()) if trainable is not None else 0
        ),
        "skipped_count": int((frame.get("status", pd.Series(dtype=str)).astype(str) == "skip").sum()),
        "ttt_oom_fallback_count": int(
            truthy_column_mask(frame, "ttt_oom_fallback").sum()
        ),
        "mean_accuracy": (
            float(accuracy.mean()) if accuracy is not None and accuracy.notna().any() else None
        ),
    }


def _write_matrix_common_results(
    matrix_rows: List[Dict[str, Any]],
    matrix_dir: Path,
) -> int:
    ensure_runtime_deps()
    frames: List[Any] = []
    dataset_sets: List[Set[str]] = []
    for item in matrix_rows:
        if item.get("status") != "success":
            continue
        csv_path = Path(str(item["output_dir"])) / "all_classification_results.csv"
        if not csv_path.is_file():
            continue
        frame = pd.read_csv(csv_path)
        if "status" not in frame or "dataset_name" not in frame:
            continue
        frame = frame[
            (frame["status"].astype(str) == "ok")
            & truthy_column_mask(frame, "ttt_applied")
        ].copy()
        frame["model_family"] = item["model_family"]
        frame["peft_method"] = item["peft_method"]
        frames.append(frame)
        dataset_sets.append(set(frame["dataset_name"].astype(str)))

    common = set.intersection(*dataset_sets) if dataset_sets else set()
    common_frames = [
        frame[frame["dataset_name"].astype(str).isin(common)] for frame in frames
    ]
    common_frame = pd.concat(common_frames, ignore_index=True) if common_frames else pd.DataFrame()
    common_frame.to_csv(matrix_dir / "matrix_common_results.csv", index=False)
    return len(common)


def run_matrix_mode(args: argparse.Namespace) -> None:
    if args.model_path is not None or args.models_dir is not None:
        raise ValueError(
            "Matrix mode requires --tabicl-model-path/--tabpfn-model-path; "
            "legacy --model-path/--models-dir are ambiguous"
        )
    if not args.ttt_enabled:
        raise ValueError("Matrix PEFT mode requires TTT/finetuning to be enabled")

    trials = expand_matrix_trials(args.model_family, args.ttt_peft_method)
    reused_trials = _load_reused_matrix_trials(args.reuse_matrix_manifest, trials)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser()
    matrix_dir = output_root / "_matrix" / run_name
    manifest_path = matrix_dir / "matrix_manifest.json"
    if matrix_dir.exists() and not args.resume:
        raise FileExistsError(f"Matrix run already exists: {matrix_dir}; use --resume")
    matrix_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = matrix_dir / "launcher.log"

    manifest: Dict[str, Any]
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "run_name": run_name,
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "reuse_matrix_manifest": args.reuse_matrix_manifest,
            "trials": [],
        }
    indexed = {
        (item.get("model_family"), item.get("peft_method")): item
        for item in manifest.get("trials", [])
    }

    matrix_rows: List[Dict[str, Any]] = []
    any_failed = False
    for model_family, peft_method in trials:
        key = (model_family, peft_method)
        if key in reused_trials:
            reused_item = dict(reused_trials[key])
            previous_item = indexed.get(key)
            if previous_item is None:
                manifest["trials"].append(reused_item)
            else:
                manifest["trials"][manifest["trials"].index(previous_item)] = reused_item
            indexed[key] = reused_item
            matrix_rows.append(dict(reused_item))
            _atomic_write_json(manifest_path, manifest)
            continue

        trial_dir = output_root / model_family / peft_method / run_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "logs").mkdir(parents=True, exist_ok=True)
        item = indexed.get(key)
        if args.resume and item and item.get("status") == "success" and _trial_is_complete(trial_dir):
            summary = _summarize_trial_output(trial_dir)
            matrix_rows.append({**item, **summary})
            continue

        command = build_matrix_trial_command(
            args,
            model_family=model_family,
            peft_method=peft_method,
            trial_dir=trial_dir,
        )
        item = {
            "model_family": model_family,
            "peft_method": peft_method,
            "output_dir": str(trial_dir),
            "command": command,
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }
        if key not in indexed:
            manifest["trials"].append(item)
        else:
            manifest["trials"][manifest["trials"].index(indexed[key])] = item
        indexed[key] = item
        _atomic_write_json(manifest_path, manifest)

        exit_code = _run_logged_command(
            command,
            [launcher_log, trial_dir / "logs" / "runner.log"],
        )
        summary = _summarize_trial_output(trial_dir)
        item.update(
            {
                "finished_at": datetime.now().isoformat(),
                "exit_code": exit_code,
                **summary,
            }
        )
        valid_peft = summary.get("peft_applied_count", 0) > 0
        if args.require_all_peft_success:
            valid_peft = (
                summary.get("row_count", 0) > 0
                and summary.get("ok_count") == summary.get("row_count")
                and summary.get("peft_applied_count") == summary.get("row_count")
                and summary.get("nonzero_trainable_count") == summary.get("row_count")
            )
        item["status"] = "success" if exit_code == 0 and valid_peft else "fail"
        matrix_rows.append(dict(item))
        _atomic_write_json(manifest_path, manifest)
        if item["status"] != "success":
            any_failed = True
            if args.matrix_fail_fast:
                break

    manifest["status"] = "fail" if any_failed else "success"
    manifest["finished_at"] = datetime.now().isoformat()
    common_dataset_count = _write_matrix_common_results(matrix_rows, matrix_dir)
    manifest["common_successful_dataset_count"] = common_dataset_count
    _atomic_write_json(manifest_path, manifest)
    pd.DataFrame(matrix_rows).to_csv(matrix_dir / "matrix_results.csv", index=False)
    summary_lines = [
        f"run_name: {run_name}",
        f"status: {manifest['status']}",
        f"planned_trials: {len(trials)}",
        f"completed_trials: {len(matrix_rows)}",
        f"successful_trials: {sum(row.get('status') == 'success' for row in matrix_rows)}",
        f"failed_trials: {sum(row.get('status') == 'fail' for row in matrix_rows)}",
        f"reused_trials: {sum(row.get('execution') == 'reused' for row in matrix_rows)}",
        f"common_successful_dataset_count: {common_dataset_count}",
    ]
    (matrix_dir / "matrix_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print(f"saved_matrix_manifest: {manifest_path}")
    if any_failed:
        raise RuntimeError(f"One or more PEFT matrix trials failed; see {manifest_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabICLv2 classification benchmarks on dataset roots with "
            "epoch-shuffled chunk FT and AMD/ROCm multi-GPU workers."
        )
    )
    parser.add_argument(
        "--model-family",
        choices=[*MODEL_FAMILIES, "all"],
        default="tabiclv2",
        help="Model backend to run. Use all to expand the TabICLv2/TabPFNv3 matrix.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--tabicl-model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tabpfn-model-path", default=str(DEFAULT_TABPFN_MODEL_PATH))
    parser.add_argument(
        "--tabpfn-v3-binary-model-path",
        default=str(DEFAULT_TABPFN_V3_BINARY_MODEL_PATH),
    )
    parser.add_argument(
        "--tabpfn-v3-multiclass-model-path",
        default=str(DEFAULT_TABPFN_V3_MULTICLASS_MODEL_PATH),
    )
    parser.add_argument("--checkpoint-version", default=DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Exact output directory for a single trial. Matrix mode assigns this "
            "internally and rejects ambiguous top-level usage."
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUT_DIR_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-matrix-manifest",
        default=None,
        help=(
            "Reuse successful TabICLv2 trials from an existing matrix manifest. "
            "Each source trial is validated independently even if the source matrix is running."
        ),
    )
    parser.add_argument(
        "--matrix-fail-fast",
        action="store_true",
        help="Stop the matrix after the first failed trial (used by smoke validation).",
    )
    parser.add_argument(
        "--require-all-peft-success",
        action="store_true",
        help="Require every result row to be status=ok with ttt_applied=true.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-groups", default="2;3")
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
        help="Compatibility alias: enable full-train epoch-chunk FT. No B/C holdout is used.",
    )
    parser.add_argument(
        "--no-ttt",
        dest="ttt_enabled",
        action="store_false",
        help="Disable FT and run ordinary TabICL inference.",
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
            "Per-step FT micro-batch size. When FT data parallel is active, "
            "this value is interpreted per GPU and the effective batch becomes "
            "micro_batch_size x number_of_gpus_in_gpu_group."
        ),
    )
    parser.add_argument("--ttt-weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--ttt-epochs",
        "--ttt-steps",
        dest="ttt_epochs",
        type=int,
        default=30,
        help="Number of epoch-shuffled chunk FT passes. --ttt-steps is kept as a compatibility alias.",
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
        "--ttt-peft-method",
        choices=[*PEFT_METHODS, "all"],
        default="lora",
        help=(
            "Paper PEFT method from On Finetuning Tabular Foundation Models: "
            "LoRA, last ICL layers, or LayerNorm/head/embedding tuning."
        ),
    )
    parser.add_argument(
        "--ttt-peft-targets",
        default="col,row,icl",
        help="Comma-separated PEFT module targets from {col,row,icl}. Default: col,row,icl.",
    )
    parser.add_argument("--ttt-lora-rank", type=int, default=4)
    parser.add_argument("--ttt-lora-alpha", type=float, default=8.0)
    parser.add_argument("--ttt-lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--ttt-last-n-icl-blocks",
        type=int,
        default=1,
        help="Number of final icl_predictor.tf_icl blocks to train for --ttt-peft-method last_layers.",
    )
    parser.add_argument(
        "--ttt-save-ckpt",
        type=parse_bool,
        default=False,
        help="Whether to save intermediate TabICL checkpoints during the FT update path.",
    )
    parser.add_argument(
        "--ttt-save-ckpt-every",
        type=int,
        default=30,
        help="Save a FT checkpoint every N optimizer steps and always save the final step.",
    )
    parser.add_argument(
        "--ttt-save-ckpt-start-step",
        type=parse_optional_int,
        default=None,
        help=(
            "First optimizer step to save a FT checkpoint. Use None to keep the legacy "
            "multiple-of --ttt-save-ckpt-every schedule."
        ),
    )
    parser.add_argument(
        "--ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_true",
        help="Enable intra-worker multi-GPU data parallelism for the FT update path.",
    )
    parser.add_argument(
        "--no-ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_false",
        help="Disable intra-worker multi-GPU data parallelism for the FT update path.",
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
        print("ft_config:")
        print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))

    system_failure_mask = (
        all_df["dataset_name"].astype(str).str.startswith("__WORKER_")
        | all_df["dataset_name"].astype(str).str.startswith("__EMPTY_RESULT__")
    )
    if integrity_errors or system_failure_mask.any():
        details = "; ".join(integrity_errors) if integrity_errors else "worker failure rows were produced"
        raise RuntimeError(f"TabICL FT worker output integrity check failed: {details}")


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
            print("ft_config:")
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

    matrix_mode = args.model_family == "all" or args.ttt_peft_method == "all"
    if matrix_mode:
        if args.out_dir is not None:
            raise ValueError("Matrix mode uses --output-root/--run-name, not --out-dir")
        run_matrix_mode(args)
        return

    if args.model_family == "tabiclv2":
        if args.model_path is not None:
            args.tabicl_model_path = args.model_path
        args.model_path = None if args.models_dir is not None else args.tabicl_model_path
    elif args.model_family == "tabpfnv2":
        if args.models_dir is not None:
            raise ValueError("--models-dir is only supported by the TabICLv2 backend")
        if args.model_path is not None:
            args.tabpfn_model_path = args.model_path
        args.model_path = None
    else:
        if args.models_dir is not None:
            raise ValueError("--models-dir is only supported by the TabICLv2 backend")
        if args.model_path is not None:
            raise ValueError(
                "--model-path is a TabICLv2/TabPFNv2 compatibility option; "
                "use the task-specific --tabpfn-v3-*-model-path arguments for TabPFNv3"
            )

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

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir is not None else (
        Path(args.output_root).expanduser()
        / args.model_family
        / args.ttt_peft_method
        / run_name
    )
    if args.out_dir is None and out_dir.exists() and not args.resume:
        raise FileExistsError(f"Trial output already exists: {out_dir}; use --resume")
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")

    gpu_ids, gpu_groups = resolve_gpu_assignments(args)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    run_config_path = out_dir / "run_config.json"
    run_config = {
        "status": "running",
        "model_family": args.model_family,
        "peft_method": args.ttt_peft_method,
        "data_root": str(data_root),
        "output_dir": str(out_dir),
        "started_at": datetime.now().isoformat(),
        "args": vars(args),
    }
    _atomic_write_json(run_config_path, run_config)
    try:
        if args.model_family in TABPFN_BACKENDS:
            run_tabpfn_single_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)
        elif args.models_dir is not None:
            run_multi_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)
        else:
            run_single_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)
    except Exception as exc:
        run_config.update(
            {
                "status": "fail",
                "finished_at": datetime.now().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_write_json(run_config_path, run_config)
        raise
    run_config.update({"status": "success", "finished_at": datetime.now().isoformat()})
    _atomic_write_json(run_config_path, run_config)


if __name__ == "__main__":
    main()
