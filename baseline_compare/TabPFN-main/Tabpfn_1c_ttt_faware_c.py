#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import importlib
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_COMPARE_ROOT = SCRIPT_DIR.parent
if str(BASELINE_COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_COMPARE_ROOT))


def add_sys_path_if_exists(path: Path | None, *, prepend: bool = True) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    if resolved.exists() and str(resolved) not in sys.path:
        if prepend:
            sys.path.insert(0, str(resolved))
        else:
            sys.path.append(str(resolved))


add_sys_path_if_exists(SCRIPT_DIR / "src", prepend=True)

extensions_path_env = os.environ.get("TABPFN_EXTENSIONS_SRC_DIR", "").strip()
extensions_candidates = [
    Path(extensions_path_env) if extensions_path_env else None,
    SCRIPT_DIR.parent / "tabpfn-extensions" / "src",
    SCRIPT_DIR.parent / "tabpfn-extensions",
    Path.home() / "pythonlibs" / "tabpfn_manyclass" / "src",
    Path.home() / "pythonlibs" / "tabpfn_manyclass",
]
for candidate in extensions_candidates:
    add_sys_path_if_exists(candidate, prepend=False)


def find_tabpfn_extensions_package_dir() -> Path | None:
    for candidate in extensions_candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        package_dir = resolved / "tabpfn_extensions"
        if package_dir.is_dir():
            return package_dir
    return None


TABPFN_EXTENSIONS_PACKAGE_DIR = find_tabpfn_extensions_package_dir()


def import_many_class_classifier():
    try:
        from tabpfn_extensions.many_class import ManyClassClassifier

        return ManyClassClassifier
    except Exception:
        if TABPFN_EXTENSIONS_PACKAGE_DIR is None:
            raise

        for module_name in list(sys.modules):
            if module_name == "tabpfn_extensions" or module_name.startswith(
                "tabpfn_extensions."
            ):
                sys.modules.pop(module_name, None)

        package = types.ModuleType("tabpfn_extensions")
        package.__file__ = str(TABPFN_EXTENSIONS_PACKAGE_DIR / "__init__.py")
        package.__package__ = "tabpfn_extensions"
        package.__path__ = [str(TABPFN_EXTENSIONS_PACKAGE_DIR)]
        sys.modules["tabpfn_extensions"] = package

        importlib.invalidate_caches()
        module = importlib.import_module("tabpfn_extensions.many_class")
        return module.ManyClassClassifier

import numpy as np
import pandas as pd

# =============================================================================
# User-configurable defaults
#
# Edit this block to change the experiment defaults. Every value remains
# overridable from the command line; build_arg_parser() only exposes this block.
# =============================================================================
@dataclass(frozen=True)
class ExperimentDefaults:
    # Data, output, and parallel execution
    data_root: str = "../../data184"
    out_dir: str | None = (
        "../../results/tabpfn/tabpfnv3_lr857e-6_f_test_centroid_reserve_wise_ft"
    )
    workers: int = 1
    gpus: str = "0"
    max_datasets: int | None = None
    random_state: int = 42
    verbose: bool = False

    # TabPFN model and base inference
    model_version: str = "v3"
    model_path: str | None = None
    v3_binary_model_path: str = "auto"
    v3_multiclass_model_path: str = "auto"
    n_estimators: int = 8
    ignore_pretraining_limits: bool = True

    # Many-class compatibility
    many_class: str = "auto"
    many_class_alphabet_size: int = 10
    many_class_redundancy: int = 4
    many_class_n_estimators: int = 16

    # TTT and F-aware-C
    ttt_enabled: bool = True
    ttt_epochs: int = 30
    ttt_max_chunk_size: int = 10_000
    ttt_grad_accumulation_steps: int = 1
    ttt_query_ratio: float = 0.2
    ttt_c_selection: str = "f_test_centroid_reserve"
    ttt_c_metric: str = "standardized_l2"
    ttt_c_reserve_ratio: float = 0.05
    ttt_lr: float = 8.57e-6
    ttt_weight_decay: float = 0.01
    ttt_wise_ft: bool = True
    ttt_wise_alpha: float = 0.1
    ttt_grad_clip: float = 1.0
    ttt_patience: int = 8
    ttt_min_delta: float = 1e-4
    ttt_eval_metric: str = "acc"
    ttt_validation_fraction: float = 0.1
    ttt_n_estimators_finetune: int = 2
    ttt_validation_n_estimators: int = 2
    ttt_lr_scheduler: bool = True
    ttt_lr_warmup_only: bool = False
    ttt_activation_checkpointing: bool = True

    # Failed-dataset retry and result merging
    retry_failed_datasets_only: bool = False
    reference_results_csv: str = "v3_results/all_classification_results.csv"
    retry_include_oom_failures: bool = False
    merge_results_from_csv: str | None = None


DEFAULTS = ExperimentDefaults()

CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"
TABPFN_CLASS_LIMIT = DEFAULTS.many_class_alphabet_size
FAWARE_C_SELECTION = "f_test_centroid_reserve"
FAWARE_C_SOURCE = "test"
FAWARE_C_RANDOM_SOURCE = "none"
WISE_FT_MAX_DATASET_ROWS = 2_000
RESULTS_ROOT = BASELINE_COMPARE_ROOT / "results"
V3_BINARY_CLASSIFIER_FILE = "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
V3_MULTICLASS_CLASSIFIER_FILE = "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
GATED_CLASSIFIER_CACHE_FILES = {
    "v2.5": "tabpfn-v2.5-classifier-v2.5_default.ckpt",
    "v2.6": "tabpfn-v2.6-classifier-v2.6_default.ckpt",
    "v3": "tabpfn-v3-classifier-v3_default.ckpt",
}

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


def sanitize_component(value: Any) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unknown"


def dataset_label(data_root: Path, dataset_dirs: Sequence[Path]) -> str:
    if len(dataset_dirs) == 1:
        return Path(dataset_dirs[0]).name
    root_name = Path(data_root).name or "datasets"
    return f"{root_name}_{len(dataset_dirs)}datasets"


def checkpoint_stem(value: Any) -> str | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if raw_value.lower() in {"", "auto", "none", "null"}:
        return None
    return Path(raw_value).stem or None


def infer_estimator_label(
    args: Any,
    *,
    attr_name: str = "n_estimators",
    all_if_zero: bool = False,
) -> str:
    value = getattr(args, attr_name)
    if all_if_zero and int(value) == 0:
        return "infer_estall"
    return f"infer_est{value}"


def seed_label(args: Any) -> str:
    seed_value = getattr(args, "random_state", None)
    if seed_value is None:
        seed_value = getattr(args, "seed")
    return f"seed{seed_value}"


def no_ttt_label() -> str:
    return "no_ttt"


def ttt_label(args: Any) -> str:
    if not bool(getattr(args, "ttt_enabled", False)):
        return no_ttt_label()
    parts = [
        f"ttt_eval-{getattr(args, 'ttt_eval_metric')}",
        f"ep{getattr(args, 'ttt_epochs')}",
        f"chunk{getattr(args, 'ttt_max_chunk_size')}",
        f"lr{getattr(args, 'ttt_lr')}",
        f"q{getattr(args, 'ttt_query_ratio')}",
    ]
    if hasattr(args, "ttt_grad_accumulation_steps"):
        parts.append(f"accum{getattr(args, 'ttt_grad_accumulation_steps')}")
    if hasattr(args, "ttt_n_estimators_finetune"):
        parts.append(f"finetuneest{getattr(args, 'ttt_n_estimators_finetune')}")
    if hasattr(args, "ttt_validation_n_estimators"):
        parts.append(f"valest{getattr(args, 'ttt_validation_n_estimators')}")
    return "_".join(parts)


def auto_out_dir(
    *,
    model_label: str,
    data_root: Path,
    dataset_dirs: Sequence[Path],
    infer_label: str,
    ttt_label: str,
    seed_label: str,
) -> Path:
    name_parts = [
        model_label,
        dataset_label(data_root, dataset_dirs),
        infer_label,
        ttt_label,
        seed_label,
    ]
    auto_name = "__".join(sanitize_component(part) for part in name_parts)
    return RESULTS_ROOT / auto_name


@dataclass
class LoadedDataset:
    dataset_name: str
    dataset_dir: Path
    task_type: Optional[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame | None
    y_val: np.ndarray | None
    X_test: pd.DataFrame
    y_test: np.ndarray
    categorical_feature_indices: list[int]

    @property
    def n_val(self) -> int:
        return 0 if self.y_val is None else int(len(self.y_val))

    @property
    def X_train_merged(self) -> pd.DataFrame:
        if self.X_val is None:
            return self.X_train
        return pd.concat([self.X_train, self.X_val], axis=0, ignore_index=True)

    @property
    def y_train_merged(self) -> np.ndarray:
        if self.y_val is None:
            return self.y_train
        return np.concatenate([np.asarray(self.y_train), np.asarray(self.y_val)], axis=0)

    @property
    def n_train_report(self) -> int:
        return int(len(self.y_train_merged))

    @property
    def n_classes(self) -> int:
        pieces = [np.asarray(self.y_train)]
        if self.y_val is not None:
            pieces.append(np.asarray(self.y_val))
        pieces.append(np.asarray(self.y_test))
        return int(len(pd.unique(pd.Series(np.concatenate(pieces, axis=0)))))


@dataclass
class PredictionResult:
    y_pred: Any
    y_proba: Any | None
    classes: Any | None
    fit_seconds: float
    predict_seconds: float
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
    ttt_c_reserve_ratio: Optional[float] = None
    ttt_c_f_distance_mean: Optional[float] = None
    ttt_c_f_distance_std: Optional[float] = None
    ttt_c_fallback_reason: Optional[str] = None
    ttt_c_label_coverage_ok: bool = True


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
    ttt_c_reserve_ratio: Optional[float] = None
    ttt_c_f_distance_mean: Optional[float] = None
    ttt_c_f_distance_std: Optional[float] = None
    ttt_c_fallback_reason: Optional[str] = None
    ttt_c_label_coverage_ok: bool = True


@dataclass
class TTTConfig:
    enabled: bool = True
    epochs: int = 3
    max_chunk_size: int = 2_000
    grad_accumulation_steps: int = 5
    query_ratio: float = 0.2
    lr: float = 1e-5
    weight_decay: float = 0.01
    wise_ft: bool = True
    wise_alpha: float = 0.1
    grad_clip: Optional[float] = 1.0
    patience: int = 8
    min_delta: float = 1e-4
    eval_metric: str = "roc_auc"
    validation_fraction: float = 0.1
    n_estimators_finetune: int = 2
    n_estimators_validation: int = 2
    n_estimators_final_inference: int = 32
    use_lr_scheduler: bool = True
    lr_warmup_only: bool = False
    use_activation_checkpointing: bool = True
    save_checkpoint_interval: Optional[int] = None
    c_selection: str = FAWARE_C_SELECTION
    c_metric: str = "standardized_l2"
    c_reserve_ratio: float = 0.05


RESULT_COLUMNS = list(ResultRow.__annotations__.keys())
OOM_ERROR_MARKERS = (
    "out of memory",
    "oom",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "tabpfncudaoutofmemoryerror",
    "tabpfnmpsoutofmemoryerror",
)


class SkipDataset(Exception):
    pass


def faware_c_source_for_selection(selection: str) -> str:
    return FAWARE_C_SOURCE if selection == FAWARE_C_SELECTION else FAWARE_C_RANDOM_SOURCE


def ttt_c_metric_for_config(config: TTTConfig) -> Optional[str]:
    return config.c_metric if config.c_selection == FAWARE_C_SELECTION else None


def ttt_c_reserve_ratio_for_config(config: TTTConfig) -> Optional[float]:
    return config.c_reserve_ratio if config.c_selection == FAWARE_C_SELECTION else None


def wise_ft_enabled_for_config(config: TTTConfig) -> bool:
    return bool(config.enabled and config.wise_ft and config.c_selection == FAWARE_C_SELECTION)


def wise_ft_enabled_for_dataset(config: TTTConfig, dataset_total_rows: int) -> bool:
    return (
        wise_ft_enabled_for_config(config)
        and int(dataset_total_rows) < WISE_FT_MAX_DATASET_ROWS
    )


def ttt_c_reason_suffix(config: TTTConfig) -> str:
    source = faware_c_source_for_selection(config.c_selection)
    parts = [f"c_selection={config.c_selection}", f"c_source={source}"]
    if config.c_selection == FAWARE_C_SELECTION:
        parts.extend(
            [
                f"c_metric={config.c_metric}",
                f"c_reserve_ratio={config.c_reserve_ratio}",
            ]
        )
    return " | " + " | ".join(parts)


def ttt_c_batch_mode(config: TTTConfig) -> str:
    base = f"tabpfn_finetuning_api_{config.c_selection}"
    accumulation = _grad_accumulation_steps(config)
    if accumulation == 1:
        return base
    return f"{base}_grad_accum{accumulation}"


def build_ttt_config(args: argparse.Namespace) -> TTTConfig:
    if int(args.ttt_epochs) < 0:
        raise ValueError("--ttt-epochs must be >= 0")
    if int(args.ttt_max_chunk_size) < 2:
        raise ValueError("--ttt-max-chunk-size must be >= 2")
    if int(args.ttt_grad_accumulation_steps) < 1:
        raise ValueError("--ttt-grad-accumulation-steps must be >= 1")
    if not 0.0 < float(args.ttt_query_ratio) < 1.0:
        raise ValueError("--ttt-query-ratio must be in (0, 1)")
    if not 0.0 < float(args.ttt_validation_fraction) < 1.0:
        raise ValueError("--ttt-validation-fraction must be in (0, 1)")
    if int(args.ttt_n_estimators_finetune) < 1:
        raise ValueError("--ttt-n-estimators-finetune must be >= 1")
    if int(args.ttt_validation_n_estimators) < 1:
        raise ValueError("--ttt-validation-n-estimators must be >= 1")
    if int(args.n_estimators) < 1:
        raise ValueError("--n-estimators must be >= 1")
    if str(args.ttt_eval_metric) not in {"roc_auc", "log_loss", "acc"}:
        raise ValueError("--ttt-eval-metric must be one of: roc_auc, log_loss, acc")
    if not 0.0 <= float(args.ttt_wise_alpha) <= 1.0:
        raise ValueError("--ttt-wise-alpha must be in [0, 1]")
    c_selection = str(args.ttt_c_selection)
    if c_selection not in {FAWARE_C_SELECTION, "random"}:
        raise ValueError(
            "--ttt-c-selection must be one of: "
            f"{FAWARE_C_SELECTION}, random"
        )
    if str(args.ttt_c_metric) not in {"raw_l2", "standardized_l2"}:
        raise ValueError("--ttt-c-metric must be one of: raw_l2, standardized_l2")
    if c_selection == FAWARE_C_SELECTION:
        if not 0.0 < float(args.ttt_c_reserve_ratio) < 1.0:
            raise ValueError("--ttt-c-reserve-ratio must be in (0, 1)")
        if float(args.ttt_c_reserve_ratio) + float(args.ttt_query_ratio) >= 1.0:
            raise ValueError(
                "--ttt-c-reserve-ratio + --ttt-query-ratio must be < 1 for "
                f"{FAWARE_C_SELECTION}"
            )
    grad_clip = None if float(args.ttt_grad_clip) <= 0 else float(args.ttt_grad_clip)
    return TTTConfig(
        enabled=bool(args.ttt),
        epochs=int(args.ttt_epochs),
        max_chunk_size=int(args.ttt_max_chunk_size),
        grad_accumulation_steps=int(args.ttt_grad_accumulation_steps),
        query_ratio=float(args.ttt_query_ratio),
        lr=float(args.ttt_lr),
        weight_decay=float(args.ttt_weight_decay),
        wise_ft=bool(args.ttt_wise_ft) and c_selection == FAWARE_C_SELECTION,
        wise_alpha=float(args.ttt_wise_alpha),
        grad_clip=grad_clip,
        patience=int(args.ttt_patience),
        min_delta=float(args.ttt_min_delta),
        eval_metric=str(args.ttt_eval_metric),
        validation_fraction=float(args.ttt_validation_fraction),
        n_estimators_finetune=int(args.ttt_n_estimators_finetune),
        n_estimators_validation=int(args.ttt_validation_n_estimators),
        n_estimators_final_inference=int(args.n_estimators),
        use_lr_scheduler=bool(args.ttt_lr_scheduler),
        lr_warmup_only=bool(args.ttt_lr_warmup_only),
        use_activation_checkpointing=bool(args.ttt_activation_checkpointing),
        save_checkpoint_interval=None,
        c_selection=c_selection,
        c_metric=str(args.ttt_c_metric),
        c_reserve_ratio=float(args.ttt_c_reserve_ratio),
    )


def get_estimator_classes(estimator: Any) -> np.ndarray | None:
    for attr_owner in (
        estimator,
        getattr(estimator, "finetuned_inference_classifier_", None),
        getattr(estimator, "estimator", None),
    ):
        if attr_owner is None:
            continue
        classes = getattr(attr_owner, "classes_", None)
        if classes is not None:
            return np.asarray(classes)
    return None


def predict_from_proba_or_model(
    estimator: Any,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    y_proba = None
    try:
        y_proba = np.asarray(estimator.predict_proba(X_test))
    except Exception:
        y_pred = np.asarray(estimator.predict(X_test))
        return y_pred, None, get_estimator_classes(estimator)

    classes = get_estimator_classes(estimator)
    if y_proba.ndim != 2 or y_proba.shape[0] != len(X_test) or y_proba.shape[1] < 1:
        y_pred = np.asarray(estimator.predict(X_test))
        return y_pred, y_proba, classes

    encoded = np.argmax(y_proba, axis=1)
    if classes is not None and len(classes) == y_proba.shape[1]:
        y_pred = classes[encoded]
    else:
        try:
            y_pred = np.asarray(estimator.predict(X_test))
        except Exception:
            y_pred = encoded
    return np.asarray(y_pred), y_proba, classes


def compute_roc_auc(
    y_true: Any,
    y_proba: Any | None,
    classes: Any | None,
) -> Optional[float]:
    if y_proba is None:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != len(y_true_arr):
            return None
        if len(np.unique(y_true_arr)) < 2:
            return None
        class_arr = None if classes is None else np.asarray(classes)
        if class_arr is not None and len(class_arr) != proba_arr.shape[1]:
            class_arr = None

        if proba_arr.shape[1] == 2:
            if class_arr is not None and len(class_arr) == 2:
                y_binary = (y_true_arr == class_arr[1]).astype(int)
                if len(np.unique(y_binary)) < 2:
                    return None
                return float(roc_auc_score(y_binary, proba_arr[:, 1]))
            return float(roc_auc_score(y_true_arr, proba_arr[:, 1]))
        if class_arr is None:
            class_arr = np.asarray(sorted(pd.unique(pd.Series(y_true_arr)).tolist()))
            if len(class_arr) != proba_arr.shape[1]:
                return None
        return float(
            roc_auc_score(
                y_true_arr,
                proba_arr,
                labels=list(class_arr),
                multi_class="ovr",
            )
        )
    except Exception:
        return None


def compute_weighted_f1(y_true: Any, y_pred: Any) -> Optional[float]:
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
    try:
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))
    except Exception:
        return None


def compute_log_loss(
    y_true: Any,
    y_proba: Any | None,
    classes: Any | None,
) -> Optional[float]:
    if y_proba is None:
        return None
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


def extract_ttt_validation_summary(
    classifier: Any,
    *,
    eval_metric: str,
    epochs: int,
) -> dict[str, Any]:
    history = getattr(classifier, "_ttt_eval_history_", None) or []
    valid_entries: list[tuple[int, dict[str, Any]]] = []
    for idx, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        primary = _optional_float(item.get("primary"))
        if primary is not None:
            valid_entries.append((idx, item))

    baseline_metric = valid_entries[0][1].get("primary") if valid_entries else None
    baseline_metric = _optional_float(baseline_metric)
    best_idx = 0
    best_metric = baseline_metric
    if valid_entries:
        best_idx, best_item = max(
            valid_entries,
            key=lambda pair: float(pair[1].get("primary")),
        )
        best_metric = _optional_float(best_item.get("primary"))

    return {
        "ttt_val_eval_metric": eval_metric,
        "ttt_val_baseline_metric": baseline_metric,
        "ttt_val_best_metric": best_metric,
        "ttt_val_baseline_accuracy": None,
        "ttt_val_best_accuracy": None,
        "ttt_best_epoch": int(best_idx) if best_metric is not None else 0,
        "ttt_stopped_early": bool(len(history) > 1 and len(history) < int(epochs) + 1),
    }


def is_oom_exception(value: BaseException) -> bool:
    return is_oom_error_message(f"{type(value).__name__}: {value}")


def format_exception_for_csv(value: BaseException) -> str:
    return " ".join(f"{type(value).__name__}: {value}".split())


def format_optional_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _optional_float(value: Any) -> Optional[float]:
    try:
        value_float = float(value)
        if not np.isfinite(value_float):
            return None
        return value_float
    except (TypeError, ValueError):
        return None


@dataclass
class FawareSplitStats:
    distance_sum: float = 0.0
    distance_sumsq: float = 0.0
    distance_count: int = 0
    fallback_reasons: list[str] | None = None
    label_coverage_ok: bool = True

    def record(
        self,
        distances: np.ndarray | None = None,
        *,
        fallback_reason: str | None = None,
        label_coverage_ok: bool = True,
    ) -> None:
        if distances is not None and len(distances):
            values = np.asarray(distances, dtype=np.float64)
            self.distance_sum += float(np.sum(values))
            self.distance_sumsq += float(np.sum(values * values))
            self.distance_count += int(len(values))
        if fallback_reason:
            if self.fallback_reasons is None:
                self.fallback_reasons = []
            if fallback_reason not in self.fallback_reasons:
                self.fallback_reasons.append(fallback_reason)
        self.label_coverage_ok = bool(self.label_coverage_ok and label_coverage_ok)


@dataclass
class FawareIndexSplit:
    ctx_idx: np.ndarray
    qry_idx: np.ndarray
    reserved_idx: np.ndarray
    reserved_scores: np.ndarray
    fallback_reason: str | None = None
    label_coverage_ok: bool = True


@dataclass(frozen=True)
class GlobalCentroidReservation:
    reserved_row_ids: np.ndarray
    scores: np.ndarray
    requested_count: int


@dataclass
class AlignedSplitResult:
    X_context: Any
    X_query: Any
    y_context: Any
    y_query: Any
    fallback_reason: str | None = None
    label_coverage_ok: bool = True


def _grad_accumulation_steps(config: Any) -> int:
    return int(getattr(config, "grad_accumulation_steps", 1))


def _take_rows(obj: Any, idx: np.ndarray) -> Any:
    return obj.iloc[idx] if hasattr(obj, "iloc") else obj[idx]


def _standardize_train_reference(
    train_arr: np.ndarray,
    ref_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train_arr, axis=0)
    scale = np.nanstd(train_arr, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (
        np.nan_to_num((train_arr - mean) / scale, nan=0.0),
        np.nan_to_num((ref_arr - mean) / scale, nan=0.0),
    )


def _selection_only_feature_space(
    X_train: Any,
    X_reference: Any,
    *,
    categorical_feature_indices: Sequence[int],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import OrdinalEncoder

    train_frame = pd.DataFrame(X_train).reset_index(drop=True)
    reference_frame = pd.DataFrame(X_reference).reset_index(drop=True)
    if train_frame.shape[1] != reference_frame.shape[1]:
        raise ValueError("Training and test selection spaces must have equal width")

    categorical = set(int(index) for index in categorical_feature_indices)
    train_columns: list[np.ndarray] = []
    reference_columns: list[np.ndarray] = []
    for column_index in range(train_frame.shape[1]):
        train_column = train_frame.iloc[:, column_index]
        reference_column = reference_frame.iloc[:, column_index]
        if column_index in categorical:
            missing_token = "__TABPFN_SELECTION_MISSING__"
            train_values = (
                train_column.astype("object")
                .where(~train_column.isna(), missing_token)
                .map(str)
                .to_numpy()
                .reshape(-1, 1)
            )
            reference_values = (
                reference_column.astype("object")
                .where(~reference_column.isna(), missing_token)
                .map(str)
                .to_numpy()
                .reshape(-1, 1)
            )
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float64,
            )
            train_encoded = encoder.fit_transform(train_values).reshape(-1)
            reference_encoded = encoder.transform(reference_values).reshape(-1)
            train_columns.append(train_encoded)
            reference_columns.append(reference_encoded)
            continue

        train_numeric = pd.to_numeric(train_column, errors="coerce").to_numpy(
            dtype=np.float64
        )
        reference_numeric = pd.to_numeric(
            reference_column, errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite_train = train_numeric[np.isfinite(train_numeric)]
        median = float(np.median(finite_train)) if len(finite_train) else 0.0
        train_columns.append(
            np.where(np.isfinite(train_numeric), train_numeric, median)
        )
        reference_columns.append(
            np.where(np.isfinite(reference_numeric), reference_numeric, median)
        )

    train_arr = np.column_stack(train_columns).astype(np.float64, copy=False)
    reference_arr = np.column_stack(reference_columns).astype(np.float64, copy=False)
    if metric == "standardized_l2":
        return _standardize_train_reference(train_arr, reference_arr)
    return train_arr, reference_arr


def _test_centroid_distance_scores(
    X_train: Any,
    X_test: Any,
    *,
    categorical_feature_indices: Sequence[int],
    metric: str,
) -> np.ndarray:
    X_sel, test_sel = _selection_only_feature_space(
        X_train,
        X_test,
        categorical_feature_indices=categorical_feature_indices,
        metric=metric,
    )
    if len(test_sel) == 0:
        return np.zeros(X_sel.shape[0], dtype=np.float64)
    test_centroid = np.mean(test_sel, axis=0)
    scores = np.mean((X_sel - test_centroid) ** 2, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64,
        copy=False,
    )


def _build_global_test_centroid_reservation(
    X_train: Any,
    X_test: Any,
    *,
    categorical_feature_indices: Sequence[int],
    metric: str,
    reserve_ratio: float,
) -> GlobalCentroidReservation:
    n_train = int(len(X_train))
    if n_train < 3:
        raise ValueError(
            f"Need at least three rows for {FAWARE_C_SELECTION} context/query split"
        )
    if X_test is None or len(X_test) == 0:
        raise ValueError(f"Missing test rows for {FAWARE_C_SELECTION}")
    scores = _test_centroid_distance_scores(
        X_train,
        X_test,
        categorical_feature_indices=categorical_feature_indices,
        metric=metric,
    )
    requested_count = max(1, int(math.ceil(float(reserve_ratio) * n_train)))
    reserve_count = min(requested_count, n_train - 1)
    ranked_row_ids = np.argsort(-scores, kind="stable")
    return GlobalCentroidReservation(
        reserved_row_ids=np.asarray(ranked_row_ids[:reserve_count], dtype=np.int64),
        scores=scores,
        requested_count=requested_count,
    )


def _has_query_label_coverage(y_chunk: Any, ctx_idx: Any, qry_idx: Any) -> bool:
    y_arr = np.asarray(y_chunk)
    ctx_labels = set(y_arr[np.asarray(ctx_idx, dtype=int)].tolist())
    qry_labels = set(y_arr[np.asarray(qry_idx, dtype=int)].tolist())
    return qry_labels.issubset(ctx_labels)


def _is_stratified_split_feasibility_error(value: ValueError) -> bool:
    message = str(value).lower()
    return any(
        marker in message
        for marker in (
            "least populated class",
            "test_size",
            "train_size",
            "number of classes",
            "minimum number of groups",
        )
    )


def _split_fn_keyword(split_fn: Any, name: str, default: Any = None) -> Any:
    keywords = getattr(split_fn, "keywords", None) or {}
    return keywords.get(name, default)


def _max_reserved_count_for_split_fn(split_fn: Any, n_samples: int) -> int:
    test_size = _split_fn_keyword(split_fn, "test_size")
    if test_size is None or isinstance(test_size, float):
        return max(1, int(n_samples) - 2)
    return max(0, int(n_samples) - int(test_size) - 1)


def _join_reasons(*reasons: str | None) -> str | None:
    clean = [str(reason) for reason in reasons if reason]
    if not clean:
        return None
    return "; ".join(clean)


def _call_original_split_fn(
    split_fn: Any,
    X: Any,
    y: Any,
    stratify: Any,
) -> tuple[Any, Any, Any, Any]:
    try:
        return split_fn(X, y, stratify=stratify)
    except TypeError:
        return split_fn(X, y)


def _aligned_split_with_stratify_fallback(
    split_fn: Any,
    X: Any,
    y: Any,
    stratify: Any,
    *,
    fallback_label: str,
) -> AlignedSplitResult:
    try:
        X_context, X_query, y_context, y_query = _call_original_split_fn(
            split_fn,
            X,
            y,
            stratify,
        )
        fallback_reason = None
    except ValueError as exc:
        if stratify is None or not _is_stratified_split_feasibility_error(exc):
            raise
        fallback_reason = (
            f"{fallback_label} stratified split failed; retried with "
            f"stratify=None: {type(exc).__name__}: {exc}"
        )
        X_context, X_query, y_context, y_query = _call_original_split_fn(
            split_fn,
            X,
            y,
            None,
        )
    label_coverage_ok = _has_query_label_coverage_arrays(y_context, y_query)
    return AlignedSplitResult(
        X_context=X_context,
        X_query=X_query,
        y_context=y_context,
        y_query=y_query,
        fallback_reason=fallback_reason,
        label_coverage_ok=label_coverage_ok,
    )


def _has_query_label_coverage_arrays(y_context: Any, y_query: Any) -> bool:
    try:
        ctx_arr = np.asarray(y_context)
        qry_arr = np.asarray(y_query)
        ctx_labels = set(ctx_arr.tolist())
        qry_labels = set(qry_arr.tolist())
        return qry_labels.issubset(ctx_labels)
    except Exception:
        return False


def _make_aligned_split_fn(
    split_fn: Any,
    stats: FawareSplitStats,
    *,
    fallback_label: str,
):
    def aligned_split_fn(X: Any, y: Any, stratify: Any = None) -> tuple[Any, Any, Any, Any]:
        result = _aligned_split_with_stratify_fallback(
            split_fn,
            X,
            y,
            stratify,
            fallback_label=fallback_label,
        )
        stats.record(
            fallback_reason=result.fallback_reason,
            label_coverage_ok=result.label_coverage_ok,
        )
        return result.X_context, result.X_query, result.y_context, result.y_query

    return aligned_split_fn


def _repair_query_label_coverage(
    y_chunk: Any,
    ctx_idx: np.ndarray,
    qry_idx: np.ndarray,
    *,
    protected_context_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    y_arr = np.asarray(y_chunk)
    context = [int(index) for index in ctx_idx.tolist()]
    query = [int(index) for index in qry_idx.tolist()]
    protected = set(np.asarray(protected_context_idx, dtype=int).tolist())
    context_label_counts: dict[Any, int] = {}
    for context_index in context:
        label = y_arr[context_index]
        context_label_counts[label] = context_label_counts.get(label, 0) + 1

    repaired = 0
    moved = 0
    moved_indices: set[int] = set()
    for query_position, query_index in enumerate(list(query)):
        query_label = y_arr[query_index]
        if context_label_counts.get(query_label, 0) > 0:
            continue
        swap_index = next(
            (
                context_index
                for context_index in context
                if context_index not in protected
                and context_label_counts.get(y_arr[context_index], 0) > 1
            ),
            None,
        )
        if swap_index is not None:
            swap_label = y_arr[swap_index]
            context.remove(swap_index)
            context.append(query_index)
            query[query_position] = swap_index
            context_label_counts[swap_label] -= 1
            context_label_counts[query_label] = (
                context_label_counts.get(query_label, 0) + 1
            )
            repaired += 1
            continue
        if len(query) - len(moved_indices) > 1:
            context.append(query_index)
            moved_indices.add(query_index)
            context_label_counts[query_label] = (
                context_label_counts.get(query_label, 0) + 1
            )
            moved += 1

    if repaired == 0 and moved == 0:
        return ctx_idx, qry_idx, None
    if moved_indices:
        query = [index for index in query if index not in moved_indices]
    actions: list[str] = []
    if repaired:
        actions.append(f"swapped {repaired}")
    if moved:
        actions.append(f"moved {moved}")
    return (
        np.asarray(sorted(set(context)), dtype=int),
        np.asarray(query, dtype=int),
        f"{FAWARE_C_SELECTION} {' and '.join(actions)} query row(s) "
        "for label coverage",
    )


def _global_test_centroid_reserve_indices(
    y_chunk: Any,
    global_row_ids: Any,
    reservation: GlobalCentroidReservation,
    split_fn: Any,
    stratify: Any,
) -> FawareIndexSplit:
    n_samples = int(len(y_chunk))
    if n_samples < 3:
        raise ValueError(
            f"Need at least three rows for {FAWARE_C_SELECTION} context/query split"
        )
    row_ids = np.asarray(global_row_ids, dtype=np.int64).reshape(-1)
    if len(row_ids) != n_samples:
        raise ValueError("Chunk rows and global row IDs are not aligned")

    reserved_global = set(reservation.reserved_row_ids.tolist())
    reserved_idx = np.asarray(
        [
            local_index
            for local_index, row_id in enumerate(row_ids.tolist())
            if row_id in reserved_global
        ],
        dtype=int,
    )
    max_reserved_for_split = _max_reserved_count_for_split_fn(split_fn, n_samples)
    reserve_adjust_reason = None
    if len(reserved_idx) > max_reserved_for_split:
        ranked_local = sorted(
            reserved_idx.tolist(),
            key=lambda local_index: (
                -float(reservation.scores[int(row_ids[local_index])]),
                int(row_ids[local_index]),
            ),
        )
        retained = ranked_local[:max_reserved_for_split]
        reserve_adjust_reason = (
            f"{FAWARE_C_SELECTION} chunk capacity released "
            f"{len(reserved_idx) - max_reserved_for_split} lowest-score reserved "
            "row(s) to preserve the official split"
        )
        reserved_idx = np.asarray(sorted(retained), dtype=int)
    reserved_set = set(reserved_idx.tolist())
    allowed_idx = np.asarray(
        [idx for idx in range(n_samples) if idx not in reserved_set],
        dtype=int,
    )
    if len(allowed_idx) < 2:
        raise ValueError("Need at least two non-reserved rows for query selection")

    y_arr = np.asarray(y_chunk)
    y_allowed = y_arr[allowed_idx]
    stratify_allowed = None if stratify is None else np.asarray(stratify)[allowed_idx]
    index_column = allowed_idx.reshape(-1, 1)
    aligned = _aligned_split_with_stratify_fallback(
        split_fn,
        index_column,
        y_allowed,
        stratify_allowed,
        fallback_label=f"{FAWARE_C_SELECTION} allowed-row",
    )
    allowed_ctx_idx = np.asarray(aligned.X_context, dtype=int).reshape(-1)
    qry_idx = np.asarray(aligned.X_query, dtype=int).reshape(-1)
    if set(reserved_idx.tolist()) & set(qry_idx.tolist()):
        raise ValueError(
            f"{FAWARE_C_SELECTION} reserved context rows leaked into query"
        )

    ctx_idx = np.unique(np.concatenate([reserved_idx, allowed_ctx_idx])).astype(int)
    ctx_idx, qry_idx, coverage_repair_reason = _repair_query_label_coverage(
        y_chunk,
        ctx_idx,
        qry_idx,
        protected_context_idx=reserved_idx,
    )
    coverage_ok = _has_query_label_coverage(y_chunk, ctx_idx, qry_idx)
    reserved_row_ids = row_ids[reserved_idx]
    return FawareIndexSplit(
        ctx_idx=ctx_idx,
        qry_idx=qry_idx.astype(int),
        reserved_idx=reserved_idx,
        reserved_scores=reservation.scores[reserved_row_ids],
        fallback_reason=_join_reasons(
            reserve_adjust_reason,
            aligned.fallback_reason,
            coverage_repair_reason,
        ),
        label_coverage_ok=coverage_ok,
    )


def _make_faware_get_preprocessed_dataset_chunks(
    *,
    global_reservation: GlobalCentroidReservation | None,
    config: TTTConfig,
    stats: FawareSplitStats,
    original_get_preprocessed_dataset_chunks: Any,
):
    def get_preprocessed_dataset_chunks(  # noqa: PLR0913
        calling_instance: Any,
        X_raw: Any,
        y_raw: Any,
        split_fn: Any,
        max_data_size: int | None,
        model_type: str,
        *,
        equal_split_size: bool,
        data_shuffle_seed: int,
        preprocessing_random_state: int | np.random.Generator,
        shuffle: bool = True,
        force_no_stratify: bool = False,
    ):
        if config.c_selection == "random":
            aligned_split_fn = _make_aligned_split_fn(
                split_fn,
                stats,
                fallback_label="random",
            )
            return original_get_preprocessed_dataset_chunks(
                calling_instance=calling_instance,
                X_raw=X_raw,
                y_raw=y_raw,
                split_fn=aligned_split_fn,
                max_data_size=max_data_size,
                model_type=model_type,
                equal_split_size=equal_split_size,
                data_shuffle_seed=data_shuffle_seed,
                preprocessing_random_state=preprocessing_random_state,
                shuffle=shuffle,
                force_no_stratify=force_no_stratify,
            )
        if model_type != "classifier":
            return original_get_preprocessed_dataset_chunks(
                calling_instance=calling_instance,
                X_raw=X_raw,
                y_raw=y_raw,
                split_fn=split_fn,
                max_data_size=max_data_size,
                model_type=model_type,
                equal_split_size=equal_split_size,
                data_shuffle_seed=data_shuffle_seed,
                preprocessing_random_state=preprocessing_random_state,
                shuffle=shuffle,
                force_no_stratify=force_no_stratify,
            )
        if global_reservation is None:
            stats.record(
                fallback_reason=f"missing global reservation for {FAWARE_C_SELECTION}",
                label_coverage_ok=False,
            )
            aligned_split_fn = _make_aligned_split_fn(
                split_fn,
                stats,
                fallback_label=f"{FAWARE_C_SELECTION} missing-global-reservation",
            )
            return original_get_preprocessed_dataset_chunks(
                calling_instance=calling_instance,
                X_raw=X_raw,
                y_raw=y_raw,
                split_fn=aligned_split_fn,
                max_data_size=max_data_size,
                model_type=model_type,
                equal_split_size=equal_split_size,
                data_shuffle_seed=data_shuffle_seed,
                preprocessing_random_state=preprocessing_random_state,
                shuffle=shuffle,
                force_no_stratify=force_no_stratify,
            )

        from tabpfn.finetuning import data_util
        from tabpfn.preprocessing.datamodel import FeatureModality

        if not isinstance(X_raw, list):
            X_raw = [X_raw]
        if not isinstance(y_raw, list):
            y_raw = [y_raw]
        assert len(X_raw) == len(y_raw), "X and y lists must have the same length."

        if not hasattr(calling_instance, "models_") or calling_instance.models_ is None:
            calling_instance._initialize_model_variables()

        X_split, y_split, row_id_split = [], [], []
        row_id_offset = 0
        for X_item, y_item in zip(X_raw, y_raw):
            item_row_ids = np.arange(
                row_id_offset,
                row_id_offset + len(y_item),
                dtype=np.int64,
            )
            row_id_offset += len(y_item)
            if max_data_size is not None:
                Xparts, yparts = data_util.shuffle_and_chunk_data(
                    X_item,
                    y_item,
                    max_chunk_size=max_data_size,
                    equal_split_size=equal_split_size,
                    seed=data_shuffle_seed,
                    task="multiclass",
                    shuffle=shuffle,
                )
                row_id_parts, row_id_y_parts = data_util.shuffle_and_chunk_data(
                    item_row_ids,
                    y_item,
                    max_chunk_size=max_data_size,
                    equal_split_size=equal_split_size,
                    seed=data_shuffle_seed,
                    task="multiclass",
                    shuffle=shuffle,
                )
                if len(yparts) != len(row_id_y_parts) or any(
                    not np.array_equal(np.asarray(y_part), np.asarray(row_id_y_part))
                    for y_part, row_id_y_part in zip(yparts, row_id_y_parts)
                ):
                    raise ValueError(
                        f"{FAWARE_C_SELECTION} row-ID chunking lost alignment"
                    )
            else:
                Xparts, yparts = [X_item], [y_item]
                row_id_parts = [item_row_ids]
            X_split.extend(Xparts)
            y_split.extend(yparts)
            row_id_split.extend(row_id_parts)
        if row_id_offset != len(global_reservation.scores):
            raise ValueError(
                f"{FAWARE_C_SELECTION} global reservation has "
                f"{len(global_reservation.scores)} rows but fit received "
                f"{row_id_offset}"
            )

        row_ids_by_x_id: dict[int, np.ndarray] = {}
        dataset_config_collection: list[Any] = []
        for X_item, y_item, row_ids in zip(X_split, y_split, row_id_split):
            ensemble_configs, X_mod, y_mod = (
                calling_instance._initialize_dataset_preprocessing(
                    X=X_item,
                    y=y_item,
                    random_state=preprocessing_random_state,
                )
            )
            if len(X_mod) != len(row_ids):
                raise ValueError(
                    f"{FAWARE_C_SELECTION} preprocessing changed row count"
                )
            row_ids_by_x_id[id(X_mod)] = np.asarray(row_ids, dtype=np.int64)
            current_cat_ix = calling_instance.inferred_feature_schema_.indices_for(
                FeatureModality.CATEGORICAL
            )
            dataset_config_collection.append(
                data_util.ClassifierDatasetConfig(
                    config=ensemble_configs,
                    X_raw=X_mod,
                    y_raw=y_mod,
                    cat_ix=current_cat_ix,
                )
            )

        def faware_split_fn(X: Any, y: Any, stratify: Any = None) -> tuple[Any, Any, Any, Any]:
            global_row_ids = row_ids_by_x_id.get(id(X))
            if global_row_ids is None:
                stats.record(
                    fallback_reason=(
                        f"missing chunk row IDs for {FAWARE_C_SELECTION}"
                    ),
                    label_coverage_ok=False,
                )
                result = _aligned_split_with_stratify_fallback(
                    split_fn,
                    X,
                    y,
                    stratify,
                    fallback_label=f"{FAWARE_C_SELECTION} missing-row-IDs",
                )
                stats.record(
                    fallback_reason=result.fallback_reason,
                    label_coverage_ok=result.label_coverage_ok,
                )
                return result.X_context, result.X_query, result.y_context, result.y_query

            try:
                split = _global_test_centroid_reserve_indices(
                    y,
                    global_row_ids,
                    global_reservation,
                    split_fn,
                    stratify,
                )
            except Exception as exc:
                stats.record(
                    fallback_reason=(
                        f"{FAWARE_C_SELECTION} split failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    label_coverage_ok=False,
                )
                raise

            stats.record(
                split.reserved_scores,
                fallback_reason=split.fallback_reason,
                label_coverage_ok=split.label_coverage_ok,
            )
            return (
                _take_rows(X, split.ctx_idx),
                _take_rows(X, split.qry_idx),
                _take_rows(y, split.ctx_idx),
                _take_rows(y, split.qry_idx),
            )

        return data_util.DatasetCollectionWithPreprocessing(
            faware_split_fn,
            random_state=preprocessing_random_state,
            dataset_config_collection=dataset_config_collection,
            stratify=False if force_no_stratify else (model_type == "classifier"),
        )

    return get_preprocessed_dataset_chunks


def _split_summary(classifier: Any) -> dict[str, Any]:
    stats = getattr(classifier, "_faware_split_stats_", None)
    if stats is None:
        return {
            "ttt_c_f_distance_mean": None,
            "ttt_c_f_distance_std": None,
            "ttt_c_fallback_reason": None,
            "ttt_c_label_coverage_ok": True,
        }
    fallback_reason = None
    if stats.fallback_reasons:
        fallback_reason = "; ".join(stats.fallback_reasons)
    if int(stats.distance_count) <= 0:
        return {
            "ttt_c_f_distance_mean": None,
            "ttt_c_f_distance_std": None,
            "ttt_c_fallback_reason": fallback_reason,
            "ttt_c_label_coverage_ok": bool(stats.label_coverage_ok),
        }
    mean = float(stats.distance_sum / stats.distance_count)
    var = max(0.0, float(stats.distance_sumsq / stats.distance_count) - mean * mean)
    return {
        "ttt_c_f_distance_mean": mean,
        "ttt_c_f_distance_std": math.sqrt(var),
        "ttt_c_fallback_reason": fallback_reason,
        "ttt_c_label_coverage_ok": bool(stats.label_coverage_ok),
    }


def _clone_model_state_dict_cpu(model: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _interpolate_state_dicts(
    initial_state: dict[str, Any],
    tuned_state: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    blended_state: dict[str, Any] = {}
    alpha = float(alpha)
    for name, tuned_tensor in tuned_state.items():
        tuned_cpu = tuned_tensor.detach().cpu()
        initial_tensor = initial_state.get(name)
        if initial_tensor is None:
            blended_state[name] = tuned_cpu.clone()
            continue
        if tuned_cpu.is_floating_point() and initial_tensor.is_floating_point():
            initial_for_tuned = initial_tensor.to(dtype=tuned_cpu.dtype)
            blended_state[name] = initial_for_tuned.mul(1.0 - alpha).add(
                tuned_cpu,
                alpha=alpha,
            )
        else:
            blended_state[name] = tuned_cpu.clone()
    return blended_state


def _apply_wise_ft_weights(
    model: Any,
    initial_state: dict[str, Any],
    alpha: float,
) -> None:
    tuned_state = _clone_model_state_dict_cpu(model)
    model.load_state_dict(
        _interpolate_state_dicts(initial_state, tuned_state, alpha)
    )


class VersionedFinetunedTabPFNClassifier:
    @staticmethod
    def build(model_version: str, **kwargs: Any):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        from tabpfn.finetuning import FinetunedTabPFNClassifier

        faware_selection = str(kwargs.pop("faware_c_selection"))
        faware_metric = str(kwargs.pop("faware_c_metric"))
        faware_reserve_ratio = float(kwargs.pop("faware_c_reserve_ratio"))
        wise_ft_enabled = bool(kwargs.pop("wise_ft_enabled"))
        wise_alpha = float(kwargs.pop("wise_alpha"))
        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
            "v2.6": ModelVersion.V2_6,
            "v3": ModelVersion.V3,
        }
        version = version_map[model_version]

        class _VersionedFinetunedTabPFNClassifier(FinetunedTabPFNClassifier):
            def _configure_model_for_optimization(self, model: Any) -> None:
                super()._configure_model_for_optimization(model)
                self._ttt_wise_applied_ = False
                self._ttt_wise_initial_state_ = (
                    _clone_model_state_dict_cpu(model)
                    if wise_ft_enabled
                    else None
                )

            def _setup_inference_model(
                self,
                final_inference_eval_config: dict[str, Any],
            ) -> None:
                initial_state = getattr(
                    self,
                    "_ttt_wise_initial_state_",
                    None,
                )
                if wise_ft_enabled and initial_state is not None:
                    _apply_wise_ft_weights(
                        self.finetuned_estimator_.model_,
                        initial_state,
                        wise_alpha,
                    )
                    self._ttt_wise_applied_ = True
                super()._setup_inference_model(final_inference_eval_config)

            def set_faware_global_reservation(
                self,
                reservation: GlobalCentroidReservation,
            ) -> None:
                self._faware_global_reservation_ = reservation

            def _create_estimator(self, config: dict[str, Any]) -> TabPFNClassifier:
                if model_version == "v3":
                    return TabPFNClassifier(
                        **config,
                        fit_mode="batched",
                        differentiable_input=False,
                    )
                return TabPFNClassifier.create_default_for_version(
                    version,
                    **config,
                    fit_mode="batched",
                    differentiable_input=False,
                )

            def _evaluate_model(self, *args: Any, **kwargs: Any):
                result = super()._evaluate_model(*args, **kwargs)
                history = getattr(self, "_ttt_eval_history_", None)
                if history is None:
                    history = []
                    setattr(self, "_ttt_eval_history_", history)
                history.append(
                    {
                        "primary": _optional_float(result.primary),
                        "secondary": {
                            key: _optional_float(value)
                            for key, value in result.secondary.items()
                        },
                    }
                )
                return result

            def fit(
                self,
                X: Any,
                y: Any,
                X_val: Any | None = None,
                y_val: Any | None = None,
                output_dir: Path | None = None,
            ):
                from tabpfn.finetuning import finetuned_base

                global_reservation = getattr(
                    self,
                    "_faware_global_reservation_",
                    None,
                )
                self._faware_split_stats_ = FawareSplitStats()
                faware_values = {
                    "enabled": True,
                    "epochs": int(self.epochs),
                    "max_chunk_size": int(self.n_finetune_ctx_plus_query_samples),
                    "query_ratio": float(self.finetune_ctx_query_split_ratio),
                    "lr": float(self.learning_rate),
                    "weight_decay": float(self.weight_decay),
                    "grad_clip": self.grad_clip_value,
                    "patience": int(self.early_stopping_patience),
                    "min_delta": float(self.min_delta),
                    "eval_metric": str(self.eval_metric or "roc_auc"),
                    "validation_fraction": float(self.validation_split_ratio),
                    "n_estimators_finetune": int(self.n_estimators_finetune),
                    "n_estimators_validation": int(self.n_estimators_validation),
                    "n_estimators_final_inference": int(
                        self.n_estimators_final_inference
                    ),
                    "use_lr_scheduler": bool(self.use_lr_scheduler),
                    "lr_warmup_only": bool(self.lr_warmup_only),
                    "use_activation_checkpointing": bool(
                        self.use_activation_checkpointing
                    ),
                    "save_checkpoint_interval": self.save_checkpoint_interval,
                    "c_selection": faware_selection,
                    "c_metric": faware_metric,
                    "c_reserve_ratio": faware_reserve_ratio,
                }
                if hasattr(self, "gradient_accumulation_steps"):
                    faware_values["grad_accumulation_steps"] = int(
                        self.gradient_accumulation_steps
                    )
                faware_config = TTTConfig(**faware_values)
                original_get = finetuned_base.get_preprocessed_dataset_chunks
                finetuned_base.get_preprocessed_dataset_chunks = (
                    _make_faware_get_preprocessed_dataset_chunks(
                        global_reservation=global_reservation,
                        config=faware_config,
                        stats=self._faware_split_stats_,
                        original_get_preprocessed_dataset_chunks=original_get,
                    )
                )
                try:
                    return super().fit(
                        X,
                        y,
                        X_val=X_val,
                        y_val=y_val,
                        output_dir=output_dir,
                    )
                finally:
                    finetuned_base.get_preprocessed_dataset_chunks = original_get

        return _VersionedFinetunedTabPFNClassifier(**kwargs)


class TabPFNAdapter:
    def __init__(self, args: argparse.Namespace, device: str) -> None:
        self.args = args
        self.device = device
        self.ttt_config = build_ttt_config(args)

    def _classifier_kwargs(
        self,
        categorical_feature_indices: list[int],
        *,
        n_estimators_override: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "device": self.device,
            "n_estimators": (
                int(n_estimators_override)
                if n_estimators_override is not None
                else int(self.args.n_estimators)
            ),
            "ignore_pretraining_limits": self.args.ignore_pretraining_limits,
        }
        if categorical_feature_indices:
            kwargs["categorical_features_indices"] = list(categorical_feature_indices)
        return kwargs

    def _v3_model_path_for_loaded(self, loaded: LoadedDataset) -> str:
        task_type = str(loaded.task_type or "").lower()
        if task_type == "binclass":
            model_path = self.args.v3_binary_model_path
        elif task_type == "multiclass":
            model_path = self.args.v3_multiclass_model_path
        else:
            raise ValueError(
                f"Cannot choose a TabPFN v3 checkpoint for task_type={loaded.task_type!r}"
            )
        if self.args.verbose:
            print(
                f"[v3_model] {loaded.dataset_name}: task_type={task_type}, "
                f"model_path={model_path}",
                flush=True,
            )
        return model_path

    def _make_classifier(
        self,
        loaded: LoadedDataset,
        *,
        n_estimators_override: int | None = None,
    ):
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except Exception as exc:
            raise RuntimeError(f"Failed to import local TabPFN: {exc}") from exc

        classifier_kwargs = self._classifier_kwargs(
            loaded.categorical_feature_indices,
            n_estimators_override=n_estimators_override,
        )
        if self.args.model_path:
            return TabPFNClassifier(model_path=self.args.model_path, **classifier_kwargs)

        if self.args.model_version == "v3":
            return TabPFNClassifier(
                model_path=self._v3_model_path_for_loaded(loaded),
                **classifier_kwargs,
            )

        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
            "v2.6": ModelVersion.V2_6,
            "v3": ModelVersion.V3,
        }
        version = version_map[self.args.model_version]
        try:
            return TabPFNClassifier.create_default_for_version(
                version,
                **classifier_kwargs,
            )
        except TypeError:
            classifier = TabPFNClassifier.create_default_for_version(version)
            if hasattr(classifier, "set_params"):
                try:
                    classifier.set_params(
                        device=self.device,
                        n_estimators=classifier_kwargs["n_estimators"],
                        ignore_pretraining_limits=self.args.ignore_pretraining_limits,
                    )
                except Exception:
                    try:
                        classifier.set_params(
                            device=self.device,
                            n_estimators=classifier_kwargs["n_estimators"],
                        )
                    except Exception:
                        pass
            elif hasattr(classifier, "device"):
                classifier.device = self.device
            if hasattr(classifier, "n_estimators"):
                classifier.n_estimators = classifier_kwargs["n_estimators"]
            if hasattr(classifier, "ignore_pretraining_limits"):
                classifier.ignore_pretraining_limits = self.args.ignore_pretraining_limits
            if loaded.categorical_feature_indices and hasattr(
                classifier,
                "categorical_features_indices",
            ):
                classifier.categorical_features_indices = list(
                    loaded.categorical_feature_indices
                )
            return classifier

    def _should_use_many_class(self, n_classes: int) -> bool:
        mode = self.args.many_class
        if mode == "off":
            return False
        if mode == "on":
            return True
        return n_classes > TABPFN_CLASS_LIMIT

    def _wrap_many_class_if_needed(self, classifier, loaded: LoadedDataset):
        if not self._should_use_many_class(loaded.n_classes):
            return classifier
        try:
            ManyClassClassifier = import_many_class_classifier()
        except Exception as exc:
            raise RuntimeError(
                "Official TabPFN many-class inference requires tabpfn-extensions. "
                "Install it with: pip install git+https://github.com/PriorLabs/tabpfn-extensions.git. "
                f"Import failed with {type(exc).__name__}: {exc}"
            ) from exc

        wrapped = ManyClassClassifier(
            estimator=classifier,
            alphabet_size=self.args.many_class_alphabet_size,
            n_estimators=self.args.many_class_n_estimators,
            n_estimators_redundancy=self.args.many_class_redundancy,
            random_state=self.args.random_state,
            verbose=1 if self.args.verbose else 0,
        )
        if loaded.categorical_feature_indices and hasattr(
            wrapped,
            "set_categorical_features",
        ):
            wrapped.set_categorical_features(loaded.categorical_feature_indices)
        if self.args.verbose:
            print(
                f"[many_class] {loaded.dataset_name}: n_classes={loaded.n_classes}, "
                f"alphabet_size={self.args.many_class_alphabet_size}, "
                f"redundancy={self.args.many_class_redundancy}",
                flush=True,
            )
        return wrapped

    def _prediction_result_with_faware_defaults(
        self,
        result: PredictionResult,
        *,
        split_summary: dict[str, Any] | None = None,
    ) -> PredictionResult:
        config = self.ttt_config
        values = result.__dict__.copy()
        values.update(
            ttt_wise_ft=wise_ft_enabled_for_config(config),
            ttt_wise_alpha=(
                float(config.wise_alpha)
                if wise_ft_enabled_for_config(config)
                else None
            ),
            ttt_wise_applied=False,
            ttt_c_selection=config.c_selection if config.enabled else None,
            ttt_c_source=(
                faware_c_source_for_selection(config.c_selection)
                if config.enabled
                else None
            ),
            ttt_c_metric=ttt_c_metric_for_config(config) if config.enabled else None,
            ttt_c_reserve_ratio=(
                ttt_c_reserve_ratio_for_config(config) if config.enabled else None
            ),
            ttt_c_f_distance_mean=None,
            ttt_c_f_distance_std=None,
            ttt_c_fallback_reason=None,
            ttt_c_label_coverage_ok=True,
        )
        if split_summary:
            values.update(split_summary)
        return PredictionResult(**values)

    def _fit_predict_baseline(
        self,
        loaded: LoadedDataset,
        *,
        ttt_reason: str | None = None,
        ttt_oom_fallback: bool = False,
        ttt_fallback_reason: str | None = None,
    ) -> PredictionResult:
        classifier = self._wrap_many_class_if_needed(
            self._make_classifier(loaded),
            loaded,
        )
        X_train = loaded.X_train_merged.to_numpy()
        y_train = loaded.y_train_merged
        X_test = loaded.X_test.to_numpy()

        fit_started = time.time()
        classifier.fit(X_train, y_train)
        fit_seconds = time.time() - fit_started

        predict_started = time.time()
        y_pred, y_proba, classes = predict_from_proba_or_model(classifier, X_test)
        predict_seconds = time.time() - predict_started
        return PredictionResult(
            y_pred=y_pred,
            y_proba=y_proba,
            classes=classes,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            n_train_a=loaded.n_train_report,
            n_train_b=0,
            n_holdout_c=0,
            n_test_d=int(len(loaded.y_test)),
            ttt_lr=self.ttt_config.lr if self.ttt_config.enabled else None,
            ttt_wise_ft=wise_ft_enabled_for_config(self.ttt_config),
            ttt_wise_alpha=(
                float(self.ttt_config.wise_alpha)
                if wise_ft_enabled_for_config(self.ttt_config)
                else None
            ),
            ttt_wise_applied=False,
            ttt_applied=False,
            ttt_split_reason=ttt_reason,
            ttt_oom_fallback=ttt_oom_fallback,
            ttt_fallback_reason=ttt_fallback_reason,
            ttt_c_selection=(
                self.ttt_config.c_selection if self.ttt_config.enabled else None
            ),
            ttt_c_source=(
                faware_c_source_for_selection(self.ttt_config.c_selection)
                if self.ttt_config.enabled
                else None
            ),
            ttt_c_metric=(
                ttt_c_metric_for_config(self.ttt_config)
                if self.ttt_config.enabled
                else None
            ),
            ttt_c_reserve_ratio=(
                ttt_c_reserve_ratio_for_config(self.ttt_config)
                if self.ttt_config.enabled
                else None
            ),
        )

    def _make_finetuned_classifier(
        self,
        loaded: LoadedDataset,
        *,
        apply_wise_ft: bool,
    ):
        config = self.ttt_config
        classifier_kwargs = self._classifier_kwargs(
            loaded.categorical_feature_indices,
            n_estimators_override=config.n_estimators_final_inference,
        )
        if self.args.model_path:
            classifier_kwargs["model_path"] = self.args.model_path
        elif self.args.model_version == "v3":
            classifier_kwargs["model_path"] = self._v3_model_path_for_loaded(loaded)

        from tabpfn.finetuning import FinetunedTabPFNClassifier

        build_kwargs: dict[str, Any] = {
            "device": self.device,
            "epochs": config.epochs,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "validation_split_ratio": config.validation_fraction,
            "n_finetune_ctx_plus_query_samples": config.max_chunk_size,
            "finetune_ctx_query_split_ratio": config.query_ratio,
            "random_state": int(self.args.random_state),
            "early_stopping": config.patience > 0,
            "early_stopping_patience": max(1, config.patience),
            "min_delta": config.min_delta,
            "grad_clip_value": config.grad_clip,
            "use_lr_scheduler": config.use_lr_scheduler,
            "lr_warmup_only": config.lr_warmup_only,
            "n_estimators_finetune": config.n_estimators_finetune,
            "n_estimators_validation": config.n_estimators_validation,
            "n_estimators_final_inference": config.n_estimators_final_inference,
            "use_activation_checkpointing": config.use_activation_checkpointing,
            "save_checkpoint_interval": config.save_checkpoint_interval,
            "extra_classifier_kwargs": classifier_kwargs,
            "eval_metric": config.eval_metric,
            "faware_c_selection": config.c_selection,
            "faware_c_metric": config.c_metric,
            "faware_c_reserve_ratio": config.c_reserve_ratio,
            "wise_ft_enabled": bool(apply_wise_ft),
            "wise_alpha": float(config.wise_alpha),
        }
        init_params = inspect.signature(FinetunedTabPFNClassifier.__init__).parameters
        if "gradient_accumulation_steps" in init_params:
            build_kwargs["gradient_accumulation_steps"] = _grad_accumulation_steps(
                config
            )
        return VersionedFinetunedTabPFNClassifier.build(
            self.args.model_version,
            **build_kwargs,
        )

    def _fit_predict_with_ttt(self, loaded: LoadedDataset) -> PredictionResult:
        config = self.ttt_config
        if not config.enabled or config.epochs < 1:
            return self._fit_predict_baseline(loaded)
        if self._should_use_many_class(loaded.n_classes):
            return self._fit_predict_baseline(
                loaded,
                ttt_reason="TTT skipped because n_classes > 10 requires ManyClass baseline inference",
            )

        X_train = loaded.X_train.to_numpy()
        y_train = np.asarray(loaded.y_train)
        X_val = loaded.X_val.to_numpy() if loaded.X_val is not None else None
        y_val = np.asarray(loaded.y_val) if loaded.y_val is not None else None
        X_test = loaded.X_test.to_numpy()
        dataset_total_rows = int(loaded.n_train_report + len(loaded.y_test))
        wise_ft_requested = wise_ft_enabled_for_config(config)
        apply_wise_ft = wise_ft_enabled_for_dataset(config, dataset_total_rows)
        chunks_per_epoch = max(
            1,
            (int(len(y_train)) + int(config.max_chunk_size) - 1)
            // int(config.max_chunk_size),
        )

        fit_started = time.time()
        classifier = self._make_finetuned_classifier(
            loaded,
            apply_wise_ft=apply_wise_ft,
        )
        if config.c_selection == FAWARE_C_SELECTION:
            global_reservation = _build_global_test_centroid_reservation(
                loaded.X_train,
                loaded.X_test,
                categorical_feature_indices=loaded.categorical_feature_indices,
                metric=config.c_metric,
                reserve_ratio=config.c_reserve_ratio,
            )
            classifier.set_faware_global_reservation(global_reservation)
        ttt_update_started = time.time()
        try:
            classifier.fit(X_train, y_train, X_val=X_val, y_val=y_val, output_dir=None)
        except Exception as exc:
            if not is_oom_exception(exc):
                raise
            fallback_reason = (
                "TTT OOM; used original model parameters for inference: "
                f"{format_exception_for_csv(exc)}"
            )
            return self._fit_predict_baseline(
                loaded,
                ttt_reason=fallback_reason,
                ttt_oom_fallback=True,
                ttt_fallback_reason=fallback_reason,
            )

        ttt_update_seconds = time.time() - ttt_update_started
        fit_seconds = time.time() - fit_started
        ttt_validation_summary = extract_ttt_validation_summary(
            classifier,
            eval_metric=config.eval_metric,
            epochs=config.epochs,
        )
        split_summary = _split_summary(classifier)
        wise_ft_applied = bool(
            getattr(classifier, "_ttt_wise_applied_", False)
        )
        ttt_reason = (
            (
                "dataset_val_split"
                if X_val is not None
                else f"internal validation_split_ratio={config.validation_fraction}"
            )
            + ttt_c_reason_suffix(config)
        )
        if wise_ft_requested:
            if wise_ft_applied:
                ttt_reason += (
                    f" | wise_ft=alpha{config.wise_alpha:g}"
                    f" rows={dataset_total_rows}<{WISE_FT_MAX_DATASET_ROWS}"
                )
            else:
                ttt_reason += (
                    f" | wise_ft_skipped rows={dataset_total_rows}"
                    f">={WISE_FT_MAX_DATASET_ROWS}"
                )

        predict_started = time.time()
        y_pred, y_proba, classes = predict_from_proba_or_model(classifier, X_test)
        predict_seconds = time.time() - predict_started
        return PredictionResult(
            y_pred=y_pred,
            y_proba=y_proba,
            classes=classes,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            n_train_a=loaded.n_train_report,
            n_train_b=int(len(y_train)),
            n_holdout_c=0,
            n_test_d=int(len(loaded.y_test)),
            ttt_lr=config.lr,
            ttt_steps=config.epochs
            * (
                (chunks_per_epoch + _grad_accumulation_steps(config) - 1)
                // _grad_accumulation_steps(config)
            ),
            ttt_applied=True,
            ttt_wise_ft=wise_ft_requested,
            ttt_wise_alpha=(
                float(config.wise_alpha) if wise_ft_requested else None
            ),
            ttt_wise_applied=wise_ft_applied,
            ttt_update_seconds=float(ttt_update_seconds),
            ttt_split_strategy=f"tabpfn_finetune_epoch_chunks_{config.c_selection}",
            ttt_split_reason=ttt_reason,
            ttt_epochs=config.epochs,
            ttt_chunks_per_epoch=chunks_per_epoch,
            ttt_batch_mode=ttt_c_batch_mode(config),
            ttt_val_eval_metric=ttt_validation_summary["ttt_val_eval_metric"],
            ttt_val_baseline_metric=ttt_validation_summary["ttt_val_baseline_metric"],
            ttt_val_best_metric=ttt_validation_summary["ttt_val_best_metric"],
            ttt_val_baseline_accuracy=ttt_validation_summary["ttt_val_baseline_accuracy"],
            ttt_val_best_accuracy=ttt_validation_summary["ttt_val_best_accuracy"],
            ttt_best_epoch=ttt_validation_summary["ttt_best_epoch"],
            ttt_stopped_early=ttt_validation_summary["ttt_stopped_early"],
            ttt_c_selection=config.c_selection,
            ttt_c_source=faware_c_source_for_selection(config.c_selection),
            ttt_c_metric=ttt_c_metric_for_config(config),
            ttt_c_reserve_ratio=ttt_c_reserve_ratio_for_config(config),
            ttt_c_f_distance_mean=split_summary["ttt_c_f_distance_mean"],
            ttt_c_f_distance_std=split_summary["ttt_c_f_distance_std"],
            ttt_c_fallback_reason=split_summary["ttt_c_fallback_reason"],
            ttt_c_label_coverage_ok=split_summary["ttt_c_label_coverage_ok"],
        )

    def fit_predict(self, loaded: LoadedDataset) -> PredictionResult:
        return self._fit_predict_with_ttt(loaded)

    def fit_predict_with_forced_ignore_limits(
        self,
        loaded: LoadedDataset,
    ) -> PredictionResult:
        original_flag = self.args.ignore_pretraining_limits
        self.args.ignore_pretraining_limits = True
        try:
            return self.fit_predict(loaded)
        finally:
            self.args.ignore_pretraining_limits = original_flag


def resolve_script_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    script_path = SCRIPT_DIR / path
    repo_path = SCRIPT_DIR.parent.parent / path

    first_part = path.parts[0] if path.parts else ""
    if first_part and first_part not in {".", ".."}:
        if (Path.cwd() / first_part).exists():
            return cwd_path.resolve()
        if (SCRIPT_DIR / first_part).exists():
            return script_path.resolve()
        if (SCRIPT_DIR.parent.parent / first_part).exists():
            return repo_path.resolve()

    if cwd_path.exists():
        return cwd_path.resolve()
    if script_path.exists():
        return script_path.resolve()
    if repo_path.exists():
        return repo_path.resolve()
    return script_path.resolve()


def tabpfn_cache_dir_from_env() -> Path:
    cache_dir = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "tabpfn"
    return Path.home() / ".cache" / "tabpfn"


def is_auto_model_path_value(value: str | Path | None) -> bool:
    if value is None:
        return True
    raw_value = str(value).strip()
    return not raw_value or raw_value.lower() in {"auto", "none", "null"}


def resolve_auto_model_file(file_name: str) -> str:
    cache_dir = tabpfn_cache_dir_from_env()
    for candidate in (
        Path.cwd() / file_name,
        SCRIPT_DIR / file_name,
        SCRIPT_DIR.parent.parent / file_name,
        cache_dir / file_name,
    ):
        if candidate.exists():
            return candidate.resolve().as_posix()
    return (cache_dir / file_name).expanduser().as_posix()


def resolve_v3_task_model_path(
    value: str | Path | None,
    default_file_name: str,
) -> str:
    if is_auto_model_path_value(value):
        return resolve_auto_model_file(default_file_name)
    resolved = resolve_model_path(value, model_version=None)
    if resolved is None:
        return resolve_auto_model_file(default_file_name)
    return resolved


def resolve_model_path(
    value: str | Path | None,
    model_version: str | None = None,
) -> str | None:
    if is_auto_model_path_value(value):
        if model_version:
            preferred_file = GATED_CLASSIFIER_CACHE_FILES.get(model_version)
            if preferred_file:
                for candidate in (
                    Path.cwd() / preferred_file,
                    SCRIPT_DIR / preferred_file,
                    SCRIPT_DIR.parent.parent / preferred_file,
                    tabpfn_cache_dir_from_env() / preferred_file,
                ):
                    if candidate.exists():
                        return candidate.resolve().as_posix()
        return None

    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.as_posix()

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve().as_posix()

    script_path = SCRIPT_DIR / path
    if script_path.exists():
        return script_path.resolve().as_posix()

    repo_path = SCRIPT_DIR.parent.parent / path
    if repo_path.exists():
        return repo_path.resolve().as_posix()

    return script_path.resolve().as_posix()


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_dataset_dirs(data_root: Path) -> list[Path]:
    return [path for path in sorted(data_root.iterdir()) if path.is_dir()]


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    string_series = series.astype("string")
    string_series = string_series.fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def make_feature_frame(values, *, kind: str, prefix: str) -> pd.DataFrame:
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
    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_array(file_path: Path) -> np.ndarray:
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


def stable_feature_prefix(context: str, fallback: str) -> str:
    stem = Path(context or fallback).stem
    for suffix in ("-train", "-test", "-val", "-single"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or fallback


def load_split(
    num_path: Path | None,
    cat_path: Path | None,
    y_path: Path,
    *,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    features: list[pd.DataFrame] = []
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


def load_classification_dataset(dataset_dir: Path) -> LoadedDataset:
    info = load_dataset_info(dataset_dir)
    task_type = str(info.get("task_type", "")).lower() if info else None
    if task_type not in CLASSIFICATION_TASKS:
        raise SkipDataset(f"Skipped due to task_type={task_type!r}")

    train_split, val_split, test_split = find_split_files(dataset_dir)
    categorical_feature_indices = categorical_indices_from_split(train_split)
    X_train, y_train = load_split(
        train_split[0],
        train_split[1],
        train_split[2],
        context=f"{dataset_dir.name}-train",
    )

    X_val = None
    y_val = None
    if val_split is not None:
        X_val, y_val = load_split(
            val_split[0],
            val_split[1],
            val_split[2],
            context=f"{dataset_dir.name}-val",
        )

    X_test, y_test = load_split(
        test_split[0],
        test_split[1],
        test_split[2],
        context=f"{dataset_dir.name}-test",
    )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]}, test has {X_test.shape[1]}"
        )
    if X_val is not None and X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]}, val has {X_val.shape[1]}"
        )

    return LoadedDataset(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        task_type=task_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        categorical_feature_indices=categorical_feature_indices,
    )


def empty_row_for_dataset(
    dataset_dir: Path,
    status: str,
    error: str,
    *,
    task_type: Optional[str] = None,
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
        f1=None,
        balanced_accuracy=None,
        roc_auc=None,
        log_loss=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status=status,
        error=error,
    )


def evaluate_one_dataset(adapter: TabPFNAdapter, dataset_dir: Path) -> ResultRow:
    task_type: Optional[str] = None
    try:
        loaded = load_classification_dataset(dataset_dir)
        task_type = loaded.task_type
        if loaded.n_classes > TABPFN_CLASS_LIMIT:
            return ResultRow(
                dataset_name=loaded.dataset_name,
                dataset_dir=loaded.dataset_dir.as_posix(),
                task_type=loaded.task_type,
                n_train=loaded.n_train_report,
                n_val=loaded.n_val,
                n_test=int(len(loaded.y_test)),
                n_features=int(loaded.X_train.shape[1]),
                n_classes=loaded.n_classes,
                accuracy=None,
                f1=None,
                balanced_accuracy=None,
                roc_auc=None,
                log_loss=None,
                fit_seconds=0.0,
                predict_seconds=0.0,
                status="skip",
                error=(
                    f"Skipped because n_classes={loaded.n_classes} exceeds "
                    f"TabPFN class limit {TABPFN_CLASS_LIMIT}; "
                    "ManyClassClassifier is not run in this benchmark."
                ),
                ttt_c_selection=adapter.ttt_config.c_selection,
                ttt_c_source=faware_c_source_for_selection(
                    adapter.ttt_config.c_selection
                ),
                ttt_c_metric=ttt_c_metric_for_config(adapter.ttt_config),
                ttt_c_reserve_ratio=ttt_c_reserve_ratio_for_config(
                    adapter.ttt_config
                ),
                ttt_wise_ft=wise_ft_enabled_for_config(adapter.ttt_config),
                ttt_wise_alpha=(
                    float(adapter.ttt_config.wise_alpha)
                    if wise_ft_enabled_for_config(adapter.ttt_config)
                    else None
                ),
            )
        try:
            result = adapter.fit_predict(loaded)
        except Exception as exc:
            if is_pretraining_limit_error(exc):
                result = adapter.fit_predict_with_forced_ignore_limits(loaded)
            else:
                raise
        y_pred = np.asarray(result.y_pred)
        y_test = np.asarray(loaded.y_test)
        if len(y_pred) != len(y_test):
            raise ValueError(f"Prediction length mismatch: got {len(y_pred)}, expected {len(y_test)}")
        accuracy = float(np.mean(y_pred == y_test))
        f1 = compute_weighted_f1(y_test, y_pred)
        balanced_accuracy = compute_balanced_accuracy(y_test, y_pred)
        roc_auc = compute_roc_auc(y_test, result.y_proba, result.classes)
        log_loss_score = compute_log_loss(y_test, result.y_proba, result.classes)
        return ResultRow(
            dataset_name=loaded.dataset_name,
            dataset_dir=loaded.dataset_dir.as_posix(),
            task_type=loaded.task_type,
            n_train=loaded.n_train_report,
            n_val=loaded.n_val,
            n_test=int(len(y_test)),
            n_features=int(loaded.X_train.shape[1]),
            n_classes=loaded.n_classes,
            accuracy=accuracy,
            f1=f1,
            balanced_accuracy=balanced_accuracy,
            roc_auc=roc_auc,
            log_loss=log_loss_score,
            fit_seconds=float(result.fit_seconds),
            predict_seconds=float(result.predict_seconds),
            status="ok",
            error=None,
            n_train_a=result.n_train_a,
            n_train_b=result.n_train_b,
            n_holdout_c=result.n_holdout_c,
            n_test_d=result.n_test_d,
            ttt_loss=result.ttt_loss,
            ttt_wise_ft=result.ttt_wise_ft,
            ttt_wise_alpha=result.ttt_wise_alpha,
            ttt_wise_applied=result.ttt_wise_applied,
            ttt_steps=result.ttt_steps,
            ttt_lr=result.ttt_lr,
            ttt_applied=result.ttt_applied,
            ttt_update_seconds=result.ttt_update_seconds,
            ttt_split_strategy=result.ttt_split_strategy,
            ttt_split_reason=result.ttt_split_reason,
            ttt_epochs=result.ttt_epochs,
            ttt_chunks_per_epoch=result.ttt_chunks_per_epoch,
            ttt_batch_mode=result.ttt_batch_mode,
            ttt_val_eval_metric=result.ttt_val_eval_metric,
            ttt_val_baseline_metric=result.ttt_val_baseline_metric,
            ttt_val_best_metric=result.ttt_val_best_metric,
            ttt_val_baseline_accuracy=result.ttt_val_baseline_accuracy,
            ttt_val_best_accuracy=result.ttt_val_best_accuracy,
            ttt_best_epoch=result.ttt_best_epoch,
            ttt_stopped_early=result.ttt_stopped_early,
            ttt_oom_fallback=result.ttt_oom_fallback,
            ttt_fallback_reason=result.ttt_fallback_reason,
            ttt_c_selection=result.ttt_c_selection,
            ttt_c_source=result.ttt_c_source,
            ttt_c_metric=result.ttt_c_metric,
            ttt_c_reserve_ratio=result.ttt_c_reserve_ratio,
            ttt_c_f_distance_mean=result.ttt_c_f_distance_mean,
            ttt_c_f_distance_std=result.ttt_c_f_distance_std,
            ttt_c_fallback_reason=result.ttt_c_fallback_reason,
            ttt_c_label_coverage_ok=result.ttt_c_label_coverage_ok,
        )
    except SkipDataset as exc:
        return empty_row_for_dataset(dataset_dir, "skip", str(exc), task_type=task_type)
    except Exception as exc:
        return empty_row_for_dataset(
            dataset_dir,
            "fail",
            f"{type(exc).__name__}: {exc}",
            task_type=task_type,
        )


def rows_to_frame(rows: Iterable[ResultRow]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[RESULT_COLUMNS]


def ensure_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    normalized = frame.copy()
    for column in RESULT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[RESULT_COLUMNS]


def is_oom_error_message(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in OOM_ERROR_MARKERS)


def is_pretraining_limit_error(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return (
        "ignore_pretraining_limits" in text
        or "officially supported by tabpfn" in text
        or "pre-training range" in text
    )


def load_failed_dataset_names_from_results_csv(
    results_csv: Path,
    *,
    include_oom_failures: bool,
) -> list[str]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Retry results CSV does not exist: {results_csv}")

    frame = pd.read_csv(results_csv)
    required_columns = {"dataset_name", "status"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Retry results CSV is missing required columns: {', '.join(missing_columns)}"
        )

    failed_df = frame[frame["status"].astype(str).str.lower() == "fail"].copy()
    if not include_oom_failures and "error" in failed_df.columns:
        failed_df = failed_df[~failed_df["error"].map(is_oom_error_message)]

    dataset_names: list[str] = []
    seen_names: set[str] = set()
    for _, row in failed_df.iterrows():
        raw_name = str(row.get("dataset_name", "") or "").strip()
        if not raw_name and "dataset_dir" in failed_df.columns:
            raw_name = Path(str(row.get("dataset_dir", "") or "")).name.strip()
        if not raw_name or raw_name in seen_names:
            continue
        dataset_names.append(raw_name)
        seen_names.add(raw_name)
    return dataset_names


def filter_dataset_dirs_by_name(
    dataset_dirs: list[Path],
    dataset_names: Iterable[str],
) -> tuple[list[Path], list[str]]:
    wanted_names = {str(name).strip() for name in dataset_names if str(name).strip()}
    filtered = [path for path in dataset_dirs if path.name in wanted_names]
    found_names = {path.name for path in filtered}
    missing_names = [name for name in dataset_names if name not in found_names]
    return filtered, missing_names


def merge_result_frames(base_df: pd.DataFrame, updated_df: pd.DataFrame) -> pd.DataFrame:
    base_df = ensure_result_columns(base_df)
    updated_df = ensure_result_columns(updated_df)
    if base_df.empty:
        return updated_df
    if updated_df.empty:
        return base_df

    base_df = base_df.copy()
    updated_df = updated_df.copy()
    order_map = {
        str(dataset_name): idx
        for idx, dataset_name in enumerate(base_df["dataset_name"].astype(str).tolist())
    }
    replacement_names = set(updated_df["dataset_name"].astype(str).tolist())
    base_keep = base_df[~base_df["dataset_name"].astype(str).isin(replacement_names)].copy()

    base_keep["_merge_order"] = base_keep["dataset_name"].astype(str).map(order_map)
    appended_start = len(order_map)
    updated_df["_merge_order"] = [
        order_map.get(str(dataset_name), appended_start + idx)
        for idx, dataset_name in enumerate(updated_df["dataset_name"].tolist())
    ]

    merged = pd.concat([base_keep, updated_df], ignore_index=True)
    merged = merged.sort_values("_merge_order", kind="stable").drop(columns="_merge_order")
    merged = merged.reset_index(drop=True)
    return merged[RESULT_COLUMNS]


def write_summary(summary_path: Path, result_df: pd.DataFrame, dataset_dirs: list[Path], wall_seconds: float) -> None:
    result_df = result_df.copy()
    for metric_column in ("accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"):
        if metric_column in result_df.columns:
            result_df[metric_column] = pd.to_numeric(result_df[metric_column], errors="coerce")
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    oom_fallback_df = (
        ok_df[ok_df["ttt_oom_fallback"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
        if len(ok_df) and "ttt_oom_fallback" in ok_df.columns
        else pd.DataFrame()
    )
    wise_applied_df = (
        ok_df[ok_df["ttt_wise_applied"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
        if len(ok_df) and "ttt_wise_applied" in ok_df.columns
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
        f"ttt_wise_applied_count: {len(wise_applied_df)}",
        mean_line("avg_accuracy_ok", "accuracy"),
        mean_line("avg_f1_ok", "f1"),
        mean_line("avg_balanced_accuracy_ok", "balanced_accuracy"),
        mean_line("avg_roc_auc_ok", "roc_auc"),
        mean_line("avg_log_loss_ok", "log_loss"),
        f"wall_seconds: {wall_seconds:.3f}",
        "failed_datasets: "
        + (", ".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "(none)"),
        "skipped_datasets: "
        + (", ".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "(none)"),
        "ttt_oom_fallback_datasets: "
        + (", ".join(oom_fallback_df["dataset_name"].astype(str).tolist()) if len(oom_fallback_df) else "(none)"),
        "ttt_wise_applied_datasets: "
        + (", ".join(wise_applied_df["dataset_name"].astype(str).tolist()) if len(wise_applied_df) else "(none)"),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_gpu_id_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def detect_env_gpu_ids() -> list[int]:
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
    return []


def detect_nvidia_gpu_ids() -> list[int]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    gpu_ids: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            gpu_ids.append(int(line.split(",")[0].strip()))
        except ValueError:
            continue
    return gpu_ids


def detect_torch_gpu_ids() -> list[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(int(torch.cuda.device_count())))
    except Exception:
        pass
    return []


def default_tabpfn_cache_dir() -> Path:
    return tabpfn_cache_dir_from_env()


def has_tabpfn_auth_token() -> bool:
    if os.environ.get("TABPFN_TOKEN") or os.environ.get("HF_TOKEN"):
        return True
    return any(
        path.expanduser().exists()
        for path in (
            default_tabpfn_cache_dir() / "auth_token",
            Path.home() / ".tabpfn" / "token",
        )
    )


def task_types_for_dataset_dirs(dataset_dirs: Iterable[Path]) -> set[str]:
    task_types: set[str] = set()
    for dataset_dir in dataset_dirs:
        info = load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else ""
        if task_type in CLASSIFICATION_TASKS:
            task_types.add(task_type)
    return task_types


def validate_checkpoint_or_auth(label: str, model_path: str) -> None:
    path = Path(model_path).expanduser()
    if path.exists() or has_tabpfn_auth_token():
        return
    raise RuntimeError(
        f"{label} weights are not available at {path} and no TabPFN/Hugging Face "
        "auth token was found. Pass an existing checkpoint path or export "
        "TABPFN_TOKEN after accepting the PriorLabs license."
    )


def validate_model_access(
    args: argparse.Namespace,
    dataset_dirs: Iterable[Path],
) -> None:
    if args.model_path:
        validate_checkpoint_or_auth("Explicit TabPFN model", args.model_path)
        return

    if args.model_version == "v3":
        task_types = task_types_for_dataset_dirs(dataset_dirs)
        if "binclass" in task_types:
            validate_checkpoint_or_auth(
                "TabPFN v3 binary classifier",
                args.v3_binary_model_path,
            )
        if "multiclass" in task_types:
            validate_checkpoint_or_auth(
                "TabPFN v3 multiclass classifier",
                args.v3_multiclass_model_path,
            )
        return

    cache_file_name = GATED_CLASSIFIER_CACHE_FILES.get(args.model_version)
    if cache_file_name is None:
        return

    cache_path = default_tabpfn_cache_dir() / cache_file_name
    if cache_path.exists() or has_tabpfn_auth_token():
        return

    raise RuntimeError(
        f"TabPFN {args.model_version} classifier weights are not cached at {cache_path} "
        "and no TabPFN/Hugging Face auth token was found. Use an older --model-version "
        "if that cache exists, pass --model-path to a local checkpoint, or export "
        "TABPFN_TOKEN after accepting the PriorLabs license."
    )


def resolve_workers_and_gpu_ids(args: argparse.Namespace) -> tuple[int, list[int]]:
    if args.gpus is None or str(args.gpus).strip().lower() == "auto":
        gpu_ids = detect_env_gpu_ids() or detect_nvidia_gpu_ids() or detect_torch_gpu_ids()
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
        raise ValueError(
            f"--gpus must contain exactly --workers ids; got {len(gpu_ids)} ids for {workers} workers"
        )
    return workers, gpu_ids


def use_rocm_gpu_visibility() -> bool:
    backend = (os.environ.get("TFM_GPU_BACKEND") or os.environ.get("TABICL_GPU_BACKEND") or "").strip().lower()
    return backend in {"rocm", "hip", "amd"}


def bind_worker_gpu(gpu_id: int) -> None:
    gpu_id_str = str(gpu_id)
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


def worker_main(
    worker_id: int,
    gpu_id: int,
    dataset_dirs: list[str],
    worker_csv: str,
    args_dict: dict[str, Any],
) -> None:
    bind_worker_gpu(gpu_id)
    args = argparse.Namespace(**args_dict)
    rows: list[ResultRow] = []
    try:
        adapter = TabPFNAdapter(args, device="cuda:0")
        for dataset_dir in dataset_dirs:
            row = evaluate_one_dataset(adapter, Path(dataset_dir))
            rows.append(row)
            if args.verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [ok] "
                        f"{row.dataset_name} "
                        f"accuracy={format_optional_float(row.accuracy)} "
                        f"f1={format_optional_float(row.f1)} "
                        f"balanced_accuracy={format_optional_float(row.balanced_accuracy)} "
                        f"roc_auc={format_optional_float(row.roc_auc)} "
                        f"log_loss={format_optional_float(row.log_loss)} "
                        f"wise_ft={row.ttt_wise_ft} "
                        f"wise_alpha={format_optional_float(row.ttt_wise_alpha)} "
                        f"wise_applied={row.ttt_wise_applied}",
                        flush=True,
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [{row.status}] "
                        f"{row.dataset_name} error={row.error}",
                        flush=True,
                    )
    except Exception:
        rows.append(
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
    rows_to_frame(rows).to_csv(worker_csv, index=False)


def run_benchmark(args: argparse.Namespace) -> None:
    data_root = resolve_script_path(args.data_root)
    raw_model_path_arg = args.model_path
    if args.model_version == "v3" and is_auto_model_path_value(args.model_path):
        args.model_path = None
    else:
        args.model_path = resolve_model_path(args.model_path, args.model_version)
    args.v3_binary_model_path = resolve_v3_task_model_path(
        args.v3_binary_model_path,
        V3_BINARY_CLASSIFIER_FILE,
    )
    args.v3_multiclass_model_path = resolve_v3_task_model_path(
        args.v3_multiclass_model_path,
        V3_MULTICLASS_CLASSIFIER_FILE,
    )
    args.ttt_enabled = bool(args.ttt)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    all_dataset_dirs = find_dataset_dirs(data_root)
    dataset_dirs = list(all_dataset_dirs)
    retry_results_csv: Path | None = None
    merge_results_csv: Path | None = None

    if args.retry_failed_datasets_only:
        retry_results_csv = resolve_script_path(args.reference_results_csv)
        failed_dataset_names = load_failed_dataset_names_from_results_csv(
            retry_results_csv,
            include_oom_failures=args.retry_include_oom_failures,
        )
        dataset_dirs, missing_dataset_names = filter_dataset_dirs_by_name(
            dataset_dirs,
            failed_dataset_names,
        )
        if missing_dataset_names:
            print(
                "warning: retry CSV referenced datasets missing under data_root: "
                + ", ".join(missing_dataset_names),
                file=sys.stderr,
                flush=True,
            )
        if args.verbose:
            print(
                f"retry_failed_datasets_only: selected {len(dataset_dirs)} datasets "
                f"from {retry_results_csv}",
                flush=True,
            )

    if args.merge_results_from_csv:
        merge_results_csv = resolve_script_path(args.merge_results_from_csv)
    elif retry_results_csv is not None:
        merge_results_csv = retry_results_csv

    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        if retry_results_csv is not None:
            raise FileNotFoundError(
                f"No retry target datasets found under {data_root} for {retry_results_csv}"
            )
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    if args.out_dir is None:
        checkpoint = checkpoint_stem(raw_model_path_arg)
        model_label = f"tabpfn-{checkpoint}" if checkpoint else f"tabpfn{args.model_version}"
        naming_dataset_dirs = all_dataset_dirs if merge_results_csv is not None else dataset_dirs
        ttt_label_value = ttt_label(args)
        if bool(args.ttt):
            c_selection_label = str(args.ttt_c_selection)
            ttt_label_value = f"{ttt_label_value}_csel{c_selection_label}"
            if c_selection_label == FAWARE_C_SELECTION:
                reserve_label = str(args.ttt_c_reserve_ratio).replace(".", "p")
                wise_alpha_label = str(args.ttt_wise_alpha).replace(".", "p")
                ttt_label_value = (
                    f"{ttt_label_value}_{args.ttt_c_metric}_reserve{reserve_label}"
                    f"_wise{int(bool(args.ttt_wise_ft))}"
                    f"_alpha{wise_alpha_label}"
                )
        out_dir = auto_out_dir(
            model_label=model_label,
            data_root=data_root,
            dataset_dirs=naming_dataset_dirs,
            infer_label=infer_estimator_label(args),
            ttt_label=ttt_label_value,
            seed_label=seed_label(args),
        )
    else:
        out_dir = resolve_script_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}", flush=True)

    args.data_root = str(data_root)
    args.out_dir = str(out_dir)
    args.workers, gpu_ids = resolve_workers_and_gpu_ids(args)
    if args.verbose:
        print(f"resolved_workers: {args.workers}", flush=True)
        print(f"resolved_gpus: {','.join(str(gpu_id) for gpu_id in gpu_ids)}", flush=True)
        print(
            f"ignore_pretraining_limits: {args.ignore_pretraining_limits}",
            flush=True,
        )
        if merge_results_csv is not None:
            print(f"merge_results_from_csv: {merge_results_csv}", flush=True)

    validate_model_access(args, dataset_dirs)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    started = time.time()
    worker_csvs: list[Path] = []
    processes: list[mp.Process] = []
    args_dict = vars(args).copy()

    for worker_id in range(args.workers):
        assigned = [str(path.resolve()) for path in dataset_dirs[worker_id :: args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csvs.append(worker_csv)
        proc = mp.Process(
            target=worker_main,
            args=(worker_id, gpu_ids[worker_id], assigned, str(worker_csv), args_dict),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    frames = [pd.read_csv(path) for path in worker_csvs if path.exists()]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
    result_df = ensure_result_columns(result_df)
    summary_dataset_dirs = dataset_dirs
    if merge_results_csv is not None:
        base_df = ensure_result_columns(pd.read_csv(merge_results_csv))
        result_df = merge_result_frames(base_df, result_df)
        summary_dataset_dirs = all_dataset_dirs
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    result_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, result_df, summary_dataset_dirs, time.time() - started)
    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabPFN 1C chunk-style finetuning TTT with configurable "
            "F-aware context/query splitting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULTS.data_root,
        help="Root directory containing data178-style dataset folders.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULTS.out_dir,
        help=(
            "Directory for worker CSVs, all_classification_results.csv, and summary.txt. "
            "If omitted, uses baseline_compare/results/<auto_name>."
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULTS.workers)
    parser.add_argument(
        "--gpus",
        default=DEFAULTS.gpus,
        help="Comma-separated physical GPU ids, or 'auto' to use detected GPUs.",
    )
    parser.add_argument("--max-datasets", type=int, default=DEFAULTS.max_datasets)
    parser.add_argument("--random-state", type=int, default=DEFAULTS.random_state)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULTS.n_estimators,
        help="Number of TabPFN ensemble estimators used by the base inference estimator.",
    )
    parser.add_argument("--verbose", action="store_true", default=DEFAULTS.verbose)
    parser.add_argument(
        "--model-version",
        choices=["v2", "v2.5", "v2.6", "v3"],
        default=DEFAULTS.model_version,
        help="TabPFN model version to use when --model-path is not provided.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULTS.model_path,
        help=(
            "Path to a local TabPFN checkpoint. Relative paths are resolved from the "
            "current working directory if they exist, otherwise from this script's "
            "directory. If omitted, the script first tries a version-matched local "
            "checkpoint filename, then falls back to the --model-version cache/token flow. "
            "Use 'auto' or 'none' to skip local checkpoint discovery explicitly."
        ),
    )
    parser.add_argument(
        "--v3-binary-model-path",
        default=DEFAULTS.v3_binary_model_path,
        help=(
            "TabPFN v3 checkpoint used automatically for data178 binclass tasks "
            f"when --model-version v3 and --model-path is omitted. 'auto' searches "
            f"for {V3_BINARY_CLASSIFIER_FILE} locally and in the TabPFN cache."
        ),
    )
    parser.add_argument(
        "--v3-multiclass-model-path",
        default=DEFAULTS.v3_multiclass_model_path,
        help=(
            "TabPFN v3 checkpoint used automatically for data178 multiclass tasks "
            f"when --model-version v3 and --model-path is omitted. 'auto' searches "
            f"for {V3_MULTICLASS_CLASSIFIER_FILE} locally and in the TabPFN cache."
        ),
    )
    parser.add_argument(
        "--many-class",
        choices=["auto", "on", "off"],
        default=DEFAULTS.many_class,
        help=(
            "Legacy many-class wrapper switch. Datasets with >10 classes are "
            "skipped by this benchmark before model fitting."
        ),
    )
    parser.add_argument(
        "--many-class-alphabet-size",
        type=int,
        default=DEFAULTS.many_class_alphabet_size,
    )
    parser.add_argument(
        "--many-class-redundancy",
        type=int,
        default=DEFAULTS.many_class_redundancy,
    )
    parser.add_argument(
        "--many-class-n-estimators",
        type=int,
        default=DEFAULTS.many_class_n_estimators,
    )
    ttt_group = parser.add_mutually_exclusive_group()
    ttt_group.add_argument(
        "--ttt",
        dest="ttt",
        action="store_true",
        help="Enable TabPFN finetuning TTT before final inference.",
    )
    ttt_group.add_argument(
        "--no-ttt",
        dest="ttt",
        action="store_false",
        help="Disable TTT and run baseline TabPFN inference.",
    )
    parser.set_defaults(ttt=DEFAULTS.ttt_enabled)
    parser.add_argument("--ttt-epochs", type=int, default=DEFAULTS.ttt_epochs)
    parser.add_argument(
        "--ttt-max-chunk-size",
        type=int,
        default=DEFAULTS.ttt_max_chunk_size,
    )
    parser.add_argument(
        "--ttt-grad-accumulation-steps",
        type=int,
        default=DEFAULTS.ttt_grad_accumulation_steps,
        help=(
            "Accumulate gradients across this many TTT chunks before one "
            "optimizer update. With the default chunk size 10000 and value 1, "
            "the effective update batch is 10000 samples without loading all "
            "10000 samples at once."
        ),
    )
    parser.add_argument(
        "--ttt-query-ratio",
        type=float,
        default=DEFAULTS.ttt_query_ratio,
    )
    parser.add_argument(
        "--ttt-c-selection",
        choices=[FAWARE_C_SELECTION, "random"],
        default=DEFAULTS.ttt_c_selection,
        help=(
            f"How to choose query/context rows for TTT. {FAWARE_C_SELECTION} "
            "scores the full training set once against the global test centroid "
            "and reserves the highest-score row IDs as context-only across all "
            "chunks/epochs; random uses the original TabPFN finetuning split."
        ),
    )
    parser.add_argument(
        "--ttt-c-metric",
        choices=["raw_l2", "standardized_l2"],
        default=DEFAULTS.ttt_c_metric,
        help=(
            f"Selection-only feature metric used when --ttt-c-selection="
            f"{FAWARE_C_SELECTION}. raw_l2 uses train-fitted numeric/categorical "
            "encodings; standardized_l2 z-scores them with full-train statistics."
        ),
    )
    parser.add_argument(
        "--ttt-c-reserve-ratio",
        type=float,
        default=DEFAULTS.ttt_c_reserve_ratio,
        help=(
            "Fraction of globally scored train rows reserved as context-only when "
            f"--ttt-c-selection={FAWARE_C_SELECTION}. Must satisfy reserve_ratio "
            f"+ --ttt-query-ratio < 1 for {FAWARE_C_SELECTION}."
        ),
    )
    parser.add_argument("--ttt-lr", type=float, default=DEFAULTS.ttt_lr)
    parser.add_argument(
        "--ttt-weight-decay",
        type=float,
        default=DEFAULTS.ttt_weight_decay,
    )
    parser.add_argument(
        "--ttt-wise-ft",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.ttt_wise_ft,
        help=(
            "Apply fixed-alpha WiSE-FT after early-stopping restoration only for "
            f"--ttt-c-selection={FAWARE_C_SELECTION} datasets with total rows "
            f"< {WISE_FT_MAX_DATASET_ROWS}. random always disables WiSE-FT."
        ),
    )
    parser.add_argument(
        "--ttt-wise-alpha",
        type=float,
        default=DEFAULTS.ttt_wise_alpha,
        help=(
            "WiSE-FT interpolation alpha in [0, 1], where 0 keeps the initial "
            "model weights and 1 keeps the fine-tuned endpoint."
        ),
    )
    parser.add_argument("--ttt-grad-clip", type=float, default=DEFAULTS.ttt_grad_clip)
    parser.add_argument("--ttt-patience", type=int, default=DEFAULTS.ttt_patience)
    parser.add_argument("--ttt-min-delta", type=float, default=DEFAULTS.ttt_min_delta)
    parser.add_argument(
        "--ttt-eval-metric",
        choices=["roc_auc", "log_loss", "acc"],
        default=DEFAULTS.ttt_eval_metric,
    )
    parser.add_argument(
        "--ttt-validation-fraction",
        type=float,
        default=DEFAULTS.ttt_validation_fraction,
    )
    parser.add_argument(
        "--ttt-n-estimators-finetune",
        type=int,
        default=DEFAULTS.ttt_n_estimators_finetune,
    )
    parser.add_argument(
        "--ttt-validation-n-estimators",
        type=int,
        default=DEFAULTS.ttt_validation_n_estimators,
    )
    parser.add_argument(
        "--ttt-lr-scheduler",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.ttt_lr_scheduler,
        help="Use the TabPFN finetuning LR scheduler.",
    )
    parser.add_argument(
        "--ttt-lr-warmup-only",
        action="store_true",
        default=DEFAULTS.ttt_lr_warmup_only,
        help="Use warmup-only LR scheduling during finetuning.",
    )
    parser.add_argument(
        "--ttt-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.ttt_activation_checkpointing,
        help="Use activation checkpointing during finetuning.",
    )
    parser.add_argument(
        "--retry-failed-datasets-only",
        action="store_true",
        default=DEFAULTS.retry_failed_datasets_only,
        help=(
            "When enabled, only rerun datasets whose rows have status=fail in "
            "--reference-results-csv. Disabled by default so the script can run "
            "full benchmarks for any model version."
        ),
    )
    parser.add_argument(
        "--reference-results-csv",
        default=DEFAULTS.reference_results_csv,
        help=(
            "Historical results CSV used by --retry-failed-datasets-only to pick "
            "failed datasets. Default points to the local v3 results file."
        ),
    )
    parser.add_argument(
        "--retry-include-oom-failures",
        action="store_true",
        default=DEFAULTS.retry_include_oom_failures,
        help="When retrying previous failures, also rerun failures whose error looks like OOM.",
    )
    parser.add_argument(
        "--merge-results-from-csv",
        default=DEFAULTS.merge_results_from_csv,
        help=(
            "Merge the current run into this existing results CSV before saving "
            "all_classification_results.csv. Defaults to --reference-results-csv "
            "when --retry-failed-datasets-only is enabled."
        ),
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--ignore-pretraining-limits",
        dest="ignore_pretraining_limits",
        action="store_true",
        help=(
            "Ignore TabPFN pretraining limits such as large sample/feature checks. "
            "Enabled by default."
        ),
    )
    limit_group.add_argument(
        "--enforce-pretraining-limits",
        dest="ignore_pretraining_limits",
        action="store_false",
        help="Enforce TabPFN pretraining limits and fail on those validation checks.",
    )
    parser.set_defaults(
        ignore_pretraining_limits=DEFAULTS.ignore_pretraining_limits,
    )
    return parser


def _run_faware_sanity_checks() -> None:
    from functools import partial

    from sklearn.model_selection import train_test_split

    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.2, 1.1],
            [8.0, 8.0],
            [9.0, 9.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.int64)
    X_reference = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1]], dtype=np.float64)
    query_ratio = 0.25
    reserve_ratio = 0.25
    if reserve_ratio + query_ratio >= 1.0:
        raise AssertionError("sanity setup violates reserve/query ratio")
    split_fn = partial(train_test_split, test_size=2, random_state=42)
    reservation = _build_global_test_centroid_reservation(
        X,
        X_reference,
        categorical_feature_indices=[],
        metric="standardized_l2",
        reserve_ratio=reserve_ratio,
    )
    split = _global_test_centroid_reserve_indices(
        y,
        np.arange(len(y)),
        reservation,
        split_fn,
        y,
    )
    if set(split.reserved_idx.tolist()) & set(split.qry_idx.tolist()):
        raise AssertionError("reserved context rows leaked into query")
    if not _has_query_label_coverage(y, split.ctx_idx, split.qry_idx):
        raise AssertionError("query labels are not covered by context")
    if split.fallback_reason is not None:
        raise AssertionError(f"unexpected fallback: {split.fallback_reason}")

    rare_y = np.array([0, 0, 0, 0, 1, 2], dtype=np.int64)
    rare_X = np.arange(len(rare_y)).reshape(-1, 1)
    rare_split_fn = partial(train_test_split, test_size=3, random_state=7)
    random_stats = FawareSplitStats()
    random_aligned = _make_aligned_split_fn(
        rare_split_fn,
        random_stats,
        fallback_label="random sanity",
    )
    random_aligned(rare_X, rare_y, stratify=rare_y)
    if not random_stats.fallback_reasons:
        raise AssertionError("random rare-class sanity did not record fallback")
    random_summary = _split_summary(
        type("_Dummy", (), {"_faware_split_stats_": random_stats})()
    )
    if random_summary["ttt_c_f_distance_mean"] is not None:
        raise AssertionError("random sanity unexpectedly recorded f-distance")

    rare_reservation = _build_global_test_centroid_reservation(
        rare_X.astype(np.float64),
        np.array([[0.0]], dtype=np.float64),
        categorical_feature_indices=[],
        metric="standardized_l2",
        reserve_ratio=0.1,
    )
    rare_faware_split = _global_test_centroid_reserve_indices(
        rare_y,
        np.arange(len(rare_y)),
        rare_reservation,
        rare_split_fn,
        rare_y,
    )
    if set(rare_faware_split.reserved_idx.tolist()) & set(
        rare_faware_split.qry_idx.tolist()
    ):
        raise AssertionError(
            f"rare {FAWARE_C_SELECTION} reserved context rows leaked into query"
        )
    if rare_faware_split.fallback_reason is None:
        raise AssertionError(
            f"rare {FAWARE_C_SELECTION} sanity did not record aligned fallback"
        )

    random_args = argparse.Namespace(
        ttt=True,
        ttt_epochs=1,
        ttt_max_chunk_size=8,
        ttt_grad_accumulation_steps=1,
        ttt_query_ratio=0.9,
        ttt_validation_fraction=0.1,
        ttt_n_estimators_finetune=1,
        ttt_validation_n_estimators=1,
        n_estimators=1,
        ttt_eval_metric="acc",
        ttt_c_selection="random",
        ttt_c_metric="standardized_l2",
        ttt_c_reserve_ratio=0.9,
        ttt_grad_clip=1.0,
        ttt_lr=1e-5,
        ttt_weight_decay=0.01,
        ttt_wise_ft=True,
        ttt_wise_alpha=0.1,
        ttt_patience=1,
        ttt_min_delta=1e-4,
        ttt_lr_scheduler=True,
        ttt_lr_warmup_only=False,
        ttt_activation_checkpointing=True,
    )
    random_config = build_ttt_config(random_args)
    if random_config.c_selection != "random":
        raise AssertionError("random selection was not preserved in TTTConfig")
    if faware_c_source_for_selection(random_config.c_selection) != FAWARE_C_RANDOM_SOURCE:
        raise AssertionError("random selection should use c_source=none")
    if ttt_c_metric_for_config(random_config) is not None:
        raise AssertionError("random selection should not report c_metric")
    if ttt_c_reserve_ratio_for_config(random_config) is not None:
        raise AssertionError("random selection should not report c_reserve_ratio")
    if wise_ft_enabled_for_config(random_config):
        raise AssertionError("random selection should always disable WiSE-FT")


def main() -> None:
    args = build_arg_parser().parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
