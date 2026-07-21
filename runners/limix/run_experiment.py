#!/usr/bin/env python3
"""Run one LimiX-2M method on data184 with resumable atomic results."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import traceback
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The vendor benchmark imports result_naming only for its own automatic output
# path branch. This runner always requires --out-dir, and the helper is absent
# from the current checkout, so an import-only shim keeps the reusable adapter
# and dataset loader available without modifying vendor code.
if "result_naming" not in sys.modules:
    sys.modules["result_naming"] = types.ModuleType("result_naming")

from baseline_compare.LimiX import benchmark_infer as base
from runners.limix import METHODS
from runners.limix.methods import (
    ContextEpisode,
    MixtureContextBuilder,
    RetrievalPreprocessor,
    configure_full_finetune,
    configure_last_block,
    configure_lora,
    configure_micp_adapters,
    configure_prefix,
    count_parameters,
    default_support_size,
    faware_reserved_indices,
    group_episodes_by_context,
    split_context_query,
)

DEFAULT_MODEL = REPO_ROOT / "baseline_compare/LimiX/LimiX-2M.ckpt"
DEFAULT_CONFIG = REPO_ROOT / "baseline_compare/LimiX/config/cls_default_noretrieval.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "data184"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/limix/data184/limix2m"
RESULT_COLUMNS = (
    "dataset_name",
    "dataset_dir",
    "task_type",
    "method",
    "seed",
    "config_hash",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "n_classes",
    "accuracy",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "log_loss",
    "fit_seconds",
    "predict_seconds",
    "wall_seconds",
    "peak_vram_mb",
    "status",
    "error",
    "ttt_applied",
    "ttt_steps",
    "ttt_loss",
    "ttt_lr",
    "trainable_params",
    "total_params",
    "trainable_ratio",
    "best_epoch",
    "baseline_val_accuracy",
    "best_val_accuracy",
    "stopped_early",
    "split_strategy",
    "selection_metric",
    "reserve_ratio",
    "retrieval_backend",
    "context_size",
    "support_size",
    "n_routes",
    "micp_gamma",
    "micp_train_batch_size",
    "micp_inference_batch_size",
    "micp_implementation",
    "fallback_count",
)


@dataclass
class AdaptationTelemetry:
    applied: bool = False
    steps: int = 0
    loss: float | None = None
    seconds: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    trainable_ratio: float = 0.0
    best_epoch: int = 0
    baseline_val_accuracy: float | None = None
    best_val_accuracy: float | None = None
    stopped_early: bool = False
    split_strategy: str | None = None
    selection_metric: str | None = None
    reserve_ratio: float | None = None
    retrieval_backend: str | None = None
    context_size: int | None = None
    support_size: int | None = None
    n_routes: int | None = None
    micp_gamma: float | None = None
    micp_train_batch_size: int | None = None
    micp_inference_batch_size: int | None = None
    micp_implementation: str | None = None
    fallback_count: int = 0


@dataclass
class DatasetResult:
    dataset_name: str
    dataset_dir: str
    task_type: str | None
    method: str
    seed: int
    config_hash: str
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    n_classes: int
    accuracy: float | None
    f1: float | None
    balanced_accuracy: float | None
    roc_auc: float | None
    log_loss: float | None
    fit_seconds: float
    predict_seconds: float
    wall_seconds: float
    peak_vram_mb: float | None
    status: str
    error: str | None
    ttt_applied: bool
    ttt_steps: int
    ttt_loss: float | None
    ttt_lr: float | None
    trainable_params: int
    total_params: int
    trainable_ratio: float
    best_epoch: int
    baseline_val_accuracy: float | None
    best_val_accuracy: float | None
    stopped_early: bool
    split_strategy: str | None
    selection_metric: str | None
    reserve_ratio: float | None
    retrieval_backend: str | None
    context_size: int | None
    support_size: int | None
    n_routes: int | None
    micp_gamma: float | None
    micp_train_batch_size: int | None
    micp_inference_batch_size: int | None
    micp_implementation: str | None
    fallback_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runner_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def semantic_config(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {
        "out_dir",
        "resume",
        "retry_failed",
        "dry_run",
        "verbose",
        "max_datasets",
        "gpu",
        "device",
    }
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }
    payload["model_path"] = str(resolve_path(args.model_path))
    payload["config_path"] = str(resolve_path(args.config_path))
    payload["data_root"] = str(resolve_path(args.data_root))
    return payload


def build_run_identity(args: argparse.Namespace) -> dict[str, Any]:
    model_path = resolve_path(args.model_path)
    config_path = resolve_path(args.config_path)
    payload = {
        "semantic_config": semantic_config(args),
        "model_sha256": sha256_file(model_path),
        "inference_config_sha256": sha256_file(config_path),
        "runner_sha256": runner_sha256(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["config_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        frame.to_csv(handle, index=False)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def normalize_probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(scores, dtype=np.float64))
    values = np.maximum(values, 0.0)
    totals = values.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0
    if np.any(empty):
        values[empty] = 1.0
        totals = values.sum(axis=1, keepdims=True)
    return values / totals


def split_adaptation_pool(
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    from sklearn.model_selection import train_test_split

    labels = np.asarray(y_pool)
    indices = np.arange(len(labels))
    counts = pd.Series(labels).value_counts()
    validation_size = max(len(counts), int(math.ceil(len(labels) * validation_fraction)))
    validation_size = min(validation_size, len(labels) - len(counts))
    if validation_size < 1:
        raise ValueError("labeled pool is too small for an internal validation split")
    stratify = labels if int(counts.min()) >= 2 else None
    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        stratify=stratify,
    )
    missing = set(np.unique(labels)) - set(labels[train_indices])
    if missing:
        train_list = list(train_indices)
        val_list = list(val_indices)
        for label in sorted(missing, key=str):
            move = next(index for index in val_list if labels[index] == label)
            val_list.remove(move)
            train_list.append(move)
        train_indices = np.asarray(train_list, dtype=int)
        val_indices = np.asarray(val_list, dtype=int)
    if len(val_indices) == 0:
        raise ValueError("internal validation split became empty after label repair")
    return (
        X_pool.iloc[train_indices].reset_index(drop=True),
        labels[train_indices],
        X_pool.iloc[val_indices].reset_index(drop=True),
        labels[val_indices],
    )


def is_oom_exception(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in ("out of memory", "cuda oom", "outofmemory"))


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class LimiXMethodRunner(base.LimiXAdapter):
    def __init__(self, args: argparse.Namespace, device: str) -> None:
        super().__init__(args, device)
        self.method = args.method

    def _init_cls_encoding(self, predictor: Any, y_context: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import LabelEncoder

        rng = np.random.default_rng(int(self.args.seed))
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(np.asarray(y_context))
        predictor.label_encoder = encoder
        predictor.classes = encoder.classes_
        predictor.n_classes = len(encoder.classes_)
        noise = rng.random(
            (
                predictor.n_estimators * predictor.class_shuffle_factor,
                predictor.n_classes,
            )
        )
        unique_permutations = np.unique(np.argsort(noise, axis=1), axis=0)
        balance_count = predictor.n_estimators // len(unique_permutations)
        predictor.class_permutations = list(
            chain.from_iterable(
                repeat(permutation, balance_count)
                for permutation in unique_permutations
            )
        )
        remainder = predictor.n_estimators % len(unique_permutations)
        if remainder:
            predictor.class_permutations += [
                unique_permutations[index]
                for index in rng.choice(
                    len(unique_permutations),
                    size=remainder,
                    replace=False,
                )
            ]
        return np.asarray(encoded)

    def _episode_loss(
        self,
        predictor: Any,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_query: pd.DataFrame,
        y_query: np.ndarray,
    ) -> Any:
        import torch
        import torch.nn.functional as functional
        from baseline_compare.LimiX.inference.inference_method import (
            InferenceAttentionMap,
        )
        from baseline_compare.LimiX.inference.preprocess import SubSampleData

        y_context = np.asarray(y_context)
        y_query = np.asarray(y_query)
        missing = set(y_query.tolist()) - set(y_context.tolist())
        if missing:
            raise ValueError(f"query labels absent from context: {sorted(missing, key=str)}")
        raw = np.concatenate([X_context.to_numpy(), X_query.to_numpy()], axis=0)
        y_encoded = self._init_cls_encoding(predictor, y_context)
        encoded_query = predictor.label_encoder.transform(y_query)
        losses = []
        categorical_indices = None
        limit = min(int(self.args.train_n_estimators), predictor.n_estimators)
        for pipeline_index in range(limit):
            pipeline = predictor.preprocess_pipelines[pipeline_index]
            transformed = predictor.convert_x_dtypes(raw.copy())
            transformed = predictor.convert_category2num(transformed)
            transformed = transformed.astype(np.float32)
            if categorical_indices is None:
                categorical_indices = predictor.get_categorical_features_indices(
                    transformed
                )
            current_categorical = list(categorical_indices)
            permuted_y = predictor.class_permutations[pipeline_index][y_encoded.copy()]
            for step_index, step in enumerate(pipeline):
                if isinstance(step, (InferenceAttentionMap, SubSampleData)):
                    raise ValueError(
                        "LimiX adaptation requires a no-retrieval inference config"
                    )
                transformed, current_categorical = step.fit_transform(
                    transformed,
                    current_categorical,
                    predictor.seeds[
                        pipeline_index * predictor.preprocess_num + step_index
                    ],
                    y=permuted_y,
                )
            x_tensor = torch.from_numpy(transformed).float().to(predictor.device)
            y_tensor = torch.from_numpy(permuted_y).float().to(predictor.device)
            query_tensor = torch.from_numpy(encoded_query).long().to(predictor.device)
            with torch.autocast(
                device_type=predictor.device.type,
                enabled=bool(
                    predictor.mix_precision and predictor.device.type == "cuda"
                ),
            ):
                output = predictor.model(
                    x=x_tensor.unsqueeze(0),
                    y=y_tensor.unsqueeze(0),
                    eval_pos=len(y_context),
                    task_type="cls",
                )
                logits = output if isinstance(output, torch.Tensor) else output["cls_output"]
                logits = logits.squeeze(0)[:, : predictor.n_classes]
                if predictor.softmax_temperature != 1:
                    logits = logits.float() / predictor.softmax_temperature
                logits = logits[..., predictor.class_permutations[pipeline_index]]
                losses.append(functional.cross_entropy(logits.float(), query_tensor))
        if not losses:
            raise RuntimeError("no preprocessing pipelines were selected for adaptation")
        return torch.stack(losses).mean()

    def _predict_proba(
        self,
        predictor: Any,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_query: pd.DataFrame,
        *,
        n_estimators: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        original_pipelines = predictor.preprocess_pipelines
        original_configs = predictor.inference_config
        original_count = predictor.n_estimators
        keep = min(max(1, int(n_estimators)), predictor.n_estimators)
        predictor.preprocess_pipelines = predictor.preprocess_pipelines[:keep]
        predictor.inference_config = predictor.inference_config[:keep]
        predictor.n_estimators = keep
        try:
            prediction = np.asarray(
                predictor.predict(
                    X_context.to_numpy(),
                    np.asarray(y_context),
                    X_query.to_numpy(),
                    task_type="Classification",
                )
            )
            classes = np.asarray(
                getattr(
                    predictor,
                    "classes",
                    pd.unique(pd.Series(np.asarray(y_context))),
                )
            )
            if prediction.ndim == 1:
                one_hot = np.zeros((len(prediction), len(classes)), dtype=np.float64)
                lookup = {label: index for index, label in enumerate(classes.tolist())}
                for row_index, label in enumerate(prediction):
                    if label in lookup:
                        one_hot[row_index, lookup[label]] = 1.0
                prediction = one_hot
            return normalize_probabilities(prediction), classes
        finally:
            predictor.preprocess_pipelines = original_pipelines
            predictor.inference_config = original_configs
            predictor.n_estimators = original_count

    def _predict_standard(
        self,
        predictor: Any,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_query: pd.DataFrame,
        *,
        n_estimators: int,
        query_micro_batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if query_micro_batch_size <= 0 or len(X_query) <= query_micro_batch_size:
            return self._predict_proba(
                predictor,
                X_context,
                y_context,
                X_query,
                n_estimators=n_estimators,
            )
        pieces = []
        classes = None
        for start in range(0, len(X_query), query_micro_batch_size):
            probabilities, batch_classes = self._predict_proba(
                predictor,
                X_context,
                y_context,
                X_query.iloc[start : start + query_micro_batch_size].reset_index(
                    drop=True
                ),
                n_estimators=n_estimators,
            )
            pieces.append(probabilities)
            if classes is None:
                classes = batch_classes
            elif not np.array_equal(classes, batch_classes):
                raise RuntimeError("class order changed across query micro-batches")
        return np.concatenate(pieces, axis=0), np.asarray(classes)

    def _predict_episodes(
        self,
        predictor: Any,
        X_context_pool: pd.DataFrame,
        y_context_pool: np.ndarray,
        X_query: pd.DataFrame,
        episodes: Sequence[ContextEpisode],
        *,
        n_estimators: int,
        query_micro_batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        global_classes = np.asarray(pd.unique(pd.Series(np.asarray(y_context_pool))))
        class_to_column = {
            label: index for index, label in enumerate(global_classes.tolist())
        }
        output = np.zeros((len(X_query), len(global_classes)), dtype=np.float64)
        fallback_count = sum(bool(episode.fallback) for episode in episodes)
        for episode in group_episodes_by_context(episodes):
            context_indices = np.asarray(episode.context_indices, dtype=int)
            query_indices = np.asarray(episode.query_indices, dtype=int)
            for start in range(0, len(query_indices), max(1, query_micro_batch_size)):
                current_queries = query_indices[
                    start : start + max(1, query_micro_batch_size)
                ]
                probabilities, classes = self._predict_proba(
                    predictor,
                    X_context_pool.iloc[context_indices].reset_index(drop=True),
                    np.asarray(y_context_pool)[context_indices],
                    X_query.iloc[current_queries].reset_index(drop=True),
                    n_estimators=n_estimators,
                )
                for local_column, label in enumerate(classes.tolist()):
                    if label in class_to_column:
                        output[current_queries, class_to_column[label]] = probabilities[
                            :, local_column
                        ]
        return normalize_probabilities(output), global_classes, fallback_count

    def _configure_method(self, predictor: Any) -> tuple[list[Any], dict[str, Any]]:
        model = predictor.model.to(predictor.device)
        if self.method in {"ft", "faware_ft"}:
            trainable = configure_full_finetune(model)
            metadata: dict[str, Any] = {}
        elif self.method == "lora":
            trainable, installed = configure_lora(
                model,
                rank=int(self.args.lora_rank),
                alpha=float(self.args.lora_alpha),
                dropout=float(self.args.lora_dropout),
            )
            metadata = {"lora_modules": installed}
        elif self.method == "prefix":
            trainable, installed = configure_prefix(
                model,
                prefix_length=int(self.args.prefix_length),
            )
            metadata = {"prefix_modules": installed}
        elif self.method == "last_block":
            trainable = configure_last_block(model)
            metadata = {}
        elif self.method == "mixturepfn":
            trainable, installed = configure_micp_adapters(
                model,
                bottleneck=int(self.args.adapter_bottleneck),
            )
            metadata = {"adapter_modules": installed}
        else:
            raise ValueError(f"method {self.method!r} does not adapt parameters")
        if not trainable:
            raise RuntimeError(f"method {self.method!r} selected no trainable parameters")
        return trainable, metadata

    @staticmethod
    def _scheduler(
        optimizer: Any,
        *,
        total_steps: int,
        warmup_proportion: float,
    ) -> Any:
        import torch

        warmup_steps = max(1, int(total_steps * warmup_proportion))

        def multiplier(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

    def _validation_accuracy_standard(
        self,
        predictor: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        *,
        n_estimators: int | None = None,
        query_micro_batch_size: int | None = None,
    ) -> float:
        import torch

        predictor.model.eval()
        with torch.inference_mode():
            probabilities, classes = self._predict_standard(
                predictor,
                X_train,
                y_train,
                X_val,
                n_estimators=int(
                    self.args.validation_n_estimators
                    if n_estimators is None
                    else n_estimators
                ),
                query_micro_batch_size=int(
                    self.args.query_micro_batch_size
                    if query_micro_batch_size is None
                    else query_micro_batch_size
                ),
            )
        prediction = classes[np.argmax(probabilities, axis=1)]
        predictor.model.train()
        return float(np.mean(prediction == np.asarray(y_val)))

    def _validation_accuracy_episodes(
        self,
        predictor: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        episodes: Sequence[ContextEpisode],
        *,
        n_estimators: int | None = None,
        query_micro_batch_size: int | None = None,
    ) -> tuple[float, int]:
        import torch

        limit = min(len(X_val), int(self.args.validation_max_queries))
        limited = list(episodes[:limit])
        predictor.model.eval()
        with torch.inference_mode():
            probabilities, classes, fallback_count = self._predict_episodes(
                predictor,
                X_train,
                y_train,
                X_val.iloc[:limit].reset_index(drop=True),
                limited,
                n_estimators=int(
                    self.args.validation_n_estimators
                    if n_estimators is None
                    else n_estimators
                ),
                query_micro_batch_size=int(
                    self.args.query_micro_batch_size
                    if query_micro_batch_size is None
                    else query_micro_batch_size
                ),
            )
        prediction = classes[np.argmax(probabilities, axis=1)]
        predictor.model.train()
        return float(np.mean(prediction == np.asarray(y_val)[:limit])), fallback_count

    def _fit_standard(
        self,
        predictor: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
    ) -> AdaptationTelemetry:
        import torch

        start = time.time()
        baseline = self._validation_accuracy_standard(
            predictor,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        reserved: set[int] = set()
        selection_metric = None
        if self.method == "faware_ft":
            preprocessor = RetrievalPreprocessor(
                self._categorical_indices_for_frame(X_train)
            ).fit(X_train)
            train_space = preprocessor.transform(X_train)
            test_space = preprocessor.transform(X_test)
            indices, _ = faware_reserved_indices(
                train_space,
                test_space,
                reserve_ratio=float(self.args.reserve_ratio),
            )
            reserved = set(indices.tolist())
            selection_metric = "f_mmd:standardized_l2:test"

        trainable, _ = self._configure_method(predictor)
        trainable_count, total_count, ratio = count_parameters(predictor.model)
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
        )
        chunks_per_epoch = max(
            1,
            int(math.ceil(len(y_train) / int(self.args.chunk_size))),
        )
        scheduler = self._scheduler(
            optimizer,
            total_steps=int(self.args.epochs) * chunks_per_epoch,
            warmup_proportion=float(self.args.warmup_proportion),
        )
        # The untouched checkpoint is reported as the baseline, but it is not
        # an adaptation checkpoint.  Early stopping selects among trained
        # epochs so an adaptation method never silently falls back to Infer.
        best_metric = float("-inf")
        best_state = None
        best_epoch = 0
        patience = 0
        steps = 0
        last_loss = None
        stopped_early = False
        split_labels: set[str] = set()
        predictor.model.train()
        for epoch in range(int(self.args.epochs)):
            rng = np.random.default_rng(int(self.args.seed) + epoch)
            order = rng.permutation(len(y_train))
            for chunk_number, start_index in enumerate(
                range(0, len(order), int(self.args.chunk_size))
            ):
                chunk = order[
                    start_index : start_index + int(self.args.chunk_size)
                ]
                if len(chunk) < int(self.args.min_chunk_size):
                    continue
                forced = [
                    local_index
                    for local_index, global_index in enumerate(chunk)
                    if int(global_index) in reserved
                ]
                context_indices, query_indices, strategy = split_context_query(
                    y_train[chunk],
                    query_ratio=float(self.args.query_ratio),
                    seed=int(self.args.seed) + epoch + chunk_number * 7919,
                    forced_context=forced,
                )
                split_labels.add(strategy)
                optimizer.zero_grad(set_to_none=True)
                loss = self._episode_loss(
                    predictor,
                    X_train.iloc[chunk[context_indices]].reset_index(drop=True),
                    y_train[chunk[context_indices]],
                    X_train.iloc[chunk[query_indices]].reset_index(drop=True),
                    y_train[chunk[query_indices]],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, float(self.args.grad_clip))
                optimizer.step()
                scheduler.step()
                steps += 1
                last_loss = float(loss.detach().cpu())
            metric = self._validation_accuracy_standard(
                predictor,
                X_train,
                y_train,
                X_val,
                y_val,
            )
            if best_state is None or metric > best_metric + float(self.args.min_delta):
                best_metric = metric
                best_epoch = epoch + 1
                best_state = copy.deepcopy(predictor.model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= int(self.args.patience):
                stopped_early = True
                break
        applied = best_state is not None
        if applied:
            predictor.model.load_state_dict(best_state)
        return AdaptationTelemetry(
            applied=applied,
            steps=steps,
            loss=last_loss,
            seconds=time.time() - start,
            trainable_params=trainable_count,
            total_params=total_count,
            trainable_ratio=ratio,
            best_epoch=best_epoch,
            baseline_val_accuracy=baseline,
            best_val_accuracy=best_metric if best_state is not None else baseline,
            stopped_early=stopped_early,
            split_strategy="|".join(sorted(split_labels)),
            selection_metric=selection_metric,
            reserve_ratio=(
                float(self.args.reserve_ratio)
                if self.method == "faware_ft"
                else None
            ),
        )

    def _categorical_indices_for_frame(self, X: pd.DataFrame) -> list[int]:
        return [
            index
            for index, dtype in enumerate(X.dtypes)
            if not pd.api.types.is_numeric_dtype(dtype)
        ]

    def _select_micp_builder(
        self,
        predictor: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        train_space: np.ndarray,
        val_space: np.ndarray,
        *,
        support_size: int,
    ) -> tuple[MixtureContextBuilder, float, int]:
        small_dataset = len(y_train) <= support_size
        candidates = (
            (1.0,)
            if small_dataset or self.args.n_clusters is not None
            else tuple(float(value) for value in self.args.micp_gammas)
        )
        best_builder = None
        best_metric = float("-inf")
        best_fallbacks = 0
        for gamma in candidates:
            builder = MixtureContextBuilder(
                train_space,
                y_train,
                support_size=support_size,
                n_clusters=1 if small_dataset else self.args.n_clusters,
                seed=int(self.args.seed),
                backend=self.args.retrieval_backend,
                gamma=gamma,
            )
            if len(y_train) <= support_size:
                metric = self._validation_accuracy_standard(
                    predictor,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_estimators=int(self.args.micp_n_estimators),
                    query_micro_batch_size=int(
                        self.args.micp_inference_batch_size
                    ),
                )
                fallbacks = 0
            else:
                metric, fallbacks = self._validation_accuracy_episodes(
                    predictor,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    builder.route(val_space),
                    n_estimators=int(self.args.micp_n_estimators),
                    query_micro_batch_size=int(
                        self.args.micp_inference_batch_size
                    ),
                )
            if best_builder is None or metric > best_metric:
                best_builder = builder
                best_metric = metric
                best_fallbacks = fallbacks
        assert best_builder is not None
        return best_builder, best_metric, best_fallbacks

    def _prepare_micp(
        self,
        predictor: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> tuple[AdaptationTelemetry, RetrievalPreprocessor]:
        started = time.time()
        preprocessor = RetrievalPreprocessor(
            self._categorical_indices_for_frame(X_train)
        ).fit(X_train)
        train_space = preprocessor.transform(X_train)
        val_space = preprocessor.transform(X_val)
        support_size = (
            int(self.args.support_size)
            if self.args.support_size is not None
            else default_support_size(len(y_train))
        )
        builder, metric, fallbacks = self._select_micp_builder(
            predictor,
            X_train,
            y_train,
            X_val,
            y_val,
            train_space,
            val_space,
            support_size=support_size,
        )
        _, total_count, _ = count_parameters(predictor.model)
        return (
            AdaptationTelemetry(
                applied=False,
                seconds=time.time() - started,
                total_params=total_count,
                baseline_val_accuracy=metric,
                best_val_accuracy=metric,
                split_strategy=(
                    "paper_micp_disabled_n_le_b"
                    if len(y_train) <= support_size
                    else "paper_micp_kmeans_route_shared_support"
                ),
                selection_metric="validation_accuracy_gamma_selection",
                retrieval_backend=(
                    f"{builder.kmeans.backend}+{builder.anchor_index.backend}"
                ),
                support_size=support_size,
                n_routes=builder.n_clusters,
                micp_gamma=builder.gamma,
                micp_train_batch_size=int(self.args.micp_train_batch_size),
                micp_inference_batch_size=int(
                    self.args.micp_inference_batch_size
                ),
                micp_implementation="paper_micp_routing_only",
                fallback_count=fallbacks,
            ),
            preprocessor,
        )

    def _fit_mixturepfn(
        self,
        predictor: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> tuple[AdaptationTelemetry, RetrievalPreprocessor]:
        import torch

        started = time.time()
        preprocessor = RetrievalPreprocessor(
            self._categorical_indices_for_frame(X_train)
        ).fit(X_train)
        train_space = preprocessor.transform(X_train)
        val_space = preprocessor.transform(X_val)
        support_size = (
            int(self.args.support_size)
            if self.args.support_size is not None
            else default_support_size(len(y_train))
        )
        builder, baseline, baseline_fallbacks = self._select_micp_builder(
            predictor,
            X_train,
            y_train,
            X_val,
            y_val,
            train_space,
            val_space,
            support_size=support_size,
        )
        trainable, _ = self._configure_method(predictor)
        trainable_count, total_count, ratio = count_parameters(predictor.model)
        optimizer = torch.optim.Adam(
            trainable,
            lr=float(self.args.micp_lr),
        )
        last_loss = None
        fallback_count = baseline_fallbacks
        completed_steps = 0
        rng = np.random.default_rng(int(self.args.seed))
        predictor.model.train()
        for step in range(int(self.args.micp_steps)):
            anchor = int(rng.integers(0, len(y_train)))
            episode = builder.bootstrap_episode(
                anchor,
                query_batch_size=int(self.args.micp_train_batch_size),
                small_train_fraction=float(self.args.micp_small_train_fraction),
            )
            fallback_count += int(bool(episode.fallback))
            optimizer.zero_grad(set_to_none=True)
            loss = self._episode_loss(
                predictor,
                X_train.iloc[episode.context_indices].reset_index(drop=True),
                y_train[episode.context_indices],
                X_train.iloc[episode.query_indices].reset_index(drop=True),
                y_train[episode.query_indices],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(self.args.grad_clip))
            optimizer.step()
            completed_steps = step + 1
            last_loss = float(loss.detach().cpu())

        # Appendix 17 fixes C_A PFN at 128 Adam iterations. Gamma is a
        # validation-selected inference hyperparameter, so select it after
        # adapter optimization rather than early-stopping adapter checkpoints.
        builder, best_metric, selection_fallbacks = self._select_micp_builder(
            predictor,
            X_train,
            y_train,
            X_val,
            y_val,
            train_space,
            val_space,
            support_size=support_size,
        )
        fallback_count += selection_fallbacks
        applied = completed_steps == int(self.args.micp_steps)
        return (
            AdaptationTelemetry(
                applied=applied,
                steps=completed_steps,
                loss=last_loss,
                seconds=time.time() - started,
                trainable_params=trainable_count,
                total_params=total_count,
                trainable_ratio=ratio,
                best_epoch=completed_steps,
                baseline_val_accuracy=baseline,
                best_val_accuracy=best_metric,
                stopped_early=False,
                split_strategy=(
                    "paper_capfn_small_random_90_10"
                    if len(y_train) <= support_size
                    else "paper_capfn_large_knn_b_then_batch64"
                ),
                selection_metric="validation_accuracy_gamma_selection",
                retrieval_backend=(
                    f"{builder.kmeans.backend}+{builder.anchor_index.backend}"
                ),
                support_size=support_size,
                n_routes=builder.n_clusters,
                micp_gamma=builder.gamma,
                micp_train_batch_size=int(self.args.micp_train_batch_size),
                micp_inference_batch_size=int(
                    self.args.micp_inference_batch_size
                ),
                micp_implementation="paper_mixturepfn_micp_plus_capfn",
                fallback_count=fallback_count,
            ),
            preprocessor,
        )

    def run_loaded(
        self,
        loaded: base.LoadedDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, AdaptationTelemetry, float, float]:
        predictor = self._load_predictor()
        # LimiXPredictor.predict() moves the model to its device inside
        # torch.inference_mode().  If that first device move allocates new
        # parameter storage, PyTorch marks it as inference-only and later
        # requires_grad updates fail.  Adaptation methods must complete the
        # device move before their baseline inference pass.
        if self.method != "infer":
            predictor.model.to(predictor.device)
        X_pool = loaded.X_train_merged.reset_index(drop=True)
        y_pool = np.asarray(loaded.y_train_merged)
        if self.method == "infer":
            total = sum(int(parameter.numel()) for parameter in predictor.model.parameters())
            predict_started = time.time()
            probabilities, classes = self._predict_standard(
                predictor,
                X_pool,
                y_pool,
                loaded.X_test,
                n_estimators=int(self.args.n_estimators),
                query_micro_batch_size=int(self.args.query_micro_batch_size),
            )
            return (
                classes[np.argmax(probabilities, axis=1)],
                probabilities,
                classes,
                AdaptationTelemetry(total_params=total),
                0.0,
                time.time() - predict_started,
            )

        X_train, y_train, X_val, y_val = split_adaptation_pool(
            X_pool,
            y_pool,
            validation_fraction=float(self.args.validation_fraction),
            seed=int(self.args.seed),
        )
        preprocessor = None
        if self.method == "micp":
            telemetry, preprocessor = self._prepare_micp(
                predictor,
                X_train,
                y_train,
                X_val,
                y_val,
            )
        elif self.method == "mixturepfn":
            telemetry, preprocessor = self._fit_mixturepfn(
                predictor,
                X_train,
                y_train,
                X_val,
                y_val,
            )
        else:
            telemetry = self._fit_standard(
                predictor,
                X_train,
                y_train,
                X_val,
                y_val,
                loaded.X_test,
            )

        if self.method != "micp" and not telemetry.applied:
            self.reset_predictor()
            predictor = self._load_predictor()
        predict_started = time.time()
        micro_batch = int(
            self.args.micp_inference_batch_size
            if self.method in {"micp", "mixturepfn"}
            else self.args.query_micro_batch_size
        )

        def predict_current(batch_size: int) -> tuple[np.ndarray, np.ndarray, int]:
            if self.method in {"micp", "mixturepfn"}:
                assert preprocessor is not None
                if len(y_pool) <= int(
                    telemetry.support_size or default_support_size(len(y_pool))
                ):
                    probabilities, classes = self._predict_standard(
                        predictor,
                        X_pool,
                        y_pool,
                        loaded.X_test,
                        n_estimators=int(self.args.micp_n_estimators),
                        query_micro_batch_size=batch_size,
                    )
                    return probabilities, classes, 0
                pool_space = preprocessor.transform(X_pool)
                query_space = preprocessor.transform(loaded.X_test)
                builder = MixtureContextBuilder(
                    pool_space,
                    y_pool,
                    support_size=int(telemetry.support_size or default_support_size(len(y_pool))),
                    n_clusters=self.args.n_clusters,
                    seed=int(self.args.seed),
                    backend=self.args.retrieval_backend,
                    gamma=float(telemetry.micp_gamma or 1.0),
                )
                telemetry.n_routes = builder.n_clusters
                return self._predict_episodes(
                    predictor,
                    X_pool,
                    y_pool,
                    loaded.X_test,
                    builder.route(query_space),
                    n_estimators=int(self.args.micp_n_estimators),
                    query_micro_batch_size=batch_size,
                )
            probabilities, classes = self._predict_standard(
                predictor,
                X_pool,
                y_pool,
                loaded.X_test,
                n_estimators=int(self.args.n_estimators),
                query_micro_batch_size=batch_size,
            )
            return probabilities, classes, 0

        try:
            probabilities, classes, prediction_fallbacks = predict_current(micro_batch)
        except Exception as exc:
            if not is_oom_exception(exc) or micro_batch <= 1:
                raise
            cleanup_cuda()
            probabilities, classes, prediction_fallbacks = predict_current(
                max(1, micro_batch // 2)
            )
        telemetry.fallback_count += prediction_fallbacks
        predict_seconds = time.time() - predict_started
        prediction = classes[np.argmax(probabilities, axis=1)]
        return (
            prediction,
            probabilities,
            classes,
            telemetry,
            telemetry.seconds,
            predict_seconds,
        )


def peak_vram_mb(device: str) -> float | None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def reset_peak_vram(device: str) -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def empty_result(
    args: argparse.Namespace,
    dataset_dir: Path,
    config_hash: str,
    *,
    status: str,
    error: str,
    wall_seconds: float,
) -> DatasetResult:
    return DatasetResult(
        dataset_name=dataset_dir.name,
        dataset_dir=str(dataset_dir),
        task_type=None,
        method=args.method,
        seed=int(args.seed),
        config_hash=config_hash,
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
        wall_seconds=wall_seconds,
        peak_vram_mb=None,
        status=status,
        error=error,
        ttt_applied=False,
        ttt_steps=0,
        ttt_loss=None,
        ttt_lr=None,
        trainable_params=0,
        total_params=0,
        trainable_ratio=0.0,
        best_epoch=0,
        baseline_val_accuracy=None,
        best_val_accuracy=None,
        stopped_early=False,
        split_strategy=None,
        selection_metric=None,
        reserve_ratio=None,
        retrieval_backend=None,
        context_size=None,
        support_size=None,
        n_routes=None,
        micp_gamma=None,
        micp_train_batch_size=None,
        micp_inference_batch_size=None,
        micp_implementation=None,
        fallback_count=0,
    )


def evaluate_dataset(
    args: argparse.Namespace,
    dataset_dir: Path,
    *,
    config_hash: str,
    device: str,
) -> DatasetResult:
    started = time.time()
    reset_peak_vram(device)
    loaded = None
    try:
        loaded = base.load_classification_dataset(dataset_dir)
        if loaded.n_classes > 10:
            raise ValueError(f"LimiX-2M supports at most 10 classes, got {loaded.n_classes}")
        runner = LimiXMethodRunner(args, device=device)
        prediction, probabilities, classes, telemetry, fit_seconds, predict_seconds = (
            runner.run_loaded(loaded)
        )
        y_test = np.asarray(loaded.y_test)
        if len(prediction) != len(y_test):
            raise ValueError(
                f"prediction length {len(prediction)} != test rows {len(y_test)}"
            )
        return DatasetResult(
            dataset_name=loaded.dataset_name,
            dataset_dir=str(loaded.dataset_dir),
            task_type=loaded.task_type,
            method=args.method,
            seed=int(args.seed),
            config_hash=config_hash,
            n_train=int(len(loaded.y_train)),
            n_val=loaded.n_val,
            n_test=int(len(y_test)),
            n_features=int(loaded.X_train.shape[1]),
            n_classes=loaded.n_classes,
            accuracy=float(np.mean(np.asarray(prediction) == y_test)),
            f1=base.compute_weighted_f1(y_test, prediction),
            balanced_accuracy=base.compute_balanced_accuracy(y_test, prediction),
            roc_auc=base.compute_roc_auc(y_test, probabilities, classes),
            log_loss=base.compute_log_loss(y_test, probabilities, classes),
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            wall_seconds=time.time() - started,
            peak_vram_mb=peak_vram_mb(device),
            status="ok",
            error=None,
            ttt_applied=telemetry.applied,
            ttt_steps=telemetry.steps,
            ttt_loss=telemetry.loss,
            ttt_lr=(
                None
                if args.method in {"infer", "micp"}
                else float(
                    args.micp_lr
                    if args.method == "mixturepfn"
                    else args.lr
                )
            ),
            trainable_params=telemetry.trainable_params,
            total_params=telemetry.total_params,
            trainable_ratio=telemetry.trainable_ratio,
            best_epoch=telemetry.best_epoch,
            baseline_val_accuracy=telemetry.baseline_val_accuracy,
            best_val_accuracy=telemetry.best_val_accuracy,
            stopped_early=telemetry.stopped_early,
            split_strategy=telemetry.split_strategy,
            selection_metric=telemetry.selection_metric,
            reserve_ratio=telemetry.reserve_ratio,
            retrieval_backend=telemetry.retrieval_backend,
            context_size=telemetry.context_size,
            support_size=telemetry.support_size,
            n_routes=telemetry.n_routes,
            micp_gamma=telemetry.micp_gamma,
            micp_train_batch_size=telemetry.micp_train_batch_size,
            micp_inference_batch_size=telemetry.micp_inference_batch_size,
            micp_implementation=telemetry.micp_implementation,
            fallback_count=telemetry.fallback_count,
        )
    except base.SkipDataset as exc:
        return empty_result(
            args,
            dataset_dir,
            config_hash,
            status="skip",
            error=str(exc),
            wall_seconds=time.time() - started,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            error += "\n" + traceback.format_exc()
        result = empty_result(
            args,
            dataset_dir,
            config_hash,
            status="error",
            error=error,
            wall_seconds=time.time() - started,
        )
        if loaded is not None:
            result.task_type = loaded.task_type
            result.n_train = int(len(loaded.y_train))
            result.n_val = loaded.n_val
            result.n_test = int(len(loaded.y_test))
            result.n_features = int(loaded.X_train.shape[1])
            result.n_classes = loaded.n_classes
            result.peak_vram_mb = peak_vram_mb(device)
        return result
    finally:
        cleanup_cuda()


def result_path(out_dir: Path, dataset_name: str) -> Path:
    safe_name = dataset_name.replace(os.sep, "_")
    return out_dir / "dataset_results" / f"{safe_name}.json"


def load_existing_result(path: Path) -> DatasetResult | None:
    if not path.is_file():
        return None
    try:
        return DatasetResult(**json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def collect_results(out_dir: Path, dataset_dirs: Sequence[Path]) -> list[DatasetResult]:
    rows = []
    for dataset_dir in dataset_dirs:
        row = load_existing_result(result_path(out_dir, dataset_dir.name))
        if row is not None:
            rows.append(row)
    return rows


def write_method_outputs(
    out_dir: Path,
    rows: Sequence[DatasetResult],
    *,
    discovered_count: int,
    wall_seconds: float,
) -> None:
    frame = pd.DataFrame([asdict(row) for row in rows], columns=RESULT_COLUMNS)
    atomic_write_csv(out_dir / "all_classification_results.csv", frame)
    ok = frame[frame["status"] == "ok"] if len(frame) else frame
    errors = frame[frame["status"] == "error"] if len(frame) else frame
    skipped = frame[frame["status"] == "skip"] if len(frame) else frame

    def metric_line(column: str) -> str:
        values = pd.to_numeric(ok[column], errors="coerce") if len(ok) else pd.Series(dtype=float)
        return f"avg_{column}_ok: {values.mean():.6f}" if values.notna().any() else f"avg_{column}_ok: (none)"

    lines = [
        f"discovered_datasets: {discovered_count}",
        f"processed_datasets: {len(frame)}",
        f"ok_count: {len(ok)}",
        f"error_count: {len(errors)}",
        f"skipped_count: {len(skipped)}",
        f"ttt_applied_count: {int(ok['ttt_applied'].fillna(False).astype(bool).sum()) if len(ok) else 0}",
        metric_line("accuracy"),
        metric_line("balanced_accuracy"),
        metric_line("f1"),
        metric_line("roc_auc"),
        metric_line("log_loss"),
        f"wall_seconds: {wall_seconds:.3f}",
        "error_datasets: "
        + (", ".join(errors["dataset_name"].astype(str)) if len(errors) else "(none)"),
        "skipped_datasets: "
        + (", ".join(skipped["dataset_name"].astype(str)) if len(skipped) else "(none)"),
    ]
    atomic_write_text(out_dir / "summary.txt", "\n".join(lines) + "\n")


def bind_device(args: argparse.Namespace) -> str:
    if args.device == "cpu":
        return "cpu"
    if args.gpu is not None:
        base.bind_worker_gpu(int(args.gpu))
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return "cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"


def preflight_environment() -> None:
    required = {
        "torch": "torch",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "einops": "einops",
        "kditransform": "kditransform",
        "hyperopt": "hyperopt",
    }
    missing = [
        distribution
        for module, distribution in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        requirements = REPO_ROOT / "baseline_compare/LimiX/requirements_benchmark.txt"
        raise RuntimeError(
            "missing LimiX runtime dependencies: "
            + ", ".join(missing)
            + f"; install with: pip install -r {requirements}"
        )


def run(args: argparse.Namespace) -> None:
    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    model_path = resolve_path(args.model_path)
    config_path = resolve_path(args.config_path)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"LimiX-2M checkpoint does not exist: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"inference config does not exist: {config_path}")
    dataset_dirs = base.find_dataset_dirs(data_root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: int(args.max_datasets)]
    if not dataset_dirs:
        raise FileNotFoundError(f"no datasets found under {data_root}")

    identity = build_run_identity(args)
    config_hash = str(identity["config_hash"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": args.method,
                    "seed": args.seed,
                    "datasets": len(dataset_dirs),
                    "out_dir": str(out_dir),
                    "config_hash": config_hash,
                },
                indent=2,
            )
        )
        return

    preflight_environment()
    manifest_path = out_dir / "run_manifest.json"
    existing_manifest = None
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_hash = existing_manifest.get("config_hash")
        if existing_hash != config_hash:
            raise RuntimeError(
                "output directory contains a different run configuration: "
                f"{existing_hash} != {config_hash}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **identity,
        "method": args.method,
        "seed": int(args.seed),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "expected_datasets": len(dataset_dirs),
        "status": "running",
        "started_at": (
            existing_manifest.get("started_at")
            if existing_manifest
            else utc_now()
        ),
        "updated_at": utc_now(),
        "command": [sys.executable, *sys.argv],
    }
    atomic_write_json(manifest_path, manifest)
    device = bind_device(args)
    started = time.time()
    for index, dataset_dir in enumerate(dataset_dirs, start=1):
        path = result_path(out_dir, dataset_dir.name)
        existing = load_existing_result(path) if args.resume else None
        if (
            existing is not None
            and existing.config_hash == config_hash
            and existing.status == "ok"
        ):
            if args.verbose:
                print(f"[{index}/{len(dataset_dirs)}] resume ok {dataset_dir.name}")
            continue
        if (
            existing is not None
            and existing.config_hash == config_hash
            and existing.status != "ok"
            and not args.retry_failed
        ):
            if args.verbose:
                print(
                    f"[{index}/{len(dataset_dirs)}] preserve {existing.status} "
                    f"{dataset_dir.name}"
                )
            continue
        row = evaluate_dataset(
            args,
            dataset_dir,
            config_hash=config_hash,
            device=device,
        )
        atomic_write_json(path, asdict(row))
        print(
            f"[{index}/{len(dataset_dirs)}] {row.status} {row.dataset_name} "
            f"accuracy={row.accuracy} ttt_applied={row.ttt_applied}",
            flush=True,
        )
        rows = collect_results(out_dir, dataset_dirs)
        write_method_outputs(
            out_dir,
            rows,
            discovered_count=len(dataset_dirs),
            wall_seconds=time.time() - started,
        )
    rows = collect_results(out_dir, dataset_dirs)
    write_method_outputs(
        out_dir,
        rows,
        discovered_count=len(dataset_dirs),
        wall_seconds=time.time() - started,
    )
    status_counts = pd.Series([row.status for row in rows]).value_counts().to_dict()
    manifest.update(
        {
            "status": (
                "complete"
                if len(rows) == len(dataset_dirs)
                else "incomplete"
            ),
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "processed_datasets": len(rows),
            "updated_at": utc_now(),
            "wall_seconds": time.time() - started,
        }
    )
    atomic_write_json(manifest_path, manifest)


def parse_float_csv(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected one or more positive comma-separated floats"
        )
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one native LimiX-2M adaptation method on data184.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", "--random-state", dest="seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--min-chunk-size", type=int, default=10)
    parser.add_argument("--query-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-max-queries", type=int, default=64)
    parser.add_argument("--train-n-estimators", type=int, default=2)
    parser.add_argument("--validation-n-estimators", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--query-micro-batch-size", type=int, default=64)

    parser.add_argument("--reserve-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument(
        "--retrieval-backend",
        choices=["auto", "faiss", "sklearn"],
        default="auto",
    )
    parser.add_argument("--support-size", type=int)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--adapter-bottleneck", type=int, default=8)
    parser.add_argument("--micp-steps", type=int, default=128)
    parser.add_argument("--micp-lr", type=float, default=1e-3)
    parser.add_argument(
        "--micp-gammas",
        type=parse_float_csv,
        default=(5.0, 1.0),
        help="paper gamma candidates selected on validation accuracy",
    )
    parser.add_argument("--micp-train-batch-size", type=int, default=64)
    parser.add_argument("--micp-inference-batch-size", type=int, default=1024)
    parser.add_argument("--micp-n-estimators", type=int, default=16)
    parser.add_argument("--micp-small-train-fraction", type=float, default=0.9)

    # Arguments consumed by baseline_compare.LimiX.benchmark_infer.LimiXAdapter.
    parser.set_defaults(
        random_state=42,
        hf_endpoint="https://hf-mirror.com",
        hf_repo="stableai-org/LimiX-2M",
        hf_filename="LimiX-2M.ckpt",
        model_cache_dir=str(DEFAULT_MODEL.parent),
        local_files_only=True,
        disable_mixed_precision=False,
        direct_max_classes=10,
        max_classes=10,
        max_train_rows=0,
        test_batch_rows=0,
        internal_ddp_dataset_dir=None,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.chunk_size < 2 or args.min_chunk_size < 1:
        raise ValueError("epochs/chunk-size/min-chunk-size must be positive")
    if not 0 < args.query_ratio < 1:
        raise ValueError("--query-ratio must be in (0, 1)")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if not 0 < args.reserve_ratio < 1:
        raise ValueError("--reserve-ratio must be in (0, 1)")
    if args.reserve_ratio + args.query_ratio >= 1:
        raise ValueError("reserve-ratio + query-ratio must be < 1")
    if args.train_n_estimators < 1 or args.n_estimators < 1:
        raise ValueError("estimator counts must be positive")
    if args.query_micro_batch_size < 1:
        raise ValueError("--query-micro-batch-size must be positive")
    if (
        args.micp_steps < 1
        or args.micp_train_batch_size < 1
        or args.micp_inference_batch_size < 1
        or args.micp_n_estimators < 1
    ):
        raise ValueError("MICP steps/batch/estimator counts must be positive")
    if not 0 < args.micp_small_train_fraction < 1:
        raise ValueError("--micp-small-train-fraction must be in (0, 1)")
    args.random_state = int(args.seed)


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
