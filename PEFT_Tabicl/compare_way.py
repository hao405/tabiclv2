#!/usr/bin/env python3
"""Run the official LoCalPFN fine-tuning recipe on TabICLv2/TabPFNv3.

The file is intentionally standalone.  In particular it neither imports nor
executes ``PEFT_Tabicl/1C_Chunk_PEFT.py``.

The retrieval and episode semantics are ported from layer6ai-labs/LoCalPFN at
commit ff8803c57cd277380b2444f0f5ed4856b46f47f5.  TabICLv2 and TabPFNv3 are
model backends only; they do not change the LoCalPFN sampling policy.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TABPFN_SRC_DIR = REPO_ROOT / "baseline_compare/TabPFN-main/src"
for _path in (SRC_DIR, TABPFN_SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DEFAULT_DATA_ROOT = Path("data184")
DEFAULT_TABICL_MODEL = Path("tabicl-classifier-v2-20260212.ckpt")
DEFAULT_TABPFN_BINARY_MODEL = Path(
    "baseline_compare/TabPFN-main/tabpfn-v3-classifier-v3_20260417_binary.ckpt"
)
DEFAULT_TABPFN_MULTICLASS_MODEL = Path(
    "baseline_compare/TabPFN-main/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
)
DEFAULT_OUT_ROOT = Path("results/PEFT_compare")
MODEL_FAMILIES = ("tabiclv2", "tabpfnv3")
METHODS = ("localpfn",)
LOCALPFN_IMPLEMENTATION = "official_ftknn_port"
LOCALPFN_SOURCE_COMMIT = "ff8803c57cd277380b2444f0f5ed4856b46f47f5"
CLASSIFICATION_TASKS = {"binclass", "multiclass"}
MISSING_CATEGORY = "__local_context_missing__"
OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "outofmemoryerror",
)


@dataclass(frozen=True)
class ContextEpisode:
    context_indices: np.ndarray
    query_indices: np.ndarray


@dataclass
class RetrievalTiming:
    preprocessing_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    routing_seconds: float = 0.0


@dataclass
class AdaptationResult:
    steps: int
    loss: float | None
    seconds: float
    trainable_params: int
    trainable_ratio: float
    best_step: int = 0
    stopped_early: bool = False


@dataclass
class DatasetResult:
    dataset_name: str
    dataset_dir: str
    task_type: str | None
    model_family: str
    method: str
    config_fingerprint: str
    status: str
    error: str | None
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_features: int = 0
    n_classes: int = 0
    accuracy: float | None = None
    f1: float | None = None
    balanced_accuracy: float | None = None
    roc_auc: float | None = None
    log_loss: float | None = None
    retrieval_backend: str | None = None
    localpfn_implementation: str = LOCALPFN_IMPLEMENTATION
    localpfn_source_commit: str = LOCALPFN_SOURCE_COMMIT
    preprocessing_scope: str = "all_unlabeled_X_vocab_train_scaler"
    retrieval_space: str = "raw"
    context_size: int = 0
    train_query_length: int = 0
    train_batch_size: int = 0
    inference_batch_size: int = 0
    best_epoch: int = 0
    fallback_count: int = 0
    ft_steps: int = 0
    ft_loss: float | None = None
    ft_applied: bool = False
    ft_best_step: int = 0
    ft_stopped_early: bool = False
    trainable_params: int = 0
    trainable_ratio: float = 0.0
    retrieval_seconds: float = 0.0
    routing_seconds: float = 0.0
    ft_seconds: float = 0.0
    predict_seconds: float = 0.0
    total_seconds: float = 0.0
    inference_fallback: bool = False


@dataclass
class LoadedDataset:
    name: str
    path: Path
    task_type: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame | None
    y_val: np.ndarray | None
    X_test: pd.DataFrame
    y_test: np.ndarray
    categorical_indices: list[int] = field(default_factory=list)


@dataclass
class PreparedLoCalData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame | None
    X_test: pd.DataFrame
    retrieval_train: np.ndarray
    retrieval_val: np.ndarray | None
    retrieval_test: np.ndarray
    categorical_indices: list[int]


def is_oom_exception(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in OOM_MARKERS)


def count_parameters(model: Any) -> tuple[int, int, float]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return trainable, total, (float(trainable) / total if total else 0.0)


def expand_matrix_trials(
    model_family: str,
    method: str,
) -> list[tuple[str, str]]:
    models = list(MODEL_FAMILIES) if model_family == "all" else [model_family]
    return [(model, method) for model in models]


def default_context_size(n_train: int) -> int:
    return min(max(1, int(10 * math.sqrt(n_train))), 1000)


def _normalize_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_CATEGORY).astype(str)


def _load_array(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.lib.npyio.NpzFile):
        value = value[value.files[0]]
    return np.asarray(value)


def _find_suffix(files: Sequence[Path], suffix: str) -> Path | None:
    suffix = suffix.lower()
    return next((path for path in files if path.name.lower().endswith(suffix)), None)


def _split_files(dataset_dir: Path, split: str) -> tuple[Path | None, Path | None, Path] | None:
    files = [path for path in dataset_dir.iterdir() if path.is_file()]
    num = _find_suffix(files, f"n_{split}.npy")
    cat = _find_suffix(files, f"c_{split}.npy")
    y = _find_suffix(files, f"y_{split}.npy")
    if y is None:
        return None
    if num is None and cat is None:
        raise FileNotFoundError(f"{dataset_dir}: {split} has labels but no features")
    return num, cat, y


def _load_split(
    split_files: tuple[Path | None, Path | None, Path],
    *,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    num_path, cat_path, y_path = split_files
    frames: list[pd.DataFrame] = []
    categorical_indices: list[int] = []
    if num_path is not None:
        num = np.asarray(_load_array(num_path))
        if num.ndim == 1:
            num = num[:, None]
        frames.append(pd.DataFrame(num).apply(pd.to_numeric, errors="coerce"))
    if cat_path is not None:
        cat = np.asarray(_load_array(cat_path))
        if cat.ndim == 1:
            cat = cat[:, None]
        cat_frame = pd.DataFrame(cat).apply(_normalize_category)
        categorical_indices = list(
            range(sum(frame.shape[1] for frame in frames), sum(frame.shape[1] for frame in frames) + cat_frame.shape[1])
        )
        frames.append(cat_frame)
    X = pd.concat(frames, axis=1, ignore_index=True)
    X.columns = columns or [f"feature_{idx}" for idx in range(X.shape[1])]
    y = np.asarray(_load_array(y_path)).reshape(-1)
    if len(X) != len(y):
        raise ValueError(f"{y_path}: X/y length mismatch {len(X)} != {len(y)}")
    return X, y, categorical_indices


def load_dataset(dataset_dir: Path) -> LoadedDataset:
    info_path = dataset_dir / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    task_type = str(info.get("task_type", "")).lower()
    train_files = _split_files(dataset_dir, "train")
    test_files = _split_files(dataset_dir, "test")
    if train_files is None or test_files is None:
        raise FileNotFoundError(f"{dataset_dir}: missing train/test split")
    X_train, y_train, categorical_indices = _load_split(train_files)
    X_test, y_test, _ = _load_split(test_files, columns=list(X_train.columns))
    val_files = _split_files(dataset_dir, "val")
    X_val = y_val = None
    if val_files is not None:
        X_val, y_val, _ = _load_split(val_files, columns=list(X_train.columns))
    return LoadedDataset(
        name=dataset_dir.name,
        path=dataset_dir,
        task_type=task_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        categorical_indices=categorical_indices,
    )


class OfficialLoCalPFNPreprocessor:
    """Port LoCalPFN's TabZilla preprocessing without using any labels.

    Category vocabularies are intentionally learned from all feature splits,
    matching ``dataset.py`` in the pinned LoCalPFN source.  StandardScaler is
    still fit only on the training rows.
    """

    def __init__(
        self,
        *,
        clipping_value: float,
        disable_normalize_data: bool,
        use_one_hot_emb: bool,
        onehot_retrieval: bool,
    ) -> None:
        self.clipping_value = float(clipping_value)
        self.disable_normalize_data = bool(disable_normalize_data)
        self.use_one_hot_emb = bool(use_one_hot_emb)
        self.onehot_retrieval = bool(onehot_retrieval)
        self.category_values_: dict[int, list[str]] = {}
        self.category_dimensions_: dict[int, int] = {}
        self.raw_scaler_: Any = None
        self.onehot_scaler_: Any = None
        self.fitted_rows: int = 0
        self.vocabulary_rows: int = 0

    def _ordinal_matrix(
        self,
        X: pd.DataFrame,
        categorical_indices: Sequence[int],
        *,
        fit_vocab: bool,
    ) -> np.ndarray:
        categorical = set(int(index) for index in categorical_indices)
        columns: list[np.ndarray] = []
        for index in range(X.shape[1]):
            if index not in categorical:
                columns.append(
                    pd.to_numeric(X.iloc[:, index], errors="coerce")
                    .to_numpy(dtype=np.float64)
                    .reshape(-1, 1)
                )
                continue
            values = _normalize_category(X.iloc[:, index])
            if fit_vocab:
                vocabulary = sorted(values.unique().tolist())
                self.category_values_[index] = vocabulary
                self.category_dimensions_[index] = len(vocabulary)
            vocabulary = self.category_values_[index]
            mapping = {value: position for position, value in enumerate(vocabulary)}
            encoded = values.map(mapping)
            if encoded.isna().any():
                unknown = sorted(values[encoded.isna()].unique().tolist())
                raise ValueError(
                    f"unseen categorical values after all-X vocabulary fit: {unknown}"
                )
            columns.append(encoded.to_numpy(dtype=np.float64).reshape(-1, 1))
        return np.concatenate(columns, axis=1)

    def _one_hot_matrix(
        self,
        ordinal: np.ndarray,
        categorical_indices: Sequence[int],
        *,
        model_embedding: bool,
    ) -> tuple[np.ndarray, list[int]]:
        categorical = set(int(index) for index in categorical_indices)
        current_dimension = int(ordinal.shape[1])
        columns: list[np.ndarray] = []
        retained_categorical: list[int] = []
        output_position = 0
        for index in range(ordinal.shape[1]):
            if index not in categorical:
                columns.append(ordinal[:, [index]])
                output_position += 1
                continue
            category_dimension = self.category_dimensions_[index]
            should_expand = (
                current_dimension + category_dimension - 1 < 100
                if model_embedding
                else category_dimension < 100
            )
            if should_expand:
                encoded = np.eye(category_dimension, dtype=np.float64)[
                    ordinal[:, index].astype(int)
                ]
                columns.append(encoded)
                current_dimension += category_dimension - 1
                output_position += category_dimension
            else:
                columns.append(ordinal[:, [index]])
                retained_categorical.append(output_position)
                output_position += 1
        return np.concatenate(columns, axis=1), retained_categorical

    def fit_transform(
        self,
        loaded: LoadedDataset,
    ) -> PreparedLoCalData:
        from sklearn.preprocessing import StandardScaler

        feature_splits = [loaded.X_train]
        if loaded.X_val is not None:
            feature_splits.append(loaded.X_val)
        feature_splits.append(loaded.X_test)
        all_X = pd.concat(feature_splits, ignore_index=True)
        self.vocabulary_rows = len(all_X)
        self._ordinal_matrix(
            all_X,
            loaded.categorical_indices,
            fit_vocab=True,
        )
        ordinal_train = self._ordinal_matrix(
            loaded.X_train,
            loaded.categorical_indices,
            fit_vocab=False,
        )
        ordinal_val = (
            self._ordinal_matrix(
                loaded.X_val,
                loaded.categorical_indices,
                fit_vocab=False,
            )
            if loaded.X_val is not None
            else None
        )
        ordinal_test = self._ordinal_matrix(
            loaded.X_test,
            loaded.categorical_indices,
            fit_vocab=False,
        )
        onehot_train, onehot_categorical = self._one_hot_matrix(
            ordinal_train,
            loaded.categorical_indices,
            model_embedding=self.use_one_hot_emb,
        )
        onehot_val = (
            self._one_hot_matrix(
                ordinal_val,
                loaded.categorical_indices,
                model_embedding=self.use_one_hot_emb,
            )[0]
            if ordinal_val is not None
            else None
        )
        onehot_test = self._one_hot_matrix(
            ordinal_test,
            loaded.categorical_indices,
            model_embedding=self.use_one_hot_emb,
        )[0]
        retrieval_onehot_train, _ = self._one_hot_matrix(
            ordinal_train,
            loaded.categorical_indices,
            model_embedding=False,
        )
        retrieval_onehot_val = (
            self._one_hot_matrix(
                ordinal_val,
                loaded.categorical_indices,
                model_embedding=False,
            )[0]
            if ordinal_val is not None
            else None
        )
        retrieval_onehot_test = self._one_hot_matrix(
            ordinal_test,
            loaded.categorical_indices,
            model_embedding=False,
        )[0]
        self.fitted_rows = len(ordinal_train)

        def scale_splits(
            train: np.ndarray,
            val: np.ndarray | None,
            test: np.ndarray,
            *,
            scaler_name: str,
        ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
            if self.disable_normalize_data:
                return train, val, test
            scaler = StandardScaler().fit(train)
            setattr(self, scaler_name, scaler)
            train_scaled = scaler.transform(train)
            val_scaled = scaler.transform(val) if val is not None else None
            test_scaled = scaler.transform(test)
            clip = self.clipping_value
            return (
                np.clip(train_scaled, -clip, clip),
                None if val_scaled is None else np.clip(val_scaled, -clip, clip),
                np.clip(test_scaled, -clip, clip),
            )

        raw_train, raw_val, raw_test = scale_splits(
            ordinal_train,
            ordinal_val,
            ordinal_test,
            scaler_name="raw_scaler_",
        )
        model_onehot_train, model_onehot_val, model_onehot_test = scale_splits(
            onehot_train,
            onehot_val,
            onehot_test,
            scaler_name="onehot_scaler_",
        )
        retrieval_onehot_train, retrieval_onehot_val, retrieval_onehot_test = (
            scale_splits(
                retrieval_onehot_train,
                retrieval_onehot_val,
                retrieval_onehot_test,
                scaler_name="retrieval_onehot_scaler_",
            )
        )
        if self.use_one_hot_emb:
            model_train, model_val, model_test = (
                model_onehot_train,
                model_onehot_val,
                model_onehot_test,
            )
            model_categorical = onehot_categorical
        else:
            model_train, model_val, model_test = raw_train, raw_val, raw_test
            model_categorical = list(loaded.categorical_indices)
        if self.onehot_retrieval:
            retrieval_train, retrieval_val, retrieval_test = (
                retrieval_onehot_train,
                retrieval_onehot_val,
                retrieval_onehot_test,
            )
        else:
            retrieval_train, retrieval_val, retrieval_test = (
                model_train,
                model_val,
                model_test,
            )

        def frame(value: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(np.asarray(value, dtype=np.float32))

        return PreparedLoCalData(
            X_train=frame(model_train),
            X_val=None if model_val is None else frame(model_val),
            X_test=frame(model_test),
            retrieval_train=np.asarray(retrieval_train, dtype=np.float32),
            retrieval_val=(
                None
                if retrieval_val is None
                else np.asarray(retrieval_val, dtype=np.float32)
            ),
            retrieval_test=np.asarray(retrieval_test, dtype=np.float32),
            # The pinned LoCalPFN PFN consumes the ordinal/one-hot matrix as
            # numeric features and does not pass categorical metadata onward.
            categorical_indices=[],
        )


# Compatibility name for downstream imports; its behavior is now official LoCalPFN.
RetrievalPreprocessor = OfficialLoCalPFNPreprocessor


class NeighborIndex:
    def __init__(self, backend: Literal["auto", "faiss", "sklearn"] = "auto") -> None:
        self.requested_backend = backend
        self.backend: str | None = None
        self.index: Any = None
        self.X: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NeighborIndex":
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.X = X
        if self.requested_backend in {"auto", "faiss"}:
            try:
                import faiss

                self.index = faiss.IndexFlatL2(X.shape[1])
                self.index.add(X)
                self.backend = "faiss"
                return self
            except ImportError:
                raise RuntimeError(
                    "official LoCalPFN retrieval requires FAISS; install faiss or "
                    "explicitly request --retrieval-backend sklearn for a "
                    "non-reference diagnostic run"
                )
        from sklearn.neighbors import NearestNeighbors

        self.index = NearestNeighbors(algorithm="brute", metric="euclidean").fit(X)
        self.backend = "sklearn"
        return self

    def query(self, X_query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None or self.index is None:
            raise RuntimeError("NeighborIndex must be fit before query")
        k = min(max(1, int(k)), len(self.X))
        X_query = np.ascontiguousarray(X_query, dtype=np.float32)
        if self.backend == "faiss":
            distances, indices = self.index.search(X_query, k)
            return np.sqrt(np.maximum(distances, 0.0)), indices
        distances, indices = self.index.kneighbors(X_query, n_neighbors=k)
        return np.asarray(distances), np.asarray(indices)


class LocalContextBuilder:
    def __init__(
        self,
        train_space: np.ndarray,
        y_train: np.ndarray,
        *,
        context_size: int,
        train_query_length: int,
        backend: Literal["auto", "faiss", "sklearn"],
        seed: int,
    ) -> None:
        self.train_space = np.asarray(train_space, dtype=np.float32)
        self.y_train = np.asarray(y_train)
        self.context_size = max(1, int(context_size))
        self.train_query_length = max(1, int(train_query_length))
        self.seed = int(seed)
        self.index = NeighborIndex(backend).fit(self.train_space)
        self.selection_counts = np.ones(len(self.y_train), dtype=np.int64)

    @property
    def backend(self) -> str:
        assert self.index.backend is not None
        return self.index.backend

    def episodes_for_external(
        self,
        query_space: np.ndarray,
        *,
        query_offset: int = 0,
    ) -> list[ContextEpisode]:
        _, neighbors = self.index.query(
            np.asarray(query_space, dtype=np.float32),
            min(self.context_size, len(self.y_train)),
        )
        return [
            ContextEpisode(
                context_indices=np.asarray(candidate, dtype=int),
                query_indices=np.asarray([query_offset + row_idx], dtype=int),
            )
            for row_idx, candidate in enumerate(neighbors)
        ]

    def training_batches(
        self,
        *,
        epoch: int,
        max_steps: int,
        batch_size: int,
        better_selection: bool,
        exact_knn: bool,
    ) -> list[list[ContextEpisode]]:
        """Create official K+Q anchor-neighborhood batches for one epoch."""

        if better_selection and batch_size != 2:
            raise ValueError("official better_selection requires train batch size 2")
        rng = np.random.default_rng(self.seed + int(epoch))
        if better_selection:
            self.selection_counts = np.maximum(self.selection_counts, 1)
            anchor_batches = [
                rng.choice(
                    len(self.y_train),
                    size=2,
                    replace=True,
                    p=self.selection_counts / self.selection_counts.sum(),
                )
                for _ in range(max_steps)
            ]
        else:
            order = rng.permutation(len(self.y_train))
            anchor_batches = [
                order[start : start + batch_size]
                for start in range(0, len(order), batch_size)
            ][:max_steps]
        retrieval_length = min(
            len(self.y_train),
            self.context_size + self.train_query_length,
        )
        batches: list[list[ContextEpisode]] = []
        for anchors in anchor_batches:
            if len(anchors) == 0:
                continue
            _, nearest = self.index.query(
                self.train_space[np.asarray(anchors, dtype=int)],
                retrieval_length,
            )
            if better_selection:
                for neighbors in nearest:
                    self.selection_counts[np.asarray(neighbors, dtype=int)] -= 1
            if exact_knn:
                if retrieval_length != self.context_size + 1:
                    raise ValueError(
                        "--local-exact-knn requires K+1 retrieved rows; set "
                        "--local-train-query-length 1 and ensure n_train >= K+1"
                    )
                permutation = np.arange(retrieval_length - 1, -1, -1)
            else:
                permutation = rng.permutation(retrieval_length)
            eval_position = min(self.context_size, retrieval_length - 1)
            batch: list[ContextEpisode] = []
            for neighbors in nearest:
                sequence = np.asarray(neighbors, dtype=int)[permutation]
                batch.append(
                    ContextEpisode(
                        context_indices=sequence[:eval_position],
                        query_indices=sequence[eval_position:],
                    )
                )
            batches.append(batch)
        return batches


def prefix_context_query_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    context_size: int,
    **_unused: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the explicit LoCalPFN prefix split used by TabPFNv3."""

    split_at = int(context_size)
    return X[:split_at], X[split_at:], y[:split_at], y[split_at:]


def set_full_finetune(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


class ModelEngine:
    def adapt(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        builder: LocalContextBuilder,
        *,
        args: argparse.Namespace,
        validation_episodes: Sequence[ContextEpisode] | None,
    ) -> AdaptationResult:
        raise NotImplementedError

    def predict_local(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        episodes: Sequence[ContextEpisode],
        *,
        args: argparse.Namespace,
    ) -> np.ndarray:
        raise NotImplementedError

def _tabicl_training_item(
    classifier: Any,
    X_encoded: np.ndarray,
    y_encoded: np.ndarray,
    episode: ContextEpisode,
    *,
    n_estimators: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    try:
        from tabicl._sklearn.preprocessing import EnsembleGenerator
    except ImportError:
        from tabicl.sklearn.preprocessing import EnsembleGenerator
    import torch

    context = np.asarray(episode.context_indices, dtype=int)
    query = np.asarray(episode.query_indices, dtype=int)
    indices = np.concatenate([context, query])
    labels = np.asarray(y_encoded[indices], dtype=int)
    generator = EnsembleGenerator(
        classification=True,
        n_estimators=int(n_estimators),
        norm_methods=getattr(classifier, "norm_methods", None),
        feat_shuffle_method=getattr(classifier, "feat_shuffle_method", "latin"),
        # LoCalPFN feeds the globally encoded labels directly.  Disabling the
        # estimator-only class permutation preserves that global label space.
        class_shuffle_method="none",
        outlier_threshold=getattr(classifier, "outlier_threshold", 4.0),
        random_state=int(seed),
    )
    generator.fit(X_encoded[indices], labels)
    # Older TabICLv2 checkpoints expose ``transform(X, mode=...)`` without a
    # default for X even though train mode uses the fitted matrix internally.
    variants = generator.transform(X_encoded[indices], mode="train")
    X_views: list[np.ndarray] = []
    y_context_views: list[np.ndarray] = []
    y_query_views: list[np.ndarray] = []
    for _norm_method, (X_variant, y_variant) in variants.items():
        X_views.append(X_variant)
        y_context_views.append(y_variant[:, : len(context)])
        y_query_views.append(y_variant[:, len(context) :])
    return (
        torch.from_numpy(np.concatenate(X_views, axis=0)).float(),
        torch.from_numpy(np.concatenate(y_context_views, axis=0)).float(),
        torch.from_numpy(np.concatenate(y_query_views, axis=0)).long(),
    )


class TabICLNativeEngine(ModelEngine):
    def __init__(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        args: argparse.Namespace,
        device: str,
    ) -> None:
        from tabicl import TabICLClassifier

        self.device = device
        self.classifier = TabICLClassifier(
            model_path=str(Path(args.tabicl_model_path).expanduser().resolve()),
            checkpoint_version=Path(args.tabicl_model_path).name,
            allow_auto_download=False,
            device=device,
            n_estimators=int(args.n_estimators),
            batch_size=int(args.local_inference_batch_size),
            random_state=int(args.seed),
            kv_cache=False,
            use_amp=args.use_amp,
        )
        self.classifier.fit(prepared.X_train, loaded.y_train)
        self.model = getattr(self.classifier.model_, "module", self.classifier.model_)

    def _prepare_inference_item(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        episode: ContextEpisode,
        X_query: pd.DataFrame,
        *,
        seed: int,
    ) -> tuple[Any, Any, np.ndarray]:
        try:
            from tabicl._sklearn.preprocessing import EnsembleGenerator
        except ImportError:
            from tabicl.sklearn.preprocessing import EnsembleGenerator
        import torch

        context = np.asarray(episode.context_indices, dtype=int)
        query = np.asarray(episode.query_indices, dtype=int)
        context_labels = np.asarray(loaded.y_train[context])
        local_classes = np.unique(context_labels)
        if len(local_classes) < 2:
            raise ValueError(
                "TabICLv2 does not support a single-class LoCalPFN context"
            )
        local_mapping = {
            label: position for position, label in enumerate(local_classes.tolist())
        }
        local_y = np.asarray(
            [local_mapping[label] for label in context_labels],
            dtype=int,
        )
        X_context_encoded = self.classifier.X_encoder_.transform(
            prepared.X_train.iloc[context]
        )
        X_query_encoded = self.classifier.X_encoder_.transform(
            X_query.iloc[query]
        )
        generator = EnsembleGenerator(
            classification=True,
            n_estimators=int(self.classifier.n_estimators),
            norm_methods=getattr(self.classifier, "norm_methods", None),
            feat_shuffle_method=getattr(
                self.classifier,
                "feat_shuffle_method",
                "latin",
            ),
            class_shuffle_method="none",
            outlier_threshold=getattr(self.classifier, "outlier_threshold", 4.0),
            random_state=int(seed),
        )
        generator.fit(X_context_encoded, local_y)
        variants = generator.transform(X_query_encoded, mode="both")
        X_views = np.concatenate(
            [value[0] for value in variants.values()],
            axis=0,
        )
        y_views = np.concatenate(
            [value[1] for value in variants.values()],
            axis=0,
        )
        return (
            torch.from_numpy(X_views).float(),
            torch.from_numpy(y_views).float(),
            local_classes,
        )

    def _predict_with_episodes(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        episodes: Sequence[ContextEpisode],
        *,
        X_query: pd.DataFrame,
        inference_batch_size: int,
    ) -> np.ndarray:
        import torch

        global_classes = np.unique(loaded.y_train)
        global_mapping = {
            label: position for position, label in enumerate(global_classes.tolist())
        }
        output = np.zeros((len(X_query), len(global_classes)), dtype=np.float64)
        self.model.eval()
        batch_size = max(1, int(inference_batch_size))
        start = 0
        retried = False
        while start < len(episodes):
            current_episodes = list(episodes[start : start + batch_size])
            prepared_items = [
                self._prepare_inference_item(
                    loaded,
                    prepared,
                    episode,
                    X_query,
                    seed=int(self.classifier.random_state) + start + offset,
                )
                for offset, episode in enumerate(current_episodes)
            ]
            groups: dict[tuple[int, int, int, int], list[tuple[int, Any, Any, np.ndarray]]] = defaultdict(list)
            for offset, (X_item, y_item, classes) in enumerate(prepared_items):
                key = (
                    int(X_item.shape[0]),
                    int(X_item.shape[1]),
                    int(X_item.shape[2]),
                    len(classes),
                )
                groups[key].append((offset, X_item, y_item, classes))
            try:
                for items in groups.values():
                    X_batch = torch.cat([item[1] for item in items], dim=0).to(
                        self.classifier.device_
                    )
                    y_batch = torch.cat([item[2] for item in items], dim=0).to(
                        self.classifier.device_
                    )
                    with torch.inference_mode():
                        logits = self.model(
                            X_batch,
                            y_batch,
                            return_logits=True,
                            softmax_temperature=getattr(
                                self.classifier,
                                "softmax_temperature",
                                0.9,
                            ),
                        )
                        probabilities = torch.softmax(
                            logits
                            / float(
                                getattr(
                                    self.classifier,
                                    "softmax_temperature",
                                    0.9,
                                )
                            ),
                            dim=-1,
                        )
                    views_per_episode = int(items[0][1].shape[0])
                    probabilities = probabilities.reshape(
                        len(items),
                        views_per_episode,
                        probabilities.shape[-2],
                        probabilities.shape[-1],
                    ).mean(dim=1)
                    for item_position, (offset, _X, _y, classes) in enumerate(items):
                        query_indices = current_episodes[offset].query_indices
                        local = probabilities[item_position].detach().cpu().numpy()
                        for local_position, label in enumerate(classes):
                            output[query_indices, global_mapping[label]] = local[
                                :,
                                local_position,
                            ]
            except Exception as exc:
                if not is_oom_exception(exc) or retried or batch_size == 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                retried = True
                continue
            start += len(current_episodes)
        return output

    @staticmethod
    def _stack_training_batch(
        items: Sequence[tuple[Any, Any, Any]],
    ) -> tuple[Any, Any, Any]:
        import torch
        import torch.nn.functional as F

        max_features = max(int(item[0].shape[-1]) for item in items)
        X_items = [
            F.pad(item[0], (0, max_features - int(item[0].shape[-1])))
            for item in items
        ]
        return (
            torch.cat(X_items, dim=0),
            torch.cat([item[1] for item in items], dim=0),
            torch.cat([item[2] for item in items], dim=0),
        )

    def _evaluate_split(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        builder: LocalContextBuilder,
        split: str,
        args: argparse.Namespace,
    ) -> tuple[float, dict[str, float | None]]:
        if split == "valid":
            if prepared.X_val is None or loaded.y_val is None or builder is None:
                raise ValueError("validation split requested but unavailable")
            X_query = prepared.X_val
            y_true = loaded.y_val
            retrieval = prepared.retrieval_val
        elif split == "test":
            X_query = prepared.X_test
            y_true = loaded.y_test
            retrieval = prepared.retrieval_test
        elif split == "train":
            X_query = prepared.X_train
            y_true = loaded.y_train
            retrieval = prepared.retrieval_train
        else:
            raise ValueError(f"unsupported evaluated split: {split}")
        assert retrieval is not None
        episodes = builder.episodes_for_external(retrieval)
        proba = self._predict_with_episodes(
            loaded,
            prepared,
            episodes,
            X_query=X_query,
            inference_batch_size=int(args.local_inference_batch_size),
        )
        metrics = _classification_metrics(y_true, proba, np.unique(loaded.y_train))
        metric = _early_stopping_value(
            args.local_early_stopping_metric,
            metrics,
        )
        return metric, metrics

    def adapt(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        builder: LocalContextBuilder,
        *,
        args: argparse.Namespace,
        validation_episodes: Sequence[ContextEpisode] | None,
    ) -> AdaptationResult:
        import torch
        import torch.nn.functional as F

        set_full_finetune(self.model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=float(args.local_lr),
            weight_decay=float(args.local_weight_decay),
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(args.local_epochs)
                * int(args.local_max_steps_per_epoch),
                eta_min=float(args.local_lr) / 10,
            )
            if args.local_scheduler
            else None
        )
        trainable, _total, ratio = count_parameters(self.model)
        X_encoded = self.classifier.X_encoder_.transform(prepared.X_train)
        y_encoded = self.classifier.y_encoder_.transform(loaded.y_train)
        device = self.classifier.device_
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        best_score = -math.inf
        best_step = 0
        best_epoch = 0
        steps = 0
        last_loss: float | None = None
        stopped_early = False
        history: dict[str, list[float]] = defaultdict(list)
        started = time.time()
        evaluated_splits = {
            "valid": ["valid"],
            "valid_test": ["valid", "test"],
            "all": ["valid", "test", "train"],
        }[args.local_splits_evaluated]
        try:
            for epoch in range(int(args.local_epochs)):
                if epoch % int(args.local_eval_interval) == 0:
                    for split in evaluated_splits:
                        metric, metrics = self._evaluate_split(
                            loaded,
                            prepared,
                            builder,
                            split,
                            args,
                        )
                        for name, value in metrics.items():
                            if value is not None:
                                history[f"{name}_{split}"].append(float(value))
                        if split == "valid" and metric > best_score:
                            best_score = metric
                            best_epoch = epoch
                            best_step = steps
                            best_state = {
                                key: value.detach().cpu().clone()
                                for key, value in self.model.state_dict().items()
                            }
                if best_epoch + int(args.local_early_stopping_rounds) < epoch:
                    stopped_early = True
                    break
                self.model.train()
                epoch_batches = builder.training_batches(
                    epoch=epoch,
                    max_steps=int(args.local_max_steps_per_epoch),
                    batch_size=int(args.local_train_batch_size),
                    better_selection=bool(args.local_better_selection),
                    exact_knn=bool(args.local_exact_knn),
                )
                for batch_index, episode_batch in enumerate(epoch_batches):
                    items = [
                        _tabicl_training_item(
                            self.classifier,
                            X_encoded,
                            y_encoded,
                            episode,
                            n_estimators=int(args.train_n_estimators),
                            seed=int(args.seed) + epoch * 10_000 + batch_index,
                        )
                        for episode in episode_batch
                    ]
                    X_batch, y_context, y_query = self._stack_training_batch(items)
                    X_batch = X_batch.to(device)
                    y_context = y_context.to(device)
                    y_query = y_query.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(X_batch, y_context)
                    n_classes = len(np.unique(loaded.y_train))
                    logits = logits[..., :n_classes].reshape(-1, n_classes)
                    loss = F.cross_entropy(logits, y_query.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    steps += 1
                    last_loss = float(loss.detach().cpu())
                if last_loss is not None:
                    history["loss_train"].append(last_loss)
            if validation_episodes is not None:
                self.model.load_state_dict(best_state)
            if args.local_save_data:
                data_dir = Path(args.out_dir) / "data" / loaded.name
                data_dir.mkdir(parents=True, exist_ok=True)
                np.savez(data_dir / "saved_data.npz", **history)
        finally:
            self.model.eval()
        self.best_epoch = best_epoch
        return AdaptationResult(
            steps=steps,
            loss=last_loss,
            seconds=time.time() - started,
            trainable_params=trainable,
            trainable_ratio=ratio,
            best_step=best_step,
            stopped_early=stopped_early,
        )

    def predict_local(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        episodes: Sequence[ContextEpisode],
        *,
        args: argparse.Namespace,
    ) -> np.ndarray:
        return self._predict_with_episodes(
            loaded,
            prepared,
            episodes,
            X_query=prepared.X_test,
            inference_batch_size=int(args.local_inference_batch_size),
        )


def _build_tabpfn_finetuned_classifier(
    *,
    loaded: LoadedDataset,
    prepared: PreparedLoCalData,
    args: argparse.Namespace,
    device: str,
    episode_size: int,
) -> Any:
    from tabpfn import TabPFNClassifier
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    model_path = (
        args.tabpfn_v3_binary_model_path
        if loaded.task_type == "binclass"
        else args.tabpfn_v3_multiclass_model_path
    )

    class LocalContextFinetunedClassifier(FinetunedTabPFNClassifier):
        def _forward_with_loss(self, batch: Any) -> Any:
            from tabpfn.finetuning.finetuned_classifier import (
                _compute_classification_loss,
            )

            logits = self._training_forward(
                batch.X_query,
                return_raw_logits=True,
            )
            query_count, batch_size, estimators, n_classes = logits.shape
            if batch.y_query.shape != (batch_size, query_count):
                raise ValueError(
                    "unexpected TabPFNv3 LoCalPFN target shape: "
                    f"{tuple(batch.y_query.shape)} != {(batch_size, query_count)}"
                )
            logits = logits.permute(1, 2, 3, 0).reshape(
                batch_size * estimators,
                n_classes,
                query_count,
            )
            targets = batch.y_query.repeat_interleave(estimators, dim=0).to(
                self.device
            )
            loss = _compute_classification_loss(
                logits_BLQ=logits,
                targets_BQ=targets,
            )
            self._local_context_optimizer_steps_ = (
                int(getattr(self, "_local_context_optimizer_steps_", 0)) + 1
            )
            return loss

        def _should_skip_batch(self, batch: Any) -> bool:
            # The official LoCalPFN code does not repair or skip neighborhoods
            # whose context omits a query class.  The episode preprocessor has
            # already seen the supervised query labels and keeps the global
            # output space available to the loss.
            return False

        def _create_estimator(self, config: dict[str, Any]) -> Any:
            return TabPFNClassifier(
                **config,
                fit_mode="batched",
                differentiable_input=False,
            )

        def _configure_model_for_optimization(self, model: Any) -> None:
            set_full_finetune(model)

        def _evaluate_model(
            self,
            eval_config: dict[str, Any],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
        ) -> Any:
            callback = getattr(self, "_local_context_eval_callback", None)
            if callback is None:
                return super()._evaluate_model(
                    eval_config,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )
            return callback(self.finetuned_estimator_, eval_config)

        def _get_initial_best_metric(self) -> float:
            return float(
                getattr(self, "_local_context_initial_metric", -math.inf)
            )

    classifier_kwargs: dict[str, Any] = {
        "model_path": str(Path(model_path).expanduser().resolve()),
        "n_estimators": int(args.n_estimators),
        "ignore_pretraining_limits": True,
    }
    if prepared.categorical_indices:
        classifier_kwargs["categorical_features_indices"] = (
            prepared.categorical_indices
        )
    eval_metric = {
        "auc": "roc_auc",
        "negloss": "log_loss",
        "acc": "acc",
        "f1": "acc",
    }[args.local_early_stopping_metric]
    classifier = LocalContextFinetunedClassifier(
        device=device,
        epochs=int(args.local_epochs),
        learning_rate=float(args.local_lr),
        weight_decay=float(args.local_weight_decay),
        validation_split_ratio=0.0,
        n_finetune_ctx_plus_query_samples=max(2, int(episode_size)),
        finetune_ctx_query_split_ratio=(
            float(args.local_train_query_length) / max(2, int(episode_size))
        ),
        random_state=int(args.seed),
        early_stopping=prepared.X_val is not None,
        early_stopping_patience=int(args.local_early_stopping_rounds),
        min_delta=0.0,
        grad_clip_value=0.0,
        use_lr_scheduler=bool(args.local_scheduler),
        n_estimators_finetune=int(args.train_n_estimators),
        n_estimators_validation=1,
        n_estimators_final_inference=int(args.n_estimators),
        use_activation_checkpointing=False,
        gradient_accumulation_steps=1,
        extra_classifier_kwargs=classifier_kwargs,
        eval_metric=eval_metric,
    )
    classifier.meta_batch_size = int(args.local_train_batch_size)
    return classifier


class TabPFNv3Engine(ModelEngine):
    def __init__(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        args: argparse.Namespace,
        device: str,
    ) -> None:
        self.device = device
        self.args = args
        self.loaded = loaded
        self.prepared = prepared
        self.finetuned: Any = None
        self.inference_classifier: Any = None

    def _make_base_classifier(self, loaded: LoadedDataset, args: argparse.Namespace) -> Any:
        from tabpfn import TabPFNClassifier

        model_path = (
            args.tabpfn_v3_binary_model_path
            if loaded.task_type == "binclass"
            else args.tabpfn_v3_multiclass_model_path
        )
        kwargs: dict[str, Any] = {
            "model_path": str(Path(model_path).expanduser().resolve()),
            "device": self.device,
            "n_estimators": int(args.n_estimators),
            "ignore_pretraining_limits": True,
        }
        if self.prepared.categorical_indices:
            kwargs["categorical_features_indices"] = (
                self.prepared.categorical_indices
            )
        return TabPFNClassifier(**kwargs)

    def _predict_with_classifier(
        self,
        classifier: Any,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        episodes: Sequence[ContextEpisode],
        X_query: pd.DataFrame,
        *,
        args: argparse.Namespace,
    ) -> np.ndarray:
        import torch

        global_classes = np.unique(loaded.y_train)
        global_mapping = {
            label: position for position, label in enumerate(global_classes.tolist())
        }
        output = np.zeros((len(X_query), len(global_classes)), dtype=float)
        groups: dict[
            tuple[int, tuple[Any, ...], int],
            list[ContextEpisode],
        ] = defaultdict(list)
        for episode in episodes:
            context = np.asarray(episode.context_indices, dtype=int)
            local_classes = tuple(
                np.unique(loaded.y_train[context]).tolist()
            )
            if len(local_classes) < 2:
                raise ValueError(
                    "TabPFNv3 does not support a single-class LoCalPFN context"
                )
            groups[
                (
                    len(context),
                    local_classes,
                    len(episode.query_indices),
                )
            ].append(episode)
        for (_context_size, local_classes, _query_size), group in groups.items():
            batch_size = max(1, int(args.local_inference_batch_size))
            start = 0
            retried = False
            while start < len(group):
                current = group[start : start + batch_size]
                contexts = [
                    prepared.X_train.iloc[episode.context_indices].to_numpy()
                    for episode in current
                ]
                labels = [
                    loaded.y_train[episode.context_indices]
                    for episode in current
                ]
                queries = [
                    X_query.iloc[episode.query_indices].to_numpy()
                    for episode in current
                ]
                try:
                    probabilities = classifier.predict_proba_batched(
                        contexts,
                        labels,
                        queries,
                    )
                except Exception as exc:
                    if not is_oom_exception(exc) or retried or batch_size == 1:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    retried = True
                    continue
                probabilities = np.asarray(probabilities)
                for position, episode in enumerate(current):
                    for local_position, label in enumerate(local_classes):
                        output[
                            episode.query_indices,
                            global_mapping[label],
                        ] = probabilities[position, :, local_position]
                start += len(current)
        return output

    def _eval_result(
        self,
        estimator: Any,
        eval_config: dict[str, Any],
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        builder: LocalContextBuilder,
        args: argparse.Namespace,
    ) -> Any:
        from tabpfn import TabPFNClassifier
        from tabpfn.finetuning.finetuned_base import EvalResult
        from tabpfn.finetuning.train_util import clone_model_for_evaluation

        classifier = clone_model_for_evaluation(
            estimator,
            eval_config,
            TabPFNClassifier,
        )
        split_payloads = {
            "valid": (
                prepared.X_val,
                loaded.y_val,
                prepared.retrieval_val,
            ),
            "test": (
                prepared.X_test,
                loaded.y_test,
                prepared.retrieval_test,
            ),
            "train": (
                prepared.X_train,
                loaded.y_train,
                prepared.retrieval_train,
            ),
        }
        evaluated_splits = {
            "valid": ["valid"],
            "valid_test": ["valid", "test"],
            "all": ["valid", "test", "train"],
        }[args.local_splits_evaluated]
        validation_metrics: dict[str, float | None] | None = None
        for split in evaluated_splits:
            X_query, y_true, retrieval = split_payloads[split]
            if X_query is None or y_true is None or retrieval is None:
                continue
            episodes = builder.episodes_for_external(retrieval)
            proba = self._predict_with_classifier(
                classifier,
                loaded,
                prepared,
                episodes,
                X_query,
                args=args,
            )
            metrics = _classification_metrics(
                y_true,
                proba,
                np.unique(loaded.y_train),
            )
            if split == "valid":
                validation_metrics = metrics
        if validation_metrics is None:
            raise ValueError("LoCalPFN validation split is unavailable")
        primary = _early_stopping_value(
            args.local_early_stopping_metric,
            validation_metrics,
        )
        return EvalResult(
            primary=primary,
            secondary={
                "log_loss": float(validation_metrics["log_loss"] or np.nan),
                "roc_auc": float(validation_metrics["roc_auc"] or np.nan),
                "accuracy": float(validation_metrics["accuracy"] or np.nan),
                "f1": float(validation_metrics["f1"] or np.nan),
            },
        )

    def adapt(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        builder: LocalContextBuilder,
        *,
        args: argparse.Namespace,
        validation_episodes: Sequence[ContextEpisode] | None,
    ) -> AdaptationResult:
        initial_batches = builder.training_batches(
            epoch=0,
            max_steps=int(args.local_max_steps_per_epoch),
            batch_size=int(args.local_train_batch_size),
            better_selection=bool(args.local_better_selection),
            exact_knn=bool(args.local_exact_knn),
        )
        initial_episodes = [
            episode for batch in initial_batches for episode in batch
        ]
        if not initial_episodes:
            raise ValueError("LoCalPFN produced no TabPFNv3 training episodes")
        episode_size = max(
            len(episode.context_indices) + len(episode.query_indices)
            for episode in initial_episodes
        )
        self.finetuned = _build_tabpfn_finetuned_classifier(
            loaded=loaded,
            prepared=prepared,
            args=args,
            device=self.device,
            episode_size=episode_size,
        )
        if validation_episodes and loaded.X_val is not None and loaded.y_val is not None:
            base_classifier = self._make_base_classifier(loaded, args)
            baseline_proba = self._predict_with_classifier(
                base_classifier,
                loaded,
                prepared,
                validation_episodes,
                prepared.X_val,
                args=args,
            )
            baseline_metrics = _classification_metrics(
                loaded.y_val,
                baseline_proba,
                np.unique(loaded.y_train),
            )
            self.finetuned._local_context_initial_metric = _early_stopping_value(
                args.local_early_stopping_metric,
                baseline_metrics,
            )
        self.finetuned._local_context_eval_callback = (
            lambda estimator, config: self._eval_result(
                estimator,
                config,
                loaded,
                prepared,
                builder,
                args,
            )
        )
        started = time.time()
        from torch.utils.data import ConcatDataset
        from tabpfn.finetuning import finetuned_base as finetuned_base_module

        original_chunk_builder = finetuned_base_module.get_preprocessed_dataset_chunks

        def episode_preserving_chunk_builder(**kwargs: Any) -> Any:
            collections = []
            base_seed = int(kwargs.get("data_shuffle_seed", args.seed))
            epoch = max(0, base_seed - int(args.seed))
            batches = builder.training_batches(
                epoch=epoch,
                max_steps=int(args.local_max_steps_per_epoch),
                batch_size=int(args.local_train_batch_size),
                better_selection=bool(args.local_better_selection),
                exact_knn=bool(args.local_exact_knn),
            )
            episodes = [episode for batch in batches for episode in batch]
            for episode_idx, episode in enumerate(episodes):
                context_size = len(episode.context_indices)
                indices = np.concatenate(
                    [episode.context_indices, episode.query_indices]
                ).astype(int)

                def prefix_split(
                    X: np.ndarray,
                    y: np.ndarray,
                    *,
                    split_at: int = context_size,
                    **_unused: Any,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                    return prefix_context_query_split(
                        X,
                        y,
                        context_size=split_at,
                    )

                episode_kwargs = dict(kwargs)
                episode_kwargs.update(
                    {
                        "X_raw": prepared.X_train.iloc[indices].to_numpy(),
                        "y_raw": loaded.y_train[indices],
                        "split_fn": prefix_split,
                        "max_data_size": None,
                        "data_shuffle_seed": base_seed + episode_idx,
                        "shuffle": False,
                        "force_no_stratify": True,
                    }
                )
                collections.append(original_chunk_builder(**episode_kwargs))
            return ConcatDataset(collections)

        finetuned_base_module.get_preprocessed_dataset_chunks = (
            episode_preserving_chunk_builder
        )
        try:
            self.finetuned.fit(
                prepared.X_train.to_numpy(),
                loaded.y_train,
                X_val=prepared.X_val.to_numpy() if prepared.X_val is not None else None,
                y_val=loaded.y_val,
                output_dir=None,
            )
        finally:
            finetuned_base_module.get_preprocessed_dataset_chunks = original_chunk_builder
        self.inference_classifier = self.finetuned.finetuned_inference_classifier_
        trainable, _total, ratio = count_parameters(self.finetuned.finetuned_estimator_.model_)
        actual_steps = int(getattr(self.finetuned, "_local_context_optimizer_steps_", 0))
        if actual_steps <= 0:
            raise RuntimeError("TabPFNv3 finetuning produced no optimizer step")
        best_step = actual_steps
        expected_steps = int(
            int(args.local_epochs)
            * math.ceil(
                len(initial_episodes) / int(args.local_train_batch_size)
            )
        )
        self.best_epoch = int(
            getattr(self.finetuned, "_local_context_best_epoch_", 0)
        )
        return AdaptationResult(
            steps=actual_steps,
            loss=None,
            seconds=time.time() - started,
            trainable_params=trainable,
            trainable_ratio=ratio,
            best_step=best_step,
            stopped_early=actual_steps < expected_steps,
        )

    def predict_local(
        self,
        loaded: LoadedDataset,
        prepared: PreparedLoCalData,
        episodes: Sequence[ContextEpisode],
        *,
        args: argparse.Namespace,
    ) -> np.ndarray:
        if self.inference_classifier is None:
            raise RuntimeError("TabPFNv3 engine must be adapted before prediction")
        return self._predict_with_classifier(
            self.inference_classifier,
            loaded,
            prepared,
            episodes,
            prepared.X_test,
            args=args,
        )


def _classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float | None]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, 0.0, None)
    row_sums = proba.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("model produced a probability row with non-positive sum")
    proba = proba / row_sums
    prediction = classes[np.argmax(proba, axis=1)]
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }
    try:
        if len(classes) == 2:
            encoded = (np.asarray(y_true) == classes[1]).astype(int)
            result["roc_auc"] = float(roc_auc_score(encoded, proba[:, 1]))
        else:
            result["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    proba,
                    labels=classes,
                    multi_class="ovo",
                )
            )
    except ValueError:
        result["roc_auc"] = None
    try:
        result["log_loss"] = float(log_loss(y_true, proba, labels=classes))
    except ValueError:
        result["log_loss"] = None
    return result


def _early_stopping_value(
    metric_name: str,
    metrics: dict[str, float | None],
) -> float:
    key = {
        "negloss": "log_loss",
        "acc": "accuracy",
        "f1": "f1",
        "auc": "roc_auc",
    }[metric_name]
    value = metrics[key]
    if value is None or not np.isfinite(value):
        return -math.inf
    return -float(value) if metric_name == "negloss" else float(value)


def _make_train_and_query_episodes(
    loaded: LoadedDataset,
    *,
    args: argparse.Namespace,
) -> tuple[
    PreparedLoCalData,
    LocalContextBuilder,
    list[ContextEpisode] | None,
    list[ContextEpisode],
    str,
    int,
    RetrievalTiming,
]:
    timing = RetrievalTiming()
    started = time.time()
    preprocessor = OfficialLoCalPFNPreprocessor(
        clipping_value=float(args.local_clipping_value),
        disable_normalize_data=bool(args.local_disable_normalize_data),
        use_one_hot_emb=bool(args.local_use_one_hot_emb),
        onehot_retrieval=bool(args.local_onehot_retrieval),
    )
    prepared = preprocessor.fit_transform(loaded)
    timing.preprocessing_seconds = time.time() - started

    context_size = (
        default_context_size(len(loaded.y_train))
        if args.context_size is None
        else int(args.context_size)
    )
    started = time.time()
    builder = LocalContextBuilder(
        prepared.retrieval_train,
        loaded.y_train,
        context_size=context_size,
        train_query_length=int(args.local_train_query_length),
        backend=args.retrieval_backend,
        seed=args.seed,
    )
    validation_episodes = (
        builder.episodes_for_external(prepared.retrieval_val)
        if prepared.retrieval_val is not None
        else None
    )
    test_episodes = builder.episodes_for_external(prepared.retrieval_test)
    timing.retrieval_seconds = time.time() - started
    return (
        prepared,
        builder,
        validation_episodes,
        test_episodes,
        builder.backend,
        context_size,
        timing,
    )


def evaluate_dataset(
    dataset_dir: Path,
    *,
    args: argparse.Namespace,
    device: str,
) -> DatasetResult:
    started = time.time()
    loaded: LoadedDataset | None = None
    try:
        loaded = load_dataset(dataset_dir)
        if loaded.task_type not in CLASSIFICATION_TASKS:
            return DatasetResult(
                dataset_name=loaded.name,
                dataset_dir=str(loaded.path),
                task_type=loaded.task_type,
                model_family=args.model_family,
                method=args.method,
                config_fingerprint=args.config_fingerprint,
                status="skip",
                error=f"task_type={loaded.task_type!r} is not classification",
                total_seconds=time.time() - started,
            )
        train_classes = np.unique(loaded.y_train)
        if len(train_classes) < 2:
            raise ValueError("training split has fewer than two classes")
        if len(loaded.y_train) <= len(train_classes):
            raise ValueError("training split is too small to form context/query episodes")

        (
            prepared,
            builder,
            validation_episodes,
            test_episodes,
            retrieval_backend,
            context_size,
            timing,
        ) = _make_train_and_query_episodes(loaded, args=args)

        if args.model_family == "tabiclv2":
            engine: ModelEngine = TabICLNativeEngine(
                loaded,
                prepared,
                args,
                device,
            )
        elif args.model_family == "tabpfnv3":
            engine = TabPFNv3Engine(
                loaded,
                prepared,
                args,
                device,
            )
        else:
            raise ValueError(f"single-cell runner got unsupported model {args.model_family!r}")
        adaptation = engine.adapt(
            loaded,
            prepared,
            builder,
            args=args,
            validation_episodes=validation_episodes,
        )
        if adaptation.steps <= 0:
            raise RuntimeError("adaptation completed without an optimizer step")
        predict_started = time.time()
        proba = engine.predict_local(
            loaded,
            prepared,
            test_episodes,
            args=args,
        )
        predict_seconds = time.time() - predict_started
        metrics = _classification_metrics(loaded.y_test, proba, train_classes)
        return DatasetResult(
            dataset_name=loaded.name,
            dataset_dir=str(loaded.path),
            task_type=loaded.task_type,
            model_family=args.model_family,
            method=args.method,
            config_fingerprint=args.config_fingerprint,
            status="ok",
            error=None,
            n_train=len(loaded.y_train),
            n_val=0 if loaded.y_val is None else len(loaded.y_val),
            n_test=len(loaded.y_test),
            n_features=loaded.X_train.shape[1],
            n_classes=len(train_classes),
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            balanced_accuracy=metrics["balanced_accuracy"],
            roc_auc=metrics["roc_auc"],
            log_loss=metrics["log_loss"],
            retrieval_backend=retrieval_backend,
            retrieval_space=(
                "onehot" if args.local_onehot_retrieval else "raw"
            ),
            context_size=context_size,
            train_query_length=int(args.local_train_query_length),
            train_batch_size=int(args.local_train_batch_size),
            inference_batch_size=int(args.local_inference_batch_size),
            best_epoch=int(getattr(engine, "best_epoch", 0)),
            fallback_count=0,
            ft_steps=adaptation.steps,
            ft_loss=adaptation.loss,
            ft_applied=True,
            ft_best_step=adaptation.best_step,
            ft_stopped_early=adaptation.stopped_early,
            trainable_params=adaptation.trainable_params,
            trainable_ratio=adaptation.trainable_ratio,
            retrieval_seconds=timing.preprocessing_seconds + timing.retrieval_seconds,
            routing_seconds=0.0,
            ft_seconds=adaptation.seconds,
            predict_seconds=predict_seconds,
            total_seconds=time.time() - started,
            inference_fallback=False,
        )
    except Exception as exc:
        if is_oom_exception(exc):
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return DatasetResult(
            dataset_name=dataset_dir.name,
            dataset_dir=str(dataset_dir),
            task_type=None if loaded is None else loaded.task_type,
            model_family=args.model_family,
            method=args.method,
            config_fingerprint=args.config_fingerprint,
            status="fail",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            n_train=0 if loaded is None else len(loaded.y_train),
            n_val=0 if loaded is None or loaded.y_val is None else len(loaded.y_val),
            n_test=0 if loaded is None else len(loaded.y_test),
            n_features=0 if loaded is None else loaded.X_train.shape[1],
            n_classes=0 if loaded is None else len(np.unique(loaded.y_train)),
            total_seconds=time.time() - started,
            inference_fallback=False,
        )


def discover_datasets(
    data_root: Path,
    *,
    dataset_names: Sequence[str],
    max_datasets: int | None,
) -> list[Path]:
    if dataset_names:
        paths = [data_root / name for name in dataset_names]
        missing = [str(path) for path in paths if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Dataset directories not found: {missing}")
    else:
        paths = sorted(path for path in data_root.iterdir() if path.is_dir())
    if max_datasets is not None:
        paths = paths[: int(max_datasets)]
    return paths


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _config_fingerprint(args: argparse.Namespace) -> str:
    ignored = {
        "config_fingerprint",
        "data_root",
        "dataset",
        "dry_run",
        "gpus",
        "max_datasets",
        "out_dir",
        "resume",
        "workers",
    }
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in ignored
    }
    payload.update(
        {
            "implementation": LOCALPFN_IMPLEMENTATION,
            "source_commit": LOCALPFN_SOURCE_COMMIT,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_results(out_dir: Path, rows: Sequence[DatasetResult], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    if not frame.empty and frame.duplicated(["dataset_name"]).any():
        duplicates = frame.loc[frame.duplicated(["dataset_name"], keep=False), "dataset_name"].tolist()
        raise RuntimeError(f"duplicate dataset rows detected: {duplicates}")
    csv_path = out_dir / "all_classification_results.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    ok = frame[frame["status"] == "ok"] if not frame.empty else frame
    lines = [
        f"model_family: {args.model_family}",
        f"method: {args.method}",
        f"datasets_total: {len(frame)}",
        f"status_ok: {len(ok)}",
        f"status_fail: {int((frame['status'] == 'fail').sum()) if not frame.empty else 0}",
        f"status_skip: {int((frame['status'] == 'skip').sum()) if not frame.empty else 0}",
    ]
    for metric in ("accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"):
        value = pd.to_numeric(ok[metric], errors="coerce").mean() if not ok.empty else float("nan")
        lines.append(f"mean_{metric}_ok: {value:.8f}" if pd.notna(value) else f"mean_{metric}_ok: nan")
    lines.extend(
        [
            f"ft_steps_total_ok: {int(pd.to_numeric(ok['ft_steps'], errors='coerce').fillna(0).sum()) if not ok.empty else 0}",
            f"inference_fallback_count: {int(ok['inference_fallback'].fillna(False).astype(bool).sum()) if not ok.empty else 0}",
            f"fallback_count_total_ok: {int(pd.to_numeric(ok['fallback_count'], errors='coerce').fillna(0).sum()) if not ok.empty else 0}",
        ]
    )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(
        out_dir / "run_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "script": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_fingerprint": args.config_fingerprint,
            "localpfn_implementation": LOCALPFN_IMPLEMENTATION,
            "localpfn_source_commit": LOCALPFN_SOURCE_COMMIT,
            "arguments": vars(args),
            "rows": len(frame),
            "ok": len(ok),
        },
    )


def _load_checkpoint_rows(
    csv_path: Path,
    *,
    allowed_dataset_names: set[str],
    config_fingerprint: str,
) -> dict[str, DatasetResult]:
    if not csv_path.exists():
        return {}
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return {}
    if frame.empty or "dataset_name" not in frame.columns:
        return {}
    if frame.duplicated(["dataset_name"]).any():
        return {}
    rows: dict[str, DatasetResult] = {}
    for record in frame.to_dict(orient="records"):
        dataset_name = str(record.get("dataset_name", ""))
        if dataset_name not in allowed_dataset_names:
            continue
        if str(record.get("config_fingerprint", "")) != config_fingerprint:
            continue
        try:
            rows[dataset_name] = DatasetResult(**record)
        except TypeError:
            return {}
    return rows


def _ordered_checkpoint_rows(
    dataset_paths: Sequence[Path],
    rows_by_name: dict[str, DatasetResult],
) -> list[DatasetResult]:
    return [
        rows_by_name[path.name]
        for path in dataset_paths
        if path.name in rows_by_name
    ]


def _evaluate_with_checkpoints(
    dataset_paths: Sequence[Path],
    *,
    args: argparse.Namespace,
    device: str,
    checkpoint_csv: Path,
    result_out_dir: Path | None,
) -> list[DatasetResult]:
    allowed_names = {path.name for path in dataset_paths}
    rows_by_name = (
        _load_checkpoint_rows(
            checkpoint_csv,
            allowed_dataset_names=allowed_names,
            config_fingerprint=args.config_fingerprint,
        )
        if args.resume
        else {}
    )
    for dataset_path in dataset_paths:
        existing = rows_by_name.get(dataset_path.name)
        if existing is not None and existing.status == "ok":
            print(f"[resume] reuse status=ok dataset={dataset_path.name}", flush=True)
            continue
        row = evaluate_dataset(dataset_path, args=args, device=device)
        rows_by_name[dataset_path.name] = row
        ordered_rows = _ordered_checkpoint_rows(dataset_paths, rows_by_name)
        if result_out_dir is not None:
            _write_results(result_out_dir, ordered_rows, args)
        else:
            checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_csv.with_name(
                f".{checkpoint_csv.name}.tmp.{os.getpid()}"
            )
            pd.DataFrame([asdict(item) for item in ordered_rows]).to_csv(
                temporary,
                index=False,
            )
            temporary.replace(checkpoint_csv)
        print(
            f"[checkpoint] dataset={dataset_path.name} status={row.status} "
            f"completed={len(ordered_rows)}/{len(dataset_paths)}",
            flush=True,
        )
    return _ordered_checkpoint_rows(dataset_paths, rows_by_name)


def _worker_entry(
    worker_id: int,
    gpu: str,
    dataset_paths: list[str],
    args_dict: dict[str, Any],
    worker_csv: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    args = argparse.Namespace(**args_dict)
    device = "cuda:0" if str(gpu).lower() != "cpu" else "cpu"
    paths = [Path(path) for path in dataset_paths]
    rows = _evaluate_with_checkpoints(
        paths,
        args=args,
        device=device,
        checkpoint_csv=Path(worker_csv),
        result_out_dir=None,
    )
    print(f"[worker {worker_id}] gpu={gpu} datasets={len(rows)} complete", flush=True)


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in str(value).split(",") if item.strip()]
    return gpus or ["0"]


def run_single_cell(args: argparse.Namespace) -> None:
    import multiprocessing as mp

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    dataset_paths = discover_datasets(
        data_root,
        dataset_names=args.dataset,
        max_datasets=args.max_datasets,
    )
    if not dataset_paths:
        raise RuntimeError(f"No datasets found under {data_root}")
    gpus = _parse_gpus(args.gpus)
    worker_count = min(max(1, int(args.workers)), len(gpus), len(dataset_paths))
    out_dir.mkdir(parents=True, exist_ok=True)
    if worker_count == 1:
        gpu = gpus[0]
        if gpu.lower() != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            device = "cuda:0"
        else:
            device = "cpu"
        rows = _evaluate_with_checkpoints(
            dataset_paths,
            args=args,
            device=device,
            checkpoint_csv=out_dir / "all_classification_results.csv",
            result_out_dir=out_dir,
        )
        failed = [row.dataset_name for row in rows if row.status == "fail"]
        if failed:
            raise RuntimeError(f"dataset evaluation failed: {failed}")
        return

    context = mp.get_context("spawn")
    processes: list[Any] = []
    worker_paths: list[Path] = []
    args_dict = vars(args).copy()
    for worker_id in range(worker_count):
        assigned = dataset_paths[worker_id::worker_count]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_paths.append(worker_csv)
        process = context.Process(
            target=_worker_entry,
            args=(
                worker_id,
                gpus[worker_id],
                [str(path) for path in assigned],
                args_dict,
                str(worker_csv),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"worker processes failed with exit codes {failed}")
    frame = pd.concat([pd.read_csv(path) for path in worker_paths], ignore_index=True)
    rows = [DatasetResult(**record) for record in frame.to_dict(orient="records")]
    _write_results(out_dir, rows, args)
    failed_datasets = [row.dataset_name for row in rows if row.status == "fail"]
    if failed_datasets:
        raise RuntimeError(f"dataset evaluation failed: {failed_datasets}")


def _cell_is_complete(
    cell_dir: Path,
    *,
    expected_dataset_names: set[str],
    expected_fingerprint: str,
) -> bool:
    csv_path = cell_dir / "all_classification_results.csv"
    manifest_path = cell_dir / "run_manifest.json"
    if not csv_path.exists() or not manifest_path.exists():
        return False
    try:
        frame = pd.read_csv(csv_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(manifest.get("config_fingerprint", "")) != expected_fingerprint:
        return False
    if frame.empty or frame.duplicated(["dataset_name"]).any():
        return False
    observed_names = set(frame["dataset_name"].astype(str))
    return (
        observed_names == expected_dataset_names
        and bool((frame["status"] == "ok").all())
    )


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_matrix_trial_command(
    args: argparse.Namespace,
    *,
    model_family: str,
    method: str,
    cell_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-family",
        model_family,
        "--method",
        method,
        "--data-root",
        str(Path(args.data_root).expanduser().resolve()),
        "--out-dir",
        str(cell_dir.resolve()),
        "--gpus",
        str(args.gpus),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--n-estimators",
        str(args.n_estimators),
        "--train-n-estimators",
        str(args.train_n_estimators),
        "--retrieval-backend",
        str(args.retrieval_backend),
        "--local-epochs",
        str(args.local_epochs),
        "--local-max-steps-per-epoch",
        str(args.local_max_steps_per_epoch),
        "--local-lr",
        str(args.local_lr),
        "--local-weight-decay",
        str(args.local_weight_decay),
        "--local-train-query-length",
        str(args.local_train_query_length),
        "--local-train-batch-size",
        str(args.local_train_batch_size),
        "--local-inference-batch-size",
        str(args.local_inference_batch_size),
        "--local-class-choice",
        str(args.local_class_choice),
        "--local-early-stopping-metric",
        str(args.local_early_stopping_metric),
        "--local-early-stopping-rounds",
        str(args.local_early_stopping_rounds),
        "--local-eval-interval",
        str(args.local_eval_interval),
        "--local-embedding",
        str(args.local_embedding),
        "--local-splits-evaluated",
        str(args.local_splits_evaluated),
        "--local-clipping-value",
        str(args.local_clipping_value),
        "--use-amp",
        str(args.use_amp).lower(),
        "--tabicl-model-path",
        str(args.tabicl_model_path),
        "--tabpfn-v3-binary-model-path",
        str(args.tabpfn_v3_binary_model_path),
        "--tabpfn-v3-multiclass-model-path",
        str(args.tabpfn_v3_multiclass_model_path),
        "--resume" if args.resume else "--no-resume",
    ]
    _append_option(command, "--max-datasets", args.max_datasets)
    _append_option(command, "--context-size", args.context_size)
    for enabled, flag in (
        (args.local_use_one_hot_emb, "--local-use-one-hot-emb"),
        (args.local_onehot_retrieval, "--local-onehot-retrieval"),
        (args.local_scheduler, "--local-scheduler"),
        (args.local_better_selection, "--local-better-selection"),
        (args.local_save_data, "--local-save-data"),
        (args.local_exact_knn, "--local-exact-knn"),
        (args.local_disable_normalize_data, "--local-disable-normalize-data"),
    ):
        if enabled:
            command.append(flag)
    for dataset_name in args.dataset:
        command.extend(["--dataset", dataset_name])
    return command


def run_matrix(args: argparse.Namespace) -> None:
    root = Path(args.out_dir).expanduser().resolve()
    trials = expand_matrix_trials(args.model_family, args.method)
    expected_dataset_names = {
        path.name
        for path in discover_datasets(
            Path(args.data_root).expanduser().resolve(),
            dataset_names=args.dataset,
            max_datasets=args.max_datasets,
        )
    }
    manifest_path = root / "matrix_manifest.json"
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "localpfn_source_commit": LOCALPFN_SOURCE_COMMIT,
        "axes": {
            "model_family": sorted({model for model, _method in trials}),
            "method": sorted({method for _model, method in trials}),
        },
        "trials": [],
    }
    for model_family, method in trials:
        cell_args = copy.copy(args)
        cell_args.model_family = model_family
        cell_args.method = method
        cell_fingerprint = _config_fingerprint(cell_args)
        cell_dir = root / model_family / method
        command = build_matrix_trial_command(
            args,
            model_family=model_family,
            method=method,
            cell_dir=cell_dir,
        )
        record = {
            "model_family": model_family,
            "method": method,
            "out_dir": str(cell_dir),
            "command": command,
            "config_fingerprint": cell_fingerprint,
            "status": "dry_run" if args.dry_run else "pending",
        }
        manifest["trials"].append(record)
        if args.dry_run:
            print(" ".join(command))
            continue
        root.mkdir(parents=True, exist_ok=True)
        if args.resume and _cell_is_complete(
            cell_dir,
            expected_dataset_names=expected_dataset_names,
            expected_fingerprint=cell_fingerprint,
        ):
            record["status"] = "reused"
            _atomic_json(manifest_path, manifest)
            continue
        log_path = root / f"{model_family}_{method}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_mode = "a" if args.resume else "w"
        with log_path.open(log_mode, encoding="utf-8") as stream:
            if args.resume:
                stream.write(
                    f"\n[resume matrix cell {time.strftime('%Y-%m-%dT%H:%M:%S%z')}]\n"
                )
                stream.flush()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["returncode"] = int(completed.returncode)
        record["log"] = str(log_path)
        record["status"] = (
            "ok"
            if completed.returncode == 0
            and _cell_is_complete(
                cell_dir,
                expected_dataset_names=expected_dataset_names,
                expected_fingerprint=cell_fingerprint,
            )
            else "fail"
        )
        _atomic_json(manifest_path, manifest)
        if record["status"] != "ok":
            raise RuntimeError(
                f"matrix cell {model_family}/{method} failed; see {log_path}"
            )
    if args.dry_run:
        return
    frames: list[pd.DataFrame] = []
    for record in manifest["trials"]:
        frame = pd.read_csv(Path(record["out_dir"]) / "all_classification_results.csv")
        frames.append(frame)
    common = pd.concat(frames, ignore_index=True)
    keys = ["model_family", "method", "dataset_name"]
    if common.duplicated(keys).any():
        duplicates = common.loc[common.duplicated(keys, keep=False), keys]
        raise RuntimeError(f"matrix produced duplicate rows:\n{duplicates}")
    common.to_csv(root / "all_classification_results.csv", index=False)
    _atomic_json(manifest_path, manifest)


def _parse_amp(value: str) -> bool | str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be auto, true, or false")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Official LoCalPFN full-finetuning runner for TabICLv2 and "
            "TabPFNv3."
        )
    )
    parser.add_argument(
        "--model-family",
        choices=[*MODEL_FAMILIES, "all"],
        default="all",
    )
    parser.add_argument(
        "--method",
        choices=list(METHODS),
        default="localpfn",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--train-n-estimators", type=int, default=2)
    parser.add_argument(
        "--retrieval-backend",
        choices=["auto", "faiss", "sklearn"],
        default="faiss",
    )
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--local-epochs", type=int, default=21)
    parser.add_argument("--local-max-steps-per-epoch", type=int, default=30)
    parser.add_argument("--local-lr", type=float, default=1e-5)
    parser.add_argument("--local-weight-decay", type=float, default=0.01)
    parser.add_argument("--local-train-query-length", type=int, default=1000)
    parser.add_argument("--local-train-batch-size", type=int, default=2)
    parser.add_argument("--local-inference-batch-size", type=int, default=512)
    parser.add_argument(
        "--local-class-choice",
        choices=["equal", "balance"],
        default="equal",
    )
    parser.add_argument(
        "--local-early-stopping-metric",
        choices=["negloss", "acc", "f1", "auc"],
        default="auc",
    )
    parser.add_argument("--local-early-stopping-rounds", type=int, default=100)
    parser.add_argument("--local-eval-interval", type=int, default=1)
    parser.add_argument("--local-embedding", choices=["raw"], default="raw")
    parser.add_argument(
        "--local-splits-evaluated",
        choices=["valid", "valid_test", "all"],
        default="valid",
    )
    parser.add_argument("--local-clipping-value", type=float, default=10.0)
    parser.add_argument("--local-use-one-hot-emb", action="store_true")
    parser.add_argument("--local-onehot-retrieval", action="store_true")
    parser.add_argument("--local-scheduler", action="store_true")
    parser.add_argument("--local-better-selection", action="store_true")
    parser.add_argument("--local-save-data", action="store_true")
    parser.add_argument("--local-exact-knn", action="store_true")
    parser.add_argument("--local-disable-normalize-data", action="store_true")
    parser.add_argument("--use-amp", type=_parse_amp, default="auto")
    parser.add_argument("--tabicl-model-path", default=str(DEFAULT_TABICL_MODEL))
    parser.add_argument(
        "--tabpfn-v3-binary-model-path",
        default=str(DEFAULT_TABPFN_BINARY_MODEL),
    )
    parser.add_argument(
        "--tabpfn-v3-multiclass-model-path",
        default=str(DEFAULT_TABPFN_MULTICLASS_MODEL),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "workers": args.workers,
        "n_estimators": args.n_estimators,
        "train_n_estimators": args.train_n_estimators,
        "local_epochs": args.local_epochs,
        "local_max_steps_per_epoch": args.local_max_steps_per_epoch,
        "local_train_query_length": args.local_train_query_length,
        "local_train_batch_size": args.local_train_batch_size,
        "local_inference_batch_size": args.local_inference_batch_size,
        "local_early_stopping_rounds": args.local_early_stopping_rounds,
        "local_eval_interval": args.local_eval_interval,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"arguments must be >= 1: {invalid}")
    if args.context_size is not None and int(args.context_size) < 1:
        raise ValueError("--context-size must be >= 1")
    if args.local_exact_knn and int(args.local_train_query_length) != 1:
        raise ValueError(
            "--local-exact-knn requires --local-train-query-length 1"
        )
    if args.model_family != "all" and args.model_family not in MODEL_FAMILIES:
        raise ValueError(f"unsupported model family {args.model_family}")
    if args.method not in METHODS:
        raise ValueError(f"unsupported method {args.method}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    is_matrix = args.model_family == "all"
    if args.out_dir is None:
        data_label = Path(args.data_root).name
        args.out_dir = str(
            DEFAULT_OUT_ROOT
            / f"{data_label}_seed{args.seed}"
            / ("matrix" if is_matrix else f"{args.model_family}_{args.method}")
        )
    args.config_fingerprint = _config_fingerprint(args)
    if args.dry_run and not is_matrix:
        print(
            json.dumps(
                {
                    "model_family": args.model_family,
                    "method": args.method,
                    "out_dir": str(Path(args.out_dir).resolve()),
                    "datasets": args.dataset or "discover",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if is_matrix:
        run_matrix(args)
    else:
        run_single_cell(args)


if __name__ == "__main__":
    main()
