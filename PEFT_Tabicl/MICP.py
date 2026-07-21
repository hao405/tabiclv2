#!/usr/bin/env python3
"""Paper-faithful MICP/CA-PFN runner for TabICLv2 and TabPFNv3.

This file implements the arXiv-v1 recipe from "Mixture of In-Context
Prompters for Tabular PFNs" (arXiv:2405.16156):

* MICP: K-Means routing with one bounded support set per prompter.
* CA-PFN: downstream bootstrap episodes and frozen-backbone adapters.
* A fixed 2 x 2 matrix: TabICLv2/TabPFNv3 x infer/mixturepfn.

The model-facing code reuses the already validated TabICLv2 and TabPFNv3
engines in ``PEFT_Tabicl.compare_way``.  It does not modify that runner.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PEFT_Tabicl import compare_way as backend  # noqa: E402


PAPER_ID = "arXiv:2405.16156v1"
MODEL_FAMILIES = ("tabiclv2", "tabpfnv3")
METHODS = ("infer", "mixturepfn")
DEFAULT_DATA_ROOT = Path("data184")
DEFAULT_OUTPUT_ROOT = Path("results/PEFT_results/MICP/data184/seed42")
DEFAULT_TABICL_MODEL = Path("tabicl-classifier-v2-20260212.ckpt")
DEFAULT_TABPFN_BINARY_MODEL = Path(
    "baseline_compare/TabPFN-main/"
    "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
)
DEFAULT_TABPFN_MULTICLASS_MODEL = Path(
    "baseline_compare/TabPFN-main/"
    "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
)
MISSING_CATEGORY = "__micp_missing__"


@dataclass(frozen=True)
class ContextEpisode:
    context_indices: np.ndarray
    query_indices: np.ndarray
    route_id: int | None = None
    bootstrap_policy: str | None = None


@dataclass
class AdaptationTelemetry:
    steps: int = 0
    loss: float | None = None
    seconds: float = 0.0
    trainable_params: int = 0
    trainable_ratio: float = 0.0


@dataclass
class ResultRow:
    dataset_name: str
    dataset_dir: str
    task_type: str | None
    model_family: str
    method: str
    status: str
    error: str | None
    config_fingerprint: str
    seed: int
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
    context_policy: str | None = None
    support_size: int = 0
    gamma: float = 0.0
    n_clusters: int = 0
    micp_applied: bool = False
    capfn_applied: bool = False
    adapter_bottleneck: int = 0
    adapter_layers: int = 0
    trainable_params: int = 0
    trainable_ratio: float = 0.0
    ca_steps: int = 0
    ca_loss: float | None = None
    preprocessing_seconds: float = 0.0
    routing_seconds: float = 0.0
    ca_seconds: float = 0.0
    predict_seconds: float = 0.0
    total_seconds: float = 0.0


class PaperRetrievalPreprocessor:
    """Train-only ordinal preprocessing for MICP routing.

    The PFN backends still receive their normal pandas inputs.  This transformed
    view is used only for K-Means and nearest-neighbour search.
    """

    def __init__(self, categorical_indices: Sequence[int]) -> None:
        self.categorical_indices = sorted({int(index) for index in categorical_indices})
        self.transformer: Any | None = None
        self.fitted_rows = 0

    def _normalise_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        output = X.copy()
        for index in self.categorical_indices:
            column = output.columns[index]
            output[column] = (
                output[column]
                .astype("string")
                .fillna(MISSING_CATEGORY)
                .astype(str)
            )
        return output

    def fit(self, X_train: pd.DataFrame) -> "PaperRetrievalPreprocessor":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler

        all_indices = list(range(X_train.shape[1]))
        categorical = set(self.categorical_indices)
        numeric_indices = [index for index in all_indices if index not in categorical]
        pieces: list[tuple[str, Any, list[int]]] = []
        if numeric_indices:
            pieces.append(
                (
                    "numeric",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                    ),
                    numeric_indices,
                )
            )
        if self.categorical_indices:
            pieces.append(
                (
                    "categorical",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                        ),
                        StandardScaler(),
                    ),
                    self.categorical_indices,
                )
            )
        if not pieces:
            raise ValueError("MICP retrieval requires at least one feature")
        self.transformer = ColumnTransformer(pieces, sparse_threshold=0.0)
        self.transformer.fit(self._normalise_categories(X_train))
        self.fitted_rows = len(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("MICP retrieval preprocessor is not fitted")
        value = self.transformer.transform(self._normalise_categories(X))
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("MICP preprocessing produced invalid values")
        return value


class NeighborIndex:
    def __init__(self, backend_name: str = "faiss") -> None:
        self.requested_backend = backend_name
        self.backend = ""
        self.index: Any | None = None
        self.train: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> "NeighborIndex":
        matrix = np.ascontiguousarray(train, dtype=np.float32)
        if self.requested_backend in {"auto", "faiss"}:
            try:
                import faiss

                index = faiss.IndexFlatL2(matrix.shape[1])
                index.add(matrix)
                self.index = index
                self.backend = "faiss"
                self.train = matrix
                return self
            except ImportError:
                if self.requested_backend == "faiss":
                    raise
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(algorithm="auto", metric="euclidean")
        index.fit(matrix)
        self.index = index
        self.backend = "sklearn"
        self.train = matrix
        return self

    def query(self, query: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.train is None:
            raise RuntimeError("MICP neighbour index is not fitted")
        count = min(max(1, int(n_neighbors)), len(self.train))
        matrix = np.ascontiguousarray(query, dtype=np.float32)
        if self.backend == "faiss":
            distances, indices = self.index.search(matrix, count)
            return distances, indices
        distances, indices = self.index.kneighbors(matrix, n_neighbors=count)
        return np.asarray(distances), np.asarray(indices)


class MICPContextBuilder:
    """MICP router/support construction plus CA-PFN bootstrap episodes."""

    def __init__(
        self,
        train_space: np.ndarray,
        y_train: np.ndarray,
        *,
        support_size: int,
        gamma: float,
        seed: int,
        retrieval_backend: str,
        require_all_classes: bool = False,
    ) -> None:
        from sklearn.cluster import KMeans

        self.train_space = np.asarray(train_space, dtype=np.float32)
        self.y_train = np.asarray(y_train)
        self.support_size = min(max(1, int(support_size)), len(self.y_train))
        if float(gamma) <= 0:
            raise ValueError("MICP gamma must be positive")
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.require_all_classes = bool(require_all_classes)
        self.classes = np.unique(self.y_train)
        if self.require_all_classes and self.support_size < len(self.classes):
            raise ValueError(
                "TabICLv2 MICP support_size must cover every training class: "
                f"support_size={self.support_size}, n_classes={len(self.classes)}"
            )
        self.micp_enabled = len(self.y_train) > self.support_size
        self.n_clusters = (
            max(
                1,
                min(
                    len(self.y_train),
                    int(
                        math.ceil(
                            self.gamma * len(self.y_train) / self.support_size
                        )
                    ),
                ),
            )
            if self.micp_enabled
            else 1
        )
        self.neighbors = NeighborIndex(retrieval_backend).fit(self.train_space)
        self.kmeans: Any | None = None
        self.assignments = np.zeros(len(self.y_train), dtype=int)
        self.supports: dict[int, np.ndarray] = {
            0: np.arange(len(self.y_train), dtype=int)
        }
        if self.micp_enabled:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.seed,
                n_init=10,
            )
            self.assignments = self.kmeans.fit_predict(self.train_space)
            self.supports = self._build_supports()

    def _nearest_index(
        self,
        candidates: np.ndarray,
        reference: np.ndarray,
    ) -> int:
        matrix = self.train_space[np.asarray(candidates, dtype=int)]
        distances = np.sum((matrix - reference.reshape(1, -1)) ** 2, axis=1)
        return int(candidates[int(np.argmin(distances))])

    def _replace_preserving_classes(
        self,
        selected: np.ndarray,
        replacement: int,
    ) -> np.ndarray:
        output = np.asarray(selected, dtype=int).copy()
        if int(replacement) in set(output.tolist()):
            return output
        replacement_label = self.y_train[int(replacement)]
        labels = self.y_train[output]
        same_label = np.flatnonzero(labels == replacement_label)
        if len(same_label):
            output[int(same_label[-1])] = int(replacement)
            return output
        counts = {
            label: int(np.sum(labels == label))
            for label in np.unique(labels)
        }
        replaceable = [
            position
            for position, label in enumerate(labels.tolist())
            if counts[label] > 1
        ]
        if not replaceable:
            raise ValueError(
                "MICP support cannot add coverage without dropping a class"
            )
        output[int(replaceable[-1])] = int(replacement)
        return output

    def _repair_support(
        self,
        indices: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        output = np.asarray(indices, dtype=int).copy()
        if self.require_all_classes:
            present = set(self.y_train[output].tolist())
            for label in self.classes.tolist():
                if label in present:
                    continue
                candidates = np.flatnonzero(self.y_train == label)
                replacement = self._nearest_index(candidates, reference)
                output = self._replace_preserving_classes(output, replacement)
                present.add(label)

        if len(output) > 1 and not np.any(
            np.ptp(self.train_space[output], axis=0) > 0
        ):
            anchor = self.train_space[int(output[0])]
            varying = np.flatnonzero(
                np.any(self.train_space != anchor.reshape(1, -1), axis=1)
            )
            if len(varying):
                replacement = self._nearest_index(varying, reference)
                output = self._replace_preserving_classes(output, replacement)
        return output

    def _build_supports(self) -> dict[int, np.ndarray]:
        if self.kmeans is None:
            return {0: np.arange(len(self.y_train), dtype=int)}
        supports: dict[int, np.ndarray] = {}
        small_routes: list[int] = []
        small_centres: list[np.ndarray] = []
        for route_id in range(self.n_clusters):
            members = np.flatnonzero(self.assignments == route_id)
            if len(members) >= self.support_size:
                rng = np.random.default_rng(self.seed + route_id)
                supports[route_id] = np.asarray(
                    rng.choice(
                        members,
                        size=self.support_size,
                        replace=False,
                    ),
                    dtype=int,
                )
            else:
                small_routes.append(route_id)
                small_centres.append(self.kmeans.cluster_centers_[route_id])
        if small_routes:
            _distances, nearest = self.neighbors.query(
                np.asarray(small_centres, dtype=np.float32),
                self.support_size,
            )
            for route_id, indices in zip(small_routes, nearest, strict=True):
                supports[route_id] = np.asarray(indices, dtype=int)
        for route_id, indices in supports.items():
            supports[route_id] = self._repair_support(
                indices,
                np.asarray(self.kmeans.cluster_centers_[route_id]),
            )
        return supports

    def route_ids(self, query_space: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            return np.zeros(len(query_space), dtype=int)
        return np.asarray(self.kmeans.predict(query_space), dtype=int)

    def episodes_for_external(
        self,
        query_space: np.ndarray,
        *,
        query_batch_size: int = 1024,
    ) -> list[ContextEpisode]:
        route_ids = self.route_ids(np.asarray(query_space, dtype=np.float32))
        grouped: dict[int, list[int]] = defaultdict(list)
        for query_index, route_id in enumerate(route_ids.tolist()):
            grouped[int(route_id)].append(int(query_index))
        episodes: list[ContextEpisode] = []
        batch_size = max(1, int(query_batch_size))
        for route_id in sorted(grouped):
            query_indices = grouped[route_id]
            for start in range(0, len(query_indices), batch_size):
                episodes.append(
                    ContextEpisode(
                        context_indices=self.supports[route_id].copy(),
                        query_indices=np.asarray(
                            query_indices[start : start + batch_size],
                            dtype=int,
                        ),
                        route_id=route_id,
                    )
                )
        return episodes

    def bootstrap_episode(
        self,
        anchor_index: int,
        *,
        query_size: int,
    ) -> ContextEpisode:
        rng = np.random.default_rng(
            self.seed + 1_000_003 * int(anchor_index)
        )
        if self.micp_enabled:
            _distances, nearest = self.neighbors.query(
                self.train_space[[int(anchor_index)]],
                self.support_size,
            )
            bootstrap = np.asarray(nearest[0], dtype=int)
            bootstrap = self._repair_support(
                bootstrap,
                self.train_space[int(anchor_index)],
            )
            rng.shuffle(bootstrap)
            requested_query = min(
                max(1, int(query_size)),
                max(1, len(bootstrap) - 1),
            )
            policy = "large_knn_b"
        else:
            bootstrap = rng.permutation(len(self.y_train)).astype(int)
            requested_query = max(
                1,
                len(bootstrap) - int(math.floor(0.9 * len(bootstrap))),
            )
            requested_query = min(requested_query, max(1, len(bootstrap) - 1))
            policy = "small_random_90_10"

        counts = {
            label: int(np.sum(self.y_train[bootstrap] == label))
            for label in np.unique(self.y_train[bootstrap])
        }
        eligible = [
            int(index)
            for index in bootstrap
            if counts[self.y_train[int(index)]] >= 2
        ]
        if not eligible:
            raise ValueError("CA-PFN bootstrap cannot form a label-covered query")
        query = np.asarray(eligible[:requested_query], dtype=int)
        query_set = set(query.tolist())
        context = np.asarray(
            [int(index) for index in bootstrap if int(index) not in query_set],
            dtype=int,
        )
        missing = set(self.y_train[query].tolist()) - set(
            self.y_train[context].tolist()
        )
        if missing:
            raise ValueError(
                f"CA-PFN query labels absent from context: {sorted(missing, key=str)}"
            )
        return ContextEpisode(
            context_indices=context,
            query_indices=query,
            route_id=(
                int(self.assignments[int(anchor_index)])
                if self.micp_enabled
                else 0
            ),
            bootstrap_policy=policy,
        )

    def training_batches(
        self,
        *,
        epoch: int,
        max_steps: int,
        batch_size: int,
        better_selection: bool = False,
        exact_knn: bool = False,
    ) -> list[list[ContextEpisode]]:
        del better_selection, exact_knn
        rng = np.random.default_rng(self.seed + int(epoch))
        required = max(1, int(max_steps)) * max(1, int(batch_size))
        anchors = rng.integers(0, len(self.y_train), size=required)
        episodes = [
            self.bootstrap_episode(
                int(anchor),
                query_size=getattr(self, "training_query_size", 64),
            )
            for anchor in anchors
        ]
        width = max(1, int(batch_size))
        return [
            episodes[start : start + width]
            for start in range(0, len(episodes), width)
        ]


def _module_width(module: Any) -> int:
    import torch

    # Both supported backends expose the residual-stream LayerNorm directly on
    # each ICL block.  Prefer it over a recursive search: TabPFNv3 also nests a
    # softmax-scaling helper whose first LayerNorm is head-sized (64) rather
    # than residual-stream-sized (512).
    for attribute in (
        "layernorm",
        "layernorm_mlp",
        "norm1",
        "norm2",
    ):
        normalisation = getattr(module, attribute, None)
        shape = getattr(normalisation, "normalized_shape", None)
        if shape is not None:
            return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
    for child in module.modules():
        if isinstance(child, torch.nn.LayerNorm):
            shape = child.normalized_shape
            return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            return int(child.out_features)
    raise RuntimeError(f"cannot infer adapter width from {type(module).__name__}")


def _adapter_block(base_layer: Any, width: int, bottleneck: int) -> Any:
    import torch

    class ResidualAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(width)
            self.down = torch.nn.Linear(width, bottleneck)
            self.activation = torch.nn.GELU()
            self.up = torch.nn.Linear(bottleneck, width)
            torch.nn.init.zeros_(self.up.weight)
            torch.nn.init.zeros_(self.up.bias)

        def forward(self, value: Any) -> Any:
            return value + self.up(self.activation(self.down(self.norm(value))))

    class AdapterBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = base_layer
            self.micp_adapter = ResidualAdapter()

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            output = self.base_layer(*args, **kwargs)
            if isinstance(output, torch.Tensor):
                return self.micp_adapter(output)
            if isinstance(output, tuple) and output and isinstance(
                output[0], torch.Tensor
            ):
                return (self.micp_adapter(output[0]), *output[1:])
            raise TypeError(
                "MICP adapter expected a Tensor or tuple whose first item is a Tensor"
            )

    return AdapterBlock()


def install_micp_adapters(
    model: Any,
    *,
    bottleneck_override: int | None = None,
) -> tuple[list[Any], dict[str, int]]:
    """Freeze a PFN and install one identity-initialised adapter per ICL block."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    tabicl_blocks = getattr(
        getattr(getattr(model, "icl_predictor", None), "tf_icl", None),
        "blocks",
        None,
    )
    tabpfn_blocks = getattr(model, "icl_blocks", None)
    # TabPFNv3 may expose compatibility submodules that resemble TabICL's
    # ``icl_predictor.tf_icl`` tree, but its actual forward path uses the
    # top-level ``icl_blocks``.  Prefer that unambiguous architecture marker.
    blocks = tabpfn_blocks if tabpfn_blocks is not None else tabicl_blocks
    if blocks is None:
        raise RuntimeError(
            "MICP adapters require TabICL tf_icl.blocks or TabPFNv3 icl_blocks"
        )
    installed = 0
    bottleneck_used = 0
    for index, layer in enumerate(list(blocks)):
        if hasattr(layer, "micp_adapter"):
            for parameter in layer.micp_adapter.parameters():
                parameter.requires_grad = True
            bottleneck_used = int(layer.micp_adapter.down.out_features)
            installed += 1
            continue
        width = _module_width(layer)
        bottleneck = (
            int(bottleneck_override)
            if bottleneck_override is not None
            else max(8, width // 16)
        )
        wrapper = _adapter_block(layer, width, bottleneck)
        reference_parameter = next(layer.parameters(), None)
        if reference_parameter is not None:
            wrapper = wrapper.to(
                device=reference_parameter.device,
                dtype=reference_parameter.dtype,
            )
        blocks[index] = wrapper
        bottleneck_used = bottleneck
        installed += 1
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("MICP adapter installation selected no parameters")
    return trainable, {
        "adapter_layers": installed,
        "adapter_bottleneck": bottleneck_used,
    }


@contextlib.contextmanager
def patched_adapter_configuration(
    *,
    bottleneck_override: int | None,
) -> Iterator[dict[str, int]]:
    """Patch only the backend's trainable-parameter callback for one fit."""

    original = backend.set_full_finetune
    telemetry: dict[str, int] = {}

    def configure(model: Any) -> None:
        _trainable, installed = install_micp_adapters(
            model,
            bottleneck_override=bottleneck_override,
        )
        telemetry.update(installed)

    backend.set_full_finetune = configure
    try:
        yield telemetry
    finally:
        backend.set_full_finetune = original


def _categorical_indices(X: pd.DataFrame) -> list[int]:
    return [
        index
        for index, dtype in enumerate(X.dtypes)
        if not pd.api.types.is_numeric_dtype(dtype)
    ]


def _prepared_data(
    loaded: Any,
    preprocessor: PaperRetrievalPreprocessor,
    *,
    include_validation: bool,
) -> Any:
    return backend.PreparedLoCalData(
        X_train=loaded.X_train,
        X_val=loaded.X_val if include_validation else None,
        X_test=loaded.X_test,
        retrieval_train=preprocessor.transform(loaded.X_train),
        retrieval_val=(
            preprocessor.transform(loaded.X_val)
            if include_validation and loaded.X_val is not None
            else None
        ),
        retrieval_test=preprocessor.transform(loaded.X_test),
        categorical_indices=_categorical_indices(loaded.X_train),
    )


def _loaded_without_validation(loaded: Any) -> Any:
    return dataclasses.replace(loaded, X_val=None, y_val=None)


def _make_engine(
    model_family: str,
    loaded: Any,
    prepared: Any,
    args: argparse.Namespace,
    device: str,
) -> Any:
    if model_family == "tabiclv2":
        return backend.TabICLNativeEngine(loaded, prepared, args, device)
    if model_family == "tabpfnv3":
        return backend.TabPFNv3Engine(loaded, prepared, args, device)
    raise ValueError(f"unsupported model family: {model_family}")


def _predict_infer(
    engine: Any,
    model_family: str,
    loaded: Any,
    prepared: Any,
    args: argparse.Namespace,
) -> np.ndarray:
    rng = np.random.default_rng(int(args.seed))
    support_size = min(int(args.support_size), len(loaded.y_train))
    context = (
        np.arange(len(loaded.y_train), dtype=int)
        if len(loaded.y_train) <= support_size
        else np.sort(
            rng.choice(
                len(loaded.y_train),
                size=support_size,
                replace=False,
            )
        ).astype(int)
    )
    episodes = [
        ContextEpisode(
            context_indices=context,
            query_indices=np.arange(
                start,
                min(start + int(args.inference_batch_size), len(loaded.y_test)),
                dtype=int,
            ),
        )
        for start in range(0, len(loaded.y_test), int(args.inference_batch_size))
    ]
    if model_family == "tabiclv2":
        return engine.predict_local(loaded, prepared, episodes, args=args)
    classifier = engine._make_base_classifier(loaded, args)
    return engine._predict_with_classifier(
        classifier,
        loaded,
        prepared,
        episodes,
        prepared.X_test,
        args=args,
    )


def _adapt_and_predict(
    engine: Any,
    model_family: str,
    loaded: Any,
    prepared: Any,
    builder: MICPContextBuilder,
    args: argparse.Namespace,
) -> tuple[np.ndarray, Any, dict[str, int]]:
    builder.training_query_size = int(args.ca_query_size)
    with patched_adapter_configuration(
        bottleneck_override=args.adapter_bottleneck,
    ) as adapter_telemetry:
        adaptation = engine.adapt(
            loaded,
            prepared,
            builder,
            args=args,
            validation_episodes=None,
        )
    episodes = builder.episodes_for_external(
        prepared.retrieval_test,
        query_batch_size=int(args.inference_batch_size),
    )
    probabilities = engine.predict_local(
        loaded,
        prepared,
        episodes,
        args=args,
    )
    return probabilities, adaptation, adapter_telemetry


def _base_result(
    loaded: Any,
    args: argparse.Namespace,
    dataset_dir: Path,
) -> ResultRow:
    return ResultRow(
        dataset_name=dataset_dir.name,
        dataset_dir=str(dataset_dir),
        task_type=getattr(loaded, "task_type", None),
        model_family=args.model_family,
        method=args.method,
        status="error",
        error=None,
        config_fingerprint=args.config_fingerprint,
        seed=int(args.seed),
        n_train=len(loaded.y_train),
        n_val=0 if loaded.y_val is None else len(loaded.y_val),
        n_test=len(loaded.y_test),
        n_features=int(loaded.X_train.shape[1]),
        n_classes=len(np.unique(loaded.y_train)),
        support_size=min(int(args.support_size), len(loaded.y_train)),
        gamma=float(args.gamma),
    )


def evaluate_dataset(
    dataset_dir: Path,
    *,
    args: argparse.Namespace,
    device: str,
) -> ResultRow:
    started = time.time()
    loaded: Any | None = None
    row: ResultRow | None = None
    try:
        loaded = backend.load_dataset(dataset_dir)
        row = _base_result(loaded, args, dataset_dir)
        if loaded.task_type not in backend.CLASSIFICATION_TASKS:
            row.status = "skip"
            row.error = f"task_type={loaded.task_type!r} is not classification"
            return row
        if len(np.unique(loaded.y_train)) < 2:
            raise ValueError("training split has fewer than two classes")

        preprocessing_started = time.time()
        preprocessor = PaperRetrievalPreprocessor(
            _categorical_indices(loaded.X_train)
        ).fit(loaded.X_train)
        include_validation = args.model_family == "tabiclv2"
        engine_loaded = (
            loaded if include_validation else _loaded_without_validation(loaded)
        )
        prepared = _prepared_data(
            engine_loaded,
            preprocessor,
            include_validation=include_validation,
        )
        row.preprocessing_seconds = time.time() - preprocessing_started

        builder: MICPContextBuilder | None = None
        if args.method == "mixturepfn":
            routing_started = time.time()
            builder = MICPContextBuilder(
                prepared.retrieval_train,
                engine_loaded.y_train,
                support_size=int(args.support_size),
                gamma=float(args.gamma),
                seed=int(args.seed),
                retrieval_backend=args.retrieval_backend,
                require_all_classes=args.model_family == "tabiclv2",
            )
            row.routing_seconds = time.time() - routing_started
            row.n_clusters = builder.n_clusters
            row.micp_applied = builder.micp_enabled
            row.context_policy = (
                "micp_cluster_support"
                if builder.micp_enabled
                else "full_context_small_dataset"
            )
        else:
            row.n_clusters = 0
            row.context_policy = "paper_random_bounded_infer"

        engine = _make_engine(
            args.model_family,
            engine_loaded,
            prepared,
            args,
            device,
        )
        predict_started = time.time()
        if args.method == "infer":
            probabilities = _predict_infer(
                engine,
                args.model_family,
                engine_loaded,
                prepared,
                args,
            )
        else:
            assert builder is not None
            ca_started = time.time()
            probabilities, adaptation, adapter_telemetry = _adapt_and_predict(
                engine,
                args.model_family,
                engine_loaded,
                prepared,
                builder,
                args,
            )
            row.ca_seconds = time.time() - ca_started
            row.capfn_applied = int(adaptation.steps) > 0
            row.ca_steps = int(adaptation.steps)
            row.ca_loss = adaptation.loss
            row.trainable_params = int(adaptation.trainable_params)
            row.trainable_ratio = float(adaptation.trainable_ratio)
            row.adapter_layers = int(
                adapter_telemetry.get("adapter_layers", 0)
            )
            row.adapter_bottleneck = int(
                adapter_telemetry.get("adapter_bottleneck", 0)
            )
            if not row.capfn_applied or row.trainable_params <= 0:
                raise RuntimeError("CA-PFN completed without adapter updates")
        row.predict_seconds = time.time() - predict_started - row.ca_seconds

        metrics = backend._classification_metrics(
            loaded.y_test,
            probabilities,
            np.unique(loaded.y_train),
        )
        row.accuracy = metrics["accuracy"]
        row.f1 = metrics["f1"]
        row.balanced_accuracy = metrics["balanced_accuracy"]
        row.roc_auc = metrics["roc_auc"]
        row.log_loss = metrics["log_loss"]
        row.status = "ok"
        row.error = None
    except Exception as exc:
        if row is None:
            row = ResultRow(
                dataset_name=dataset_dir.name,
                dataset_dir=str(dataset_dir),
                task_type=getattr(loaded, "task_type", None),
                model_family=args.model_family,
                method=args.method,
                status="error",
                error=None,
                config_fingerprint=args.config_fingerprint,
                seed=int(args.seed),
            )
        row.status = "error"
        row.error = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        traceback.print_exc()
    finally:
        if row is not None:
            row.total_seconds = time.time() - started
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    assert row is not None
    return row


def discover_datasets(args: argparse.Namespace) -> list[Path]:
    return backend.discover_datasets(
        Path(args.data_root).expanduser().resolve(),
        dataset_names=args.dataset,
        max_datasets=args.max_datasets,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame([asdict(row) for row in rows]).to_csv(temporary, index=False)
    temporary.replace(path)


def write_summary(
    summary_path: Path,
    result_df: pd.DataFrame,
    *,
    discovered_datasets: int,
) -> None:
    """Write the repository-standard ``summary.txt`` result surface."""

    frame = result_df.copy()
    for column in (
        "accuracy",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "log_loss",
        "total_seconds",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if len(frame) and "status" in frame.columns:
        statuses = frame["status"].astype(str).str.lower()
        ok_df = frame[statuses == "ok"].copy()
        skipped_df = frame[statuses == "skip"].copy()
        failed_df = frame[~statuses.isin(("ok", "skip"))].copy()
    else:
        ok_df = pd.DataFrame()
        failed_df = pd.DataFrame()
        skipped_df = pd.DataFrame()

    def mean_line(label: str, column: str) -> str:
        if (
            len(ok_df)
            and column in ok_df.columns
            and ok_df[column].notna().any()
        ):
            return f"{label}: {ok_df[column].mean():.6f}"
        return f"{label}: (none)"

    wall_seconds = (
        float(frame["total_seconds"].fillna(0.0).sum())
        if "total_seconds" in frame.columns
        else 0.0
    )
    lines = [
        f"discovered_datasets: {int(discovered_datasets)}",
        f"processed_datasets: {len(frame)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        "ft_oom_fallback_count: 0",
        mean_line("avg_accuracy_ok", "accuracy"),
        mean_line("avg_f1_ok", "f1"),
        mean_line("avg_balanced_accuracy_ok", "balanced_accuracy"),
        mean_line("avg_roc_auc_ok", "roc_auc"),
        mean_line("avg_log_loss_ok", "log_loss"),
        f"wall_seconds: {wall_seconds:.3f}",
        (
            "failed_datasets: "
            + (
                ", ".join(failed_df["dataset_name"].astype(str).tolist())
                if len(failed_df)
                else "(none)"
            )
        ),
        (
            "skipped_datasets: "
            + (
                ", ".join(skipped_df["dataset_name"].astype(str).tolist())
                if len(skipped_df)
                else "(none)"
            )
        ),
        "ft_oom_fallback_datasets: (none)",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(summary_path)


def _write_cell_outputs(
    cell_dir: Path,
    rows: Sequence[ResultRow],
    *,
    discovered_datasets: int,
) -> None:
    _atomic_csv(cell_dir / "all_classification_results.csv", rows)
    write_summary(
        cell_dir / "summary.txt",
        pd.DataFrame([asdict(row) for row in rows]),
        discovered_datasets=discovered_datasets,
    )


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    ignored = {
        "dry_run",
        "resume",
        "retry_failed",
        "out_dir",
        "config_fingerprint",
    }
    return {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in ignored
    }


def config_fingerprint(args: argparse.Namespace) -> str:
    encoded = json.dumps(
        _config_payload(args),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_existing(path: Path) -> dict[str, ResultRow]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    fields = set(ResultRow.__dataclass_fields__)
    output: dict[str, ResultRow] = {}
    for record in frame.to_dict(orient="records"):
        cleaned = {
            key: (None if pd.isna(value) else value)
            for key, value in record.items()
            if key in fields
        }
        row = ResultRow(**cleaned)
        output[row.dataset_name] = row
    return output


def run_cell(args: argparse.Namespace, cell_dir: Path) -> dict[str, Any]:
    cell_dir = cell_dir.resolve()
    args.out_dir = str(cell_dir)
    args.config_fingerprint = config_fingerprint(args)
    dataset_paths = discover_datasets(args)
    if not dataset_paths:
        raise RuntimeError(f"no datasets found under {args.data_root}")
    csv_path = cell_dir / "all_classification_results.csv"
    existing = _read_existing(csv_path) if args.resume else {}
    rows: dict[str, ResultRow] = {}
    for dataset_path in dataset_paths:
        previous = existing.get(dataset_path.name)
        can_reuse = (
            previous is not None
            and previous.config_fingerprint == args.config_fingerprint
            and (
                previous.status == "ok"
                or (
                    previous.status != "ok"
                    and not bool(args.retry_failed)
                )
            )
        )
        if can_reuse:
            rows[dataset_path.name] = previous
            print(
                f"[resume] {args.model_family}/{args.method} "
                f"{dataset_path.name} status={previous.status}",
                flush=True,
            )
            continue
        row = evaluate_dataset(
            dataset_path,
            args=args,
            device="cuda:0" if args.gpus != "cpu" else "cpu",
        )
        rows[dataset_path.name] = row
        ordered = [rows[path.name] for path in dataset_paths if path.name in rows]
        _write_cell_outputs(
            cell_dir,
            ordered,
            discovered_datasets=len(dataset_paths),
        )
        print(
            f"[checkpoint] {args.model_family}/{args.method} "
            f"dataset={dataset_path.name} status={row.status} "
            f"completed={len(ordered)}/{len(dataset_paths)}",
            flush=True,
        )
    ordered = [rows[path.name] for path in dataset_paths]
    _write_cell_outputs(
        cell_dir,
        ordered,
        discovered_datasets=len(dataset_paths),
    )
    ok_count = sum(row.status == "ok" for row in ordered)
    manifest = {
        "paper": PAPER_ID,
        "model_family": args.model_family,
        "method": args.method,
        "config": _config_payload(args),
        "config_fingerprint": args.config_fingerprint,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_count": len(ordered),
        "ok_count": ok_count,
        "error_count": sum(row.status == "error" for row in ordered),
        "skip_count": sum(row.status == "skip" for row in ordered),
        "status": "ok" if ok_count == len(ordered) else "partial",
        "result_csv": str(csv_path),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(cell_dir / "run_manifest.json", manifest)
    return manifest


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_cell_command(
    args: argparse.Namespace,
    *,
    model_family: str,
    method: str,
    gpu: str,
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
        gpu,
        "--seed",
        str(args.seed),
        "--support-size",
        str(args.support_size),
        "--gamma",
        str(args.gamma),
        "--ca-steps",
        str(args.ca_steps),
        "--ca-query-size",
        str(args.ca_query_size),
        "--ca-lr",
        str(args.ca_lr),
        "--inference-batch-size",
        str(args.inference_batch_size),
        "--n-estimators",
        str(args.n_estimators),
        "--train-n-estimators",
        str(args.train_n_estimators),
        "--retrieval-backend",
        str(args.retrieval_backend),
        "--tabicl-model-path",
        str(args.tabicl_model_path),
        "--tabpfn-v3-binary-model-path",
        str(args.tabpfn_v3_binary_model_path),
        "--tabpfn-v3-multiclass-model-path",
        str(args.tabpfn_v3_multiclass_model_path),
        "--use-amp",
        str(args.use_amp).lower(),
        "--resume" if args.resume else "--no-resume",
    ]
    _append_option(command, "--adapter-bottleneck", args.adapter_bottleneck)
    _append_option(command, "--max-datasets", args.max_datasets)
    if args.retry_failed:
        command.append("--retry-failed")
    for dataset in args.dataset:
        command.extend(["--dataset", dataset])
    return command


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in str(value).split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one device")
    if any(item != "cpu" and not item.isdigit() for item in gpus):
        raise ValueError("--gpus must be comma-separated physical ids or cpu")
    return gpus


def matrix_trials(
    model_family: str,
    method: str,
) -> list[tuple[str, str]]:
    models = MODEL_FAMILIES if model_family == "all" else (model_family,)
    methods = METHODS if method == "all" else (method,)
    return [(model, method_name) for model in models for method_name in methods]


def run_matrix(args: argparse.Namespace, root: Path) -> None:
    trials = matrix_trials(args.model_family, args.method)
    gpus = _parse_gpus(args.gpus)
    gpu_by_model = {
        model: gpus[min(index, len(gpus) - 1)]
        for index, model in enumerate(MODEL_FAMILIES)
    }
    records: list[dict[str, Any]] = []
    commands: dict[str, list[list[str]]] = defaultdict(list)
    for model_family, method in trials:
        cell_dir = root / model_family / method
        command = build_cell_command(
            args,
            model_family=model_family,
            method=method,
            gpu=gpu_by_model[model_family],
            cell_dir=cell_dir,
        )
        record = {
            "model_family": model_family,
            "method": method,
            "gpu": gpu_by_model[model_family],
            "out_dir": str(cell_dir),
            "command": command,
            "status": "dry_run" if args.dry_run else "pending",
        }
        records.append(record)
        commands[model_family].append(command)
    if args.dry_run:
        for record in records:
            print(shlex.join(record["command"]))
        print(
            json.dumps(
                {
                    "planned_cells": len(records),
                    "datasets_per_cell": len(discover_datasets(args)),
                    "planned_dataset_tasks": (
                        len(records) * len(discover_datasets(args))
                    ),
                    "gpu_by_model": gpu_by_model,
                },
                indent=2,
            )
        )
        return

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "matrix_manifest.json"
    manifest = {
        "paper": PAPER_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "axes": {
            "model_family": sorted({model for model, _ in trials}),
            "method": sorted({method for _, method in trials}),
        },
        "gpu_by_model": gpu_by_model,
        "schedule": "model_lanes_each_serial",
        "trials": records,
        "status": "running",
    }
    _atomic_json(manifest_path, manifest)
    processes: dict[str, subprocess.Popen[Any]] = {}
    logs: dict[str, Any] = {}
    try:
        for model_family, lane_commands in commands.items():
            lane_script = "set -o pipefail; " + " && ".join(
                shlex.join(command) for command in lane_commands
            )
            log_path = root / f"{model_family}.log"
            stream = log_path.open("a" if args.resume else "w", encoding="utf-8")
            logs[model_family] = stream
            processes[model_family] = subprocess.Popen(
                ["bash", "-lc", lane_script],
                cwd=REPO_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            for record in records:
                if record["model_family"] == model_family:
                    record["log"] = str(log_path)
                    record["lane_pid"] = processes[model_family].pid
        _atomic_json(manifest_path, manifest)
        returncodes = {
            model: process.wait() for model, process in processes.items()
        }
    finally:
        for stream in logs.values():
            stream.close()

    frames: list[pd.DataFrame] = []
    for record in records:
        csv_path = Path(record["out_dir"]) / "all_classification_results.csv"
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
            frames.append(frame)
            ok = int((frame["status"] == "ok").sum())
            record["row_count"] = len(frame)
            record["ok_count"] = ok
            record["status"] = "ok" if ok == len(frame) else "partial"
        else:
            record["status"] = "missing"
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(root / "all_classification_results.csv", index=False)
        write_summary(
            root / "summary.txt",
            combined,
            discovered_datasets=int(combined["dataset_name"].nunique()),
        )
    manifest["lane_returncodes"] = returncodes
    manifest["status"] = (
        "ok"
        if all(record["status"] == "ok" for record in records)
        else "partial"
    )
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _atomic_json(manifest_path, manifest)


def _parse_amp(value: str) -> bool | str:
    normalised = str(value).strip().lower()
    if normalised == "auto":
        return "auto"
    if normalised in {"true", "1", "yes", "on"}:
        return True
    if normalised in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be auto, true, or false")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the arXiv-v1 MICP/CA-PFN method on TabICLv2 and TabPFNv3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-family",
        choices=[*MODEL_FAMILIES, "all"],
        default="all",
    )
    parser.add_argument(
        "--method",
        choices=[*METHODS, "all"],
        default="all",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--gpus",
        default="1,2",
        help=(
            "Physical GPU ids. In the 2x2 matrix GPU1 is assigned to "
            "TabICLv2 and GPU2 to TabPFNv3."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--support-size", type=int, default=3000)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--ca-steps", type=int, default=128)
    parser.add_argument("--ca-query-size", type=int, default=64)
    parser.add_argument("--ca-lr", type=float, default=1e-3)
    parser.add_argument("--adapter-bottleneck", type=int)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    parser.add_argument("--n-estimators", type=int, default=16)
    parser.add_argument("--train-n-estimators", type=int, default=2)
    parser.add_argument(
        "--retrieval-backend",
        choices=["faiss", "sklearn", "auto"],
        default="faiss",
    )
    parser.add_argument(
        "--tabicl-model-path",
        default=str(DEFAULT_TABICL_MODEL),
    )
    parser.add_argument(
        "--tabpfn-v3-binary-model-path",
        default=str(DEFAULT_TABPFN_BINARY_MODEL),
    )
    parser.add_argument(
        "--tabpfn-v3-multiclass-model-path",
        default=str(DEFAULT_TABPFN_MULTICLASS_MODEL),
    )
    parser.add_argument("--use-amp", type=_parse_amp, default="auto")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "support_size": args.support_size,
        "gamma": args.gamma,
        "ca_steps": args.ca_steps,
        "ca_query_size": args.ca_query_size,
        "ca_lr": args.ca_lr,
        "inference_batch_size": args.inference_batch_size,
        "n_estimators": args.n_estimators,
        "train_n_estimators": args.train_n_estimators,
    }
    invalid = [key for key, value in positive.items() if float(value) <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.adapter_bottleneck is not None and args.adapter_bottleneck < 1:
        raise ValueError("--adapter-bottleneck must be positive")
    gpus = _parse_gpus(args.gpus)
    if args.model_family == "all" and len(gpus) < 2:
        raise ValueError("the 2x2 matrix requires two physical GPU ids")

    # Names expected by the validated backend engines.
    args.local_inference_batch_size = int(args.inference_batch_size)
    args.local_epochs = int(args.ca_steps)
    args.local_max_steps_per_epoch = 1
    args.local_train_batch_size = 1
    args.local_train_query_length = int(args.ca_query_size)
    args.local_lr = float(args.ca_lr)
    args.local_weight_decay = 0.0
    args.local_scheduler = False
    args.local_better_selection = False
    args.local_exact_knn = False
    args.local_early_stopping_metric = "acc"
    args.local_early_stopping_rounds = int(args.ca_steps) + 1
    args.local_eval_interval = int(args.ca_steps)
    args.local_splits_evaluated = "valid"
    args.local_save_data = False
    args.context_size = int(args.support_size)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    root = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (REPO_ROOT / DEFAULT_OUTPUT_ROOT).resolve()
    )
    trials = matrix_trials(args.model_family, args.method)
    if len(trials) > 1:
        run_matrix(args, root)
        return

    model_family, method = trials[0]
    args.model_family = model_family
    args.method = method
    gpu = _parse_gpus(args.gpus)[0]
    args.gpus = gpu
    if gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cell_dir = root
    manifest = run_cell(args, cell_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
