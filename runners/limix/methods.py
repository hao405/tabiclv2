"""Method primitives for the LimiX-2M data184 matrix.

This module is deliberately independent from the benchmark process manager so
that parameter-selection and context-construction behavior can be tested with
small fake models and arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextEpisode:
    context_indices: np.ndarray
    query_indices: np.ndarray
    route_id: int | None = None
    fallback: str | None = None
    fallback_reason: str | None = None


def count_parameters(model: Any) -> tuple[int, int, float]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return trainable, total, float(trainable) / float(total) if total else 0.0


def freeze_all(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def configure_full_finetune(model: Any) -> list[Any]:
    for parameter in model.parameters():
        parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def configure_last_block(model: Any) -> list[Any]:
    freeze_all(model)
    layers = getattr(getattr(model, "transformer_encoder", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("LimiX Last-block requires transformer_encoder.layers")
    for parameter in layers[-1].parameters():
        parameter.requires_grad = True
    decoder = getattr(model, "cls_y_decoder", None)
    if decoder is None:
        raise RuntimeError("LimiX Last-block requires cls_y_decoder")
    for parameter in decoder.parameters():
        parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _torch():
    import torch

    return torch


class LoRAWeightParametrization:
    """Factory namespace for reshape-safe LoRA parametrizations."""

    @staticmethod
    def create(weight: Any, *, rank: int, alpha: float, dropout: float) -> Any:
        torch = _torch()

        class _LoRA(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                output_size = int(np.prod(tuple(weight.shape[:-1])))
                input_size = int(weight.shape[-1])
                self.original_shape = tuple(int(value) for value in weight.shape)
                self.scaling = float(alpha) / float(rank)
                self.dropout = float(dropout)
                self.lora_A = torch.nn.Parameter(
                    torch.empty(
                        int(rank),
                        input_size,
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                )
                self.lora_B = torch.nn.Parameter(
                    torch.zeros(
                        output_size,
                        int(rank),
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                )
                torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            def forward(self, base_weight: Any) -> Any:
                lora_a = self.lora_A
                if self.dropout > 0:
                    lora_a = torch.nn.functional.dropout(
                        lora_a,
                        p=self.dropout,
                        training=self.training,
                    )
                delta = (self.lora_B @ lora_a).reshape(self.original_shape)
                return base_weight + delta.to(base_weight.dtype) * self.scaling

        return _LoRA()


def _install_lora_parameter(
    module: Any,
    parameter_name: str,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> bool:
    torch = _torch()
    from torch.nn.utils import parametrize

    if not hasattr(module, parameter_name):
        return False
    weight = getattr(module, parameter_name)
    if not isinstance(weight, torch.Tensor) or weight.ndim < 2:
        return False
    if parametrize.is_parametrized(module, parameter_name):
        return False
    parametrization = LoRAWeightParametrization.create(
        weight,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    parametrize.register_parametrization(module, parameter_name, parametrization)
    getattr(module.parametrizations, parameter_name).original.requires_grad = False
    return True


def configure_lora(
    model: Any,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> tuple[list[Any], int]:
    """Install LoRA on all compatible LimiX linear and attention projections."""

    if rank < 1:
        raise ValueError("LoRA rank must be >= 1")
    if alpha <= 0:
        raise ValueError("LoRA alpha must be > 0")
    if not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")

    torch = _torch()
    freeze_all(model)
    installed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            installed += int(
                _install_lora_parameter(
                    module,
                    "weight",
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
            )
        for parameter_name in ("qkv_proj_weight", "out_proj_weight"):
            installed += int(
                _install_lora_parameter(
                    module,
                    parameter_name,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
            )
    if installed == 0:
        raise RuntimeError("LoRA found no compatible LimiX weights")

    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_A" in name or ".lora_B" in name
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA installed parametrizations but selected no parameters")
    return trainable, installed


def is_lora_trainable_name(name: str) -> bool:
    return ".lora_A" in name or ".lora_B" in name


def _prefix_attention_wrapper(base_attention: Any, prefix_length: int) -> Any:
    torch = _torch()

    class PrefixSequenceAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_attention = base_attention
            self.embed_dim = int(base_attention.embed_dim)
            self.num_heads = int(base_attention.num_heads)
            self.head_dim = int(base_attention.head_dim)
            self.dropout = float(base_attention.dropout)
            dtype = base_attention.qkv_proj_weight.dtype
            device = base_attention.qkv_proj_weight.device
            self.prefix_key = torch.nn.Parameter(
                torch.empty(
                    prefix_length,
                    self.num_heads,
                    self.head_dim,
                    dtype=dtype,
                    device=device,
                )
            )
            self.prefix_value = torch.nn.Parameter(
                torch.empty(
                    prefix_length,
                    self.num_heads,
                    self.head_dim,
                    dtype=dtype,
                    device=device,
                )
            )
            torch.nn.init.normal_(self.prefix_key, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.prefix_value, mean=0.0, std=0.02)

        def forward(
            self,
            x: Any,
            x_kv: Any | None = None,
            copy_first_head_kv: bool = False,
            attn_mask: Any | None = None,
            calculate_sample_attention: bool = False,
            calculate_feature_attention: bool = False,
        ) -> tuple[Any, Any | None, Any | None]:
            if x_kv is None:
                raise ValueError("Prefix sequence attention requires x_kv")
            if attn_mask is not None:
                raise ValueError("Prefix sequence attention does not support an explicit mask")

            batch, sequence, _, _ = x.shape
            x_flat = x.reshape(-1, *x.shape[-2:])
            kv_flat = x_kv.reshape(-1, *x_kv.shape[-2:])
            batch_features, query_length, _ = x_flat.shape
            q_weight = self.base_attention.qkv_proj_weight[0]
            kv_weight = self.base_attention.qkv_proj_weight[1:]
            q = torch.einsum("... s, h d s -> ... h d", x_flat, q_weight)
            if copy_first_head_kv:
                kv = torch.einsum(
                    "... s, j h d s -> ... j h d",
                    kv_flat,
                    kv_weight[:, :1],
                )
                expand_shape = [-1 for _ in kv.shape]
                expand_shape[-2] = self.num_heads
                kv = kv.expand(*expand_shape)
            else:
                kv = torch.einsum(
                    "... s, j h d s -> ... j h d",
                    kv_flat,
                    kv_weight,
                )
            key, value = kv.unbind(dim=2)
            prefix_key = self.prefix_key.unsqueeze(0).expand(batch_features, -1, -1, -1)
            prefix_value = self.prefix_value.unsqueeze(0).expand(
                batch_features,
                -1,
                -1,
                -1,
            )
            key = torch.cat([prefix_key, key], dim=1)
            value = torch.cat([prefix_value, value], dim=1)
            attention = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
            ).transpose(1, 2)
            output = torch.einsum(
                "... h d, h d s -> ... s",
                attention,
                self.base_attention.out_proj_weight,
            )
            sample_attention = None
            if calculate_sample_attention:
                logits = torch.einsum("b q h d,b k h d->b q k h", q[-1:], key[-1:])
                sample_attention = torch.softmax(
                    logits.float() / math.sqrt(self.head_dim),
                    dim=2,
                ).mean(dim=-1)
            return (
                output.reshape(batch, sequence, *output.shape[1:]),
                None if not calculate_feature_attention else None,
                sample_attention,
            )

    return PrefixSequenceAttention()


def configure_prefix(
    model: Any,
    *,
    prefix_length: int = 8,
) -> tuple[list[Any], int]:
    if prefix_length < 1:
        raise ValueError("prefix_length must be >= 1")
    freeze_all(model)
    layers = getattr(getattr(model, "transformer_encoder", None), "layers", None)
    if layers is None:
        raise RuntimeError("LimiX Prefix requires transformer_encoder.layers")
    installed = 0
    for layer in layers:
        attentions = getattr(layer, "sequence_attentions", None)
        if attentions is None:
            continue
        for index, attention in enumerate(list(attentions)):
            attentions[index] = _prefix_attention_wrapper(attention, prefix_length)
            installed += 1
    if installed == 0:
        raise RuntimeError("Prefix found no LimiX sequence attention modules")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.endswith("prefix_key") or name.endswith("prefix_value")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return trainable, installed


def _residual_adapter(width: int, bottleneck: int) -> Any:
    torch = _torch()

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

    return ResidualAdapter()


def _layer_with_adapter(base_layer: Any, width: int, bottleneck: int) -> Any:
    torch = _torch()

    class LayerWithAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = base_layer
            self.adapter = _residual_adapter(width, bottleneck)

        def forward(self, value: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
            output = self.base_layer(value, **kwargs)
            if not isinstance(output, tuple) or len(output) != 3:
                raise TypeError("LimiX encoder layer must return a three-item tuple")
            return self.adapter(output[0]), output[1], output[2]

    return LayerWithAdapter()


def configure_micp_adapters(
    model: Any,
    *,
    bottleneck: int | None = None,
) -> tuple[list[Any], int]:
    freeze_all(model)
    layer_stack = getattr(model, "transformer_encoder", None)
    layers = getattr(layer_stack, "layers", None)
    if layers is None:
        raise RuntimeError("MICP requires transformer_encoder.layers")
    width = int(getattr(model, "embed_dim", 0))
    if width <= 0:
        raise RuntimeError("MICP requires model.embed_dim")
    hidden = int(bottleneck) if bottleneck is not None else max(8, width // 16)
    for index, layer in enumerate(list(layers)):
        layers[index] = _layer_with_adapter(layer, width, hidden)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".adapter." in name
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return trainable, len(layers)


MISSING_CATEGORY = "__limix_context_missing__"


class RetrievalPreprocessor:
    def __init__(self, categorical_indices: Sequence[int]) -> None:
        self.categorical_indices = sorted({int(index) for index in categorical_indices})
        self.transformer: Any | None = None
        self.fitted_rows = 0

    def _normalized_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        normalized = X.copy()
        for index in self.categorical_indices:
            column = normalized.columns[index]
            normalized[column] = (
                normalized[column]
                .astype("string")
                .fillna(MISSING_CATEGORY)
                .astype(str)
            )
        return normalized

    def fit(self, X_train: pd.DataFrame) -> "RetrievalPreprocessor":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        all_indices = list(range(X_train.shape[1]))
        numeric_indices = [
            index for index in all_indices if index not in set(self.categorical_indices)
        ]
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
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    self.categorical_indices,
                )
            )
        if not pieces:
            raise ValueError("Retrieval preprocessing requires at least one feature")
        self.transformer = ColumnTransformer(pieces, sparse_threshold=0.0)
        self.transformer.fit(self._normalized_frame(X_train))
        self.fitted_rows = len(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("RetrievalPreprocessor must be fit before transform")
        return np.asarray(
            self.transformer.transform(self._normalized_frame(X)),
            dtype=np.float32,
        )


class NeighborIndex:
    def __init__(
        self,
        backend: Literal["auto", "faiss", "sklearn"] = "auto",
    ) -> None:
        self.requested_backend = backend
        self.backend: str | None = None
        self.index: Any | None = None
        self.X: np.ndarray | None = None
        self._squared_norms: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NeighborIndex":
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.X = X
        self._squared_norms = np.sum(X * X, axis=1)
        if self.requested_backend in {"auto", "faiss"}:
            try:
                import faiss

                self.index = faiss.IndexFlatL2(X.shape[1])
                self.index.add(X)
                self.backend = "faiss"
                return self
            except ImportError:
                if self.requested_backend == "faiss":
                    raise RuntimeError("FAISS requested but faiss is not installed")
        # NumPy exact search avoids the native pairwise-distance dispatcher,
        # which segfaults in the repository's current Python 3.13 sklearn
        # build. It preserves the same exact Euclidean-neighbor semantics.
        self.index = X
        self.backend = "numpy_exact"
        return self

    def query(self, X_query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.X is None:
            raise RuntimeError("NeighborIndex must be fit before query")
        count = min(max(1, int(k)), len(self.X))
        query = np.ascontiguousarray(X_query, dtype=np.float32)
        if self.backend == "faiss":
            distances, indices = self.index.search(query, count)
            return np.sqrt(np.maximum(distances, 0.0)), indices
        assert self._squared_norms is not None
        all_distances: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        for start in range(0, len(query), 256):
            batch = query[start : start + 256]
            squared = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + self._squared_norms[None, :]
                - 2.0 * (batch @ self.X.T)
            )
            # argpartition limits sorting work, followed by a stable local sort.
            partition = np.argpartition(squared, kth=count - 1, axis=1)[:, :count]
            partition_distances = np.take_along_axis(squared, partition, axis=1)
            local_order = np.argsort(partition_distances, axis=1, kind="stable")
            indices = np.take_along_axis(partition, local_order, axis=1)
            selected = np.take_along_axis(squared, indices, axis=1)
            all_indices.append(indices)
            all_distances.append(np.sqrt(np.maximum(selected, 0.0)))
        return np.concatenate(all_distances), np.concatenate(all_indices)


def default_support_size(n_train: int) -> int:
    return min(3000, n_train)


def stratified_context_indices(
    y_train: np.ndarray,
    size: int,
    *,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(y_train)
    size = min(max(int(size), len(np.unique(labels))), len(labels))
    rng = np.random.default_rng(seed)
    selected = [
        int(rng.choice(np.flatnonzero(labels == label)))
        for label in np.unique(labels)
    ]
    remaining = np.setdiff1d(np.arange(len(labels)), selected)
    if len(selected) < size:
        selected.extend(
            rng.choice(remaining, size=size - len(selected), replace=False).tolist()
        )
    rng.shuffle(selected)
    return np.asarray(selected[:size], dtype=int)


def ensure_label_coverage(
    candidate_indices: Sequence[int],
    *,
    y_train: np.ndarray,
    train_space: np.ndarray,
    query_vector: np.ndarray | None,
    target_size: int,
    seed: int,
    exclude_indices: Sequence[int] = (),
) -> tuple[np.ndarray, str | None, str | None]:
    labels = np.asarray(y_train)
    target_size = min(max(1, int(target_size)), len(labels))
    excluded = {int(index) for index in exclude_indices}
    chosen = list(
        dict.fromkeys(
            int(index)
            for index in candidate_indices
            if 0 <= int(index) < len(labels) and int(index) not in excluded
        )
    )
    reference = (
        np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query_vector is not None
        else np.mean(train_space[chosen or [0]], axis=0, keepdims=True)
    )
    fallback = None
    reason = None
    missing = set(np.unique(labels)) - set(labels[np.asarray(chosen, dtype=int)])
    if missing:
        fallback = "class_aware_repair"
        reason = f"missing labels before repair: {sorted(missing, key=str)}"
    for label in sorted(missing, key=str):
        candidates = np.asarray(
            [
                index
                for index in np.flatnonzero(labels == label)
                if int(index) not in excluded
            ],
            dtype=int,
        )
        if not len(candidates):
            continue
        distances = np.sum((train_space[candidates] - reference) ** 2, axis=1)
        replacement = int(candidates[int(np.argmin(distances))])
        if len(chosen) < target_size:
            chosen.append(replacement)
        else:
            replace_position = next(
                (
                    position
                    for position in range(len(chosen) - 1, -1, -1)
                    if np.sum(labels[np.asarray(chosen)] == labels[chosen[position]]) > 1
                ),
                None,
            )
            if replace_position is not None:
                chosen[replace_position] = replacement
    chosen = list(dict.fromkeys(chosen))
    order = np.argsort(np.sum((train_space - reference) ** 2, axis=1))
    for index in order:
        if len(chosen) >= target_size:
            break
        if int(index) not in chosen and int(index) not in excluded:
            chosen.append(int(index))
    if set(np.unique(labels)) - set(labels[np.asarray(chosen, dtype=int)]):
        repaired = [
            int(index)
            for index in stratified_context_indices(
                labels,
                min(len(labels), target_size + len(excluded)),
                seed=seed,
            )
            if int(index) not in excluded
        ][:target_size]
        return (
            np.asarray(repaired, dtype=int),
            "stratified_context",
            "nearest-neighbor repair could not cover all train labels",
        )
    return np.asarray(chosen[:target_size], dtype=int), fallback, reason


class MICPClusterRouter:
    """FAISS-first K-Means router with a deterministic NumPy fallback."""

    def __init__(
        self,
        n_clusters: int,
        *,
        seed: int,
        n_init: int = 10,
        max_iter: int = 100,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.cluster_centers_: np.ndarray | None = None
        self.backend: str | None = None

    @staticmethod
    def _assign(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
        assignments: list[np.ndarray] = []
        inertia = 0.0
        center_norms = np.sum(centers * centers, axis=1)
        for start in range(0, len(X), 1024):
            batch = X[start : start + 1024]
            squared = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + center_norms[None, :]
                - 2.0 * (batch @ centers.T)
            )
            current = np.argmin(squared, axis=1)
            assignments.append(current)
            inertia += float(
                np.maximum(squared[np.arange(len(batch)), current], 0.0).sum()
            )
        return np.concatenate(assignments).astype(int), inertia

    def _fit_numpy(self, X: np.ndarray) -> np.ndarray:
        best_centers = None
        best_assignments = None
        best_inertia = float("inf")
        for init_index in range(self.n_init):
            rng = np.random.default_rng(self.seed + init_index)
            centers = X[
                rng.choice(len(X), size=self.n_clusters, replace=False)
            ].copy()
            assignments = np.full(len(X), -1, dtype=int)
            for _ in range(self.max_iter):
                new_assignments, _ = self._assign(X, centers)
                if np.array_equal(assignments, new_assignments):
                    assignments = new_assignments
                    break
                assignments = new_assignments
                new_centers = centers.copy()
                for cluster_id in range(self.n_clusters):
                    members = X[assignments == cluster_id]
                    if len(members):
                        new_centers[cluster_id] = members.mean(axis=0)
                    else:
                        nearest_distance = np.sum(
                            (X - centers[assignments]) ** 2,
                            axis=1,
                        )
                        new_centers[cluster_id] = X[int(np.argmax(nearest_distance))]
                if np.allclose(centers, new_centers, rtol=0.0, atol=1e-6):
                    centers = new_centers
                    break
                centers = new_centers
            assignments, inertia = self._assign(X, centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_assignments = assignments.copy()
        assert best_centers is not None and best_assignments is not None
        self.cluster_centers_ = np.asarray(best_centers, dtype=np.float32)
        self.backend = "numpy_lloyd"
        return best_assignments

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(X, dtype=np.float32)
        try:
            import faiss

            trainer = faiss.Kmeans(
                values.shape[1],
                self.n_clusters,
                niter=self.max_iter,
                nredo=self.n_init,
                seed=self.seed,
                verbose=False,
                gpu=False,
            )
            trainer.train(values)
            self.cluster_centers_ = np.asarray(
                trainer.centroids,
                dtype=np.float32,
            ).reshape(self.n_clusters, values.shape[1])
            self.backend = "faiss_kmeans"
            assignments, _ = self._assign(values, self.cluster_centers_)
            return assignments
        except ImportError:
            return self._fit_numpy(values)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("MICPClusterRouter must be fit before predict")
        assignments, _ = self._assign(
            np.ascontiguousarray(X, dtype=np.float32),
            self.cluster_centers_,
        )
        return assignments


class MixtureContextBuilder:
    def __init__(
        self,
        train_space: np.ndarray,
        y_train: np.ndarray,
        *,
        support_size: int,
        n_clusters: int | None,
        seed: int,
        backend: Literal["auto", "faiss", "sklearn"],
        gamma: float = 1.0,
    ) -> None:
        self.train_space = np.asarray(train_space, dtype=np.float32)
        self.y_train = np.asarray(y_train)
        self.support_size = min(max(1, int(support_size)), len(y_train))
        if gamma <= 0:
            raise ValueError("MICP gamma must be positive")
        self.gamma = float(gamma)
        self.n_clusters = max(
            1,
            min(
                len(y_train),
                int(n_clusters)
                if n_clusters is not None
                else int(
                    math.ceil(
                        self.gamma * len(y_train) / self.support_size
                    )
                ),
            ),
        )
        self.seed = int(seed)
        self.kmeans = MICPClusterRouter(
            n_clusters=self.n_clusters,
            seed=seed,
            n_init=10,
        )
        self.assignments = self.kmeans.fit_predict(self.train_space)
        self.supports = self._build_supports()
        self.anchor_index = NeighborIndex(backend).fit(self.train_space)

    def _build_supports(self) -> dict[int, np.ndarray]:
        supports: dict[int, np.ndarray] = {}
        for route_id in range(self.n_clusters):
            members = np.flatnonzero(self.assignments == route_id)
            centroid = self.kmeans.cluster_centers_[route_id]
            if len(members) > self.support_size:
                # MICP Eq. (2): oversized clusters are randomly sampled.
                rng = np.random.default_rng(self.seed + route_id)
                support = rng.choice(
                    members,
                    size=self.support_size,
                    replace=False,
                )
            elif len(members) < self.support_size:
                # MICP Eq. (2): undersized clusters are expanded with the
                # B nearest training examples to the cluster centroid.
                order = np.argsort(
                    np.sum((self.train_space - centroid) ** 2, axis=1),
                    kind="stable",
                )
                support = order[: self.support_size]
            else:
                support = members
            supports[route_id] = np.asarray(support, dtype=int)
        return supports

    def route(self, query_space: np.ndarray) -> list[ContextEpisode]:
        route_ids = self.kmeans.predict(np.asarray(query_space, dtype=np.float32))
        return [
            ContextEpisode(
                context_indices=self.supports[int(route_id)].copy(),
                query_indices=np.asarray([query_index], dtype=int),
                route_id=int(route_id),
            )
            for query_index, route_id in enumerate(route_ids)
        ]

    def bootstrap_episode(
        self,
        anchor_index: int,
        *,
        query_batch_size: int = 64,
        small_train_fraction: float = 0.9,
    ) -> ContextEpisode:
        if query_batch_size < 1:
            raise ValueError("MICP query batch size must be positive")
        if not 0 < small_train_fraction < 1:
            raise ValueError("MICP small-train fraction must be in (0, 1)")

        rng = np.random.default_rng(self.seed + int(anchor_index))
        if len(self.y_train) > self.support_size:
            # Section 3.3.1: sample x, obtain exactly B nearest neighbors,
            # then split that bootstrap set into labelled prompt and decoder
            # queries. N_batch=64 is fixed by Appendix 17.
            _, nearest = self.anchor_index.query(
                self.train_space[[int(anchor_index)]],
                self.support_size,
            )
            bootstrap = np.asarray(nearest[0], dtype=int)
            rng.shuffle(bootstrap)
            query_size = min(
                int(query_batch_size),
                max(1, len(bootstrap) - len(np.unique(self.y_train[bootstrap]))),
            )
            strategy = "large_knn_b_then_batch64"
        else:
            # Section 3.3.2: MICP routing is disabled for N<=B and C_A PFN
            # uses a random 90/10 downstream bootstrap.
            bootstrap = rng.permutation(len(self.y_train)).astype(int)
            query_size = max(
                1,
                len(bootstrap) - int(math.floor(len(bootstrap) * small_train_fraction)),
            )
            query_size = min(
                query_size,
                max(1, len(bootstrap) - len(np.unique(self.y_train[bootstrap]))),
            )
            strategy = "small_random_90_10"

        # PFN decoder labels must be represented in the labelled context.
        # Prefer query rows whose class has another occurrence in D_bootstrap;
        # this is a feasibility constraint, not a support-set modification.
        remaining = {
            label: int(np.sum(self.y_train[bootstrap] == label))
            for label in np.unique(self.y_train[bootstrap])
        }
        selected_query: list[int] = []
        for index in bootstrap:
            label = self.y_train[int(index)]
            if remaining[label] <= 1:
                continue
            selected_query.append(int(index))
            remaining[label] -= 1
            if len(selected_query) == query_size:
                break
        if not selected_query:
            raise ValueError("MICP bootstrap cannot form a label-covered query")
        query = np.asarray(selected_query, dtype=int)
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
                f"MICP bootstrap query labels absent from context: {sorted(missing, key=str)}"
            )
        return ContextEpisode(
            context_indices=context,
            query_indices=query,
            route_id=int(self.assignments[int(anchor_index)]),
            fallback=None,
            fallback_reason=strategy,
        )


def group_episodes_by_context(
    episodes: Iterable[ContextEpisode],
) -> list[ContextEpisode]:
    groups: dict[tuple[int, ...], dict[str, Any]] = {}
    for episode in episodes:
        key = tuple(int(index) for index in episode.context_indices)
        item = groups.setdefault(
            key,
            {"queries": [], "route_id": episode.route_id, "fallbacks": [], "reasons": []},
        )
        item["queries"].extend(int(index) for index in episode.query_indices)
        if episode.fallback:
            item["fallbacks"].append(episode.fallback)
        if episode.fallback_reason:
            item["reasons"].append(episode.fallback_reason)
    return [
        ContextEpisode(
            context_indices=np.asarray(key, dtype=int),
            query_indices=np.asarray(item["queries"], dtype=int),
            route_id=item["route_id"],
            fallback="|".join(sorted(set(item["fallbacks"]))) or None,
            fallback_reason="|".join(sorted(set(item["reasons"]))) or None,
        )
        for key, item in groups.items()
    ]


def faware_reserved_indices(
    train_space: np.ndarray,
    test_space: np.ndarray,
    *,
    reserve_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve train rows farthest from the unlabeled test distribution mean."""

    if not 0 < reserve_ratio < 1:
        raise ValueError("reserve_ratio must be in (0, 1)")
    train_space = np.asarray(train_space, dtype=np.float32)
    test_space = np.asarray(test_space, dtype=np.float32)
    test_mean = np.mean(test_space, axis=0, keepdims=True)
    scores = np.sqrt(np.sum((train_space - test_mean) ** 2, axis=1))
    count = min(len(train_space) - 1, max(1, int(math.ceil(len(train_space) * reserve_ratio))))
    reserved = np.argsort(-scores, kind="stable")[:count]
    return np.sort(reserved.astype(int)), scores


def split_context_query(
    y: np.ndarray,
    *,
    query_ratio: float,
    seed: int,
    forced_context: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split one chunk while preserving every query label in context."""

    from sklearn.model_selection import train_test_split

    labels = np.asarray(y)
    all_indices = np.arange(len(labels))
    forced = np.asarray(sorted({int(index) for index in forced_context}), dtype=int)
    allowed = np.setdiff1d(all_indices, forced)
    if len(allowed) < 2:
        raise ValueError("fewer than two non-reserved rows remain for query selection")
    query_size = min(
        len(allowed) - 1,
        max(1, int(math.ceil(len(labels) * query_ratio))),
    )
    stratify = labels[allowed]
    counts = pd.Series(stratify).value_counts()
    can_stratify = (
        len(counts) > 1
        and int(counts.min()) >= 2
        and query_size >= len(counts)
    )
    if can_stratify:
        context_allowed, query = train_test_split(
            allowed,
            test_size=query_size,
            random_state=seed,
            stratify=stratify,
        )
        strategy = "stratified"
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(allowed)
        query = shuffled[:query_size]
        context_allowed = shuffled[query_size:]
        strategy = "random"
    context = np.unique(np.concatenate([forced, np.asarray(context_allowed, dtype=int)]))
    missing = set(labels[np.asarray(query, dtype=int)]) - set(labels[context])
    if missing:
        query_list = list(np.asarray(query, dtype=int))
        for label in sorted(missing, key=str):
            candidate = next(
                (index for index in query_list if labels[index] == label),
                None,
            )
            if candidate is not None:
                query_list.remove(candidate)
                context = np.append(context, candidate)
        query = np.asarray(query_list, dtype=int)
        strategy += "_coverage_repair"
    if len(query) == 0:
        raise ValueError("context/query split left no query rows")
    return np.asarray(context, dtype=int), np.asarray(query, dtype=int), strategy
