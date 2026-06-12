"""Meta-dataset construction for TabICL single-dataset fine-tuning.
用于TabICL单数据集微调的元数据集（Meta-dataset）构建。

A fine-tuning *meta-batch* corresponds to one (context, query) split of a chunk
of the real training dataset, preprocessed into ``n_estimators_finetune``
ensemble variants (different normalizations, feature shuffles, and — for
classification — class-label shuffles). The meta-batch is the unit a single
forward pass consumes: the ensemble dimension is treated as the batch dimension
of :class:`tabicl._model.tabicl.TabICL`, so one meta-batch ≈ one
``TabICL.forward`` call.
微调时的“元批次”（meta-batch）对应于真实训练数据集一个数据块（chunk）的（上下文，查询）划分。
它被预处理成 ``n_estimators_finetune`` 个集成变体（包含了不同的标准化方法、特征打乱，对于分类任务还有标签打乱）。
元批次是单次前向传播所消耗的数据单位：集成（ensemble）维度被当作批次（batch）维度。

Per-epoch chunking with fresh context/query splits keeps the in-context
signal diverse across epochs while reusing the pretrained preprocessing
pipeline from :class:`tabicl._sklearn.preprocessing.EnsembleGenerator`.
每个epoch都会重新进行数据分块，并进行新的上下文/查询（context/query）划分，
这保证了不同epoch之间上下文信号的多样性，同时复用了预训练好的预处理流水线。
"""

from __future__ import annotations # 允许在类定义内部使用该类名作为类型提示

from collections.abc import Iterator # 迭代器类型，用于类型提示
from dataclasses import dataclass # 用于创建只包含数据的类（数据类）
from typing import List, Optional # 列表和可选类型提示

import numpy as np # 导入NumPy，用于高效的数值和数组计算
import torch # 导入PyTorch，深度学习框架
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit # 导入scikit-learn中的数据划分工具（分层随机划分、普通随机划分）

from tabicl._sklearn.preprocessing import EnsembleGenerator, TransformToNumerical # 导入项目内部的预处理模块


@dataclass
class MetaBatch:
    """A single context+query meta-batch for fine-tuning.
    微调时使用的单个上下文+查询的元批次（数据类）。

    The ensemble dimension ``E`` is the outer (batch) dimension of the tensors.
    Each of the ``E`` ensemble members sees the same underlying (context, query)
    row split but a different preprocessing pipeline + feature/class shuffle.
    集成维度 `E` 是张量的最外层（批次）维度。
    这 `E` 个集成成员看到的是相同的底层（上下文，查询）行划分，但预处理流程和特征/类别打乱方式不同。

    Attributes (属性说明)
    ----------
    X : Tensor, shape ``(E, T, H)``
        Concatenated context+query features, with ``X[:, :train_size]`` the
        context and ``X[:, train_size:]`` the query. Always float32.
        拼接后的 上下文+查询 特征。前 `train_size` 个是上下文特征，后面的是查询特征。数据类型总是float32。

    y_train : Tensor, shape ``(E, train_size)``
        Context labels (per-estimator, with class shuffle already applied for
        classification). Always float32 — the classifier forward casts to long
        where cross-entropy needs integer indices.
        上下文标签（对于分类任务已经应用了类别打乱）。总是 float32 类型。

    y_query : Tensor, shape ``(E, test_size)``
        Ground-truth query labels aligned to each ensemble member's shuffle
        pattern. Long dtype for classification; float32 (z-normalized) for
        regression.
        真实的查询标签（也就是我们需要预测的目标）。分类任务是 Long（整数）类型，回归任务是 float32 类型。

    train_size : int
        Number of context samples (``X[:, :train_size]``).
        上下文样本的数量。

    y_scaler_mean, y_scaler_std : float or None
        Regression-only: per-chunk z-norm statistics computed on the context.
        Unused (``None``) for classification.
        仅用于回归任务：在上下文数据上计算得到的均值和标准差，用于标准化。分类任务中为 None。
    """

    X: torch.Tensor
    y_train: torch.Tensor
    y_query: torch.Tensor
    train_size: int
    y_scaler_mean: Optional[float] = None
    y_scaler_std: Optional[float] = None


def count_chunks(
    n_samples: int,
    max_chunk_size: int,
    min_chunk_size: int = 50,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> int:
    """Return how many meta-batch chunks this rank will yield per epoch.
    计算每个epoch中，当前进程（rank）会产生多少个元批次数据块（chunks）。
    这主要用于分布式训练中确定进度条长度和学习率调度。

    Deterministic and cheap (no actual permutation / tensor work). Useful to
    size a tqdm progress bar and the LR scheduler before starting the epoch.
    计算过程是确定的且开销很小（没有实际的数据打乱或张量计算）。

    参数：
    n_samples: 样本总数
    max_chunk_size: 每个块的最大尺寸
    min_chunk_size: 每个块的最小尺寸，默认50
    rank: 当前进程ID（分布式训练用，单机默认为0）
    world_size: 总进程数（分布式训练用，单机默认为1）
    """
    if n_samples <= 0: # 如果没有样本，返回0
        return 0
    if n_samples <= max_chunk_size: # 如果样本总数不超过最大块大小，则只有1个块
        global_n = 1
    else:
        n_full = n_samples // max_chunk_size # 计算可以分出多少个完整大小的块
        remainder = n_samples - n_full * max_chunk_size # 计算剩下的样本数
        # 如果剩下的样本数大于等于最小块大小，则额外增加1个块，否则舍弃剩余部分
        global_n = n_full + (1 if remainder >= min_chunk_size else 0)
        
    del rank  # shard count is the same on every rank under drop_last (在使用 drop_last 的情况下，每个进程的块数是一样的)
    
    # 如果是单机（world_size <= 1）或者总块数比进程数还少（不够分），直接返回全局块数
    if world_size <= 1 or global_n < world_size:
        return global_n
        
    # 分布式情况下，平分全局块数（向下取整，即 drop_last 语义）
    return global_n // world_size


def _chunk_indices(
    n_samples: int,
    max_chunk_size: int,
    rng: np.random.Generator,
    *,
    min_chunk_size: int = 50,
) -> List[np.ndarray]:
    """Split ``range(n_samples)`` into randomly-shuffled chunks of at most
    ``max_chunk_size`` samples.
    将从 0 到 n_samples-1 的索引随机打乱，并分割成多个大小不超过 max_chunk_size 的数据块。

    Tail chunks smaller than ``min_chunk_size`` are dropped unless the whole
    dataset is smaller than ``max_chunk_size`` (in which case we keep the
    single chunk).
    如果最后一块的大小小于 min_chunk_size，则会被丢弃；但如果整个数据集都不够一个 max_chunk_size，则保留这仅有的一块。
    """
    # 随机打乱 0 到 n_samples-1 的索引
    perm = rng.permutation(n_samples)
    
    # 如果总样本数小于最大块大小，直接返回整个打乱的索引作为一个块
    if n_samples <= max_chunk_size:
        return [perm]
        
    # 计算可以分成多少个完整的块
    n_full = n_samples // max_chunk_size
    remainder = n_samples - n_full * max_chunk_size # 计算剩余的样本数
    
    # 将完整的块截取出来保存到列表中
    chunks = [perm[i * max_chunk_size : (i + 1) * max_chunk_size] for i in range(n_full)]
    
    # 如果剩余的样本数大于等于最小块大小，将其作为一个单独的块追加进去
    if remainder >= min_chunk_size:
        chunks.append(perm[n_full * max_chunk_size :])
        
    return chunks


def _split_ctx_query(
    y_chunk: np.ndarray,
    *,
    query_size: int,
    seed: int,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(context_idx, query_idx)`` into the chunk for a single split.
    将一个数据块划分为上下文（context）和查询（query），返回对应的索引。
    
    参数：
    y_chunk: 数据块的标签
    query_size: 查询部分的样本数量
    seed: 随机种子
    stratify: 是否进行分层抽样（分类任务通常需要保证各种类别的比例一致）
    """
    n = len(y_chunk) # 数据块的总长度
    query_size = max(1, min(query_size, n - 1)) # 保证查询部分的长度至少为1，且不超过 n-1
    
    # 如果是分类任务(stratify=True)则使用分层抽样，否则使用普通的随机抽样
    splitter_cls = StratifiedShuffleSplit if stratify else ShuffleSplit
    splitter = splitter_cls(n_splits=1, test_size=query_size, random_state=seed)
    
    # StratifiedShuffleSplit.split only uses X for its length — pass a dummy.
    # 分割器需要传入特征数据X和标签y，但因为这里只关心标签比例和长度，所以传一个占位的假X（dummy_X）即可
    dummy_X = np.zeros((n, 1))
    
    # 得到划分后的上下文索引和查询索引
    ctx_idx, qry_idx = next(splitter.split(dummy_X, y_chunk))
    return ctx_idx, qry_idx


def _take_rows(X, indices: np.ndarray):
    """根据给定的索引数组从特征数据X中提取特定的行。"""
    # 如果X是pandas的DataFrame或Series，使用 .iloc 按位置提取
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    # 如果X是numpy数组或其他支持切片的数据结构，直接用索引提取
    return X[indices]


def _build_ensemble_generator(
    *,
    classification: bool,
    n_estimators: int,
    norm_methods,
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    random_state: int,
) -> EnsembleGenerator:
    """构建并返回一个 EnsembleGenerator（集成生成器）对象，用于特征预处理、标准化和打乱。"""
    return EnsembleGenerator(
        classification=classification,    # 是否是分类任务
        n_estimators=n_estimators,        # 集成成员（模型）的数量
        norm_methods=norm_methods,        # 标准化方法的集合
        feat_shuffle_method=feat_shuffle_method, # 特征打乱的方法
        class_shuffle_method=class_shuffle_method, # 类别标签打乱的方法
        outlier_threshold=outlier_threshold, # 异常值阈值
        random_state=random_state,        # 随机种子
    )


def _build_meta_batch(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    *,
    classification: bool,
    n_estimators: int,
    query_size: int,
    epoch_seed: int,
    chunk_idx: int,
    norm_methods,
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    preprocessing_seed: int,
) -> MetaBatch:
    """Build one MetaBatch from one chunk of the training set.
    从训练集的一个数据块（chunk）中构建一个元批次（MetaBatch）。

    Implements the context/query split + per-member preprocessing + class/feature
    shuffling + z-normalization (regression). Returns CPU tensors — the caller
    is responsible for moving them to the device.
    实现了 上下文/查询 划分 + 每个集成成员的预处理 + 类别/特征打乱 + z标准化（针对回归）。
    返回的是存放在 CPU 上的张量（tensors），调用者需要自己负责把它们移动到 GPU 上。
    """
    # 结合 epoch 种子和 chunk 索引生成一个特定于此 chunk 的随机种子，确保不同 chunk 划分不同
    split_seed = epoch_seed + chunk_idx * 7919  # prime offset to decorrelate chunks (使用质数偏移解除相关性)
    
    if classification:
        n_classes = int(y_chunk.max()) + 1 # 计算类别数量
        query_size = max(query_size, n_classes) # 查询集大小至少要等于类别数量，以确保每个类都能出现
        
    # 将当前数据块划分为上下文索引和查询索引
    ctx_idx, qry_idx = _split_ctx_query(y_chunk, query_size=query_size, seed=split_seed, stratify=classification)

    # 根据索引获取对应的上下文和查询特征及标签
    X_ctx = _take_rows(X_chunk, ctx_idx)
    y_ctx = y_chunk[ctx_idx]
    X_qry = _take_rows(X_chunk, qry_idx)
    y_qry = y_chunk[qry_idx]

    y_mean: Optional[float] = None
    y_std: Optional[float] = None
    
    # 如果不是分类任务（即回归任务），需要对目标值 y 进行标准化处理
    if not classification:
        y_mean = float(np.mean(y_ctx)) # 计算上下文 y 的均值
        y_std = float(np.std(y_ctx))   # 计算上下文 y 的标准差
        if y_std < 1e-8: # 防止除以 0
            y_std = 1e-8
        # 使用上下文计算出的均值和标准差来标准化上下文和查询的标签
        y_ctx = (y_ctx - y_mean) / y_std
        y_qry = (y_qry - y_mean) / y_std

    # 使用 TransformToNumerical 将分类等非数值特征转化为数值特征
    x_encoder = TransformToNumerical()
    X_ctx = x_encoder.fit_transform(X_ctx) # 在上下文上拟合（fit）并转换（transform）
    X_qry = x_encoder.transform(X_qry)     # 直接应用到查询数据上（transform）

    # 构建并使用 EnsembleGenerator 进行进一步的预处理（多种标准化和特征打乱等）
    gen = _build_ensemble_generator(
        classification=classification,
        n_estimators=n_estimators,
        norm_methods=norm_methods,
        feat_shuffle_method=feat_shuffle_method,
        class_shuffle_method=class_shuffle_method,
        outlier_threshold=outlier_threshold,
        random_state=preprocessing_seed,
    )
    gen.fit(X_ctx, y_ctx) # 用上下文特征拟合
    variants = gen.transform(X_qry, mode="both") # 将转换应用到整个查询上，获取所有集成变体

    X_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    y_query_list: list[np.ndarray] = []
    train_size = len(ctx_idx) # 上下文大小

    # 遍历不同的标准化方法的变体
    for norm_method, (X_variant, y_variant) in variants.items():
        # X_variant: (E_m, T, H) 其中 T = train_size + test_size
        # y_variant: (E_m, train_size) 对于分类任务，此时标签已经被打乱过了
        X_list.append(X_variant)
        y_train_list.append(y_variant)

        # 获取当前标准化方法下特征和类别打乱的配置
        shuffle_configs = gen.ensemble_configs_[norm_method]
        for _feat_shuffle, y_pattern in shuffle_configs:
            if classification and y_pattern is not None:
                # Apply the same class remap to the query ground truth so that
                # cross-entropy compares logits to targets in the shuffled label
                # space.
                # 对于分类任务，如果进行了类别打乱（y_pattern），查询目标的标签也需要应用同样的映射，
                # 这样交叉熵计算时，预测结果才能和打乱后的目标匹配上。
                y_query_list.append(np.asarray(y_pattern)[y_qry.astype(int)])
            else:
                # 否则，直接使用未改变的查询目标标签
                y_query_list.append(y_qry)

    # 组合成最终的张量格式（Tensor）
    X_tensor = torch.from_numpy(np.concatenate(X_list, axis=0)).float()
    
    # ``y_train`` is always float32: classification labels get cast to long
    # by the cross-entropy loss site, and the model's train-time forward
    # expects float ICL labels. ``y_query`` uses long for classification
    # (so CE can index) and float32 for regression (z-normalized targets).
    # y_train 总是浮点类型（float32）。y_query 在分类任务中是整型（Long，为了交叉熵使用索引），回归任务是浮点型。
    y_train_tensor = torch.from_numpy(np.concatenate(y_train_list, axis=0)).float()
    y_query_dtype = torch.long if classification else torch.float32
    y_query_tensor = torch.from_numpy(np.stack(y_query_list, axis=0)).to(dtype=y_query_dtype)

    # 将打包好的数据实例化为 MetaBatch 对象并返回
    return MetaBatch(
        X=X_tensor,
        y_train=y_train_tensor,
        y_query=y_query_tensor,
        train_size=train_size,
        y_scaler_mean=y_mean,
        y_scaler_std=y_std,
    )


def iter_epoch_meta_batches(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    n_estimators: int,
    max_chunk_size: int,
    query_ratio: float,
    epoch_seed: int,
    preprocessing_seed: int,
    norm_methods,
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    min_chunk_size: int = 50,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[MetaBatch]:
    """Yield one :class:`MetaBatch` per chunk of ``(X, y)`` for a single epoch.
    在一个 epoch 中，针对数据集 (X, y) 的每一个 chunk 产出一个 MetaBatch（生成器函数）。

    Each call regenerates the chunking permutation using ``epoch_seed``, so
    successive epochs see different random chunks / different (context, query)
    splits within each chunk. Preprocessors inside each chunk are seeded with
    ``preprocessing_seed`` (fixed across epochs) so normalization and shuffles
    are stable — only the *samples* seen as context vs query change.
    每次调用都会用 epoch_seed 重新打乱生成 chunk，所以不同 epoch 的划分是不同的。
    但是 chunk 内部预处理的种子 preprocessing_seed 是固定的，因此只要输入不变，预处理行为（标准化和打乱）是一致的。

    Under DDP (``world_size > 1``) the chunk list is split across ranks with
    drop_last semantics: rank ``r`` yields chunks ``[r * k, (r + 1) * k)``
    where ``k = global_n_chunks // world_size``. The per-chunk ``chunk_idx``
    passed into :func:`_build_meta_batch` is the *global* index, so every
    rank's sharded output is a bit-identical subset of the single-GPU
    stream (preprocessing seeds derive from ``chunk_idx`` and must not
    depend on rank). If ``global_n_chunks < world_size`` every rank falls
    back to the whole list (replication).
    在分布式数据并行（DDP）中，各个 chunk 会被平均分配到每个 GPU（进程 rank）上。
    """
    # 初始化随机器，根据 epoch 种子设定状态
    rng = np.random.default_rng(epoch_seed)
    
    # 按照设定的最大和最小大小，将整个数据集划分为多个 chunk，得到它们在数据集中的索引
    chunks = _chunk_indices(len(y), max_chunk_size=max_chunk_size, rng=rng, min_chunk_size=min_chunk_size)

    # 针对分布式训练，将所有的 chunk 分摊给每个进程（rank）
    if world_size > 1 and len(chunks) >= world_size:
        per_rank = len(chunks) // world_size # 每个进程应该处理的 chunk 数量
        start = rank * per_rank              # 当前进程负责的起始位置
        # 截取分配给当前进程的那部分 chunk（带有全局索引枚举）
        sharded = list(enumerate(chunks))[start : start + per_rank]
    else:
        # 如果是单机或者是数据块太少，每个进程处理所有的 chunk
        sharded = list(enumerate(chunks))

    # 迭代当前进程所负责的 chunk 列表
    for chunk_idx, indices in sharded:
        # 取出当前 chunk 的特征和标签
        X_chunk = _take_rows(X, indices)
        y_chunk = y[indices]
        
        # 根据 query_ratio 计算查询样本的数量
        query_size = max(1, int(len(indices) * query_ratio))
        
        # 生成并 yield (产生) 一个 MetaBatch
        yield _build_meta_batch(
            X_chunk,
            y_chunk,
            classification=classification,
            n_estimators=n_estimators,
            query_size=query_size,
            epoch_seed=epoch_seed,
            chunk_idx=chunk_idx,  # 传入全局索引，确保分布式下不同 rank 处理时的状态也是确定可复现的
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            class_shuffle_method=class_shuffle_method,
            outlier_threshold=outlier_threshold,
            preprocessing_seed=preprocessing_seed,
        )


def move_meta_batch(batch: MetaBatch, device: torch.device) -> MetaBatch:
    """Return a copy of ``batch`` with tensors on ``device``.
    工具函数：将 MetaBatch 中的所有张量（Tensors）移动到指定的设备（比如 GPU 或 CPU）上。
    并返回一个新的 MetaBatch。non_blocking=True 可以让数据传输在后台异步执行以提升效率。
    """
    return MetaBatch(
        X=batch.X.to(device, non_blocking=True),
        y_train=batch.y_train.to(device, non_blocking=True),
        y_query=batch.y_query.to(device, non_blocking=True),
        train_size=batch.train_size,
        y_scaler_mean=batch.y_scaler_mean,
        y_scaler_std=batch.y_scaler_std,
    )
