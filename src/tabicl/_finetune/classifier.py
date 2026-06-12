"""TabICL classifier fine-tuning wrapper.
TabICL 分类器微调包装类。
"""

from __future__ import annotations # 允许在类定义内部使用该类名作为类型提示

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import ClassifierMixin # scikit-learn 分类器基础混合类
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score # 导入评估指标
from sklearn.utils.validation import check_is_fitted # 用于检查模型是否已经调用过 fit 方法拟合

# 导入内部模块
from tabicl._sklearn.classifier import TabICLClassifier
from tabicl._finetune.base import ValidationMetrics, FinetunedTabICLBase
from tabicl._finetune.data import MetaBatch


class FinetunedTabICLClassifier(ClassifierMixin, FinetunedTabICLBase):
    """Fine-tune a pretrained TabICL for single-dataset classification.
    对预训练的 TabICL 模型进行针对单一数据集的分类任务微调。

    Subclass of :class:`FinetunedTabICLBase` that implements cross-entropy loss on
    the raw TabICL logits and ROC-AUC / log-loss / accuracy evaluation metrics.
    它是 `FinetunedTabICLBase` 的子类，实现了在原始 TabICL 预测值 (logits) 上的交叉熵损失 (cross-entropy loss) ，
    以及 ROC-AUC、对数损失 (log-loss)、准确率 (accuracy) 等评估指标。

    Minimal usage:: (最简用法示例)

        from tabicl import FinetunedTabICLClassifier
        clf = FinetunedTabICLClassifier(epochs=30, device="cuda", verbose=True)
        # 训练模型，可以选择提供验证集
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        # 预测测试集的分类概率
        y_proba = clf.predict_proba(X_test)

    Parameters (参数说明)
    ----------

    **Optimization (优化相关设置)**

    epochs : int, default=30
        Number of passes through the fine-tuning meta-batches.
        训练的总轮数。

    learning_rate : float, default=1e-5
        AdamW learning rate.
        AdamW 优化器的学习率。

    weight_decay : float, default=0.01
        AdamW weight decay.
        AdamW 优化器的权重衰减（用于正则化，防止过拟合）。

    grad_clip : float, default=1.0
        Max global gradient norm (``0`` disables).
        全局梯度裁剪的最大范数（用于防止梯度爆炸）。如果设为 0，则不裁剪。

    amp : bool, default=True
        Use FP16 automatic mixed precision on CUDA.
        如果使用 CUDA（GPU），是否启用 FP16 自动混合精度训练以加速并节省显存。

    use_lr_scheduler : bool, default=True
        Cosine-with-warmup LR schedule.
        是否使用带有预热（warmup）的余弦退火学习率调度器。

    warmup_proportion : float, default=0.1
        Warmup fraction of total steps.
        预热阶段占总训练步数的比例。

    **Data pipeline (数据处理流水线)**

    n_estimators_finetune : int, default=2
        Ensemble size during training meta-batches.
        在训练元批次时使用的集成规模（模型并行变体数量）。

    n_estimators_validation : int, default=2
        Ensemble size during end-of-epoch validation.
        在每个 epoch 结束时进行验证使用的集成规模。

    n_estimators_inference : int, default=8
        Ensemble size of the final inner estimator used by
        :meth:`predict` / :meth:`predict_proba`.
        最终调用 `predict` 或 `predict_proba` 进行推理时使用的集成规模。
        通常推理时会使用更大的集成来提高稳定性。

    max_data_size : int, default=10_000
        Max samples per meta-dataset chunk.
        每个元数据集分块（chunk）的最大样本数。

    finetune_ctx_query_ratio : float, default=0.2
        Query fraction inside each chunk.
        每个数据块中被划分为“查询（query）”的比例（剩余的是“上下文（context）”）。

    validation_split_ratio : float, default=0.1
        Size of auto-split validation set when ``X_val`` / ``y_val`` are not
        passed to :meth:`fit`.
        如果在调用 `fit` 时没有传入验证集，默认自动从训练集中分出多大比例作为验证集。

    **Early stopping & time budget (早停与时间预算)**

    early_stopping : bool, default=True
        Stop after ``patience`` non-improving epochs.
        是否启用早停（如果连续 `patience` 个 epoch 验证指标没有提升，则停止训练）。

    patience : int, default=8
        Number of non-improving epochs tolerated.
        容忍多少个 epoch 性能不提升。

    min_delta : float, default=1e-4
        Minimum metric improvement that counts as an improvement.
        被算作“性能提升”的最小指标增量。

    time_limit : float or None, default=None
        Wall-clock budget in seconds; ``None`` disables.
        训练的挂钟时间预算（秒）；`None` 表示不限制时间。

    save_interval : int, default=1
        Write an interval checkpoint every N epochs; best is always saved.
        每隔几个 epoch 保存一次常规检查点；最好的模型权重会一直被单独保存。

    **Preprocessing (数据预处理)**

    norm_methods : str, list[str] or None, default=None
        Normalization methods forwarded to
        :class:`tabicl._sklearn.preprocessing.EnsembleGenerator`.
        传递给集成生成器的标准化方法。

    feat_shuffle_method : str, default="latin"
        Feature-permutation strategy for ensemble diversity.
        用于增加集成多样性的特征打乱策略（例如 "latin" 超拉丁方抽样）。

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier clipping during preprocessing.
        预处理时使用的异常值截断的 Z-score 阈值。

    **Model loading (模型加载)**

    model_path : str, Path or None, default=None
        Checkpoint file to fine-tune from. ``None`` → download the default
        TabICLv2 classifier checkpoint from Hugging Face Hub.
        要从中恢复或微调的检查点文件路径。如果不填，默认从 Hugging Face 自动下载。

    allow_auto_download : bool, default=True
        Permit downloading the pretrained checkpoint when it isn't cached.
        当本地没有缓存时，是否允许自动下载预训练的检查点。

    checkpoint_version : str, default="tabicl-classifier-v2-20260212.ckpt"
        Pretrained checkpoint version identifier.
        预训练检查点的版本标识符。

    **Freezing (冻结参数设定)**

    freeze_col : bool, default=False
        Freeze the column-embedding sub-module (weights and dropout/BN).
        是否冻结列嵌入（column-embedding）子模块，冻结后这部分参数在微调中不会改变。

    freeze_row : bool, default=False
        Freeze the row-interaction sub-module.
        是否冻结行交互（row-interaction）子模块。

    freeze_icl : bool, default=False
        Freeze the in-context-learning predictor.
        是否冻结上下文学习（ICL）预测器。

    **Device & logging (设备与日志)**

    device : str, torch.device or None, default=None
        Compute device; ``None`` auto-selects ``cuda`` when available.
        计算设备，如果是 None 会自动在有条件时选择 `cuda` (GPU)。

    random_state : int, default=42
        Seed for data splits and ensemble shuffle patterns.
        数据划分和集成打乱模式的随机种子，保证可复现性。

    verbose : bool, default=False
        Print a tqdm progress bar and one-line per-epoch summary.
        是否打印训练进度条和每轮的简要总结。

    wandb_kwargs : dict or None, default=None
        When provided, enables Weights & Biases tracking by instantiating
        :class:`WandbLogger(**wandb_kwargs)` on rank 0.
        如果提供了，会在主进程中启用 Weights & Biases(W&B) 实验追踪。

    **Classifier-specific (分类器特有参数)**

    class_shuffle_method : str, default="shift"
        Class-label shuffle strategy for ensemble diversity.
        增加集成多样性的类别标签打乱策略（如 "shift" 循环移位）。

    softmax_temperature : float, default=0.9
        Softmax temperature used by the inner :class:`TabICLClassifier` at
        inference time.
        推理时内部 TabICLClassifier 使用的 Softmax 温度系数（影响概率输出的平滑度）。

    average_logits : bool, default=True
        If True, ensemble averaging is done on logits; else on probabilities.
        如果为 True，多模型集成平均在 logits（未归一化的原始输出）上进行；否则在概率上进行。

    support_many_classes : bool, default=True
        Enable TabICL's mixed-radix ensembling when the dataset has more
        classes than the pretrained head's native ``max_classes``.
        如果数据集的类别数超过了预训练模型本身支持的最大类别数，是否启用混合基数集成。

    eval_metric : {"roc_auc", "log_loss", "accuracy"}, default="roc_auc"
        Primary validation metric driving early stopping and best-weight
        selection. ``log_loss`` is internally negated so "higher is better"
        holds uniformly.
        用于驱动早停策略和选择最佳权重的首要验证指标。
        如果是对数损失(`log_loss`)，代码内部会取反，从而保证所有的指标都是“越大越好”。

    extra_classifier_kwargs : dict or None, default=None
        Additional kwargs forwarded to the inner :class:`TabICLClassifier`.
        转发给内部 :class:`TabICLClassifier` 的其它参数。
    """

    def __init__(
        self,
        *,
        # Optimization
        epochs: int = 30,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        amp: bool = True,
        use_lr_scheduler: bool = True,
        warmup_proportion: float = 0.1,
        # Data pipeline
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_inference: int = 8,
        max_data_size: int = 10_000,
        finetune_ctx_query_ratio: float = 0.2,
        validation_split_ratio: float = 0.1,
        # Early stopping & time budget
        early_stopping: bool = True,
        patience: int = 8,
        min_delta: float = 1e-4,
        time_limit: Optional[float] = None,
        save_interval: int = 1,
        # Preprocessing
        norm_methods=None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        # Model loading
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt",
        # Freezing
        freeze_col: bool = False,
        freeze_row: bool = False,
        freeze_icl: bool = False,
        # Device & logging
        device: Optional[str | torch.device] = None,
        random_state: int = 42,
        verbose: bool = False,
        wandb_kwargs: Optional[dict[str, Any]] = None,
        # Classifier-specific
        class_shuffle_method: str = "shift",
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        eval_metric: Literal["roc_auc", "log_loss", "accuracy"] = "roc_auc",
        extra_classifier_kwargs: Optional[dict[str, Any]] = None,
    ):
        # 调用父类 FinetunedTabICLBase 的初始化方法
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            amp=amp,
            use_lr_scheduler=use_lr_scheduler,
            warmup_proportion=warmup_proportion,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_inference=n_estimators_inference,
            max_data_size=max_data_size,
            finetune_ctx_query_ratio=finetune_ctx_query_ratio,
            validation_split_ratio=validation_split_ratio,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            time_limit=time_limit,
            save_interval=save_interval,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            outlier_threshold=outlier_threshold,
            model_path=model_path,
            allow_auto_download=allow_auto_download,
            checkpoint_version=checkpoint_version,
            freeze_col=freeze_col,
            freeze_row=freeze_row,
            freeze_icl=freeze_icl,
            device=device,
            random_state=random_state,
            verbose=verbose,
            wandb_kwargs=wandb_kwargs,
        )
        # 保存分类器专属的配置参数
        self.class_shuffle_method = class_shuffle_method
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.support_many_classes = support_many_classes
        self.eval_metric = eval_metric
        self.extra_classifier_kwargs = extra_classifier_kwargs

    # ---- base hooks (基础钩子方法，供父类调用) ----

    @property
    def _model_type(self) -> Literal["classifier", "regressor"]:
        """指明这是一个分类器"""
        return "classifier"

    @property
    def _metric_name(self) -> str:
        """返回当前的评估指标名称（用于日志和最佳模型选择）"""
        return self.eval_metric

    def _create_inner_estimator(self, *, n_estimators: int, device: torch.device) -> TabICLClassifier:
        """Construct a fresh :class:`TabICLClassifier` with matching config.
        构造并返回一个新的底层 TabICLClassifier 模型实例，包含相关的配置。

        Note: ``self.verbose`` drives only the outer fine-tuning tqdm bar and
        epoch summary. The inner estimator defaults to ``verbose=False`` so
        its per-inference "Available GPU memory / Offload decision" logs do
        not fire on every end-of-epoch validation pass.
        注意：self.verbose 控制的是微调过程的进度条。内部的分类器默认 verbose=False，
        为了避免它在每次验证时大量输出显存/日志信息。
        """
        kwargs = dict(self.extra_classifier_kwargs or {})
        kwargs.setdefault("verbose", False) # 强制内部估计器静默，除非外部明确提供 kwargs
        kwargs.update(
            dict(
                n_estimators=n_estimators,
                norm_methods=self.norm_methods,
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                softmax_temperature=self.softmax_temperature,
                average_logits=self.average_logits,
                support_many_classes=self.support_many_classes,
                model_path=self.model_path,
                allow_auto_download=self.allow_auto_download,
                checkpoint_version=self.checkpoint_version,
                device=device,
                random_state=self.random_state,
            )
        )
        return TabICLClassifier(**kwargs)

    def _task_skip_batch(self, batch: MetaBatch) -> bool:
        """Skip batches where the query contains classes missing from context.
        决定是否跳过当前批次。
        如果当前查询集（需要预测的数据）包含在上下文（模型参考的数据）中未曾见过的类别，则跳过。

        Cross-entropy is undefined for a class index the model never saw in
        context (its logits aren't calibrated for it); skipping preserves
        training stability.
        这是因为如果模型在上下文中没见过某个类，它的输出（logits）就没针对那个类校准，
        算交叉熵就没有意义。跳过可以维持训练稳定性。
        """
        ctx = torch.unique(batch.y_train.reshape(-1)) # 上下文中的唯一类别标签
        qry = torch.unique(batch.y_query.reshape(-1)) # 查询集中的唯一类别标签
        # 判断 query 里的所有类别是不是都在 context 里。如果不是，就跳过这个 batch。
        return not bool(torch.isin(qry, ctx, assume_unique=True).all())

    def _compute_batch_loss(self, batch: MetaBatch, model) -> torch.Tensor:
        """Cross-entropy on the TabICL classifier logits.
        在 TabICL 分类器的 logits 输出上计算交叉熵损失。
        """
        # TabICL.forward signature: (X, y_train) -> (E, test_size, max_classes)
        # where E = n_estimators_finetune (ensemble as the batch dim). The
        # active-feature-count tensor ``d`` is omitted: all ensemble members
        # in a fine-tuning batch share the same feature count, so ``d=None``
        # short-circuits the mask math inside TabICL.
        # 调用模型前向传播。TabICL 前向计算返回形状为 (E, test_size, max_classes) 的 logits。
        # 这里的 E 表示集成规模，在微调时被当作批次(batch)维度处理。
        logits = model(batch.X, batch.y_train.float())
        
        # Slice down to the number of classes actually in this dataset so
        # cross-entropy targets are valid indices into the sliced logits.
        # 由于模型的输出维度可能(max_classes)比当前数据集的实际类别数大，
        # 为了计算交叉熵损失，切取实际使用的类别数的部分。
        n_classes = int(batch.y_train.max().item()) + 1 # 计算当前实际包含多少类别
        logits_used = logits[..., :n_classes].reshape(-1, n_classes) # 切片并重塑为2维，适合 PyTorch 的损失函数
        targets = batch.y_query.long().reshape(-1) # 查询集的目标标签也平铺成1维
        
        # 计算交叉熵损失
        return F.cross_entropy(logits_used, targets)

    def _run_validation(
        self,
        inner: TabICLClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ValidationMetrics:
        """Fit ``inner`` on train, predict on val, return ROC-AUC/log-loss/accuracy.
        执行验证。使用训练集让底层模型(inner)拟合，在验证集上预测，并返回 ROC-AUC / log-loss / accuracy 指标。
        """
        # The caller (:meth:`FinetunedTabICLBase._validate_current_model`) has
        # already switched the underlying module to eval mode.
        # 此时底层模块已经由父类切换到验证模式 (eval mode)。
        try:
            inner.fit(X_train, y_train) # 拟合上下文
            proba = inner.predict_proba(X_val) # 输出概率预测
        except (ValueError, RuntimeError) as e:
            # 如果在拟合或预测时出错（例如因为数据极端不平衡等问题），捕获错误并返回 NaN
            if self.verbose:
                import logging

                logging.getLogger(__name__).warning("Validation failed: %s", e)
            return ValidationMetrics(primary=float("nan"))

        secondary: dict[str, float] = {}
        try:
            # 计算 ROC-AUC 指标
            if proba.shape[1] == 2:
                # 二分类情况
                roc = float(roc_auc_score(y_val, proba[:, 1]))
            else:
                # 多分类情况（使用 ovr 策略）
                roc = float(roc_auc_score(y_val, proba, multi_class="ovr"))
            secondary["roc_auc"] = roc
        except ValueError:
            roc = float("nan")

        # 计算对数损失 (Log Loss)
        ll = float(log_loss(y_val, proba, labels=inner.classes_))
        secondary["log_loss"] = ll
        
        # 计算准确率 (Accuracy)
        acc = float(accuracy_score(y_val, np.asarray(inner.classes_)[proba.argmax(axis=1)]))
        secondary["accuracy"] = acc

        # 根据初始化时选择的评估指标 (eval_metric)，确定哪个是首要指标（用于决定模型好坏）
        if self.eval_metric == "roc_auc":
            primary = roc
        elif self.eval_metric == "log_loss":
            # 注意：因为整个系统假设"主指标越大越好"，所以这里对 loss 取负值
            primary = -ll  # higher-is-better sign convention
        elif self.eval_metric == "accuracy":
            primary = acc
        else:
            raise ValueError(f"Unsupported eval_metric: {self.eval_metric!r}")

        return ValidationMetrics(primary=primary, secondary=secondary)

    def _metric_improved(self, current: float, best: float) -> bool:
        """Higher-is-better comparison (``log_loss`` is pre-negated).
        比较当前指标是否比历史最佳指标好。因为前面把 log_loss 取负数了，所以这里总是“当前值 > 最佳值”算变好。
        """
        return current > best + self.min_delta

    def _initial_best_metric(self) -> float:
        """Sentinel for "no baseline yet" under the higher-is-better convention.
        初始化历史最佳指标为负无穷大，适用于“越大越好”的规则。
        """
        return -np.inf

    # ---- prediction surface (对外暴露的预测接口) ----

    def predict_proba(self, X):
        """Predict class probabilities for ``X``.
        预测输入数据 `X` 属于每个类别的概率。

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Probability that each sample belongs to each class, in the order
            given by :attr:`classes_`.
            返回形状为 (n_samples, n_classes) 的数组，包含预测的概率。类别顺序同 `classes_` 属性。
        """
        check_is_fitted(self, "_final_estimator_") # 检查是否已经调用过 fit 进行微调训练
        return self._final_estimator_.predict_proba(X) # 调用底层微调后的模型进行预测

    @property
    def classes_(self):
        """Class labels in the order used by :meth:`predict_proba`.
        返回模型使用的类别标签列表，顺序与 `predict_proba` 返回概率的列顺序一致。
        """
        check_is_fitted(self, "_final_estimator_")
        return self._final_estimator_.classes_
