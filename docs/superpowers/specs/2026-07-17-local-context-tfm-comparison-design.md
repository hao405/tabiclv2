# LoCalPFN / MixturePFN 双模型比较 Runner 设计

## 目标与边界

在 `PEFT_Tabicl/compare_way.py` 中实现一个完整、独立的分类 benchmark runner，比较
LoCalPFN 风格的局部检索全量微调与 MixturePFN 风格的路由 context + CaPFN adapter
微调。runner 只支持：

- 模型：`tabiclv2`、`tabpfnv3`
- 方法：`localpfn`、`mixturepfn`
- `--model-family all`：展开为两个模型与两个方法的四个实验 cell

实现不导入、不调用、也不修改 `PEFT_Tabicl/1C_Chunk_PEFT.py`。可以复刻该 runner
已经验证的数据读取、模型加载、并行 worker、结果写入和诊断行为，但新文件必须独立运行。

## 统一 episode 与模型接口

`ContextEpisode` 记录 `context_indices`、`query_indices`、`route_id`、fallback 类型与
说明。检索、路由和 fallback 只返回 episode，不直接耦合具体模型。

TabICLv2 和 TabPFNv3 各自提供 native forward adapter，将同一个 episode 转换为模型
所需的 context/query tensor，并返回 query logits。训练和 validation/test prediction
都必须经过局部 context，不允许最终预测退回普通全训练集 inference。

## LoCalPFN

- 仅用 train split 拟合检索预处理器：数值列 median imputation + standardization，
  类别列 most-frequent imputation + one-hot。
- context size 默认 `min(10 * sqrt(n_train), 1000)`。
- 每个训练 query/anchor 构造自己的 class-aware KNN context，使用 AdamW、`lr=1e-5`、
  21 epochs、每 epoch 最多 30 个 optimizer steps 做 full FT。
- validation/test 每个 query 分别检索 context，并按 context signature 合并可共同
  forward 的 query；语义上仍是 per-query context。
- 邻域未覆盖训练标签时，先按缺失类别选取最近 train row 扩展/替换；仍不能满足时，
  使用等长 stratified train context，并记录 fallback。

## MixturePFN

- 仅在 train retrieval space 上执行 KMeans。
- support size 默认 `min(3000, n_train)`；cluster 数默认
  `max(1, ceil(n_train / support_size))`。
- cluster 大于 support 时分层下采样；小于 support 时按到 centroid 的距离从全部 train
  rows 补齐，最终再执行标签覆盖修复。
- 每个 query 路由到最近 centroid，同一 route 始终使用同一固定 support context。
- CaPFN 训练从 train 中采样 anchor，取 anchor 的 KNN bootstrap 邻域，再切分
  context/query；默认执行 128 optimizer steps，Adam `lr=1e-3`。
- 冻结 backbone，只训练插在每个 ICL block 输出后的 residual adapter：
  `LayerNorm -> Linear(d, max(8, d/16)) -> GELU -> Linear -> residual`。最后一层
  projection 零初始化，使 step 0 严格为恒等映射。
- TabICLv2 adapter 目标为 `icl_predictor.tf_icl.blocks`；TabPFNv3 目标为
  `icl_blocks`。

## 数据隔离、容错与输出

- train：预处理、索引、KMeans、support 构造与微调。
- validation：仅 early stopping / 最优 checkpoint 选择。
- test features：仅检索或路由；test labels：仅最终 metrics。
- 小数据退化为单 route；标签覆盖 fallback 必须显式计数和落盘。
- OOM 仅允许 query micro-batch 减半后重试一次，不缩小 context，不退回普通 inference。
  训练 OOM 或第二次预测 OOM 将该 dataset 标记失败。
- validation 最优状态从 step 0/base 状态开始比较，允许阻止负迁移。

除标准分类结果外，CSV/summary/manifest 还记录 method、retrieval backend、
context/support size、route 数、fallback 数、实际 FT steps、trainable params/ratio，
以及 retrieval、routing、FT、predict 分阶段耗时。

## CLI 与矩阵

核心参数为：

- `--model-family tabiclv2|tabpfnv3|all`
- `--method localpfn|mixturepfn`
- retrieval backend：`auto|faiss|sklearn`，`auto` 优先 FAISS，否则使用 sklearn exact
  neighbors
- context/support、cluster、adapter、训练步数/epoch/lr、checkpoint 和 benchmark
  输出参数

矩阵模式严格生成：

1. `tabiclv2/localpfn`
2. `tabiclv2/mixturepfn`
3. `tabpfnv3/localpfn`
4. `tabpfnv3/mixturepfn`

每个 cell 使用独立目录，matrix manifest 记录命令、状态和产物，resume 只复用完整且
`status=ok` 的 cell，合并结果不得产生重复 dataset row。

## 验证

本地验证包括：

- retrieval preprocessing 无泄漏、KNN、KMeans support、bootstrap split、标签覆盖
- Fake TabICLv2/TabPFNv3 block 的 adapter 初始恒等、仅 adapter 可训练、checkpoint
  重载一致
- LoCalPFN query-local context 与 MixturePFN route-shared support
- `all` 精确四 cell、parser/dry-run、CSV 唯一性
- `py_compile` 与 focused pytest
- 验证 `PEFT_Tabicl/1C_Chunk_PEFT.py` 未被本任务修改

远程 smoke 在 `jiqun` 的 `tmux zh` 中执行。先检查已有窗口与 GPU，不中断任务；新建
`zh:compare_way_smoke`，选择显存占用低于 1 GiB 的最低编号 GPU。使用
`Basketball_c`、seed 42 跑四个 cell，要求每个 cell `status=ok`、FT steps 大于零、
LoCalPFN 为 full FT、MixturePFN 仅 adapter 可训练、无 inference fallback、产物完整，
且本地/远程脚本 SHA-256 一致。通过 smoke 后停止，不在本次验收中运行完整 `data184`。
