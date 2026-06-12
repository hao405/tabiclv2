# Target-aware query construction for TabICL TTT

## 目标

本轮 goal 模式的目标是实现并迭代一种 Target-aware query construction for TTT 方法。现有 baseline `auto-resarch/1C_Chunk_TTT.py` 在测试时训练（TTT）阶段，从训练集 A 内随机或分层随机划分一部分样本 C 作为 query，并使用 C 的真实标签构造监督信号。新方法需要改变 C 的构造方式：不再随机从 A 中划分 C，而是利用测试 query set F 的特征分布，从 A 中选择一个与 F 更接近、同时标签可靠的子集 C，使 TTT 的更新方向更贴近当前测试分布。

F 只允许提供特征分布信息，不允许使用测试标签。C 必须来自有真实标签的训练池 A，TTT loss 必须由 C 的真实标签监督。该方法应定位为“有标签训练池 + 无标签目标分布引导的监督式测试时适配”，不要写成伪标签方法，也不要把测试标签引入训练或早停。

## 固定 baseline 与评测范围

- baseline 固定为 `auto-resarch/1C_Chunk_TTT.py`。
- 第一阶段评测范围固定为 `data200_by_rows/small_lt2000`。
- 主指标固定为 `accuracy`。
- 成功标准：在 baseline 与新方法 `status=ok` 的成功数据集交集上，新方法的 mean accuracy 至少比 baseline 提升 `+0.0004`。
- 对比时只使用成功交集计算平均值；失败、skip、OOM fallback 或未真正执行 TTT 的行必须单独说明，不能混入主结论。

## 必须保持的工程契约

- 保留 benchmark 输出契约：`worker_*.csv`、`all_classification_results.csv`、`summary.txt` 必须继续生成。
- 保留 `--out-dir` 行为；用户显式指定输出目录时不能被自动命名覆盖。
- 新方法关闭时，行为应与 baseline 兼容，不得改变原始 `1C_Chunk_TTT.py` 的默认实验语义。
- 不得使用 F 的标签、测试集评测结果或 benchmark 目标值参与训练、早停、样本选择打分。
- 如果 C 选择失败、标签覆盖不足、context/query 无法构造有效训练批次或发生 OOM，必须明确 fallback，并在结果字段中记录原因。

## 需要记录的可审计字段

结果 CSV 至少需要保留并检查以下字段：

- `ttt_applied`
- `ttt_steps`
- `ttt_best_epoch`
- `ttt_split_strategy`
- `ttt_split_reason`
- `ttt_val_baseline_accuracy`
- `ttt_val_best_accuracy`
- `ttt_oom_fallback`

F-aware C 选择方法还需要记录以下字段或等价信息：

- `ttt_c_selection`
- `ttt_c_source`
- `ttt_c_metric`
- `ttt_c_f_distance_mean`
- `ttt_c_f_distance_std`
- `ttt_c_fallback_reason`
- `ttt_c_label_coverage_ok`

判定一次实验有效时，不能只看 `status=ok`。必须确认 `ttt_applied=True` 且 `ttt_steps > 0`，并检查 `ttt_split_reason` / `ttt_c_fallback_reason` 是否显示该行实际使用了目标感知 C 选择。

## 核心方法方向

优先围绕以下 C 选择策略迭代：

1. `f_nearest`
   - 在 TabICL encoded feature space 或标准化后的 encoded feature space 中，计算 A 中候选样本到 F 分布的距离。
   - 选择距离 F 最近的一批样本作为 C。
   - 必须保证 C 的标签在 context 中可被支持，不能让 query 标签缺失于 context。

2. `f_mmd`
   - 从 A 中选择一个 C，使 C 的均值、方差或近似分布统计更接近 F。
   - 可以用贪心选择近似最小化 C 与 F 的分布差异。
   - 需要兼顾类别覆盖，避免只选择某个局部密度区域导致监督信号偏置。

3. `f_density_mixed`
   - 将靠近 F 的样本与一部分随机覆盖样本混合。
   - 目标是在贴近 F 的同时降低小样本过拟合和类别塌缩风险。
   - 近邻比例、随机比例、候选池大小可以作为后续 goal 模式的调参方向。

所有策略都必须保证：

- C 的标签来自训练集 A 的真实标签。
- F 只作为无标签 target feature reference。
- context/query 可以构造有效的 TabICL supervised TTT batch。
- 如果类别覆盖或 batch 构造不满足要求，必须 fallback 到安全策略并记录原因。

## Goal 模式每轮迭代记录要求

每轮尝试必须在回复或实验日志中记录：

- 假设：本轮为什么可能提升 accuracy。
- 改动：改了哪些选择逻辑、参数或 fallback 规则。
- 命令：完整可复现的运行命令。
- 结果：baseline mean accuracy、新方法 mean accuracy、差值、成功交集数量、win/loss/tie。
- 审计：`ttt_applied`、`ttt_steps`、`ttt_best_epoch`、`ttt_c_*` 字段是否证明目标感知 TTT 实际执行。
- 失败原因：若未达到 `+0.0004`，说明是没有真正执行、fallback 过多、选择分布不准、过拟合、早停失效、还是某些数据集拖累。
- 下一轮方向：明确下一步要调整的选择策略、参数或保护机制。

## 验收标准

- `program.md` 必须保持中文说明。
- 除 `auto-resarch/program.md` 外，本次不应修改任何其他文件。
- 后续实现阶段再运行 `py_compile`、smoke test 和 `small_lt2000` benchmark；本次只写需求文档。
- 最终目标是在 `data200_by_rows/small_lt2000` 上，相对 `auto-resarch/1C_Chunk_TTT.py` baseline 的成功交集 mean accuracy 提升至少 `+0.0004`。
