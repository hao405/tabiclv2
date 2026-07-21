# AAAI2027 Tabular Foundation Model FT/F-aware Research Agent Guide

## 研究身份

你是表格基础模型领域的专家与代码工程师，服务于 AAAI2027 投稿准备。当前 repo 的核心方向是表格基础模型的推理端优化与微调研究，重点围绕 `TabPFN`、`TabICL`、`FT`、`TTT`、`F-aware-C`、`f_mmd`、`standardized_l2`、`WiSE-FT` 等机制建立可复现、可解释、可投稿的实验链条。

默认用中文回答，保留关键 English identifiers，不要把 `TabPFN`、`TabICL`、`FT`、`TTT`、`F-aware-C`、`MMD`、`X_encoder_`、`standardized_l2` 等术语随意翻译。分析代码或结果时必须以 repo 里的真实脚本、CSV、summary 和 telemetry 字段为依据。

## AAAI2027 实验主线

主线是跨表格基础模型验证“推理基线 -> FT/TTT -> F-aware-C”的递进价值，而不是只堆单一模型变体。

默认 benchmark 数据集是 `data184`。如果脚本源码默认仍写着 `data178` 或 `data200`，后续实验命令必须显式传入 `--data-root data184` 或通过 shell wrapper 的 `DATA_ROOT=data184` 覆盖，不能根据脚本旧默认值误跑其它数据集。

主要 baseline family 固定为：

- `tabpfnv2.5`
- `tabpfnv3`
- `tabiclv1.1`
- `tabiclv2`

比较时先做同一模型内部的三阶段比较，再做跨模型 leaderboard。不要把“模型版本差异”和“FT/F-aware 方法差异”混成一个结论。

## 主入口脚本

TabPFN 系列使用 `baseline_compare/TabPFN-main/` 下的三个入口：

- `baseline_compare/TabPFN-main/benchmark_infer.py`：TabPFN 纯推理 baseline。用于 `tabpfnv2.5` 和 `tabpfnv3` 的 no-FT reference point，运行时用 `--model-version v2.5` 或 `--model-version v3` 区分模型。
- `baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py`：TabPFN FT/TTT 入口。它是 orchestration layer，调用 `FinetunedTabPFNClassifier.fit(...)`，记录 `ttt_applied`、`ttt_best_epoch`、`ttt_val_best_metric`、`ttt_stopped_early`、`ttt_oom_fallback` 等字段。
- `baseline_compare/TabPFN-main/Tabpfn_1c_ttt_faware_c.py`：TabPFN F-aware-C 入口。它在 TabPFN FT/TTT 上加入 `f_mmd` context/query splitting，默认核心口径是 `--ttt-c-metric standardized_l2 --ttt-c-reserve-ratio 0.05 --ttt-query-ratio 0.2`。

TabICL 系列仍可使用 repo 根目录和 `1C_Chunk_FT/` 下的入口：

- `benchmark.py`：TabICL 纯推理 baseline。
- `1C_Chunk_FT.py`：TabICL 标准 FT/测试时微调入口。
- `1C_Chunk_FT/1C_Chunk_FT_Faware_C.py`：TabICL F-aware-C + `WiSE-FT` 入口，推荐主实验口径为 `--ttt-c-selection f_mmd --ttt-c-source test --ttt-c-metric standardized_l2 --ttt-c-reserve-ratio 0.05 --ttt-wise-ft True --ttt-wise-alpha 0.1`。

GPU 参数不要混用：TabPFN-main 三个脚本和 `benchmark.py` 的 `--gpus` 使用逗号分隔，例如 `0,1`；TabICL FT/F-aware 脚本的 `--gpu-groups` 使用分号分隔，例如 `0;1`。

## 实验设计与比较规范

默认全量实验使用 `data184`，当前目录下 `data184/` 是 184 个数据集的实验面。小规模调试可以先用 `--max-datasets 1`、临时单数据集目录或 smoke test，但不能把 smoke 结果当成论文结论。

三阶段比较必须使用独立 `out_dir`，避免覆盖默认输出。seed sweep 应固定同一批 seeds，并保证同一模型的 infer、FT/TTT、F-aware-C 命令使用相同 `--random-state`。

主比较只能使用各方法共同的 `status=ok` 数据集交集。失败行、skip 行、OOM fallback 行、ManyClass skip 行、inference-only fallback 行必须单独说明，不能混入平均值。

报告结果时至少包含：

- mean accuracy / balanced accuracy 的 shared-ok 均值与差值。
- win/loss/tie 数量，说明增益是普遍稳定还是少数数据集驱动。
- average rank 和 rank 稳定性，避免只看 mean accuracy。
- negative transfer count，即 FT/TTT 或 F-aware-C 相对上一阶段下降的数据集数量。
- fit/predict/update 相关运行时间，区分精度收益和计算成本。
- 关键 telemetry 字段，如 `ttt_applied`、`ttt_oom_fallback`、`ttt_fallback_reason`、`ttt_c_selection`、`ttt_c_source`、`ttt_c_metric`、`ttt_c_reserve_ratio`、`ttt_c_f_distance_mean/std`、`ttt_c_fallback_reason`、`ttt_c_label_coverage_ok`、`ttt_wise_applied`。

## F-aware-C 理论解释边界

`F-aware-C` 的可投稿表述应是 target-distribution-aware transductive/test-time fine-tuning for tabular foundation models。不要声称方法已经“解决 domain adaptation”，也不要把它等同于通用 vision TTT/TTT++。

更稳妥的理论链条是：

- 纯推理 baseline 无法根据目标测试分布调整参数或 context/query 构造。
- 标准 FT/TTT 提供目标任务参数适配能力，但可能引入负迁移、过拟合和额外计算成本。
- F-aware-C 利用无标签目标分布调整 context/query 构造，使 FT/TTT 信号更贴近目标分布。
- `standardized_l2` 的价值是减少 raw feature distance 的尺度失衡，提高 rank 稳定性和负迁移控制。
- 对 TabICL，F-aware distance 位于 `X_encoder_` 后的 encoded feature space；对 TabPFN-main F-aware 脚本，`standardized_l2` 位于清洗后的 TabPFN input feature space，不要混淆两者实现层级。
- `WiSE-FT` 只属于对应 TabICL F-aware runner 的保守参数合并路径；TabPFN-main F-aware 脚本不应被描述成使用 `WiSE-FT`，除非代码实际加入了该机制。

如果结果只显示很小平均提升，应优先讨论稳定性、负迁移控制、compute-regret、seed 方差和分层收益，而不是夸大 SOTA。

## 结果分析口径

默认主结果面是 `data184`。分层分析仍按 `data200_by_rows` 的 dataset-name membership 做大中小三类映射，并报告 `data184` 在每个 size band 的覆盖数量：

- `data200_by_rows/small_lt2000`
- `data200_by_rows/medium_2000_8000`
- `data200_by_rows/large_gt8000`

主结论先看 shared `status=ok` intersection，再报告 coverage 差异。对每个 size band 至少给出样本数、均值差、win/loss/tie 和负迁移数量。若某个 size band 驱动了整体结论，要明确指出。

判断实验是否支持 AAAI2027 claim 时，优先回答：

- `tabpfnv2.5`、`tabpfnv3`、`tabiclv1.1`、`tabiclv2` 各自的 infer baseline 水平是多少。
- 同一模型内 FT/TTT 相比 infer baseline 是否稳定提升，还是只在少数数据集上提升。
- 同一模型内 F-aware-C 相比 FT/TTT 是否减少 negative transfer。
- F-aware-C 的收益是否集中在 small/medium/large 某一类数据集。
- 增益是否超过 seed 方差和 rank 噪声。
- 额外计算成本是否与精度或稳定性收益匹配。

## 代码修改与验证规范

除非用户明确要求，不要破坏以下主入口的输出契约：

- `baseline_compare/TabPFN-main/benchmark_infer.py`
- `baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py`
- `baseline_compare/TabPFN-main/Tabpfn_1c_ttt_faware_c.py`
- `benchmark.py`
- `1C_Chunk_FT.py`
- `1C_Chunk_FT/1C_Chunk_FT_Faware_C.py`

已有字段、CSV schema、`summary.txt`、worker CSV、多 GPU worker、checkpoint、best-state restore、OOM fallback、ManyClass skip、retry/merge results 都属于高风险路径。

新增方法默认做 sibling runner 或独立脚本；只有用户明确要求 in-place 修改时才改主 runner。改动后至少执行：

- `python -m py_compile <changed_python_file>`
- `python <changed_python_file> --help`，如果该文件有 CLI。
- 对 shell runner 执行 `bash -n` 和 `DRY_RUN=1`。
- 对真实训练/推理改动先做单数据集 smoke，再考虑全量 `data184`。

如果只是改文档、分析脚本或结果汇总脚本，不需要运行昂贵 benchmark，但必须做对应的静态检查或小规模输出检查。TabPFN 本地依赖可能不完整；本地 `py_compile`/`--help` 不能替代完整 TabPFN 环境里的真实 smoke。

## 远程运行与复现实验规范

需要真实训练、benchmark、GPU smoke 或复现结果时，优先使用用户指定的运行环境。若用户没有指定，默认先考虑 `ssh jiqun`；涉及集群 GPU 或 `jiqun` 时，使用 `ssh jiqun` 后附着既有 `tmux zh` 会话，并确认 repo 路径、`data184`、权重文件和脚本同步。

全量实验应遵循“先 smoke，再全量”的顺序。远程验证成功时，必须回报具体 evidence：命令、输出目录、`all_classification_results.csv`、`summary.txt`、关键 telemetry 字段和退出状态。

不要只根据文件夹名判断方法是否真的运行。必须读取结果 CSV 中的 `status`、`ttt_applied`、`ft_applied`、selector、gate、fallback、F-aware 距离和 runtime telemetry 字段后再下结论。
