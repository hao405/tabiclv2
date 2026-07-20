# TabICLv2 / TabPFNv3 混合 Seed 方法级队列设计

## 目标与矩阵

统一 seed-search queue 保持现有 25 个方法级 cell，不重新加入已完成的
`TabICLv2/data184 infer、ft、faware_ft`。

- `infer、ft、faware_ft` 共 9 个 cell，使用 15 个 seeds：
  `3,10,16,42,2025–2035`。
- `lora、last_layers、ln_head_embedding、micp` 共 16 个 cell，使用
  `3,10,16`。
- 完整队列固定展开为 25 cells、183 seed slots。

每个方法 cell 仍是不可拆分的最小调度单位；GPU0、GPU1、GPU2 各运行一个
worker，在同一 GPU 上串行完成领取 cell 的全部 seeds。

## 接口与 Manifest

Queue CLI 使用 `--long-seeds` 和 `--short-seeds`；远程 wrapper 对应暴露
`LONG_SEEDS` 与 `SHORT_SEEDS`。`build_specs()` 根据方法族选择实际 seed
列表，并将其写入每个 `CellSpec`。

不可变 `queue_manifest.json` 同时记录两套 seed policy、长短方法集合、
长短 cell 数和实际 slot 总数。配置指纹继续描述 model、dataset、method、
runner、数据 manifest 和非 seed 参数；复用在单个 seed 层面额外严格校验
dataset、model、method、seed 和配置指纹。

Preflight 和 dry-run 必须验证：

- 25 cells、9 个 long-policy cells、16 个 short-policy cells、183 slots；
- 两个 seed 列表均无重复，short seeds 是 long seeds 的子集；
- TFM 和 PEFT 命令使用 `--random-state`，MICP 使用 `--seed`；
- `data184` 为 184 个任务，OpenML-CC18 view 为现有 67 个任务。

## 切换、恢复与失败

旧 run `tabular_15seed_20260720_203346` 通过精确进程树审计后停止，旧结果和
manifest 保留，并写入 `superseded.json` 记录停止原因、时间、旧进程信息和
替代 run ID。新 run ID 使用 `tabular_mixedseed_<timestamp>`。

新 full/smoke queue 可通过多个 `reuse-root` 引用旧 full/smoke 根目录。只有
存在 `queue_seed_terminal.json` 且身份与指纹完全一致时才复用；未完成的
seed 在新目录重新运行。已有失败终态默认保留，只有显式启用
`retry_failed` 才重跑。

异构 seed 下的共享成功覆盖按 dataset、model、seed 计算：

- seed `3、10、16` 比较该 dataset/model 下所有已计划方法；
- 其余 seeds 只比较 `infer、ft、faware_ft`；
- 汇总显式记录 `seed_policy`、`expected_method_count`、实际方法数和共享
  `status=ok` 数据集。

## 验证与完成标准

本地执行语法检查、聚焦单元测试和 dry-run。远程依次执行代码同步、同环境
测试、183-slot dry-run、三类 smoke gate 和三 GPU 正式启动检查。

启动成功要求三个 worker 分别领取一个完整方法 cell。最终完成要求 183 个
slots 全部具有 `success` 或 `failed` 终态，无遗留 `pending/running`，并
分别汇总成功、失败、复用数量及共享成功覆盖；失败不得计为成功。
