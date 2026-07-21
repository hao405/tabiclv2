# TabICLv2 / TabPFNv3 15-Seed 方法级排队设计

## 目标矩阵

- 固定 seeds：`3, 10, 16, 42, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035`。
- 方法轴：`infer`、`ft`、`faware_ft`、`lora`、`last_layers`、`ln_head_embedding`、`micp`。`micp` 映射到 `PEFT_Tabicl/MICP.py --method mixturepfn`。
- `TabICLv2/data184` 只运行三个 PEFT 方法和 MICP；已完成的 `infer`、`ft`、`faware_ft` 不重复运行。
- `TabPFNv3/data184`、`TabICLv2/OpenML-CC18`、`TabPFNv3/OpenML-CC18` 均运行完整七方法。
- 总计 25 个方法级 cell、375 个 seed slot。一个 cell 是不可拆分的调度最小单位，同一 worker 在同一 GPU 上串行完成该 cell 的 15 个 seeds。

## 配置与复用

- 启动前生成不可变 `queue_manifest.json`，记录 runner、checkpoint、数据 manifest、参数来源、非 seed 参数及规范化 SHA256 配置指纹。
- 参数来源按以下顺序解析：
  1. 兼容且已完成的调参产物中目标指标最佳者；并列时选择 `finished_at` 最新者。
  2. 正式 seed42 的 `manager_manifest.json` 或 `run_config.json`。
  3. runner 默认配置快照。
- TabPFNv3 F-aware 的调参产物未完成时，对应 cell 保持 `blocked_config`，不得自动回退。
- 复用必须同时匹配 model、dataset、method、seed、数据 manifest 与配置指纹，并存在可审计终态。失败终态保留且默认不重试；仅显式 `retry_failed` 时重跑。
- 结果统一写入 `results/seed_search/<run_id>/<dataset>/<model>/<method>/seed<seed>/`，同时保存 cell 汇总、总体汇总、解析命令、退出码和日志。

## 调度与恢复

- 在 `jiqun` 的 `tmux zh` 中创建命名窗口 `seed_search`。
- 当前 Optuna 完成并写出最佳配置后，GPU0/GPU1/GPU2 各运行一个 worker，占用全部三张物理 GPU；每个 worker 必须先等待对应卡上的已有计算进程退出且显存连续满足阈值，再通过锁占用 GPU。
- worker 原子领取完整 cell；队列优先级为 `data184` 后 `OpenML-CC18`。
- 状态为 `pending -> running -> complete/complete_with_failures`。异常退出后从未终态 seed 继续，不重复已完成或已记录失败的 seed。
- GPU0 以及 GPU1/GPU2 上已有任务均不得被中断或抢占。

## 验证与验收

- 本地验证矩阵恰好包含 25 cells / 375 slots，且不存在三个已排除的 `TabICLv2/data184` cell。
- dry-run 展开全部 cell，并验证基础方法/PEFT 使用 `--random-state`，MICP 使用 `--seed`。
- 远程 preflight 验证 `data184` 为 184 个任务，OpenML-CC18 使用现有 67-task、类别数不超过 10 的数据 view，并验证 checkpoint、runner、配置来源和输出隔离。
- 全量入队前，基础方法、PEFT、MICP 各完成一次 `1 dataset x 1 seed` 真实 smoke。任一 smoke 失败都阻止全量启动。
- 排队成功要求 manager 为 `running` 或 `waiting`、25 个 cells 全部登记、两个 worker 仅在 GPU 可用时领取任务。
- 搜索完成要求 375 个 slots 全部有明确终态，无遗留 `pending/running`，并输出成功、失败、复用及共享成功覆盖审计。
