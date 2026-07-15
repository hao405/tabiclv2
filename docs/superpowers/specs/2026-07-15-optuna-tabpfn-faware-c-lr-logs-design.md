# TabPFN F-aware-C Optuna Trial Logs 恢复设计

## 目标

在 `scripts/optuna_tabpfn_faware_c_lr.py` 中恢复每个 Optuna trial 的独立日志输出，不改变搜索空间、结果目录层级、缓存语义或当前参数默认值。

## 输出契约

- 每个 trial 的结果继续写入 `<output_root>/trials/<trial_slug>/`。
- 每个实际执行的 trial 将 stdout 和 stderr 合并写入 `<output_root>/logs/<trial_slug>.log`。
- trial 元数据及 `best_summary.json` 中记录 `log_path`，用于从最优 trial 直接定位日志。
- dry-run 在计划记录中写入 `log_path`，但不创建空日志文件。
- 命中已有有效结果且未指定 `--force` 时沿用缓存，不执行 runner，也不覆盖已有日志。

## 实现边界

1. 扩展 `TrialPaths`，同时携带 `out_dir` 和 `log_path`。
2. `trial_paths(...)` 使用同一个可读 trial slug 生成结果目录和日志文件名。
3. `ensure_trial_outputs(...)` 接收 `log_path`，执行 runner 前创建日志父目录，并将 runner 的 stdout/stderr 合并写入日志。
4. objective、dry-run 行记录和最终 summary 统一传递并持久化 `log_path`。
5. 保留工作区中现有的 `--n-trials` 默认值 `6`，不做无关重构。

## 错误与缓存行为

- runner 返回非零时仍抛出 `RuntimeError`；对应日志保留，便于诊断。
- runner 成功但缺少 CSV 或 summary 时仍抛出 `FileNotFoundError`；日志同样保留。
- 日志文件采用覆盖写入，使 `--force` 重跑不会把不同执行混在同一日志中。

## 验证

- 单元测试覆盖 trial 日志路径、stdout/stderr 合并、缓存命中不执行且不覆盖日志、dry-run 只记录路径，以及真实模式 summary 中的 `log_path`。
- 运行 `python -m py_compile scripts/optuna_tabpfn_faware_c_lr.py tests/test_optuna_tabpfn_faware_c_lr.py`。
- 运行 `python -m pytest tests/test_optuna_tabpfn_faware_c_lr.py -q`。
- 执行一个临时目录 dry-run，核对 `best_summary.json` 中的路径且没有生成空 `.log`。
