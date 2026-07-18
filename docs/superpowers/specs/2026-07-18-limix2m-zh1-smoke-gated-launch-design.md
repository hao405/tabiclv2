# LimiX-2M `zh:1` 双 GPU smoke-gated 启动设计

## 目标

在 `jiqun` 的现有 `tmux zh:1` window 中，先用同一个小数据集
`eucalyptus` 并行验证 LimiX-2M 的 `infer` 与 `ft`，通过后再用物理
GPU1、GPU2 启动 seed42 的 data184 八方法完整矩阵。

本次运行复用现有：

- `runners/limix/run_experiment.py`
- `scripts/run_limix2m_data184_matrix.sh`
- `baseline_compare/LimiX/LimiX-2M.ckpt`
- `baseline_compare/LimiX/config/cls_default_noretrieval.json`

不修改八种方法的实验语义或超参数。

## 执行面与 GPU 契约

- 唯一远端执行面为 `ssh jiqun` 下的 `tmux zh:1`。
- 保留 `zh:1` 原 pane；smoke 阶段新增两个 split pane。
- 物理 GPU1 只运行 `infer` smoke。
- 物理 GPU2 只运行 `ft` smoke。
- 正式矩阵固定 `GPUS=1,2`、`MAX_PARALLEL=2`、`SEEDS=42`。
- 启动命令和结果 manifest 中的 GPU 编号必须与上述绑定一致。

## 运行前检查

1. 确认 `zh:1` 当前位于 `gpu2` 的 `~/zh/tabiclv2`，且 GPU1、GPU2
   没有被本任务之外的进程占用。
2. 比较本地与远端以下文件的 SHA-256；不一致时只同步这些目标文件：
   - `runners/limix/`
   - `scripts/run_limix2m_data184_matrix.sh`
3. 在当前 `tabicl` 环境中只安装缺失依赖：
   - `einops==0.8.1`
   - `kditransform==1.1.0`
   - `hyperopt==0.2.7`
4. 不执行完整 `requirements_benchmark.txt`，避免升级现有
   `torch`、`numpy`、`pandas`。
5. 用当前环境完成 import、checkpoint、config、data184 数量和 CUDA
   可见性检查。

任一检查失败时停止，不创建正式结果根。

## Smoke 数据与输出隔离

在远端仓库的 `.tmp/limix2m_smoke_data/` 下建立只包含
`data184/eucalyptus` 的数据视图。数据视图使用符号链接，不复制或修改
原数据。

每次 smoke 使用带时间戳的独立输出根：

```text
results/limix/data184/limix2m_smoke/<run_id>/
├── infer/
└── ft/
```

对应日志写入：

```text
logs/limix/<run_id>/
├── infer_gpu1.log
└── ft_gpu2.log
```

smoke 结果不得写入正式的
`results/limix/data184/limix2m/seed42/`。

## Smoke 放行门

两个 smoke 并行启动，全部满足以下条件才允许启动完整矩阵：

- 两个进程退出码均为 `0`；
- `infer` 与 `ft` 各自恰有一个 `eucalyptus` 结果；
- 两个结果均为 `status=ok`；
- `infer.ttt_applied=False`；
- `ft.ttt_applied=True`；
- `all_classification_results.csv`、逐数据集 JSON、`summary.txt`、
  `run_manifest.json` 均存在；
- manifest 的 seed、method、checkpoint/config/runner hash 与命令一致；
- 日志中没有未记录的 traceback、OOM 或 CUDA 设备错误。

任一条件失败时保留 smoke 日志和结果用于排查，但不归档旧正式结果、
不启动完整矩阵，也不回退为 Infer。

## 历史残留处理

远端标准结果根已有一次中断运行留下的部分 `infer` JSON，且缺少完整
manifest。smoke 通过后，将现有目录非破坏性重命名为：

```text
results/limix/data184/limix2m_interrupted_20260717_<timestamp>/
```

若目标名已存在，则生成新的唯一时间戳；不删除、不覆盖残留结果。
归档完成后重新创建标准结果根。

## 正式矩阵启动

关闭 smoke split panes 后，在 `tmux zh:1` 中运行：

```bash
GPUS=1,2 \
MAX_PARALLEL=2 \
SEEDS=42 \
RESUME=1 \
RETRY_FAILED=1 \
bash scripts/run_limix2m_data184_matrix.sh
```

方法按 shell 既定顺序分批运行：

```text
infer, ft, faware_ft, lora, prefix, last_block, localpfn, micp
```

标准输出为：

```text
results/limix/data184/limix2m/seed42/<method>/
```

## 启动后验证

正式启动后的即时验证仅声明“矩阵已启动”，不声明“矩阵已完成”。验证项：

- `zh:1` 中正式 launcher 仍存活；
- 首批 `infer`、`ft` 分别绑定物理 GPU1、GPU2；
- 两个方法目录均生成匹配的 `run_manifest.json`；
- 日志开始产生逐数据集进度；
- 未出现配置哈希冲突、依赖错误、CUDA 不可用或立即 OOM。

最终完成状态以后续 `matrix_manifest.json`、八个方法 CSV 和
shared-`status=ok` 汇总为准，不能以 pane 空闲或进程退出单独判断。

## 范围边界

- 本设计授权安装上述三个缺失依赖、同步 LimiX runner/shell、创建 smoke
  数据视图、split pane、smoke 结果、非破坏性归档和正式矩阵启动。
- 不修改原始 data184、LimiX checkpoint、方法超参数或第三方 vendor
  实现。
- 不中断 `zh:1` 之外的任务，不使用物理 GPU0，不删除任何历史结果。
