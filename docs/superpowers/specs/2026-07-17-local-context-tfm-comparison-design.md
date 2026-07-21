# 官方 LoCalPFN 双模型 Runner 设计

## 目标与边界

`PEFT_Tabicl/compare_way.py` 是独立的 LoCalPFN 分类 benchmark runner，只支持：

- 模型：`tabiclv2`、`tabpfnv3`
- 方法：`localpfn`
- `--model-family all`：展开为两个模型的两个实验 cell

LoCalPFN 的 retrieval、episode 和训练默认值固定对齐
`layer6ai-labs/LoCalPFN` commit
`ff8803c57cd277380b2444f0f5ed4856b46f47f5`。TabICLv2 和 TabPFNv3 仅替换模型后端。
runner 不包含 MixturePFN/MICP、KMeans route、support context 或 residual adapter。

## 官方数据与 episode 语义

- 类别词表与 one-hot 维度由 train/valid/test 的全体无标签 `X` 确定。
- `StandardScaler` 只在 train 上拟合，全部 split 转换后裁剪到 `[-10, 10]`。
- 默认使用 standardized raw/ordinal retrieval space；`onehot_retrieval` 和
  `use_one_hot_emb` 遵循官方类别维度与 100-feature budget。
- 正式实验使用 FAISS `IndexFlatL2`。显式 `sklearn` 只用于非 reference 诊断，不允许
  `auto` 静默 fallback。
- 默认 `K=min(10*sqrt(n_train),1000)`、`Q=1000`。每个训练 step 选择两个 anchor，
  分别检索 `K+Q` 个最近 train rows，两个邻域使用相同的位置 permutation；前 `K`
  行为 context，剩余行为监督 query。
- anchor 不从邻域中移除。小数据集使用全部 train rows，并将最后至少一行保留为 query。
- validation/test 的每个 query 使用独立纯 KNN context。不存在 class-aware repair、
  stratified replacement 或普通 inference fallback。

## 双模型训练与预测

- 两个模型均进行 full FT，默认 AdamW、`lr=1e-5`、`weight_decay=0.01`、21 epochs、
  每 epoch 最多 30 steps、train batch size 2。
- TabICLv2 将两个 anchor episode 合并为一次 optimizer step，并对所有 query logits
  计算交叉熵。
- TabPFNv3 的 dataset chunk 使用显式 prefix split，严格保证前 `K` 行为 context、
  后 `Q` 行为 query，不执行内部随机比例重划分；meta batch size 为 2。
- 最终 prediction 对不同 query-local contexts 分组批量执行，并将局部概率列映射到
  train 的全局类别空间。单类 context 不受后端支持时 dataset 明确失败。
- validation 默认使用 AUC 选择 best checkpoint；同时支持官方 `negloss`、`acc`、
  `f1`、scheduler、better selection、exact KNN、save data 与 evaluated-splits 开关。

## CLI、结果与恢复

- `--method` 只接受 `localpfn`；`mixturepfn` 和 `all` 均为非法值。
- 官方主要参数使用 `--local-*` 前缀；`--context-size` 保留为 context length
  兼容入口。
- CSV、summary 与 manifest 记录官方 source commit、K/Q、train/inference batch、
  retrieval space、best epoch、FT steps、trainable ratio 和分阶段耗时。
- 每个 cell 使用包含脚本 SHA-256 的配置 fingerprint。resume 只复用 fingerprint
  完全一致且 `status=ok` 的行；旧 LoCalPFN-inspired 或 MixturePFN 结果不能混用。
- OOM 仅允许 inference batch 减半重试一次，不缩小 context、不修改邻域、不 fallback。

## 验证与运行

- 单元测试覆盖 all-X vocabulary/train-only scaler、one-hot budget、纯 KNN、anchor
  inclusion、共享 permutation、K/Q split、小数据、概率列对齐、单类失败、双 cell
  matrix 和 fingerprint resume。
- 本地依次执行 `py_compile`、focused pytest、双 cell dry-run。
- 旧语义远程结果非破坏性归档为 legacy。新版本先在 `jiqun` 的 `tmux zh:0`、物理
  GPU0 上运行 Basketball_c 双模型 smoke，再在新结果根目录运行
  `data184 × {tabiclv2,tabpfnv3} × localpfn × seed42`。
