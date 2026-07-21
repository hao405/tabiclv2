## 实验思路
研究 TFM 中由两层 distribution mismatch 引起的推理性能损失：

1. TFM 在 synthetic prior 上预训练，但部署到真实表格任务时存在
   `synthetic-target mismatch`。使用 target-task FT 适配真实数据分布。
2. FT 内部用于优化的 context/holdout 与最终 test query 之间仍可能存在
   `context-query mismatch`。使用 F-aware context/query construction 缓解该失配。

实验模型：`TabICLv2` 和 `TabPFNv3`和`limix-2M`。
实验数据集：`data184 (Talent)`、`OpenML-CC18` 和 Graph-SCM synthetic tasks。
实验方法：`Infer`、`FT`、`F-aware FT`、PEFT（LoRA、Prefix-tuning、Last-block）、
`MICP`。
核心假设：`FT` 相比 `Infer` 缓解 synthetic-target mismatch；`F-aware FT` 相比随机
context/query FT 进一步缓解 context-query mismatch。PEFT 作为第一层 mismatch 的
低成本/参数高效适配对比；MICP 作为第二层 mismatch 的路由策略对比。

## 实验进度
tabiclv2 talent184 ✅  lr1e-5，reverse_ratio=0.05
tabpfnv3 talent184 ❌还在搜参 lr 8.57e-6，reverse_ratio=
tabiclv2 openmlcc18 
tabpfnv3 openmlcc18 
tabiclv2 peft talent184✅
tabpfnv3 peft talent184，性能提升不明显

tabiclv2 synthic 完成
tabpfnv3 synthic 搜参 faware一般✅

tabiclv2  tabpfnv3 micp 在跑


## 实验数据

### TabICLv2 最优种子对比（data184）

按同一 seed 下两种方法共同 `status=ok` 的 184 个数据集计算 mean accuracy，提升为绝对百分点（pp）。

| 对比 | seed | 前方法 | 后方法 | 提升 | 胜/负/平 |
|---|---:|---:|---:|---:|---:|
| FT vs. base_infer | 2027 | 85.5337% | 85.9199% | +0.3862 pp | 59/41/84 |
| FT vs. base_infer | 16 | 85.5313% | 85.9050% | +0.3737 pp | 58/43/83 |
| FT vs. base_infer | 3 | 85.5280% | 85.8417% | +0.3137 pp | 65/36/83 |
| FawareC vs. FT | 2025 | 85.8368% | 85.9290% | +0.0922 pp | 65/54/65 |
| FawareC vs. FT | 42 | 85.8200% | 85.8872% | +0.0671 pp | 58/48/78 |
| FawareC vs. FT | 3 | 85.8417% | 85.9086% | +0.0669 pp | 51/55/78 |

注：`faware_c/seed2029` 尚未生成汇总 CSV，因此 FawareC 排名基于其余 8 个已完成 seed。

### TabICLv2 / TabPFNv3 Graph-SCM 合成数据默认矩阵（seed 42）

结果来自 `tabiclv2_graph_scm_2x3_full_20260716_090925`。矩阵为
`TabICLv2 / TabPFNv3 × infer / ft / faware_ft`，数据为 512 个 Graph-SCM
合成分类任务（Stage1/2/3 分别为 171/171/170 个）。以下统计严格使用每次比较两侧
共同的 `status=ok` 数据集；六个 cell 均为 512/512 `status=ok`。提升为后方法减前方法
的绝对百分点（pp）。TabPFNv3 的 F-aware 结果已替换为 reserve-ratio Optuna 搜参的最佳
Trial 3（`ttt_c_reserve_ratio=0.2908641795`）；完整归档位于 `results/合成数据实验/`。

| 模型 | 对比 | 指标 | 前方法均值 | 后方法均值 | 提升 | 胜/负/平 |
|---|---|---|---:|---:|---:|---:|
| TabICLv2 | FT vs. base_infer | Accuracy | 81.9711% | 81.9975% | +0.0265 pp | 63/41/408 |
| TabICLv2 | FT vs. base_infer | Balanced Accuracy | 76.4844% | 76.5197% | +0.0353 pp | 78/41/393 |
| TabICLv2 | F-aware FT vs. FT | Accuracy | 81.9975% | 82.0128% | +0.0153 pp | 57/55/400 |
| TabICLv2 | F-aware FT vs. FT | Balanced Accuracy | 76.5197% | 76.5159% | −0.0038 pp | 67/64/381 |
| TabPFNv3 | FT vs. base_infer | Accuracy | 81.3901% | 81.4843% | +0.0942 pp | 166/122/224 |
| TabPFNv3 | FT vs. base_infer | Balanced Accuracy | 75.4918% | 75.6705% | +0.1788 pp | 181/134/197 |
| TabPFNv3 | 最佳 F-aware FT vs. FT | Accuracy | 81.4843% | 81.4942% | +0.0099 pp | 108/104/300 |
| TabPFNv3 | 最佳 F-aware FT vs. FT | Balanced Accuracy | 75.6705% | 75.6943% | +0.0238 pp | 116/114/282 |

TTT 与 fallback 审计：

- TabICLv2 FT 使用 `random`，506/512 个任务实际应用 TTT，6 个任务发生 OOM fallback；
  F-aware FT 使用 `f_mmd + standardized_l2`，508/512 个任务实际应用 TTT，4 个任务发生
  OOM fallback。只保留实际应用 TTT 的任务后，FT 仍略优于 infer，而 F-aware FT 仍表现为
  Accuracy 微升、Balanced Accuracy 微降，结论不变。
- TabPFNv3 FT 和 F-aware FT 均为 512/512 个任务实际应用 TTT，且没有 OOM fallback；
  最佳 F-aware FT 使用 `f_mmd + standardized_l2`。4 个 Optuna trials 均为 512/512
  `status=ok`，最佳 Trial 3 的 mean Accuracy 为 `0.8149421368`。

分阶段结果与统计边界：

- TabICLv2 的 `FT − infer` 在三个 Stage 的均值均为正，但 Stage1 基本为零；总体 Accuracy
  的配对正态近似 95% CI 为 `[−0.0046, +0.0575] pp`，因此只能认为是很弱的改善。
- TabICLv2 的 `F-aware FT − FT` Accuracy 在三个 Stage 均为正，但 Balanced Accuracy
  在 Stage2 为 `−0.0523 pp`；两个总体指标的 95% CI 均跨过零，未形成稳定的二级提升。
- TabPFNv3 的 `FT − infer` 在总体 Accuracy 和 Balanced Accuracy 上分别提高
  `+0.0942 pp` 和 `+0.1788 pp`，对应 95% CI 分别为 `[+0.0210, +0.1673] pp`
  和 `[+0.0678, +0.2898] pp`，是当前矩阵中最明确的正向结果。
- 调参后 TabPFNv3 的 `F-aware FT − FT` 在总体 Accuracy 和 Balanced Accuracy 上分别为
  `+0.0099 pp` 和 `+0.0238 pp`，相对默认 ratio=0.05 分别改善 `+0.0532 pp` 和
  `+0.0706 pp`。但两个指标的 dataset-bootstrap 95% CI 分别为
  `[−0.0606, +0.0807] pp` 和 `[−0.0859, +0.1364] pp`，均跨过零。

当前结论：Graph-SCM 合成数据实验支持 `FT > base_infer`，其中 TabPFNv3 的证据更明确；
Optuna 调参使 TabPFNv3 达到均值意义上的 `F-aware FT > FT`，但提升仅约
`0.01–0.02 pp` 且置信区间跨零，尚不能认为二级提升具有统计稳定性。TabICLv2 的
F-aware 结果仍近似持平，因此整体尚未稳定完成预设的两级提升目标。

## 最终实验结果

最终口径使用 `results/合成数据实验/best_matrix/`：TabICLv2 保持原 2×3 矩阵结果，
TabPFNv3 的 F-aware cell 使用 Optuna 最佳 Trial 3。所有比较均基于 512 个
dataset-name shared-`status=ok` 任务。

| 模型 | 比较 | Accuracy Δ | Balanced Accuracy Δ | W/L/T（Accuracy） | 最终判断 |
|---|---|---:|---:|---:|---|
| TabICLv2 | FT − Infer | +0.0265 pp | +0.0353 pp | 63/41/408 | 基本满足，但提升很弱 |
| TabICLv2 | F-aware − FT | +0.0153 pp | −0.0038 pp | 57/55/400 | 未稳定满足 |
| TabPFNv3 | FT − Infer | +0.0942 pp | +0.1788 pp | 166/122/224 | 明确满足 |
| TabPFNv3 | 最佳 F-aware − FT | +0.0099 pp | +0.0238 pp | 108/104/300 | 均值满足，但不稳定 |

最终判断：`FT > base_infer` 在两个模型上均成立，其中 TabPFNv3 的证据最明确。
Optuna 搜参使 TabPFNv3 的 `F-aware > FT` 从负收益转为轻微正收益，但 Accuracy 与
Balanced Accuracy 的 bootstrap 95% CI 均跨过零；同时 TabICLv2 的 Balanced Accuracy
仍略低于 FT。因此当前实验实现了平均值层面的部分两级提升，但尚未证明
`F-aware > FT` 在两个模型上稳定成立。

## 建议的实验章节 Organization（两层 Mismatch 主线）

整篇实验围绕一条递进逻辑组织：

`Synthetic-pretrained TFM`
→ **synthetic-target mismatch**
→ `FT`
→ **FT holdout-query mismatch**
→ `F-aware FT`
→ `final test prediction`。

形式上区分两个 discrepancy：

- `Disc_syn(P_syn, P_T)`：synthetic pretraining prior `P_syn` 与真实 target
  distribution `P_T` 之间的差异；
- `Disc_phi(H, Q; θ, C)`：FT 中承担优化监督的 holdout/query subset `H` 与最终
  test query distribution `Q` 在模型表示空间中的差异。

### 5.1 Experimental Setup

- **Models.** `TabICLv2` 和 `TabPFNv3`。
- **Real-world benchmarks.** `data184 (Talent)` 与 `OpenML-CC18` 用于验证从
  synthetic pretraining 到真实任务的适配。若 OpenML 只使用筛选子集，应报告筛选规则。
- **Controlled synthetic tasks.** Graph-SCM 用于构造或分析可控 shift。只有当
  Stage1/2/3 明确定义为递增 mismatch 时，才能将其作为 mismatch-strength 实验；否则
  只能作为跨生成机制的 robustness benchmark。
- **Methods.** `Infer` 是无适配基线；`FT` 处理第一层 mismatch；PEFT 是第一层的
  参数高效适配对比；`F-aware FT` 处理第二层 mismatch；MICP 是第二层的 routed
  context 对比方法。
- **Metrics.** Accuracy、Balanced Accuracy、Average Rank、W/L/T、paired bootstrap
  95% CI、wall time、峰值显存、失败率与 `ttt_applied` coverage。
- **Protocol.** 所有比较固定 backbone、data split、preprocessing 和 seed。主结果使用
  shared-`status=ok` intersection，OOM/fallback 与 `ttt_applied=False` 单独审计。

### 5.2 Diagnosing the Two Distribution Mismatches

这一节先证明论文要解决的两个问题确实存在，并与性能损失相关。

#### 5.2.1 Synthetic-to-Target Mismatch

- 在可获得对应 synthetic prior sampler 时，估计 `P_syn` 与目标 benchmark 的
  discrepancy。不同表格维度不能直接做 raw-feature MMD，应使用统一维度的 encoder
  representation 或 dataset meta-feature representation。
- 在 controlled synthetic tasks 上逐步改变生成机制、非线性、噪声、类别不平衡或缺失率，
  构造从 prior-like 到 shifted 的难度轴。
- 验证两个关系：mismatch 越大，`Infer` 性能越低；mismatch 越大，`FT − Infer`
  的潜在收益越大。

**推荐 Figure 1：** 横轴为 `Disc_syn` 或 controlled shift strength，纵轴同时展示
`Infer` performance 与 `FT − Infer`。该图证明 FT 的必要性来自 synthetic-target
mismatch，而不只是“多训练几步通常会变好”。

#### 5.2.2 Context/Holdout-to-Query Mismatch during FT

- 区分三层数据角色：外层 target train/test split、FT 内部 context/holdout split，以及
  early-stopping validation split，避免把它们混成同一个 query。
- 对随机 FT，测量内部 holdout `H` 与最终 test query `Q` 的
  `Disc_phi(H,Q;θ,C)`。
- 检查 `Disc_phi` 是否与 `FT` 的 dataset-level negative transfer 或性能波动相关。

**推荐 Figure 2：** 随机 FT 的 `Disc_phi` 与 `FT − Infer` 散点图。若 discrepancy
较大时 FT 更容易退化，就形成提出 F-aware 的直接实验证据。

### 5.3 Addressing Synthetic-to-Target Mismatch with FT

这一节验证第一层核心主张：利用真实 target-task supervision 进行 FT，可以缓解
synthetic pretraining prior 与真实任务之间的 mismatch。

#### 5.3.1 Main Results on Real-world Benchmarks

在 `TabICLv2 / TabPFNv3 × data184 / OpenML-CC18` 上比较：

1. `Infer`；
2. `FT`；
3. PEFT-LoRA、PEFT-Prefix、PEFT-LastBlock。

**Main Table 1：** 以 `Infer` 为 reference，报告 FT 与各 PEFT 方法的 Accuracy、
Balanced Accuracy、Average Rank 和性能增量。主比较是 `FT vs. Infer`；PEFT 用于
说明以更低 trainable-parameter budget 缓解 synthetic-target mismatch 的效果与代价。

#### 5.3.2 Controlled Shift Results

在 Graph-SCM 或重新构造的 controlled-shift tasks 上报告 `Infer` 与 `FT`。重点不是
简单重复总体均值，而是验证 `FT` 的收益是否随 synthetic-target mismatch 增大。

**Main Table 2 / Figure 3：** 按 shift level 报告 `Infer`、`FT`、`FT − Infer`
和 95% CI。若现有 Stage 并非 mismatch level，应避免把 Stage-wise 差异解释成机制证据。

#### 5.3.3 Full FT versus Parameter-Efficient Adaptation

PEFT 在这里作为第一层 mismatch 的低成本适配方案，回答：缓解 synthetic-target
mismatch 需要多大的参数适配能力。

- Full FT vs. LoRA；
- Full FT vs. Prefix-tuning；
- Full FT vs. Last-block FT；
- 同时报告 trainable parameters、时间和显存。

因此 `FT > PEFT` 不是独立论文故事，而是第一层 mismatch 中的 adaptation-capacity
消融。如果 PEFT 更优，也可以形成“较小更新已足以完成 target adaptation”的有效结论。

### 5.4 Addressing Context-Query Mismatch with F-aware FT

这一节验证第二层核心主张：随机 context/holdout construction 不能保证内部优化目标与最终
test query 对齐，F-aware FT 通过 query-aware construction 减少该 discrepancy。

#### 5.4.1 Main Comparison

第二层的核心比较组固定为：

1. random-context FT；
2. F-aware FT；
3. MICP。

其中 random FT 是共同 reference：

- `F-aware FT` 通过 test-distribution-aware context/holdout construction 缓解 mismatch；
- MICP 通过 query routing 与 route-shared support/context 缓解 mismatch。

**Main Table 3：** 报告 F-aware FT、MICP 相对 random FT 的 Accuracy、
Balanced Accuracy、Average Rank、W/L/T、95% CI 和 negative-transfer count，覆盖两个
backbone 和两个真实 benchmark。Graph-SCM 作为补充机制或 robustness 证据。

由于这些方法可能同时改变 context construction、trainable scope 和优化预算，表中必须同时
报告 full FT / adapter FT、trainable parameters、context size、optimization steps、
wall time 与显存。它们是第二层 mismatch 的 competing approaches，而不是只改变单一变量
的纯消融。

#### 5.4.2 Does F-aware Actually Reduce the Mismatch?

除最终 accuracy 外，必须直接报告：

1. random FT 下的 `Disc_phi(H,Q)`；
2. F-aware FT 下的 `Disc_phi(H,Q)`；
3. discrepancy reduction；
4. discrepancy reduction 与 `F-aware FT − FT` 的相关性。

如果 F-aware 使用 MMD 做选择，机制评估最好再加入一个未被直接优化的 two-sample metric，
例如 classifier two-sample AUC、energy distance 或 sliced Wasserstein distance，避免
“用 MMD 选择后再用同一个 MMD 证明 MMD 下降”的循环论证。

**推荐 Figure 4：** 左图比较 random/F-aware 的 mismatch；右图展示 mismatch reduction
与 accuracy gain 的关系。

#### 5.4.3 Where Does F-aware Help?

首先按原始 `Disc_phi` 大小分组：

- low-mismatch tasks：理论上 F-aware 不应带来明显收益；
- medium-mismatch tasks：检查是否稳定改善；
- high-mismatch tasks：验证是否收益最大，以及是否受到 label coverage 或样本不足限制。

再辅以样本数、特征数、cell count、类别数和 imbalance 分层，用于解释失败原因。

### 5.5 Ablation Studies

消融也按两层 mismatch 分组。

#### FT / Synthetic-target Adaptation Ablations

1. full FT vs. LoRA / Prefix / LastBlock；
2. update steps、learning rate 与 trainable scope；
3. performance gain 与 `Disc_syn` 的关系；
4. 相同 compute budget 下的适配效果。

#### F-aware / Context-query Alignment Ablations

1. `random` vs. `random_reserve` vs. `f_mmd`：隔离“reserve 机制”和“F-aware
   选择依据”的贡献；
2. `standardized_l2` vs. encoder-space distance；
3. `reserve_ratio` 与 `query_ratio`；
4. 是否使用 test feature distribution 作为无标签 reference；
5. 固定 downstream context/query splitter，只改变 reserved-set selection，避免混入
   supervision amount 和 split semantics 的变化。

#### Competing Context-Adaptation Strategies

将第二层方法进一步按 context granularity 对齐：

1. random FT：global random context；
2. F-aware FT：global query-distribution-aware context；
3. MICP：route-level shared support/context。

建议同时报告两种口径：各方法论文/实现的 native configuration，以及尽可能对齐
context size、trainable scope 或 optimization budget 的 controlled comparison。

超参数搜索任务与最终 evaluation tasks 必须分开，不能在同一批 test tasks 上选参后再将
同批结果作为无偏主结果。

### 5.6 Robustness, Efficiency, and Failure Analysis

- 多 seed mean ± std、paired bootstrap CI、W/L/T 与 Average Rank；
- shared-`status=ok`、`ttt_applied`、fallback 和 OOM 分开报告；
- 检查结果是否由少数 outlier 数据集主导；
- 报告 FT、PEFT、F-aware 和 MICP 的额外 wall time、显存和失败风险；
- 明确 F-aware 使用最终 test features、但不使用 test labels，属于 transductive
  test-time adaptation setting；
- 分析两种 mismatch 指标在哪些数据属性下失效，以及 FT/F-aware 的负迁移边界。

### 推荐的正文图表顺序

| 顺序 | 图表 | 核心作用 |
|---|---|---|
| Figure 1 | `Disc_syn`、Infer 性能与 FT gain | 证明第一层 mismatch 与 FT 动机 |
| Figure 2 | 随机 FT 的 `Disc_phi` 与负迁移 | 证明第二层 mismatch 与 F-aware 动机 |
| Table 1 | 真实 benchmark：Infer / FT | 验证 FT 缓解 synthetic-target mismatch |
| Figure 3 | Controlled shift 下的 FT gain | 提供第一层机制证据 |
| Table 2 | Full FT 与 PEFT 的效果—成本比较 | 验证所需 adaptation capacity |
| Table 3 | Random FT / F-aware / MICP | 比较第二层 mismatch 的不同处理方式 |
| Figure 4 | F-aware 前后的 `Disc_phi` 及其与 gain 的关系 | 直接验证 mismatch reduction |
| Table 4 | 两层 mismatch 对应的消融实验 | 定位增益来源 |
| Table 5 | 稳健性、效率和 failure audit | 给出方法边界 |

### Claim–Evidence 对齐

| 论文 Claim | 必需证据 | 当前状态 |
|---|---|---|
| Synthetic pretraining 与真实任务存在影响性能的 mismatch | `Disc_syn`/controlled shift 与 Infer performance 的关系 | 待补直接诊断 |
| FT 缓解 synthetic-target mismatch | 真实 benchmark 的 `FT − Infer`，以及 gain 随 mismatch 增大的趋势 | 已有部分效果证据，机制证据待补 |
| PEFT 能以更低适配成本缓解第一层 mismatch | PEFT 相对 Infer/FT 的性能、参数量、时间和显存 | 已有部分结果，待统一汇总 |
| 随机 FT 存在 context/holdout-query mismatch | `Disc_phi(H,Q)` 与 FT negative transfer 的关系 | 待补直接诊断 |
| F-aware FT 缓解 context-query mismatch | F-aware 降低 `Disc_phi`，且 mismatch reduction 对应 accuracy gain | 当前只有弱效果证据，机制证据待补 |
| F-aware 相比其他第二层方法具有优势 | 与 MICP 的 native 与 controlled comparison | 实验在跑 |
| 方法跨 backbone 和 benchmark 泛化 | TabICLv2、TabPFNv3、data184、OpenML-CC18 的一致结果 | 待补齐 |
