## 实验思路
实现对 TFM 的测试时计算的研究。
实验模型： Tabiclv2 和 Tabpfnv3 
实验数据集：data184（talent），openmlcc18
实验方法：ft，faware-ft，infer
需要完成效果：ft>base_infer,faware-ft>ft.

## 实验进度
tabiclv2 talent184 ✅
tabpfnv3 talent184 



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
