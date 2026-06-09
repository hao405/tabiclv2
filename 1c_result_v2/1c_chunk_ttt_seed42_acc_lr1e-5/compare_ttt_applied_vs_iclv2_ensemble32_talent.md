# TabICLv2 TTT accuracy_42 vs ICLv2 ensemble32_talent

分析日期：2026-06-05

## 输入与对比口径

- TTT run：`1c_result_v2/Tabiclv2_ttt_accuracy_42/all_classification_results.csv`
- 对照 run：`1c_result_v2/iclv2_ensemble32_talent/all_classification_results.csv`
- 两边共同数据集：200 个，且两边均为 `status=ok`。
- 主分析对象：TTT run 中 `ttt_applied=True` 的数据集，即真正执行了 TTT update 的 184 个数据集。
- `iclv2_ensemble32_talent` 没有 TTT 元信息，因此这里只把它作为同名数据集的 ensemble32 对照。
- 本地没有 `data200/<dataset>/info.json`，所以样本规模用结果 CSV 中的 `n_train + n_val + n_test` 表示，下面标为 `rows_csv`。

## 核心结论

这轮 `Tabiclv2_ttt_accuracy_42` 相对 `iclv2_ensemble32_talent` 是弱正收益，但不是均匀提升。真正执行 TTT 的 184 个数据集上，平均 accuracy 从 `0.855443` 提升到 `0.858431`，平均变化 `+0.2987 pp`；按 test 样本数加权后，变化为 `+0.5790 pp`，约等于净多对 `+3017` 个 test 样本。

收益主要来自少数大幅提升数据集。184 个 TTT 数据集中，64 个上升、35 个下降、85 个持平；去掉 top 3 个 accuracy gain 后，平均提升从 `+0.2987 pp` 降到 `+0.1265 pp`，去掉 top 10 后只剩 `+0.0227 pp`。

## 主指标汇总

| subset | n | TTT acc | ensemble acc | mean delta | weighted delta | delta_correct | win/loss/tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_success_intersection | 200 | 0.854790 | 0.852041 | +0.2749 pp | +0.5072 pp | +3020 | 65/35/100 |
| `ttt_applied=True` | 184 | 0.858431 | 0.855443 | +0.2987 pp | +0.5790 pp | +3017 | 64/35/85 |
| validation accepted update | 132 | 0.857144 | 0.852980 | +0.4164 pp | +0.6602 pp | +3017 | 64/35/33 |
| `best_epoch=0` / no val gain | 52 | 0.861698 | 0.861698 | +0.0000 pp | +0.0000 pp | 0 | 0/0/52 |
| skipped non-OOM | 12 | 0.765340 | 0.765321 | +0.0019 pp | +0.0048 pp | +3 | 1/0/11 |
| OOM fallback | 4 | 0.955694 | 0.955694 | +0.0000 pp | +0.0000 pp | 0 | 0/0/4 |

`ttt_applied=True` 的其他指标：

| metric | TTT mean | ensemble mean | delta |
|---|---:|---:|---:|
| accuracy | 0.858431 | 0.855443 | +0.002987 |
| f1 | 0.850377 | 0.847737 | +0.002640 |
| balanced_accuracy | 0.785027 | 0.781775 | +0.003252 |
| roc_auc | 0.905194 | 0.903920 | +0.001275 |
| log_loss | 0.317180 | 0.322185 | -0.005005 |

## TTT 覆盖与 fallback

TTT run 中 200 个数据集全部完成，但不是全部真正执行 TTT：

- `ttt_applied=True`：184 个
- `ttt_applied=False`：16 个
- `ttt_oom_fallback=True`：4 个
- 因 `n_classes > 10` 跳过 TTT：12 个

OOM fallback 数据集：

- `Indian_pines`
- `philippine`
- `gas-drift`
- `naticusdroid+android+permissions+dataset`

类别数超过 `max_classes=10` 而跳过 TTT 的数据集：

- `ASP-POTASSCO-classification`
- `UJI_Pen_Characters`
- `one-hundred-plants-margin`
- `one-hundred-plants-texture`
- `walking-activity`
- `helena`
- `internet_usage`
- `kr-vs-k`
- `kropt`
- `letter`
- `one-hundred-plants-shape`
- `texture`

这些非 TTT 数据集不应计入 TTT 收益解释。OOM fallback 的 4 个数据集和 ensemble32_talent 指标完全一致；12 个 max-class skip 基本也一致，只有 `helena` 有 `+0.000230` 的微小差异，约等于 3 个 test 样本，不能解释为 TTT 收益。

## 最大 accuracy 提升

| dataset | task | rows_csv | test | feat | cls | TTT acc | ensemble acc | delta | delta_correct | delta_bal_acc | delta_log_loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eye_movements | multiclass | 12686 | 2188 | 27 | 3 | 0.961152 | 0.835466 | +12.569 pp | +275 | +11.836 pp | -0.295527 |
| Click_prediction_small | binclass | 46340 | 7990 | 3 | 2 | 0.833041 | 0.717897 | +11.514 pp | +920 | -5.577 pp | -0.047438 |
| jungle_chess_2pcs_raw_endgame_complete | multiclass | 51990 | 8964 | 6 | 3 | 0.972557 | 0.892682 | +7.988 pp | +716 | +10.395 pp | -0.087630 |
| compass | binclass | 19307 | 3329 | 17 | 2 | 0.885251 | 0.830580 | +5.467 pp | +182 | +5.467 pp | -0.096991 |
| eye_movements_bin | binclass | 8826 | 1522 | 20 | 2 | 0.693824 | 0.655059 | +3.876 pp | +59 | +3.876 pp | -0.066070 |
| BLE_RSSI_dataset_for_Indoor_localization | multiclass | 11582 | 1997 | 3 | 3 | 0.765148 | 0.733600 | +3.155 pp | +63 | +3.200 pp | -0.042284 |
| artificial-characters | multiclass | 11853 | 2044 | 7 | 10 | 0.966243 | 0.940802 | +2.544 pp | +52 | +2.536 pp | -0.049277 |
| GesturePhaseSegmentationProcessed | multiclass | 11453 | 1975 | 32 | 5 | 0.846076 | 0.827342 | +1.873 pp | +37 | +2.498 pp | -0.047124 |
| electricity | binclass | 52562 | 9063 | 8 | 2 | 0.937107 | 0.926183 | +1.092 pp | +99 | +1.116 pp | -0.022068 |
| connect-4 | multiclass | 78366 | 13512 | 42 | 3 | 0.865601 | 0.856202 | +0.940 pp | +127 | +1.716 pp | -0.020677 |
| car-evaluation | multiclass | 2005 | 346 | 21 | 4 | 0.994220 | 0.985549 | +0.867 pp | +3 | +0.753 pp | -0.009989 |
| hill-valley | binclass | 1406 | 243 | 100 | 2 | 0.979424 | 0.971193 | +0.823 pp | +2 | +0.830 pp | -0.009203 |

注意：`Click_prediction_small` 的 accuracy 大幅提升，但 balanced accuracy 下降 `-5.577 pp`。这说明当前 TTT 的验证指标是 accuracy 时，可能会在类别不平衡任务上偏向多数类；它是最需要单独排查的正收益样本。

## 最大 accuracy 下降

| dataset | task | rows_csv | test | feat | cls | TTT acc | ensemble acc | delta | delta_correct | delta_bal_acc | delta_log_loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| autoUniv-au4-2500 | multiclass | 2900 | 500 | 100 | 3 | 0.712000 | 0.728000 | -1.600 pp | -8 | -1.146 pp | -0.007028 |
| Gender_Gap_in_Spanish_WP | multiclass | 5506 | 950 | 13 | 3 | 0.602105 | 0.614737 | -1.263 pp | -12 | -1.981 pp | +0.001050 |
| credit-g | binclass | 1160 | 200 | 20 | 2 | 0.785000 | 0.795000 | -1.000 pp | -2 | -1.667 pp | -0.000080 |
| wine-quality-red | multiclass | 1855 | 320 | 4 | 6 | 0.671875 | 0.681250 | -0.938 pp | -3 | +4.764 pp | -0.021069 |
| contraceptive_method_choice | multiclass | 1709 | 295 | 9 | 3 | 0.630508 | 0.637288 | -0.678 pp | -2 | -0.762 pp | -0.002320 |
| Fitness_Club_c | binclass | 1740 | 300 | 6 | 2 | 0.796667 | 0.803333 | -0.667 pp | -2 | +0.452 pp | +0.001783 |
| Water_Quality_and_Potability | binclass | 3800 | 656 | 8 | 2 | 0.650915 | 0.657012 | -0.610 pp | -4 | -0.430 pp | +0.000324 |
| PhishingWebsites | binclass | 12824 | 2211 | 30 | 2 | 0.976029 | 0.980552 | -0.452 pp | -10 | -0.354 pp | +0.000939 |
| Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.5GHz(Urbinati) | binclass | 2784 | 480 | 30 | 2 | 0.962500 | 0.966667 | -0.417 pp | -2 | -0.417 pp | +0.000258 |
| pc4 | binclass | 1692 | 292 | 37 | 2 | 0.907534 | 0.910959 | -0.342 pp | -1 | -0.195 pp | +0.013187 |
| waveform_database_generator_version_1 | multiclass | 5800 | 1000 | 21 | 3 | 0.868000 | 0.871000 | -0.300 pp | -3 | -0.300 pp | +0.001251 |
| water_quality | binclass | 9276 | 1600 | 20 | 2 | 0.909375 | 0.911875 | -0.250 pp | -4 | +0.098 pp | -0.000333 |

下降幅度整体小于最大提升幅度。按 `delta_correct` 看，最大负贡献是 `dabetes_130-us_hospitals`，accuracy 只下降 `-0.113 pp`，但 test 较大，净少对约 23 个样本。

## 数据属性分组

### 按任务类型

| task_type | n | TTT acc | ensemble acc | mean delta | win/loss/tie |
|---|---:|---:|---:|---:|---:|
| binclass | 118 | 0.865950 | 0.863749 | +0.2200 pp | 39/22/57 |
| multiclass | 66 | 0.844988 | 0.840593 | +0.4394 pp | 25/13/28 |

多分类数据集的平均收益更大，尤其是 `eye_movements`、`jungle_chess_2pcs_raw_endgame_complete`、`BLE_RSSI_dataset_for_Indoor_localization`、`artificial-characters` 等数据集贡献明显。

### 按样本规模

| rows_csv bin | loss | tie | gain |
|---|---:|---:|---:|
| <=1k | 0 | 3 | 1 |
| 1k-5k | 17 | 49 | 17 |
| 5k-20k | 12 | 27 | 23 |
| 20k-100k | 5 | 6 | 18 |
| >100k | 1 | 0 | 5 |

提升组的样本规模明显更大：提升组 `rows_csv` 中位数为 `12578`，下降组为 `5132`，持平组为 `2940`。这说明本轮 TTT 更像在中大样本任务上有选择性收益，而不是小数据集普遍受益。

### 按特征数

| n_features bin | loss | tie | gain |
|---|---:|---:|---:|
| <=10 | 14 | 21 | 22 |
| 11-30 | 13 | 43 | 32 |
| 31-100 | 6 | 16 | 10 |
| 101-300 | 2 | 5 | 0 |

`<=50` 特征附近整体更容易受益；`101-300` 特征段没有 gain，且 OOM fallback 数据集也多是较宽表格。这提示 TTT 的显存和适配稳定性都可能受特征数影响。

## 验证集选择与泛化风险

TTT run 使用 validation accuracy 选择最佳 epoch。统计上，validation accuracy 改善和 test accuracy 改善的相关系数约为 `0.59`，说明验证集选择确实有信号，但不是可靠到可以忽略风险。

关键现象：

- 52 个 `best_epoch=0` 或 validation 无改善的数据集完全回退到原模型，test 也完全持平。
- 132 个 validation accepted update 数据集贡献了全部净收益，平均 accuracy 提升 `+0.4164 pp`。
- 35 个 test 下降的数据集也都有 validation 提升，说明存在验证集选择偏差或局部过拟合。
- 对 accuracy 作为 TTT 选择指标的依赖较强；在不平衡数据集上，balanced accuracy 可能与 accuracy 方向相反。

因此，这轮结果可以作为 TTT 有效性的正证据，但不能简单解释为“TTT 在所有数据上稳定提升”。更准确的表述是：accuracy-gated TTT 在部分中大样本、低到中等特征数任务上有明显收益，但仍会在一批小样本或验证集不稳定任务上产生小幅负迁移。

## 运行成本

从 summary 看：

- TTT run wall time：`4882.721s`
- ensemble32_talent wall time：`2988.892s`
- wall time 增加约 `1883.829s`，约为 `1.63x`

按 CSV 中单数据集耗时看，`ttt_applied=True` 数据集：

- TTT 平均 fit time：`27.76s`
- ensemble 平均 fit time：`0.81s`
- TTT 平均 total time：`38.99s`
- ensemble 平均 total time：`12.00s`
- TTT update seconds 中位数：`12.30s`，均值：`27.51s`，最大：`394.28s`

收益和成本之间的 trade-off 很明显：当前 TTT 不是廉价推理增强，而是需要用验证门控和失败回退来筛掉无效更新的测试时适配过程。

## 上升、下降、持平数据集列表

<details>
<summary>上升数据集：64 个</summary>

- eye_movements, Click_prediction_small, jungle_chess_2pcs_raw_endgame_complete, compass, eye_movements_bin, BLE_RSSI_dataset_for_Indoor_localization, artificial-characters, GesturePhaseSegmentationProcessed
- electricity, connect-4, car-evaluation, hill-valley, FOREX_cadjpy-day-High, airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True, Basketball_c, turiye_student_evaluation
- Pima_Indians_Diabetes_Database, Firm-Teacher_Clave-Direction_Classification, internet_firewall, FOREX_audchf-day-High, accelerometer, statlog, mfeat-zernike, PizzaCutter3
- California-Housing-Classification, Credit_c, baseball, MagicTelescope, rl, adult, microaggregation2, steel_plates_faults
- ada, satimage, FOREX_audusd-hour-High, Intersectional-Bias-Assessment, Marketing_Campaign, Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.0GHz(Urbinati), okcupid_stem, in_vehicle_coupon_recommendation
- FOREX_audsgd-hour-High, thyroid-ann, allrep, FOREX_cadjpy-hour-High, abalone, customer_satisfaction_in_airline, ada_agnostic, spambase
- Bank_Customer_Churn_Dataset, sylvine, kdd_ipums_la_97-small, wall-robot-navigation, E-CommereShippingData, bank, INNHotelsGroup, online_shoppers
- house_16H, Rain_in_Australia, credit, Cardiovascular-Disease-dataset, jm1, BNG(breast-w), shuttle, SDSS17

</details>

<details>
<summary>下降数据集：35 个</summary>

- autoUniv-au4-2500, Gender_Gap_in_Spanish_WP, credit-g, wine-quality-red, contraceptive_method_choice, Fitness_Club_c, Water_Quality_and_Potability, PhishingWebsites
- Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.5GHz(Urbinati), pc4, waveform_database_generator_version_1, mfeat-fourier, water_quality, first-order-theorem-proving, segment, Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.0GHz(Urbinati)
- wine, madeline, splice, dna, led7, shrutime, pol, FOREX_audcad-hour-High
- ringnorm, rice_cammeo_and_osmancik, dabetes_130-us_hospitals, predict_students_dropout_and_academic_success, Employee, shill-bidding, HR_Analytics_Job_Change_of_Data_Scientists, dry_bean_dataset
- mobile_c36_oversampling, BNG(tic-tac-toe), htru

</details>

<details>
<summary>持平数据集：85 个</summary>

- BNG(cmc), Performance-Prediction, banknote_authentication, autoUniv-au7-1100, ada_prior, Waterstress, VulNoneVul, Telecom_Churn_Dataset
- QSAR_biodegradation, PieChart3, Mobile_Price_Classification, Amazon_employee_access, Long, KDD, Is-this-a-good-customer, Insurance
- GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1, FOREX_audjpy-day-High, FICO-HELOC-cleaned, Customer_Personality_Analysis, cmc, delta_ailerons, eucalyptus, kc1
- website_phishing, waveform-5000, vehicle, taiwanese_bankruptcy_prediction, svmguide3, seismic+bumps, phoneme, pendigits
- pc3, page-blocks, ozone-level-8hr, mice_protein_expression, mfeat-pixel, mfeat-karhunen, mfeat-factors, mammography
- led24, Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-11.0GHz(Urbinati), wine-quality-white, Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.5GHz(Urbinati), Pumpkin_Seeds, dis, default_of_credit_card_clients, company_bankruptcy_prediction
- churn, analcatdata_authorship, allbp, Wilt, Shipping, National_Health_and_Nutrition_Health_Survey, twonorm, MIC
- KDDCup09_upselling, JapaneseVowels, Heart-Disease-Dataset-(Comprehensive), GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001, FOREX_audjpy-hour-High, FOREX_audcad-day-High, Diabetic_Retinopathy_Debrecen, drug_consumption
- eeg-eye-state, estimation_of_obesity_levels, golf_play_dataset_extended, thyroid-dis, thyroid, telco-customer-churn, sports_articles_for_objectivity_analysis, semeion
- pc1, ozone_level, optdigits, national-longitudinal-survey-binary, mozilla4, mfeat-morphological, maternal_health_risk, law-school-admission-bianry
- kr-vs-kp, jasmine, ibm-employee-performance, heloc, yeast

</details>

## 研究判断与下一轮建议

1. 保留 validation-gated TTT 作为有效方向：当前结果有明确正收益，尤其是 weighted accuracy 和 net correct 都为正。
2. 单独复查 `Click_prediction_small` 与 `internet_firewall`：accuracy 上升但 balanced accuracy 下降，可能存在多数类偏置。
3. 对 top loss 做 seed/val split 稳定性复查：`autoUniv-au4-2500`、`Gender_Gap_in_Spanish_WP`、`credit-g`、`wine-quality-red` 是优先对象。
4. 尝试 balanced accuracy 或 log loss 作为 TTT validation metric 的对照 run：当前 accuracy-gated 策略可能牺牲少数类表现。
5. 对高特征数据集做显存优化或更细 chunk：`Indian_pines`、`philippine`、`gas-drift`、`naticusdroid+android+permissions+dataset` 的 OOM fallback 说明宽表 TTT 仍是工程瓶颈。
