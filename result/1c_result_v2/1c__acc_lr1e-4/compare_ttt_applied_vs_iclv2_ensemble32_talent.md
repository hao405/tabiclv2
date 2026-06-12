# tabicl-v2-20260212 TTT seed42 2 vs ICLv2 ensemble32_talent

分析日期：2026-06-05

## 输入与对比口径

- TTT run：`1c_result_v2/tabicl-v2-20260212__data200_200datasets__inferest32_bs8_kv0_ampauto_fa3auto_offloadauto__ttt_eval-accuracy_finetuneest2_valest2_ep30_lr0.0001_q0.2__seed42 2/all_classification_results.csv`
- 对照 run：`1c_result_v2/iclv2_ensemble32_talent/all_classification_results.csv` 
- 主比较口径：两边 `status=ok` 的共同数据集里，TTT run 的 `ttt_applied=True` 数据集。
- 本地没有 `data200/<dataset>/info.json`，所以样本规模用结果 CSV 中的 `n_train + n_val + n_test` 表示，下面标为 `rows_csv`。

## 运行概况

这个 `seed42 2` run 比无空格 `seed42` 那份健康很多：200 个数据集全部 `ok`，没有 fail；只有 4 个 TTT OOM fallback。严格按“交集且用了 TTT”看，主比较有 184 个数据集。

| method | rows | ok | fail | ttt_applied | ttt_oom_fallback | avg_accuracy_ok | wall_seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| tabicl-v2-20260212 TTT seed42 2 | 200 | 200 | 0 | 184 | 4 | 0.853168 | 8350.665 |
| iclv2_ensemble32_talent | 200 | 200 | 0 | n/a | n/a | 0.852041 | 2988.892 |

## 核心结论

在 184 个真正 `ttt_applied=True` 的共同成功数据集上，这个 TTT run 相对 `iclv2_ensemble32_talent` 是弱正收益：TTT 平均 accuracy 为 `0.856667`，ensemble 平均为 `0.855443`，平均变化 `+0.1224 pp`。按 test 样本数加权后，weighted delta 为 `+0.5729 pp`，约等于净多对 `+2985` 个 test 样本。

但这不是普遍提升。184 个 TTT 数据集中，accuracy 胜/负/平为 `48/48/88`；提升和下降数量完全相同，正收益主要来自少数大幅提升或大 test 数据集。去掉 top 3 accuracy gain 后，平均 delta 从 `+0.1224 pp` 变成 `-0.0659 pp`；去掉 top 10 后变成 `-0.1634 pp`。

因此更准确的判断是：这个 run 在 weighted accuracy 上有明显正收益，但逐数据集稳定性一般，属于“少数大收益拉正整体”的 TTT 结果。

## 主指标汇总

| subset | n | TTT acc | ensemble acc | mean delta | weighted delta | delta_correct | win/loss/tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| ok intersection | 200 | 0.853168 | 0.852041 | +0.1127 pp | +0.5018 pp | +2988 | 49/48/103 |
| ok & `ttt_applied=True` | 184 | 0.856667 | 0.855443 | +0.1224 pp | +0.5729 pp | +2985 | 48/48/88 |
| TTT validation accepted update | 113 | 0.872155 | 0.870162 | +0.1993 pp | +0.7025 pp | +2985 | 48/48/17 |
| TTT no val gain / best_epoch=0 | 71 | 0.832018 | 0.832018 | +0.0000 pp | +0.0000 pp | 0 | 0/0/71 |
| ok & skipped non-fallback | 12 | 0.765340 | 0.765321 | +0.0019 pp | +0.0048 pp | +3 | 1/0/11 |
| ok & OOM fallback | 4 | 0.955694 | 0.955694 | +0.0000 pp | +0.0000 pp | 0 | 0/0/4 |

`ok & ttt_applied=True` 的其他指标：

| metric | TTT mean | ensemble mean | delta |
|---|---:|---:|---:|
| accuracy | 0.856667 | 0.855443 | +0.001224 |
| f1 | 0.848126 | 0.847737 | +0.000389 |
| balanced_accuracy | 0.782562 | 0.781775 | +0.000787 |
| roc_auc | 0.903899 | 0.903920 | -0.000021 |
| log_loss | 0.319197 | 0.322185 | -0.002988 |

accuracy、F1、balanced accuracy、log loss 都是正向；ROC-AUC 基本持平略负。整体信号比无空格 `seed42` 那份稳定，但仍然不是逐数据集全面胜出。

## TTT 覆盖与 fallback

TTT run 中：

- `status=ok`：200 个
- `status=fail`：0 个
- `ttt_applied=True`：184 个
- `ttt_oom_fallback=True`：4 个
- `ttt_applied=False` 且非 OOM fallback：12 个，主要是类别数超过 `max_classes=10` 的 skip。

TTT OOM fallback 数据集：

- `Indian_pines`
- `gas-drift`
- `naticusdroid+android+permissions+dataset`
- `philippine`

类别数超过 `max_classes=10` 而跳过 TTT 的 12 个数据集：

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

这些非 TTT 数据集不应计入 TTT 收益解释。主结论只基于 184 个 `ttt_applied=True` 数据集。

## 最大 accuracy 提升

| dataset | task | rows_csv | test | feat | cls | TTT acc | ensemble acc | delta | delta_correct | delta_bal_acc | delta_log_loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eye_movements | multiclass | 12686 | 2188 | 27 | 3 | 0.975320 | 0.835466 | +13.985 pp | +306 | +13.270 pp | -0.341610 |
| Click_prediction_small | binclass | 46340 | 7990 | 3 | 2 | 0.832416 | 0.717897 | +11.452 pp | +915 | -5.822 pp | -0.047968 |
| jungle_chess_2pcs_raw_endgame_complete | multiclass | 51990 | 8964 | 6 | 3 | 0.982820 | 0.892682 | +9.014 pp | +808 | +11.663 pp | -0.119977 |
| eye_movements_bin | binclass | 8826 | 1522 | 20 | 2 | 0.703679 | 0.655059 | +4.862 pp | +74 | +4.862 pp | -0.091433 |
| electricity | binclass | 52562 | 9063 | 8 | 2 | 0.963919 | 0.926183 | +3.774 pp | +342 | +3.943 pp | -0.083006 |
| BLE_RSSI_dataset_for_Indoor_localization | multiclass | 11582 | 1997 | 3 | 3 | 0.754131 | 0.733600 | +2.053 pp | +41 | +2.098 pp | -0.022604 |
| GesturePhaseSegmentationProcessed | multiclass | 11453 | 1975 | 32 | 5 | 0.845063 | 0.827342 | +1.772 pp | +35 | +2.491 pp | -0.043971 |
| hill-valley | binclass | 1406 | 243 | 100 | 2 | 0.987654 | 0.971193 | +1.646 pp | +4 | +1.653 pp | -0.037919 |
| Pima_Indians_Diabetes_Database | binclass | 891 | 154 | 8 | 2 | 0.740260 | 0.727273 | +1.299 pp | +2 | +1.852 pp | -0.001204 |
| FOREX_cadjpy-day-High | binclass | 2128 | 367 | 10 | 2 | 0.738420 | 0.727520 | +1.090 pp | +4 | +1.087 pp | -0.003049 |
| connect-4 | multiclass | 78366 | 13512 | 42 | 3 | 0.866785 | 0.856202 | +1.058 pp | +143 | +2.625 pp | -0.022024 |
| mfeat-zernike | multiclass | 2320 | 400 | 47 | 10 | 0.905000 | 0.897500 | +0.750 pp | +3 | +0.750 pp | -0.006815 |

`Click_prediction_small` 仍然是一个需要单独解释的样本：accuracy 大幅提升，但 balanced accuracy 下降 `-5.822 pp`，说明 accuracy gate 可能强化了多数类方向。

## 最大 accuracy 下降

| dataset | task | rows_csv | test | feat | cls | TTT acc | ensemble acc | delta | delta_correct | delta_bal_acc | delta_log_loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cmc | multiclass | 1709 | 295 | 9 | 3 | 0.450847 | 0.603390 | -15.254 pp | -45 | -20.841 pp | +0.133868 |
| Performance-Prediction | binclass | 1555 | 268 | 19 | 2 | 0.712687 | 0.746269 | -3.358 pp | -9 | -1.617 pp | +0.058469 |
| website_phishing | multiclass | 1570 | 271 | 9 | 3 | 0.900369 | 0.926199 | -2.583 pp | -7 | -2.186 pp | +0.018059 |
| Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.5GHz(Urbinati) | binclass | 2784 | 480 | 30 | 2 | 0.952083 | 0.966667 | -1.458 pp | -7 | -1.397 pp | +0.013083 |
| Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.5GHz(Urbinati) | binclass | 2784 | 480 | 30 | 2 | 0.970833 | 0.983333 | -1.250 pp | -6 | -1.224 pp | +0.032963 |
| HR_Analytics_Job_Change_of_Data_Scientists | binclass | 22224 | 3832 | 13 | 2 | 0.789927 | 0.800626 | -1.070 pp | -41 | -5.539 pp | +0.009040 |
| wine-quality-red | multiclass | 1855 | 320 | 4 | 6 | 0.671875 | 0.681250 | -0.938 pp | -3 | +4.748 pp | -0.009543 |
| Diabetic_Retinopathy_Debrecen | binclass | 1335 | 231 | 19 | 2 | 0.731602 | 0.740260 | -0.866 pp | -2 | -0.870 pp | +0.003345 |
| FOREX_audjpy-day-High | binclass | 2125 | 367 | 10 | 2 | 0.779292 | 0.787466 | -0.817 pp | -3 | -0.843 pp | +0.003652 |
| ada_agnostic | binclass | 5292 | 913 | 48 | 2 | 0.843373 | 0.851041 | -0.767 pp | -7 | +2.732 pp | +0.010856 |
| waveform_database_generator_version_1 | multiclass | 5800 | 1000 | 21 | 3 | 0.864000 | 0.871000 | -0.700 pp | -7 | -0.692 pp | +0.004808 |
| predict_students_dropout_and_academic_success | multiclass | 5132 | 885 | 34 | 3 | 0.776271 | 0.783051 | -0.678 pp | -6 | -0.729 pp | -0.001401 |

`cmc` 是最需要优先排查的负迁移样本：accuracy 下降 `-15.254 pp`，balanced accuracy 下降 `-20.841 pp`，log loss 也明显变差。它不是小数值噪声，而是显著失败案例。

## 加权贡献

最大正贡献来自较大 test 集：

- `Click_prediction_small`：约 `+915` 个 test 样本
- `jungle_chess_2pcs_raw_endgame_complete`：约 `+808`
- `electricity`：约 `+342`
- `eye_movements`：约 `+306`
- `connect-4`：约 `+143`
- `Credit_c`：约 `+114`
- `accelerometer`：约 `+113`

最大负贡献：

- `cmc`：约 `-45`
- `HR_Analytics_Job_Change_of_Data_Scientists`：约 `-41`
- `FOREX_audcad-hour-High`：约 `-14`
- `default_of_credit_card_clients`：约 `-11`
- `mobile_c36_oversampling`：约 `-11`
- `Performance-Prediction`：约 `-9`

这解释了为什么胜负数量是 `48/48`，但 weighted delta 仍然明显为正：大 test 集上的正贡献更大。

## 数据属性分组

### 按任务类型

| task_type | n | TTT acc | ensemble acc | mean delta | win/loss/tie |
|---|---:|---:|---:|---:|---:|
| binclass | 118 | 0.864915 | 0.863749 | +0.1165 pp | 31/32/55 |
| multiclass | 66 | 0.841921 | 0.840593 | +0.1328 pp | 17/16/33 |

二分类和多分类的平均收益都很小且为正；二分类胜负略偏负，多分类胜负略偏正。

### 按样本规模

| rows_csv bin | loss | tie | gain |
|---|---:|---:|---:|
| <=1k | 1 | 2 | 1 |
| 1k-5k | 24 | 49 | 10 |
| 5k-20k | 16 | 31 | 15 |
| 20k-100k | 7 | 5 | 17 |
| >100k | 0 | 1 | 5 |

提升组的规模明显更大：提升组 `rows_csv` 中位数 `12656.5`，下降组 `4143.5`，持平组 `3702.0`。这说明本轮 TTT 的 weighted 收益主要来自中大样本数据集。

### 按特征数

| n_features bin | loss | tie | gain |
|---|---:|---:|---:|
| <=10 | 16 | 23 | 18 |
| 11-30 | 25 | 41 | 22 |
| 31-100 | 6 | 18 | 8 |
| 101-300 | 1 | 6 | 0 |

高特征数段没有明显收益。`101-300` 特征段没有 gain，且 OOM fallback 仍集中在部分宽表任务上。

## 验证集选择与泛化风险

`ttt_applied=True` 中，validation accepted update 有 113 个，test accuracy 胜/负/平为 `48/48/17`。也就是说，验证集接受 TTT 更新后，test 上提升和下降的数据集数量相等。

不过 validation accepted 子集的 weighted delta 是 `+0.7025 pp`，净多对约 `+2985` 个样本；这说明验证集 gate 对大收益样本有筛选作用，但不能阻止所有负迁移。`cmc`、`Performance-Prediction`、`website_phishing` 这类明显下降样本，说明 accuracy gate 仍然会接受对 test 泛化不利的更新。

`no val gain / best_epoch=0` 的 71 个数据集全部持平；这部分基本是回退原模型，不贡献收益或损失。

## 运行成本

从 summary 看：

- TTT run wall time：`8350.665s`
- ensemble32_talent wall time：`2988.892s`
- wall time 增加约 `5361.773s`，约为 `2.79x`

在 `ok & ttt_applied=True` 主子集上：

- TTT 平均 total time：`40.78s`
- ensemble 平均 total time：`12.00s`
- 平均额外耗时：`+28.78s`

相比无空格 `seed42` 那份，这个 run 的覆盖率显著更好，但代价是 wall time 更高。它更适合作为算法效果分析样本，但如果用于生产式 benchmark，需要继续优化 TTT update 成本。

## 上升、下降、持平数据集列表

<details>
<summary>上升数据集：48 个</summary>

- eye_movements, Click_prediction_small, jungle_chess_2pcs_raw_endgame_complete, eye_movements_bin, electricity, BLE_RSSI_dataset_for_Indoor_localization, GesturePhaseSegmentationProcessed, hill-valley
- Pima_Indians_Diabetes_Database, FOREX_cadjpy-day-High, connect-4, mfeat-zernike, Basketball_c, Pumpkin_Seeds, car-evaluation, Credit_c
- adult, BNG(tic-tac-toe), baseball, Firm-Teacher_Clave-Direction_Classification, accelerometer, turiye_student_evaluation, pc4, FOREX_audsgd-hour-High
- FOREX_cadjpy-hour-High, Bank_Customer_Churn_Dataset, satimage, National_Health_and_Nutrition_Health_Survey, sylvine, INNHotelsGroup, kdd_ipums_la_97-small, okcupid_stem
- dabetes_130-us_hospitals, taiwanese_bankruptcy_prediction, E-CommereShippingData, thyroid-ann, California-Housing-Classification, FOREX_audusd-hour-High, Long, Cardiovascular-Disease-dataset
- wall-robot-navigation, customer_satisfaction_in_airline, bank, Amazon_employee_access, BNG(breast-w), Rain_in_Australia, jm1, MagicTelescope

</details>

<details>
<summary>下降数据集：48 个</summary>

- cmc, Performance-Prediction, website_phishing, Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.5GHz(Urbinati), Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.5GHz(Urbinati), HR_Analytics_Job_Change_of_Data_Scientists, wine-quality-red, Diabetic_Retinopathy_Debrecen
- FOREX_audjpy-day-High, ada_agnostic, waveform_database_generator_version_1, predict_students_dropout_and_academic_success, segment, GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001, vehicle, FOREX_audchf-day-High
- PizzaCutter3, led24, pc1, Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.0GHz(Urbinati), pol, Fitness_Club_c, pc3, splice
- led7, MIC, house_16H, rice_cammeo_and_osmancik, steel_plates_faults, water_quality, Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-11.0GHz(Urbinati), default_of_credit_card_clients
- FOREX_audcad-hour-High, shrutime, Telecom_Churn_Dataset, dry_bean_dataset, PhishingWebsites, ringnorm, mobile_c36_oversampling, page-blocks
- mammography, htru, company_bankruptcy_prediction, thyroid, twonorm, BNG(cmc), internet_firewall, online_shoppers

</details>

<details>
<summary>持平数据集：88 个</summary>

- Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.0GHz(Urbinati), Customer_Personality_Analysis, mice_protein_expression, mfeat-pixel, mfeat-morphological, mfeat-karhunen, mfeat-fourier, mfeat-factors
- maternal_health_risk, madeline, law-school-admission-bianry, kr-vs-kp, kc1, jasmine, in_vehicle_coupon_recommendation, ibm-employee-performance
- heloc, golf_play_dataset_extended, first-order-theorem-proving, eucalyptus, estimation_of_obesity_levels, microaggregation2, mozilla4, national-longitudinal-survey-binary
- spambase, wine-quality-white, wine, waveform-5000, thyroid-dis, telco-customer-churn, svmguide3, statlog
- sports_articles_for_objectivity_analysis, shuttle, optdigits, shill-bidding, semeion, seismic+bumps, rl, phoneme
- pendigits, ozone_level, ozone-level-8hr, eeg-eye-state, drug_consumption, dna, Is-this-a-good-customer, SDSS17
- QSAR_biodegradation, PieChart3, Mobile_Price_Classification, Marketing_Campaign, KDDCup09_upselling, KDD, JapaneseVowels, Intersectional-Bias-Assessment
- VulNoneVul, Insurance, Heart-Disease-Dataset-(Comprehensive), Gender_Gap_in_Spanish_WP, GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1, FOREX_audjpy-hour-High, FOREX_audcad-day-High, FICO-HELOC-cleaned
- Employee, Shipping, Water_Quality_and_Potability, dis, autoUniv-au4-2500, delta_ailerons, credit-g, credit
- contraceptive_method_choice, compass, churn, banknote_authentication, autoUniv-au7-1100, artificial-characters, Waterstress, analcatdata_authorship
- allrep, allbp, airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True, ada_prior, ada, abalone, Wilt, yeast

</details>

## 研究判断与下一轮建议

1. 这份 `seed42 2` run 可以作为 TTT 有效性的正证据，但结论应写成“weighted 正收益、逐数据集不稳定”，而不是“普遍提升”。
2. 优先排查 `cmc`：这是最严重负迁移样本，accuracy、balanced accuracy、log loss 同时明显变差。
3. 单独复核 `Click_prediction_small`：它贡献了最大 net correct，但 balanced accuracy 明显下降，可能有类别不平衡下的 accuracy gate 偏置。
4. 对 `eye_movements`、`jungle_chess_2pcs_raw_endgame_complete`、`electricity` 做机制复查：这些是大收益核心来源，决定了 weighted 口径。
5. 下一轮建议同时跑 balanced accuracy / log loss gate，对比是否能减少 `cmc`、`Performance-Prediction`、`website_phishing` 这类负迁移。
