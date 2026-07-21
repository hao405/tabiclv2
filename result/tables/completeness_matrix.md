# Completeness Matrix

| Dataset | Model | Method | Status | Coverage | status=ok | Adaptation | Note |
|---|---|---|---|---:|---:|---:|---|
| data184 | TabICLv2 | Infer | complete | 184/184 | 184 | 0/0 | — |
| data184 | TabICLv2 | FT | complete | 184/184 | 184 | 184/184 | selected_on_eval;middle3_avg_accuracy_ok |
| data184 | TabICLv2 | F-aware FT | complete | 184/184 | 184 | 184/184 | selected_on_eval;top3_avg_accuracy_ok |
| data184 | TabICLv2 | LoRA | complete | 184/184 | 184 | 184/184 | — |
| data184 | TabICLv2 | ln_head_embedding | complete | 184/184 | 184 | 184/184 | — |
| data184 | TabICLv2 | Last-block | complete | 184/184 | 184 | 184/184 | — |
| data184 | TabICLv2 | LoCalPFN | complete_with_failures | 184/184 | 95 | 95/184 | — |
| data184 | TabICLv2 | MICP | missing | 0/184 | 0 | 0/184 | — |
| OpenML-CC18 | TabICLv2 | Infer | complete_with_failures | 67/67 | 66 | 0/0 | — |
| OpenML-CC18 | TabICLv2 | FT | complete_with_failures | 67/67 | 64 | 59/67 | — |
| OpenML-CC18 | TabICLv2 | F-aware FT | complete_with_failures | 67/67 | 64 | 59/67 | — |
| OpenML-CC18 | TabICLv2 | LoRA | complete_with_failures | 67/67 | 65 | 59/67 | — |
| OpenML-CC18 | TabICLv2 | ln_head_embedding | complete_with_failures | 67/67 | 65 | 60/67 | — |
| OpenML-CC18 | TabICLv2 | Last-block | complete_with_failures | 67/67 | 65 | 64/67 | — |
| OpenML-CC18 | TabICLv2 | LoCalPFN | missing | 0/67 | 0 | 0/67 | — |
| OpenML-CC18 | TabICLv2 | MICP | missing | 0/67 | 0 | 0/67 | — |
| Graph-SCM | TabICLv2 | Infer | complete | 512/512 | 512 | 0/0 | — |
| Graph-SCM | TabICLv2 | FT | complete_with_failures | 512/512 | 512 | 506/512 | — |
| Graph-SCM | TabICLv2 | F-aware FT | complete_with_failures | 512/512 | 512 | 508/512 | — |
| Graph-SCM | TabICLv2 | LoRA | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | TabICLv2 | ln_head_embedding | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | TabICLv2 | Last-block | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | TabICLv2 | LoCalPFN | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | TabICLv2 | MICP | missing | 0/512 | 0 | 0/512 | — |
| data184 | TabPFNv3 | Infer | complete | 184/184 | 184 | 0/0 | — |
| data184 | TabPFNv3 | FT | complete_with_failures | 184/184 | 183 | 183/184 | selected_on_eval |
| data184 | TabPFNv3 | F-aware FT | complete_with_failures | 184/184 | 183 | 183/184 | — |
| data184 | TabPFNv3 | LoRA | complete_with_failures | 184/184 | 178 | 177/184 | — |
| data184 | TabPFNv3 | ln_head_embedding | complete_with_failures | 184/184 | 182 | 181/184 | — |
| data184 | TabPFNv3 | Last-block | complete_with_failures | 184/184 | 183 | 183/184 | — |
| data184 | TabPFNv3 | LoCalPFN | partial | 3/184 | 2 | 2/184 | missing=181 |
| data184 | TabPFNv3 | MICP | missing | 0/184 | 0 | 0/184 | — |
| OpenML-CC18 | TabPFNv3 | Infer | complete | 67/67 | 67 | 0/0 | — |
| OpenML-CC18 | TabPFNv3 | FT | complete_with_failures | 67/67 | 65 | 61/67 | — |
| OpenML-CC18 | TabPFNv3 | F-aware FT | complete_with_failures | 67/67 | 65 | 61/67 | — |
| OpenML-CC18 | TabPFNv3 | LoRA | complete_with_failures | 67/67 | 62 | 56/67 | — |
| OpenML-CC18 | TabPFNv3 | ln_head_embedding | complete_with_failures | 67/67 | 65 | 58/67 | — |
| OpenML-CC18 | TabPFNv3 | Last-block | complete_with_failures | 67/67 | 63 | 62/67 | — |
| OpenML-CC18 | TabPFNv3 | LoCalPFN | missing | 0/67 | 0 | 0/67 | — |
| OpenML-CC18 | TabPFNv3 | MICP | missing | 0/67 | 0 | 0/67 | — |
| Graph-SCM | TabPFNv3 | Infer | complete | 512/512 | 512 | 0/0 | — |
| Graph-SCM | TabPFNv3 | FT | complete | 512/512 | 512 | 512/512 | — |
| Graph-SCM | TabPFNv3 | F-aware FT | complete | 512/512 | 512 | 512/512 | selected_on_eval |
| Graph-SCM | TabPFNv3 | LoRA | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | TabPFNv3 | ln_head_embedding | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | TabPFNv3 | Last-block | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | TabPFNv3 | LoCalPFN | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | TabPFNv3 | MICP | missing | 0/512 | 0 | 0/512 | — |
| data184 | LimiX-2M | Infer | complete_with_failures | 184/184 | 178 | 0/0 | — |
| data184 | LimiX-2M | FT | partial | 87/184 | 83 | 83/184 | missing=97 |
| data184 | LimiX-2M | F-aware FT | missing | 0/184 | 0 | 0/184 | — |
| data184 | LimiX-2M | LoRA | missing | 0/184 | 0 | 0/184 | — |
| data184 | LimiX-2M | ln_head_embedding | missing | 0/184 | 0 | 0/184 | — |
| data184 | LimiX-2M | Last-block | missing | 0/184 | 0 | 0/184 | — |
| data184 | LimiX-2M | LoCalPFN | missing | 0/184 | 0 | 0/184 | — |
| data184 | LimiX-2M | MICP | missing | 0/184 | 0 | 0/184 | — |
| OpenML-CC18 | LimiX-2M | Infer | missing | 0/67 | 0 | 0/0 | — |
| OpenML-CC18 | LimiX-2M | FT | missing | 0/67 | 0 | 0/67 | — |
| OpenML-CC18 | LimiX-2M | F-aware FT | missing | 0/67 | 0 | 0/67 | — |
| OpenML-CC18 | LimiX-2M | LoRA | missing | 0/67 | 0 | 0/67 | — |
| OpenML-CC18 | LimiX-2M | ln_head_embedding | missing | 0/67 | 0 | 0/67 | — |
| OpenML-CC18 | LimiX-2M | Last-block | missing | 0/67 | 0 | 0/67 | — |
| OpenML-CC18 | LimiX-2M | LoCalPFN | missing | 0/67 | 0 | 0/67 | — |
| OpenML-CC18 | LimiX-2M | MICP | missing | 0/67 | 0 | 0/67 | — |
| Graph-SCM | LimiX-2M | Infer | missing | 0/512 | 0 | 0/0 | — |
| Graph-SCM | LimiX-2M | FT | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | LimiX-2M | F-aware FT | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | LimiX-2M | LoRA | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | LimiX-2M | ln_head_embedding | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | LimiX-2M | Last-block | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | LimiX-2M | LoCalPFN | missing | 0/512 | 0 | 0/512 | — |
| Graph-SCM | LimiX-2M | MICP | missing | 0/512 | 0 | 0/512 | — |

## missing

- `data184 / TabICLv2 / MICP`: 0/184, `no qualifying artifact`
- `OpenML-CC18 / TabICLv2 / LoCalPFN`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / TabICLv2 / MICP`: 0/67, `no qualifying artifact`
- `Graph-SCM / TabICLv2 / LoRA`: 0/512, `no qualifying artifact`
- `Graph-SCM / TabICLv2 / ln_head_embedding`: 0/512, `no qualifying artifact`
- `Graph-SCM / TabICLv2 / Last-block`: 0/512, `no qualifying artifact`
- `Graph-SCM / TabICLv2 / LoCalPFN`: 0/512, `no qualifying artifact`
- `Graph-SCM / TabICLv2 / MICP`: 0/512, `no qualifying artifact`
- `data184 / TabPFNv3 / MICP`: 0/184, `no qualifying artifact`
- `OpenML-CC18 / TabPFNv3 / LoCalPFN`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / TabPFNv3 / MICP`: 0/67, `no qualifying artifact`
- `Graph-SCM / TabPFNv3 / LoRA`: 0/512, `no qualifying artifact`
- `Graph-SCM / TabPFNv3 / ln_head_embedding`: 0/512, `no qualifying artifact`
- `Graph-SCM / TabPFNv3 / Last-block`: 0/512, `no qualifying artifact`
- `Graph-SCM / TabPFNv3 / LoCalPFN`: 0/512, `no qualifying artifact`
- `Graph-SCM / TabPFNv3 / MICP`: 0/512, `no qualifying artifact`
- `data184 / LimiX-2M / F-aware FT`: 0/184, `no qualifying artifact`
- `data184 / LimiX-2M / LoRA`: 0/184, `no qualifying artifact`
- `data184 / LimiX-2M / ln_head_embedding`: 0/184, `no qualifying artifact`
- `data184 / LimiX-2M / Last-block`: 0/184, `no qualifying artifact`
- `data184 / LimiX-2M / LoCalPFN`: 0/184, `no qualifying artifact`
- `data184 / LimiX-2M / MICP`: 0/184, `no qualifying artifact`
- `OpenML-CC18 / LimiX-2M / Infer`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / LimiX-2M / FT`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / LimiX-2M / F-aware FT`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / LimiX-2M / LoRA`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / LimiX-2M / ln_head_embedding`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / LimiX-2M / Last-block`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / LimiX-2M / LoCalPFN`: 0/67, `no qualifying artifact`
- `OpenML-CC18 / LimiX-2M / MICP`: 0/67, `no qualifying artifact`
- `Graph-SCM / LimiX-2M / Infer`: 0/512, `no qualifying artifact`
- `Graph-SCM / LimiX-2M / FT`: 0/512, `no qualifying artifact`
- `Graph-SCM / LimiX-2M / F-aware FT`: 0/512, `no qualifying artifact`
- `Graph-SCM / LimiX-2M / LoRA`: 0/512, `no qualifying artifact`
- `Graph-SCM / LimiX-2M / ln_head_embedding`: 0/512, `no qualifying artifact`
- `Graph-SCM / LimiX-2M / Last-block`: 0/512, `no qualifying artifact`
- `Graph-SCM / LimiX-2M / LoCalPFN`: 0/512, `no qualifying artifact`
- `Graph-SCM / LimiX-2M / MICP`: 0/512, `no qualifying artifact`

## running

—

## partial

- `data184 / TabPFNv3 / LoCalPFN`: 3/184, `/home/crc00006699/zh/tabiclv2/results/PEFT_compare/localpfn_fast_explore_k256_q128_e5_s5_data184_seed42_20260719/tabpfnv3/localpfn/all_classification_results.csv`
- `data184 / LimiX-2M / FT`: 87/184, `/home/crc00006699/zh/tabiclv2/results/limix/data184/limix2m/seed42/ft/all_classification_results.csv`

## complete_with_failures

- `data184 / TabICLv2 / LoCalPFN`: 184/184, `/home/crc00006699/zh/tabiclv2/results/PEFT_compare/localpfn_fast_explore_k256_q128_e5_s5_data184_seed42_20260719/tabiclv2/localpfn/all_classification_results.csv`
- `OpenML-CC18 / TabICLv2 / Infer`: 67/67, `/home/crc00006699/zh/tabiclv2/results/managed_experiments/openmlcc18_2x3_full_20260715_223032/tabicl-v2/infer/seed42/all_classification_results.csv`
- `OpenML-CC18 / TabICLv2 / FT`: 67/67, `/home/crc00006699/zh/tabiclv2/results/managed_experiments/openmlcc18_2x3_full_20260715_223032/tabicl-v2/ft/seed42/all_classification_results.csv`
- `OpenML-CC18 / TabICLv2 / F-aware FT`: 67/67, `/home/crc00006699/zh/tabiclv2/results/managed_experiments/openmlcc18_2x3_full_20260715_223032/tabicl-v2/faware_ft/seed42/all_classification_results.csv`
- `OpenML-CC18 / TabICLv2 / LoRA`: 67/67, `/home/crc00006699/zh/tabiclv2/results/PEFT/tabiclv2/lora/openmlcc18_peft_2x3_full_20260717_113939/all_classification_results.csv`
- `OpenML-CC18 / TabICLv2 / ln_head_embedding`: 67/67, `/home/crc00006699/zh/tabiclv2/results/PEFT/tabiclv2/ln_head_embedding/openmlcc18_peft_2x3_full_20260717_113939/all_classification_results.csv`
- `OpenML-CC18 / TabICLv2 / Last-block`: 67/67, `/home/crc00006699/zh/tabiclv2/results/PEFT/tabiclv2/last_layers/openmlcc18_peft_2x3_full_20260717_113939/all_classification_results.csv`
- `Graph-SCM / TabICLv2 / FT`: 512/512, `/Users/zhuhao/experiment/second_learning/tabiclv2_test/results/合成数据实验/matrix/tabiclv2_graph_scm_2x3_full_20260716_090925/tabicl-v2/ft/seed42/all_classification_results.csv`
- `Graph-SCM / TabICLv2 / F-aware FT`: 512/512, `/Users/zhuhao/experiment/second_learning/tabiclv2_test/results/合成数据实验/matrix/tabiclv2_graph_scm_2x3_full_20260716_090925/tabicl-v2/faware_ft/seed42/all_classification_results.csv`
- `data184 / TabPFNv3 / FT`: 184/184, `/Users/zhuhao/experiment/second_learning/tabiclv2_test/results/tabpfn/v3/tabpfnv3_best_para/all_classification_results.csv`
- `data184 / TabPFNv3 / F-aware FT`: 184/184, `/home/crc00006699/zh/tabiclv2/results/tabpfn/tabpfn_v3_Data184/f_mmd/tabpfn_v3/seed42/all_classification_results.csv`
- `data184 / TabPFNv3 / LoRA`: 184/184, `/Users/zhuhao/experiment/second_learning/tabiclv2_test/results/PEFT_results/talent184/tabpfnv3/lora/tabpfnv3_peft_data184_direct_20260716_122453/all_classification_results.csv`
- `data184 / TabPFNv3 / ln_head_embedding`: 184/184, `/Users/zhuhao/experiment/second_learning/tabiclv2_test/results/PEFT_results/talent184/tabpfnv3/ln_head_embedding/tabpfnv3_peft_data184_direct_20260716_122453/all_classification_results.csv`
- `data184 / TabPFNv3 / Last-block`: 184/184, `/Users/zhuhao/experiment/second_learning/tabiclv2_test/results/PEFT_results/talent184/tabpfnv3/last_layers/tabpfnv3_peft_data184_direct_20260716_122453/all_classification_results.csv`
- `OpenML-CC18 / TabPFNv3 / FT`: 67/67, `/home/crc00006699/zh/tabiclv2/results/managed_experiments/openmlcc18_2x3_full_20260715_223032/tabpfn-v3/ft/seed42/all_classification_results.csv`
- `OpenML-CC18 / TabPFNv3 / F-aware FT`: 67/67, `/home/crc00006699/zh/tabiclv2/results/managed_experiments/openmlcc18_2x3_full_20260715_223032/tabpfn-v3/faware_ft/seed42/all_classification_results.csv`
- `OpenML-CC18 / TabPFNv3 / LoRA`: 67/67, `/home/crc00006699/zh/tabiclv2/results/PEFT/tabpfnv3/lora/openmlcc18_peft_2x3_full_20260717_113939/all_classification_results.csv`
- `OpenML-CC18 / TabPFNv3 / ln_head_embedding`: 67/67, `/home/crc00006699/zh/tabiclv2/results/PEFT/tabpfnv3/ln_head_embedding/openmlcc18_peft_2x3_full_20260717_113939/all_classification_results.csv`
- `OpenML-CC18 / TabPFNv3 / Last-block`: 67/67, `/home/crc00006699/zh/tabiclv2/results/PEFT/tabpfnv3/last_layers/openmlcc18_peft_2x3_full_20260717_113939/all_classification_results.csv`
- `data184 / LimiX-2M / Infer`: 184/184, `/home/crc00006699/zh/tabiclv2/results/limix/data184/limix2m/seed42/infer/all_classification_results.csv`

## invalid

—
