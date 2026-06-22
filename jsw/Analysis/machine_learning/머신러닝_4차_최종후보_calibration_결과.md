# 머신러닝 4차 최종 후보 calibration 결과

## 1. 실행 목적

- lockbox test 전에 model-level ensemble 후보의 score 사용 방식을 확정했다.
- rank 평균은 순위 성능용, geometric mean 계열은 확률/threshold 운영용으로 분리해 평가했다.
- lockbox test는 사용하지 않았다.

## 2. 기준선

- 로지스틱 Stage17 AUPRC: 0.2398
- 1차 ML 최고 AUPRC: 0.2984
- 1차 ML 최고 Brier/log loss: 0.07415 / 0.25806

## 3. 최종 선택

- 순위용 score: `model_level_top5_rank_average / raw`
  - AUPRC 0.3068, ROC AUC 0.8187, top-risk 우선순위용
- 확률/threshold용 score: `model_level_top5_geometric_mean / raw`
  - AUPRC 0.3034, Brier 0.07328, log loss 0.24994
  - 1차 최고 대비 ΔAUPRC +0.0050

## 4. calibration 비교

| model                           | score_type    |   auprc |   auroc |   brier |   log_loss |   best_f1_f1 |   best_f1_precision |   best_f1_recall |   delta_auprc_vs_stage1_best |   delta_brier_vs_stage1_best |   delta_log_loss_vs_stage1_best |
|:--------------------------------|:--------------|--------:|--------:|--------:|-----------:|-------------:|--------------------:|-----------------:|-----------------------------:|-----------------------------:|--------------------------------:|
| model_level_top5_rank_average   | raw           | 0.30682 | 0.81865 | 0.2751  |    0.80025 |      0.34722 |             0.25154 |          0.56039 |                      0.00843 |                      0.20095 |                         0.54218 |
| model_level_top5_rank_average   | sigmoid_logit | 0.30599 | 0.81807 | 0.07315 |    0.24938 |      0.34737 |             0.26391 |          0.50805 |                      0.00759 |                     -0.001   |                        -0.00868 |
| model_level_top5_geometric_mean | raw           | 0.30343 | 0.81779 | 0.07328 |    0.24994 |      0.34689 |             0.2444  |          0.59742 |                      0.00504 |                     -0.00086 |                        -0.00812 |
| model_level_top5_perf_weighted  | raw           | 0.30284 | 0.81694 | 0.07297 |    0.24846 |      0.34789 |             0.24499 |          0.59984 |                      0.00444 |                     -0.00118 |                        -0.0096  |
| model_level_top5_average        | raw           | 0.30283 | 0.81692 | 0.07296 |    0.2484  |      0.34766 |             0.24436 |          0.60225 |                      0.00443 |                     -0.00119 |                        -0.00967 |
| model_level_top5_logit_average  | raw           | 0.30251 | 0.81788 | 0.07321 |    0.24964 |      0.34679 |             0.25795 |          0.52899 |                      0.00411 |                     -0.00094 |                        -0.00842 |
| model_level_top5_geometric_mean | sigmoid_raw   | 0.29757 | 0.81622 | 0.07413 |    0.26108 |      0.34706 |             0.24135 |          0.61755 |                     -0.00082 |                     -2e-05   |                         0.00302 |
| model_level_top5_logit_average  | sigmoid_raw   | 0.29754 | 0.81626 | 0.07417 |    0.26124 |      0.34711 |             0.25551 |          0.54106 |                     -0.00086 |                      2e-05   |                         0.00318 |
| model_level_top5_average        | sigmoid_logit | 0.29716 | 0.81597 | 0.07258 |    0.24645 |      0.34813 |             0.24469 |          0.60306 |                     -0.00123 |                     -0.00157 |                        -0.01161 |
| model_level_top5_logit_average  | sigmoid_logit | 0.29715 | 0.81715 | 0.07253 |    0.24555 |      0.3468  |             0.24726 |          0.58052 |                     -0.00124 |                     -0.00162 |                        -0.01251 |
| model_level_top5_perf_weighted  | sigmoid_logit | 0.29714 | 0.816   | 0.07258 |    0.24645 |      0.34748 |             0.24405 |          0.60306 |                     -0.00125 |                     -0.00157 |                        -0.01161 |
| model_level_top5_perf_weighted  | sigmoid_raw   | 0.29709 | 0.81517 | 0.07414 |    0.26089 |      0.34962 |             0.24685 |          0.59903 |                     -0.00131 |                     -0       |                         0.00282 |
| model_level_top5_average        | sigmoid_raw   | 0.29708 | 0.81514 | 0.07413 |    0.26078 |      0.34926 |             0.24676 |          0.59742 |                     -0.00131 |                     -2e-05   |                         0.00272 |
| model_level_top5_geometric_mean | sigmoid_logit | 0.29708 | 0.81707 | 0.07254 |    0.24558 |      0.34687 |             0.25709 |          0.53301 |                     -0.00132 |                     -0.00161 |                        -0.01248 |
| model_level_top5_rank_average   | isotonic      | 0.28537 | 0.81467 | 0.07246 |    0.24803 |      0.34478 |             0.22475 |          0.73994 |                     -0.01302 |                     -0.00169 |                        -0.01003 |
| model_level_top5_rank_average   | sigmoid_raw   | 0.2827  | 0.81773 | 0.0732  |    0.24711 |      0.34775 |             0.26613 |          0.50161 |                     -0.01569 |                     -0.00095 |                        -0.01095 |
| model_level_top5_geometric_mean | isotonic      | 0.28082 | 0.81413 | 0.07315 |    0.24988 |      0.34292 |             0.22325 |          0.73913 |                     -0.01757 |                     -0.00099 |                        -0.00818 |
| model_level_top5_average        | isotonic      | 0.2804  | 0.81289 | 0.07318 |    0.25111 |      0.34403 |             0.26201 |          0.50081 |                     -0.018   |                     -0.00097 |                        -0.00696 |
| model_level_top5_perf_weighted  | isotonic      | 0.27995 | 0.81299 | 0.07319 |    0.25109 |      0.34581 |             0.26276 |          0.50564 |                     -0.01844 |                     -0.00095 |                        -0.00697 |
| model_level_top5_logit_average  | isotonic      | 0.2791  | 0.81374 | 0.07325 |    0.24843 |      0.34324 |             0.22352 |          0.73913 |                     -0.01929 |                     -0.0009  |                        -0.00963 |

## 5. 최종 후보 threshold

| feature_set          | feature_group   | model                           | score_type   | operating_point   |   threshold |   selected_n |   selected_rate |   tp |   fp |   fn |    tn |   precision |   recall |   specificity |      f1 |   accuracy |   balanced_accuracy |     mcc |
|:---------------------|:----------------|:--------------------------------|:-------------|:------------------|------------:|-------------:|----------------:|-----:|-----:|-----:|------:|------------:|---------:|--------------:|--------:|-----------:|--------------------:|--------:|
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_geometric_mean | raw          | fixed_0.50        |     0.5     |           60 |         0.0044  |   52 |    8 | 1190 | 12382 |     0.86667 |  0.04187 |       0.99935 | 0.07988 |    0.91212 |             0.52061 | 0.1792  |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_geometric_mean | raw          | best_f1           |     0.11822 |         3036 |         0.22271 |  742 | 2294 |  500 | 10096 |     0.2444  |  0.59742 |       0.81485 | 0.34689 |    0.79504 |             0.70614 | 0.28514 |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_geometric_mean | raw          | recall_ge_0.50    |     0.13757 |         2401 |         0.17613 |  621 | 1780 |  621 | 10610 |     0.25864 |  0.5     |       0.85634 | 0.34093 |    0.82387 |             0.67817 | 0.26918 |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_geometric_mean | raw          | recall_ge_0.70    |     0.09195 |         3880 |         0.28462 |  870 | 3010 |  372 |  9380 |     0.22423 |  0.70048 |       0.75706 | 0.33971 |    0.75191 |             0.72877 | 0.29179 |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_geometric_mean | raw          | recall_ge_0.90    |     0.03828 |         6155 |         0.45151 | 1118 | 5037 |  124 |  7353 |     0.18164 |  0.90016 |       0.59346 | 0.30228 |    0.62141 |             0.74681 | 0.28544 |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_rank_average   | raw          | fixed_0.50        |     0.5     |         6700 |         0.49149 | 1157 | 5543 |   85 |  6847 |     0.17269 |  0.93156 |       0.55262 | 0.29136 |    0.58715 |             0.74209 | 0.2787  |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_rank_average   | raw          | best_f1           |     0.79385 |         2767 |         0.20298 |  696 | 2071 |  546 | 10319 |     0.25154 |  0.56039 |       0.83285 | 0.34722 |    0.80803 |             0.69662 | 0.28134 |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_rank_average   | raw          | recall_ge_0.50    |     0.81834 |         2361 |         0.1732  |  621 | 1740 |  621 | 10650 |     0.26302 |  0.5     |       0.85956 | 0.34471 |    0.8268  |             0.67978 | 0.27343 |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_rank_average   | raw          | recall_ge_0.70    |     0.72295 |         3835 |         0.28132 |  870 | 2965 |  372 |  9425 |     0.22686 |  0.70048 |       0.76069 | 0.34272 |    0.75521 |             0.73059 | 0.29514 |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_rank_average   | raw          | recall_ge_0.90    |     0.54726 |         6077 |         0.44579 | 1118 | 4959 |  124 |  7431 |     0.18397 |  0.90016 |       0.59976 | 0.30551 |    0.62713 |             0.74996 | 0.28942 |

## 6. 확률 후보 top-risk capture

| feature_set          | feature_group   | model                           | score_type   |   top_pct |   selected_n |   selected_rate |   threshold_min |   captured_positive_n |   total_positive_n |   capture_rate_recall |   precision |   base_positive_rate |   lift_vs_base |
|:---------------------|:----------------|:--------------------------------|:-------------|----------:|-------------:|----------------:|----------------:|----------------------:|-------------------:|----------------------:|------------:|---------------------:|---------------:|
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_geometric_mean | raw          |      0.05 |          682 |         0.05003 |         0.23256 |                   220 |               1242 |               0.17713 |     0.32258 |              0.09111 |        3.5406  |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_geometric_mean | raw          |      0.1  |         1364 |         0.10006 |         0.1828  |                   395 |               1242 |               0.31804 |     0.28959 |              0.09111 |        3.17849 |
| ENSEMBLE_MODEL_LEVEL | stage3_ensemble | model_level_top5_geometric_mean | raw          |      0.2  |         2727 |         0.20004 |         0.12736 |                   681 |               1242 |               0.54831 |     0.24972 |              0.09111 |        2.74094 |

## 7. 확률 후보 hard-negative subgroup

| model                           | score_type   | subgroup   |   auprc |   auroc |   brier |   log_loss |
|:--------------------------------|:-------------|:-----------|--------:|--------:|--------:|-----------:|
| model_level_top5_geometric_mean | raw          | Target_0A  | 0.30979 | 0.66459 | 0.13392 |    0.4473  |
| model_level_top5_geometric_mean | raw          | Target_0B1 | 0.921   | 0.96727 | 0.18    |    0.54139 |
| model_level_top5_geometric_mean | raw          | Target_0B2 | 0.95659 | 0.97568 | 0.23865 |    0.71203 |

## 8. 해석

- calibration을 적용하면 일부 Brier/log loss는 약간 개선될 수 있지만, AUPRC가 떨어지는 경우가 있었다.
- 최종 ranking score는 AUPRC가 가장 높은 raw rank average로 둔다.
- 최종 probability score는 AUPRC +0.005 기준을 넘고 확률 지표가 안정적인 raw geometric mean으로 둔다.
- Step5에서는 이 선택을 고정한 뒤 lockbox test를 최초로 평가한다.

## 9. 검증

| check                            |   value |   expected | passed   |
|:---------------------------------|--------:|-----------:|:---------|
| candidate_models_n               |       5 |          5 | True     |
| development_row_count_per_model  |   13632 |      13632 | True     |
| development_positive_n_per_model |    1242 |       1242 | True     |
| lockbox_overlap                  |       0 |          0 | True     |
| development_manifest_match       |       0 |          0 | True     |
| selection_rows                   |       2 |          2 | True     |
| score_raw_nan                    |       0 |          0 | True     |
| score_raw_inf                    |       0 |          0 | True     |
| score_raw_range                  |       0 |          0 | True     |
| score_sigmoid_nan                |       0 |          0 | True     |
| score_sigmoid_inf                |       0 |          0 | True     |
| score_sigmoid_range              |       0 |          0 | True     |
| score_sigmoid_logit_nan          |       0 |          0 | True     |
| score_sigmoid_logit_inf          |       0 |          0 | True     |
| score_sigmoid_logit_range        |       0 |          0 | True     |
| score_isotonic_nan               |       0 |          0 | True     |
| score_isotonic_inf               |       0 |          0 | True     |
| score_isotonic_range             |       0 |          0 | True     |

## 10. 산출물

- `outputs/metrics/ml_stage4_calibration_comparison.csv`
- `outputs/metrics/ml_stage4_final_candidate_selection.csv`
- `outputs/metrics/ml_stage4_final_thresholds.csv`
- `outputs/metrics/ml_stage4_final_top_risk_capture.csv`
- `outputs/metrics/ml_stage4_final_subgroup_metrics.csv`
- `outputs/predictions/ml_stage4_final_candidate_oof_predictions.csv`
- `outputs/models/stage4_final_selection/stage4_final_selection_manifest.json`
