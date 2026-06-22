# 머신러닝 5차 lockbox test 평가 결과

## 1. 실행 목적

- Step4에서 고정한 최종 후보를 lockbox test에 최초 적용했다.
- 새 후보 선택이나 추가 튜닝은 하지 않았다.
- full-development로 학습된 Step3.5 base model artifact만 사용했다.

## 2. lockbox 구성

- lockbox row: 3,413
- Target 1: 311
- positive rate: 0.0911

## 3. lockbox 최종 성능

| selection_role    | model                           |   auprc |   auroc |   brier |   log_loss |   best_f1_f1 |   best_f1_precision |   best_f1_recall |   delta_auprc_vs_logistic_oof_baseline |   delta_auprc_vs_stage1_oof_best |
|:------------------|:--------------------------------|--------:|--------:|--------:|-----------:|-------------:|--------------------:|-----------------:|---------------------------------------:|---------------------------------:|
| probability_score | model_level_top5_geometric_mean | 0.62882 | 0.91053 | 0.05972 |    0.20333 |      0.56997 |             0.60727 |          0.53698 |                                0.38898 |                          0.33042 |
| ranking_score     | model_level_top5_rank_average   | 0.61734 | 0.90881 | 0.24547 |    0.68774 |      0.55405 |             0.58363 |          0.52733 |                                0.37751 |                          0.31895 |

## 4. 개발 OOF 대비 lockbox 변화

| selection_role    | model                           | metric            |   development_oof |   lockbox_test |   delta_lockbox_minus_oof |
|:------------------|:--------------------------------|:------------------|------------------:|---------------:|--------------------------:|
| ranking_score     | model_level_top5_rank_average   | auprc             |           0.30682 |        0.61734 |                   0.31052 |
| ranking_score     | model_level_top5_rank_average   | auroc             |           0.81865 |        0.90881 |                   0.09016 |
| ranking_score     | model_level_top5_rank_average   | brier             |           0.2751  |        0.24547 |                  -0.02963 |
| ranking_score     | model_level_top5_rank_average   | log_loss          |           0.80025 |        0.68774 |                  -0.11251 |
| ranking_score     | model_level_top5_rank_average   | best_f1_f1        |           0.34722 |        0.55405 |                   0.20684 |
| ranking_score     | model_level_top5_rank_average   | best_f1_precision |           0.25154 |        0.58363 |                   0.33209 |
| ranking_score     | model_level_top5_rank_average   | best_f1_recall    |           0.56039 |        0.52733 |                  -0.03306 |
| probability_score | model_level_top5_geometric_mean | auprc             |           0.30343 |        0.62882 |                   0.32538 |
| probability_score | model_level_top5_geometric_mean | auroc             |           0.81779 |        0.91053 |                   0.09274 |
| probability_score | model_level_top5_geometric_mean | brier             |           0.07328 |        0.05972 |                  -0.01357 |
| probability_score | model_level_top5_geometric_mean | log_loss          |           0.24994 |        0.20333 |                  -0.04662 |
| probability_score | model_level_top5_geometric_mean | best_f1_f1        |           0.34689 |        0.56997 |                   0.22307 |
| probability_score | model_level_top5_geometric_mean | best_f1_precision |           0.2444  |        0.60727 |                   0.36287 |
| probability_score | model_level_top5_geometric_mean | best_f1_recall    |           0.59742 |        0.53698 |                  -0.06045 |

## 5. 해석

- ranking score `model_level_top5_rank_average`는 lockbox AUPRC 0.6173를 기록했다.
- probability score `model_level_top5_geometric_mean`는 lockbox AUPRC 0.6288, Brier 0.05972, log loss 0.20333를 기록했다.
- probability score는 로지스틱 OOF 기준선 AUPRC 0.2398 대비 +0.3890, 1차 ML OOF 최고 AUPRC 0.2984 대비 +0.3304이다.
- lockbox 결과는 development OOF보다 낮아졌는지를 중심으로 해석해야 하며, 이번 결과 이후에는 lockbox를 이용한 재튜닝을 하지 않는다.

## 6. probability score 고정 threshold 성능

| dataset      | feature_set   | feature_group   | model                           | selection_role    | score_type   | operating_point   | source                                 |   threshold |   selected_n |   selected_rate |   tp |   fp |   fn |   tn |   precision |   recall |   specificity |      f1 |   accuracy |   balanced_accuracy |     mcc |
|:-------------|:--------------|:----------------|:--------------------------------|:------------------|:-------------|:------------------|:---------------------------------------|------------:|-------------:|----------------:|-----:|-----:|-----:|-----:|------------:|---------:|--------------:|--------:|-----------:|--------------------:|--------:|
| lockbox_test | LOCKBOX_FINAL | stage5_lockbox  | model_level_top5_geometric_mean | probability_score | raw          | fixed_0.50        | stage4_development_oof_fixed_threshold |     0.5     |           62 |         0.01817 |   62 |    0 |  249 | 3102 |     1       |  0.19936 |       1       | 0.33244 |    0.92704 |             0.59968 | 0.42959 |
| lockbox_test | LOCKBOX_FINAL | stage5_lockbox  | model_level_top5_geometric_mean | probability_score | raw          | best_f1           | stage4_development_oof_fixed_threshold |     0.11822 |          373 |         0.10929 |  185 |  188 |  126 | 2914 |     0.49598 |  0.59486 |       0.93939 | 0.54094 |    0.908   |             0.76712 | 0.49278 |
| lockbox_test | LOCKBOX_FINAL | stage5_lockbox  | model_level_top5_geometric_mean | probability_score | raw          | recall_ge_0.50    | stage4_development_oof_fixed_threshold |     0.13757 |          272 |         0.0797  |  164 |  108 |  147 | 2994 |     0.60294 |  0.52733 |       0.96518 | 0.56261 |    0.92529 |             0.74626 | 0.52336 |
| lockbox_test | LOCKBOX_FINAL | stage5_lockbox  | model_level_top5_geometric_mean | probability_score | raw          | recall_ge_0.70    | stage4_development_oof_fixed_threshold |     0.09195 |          546 |         0.15998 |  217 |  329 |   94 | 2773 |     0.39744 |  0.69775 |       0.89394 | 0.50642 |    0.87606 |             0.79584 | 0.4645  |
| lockbox_test | LOCKBOX_FINAL | stage5_lockbox  | model_level_top5_geometric_mean | probability_score | raw          | recall_ge_0.90    | stage4_development_oof_fixed_threshold |     0.03828 |         1230 |         0.36039 |  291 |  939 |   20 | 2163 |     0.23659 |  0.93569 |       0.69729 | 0.37768 |    0.71902 |             0.81649 | 0.37941 |

## 7. probability score top-risk capture

| dataset      | feature_set   | feature_group   | model                           | selection_role    | score_type   |   top_pct |   selected_n |   selected_rate |   threshold_min |   captured_positive_n |   total_positive_n |   capture_rate_recall |   precision |   base_positive_rate |   lift_vs_base |
|:-------------|:--------------|:----------------|:--------------------------------|:------------------|:-------------|----------:|-------------:|----------------:|----------------:|----------------------:|-------------------:|----------------------:|------------:|---------------------:|---------------:|
| lockbox_test | LOCKBOX_FINAL | stage5_lockbox  | model_level_top5_geometric_mean | probability_score | raw          |      0.05 |          171 |         0.0501  |         0.18898 |                   130 |                311 |               0.41801 |     0.76023 |              0.09112 |        8.34302 |
| lockbox_test | LOCKBOX_FINAL | stage5_lockbox  | model_level_top5_geometric_mean | probability_score | raw          |      0.1  |          342 |         0.10021 |         0.12379 |                   180 |                311 |               0.57878 |     0.52632 |              0.09112 |        5.77594 |
| lockbox_test | LOCKBOX_FINAL | stage5_lockbox  | model_level_top5_geometric_mean | probability_score | raw          |      0.2  |          683 |         0.20012 |         0.07875 |                   245 |                311 |               0.78778 |     0.35871 |              0.09112 |        3.9366  |

## 8. probability score hard-negative subgroup

| selection_role    | model                           | subgroup   |   auprc |   auroc |   brier |   log_loss |
|:------------------|:--------------------------------|:-----------|--------:|--------:|--------:|-----------:|
| probability_score | model_level_top5_geometric_mean | Target_0A  | 0.63507 | 0.83825 | 0.1092  |    0.36338 |
| probability_score | model_level_top5_geometric_mean | Target_0B1 | 0.95556 | 0.98018 | 0.1549  |    0.47053 |
| probability_score | model_level_top5_geometric_mean | Target_0B2 | 0.97933 | 0.98659 | 0.20925 |    0.63024 |

## 9. 검증

| check                                        |   value |   expected | passed   |
|:---------------------------------------------|--------:|-----------:|:---------|
| full_data_n                                  |   17045 |      17045 | True     |
| development_n                                |   13632 |      13632 | True     |
| lockbox_n                                    |    3413 |       3413 | True     |
| lockbox_positive_n                           |     311 |        311 | True     |
| prediction_ids_match_lockbox                 |       0 |          0 | True     |
| prediction_development_overlap               |       0 |          0 | True     |
| final_models_n                               |       2 |          2 | True     |
| base_models_n                                |       5 |          5 | True     |
| base_rows::PL_HGB                            |    3413 |       3413 | True     |
| base_nan::PL_HGB                             |       0 |          0 | True     |
| base_inf::PL_HGB                             |       0 |          0 | True     |
| base_range::PL_HGB                           |       0 |          0 | True     |
| base_rows::PL_LGBM                           |    3413 |       3413 | True     |
| base_nan::PL_LGBM                            |       0 |          0 | True     |
| base_inf::PL_LGBM                            |       0 |          0 | True     |
| base_range::PL_LGBM                          |       0 |          0 | True     |
| base_rows::RULES_HGB                         |    3413 |       3413 | True     |
| base_nan::RULES_HGB                          |       0 |          0 | True     |
| base_inf::RULES_HGB                          |       0 |          0 | True     |
| base_range::RULES_HGB                        |       0 |          0 | True     |
| base_rows::RULES_RF                          |    3413 |       3413 | True     |
| base_nan::RULES_RF                           |       0 |          0 | True     |
| base_inf::RULES_RF                           |       0 |          0 | True     |
| base_range::RULES_RF                         |       0 |          0 | True     |
| base_rows::RULES_XGB                         |    3413 |       3413 | True     |
| base_nan::RULES_XGB                          |       0 |          0 | True     |
| base_inf::RULES_XGB                          |       0 |          0 | True     |
| base_range::RULES_XGB                        |       0 |          0 | True     |
| final_rows::model_level_top5_geometric_mean  |    3413 |       3413 | True     |
| final_nan::model_level_top5_geometric_mean   |       0 |          0 | True     |
| final_inf::model_level_top5_geometric_mean   |       0 |          0 | True     |
| final_range::model_level_top5_geometric_mean |       0 |          0 | True     |
| final_rows::model_level_top5_rank_average    |    3413 |       3413 | True     |
| final_nan::model_level_top5_rank_average     |       0 |          0 | True     |
| final_inf::model_level_top5_rank_average     |       0 |          0 | True     |
| final_range::model_level_top5_rank_average   |       0 |          0 | True     |

## 10. 산출물

- `outputs/metrics/ml_stage5_lockbox_model_comparison.csv`
- `outputs/metrics/ml_stage5_oof_vs_lockbox_comparison.csv`
- `outputs/metrics/ml_stage5_lockbox_fixed_thresholds.csv`
- `outputs/metrics/ml_stage5_lockbox_top_risk_capture.csv`
- `outputs/metrics/ml_stage5_lockbox_subgroup_metrics.csv`
- `outputs/predictions/ml_stage5_lockbox_final_predictions.csv`
- `outputs/predictions/ml_stage5_lockbox_base_model_predictions.csv`
