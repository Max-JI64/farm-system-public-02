# 머신러닝 모델링 진행 로그

## 2026-06-20

### 1차 v2 실행

- 로지스틱 Step17 최종 결과 기반으로 1차 ML screening을 재구성했다.
- Optuna는 사용하지 않았다.
- 모든 피처셋에 모든 사용 가능 모델을 적용했다.
- 로지스틱 Stage17과 같은 metric/threshold/top-risk/subgroup 지표를 산출했다.

### 최고 결과

- 전체 최고: `PLUS_LANDCOVER / HistGradientBoosting / raw`
- AUPRC 0.2984, ROC AUC 0.8123, Brier 0.07415, ΔAUPRC +0.0586

### 그룹별 최고

| result_group          | feature_set    | model                |   auprc |   auroc |   brier |   log_loss |   delta_auprc_vs_logistic |
|:----------------------|:---------------|:---------------------|--------:|--------:|--------:|-----------:|--------------------------:|
| diagnostic_m123       | M1             | RandomForest         | 0.26014 | 0.76791 | 0.07552 |    0.26615 |                   0.0203  |
| logistic_final_guided | PLUS_LANDCOVER | HistGradientBoosting | 0.2984  | 0.81235 | 0.07415 |    0.25806 |                   0.05856 |
| overall               | PLUS_LANDCOVER | HistGradientBoosting | 0.2984  | 0.81235 | 0.07415 |    0.25806 |                   0.05856 |

### 실행 검증

- 실행 시간: 약 26분 33초
- 전체 결합 행 수: 17,045
- development 행 수: 13,632
- development Target 1: 1,242
- strict leak audit: model group/exposure/date/positive actual date leak 모두 0
- OOF prediction rows: 572,544
- raw feature/model 조합: 42개
- 각 조합별 행 수: 13,632
- 각 조합별 Target 1 수: 1,242
- raw score 범위: 0.000000000001 ~ 0.947359
- calibrated score 범위: 0.000000000001 ~ 0.999999999999
- raw/calibrated score NaN: 0

### 모델 포함 확인

모든 모델이 7개 피처셋에 대해 실행됐다.

| model | raw result rows |
|---|---:|
| CatBoost | 7 |
| ExtraTrees | 7 |
| HistGradientBoosting | 7 |
| LightGBM | 7 |
| RandomForest | 7 |
| XGBoost | 7 |

XGBoost는 누락되지 않았다. 주요 결과는 다음과 같다.

| feature_set | rank | AUPRC | ROC AUC | Brier | ΔAUPRC vs logistic |
|---|---:|---:|---:|---:|---:|
| M1 | 3 | 0.24437 | 0.77908 | 0.07552 | +0.00453 |
| M2 | 2 | 0.24660 | 0.77526 | 0.07563 | +0.00677 |
| M3 | 4 | 0.16869 | 0.69402 | 0.13923 | -0.07115 |
| STAGE7_RECOMMENDED | 6 | 0.18036 | 0.74552 | 0.08869 | -0.05947 |
| PLUS_LANDCOVER | 5 | 0.19950 | 0.75687 | 0.12069 | -0.04034 |
| PLUS_LANDCOVER_RULES_ANOVA_PROXY | 3 | 0.28014 | 0.81380 | 0.07333 | +0.04030 |
| FINAL_REDUCED_WITH_FWI_PROXY | 5 | 0.15705 | 0.69729 | 0.10714 | -0.08279 |

### 결과 해석

1차 v2 결과는 로지스틱 대비 명확한 개선이 확인된 screening으로 해석된다. 기준 로지스틱의 AUPRC는 0.2398이었고, 최고 ML 후보인 `PLUS_LANDCOVER / HistGradientBoosting / raw`는 AUPRC 0.2984를 기록했다. 절대 개선폭은 +0.0586이고, 상대적으로는 약 24.4% 개선이다. ROC AUC도 0.7768에서 0.8123으로 상승했고, Brier와 log loss도 낮아져서 순위 성능뿐 아니라 확률 품질도 같이 좋아졌다.

개선의 핵심은 단순히 모델을 로지스틱에서 머신러닝으로 바꾼 것이 아니라, 로지스틱 최종 분석에서 확인된 피처 구조를 비선형 모델이 더 잘 활용한 데 있다. M1/M2/M3 재비교에서는 M1 RandomForest가 AUPRC 0.2601로 기준선을 넘었지만 개선폭은 +0.0203에 그쳤다. 반면 Stage7 파생 피처에 토지피복을 더한 `PLUS_LANDCOVER`에서는 HistGradientBoosting이 +0.0586까지 개선했다. 따라서 2차는 M1/M2/M3 자체보다 로지스틱 최종 결과에서 얻은 피처 자산을 ML 모델이 더 잘 활용하도록 튜닝하는 방향이 맞다.

`PLUS_LANDCOVER_RULES_ANOVA_PROXY`도 좋은 후보였지만, 1차 최고 성능은 오히려 `PLUS_LANDCOVER`에서 나왔다. 이는 로지스틱에서는 명시적인 규칙/ANOVA 축약이 도움이 되었더라도, 트리 부스팅 모델에서는 Stage7+토지피복 원 피처만으로도 임계값과 상호작용을 직접 학습할 수 있음을 의미한다. 규칙/ANOVA proxy는 버릴 후보가 아니라 보조 후보로 유지하되, 2차의 1순위는 `PLUS_LANDCOVER / HistGradientBoosting`으로 두는 것이 타당하다.

모델별로는 HistGradientBoosting이 AUPRC 최상위이고, LightGBM은 ROC AUC와 best-F1/recall 측면에서 강하다. XGBoost는 모든 피처셋에서 실행됐고, 특히 `PLUS_LANDCOVER_RULES_ANOVA_PROXY`에서 AUPRC 0.2801, ROC AUC 0.8138, Brier 0.07333, log loss 0.24817로 확률 지표가 좋았다. 따라서 2차에서는 XGBoost를 제외하지 말고 규칙/ANOVA proxy 조합 중심으로 튜닝 후보에 포함한다.

운영점 관점에서는 fixed 0.50 threshold가 너무 보수적이다. precision은 0.7374로 높지만 recall이 0.0588에 그쳐 실제 양성의 대부분을 놓친다. best-F1 운영점은 threshold 0.1096에서 recall 0.5564, precision 0.2476이고, recall 0.70 운영점은 threshold 0.0743에서 recall 0.7005, precision 0.2216이다. 산불 위험 선별 목적이라면 0.50 고정값보다 best-F1 또는 recall 0.70 수준의 운영점이 더 현실적이다.

Top-risk capture는 모델을 위험도 순위화 도구로 사용할 수 있음을 보여준다. 개발셋 양성률은 9.1%인데, 상위 5% 위험군의 precision은 32.1%로 기본 양성률의 3.52배이다. 상위 10%만 보아도 전체 양성의 31.3%를 포착하고, 상위 20%에서는 54.3%를 포착한다. 즉 제한된 감시 자원을 우선순위가 높은 지역에 배분하는 용도에서는 실질적인 활용 가능성이 있다.

Hard-negative 결과에서는 `Target_0B1`, `Target_0B2` 구분 성능이 매우 높고 `Target_0A`가 여전히 가장 어렵다. 이는 모델이 쉬운 배경 음성은 잘 분리하지만, 실제 양성과 시간/공간적으로 가까운 0A 음성에서는 아직 혼동이 남아 있다는 뜻이다. 2차에서는 전체 AUPRC뿐 아니라 0A subgroup 성능을 같이 보면서, 단순 배경 분리보다 실제 발생 전조를 더 잘 잡는 방향으로 튜닝해야 한다.

Calibration은 1차 기준에서 최종 선택 기준으로 삼기 어렵다. Raw score가 AUPRC 기준으로 가장 좋았고, calibrated score는 일부 Brier/log loss를 개선하더라도 ranking 성능은 낮아졌다. 따라서 1차에서는 raw score로 모델을 고르고, calibration은 최종 후보가 좁혀진 뒤 3차에서 lockbox 평가와 함께 다시 판단한다.

2차 후보는 `PLUS_LANDCOVER / HistGradientBoosting`을 1순위로 두고, `PLUS_LANDCOVER / LightGBM`, `PLUS_LANDCOVER_RULES_ANOVA_PROXY / HistGradientBoosting`, `PLUS_LANDCOVER_RULES_ANOVA_PROXY / RandomForest`, `PLUS_LANDCOVER_RULES_ANOVA_PROXY / XGBoost`를 유지한다. M1 RandomForest는 진단용 기준선으로 남기되, 최종 성능 개선의 주축으로 보기는 어렵다.

### 산출물

- `outputs/ml_stage1_v2_summary.md`
- `outputs/metrics/ml_stage1_v2_all_model_comparison.csv`
- `outputs/metrics/ml_stage1_v2_model_rank_by_feature_set.csv`
- `outputs/predictions/ml_stage1_v2_oof_predictions.csv`

### 2차 계획 작성

- 1차 최고 후보 `PLUS_LANDCOVER / HistGradientBoosting / raw`의 AUPRC 0.2984를 2차 내부 기준선으로 설정했다.
- 2차는 모든 조합 전수 screening이 아니라 Optuna 기반 집중 튜닝 단계로 계획했다.
- 주 튜닝 후보는 `PLUS_LANDCOVER / HistGradientBoosting`, `PLUS_LANDCOVER / LightGBM`, `PLUS_LANDCOVER_RULES_ANOVA_PROXY / HistGradientBoosting`, `PLUS_LANDCOVER_RULES_ANOVA_PROXY / XGBoost`, `PLUS_LANDCOVER_RULES_ANOVA_PROXY / RandomForest`로 정했다.
- M1 RandomForest와 PLUS_LANDCOVER XGBoost는 보조 확인 후보로만 유지한다.
- 세부 계획은 `머신러닝_2차_진행_계획.md`에 작성했다.

### 2차 quick pilot 실행 결과

- 이번 실행은 full 2차 Optuna 튜닝이 아니라 파이프라인 검증용 quick pilot이다.
- 실행 설정: main 후보 1 trial per outer, auxiliary 후보 1 trial per outer, focused 추가 0, calibration skip.
- 따라서 이 결과는 1차 최고 모델을 대체하는 최종 2차 성능으로 해석하지 않는다.

- 전체 최고: `PLUS_LANDCOVER_RULES_ANOVA_PROXY / HistGradientBoosting / raw`
- AUPRC 0.2248, ROC AUC 0.7468, Brier 0.12384, log loss 0.38487
- 로지스틱 대비 ΔAUPRC -0.0151
- 1차 최고 대비 ΔAUPRC -0.0736

해석:

- quick pilot 최고 후보도 1차 최고 AUPRC 0.2984와 로지스틱 기준선 0.2398을 모두 넘지 못했다.
- 이번 결과는 trial 수 부족에 따른 임의 파라미터 평가에 가깝다.
- 1차 최고 후보 `PLUS_LANDCOVER / HistGradientBoosting / raw`는 유지한다.
- full 2차는 수정된 가벼운 탐색 공간으로 다시 실행해야 한다.

2차 후보별 raw AUPRC:

| feature_set                      | model                |   auprc |   auroc |   brier |   log_loss |   delta_auprc_vs_stage1_best |
|:---------------------------------|:---------------------|--------:|--------:|--------:|-----------:|-----------------------------:|
| PLUS_LANDCOVER_RULES_ANOVA_PROXY | HistGradientBoosting | 0.22477 | 0.74679 | 0.12384 |    0.38487 |                     -0.07363 |
| PLUS_LANDCOVER                   | XGBoost              | 0.20467 | 0.75318 | 0.10674 |    0.33772 |                     -0.09373 |
| PLUS_LANDCOVER_RULES_ANOVA_PROXY | XGBoost              | 0.20362 | 0.77431 | 0.10033 |    0.31493 |                     -0.09478 |
| PLUS_LANDCOVER                   | LightGBM             | 0.20087 | 0.72656 | 0.13788 |    0.42285 |                     -0.09752 |
| PLUS_LANDCOVER_RULES_ANOVA_PROXY | RandomForest         | 0.19874 | 0.77118 | 0.09418 |    0.30339 |                     -0.09966 |
| PLUS_LANDCOVER                   | HistGradientBoosting | 0.19108 | 0.72728 | 0.14087 |    0.41436 |                     -0.10732 |
| M1                               | RandomForest         | 0.16395 | 0.70727 | 0.09394 |    0.3149  |                     -0.13445 |

산출물:

- `outputs/ml_stage2_summary.md`
- `머신러닝_2차_진행_결과.md`
- `outputs/metrics/ml_stage2_model_comparison.csv`
- `outputs/predictions/ml_stage2_oof_predictions.csv`

### 3차 앙상블 계획 작성

- 2차 Optuna 결과는 1차 최고와 로지스틱 기준선을 모두 넘지 못했으므로 주 후보로 쓰지 않는다.
- 3차 앙상블의 주 재료는 1차 OOF 예측으로 정했다.
- 1차 최고 `PLUS_LANDCOVER / HistGradientBoosting / raw` AUPRC 0.2984를 앙상블 내부 기준선으로 둔다.
- Step3-A 후보 정리/상관 진단, Step3-B 단순 앙상블, Step3-C 제한적 weight optimization, Step3-D stacking, Step3-E calibration/운영점, Step3-F 최종 후보 선정 순서로 계획했다.
- 세부 계획은 `머신러닝_3차_앙상블_진행_계획.md`에 작성했다.

### 2차 실행 결과

- full 2차를 직접 실행했다.
- 실행 설정: main 후보 4 trials per outer, auxiliary 후보 2 trials per outer, focused 상위 2개 후보 추가 4 trials per outer, calibration top 3.
- 전체 최고: `PLUS_LANDCOVER_RULES_ANOVA_PROXY / HistGradientBoosting / raw`
- AUPRC 0.2200, ROC AUC 0.7467, Brier 0.13507, log loss 0.40192
- 로지스틱 대비 ΔAUPRC -0.0198
- 1차 최고 대비 ΔAUPRC -0.0784

결론:

- full 2차는 1차 최고 AUPRC 0.2984와 로지스틱 기준선 0.2398을 모두 넘지 못했다.
- 2차 Optuna 결과는 최종 후보로 채택하지 않는다.
- 현재 최고 모델은 1차 `PLUS_LANDCOVER / HistGradientBoosting / raw`로 유지한다.
- 다음 단계는 3차가 아니라 2차 재설계가 맞다.

2차 후보별 raw AUPRC:

| feature_set                      | model                |   auprc |   auroc |   brier |   log_loss |   delta_auprc_vs_stage1_best |
|:---------------------------------|:---------------------|--------:|--------:|--------:|-----------:|-----------------------------:|
| PLUS_LANDCOVER_RULES_ANOVA_PROXY | HistGradientBoosting | 0.22002 | 0.74671 | 0.13507 |    0.40192 |                     -0.07838 |
| PLUS_LANDCOVER                   | XGBoost              | 0.21807 | 0.7668  | 0.17281 |    0.49546 |                     -0.08033 |
| PLUS_LANDCOVER_RULES_ANOVA_PROXY | RandomForest         | 0.20734 | 0.77504 | 0.08357 |    0.27902 |                     -0.09106 |
| PLUS_LANDCOVER                   | LightGBM             | 0.20715 | 0.73542 | 0.14096 |    0.41848 |                     -0.09124 |
| PLUS_LANDCOVER                   | HistGradientBoosting | 0.20333 | 0.75453 | 0.12886 |    0.38441 |                     -0.09506 |
| PLUS_LANDCOVER_RULES_ANOVA_PROXY | XGBoost              | 0.20324 | 0.77137 | 0.09416 |    0.30153 |                     -0.09516 |
| M1                               | RandomForest         | 0.18303 | 0.74484 | 0.08128 |    0.27944 |                     -0.11536 |

산출물:

- `outputs/ml_stage2_summary.md`
- `머신러닝_2차_진행_결과.md`
- `outputs/metrics/ml_stage2_model_comparison.csv`
- `outputs/predictions/ml_stage2_oof_predictions.csv`

## 2026-06-21

### 3차 앙상블 실행

- 1차 OOF 예측을 주 재료로 단순 평균, 제한 weight search, stacking, calibration을 수행했다.
- 2차 Optuna 후보는 성능이 낮아 주 후보가 아니라 진단 후보로만 사용했다.
- lockbox test는 사용하지 않았다.

### 3차 최고 결과

- `ENSEMBLE_SIMPLE / simple_top5_rank_avg / raw`
- AUPRC 0.3068, ROC AUC 0.8187, Brier 0.27510, log loss 0.80025
- 1차 최고 대비 ΔAUPRC +0.0084, 로지스틱 대비 ΔAUPRC +0.0670
- 판단: 3차 앙상블 후보를 채택할 수 있다.
- 단, rank 평균은 확률 calibration 지표가 나쁘므로 보고서용 확률 후보는 `ENSEMBLE_SIMPLE / simple_top5_geometric_mean / raw`: AUPRC 0.3034, Brier 0.07328, log loss 0.24994, 1차 대비 ΔAUPRC +0.0050.

### 산출물

- `outputs/ml_stage3_ensemble_summary.md`
- `머신러닝_3차_앙상블_진행_결과.md`
- `outputs/metrics/ml_stage3_final_comparison_with_stage1_stage2_logistic.csv`
- `outputs/predictions/ml_stage3_ensemble_oof_predictions.csv`

## 2026-06-21

### 3.5차 모델 레벨 앙상블 실행

- 3차 OOF blending을 재학습 가능한 model-level ensemble 파이프라인으로 전환했다.
- 각 outer fold에서 base model 5개를 실제로 다시 학습하고 validation 예측을 결합했다.
- full development 재학습 model artifact도 저장했다. lockbox 평가는 아직 하지 않았다.

### 3.5차 결과

- AUPRC 최고: `ENSEMBLE_MODEL_LEVEL / model_level_top5_rank_average / raw`, AUPRC 0.3068, Δstage1 +0.0084
- 보고서용 후보: `ENSEMBLE_MODEL_LEVEL / model_level_top5_geometric_mean / raw`, AUPRC 0.3034, Brier 0.07328, log loss 0.24994

### 산출물

- `outputs/ml_stage35_model_level_ensemble_summary.md`
- `머신러닝_3_5차_모델레벨_앙상블_진행_결과.md`
- `outputs/metrics/ml_stage35_ensemble_model_comparison.csv`
- `outputs/models/stage35_model_level_ensemble/`

## 2026-06-21

### 4차 최종 후보 calibration 실행

- lockbox test 전 최종 ranking score와 probability score를 분리해 확정했다.
- raw/sigmoid_raw/sigmoid_logit/isotonic calibration을 development OOF 기준으로 비교했다.
- lockbox test는 사용하지 않았다.

### 4차 선택

- 순위용: `model_level_top5_rank_average / raw`, AUPRC 0.3068
- 확률/threshold용: `model_level_top5_geometric_mean / raw`, AUPRC 0.3034, Brier 0.07328, log loss 0.24994

### 산출물

- `outputs/ml_stage4_final_calibration_summary.md`
- `머신러닝_4차_최종후보_calibration_결과.md`
- `outputs/models/stage4_final_selection/stage4_final_selection_manifest.json`

## 2026-06-21

### 5차 lockbox test 최초 평가

- Step4에서 고정한 ranking/probability score를 lockbox test에 최초 적용했다.
- lockbox를 이용한 추가 후보 선택이나 튜닝은 하지 않았다.

### 5차 결과

- ranking score `model_level_top5_rank_average`: AUPRC 0.6173, ROC AUC 0.9088
- probability score `model_level_top5_geometric_mean`: AUPRC 0.6288, Brier 0.05972, log loss 0.20333

### 산출물

- `outputs/ml_stage5_lockbox_summary.md`
- `머신러닝_5차_lockbox_평가_결과.md`
- `outputs/metrics/ml_stage5_lockbox_model_comparison.csv`
- `outputs/predictions/ml_stage5_lockbox_final_predictions.csv`

## 2026-06-21

### 6차 최종보고서 정리

- Step1~5 산출물을 최종 보고서용 비교표와 해석 문서로 통합했다.
- 새 모델 학습, 추가 튜닝, lockbox 기반 후보 재선택은 하지 않았다.
- development OOF와 lockbox test를 분리해 표기했다.

### 6차 최종 선택

- 최종 보고서/운영 threshold 모델: `model_level_top5_geometric_mean / raw`
- development OOF: AUPRC 0.3034, ROC AUC 0.8178, Brier 0.07328, log loss 0.24994
- lockbox test: AUPRC 0.6288, ROC AUC 0.9105, Brier 0.05972, log loss 0.20333
- 운영 후보: `recall_ge_0.70` threshold 0.09195, lockbox precision 0.3974, recall 0.6977, selected rate 0.1600
- top 10% capture: lockbox 양성 180/311 포착, capture rate 0.5788, precision 0.5263

### 산출물

- `outputs/ml_stage6_final_report.md`
- `머신러닝_6차_최종보고서_정리_결과.md`
- `outputs/metrics/ml_stage6_final_model_comparison.csv`
- `outputs/metrics/ml_stage6_final_model_decision.csv`
- `outputs/metrics/ml_stage6_validation_checks.csv`

## 2026-06-21

### 7차 최종모델 해석/오류분석

- Step6 최종 모델 `model_level_top5_geometric_mean / raw`를 대상으로 score decile, threshold 오류, base model agreement, native feature importance를 산출했다.
- 새 모델 학습, 추가 튜닝, lockbox 기반 후보 재선택은 하지 않았다.
- Step6 권장 threshold `0.09195`를 그대로 사용했다.

### 7차 결과

- lockbox top 10% decile: 양성률 0.5279, lift 5.79, 누적 capture 0.5788
- lockbox top 30% 누적 capture: 0.8939
- threshold 0.09195 적용 lockbox: TP 217, FP 329, FN 94, TN 2773
- base model 5개가 모두 threshold 이상인 lockbox 샘플: 259개, 양성률 0.6139
- native feature importance 상위: `D1_DC`, `토지피복_L1_NAME`, `D1_FFMC_10일평균`, `D-1_평균습도_pct`, `log1p_도로거리_m`
- native feature importance는 LightGBM, RandomForest, XGBoost 3개 모델에서 산출했다. HistGradientBoosting 2개는 `feature_importances_`를 제공하지 않아 제외했다.

### 산출물

- `outputs/ml_stage7_final_model_diagnostics.md`
- `머신러닝_7차_최종모델_해석_결과.md`
- `outputs/metrics/ml_stage7_score_decile_analysis.csv`
- `outputs/metrics/ml_stage7_threshold_error_profile.csv`
- `outputs/metrics/ml_stage7_base_model_agreement.csv`
- `outputs/metrics/ml_stage7_native_feature_importance.csv`
- `outputs/metrics/ml_stage7_validation_checks.csv`

## 2026-06-21

### 8차 전체결과 정리/해석/분석

- 로지스틱 기준선부터 ML Step1~7까지의 결과를 하나의 종합 보고서로 정리했다.
- 단순 지표 나열이 아니라 단계별 의사결정, 최종 모델 선택 근거, 운영 해석, 오류분석, 중요 피처, 보고서 문구 주의점을 함께 정리했다.
- 새 모델 학습, 추가 튜닝, lockbox 기반 후보 재선택은 하지 않았다.

### 8차 종합 결론

- 최종 보고서/운영 threshold 모델: `model_level_top5_geometric_mean / raw`
- development OOF AUPRC: 0.3034
- 로지스틱 비교 기준선 AUPRC: 0.2398, Δ +0.0636
- 1차 단일 ML 최고 AUPRC: 0.2984, Δ +0.0050
- lockbox AUPRC: 0.6288, ROC AUC 0.9105, Brier 0.05972, log loss 0.20333
- 권장 운영점: `recall_ge_0.70`, threshold 0.09195
- lockbox top 10% capture: 0.5788
- lockbox top 30% decile capture: 0.8939

### 8차 해석 원칙

- lockbox 성능은 고정 후보의 최초 평가 결과로만 해석한다.
- lockbox 결과가 development OOF보다 높더라도 일반화 성능이 OOF보다 높다고 단정하지 않는다.
- native feature importance는 인과 효과가 아니라 모델 내부 분할 중요도로 해석한다.

### 산출물

- `outputs/ml_stage8_comprehensive_analysis.md`
- `머신러닝_8차_전체결과_정리_해석_분석.md`
- `outputs/metrics/ml_stage8_stage_timeline.csv`
- `outputs/metrics/ml_stage8_key_metric_summary.csv`
- `outputs/metrics/ml_stage8_report_claims.csv`
- `outputs/metrics/ml_stage8_validation_checks.csv`
