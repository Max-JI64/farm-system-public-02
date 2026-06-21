# 머신러닝 4차 최종 후보 calibration 진행 계획

## 1. 목적

4차의 목적은 lockbox test를 열기 전에 development OOF 기준으로 최종 후보의 score 사용 방식을 확정하는 것이다.

3.5차에서 model-level ensemble 파이프라인을 만들었고, 다음 두 후보가 남았다.

| 용도 | 후보 | 이유 |
|---|---|---|
| 순위 성능용 | `model_level_top5_rank_average` | AUPRC 0.3068로 최고 |
| 보고서/운영 확률용 | `model_level_top5_geometric_mean` | AUPRC 0.3034, Brier/log loss 양호 |

4차에서는 이 둘을 중심으로 raw/sigmoid/isotonic calibration을 비교하고, 최종 보고서와 lockbox 평가에서 어떤 score와 threshold를 사용할지 고정한다.

## 2. 중요한 원칙

- lockbox test는 열지 않는다.
- 새로운 base model 튜닝은 하지 않는다.
- 3.5차에서 저장한 model-level ensemble OOF prediction을 사용한다.
- calibration 성능은 outer-fold OOF 방식으로 평가한다.
- `rank_average`는 확률값이 아니라 순위화 score이므로 Brier/log loss를 확률 품질로 해석하지 않는다.

## 3. 비교할 후보

### 3-1. ranking 후보

- `model_level_top5_rank_average`
- 비교 score: raw, sigmoid, isotonic
- 최종 판단 기준: AUPRC, ROC AUC, top 5/10/20% capture, 0A AUPRC

### 3-2. probability 후보

- `model_level_top5_geometric_mean`
- `model_level_top5_average`
- `model_level_top5_perf_weighted`
- `model_level_top5_logit_average`

최종 판단 기준:

1. AUPRC가 1차 최고 0.2984를 넘는지
2. Brier/log loss가 1차 최고와 비슷하거나 개선되는지
3. top-risk capture가 유지되는지
4. 0A hard-negative AUPRC가 악화되지 않는지
5. calibration으로 AUPRC가 크게 낮아지지 않는지

## 4. calibration 방식

각 outer fold마다 다음 방식으로 calibration OOF score를 만든다.

1. outer train row의 ensemble raw score와 Target으로 calibrator를 학습한다.
2. outer validation row의 ensemble raw score에 calibrator를 적용한다.
3. 모든 fold를 합쳐 OOF calibrated score를 평가한다.

비교 방법:

- raw
- sigmoid on raw score
- sigmoid on logit score
- isotonic

## 5. threshold 확정

최종 확률 후보에 대해 다음 운영점을 저장한다.

- fixed 0.50
- best-F1
- recall >= 0.50
- recall >= 0.70
- recall >= 0.90
- top 5%, 10%, 20%

보고서에는 fixed 0.50보다 best-F1, recall 0.70, top-risk capture를 중심으로 제시한다.

## 6. lockbox-ready manifest

Step4는 lockbox 평가를 하지 않지만, Step5에서 바로 적용할 수 있도록 다음을 고정한다.

- base model artifact 위치
- 최종 ranking ensemble recipe
- 최종 probability ensemble recipe
- 선택한 calibration method
- OOF 기준 threshold
- 산출 metric 기준선

## 7. 산출물

- `stage4_final_calibration.py`
- `outputs/metrics/ml_stage4_calibration_comparison.csv`
- `outputs/metrics/ml_stage4_final_candidate_selection.csv`
- `outputs/metrics/ml_stage4_final_thresholds.csv`
- `outputs/metrics/ml_stage4_final_top_risk_capture.csv`
- `outputs/metrics/ml_stage4_final_subgroup_metrics.csv`
- `outputs/metrics/ml_stage4_validation_checks.csv`
- `outputs/predictions/ml_stage4_final_candidate_oof_predictions.csv`
- `outputs/models/stage4_final_selection/`
- `outputs/ml_stage4_final_calibration_summary.md`
- `머신러닝_4차_최종후보_calibration_결과.md`

## 8. 성공 기준

- ranking 후보는 AUPRC 0.3068 수준을 유지한다.
- probability 후보는 AUPRC 0.3034 이상 또는 최소 1차 최고 0.2984 이상을 유지한다.
- probability 후보의 Brier/log loss가 1차 최고보다 나쁘지 않거나 비슷해야 한다.
- lockbox row가 calibration 학습/평가에 들어가지 않는다.
- 최종 manifest만 만들고 lockbox 평가는 Step5로 넘긴다.
