# 머신러닝 5차 lockbox test 평가 계획

## 1. 목적

5차의 목적은 지금까지 보존해온 lockbox test를 처음으로 열고, Step4에서 고정한 최종 후보가 development OOF 밖에서도 성능을 유지하는지 확인하는 것이다.

Step5에서는 새 후보를 고르지 않는다. Step4에서 이미 고정한 다음 두 score만 평가한다.

| 역할 | 후보 | score_type | 용도 |
|---|---|---|---|
| ranking score | `model_level_top5_rank_average` | raw | 위험도 순위화, top-risk capture |
| probability score | `model_level_top5_geometric_mean` | raw | 확률/threshold 운영, 보고서용 최종 후보 |

## 2. 입력

- Step4 final manifest: `outputs/models/stage4_final_selection/stage4_final_selection_manifest.json`
- Step4 final selection: `outputs/metrics/ml_stage4_final_candidate_selection.csv`
- Step3.5 full-development base model artifacts: `outputs/models/stage35_model_level_ensemble/*.joblib`
- Step3.5 full-development base score reference: `outputs/predictions/ml_stage35_full_development_base_scores.csv`
- lockbox manifest: `jsw/Analysis/logistic/outputs/splits/lockbox_manifest.csv`
- 원 데이터: `data/학습데이터/학습데이터_로지스틱_D2D3.csv`
- engineered feature: `jsw/Analysis/logistic/outputs/features/stage7_engineered_features.csv`

## 3. 평가 방식

1. 전체 원 데이터를 로드하고 Stage7 engineered feature를 `샘플ID` 기준으로 결합한다.
2. Stage9 규칙 피처를 동일하게 재생성한다.
3. lockbox row만 분리한다.
4. Step3.5에서 full development로 학습해 저장한 base model 5개를 로드한다.
5. lockbox row에 대해 base model별 score를 생성한다.
6. Step4에서 고정한 ensemble recipe를 적용한다.
7. lockbox Target으로 최종 성능을 평가한다.

## 4. rank average 적용 방식

`model_level_top5_rank_average`는 확률값이 아니라 순위화 score이다. lockbox 내부 분포만으로 rank를 다시 매기면 test set 분포를 사용한 transductive score가 될 수 있으므로, Step5에서는 development full-model score 분포를 reference로 사용해 percentile rank를 계산한다.

즉, lockbox score는 다음 방식으로 만든다.

1. full development base score별 empirical CDF를 만든다.
2. lockbox base score를 development reference CDF에 넣어 percentile rank로 변환한다.
3. 5개 base model percentile rank를 평균한다.

## 5. 평가 지표

로지스틱/ML OOF와 동일한 지표를 산출한다.

- AUPRC
- ROC AUC
- Brier
- log loss
- best-F1
- Step4에서 고정한 threshold 운영점
  - fixed 0.50
  - best-F1
  - recall >= 0.50
  - recall >= 0.70
  - recall >= 0.90
- top 5/10/20% capture
- hard-negative subgroup: 0A/0B1/0B2
- 기후지형유형 subgroup

## 6. 해석 기준

Step5는 최종 일반화 성능 확인 단계이다. development OOF보다 성능이 낮아지는 것은 자연스럽지만, 다음을 확인한다.

| 기준 | 해석 |
|---|---|
| lockbox AUPRC가 로지스틱 OOF 기준선 0.2398 이상 | 최소 성공 |
| lockbox AUPRC가 1차 ML OOF 최고 0.2984 근처 | 강한 일반화 |
| top 10/20% capture 유지 | 위험도 우선순위 활용 가능 |
| Brier/log loss 급격 악화 없음 | 확률 후보 안정성 |
| 0A hard-negative 급락 없음 | 실제 어려운 음성군에서 유지 |

## 7. 산출물

- `stage5_lockbox_evaluation.py`
- `outputs/metrics/ml_stage5_lockbox_model_comparison.csv`
- `outputs/metrics/ml_stage5_oof_vs_lockbox_comparison.csv`
- `outputs/metrics/ml_stage5_lockbox_fixed_thresholds.csv`
- `outputs/metrics/ml_stage5_lockbox_top_risk_capture.csv`
- `outputs/metrics/ml_stage5_lockbox_subgroup_metrics.csv`
- `outputs/metrics/ml_stage5_lockbox_base_model_metrics.csv`
- `outputs/metrics/ml_stage5_validation_checks.csv`
- `outputs/predictions/ml_stage5_lockbox_base_model_predictions.csv`
- `outputs/predictions/ml_stage5_lockbox_final_predictions.csv`
- `outputs/ml_stage5_lockbox_summary.md`
- `머신러닝_5차_lockbox_평가_결과.md`

## 8. 테스트 계획

- 전체 데이터 17,045행 유지
- development 13,632행, lockbox 3,413행 확인
- lockbox prediction이 lockbox row만 포함하는지 확인
- full-development 학습 artifact만 사용하고 lockbox로 모델을 재학습하지 않는지 확인
- base/ensemble score NaN/inf 없음
- score 범위 `[0, 1]`
- Step4 threshold가 그대로 적용되는지 확인
- lockbox 결과를 기준으로 새 후보 선택을 하지 않는다.
