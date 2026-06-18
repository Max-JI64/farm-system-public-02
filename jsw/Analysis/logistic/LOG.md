# 로지스틱 모델링 진행 로그

## 2026-06-18

### 작업 목표

- 단계별 전체 진행 순서를 모델링 계획 문서에 반영한다.
- `26.06.18_로지스틱_모델링.ipynb`에 1~4단계를 구현한다.
- 실제 Python 3.13 환경에서 노트북을 실행해 산출물과 분할 누수 여부를 검증한다.

### 계획 문서 갱신

완료:

- `로지스틱_모델링_계획.md` 상단에 1~10단계 진행 순서를 추가했다.
- 1~4단계는 데이터 준비와 검증 구조 고정, 5단계부터 모델 학습으로 구분했다.

### 1단계. 캐나다 지수 시간 누수 수정

상태: 완료

기존 문제:

- `학습데이터_최종_캐나다지수.csv`는 같은 날 12시 캐나다 지수를 날짜 기준으로 병합했다.
- 기준시각 12:00 이전 행 7,777건과 그중 Target 1 693건에서 미래 정보가 될 수 있다.

적용 정책:

- 기존 같은 날 캐나다 지수 9개를 모델 입력에서 제거했다.
- 모든 표본에 D-1 정오 `FFMC`, `FFMC_10일평균`, `DMC`, `DC`, `ISI`, `BUI`, `FWI`를 재병합했다.
- `Indexed_FFMC`, `FFMC_논문식_발생확률`도 D-1 benchmark용으로 별도 보존했다.

검증 결과:

- 행 수: 17,045
- Target 1: 1,553
- Target 0: 15,492
- D-1 지수 결측: 0
- 모든 지수 기준 날짜가 기준 날짜보다 정확히 1일 이전
- FFMC 선행연구 산출물과 FWI 산출물의 FFMC 최대 차이: 1e-10 미만

### 2단계. 그룹 ID 복원

상태: 완료

처리:

- `샘플ID`로 `학습데이터_병합.csv`의 `fire_id`, `원본_fire_id`, `후보점ID`를 복원했다.
- Target 1과 Target 0-A는 `source_fire_id`를 이용해 같은 `FIRE_*` 그룹으로 묶었다.
- Target 0-B1/0-B2는 `후보점ID`별 `BG_*` 그룹으로 만들었다.

검증 결과:

- 전체 모델링 그룹: 9,307개
- 산불 source 그룹: 1,553개
- 공간 배경 그룹: 7,754개
- 그룹 ID 결측: 0건
- 산불 그룹당 Target 1이 1건이 아닌 그룹: 0개
- 공간 배경 그룹 중복: 0건

### 3단계. M1/M2/M3 변수 확정

상태: 완료

확정한 모델 세트:

| 모델 | 변수 구성 | 변수 수 |
|---|---|---:|
| M1 | 날씨·공간 + FA + D-1 캐나다 지수 | 53 |
| M2 | 날씨·공간 + D-1 캐나다 지수 | 48 |
| M3 | 날씨·공간 | 41 |

추가 처리:

- 도로·시가화·농업·임도·등산로·산림 거리의 `log1p` 변수를 생성했다.
- 월과 시간대의 sin/cos 순환형 변수를 생성했다.
- `기후지형유형`은 후속 모델 pipeline 내부 one-hot 대상 범주형으로 유지했다.
- ID, 좌표, Target, 샘플유형, 공간층, 시간샘플링방식 등 누수·식별자 변수는 feature manifest에서 제외했다.

저장:

- `data/학습데이터/학습데이터_로지스틱_D1.csv`
- `jsw/Analysis/logistic/outputs/manifests/feature_manifest.json`

### 4단계. 그룹 기반 검증 분할 고정

상태: 완료

분할 정책:

- 그룹 기준 5개 후보 중 크기 20%와 전체 양성률에 가장 가까운 후보 fold 2를 lockbox로 선택했다.
- development 데이터에 grouped outer 5-fold를 배정했다.
- 각 outer training 구간 안에 grouped inner 4-fold를 배정했다.

Lockbox 결과:

| 구분 | 행 수 | Target 1 | 양성률 | 그룹 수 |
|---|---:|---:|---:|---:|
| development | 13,632 | 1,242 | 9.1109% | 7,444 |
| lockbox test | 3,413 | 311 | 9.1122% | 1,863 |

Outer fold 결과:

| outer fold | 행 수 | Target 1 | 양성률 | 그룹 수 |
|---:|---:|---:|---:|---:|
| 0 | 2,728 | 248 | 9.0909% | 1,489 |
| 1 | 2,723 | 248 | 9.1076% | 1,489 |
| 2 | 2,725 | 248 | 9.1009% | 1,488 |
| 3 | 2,726 | 249 | 9.1343% | 1,488 |
| 4 | 2,730 | 249 | 9.1209% | 1,490 |

분할 검증:

- development와 lockbox 그룹 중복: 0
- outer fold 간 그룹 분할 위반: 0
- inner fold 간 그룹 분할 위반: 0
- inner manifest 행 수: 54,528
- outer×inner 조합: 20개

저장:

- `jsw/Analysis/logistic/outputs/splits/lockbox_manifest.csv`
- `jsw/Analysis/logistic/outputs/splits/outer_cv_manifest.csv`
- `jsw/Analysis/logistic/outputs/splits/inner_cv_manifest.csv`
- `jsw/Analysis/logistic/outputs/splits/lockbox_candidate_summary.csv`

### 노트북 실행 검증

상태: 완료

- `26.06.18_로지스틱_모델링.ipynb`의 모든 셀을 Python 3.13.1 환경에서 실행했다.
- 실행 오류 없이 완료했고 결과 출력도 노트북에 저장했다.
- `git diff --check` 오류 없음.

### 다음 작업

상태: 5단계 완료

## 2026-06-18 — 5단계 기준 모델 학습

### 수행 내용

상태: 완료

- lockbox test 3,413행은 사용하지 않았다.
- development 13,632행에 대해 outer 5-fold OOF 예측을 생성했다.
- Dummy classifier와 M1/M2/M3 L2 로지스틱을 동일한 fold에서 비교했다.
- M1/M2/M3는 각 outer training 안의 inner 4-fold AUPRC로 `C ∈ {0.01, 0.1, 1, 10}`을 선택했다.
- 현재 기준 모델은 `class_weight=None`이며, L1·Elastic Net·class weight 비교는 6단계로 남겼다.

### 전체 development OOF 성능

| 모델 | AUPRC | AUROC | Brier | Log loss | Calibration intercept | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0.0910 | 0.4995 | 0.08281 | 0.30510 | -0.3657 | 0.8412 |
| M1 원본+FA+캐나다 | 0.3335 | 0.7877 | 0.07200 | 0.25624 | -0.0611 | 0.9676 |
| M2 원본+캐나다 | 0.3335 | 0.7877 | 0.07200 | 0.25623 | -0.0601 | 0.9682 |
| M3 원본 | 0.2765 | 0.7661 | 0.07494 | 0.26562 | -0.0712 | 0.9631 |

핵심 비교:

- M1-M2 전체 AUPRC 차이: `-0.000036`
- M1-M2 outer-fold paired AUPRC 차이: 평균 `+0.00026`, SD `0.00136`, 5개 fold 중 M1 우세 3개
- M2-M3 전체 AUPRC 차이: `+0.05704`
- M2-M3 outer-fold paired AUPRC 차이: 평균 `+0.05618`, SD `0.01548`, 범위 `+0.03453~+0.07526`
- M2-M3 Brier 차이: 평균 `-0.00294`로 M2가 더 낮음

현재 판단:

- 원본 변수와 D-1 캐나다 지수가 있는 상태에서 FA 5개를 추가한 M1은 M2보다 좋아지지 않았다.
- 이번 L2 baseline만 보면 FA의 추가 예측력은 사실상 0이다.
- D-1 캐나다 지수를 추가한 M2는 M3보다 순위 성능과 확률 오차가 모두 개선됐다.
- FA 제거 최종 결정은 6단계 L1/Elastic Net과 이후 로지스틱 계열 확장까지 확인한 뒤 내린다.

### Outer fold 안정성

| 모델 | AUPRC 평균 ± SD | AUROC 평균 ± SD | Brier 평균 ± SD |
|---|---:|---:|---:|
| M1 | 0.3378 ± 0.0142 | 0.7881 ± 0.0084 | 0.07200 ± 0.00086 |
| M2 | 0.3375 ± 0.0136 | 0.7881 ± 0.0082 | 0.07200 ± 0.00086 |
| M3 | 0.2813 ± 0.0060 | 0.7664 ± 0.0049 | 0.07494 ± 0.00033 |

- train-validation AUPRC gap은 대부분 0 부근이었다.
- 가장 큰 gap은 약 0.033으로, 전체적으로 심한 과적합 신호는 아직 없다.
- M1과 M2는 fold별 성능 곡선도 거의 겹친다.

### 대조군 유형별 OOF 성능

| 모델 | Target 0-A AUPRC | Target 0-B1 AUPRC | Target 0-B2 AUPRC |
|---|---:|---:|---:|
| M1 | 0.3630 | 0.7469 | 0.8601 |
| M2 | 0.3629 | 0.7470 | 0.8603 |
| M3 | 0.3033 | 0.7239 | 0.8608 |

해석:

- 가장 어려운 구분은 같은 실제 산불 위치의 다른 시각을 비교하는 0-A이다.
- M2는 M3보다 0-A AUPRC가 약 `+0.0596` 높아 D-1 캐나다 지수가 시간적 기상 구분에 추가 정보를 준다.
- 0-B2는 세 모델 모두 약 0.86으로 높아 쉬운 전역 배경 대조군의 영향이 크다.
- 최종 모델 선택에서는 전체 성능보다 0-A와 0-B1 성능을 반드시 함께 본다.

### 기후지형유형별 M2 성능

| 기후지형유형 | AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| 고지·산간형 | 0.2791 | 0.7907 | 0.07381 |
| 영동 해안형 | 0.4368 | 0.8216 | 0.06726 |
| 영서 내륙형 | 0.2631 | 0.7582 | 0.07502 |

- 영동 해안형의 구분 성능이 가장 높다.
- 영서 내륙형은 AUPRC와 AUROC가 가장 낮아 후속 상호작용·건조도 변수 보강 우선 대상이다.

### 임계값 결과

M2 기준:

- 고정 threshold 0.5: precision 0.7304, recall 0.0676, F1 0.1238
- OOF F1 최적 threshold 0.2258: precision 0.3440, recall 0.3720, F1 0.3574
- OOF Recall 90% threshold 0.0424: precision 0.1430, recall 0.9002

해석:

- 양성 비율이 9.11%이므로 threshold 0.5는 지나치게 보수적이며 대부분의 산불을 놓친다.
- 최종 threshold는 운영 목적과 lockbox 이전 validation 정책으로 별도 고정해야 한다.
- OOF에서 찾은 threshold는 탐색값이며 lockbox 성능으로 직접 보고하지 않는다.

### 규제와 계수 진단

- M1/M2/M3 모두 5개 outer fold에서 `C=10`이 선택됐다.
- 현재 탐색 범위의 상한에 걸렸으므로 6단계에서는 `C=100` 이상까지 확장해야 한다.
- M1 상위 계수에서 `D1_FWI`는 큰 양수, `D1_BUI`와 `D1_ISI`는 큰 음수로 동시에 나타났다.
- 중복 습도·풍속 변수도 서로 반대 부호가 나타났다.
- 이는 FWI 구성지수와 원변수의 강한 다중공선성 신호이다.
- 따라서 5단계 L2 계수는 개별 오즈비로 해석하지 않고 방향·안정성 진단용으로만 사용한다.

### 생성 산출물

상세 표:

- `outputs/metrics/stage5_overall_metrics.csv`
- `outputs/metrics/stage5_outer_fold_metrics.csv`
- `outputs/metrics/stage5_fold_summary.csv`
- `outputs/metrics/stage5_threshold_metrics.csv`
- `outputs/metrics/stage5_negative_type_metrics.csv`
- `outputs/metrics/stage5_climate_metrics.csv`
- `outputs/metrics/stage5_inner_c_tuning.csv`
- `outputs/metrics/stage5_selected_c.csv`
- `outputs/metrics/stage5_generalization_gap.csv`
- `outputs/metrics/stage5_paired_model_differences.csv`

예측·계수:

- `outputs/predictions/stage5_oof_predictions.csv`
- `outputs/coefficients/stage5_fold_coefficients.csv`
- `outputs/coefficients/stage5_coefficient_summary.csv`

플롯:

- `outputs/plots/stage5_01_pr_roc_curves.png`
- `outputs/plots/stage5_02_calibration.png`
- `outputs/plots/stage5_03_fold_stability.png`
- `outputs/plots/stage5_04_negative_type_auprc.png`
- `outputs/plots/stage5_05_confusion_matrices.png`
- `outputs/plots/stage5_06_probability_distribution.png`
- `outputs/plots/stage5_07_m1_top_coefficients.png`

자동 요약:

- `outputs/stage5_result_summary.md`

### 다음 작업

상태: 대기

6단계에서 수행:

1. `C` 탐색 범위를 확장한 L2 재검증
2. L1 및 Elastic Net 비교
3. `class_weight=None`과 `balanced` 비교
4. 변수 선택 빈도와 계수 부호 안정성 산출

## 2026-06-18 — 6단계 정규화·캐나다 지수 축소

### 실행 상태

상태: 완료

- 실행 시작: 약 16:53
- 실행 완료: 약 17:15
- 소요 시간: 약 22분
- 노트북 코드 셀: 17개 실행
- 실행 오류: 0
- lockbox test 사용: 없음

오래 걸린 이유:

- `L1`, `L2`, `Elastic Net`과 class weight 2개 조합을 outer 5-fold × inner 4-fold로 반복 검증했다.
- Elastic Net은 여러 `C`와 `l1_ratio` 조합을 `saga` solver로 반복 적합했다.
- 이후 캐나다 지수 5개 조합을 다시 nested CV로 검증했다.

### FA 제외

5단계 결과에 따라 6단계부터 FA 요인점수는 사용하지 않았다.

- M1과 M2의 성능 차이가 사실상 0이었다.
- 이후 주 모델은 `원본 날씨·공간 + D-1 캐나다 지수`를 기준으로 한다.
- FA 결과는 “추가 예측력 없음”을 보여주는 비교 근거로만 보존한다.

### 정규화·class weight 비교

| 후보 | AUPRC | AUROC | Brier | Calibration slope | 0-A AUPRC |
|---|---:|---:|---:|---:|---:|
| L2, weight 없음 | 0.3339 | 0.7852 | 0.07202 | 0.9625 | 0.3628 |
| Elastic Net, weight 없음 | 0.3325 | 0.7874 | 0.07203 | 0.9736 | 0.3616 |
| L1, weight 없음 | 0.3324 | 0.7874 | 0.07204 | 0.9741 | 0.3614 |
| L1, balanced | 0.2785 | 0.7902 | 0.19149 | 0.9222 | 0.3070 |
| Elastic Net, balanced | 0.2785 | 0.7902 | 0.19149 | 0.9214 | 0.3069 |
| L2, balanced | 0.2781 | 0.7894 | 0.19155 | 0.9182 | 0.3054 |

판단:

- 주 지표 AUPRC와 Brier 기준 승자는 `L2 + class_weight=None`이다.
- balanced weight는 AUROC를 약간 높였지만 AUPRC와 0-A AUPRC를 크게 낮췄고 확률 보정을 심각하게 악화시켰다.
- 따라서 현재 샘플링 구조에서는 `class_weight="balanced"`를 사용하지 않는다.
- L1과 Elastic Net은 L2와 거의 같은 성능이지만 더 좋지는 않았다.

### L1·Elastic Net 변수 축소 여부

- `L1_none`: 49개 변수가 모든 outer fold에서 선택됨
- `ElasticNet_none`: 49개 변수가 모든 outer fold에서 선택됨
- `L1_balanced`: 49개 중 47개만 모든 fold에서 선택됐으나 평균 선택률은 0.992

판단:

- 선택된 `C`가 대부분 10~1000으로 커서 규제가 약했다.
- L1과 Elastic Net이 실질적인 희소 변수 선택을 만들지 못했다.
- 성능을 유지하면서 자동으로 변수를 제거하는 효과는 이번 단계에서 확인되지 않았다.

### 선택된 규제 강도

`L2_none`의 outer-fold별 선택:

- fold 0: C=100
- fold 1: C=100
- fold 2: C=1000
- fold 3: C=1000
- fold 4: C=1000

해석:

- 5단계의 C=10 상한 문제는 해결됐다.
- 최적점은 강한 규제보다 약한 L2 규제 쪽에 있다.
- 다만 C=100과 1000의 실질 성능 차이가 작으므로 최종 모델에서는 단순성과 수치 안정성을 고려해 C 선택 규칙을 보수적으로 고정할 필요가 있다.

### 캐나다 지수 조합 축소

정규화 승자인 `L2 + class_weight=None`으로 비교했다.

| 캐나다 지수 조합 | AUPRC | AUROC | Brier | 0-A AUPRC |
|---|---:|---:|---:|---:|
| 전체 7개 지수 | 0.3339 | 0.7852 | 0.07202 | 0.3628 |
| FWI 단독 | 0.2773 | 0.7714 | 0.07471 | 0.3042 |
| 캐나다 미사용 | 0.2763 | 0.7647 | 0.07493 | 0.3034 |
| FFMC·ISI·BUI | 0.2751 | 0.7729 | 0.07459 | 0.3022 |
| FFMC·DMC·DC | 0.2733 | 0.7763 | 0.07459 | 0.3006 |

판단:

- 전체 7개 지수를 함께 사용한 경우에만 성능 향상이 크게 유지됐다.
- FWI 단독이나 일부 구성지수만으로는 캐나다 지수 미사용 모델과 거의 비슷했다.
- 전체 지수는 캐나다 미사용 대비 전체 AUPRC `+0.0576`, 0-A AUPRC `+0.0594`를 개선했다.
- 예측 성능 기준으로는 현재 `CAN_ALL`을 유지한다.

### 계수 해석 주의

`CAN_ALL`에서 주요 계수 방향은 outer fold 간 대체로 안정적이었지만 다음 충돌이 유지됐다.

- `D1_FWI`: 큰 양수
- `D1_ISI`, `D1_BUI`: 큰 음수
- `D1_DC`, `D1_FFMC`: 양수
- `D1_DMC`, `D1_FFMC_10일평균`: 음수
- 24시간/48시간 습도와 풍속 변수도 서로 다른 부호가 나타남

이는 구성지수와 원본 변수 간 다중공선성 및 억제효과에 가깝다.

- 49개 변수 중 8개는 outer fold에서 계수 부호가 바뀌었다.
- 현재 전체 모델 계수는 개별 인과효과나 최종 오즈비로 사용하지 않는다.
- 오즈비 분석은 상관 블록을 정리한 별도의 축소 모델로 수행해야 한다.

### 6단계 최종 후보

- FA: 제외
- 변수 구성: 원본 날씨·공간 + D-1 캐나다 전체 7개 지수
- 정규화: L2
- class weight: 없음
- lockbox: 계속 미사용

### 생성 산출물

요약:

- `outputs/stage6_result_summary.md`

정규화 비교:

- `outputs/metrics/stage6_regularization_overall_metrics.csv`
- `outputs/metrics/stage6_regularization_negative_type_metrics.csv`
- `outputs/metrics/stage6_regularization_fold_metrics.csv`
- `outputs/metrics/stage6_regularization_fold_summary.csv`
- `outputs/metrics/stage6_regularization_inner_tuning.csv`
- `outputs/metrics/stage6_regularization_selected_params.csv`
- `outputs/metrics/stage6_regularization_threshold_metrics.csv`

캐나다 지수 축소:

- `outputs/metrics/stage6_canadian_subset_overall_metrics.csv`
- `outputs/metrics/stage6_canadian_subset_negative_type_metrics.csv`
- `outputs/metrics/stage6_canadian_subset_fold_metrics.csv`
- `outputs/metrics/stage6_canadian_subset_fold_summary.csv`
- `outputs/metrics/stage6_canadian_subset_inner_tuning.csv`
- `outputs/metrics/stage6_canadian_subset_selected_params.csv`
- `outputs/metrics/stage6_canadian_subset_threshold_metrics.csv`

플롯:

- `outputs/plots/stage6_01_regularization_auprc.png`
- `outputs/plots/stage6_02_regularization_calibration.png`
- `outputs/plots/stage6_03_selected_hyperparameters.png`
- `outputs/plots/stage6_04_canadian_subsets_auprc.png`
- `outputs/plots/stage6_05_canadian_subsets_calibration.png`
- `outputs/plots/stage6_06_best_subset_coefficients.png`
- `outputs/plots/stage6_07_best_subset_selection_rate.png`

### 다음 작업

상태: 대기

7단계에서 수행:

1. 상관 블록 기반 변수 축소
2. EDA 사전가설 상호작용 추가
3. 특히 0-A 성능 개선 여부 확인
4. 예측 모델과 별도로 해석 가능한 오즈비 모델 후보 구성

## 2026-06-18 — 7단계 EDA 피처 확장

### 실행 상태

상태: 완료

- 실행 시간: 약 68초
- lockbox test 사용: 없음
- 정규화: 6단계 승자인 L2, class weight 없음
- 캐나다 지수: 전체 7개 유지
- FA 요인점수: 미사용

### 추가한 피처

연속형:

- `rh_minus_local_q05`
- `wind_mean_6h`
- `wind_max_6h`
- `dry_spell_h_0p1`
- `dry_spell_h_5p0`

EDA 임계치 플래그:

- `rh_local_q05`
- `wind_max_6h_ge_5`
- `dry_spell_0p1_gt_24h`
- `dry_spell_5p0_gt_240h`
- `rh_local_q05_AND_ffmc_ge_90`
- `rh_local_q05_AND_isi_ge_10`
- `rh_local_q05_AND_wind_max_6h_ge_5`
- `rh_local_q05_AND_westerly_strong_max_6h`

권역·복합 상호작용:

- 영동×6시간 평균·최대풍속
- 영동×6시간 최대풍속 시점 서풍 여부
- 영동×국지 하위 5% 저습
- 영서·고지산간×24시간 건조도
- 24시간 건조도×6시간 최대풍속

### 누수 방지 계산 정책

- 6시간 풍속은 `[기준시각-6h, 기준시각)`만 사용했다.
- 무강수 지속시간은 기준시각 이전 마지막 시간강수를 기준으로 계산했다.
- FFMC/ISI 결합 조건은 D-1 정오 지수를 사용했다.
- 국지 하위 5% 기준은 라벨을 사용하지 않은 `기상셀×월×시간대` 고정 기후분포로 계산했다.

### 피처 세트별 결과

| 피처 세트 | 전체 AUPRC | 0-A AUPRC | AUROC | Brier | BASE 대비 전체 Δ |
|---|---:|---:|---:|---:|---:|
| BASE_CAN_ALL | 0.3339 | 0.3628 | 0.7852 | 0.07202 | 0 |
| PLUS_CONTINUOUS | 0.3514 | 0.3816 | 0.7895 | 0.07115 | +0.0175 |
| PLUS_FLAGS | 0.3646 | 0.3986 | 0.7939 | 0.07031 | +0.0307 |
| PLUS_CONTINUOUS_FLAGS | 0.3751 | 0.4088 | 0.7953 | 0.06982 | +0.0412 |
| PLUS_ALL_INTERACTIONS | 0.3759 | 0.4103 | 0.7983 | 0.06967 | +0.0420 |

### 최종 추천 세트

수치상 1위:

- `PLUS_ALL_INTERACTIONS`
- 전체 AUPRC 0.3759
- 0-A AUPRC 0.4103

복잡도 반영 추천:

- `PLUS_CONTINUOUS_FLAGS`
- 전체 AUPRC 0.3751
- 0-A AUPRC 0.4088

추천 이유:

- 전체 상호작용을 추가한 이득은 전체 AUPRC `+0.0008`, 0-A AUPRC `+0.0015`에 불과하다.
- outer fold에서 `PLUS_ALL_INTERACTIONS`가 `PLUS_CONTINUOUS_FLAGS`보다 전체 AUPRC가 높았던 fold는 5개 중 3개뿐이다.
- 상호작용 추가 효과는 사전 기준 0.005보다 작아 더 단순한 연속형+플래그 세트를 후속 주 모델로 선택한다.

### 대조군 유형별 추천 세트 성능

`PLUS_CONTINUOUS_FLAGS`:

| 대조군 | AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| Target 0-A | 0.4088 | 0.6960 | 0.12412 |
| Target 0-B1 | 0.7568 | 0.8827 | 0.16230 |
| Target 0-B2 | 0.8472 | 0.9118 | 0.21300 |

- 가장 어려운 0-A AUPRC가 기준 모델보다 `+0.0460` 개선됐다.
- 0-B2 AUPRC는 소폭 낮아졌지만, 어려운 시간 대조군과 0-B1 성능이 개선돼 목적에 더 적합하다.

### 기후지형유형별 결과

`PLUS_CONTINUOUS_FLAGS`:

| 유형 | AUPRC | 기준 모델 대비 |
|---|---:|---:|
| 영동 해안형 | 0.5268 | +0.0863 |
| 영서 내륙형 | 0.2599 | -0.0009 |
| 고지·산간형 | 0.2613 | -0.0094 |

해석:

- 전체 개선의 중심은 영동 해안형이다.
- 국지 저습·6시간 풍속·무강수 피처가 EDA의 영동 건조강풍 패턴을 모델에서 포착했다.
- 영서와 고지·산간형은 개선이 없거나 소폭 악화돼 후속 단계에서 권역별 피처 또는 층화 모델을 검토해야 한다.

### 후보 플래그 감사

- `rh_local_q05`: Target 1 17.71%, 0-A 4.93%
- `dry_spell_0p1_gt_24h`: Target 1 90.98%, 0-A 78.09%
- `dry_spell_5p0_gt_240h`: Target 1 89.94%, 0-A 80.37%
- `rh_local_q05_AND_ffmc_ge_90`: Target 1 8.45%, 0-A 0.55%
- `rh_local_q05_AND_isi_ge_10`: Target 1 6.44%, 0-A 0.21%
- `rh_local_q05_AND_wind_max_6h_ge_5`: Target 1 2.66%, 0-A 0.74%

EDA에서 확인한 위험방향과 모델링 표본의 충족률 방향이 일치했다.

### 생성 산출물

- `outputs/stage7_result_summary.md`
- `outputs/features/stage7_engineered_features.csv`
- `outputs/features/stage7_feature_sets.json`
- `outputs/features/stage7_recommended_feature_set.json`
- `outputs/metrics/stage7_feature_set_comparison.csv`
- `outputs/metrics/stage7_feature_set_overall_metrics.csv`
- `outputs/metrics/stage7_feature_set_negative_type_metrics.csv`
- `outputs/metrics/stage7_feature_set_climate_metrics.csv`
- `outputs/metrics/stage7_candidate_feature_prevalence.csv`
- `outputs/predictions/stage7_feature_set_oof_predictions.csv`
- `outputs/coefficients/stage7_feature_set_fold_coefficients.csv`
- `outputs/plots/stage7_01_feature_sets_auprc.png`
- `outputs/plots/stage7_02_delta_vs_baseline.png`
- `outputs/plots/stage7_03_climate_auprc.png`
- `outputs/plots/stage7_04_candidate_prevalence.png`

### 다음 작업

상태: 대기

8단계:

1. `PLUS_CONTINUOUS_FLAGS`를 기준으로 상관 블록 대표변수를 선택한다.
2. 예측 모델과 분리된 축소 오즈비 모델을 구성한다.
3. 변수별 표준화 OR, 95% CI, fold 부호 안정성을 산출한다.
4. 이후 동일 피처로 class-weight, Elastic Net, spline/GEE 등 로지스틱 계열 후보를 비교한다.

## 2026-06-18 — Stage 5~7 평가 보고서 상세화

### 수정 배경

- 기존 요약문이 AUPRC 중심으로 지나치게 압축되어 F1, 정확도, 정밀도, 재현율, ROC AUC와 threshold의 관계를 파악하기 어려웠다.
- AUPRC는 양성률 9.11%의 불균형 자료에서 주 모델 선택지표로 유지하되, 다른 지표를 생략하는 의미가 아니라는 점을 명시했다.
- 정확도·정밀도·재현율·F1은 threshold 의존 지표이고 AUPRC·ROC AUC·Brier는 threshold 비의존 지표이므로 두 표를 분리했다.

### 수행 내용

상태: 완료

- `outputs/stage5_result_summary.md`를 8개 절의 상세 결과문으로 개정했다.
- `outputs/stage6_result_summary.md`를 9개 절의 상세 결과문으로 개정했다.
- `outputs/stage7_result_summary.md`를 13개 절의 상세 결과문으로 개정했다.
- 각 단계에 데이터 구성, AUPRC 사용 이유, ROC AUC, Brier, Log loss, calibration, 대조군별 성능, fold 안정성, 변수·오즈비 해석 제한을 추가했다.
- threshold 0.5, OOF F1 최적, OOF Recall 90%에서 Accuracy, Balanced accuracy, Precision, Recall, Specificity, F1, MCC와 confusion matrix를 보고하도록 통일했다.
- Stage 7에는 상위 5%·10%·20% 위험군의 양성률, lift, 산불 포착률을 추가했다.
- 현재 분할에서 동일 기상노출과 집단발생 날짜가 여러 fold에 분산될 수 있다는 검증 한계를 Stage 5와 Stage 7 요약에 명시했다.

### 새 산출물

- `refresh_result_summaries.py`
- `outputs/metrics/stage7_feature_set_threshold_metrics.csv`
- `outputs/metrics/stage7_feature_set_overall_metrics_detailed.csv`
- `outputs/metrics/stage7_recommended_top_risk_metrics.csv`

### Stage 7 추천모델의 임계값별 핵심 결과

| 기준 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| threshold 0.5 | 0.9152 | 0.7529 | 0.1031 | 0.1813 |
| OOF F1 최적 threshold 0.2499 | 0.8926 | 0.3976 | 0.3470 | 0.3706 |
| OOF Recall 90% threshold 0.0428 | 0.5215 | 0.1487 | 0.9002 | 0.2553 |

추가 해석:

- threshold 0.5의 Accuracy 91.52%는 높지만 산불 재현율이 10.31%에 불과하다.
- Accuracy가 높은 이유 중 상당 부분은 음성이 90.89%인 자료 구성 때문이다.
- Stage 6 기준모델 대비 Stage 7 추천모델의 F1은 0.3574에서 0.3706으로 +0.0132 개선됐다.
- 전체 AUPRC는 0.3339에서 0.3751로 +0.0412, ROC AUC는 0.7852에서 0.7953으로 +0.0101 개선됐다.

### 계획 문서 반영

- 모든 단계별 요약에 threshold 비의존 지표와 threshold 의존 지표를 분리해 보고하는 규칙을 추가했다.
- Step 7.5로 날짜·기상노출 단위의 엄격한 재검증 단계를 추가했다.

## 2026-06-18 — Stage 8 D2D3 토지피복·임상도 보강 데이터 및 로지스틱 비교

### 수행 배경

- 사용자가 `D2` 토지피복 변수와 `D3` 임상도 변수를 합쳐 새 학습데이터를 만들고 분석을 이어서 진행하라고 요청했다.
- 외부 신규 데이터는 사용하지 않고 `D:\farm-system-public-02\원천데이터`와 기존 학습데이터 산출물만 사용했다.
- Target 1에만 임상도를 붙이면 누수가 되므로, 모든 학습 행에 동일한 공간조인 방식으로 토지피복·임상도 속성을 부여하는 방식으로 처리했다.

### 생성 데이터

상태: 완료

- 생성 스크립트: `build_d2d3_dataset.py`
- 생성 데이터: `data/학습데이터/학습데이터_로지스틱_D2D3.csv`
- 행/열: 17,045행 × 126열
- Target 분포: Target 1 = 1,553건, Target 0 = 15,492건
- D1의 `샘플ID`, `Target`, `샘플유형`, `모델링_그룹ID`는 유지했다.

토지피복 매칭 결과:

| 매칭방식 | 건수 |
|---|---:|
| within | 16,068 |
| nearest_30m | 90 |
| unmatched | 887 |

토지피복 주요 분포:

| L1 | 건수 |
|---|---:|
| 산림지역 | 7,357 |
| 시가화건조지역 | 3,686 |
| 초지 | 2,367 |
| 농업지역 | 1,562 |
| 미상 | 887 |
| 나지 | 823 |

임상도 매칭 결과:

| 출처 | 건수 |
|---|---:|
| 미매칭 | 17,017 |
| 2020_수종별zip | 28 |

해석:

- D2 토지피복 변수는 대부분 정상 부여됐다.
- D3 수종별 임상도는 현재 접근 가능한 2020 zip 기준 매칭률이 매우 낮아, 상세 임상도 변수로서 모델에 강하게 쓰기 어렵다.
- 2023 GDB는 pyogrio 접근 시 `Permission denied`가 발생했고, fiona는 현재 Python 환경에 설치되어 있지 않아 직접 공간조인에 사용하지 못했다.

### Stage 8 로지스틱 비교

상태: 완료

- 분석 스크립트: `stage8_d2d3_logistic_analysis.py`
- 비교 기준: Stage 7 추천 피처 `PLUS_CONTINUOUS_FLAGS`
- 분할: 기존 development outer/inner grouped CV 유지
- lockbox test는 사용하지 않았다.

| 피처 세트 | AUPRC | ROC AUC | Brier | Stage7 대비 AUPRC |
|---|---:|---:|---:|---:|
| PLUS_LANDCOVER | 0.3952 | 0.8294 | 0.06818 | +0.0201 |
| PLUS_LANDCOVER_FOREST | 0.3946 | 0.8292 | 0.06822 | +0.0195 |
| STAGE7_RECOMMENDED | 0.3751 | 0.7953 | 0.06982 | 0.0000 |
| PLUS_FOREST_STAND | 0.3746 | 0.7950 | 0.06986 | -0.0006 |

최고 모델 `PLUS_LANDCOVER`의 F1 최적 운영점:

| threshold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.2077 | 0.8622 | 0.3225 | 0.4654 | 0.3810 |

대조군별 해석:

- `Target_0A`: Stage7 0.4088 → PLUS_LANDCOVER 0.4049로 소폭 하락했다.
- `Target_0B1`: Stage7 0.7568 → PLUS_LANDCOVER 0.9156으로 크게 상승했다.
- `Target_0B2`: Stage7 0.8472 → PLUS_LANDCOVER 0.9313으로 크게 상승했다.
- 따라서 토지피복은 공간 배경 대조군 구분에는 강하지만, 같은 위치의 다른 시간대인 0-A 구분에는 거의 도움을 주지 않는다.

기후지형유형별 해석:

- 영동 해안형 AUPRC: 0.5268 → 0.5524
- 영서 내륙형 AUPRC: 0.2599 → 0.2751
- 고지·산간형 AUPRC: 0.2613 → 0.2743

### 산출물

- `data/학습데이터/학습데이터_로지스틱_D2D3.csv`
- `outputs/stage8_d2d3_result_summary.md`
- `outputs/features/d2d3_feature_audit.csv`
- `outputs/features/d2d3_dataset_summary.json`
- `outputs/features/stage8_d2d3_feature_sets.json`
- `outputs/metrics/stage8_d2d3_overall_metrics.csv`
- `outputs/metrics/stage8_d2d3_threshold_metrics.csv`
- `outputs/metrics/stage8_d2d3_sample_type_metrics.csv`
- `outputs/metrics/stage8_d2d3_climate_metrics.csv`
- `outputs/metrics/stage8_d2d3_top_risk_metrics.csv`
- `outputs/predictions/stage8_d2d3_oof_predictions.csv`
- `outputs/coefficients/stage8_d2d3_fold_coefficients.csv`
- `outputs/plots/stage8_d2d3_01_auprc.png`
- `outputs/plots/stage8_d2d3_02_delta_auprc.png`

### 다음 작업

상태: 대기

1. 최종 로지스틱 주 모델은 현재 기준 `PLUS_LANDCOVER`를 우선 후보로 둔다.
2. 다만 성능 개선이 B1/B2 공간 대조군에서 주로 발생했으므로, 최종 결론 전에는 0-A와 날짜·기상노출 엄격 검증을 다시 확인해야 한다.
3. D3 상세 임상도는 현재 매칭률이 낮아 최종 보고서 핵심 변수로 주장하지 않는다.
4. 이후 Step 7.5 엄격 검증 또는 로지스틱 계열 확장 시 D2 토지피복 변수는 유지하고, D3 임상도 상세 변수는 제외하거나 보조 품질 플래그로만 둔다.

## 2026-06-18 — Step 7.5 날짜·기상노출 엄격 검증

### 수행 배경

- Stage 7/8 grouped CV는 `모델링_그룹ID`를 분리했지만 동일 `기상셀ID×기준시각`과 집단 발생 날짜가 fold 사이에 나뉠 수 있었다.
- lockbox를 열기 전 development 내부에서 더 엄격한 검증을 수행했다.
- 비교 모델은 `STAGE7_RECOMMENDED`와 Stage 8 후보 `PLUS_LANDCOVER` 두 개로 제한했다.

### 검증 방식

상태: 완료

1. `exposure_component_cv`
   - `모델링_그룹ID`와 동일 `기상셀ID×기준시각`을 연결 성분으로 묶어 fold를 구성했다.
   - 모델 그룹 누수 0건, 동일 기상노출 누수 0건.
2. `date_exposure_component_cv`
   - 위 조건에 더해 Target 1/0-A는 원 산불 발생 날짜, B계열 대조군은 실제 기준날짜를 날짜 블록으로 묶었다.
   - 모델 그룹 누수 0건, 동일 기상노출 누수 0건, 날짜 블록 누수 0건, 양성 실제발생일 누수 0건.
   - 단, 가장 큰 날짜 성분이 5,140행과 양성 558건을 포함해 매우 강한 stress test 성격을 가진다.

### 전체 성능

| 검증 방식 | 모델 | AUPRC | ROC AUC | Brier |
|---|---|---:|---:|---:|
| current grouped CV | STAGE7_RECOMMENDED | 0.3751 | 0.7953 | 0.06982 |
| current grouped CV | PLUS_LANDCOVER | 0.3952 | 0.8294 | 0.06818 |
| exposure_component_cv | STAGE7_RECOMMENDED | 0.3437 | 0.7851 | 0.07145 |
| exposure_component_cv | PLUS_LANDCOVER | 0.3665 | 0.8213 | 0.06970 |
| date_exposure_component_cv | STAGE7_RECOMMENDED | 0.1961 | 0.7315 | 0.07975 |
| date_exposure_component_cv | PLUS_LANDCOVER | 0.2300 | 0.7763 | 0.07757 |

### 해석

- 동일 기상노출만 막으면 `PLUS_LANDCOVER` AUPRC는 0.3952에서 0.3665로 낮아진다.
- 날짜와 기상노출을 모두 막으면 `PLUS_LANDCOVER` AUPRC는 0.2300까지 낮아진다.
- 따라서 기존 Stage 8 점수는 날짜·기상노출 공유에 의해 낙관적이었을 가능성이 크다.
- 그래도 `PLUS_LANDCOVER`는 모든 엄격 검증에서 `STAGE7_RECOMMENDED`보다 높았다.
- 가장 엄격한 기준에서 AUPRC 차이는 +0.0340이다.

### F1 운영점

| 기준 | threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| current grouped CV / PLUS_LANDCOVER | 0.2077 | 0.8622 | 0.3225 | 0.4654 | 0.3810 |
| date_exposure_component_cv / PLUS_LANDCOVER | 0.1645 | 0.7940 | 0.2210 | 0.4992 | 0.3063 |

### 대조군별 해석

- 가장 엄격한 기준에서 `Target_0A` AUPRC는 Stage7 0.2309 → PLUS_LANDCOVER 0.2412로 소폭 개선에 그쳤다.
- `Target_0B1` AUPRC는 0.6341 → 0.8762로 크게 개선됐다.
- `Target_0B2` AUPRC는 0.7893 → 0.9010으로 크게 개선됐다.
- 토지피복은 공간 배경 대조군 구분에는 강하지만, 같은 위치의 다른 시간대 구분에는 제한적이라는 Stage 8 해석이 유지된다.

### 산출물

- `stage75_strict_validation.py`
- `outputs/stage75_strict_validation_summary.md`
- `outputs/metrics/stage75_current_vs_strict_overall.csv`
- `outputs/metrics/stage75_strict_group_audit.csv`
- `outputs/metrics/stage75_strict_threshold_metrics.csv`
- `outputs/metrics/stage75_strict_sample_type_metrics.csv`
- `outputs/metrics/stage75_strict_climate_metrics.csv`
- `outputs/predictions/stage75_strict_oof_predictions.csv`
- `outputs/coefficients/stage75_strict_fold_coefficients.csv`
- `outputs/plots/stage75_01_current_vs_strict_auprc.png`

### 다음 작업

상태: 대기

1. 로지스틱 주 후보는 `PLUS_LANDCOVER`로 유지한다.
2. 최종 보고서에는 current grouped CV 점수와 strict CV 점수를 분리해서 보고한다.
3. 다음 단계는 같은 strict split 기준으로 로지스틱 계열의 통계적 이진분류 모델을 추가 비교하는 것이다.
4. lockbox는 아직 열지 않는다.

## 2026-06-18 — Stage 9 로지스틱 추가 개선 실험

### 수행 배경

- 사용자가 로지스틱에서 더 할 수 있는 통계적 개선이 있는지 요청했다.
- lockbox는 열지 않고, Step 7.5에서 만든 가장 엄격한 `date_exposure_component_cv`를 그대로 사용했다.
- 비교 기준은 Stage 8/7.5에서 가장 안정적이었던 `PLUS_LANDCOVER`로 두었다.
- 요인점수는 이전 결과에서 실질적 개선이 작았으므로 제외했다.

### 실험 내용

상태: 완료

1. EDA 기반 rule/interaction 변수 추가
   - 도로 10m/30m 이내, 시가화 10m/30m 이내
   - 비산림 WUI × 도로접근성
   - 비산림 WUI × 국지저습/무강수/풍속
   - 산림·침엽수림 × 국지저습
   - 영동 × 서풍성 강풍·토지피복 후보
2. 고정 bin 비선형화
   - 습도, 6시간 최대풍속, 무강수 지속, 도로거리, D1 FWI/ISI 구간화
3. ANOVA F-test 변수선택
   - `SelectKBest(f_classif)`를 nested CV 내부에서만 수행해 leakage를 방지했다.
4. L1 정규화
   - sparse logistic 후보로 변수 제거 효과를 확인했다.

### 전체 성능

가장 엄격한 `date_exposure_component_cv` 기준 결과:

| 모델 | AUPRC | ROC AUC | Brier | PLUS_LANDCOVER 대비 AUPRC |
|---|---:|---:|---:|---:|
| PLUS_LANDCOVER_RULES_ANOVA | 0.2398 | 0.7768 | 0.07697 | +0.0098 |
| PLUS_LANDCOVER_RULES_L2 | 0.2307 | 0.7762 | 0.07765 | +0.0006 |
| PLUS_LANDCOVER | 0.2300 | 0.7763 | 0.07757 | 기준 |
| PLUS_LANDCOVER_RULES_BINS_ANOVA | 0.2298 | 0.7733 | 0.07772 | -0.0002 |
| PLUS_LANDCOVER_RULES_BINS_L2 | 0.2209 | 0.7715 | 0.07862 | -0.0092 |
| PLUS_LANDCOVER_RULES_L1 | 0.2183 | 0.7687 | 0.07790 | -0.0117 |
| STAGE7_RECOMMENDED | 0.1961 | 0.7315 | 0.07975 | -0.0340 |

### F1 운영점

| 모델 | threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| PLUS_LANDCOVER | 0.1645 | 0.7940 | 0.2210 | 0.4992 | 0.3063 |
| PLUS_LANDCOVER_RULES_ANOVA | 0.1483 | 0.7711 | 0.2107 | 0.5507 | 0.3047 |

### 대조군별 해석

- `Target_0A` AUPRC는 `PLUS_LANDCOVER` 0.2412 → `PLUS_LANDCOVER_RULES_ANOVA` 0.2500으로 +0.0087 개선됐다.
- `Target_0B1` AUPRC는 0.8762 → 0.8846으로 소폭 개선됐다.
- `Target_0B2` AUPRC는 0.9010 → 0.9063으로 소폭 개선됐다.
- 즉 ANOVA 변수선택은 전체적으로 약간 도움이 되지만, 같은 위치의 다른 시간대를 구분하는 0-A 문제를 크게 해결하지는 못했다.

### 판단

- 로지스틱에서 ANOVA 변수선택은 남길 수 있는 후보이지만 개선폭은 작다.
- EDA rule 변수만 추가한 L2는 거의 변화가 없었다.
- L1 정규화와 단순 binning은 성능을 낮췄다.
- 따라서 단순 L1/L2/ANOVA 수준의 로지스틱 튜닝은 수확체감 구간에 들어갔다고 판단한다.
- 다음 단계는 통계적 이진분류 범위 안에서 elastic net, class-weight/cost-sensitive logistic, 0-A 중심 로지스틱, GAM/스플라인 로지스틱, GEE/혼합효과 로지스틱 가능성을 비교하는 것이다.

### 산출물

- `stage9_logistic_enhancement.py`
- `outputs/stage9_logistic_enhancement_summary.md`
- `outputs/metrics/stage9_logistic_enhancement_overall_metrics.csv`
- `outputs/metrics/stage9_logistic_enhancement_threshold_metrics.csv`
- `outputs/metrics/stage9_logistic_enhancement_sample_type_metrics.csv`
- `outputs/metrics/stage9_logistic_enhancement_climate_metrics.csv`
- `outputs/metrics/stage9_logistic_enhancement_top_risk_metrics.csv`
- `outputs/predictions/stage9_logistic_enhancement_oof_predictions.csv`
- `outputs/predictions/stage9_new_candidate_oof_predictions.csv`
- `outputs/coefficients/stage9_logistic_enhancement_fold_coefficients.csv`
- `outputs/plots/stage9_01_logistic_enhancement_auprc.png`

## 2026-06-18 — Stage 10 경량 로지스틱 계열 통계모델 확장

### 수행 배경

- 비통계적 이진분류 모델은 제외한다는 방향으로 수정했다.
- 이전 Stage 10~12 비통계적 모델 관련 스크립트, 산출물, 로그는 삭제했다.
- spline/GAM까지 한 번에 포함한 첫 Stage 10 시도는 실행 시간이 과도해 중단했다.
- 이번 Stage 10은 빠른 스크리닝으로 범위를 줄였다.
- strict `date_exposure_component_cv`를 그대로 사용했고 lockbox는 열지 않았다.

### 실험 내용

상태: 완료

비교 후보:

1. `LOGIT_L2_RULES_C1`
   - Stage 9 rule 피처 전체를 넣은 고정 C=1 L2 로지스틱
2. `LOGIT_BALANCED_L2_C1`
   - class-balanced sample weight 적용
3. `LOGIT_0A_WEIGHT_X2_L2_C1`
   - 0-A 음성 표본 가중치 2배
4. `LOGIT_0A_WEIGHT_X4_L2_C1`
   - 0-A 음성 표본 가중치 4배
5. `LOGIT_LIGHT_INTERACTIONS_L2_C1`
   - 습도×풍속, 습도×무강수, D1 지수×습도, 영동×풍속 등 간단 상호작용 추가
6. `LOGIT_ELASTICNET_C01_L1R05`
   - C=0.1, l1_ratio=0.5
7. `LOGIT_ELASTICNET_C1_L1R05`
   - C=1.0, l1_ratio=0.5

### 전체 성능

| 모델 | AUPRC | ROC AUC | Brier | Stage 9 최고 대비 AUPRC |
|---|---:|---:|---:|---:|
| PLUS_LANDCOVER_RULES_ANOVA | 0.2398 | 0.7768 | 0.07697 | 기준 |
| PLUS_LANDCOVER | 0.2300 | 0.7763 | 0.07757 | -0.0098 |
| LOGIT_LIGHT_INTERACTIONS_L2_C1 | 0.2295 | 0.7742 | 0.07815 | -0.0103 |
| LOGIT_ELASTICNET_C1_L1R05 | 0.2286 | 0.7760 | 0.07776 | -0.0112 |
| LOGIT_L2_RULES_C1 | 0.2224 | 0.7731 | 0.07848 | -0.0174 |
| LOGIT_0A_WEIGHT_X2_L2_C1 | 0.2217 | 0.7684 | 0.14807 | -0.0181 |
| LOGIT_ELASTICNET_C01_L1R05 | 0.2216 | 0.7745 | 0.07753 | -0.0182 |
| LOGIT_0A_WEIGHT_X4_L2_C1 | 0.2161 | 0.7528 | 0.10828 | -0.0238 |
| LOGIT_BALANCED_L2_C1 | 0.2154 | 0.7767 | 0.20480 | -0.0244 |

### F1 운영점

| 모델 | threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| PLUS_LANDCOVER | 0.1645 | 0.7940 | 0.2210 | 0.4992 | 0.3063 |
| LOGIT_BALANCED_L2_C1 | 0.5186 | 0.6866 | 0.1913 | 0.7560 | 0.3053 |
| PLUS_LANDCOVER_RULES_ANOVA | 0.1483 | 0.7711 | 0.2107 | 0.5507 | 0.3047 |
| LOGIT_ELASTICNET_C1_L1R05 | 0.1908 | 0.8223 | 0.2361 | 0.4251 | 0.3036 |
| LOGIT_LIGHT_INTERACTIONS_L2_C1 | 0.1663 | 0.7969 | 0.2200 | 0.4831 | 0.3023 |

### 0-A 성능

| 모델 | 0-A AUPRC | 0-A ROC AUC | 0-A Brier |
|---|---:|---:|---:|
| PLUS_LANDCOVER_RULES_ANOVA | 0.2500 | 0.6054 | 0.13979 |
| LOGIT_0A_WEIGHT_X4_L2_C1 | 0.2477 | 0.6169 | 0.17447 |
| PLUS_LANDCOVER | 0.2412 | 0.6084 | 0.14082 |
| LOGIT_LIGHT_INTERACTIONS_L2_C1 | 0.2411 | 0.6042 | 0.14176 |
| LOGIT_0A_WEIGHT_X2_L2_C1 | 0.2403 | 0.6112 | 0.23921 |
| LOGIT_ELASTICNET_C1_L1R05 | 0.2395 | 0.6051 | 0.14121 |

### 판단

- 경량 로지스틱 확장에서는 Stage 9 최고 후보 `PLUS_LANDCOVER_RULES_ANOVA`를 넘는 모델이 없었다.
- Elastic Net은 성능을 유지하지 못했고, 전체 AUPRC와 0-A AUPRC가 모두 낮아졌다.
- class-balanced와 0-A 가중 로지스틱은 recall을 올릴 수 있지만 Brier가 크게 나빠져 확률모델로 부적절하다.
- 간단 상호작용 추가도 Stage 9 ANOVA 후보보다 낮았다.
- 따라서 예측 성능 개선 측면에서는 로지스틱 계열이 거의 한계에 도달한 것으로 본다.
- 다만 통계 해석 측면에서는 아직 축소 오즈비 모델, cluster-robust CI, GEE/혼합효과 로지스틱, 제한적 spline/GAM 단독 실험은 남아 있다.

### 산출물

- `stage10_logistic_stat_extensions.py`
- `outputs/stage10_logistic_stat_extensions_summary.md`
- `outputs/metrics/stage10_logistic_stat_extensions_overall_metrics.csv`
- `outputs/metrics/stage10_logistic_stat_extensions_threshold_metrics.csv`
- `outputs/metrics/stage10_logistic_stat_extensions_sample_type_metrics.csv`
- `outputs/metrics/stage10_logistic_stat_extensions_climate_metrics.csv`
- `outputs/metrics/stage10_logistic_stat_extensions_top_risk_metrics.csv`
- `outputs/metrics/stage10_logistic_stat_extensions_fold_metrics.csv`
- `outputs/predictions/stage10_logistic_stat_extensions_oof_predictions.csv`
- `outputs/coefficients/stage10_logistic_stat_extensions_fold_coefficients.csv`
- `outputs/plots/stage10_logistic_01_auprc.png`
- `outputs/plots/stage10_logistic_02_best_f1.png`
- `outputs/plots/stage10_logistic_03_0a_auprc.png`

## 2026-06-18 — 로지스틱 모델 비교용 통합 성능표 생성

### 수행 배경

- 다른 사람이 산출한 모델 성능과 직접 비교할 수 있도록 로지스틱 결과를 한 표로 통합했다.
- Stage 9와 Stage 10의 strict `date_exposure_component_cv` development OOF 결과를 사용했다.
- lockbox는 사용하지 않았다.

### 포함 지표

상태: 완료

- 전체 성능: AUPRC, ROC AUC, Brier score, log loss
- best-F1 운영점: threshold, accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, TN/FP/FN/TP
- fixed 0.5 운영점: accuracy, precision, recall, F1
- recall 90% 운영점: threshold, accuracy, precision, recall, F1
- 대조군별 성능: `Target_0A`, `Target_0B1`, `Target_0B2` 각각 AUPRC, ROC AUC, Brier, log loss

### 주요 결과

| 모델 | AUPRC | ROC AUC | Brier | best-F1 | Precision | Recall | 0-A AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| PLUS_LANDCOVER_RULES_ANOVA | 0.2398 | 0.7768 | 0.07697 | 0.3047 | 0.2107 | 0.5507 | 0.2500 |
| PLUS_LANDCOVER | 0.2300 | 0.7763 | 0.07757 | 0.3063 | 0.2210 | 0.4992 | 0.2412 |
| LOGIT_LIGHT_INTERACTIONS_L2_C1 | 0.2295 | 0.7742 | 0.07815 | 0.3023 | 0.2200 | 0.4831 | 0.2411 |
| LOGIT_ELASTICNET_C1_L1R05 | 0.2286 | 0.7760 | 0.07776 | 0.3036 | 0.2361 | 0.4251 | 0.2395 |
| LOGIT_BALANCED_L2_C1 | 0.2154 | 0.7767 | 0.20480 | 0.3053 | 0.1913 | 0.7560 | 0.2262 |

### 산출물

- `make_logistic_benchmark_tables.py`
- `outputs/logistic_benchmark_for_model_comparison.csv`
- `outputs/logistic_benchmark_for_model_comparison.md`

## 2026-06-19 — 로지스틱 노트북 재현 블록 추가

### 수행 배경

- `26.06.18_로지스틱_모델링.ipynb`에 Stage 7.5 이후 코드가 포함되어 있는지 확인했다.
- 확인 결과 기존 노트북은 Stage 5~7 중심이고, `stage75`, `stage8`, `stage9`, `stage10`, 통합 비교표 생성 호출은 없었다.
- 재현 가능성을 위해 노트북 뒤쪽에 최종 로지스틱 재현 실행 블록을 추가했다.

### 추가 내용

상태: 완료

노트북에 다음 실행 순서를 추가했다.

1. `stage7_feature_extension.py`
2. `build_d2d3_dataset.py`
3. `stage8_d2d3_logistic_analysis.py`
4. `stage75_strict_validation.py`
5. `stage9_logistic_enhancement.py`
6. `stage10_logistic_stat_extensions.py`
7. `make_logistic_benchmark_tables.py`

### 실행 방식

- 노트북의 `RUN_REPRODUCE_STAGE7_TO_STAGE10 = True` 상태에서 Run All 하면 Stage 7~10과 통합 비교표를 재생성한다.
- 이미 산출된 결과만 빠르게 확인하려면 해당 값을 `False`로 바꾸면 된다.
- lockbox는 열지 않는다.
- 비통계적 이진분류 모델은 포함하지 않는다.

### 산출물

- `26.06.18_로지스틱_모델링.ipynb`
- `_append_reproducibility_sections.py`

## 2026-06-19 — 다음 단계 인수인계 계획: 최종 보고서용 로지스틱 통계 모델링

### 현재 판단

상태: 계획 확정, 미실행

- Stage 9~10 결과 기준으로 로지스틱의 예측 성능 개선 폭은 제한적이다.
- `PLUS_LANDCOVER_RULES_ANOVA`가 AUPRC, ROC AUC, Brier, 0-A AUPRC 기준에서 가장 안정적인 대표 성능 모델이다.
- `PLUS_LANDCOVER`는 best-F1이 아주 조금 더 높으므로 운영 threshold 비교용 보조 모델로 유지한다.
- 요인점수는 앞으로 제외한다. 지금까지의 비교에서 요인점수를 포함할 실익이 충분하지 않았고, 최종 보고서에서도 해당 파트를 빼는 방향으로 정리한다.
- 이후 작업은 “성능을 무리하게 더 올리는 단계”가 아니라, 최종 보고서에 넣을 수 있는 “해석 가능하고 통계적으로 방어 가능한 로지스틱 분석”을 만드는 단계이다.
- 비통계적 이진분류 모델은 추가하지 않는다.

### 다음 단계 전체 순서

다른 세션에서 이어서 진행할 경우 아래 순서로 진행한다.

1. Step 11: 최종 성능 비교용 로지스틱 기준 모델 확정
2. Step 12: 해석용 로지스틱 변수셋 구성
3. Step 13: 오즈비, 신뢰구간, p-value, FDR q-value 산출
4. Step 14: 변수군 제거 실험
5. Step 15: 다중공선성, 상관관계, 비선형성 확인
6. Step 16: 필요한 로지스틱 계열 확장 모델만 선택적으로 수행

---

## Step 11 — 최종 성능 비교용 로지스틱 기준 모델 확정

### 목적

- 다른 사람이 산출한 모델 성능과 비교할 로지스틱 기준선을 고정한다.
- 최종 보고서에서 “로지스틱 대표 성능”과 “F1 운영점”을 분리해 설명할 수 있게 만든다.
- lockbox는 열지 않고 strict `date_exposure_component_cv` development OOF 결과를 기준으로 한다.

### 사용할 모델

| 용도 | 모델 | 이유 |
|---|---|---|
| 대표 성능 모델 | `PLUS_LANDCOVER_RULES_ANOVA` | AUPRC, ROC AUC, Brier, 0-A AUPRC가 가장 안정적 |
| 운영 threshold 보조 모델 | `PLUS_LANDCOVER` | best-F1이 가장 높거나 거의 최고 수준 |

### 진행할 작업

- `outputs/logistic_benchmark_for_model_comparison.csv`를 기준으로 최종 비교표를 정리한다.
- 포함 지표:
  - AUPRC
  - ROC AUC
  - Brier score
  - log loss
  - best-F1 threshold
  - Accuracy
  - Balanced accuracy
  - Precision
  - Recall
  - Specificity
  - F1-score
  - MCC
  - TN/FP/FN/TP
  - fixed 0.5 threshold 기준 Accuracy, Precision, Recall, F1
  - recall 90% 운영점 기준 Accuracy, Precision, Recall, F1
  - `Target_0A`, `Target_0B1`, `Target_0B2`별 AUPRC, ROC AUC, Brier, log loss

### 산출물

- `outputs/stage11_final_logistic_benchmark_summary.md`
- `outputs/tables/stage11_final_logistic_benchmark.csv`
- `outputs/tables/stage11_final_logistic_benchmark.md`

### 판단 기준

- 최종 성능 대표값은 `PLUS_LANDCOVER_RULES_ANOVA`를 우선한다.
- F1만 비교할 때는 `PLUS_LANDCOVER`도 함께 제시한다.
- “로지스틱 최고 성능”을 하나로만 말해야 하면 `PLUS_LANDCOVER_RULES_ANOVA`를 사용한다.

---

## Step 12 — 해석용 로지스틱 변수셋 구성

### 목적

- 성능 최고 모델을 그대로 오즈비 해석에 쓰지 않는다.
- ANOVA 변수선택, 정규화, 파생변수가 섞인 성능 모델은 예측용으로는 적절하지만, 개별 계수와 오즈비 해석에는 부적절할 수 있다.
- EDA 결론과 직접 연결되는 변수만 골라 별도의 해석용 로지스틱 모델을 만든다.

### 제외 원칙

- 요인점수는 제외한다.
- 비통계적 모델 변수중요도는 사용하지 않는다.
- 서로 거의 같은 의미의 변수는 동시에 많이 넣지 않는다.
  - 예: 습도 관련 변수가 여러 개면 대표 변수 1~2개만 선택
  - 예: FFMC, FWI, ISI가 강하게 중복되면 해석 목적에 맞춰 축소

### 포함 후보 변수군

1. 습도 / 국지저습
   - `직전24h_최소습도` 또는 이에 대응하는 학습데이터 변수
   - `직전24h_평균습도`
   - `rh_local_q05`
   - 필요 시 습도 5%p 감소 단위 OR 산출

2. 강수 / 무강수
   - `직전24h_강수량합`
   - `rain_sum_48h`
   - `rain_sum_72h`
   - `dry_spell_0p1_gt_24h`
   - `dry_spell_5p0_gt_240h`

3. 풍속 / 풍향
   - `wind_mean_6h`
   - `wind_max_6h`
   - `직전24h_평균풍속`
   - `직전24h_최대풍속`
   - 영동 해안형에서 서풍계열 강풍 조건이 있으면 후보로 포함

4. 공간 / 접근성
   - 도로 최단거리
   - 도로 10m 이내 여부
   - 시가화 최단거리
   - 산림 최단거리
   - WUI 여부 또는 공간층
   - 토지피복 대분류

5. 지형
   - 고도
   - 경사도
   - TPI
   - TWI는 기존 EDA에서 결측과 극단값 문제가 있었으므로 핵심 변수로 쓰지 않는다.

6. 캐나다 산불지수
   - FFMC
   - DMC
   - DC
   - ISI
   - BUI
   - FWI
   - 서로 강한 공선성이 있으면 대표 지수만 선택한다.

7. 층화 / 집단 변수
   - 기후지형유형
   - 필요 시 번지유형은 보조로만 검토
   - 주소 기반 임야번지는 실제 산림 피복과 불일치가 있으므로 최종 핵심 해석 변수로 과신하지 않는다.

8. EDA 기반 핵심 상호작용
   - `영동 해안형 × 풍속`
   - `국지저습 × 6시간 최대풍속`
   - `WUI × 도로 최단거리`
   - 필요 시 `기후지형유형 × 캐나다지수`

### 진행할 작업

- 학습데이터 컬럼명을 확인하고 위 후보와 실제 컬럼을 매핑한다.
- 결측률, 분포, 유일값 수를 확인한다.
- 수치형 변수는 표준화 버전과 실제 단위 OR 산출용 원본 단위를 함께 관리한다.
- 범주형 변수는 기준 범주를 명확히 지정한다.
- 해석용 모델 후보를 2~3개로 나눈다.
  - `INTERPRET_WEATHER_SPACE`: 날씨 + 공간 + 지형
  - `INTERPRET_WEATHER_SPACE_CANADA`: 날씨 + 공간 + 지형 + 캐나다지수
  - `INTERPRET_EDA_INTERACTIONS`: 위 변수 + 제한적 상호작용

### 산출물

- `stage12_interpret_feature_set.py`
- `outputs/stage12_interpret_feature_set_summary.md`
- `outputs/tables/stage12_feature_mapping.csv`
- `outputs/tables/stage12_missing_and_distribution.csv`

### 판단 기준

- 해석용 모델은 성능 최고가 목적이 아니다.
- 계수 방향, 오즈비, 신뢰구간이 EDA 결론과 일관되는지 확인하는 것이 목적이다.
- 너무 많은 변수를 넣어 표준오차가 커지거나 부호가 불안정해지면 변수셋을 줄인다.

---

## Step 13 — 오즈비, 신뢰구간, p-value, FDR q-value 산출

### 목적

- 최종 보고서에 들어갈 변수별 통계 해석 표를 만든다.
- 단순 계수표가 아니라 실제 해석 가능한 단위의 오즈비를 제공한다.

### 진행할 작업

- 해석용 모델에 대해 비정규화 로지스틱 또는 약한 L2 로지스틱을 적합한다.
- 가능하면 `statsmodels` GLM Binomial을 사용해 계수, 표준오차, p-value를 산출한다.
- 날짜 또는 기상셀 기준 cluster-robust 표준오차를 시도한다.
  - 1순위: 발생일 또는 기준일 단위 clustering
  - 2순위: 기상셀 단위 clustering
  - 데이터 구조상 실패하면 일반 robust HC 표준오차를 대안으로 사용하고 로그에 명시한다.
- 각 변수에 대해 다음을 산출한다.
  - coefficient
  - standard error
  - z-value 또는 Wald statistic
  - p-value
  - FDR q-value
  - odds ratio
  - 95% CI lower/upper
  - 표준화 1SD 증가 기준 OR
  - 실제 단위 기준 OR

### 실제 단위 OR 예시

- 습도: 5%p 감소 시 OR
- 풍속: 1m/s 증가 시 OR
- 강수량: 1mm 또는 5mm 증가 시 OR
- 도로거리: 10m 증가 또는 log-distance 증가 시 OR
- 고도: 100m 증가 시 OR
- FFMC/FWI: 1점 또는 5점 증가 시 OR

### 산출물

- `stage13_logistic_or_inference.py`
- `outputs/stage13_logistic_or_inference_summary.md`
- `outputs/tables/stage13_odds_ratios.csv`
- `outputs/tables/stage13_odds_ratios.md`
- `outputs/tables/stage13_cluster_robust_odds_ratios.csv`
- `outputs/plots/stage13_or_forestplot.png`

### 판단 기준

- p-value만으로 의미를 판단하지 않는다.
- OR 방향, CI 폭, EDA와의 일관성, 변수군 제거 실험에서의 안정성을 함께 본다.
- 정규화 모델의 계수는 예측 보조로만 보고, 최종 오즈비 해석은 비정규화 또는 해석용 모델 기준으로 한다.

---

## Step 14 — 변수군 제거 실험

### 목적

- 어떤 변수군이 실제 성능과 해석에 기여하는지 확인한다.
- 최종 보고서에서 “습도·무강수·공간 접근성·캐나다지수 중 무엇이 중요한가”를 근거 있게 설명한다.

### 비교할 모델

기준 모델은 Step 12의 해석용 전체 모델로 둔다.

| 실험명 | 제거 변수군 | 목적 |
|---|---|---|
| `DROP_WEATHER` | 습도, 강수, 풍속 등 날씨 변수 | 기상 조건의 기여도 확인 |
| `DROP_SPACE` | 도로거리, WUI, 토지피복, 시가화/산림거리 | 공간 접근성의 기여도 확인 |
| `DROP_CANADA` | FFMC, DMC, DC, ISI, BUI, FWI | 캐나다 산불지수의 추가 기여도 확인 |
| `DROP_LANDCOVER` | 토지피복 범주 | 토지피복 효과 확인 |
| `DROP_INTERACTIONS` | EDA 기반 상호작용 | 상호작용의 실질 기여도 확인 |

### 평가 지표

- AUPRC
- ROC AUC
- Brier score
- log loss
- best-F1
- Precision
- Recall
- MCC
- likelihood ratio test 가능 여부
- AIC / BIC 가능 여부

### 진행 방식

- strict `date_exposure_component_cv` OOF 구조를 유지한다.
- lockbox는 열지 않는다.
- 성능 차이는 단일 숫자만 보지 말고 fold별 차이도 확인한다.
- GLM 기반 nested model이면 likelihood ratio test를 시도한다.
- 완전히 nested가 아니거나 정규화가 섞이면 LRT 해석을 제한한다.

### 산출물

- `stage14_ablation_tests.py`
- `outputs/stage14_ablation_tests_summary.md`
- `outputs/tables/stage14_ablation_metrics.csv`
- `outputs/tables/stage14_likelihood_ratio_tests.csv`
- `outputs/plots/stage14_ablation_auprc.png`
- `outputs/plots/stage14_ablation_brier.png`

### 판단 기준

- 성능 하락이 작아도 OR 해석상 중요한 변수군이면 유지할 수 있다.
- 캐나다지수 제거 후 성능이 거의 변하지 않으면, 최종 보고서에서는 “캐나다지수는 보조 설명 변수”로 낮춰 쓴다.
- 공간 변수 제거 후 성능이 크게 나빠지면, EDA의 도로/WUI 결론과 강하게 연결한다.

---

## Step 15 — 다중공선성, 상관관계, 비선형성 확인

### 목적

- 해석용 로지스틱 모델의 계수 안정성을 점검한다.
- 습도, 풍속, 도로거리, 캐나다지수처럼 비선형 가능성이 큰 변수를 선형항 하나로만 해석해도 되는지 확인한다.

### 다중공선성 확인

진행할 작업:

- 수치형 변수 상관행렬 산출
- Spearman/Pearson 상관 확인
- VIF 산출
- 캐나다지수끼리의 중복성 확인
- 습도 계열 변수끼리의 중복성 확인
- 거리 변수끼리의 중복성 확인

판단 기준:

- VIF가 과도하게 높으면 같은 의미의 변수 중 하나를 제거한다.
- FFMC/FWI/ISI 등 캐나다지수가 강하게 묶이면 대표 지수만 남긴다.
- 습도 변수는 EDA 결론과 가장 직접 연결되는 `직전24h_최소습도` 또는 `rh_local_q05`를 우선한다.

### 비선형성 확인

대상 변수:

- 습도
- 도로거리
- 풍속
- FFMC
- FWI
- 고도

진행할 작업:

- 분위수 binning 기반 로지스틱 효과 확인
- 제한적 spline logistic 검토
- 각 bin별 산불 비율과 예측확률 확인
- 비선형성이 강하면 최종 해석 모델에 bin 또는 spline을 보조로 추가

### 산출물

- `stage15_collinearity_nonlinearity.py`
- `outputs/stage15_collinearity_nonlinearity_summary.md`
- `outputs/tables/stage15_correlations.csv`
- `outputs/tables/stage15_vif.csv`
- `outputs/tables/stage15_binned_effects.csv`
- `outputs/plots/stage15_corr_heatmap.png`
- `outputs/plots/stage15_vif_bar.png`
- `outputs/plots/stage15_binned_humidity.png`
- `outputs/plots/stage15_binned_road_distance.png`
- `outputs/plots/stage15_binned_wind.png`
- `outputs/plots/stage15_binned_fwi.png`

### 판단 기준

- 비선형성이 확인되어도 복잡한 모델을 무조건 최종 모델로 쓰지는 않는다.
- 보고서에서는 선형 로지스틱의 OR 표를 중심으로 두고, bin/spline 결과는 “선형 가정 점검” 또는 “임계구간 후보”로 제시한다.

---

## Step 16 — 선택적 로지스틱 계열 확장

### 목적

- 일반 로지스틱만으로 불안정한 부분을 보완한다.
- 단, 성능 개선 목적의 무리한 확장은 하지 않는다.

### 우선순위

1. 일반 로지스틱
   - 최종 OR 해석의 기본 모델

2. 약한 L2 로지스틱
   - 공선성 또는 계수 불안정이 있을 때 비교용
   - 오즈비 해석은 보조로 제한

3. GEE logistic 또는 cluster-robust logistic
   - 날짜 군집, 기상셀 군집을 반영하기 위한 통계 모델
   - 가능하면 Step 13에서 cluster-robust SE를 우선 시도하고, 부족하면 GEE로 확장

4. spline/bin logistic
   - 습도, 도로거리, 풍속, FFMC/FWI의 비선형 해석 보조

5. rare-event 또는 Firth logistic
   - 특정 층화에서 separation 또는 희소 문제가 발생할 때만 검토
   - 전체 모델 성능 개선용으로 무리하게 사용하지 않는다.

### 보류할 모델

- mixed-effect logistic
  - 데이터 구조와 계산 안정성이 충분히 확인될 때만 수행한다.
  - 무리하게 넣으면 해석과 수렴 문제가 커질 수 있다.

- conditional logistic regression
  - 매칭 세트 ID가 명확하고 조건부 로지스틱 구조가 필요한 경우에만 수행한다.

### 산출물

- 필요 시에만 생성한다.
- 후보 파일명:
  - `stage16_logistic_extensions.py`
  - `outputs/stage16_logistic_extensions_summary.md`
  - `outputs/tables/stage16_extension_comparison.csv`

### 판단 기준

- Step 16은 필수 단계가 아니다.
- Step 13~15에서 해석과 통계 검정이 충분하면 Step 16은 생략 가능하다.
- 추가 모델이 성능 또는 해석 안정성을 명확히 개선하지 않으면 최종 보고서에 넣지 않는다.

---

## 다음 세션 시작 시 권장 첫 작업

1. `outputs/logistic_benchmark_for_model_comparison.csv`가 존재하는지 확인한다.
2. `stage11_final_logistic_benchmark.py`를 새로 작성한다.
3. Step 11 산출물을 만든다.
4. 이어서 실제 학습데이터 컬럼명을 확인하고 Step 12의 변수 매핑표를 만든다.

### 주의사항

- 요인점수는 사용하지 않는다.
- 비통계적 이진분류 모델은 추가하지 않는다.
- lockbox는 사용하지 않는다.
- strict `date_exposure_component_cv` 기준을 유지한다.
- 기존 EDA인 `jsw/final_eda.md`와 연결되는 방향으로 해석한다.
- 성능 모델과 해석 모델을 혼동하지 않는다.
