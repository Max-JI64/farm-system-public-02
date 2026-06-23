# Pole 산불 취약성 예측 계획

## 1. 문서 목적

이 문서는 `jsw/Analysis/new_machine_learning/머신러닝_학습_로그.md`의 STEP 5 내용을 독립적인 예측 실행 계획으로 옮긴 것이다.

예측 대상은 한전 전력설비 Pole이며, 목표는 특정 날짜의 실제 산불 발생확률을 계산하는 것이 아니라 `10월~5월 09~16시` 산불 위험관리 기간의 실제 기상 조건에서 각 Pole이 얼마나 반복적으로 높은 산불 취약도 score를 받는지 평가하는 것이다.

중요한 전제는 다음과 같다.

- 학습데이터 행을 Pole에 직접 붙여서 예측데이터를 만들지 않는다.
- Pole 예측데이터는 `data/예측데이터`의 Pole 위치자료와 원천 기상·공간·토지피복·캐나다 산불지수 자료에서 새로 생성한다.
- 다만 최종 LightGBM 학습에 사용한 변수명, 단위, rolling 방식, `log1p` 변환, 캐나다지수 as-of 정책, 토지피복 binary 정의는 학습데이터 생성 방식과 동일하게 맞춘다.
- 결과는 실제 발생확률이 아니라 top-risk 기반의 상대 취약성·점검 우선순위로 해석한다.

---

## 2. 현재 입력 데이터

현재 `data/예측데이터`에는 다음 Pole 자료가 있다.

| 파일 | 역할 |
| --- | --- |
| `gangwon_poles_4326.csv` | Pole 좌표 CSV |
| `gangwon_poles_4326.shp` | Pole 위치 shapefile 본체 |
| `gangwon_poles_4326.dbf` | shapefile 속성 테이블 |
| `gangwon_poles_4326.shx` | shapefile index |
| `gangwon_poles_4326.prj` | 좌표계 정의 |

우선 `shp/dbf/shx/prj` 세트를 원본 공간자료로 보고, CSV는 동일 내용을 저장한 보조 입력으로 사용한다. 두 자료의 `pole_id`, 좌표, 행 수가 다르면 shapefile을 기준으로 CSV를 재생성하거나 차이를 감사표에 남긴다.

예측 입력자료 품질검사에서 확인할 항목은 다음이다.

| 항목 | 확인 내용 |
| --- | --- |
| 좌표계 | EPSG:4326 여부 |
| 필수 열 | `pole_id`, `lon`, `lat` 또는 geometry |
| 중복 | `pole_id` 중복 여부 |
| 결측 | 좌표 결측, geometry 결측 |
| 공간 범위 | 강원도 경계 또는 분석 대상권 안에 있는지 |
| CSV-SHP 정합성 | CSV와 shapefile의 행 수·좌표 일치 |

---

## 3. 학습데이터와 Pole 예측데이터의 차이

학습데이터 생성 흐름은 다음과 같았다.

```text
산불 발생점/대조점 샘플
→ 해당 위치·시각의 기상·공간·토지피복·캐나다지수 결합
→ Target이 있는 모델 학습용 데이터 생성
```

Pole 예측데이터 생성 흐름은 다르다.

```text
Pole 위치 목록
→ 각 Pole에 공간·지형·토지피복·기상셀 매칭
→ 10월~5월 09~16시의 실제 기상 row 생성
→ Pole × 기준시각 입력행 생성
→ 최종 LightGBM score 산출
→ Pole 단위 취약성 요약
```

따라서 학습데이터는 예측 입력으로 재사용하지 않는다. 학습데이터는 다음 항목을 맞추기 위한 기준 계약서로만 사용한다.

| 학습데이터에서 재사용할 것 | 직접 재사용하지 않을 것 |
| --- | --- |
| 최종 58개 입력 변수명 | 학습 row 자체 |
| 변수 단위와 자료형 | Target, 샘플유형 |
| rolling 기상 피처 계산 방식 | 산불·대조 표본 위치 |
| 캐나다지수 as-of 병합 정책 | OOF 예측값 |
| 거리 변수 `log1p` 변환 방식 | CV fold, 모델링 그룹 |
| 토지피복 binary 정의 | 학습용 대조군 설계 |

---

## 4. 최종 예측 단위

최종 LightGBM의 입력 단위는 Pole 하나가 아니라 다음이다.

```text
pole_id × 기준시각
```

주 분석 모집단은 다음으로 정의한다.

```text
10월~5월의 사용 가능한 과거 날짜 × 09~16시 × Pole에 매칭된 기상셀의 실제 날씨 row
```

이 설정의 의미는 다음과 같다.

- 월 범위는 `10월~5월`로 제한한다.
- 시간대는 `09시, 10시, 11시, 12시, 13시, 14시, 15시, 16시`로 제한한다.
- 기상값은 임의 시나리오가 아니라 실제 과거 관측·파생 기상 row를 사용한다.
- 각 Pole은 자신에게 매칭된 기상셀의 날씨 시계열과 결합한다.
- 최종 Pole 취약성은 여러 시간별 score를 Pole 단위로 요약해 산출한다.

---

## 5. 최종 LightGBM 입력 피처 계약

Pole 예측 feature matrix는 최종 LightGBM이 사용한 58개 피처와 동일한 이름·의미·단위로 만들어야 한다.

### 5-1. 계절·시간·기후지형유형

| 변수 |
| --- |
| `기후지형유형` |
| `월_sin` |
| `월_cos` |
| `시간_sin` |
| `시간_cos` |

### 5-2. 습도·강수 핵심 변수

| 변수 |
| --- |
| `시점_습도_pct` |
| `직전24h_평균습도` |
| `직전24h_최소습도` |
| `직전48h_평균습도` |
| `직전48h_최소습도` |
| `D-1_최소습도_pct` |
| `D-1_평균습도_pct` |
| `D-2_최소습도_pct` |
| `D-3_최소습도_pct` |
| `직전24h_강수량합` |
| `직전48h_강수량합` |
| `D-1_강수량합_mm` |

### 5-3. 추가 기상 변수

| 변수 |
| --- |
| `시점_기온_C` |
| `직전24h_평균기온_C` |
| `시점_풍속_m_s` |
| `직전24h_평균풍속` |
| `직전24h_최대풍속` |
| `직전48h_평균풍속` |
| `직전48h_최대풍속` |
| `풍향_sin` |
| `풍향_cos` |
| `서풍계열_여부` |
| `시점_현지기압_hPa` |
| `시점_해면기압_hPa` |
| `기압변동_3h` |

### 5-4. 공간·지형·접근성 변수

| 변수 |
| --- |
| `log1p_도로_최단거리_m` |
| `log1p_시가화거리_m` |
| `log1p_산림지역_최단거리_m` |
| `log1p_농업거리_m` |
| `log1p_임도_최단거리_m` |
| `log1p_등산로거리_m` |
| `고도(m)` |
| `경사도(도)` |
| `TPI(지형위치지수)` |
| `사면방향_sin` |
| `사면방향_cos` |

### 5-5. 캐나다 산불지수

| 변수 |
| --- |
| `FFMC` |
| `FFMC_10일평균` |
| `DMC` |
| `DC` |
| `ISI` |
| `BUI` |
| `FWI` |

### 5-6. 토지피복 변수

| 변수 |
| --- |
| `토지피복_산림지역` |
| `토지피복_시가화건조지역` |
| `토지피복_농업지역` |
| `토지피복_초지` |
| `토지피복_나지` |
| `토지피복_도로` |
| `토지피복_활엽수림` |
| `토지피복_침엽수림` |
| `토지피복_혼효림` |
| `비산림_WUI_접경후보` |

다음 변수는 예측자료에 보존할 수 있지만 모델 입력에는 넣지 않는다.

| 메타 변수 | 용도 |
| --- | --- |
| `pole_id` | Pole 식별 |
| `lon`, `lat` | 위치 추적 |
| `기준시각` | 시간 추적 |
| `기상셀ID` | 기상 join 감사 |
| `캐나다지수_기준날짜` | 캐나다지수 as-of 감사 |
| `캐나다지수_정책` | 캐나다지수 병합 정책 감사 |

---

## 6. 데이터 생성 단계

### 6-1. 입력 Pole 자료 감사

첫 단계에서는 `data/예측데이터/gangwon_poles_4326.*`를 검사한다.

산출물:

```text
jsw/predict/outputs/input_audit__gangwon_poles.csv
```

검사 항목:

- CSV와 shapefile 행 수 비교
- `pole_id` 중복 수
- 좌표 결측 수
- 강원도 경계 밖 Pole 수
- CRS 확인
- 좌표 범위 최소·최대

### 6-2. Pole별 기상셀·기후지형유형 매칭

각 Pole을 학습데이터에서 사용한 기상셀 체계에 매칭한다.

원칙:

- 가능하면 기상셀 polygon 안에 포함되는 방식으로 매칭한다.
- polygon 경계자료가 없거나 포함 실패 시 가장 가까운 기상셀 중심점으로 매칭한다.
- 매칭 거리와 매칭 방식은 반드시 저장한다.
- 매칭된 기상셀의 `기후지형유형`을 Pole에 부여한다.

산출물:

```text
jsw/predict/outputs/pole_weather_cell_match.csv
```

필수 열:

| 열 | 설명 |
| --- | --- |
| `pole_id` | Pole ID |
| `lon` | 경도 |
| `lat` | 위도 |
| `기상셀ID` | 매칭된 기상셀 |
| `기후지형유형` | 영동 해안형·영서 내륙형·고지·산간형 |
| `weather_cell_match_method` | polygon 또는 nearest |
| `weather_cell_distance_m` | nearest 사용 시 거리 |

### 6-3. Pole 정적 공간 피처 생성

Pole 위치에서 공간·지형·토지피복 피처를 만든다.

동일성 원칙:

- DEM에서 고도, 경사도, 사면방향, TPI를 계산할 때 학습데이터 생성 코드의 계산 방식과 동일하게 한다.
- 단, 예측용 Pole 전체 적용에서는 DEM 3×3 window 계산이 실패한 위치가 생길 수 있다. 이 경우 학습데이터의 기본 산출 로직을 우선 적용하되, 실패한 Pole에 한해 가장 가까운 유효 DEM 3×3 window를 찾아 `고도(m)`, `경사도(도)`, `사면방향_sin`, `사면방향_cos`, `TPI(지형위치지수)`를 보완한다.
- DEM 보완 여부는 숨기지 않고 `DEM_보완방식`, `DEM_보완거리_m`으로 저장한다.
- DEM 보완은 정적 지형 피처에만 적용하며, 토지피복 미매칭은 임의 보완하지 않고 `미상` 및 binary 0 상태로 유지한다.
- 거리 변수는 원거리 원본값을 만든 뒤 학습과 동일하게 `log1p` 변환한다.
- 토지피복 binary 변수는 학습데이터의 정의와 동일하게 만든다.
- `비산림_WUI_접경후보`도 학습데이터와 같은 규칙으로 만든다.

산출물:

```text
jsw/predict/outputs/pole_static_features.csv
```

주요 열:

| 열 |
| --- |
| `pole_id` |
| `기상셀ID` |
| `기후지형유형` |
| `고도(m)` |
| `경사도(도)` |
| `사면방향_sin` |
| `사면방향_cos` |
| `TPI(지형위치지수)` |
| `log1p_도로_최단거리_m` |
| `log1p_시가화거리_m` |
| `log1p_산림지역_최단거리_m` |
| `log1p_농업거리_m` |
| `log1p_임도_최단거리_m` |
| `log1p_등산로거리_m` |
| 토지피복 binary 10개 |
| `DEM_보완방식` |
| `DEM_보완거리_m` |

추가 산출물:

```text
jsw/predict/outputs/pole_static_features_model_ready.csv
jsw/predict/outputs/pole_static_feature_audit.csv
jsw/predict/outputs/pole_static_feature_dem_fallback_summary.csv
```

현재 전체 Pole 실행 기준 정적 피처 검증 결과는 다음과 같다.

| 항목 | 결과 |
| --- | ---: |
| 전체 Pole 수 | 1,387,831 |
| 모델 입력 가능 Pole 수 | 1,387,831 |
| 정적 피처 결측 Pole 수 | 0 |
| DEM nearest fallback 적용 Pole 수 | 9,863 |
| DEM 미해결 Pole 수 | 0 |
| 기상셀 누락 | 0 |
| 기후지형유형 누락 | 0 |
| 토지피복 미매칭 | 67,499 |

토지피복 미매칭은 모델 입력 결측은 아니지만, 해석과 지도화 단계에서는 `토지피복_매칭방식`, `토지피복_L1_NAME`, `토지피복_L2_NAME`을 함께 확인한다.

### 6-4. Pole용 시간기상 모집단 생성

학습데이터 row가 아니라 시간단위 기상 원천 또는 학습용 시간기상 파생자료에서 Pole 적용용 날씨 모집단을 만든다.

주 분석 필터:

```text
월: 10월~5월
시간: 09~16시
단위: 기상셀ID × 기준시각
```

산출물:

```text
jsw/predict/outputs/weather_population_10to5_09to16.csv
```

이 파일은 기상셀별 전체 후보 날씨 row이며 Pole을 아직 붙이지 않은 상태다.

필수 열:

| 열 |
| --- |
| `기상셀ID` |
| `기준시각` |
| `월_sin` |
| `월_cos` |
| `시간_sin` |
| `시간_cos` |
| 기상 피처 전체 |

검증:

- 10월~5월 외 월 없음
- 09~16시 외 시간 없음
- `기상셀ID × 기준시각` 중복 없음
- rolling 피처 결측 확인
- 기상셀별 row 수 분포 확인

### 6-5. 캐나다 산불지수 결합

캐나다 산불지수는 학습데이터와 동일한 as-of 정책으로 붙인다.

사용 파일:

```text
data/학습데이터/캐나다_FFMC_선행연구_일단위.csv
data/학습데이터/캐나다_FWI_일단위.csv
```

정책:

| 기준시각 | 사용할 캐나다지수 기준날짜 |
| --- | --- |
| 09~11시 | 전날 정오 기준 지수 |
| 12~16시 | 당일 정오 기준 지수 |

생성 열:

| 열 |
| --- |
| `캐나다지수_기준날짜` |
| `캐나다지수_정책` |
| `FFMC` |
| `FFMC_10일평균` |
| `DMC` |
| `DC` |
| `ISI` |
| `BUI` |
| `FWI` |

검증:

- 캐나다지수 결측 0
- 09~11시는 기준날짜가 기준시각 날짜보다 1일 이전
- 12~16시는 기준날짜가 기준시각 날짜와 동일
- 기상셀ID+기준날짜 중복 없음

### 6-6. Pole × 기준시각 feature matrix 생성

정적 Pole 피처와 날씨 모집단을 `기상셀ID`로 결합한다.

개념:

```text
pole_static_features
JOIN weather_population_10to5_09to16
ON 기상셀ID
```

결과 단위:

```text
pole_id × 기준시각
```

주의:

- 전체 feature matrix가 매우 클 수 있으므로 한 번에 모두 만들지 않는다.
- 기상셀, 연도, 월 또는 날짜 chunk 단위로 나눠 생성한다.
- 최종 예측 score와 Pole별 요약에 필요한 최소 열만 저장한다.
- 모델 입력 58개 피처가 완성되지 않은 row는 예측하지 않고 결측 감사표에 남긴다.

산출물:

```text
jsw/predict/outputs/pole_feature_matrix_sample.csv
jsw/predict/outputs/feature_matrix_audit.csv
```

전체 feature matrix는 크기가 크면 chunk 파일 또는 parquet로 저장한다.

---

## 7. 모델 적용 계획

### 7-1. 모델 artifact 선택

현재 STEP 2~4의 성능은 5-fold OOF 결과다. 실제 Pole 예측을 위해서는 배포용 모델 artifact가 필요하다.

STEP 2의 최종 후보 `TUNE_LGBM_ALL_ALL_LC_NONE`은 하나의 단일 하이퍼파라미터가 아니라 outer fold별로 선택된 5개 하이퍼파라미터 세트를 가진다.

```text
jsw/Analysis/new_machine_learning/outputs/step2_tuned_single_models/
selected_params__TUNE_LGBM_ALL_ALL_LC_NONE.json
```

따라서 배포용 모델 artifact는 다음 두 방식 중 하나로 만든다.

| 방식 | 설명 | 판단 |
| --- | --- | --- |
| `single_median_params` | 5개 outer fold 하이퍼파라미터 세트의 중앙값 대표값을 만들고, 전체 development 17,045행으로 LightGBM 1개를 재학습 | 기본값. 계산량이 현실적이고 배포 해석이 단순함 |
| `five_param_ensemble` | 5개 outer fold 하이퍼파라미터 세트를 각각 전체 development 17,045행으로 재학습하고, 5개 LightGBM score 평균 사용 | OOF 구조에 더 충실하지만 전체 Pole 예측 계산량이 약 5배 증가 |

전체 Pole 예측은 기본적으로 `single_median_params`를 사용한다. 전체 score row 수가 약 53.7억 개이므로 5개 모델 평균은 계산량 측면에서 부담이 크다. 5개 모델 평균을 민감도 분석으로 별도 수행하려면 `--model-mode five_param_ensemble`을 명시한다.

기본 배포 모델 캐시는 다음 규칙으로 저장한다.

```text
jsw/predict/outputs/deployment_lgbm__single_median_params.joblib
```

5개 모델 평균을 사용할 때는 다음 이름을 사용한다.

```text
jsw/predict/outputs/deployment_lgbm__five_param_ensemble.joblib
```

운영 threshold는 STEP 3에서 선정한 F2 최대점 threshold를 사용한다.

```text
best_f2 threshold = 0.0199793962581399
```

단, 이 threshold는 Pole별 최종 등급을 직접 나누는 절대 위험 기준이 아니라 `f2_threshold_exceed_rate` 계산용 보조 기준이다. Pole별 최종 주 순위는 `p95_score`를 사용한다.

### 7-2. score 산출

각 chunk별로 최종 LightGBM 58개 피처를 모델에 넣어 score를 계산한다. 하지만 전체 `Pole × 기준시각` row를 CSV로 저장하지 않는다.

현재 입력 규모는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| 전체 Pole 수 | 1,387,831 |
| 날씨 모집단 row 수 | 356,224 |
| 기상셀 수 | 92 |
| 예상 score row 수 | 5,373,681,632 |

따라서 구현 원칙은 다음과 같다.

- `pole_static_features_model_ready.csv`와 `weather_population_10to5_09to16.csv`를 `기상셀ID` 기준으로 결합한다.
- 전체 feature matrix를 한 번에 만들지 않는다.
- `기상셀ID → Pole chunk → weather chunk` 순서로 나눠 score를 계산한다.
- 각 Pole chunk에 대해 시간별 score matrix를 만들고, 즉시 Pole별 요약값을 계산한다.
- 원시 `Pole × 기준시각` score 전체 파일은 기본 생성하지 않는다.
- 중간 산출은 checkpoint part로 저장하고, 최종 단계에서 part를 병합해 Pole별 요약 파일을 만든다.

산출물:

```text
jsw/predict/outputs/pole_vulnerability_summary_parts__single_median_params/
jsw/predict/outputs/pole_vulnerability_summary.csv
jsw/predict/outputs/pole_vulnerability_groups.csv
jsw/predict/outputs/pole_scoring_audit.csv
jsw/predict/outputs/run_manifest__04_score_pole_time_rows.json
```

checkpoint part의 주요 열:

| 열 | 설명 |
| --- | --- |
| `pole_id` | Pole ID |
| `기상셀ID` | 매칭된 기상셀 |
| `mean_score` | 해당 Pole의 전체 분석 기간 평균 score |
| `p90_score` | 해당 Pole의 score 90분위수 |
| `p95_score` | 해당 Pole의 score 95분위수 |
| `p99_score` | 해당 Pole의 score 99분위수 |
| `max_score` | 해당 Pole의 최대 score |
| `f2_threshold_exceed_count` | F2 threshold 이상 시간 row 수 |
| `f2_threshold_exceed_rate` | F2 threshold 이상 시간 row 비율 |
| `scored_time_rows` | 해당 Pole에 적용된 시간 row 수 |

`score`는 실제 발생확률로 표현하지 않는다. 현재 모델은 calibration을 최종 보고 기준으로 사용하지 않으므로, score는 상대적 산불 취약도 점수로 해석한다.

### 7-3. checkpoint와 재시작 정책

전체 scoring은 장시간 실행될 수 있으므로 checkpoint/resume을 기본으로 사용한다.

checkpoint 단위:

```text
기상셀ID + Pole chunk
```

예시 파일명:

```text
part__YS_0001__pole_00000000_00001000.csv
```

재시작 규칙:

- 이미 존재하는 checkpoint part는 재계산하지 않고 재사용한다.
- 오류나 중단이 발생하면 같은 명령을 다시 실행한다.
- `--reuse-model-cache`를 붙이면 배포용 LightGBM 모델도 다시 학습하지 않고 재사용한다.
- 강제로 checkpoint를 무시하고 다시 계산하려면 `--no-resume-checkpoints`를 명시한다.

터미널 로그에는 다음 진행 정보가 출력된다.

```text
[cell 1/92] YS_0001 | pole=62,321 | weather=3,872 | rows=241,306,912 | elapsed=...
  [chunk 1/...] 계산 시작 | pole=0:1000 | score_rows=3,872,000
    weather 20/61 | chunk_rows=... | global_progress~...% | elapsed=...
  [chunk 1/...] 저장 완료 | completed_rows=.../5,373,681,632 (...%) | elapsed=...
```

checkpoint 재사용 시에는 다음처럼 출력된다.

```text
[chunk 1/...] checkpoint 재사용 | completed_rows=... (...%)
```

### 7-4. 실행 명령

전체 실행 전 규모 확인:

```powershell
& 'C:\Program Files\Python313\python.exe' 'jsw\predict\04_score_pole_time_rows.py' --estimate-only
```

전체 기본 실행:

```powershell
& 'C:\Program Files\Python313\python.exe' 'jsw\predict\04_score_pole_time_rows.py' --confirm-full-run
```

중단 후 재시작:

```powershell
& 'C:\Program Files\Python313\python.exe' 'jsw\predict\04_score_pole_time_rows.py' --confirm-full-run --reuse-model-cache
```

5개 LightGBM 평균 score를 사용할 때:

```powershell
& 'C:\Program Files\Python313\python.exe' 'jsw\predict\04_score_pole_time_rows.py' --confirm-full-run --model-mode five_param_ensemble
```

checkpoint를 무시하고 처음부터 다시 계산할 때:

```powershell
& 'C:\Program Files\Python313\python.exe' 'jsw\predict\04_score_pole_time_rows.py' --confirm-full-run --no-resume-checkpoints
```

---

## 8. Pole 단위 취약성 요약

Pole별 시간 score는 다음 지표로 요약한다.

| 지표 | 정의 | 역할 |
| --- | --- | --- |
| `p95_score` | Pole별 10월~5월 09~16시 score의 95분위수 | 최종 취약성 주 순위 |
| `p90_score` | Pole별 score의 90분위수 | 보조 고위험 조건 |
| `f2_threshold_exceed_rate` | F2 threshold 이상 row 비율 | 반복 위험군 여부 |
| `top10_repeat_rate` | 같은 기준시각 내 top 10% 포함 비율 | 반복 우선점검 후보 |
| `mean_score` | Pole별 평균 score | 평균적 위험 보조 설명 |
| `max_score` | Pole별 최대 score | 최악 조건 참고 |

최종 주 순위는 `p95_score`로 정한다.

이유:

- 10월~5월에도 비 오는 날, 습한 날, 상대적으로 저위험인 날이 포함된다.
- 평균 score는 고위험 조건에서 취약한 Pole의 신호를 희석할 수 있다.
- 전력설비 취약성 평가는 평상시 평균 위험보다 고위험 조건에서 반복적으로 높아지는 Pole을 찾는 목적에 가깝다.

등급:

| 등급 | 정의 |
| --- | --- |
| 매우 높음 | `p95_score` top 5% |
| 높음 | top 5~10% |
| 중간 | top 10~20% |
| 낮음 | 나머지 |
| 반복 고위험 | `f2_threshold_exceed_rate` 상위권 |

`top 5%`, `top 10%`, `top 20%`는 절대 위험 임계값이 아니라 우선점검 등급이다. STEP 3 top-risk 분석에서 최종 LightGBM의 OOF score 상위 구간이 실제 산불을 강하게 농축했기 때문에, Pole 예측에서도 top-risk 방식으로 우선순위를 정한다.

산출물:

```text
jsw/predict/outputs/pole_vulnerability_summary.csv
jsw/predict/outputs/pole_vulnerability_groups.csv
```

---

## 9. 검증 계획

### 9-1. 입력 검증

| 검증 항목 | 기준 |
| --- | --- |
| Pole ID | 중복 0 |
| 좌표 | 결측 0 |
| CRS | EPSG:4326 확인 또는 변환 |
| CSV/SHP 정합성 | 행 수·좌표 차이 보고 |
| 공간 범위 | 강원도 또는 분석 대상권 안 |

### 9-2. 정적 피처 검증

| 검증 항목 | 기준 |
| --- | --- |
| 기상셀 매칭 | 모든 Pole에 `기상셀ID` 존재 |
| 기후지형유형 | 모든 Pole에 값 존재 |
| DEM 피처 | 고도·경사·사면방향·TPI 결측 점검 |
| 거리 피처 | 음수 없음, `log1p` 변환 확인 |
| 토지피복 | binary 변수 0/1 범위 확인 |

### 9-3. 날씨 모집단 검증

| 검증 항목 | 기준 |
| --- | --- |
| 월 필터 | 10월~5월만 존재 |
| 시간 필터 | 09~16시만 존재 |
| 중복 | `기상셀ID × 기준시각` 중복 0 |
| rolling 피처 | 핵심 rolling 변수 결측 점검 |
| 캐나다지수 | 결측 0 |
| as-of 정책 | 09~11시 D-1, 12~16시 당일 |

### 9-4. 모델 입력 검증

| 검증 항목 | 기준 |
| --- | --- |
| 피처 수 | 최종 LightGBM 58개 피처 모두 존재 |
| 이름 | 학습 피처명과 완전 일치 |
| 자료형 | 수치형·범주형 처리 일치 |
| 결측 | 모델 입력 NaN·inf 없음 또는 처리 규칙 명시 |
| score | 0~1 범위, NaN·inf 없음 |
| 모델 artifact | `single_median_params` 또는 `five_param_ensemble` 모드 명시 |
| threshold | STEP 3 F2 threshold `0.0199793962581399` 사용 여부 확인 |

### 9-5. 결과 검증

| 검증 항목 | 기준 |
| --- | --- |
| row 수 | 예상 score row 수와 완료 score row 수 일치 |
| checkpoint | expected chunk 수와 completed chunk 수 일치 |
| resume | 중단 후 재실행 시 기존 checkpoint part 재사용 |
| Pole 집계 | 모든 Pole에 요약 결과 존재 |
| 등급 | top 5%, 10%, 20% 개수 검증 |
| score 분포 | STEP 3 OOF score와 Pole score 분포 비교 |
| 고위험 후보 | 상위 Pole의 공간 분포와 기상셀 편중 확인 |

---

## 10. 구현 파일 계획

`jsw/predict` 아래에 다음 스크립트를 만든다.

| 파일 | 역할 |
| --- | --- |
| `01_audit_pole_inputs.py` | CSV/SHP 입력 감사 |
| `02_build_pole_static_features.py` | 기상셀·기후지형유형·공간·토지피복 피처 생성 |
| `03_build_weather_population.py` | 10월~5월 09~16시 기상·캐나다지수 모집단 생성 |
| `04_score_pole_time_rows.py` | chunk 단위 feature matrix 생성, LightGBM score 산출, checkpoint 저장, Pole별 p95·threshold 초과율·top-risk 등급 생성 |
| `05_summarize_pole_vulnerability.py` | 선택 사항. 04 결과를 이용한 추가 지도화·보고서용 표·시각화 생성 |

대용량 중간 산출물은 `jsw/predict/outputs/` 아래에 저장한다. 전체 `Pole × 기준시각` score raw table은 기본 저장하지 않고, checkpoint part와 Pole별 요약만 저장한다.

---

## 11. 최종 산출물

| 산출물 | 내용 |
| --- | --- |
| `input_audit__gangwon_poles.csv` | Pole 입력자료 감사 |
| `pole_weather_cell_match.csv` | Pole별 기상셀·기후지형유형 매칭 |
| `pole_static_features.csv` | Pole별 공간·지형·토지피복 피처 |
| `weather_population_10to5_09to16.csv` | 기상셀×기준시각 날씨 모집단 |
| `weather_population_audit.csv` | 날씨 row 수, 결측, 필터 검증 |
| `weather_population_incomplete_rows.csv` | 날씨·캐나다지수 결측으로 제거된 row 목록 |
| `pole_time_score_row_estimate.csv` | 기상셀별 Pole 수, 날씨 row 수, 예상 score row 수 |
| `pole_vulnerability_summary_parts__single_median_params/` | checkpoint part 파일 |
| `pole_vulnerability_summary.csv` | Pole별 p95, p90, 초과율, 반복 top-risk |
| `pole_vulnerability_groups.csv` | top 5%, 10%, 20% 등급 |
| `pole_scoring_audit.csv` | score row 완료 수, checkpoint 완료 수, summary row 수 검증 |
| `deployment_lgbm__single_median_params.joblib` | 기본 배포용 단일 LightGBM 모델 artifact |
| `run_manifest__04_score_pole_time_rows.json` | 입력자료, 모델 모드, threshold, checkpoint, 실행시간 |

---

## 12. 보고서 해석 문장

최종 보고서에서는 다음처럼 해석한다.

> Pole 산불 취약성 평가는 학습데이터 행을 재사용해 산출한 값이 아니라, Pole 위치자료와 원천 기상·공간·토지피복·캐나다 산불지수 자료에서 학습과 동일한 방식으로 생성한 `Pole × 기준시각` feature matrix에 최종 LightGBM을 적용한 결과이다. 본 분석은 10월~5월 09~16시의 실제 과거 기상 조건을 사용하며, Pole별 최종 취약성은 평균 score가 아니라 고위험 조건을 대표하는 `p95_score`와 반복 고위험 비율로 요약한다. 이 값은 실제 산불 발생확률이 아니라 산불 위험관리 기간의 상대적 점검 우선순위로 해석한다.
