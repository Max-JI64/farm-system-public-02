from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage9_logistic_enhancement import add_stage9_features


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "학습데이터" / "학습데이터_로지스틱_D2D3.csv").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "학습데이터"
LOGISTIC_DIR = ROOT / "jsw" / "Analysis" / "logistic"
OUTPUT_DIR = LOGISTIC_DIR / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
TABLE_DIR = OUTPUT_DIR / "tables"
SPLIT_DIR = OUTPUT_DIR / "splits"

DATA_PATH = DATA_DIR / "학습데이터_로지스틱_D2D3.csv"
ENGINEERED_PATH = FEATURE_DIR / "stage7_engineered_features.csv"
OUTER_PATH = SPLIT_DIR / "outer_cv_manifest.csv"

TARGET = "Target"
ID_COL = "샘플ID"

FA_FEATURES = ["F1_score", "F2_score", "F3_score", "F4_score", "F5_score"]

EXCLUDED_ALWAYS = {
    "샘플ID",
    "기준시각",
    "위도",
    "경도",
    "기상셀ID",
    "Target",
    "샘플유형",
    "실험안",
    "샘플가중치",
    "시간샘플링방식",
    "공간층",
    "기준날짜",
    "D1_지수기준날짜",
    "캐나다지수_정책",
    "fire_id",
    "원본_fire_id",
    "후보점ID",
    "source_fire_id",
    "모델링_그룹ID",
}


def ensure_dirs() -> None:
    for directory in [FEATURE_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def feature_spec(
    group: str,
    feature: str,
    role: str,
    feature_type: str,
    unit_for_or: str,
    eda_link: str,
    expected_direction: str,
    note: str = "",
) -> dict[str, str]:
    return {
        "concept_group": group,
        "feature": feature,
        "role": role,
        "feature_type": feature_type,
        "unit_for_or": unit_for_or,
        "eda_link": eda_link,
        "expected_direction": expected_direction,
        "note": note,
    }


def build_feature_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = [
        feature_spec(
            "층화/통제",
            "기후지형유형",
            "core",
            "categorical",
            "기준범주 대비",
            "EDA의 영동 해안형/영서 내륙형/고지·산간형 층화 축",
            "유형별 방향은 기준범주에 의존",
            "해석용 모델의 기본 층화 변수",
        ),
        feature_spec("시간 통제", "월_sin", "control", "numeric", "1 단위", "월별 발생 집중 보정", "통제변수", ""),
        feature_spec("시간 통제", "월_cos", "control", "numeric", "1 단위", "월별 발생 집중 보정", "통제변수", ""),
        feature_spec("시간 통제", "시간_sin", "control", "numeric", "1 단위", "시간대 집중 보정", "통제변수", ""),
        feature_spec("시간 통제", "시간_cos", "control", "numeric", "1 단위", "시간대 집중 보정", "통제변수", ""),
        feature_spec(
            "습도/건조",
            "직전24h_최소습도",
            "core",
            "numeric",
            "5%p 감소",
            "직전 24시간 최소습도 저하가 가장 안정적인 EDA 신호",
            "습도 감소 시 산불 odds 증가 예상",
            "Step13에서 OR은 감소 방향으로 재표현",
        ),
        feature_spec(
            "습도/건조",
            "직전24h_평균습도",
            "secondary",
            "numeric",
            "5%p 감소",
            "직전 24시간 평균습도 저하",
            "습도 감소 시 산불 odds 증가 예상",
            "최소습도와 공선성 확인 필요",
        ),
        feature_spec(
            "습도/건조",
            "D-1_최소습도_pct",
            "secondary",
            "numeric",
            "5%p 감소",
            "전일 최소습도 저하",
            "습도 감소 시 산불 odds 증가 예상",
            "직전24h_최소습도와 대체 후보",
        ),
        feature_spec(
            "습도/건조",
            "rh_local_q05",
            "core",
            "binary",
            "조건 충족 vs 미충족",
            "국지·시간대 기준 하위 5% 상대건조",
            "충족 시 odds 증가 예상",
            "고정 절대습도보다 안정적인 EDA 신호",
        ),
        feature_spec(
            "습도/건조",
            "rh_minus_local_q05",
            "diagnostic",
            "numeric",
            "5%p 감소",
            "국지 하위 5% 기준 대비 습도 차이",
            "값이 낮을수록 odds 증가 예상",
            "rh_local_q05와 동시 투입 시 공선성 주의",
        ),
        feature_spec(
            "습도/건조",
            "시점_습도_pct",
            "diagnostic_not_core",
            "numeric",
            "5%p 감소",
            "발생 시점 상태 기술값",
            "습도 감소 시 odds 증가 예상",
            "EDA에서 발화 전 예측변수보다 상태 기술값으로 해석",
        ),
        feature_spec(
            "강수/무강수",
            "직전48h_강수량합",
            "core",
            "numeric",
            "5mm 감소",
            "48~72시간 강수 결핍 신호",
            "강수량 감소 시 odds 증가 예상",
            "직전24h보다 누적 결핍 쪽을 우선",
        ),
        feature_spec(
            "강수/무강수",
            "직전24h_강수량합",
            "secondary",
            "numeric",
            "1mm 감소",
            "직전24시간 강수 결핍",
            "강수량 감소 시 odds 증가 예상",
            "직전48h와 공선성 확인 필요",
        ),
        feature_spec(
            "강수/무강수",
            "D-1_강수량합_mm",
            "secondary",
            "numeric",
            "1mm 감소",
            "전일 강수 결핍",
            "강수량 감소 시 odds 증가 예상",
            "직전24h 강수와 대체 후보",
        ),
        feature_spec(
            "강수/무강수",
            "dry_spell_0p1_gt_24h",
            "core",
            "binary",
            "조건 충족 vs 미충족",
            "0.1mm 기준 24시간 초과 무강수 지속",
            "충족 시 odds 증가 예상",
            "EDA에서 OR이 안정적으로 남은 무강수 조건",
        ),
        feature_spec(
            "강수/무강수",
            "dry_spell_5p0_gt_240h",
            "core",
            "binary",
            "조건 충족 vs 미충족",
            "5mm 이상 젖힘 강수 후 10일 초과",
            "충족 시 odds 증가 예상",
            "장기 재건조 후보",
        ),
        feature_spec(
            "강수/무강수",
            "dry_spell_h_0p1",
            "diagnostic",
            "numeric",
            "24시간 증가",
            "0.1mm 기준 무강수 지속시간",
            "증가 시 odds 증가 예상",
            "이진 조건과 대체 후보",
        ),
        feature_spec(
            "강수/무강수",
            "dry_spell_h_5p0",
            "diagnostic",
            "numeric",
            "24시간 증가",
            "5mm 기준 무강수 지속시간",
            "증가 시 odds 증가 예상",
            "이진 조건과 대체 후보",
        ),
        feature_spec(
            "풍속/풍향",
            "wind_max_6h",
            "core",
            "numeric",
            "1m/s 증가",
            "6시간 최대풍속, 영동 해안형에서 강한 신호",
            "증가 시 odds 증가 예상",
            "직전24h 최대풍속보다 EDA 결합조건과 연결이 쉬움",
        ),
        feature_spec(
            "풍속/풍향",
            "wind_mean_6h",
            "secondary",
            "numeric",
            "1m/s 증가",
            "6시간 평균풍속",
            "증가 시 odds 증가 예상",
            "wind_max_6h와 공선성 확인 필요",
        ),
        feature_spec(
            "풍속/풍향",
            "직전24h_최대풍속",
            "secondary",
            "numeric",
            "1m/s 증가",
            "직전24시간 최대풍속",
            "증가 시 odds 증가 예상",
            "wind_max_6h와 대체 후보",
        ),
        feature_spec(
            "풍속/풍향",
            "서풍계열_여부",
            "core",
            "binary",
            "서풍계열 vs 기타",
            "영동 해안형 서풍계열 강풍 후보",
            "서풍계열에서 odds 증가 가능",
            "단독보다 영동×풍속 상호작용에서 중요",
        ),
        feature_spec(
            "풍속/풍향",
            "westerly_at_wind_max_6h",
            "secondary",
            "binary",
            "조건 충족 vs 미충족",
            "6시간 최대풍속 시점의 서풍계열 여부",
            "충족 시 odds 증가 가능",
            "현재 서풍계열과 대체 후보",
        ),
        feature_spec(
            "풍속/풍향",
            "wind_max_6h_ge_5",
            "secondary",
            "binary",
            "조건 충족 vs 미충족",
            "6시간 최대풍속 5m/s 이상",
            "충족 시 odds 증가 예상",
            "국지저습과 결합할 때 우선",
        ),
        feature_spec(
            "기압/기온",
            "기압변동_3h",
            "core_control",
            "numeric",
            "1hPa 변화",
            "약한 기압 하강 이상성",
            "하강 방향에서 odds 증가 가능",
            "보조 프록시",
        ),
        feature_spec(
            "기압/기온",
            "시점_기온_C",
            "control",
            "numeric",
            "1°C 증가",
            "기온 효과는 매칭 후 0에 수렴",
            "명확한 방향 없음",
            "계절·시간대 보정 통제 성격",
        ),
        feature_spec(
            "기압/기온",
            "직전24h_평균기온_C",
            "control",
            "numeric",
            "1°C 증가",
            "기온 효과는 계절 구성 영향이 큼",
            "명확한 방향 없음",
            "시점기온과 공선성 확인 필요",
        ),
        feature_spec(
            "공간/접근성",
            "log1p_도로거리_m",
            "core",
            "numeric",
            "log1p 거리 1 증가 또는 10m 증가 재표현",
            "도로 초접경 편향이 공간 EDA의 가장 강한 신호",
            "거리 증가 시 odds 감소 예상",
            "원거리 꼬리 때문에 log1p 사용",
        ),
        feature_spec(
            "공간/접근성",
            "도로_10m_이내",
            "core_interaction",
            "binary",
            "10m 이내 vs 초과",
            "WUI 발생지의 약 72%가 도로 10m 이내",
            "충족 시 odds 증가 예상",
            "log 거리와 동시 투입 시 공선성 주의",
        ),
        feature_spec(
            "공간/접근성",
            "도로_30m_이내",
            "diagnostic",
            "binary",
            "30m 이내 vs 초과",
            "도로 인접성 민감도",
            "충족 시 odds 증가 예상",
            "10m 기준의 보조 민감도",
        ),
        feature_spec(
            "공간/접근성",
            "log1p_시가화거리_m",
            "core",
            "numeric",
            "log1p 거리 1 증가",
            "시가화 접경 편향",
            "거리 증가 시 odds 감소 예상",
            "",
        ),
        feature_spec(
            "공간/접근성",
            "log1p_산림거리_m",
            "core",
            "numeric",
            "log1p 거리 1 증가",
            "산불 발생지는 산림 내부보다 접경부에 치우침",
            "거리 증가는 맥락 의존",
            "산림 내부성 해석에는 주의",
        ),
        feature_spec(
            "공간/접근성",
            "비산림_WUI_접경후보",
            "excluded_artifact",
            "binary",
            "사용하지 않음",
            "EDA의 WUI 결론과 이름은 비슷하지만 학습데이터의 Target 1에는 동일하게 적용되지 않음",
            "해석하지 않음",
            "Target 1의 공간층이 실제발생위치라 이 플래그가 Target 1에서 모두 0이 되는 구조적 산출물",
        ),
        feature_spec(
            "공간/접근성",
            "log1p_임도거리_m",
            "diagnostic",
            "numeric",
            "log1p 거리 1 증가",
            "임도는 보조 신호",
            "명확한 방향 약함",
            "핵심 모델에는 넣지 않음",
        ),
        feature_spec(
            "공간/접근성",
            "log1p_등산로거리_m",
            "diagnostic",
            "numeric",
            "log1p 거리 1 증가",
            "등산로는 보조 신호",
            "명확한 방향 약함",
            "핵심 모델에는 넣지 않음",
        ),
        feature_spec(
            "지형",
            "고도(m)",
            "core",
            "numeric",
            "100m 증가",
            "같은 공간층 안에서도 저고도 편향",
            "고도 증가 시 odds 감소 가능",
            "기후지형유형과 함께 해석",
        ),
        feature_spec(
            "지형",
            "경사도(도)",
            "core",
            "numeric",
            "5도 증가",
            "완경사·하부사면 편향",
            "경사 증가 시 odds 감소 가능",
            "",
        ),
        feature_spec(
            "지형",
            "TPI(지형위치지수)",
            "core",
            "numeric",
            "1 단위 증가",
            "하부 사면/계곡부 편향",
            "증가 시 odds 감소 가능",
            "공간층별 방향 차이 주의",
        ),
        feature_spec(
            "토지피복",
            "토지피복_L1_NAME",
            "core",
            "categorical",
            "기준범주 대비",
            "WUI 내부에서도 비산림 접경 피복 편향",
            "범주별 방향 다름",
            "주요 토지피복 대분류",
        ),
        feature_spec(
            "토지피복",
            "토지피복_산림유형",
            "core",
            "categorical",
            "기준범주 대비",
            "활엽수림·침엽수림·혼효림 구분",
            "범주별 방향 다름",
            "D3 임상도보다 안정적인 간이 산림유형",
        ),
        feature_spec(
            "토지피복",
            "토지피복_시가화건조지역",
            "secondary",
            "binary",
            "조건 충족 vs 미충족",
            "시가화건조지역 접경 편향",
            "충족 시 odds 증가 가능",
            "L1_NAME과 대체 후보",
        ),
        feature_spec(
            "토지피복",
            "토지피복_초지",
            "secondary",
            "binary",
            "조건 충족 vs 미충족",
            "초지/비산림 접경 편향",
            "충족 시 odds 증가 가능",
            "L1_NAME과 대체 후보",
        ),
        feature_spec(
            "토지피복",
            "토지피복_산림지역",
            "secondary",
            "binary",
            "조건 충족 vs 미충족",
            "산림 내부보다 접경부 편향",
            "방향은 대조군 구성에 의존",
            "L1_NAME과 대체 후보",
        ),
        feature_spec(
            "캐나다 산불지수",
            "D1_FFMC",
            "canada_core",
            "numeric",
            "5점 증가",
            "D-1 정오 미세연료 건조도",
            "증가 시 odds 증가 예상",
            "미래 정보 누수 방지를 위해 D-1만 사용",
        ),
        feature_spec(
            "캐나다 산불지수",
            "D1_DMC",
            "canada_core",
            "numeric",
            "10점 증가",
            "중간 깊이 연료층 누적 건조도",
            "증가 시 odds 증가 예상",
            "D1_BUI/DC와 공선성 확인 필요",
        ),
        feature_spec(
            "캐나다 산불지수",
            "D1_ISI",
            "canada_core",
            "numeric",
            "1점 증가",
            "초기확산지수, FFMC와 풍속 결합",
            "증가 시 odds 증가 예상",
            "wind 변수와 공선성 확인 필요",
        ),
        feature_spec(
            "캐나다 산불지수",
            "D1_FWI",
            "canada_core",
            "numeric",
            "5점 증가",
            "최종 화재기상지수",
            "증가 시 odds 증가 예상",
            "종합지수로 해석성은 높지만 하위지수와 중복 가능",
        ),
        feature_spec(
            "캐나다 산불지수",
            "D1_DC",
            "canada_diagnostic",
            "numeric",
            "100점 증가",
            "장기 가뭄지수",
            "증가 시 odds 증가 예상",
            "DMC/BUI/FWI와 공선성 확인 필요",
        ),
        feature_spec(
            "캐나다 산불지수",
            "D1_BUI",
            "canada_diagnostic",
            "numeric",
            "10점 증가",
            "누적연료건조도",
            "증가 시 odds 증가 예상",
            "DMC/DC/FWI와 공선성 확인 필요",
        ),
        feature_spec(
            "상호작용",
            "영동_x_wind_max_6h",
            "interaction_core",
            "numeric",
            "1m/s 증가",
            "영동 해안형 강풍 주도형",
            "증가 시 odds 증가 예상",
            "",
        ),
        feature_spec(
            "상호작용",
            "rh_local_q05_AND_wind_max_6h_ge_5",
            "interaction_core",
            "binary",
            "조건 충족 vs 미충족",
            "국지저습 × 6시간 최대풍속 5m/s 이상",
            "충족 시 odds 증가 예상",
            "전체에서 안정적으로 남은 복합 후보",
        ),
        feature_spec(
            "상호작용",
            "rh_local_q05_AND_westerly_strong_max_6h",
            "interaction_core",
            "binary",
            "조건 충족 vs 미충족",
            "영동 서풍계열 건조강풍 후보",
            "충족 시 odds 증가 예상",
            "특정 국지풍 직접 증거로 과해석 금지",
        ),
        feature_spec(
            "상호작용",
            "비산림WUI_x_도로10m",
            "excluded_artifact",
            "binary",
            "사용하지 않음",
            "비산림_WUI_접경후보 기반 상호작용",
            "해석하지 않음",
            "기반 플래그가 Target 1에서 모두 0이므로 해석용 모델에서 제외",
        ),
        feature_spec(
            "상호작용",
            "시가화_x_도로10m",
            "interaction_core",
            "binary",
            "조건 충족 vs 미충족",
            "시가화건조지역 × 도로 10m 초접경",
            "충족 시 odds 증가 예상",
            "비산림 WUI 대체 프록시로 사용",
        ),
        feature_spec(
            "상호작용",
            "초지_x_dry0p1",
            "interaction_core",
            "binary",
            "조건 충족 vs 미충족",
            "초지 × 0.1mm 기준 24시간 초과 무강수",
            "충족 시 odds 증가 예상",
            "비산림 접경 건조 조건의 대체 프록시",
        ),
        feature_spec(
            "상호작용",
            "영동_x_rh_q05_x_wind5",
            "interaction_secondary",
            "binary",
            "조건 충족 vs 미충족",
            "영동 × 국지저습 × 강풍",
            "충족 시 odds 증가 예상",
            "희소성 확인 필요",
        ),
        feature_spec(
            "상호작용",
            "영서_x_비산림WUI_x_dry0p1",
            "excluded_artifact",
            "binary",
            "사용하지 않음",
            "영서 내륙형 × 비산림 WUI × 단기 무강수",
            "해석하지 않음",
            "비산림_WUI_접경후보 기반이므로 해석용 모델에서 제외",
        ),
    ]

    for feature in FA_FEATURES:
        specs.append(
            feature_spec(
                "요인점수",
                feature,
                "excluded",
                "numeric",
                "사용하지 않음",
                "요인점수 파트 제외 결정",
                "해석하지 않음",
                "최종 보고서에서 요인점수 파트 제외",
            )
        )
    return specs


def load_modeling_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["기준시각"], low_memory=False)
    engineered = pd.read_csv(ENGINEERED_PATH, encoding="utf-8-sig", low_memory=False)
    drop_engineered = {"Target", "샘플유형"}
    engineered_cols = [
        col
        for col in engineered.columns
        if col not in drop_engineered and not col.startswith(f"{ID_COL}.")
    ]
    engineered = engineered[engineered_cols].copy()
    data = data.merge(engineered, on=ID_COL, how="left", validate="one_to_one")
    data = add_stage9_features(data)

    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    development_ids = set(outer[ID_COL])
    development = data.loc[data[ID_COL].isin(development_ids)].copy()
    if len(development) != len(outer):
        raise ValueError(f"development 행 수 불일치: data={len(development)}, outer={len(outer)}")
    return data, development


def make_feature_mapping(specs: list[dict[str, str]], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in specs:
        feature = spec["feature"]
        exists = feature in df.columns
        row: dict[str, Any] = dict(spec)
        row["exists"] = exists
        row["missing_n"] = int(df[feature].isna().sum()) if exists else np.nan
        row["missing_rate"] = float(df[feature].isna().mean()) if exists else np.nan
        row["unique_n"] = int(df[feature].nunique(dropna=True)) if exists else np.nan
        row["include_weather_space"] = feature in INTERPRET_WEATHER_SPACE
        row["include_weather_space_canada"] = feature in INTERPRET_WEATHER_SPACE_CANADA
        row["include_eda_interactions"] = feature in INTERPRET_EDA_INTERACTIONS
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_summary(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    quantiles = numeric.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "min": float(numeric.min()) if numeric.notna().any() else np.nan,
        "q01": float(quantiles.loc[0.01]) if numeric.notna().any() else np.nan,
        "q05": float(quantiles.loc[0.05]) if numeric.notna().any() else np.nan,
        "q25": float(quantiles.loc[0.25]) if numeric.notna().any() else np.nan,
        "median": float(quantiles.loc[0.5]) if numeric.notna().any() else np.nan,
        "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
        "q75": float(quantiles.loc[0.75]) if numeric.notna().any() else np.nan,
        "q95": float(quantiles.loc[0.95]) if numeric.notna().any() else np.nan,
        "q99": float(quantiles.loc[0.99]) if numeric.notna().any() else np.nan,
        "max": float(numeric.max()) if numeric.notna().any() else np.nan,
        "std": float(numeric.std()) if numeric.notna().any() else np.nan,
    }


def make_distribution_table(mapping: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = pd.to_numeric(df[TARGET], errors="coerce")

    for _, meta in mapping.loc[mapping["exists"].eq(True)].iterrows():
        feature = meta["feature"]
        series = df[feature]
        row: dict[str, Any] = {
            "feature": feature,
            "concept_group": meta["concept_group"],
            "role": meta["role"],
            "feature_type": meta["feature_type"],
            "n": int(len(series)),
            "missing_n": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_n": int(series.nunique(dropna=True)),
        }

        if meta["feature_type"] in {"numeric", "binary"}:
            numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            row.update(numeric_summary(numeric))
            row["target1_mean"] = float(numeric[target.eq(1)].mean())
            row["target0_mean"] = float(numeric[target.eq(0)].mean())
            row["target1_median"] = float(numeric[target.eq(1)].median())
            row["target0_median"] = float(numeric[target.eq(0)].median())
            if meta["feature_type"] == "binary":
                row["target1_rate"] = row["target1_mean"]
                row["target0_rate"] = row["target0_mean"]
        else:
            text = series.fillna("결측").astype(str)
            counts = text.value_counts(dropna=False)
            top_value = counts.index[0] if len(counts) else ""
            row["top_value"] = top_value
            row["top_count"] = int(counts.iloc[0]) if len(counts) else 0
            row["top_rate"] = float(counts.iloc[0] / len(text)) if len(text) else np.nan
            row["target1_top_value"] = (
                text[target.eq(1)].value_counts(dropna=False).index[0]
                if len(text[target.eq(1)].value_counts(dropna=False))
                else ""
            )
            row["target0_top_value"] = (
                text[target.eq(0)].value_counts(dropna=False).index[0]
                if len(text[target.eq(0)].value_counts(dropna=False))
                else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_categorical_levels(mapping: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categorical_features = mapping.loc[
        mapping["exists"].eq(True) & mapping["feature_type"].eq("categorical"), "feature"
    ].tolist()
    for feature in categorical_features:
        temp = df[[feature, TARGET]].copy()
        temp[feature] = temp[feature].fillna("결측").astype(str)
        grouped = temp.groupby(feature, dropna=False)[TARGET].agg(["count", "sum", "mean"]).reset_index()
        grouped = grouped.rename(
            columns={
                feature: "level",
                "count": "n",
                "sum": "target1_n",
                "mean": "target1_rate",
            }
        )
        grouped.insert(0, "feature", feature)
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["feature", "level", "n", "target1_n", "target1_rate"])
    return pd.concat(rows, ignore_index=True)


INTERPRET_WEATHER_SPACE = [
    "기후지형유형",
    "월_sin",
    "월_cos",
    "시간_sin",
    "시간_cos",
    "직전24h_최소습도",
    "rh_local_q05",
    "직전48h_강수량합",
    "dry_spell_0p1_gt_24h",
    "dry_spell_5p0_gt_240h",
    "wind_max_6h",
    "서풍계열_여부",
    "기압변동_3h",
    "log1p_도로거리_m",
    "log1p_시가화거리_m",
    "log1p_산림거리_m",
    "고도(m)",
    "경사도(도)",
    "TPI(지형위치지수)",
    "토지피복_L1_NAME",
    "토지피복_산림유형",
]

CANADA_CORE = ["D1_FFMC", "D1_DMC", "D1_ISI", "D1_FWI"]

INTERPRET_WEATHER_SPACE_CANADA = INTERPRET_WEATHER_SPACE + CANADA_CORE

INTERACTION_CORE = [
    "영동_x_wind_max_6h",
    "rh_local_q05_AND_wind_max_6h_ge_5",
    "rh_local_q05_AND_westerly_strong_max_6h",
    "시가화_x_도로10m",
    "초지_x_dry0p1",
    "영동_x_rh_q05_x_wind5",
]

INTERPRET_EDA_INTERACTIONS = INTERPRET_WEATHER_SPACE_CANADA + INTERACTION_CORE


def validate_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    feature_sets = {
        "INTERPRET_WEATHER_SPACE": INTERPRET_WEATHER_SPACE,
        "INTERPRET_WEATHER_SPACE_CANADA": INTERPRET_WEATHER_SPACE_CANADA,
        "INTERPRET_EDA_INTERACTIONS": INTERPRET_EDA_INTERACTIONS,
    }
    missing_by_set = {
        name: [feature for feature in features if feature not in df.columns]
        for name, features in feature_sets.items()
    }
    missing_any = {name: missing for name, missing in missing_by_set.items() if missing}
    if missing_any:
        raise KeyError(f"해석용 변수셋에 없는 컬럼이 있습니다: {missing_any}")
    return feature_sets


def make_feature_set_long(feature_sets: dict[str, list[str]], mapping: pd.DataFrame) -> pd.DataFrame:
    meta = mapping.set_index("feature")
    rows = []
    for set_name, features in feature_sets.items():
        for order, feature in enumerate(features, start=1):
            rows.append(
                {
                    "feature_set": set_name,
                    "order": order,
                    "feature": feature,
                    "concept_group": meta.loc[feature, "concept_group"] if feature in meta.index else "",
                    "role": meta.loc[feature, "role"] if feature in meta.index else "",
                    "feature_type": meta.loc[feature, "feature_type"] if feature in meta.index else "",
                }
            )
    return pd.DataFrame(rows)


def make_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_outputs(
    full_data: pd.DataFrame,
    dev: pd.DataFrame,
    mapping: pd.DataFrame,
    distribution: pd.DataFrame,
    categorical_levels: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> None:
    ensure_dirs()

    feature_set_long = make_feature_set_long(feature_sets, mapping)

    mapping_path = TABLE_DIR / "stage12_feature_mapping.csv"
    distribution_path = TABLE_DIR / "stage12_missing_and_distribution.csv"
    category_path = TABLE_DIR / "stage12_categorical_levels.csv"
    feature_sets_path = FEATURE_DIR / "stage12_interpret_feature_sets.json"
    feature_set_long_path = TABLE_DIR / "stage12_interpret_feature_sets_long.csv"
    summary_path = OUTPUT_DIR / "stage12_interpret_feature_set_summary.md"

    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    distribution.to_csv(distribution_path, index=False, encoding="utf-8-sig")
    categorical_levels.to_csv(category_path, index=False, encoding="utf-8-sig")
    feature_set_long.to_csv(feature_set_long_path, index=False, encoding="utf-8-sig")

    categorical_features = sorted(
        set(
            mapping.loc[
                mapping["feature"].isin(sum(feature_sets.values(), []))
                & mapping["feature_type"].eq("categorical"),
                "feature",
            ]
        )
    )

    payload = {
        "version": "2026-06-20",
        "stage": "stage12_interpret_feature_set",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "engineered_features": str(ENGINEERED_PATH.relative_to(ROOT)),
        "split_policy": "strict date_exposure_component_cv development only",
        "target": TARGET,
        "id": ID_COL,
        "excluded_features": {
            "factor_scores": FA_FEATURES,
            "identifiers_or_leakage": sorted(EXCLUDED_ALWAYS),
            "forest_stand_detail": "D3 임상도 상세 변수는 매칭률이 낮아 Step12 핵심 해석 변수셋에서 제외",
        },
        "categorical_features": categorical_features,
        "feature_sets": feature_sets,
        "notes": [
            "성능 모델과 해석 모델을 분리한다.",
            "요인점수는 제외한다.",
            "시점_습도_pct는 상태 기술값으로만 보고 핵심 해석 모델에는 넣지 않는다.",
            "비산림_WUI_접경후보는 Target 1에서 모두 0이 되는 학습데이터 구조 산출물이므로 해석용 모델에서 제외한다.",
            "캐나다 지수는 D-1 정오 지수만 사용한다.",
            "Step13에서 이 변수셋으로 OR/CI/p/q-value를 산출한다.",
        ],
    }
    feature_sets_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    target_counts = dev[TARGET].value_counts().sort_index()
    missing_problem = mapping.loc[mapping["exists"].eq(False)]
    feature_set_summary = pd.DataFrame(
        [
            {
                "feature_set": name,
                "n_features": len(features),
                "n_categorical": sum(1 for f in features if f in categorical_features),
                "n_numeric_or_binary": len(features) - sum(1 for f in features if f in categorical_features),
            }
            for name, features in feature_sets.items()
        ]
    )
    role_summary = mapping.groupby(["concept_group", "role"], dropna=False).size().reset_index(name="n")

    top_missing = (
        mapping.loc[mapping["exists"].eq(True), ["feature", "concept_group", "role", "missing_rate"]]
        .sort_values("missing_rate", ascending=False)
        .head(10)
        .copy()
    )

    binary_preview_cols = [
        "feature",
        "concept_group",
        "role",
        "target1_mean",
        "target0_mean",
        "mean",
    ]
    binary_preview = distribution.loc[
        distribution["feature_type"].eq("binary")
        & distribution["role"].isin(["core", "core_interaction", "interaction_core"]),
        [col for col in binary_preview_cols if col in distribution.columns],
    ].copy()

    numeric_preview_cols = [
        "feature",
        "concept_group",
        "role",
        "target1_median",
        "target0_median",
        "median",
        "q05",
        "q95",
    ]
    numeric_preview = distribution.loc[
        distribution["feature_type"].eq("numeric")
        & distribution["role"].isin(["core", "core_control", "canada_core", "interaction_core"]),
        [col for col in numeric_preview_cols if col in distribution.columns],
    ].copy()

    lines = [
        "# Stage 12 해석용 로지스틱 변수셋 구성",
        "",
        "## 1. 목적",
        "",
        "- 성능 최고 모델의 ANOVA 선택 계수를 그대로 오즈비 해석에 사용하지 않기 위해 해석용 변수셋을 별도로 구성했다.",
        "- 기존 EDA 결론과 직접 연결되는 변수만 우선했다.",
        "- 요인점수는 제외했다.",
        "- 비통계적 이진분류 모델은 포함하지 않았다.",
        "- lockbox는 열지 않고 strict development 표본만 진단했다.",
        "",
        "## 2. 입력 데이터와 병합",
        "",
        f"- 기본 데이터: `{DATA_PATH.relative_to(ROOT)}`",
        f"- Stage7 파생피처: `{ENGINEERED_PATH.relative_to(ROOT)}`",
        f"- 전체 병합 후 행/열: {len(full_data):,}행 × {full_data.shape[1]:,}열",
        f"- development 표본: {len(dev):,}행",
        f"- Target 0: {int(target_counts.get(0, 0)):,}건",
        f"- Target 1: {int(target_counts.get(1, 0)):,}건",
        f"- development 양성 비율: {float(dev[TARGET].mean()):.4f}",
        "",
        "## 3. 해석용 변수셋",
        "",
        make_markdown_table(feature_set_summary),
        "",
        "### 변수셋 정의",
        "",
        "- `INTERPRET_WEATHER_SPACE`: 습도·강수·풍속·공간접근성·지형·토지피복 중심의 기본 해석 모델",
        "- `INTERPRET_WEATHER_SPACE_CANADA`: 기본 해석 모델에 D-1 캐나다 산불지수 핵심 후보를 추가",
        "- `INTERPRET_EDA_INTERACTIONS`: 캐나다 지수 포함 모델에 EDA 기반 제한적 상호작용을 추가",
        "",
        "## 4. 변수군/역할 요약",
        "",
        make_markdown_table(role_summary),
        "",
        "## 5. 결측 및 존재 여부",
        "",
    ]
    if len(missing_problem):
        lines += [
            "다음 후보는 실제 데이터에 없어 변수셋에서 사용할 수 없다.",
            "",
            make_markdown_table(missing_problem[["concept_group", "feature", "role"]]),
            "",
        ]
    else:
        lines += ["- Step12 후보 변수는 모두 실제 병합 데이터에 존재한다.", ""]

    lines += [
        "결측률 상위 후보:",
        "",
        make_markdown_table(top_missing.round(5)),
        "",
        "## 6. 핵심 이진 변수의 Target별 평균",
        "",
        make_markdown_table(binary_preview.round(5)),
        "",
        "## 7. 핵심 수치 변수의 Target별 중앙값",
        "",
        make_markdown_table(numeric_preview.round(5)),
        "",
        "## 8. 사용/제외 방침",
        "",
        "- 최종 성능 비교 모델과 해석용 모델을 분리한다.",
        "- 요인점수 `F1_score`~`F5_score`는 사용하지 않는다.",
        "- D3 임상도 상세 변수는 매칭률이 낮아 핵심 해석 변수셋에서 제외한다.",
        "- `시점_습도_pct`는 발생 시점 상태 기술값이므로 핵심 해석 모델에서는 제외하고 보조 진단으로만 둔다.",
        "- `공간층`은 Target 1에 `실제발생위치`가 들어 있는 구조라 모델 변수로 쓰지 않는다.",
        "- `비산림_WUI_접경후보`는 이 학습데이터에서 Target 1이 모두 0으로 계산되는 구조적 산출물이므로 해석용 모델에서 제외한다.",
        "- 캐나다 산불지수는 같은 날 12시 값이 아니라 D-1 정오 지수만 사용한다.",
        "- Step13에서는 이 변수셋으로 비정규화 또는 약한 정규화 로지스틱의 OR, 95% CI, p-value, FDR q-value를 산출한다.",
        "",
        "## 9. 산출물",
        "",
        f"- `{mapping_path.relative_to(ROOT)}`",
        f"- `{distribution_path.relative_to(ROOT)}`",
        f"- `{category_path.relative_to(ROOT)}`",
        f"- `{feature_sets_path.relative_to(ROOT)}`",
        f"- `{feature_set_long_path.relative_to(ROOT)}`",
        "",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    full_data, dev = load_modeling_frame()
    specs = build_feature_specs()
    feature_sets = validate_feature_sets(dev)
    mapping = make_feature_mapping(specs, dev)
    distribution = make_distribution_table(mapping, dev)
    categorical_levels = make_categorical_levels(mapping, dev)
    write_outputs(full_data, dev, mapping, distribution, categorical_levels, feature_sets)

    print("Stage12 완료")
    print(f"전체 병합 데이터: {full_data.shape[0]:,}행 × {full_data.shape[1]:,}열")
    print(f"development: {dev.shape[0]:,}행, 양성비율 {dev[TARGET].mean():.4f}")
    print(f"요약: {OUTPUT_DIR / 'stage12_interpret_feature_set_summary.md'}")


if __name__ == "__main__":
    main()
