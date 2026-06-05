from __future__ import annotations

"""논문2 방식의 산불 피해 규모 분류 모델 입력 데이터 생성.

논문2:
    "기계학습을 활용한 산불 피해 규모 예측: 기상 및 환경 변수를 중심으로"

논문2의 핵심은 산불 피해 규모를 소형/중형/대형 3개 클래스로 분류하는 것입니다.
비교 모델은 Random Forest, XGBoost, SVM이고, 가장 좋은 모델은 XGBoost입니다.

이 스크립트는 강원도 통합 시간단위 날씨 데이터를 논문2에서 사용한 변수 구조에
가깝게 변환합니다. 현재 프로젝트에는 SPI 원자료, 침엽수 비율, 피해 규모 타깃이
완전히 준비되어 있지 않으므로, 이 파일은 "학습 직전 입력 피처"를 만드는 단계입니다.

출력:
    data/modeling/paper2_xgboost_gangwon_weather_input.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = ROOT / "data" / "강원도_날씨데이터"
OUT_DIR = ROOT / "data" / "modeling"

INTEGRATED_WEATHER = WEATHER_DIR / "강원도날씨_통합_시간단위.csv"
OUT_PATH = OUT_DIR / "paper2_xgboost_gangwon_weather_input.csv"

CELL_ID = "기상셀ID"


def season_from_month(month: int) -> str:
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    if month in (9, 10, 11):
        return "가을"
    return "겨울"


def add_spi_like_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create simple monthly drought-index proxies from rainfall.

    논문2는 SPI1, SPI2, SPI3를 사용하지만, 현재 프로젝트에는 정식 SPI 데이터가
    없습니다. 그래서 월 강수량의 1/2/3개월 누적값을 셀별 z-score로 표준화해
    SPI에 가까운 대체 피처를 만듭니다. 최종 연구에서는 기상청/가뭄지수의 정식
    SPI 값을 쓰는 것이 맞습니다.
    """
    monthly = (
        features.assign(month_period=pd.to_datetime(features["date"]).dt.to_period("M"))
        .groupby([CELL_ID, "month_period"], as_index=False)["rain_sum_mm"]
        .sum()
        .sort_values([CELL_ID, "month_period"])
    )

    for window in (1, 2, 3):
        rolling_col = f"rain_{window}m_sum_mm"
        spi_col = f"spi{window}_proxy"
        monthly[rolling_col] = (
            monthly.groupby(CELL_ID)["rain_sum_mm"]
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        mean = monthly.groupby(CELL_ID)[rolling_col].transform("mean")
        std = monthly.groupby(CELL_ID)[rolling_col].transform("std").replace(0, np.nan)
        monthly[spi_col] = ((monthly[rolling_col] - mean) / std).fillna(0)

    spi_cols = [CELL_ID, "month_period", "spi1_proxy", "spi2_proxy", "spi3_proxy"]
    features = features.assign(month_period=pd.to_datetime(features["date"]).dt.to_period("M"))
    features = features.merge(monthly[spi_cols], on=[CELL_ID, "month_period"], how="left")
    return features.drop(columns=["month_period"])


def aggregate_weather() -> pd.DataFrame:
    usecols = [
        CELL_ID,
        "일시",
        "기온_C",
        "풍속_m_s",
        "습도_pct",
        "현지기압_hPa",
        "해면기압_hPa",
        "강수량_mm",
        "기후권역",
        "기후지형유형",
        "중심경도_wgs84",
        "중심위도_wgs84",
    ]
    weather = pd.read_csv(INTEGRATED_WEATHER, encoding="utf-8-sig", usecols=usecols)
    weather["datetime"] = pd.to_datetime(weather["일시"])
    weather["date"] = weather["datetime"].dt.strftime("%Y-%m-%d")
    weather["month"] = weather["datetime"].dt.month
    weather["dayofweek"] = weather["datetime"].dt.dayofweek
    weather["pressure_diff_hPa"] = weather["해면기압_hPa"] - weather["현지기압_hPa"]

    sort_cols = [CELL_ID, "date", "datetime"]
    weather = weather.sort_values(sort_cols)
    group_keys = [CELL_ID, "date"]

    for column in ["기온_C", "풍속_m_s", "습도_pct", "pressure_diff_hPa"]:
        weather[f"{column}_diff"] = weather.groupby(group_keys)[column].diff()

    features = (
        weather.groupby(group_keys, as_index=False)
        .agg(
            temp_mean_C=("기온_C", "mean"),
            temp_range_C=("기온_C", lambda s: s.max() - s.min()),
            temp_diff_mean_C=("기온_C_diff", "mean"),
            temp_abs_diff_mean_C=("기온_C_diff", lambda s: s.abs().mean()),
            wind_mean_m_s=("풍속_m_s", "mean"),
            wind_range_m_s=("풍속_m_s", lambda s: s.max() - s.min()),
            wind_diff_mean_m_s=("풍속_m_s_diff", "mean"),
            wind_abs_diff_mean_m_s=("풍속_m_s_diff", lambda s: s.abs().mean()),
            humidity_mean_pct=("습도_pct", "mean"),
            humidity_range_pct=("습도_pct", lambda s: s.max() - s.min()),
            humidity_diff_mean_pct=("습도_pct_diff", "mean"),
            humidity_abs_diff_mean_pct=("습도_pct_diff", lambda s: s.abs().mean()),
            pressure_diff_mean_hPa=("pressure_diff_hPa", "mean"),
            pressure_diff_range_hPa=("pressure_diff_hPa", lambda s: s.max() - s.min()),
            pressure_diff_delta_mean_hPa=("pressure_diff_hPa_diff", "mean"),
            pressure_diff_abs_delta_mean_hPa=("pressure_diff_hPa_diff", lambda s: s.abs().mean()),
            rain_sum_mm=("강수량_mm", "sum"),
            month=("month", "first"),
            dayofweek=("dayofweek", "first"),
        )
    )

    features["wind_change_ratio"] = features["wind_range_m_s"] / (features["wind_mean_m_s"].abs() + 1e-6)
    features["humidity_change_ratio"] = features["humidity_range_pct"] / (features["humidity_mean_pct"].abs() + 1e-6)
    features["pressure_change_ratio"] = features["pressure_diff_range_hPa"] / (
        features["pressure_diff_mean_hPa"].abs() + 1e-6
    )
    features["season"] = features["month"].map(season_from_month)

    meta_cols = [
        CELL_ID,
        "기후권역",
        "기후지형유형",
        "중심경도_wgs84",
        "중심위도_wgs84",
    ]
    meta = weather[meta_cols].drop_duplicates(CELL_ID)
    features = features.merge(meta, on=CELL_ID, how="left")
    features = add_spi_like_features(features)

    return features


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = aggregate_weather()
    features.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"created={OUT_PATH}")
    print(f"shape={features.shape}")
    print(f"date_range={features['date'].min()}..{features['date'].max()}")
    print("target=not included; need wildfire size label small/medium/large")


if __name__ == "__main__":
    main()
