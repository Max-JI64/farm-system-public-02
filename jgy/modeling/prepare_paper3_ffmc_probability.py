from __future__ import annotations

"""논문3 방식의 FFMC 기반 산불발생확률 임시 예측.

논문3:
    "캐나다 산불 기상지수를 이용한 산불발생확률모형 개발
     - 강원도 지역 산불발생을 중심으로 -"

논문3의 핵심:
    1. 기온, 습도, 풍속, 강수량으로 FFMC(Fine Fuel Moisture Code)를 계산
    2. 봄철 산불조심기간의 10일 평균 FFMC를 4단계 Index로 변환
    3. 로지스틱 회귀식으로 산불발생확률 계산

논문 로지스틱 식:
    logit(P) = -0.529 + 0.422 * Indexed_FFMC
    P = exp(logit) / (1 + exp(logit))

Index 기준:
    0.00  ~ 58.00  -> 1
    58.01 ~ 70.50  -> 2
    70.51 ~ 80.00  -> 3
    80.01 ~ 99.00  -> 4

출력:
    data/modeling/paper3_ffmc_gangwon_occurrence_probability.csv

주의:
    이 결과는 논문3 식을 강원도 현재 데이터에 적용한 "임시 발생확률"입니다.
    정식 검증 모델이 아니라, FFMC 기반 위험 조건을 보기 위한 참고용 결과입니다.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "modeling"
DAILY_FEATURE_PATH = OUT_DIR / "gangwon_weather_daily_model_features.csv"
OUT_PATH = OUT_DIR / "paper3_ffmc_gangwon_occurrence_probability.csv"


def calculate_ffmc_day(
    prev_ffmc: float,
    temp_c: float,
    rh_pct: float,
    wind_kmh: float,
    rain_mm: float,
) -> float:
    """Calculate daily FFMC with the Van Wagner-style CFFDRS formula."""
    ffmc = float(np.clip(prev_ffmc, 0, 101))
    temp_c = float(temp_c)
    rh_pct = float(np.clip(rh_pct, 0, 100))
    wind_kmh = float(max(wind_kmh, 0))
    rain_mm = float(max(rain_mm, 0))

    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)

    if rain_mm > 0.5:
        rf = rain_mm - 0.5
        rain_effect = 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        if mo > 150:
            rain_effect += 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf)
        mo = min(mo + rain_effect, 250.0)

    ed = (
        0.942 * (rh_pct**0.679)
        + 11.0 * np.exp((rh_pct - 100.0) / 10.0)
        + 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh_pct))
    )

    if mo < ed:
        ew = (
            0.618 * (rh_pct**0.753)
            + 10.0 * np.exp((rh_pct - 100.0) / 10.0)
            + 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh_pct))
        )
        if mo <= ew:
            kl = 0.424 * (1.0 - ((100.0 - rh_pct) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (
                1.0 - ((100.0 - rh_pct) / 100.0) ** 8
            )
            kw = kl * 0.581 * np.exp(0.0365 * temp_c)
            m = ew - (ew - mo) * np.exp(-kw)
        else:
            m = mo
    else:
        kl = 0.424 * (1.0 - (rh_pct / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (
            1.0 - (rh_pct / 100.0) ** 8
        )
        kw = kl * 0.581 * np.exp(0.0365 * temp_c)
        m = ed + (mo - ed) * np.exp(-kw)

    new_ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return float(np.clip(new_ffmc, 0, 101))


def ffmc_index(ffmc_10day_mean: pd.Series) -> pd.Series:
    return pd.cut(
        ffmc_10day_mean,
        bins=[-np.inf, 58.0, 70.5, 80.0, np.inf],
        labels=[1, 2, 3, 4],
    ).astype("int8")


def logistic_probability(indexed_ffmc: pd.Series) -> pd.Series:
    logit = -0.529 + 0.422 * indexed_ffmc.astype(float)
    return 1.0 / (1.0 + np.exp(-logit))


def build_probability() -> pd.DataFrame:
    df = pd.read_csv(DAILY_FEATURE_PATH, encoding="utf-8-sig")
    df = df.sort_values(["기상셀ID", "날짜"]).copy()

    df["temp_c_for_ffmc"] = df["daily_temp_mean_C"]
    df["rh_pct_for_ffmc"] = df["hourly_humidity_mean_pct"]
    df["wind_kmh_for_ffmc"] = df["hourly_wind_mean_m_s"] * 3.6
    df["rain_mm_for_ffmc"] = df["daily_rain_mm"].fillna(df["hourly_rain_sum_mm"]).fillna(0)

    ffmc_values = []
    for _, group in df.groupby("기상셀ID", sort=False):
        prev = 85.0
        for row in group.itertuples(index=False):
            prev = calculate_ffmc_day(
                prev_ffmc=prev,
                temp_c=getattr(row, "temp_c_for_ffmc"),
                rh_pct=getattr(row, "rh_pct_for_ffmc"),
                wind_kmh=getattr(row, "wind_kmh_for_ffmc"),
                rain_mm=getattr(row, "rain_mm_for_ffmc"),
            )
            ffmc_values.append(prev)

    df["ffmc"] = ffmc_values
    df["ffmc_10day_mean"] = (
        df.groupby("기상셀ID")["ffmc"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["indexed_ffmc"] = ffmc_index(df["ffmc_10day_mean"])
    df["paper3_occurrence_probability"] = logistic_probability(df["indexed_ffmc"]).round(4)

    out_cols = [
        "기상셀ID",
        "날짜",
        "기후권역",
        "기후지형유형",
        "temp_c_for_ffmc",
        "rh_pct_for_ffmc",
        "wind_kmh_for_ffmc",
        "rain_mm_for_ffmc",
        "ffmc",
        "ffmc_10day_mean",
        "indexed_ffmc",
        "paper3_occurrence_probability",
    ]
    return df[out_cols]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = build_probability()
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"created={OUT_PATH}")
    print(f"shape={result.shape}")
    print(f"date_range={result['날짜'].min()}..{result['날짜'].max()}")
    print(result["indexed_ffmc"].value_counts().sort_index().to_string())
    print(result["paper3_occurrence_probability"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
