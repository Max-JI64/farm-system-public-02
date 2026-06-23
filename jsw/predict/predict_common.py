from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "data").exists() and (candidate / "jsw").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
DATA_DIR = ROOT / "data"
PREDICT_DIR = ROOT / "jsw" / "predict"
OUTPUT_DIR = PREDICT_DIR / "outputs"

PREDICT_DATA_DIR = DATA_DIR / "예측데이터"
WEATHER_DIR = DATA_DIR / "강원도_날씨데이터"
SPATIAL_DIR = DATA_DIR / "강원도_데이터" / "강원도_공간데이터"
TRAIN_DATA_DIR = DATA_DIR / "학습데이터"

POLE_CSV = PREDICT_DATA_DIR / "gangwon_poles_4326.csv"
POLE_SHP = PREDICT_DATA_DIR / "gangwon_poles_4326.shp"
WEATHER_HOURLY_RAW = WEATHER_DIR / "강원도날씨_격자_시간단위.csv"
WEATHER_GRID = WEATHER_DIR / "강원도날씨_격자.geojson"
CLIMATE_TYPE = WEATHER_DIR / "강원도날씨_기후지형유형_셀분류.csv"

FFMC_DAILY_PATH = TRAIN_DATA_DIR / "캐나다_FFMC_선행연구_일단위.csv"
FWI_DAILY_PATH = TRAIN_DATA_DIR / "캐나다_FWI_일단위.csv"


CANADIAN_INDEX_COLUMNS = [
    "FFMC",
    "FFMC_10일평균",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI",
]

WEATHER_REQUIRED_COLUMNS = [
    "시점_기온_C",
    "직전24h_평균기온_C",
    "시점_풍속_m_s",
    "시점_습도_pct",
    "직전24h_평균습도",
    "직전24h_최소습도",
    "직전48h_평균습도",
    "직전48h_최소습도",
    "D-1_최소습도_pct",
    "D-1_평균습도_pct",
    "D-2_최소습도_pct",
    "D-3_최소습도_pct",
    "직전24h_강수량합",
    "직전48h_강수량합",
    "D-1_강수량합_mm",
    "직전24h_평균풍속",
    "직전24h_최대풍속",
    "직전48h_평균풍속",
    "직전48h_최대풍속",
    "풍향_sin",
    "풍향_cos",
    "서풍계열_여부",
    "시점_현지기압_hPa",
    "시점_해면기압_hPa",
    "기압변동_3h",
    "월_sin",
    "월_cos",
    "시간_sin",
    "시간_cos",
]

STATIC_REQUIRED_COLUMNS = [
    "기후지형유형",
    "log1p_도로_최단거리_m",
    "log1p_시가화거리_m",
    "log1p_산림지역_최단거리_m",
    "log1p_농업거리_m",
    "log1p_임도_최단거리_m",
    "log1p_등산로거리_m",
    "고도(m)",
    "경사도(도)",
    "TPI(지형위치지수)",
    "사면방향_sin",
    "사면방향_cos",
    "토지피복_산림지역",
    "토지피복_시가화건조지역",
    "토지피복_농업지역",
    "토지피복_초지",
    "토지피복_나지",
    "토지피복_도로",
    "토지피복_활엽수림",
    "토지피복_침엽수림",
    "토지피복_혼효림",
    "비산림_WUI_접경후보",
]

FINAL_LGBM_FEATURES = [
    "기후지형유형",
    "월_sin",
    "월_cos",
    "시간_sin",
    "시간_cos",
    "시점_습도_pct",
    "직전24h_평균습도",
    "직전24h_최소습도",
    "직전48h_평균습도",
    "직전48h_최소습도",
    "D-1_최소습도_pct",
    "D-1_평균습도_pct",
    "D-2_최소습도_pct",
    "D-3_최소습도_pct",
    "직전24h_강수량합",
    "직전48h_강수량합",
    "D-1_강수량합_mm",
    "시점_기온_C",
    "직전24h_평균기온_C",
    "시점_풍속_m_s",
    "직전24h_평균풍속",
    "직전24h_최대풍속",
    "직전48h_평균풍속",
    "직전48h_최대풍속",
    "풍향_sin",
    "풍향_cos",
    "서풍계열_여부",
    "시점_현지기압_hPa",
    "시점_해면기압_hPa",
    "기압변동_3h",
    "log1p_도로_최단거리_m",
    "log1p_시가화거리_m",
    "log1p_산림지역_최단거리_m",
    "log1p_농업거리_m",
    "log1p_임도_최단거리_m",
    "log1p_등산로거리_m",
    "고도(m)",
    "경사도(도)",
    "TPI(지형위치지수)",
    "사면방향_sin",
    "사면방향_cos",
    "FFMC",
    "FFMC_10일평균",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI",
    "토지피복_산림지역",
    "토지피복_시가화건조지역",
    "토지피복_농업지역",
    "토지피복_초지",
    "토지피복_나지",
    "토지피복_도로",
    "토지피복_활엽수림",
    "토지피복_침엽수림",
    "토지피복_혼효림",
    "비산림_WUI_접경후보",
]


def ensure_output_dir(path: Path = OUTPUT_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def add_month_hour_features(frame: pd.DataFrame, datetime_col: str = "기준시각") -> pd.DataFrame:
    out = frame.copy()
    dt = pd.to_datetime(out[datetime_col], errors="raise")
    month = dt.dt.month
    hour = dt.dt.hour
    out["월_sin"] = np.sin(2 * np.pi * month / 12)
    out["월_cos"] = np.cos(2 * np.pi * month / 12)
    out["시간_sin"] = np.sin(2 * np.pi * hour / 24)
    out["시간_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def is_main_prediction_period(dt: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dt, errors="raise")
    month = parsed.dt.month
    hour = parsed.dt.hour
    return ((month >= 10) | (month <= 5)) & hour.between(9, 16)


def add_refa_rolling_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """학습데이터 step3_weather_rolling.py와 같은 shift(1) 기반 rolling."""
    hourly = hourly.sort_values(["기상셀ID", "일시"]).reset_index(drop=True)
    shifted = hourly.groupby("기상셀ID", sort=False)[
        ["기온_C", "풍속_m_s", "습도_pct", "강수량_mm"]
    ].shift(1)
    grouped = shifted.groupby(hourly["기상셀ID"], sort=False)
    for window in [24, 48]:
        hourly[f"직전{window}h_평균풍속"] = (
            grouped["풍속_m_s"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        hourly[f"직전{window}h_최대풍속"] = (
            grouped["풍속_m_s"].rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
        )
        hourly[f"직전{window}h_평균기온_C"] = (
            grouped["기온_C"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        hourly[f"직전{window}h_평균습도"] = (
            grouped["습도_pct"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        hourly[f"직전{window}h_최소습도"] = (
            grouped["습도_pct"].rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
        )
        hourly[f"직전{window}h_강수량합"] = (
            grouped["강수량_mm"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        )
    return hourly


def build_hourly_weather_features(hourly_raw: pd.DataFrame) -> pd.DataFrame:
    """data/학습데이터/step3_weather_rolling.py 로직을 예측용으로 재사용."""
    required = [
        "기상셀ID",
        "일시",
        "기온_C",
        "풍속_m_s",
        "풍향_deg",
        "강수량_mm",
        "현지기압_hPa",
        "해면기압_hPa",
        "습도_pct",
    ]
    missing = sorted(set(required) - set(hourly_raw.columns))
    if missing:
        raise KeyError(f"원천 시간기상 누락 컬럼: {missing}")

    hourly = hourly_raw[required].copy()
    hourly["일시"] = pd.to_datetime(hourly["일시"], errors="raise")
    hourly["날짜"] = hourly["일시"].dt.floor("D")

    hourly["풍향_sin"] = np.sin(np.radians(hourly["풍향_deg"]))
    hourly["풍향_cos"] = np.cos(np.radians(hourly["풍향_deg"]))
    hourly["서풍계열_여부"] = np.where(hourly["풍향_deg"].between(202.5, 292.5), 1, 0)

    hourly = hourly.sort_values(["기상셀ID", "일시"]).reset_index(drop=True)
    grouped_for_press = hourly.groupby("기상셀ID", sort=False)["현지기압_hPa"]
    hourly["기압변동_3h"] = hourly["현지기압_hPa"] - grouped_for_press.shift(3)

    hourly = add_refa_rolling_weather(hourly)

    hourly_day = hourly.groupby(["기상셀ID", "날짜"], as_index=False).agg(
        평균습도_pct=("습도_pct", "mean"),
        최소습도_pct=("습도_pct", "min"),
        강수량합_mm=("강수량_mm", "sum"),
    )

    hourly_features = hourly[
        [
            "기상셀ID",
            "일시",
            "날짜",
            "기온_C",
            "풍속_m_s",
            "습도_pct",
            "직전24h_평균풍속",
            "직전24h_최대풍속",
            "직전48h_평균풍속",
            "직전48h_최대풍속",
            "직전24h_평균기온_C",
            "직전24h_평균습도",
            "직전24h_최소습도",
            "직전48h_평균습도",
            "직전48h_최소습도",
            "직전24h_강수량합",
            "직전48h_강수량합",
            "풍향_deg",
            "풍향_sin",
            "풍향_cos",
            "서풍계열_여부",
            "현지기압_hPa",
            "해면기압_hPa",
            "기압변동_3h",
        ]
    ].copy()

    hourly_features = hourly_features.rename(
        columns={
            "일시": "기준시각",
            "기온_C": "시점_기온_C",
            "풍속_m_s": "시점_풍속_m_s",
            "습도_pct": "시점_습도_pct",
            "풍향_deg": "시점_풍향_deg",
            "현지기압_hPa": "시점_현지기압_hPa",
            "해면기압_hPa": "시점_해면기압_hPa",
        }
    )

    for lag in [1, 2, 3]:
        lag_df = hourly_day.copy()
        lag_df["날짜"] = lag_df["날짜"] + pd.Timedelta(days=lag)
        rename_cols = {
            "평균습도_pct": f"D-{lag}_평균습도_pct",
            "최소습도_pct": f"D-{lag}_최소습도_pct",
            "강수량합_mm": f"D-{lag}_강수량합_mm",
        }
        lag_df = lag_df.rename(columns=rename_cols)
        keep = ["기상셀ID", "날짜", f"D-{lag}_최소습도_pct"]
        if lag == 1:
            keep.extend([f"D-{lag}_평균습도_pct", f"D-{lag}_강수량합_mm"])
        hourly_features = hourly_features.merge(
            lag_df[keep], on=["기상셀ID", "날짜"], how="left"
        )

    hourly_features = hourly_features.drop(columns=["날짜"])
    hourly_features = add_month_hour_features(hourly_features, "기준시각")
    return hourly_features


def load_daily_canadian_indices() -> pd.DataFrame:
    ffmc = read_csv(FFMC_DAILY_PATH)
    fwi = read_csv(FWI_DAILY_PATH)
    for name, frame in [("FFMC", ffmc), ("FWI", fwi)]:
        frame["날짜"] = pd.to_datetime(frame["날짜"], errors="raise").dt.normalize()
        duplicated = int(frame.duplicated(["기상셀ID", "날짜"]).sum())
        if duplicated:
            raise ValueError(f"{name} 일단위 캐나다지수 중복: {duplicated}")

    ffmc_check = ffmc[["기상셀ID", "날짜", "FFMC"]].merge(
        fwi[["기상셀ID", "날짜", "FFMC"]],
        on=["기상셀ID", "날짜"],
        how="inner",
        suffixes=("_FFMC트랙", "_FWI트랙"),
        validate="one_to_one",
    )
    max_diff = (
        ffmc_check["FFMC_FFMC트랙"] - ffmc_check["FFMC_FWI트랙"]
    ).abs().max()
    if float(max_diff) > 1e-9:
        raise ValueError(f"FFMC 트랙 간 값이 다릅니다: max_abs_diff={max_diff}")

    daily = ffmc[["기상셀ID", "날짜", "FFMC", "FFMC_10일평균"]].merge(
        fwi[["기상셀ID", "날짜", "DMC", "DC", "ISI", "BUI", "FWI"]],
        on=["기상셀ID", "날짜"],
        how="inner",
        validate="one_to_one",
    )
    return daily


def attach_canadian_indices(frame: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    data = frame.copy().reset_index(drop=True)
    data["기준시각"] = pd.to_datetime(data["기준시각"], errors="raise")

    before_noon = data["기준시각"].dt.hour.lt(12)
    before_noon_mask = before_noon.to_numpy()
    reference_date = data["기준시각"].dt.normalize()
    reference_date = reference_date.where(~before_noon, reference_date - pd.Timedelta(days=1))
    data["캐나다지수_기준날짜"] = reference_date
    data["캐나다지수_정책"] = "latest_available_noon_LST"

    daily_for_merge = daily.rename(columns={"날짜": "캐나다지수_기준날짜"})
    data = data.merge(
        daily_for_merge,
        on=["기상셀ID", "캐나다지수_기준날짜"],
        how="left",
        validate="many_to_one",
    )

    current_date = data["기준시각"].dt.normalize()
    day_gap = (current_date - data["캐나다지수_기준날짜"]).dt.days
    audit = {
        "rows": int(len(data)),
        "before_noon_n": int(before_noon_mask.sum()),
        "at_or_after_noon_n": int((~before_noon_mask).sum()),
        "before_noon_bad_reference_n": int((~day_gap.iloc[before_noon_mask].eq(1)).sum()),
        "at_or_after_noon_bad_reference_n": int((~day_gap.iloc[~before_noon_mask].eq(0)).sum()),
        "canadian_missing_cells": int(data[CANADIAN_INDEX_COLUMNS].isna().any(axis=1).sum()),
    }
    return data, audit


def validation_frame(items: list[tuple[str, Any, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"검증항목": name, "결과": result, "통과": "예" if passed else "아니오"} for name, result, passed in items]
    )
