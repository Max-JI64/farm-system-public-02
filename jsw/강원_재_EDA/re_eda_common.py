from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_ENCODING = "utf-8-sig"

DATA_PATHS = {
    "fire": REPO_ROOT / "data/강원도_데이터/강원도_산불발생.csv",
    "weather_cells": REPO_ROOT / "data/강원도_날씨데이터/강원도날씨_격자.csv",
    "weather_grid": REPO_ROOT / "data/강원도_날씨데이터/강원도날씨_격자.geojson",
    "weather_hourly_raw": REPO_ROOT / "data/강원도_날씨데이터/강원도날씨_격자_시간단위.csv",
    "weather_hourly_derived": REPO_ROOT / "data/학습데이터/기상_시간단위_파생.csv",
    "climate_type": REPO_ROOT / "data/강원도_날씨데이터/강원도날씨_기후지형유형_셀분류.csv",
    "canada_ffmc": REPO_ROOT / "data/학습데이터/캐나다_FFMC_선행연구_일단위.csv",
    "canada_fwi": REPO_ROOT / "data/학습데이터/캐나다_FWI_일단위.csv",
    "terrain": REPO_ROOT / "data/강원도_데이터/산불_공간데이터/강원도_산불_지형특성계산.csv",
    "dem": REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_DEM_데이터.tif",
    "landcover": REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_토지피복도_세분류_병합_1m.gpkg",
    "roads": REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_병합_도로.gpkg",
    "trails": REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_등산로.csv",
    "forest_roads": REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_임도망도.csv",
    "fire_stations": REPO_ROOT / "data/강원도_데이터/산불_공간데이터/강원도_근방_소방서_위치.csv",
    "fire_water": REPO_ROOT / "data/강원도_데이터/산불_공간데이터/강원도_근방_소방용수시설_위치.csv",
}

RAW_WEATHER_COLUMNS = [
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

DERIVED_WEATHER_COLUMNS = [
    "기상셀ID",
    "일시",
    "시점_기온_C",
    "시점_풍속_m_s",
    "시점_습도_pct",
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
    "시점_풍향_deg",
    "풍향_sin",
    "풍향_cos",
    "서풍계열_여부",
    "시점_현지기압_hPa",
    "시점_해면기압_hPa",
    "기압변동_3h",
    "D-1_최소습도_pct",
    "D-1_평균습도_pct",
    "D-1_강수량합_mm",
    "D-2_최소습도_pct",
    "D-3_최소습도_pct",
]


def configure_notebook() -> None:
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "DejaVu Sans"

    sns.set_theme(
        style="whitegrid",
        font=font_name,
        rc={
            "font.family": font_name,
            "font.sans-serif": [font_name],
            "axes.unicode_minus": False,
        },
    )
    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"matplotlib font.family: {plt.rcParams['font.family']}")


def check_sources(names: Iterable[str]) -> pd.DataFrame:
    records = []
    for name in names:
        if name not in DATA_PATHS:
            raise KeyError(f"등록되지 않은 데이터 키입니다: {name}")
        path = DATA_PATHS[name]
        records.append(
            {
                "데이터키": name,
                "존재": path.exists(),
                "크기_MB": round(path.stat().st_size / 1024**2, 2) if path.exists() else np.nan,
                "경로": str(path),
            }
        )
    result = pd.DataFrame(records)
    missing = result.loc[~result["존재"], "경로"].tolist()
    if missing:
        raise FileNotFoundError("필수 원천 파일이 없습니다:\n" + "\n".join(missing))
    return result


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    frame_name: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} 필수 컬럼 누락: {missing}")


def _read_csv(
    path_key: str,
    *,
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    path = DATA_PATHS[path_key]
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        encoding=CSV_ENCODING,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )


def load_grid_bundle() -> dict[str, pd.DataFrame]:
    cells = _read_csv("weather_cells")
    climate = _read_csv("climate_type")
    grid = gpd.read_file(DATA_PATHS["weather_grid"])

    require_columns(cells, ["기상셀ID", "기후권역", "중심경도_wgs84", "중심위도_wgs84"], "기상셀 메타")
    require_columns(climate, ["기상셀ID", "기후권역", "기후지형유형"], "기후지형유형")
    require_columns(grid, ["기상셀ID", "geometry"], "기상 격자")

    for frame in (cells, climate, grid):
        frame["기상셀ID"] = frame["기상셀ID"].astype("string")

    if cells["기상셀ID"].duplicated().any():
        raise ValueError("기상셀 메타에 중복 기상셀ID가 있습니다.")
    if climate["기상셀ID"].duplicated().any():
        raise ValueError("기후지형유형에 중복 기상셀ID가 있습니다.")
    if grid["기상셀ID"].duplicated().any():
        raise ValueError("기상 격자에 중복 기상셀ID가 있습니다.")

    return {"cells": cells, "climate": climate, "grid": grid}


def load_hourly_weather(
    *,
    derived: bool,
    columns: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    path_key = "weather_hourly_derived" if derived else "weather_hourly_raw"
    default_columns = DERIVED_WEATHER_COLUMNS if derived else RAW_WEATHER_COLUMNS
    selected = list(columns) if columns is not None else default_columns
    required = ["기상셀ID", "일시"]
    missing_keys = sorted(set(required) - set(selected))
    if missing_keys:
        raise ValueError(f"시간 기상 로딩 컬럼에는 {missing_keys}가 필요합니다.")

    weather = _read_csv(path_key, usecols=selected, nrows=nrows)
    require_columns(weather, selected, path_key)
    weather["기상셀ID"] = weather["기상셀ID"].astype("string")
    weather["일시"] = pd.to_datetime(weather["일시"], errors="coerce")
    if weather["일시"].isna().any():
        raise ValueError(f"{path_key}에 파싱할 수 없는 일시가 있습니다.")
    return weather.sort_values(["기상셀ID", "일시"], ignore_index=True)


def load_canadian_indices(*, nrows: int | None = None) -> pd.DataFrame:
    ffmc_columns = [
        "기상셀ID",
        "날짜",
        "기온_C",
        "습도_pct",
        "풍속_km_h",
        "강수량_mm",
        "FFMC",
        "FFMC_10일평균",
        "Indexed_FFMC",
        "FFMC_논문식_발생확률",
    ]
    fwi_columns = ["기상셀ID", "날짜", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

    ffmc = _read_csv("canada_ffmc", usecols=ffmc_columns, nrows=nrows)
    fwi = _read_csv("canada_fwi", usecols=fwi_columns, nrows=nrows)
    ffmc["날짜"] = pd.to_datetime(ffmc["날짜"], errors="coerce")
    fwi["날짜"] = pd.to_datetime(fwi["날짜"], errors="coerce")
    ffmc["기상셀ID"] = ffmc["기상셀ID"].astype("string")
    fwi["기상셀ID"] = fwi["기상셀ID"].astype("string")

    keys = ["기상셀ID", "날짜"]
    if ffmc[keys].duplicated().any() or fwi[keys].duplicated().any():
        raise ValueError("캐나다 지수에 중복 기상셀ID-날짜 키가 있습니다.")

    fwi = fwi.rename(columns={"FFMC": "FFMC_FWI산출"})
    result = ffmc.merge(fwi, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not (result["_merge"] == "both").all():
        counts = result["_merge"].value_counts().to_dict()
        raise ValueError(f"FFMC/FWI 키 불일치: {counts}")
    result = result.drop(columns="_merge")

    comparable = result[["FFMC", "FFMC_FWI산출"]].dropna()
    if not comparable.empty:
        max_diff = (comparable["FFMC"] - comparable["FFMC_FWI산출"]).abs().max()
        if max_diff > 1e-8:
            raise ValueError(f"두 산출물의 FFMC가 일치하지 않습니다. 최대 차이={max_diff}")
    return result.sort_values(keys, ignore_index=True)


def load_fire() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    fire = _read_csv("fire")
    required = [
        "fire_id",
        "연도",
        "월",
        "일",
        "시간",
        "위도",
        "경도",
        "발생지역번지",
        "발생원인명",
        "피해면적(ha)",
        "피해금액",
        "진화소요시간(HH)",
    ]
    require_columns(fire, required, "산불")
    if fire["fire_id"].duplicated().any():
        raise ValueError("산불 자료에 중복 fire_id가 있습니다.")

    date_part = pd.to_datetime(
        {
            "year": pd.to_numeric(fire["연도"], errors="coerce"),
            "month": pd.to_numeric(fire["월"], errors="coerce"),
            "day": pd.to_numeric(fire["일"], errors="coerce"),
        },
        errors="coerce",
    )
    hour_part = pd.to_numeric(fire["시간"], errors="coerce")
    fire["발생일시"] = date_part + pd.to_timedelta(hour_part, unit="h")
    fire["기준시각"] = fire["발생일시"].dt.floor("h")
    fire["발생날짜"] = fire["기준시각"].dt.normalize()
    address = fire["발생지역번지"].fillna("").astype(str).str.strip()
    fire["번지유형"] = np.where(address.str.startswith("산"), "임야번지(산)", "일반번지")

    valid_xy = fire["경도"].notna() & fire["위도"].notna()
    fire_points = gpd.GeoDataFrame(
        fire.loc[valid_xy].copy(),
        geometry=gpd.points_from_xy(
            fire.loc[valid_xy, "경도"],
            fire.loc[valid_xy, "위도"],
        ),
        crs="EPSG:4326",
    )
    return fire, fire_points


def add_canadian_asof_keys(
    frame: pd.DataFrame,
    *,
    time_column: str = "기준시각",
) -> pd.DataFrame:
    require_columns(frame, [time_column], "as-of 대상")
    result = frame.copy()
    timestamp = pd.to_datetime(result[time_column], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"{time_column}에 파싱할 수 없는 값이 있습니다.")

    use_previous_day = timestamp.dt.hour < 12
    result["캐나다지수_기준날짜"] = (
        timestamp.dt.normalize() - pd.to_timedelta(use_previous_day.astype(int), unit="D")
    )
    result["캐나다지수_기준시각"] = result["캐나다지수_기준날짜"] + pd.Timedelta(hours=12)
    result["캐나다지수_시차시간"] = (
        timestamp - result["캐나다지수_기준시각"]
    ).dt.total_seconds() / 3600
    result["캐나다지수_시차일"] = (
        timestamp.dt.normalize() - result["캐나다지수_기준날짜"]
    ).dt.days
    result["캐나다지수_당일사용여부"] = ~use_previous_day

    violations = result["캐나다지수_기준시각"] > timestamp
    if violations.any():
        raise ValueError(f"캐나다 지수 미래시점 위반: {int(violations.sum())}건")
    return result


def load_terrain() -> pd.DataFrame:
    terrain = _read_csv("terrain")
    required = [
        "fire_id",
        "위도",
        "경도",
        "고도(m)",
        "경사도(도)",
        "사면방향_sin",
        "사면방향_cos",
        "TPI(지형위치지수)",
        "TWI(지형다습지수)",
    ]
    require_columns(terrain, required, "산불 지형 특성")
    return terrain


def load_landcover() -> gpd.GeoDataFrame:
    path = DATA_PATHS["landcover"]
    layer = pyogrio.list_layers(path)[0, 0]
    landcover = gpd.read_file(
        path,
        layer=layer,
        columns=["L1_CODE", "L1_NAME", "L2_CODE", "L2_NAME"],
        engine="pyogrio",
    )
    require_columns(landcover, ["L1_NAME", "L2_NAME", "geometry"], "토지피복")
    return landcover


def load_roads() -> gpd.GeoDataFrame:
    path = DATA_PATHS["roads"]
    layer = pyogrio.list_layers(path)[0, 0]
    roads = gpd.read_file(
        path,
        layer=layer,
        columns=["source"],
        engine="pyogrio",
    )
    require_columns(roads, ["source", "geometry"], "도로")
    return roads


def load_access_lines() -> dict[str, pd.DataFrame]:
    trails = _read_csv("trails")
    forest_roads = _read_csv("forest_roads")
    require_columns(trails, ["공간좌표"], "등산로")
    require_columns(forest_roads, ["공간좌표"], "임도")
    return {"trails": trails, "forest_roads": forest_roads}


def load_infrastructure() -> dict[str, gpd.GeoDataFrame]:
    result = {}
    for key in ("fire_stations", "fire_water"):
        frame = _read_csv(key)
        require_columns(frame, ["위도", "경도"], key)
        valid_xy = frame["경도"].notna() & frame["위도"].notna()
        result[key] = gpd.GeoDataFrame(
            frame.loc[valid_xy].copy(),
            geometry=gpd.points_from_xy(
                frame.loc[valid_xy, "경도"],
                frame.loc[valid_xy, "위도"],
            ),
            crs="EPSG:4326",
        )
    return result


def load_dem_metadata() -> Mapping[str, object]:
    with rasterio.open(DATA_PATHS["dem"]) as dataset:
        return {
            "path": str(DATA_PATHS["dem"]),
            "crs": str(dataset.crs),
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "nodata": dataset.nodata,
            "bounds": tuple(dataset.bounds),
            "resolution": dataset.res,
        }


def frame_inventory(frames: Mapping[str, object]) -> pd.DataFrame:
    records = []
    for name, value in frames.items():
        if isinstance(value, (pd.DataFrame, gpd.GeoDataFrame)):
            records.append(
                {
                    "이름": name,
                    "행": len(value),
                    "열": len(value.columns),
                    "메모리_MB": round(value.memory_usage(deep=True).sum() / 1024**2, 2),
                    "CRS": str(value.crs) if isinstance(value, gpd.GeoDataFrame) else None,
                }
            )
        else:
            records.append(
                {
                    "이름": name,
                    "행": np.nan,
                    "열": np.nan,
                    "메모리_MB": np.nan,
                    "CRS": None,
                }
            )
    return pd.DataFrame(records)
