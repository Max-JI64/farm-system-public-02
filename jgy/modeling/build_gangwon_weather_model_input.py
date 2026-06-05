from __future__ import annotations

"""강원도 날씨 데이터를 산불 위험도 모델 입력용으로 만드는 스크립트.

이 파일은 "모델을 직접 학습하는 코드"라기보다는, 산불 위험도 예측 모델에
넣을 수 있는 입력 데이터를 만드는 전처리/피처 생성 코드입니다.

사용한 원천 데이터
----------------
1. 강원도 날씨 통합 시간단위 데이터
   - data/강원도_날씨데이터/강원도날씨_통합_시간단위.csv
   - 기상셀ID, 일시, 기온, 풍속, 강수량, 습도 등이 들어 있습니다.

2. 강원도 일단위 날씨 데이터
   - data/강원도_날씨데이터/강원도날씨_격자_일단위.csv
   - 하루 평균/최저/최고기온, 일강수량, 최대순간풍속 등이 들어 있습니다.

3. 강원도 날씨 격자 정보
   - data/강원도_날씨데이터/강원도날씨_격자.csv
   - data/강원도_날씨데이터/강원도날씨_격자.geojson
   - 각 기상셀의 위치, 중심 좌표, 대표 관측소 정보가 들어 있습니다.

4. 강원도 전신주 예측 대상 데이터
   - data/예측데이터/gangwon_poles_4326.csv
   - pole_id, lon, lat 컬럼만 있는 전신주 좌표 데이터입니다.

이 코드가 만드는 결과 파일
----------------------
1. data/modeling/gangwon_weather_daily_model_features.csv
   - 기상셀ID + 날짜 기준의 일단위 날씨 피처 파일입니다.
   - 결과 크기: 67,252행, 34컬럼
   - 날짜 범위: 2020-01-01 ~ 2021-12-31

2. data/modeling/gangwon_pole_weather_cell_map.csv
   - 강원도 전신주 1,387,831개를 날씨 격자에 공간 매칭한 파일입니다.
   - 전신주 좌표가 어떤 기상셀ID에 속하는지 알려줍니다.

3. data/modeling/gangwon_poles_weather_model_input_YYYY-MM-DD.csv
   - 특정 날짜 기준으로 전신주별 날씨 피처를 붙인 최종 모델 입력 파일입니다.
   - 예: gangwon_poles_weather_model_input_2021-12-31.csv
   - 결과 크기: 1,387,831행, 37컬럼
   - 날씨 매칭 누락: 0건

주요 피처 의미
------------
- daily_temp_mean_C: 일평균기온
- daily_temp_min_C: 일최저기온
- daily_temp_max_C: 일최고기온
- daily_rain_mm: 일강수량
- daily_gust_max_m_s: 일최대순간풍속
- hourly_wind_mean_m_s: 시간단위 풍속의 일평균
- hourly_wind_max_m_s: 시간단위 풍속의 일최대
- hourly_humidity_mean_pct: 시간단위 습도의 일평균
- hourly_humidity_min_pct: 시간단위 습도의 일최저
- dry_day: 비가 오지 않은 날이면 1
- low_humidity_day: 최저습도 30% 이하이면 1
- windy_day: 최대순간풍속 7m/s 이상이면 1
- very_windy_day: 최대순간풍속 12m/s 이상이면 1
- fire_weather_flag: 건조하면서 저습 또는 강풍 조건이면 1
- weather_risk_score_0_1: 풍속, 습도, 강수량으로 만든 0~1 날씨 위험 점수

현재 결과 해석
------------
이 스크립트 결과는 "강원도 전신주별로 특정 날짜의 날씨 위험 조건을 붙인
모델 입력 테이블"입니다. 예를 들어 2021-12-31 입력 파일을 보면 전신주마다
어느 기상셀에 속하는지, 그날의 기온/풍속/습도/강수량 조건이 어떤지, 그리고
간단한 날씨 위험 점수가 얼마인지 확인할 수 있습니다.

현재 문제점과 한계
----------------
1. 아직 최종 산불 위험도 예측 모델을 학습한 것은 아닙니다.
   - 이 파일은 모델 학습/예측 전에 필요한 입력 데이터를 만드는 단계입니다.

2. 논문1 방식처럼 피해면적을 예측하려면 타깃인 피해면적(ha)이 필요합니다.
   - 현재 강원도 산불 데이터의 2020~2021년 행은 피해면적(ha)이 비어 있어서
     논문1의 DNN 회귀 모델을 그대로 학습하기 어렵습니다.

3. weather_risk_score_0_1은 임시 규칙 기반 점수입니다.
   - 풍속, 습도, 강수량을 조합한 참고용 점수이며, 실제 학습된 모델의
     예측 확률이나 최종 위험도는 아닙니다.

4. 전신주와 날씨는 날씨 격자 단위로 연결됩니다.
   - 전신주 주변의 지형, 토지피복, 산림 인접도, 소방 접근성 같은 공간 피처는
     별도 데이터와 결합해야 더 완성도 높은 산불 위험도 모델이 됩니다.

실행 예시
--------
python jgy/modeling/build_gangwon_weather_model_input.py --date latest
python jgy/modeling/build_gangwon_weather_model_input.py --date 2021-03-15
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = ROOT / "data" / "\uac15\uc6d0\ub3c4_\ub0a0\uc528\ub370\uc774\ud130"
PREDICT_DIR = ROOT / "data" / "\uc608\uce21\ub370\uc774\ud130"
OUT_DIR = ROOT / "data" / "modeling"

CELL_ID = "\uae30\uc0c1\uc140ID"
DATE = "\ub0a0\uc9dc"


def build_daily_weather_features() -> pd.DataFrame:
    """Create one daily feature row for each Gangwon weather cell."""
    hourly_cols = [
        CELL_ID,
        "\uc77c\uc2dc",
        "\uae30\uc628_C",
        "\ud48d\uc18d_m_s",
        "\ud48d\ud5a5_deg",
        "\uac15\uc218\ub7c9_mm",
        "\uc2b5\ub3c4_pct",
    ]
    hourly = pd.read_csv(
        WEATHER_DIR / "\uac15\uc6d0\ub3c4\ub0a0\uc528_\ud1b5\ud569_\uc2dc\uac04\ub2e8\uc704.csv",
        encoding="utf-8-sig",
        usecols=hourly_cols,
    )
    hourly[DATE] = pd.to_datetime(hourly["\uc77c\uc2dc"]).dt.strftime("%Y-%m-%d")

    hourly_daily = (
        hourly.groupby([CELL_ID, DATE], as_index=False)
        .agg(
            hourly_temp_mean_C=("\uae30\uc628_C", "mean"),
            hourly_temp_min_C=("\uae30\uc628_C", "min"),
            hourly_temp_max_C=("\uae30\uc628_C", "max"),
            hourly_wind_mean_m_s=("\ud48d\uc18d_m_s", "mean"),
            hourly_wind_max_m_s=("\ud48d\uc18d_m_s", "max"),
            hourly_rain_sum_mm=("\uac15\uc218\ub7c9_mm", "sum"),
            hourly_humidity_mean_pct=("\uc2b5\ub3c4_pct", "mean"),
            hourly_humidity_min_pct=("\uc2b5\ub3c4_pct", "min"),
        )
    )

    daily = pd.read_csv(
        WEATHER_DIR / "\uac15\uc6d0\ub3c4\ub0a0\uc528_\uaca9\uc790_\uc77c\ub2e8\uc704.csv",
        encoding="utf-8-sig",
    ).rename(
        columns={
            "\ud3c9\uade0\uae30\uc628_C": "daily_temp_mean_C",
            "\ucd5c\uc800\uae30\uc628_C": "daily_temp_min_C",
            "\ucd5c\uace0\uae30\uc628_C": "daily_temp_max_C",
            "\uac15\uc218\ub7c9_mm": "daily_rain_mm",
            "\ucd5c\ub300\uc21c\uac04\ud48d\uc18d_m_s": "daily_gust_max_m_s",
            "\ucd5c\ub300\uc21c\uac04\ud48d\ud5a5_deg": "daily_gust_dir_deg",
        }
    )

    grid = pd.read_csv(
        WEATHER_DIR / "\uac15\uc6d0\ub3c4\ub0a0\uc528_\uaca9\uc790.csv",
        encoding="utf-8-sig",
    ).drop(columns=["\uc601\uc5edWKT_wgs84"], errors="ignore")
    cell_type = pd.read_csv(
        WEATHER_DIR / "\uac15\uc6d0\ub3c4\ub0a0\uc528_\uae30\ud6c4\uc9c0\ud615\uc720\ud615_\uc140\ubd84\ub958.csv",
        encoding="utf-8-sig",
    ).drop(columns=["\uae30\ud6c4\uad8c\uc5ed"], errors="ignore")

    features = daily.merge(hourly_daily, on=[CELL_ID, DATE], how="left")
    features = features.merge(grid, on=CELL_ID, how="left")
    features = features.merge(cell_type, on=CELL_ID, how="left")

    # Lightweight rule-based weather indicators for downstream risk scoring.
    features["temp_range_C"] = features["daily_temp_max_C"] - features["daily_temp_min_C"]
    features["dry_day"] = features["daily_rain_mm"].fillna(0).eq(0).astype("int8")
    features["low_humidity_day"] = features["hourly_humidity_min_pct"].le(30).astype("int8")
    features["windy_day"] = features["daily_gust_max_m_s"].ge(7).astype("int8")
    features["very_windy_day"] = features["daily_gust_max_m_s"].ge(12).astype("int8")
    features["fire_weather_flag"] = (
        features["dry_day"].eq(1)
        & (features["low_humidity_day"].eq(1) | features["windy_day"].eq(1))
    ).astype("int8")

    wind_score = (features["daily_gust_max_m_s"].clip(lower=0, upper=15) / 15).fillna(0)
    dry_score = (1 - (features["daily_rain_mm"].clip(lower=0, upper=10) / 10)).fillna(0)
    humidity_score = (1 - (features["hourly_humidity_min_pct"].clip(lower=0, upper=80) / 80)).fillna(0)
    features["weather_risk_score_0_1"] = (0.45 * wind_score + 0.30 * humidity_score + 0.25 * dry_score).round(4)

    return features


def build_pole_cell_map() -> pd.DataFrame:
    """Spatially map every Gangwon pole to a weather cell."""
    map_path = OUT_DIR / "gangwon_pole_weather_cell_map.csv"
    if map_path.exists():
        return pd.read_csv(map_path, encoding="utf-8-sig")

    poles = pd.read_csv(PREDICT_DIR / "gangwon_poles_4326.csv")
    pole_gdf = gpd.GeoDataFrame(
        poles,
        geometry=gpd.points_from_xy(poles["lon"], poles["lat"]),
        crs="EPSG:4326",
    )
    cells = gpd.read_file(WEATHER_DIR / "\uac15\uc6d0\ub3c4\ub0a0\uc528_\uaca9\uc790.geojson")[
        [CELL_ID, "geometry"]
    ]
    joined = gpd.sjoin(pole_gdf, cells, how="left", predicate="within")
    result = joined.drop(columns=["geometry", "index_right"], errors="ignore")

    # A few boundary points can fall outside polygons; assign nearest cell center.
    missing = result[CELL_ID].isna()
    if missing.any():
        cell_centers = gpd.read_file(WEATHER_DIR / "\uac15\uc6d0\ub3c4\ub0a0\uc528_\uaca9\uc790.geojson")[
            [CELL_ID, "\uc911\uc2ec\uacbd\ub3c4_wgs84", "\uc911\uc2ec\uc704\ub3c4_wgs84"]
        ]
        centers = gpd.GeoDataFrame(
            cell_centers,
            geometry=gpd.points_from_xy(
                cell_centers["\uc911\uc2ec\uacbd\ub3c4_wgs84"],
                cell_centers["\uc911\uc2ec\uc704\ub3c4_wgs84"],
            ),
            crs="EPSG:4326",
        ).to_crs("EPSG:5186")
        nearest = gpd.sjoin_nearest(
            pole_gdf.loc[missing].to_crs("EPSG:5186"),
            centers[[CELL_ID, "geometry"]],
            how="left",
        )
        result.loc[missing, CELL_ID] = nearest[CELL_ID].to_numpy()

    result.to_csv(map_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    """Create daily weather features and, optionally, pole-level model input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="latest", help="YYYY-MM-DD, or latest")
    parser.add_argument("--skip-poles", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weather_features = build_daily_weather_features()
    weather_path = OUT_DIR / "gangwon_weather_daily_model_features.csv"
    weather_features.to_csv(weather_path, index=False, encoding="utf-8-sig")
    print(f"created={weather_path}")
    print(f"weather_shape={weather_features.shape}")

    if args.skip_poles:
        return

    predict_date = weather_features[DATE].max() if args.date == "latest" else args.date
    date_weather = weather_features[weather_features[DATE].eq(predict_date)].copy()
    if date_weather.empty:
        raise ValueError(f"No weather rows found for date: {predict_date}")

    pole_map = build_pole_cell_map()
    model_input = pole_map.merge(date_weather, on=CELL_ID, how="left")
    out_path = OUT_DIR / f"gangwon_poles_weather_model_input_{predict_date}.csv"
    model_input.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"created={out_path}")
    print(f"pole_model_input_shape={model_input.shape}")
    print(f"weather_missing_rows={model_input['weather_risk_score_0_1'].isna().sum()}")


if __name__ == "__main__":
    main()
