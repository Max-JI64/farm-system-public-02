from __future__ import annotations

"""Prepare Gangwon weather features in the format used by Paper 1.

Paper 1 predicts burned area with data-mining models and reports the best result
from a DNN using only five meteorological features:

    avg_temp, min_temp, max_temp, max_wind_speed, avg_wind

This script reads the integrated Gangwon hourly weather file and aggregates it to
one row per weather cell and date. Wind speed is converted from m/s to km/h to
match the paper's feature definition.

Output:
    data/modeling/paper1_dnn_gangwon_weather_input_from_integrated_time.csv

Note:
    This creates model input features only. Training the exact paper model also
    requires a valid burned-area target for matching fire events.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = ROOT / "data" / "\uac15\uc6d0\ub3c4_\ub0a0\uc528\ub370\uc774\ud130"
OUT_DIR = ROOT / "data" / "modeling"

INTEGRATED_WEATHER = WEATHER_DIR / "\uac15\uc6d0\ub3c4\ub0a0\uc528_\ud1b5\ud569_\uc2dc\uac04\ub2e8\uc704.csv"
OUT_PATH = OUT_DIR / "paper1_dnn_gangwon_weather_input_from_integrated_time.csv"

CELL_ID = "\uae30\uc0c1\uc140ID"


def add_7x7_grid_index(cell_meta: pd.DataFrame) -> pd.DataFrame:
    """Match the paper's coarse longitude/latitude grid idea with 1-7 bins."""
    cell_meta = cell_meta.copy()
    lon = "\uc911\uc2ec\uacbd\ub3c4_wgs84"
    lat = "\uc911\uc2ec\uc704\ub3c4_wgs84"
    cell_meta["longitude_grid_1_7"] = pd.qcut(
        cell_meta[lon].rank(method="first"),
        q=7,
        labels=range(1, 8),
    ).astype("int8")
    cell_meta["latitude_grid_1_7"] = pd.qcut(
        cell_meta[lat].rank(method="first"),
        q=7,
        labels=range(1, 8),
    ).astype("int8")
    return cell_meta


def main() -> None:
    """Aggregate hourly Gangwon weather into the paper's daily DNN input."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    usecols = [
        CELL_ID,
        "\uc77c\uc2dc",
        "\uae30\uc628_C",
        "\ud48d\uc18d_m_s",
        "\uae30\ud6c4\uad8c\uc5ed",
        "\uae30\ud6c4\uc9c0\ud615\uc720\ud615",
        "\ub300\ud45c\uad00\uce21\ub9dd",
        "\ub300\ud45c\uc9c0\uc810\ubc88\ud638",
        "\ub300\ud45c\uc9c0\uc810\uba85",
        "\uc911\uc2ec\uacbd\ub3c4_wgs84",
        "\uc911\uc2ec\uc704\ub3c4_wgs84",
    ]
    weather = pd.read_csv(INTEGRATED_WEATHER, encoding="utf-8-sig", usecols=usecols)
    weather["datetime"] = pd.to_datetime(weather["\uc77c\uc2dc"])
    weather["date"] = weather["datetime"].dt.strftime("%Y-%m-%d")
    weather["month"] = weather["datetime"].dt.month.astype("int8")
    weather["day"] = weather["datetime"].dt.dayofweek.astype("int8")

    weather_features = (
        weather.groupby([CELL_ID, "date"], as_index=False)
        .agg(
            avg_temp=("\uae30\uc628_C", "mean"),
            min_temp=("\uae30\uc628_C", "min"),
            max_temp=("\uae30\uc628_C", "max"),
            max_wind_speed_m_s=("\ud48d\uc18d_m_s", "max"),
            avg_wind_m_s=("\ud48d\uc18d_m_s", "mean"),
            month=("month", "first"),
            day=("day", "first"),
        )
    )

    # The paper uses wind speed in km/h.
    weather_features["max_wind_speed"] = weather_features["max_wind_speed_m_s"] * 3.6
    weather_features["avg_wind"] = weather_features["avg_wind_m_s"] * 3.6
    weather_features = weather_features.drop(columns=["max_wind_speed_m_s", "avg_wind_m_s"])

    cell_meta_cols = [
        CELL_ID,
        "\uae30\ud6c4\uad8c\uc5ed",
        "\uae30\ud6c4\uc9c0\ud615\uc720\ud615",
        "\ub300\ud45c\uad00\uce21\ub9dd",
        "\ub300\ud45c\uc9c0\uc810\ubc88\ud638",
        "\ub300\ud45c\uc9c0\uc810\uba85",
        "\uc911\uc2ec\uacbd\ub3c4_wgs84",
        "\uc911\uc2ec\uc704\ub3c4_wgs84",
    ]
    cell_meta = weather[cell_meta_cols].drop_duplicates(CELL_ID)
    cell_meta = add_7x7_grid_index(cell_meta)

    output = weather_features.merge(cell_meta, on=CELL_ID, how="left")

    ordered = [
        CELL_ID,
        "date",
        "longitude_grid_1_7",
        "latitude_grid_1_7",
        "month",
        "day",
        "avg_temp",
        "min_temp",
        "max_temp",
        "max_wind_speed",
        "avg_wind",
        "\uae30\ud6c4\uad8c\uc5ed",
        "\uae30\ud6c4\uc9c0\ud615\uc720\ud615",
        "\ub300\ud45c\uad00\uce21\ub9dd",
        "\ub300\ud45c\uc9c0\uc810\ubc88\ud638",
        "\ub300\ud45c\uc9c0\uc810\uba85",
        "\uc911\uc2ec\uacbd\ub3c4_wgs84",
        "\uc911\uc2ec\uc704\ub3c4_wgs84",
    ]
    output = output[[column for column in ordered if column in output.columns]]
    output.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"created={OUT_PATH}")
    print(f"shape={output.shape}")
    print(f"date_range={output['date'].min()}..{output['date'].max()}")
    print("paper_m_features=avg_temp,min_temp,max_temp,max_wind_speed,avg_wind")


if __name__ == "__main__":
    main()
