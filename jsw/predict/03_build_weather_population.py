from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predict_common import (
    CANADIAN_INDEX_COLUMNS,
    OUTPUT_DIR,
    WEATHER_HOURLY_RAW,
    WEATHER_REQUIRED_COLUMNS,
    attach_canadian_indices,
    build_hourly_weather_features,
    ensure_output_dir,
    is_main_prediction_period,
    load_daily_canadian_indices,
    read_csv,
    validation_frame,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="학습데이터와 동일한 기상 rolling/as-of 로직으로 10월~5월 09~16시 예측용 날씨 모집단 생성"
    )
    parser.add_argument("--weather-raw", type=Path, default=WEATHER_HOURLY_RAW)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="기상/캐나다지수 결측 row를 제거하지 않고 보존합니다.",
    )
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    print(f"시간단위 원천 날씨 로드: {args.weather_raw}")
    hourly_raw = read_csv(args.weather_raw)
    print(f"원천 날씨 행 수: {len(hourly_raw):,}")

    print("학습데이터 step3_weather_rolling.py와 동일한 rolling 기상 피처 생성 중...")
    weather = build_hourly_weather_features(hourly_raw)
    total_after_feature = len(weather)

    weather = weather[is_main_prediction_period(weather["기준시각"])].copy()
    filtered_rows = len(weather)

    print("캐나다 산불지수 as-of 결합 중...")
    daily = load_daily_canadian_indices()
    weather, canadian_audit = attach_canadian_indices(weather, daily)

    required_for_model = WEATHER_REQUIRED_COLUMNS + CANADIAN_INDEX_COLUMNS
    missing_cols = sorted(set(required_for_model) - set(weather.columns))
    if missing_cols:
        raise KeyError(f"예측용 날씨 모집단 누락 컬럼: {missing_cols}")

    incomplete_mask = weather[required_for_model].replace([float("inf"), float("-inf")], pd.NA).isna().any(axis=1)
    incomplete_rows = int(incomplete_mask.sum())
    if not args.keep_incomplete and incomplete_rows:
        weather = weather.loc[~incomplete_mask].copy()

    duplicated = int(weather.duplicated(["기상셀ID", "기준시각"]).sum())
    if duplicated:
        raise ValueError(f"기상셀ID+기준시각 중복이 있습니다: {duplicated}")

    audit = validation_frame(
        [
            ("source_weather_file", str(args.weather_raw), args.weather_raw.exists()),
            ("raw_rows", f"{len(hourly_raw):,}", len(hourly_raw) > 0),
            ("rows_after_feature_engineering", f"{total_after_feature:,}", total_after_feature > 0),
            ("rows_after_10to5_09to16_filter", f"{filtered_rows:,}", filtered_rows > 0),
            ("rows_written", f"{len(weather):,}", len(weather) > 0),
            ("dropped_incomplete_rows", incomplete_rows if not args.keep_incomplete else 0, True),
            ("kept_incomplete_rows", incomplete_rows if args.keep_incomplete else 0, incomplete_rows == 0 or args.keep_incomplete),
            ("weather_cell_count", int(weather["기상셀ID"].nunique()), weather["기상셀ID"].nunique() > 0),
            ("duplicated_cell_datetime", duplicated, duplicated == 0),
            ("outside_month_filter", int((~pd.to_datetime(weather["기준시각"]).dt.month.isin([10, 11, 12, 1, 2, 3, 4, 5])).sum()), True),
            ("outside_hour_filter", int((~pd.to_datetime(weather["기준시각"]).dt.hour.between(9, 16)).sum()), True),
            ("canadian_missing_cells_before_drop", canadian_audit["canadian_missing_cells"], canadian_audit["canadian_missing_cells"] == 0 or not args.keep_incomplete),
            ("before_noon_bad_reference_n", canadian_audit["before_noon_bad_reference_n"], canadian_audit["before_noon_bad_reference_n"] == 0),
            ("at_or_after_noon_bad_reference_n", canadian_audit["at_or_after_noon_bad_reference_n"], canadian_audit["at_or_after_noon_bad_reference_n"] == 0),
        ]
    )

    out_path = output_dir / "weather_population_10to5_09to16.csv"
    audit_path = output_dir / "weather_population_audit.csv"
    write_csv(weather, out_path)
    write_csv(audit, audit_path)
    write_json(
        {
            "script": "03_build_weather_population.py",
            "weather_source": str(args.weather_raw),
            "weather_source_note": "학습데이터 step3_weather_rolling.py와 동일하게 data/강원도_날씨데이터/강원도날씨_격자_시간단위.csv 사용",
            "months": [10, 11, 12, 1, 2, 3, 4, 5],
            "hours": list(range(9, 17)),
            "rows_written": int(len(weather)),
            "canadian_policy": "09~11시 D-1 정오, 12~16시 당일 정오",
            "keep_incomplete": bool(args.keep_incomplete),
        },
        output_dir / "run_manifest__03_build_weather_population.json",
    )
    print(f"완료: {out_path}")


if __name__ == "__main__":
    main()
