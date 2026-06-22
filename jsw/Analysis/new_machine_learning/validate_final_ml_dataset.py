from __future__ import annotations

import numpy as np
import pandas as pd

from build_final_ml_dataset import (
    BASE_PATH,
    CANADIAN_INDEX_COLUMNS,
    D2D3_ADDITION_COLUMNS,
    D2D3_PATH,
    DICTIONARY_PATH,
    FACTOR_COLUMNS,
    OUTPUT_PATH,
    load_daily_canadian_indices,
    read_csv,
)


def assert_series_equal(left: pd.Series, right: pd.Series, label: str) -> None:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        if not np.allclose(
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
            equal_nan=True,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"값 불일치: {label}")
        return

    left_text = left.astype("string").fillna("<NA>")
    right_text = right.astype("string").fillna("<NA>")
    if not left_text.equals(right_text):
        raise AssertionError(f"값 불일치: {label}")


def main() -> None:
    final = read_csv(OUTPUT_PATH)
    base = read_csv(BASE_PATH)
    d2d3 = read_csv(D2D3_PATH)
    dictionary = read_csv(DICTIONARY_PATH)
    daily = load_daily_canadian_indices()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, result: str) -> None:
        checks.append((name, bool(condition), result))
        if not condition:
            raise AssertionError(f"{name}: {result}")

    check("행 수", len(final) == 17045, f"{len(final):,}")
    check("변수 수", final.shape[1] == 117, f"{final.shape[1]:,}")
    check("샘플ID 유일성", final["샘플ID"].is_unique, str(final["샘플ID"].is_unique))
    check(
        "샘플ID 순서 보존",
        final["샘플ID"].tolist() == base["샘플ID"].tolist(),
        "원본 순서와 비교",
    )
    check(
        "Target 보존",
        final["Target"].tolist() == base["Target"].tolist(),
        str(final["Target"].value_counts().sort_index().to_dict()),
    )
    check(
        "요인점수 제거",
        not (set(FACTOR_COLUMNS) & set(final.columns)),
        str(sorted(set(FACTOR_COLUMNS) & set(final.columns))),
    )
    check(
        "변수사전 행 수",
        len(dictionary) == final.shape[1],
        f"{len(dictionary):,}",
    )
    check(
        "변수사전 변수명 일치",
        dictionary["변수명"].tolist() == final.columns.tolist(),
        "순서 포함 비교",
    )
    check(
        "D2D3 추가 변수 존재",
        set(D2D3_ADDITION_COLUMNS).issubset(final.columns),
        f"{len(D2D3_ADDITION_COLUMNS)}개",
    )

    final_time = pd.to_datetime(final["기준시각"], errors="raise")
    reference_date = pd.to_datetime(final["캐나다지수_기준날짜"], errors="raise")
    before_noon = final_time.dt.hour.lt(12)
    day_gap = (final_time.dt.normalize() - reference_date.dt.normalize()).dt.days
    check(
        "12시 이전 D-1 적용",
        day_gap.loc[before_noon].eq(1).all(),
        f"{int(before_noon.sum()):,}행",
    )
    check(
        "12시 이후 D0 적용",
        day_gap.loc[~before_noon].eq(0).all(),
        f"{int((~before_noon).sum()):,}행",
    )
    check(
        "캐나다지수 정책값",
        final["캐나다지수_정책"].eq("latest_available_noon_LST").all(),
        str(final["캐나다지수_정책"].value_counts().to_dict()),
    )
    check(
        "캐나다지수 결측",
        int(final[CANADIAN_INDEX_COLUMNS].isna().sum().sum()) == 0,
        str(final[CANADIAN_INDEX_COLUMNS].isna().sum().to_dict()),
    )

    expected = final[["샘플ID", "기상셀ID", "캐나다지수_기준날짜"]].copy()
    expected["캐나다지수_기준날짜"] = pd.to_datetime(
        expected["캐나다지수_기준날짜"], errors="raise"
    ).dt.normalize()
    expected = expected.merge(
        daily.rename(columns={"날짜": "캐나다지수_기준날짜"}),
        on=["기상셀ID", "캐나다지수_기준날짜"],
        how="left",
        validate="many_to_one",
    ).set_index("샘플ID")
    final_indexed = final.set_index("샘플ID")
    for column in CANADIAN_INDEX_COLUMNS:
        assert_series_equal(
            final_indexed.loc[expected.index, column],
            expected[column],
            f"캐나다지수 {column}",
        )
    checks.append(("캐나다지수 원천값 일치", True, f"{len(CANADIAN_INDEX_COLUMNS)}개"))

    d2d3_indexed = d2d3.set_index("샘플ID")
    for column in D2D3_ADDITION_COLUMNS:
        assert_series_equal(
            final_indexed.loc[d2d3_indexed.index, column],
            d2d3_indexed[column],
            f"D2D3 {column}",
        )
    checks.append(("D2D3 추가값 일치", True, f"{len(D2D3_ADDITION_COLUMNS)}개"))

    check_frame = pd.DataFrame(checks, columns=["검증항목", "통과", "결과"])
    print(check_frame.to_string(index=False))
    print("\n최종 머신러닝 학습데이터 검증 완료")


if __name__ == "__main__":
    main()
