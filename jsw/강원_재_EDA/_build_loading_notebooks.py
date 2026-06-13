from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def base_cells(step: int, title: str, purpose: str, source_keys: list[str]) -> list[dict]:
    return [
        markdown_cell(
            f"""
            # Step {step}. {title}

            {purpose}

            ## 현재 구현 범위

            이 노트북은 **원천 데이터 경로 확인, 로딩, 필수 스키마 검증, 시간 파싱,
            기본 행 수 감사**까지 구현한다. 통계 분석과 시각화는 이후 셀에서 이어서 작성한다.

            공통 데이터 계약은 `README.md`를 따른다. Step 2~6에서 캐나다 지수를 사건 시각에
            결합할 때는 12시 이전이면 전일 정오, 12시 이후이면 당일 정오 자료만 사용한다.

            결과 해석은 이 노트북에 작성하지 않는다. 결과표와 플롯을 함께 검토한 해석,
            한계와 다음 코드 반영사항은 대응 `진행예정로그.md`에만 기록한다.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            NOTEBOOK_DIR = Path.cwd().resolve()
            candidates = [NOTEBOOK_DIR, *NOTEBOOK_DIR.parents]
            REPO_ROOT = next(
                (path for path in candidates if (path / "jsw/강원_재_EDA/re_eda_common.py").exists()),
                Path(r"D:/farm-system-public-02"),
            )
            MODULE_DIR = REPO_ROOT / "jsw/강원_재_EDA"
            if str(MODULE_DIR) not in sys.path:
                sys.path.insert(0, str(MODULE_DIR))

            from re_eda_common import (
                DATA_PATHS,
                DERIVED_WEATHER_COLUMNS,
                RAW_WEATHER_COLUMNS,
                add_canadian_asof_keys,
                check_sources,
                configure_notebook,
                frame_inventory,
                load_access_lines,
                load_canadian_indices,
                load_dem_metadata,
                load_fire,
                load_grid_bundle,
                load_hourly_weather,
                load_infrastructure,
                load_landcover,
                load_roads,
                load_terrain,
            )

            configure_notebook()
            """
        ),
        markdown_cell("## 1. 원천 파일 존재 여부"),
        code_cell(
            f"""
            SOURCE_KEYS = {source_keys!r}
            source_audit = check_sources(SOURCE_KEYS)
            display(source_audit)
            """
        ),
    ]


def final_cells(loaded_names: list[str], next_work: str) -> list[dict]:
    inventory_mapping = "\n".join(f'    "{name}": {name},' for name in loaded_names)
    inventory_code = (
        "loaded_frames = {\n"
        f"{inventory_mapping}\n"
        "}\n"
        "display(frame_inventory(loaded_frames))"
    )
    return [
        markdown_cell("## 로딩 결과 요약"),
        code_cell(inventory_code),
        markdown_cell(
            f"""
            ## 다음 구현 범위

            {next_work}

            현재 노트북은 로딩과 입력 감사까지만 실행한다. 이후 분석 셀에서도
            `README.md`의 미래 정보 누수 방지 규칙과 대조군 정의를 유지해야 한다.
            실행 결과의 해석은 대응 `진행예정로그.md`에 작성한다.
            """
        ),
    ]


def step1_cells() -> list[dict]:
    cells = base_cells(
        1,
        "강원도 기후지형 및 캐나다 지수 기본분석",
        "산불 Target을 사용하지 않고 강원도 기상셀의 평상시 기후, 지형, 캐나다 지수 배경을 준비한다.",
        [
            "weather_cells",
            "weather_grid",
            "weather_hourly_raw",
            "climate_type",
            "canada_ffmc",
            "canada_fwi",
            "dem",
        ],
    )
    cells += [
        markdown_cell("## 2. 기상셀, 기후지형유형, 공간격자 로딩"),
        code_cell(
            """
            grid_bundle = load_grid_bundle()
            weather_cells = grid_bundle["cells"]
            climate_type = grid_bundle["climate"]
            weather_grid = grid_bundle["grid"]

            display(weather_cells.head())
            display(climate_type["기후지형유형"].value_counts(dropna=False))
            print("기상 격자 CRS:", weather_grid.crs)
            """
        ),
        markdown_cell("## 3. 원본 시간단위 기상 로딩"),
        code_cell(
            """
            hourly_weather = load_hourly_weather(
                derived=False,
                columns=RAW_WEATHER_COLUMNS,
            )
            print("기간:", hourly_weather["일시"].min(), "~", hourly_weather["일시"].max())
            print("기상셀 수:", hourly_weather["기상셀ID"].nunique())
            display(hourly_weather.head())
            """
        ),
        markdown_cell("## 4. 캐나다 산불위험지수 로딩"),
        code_cell(
            """
            canadian_indices = load_canadian_indices()
            print("기간:", canadian_indices["날짜"].min(), "~", canadian_indices["날짜"].max())
            print("기상셀 수:", canadian_indices["기상셀ID"].nunique())
            display(canadian_indices.head())
            """
        ),
        markdown_cell("## 5. DEM 메타데이터 로딩"),
        code_cell(
            """
            dem_metadata = load_dem_metadata()
            display(dem_metadata)
            """
        ),
    ]
    cells += final_cells(
        [
            "weather_cells",
            "climate_type",
            "weather_grid",
            "hourly_weather",
            "canadian_indices",
        ],
        "기상셀 중심점의 DEM 고도 샘플링, 월·계절 파생, 영동/영서 검정과 K=2·3 군집 탐색을 구현한다.",
    )
    return cells


def event_weather_load_cells(include_terrain: bool = False) -> list[dict]:
    cells = [
        markdown_cell("## 2. 산불 및 기상격자 로딩"),
        code_cell(
            """
            fire, fire_points = load_fire()
            grid_bundle = load_grid_bundle()
            weather_cells = grid_bundle["cells"]
            climate_type = grid_bundle["climate"]
            weather_grid = grid_bundle["grid"]

            print("산불:", len(fire), "건")
            print("좌표 유효 산불:", len(fire_points), "건")
            print("발생시각 결측:", int(fire["기준시각"].isna().sum()), "건")
            display(fire.head())
            """
        ),
        markdown_cell("## 3. 누수 방지 시간기상 파생자료 로딩"),
        code_cell(
            """
            hourly_weather = load_hourly_weather(
                derived=True,
                columns=DERIVED_WEATHER_COLUMNS,
            )
            print("기간:", hourly_weather["일시"].min(), "~", hourly_weather["일시"].max())
            print("기상셀 수:", hourly_weather["기상셀ID"].nunique())
            display(hourly_weather.head())
            """
        ),
        markdown_cell("## 4. 캐나다 지수 및 사건별 as-of 키 준비"),
        code_cell(
            """
            canadian_indices = load_canadian_indices()
            fire = add_canadian_asof_keys(fire, time_column="기준시각")

            asof_violations = (
                fire["캐나다지수_기준시각"] > fire["기준시각"]
            ).sum()
            assert asof_violations == 0
            display(
                fire[
                    [
                        "fire_id",
                        "기준시각",
                        "캐나다지수_기준날짜",
                        "캐나다지수_기준시각",
                        "캐나다지수_시차시간",
                        "캐나다지수_당일사용여부",
                    ]
                ].head()
            )
            """
        ),
    ]
    if include_terrain:
        cells += [
            markdown_cell("## 5. 산불 지형 특성 로딩"),
            code_cell(
                """
                terrain = load_terrain()
                display(terrain.head())
                """
            ),
        ]
    return cells


def step2_cells() -> list[dict]:
    cells = base_cells(
        2,
        "강원도 산불발생 날씨·지수 대조군 재분석",
        "산불 발생과 비발생 비교를 위한 사건, 격자, 누수 방지 기상, 캐나다 지수를 준비한다.",
        [
            "fire",
            "weather_cells",
            "weather_grid",
            "weather_hourly_derived",
            "climate_type",
            "canada_ffmc",
            "canada_fwi",
        ],
    )
    cells += event_weather_load_cells()
    cells += final_cells(
        [
            "fire",
            "fire_points",
            "weather_cells",
            "climate_type",
            "weather_grid",
            "hourly_weather",
            "canadian_indices",
        ],
        "산불-기상셀 공간조인과 미매칭 감사 후 전체 비발생 및 동일 셀·월·시간대 1:5 매칭 대조군을 생성한다.",
    )
    return cells


def step3_cells() -> list[dict]:
    cells = base_cells(
        3,
        "산불발생 선행기상 및 국지임계치 심화분석",
        "직전 24·48·72시간과 D-1~D-3 선행 기상, 번지유형, 캐나다 지수 분석에 필요한 자료를 준비한다.",
        [
            "fire",
            "weather_cells",
            "weather_grid",
            "weather_hourly_derived",
            "climate_type",
            "canada_ffmc",
            "canada_fwi",
        ],
    )
    cells += event_weather_load_cells()
    cells += [
        markdown_cell("## 5. 선행기상 컬럼 가용성 감사"),
        code_cell(
            """
            expected_lead_columns = [
                "직전24h_평균풍속",
                "직전24h_최대풍속",
                "직전48h_평균풍속",
                "직전48h_최대풍속",
                "직전24h_평균습도",
                "직전24h_최소습도",
                "직전48h_평균습도",
                "직전48h_최소습도",
                "직전24h_강수량합",
                "직전48h_강수량합",
                "D-1_최소습도_pct",
                "D-1_평균습도_pct",
                "D-1_강수량합_mm",
                "D-2_최소습도_pct",
                "D-3_최소습도_pct",
            ]
            display(hourly_weather[expected_lead_columns].isna().mean().rename("결측률").to_frame())
            print("주의: 직전72h 및 추가 D-2/D-3 집계는 원본 시간자료에서 누수 없이 후속 계산합니다.")
            """
        ),
    ]
    cells += final_cells(
        [
            "fire",
            "fire_points",
            "weather_cells",
            "climate_type",
            "weather_grid",
            "hourly_weather",
            "canadian_indices",
        ],
        "직전72시간·D-1~D-3 파생을 보강하고, 매칭 대조군 기준 상대 분위수와 지역·번지별 임계치를 계산한다.",
    )
    return cells


def spatial_load_cells(include_weather: bool) -> list[dict]:
    cells = [
        markdown_cell("## 2. 산불, 격자, 지형 자료 로딩"),
        code_cell(
            """
            fire, fire_points = load_fire()
            grid_bundle = load_grid_bundle()
            weather_cells = grid_bundle["cells"]
            climate_type = grid_bundle["climate"]
            weather_grid = grid_bundle["grid"]
            terrain = load_terrain()
            dem_metadata = load_dem_metadata()

            print("산불 포인트 CRS:", fire_points.crs)
            print("기상 격자 CRS:", weather_grid.crs)
            display(terrain.head())
            display(dem_metadata)
            """
        ),
        markdown_cell(
            """
            ## 3. 대용량 공간 원천 로딩

            토지피복 약 1.2GB, 도로 약 307MB 파일을 실제로 읽는다.
            메모리가 부족한 환경에서는 두 줄을 각각 실행해 사용량을 확인한다.
            """
        ),
        code_cell(
            """
            landcover = load_landcover()
            roads = load_roads()

            print("토지피복:", len(landcover), "건 / CRS:", landcover.crs)
            print("도로:", len(roads), "건 / CRS:", roads.crs)
            """
        ),
        markdown_cell("## 4. 등산로, 임도, 소방 인프라 로딩"),
        code_cell(
            """
            access_lines = load_access_lines()
            trails = access_lines["trails"]
            forest_roads = access_lines["forest_roads"]

            infrastructure = load_infrastructure()
            fire_stations = infrastructure["fire_stations"]
            fire_water = infrastructure["fire_water"]

            print("등산로:", len(trails), "건")
            print("임도:", len(forest_roads), "건")
            print("소방서:", len(fire_stations), "건")
            print("소방용수:", len(fire_water), "건")
            """
        ),
    ]
    if include_weather:
        cells += [
            markdown_cell("## 5. 누수 방지 기상 및 캐나다 지수 로딩"),
            code_cell(
                """
                hourly_weather = load_hourly_weather(
                    derived=True,
                    columns=DERIVED_WEATHER_COLUMNS,
                )
                canadian_indices = load_canadian_indices()
                fire = add_canadian_asof_keys(fire, time_column="기준시각")
                assert (fire["캐나다지수_기준시각"] <= fire["기준시각"]).all()
                """
            ),
        ]
    else:
        cells += [
            markdown_cell("## 5. 풍향-사면 분석용 최소 시간기상 로딩"),
            code_cell(
                """
                hourly_weather = load_hourly_weather(
                    derived=True,
                    columns=["기상셀ID", "일시", "시점_풍향_deg"],
                )
                display(hourly_weather.head())
                """
            ),
        ]
    return cells


def step4_cells() -> list[dict]:
    cells = base_cells(
        4,
        "산불발생 공간지형 및 대조군 분석",
        "토지피복, 지형, 접근성, 소방 인프라와 동일 기상셀 공간 대조 분석에 필요한 자료를 준비한다.",
        [
            "fire",
            "weather_cells",
            "weather_grid",
            "climate_type",
            "weather_hourly_derived",
            "terrain",
            "dem",
            "landcover",
            "roads",
            "trails",
            "forest_roads",
            "fire_stations",
            "fire_water",
        ],
    )
    cells += spatial_load_cells(include_weather=False)
    cells += final_cells(
        [
            "fire",
            "fire_points",
            "weather_cells",
            "climate_type",
            "weather_grid",
            "terrain",
            "landcover",
            "roads",
            "trails",
            "forest_roads",
            "fire_stations",
            "fire_water",
            "hourly_weather",
        ],
        "모든 공간 자료를 EPSG:5186으로 정렬한 뒤 토지피복 공간조인, 최단거리, 동일셀 랜덤 공간·산림 대조군을 생성한다.",
    )
    return cells


def step5_cells() -> list[dict]:
    cells = base_cells(
        5,
        "산불발생 기후·공간·지수 융합분석",
        "누수 없는 기상, 캐나다 지수, 지형, 토지피복, 접근성을 통합하기 위한 전체 원천 자료를 준비한다.",
        [
            "fire",
            "weather_cells",
            "weather_grid",
            "weather_hourly_derived",
            "climate_type",
            "canada_ffmc",
            "canada_fwi",
            "terrain",
            "dem",
            "landcover",
            "roads",
            "trails",
            "forest_roads",
            "fire_stations",
            "fire_water",
        ],
    )
    cells += spatial_load_cells(include_weather=True)
    cells += final_cells(
        [
            "fire",
            "fire_points",
            "weather_cells",
            "climate_type",
            "weather_grid",
            "terrain",
            "landcover",
            "roads",
            "trails",
            "forest_roads",
            "fire_stations",
            "fire_water",
            "hourly_weather",
            "canadian_indices",
        ],
        "Step 2의 사건·대조군, Step 3의 상대 기상 이상성, Step 4의 공간 변수를 같은 관측 단위로 결합한다.",
    )
    return cells


def step6_cells() -> list[dict]:
    cells = base_cells(
        6,
        "산불발생 인간활동 원인프록시 분석",
        "발생원인 결측 감사와 등산로·임도·생활권 접근성 프록시 분석에 필요한 전체 자료를 준비한다.",
        [
            "fire",
            "weather_cells",
            "weather_grid",
            "weather_hourly_derived",
            "climate_type",
            "canada_ffmc",
            "canada_fwi",
            "terrain",
            "dem",
            "landcover",
            "roads",
            "trails",
            "forest_roads",
            "fire_stations",
            "fire_water",
        ],
    )
    cells += spatial_load_cells(include_weather=True)
    cells += [
        markdown_cell("## 6. 원인·피해·진화시간 결측 감사"),
        code_cell(
            """
            audit_columns = ["발생원인명", "피해면적(ha)", "피해금액", "진화소요시간(HH)"]
            missing_audit = (
                fire[audit_columns]
                .isna()
                .agg(["sum", "mean"])
                .T
                .rename(columns={"sum": "결측건수", "mean": "결측률"})
            )
            display(missing_audit)
            """
        ),
    ]
    cells += final_cells(
        [
            "fire",
            "fire_points",
            "weather_cells",
            "climate_type",
            "weather_grid",
            "terrain",
            "landcover",
            "roads",
            "trails",
            "forest_roads",
            "fire_stations",
            "fire_water",
            "hourly_weather",
            "canadian_indices",
        ],
        "공간 최단거리를 계산해 250m·500m·1,000m 입산활동 프록시와 생활권 프록시를 만들고 매칭 비발생 및 선형 공간 대조군과 비교한다.",
    )
    return cells


NOTEBOOKS = {
    "Step1_강원도_기후지형및캐나다지수_기본분석.ipynb": step1_cells,
    "Step2_강원도_산불발생_날씨지수_대조군재분석.ipynb": step2_cells,
    "Step3_산불발생_선행기상및국지임계치_심화분석.ipynb": step3_cells,
    "Step4_산불발생_공간지형및대조군_분석.ipynb": step4_cells,
    "Step5_산불발생_기후공간지수_융합분석.ipynb": step5_cells,
    "Step6_산불발생_인간활동원인프록시_분석.ipynb": step6_cells,
}


def build_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.13.1",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.1",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for filename, cell_factory in NOTEBOOKS.items():
        path = HERE / filename
        notebook = build_notebook(cell_factory())
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"written: {path.name} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
