from __future__ import annotations

import argparse
import ast
import gc
import json
from pathlib import Path

import re_eda_common as common


HERE = Path(__file__).resolve().parent


def validate_notebooks() -> None:
    notebook_paths = sorted(HERE.glob("Step*.ipynb"))
    if len(notebook_paths) != 6:
        raise AssertionError(f"Step 노트북 수가 6개가 아닙니다: {len(notebook_paths)}")

    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise AssertionError(f"{path.name}: nbformat이 4가 아닙니다.")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse(
                    "".join(cell["source"]),
                    filename=f"{path.name}:cell{index}",
                )
        print(f"notebook ok: {path.name} ({len(notebook['cells'])} cells)")


def validate_core(*, full_hourly: bool) -> None:
    print(common.check_sources(common.DATA_PATHS.keys()).to_string(index=False))

    grid_bundle = common.load_grid_bundle()
    print("grid bundle:", {key: value.shape for key, value in grid_bundle.items()})

    fire, fire_points = common.load_fire()
    print("fire:", fire.shape, "points:", fire_points.shape)

    nrows = None if full_hourly else 10_000
    raw_weather = common.load_hourly_weather(derived=False, nrows=nrows)
    print("raw hourly:", raw_weather.shape)
    del raw_weather
    gc.collect()

    derived_weather = common.load_hourly_weather(derived=True, nrows=nrows)
    print("derived hourly:", derived_weather.shape)
    del derived_weather
    gc.collect()

    canadian = common.load_canadian_indices()
    print("canadian:", canadian.shape)

    fire_asof = common.add_canadian_asof_keys(fire)
    violations = (
        fire_asof["캐나다지수_기준시각"] > fire_asof["기준시각"]
    ).sum()
    if violations:
        raise AssertionError(f"캐나다 지수 미래시점 위반: {violations}건")
    print("as-of violations: 0")

    terrain = common.load_terrain()
    print("terrain:", terrain.shape)
    print("dem:", common.load_dem_metadata())

    access_lines = common.load_access_lines()
    print("access lines:", {key: value.shape for key, value in access_lines.items()})

    infrastructure = common.load_infrastructure()
    print("infrastructure:", {key: value.shape for key, value in infrastructure.items()})


def validate_heavy_spatial() -> None:
    landcover = common.load_landcover()
    print("landcover:", landcover.shape, "CRS:", landcover.crs)
    del landcover
    gc.collect()

    roads = common.load_roads()
    print("roads:", roads.shape, "CRS:", roads.crs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-hourly",
        action="store_true",
        help="시간 기상 CSV를 표본이 아닌 전체 행으로 검증합니다.",
    )
    parser.add_argument(
        "--heavy-spatial",
        action="store_true",
        help="토지피복과 도로 GPKG를 실제로 전체 로딩합니다.",
    )
    args = parser.parse_args()

    validate_notebooks()
    validate_core(full_hourly=args.full_hourly)
    if args.heavy_spatial:
        validate_heavy_spatial()
    print("all requested loading validations passed")


if __name__ == "__main__":
    main()
