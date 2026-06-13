from __future__ import annotations

import argparse
import ast
import gc
import json
import re
from pathlib import Path

import re_eda_common as common


HERE = Path(__file__).resolve().parent

EXPECTED_LOG_COUNTS = {
    "Step1_강원도_기후지형및캐나다지수_기본분석": 10,
    "Step2_강원도_산불발생_날씨지수_대조군재분석": 10,
    "Step3_산불발생_선행기상및국지임계치_심화분석": 13,
    "Step4_산불발생_공간지형및대조군_분석": 13,
    "Step5_산불발생_기후공간지수_융합분석": 12,
    "Step6_산불발생_인간활동원인프록시_분석": 10,
}


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
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        if "결과 해석은 이 노트북에 작성하지 않는다." not in markdown:
            raise AssertionError(f"{path.name}: 외부 로그 해석 원칙이 없습니다.")
        if re.search(r"^#{1,6}\s+.*결과\s*해석", markdown, flags=re.MULTILINE):
            raise AssertionError(f"{path.name}: 결과 해석 Markdown 제목이 있습니다.")
        print(f"notebook ok: {path.name} ({len(notebook['cells'])} cells)")


def validate_progress_logs() -> None:
    for stem, expected_count in EXPECTED_LOG_COUNTS.items():
        notebook_path = HERE / f"{stem}.ipynb"
        log_path = HERE / f"{stem}_진행예정로그.md"
        if not notebook_path.exists():
            raise FileNotFoundError(notebook_path)
        if not log_path.exists():
            raise FileNotFoundError(log_path)

        content = log_path.read_text(encoding="utf-8")
        analysis_rows = re.findall(r"\| S[1-6]-\d{2} \|", content)
        if len(analysis_rows) != expected_count:
            raise AssertionError(
                f"{log_path.name}: 분석 ID {len(analysis_rows)}개, 예상 {expected_count}개"
            )
        required_phrases = [
            "결과표",
            "플롯",
            "통합 해석",
            "다음 분석 코드",
            "노트북 Markdown에는 결과 해석을 작성하지 않는다.",
            "시각화 중심",
            "결과 기반 추가 심화 분석 큐",
            "추가 심화 분석 필요 여부와 근거",
            "등록한 모든",
            "Axx 심화 분석",
        ]
        missing = [phrase for phrase in required_phrases if phrase not in content]
        if missing:
            raise AssertionError(f"{log_path.name}: 필수 문구 누락 {missing}")
        print(f"progress log ok: {log_path.name} ({expected_count} analysis IDs)")


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
    validate_progress_logs()
    validate_core(full_hourly=args.full_hourly)
    if args.heavy_spatial:
        validate_heavy_spatial()
    print("all requested loading validations passed")


if __name__ == "__main__":
    main()
