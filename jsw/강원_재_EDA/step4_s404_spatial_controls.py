from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely import wkt
from shapely.geometry import Point
from shapely.prepared import prep
from shapely.strtree import STRtree

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "jsw/강원_재_EDA"
OUT_DIR = MODULE_DIR / "outputs/Step4"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
TARGET_CRS = "EPSG:5186"

SEED = 20260614
CONTROLS_PER_FIRE = 3
EXCLUDE_FIRE_BUFFER_M = 500.0
WUI_THRESHOLD_M = 100.0
ACCESS_THRESHOLD_M = 500.0

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

font_path = "C:/Windows/Fonts/malgun.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

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


def read_wkt_csv(path: Path, *, geom_col: str = "공간좌표") -> gpd.GeoDataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    parsed = df[geom_col].map(lambda value: wkt.loads(value) if isinstance(value, str) and value else None)
    return gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=parsed, crs="EPSG:4326")


def build_nearest_distance_calculator(gdf: gpd.GeoDataFrame) -> Any:
    valid = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid].copy()
    geom_list = valid.geometry.to_numpy()
    if len(geom_list) == 0:
        return lambda point: float("nan")
    tree = STRtree(geom_list)

    def calculate(point: Point) -> float:
        if point is None or point.is_empty:
            return float("nan")
        idx = tree.nearest(point)
        return float(point.distance(geom_list[idx]))

    return calculate


def polygon_parts(geometry) -> list[Any]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if not part.is_empty]
    if geometry.geom_type == "GeometryCollection":
        parts: list[Any] = []
        for part in geometry.geoms:
            parts.extend(polygon_parts(part))
        return parts
    return []


def sample_points_from_geometries(geometries: list[Any], n: int, rng) -> list[Point]:
    geometries = [geom for geom in geometries if geom is not None and not geom.is_empty and geom.area > 0]
    if not geometries or n <= 0:
        return []
    areas = np.array([geom.area for geom in geometries], dtype=float)
    probabilities = areas / areas.sum()
    prepared = [prep(geom) for geom in geometries]
    points: list[Point] = []
    max_attempts = max(n * 250, 2_000)
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        geom_idx = int(rng.choice(len(geometries), p=probabilities))
        polygon = geometries[geom_idx]
        minx, miny, maxx, maxy = polygon.bounds
        point = Point(float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy)))
        if prepared[geom_idx].contains(point):
            points.append(point)
    return points


def build_cell_sampling_geometries(
    grid_gdf: gpd.GeoDataFrame,
    landcover: gpd.GeoDataFrame,
) -> dict[str, list[Any]]:
    sampled_landcover = landcover[
        landcover.geometry.notna()
        & ~landcover.geometry.is_empty
        & landcover.geometry.is_valid
        & ~landcover["L1_NAME"].eq("수역")
    ].copy()
    spatial_index = sampled_landcover.sindex
    result: dict[str, list[Any]] = {}
    for _, cell in grid_gdf.iterrows():
        cell_geometry = cell.geometry
        candidate_idx = spatial_index.query(cell_geometry, predicate="intersects")
        parts: list[Any] = []
        for geometry in sampled_landcover.geometry.iloc[candidate_idx]:
            intersection = geometry.intersection(cell_geometry)
            parts.extend(part for part in polygon_parts(intersection) if part.area > 1.0)
        result[cell["기상셀ID"]] = parts if parts else polygon_parts(cell_geometry)
    return result


def assign_spatial_layer(row: pd.Series) -> str:
    if (
        row["도로_최단거리_m"] <= WUI_THRESHOLD_M
        or row["시가화_최단거리_m"] <= WUI_THRESHOLD_M
        or row["농업_최단거리_m"] <= WUI_THRESHOLD_M
    ):
        return "생활권-WUI"
    if row["임도_최단거리_m"] <= ACCESS_THRESHOLD_M or row["등산로_최단거리_m"] <= ACCESS_THRESHOLD_M:
        return "산림 접근권"
    return "산림 내부"


def attach_landcover(points: gpd.GeoDataFrame, landcover: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        points,
        landcover[["L1_CODE", "L1_NAME", "L2_CODE", "L2_NAME", "polygon_area", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.sort_values(["point_idx", "polygon_area"], na_position="last")
    joined = joined.drop_duplicates("point_idx", keep="first").copy()
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined


def classify_candidates(
    points: gpd.GeoDataFrame,
    landcover: gpd.GeoDataFrame,
    calculators: dict[str, Any],
    fire_distance_calculator: Any,
) -> gpd.GeoDataFrame:
    joined = attach_landcover(points, landcover)
    for column, calculator in calculators.items():
        joined[column] = joined.geometry.map(calculator)
    joined["nearest_fire_distance_m"] = joined.geometry.map(fire_distance_calculator)
    joined["landcover_unmatched"] = joined["L1_NAME"].isna()
    joined["is_water"] = joined["L1_NAME"].eq("수역")
    joined["is_urban_core"] = joined["L1_NAME"].eq("시가화건조지역") & (
        joined["산림_최단거리_m"] > WUI_THRESHOLD_M
    )
    joined["too_close_to_fire"] = joined["nearest_fire_distance_m"] < EXCLUDE_FIRE_BUFFER_M
    joined["eligible"] = ~(
        joined["landcover_unmatched"]
        | joined["is_water"]
        | joined["is_urban_core"]
        | joined["too_close_to_fire"]
    )
    joined["spatial_layer_500"] = joined.apply(assign_spatial_layer, axis=1)
    return joined


def build_inputs() -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any], Any]:
    fire_path = TABLE_DIR / "S4-03_spatial_layer_assignment.csv"
    if not fire_path.exists():
        raise FileNotFoundError(f"S4-03 output is required before S4-04: {fire_path}")

    fire = pd.read_csv(fire_path, encoding="utf-8-sig")
    fire_gdf = gpd.GeoDataFrame(
        fire,
        geometry=gpd.points_from_xy(fire["경도"], fire["위도"]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    grid_df = pd.read_csv(REPO_ROOT / "data/강원도_날씨데이터/강원도날씨_격자.csv", encoding="utf-8-sig")
    grid_gdf = gpd.GeoDataFrame(
        grid_df.drop(columns=["영역WKT_wgs84"]),
        geometry=grid_df["영역WKT_wgs84"].map(wkt.loads),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    grid_gdf = grid_gdf.merge(
        pd.read_csv(
            REPO_ROOT / "data/강원도_날씨데이터/강원도날씨_기후지형유형_셀분류.csv",
            encoding="utf-8-sig",
        )[["기상셀ID", "기후지형유형"]],
        on="기상셀ID",
        how="left",
        suffixes=("", "_climate"),
    )
    if "기후지형유형_climate" in grid_gdf.columns:
        grid_gdf["기후지형유형"] = grid_gdf["기후지형유형"].fillna(grid_gdf["기후지형유형_climate"])
        grid_gdf = grid_gdf.drop(columns=["기후지형유형_climate"])

    landcover = gpd.read_file(
        REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_토지피복도_세분류_병합_1m.gpkg",
        engine="pyogrio",
    ).to_crs(TARGET_CRS)
    landcover["polygon_area"] = landcover.geometry.area

    roads = gpd.read_file(
        REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_병합_도로.gpkg",
        engine="pyogrio",
    ).to_crs(TARGET_CRS)
    trails = read_wkt_csv(REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_등산로.csv").to_crs(TARGET_CRS)
    forest_roads = read_wkt_csv(REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_임도망도.csv").to_crs(TARGET_CRS)

    calculators = {
        "도로_최단거리_m": build_nearest_distance_calculator(roads),
        "시가화_최단거리_m": build_nearest_distance_calculator(landcover[landcover["L1_NAME"].eq("시가화건조지역")]),
        "농업_최단거리_m": build_nearest_distance_calculator(landcover[landcover["L1_NAME"].eq("농업지역")]),
        "산림_최단거리_m": build_nearest_distance_calculator(landcover[landcover["L1_NAME"].eq("산림지역")]),
        "임도_최단거리_m": build_nearest_distance_calculator(forest_roads),
        "등산로_최단거리_m": build_nearest_distance_calculator(trails),
    }
    fire_distance_calculator = build_nearest_distance_calculator(fire_gdf)
    return fire, fire_gdf, grid_gdf, landcover, calculators, fire_distance_calculator


def generate_cell_candidates(
    cell_id: str,
    sampling_geometries: list[Any],
    fire_quota: pd.Series,
    climate_type: str,
    rng,
    landcover: gpd.GeoDataFrame,
    calculators: dict[str, Any],
    fire_distance_calculator: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_by_layer = (fire_quota * CONTROLS_PER_FIRE).to_dict()
    required_layers = list(target_by_layer.keys())
    collected: list[pd.DataFrame] = []
    exclusion_totals = defaultdict(int)
    raw_sampled = 0
    batch_index = 0
    max_raw = max(4_000, min(45_000, int(sum(target_by_layer.values()) * 600)))

    while raw_sampled < max_raw:
        current_counts = (
            pd.concat(collected, ignore_index=True)["spatial_layer_500"].value_counts().to_dict()
            if collected
            else {}
        )
        if all(current_counts.get(layer, 0) >= target_by_layer[layer] for layer in required_layers):
            break

        remaining = sum(max(target_by_layer[layer] - current_counts.get(layer, 0), 0) for layer in required_layers)
        batch_n = int(min(3_000, max(300, remaining * 30)))
        points = sample_points_from_geometries(sampling_geometries, batch_n, rng)
        if not points:
            break
        raw_sampled += len(points)
        point_gdf = gpd.GeoDataFrame(
            {
                "point_idx": [f"{cell_id}_{batch_index}_{i}" for i in range(len(points))],
                "기상셀ID": cell_id,
                "기후지형유형": climate_type,
            },
            geometry=points,
            crs=TARGET_CRS,
        )
        classified = classify_candidates(point_gdf, landcover, calculators, fire_distance_calculator)
        for key in ["landcover_unmatched", "is_water", "is_urban_core", "too_close_to_fire"]:
            exclusion_totals[key] += int(classified[key].sum())
        eligible = classified[classified["eligible"] & classified["spatial_layer_500"].isin(required_layers)].copy()
        if not eligible.empty:
            collected.append(eligible)
        batch_index += 1

    if collected:
        candidates = pd.concat(collected, ignore_index=True)
        candidates = candidates.drop_duplicates("point_idx").copy()
    else:
        candidates = pd.DataFrame()

    audit = {
        "기상셀ID": cell_id,
        "기후지형유형": climate_type,
        "raw_sampled_n": raw_sampled,
        "eligible_candidate_n": len(candidates),
        "max_raw_n": max_raw,
        "sampling_geometry_part_n": len(sampling_geometries),
        "sampling_geometry_area_km2": float(sum(geom.area for geom in sampling_geometries) / 1_000_000),
        **{f"excluded_{key}_n": value for key, value in exclusion_totals.items()},
    }
    for layer, target in target_by_layer.items():
        audit[f"target_{layer}_n"] = int(target)
        audit[f"eligible_{layer}_n"] = int(
            candidates["spatial_layer_500"].eq(layer).sum() if len(candidates) else 0
        )
    return candidates, audit


def assign_controls(fire: pd.DataFrame, candidates: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []

    quota = (
        fire.groupby(["기상셀ID", "기후지형유형", "spatial_layer_500"], dropna=False)
        .size()
        .reset_index(name="fire_n")
    )
    for _, group in quota.iterrows():
        cell_id = group["기상셀ID"]
        climate_type = group["기후지형유형"]
        layer = group["spatial_layer_500"]
        fire_ids = fire.loc[
            fire["기상셀ID"].eq(cell_id) & fire["spatial_layer_500"].eq(layer),
            "fire_id",
        ].tolist()
        subset = candidates[
            candidates["기상셀ID"].eq(cell_id) & candidates["spatial_layer_500"].eq(layer)
        ].copy()
        if len(subset):
            subset = subset.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

        target_controls = len(fire_ids) * CONTROLS_PER_FIRE
        available = min(len(subset), target_controls)
        usable = subset.iloc[:available].copy()
        assigned_fire_count = 0
        rows = []
        for i, fire_id in enumerate(fire_ids):
            start = i * CONTROLS_PER_FIRE
            end = start + CONTROLS_PER_FIRE
            block = usable.iloc[start:end].copy()
            if block.empty:
                continue
            assigned_fire_count += 1
            block["matched_fire_id"] = fire_id
            block["control_index_for_fire"] = range(1, len(block) + 1)
            rows.append(block)
        if rows:
            assignments.append(pd.concat(rows, ignore_index=True))

        support_rows.append(
            {
                "기상셀ID": cell_id,
                "기후지형유형": climate_type,
                "spatial_layer_500": layer,
                "fire_n": len(fire_ids),
                "target_control_n": target_controls,
                "eligible_candidate_n": len(subset),
                "assigned_control_n": int(available),
                "control_per_fire_mean": available / len(fire_ids) if fire_ids else 0.0,
                "fires_with_any_control_n": assigned_fire_count,
                "fires_without_control_n": len(fire_ids) - assigned_fire_count,
                "shortage_n": max(target_controls - len(subset), 0),
            }
        )

    controls = pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame()
    if len(controls):
        controls = controls.reset_index(drop=True)
        controls["control_id"] = [f"C_{i + 1:06d}" for i in range(len(controls))]
        controls["x_5186"] = controls.geometry.x
        controls["y_5186"] = controls.geometry.y
        controls_wgs = gpd.GeoDataFrame(controls, geometry="geometry", crs=TARGET_CRS).to_crs("EPSG:4326")
        controls["경도"] = controls_wgs.geometry.x
        controls["위도"] = controls_wgs.geometry.y

    return controls, pd.DataFrame(support_rows)


def save_outputs(
    fire: pd.DataFrame,
    fire_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    controls: pd.DataFrame,
    support: pd.DataFrame,
    cell_audit: pd.DataFrame,
) -> None:
    output_cols = [
        "control_id",
        "matched_fire_id",
        "control_index_for_fire",
        "기상셀ID",
        "기후지형유형",
        "spatial_layer_500",
        "경도",
        "위도",
        "x_5186",
        "y_5186",
        "L1_CODE",
        "L1_NAME",
        "L2_CODE",
        "L2_NAME",
        "도로_최단거리_m",
        "시가화_최단거리_m",
        "농업_최단거리_m",
        "산림_최단거리_m",
        "임도_최단거리_m",
        "등산로_최단거리_m",
        "nearest_fire_distance_m",
        "landcover_unmatched",
        "is_water",
        "is_urban_core",
        "too_close_to_fire",
        "eligible",
    ]
    controls_for_csv = pd.DataFrame(controls.drop(columns=["geometry"], errors="ignore"))
    controls_for_csv = controls_for_csv[[col for col in output_cols if col in controls_for_csv.columns]]
    controls_for_csv.to_csv(TABLE_DIR / "S4-04_spatial_control_pool.csv", index=False, encoding="utf-8-sig")

    duplicate_keys = int(
        controls_for_csv.duplicated(["matched_fire_id", "control_index_for_fire"]).sum()
        if len(controls_for_csv)
        else 0
    )
    duplicate_xy = int(
        controls_for_csv.assign(
            x_round=controls_for_csv["x_5186"].round(3),
            y_round=controls_for_csv["y_5186"].round(3),
        ).duplicated(["x_round", "y_round"]).sum()
        if len(controls_for_csv)
        else 0
    )
    fire_control_counts = controls_for_csv.groupby("matched_fire_id").size() if len(controls_for_csv) else pd.Series(dtype=int)
    fire_missing_controls = int((~fire["fire_id"].isin(fire_control_counts.index)).sum())
    fires_less_than_target = int((fire_control_counts < CONTROLS_PER_FIRE).sum()) + fire_missing_controls
    missing_required = {
        col: int(controls_for_csv[col].isna().sum())
        for col in ["기상셀ID", "기후지형유형", "spatial_layer_500", "경도", "위도", "L1_NAME"]
        if col in controls_for_csv.columns
    }

    audit_rows = [
        ("analysis_id", "S4-04", ""),
        ("seed", SEED, ""),
        ("analysis_unit", "공간점 1개", ""),
        ("population", "정제 산불 발생지 1,558건이 속한 동일 기상셀 내부의 공간 배경", ""),
        ("comparison_group", "발생지별 동일 기상셀·동일 spatial_layer_500 대조군 최대 3점", ""),
        ("period", f"{fire['기준시각'].min()} ~ {fire['기준시각'].max()}", ""),
        ("controls_per_fire_target", CONTROLS_PER_FIRE, ""),
        ("exclude_fire_buffer_m", EXCLUDE_FIRE_BUFFER_M, ""),
        ("fire_rows_before", len(fire), ""),
        ("fire_unique_ids", fire["fire_id"].nunique(), ""),
        ("fire_duplicate_id_n", int(fire["fire_id"].duplicated().sum()), ""),
        ("target_control_n", len(fire) * CONTROLS_PER_FIRE, ""),
        ("raw_sampled_n", int(cell_audit["raw_sampled_n"].sum()), ""),
        ("eligible_candidate_n", int(cell_audit["eligible_candidate_n"].sum()), ""),
        ("assigned_control_n", len(controls_for_csv), ""),
        ("control_unique_ids", controls_for_csv["control_id"].nunique() if len(controls_for_csv) else 0, ""),
        ("control_duplicate_key_n", duplicate_keys, ""),
        ("control_duplicate_xy_rounded_1mm_n", duplicate_xy, ""),
        ("fires_with_3_controls_n", int((fire_control_counts == CONTROLS_PER_FIRE).sum()), ""),
        ("fires_less_than_3_controls_n", fires_less_than_target, ""),
        ("fires_without_control_n", fire_missing_controls, ""),
        ("control_min_nearest_fire_distance_m", controls_for_csv["nearest_fire_distance_m"].min() if len(controls_for_csv) else pd.NA, ""),
        ("control_landcover_missing_n", missing_required.get("L1_NAME", 0), ""),
    ]
    for col, value in missing_required.items():
        audit_rows.append((f"control_missing_{col}_n", value, ""))
    pd.DataFrame(audit_rows, columns=["metric", "value", "note"]).to_csv(
        TABLE_DIR / "S4-04_spatial_control_pool_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    support = support.sort_values(["기후지형유형", "기상셀ID", "spatial_layer_500"]).reset_index(drop=True)
    support.to_csv(TABLE_DIR / "S4-04_control_sampling_balance.csv", index=False, encoding="utf-8-sig")
    cell_audit.to_csv(TABLE_DIR / "S4-04_control_exclusion_audit.csv", index=False, encoding="utf-8-sig")

    plot_fire_and_controls(fire_gdf, grid_gdf, controls)
    plot_cell_layer_balance(support)


def plot_fire_and_controls(fire_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame, controls: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    grid_gdf.boundary.plot(ax=ax, linewidth=0.45, color="#b8b8b8", alpha=0.85)
    fire_gdf.plot(ax=ax, markersize=7, color="#d73027", alpha=0.45, label="산불 발생지")
    if len(controls):
        control_gdf = gpd.GeoDataFrame(controls, geometry="geometry", crs=TARGET_CRS)
        control_gdf.plot(ax=ax, markersize=5, color="#1f78b4", alpha=0.30, label="공간 대조군")
    ax.set_title("S4-04 산불 발생지와 동일셀·동일공간층 대조군 위치")
    ax.set_xlabel("X 좌표 (EPSG:5186)")
    ax.set_ylabel("Y 좌표 (EPSG:5186)")
    ax.legend(loc="lower left", frameon=True)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "S4-04_fire_and_control_pool_map.png", dpi=200)
    plt.close(fig)


def plot_cell_layer_balance(support: pd.DataFrame) -> None:
    plot_df = support.copy()
    plot_df["cell_label"] = plot_df["기후지형유형"] + " | " + plot_df["기상셀ID"]
    order = (
        plot_df.groupby(["cell_label"], sort=False)["assigned_control_n"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    pivot = (
        plot_df.pivot_table(
            index="cell_label",
            columns="spatial_layer_500",
            values="assigned_control_n",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(order)
    )
    layer_order = [layer for layer in ["생활권-WUI", "산림 접근권", "산림 내부"] if layer in pivot.columns]
    pivot = pivot[layer_order]
    height = max(7, min(18, 0.22 * len(pivot) + 2))
    fig, ax = plt.subplots(figsize=(8, height))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        linewidths=0.25,
        linecolor="#f0f0f0",
        cbar_kws={"label": "배정 대조군 수"},
        ax=ax,
    )
    ax.set_title("S4-04 기상셀·공간층별 배정 대조군 수")
    ax.set_xlabel("공간층")
    ax.set_ylabel("기후지형유형 | 기상셀ID")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "S4-04_control_pool_by_cell_layer.png", dpi=200)
    plt.close(fig)


def main() -> None:
    print("--- S4-04: 공간 대조군 후보풀 생성 시작 ---")
    rng = np.random.default_rng(SEED)
    fire, fire_gdf, grid_gdf, landcover, calculators, fire_distance_calculator = build_inputs()
    cell_sampling_geometries = build_cell_sampling_geometries(grid_gdf, landcover)

    print("분석 ID: S4-04")
    print("분석 질문: 발생지별 동일 기상셀·동일 공간층 공간 대조군을 충분하고 균형 있게 만들 수 있는가?")
    print("분석단위: 공간점 1개")
    print("모집단: 정제 산불 발생지 1,558건이 속한 동일 기상셀 내부의 공간 배경")
    print("비교집단: 발생지별 동일 기상셀·동일 spatial_layer_500 대조군 최대 3점")
    print(f"기간: {fire['기준시각'].min()} ~ {fire['기준시각'].max()}")
    print(
        "통제·층화·매칭: 동일 기상셀ID, 동일 spatial_layer_500, 실제 산불 500m 이내 제외, "
        "수역·토지피복 미매칭·산림 100m 초과 완전 시가화 중심부 제외"
    )

    quotas = fire.groupby(["기상셀ID", "spatial_layer_500"]).size()
    cell_audits: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    for cell_id in sorted(fire["기상셀ID"].dropna().unique()):
        cell_row = grid_gdf[grid_gdf["기상셀ID"].eq(cell_id)]
        if cell_row.empty:
            continue
        cell_quota = quotas.loc[cell_id]
        climate_type = str(cell_row["기후지형유형"].iloc[0])
        candidates, audit = generate_cell_candidates(
            cell_id,
            cell_sampling_geometries.get(cell_id, polygon_parts(cell_row.geometry.iloc[0])),
            cell_quota,
            climate_type,
            rng,
            landcover,
            calculators,
            fire_distance_calculator,
        )
        if len(candidates):
            candidate_frames.append(candidates)
        cell_audits.append(audit)
        print(
            f"{cell_id}: raw={audit['raw_sampled_n']}, eligible={audit['eligible_candidate_n']}, "
            f"target={int(cell_quota.sum() * CONTROLS_PER_FIRE)}"
        )

    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    controls, support = assign_controls(fire, candidates, rng)
    cell_audit = pd.DataFrame(cell_audits)
    save_outputs(fire, fire_gdf, grid_gdf, controls, support, cell_audit)

    print("\n--- S4-04 생성 요약 ---")
    print(f"산불 사건 수: {len(fire):,}")
    print(f"목표 대조군 수: {len(fire) * CONTROLS_PER_FIRE:,}")
    print(f"원시 샘플 수: {int(cell_audit['raw_sampled_n'].sum()):,}")
    print(f"후보 적격 수: {int(cell_audit['eligible_candidate_n'].sum()):,}")
    print(f"배정 대조군 수: {len(controls):,}")
    if len(controls):
        print("공간층별 대조군 수:")
        print(controls["spatial_layer_500"].value_counts())
    print("--- S4-04: 공간 대조군 후보풀 생성 완료 ---")


if __name__ == "__main__":
    main()
