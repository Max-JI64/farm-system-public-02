from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.ops import nearest_points


BASE_DIR = Path("D:/farm-system-public-02")
EDA_DIR = BASE_DIR / "jsw" / "EDA"
SPATIAL_DIR = BASE_DIR / "data" / "산불_공간DB"
LANDCOVER_DIR = SPATIAL_DIR / "산불발생_토지피복도"
FIRE_FEATURE_PATH = EDA_DIR / "fire_spatial_features.csv"
OUT_PATH = EDA_DIR / "landcover_context_features.csv"

MAJOR_CLASSES = [
    "산림지역",
    "농업지역",
    "시가화건조지역",
    "초지",
    "나지",
    "수역",
    "습지",
]

SECTORS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def normalize_major(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text == "시가화/건조지역":
        return "시가화건조지역"
    return text


def angle_to_sector(dx: float, dy: float) -> str:
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx) < 1e-9 and abs(dy) < 1e-9):
        return "CENTER"
    angle = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    idx = int(((angle + 22.5) % 360) // 45)
    return SECTORS_8[idx]


def area_entropy(area_by_sector: dict[str, float]) -> float:
    values = np.array([area_by_sector.get(s, 0.0) for s in SECTORS_8], dtype=float)
    total = values.sum()
    if total <= 0:
        return np.nan
    p = values[values > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(len(SECTORS_8)))


def dominant_sector(area_by_sector: dict[str, float]) -> str:
    values = {k: float(v) for k, v in area_by_sector.items() if k in SECTORS_8}
    if not values or max(values.values()) <= 0:
        return "없음"
    return max(values, key=values.get)


def nearest_category_stats(group: gpd.GeoDataFrame, point_geom, major_name: str) -> tuple[float, str]:
    sub = group[group["major_norm"].eq(major_name)]
    if sub.empty:
        return np.nan, "없음"
    distances = sub.geometry.distance(point_geom)
    idx = distances.idxmin()
    min_dist = float(distances.loc[idx])
    if min_dist <= 1e-7:
        return 0.0, "발생점 포함"
    _, nearest_on_poly = nearest_points(point_geom, sub.loc[idx, "geometry"])
    direction = angle_to_sector(nearest_on_poly.x - point_geom.x, nearest_on_poly.y - point_geom.y)
    return min_dist, direction


def classify_context(row: dict[str, object]) -> str:
    forest = float(row.get("raw_forest_ratio", 0.0) or 0.0)
    built = float(row.get("raw_built_ratio", 0.0) or 0.0)
    agri = float(row.get("raw_agri_ratio", 0.0) or 0.0)
    grass = float(row.get("raw_grass_ratio", 0.0) or 0.0)
    point_major = str(row.get("point_landcover_major", "미확인"))

    if point_major == "시가화건조지역" or (built >= 0.30 and forest < 0.50):
        return "도심/시가화 인접형"
    if point_major == "농업지역" or (agri >= 0.35 and forest < 0.60):
        return "농지/소각 인접형"
    if forest >= 0.80 and built < 0.05 and agri < 0.10:
        return "산림 내부형"
    if forest >= 0.30 and (built >= 0.05 or agri >= 0.10 or grass >= 0.10):
        return "산림 경계/혼재형"
    if forest >= 0.30:
        return "산림 우세형"
    return "비산림 주변형"


def build_features() -> pd.DataFrame:
    fires = pd.read_csv(
        FIRE_FEATURE_PATH,
        usecols=[
            "fire_id",
            "위도",
            "경도",
            "발생지역시도명",
            "발생지역시군구명",
            "발생지역번지",
            "발생원인명",
            "진화소요시간(HH)",
            "피해면적(ha)",
            "is_gangwon",
        ],
    )
    points = gpd.GeoDataFrame(
        fires[["fire_id", "위도", "경도"]],
        geometry=gpd.points_from_xy(fires["경도"], fires["위도"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5186")
    point_map = dict(zip(points["fire_id"], points.geometry))

    rows: list[dict[str, object]] = []
    gpkg_paths = sorted(LANDCOVER_DIR.glob("landcover_300m_clipped_*.gpkg"))
    columns = [
        "산불ID",
        "자료출처",
        "대분류명",
        "중분류명",
        "세분류명",
        "면적_제곱미터",
    ]

    for gpkg_path in gpkg_paths:
        print(f"loading {gpkg_path.name}", flush=True)
        gdf = pyogrio.read_dataframe(gpkg_path, layer="landcover_clipped", columns=columns)
        gdf = gdf.rename(columns={"산불ID": "fire_id"})
        gdf = gdf[gdf["fire_id"].isin(point_map)].copy()
        if gdf.empty:
            continue
        gdf["major_norm"] = gdf["대분류명"].map(normalize_major)
        gdf["area_m2"] = pd.to_numeric(gdf["면적_제곱미터"], errors="coerce").fillna(0.0)
        gdf = gdf[gdf["area_m2"] > 0].copy()

        for fire_id, group in gdf.groupby("fire_id", sort=False):
            point_geom = point_map[fire_id]
            total_area = float(group["area_m2"].sum())
            if total_area <= 0:
                continue

            row: dict[str, object] = {"fire_id": fire_id, "raw_total_lc_area": total_area}
            source_area = group.groupby("자료출처", dropna=False)["area_m2"].sum()
            row["sg05_area_ratio"] = float(source_area.get("SG05", 0.0) / total_area)
            row["sg03_area_ratio"] = float(source_area.get("SG03", 0.0) / total_area)

            class_area = group.groupby("major_norm", dropna=False)["area_m2"].sum()
            for major in MAJOR_CLASSES:
                row[f"raw_{major}_area"] = float(class_area.get(major, 0.0))
                row[f"raw_{major}_ratio"] = float(class_area.get(major, 0.0) / total_area)

            row["raw_forest_ratio"] = row["raw_산림지역_ratio"]
            row["raw_built_ratio"] = row["raw_시가화건조지역_ratio"]
            row["raw_agri_ratio"] = row["raw_농업지역_ratio"]
            row["raw_grass_ratio"] = row["raw_초지_ratio"]

            covers = group[group.geometry.covers(point_geom)]
            if not covers.empty:
                cover_idx = covers["area_m2"].idxmax()
                row["point_landcover_major"] = normalize_major(group.loc[cover_idx, "대분류명"])
                row["point_landcover_middle"] = "" if pd.isna(group.loc[cover_idx, "중분류명"]) else str(group.loc[cover_idx, "중분류명"])
                row["point_landcover_source"] = "" if pd.isna(group.loc[cover_idx, "자료출처"]) else str(group.loc[cover_idx, "자료출처"])
            else:
                row["point_landcover_major"] = "미확인"
                row["point_landcover_middle"] = ""
                row["point_landcover_source"] = ""

            for major, prefix in [
                ("산림지역", "forest"),
                ("시가화건조지역", "built"),
                ("농업지역", "agri"),
                ("초지", "grass"),
            ]:
                dist_m, direction = nearest_category_stats(group, point_geom, major)
                row[f"nearest_{prefix}_dist_m"] = dist_m
                row[f"nearest_{prefix}_direction"] = direction

            forest_group = group[group["major_norm"].eq("산림지역")].copy()
            forest_sector_area = {s: 0.0 for s in SECTORS_8}
            built_sector_area = {s: 0.0 for s in SECTORS_8}
            agri_sector_area = {s: 0.0 for s in SECTORS_8}

            centroids = group.geometry.representative_point()
            sectors = [
                angle_to_sector(geom.x - point_geom.x, geom.y - point_geom.y)
                for geom in centroids
            ]
            tmp = group.assign(_sector=sectors)
            for sector, area in tmp[tmp["major_norm"].eq("산림지역")].groupby("_sector")["area_m2"].sum().items():
                if sector in forest_sector_area:
                    forest_sector_area[sector] = float(area)
            for sector, area in tmp[tmp["major_norm"].eq("시가화건조지역")].groupby("_sector")["area_m2"].sum().items():
                if sector in built_sector_area:
                    built_sector_area[sector] = float(area)
            for sector, area in tmp[tmp["major_norm"].eq("농업지역")].groupby("_sector")["area_m2"].sum().items():
                if sector in agri_sector_area:
                    agri_sector_area[sector] = float(area)

            forest_area = float(sum(forest_sector_area.values()))
            for sector in SECTORS_8:
                row[f"forest_area_{sector}"] = forest_sector_area[sector]
                row[f"forest_ratio_{sector}"] = forest_sector_area[sector] / total_area
                row[f"built_ratio_{sector}"] = built_sector_area[sector] / total_area
                row[f"agri_ratio_{sector}"] = agri_sector_area[sector] / total_area

            row["forest_dominant_direction"] = dominant_sector(forest_sector_area)
            row["forest_direction_entropy"] = area_entropy(forest_sector_area)
            if forest_area > 0:
                east = forest_sector_area["NE"] + forest_sector_area["E"] + forest_sector_area["SE"]
                west = forest_sector_area["NW"] + forest_sector_area["W"] + forest_sector_area["SW"]
                north = forest_sector_area["NW"] + forest_sector_area["N"] + forest_sector_area["NE"]
                south = forest_sector_area["SW"] + forest_sector_area["S"] + forest_sector_area["SE"]
                row["forest_east_west_gradient"] = (east - west) / forest_area
                row["forest_north_south_gradient"] = (north - south) / forest_area
            else:
                row["forest_east_west_gradient"] = np.nan
                row["forest_north_south_gradient"] = np.nan
            row["landcover_context_type"] = classify_context(row)
            row["forest_built_interface"] = bool(row["raw_forest_ratio"] >= 0.20 and row["raw_built_ratio"] >= 0.05)
            row["forest_agri_interface"] = bool(row["raw_forest_ratio"] >= 0.20 and row["raw_agri_ratio"] >= 0.10)
            row["low_forest_built_candidate"] = bool(row["raw_forest_ratio"] < 0.20 and row["raw_built_ratio"] >= 0.10)
            row["forest_absent_300m"] = bool(row["raw_forest_ratio"] <= 0.0)
            row["forest_core_candidate"] = bool(row["raw_forest_ratio"] >= 0.80 and row["raw_built_ratio"] < 0.05 and row["raw_agri_ratio"] < 0.10)
            rows.append(row)

    features = pd.DataFrame(rows)
    out = fires.merge(features, on="fire_id", how="left")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = build_features()
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {OUT_PATH} shape={out.shape}")
    print(out["landcover_context_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
