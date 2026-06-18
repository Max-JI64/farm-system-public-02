from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pyogrio
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import Point, box
from shapely import wkt


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "Step4"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
TARGET_CRS = "EPSG:5186"


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    path: Path
    kind: str
    note: str = ""


SOURCES = [
    Source(
        "fire",
        "raw_fire_events",
        REPO_ROOT / "data" / "강원도_데이터" / "강원도_산불발생.csv",
        "csv_point_wgs84",
    ),
    Source(
        "clean_fire",
        "clean_fire_events",
        REPO_ROOT / "data" / "학습데이터" / "산불발생_정제.csv",
        "csv_point_wgs84",
        "reference only; Step 4 spatial audit uses raw 3,405 events",
    ),
    Source(
        "weather_grid",
        "weather_cell_polygons",
        REPO_ROOT / "data" / "강원도_날씨데이터" / "강원도날씨_격자.csv",
        "csv_wkt_polygon_wgs84",
    ),
    Source(
        "climate_type",
        "climate_topography_type",
        REPO_ROOT / "data" / "강원도_날씨데이터" / "강원도날씨_기후지형유형_셀분류.csv",
        "csv_table",
    ),
    Source(
        "terrain",
        "fire_terrain_features",
        REPO_ROOT / "data" / "강원도_데이터" / "산불_공간데이터" / "강원도_산불_지형특성계산.csv",
        "csv_point_wgs84",
    ),
    Source(
        "dem",
        "gangwon_dem",
        REPO_ROOT / "data" / "강원도_데이터" / "강원도_공간데이터" / "강원도_DEM_데이터.tif",
        "raster",
    ),
    Source(
        "landcover",
        "landcover_fine_gpkg",
        REPO_ROOT / "data" / "강원도_데이터" / "강원도_공간데이터" / "강원도_토지피복도_세분류_병합_1m.gpkg",
        "gpkg",
        "large file; full metadata plus sample geometry audit",
    ),
    Source(
        "roads",
        "merged_roads_gpkg",
        REPO_ROOT / "data" / "강원도_데이터" / "강원도_공간데이터" / "강원도_병합_도로.gpkg",
        "gpkg",
        "large file; full metadata plus sample geometry audit",
    ),
    Source(
        "trails",
        "hiking_trails",
        REPO_ROOT / "data" / "강원도_데이터" / "강원도_공간데이터" / "강원도_등산로.csv",
        "csv_wkt_line_wgs84",
    ),
    Source(
        "forest_roads",
        "forest_roads",
        REPO_ROOT / "data" / "강원도_데이터" / "강원도_공간데이터" / "강원도_임도망도.csv",
        "csv_wkt_line_wgs84",
    ),
    Source(
        "fire_stations",
        "nearby_fire_stations",
        REPO_ROOT / "data" / "강원도_데이터" / "산불_공간데이터" / "강원도_근방_소방서_위치.csv",
        "csv_point_wgs84",
    ),
    Source(
        "fire_water",
        "nearby_fire_water",
        REPO_ROOT / "data" / "강원도_데이터" / "산불_공간데이터" / "강원도_근방_소방용수시설_위치.csv",
        "csv_point_wgs84",
    ),
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def source_audit() -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        stat = source.path.stat() if source.path.exists() else None
        rows.append(
            {
                "key": source.key,
                "label": source.label,
                "kind": source.kind,
                "path": str(source.path.relative_to(REPO_ROOT)),
                "exists": source.path.exists(),
                "bytes": stat.st_size if stat else pd.NA,
                "modified": pd.Timestamp(stat.st_mtime, unit="s").isoformat()
                if stat
                else pd.NA,
                "note": source.note,
            }
        )
    return pd.DataFrame(rows)


def read_csv_point(path: Path, *, x_col: str = "경도", y_col: str = "위도") -> gpd.GeoDataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    valid = df[x_col].notna() & df[y_col].notna()
    gdf = gpd.GeoDataFrame(
        df.loc[valid].copy(),
        geometry=gpd.points_from_xy(df.loc[valid, x_col], df.loc[valid, y_col]),
        crs="EPSG:4326",
    )
    return gdf


def read_weather_grid(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    geom = df["영역WKT_wgs84"].map(wkt.loads)
    return gpd.GeoDataFrame(df.drop(columns=["영역WKT_wgs84"]), geometry=geom, crs="EPSG:4326")


def read_wkt_csv(path: Path, *, geom_col: str = "공간좌표") -> gpd.GeoDataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    parsed = df[geom_col].map(lambda value: wkt.loads(value) if isinstance(value, str) and value else None)
    return gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=parsed, crs="EPSG:4326")


def crs_to_str(crs: Any) -> str:
    if crs is None:
        return ""
    try:
        return CRS.from_user_input(crs).to_string()
    except Exception:
        return str(crs)


def target_crs_convertible(crs: Any) -> bool:
    if crs is None or crs == "":
        return False
    try:
        CRS.from_user_input(crs)
        Transformer.from_crs(crs, TARGET_CRS, always_xy=True)
        return True
    except Exception:
        return False


def geom_audit_row(label: str, gdf: gpd.GeoDataFrame, source_key: str, audit_scope: str) -> dict[str, Any]:
    geom = gdf.geometry
    bounds = gdf.total_bounds if len(gdf) else [pd.NA, pd.NA, pd.NA, pd.NA]
    geom_types = ",".join(sorted(map(str, geom.geom_type.dropna().unique()))) if len(gdf) else ""
    return {
        "key": source_key,
        "label": label,
        "audit_scope": audit_scope,
        "feature_count": len(gdf),
        "crs": crs_to_str(gdf.crs),
        "target_crs_convertible": target_crs_convertible(gdf.crs),
        "geometry_types": geom_types,
        "null_geometry_count": int(geom.isna().sum()),
        "empty_geometry_count": int(geom.is_empty.fillna(False).sum()),
        "invalid_geometry_count": int((~geom.is_valid.fillna(False)).sum()),
        "minx": bounds[0],
        "miny": bounds[1],
        "maxx": bounds[2],
        "maxy": bounds[3],
    }


def gpkg_metadata_row(source: Source) -> tuple[dict[str, Any], dict[str, Any] | None]:
    layers = pyogrio.list_layers(source.path)
    layer = str(layers[0][0])
    info = pyogrio.read_info(source.path, layer=layer)
    crs = info.get("crs")
    bounds = info.get("total_bounds") or info.get("bbox") or [pd.NA, pd.NA, pd.NA, pd.NA]
    row = {
        "key": source.key,
        "label": source.label,
        "audit_scope": "full_metadata",
        "feature_count": info.get("features", pd.NA),
        "crs": crs_to_str(crs),
        "target_crs_convertible": target_crs_convertible(crs),
        "geometry_types": str(info.get("geometry_type", "")),
        "null_geometry_count": pd.NA,
        "empty_geometry_count": pd.NA,
        "invalid_geometry_count": pd.NA,
        "minx": bounds[0],
        "miny": bounds[1],
        "maxx": bounds[2],
        "maxy": bounds[3],
    }

    sample_row = None
    try:
        sample = pyogrio.read_dataframe(source.path, layer=layer, max_features=5000)
        sample_row = geom_audit_row(source.label, sample, source.key, "first_5000_sample")
    except Exception as exc:
        row["sample_error"] = repr(exc)
    return row, sample_row


def raster_audit_row(source: Source) -> dict[str, Any]:
    with rasterio.open(source.path) as dataset:
        bounds = dataset.bounds
        return {
            "key": source.key,
            "label": source.label,
            "audit_scope": "raster_metadata",
            "feature_count": pd.NA,
            "crs": crs_to_str(dataset.crs),
            "target_crs_convertible": target_crs_convertible(dataset.crs),
            "geometry_types": "Raster",
            "null_geometry_count": pd.NA,
            "empty_geometry_count": pd.NA,
            "invalid_geometry_count": pd.NA,
            "minx": bounds.left,
            "miny": bounds.bottom,
            "maxx": bounds.right,
            "maxy": bounds.top,
            "width": dataset.width,
            "height": dataset.height,
            "resolution_x": dataset.res[0],
            "resolution_y": dataset.res[1],
            "nodata": dataset.nodata,
        }


def build_audits() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, gpd.GeoDataFrame]]:
    loaded: dict[str, gpd.GeoDataFrame] = {}
    rows: list[dict[str, Any]] = []

    fire_points = read_csv_point(SOURCES[0].path)
    loaded["raw_fire_events"] = fire_points
    rows.append(geom_audit_row("raw_fire_events", fire_points, "fire", "full"))

    clean_fire = read_csv_point(SOURCES[1].path)
    loaded["clean_fire_events"] = clean_fire
    rows.append(geom_audit_row("clean_fire_events", clean_fire, "clean_fire", "full"))

    weather_grid = read_weather_grid(SOURCES[2].path)
    loaded["weather_cell_polygons"] = weather_grid
    rows.append(geom_audit_row("weather_cell_polygons", weather_grid, "weather_grid", "full"))

    terrain_points = read_csv_point(SOURCES[4].path)
    loaded["fire_terrain_features"] = terrain_points
    rows.append(geom_audit_row("fire_terrain_features", terrain_points, "terrain", "full"))

    rows.append(raster_audit_row(SOURCES[5]))

    for source in [SOURCES[6], SOURCES[7]]:
        metadata_row, sample_row = gpkg_metadata_row(source)
        rows.append(metadata_row)
        if sample_row:
            rows.append(sample_row)

    trails = read_wkt_csv(SOURCES[8].path)
    loaded["hiking_trails"] = trails
    rows.append(geom_audit_row("hiking_trails", trails, "trails", "full"))

    forest_roads = read_wkt_csv(SOURCES[9].path)
    loaded["forest_roads"] = forest_roads
    rows.append(geom_audit_row("forest_roads", forest_roads, "forest_roads", "full"))

    fire_stations = read_csv_point(SOURCES[10].path)
    loaded["nearby_fire_stations"] = fire_stations
    rows.append(geom_audit_row("nearby_fire_stations", fire_stations, "fire_stations", "full"))

    fire_water = read_csv_point(SOURCES[11].path)
    loaded["nearby_fire_water"] = fire_water
    rows.append(geom_audit_row("nearby_fire_water", fire_water, "fire_water", "full"))

    geom_audit = pd.DataFrame(rows)
    bounds = geom_audit[
        [
            "key",
            "label",
            "audit_scope",
            "crs",
            "target_crs_convertible",
            "minx",
            "miny",
            "maxx",
            "maxy",
        ]
    ].copy()
    return geom_audit, bounds, loaded


def plot_overlay(loaded: dict[str, gpd.GeoDataFrame]) -> None:
    target_layers = {
        name: gdf.to_crs(TARGET_CRS)
        for name, gdf in loaded.items()
        if target_crs_convertible(gdf.crs)
    }
    fig, ax = plt.subplots(figsize=(9, 9))
    grid = target_layers["weather_cell_polygons"]
    grid.boundary.plot(ax=ax, linewidth=0.6, color="#9aa0a6", label="weather cells")

    target_layers["raw_fire_events"].plot(
        ax=ax,
        markersize=6,
        color="#d62728",
        alpha=0.45,
        label="raw fire events",
    )

    for name, color in [
        ("hiking_trails", "#2ca02c"),
        ("forest_roads", "#9467bd"),
    ]:
        layer = target_layers[name]
        layer.sample(min(len(layer), 500), random_state=42).plot(
            ax=ax,
            linewidth=0.35,
            color=color,
            alpha=0.35,
            label=name,
        )

    for name, color in [
        ("nearby_fire_stations", "#1f77b4"),
        ("nearby_fire_water", "#17becf"),
    ]:
        target_layers[name].plot(
            ax=ax,
            markersize=4,
            color=color,
            alpha=0.35,
            label=name,
        )

    ax.set_title("S4-01 Spatial Layer Overlay Quality Map")
    ax.set_xlabel("X (EPSG:5186)")
    ax.set_ylabel("Y (EPSG:5186)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "S4-01_layer_overlay_quality_map.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    src = source_audit()
    if not src["exists"].all():
        missing = src.loc[~src["exists"], "path"].tolist()
        raise FileNotFoundError(f"Missing S4-01 source files: {missing}")
    src.to_csv(TABLE_DIR / "S4-01_spatial_source_audit.csv", index=False, encoding="utf-8-sig")

    geom_audit, bounds, loaded = build_audits()
    geom_audit.to_csv(TABLE_DIR / "S4-01_geometry_validity_audit.csv", index=False, encoding="utf-8-sig")
    bounds.to_csv(TABLE_DIR / "S4-01_layer_bounds_audit.csv", index=False, encoding="utf-8-sig")
    plot_overlay(loaded)

    summary = {
        "source_rows": len(src),
        "geometry_rows": len(geom_audit),
        "non_convertible_layers": int((~geom_audit["target_crs_convertible"].fillna(False)).sum()),
        "invalid_full_geometry_count": int(
            geom_audit.loc[geom_audit["audit_scope"].eq("full"), "invalid_geometry_count"]
            .fillna(0)
            .sum()
        ),
        "plot": str(PLOT_DIR / "S4-01_layer_overlay_quality_map.png"),
    }
    print(summary)


if __name__ == "__main__":
    main()
