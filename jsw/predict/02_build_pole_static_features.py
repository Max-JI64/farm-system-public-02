from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict_common import (
    CLIMATE_TYPE,
    OUTPUT_DIR,
    SPATIAL_DIR,
    STATIC_REQUIRED_COLUMNS,
    WEATHER_GRID,
    ensure_output_dir,
    validation_frame,
    write_csv,
    write_json,
)


def read_gpkg(path: Path):
    import geopandas as gpd

    try:
        return gpd.read_file(path, engine="pyogrio")
    except Exception:
        return gpd.read_file(path)


def load_points(input_path: Path, limit: int | None = None):
    import geopandas as gpd

    if input_path.suffix.lower() == ".shp":
        gdf = gpd.read_file(input_path)
        if gdf.crs is None:
            raise ValueError(f"CRS가 없는 shapefile입니다: {input_path}")
        gdf = gdf.to_crs("EPSG:4326")
        if "pole_id" not in gdf.columns:
            gdf.insert(0, "pole_id", np.arange(len(gdf), dtype=np.int64))
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
    else:
        frame = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
        rename_map = {}
        if "lon" not in frame.columns and "경도" in frame.columns:
            rename_map["경도"] = "lon"
        if "lat" not in frame.columns and "위도" in frame.columns:
            rename_map["위도"] = "lat"
        frame = frame.rename(columns=rename_map)
        if "pole_id" not in frame.columns:
            frame.insert(0, "pole_id", np.arange(len(frame), dtype=np.int64))
        missing = sorted({"pole_id", "lon", "lat"} - set(frame.columns))
        if missing:
            raise KeyError(f"Pole 입력 필수 컬럼 누락: {missing}")
        gdf = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
            crs="EPSG:4326",
        )
    gdf["pole_id"] = gdf["pole_id"].astype(str)
    gdf = gdf.drop_duplicates("pole_id", keep="first").reset_index(drop=True)
    if limit is not None:
        gdf = gdf.head(limit).copy()
    return gdf


def build_valid_dem_center_mask(band: np.ndarray) -> np.ndarray:
    valid = np.isfinite(band) & (band >= -1000)
    valid_center = np.zeros(valid.shape, dtype=bool)
    center = valid[1:-1, 1:-1] & (band[1:-1, 1:-1] >= -100)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            center &= valid[1 + dr : band.shape[0] - 1 + dr, 1 + dc : band.shape[1] - 1 + dc]
    valid_center[1:-1, 1:-1] = center
    return valid_center


def dem_metrics_from_window(data: np.ndarray, dx: float, dy: float):
    if data.shape != (3, 3) or np.any(~np.isfinite(data)) or np.any(data < -1000):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    elevation = float(data[1, 1]) if data[1, 1] >= -100 else np.nan
    tpi = float(data[1, 1] - np.mean(data))
    dz_dx = (data[1, 2] - data[1, 0]) / (2 * dx)
    dz_dy = (data[2, 1] - data[0, 1]) / (2 * dy)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect = (np.degrees(aspect_rad) + 360) % 360

    return (
        elevation,
        float(slope_deg),
        float(np.sin(np.radians(aspect))),
        float(np.cos(np.radians(aspect))),
        tpi,
    )


def calculate_dem_features(points_wgs84, dem_path: Path, use_nearest_fallback: bool = True):
    import rasterio

    with rasterio.open(dem_path) as src:
        pts_proj = points_wgs84.to_crs(src.crs)
        coords = [(pt.x, pt.y) for pt in pts_proj.geometry]

        elevations = np.array([val[0] for val in src.sample(coords)], dtype=float)
        elevations[elevations < -100] = np.nan

        band = src.read(1).astype(float)
        valid_center = build_valid_dem_center_mask(band)
        if use_nearest_fallback and valid_center.any():
            from scipy import ndimage

            nearest_dist_px, nearest_index = ndimage.distance_transform_edt(
                ~valid_center,
                return_indices=True,
            )
        else:
            nearest_dist_px = None
            nearest_index = None

        slopes: list[float] = []
        aspect_sin: list[float] = []
        aspect_cos: list[float] = []
        tpis: list[float] = []
        dem_methods: list[str] = []
        dem_fallback_distances_m: list[float] = []
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)
        pixel_size_m = float((dx + dy) / 2)
        height, width = band.shape

        for i, pt in enumerate(pts_proj.geometry):
            row, col = src.index(pt.x, pt.y)

            original_valid = 1 <= row < height - 1 and 1 <= col < width - 1 and valid_center[row, col]
            if original_valid:
                data = band[row - 1 : row + 2, col - 1 : col + 2]
                elevation, slope, asp_sin, asp_cos, tpi = dem_metrics_from_window(data, dx, dy)
                elevations[i] = elevation
                slopes.append(slope)
                aspect_sin.append(asp_sin)
                aspect_cos.append(asp_cos)
                tpis.append(tpi)
                dem_methods.append("original")
                dem_fallback_distances_m.append(0.0)
                continue

            if use_nearest_fallback and nearest_index is not None:
                clipped_row = int(np.clip(row, 0, height - 1))
                clipped_col = int(np.clip(col, 0, width - 1))
                fallback_row = int(nearest_index[0, clipped_row, clipped_col])
                fallback_col = int(nearest_index[1, clipped_row, clipped_col])
                fallback_valid = bool(valid_center[fallback_row, fallback_col])

                if fallback_valid:
                    data = band[fallback_row - 1 : fallback_row + 2, fallback_col - 1 : fallback_col + 2]
                    fallback_elev, slope, asp_sin, asp_cos, tpi = dem_metrics_from_window(data, dx, dy)

                    raw_elevation_valid = np.isfinite(elevations[i])
                    if not raw_elevation_valid:
                        elevations[i] = fallback_elev

                    out_of_bounds_dr = max(0, -row, row - (height - 1))
                    out_of_bounds_dc = max(0, -col, col - (width - 1))
                    fallback_distance_px = float(
                        np.sqrt(
                            nearest_dist_px[clipped_row, clipped_col] ** 2
                            + out_of_bounds_dr**2
                            + out_of_bounds_dc**2
                        )
                    )

                    slopes.append(slope)
                    aspect_sin.append(asp_sin)
                    aspect_cos.append(asp_cos)
                    tpis.append(tpi)
                    dem_methods.append(
                        "nearest_valid_window_for_terrain"
                        if raw_elevation_valid
                        else "nearest_valid_window_for_all_dem"
                    )
                    dem_fallback_distances_m.append(fallback_distance_px * pixel_size_m)
                    continue

            if np.isfinite(elevations[i]):
                dem_methods.append("unresolved_terrain")
            else:
                dem_methods.append("unresolved_all_dem")
            dem_fallback_distances_m.append(np.nan)
            slopes.append(np.nan)
            aspect_sin.append(np.nan)
            aspect_cos.append(np.nan)
            tpis.append(np.nan)

    return (
        elevations,
        np.array(slopes),
        np.array(aspect_sin),
        np.array(aspect_cos),
        np.array(tpis),
        np.array(dem_methods, dtype=object),
        np.array(dem_fallback_distances_m, dtype=float),
    )


def normalize_landcover_columns(land):
    rename = {}
    if "L1_NAME" not in land.columns and "L1" in land.columns:
        rename["L1"] = "L1_NAME"
    if "L2_NAME" not in land.columns and "L2" in land.columns:
        rename["L2"] = "L2_NAME"
    if rename:
        land = land.rename(columns=rename)
    for col in ["L1_CODE", "L1_NAME", "L2_CODE", "L2_NAME"]:
        if col not in land.columns:
            land[col] = np.nan
    return land


def force_landcover_crs_5186(land):
    """강원도 토지피복 GPKG는 PCS_ITRF2000_TM 계열로 EPSG:5186과 같은 축척이다.

    GeoPandas가 이 WKT를 EPSG:5186으로 인식하지 못하면 전체 geometry 좌표변환을
    시도하면서 bad allocation이 발생할 수 있다. 이 자료는 좌표값 자체가 5186
    범위(약 x=217k~411k, y=493k~643k)이므로 CRS metadata만 EPSG:5186으로
    표준화한다.
    """

    crs_text = "" if land.crs is None else str(land.crs)
    if land.crs is None or "PCS_ITRF2000_TM" in crs_text:
        return land.set_crs("EPSG:5186", allow_override=True)
    try:
        if land.crs.to_epsg() == 5186:
            return land
    except Exception:
        pass
    return land.to_crs("EPSG:5186")


def buffered_bounds(bounds: np.ndarray, buffer_m: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = [float(value) for value in bounds]
    return (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)


def read_landcover_subset(points_5186, buffer_m: float):
    """Pole 주변 bbox의 토지피복만 읽는다.

    학습 코드에서는 전체 토지피복을 읽은 뒤 EPSG:5186으로 맞췄지만,
    Pole 예측에서는 전체 GPKG 변환이 메모리 오류를 일으킬 수 있으므로
    pyogrio bbox filter를 먼저 적용한다. full run에서 Pole bbox가 강원도
    전체에 가까우면 사실상 전체를 읽지만, CRS 전체 변환은 수행하지 않는다.
    """

    import geopandas as gpd

    path = SPATIAL_DIR / "강원도_토지피복도_세분류_병합_1m.gpkg"
    bbox = buffered_bounds(points_5186.total_bounds, buffer_m)
    columns = ["L1_CODE", "L1_NAME", "L2_CODE", "L2_NAME"]

    try:
        import pyogrio

        land = pyogrio.read_dataframe(path, columns=columns, bbox=bbox)
    except Exception:
        land = gpd.read_file(path, engine="pyogrio", bbox=bbox)
        keep = [c for c in [*columns, "geometry"] if c in land.columns]
        land = land[keep].copy()

    land = normalize_landcover_columns(land)
    land = force_landcover_crs_5186(land)
    return land


def attach_weather_cell(points_wgs84):
    import geopandas as gpd

    grid = gpd.read_file(WEATHER_GRID).to_crs("EPSG:4326")
    climate = pd.read_csv(CLIMATE_TYPE, encoding="utf-8-sig")

    joined = gpd.sjoin(
        points_wgs84,
        grid[["기상셀ID", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    joined["weather_cell_match_method"] = np.where(joined["기상셀ID"].notna(), "within", "unmatched")
    joined["weather_cell_distance_m"] = 0.0

    missing_mask = joined["기상셀ID"].isna()
    if missing_mask.any():
        points_5186 = joined.loc[missing_mask].to_crs("EPSG:5186")
        grid_5186 = grid.to_crs("EPSG:5186")
        nearest = gpd.sjoin_nearest(
            points_5186.drop(columns=["기상셀ID"], errors="ignore"),
            grid_5186[["기상셀ID", "geometry"]],
            how="left",
            distance_col="weather_cell_distance_m",
        )
        nearest = nearest.to_crs("EPSG:4326").drop(columns=["index_right"], errors="ignore")
        joined.loc[missing_mask, "기상셀ID"] = nearest["기상셀ID"].values
        joined.loc[missing_mask, "weather_cell_distance_m"] = nearest["weather_cell_distance_m"].values
        joined.loc[missing_mask, "weather_cell_match_method"] = "nearest"

    joined = joined.merge(climate[["기상셀ID", "기후지형유형"]], on="기상셀ID", how="left")
    return joined


def attach_landcover(points_5186, land_5186):
    import geopandas as gpd

    land_keep = land_5186[["L1_CODE", "L1_NAME", "L2_CODE", "L2_NAME", "geometry"]].copy()
    joined = gpd.sjoin(points_5186, land_keep, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].drop(columns=["index_right"], errors="ignore")
    joined = joined.rename(
        columns={
            "L1_CODE": "토지피복_L1_CODE",
            "L1_NAME": "토지피복_L1_NAME",
            "L2_CODE": "토지피복_L2_CODE",
            "L2_NAME": "토지피복_L2_NAME",
        }
    )
    for col in ["토지피복_L1_CODE", "토지피복_L1_NAME", "토지피복_L2_CODE", "토지피복_L2_NAME"]:
        joined[col] = joined[col].fillna("미상")
    joined["토지피복_매칭방식"] = np.where(joined["토지피복_L1_NAME"].eq("미상"), "unmatched", "within")
    return joined


def build_spatial_index(geom_series):
    from shapely.strtree import STRtree

    arr = np.array(list(geom_series), dtype=object)
    tree = STRtree(arr)
    return arr, tree


def nearest_distance(points_gdf, target_gdf) -> np.ndarray:
    if target_gdf.empty:
        return np.full(len(points_gdf), np.nan)

    target_arr, target_tree = build_spatial_index(target_gdf.geometry)
    distances: list[float] = []
    for point in points_gdf.geometry.values:
        nearest = target_tree.nearest(point)
        if nearest is None:
            distances.append(np.nan)
            continue
        if isinstance(nearest, (int, np.integer)):
            geom = target_arr[int(nearest)]
        else:
            geom = nearest
        distances.append(float(point.distance(geom)))
    return np.array(distances, dtype=float)


def read_wkt_csv(path: Path):
    import geopandas as gpd
    from shapely import wkt

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "공간좌표" not in frame.columns:
        raise KeyError(f"{path.name}에 공간좌표 컬럼이 없습니다.")
    return gpd.GeoDataFrame(
        frame,
        geometry=frame["공간좌표"].apply(wkt.loads),
        crs="EPSG:4326",
    ).to_crs("EPSG:5186")


def add_distances(points_5186, land_5186):
    roads = read_gpkg(SPATIAL_DIR / "강원도_병합_도로.gpkg").to_crs("EPSG:5186")
    imdo = read_wkt_csv(SPATIAL_DIR / "강원도_임도망도.csv")
    trail = read_wkt_csv(SPATIAL_DIR / "강원도_등산로.csv")

    urban = land_5186[land_5186["L1_NAME"].eq("시가화건조지역")].copy()
    agri = land_5186[land_5186["L1_NAME"].eq("농업지역")].copy()
    forest = land_5186[land_5186["L1_NAME"].eq("산림지역")].copy()

    points_5186["도로_최단거리_m"] = nearest_distance(points_5186, roads)
    points_5186["시가화_최단거리_m"] = nearest_distance(points_5186, urban)
    points_5186["농업_최단거리_m"] = nearest_distance(points_5186, agri)
    points_5186["임도_최단거리_m"] = nearest_distance(points_5186, imdo)
    points_5186["등산로_최단거리_m"] = nearest_distance(points_5186, trail)
    points_5186["산림_최단거리_m"] = nearest_distance(points_5186, forest)
    return points_5186


def add_landcover_binaries(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["토지피복_산림지역"] = out["토지피복_L1_NAME"].eq("산림지역").astype(np.int8)
    out["토지피복_시가화건조지역"] = out["토지피복_L1_NAME"].eq("시가화건조지역").astype(np.int8)
    out["토지피복_농업지역"] = out["토지피복_L1_NAME"].eq("농업지역").astype(np.int8)
    out["토지피복_초지"] = out["토지피복_L1_NAME"].eq("초지").astype(np.int8)
    out["토지피복_나지"] = out["토지피복_L1_NAME"].eq("나지").astype(np.int8)
    out["토지피복_도로"] = out["토지피복_L2_NAME"].eq("도로").astype(np.int8)
    out["토지피복_활엽수림"] = out["토지피복_L2_NAME"].eq("활엽수림").astype(np.int8)
    out["토지피복_침엽수림"] = out["토지피복_L2_NAME"].eq("침엽수림").astype(np.int8)
    out["토지피복_혼효림"] = out["토지피복_L2_NAME"].eq("혼효림").astype(np.int8)
    out["토지피복_산림유형"] = np.select(
        [
            out["토지피복_L2_NAME"].eq("활엽수림"),
            out["토지피복_L2_NAME"].eq("침엽수림"),
            out["토지피복_L2_NAME"].eq("혼효림"),
            out["토지피복_L1_NAME"].eq("미상"),
        ],
        ["활엽수림", "침엽수림", "혼효림", "미상"],
        default="비산림",
    )
    return out


def assign_spatial_layer(frame: pd.DataFrame) -> pd.Series:
    wui = (
        frame["도로_최단거리_m"].le(100)
        | frame["시가화_최단거리_m"].le(100)
        | frame["농업_최단거리_m"].le(100)
    )
    access = frame["임도_최단거리_m"].le(500) | frame["등산로_최단거리_m"].le(500)
    return pd.Series(
        np.select([wui, access], ["생활권-WUI", "산림 접근권"], default="산림 내부"),
        index=frame.index,
    )


def add_model_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_landcover_binaries(frame)
    out["공간층"] = assign_spatial_layer(out)
    out["비산림_WUI_접경후보"] = (
        out["공간층"].eq("생활권-WUI") & out["토지피복_산림지역"].eq(0)
    ).astype(np.int8)

    out["log1p_도로_최단거리_m"] = np.log1p(out["도로_최단거리_m"])
    out["log1p_시가화거리_m"] = np.log1p(out["시가화_최단거리_m"])
    out["log1p_산림지역_최단거리_m"] = np.log1p(out["산림_최단거리_m"])
    out["log1p_농업거리_m"] = np.log1p(out["농업_최단거리_m"])
    out["log1p_임도_최단거리_m"] = np.log1p(out["임도_최단거리_m"])
    out["log1p_등산로거리_m"] = np.log1p(out["등산로_최단거리_m"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pole별 기상셀·기후지형유형·DEM·거리·토지피복 정적 피처 생성"
    )
    parser.add_argument("--input", type=Path, default=OUTPUT_DIR / "pole_input_points.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit-poles", type=int, default=None, help="샘플 실행용 Pole 개수 제한")
    parser.add_argument(
        "--landcover-buffer-m",
        type=float,
        default=50000.0,
        help="Pole bbox 주변에서 읽을 토지피복 여유거리(m). full run은 대부분 전체 범위가 됩니다.",
    )
    parser.add_argument(
        "--no-dem-nearest-fallback",
        action="store_true",
        help="DEM 3x3 계산 실패 시 가장 가까운 유효 DEM window로 보완하지 않고 기존 방식처럼 결측으로 남김",
    )
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    if not args.input.exists():
        raise FileNotFoundError(f"입력 Pole 표준 좌표 파일이 없습니다. 먼저 01 스크립트를 실행하세요: {args.input}")

    print(f"Pole 입력 로드: {args.input}")
    points = load_points(args.input, args.limit_poles)
    print(f"Pole 수: {len(points):,}")

    print("기상셀 및 기후지형유형 매칭 중...")
    matched = attach_weather_cell(points)

    points_5186 = matched.to_crs("EPSG:5186")

    print("토지피복도 부분 로드 중: 파일이 커서 시간이 걸릴 수 있습니다.")
    land = read_landcover_subset(points_5186, args.landcover_buffer_m)
    print(f"토지피복 subset rows: {len(land):,} (buffer={args.landcover_buffer_m:,.0f}m)")

    print("토지피복 공간조인 중...")
    points_5186 = attach_landcover(points_5186, land)

    print("도로/시가화/농업/산림/임도/등산로 최단거리 계산 중...")
    points_5186 = add_distances(points_5186, land)

    print("DEM 고도·경사·사면방향·TPI 계산 중...")
    points_wgs84 = points_5186.to_crs("EPSG:4326")
    elevs, slopes, asp_sin, asp_cos, tpis, dem_methods, dem_fallback_distances_m = calculate_dem_features(
        points_wgs84,
        SPATIAL_DIR / "강원도_DEM_데이터.tif",
        use_nearest_fallback=not args.no_dem_nearest_fallback,
    )
    points_5186["고도(m)"] = elevs
    points_5186["경사도(도)"] = slopes
    points_5186["사면방향_sin"] = asp_sin
    points_5186["사면방향_cos"] = asp_cos
    points_5186["TPI(지형위치지수)"] = tpis
    points_5186["DEM_보완방식"] = dem_methods
    points_5186["DEM_보완거리_m"] = dem_fallback_distances_m

    result = pd.DataFrame(points_5186.drop(columns="geometry")).copy()
    result = add_model_static_features(result)

    missing_required = sorted(set(STATIC_REQUIRED_COLUMNS) - set(result.columns))
    if missing_required:
        raise KeyError(f"정적 피처 누락 컬럼: {missing_required}")

    static_values = result[STATIC_REQUIRED_COLUMNS].replace([np.inf, -np.inf], np.nan)
    static_missing = static_values.isna()
    static_incomplete_mask = static_missing.any(axis=1)
    model_static_nan = static_missing.sum()
    result["static_feature_complete"] = (~static_incomplete_mask).astype(np.int8)
    dem_method_counts = (
        result["DEM_보완방식"]
        .value_counts(dropna=False)
        .rename_axis("DEM_보완방식")
        .reset_index(name="count")
    )
    dem_fallback_rows = int(result["DEM_보완방식"].astype(str).str.startswith("nearest_valid").sum())
    dem_unresolved_rows = int(result["DEM_보완방식"].astype(str).str.startswith("unresolved").sum())

    model_ready = result.loc[~static_incomplete_mask].copy()
    incomplete_rows = result.loc[static_incomplete_mask].copy()
    if len(incomplete_rows) > 0:
        incomplete_rows["missing_static_features"] = [
            "|".join(static_missing.columns[row_missing].tolist())
            for row_missing in static_missing.loc[static_incomplete_mask].to_numpy()
        ]

    audit = validation_frame(
        [
            ("input_file", str(args.input), True),
            ("limit_poles", args.limit_poles if args.limit_poles is not None else "없음", True),
            ("dem_nearest_fallback_enabled", not args.no_dem_nearest_fallback, True),
            ("pole_rows", f"{len(result):,}", len(result) > 0),
            ("model_ready_rows", f"{len(model_ready):,}", len(model_ready) > 0),
            ("static_incomplete_rows", int(static_incomplete_mask.sum()), int(static_incomplete_mask.sum()) == 0),
            ("dem_nearest_fallback_rows", dem_fallback_rows, True),
            ("dem_unresolved_rows", dem_unresolved_rows, dem_unresolved_rows == 0),
            ("weather_cell_missing", int(result["기상셀ID"].isna().sum()), int(result["기상셀ID"].isna().sum()) == 0),
            ("climate_type_missing", int(result["기후지형유형"].isna().sum()), int(result["기후지형유형"].isna().sum()) == 0),
            ("landcover_unmatched", int(result["토지피복_L1_NAME"].eq("미상").sum()), True),
            ("dem_or_static_model_nan_cells", int(model_static_nan.sum()), int(model_static_nan.sum()) == 0),
        ]
    )
    nan_details = model_static_nan.rename("nan_count").reset_index().rename(columns={"index": "feature"})

    match_cols = [
        "pole_id",
        "lon",
        "lat",
        "기상셀ID",
        "기후지형유형",
        "weather_cell_match_method",
        "weather_cell_distance_m",
    ]
    write_csv(result[match_cols], output_dir / "pole_weather_cell_match.csv")
    write_csv(result, output_dir / "pole_static_features.csv")
    write_csv(model_ready, output_dir / "pole_static_features_model_ready.csv")
    write_csv(incomplete_rows, output_dir / "pole_static_feature_incomplete_rows.csv")
    write_csv(audit, output_dir / "pole_static_feature_audit.csv")
    write_csv(nan_details, output_dir / "pole_static_feature_nan_details.csv")
    write_csv(dem_method_counts, output_dir / "pole_static_feature_dem_fallback_summary.csv")
    write_json(
        {
            "script": "02_build_pole_static_features.py",
            "input": str(args.input),
            "limit_poles": args.limit_poles,
            "landcover_buffer_m": args.landcover_buffer_m,
            "dem_nearest_fallback_enabled": not args.no_dem_nearest_fallback,
            "landcover_subset_rows": int(len(land)),
            "output_dir": str(output_dir),
            "rows": int(len(result)),
            "model_ready_rows": int(len(model_ready)),
            "static_incomplete_rows": int(static_incomplete_mask.sum()),
            "dem_nearest_fallback_rows": dem_fallback_rows,
            "dem_unresolved_rows": dem_unresolved_rows,
            "spatial_logic_note": "학습데이터 step1_spatial_pool.py, step6_feature_engineering.py 및 logistic/build_d2d3_dataset.py 정의를 예측용 Pole에 재적용",
        },
        output_dir / "run_manifest__02_build_pole_static_features.json",
    )
    print(f"완료: {output_dir / 'pole_static_features.csv'}")
    print(f"모델 입력용 완료 행: {output_dir / 'pole_static_features_model_ready.csv'}")
    if len(incomplete_rows) > 0:
        print(f"정적 피처 결측 Pole 목록: {output_dir / 'pole_static_feature_incomplete_rows.csv'}")


if __name__ == "__main__":
    main()
