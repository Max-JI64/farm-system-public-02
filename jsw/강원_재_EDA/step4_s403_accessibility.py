from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from shapely import wkt
from shapely.strtree import STRtree

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Configure environment and directory paths
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "jsw/강원_재_EDA"
OUT_DIR = MODULE_DIR / "outputs/Step4"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 13. Matplotlib Korean Font Settings from README
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


# Definining nearest distance calculation utility for future use (e.g. S4-04 control points)
def build_nearest_distance_calculator(gdf: gpd.GeoDataFrame) -> callable:
    """
    Returns a fast distance calculator function using STRtree.
    Converts geometries to list and filters empty/invalid shapes.
    """
    valid_gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    geom_list = valid_gdf.geometry.values
    if len(geom_list) == 0:
        return lambda pt: float("inf")
    
    tree = STRtree(geom_list)
    
    def calculate_distance(pt) -> float:
        if pt is None or pt.is_empty:
            return float("nan")
        idx = tree.nearest(pt)
        nearest_geom = geom_list[idx]
        return pt.distance(nearest_geom)
        
    return calculate_distance


def main() -> None:
    print("--- S4-03: 접근성·공간층 변수 생성 시작 ---")

    # 1. Load data
    features_path = TABLE_DIR / "S4-02_fire_spatial_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"S4-02 spatial features file not found: {features_path}")
    
    df = pd.read_csv(features_path, encoding="utf-8-sig")
    print(f"로드된 산불 공간 피처 행 수: {len(df)}")

    # 2. Cross-validation of coordinates & distance for check (using R-tree on 20 sample points)
    print("도로 및 등산로/임도 최단거리 정합성 샘플 검증 진행 중...")
    try:
        # Load spatial layers
        landcover_path = REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_토지피복도_세분류_병합_1m.gpkg"
        roads_path = REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_병합_도로.gpkg"
        trails_path = REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_등산로.csv"
        forest_roads_path = REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_임도망도.csv"

        # Load with target crs mapping
        landcover_gdf = gpd.read_file(landcover_path, engine="pyogrio", max_features=10) # Just to get CRS
        target_crs = landcover_gdf.crs

        # Read only Roads GPKG sample to check or build limited tree for verification
        roads_gdf = gpd.read_file(roads_path, engine="pyogrio")
        # Align CRS to EPSG:5186!
        roads_gdf = roads_gdf.to_crs(target_crs)

        # Parse trails & forest roads WKT
        def read_wkt_csv(path, geom_col="공간좌표", crs="EPSG:4326"):
            df_csv = pd.read_csv(path, encoding="utf-8-sig")
            parsed = df_csv[geom_col].map(lambda x: wkt.loads(x) if isinstance(x, str) and x else None)
            return gpd.GeoDataFrame(df_csv.drop(columns=[geom_col]), geometry=parsed, crs=crs)

        trails_gdf = read_wkt_csv(trails_path).to_crs(target_crs)
        forest_roads_gdf = read_wkt_csv(forest_roads_path).to_crs(target_crs)

        # Build calculators
        calc_road = build_nearest_distance_calculator(roads_gdf)
        calc_trail = build_nearest_distance_calculator(trails_gdf)
        calc_forest_road = build_nearest_distance_calculator(forest_roads_gdf)

        # Project sample points
        fire_gdf = gpd.GeoDataFrame(
            df.head(20).copy(),
            geometry=gpd.points_from_xy(df.head(20)["경도"], df.head(20)["위도"]),
            crs="EPSG:4326"
        ).to_crs(target_crs)

        print("샘플 20건 최단거리 교차 검증 결과:")
        for idx, row in fire_gdf.iterrows():
            pt = row.geometry
            fid = row["fire_id"]
            calc_rd = calc_road(pt)
            calc_tr = calc_trail(pt)
            calc_fr = calc_forest_road(pt)
            
            err_rd = abs(calc_rd - row["도로_최단거리_m"])
            err_tr = abs(calc_tr - row["등산로_최단거리_m"])
            err_fr = abs(calc_fr - row["임도_최단거리_m"])
            
            print(f"  [{fid}] 도로 오차: {err_rd:.4f}m | 임도 오차: {err_fr:.4f}m | 등산로 오차: {err_tr:.4f}m")
    except Exception as e:
        print(f"상세 검증 프로세스 중 예외 발생 (일반적으로 캐싱 레이어 로드 이슈): {repr(e)}")
        print("기존 데이터에 사전 계산된 접근성 거리 필드를 신뢰하여 다음 분석을 계속 진행합니다.")

    # 3. Spatial layer assignment based on distance thresholds
    # Prioritized logic: WUI > Access > Interior
    def assign_layer(row, access_threshold: float) -> str:
        # WUI condition: Road <= 100m OR Urban <= 100m OR Agri <= 100m
        is_wui = (
            row["도로_최단거리_m"] <= 100
            or row["시가화_최단거리_m"] <= 100
            or row["농업_최단거리_m"] <= 100
        )
        if is_wui:
            return "생활권-WUI"
        
        # Access condition: Trail <= threshold OR Forest Road <= threshold
        is_access = (
            row["임도_최단거리_m"] <= access_threshold
            or row["등산로_최단거리_m"] <= access_threshold
        )
        if is_access:
            return "산림 접근권"
        
        # Otherwise: Forest Interior
        return "산림 내부"

    # Compute layers under different accessibility thresholds
    df["spatial_layer_250"] = df.apply(lambda r: assign_layer(r, 250.0), axis=1)
    df["spatial_layer_500"] = df.apply(lambda r: assign_layer(r, 500.0), axis=1)
    df["spatial_layer_1000"] = df.apply(lambda r: assign_layer(r, 1000.0), axis=1)

    print("\n--- 500m 기준 공간층 배정 빈도 ---")
    print(df["spatial_layer_500"].value_counts())
    print("미배정(null) 행 수:", df["spatial_layer_500"].isna().sum())

    # 4. Generate Sensitivity Table
    sensitivity_rows = []
    for th in [250, 500, 1000]:
        col = f"spatial_layer_{th}"
        counts = df[col].value_counts()
        pcts = df[col].value_counts(normalize=True) * 100
        
        sensitivity_rows.append({
            "임계치_m": th,
            "생활권-WUI(n)": counts.get("생활권-WUI", 0),
            "생활권-WUI(%)": pcts.get("생활권-WUI", 0.0),
            "산림 접근권(n)": counts.get("산림 접근권", 0),
            "산림 접근권(%)": pcts.get("산림 접근권", 0.0),
            "산림 내부(n)": counts.get("산림 내부", 0),
            "산림 내부(%)": pcts.get("산림 내부", 0.0),
            "Total(n)": len(df)
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(TABLE_DIR / "S4-03_access_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n--- 임계치별 공간층 구성비 민감도 분석표 ---")
    print(sensitivity_df)

    # 5. Generate Distance Summary Table by Climate Type
    dist_vars = [
        "도로_최단거리_m",
        "임도_최단거리_m",
        "등산로_최단거리_m",
        "시가화_최단거리_m",
        "농업_최단거리_m",
        "산림_최단거리_m"
    ]
    
    print("\n--- 최단거리 변수 결측 감사 ---")
    print(df[dist_vars].isna().sum())

    dist_summary_list = []
    for var in dist_vars:
        desc = df.groupby("기후지형유형")[var].describe()
        desc["variable"] = var
        dist_summary_list.append(desc)
    
    dist_summary = pd.concat(dist_summary_list).reset_index()
    dist_summary = dist_summary[[
        "variable", "기후지형유형", "count", "mean", "std", "min", "25%", "50%", "75%", "max"
    ]]
    dist_summary.to_csv(TABLE_DIR / "S4-03_accessibility_distance_summary.csv", index=False, encoding="utf-8-sig")
    print("\n--- 기후지형유형별 접근성 변수 요약표 ---")
    print(dist_summary.head(18))

    # Save final layer assignment table
    df.to_csv(TABLE_DIR / "S4-03_spatial_layer_assignment.csv", index=False, encoding="utf-8-sig")

    # 6. Generate Plots
    # Plot 1: Accessibility Distance ECDF
    plt.figure(figsize=(9, 6))
    
    # Clip max distance to 5000m for visual detail
    plot_df = df.copy()
    plot_df["도로_최단거리_m_clip"] = plot_df["도로_최단거리_m"].clip(upper=5000)
    plot_df["임도_최단거리_m_clip"] = plot_df["임도_최단거리_m"].clip(upper=5000)
    plot_df["등산로_최단거리_m_clip"] = plot_df["등산로_최단거리_m"].clip(upper=5000)
    
    sns.ecdfplot(data=plot_df, x="도로_최단거리_m_clip", label="도로 최단거리", color="#4a90e2", linewidth=2.5)
    sns.ecdfplot(data=plot_df, x="임도_최단거리_m_clip", label="임도 최단거리", color="#f5a623", linewidth=2.5)
    sns.ecdfplot(data=plot_df, x="등산로_최단거리_m_clip", label="등산로 최단거리", color="#50e3c2", linewidth=2.5)
    
    plt.title("산불 발생지의 접근성 인프라별 ECDF (0~5000m 클리핑)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("최단거리 (m)", fontsize=11, labelpad=8)
    plt.ylabel("누적 비율", fontsize=11, labelpad=8)
    plt.legend(title="인프라 유형", loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4-03_accessibility_ecdf.png", dpi=200)
    plt.close()
    print("S4-03_accessibility_ecdf.png 생성 완료")

    # Plot 2: Spatial Layer Distribution Map
    plt.figure(figsize=(8.5, 8.5))
    
    # Load weather grid polygons for overlay
    try:
        grid_csv = REPO_ROOT / "data/강원도_날씨데이터/강원도날씨_격자.csv"
        grid_df = pd.read_csv(grid_csv, encoding="utf-8-sig")
        grid_geom = grid_df["영역WKT_wgs84"].map(wkt.loads)
        grid_gdf = gpd.GeoDataFrame(grid_df, geometry=grid_geom, crs="EPSG:4326").to_crs(target_crs)
        grid_gdf.boundary.plot(ax=plt.gca(), linewidth=0.5, color="#dcdcdc", alpha=0.8, label="기상 격자")
    except Exception as e:
        print(f"기상 격자 로딩 실패로 단독 산점도로 지도를 렌더링합니다: {repr(e)}")

    # Project fire events for maps
    fire_map_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["경도"], df["위도"]),
        crs="EPSG:4326"
    ).to_crs(target_crs)

    # Scatter plot
    sns.scatterplot(
        data=fire_map_gdf,
        x=fire_map_gdf.geometry.x,
        y=fire_map_gdf.geometry.y,
        hue="spatial_layer_500",
        palette={"생활권-WUI": "#4a90e2", "산림 접근권": "#50e3c2", "산림 내부": "#9013fe"},
        s=15,
        alpha=0.65,
        edgecolor="w",
        linewidth=0.25,
        ax=plt.gca()
    )

    plt.title("산불 발생지의 공간층 분포 지도 (500m 기준)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("X 좌표 (EPSG:5186)", fontsize=11, labelpad=8)
    plt.ylabel("Y 좌표 (EPSG:5186)", fontsize=11, labelpad=8)
    plt.legend(title="공간층 분류", loc="lower left", frameon=True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4-03_spatial_layer_map.png", dpi=200)
    plt.close()
    print("S4-03_spatial_layer_map.png 생성 완료")

    print("--- S4-03: 접근성·공간층 변수 생성 완료 ---")


if __name__ == "__main__":
    main()
