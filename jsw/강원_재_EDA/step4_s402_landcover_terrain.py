from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import pyogrio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

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

pd.set_option("display.max_columns", None)


def main() -> None:
    print("--- S4-02: 발생지 토지피복·지형 변수 결합 시작 ---")

    # 1. Load data
    fire_clean_path = REPO_ROOT / "data/학습데이터/산불발생_정제.csv"
    terrain_path = REPO_ROOT / "data/강원도_데이터/산불_공간데이터/강원도_산불_지형특성계산.csv"
    landcover_path = REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_토지피복도_세분류_병합_1m.gpkg"

    print("데이터 로딩 중...")
    fire_clean = pd.read_csv(fire_clean_path, encoding="utf-8-sig")
    terrain = pd.read_csv(terrain_path, encoding="utf-8-sig")
    landcover = gpd.read_file(landcover_path, engine="pyogrio")

    print(f"정제 산불 행 수: {len(fire_clean)}")
    print(f"지형특성 행 수: {len(terrain)}")
    print(f"토지피복 폴리곤 수: {len(landcover)}")

    # 2. Merge terrain features
    # Suffix coordinates to avoid overlap in merge
    terrain_subset = terrain.drop(columns=["위도", "경도"], errors="ignore")
    fire_merged = pd.merge(fire_clean, terrain_subset, on="fire_id", how="left")
    print(f"지형특성 결합 후 행 수: {len(fire_merged)}")

    # 3. Spatial join with landcover
    # Calculate landcover polygon areas for deduplication priority
    landcover = landcover.copy()
    landcover["polygon_area"] = landcover.geometry.area

    # Convert fire coordinates to GeoDataFrame in landcover CRS
    fire_gdf = gpd.GeoDataFrame(
        fire_merged,
        geometry=gpd.points_from_xy(fire_merged["경도"], fire_merged["위도"]),
        crs="EPSG:4326"
    ).to_crs(landcover.crs)

    print("토지피복도 공간조인 실행 중...")
    joined = gpd.sjoin(fire_gdf, landcover, how="left", predicate="within")
    
    # Audit duplicates and unmatched cases
    raw_sjoin_rows = len(joined)
    duplicated_fire_ids = joined[joined["fire_id"].duplicated(keep=False)]["fire_id"].unique()
    num_duplicates = len(duplicated_fire_ids)
    
    # Deduplicate: Keep the polygon with the minimum area
    joined = joined.sort_values(by=["fire_id", "polygon_area"], na_position="last")
    deduped = joined.drop_duplicates(subset=["fire_id"], keep="first").copy()
    
    # Check unmatched count
    unmatched_mask = deduped["L1_NAME"].isna()
    num_unmatched = unmatched_mask.sum()
    
    # Fill unmatched landcover values
    deduped.loc[unmatched_mask, "L1_NAME"] = "미분류"
    deduped.loc[unmatched_mask, "L2_NAME"] = "미분류"
    deduped.loc[unmatched_mask, "L1_CODE"] = "-1"
    deduped.loc[unmatched_mask, "L2_CODE"] = "-1"

    # Final row counts validation
    final_rows = len(deduped)
    print(f"최종 처리 후 행 수: {final_rows}")
    assert final_rows == len(fire_clean), f"행 수 불일치! 기대치: {len(fire_clean)}, 실제: {final_rows}"

    # Drop index_right column if exists
    if "index_right" in deduped.columns:
        deduped = deduped.drop(columns=["index_right"])

    # 4. Generate Audit Table
    audit_data = {
        "metric": [
            "모집단 행 수 (정제 산불)",
            "공간조인 전 임시 매칭 행 수",
            "다중 매칭(중복) 발생 사건 수",
            "토지피복 공간조인 실패(미매칭) 사건 수",
            "토지피복 공간조인 성공(매칭) 사건 수",
            "최종 처리 후 행 수"
        ],
        "value": [
            len(fire_clean),
            raw_sjoin_rows,
            num_duplicates,
            num_unmatched,
            final_rows - num_unmatched,
            final_rows
        ],
        "ratio_pct": [
            100.0,
            (raw_sjoin_rows / len(fire_clean)) * 100,
            (num_duplicates / len(fire_clean)) * 100,
            (num_unmatched / len(fire_clean)) * 100,
            ((final_rows - num_unmatched) / len(fire_clean)) * 100,
            (final_rows / len(fire_clean)) * 100
        ]
    }
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv(TABLE_DIR / "S4-02_fire_landcover_join_audit.csv", index=False, encoding="utf-8-sig")
    print("\n--- 공간조인 감사 결과 ---")
    print(audit_df)

    # 5. Generate Crosstab Table (Address forest vs Landcover forest)
    # Define address forest flag: addr_type is '임야번지(산)'
    deduped["is_addr_forest"] = deduped["addr_type"] == "임야번지(산)"
    # Define landcover forest flag: L1_NAME is '산림지역'
    deduped["is_landcover_forest"] = deduped["L1_NAME"] == "산림지역"

    crosstab_count = pd.crosstab(
        deduped["addr_type"],
        deduped["is_landcover_forest"],
        margins=True,
        margins_name="Total"
    )
    crosstab_pct = pd.crosstab(
        deduped["addr_type"],
        deduped["is_landcover_forest"],
        normalize="index"
    ) * 100

    # Combine counts and percentages into a neat format
    crosstab_combined = pd.DataFrame(index=crosstab_count.index)
    crosstab_combined["산림 아님(n)"] = crosstab_count[False]
    crosstab_combined["산림지역(n)"] = crosstab_count[True]
    crosstab_combined["Total(n)"] = crosstab_count["Total"]
    
    # Fill percentage for index-based margins
    crosstab_combined.loc["일반번지", "산림 아님(%)"] = crosstab_pct.loc["일반번지", False]
    crosstab_combined.loc["일반번지", "산림지역(%)"] = crosstab_pct.loc["일반번지", True]
    crosstab_combined.loc["임야번지(산)", "산림 아님(%)"] = crosstab_pct.loc["임야번지(산)", False]
    crosstab_combined.loc["임야번지(산)", "산림지역(%)"] = crosstab_pct.loc["임야번지(산)", True]
    
    # For Total row
    total_false_pct = (crosstab_count.loc["Total", False] / crosstab_count.loc["Total", "Total"]) * 100
    total_true_pct = (crosstab_count.loc["Total", True] / crosstab_count.loc["Total", "Total"]) * 100
    crosstab_combined.loc["Total", "산림 아님(%)"] = total_false_pct
    crosstab_combined.loc["Total", "산림지역(%)"] = total_true_pct

    # Reorder columns
    crosstab_combined = crosstab_combined[[
        "산림 아님(n)", "산림 아님(%)", "산림지역(n)", "산림지역(%)", "Total(n)"
    ]]
    crosstab_combined.to_csv(TABLE_DIR / "S4-02_address_forest_landcover_crosstab.csv", encoding="utf-8-sig")
    print("\n--- 번지유형과 실제 토지피복 교차표 ---")
    print(crosstab_combined)

    # 6. Generate Terrain Summary Table by Climate/Topography Type
    terrain_vars = ["고도(m)", "경사도(도)", "TPI(지형위치지수)", "TWI(지형다습지수)"]
    
    # Verify missing values in terrain variables
    print("\n--- 지형 변수 결측 감사 ---")
    print(deduped[terrain_vars].isna().sum())

    terrain_summary_list = []
    for var in terrain_vars:
        desc = deduped.groupby("기후지형유형")[var].describe()
        desc["variable"] = var
        terrain_summary_list.append(desc)
    
    terrain_summary = pd.concat(terrain_summary_list).reset_index()
    # Reorder columns to make it clean
    terrain_summary = terrain_summary[[
        "variable", "기후지형유형", "count", "mean", "std", "min", "25%", "50%", "75%", "max"
    ]]
    terrain_summary.to_csv(TABLE_DIR / "S4-02_fire_terrain_summary.csv", index=False, encoding="utf-8-sig")
    print("\n--- 기후지형유형별 지형 변수 요약표 ---")
    print(terrain_summary.head(12))

    # Save spatial features dataframe
    deduped_df = pd.DataFrame(deduped.drop(columns=["geometry"]))
    deduped_df.to_csv(TABLE_DIR / "S4-02_fire_spatial_features.csv", index=False, encoding="utf-8-sig")

    # 7. Generate Plots
    # Plot 1: Landcover Composition by Climate/Topography Type
    # Avoid nested subplots. Create a single figure.
    plt.figure(figsize=(9, 6))
    
    # Calculate percentage for plot
    lc_comp = deduped.groupby(["기후지형유형", "L1_NAME"]).size().unstack(fill_value=0)
    lc_comp_pct = lc_comp.div(lc_comp.sum(axis=1), axis=0) * 100
    
    # Stacked bar plot
    colors = ["#4a90e2", "#50e3c2", "#b8e986", "#f8e71c", "#f5a623", "#d0021b", "#9b9b9b"]
    lc_comp_pct.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca(),
        color=colors[:len(lc_comp_pct.columns)],
        edgecolor="w",
        linewidth=0.5
    )
    
    plt.title("기후지형유형별 토지피복 대분류 구성비 (%)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("기후지형유형", fontsize=11, labelpad=10)
    plt.ylabel("구성비 (%)", fontsize=11, labelpad=10)
    plt.xticks(rotation=0)
    plt.legend(title="토지피복 대분류", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4-02_landcover_composition.png", dpi=200)
    plt.close()
    print("S4-02_landcover_composition.png 생성 완료")

    # Plot 2: Terrain variables ECDF
    # ECDF plot for Elevation (primary terrain ECDF)
    plt.figure(figsize=(8, 5.5))
    sns.ecdfplot(
        data=deduped,
        x="고도(m)",
        hue="기후지형유형",
        palette="muted",
        linewidth=2
    )
    plt.title("기후지형유형별 고도(m) 누적분포함수(ECDF)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("고도 (m)", fontsize=11)
    plt.ylabel("누적 비율", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4-02_terrain_ecdf_by_type.png", dpi=200)
    plt.savefig(PLOT_DIR / "S4-02_terrain_ecdf_by_type_elevation.png", dpi=200)
    plt.close()
    print("S4-02_terrain_ecdf_by_type_elevation.png 생성 완료")

    # ECDF plots for other terrain variables (slope, tpi, twi)
    for var, name_key in [
        ("경사도(도)", "slope"),
        ("TPI(지형위치지수)", "tpi"),
        ("TWI(지형다습지수)", "twi")
    ]:
        plt.figure(figsize=(8, 5.5))
        sns.ecdfplot(
            data=deduped,
            x=var,
            hue="기후지형유형",
            palette="muted",
            linewidth=2
        )
        plt.title(f"기후지형유형별 {var} 누적분포함수(ECDF)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel(var, fontsize=11)
        plt.ylabel("누적 비율", fontsize=11)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"S4-02_terrain_ecdf_by_type_{name_key}.png", dpi=200)
        plt.close()
        print(f"S4-02_terrain_ecdf_by_type_{name_key}.png 생성 완료")

    print("--- S4-02: 발생지 토지피복·지형 변수 결합 완료 ---")


if __name__ == "__main__":
    main()
