import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

REPO_ROOT = pd.Path if hasattr(pd, "Path") else None # standard path
# We will resolve paths relative to current file
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "jsw/강원_재_EDA"
OUT_DIR = MODULE_DIR / "outputs/Step4"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Font settings
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


def calculate_cliffs_delta(x, y):
    """
    Calculate Cliff's delta effect size.
    numpy vectorized version for performance.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    # Use broadcasting to compare all pairs
    # matrix of shape (n1, n2)
    diff = x[:, None] - y[None, :]
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)
    return (greater - less) / (n1 * n2)


def calculate_cohens_d(x, y):
    """Calculate Cohen's d effect size."""
    x = np.asarray(x)
    y = np.asarray(y)
    n1, n2 = len(x), len(y)
    if n1 <= 1 or n2 <= 1:
        return np.nan
    mean1, mean2 = np.mean(x), np.mean(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def calculate_welch_ci_and_p(x, y):
    """
    Calculate Welch's t-test p-value and 95% confidence interval for mean difference.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n1, n2 = len(x), len(y)
    if n1 <= 1 or n2 <= 1:
        return np.nan, (np.nan, np.nan)
    
    mean1, mean2 = np.mean(x), np.mean(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    
    se = np.sqrt(var1 / n1 + var2 / n2)
    mean_diff = mean1 - mean2
    
    # Welch-Satterthwaite degrees of freedom
    numerator = (var1 / n1 + var2 / n2) ** 2
    denominator = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
    df = numerator / denominator
    
    t_stat = mean_diff / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    
    # 95% CI
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    
    return p_val, (ci_lower, ci_upper)


def extract_terrain_features(df_controls, dem_path):
    print("DEM에서 대조군 포인트의 고도, 경사, TPI 추출 중...")
    elevations = []
    slopes = []
    tpis = []

    with rasterio.open(dem_path) as dataset:
        nodata = dataset.nodata
        res_x, res_y = dataset.res
        dem_arr = dataset.read(1)
        height, width = dem_arr.shape

        for idx, row in df_controls.iterrows():
            x, y = row['x_5186'], row['y_5186']
            r, c = dataset.index(x, y)

            if 1 <= r < height - 1 and 1 <= c < width - 1:
                win = dem_arr[r-1:r+2, c-1:c+2]
                z5 = win[1, 1]

                if z5 == nodata or np.isnan(z5):
                    elevations.append(np.nan)
                    slopes.append(np.nan)
                    tpis.append(np.nan)
                    continue

                # Handle nodata in 3x3 window
                win_clean = np.where((win == nodata) | np.isnan(win), z5, win)

                z1, z2, z3 = win_clean[0, 0], win_clean[0, 1], win_clean[0, 2]
                z4, z5_val, z6 = win_clean[1, 0], win_clean[1, 1], win_clean[1, 2]
                z7, z8, z9 = win_clean[2, 0], win_clean[2, 1], win_clean[2, 2]

                # Horn's method for slope
                dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8.0 * res_x)
                dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8.0 * res_y)
                slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
                slope_deg = np.degrees(slope_rad)

                # TPI
                surrounding_mean = (z1 + z2 + z3 + z4 + z6 + z7 + z8 + z9) / 8.0
                tpi = z5 - surrounding_mean

                elevations.append(z5)
                slopes.append(slope_deg)
                tpis.append(tpi)
            else:
                elevations.append(np.nan)
                slopes.append(np.nan)
                tpis.append(np.nan)

    return elevations, slopes, tpis


def main() -> None:
    print("--- S4-05: 발생지 대 공간 대조군 최소 비교 시작 ---")
    
    # 1. Load inputs
    fire_path = TABLE_DIR / "S4-03_spatial_layer_assignment.csv"
    control_path = TABLE_DIR / "S4-04_spatial_control_pool.csv"
    balance_path = TABLE_DIR / "S4-04_control_sampling_balance.csv"
    dem_path = REPO_ROOT / "data/강원도_데이터/강원도_공간데이터/강원도_DEM_데이터.tif"
    
    if not (fire_path.exists() and control_path.exists() and balance_path.exists() and dem_path.exists()):
        raise FileNotFoundError("필수 입력 파일 중 일부가 누락되었습니다.")
        
    df_fire = pd.read_csv(fire_path, encoding="utf-8-sig")
    df_control = pd.read_csv(control_path, encoding="utf-8-sig")
    df_balance = pd.read_csv(balance_path, encoding="utf-8-sig")
    
    print(f"로드 완료: 발생지 {len(df_fire)}행, 대조군 {len(df_control)}행")
    
    # 2. Extract terrain features for controls
    elevs, slopes, tpis = extract_terrain_features(df_control, dem_path)
    df_control["고도(m)"] = elevs
    df_control["경사도(도)"] = slopes
    df_control["TPI(지형위치지수)"] = tpis
    
    # 3. Shortage stratum flag
    # Find cell x layer strata with shortage
    shortage_strata = df_balance[df_balance["shortage_n"] > 0]
    shortage_keys = set(zip(shortage_strata["기상셀ID"], shortage_strata["spatial_layer_500"]))
    
    df_fire["control_shortage_stratum"] = df_fire.apply(
        lambda r: (r["기상셀ID"], r["spatial_layer_500"]) in shortage_keys, axis=1
    )
    df_control["control_shortage_stratum"] = df_control.apply(
        lambda r: (r["기상셀ID"], r["spatial_layer_500"]) in shortage_keys, axis=1
    )
    
    # 4. Filter Primary Analysis Samples
    # High quality matched fires
    matched_fire_ids = df_control["matched_fire_id"].dropna().unique()
    df_fire_primary = df_fire[df_fire["fire_id"].isin(matched_fire_ids)].copy()
    df_control_primary = df_control[df_control["matched_fire_id"].isin(matched_fire_ids)].copy()
    
    # Unmatched fires (shortage cases)
    df_fire_unmatched = df_fire[~df_fire["fire_id"].isin(matched_fire_ids)].copy()
    
    print(f"주 분석 대상 발생지 수: {len(df_fire_primary)}행, 대조군 수: {len(df_control_primary)}행")
    print(f"대조군 미배정 발생지 수: {len(df_fire_unmatched)}행")
    
    # 5. Sensitivity Analysis Table (S4-05_control_shortage_sensitivity.csv)
    # Compare 1487 matched fires vs 1558 all fires
    num_vars = ["고도(m)", "경사도(도)", "TPI(지형위치지수)", "도로_최단거리_m", "시가화_최단거리_m", 
                "농업_최단거리_m", "산림_최단거리_m", "임도_최단거리_m", "등산로_최단거리_m"]
    
    sens_rows = []
    for var in num_vars:
        for group_name, df_group in [("Matched (n=1487)", df_fire_primary), ("All (n=1558)", df_fire)]:
            valid_vals = df_group[var].dropna()
            sens_rows.append({
                "variable": var,
                "group": group_name,
                "count": len(valid_vals),
                "mean": np.mean(valid_vals),
                "std": np.std(valid_vals, ddof=1),
                "min": np.min(valid_vals),
                "25%": np.percentile(valid_vals, 25),
                "50%": np.median(valid_vals),
                "75%": np.percentile(valid_vals, 75),
                "max": np.max(valid_vals)
            })
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(TABLE_DIR / "S4-05_control_shortage_sensitivity.csv", index=False, encoding="utf-8-sig")
    
    # 6. Stratified Support Audit Table (S4-05_stratified_support_audit.csv)
    support_summary = df_balance.copy()
    support_summary["control_shortage_stratum"] = support_summary["shortage_n"] > 0
    support_summary.to_csv(TABLE_DIR / "S4-05_stratified_support_audit.csv", index=False, encoding="utf-8-sig")
    
    # 7. Statistics Analysis by Variable & Stratum (S4-05_effect_size_by_variable.csv & S4-05_fire_vs_spatial_control_summary.csv)
    # Compare Fire vs Control in Primary Analysis
    # We will analyze overall and by spatial layer
    layers = ["전체", "생활권-WUI", "산림 접근권", "산림 내부"]
    
    stats_rows = []
    summary_rows = []
    
    for layer in layers:
        if layer == "전체":
            f_sub = df_fire_primary
            c_sub = df_control_primary
        else:
            f_sub = df_fire_primary[df_fire_primary["spatial_layer_500"] == layer]
            c_sub = df_control_primary[df_control_primary["spatial_layer_500"] == layer]
            
        n_fire = len(f_sub)
        n_control = len(c_sub)
        print(f"층: {layer} -> 발생지 수: {n_fire}, 대조군 수: {n_control}")
        if n_fire == 0 or n_control == 0:
            continue
            
        p_vals_mwu = []
        p_vals_welch = []
        eff_sizes_d = []
        eff_sizes_cliff = []
        ci_lowers = []
        ci_uppers = []
        var_names = []
        
        for var in num_vars:
            x = f_sub[var].dropna().to_numpy()
            y = c_sub[var].dropna().to_numpy()
            
            # 1) Descriptives
            mean_f, std_f = np.mean(x), np.std(x, ddof=1)
            mean_c, std_c = np.mean(y), np.std(y, ddof=1)
            med_f, med_c = np.median(x), np.median(y)
            
            # Save descriptive comparison
            summary_rows.append({
                "spatial_layer": layer,
                "variable": var,
                "fire_n": len(x),
                "fire_mean": mean_f,
                "fire_std": std_f,
                "fire_median": med_f,
                "control_n": len(y),
                "control_mean": mean_c,
                "control_std": std_c,
                "control_median": med_c,
                "difference": mean_f - mean_c
            })
            
            # 2) Statistical Tests
            if len(x) > 1 and len(y) > 1:
                # Mann-Whitney U
                try:
                    stat_mwu, p_mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
                except Exception:
                    p_mwu = np.nan
                # Welch
                p_welch, (ci_l, ci_u) = calculate_welch_ci_and_p(x, y)
                # Effect sizes
                cohens_d = calculate_cohens_d(x, y)
                cliffs_d = calculate_cliffs_delta(x, y)
            else:
                p_mwu, p_welch = np.nan, np.nan
                cohens_d, cliffs_d = np.nan, np.nan
                ci_l, ci_u = np.nan, np.nan
                
            p_vals_mwu.append(p_mwu)
            p_vals_welch.append(p_welch)
            eff_sizes_d.append(cohens_d)
            eff_sizes_cliff.append(cliffs_d)
            ci_lowers.append(ci_l)
            ci_uppers.append(ci_u)
            var_names.append(var)
            
        # FDR correction per layer
        valid_idx = [i for i, p in enumerate(p_vals_welch) if not np.isnan(p)]
        q_vals_welch = [np.nan] * len(p_vals_welch)
        if valid_idx:
            p_to_correct = [p_vals_welch[i] for i in valid_idx]
            _, corrected_q, _, _ = multipletests(p_to_correct, method="fdr_bh")
            for local_idx, global_idx in enumerate(valid_idx):
                q_vals_welch[global_idx] = corrected_q[local_idx]
                
        # Append stats results
        for i, var in enumerate(var_names):
            stats_rows.append({
                "spatial_layer": layer,
                "variable": var,
                "fire_n": len(f_sub[var].dropna()),
                "control_n": len(c_sub[var].dropna()),
                "difference_in_means": (np.mean(f_sub[var].dropna()) - np.mean(c_sub[var].dropna())) if len(f_sub[var].dropna()) else np.nan,
                "ci_95_lower": ci_lowers[i],
                "ci_95_upper": ci_uppers[i],
                "welch_p_value": p_vals_welch[i],
                "welch_q_value_fdr": q_vals_welch[i],
                "mwu_p_value": p_vals_mwu[i],
                "cohens_d": eff_sizes_d[i],
                "cliffs_delta": eff_sizes_cliff[i]
            })
            
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(TABLE_DIR / "S4-05_effect_size_by_variable.csv", index=False, encoding="utf-8-sig")
    
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(TABLE_DIR / "S4-05_fire_vs_spatial_control_summary.csv", index=False, encoding="utf-8-sig")
    
    # 8. Categorical Variable Crosstab Analysis
    # Compare L1_NAME composition
    lc_crosstab_list = []
    for layer in ["전체", "생활권-WUI"]:
        if layer == "전체":
            f_sub = df_fire_primary
            c_sub = df_control_primary
        else:
            f_sub = df_fire_primary[df_fire_primary["spatial_layer_500"] == layer]
            c_sub = df_control_primary[df_control_primary["spatial_layer_500"] == layer]
            
        f_counts = f_sub["L1_NAME"].value_counts(dropna=False).rename("fire_n")
        c_counts = c_sub["L1_NAME"].value_counts(dropna=False).rename("control_n")
        
        lc_df = pd.concat([f_counts, c_counts], axis=1).fillna(0).astype(int)
        lc_df["fire_pct"] = (lc_df["fire_n"] / lc_df["fire_n"].sum()) * 100
        lc_df["control_pct"] = (lc_df["control_n"] / lc_df["control_n"].sum()) * 100
        lc_df["spatial_layer"] = layer
        lc_crosstab_list.append(lc_df.reset_index().rename(columns={"index": "L1_NAME"}))
        
    df_lc_crosstab = pd.concat(lc_crosstab_list)
    df_lc_crosstab.to_csv(TABLE_DIR / "S4-05_landcover_crosstab.csv", index=False, encoding="utf-8-sig")
    
    # 9. Key Plots Generation
    # Plot 1: ECDF of Key Variable (e.g. 고도(m) 또는 도로_최단거리_m)
    # We will plot ECDF for '도로_최단거리_m' for Fire vs Control in WUI layer
    print("S4-05 주요 변수 ECDF 플롯 생성 중...")
    plt.figure(figsize=(8, 5.5))
    
    # Get WUI samples
    f_wui = df_fire_primary[df_fire_primary["spatial_layer_500"] == "생활권-WUI"].copy()
    c_wui = df_control_primary[df_control_primary["spatial_layer_500"] == "생활권-WUI"].copy()
    
    f_wui["group"] = "산불 발생지 (n=1401)"
    c_wui["group"] = "공간 대조군 (n=4224)"
    
    df_plot_ecdf = pd.concat([f_wui[["도로_최단거리_m", "group"]], c_wui[["도로_최단거리_m", "group"]]])
    
    sns.ecdfplot(
        data=df_plot_ecdf,
        x="도로_최단거리_m",
        hue="group",
        palette={"산불 발생지 (n=1401)": "#d73027", "공간 대조군 (n=4224)": "#1f78b4"},
        linewidth=2.5
    )
    plt.title("생활권-WUI 공간층 내 도로 최단거리(m) 누적분포함수(ECDF) 비교", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("도로 최단거리 (m)", fontsize=11)
    plt.ylabel("누적 비율", fontsize=11)
    plt.xlim(-5, 200) # focus on WUI range
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4-05_key_variable_ecdf.png", dpi=200)
    plt.close()
    
    # Plot 2: Forest Plot of Effect Sizes (Cohen's d or Cliff's delta) by Variable & Layer
    # We will display Cliff's delta for all variables across 3 layers
    print("S4-05 효과크기 포레스트 플롯 생성 중...")
    
    df_plot_eff = df_stats[df_stats["spatial_layer"] != "전체"].copy()
    
    plt.figure(figsize=(9, 7))
    sns.barplot(
        data=df_plot_eff,
        x="cliffs_delta",
        y="variable",
        hue="spatial_layer",
        palette="muted"
    )
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("공간층별/변수별 Cliff's delta 효과크기 (발생지 - 대조군)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Cliff's delta", fontsize=11)
    plt.ylabel("변수명", fontsize=11)
    plt.legend(title="공간층 (500m)", loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4-05_effect_size_forest.png", dpi=200)
    plt.close()
    
    print("--- S4-05: 발생지 대 공간 대조군 최소 비교 완료 ---")


if __name__ == "__main__":
    main()
