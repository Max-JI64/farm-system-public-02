from __future__ import annotations

from itertools import combinations
from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests


TYPE_ORDER = ["영동 해안형", "고지·산간형", "영서 내륙형"]
TYPE_COLORS = {
    "영동 해안형": "#E45756",
    "고지·산간형": "#54A24B",
    "영서 내륙형": "#4C78A8",
}
SEASON_ORDER = ["겨울", "봄", "여름", "가을"]


def _prepare_dirs(table_dir: Path, plot_dir: Path) -> tuple[Path, Path]:
    table_dir = Path(table_dir)
    plot_dir = Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, plot_dir


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _fdr(values: pd.Series | list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return multipletests(values, method="fdr_bh")[1]


def _cohen_d_independent(a: pd.Series, b: pd.Series) -> float:
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def _cohen_dz(diff: pd.Series) -> float:
    diff = pd.Series(diff).dropna().astype(float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def _cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if not len(a) or not len(b):
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def _epsilon_squared(h_stat: float, n_total: int) -> float:
    return float(h_stat / (n_total - 1)) if n_total > 1 else np.nan


def _season_from_month(month: pd.Series) -> pd.Series:
    return month.map(
        {
            12: "겨울",
            1: "겨울",
            2: "겨울",
            3: "봄",
            4: "봄",
            5: "봄",
            6: "여름",
            7: "여름",
            8: "여름",
            9: "가을",
            10: "가을",
            11: "가을",
        }
    )


def _merge_climate_type(frame: pd.DataFrame, climate_type: pd.DataFrame) -> pd.DataFrame:
    if "기후지형유형" in frame.columns:
        return frame.copy()
    return frame.merge(
        climate_type[["기상셀ID", "기후지형유형"]],
        on="기상셀ID",
        how="left",
        validate="many_to_one",
    )


def run_s103(
    hourly_weather: pd.DataFrame,
    climate_type: pd.DataFrame,
    table_dir: Path,
    plot_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Compare mutually exclusive fire-caution and non-fire-caution months."""
    table_dir, plot_dir = _prepare_dirs(table_dir, plot_dir)
    variables = ["기온_C", "최저기온_C", "일교차_C", "습도_pct", "풍속_m_s"]

    hourly = hourly_weather.copy()
    hourly["날짜"] = hourly["일시"].dt.normalize()
    daily = (
        hourly.groupby(["기상셀ID", "날짜"], as_index=False)
        .agg(
            기온_C=("기온_C", "mean"),
            최저기온_C=("기온_C", "min"),
            최고기온_C=("기온_C", "max"),
            습도_pct=("습도_pct", "mean"),
            풍속_m_s=("풍속_m_s", "mean"),
            강수량_mm=("강수량_mm", "sum"),
        )
        .merge(
            climate_type[["기상셀ID", "기후권역"]],
            on="기상셀ID",
            how="left",
            validate="many_to_one",
        )
    )
    daily["일교차_C"] = daily["최고기온_C"] - daily["최저기온_C"]
    daily["계절구분"] = np.where(
        daily["날짜"].dt.month.isin([11, 12, 1, 2, 3, 4, 5]),
        "산불조심기간(11~5월)",
        "비산불기(6~10월)",
    )

    cell_season = (
        daily.groupby(["기상셀ID", "기후권역", "계절구분"], as_index=False)[variables]
        .mean()
    )
    paired = cell_season.pivot(
        index=["기상셀ID", "기후권역"],
        columns="계절구분",
        values=variables,
    )

    records: list[dict] = []
    for variable in variables:
        fire = paired[(variable, "산불조심기간(11~5월)")]
        non_fire = paired[(variable, "비산불기(6~10월)")]
        diff = non_fire - fire
        paired_t = stats.ttest_rel(non_fire, fire)
        wilcoxon = stats.wilcoxon(non_fire, fire, alternative="two-sided")
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        margin = stats.t.ppf(0.975, len(diff) - 1) * se
        records.append(
            {
                "변수": variable,
                "N_셀": len(diff),
                "비산불기_평균": non_fire.mean(),
                "산불조심기간_평균": fire.mean(),
                "평균차_비산불기_minus_산불조심기간": diff.mean(),
                "평균차_CI95_하한": diff.mean() - margin,
                "평균차_CI95_상한": diff.mean() + margin,
                "Paired_t": paired_t.statistic,
                "Paired_p": paired_t.pvalue,
                "Wilcoxon_W": wilcoxon.statistic,
                "Wilcoxon_p": wilcoxon.pvalue,
                "Cohen_dz": _cohen_dz(diff),
            }
        )
    paired_tests = pd.DataFrame(records)
    paired_tests["Paired_q_FDR"] = _fdr(paired_tests["Paired_p"])
    paired_tests["Wilcoxon_q_FDR"] = _fdr(paired_tests["Wilcoxon_p"])
    paired_tests.to_csv(
        table_dir / "S1-03_fire_vs_nonfire_paired_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    date_region = (
        daily.groupby(["날짜", "기후권역"], as_index=False)[variables]
        .mean()
        .pivot(index="날짜", columns="기후권역", values=variables)
        .dropna()
    )
    date_records: list[dict] = []
    for variable in variables:
        yd = date_region[(variable, "영동")]
        ys = date_region[(variable, "영서")]
        diff = yd - ys
        paired_t = stats.ttest_rel(yd, ys)
        wilcoxon = stats.wilcoxon(yd, ys, alternative="two-sided")
        date_records.append(
            {
                "변수": variable,
                "N_날짜": len(diff),
                "영동_일평균": yd.mean(),
                "영서_일평균": ys.mean(),
                "평균차_영동_minus_영서": diff.mean(),
                "Paired_t": paired_t.statistic,
                "Paired_p": paired_t.pvalue,
                "Wilcoxon_W": wilcoxon.statistic,
                "Wilcoxon_p": wilcoxon.pvalue,
                "Cohen_dz": _cohen_dz(diff),
                "주의": "날짜 간 자기상관을 고려하지 않은 기술적 대응 비교",
            }
        )
    date_tests = pd.DataFrame(date_records)
    date_tests["Paired_q_FDR"] = _fdr(date_tests["Paired_p"])
    date_tests["Wilcoxon_q_FDR"] = _fdr(date_tests["Wilcoxon_p"])
    date_tests.to_csv(
        table_dir / "S1-03_date_region_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for variable in variables:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(
            data=daily,
            x="기후권역",
            y=variable,
            hue="계절구분",
            split=True,
            inner="quart",
            ax=ax,
        )
        ax.set_title(f"S1-03 {variable}: 산불조심기간과 비산불기 분포")
        ax.set_xlabel("기후권역")
        _save(fig, plot_dir / f"S1-03_seasonal_{variable}.png")

        plot_data = paired.reset_index()
        fig, ax = plt.subplots(figsize=(6.5, 6))
        sns.scatterplot(
            data=plot_data,
            x=(variable, "비산불기(6~10월)"),
            y=(variable, "산불조심기간(11~5월)"),
            hue=("기후권역", ""),
            palette={"영동": "#E45756", "영서": "#4C78A8"},
            ax=ax,
            s=45,
        )
        low = min(
            plot_data[(variable, "비산불기(6~10월)")].min(),
            plot_data[(variable, "산불조심기간(11~5월)")].min(),
        )
        high = max(
            plot_data[(variable, "비산불기(6~10월)")].max(),
            plot_data[(variable, "산불조심기간(11~5월)")].max(),
        )
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        ax.set_title(f"S1-03 셀별 대응 비교: {variable}")
        ax.set_xlabel("비산불기(6~10월) 평균")
        ax.set_ylabel("산불조심기간(11~5월) 평균")
        _save(fig, plot_dir / f"S1-03_paired_{variable}.png")

    return {
        "daily": daily,
        "cell_season": cell_season,
        "paired_tests": paired_tests,
        "date_tests": date_tests,
    }


def run_s104(
    cell_climate: pd.DataFrame,
    climate_type: pd.DataFrame,
    weather_grid: gpd.GeoDataFrame,
    table_dir: Path,
    plot_dir: Path,
) -> dict[str, object]:
    """Audit outliers, then compare K=2..4 clustering with and without flags."""
    table_dir, plot_dir = _prepare_dirs(table_dir, plot_dir)
    features = [
        "평균기온_C",
        "평균_일교차_C",
        "평균습도_pct",
        "평균풍속_m_s",
        "연평균강수량_mm",
    ]
    data = _merge_climate_type(cell_climate, climate_type)

    robust = pd.DataFrame({"기상셀ID": data["기상셀ID"]})
    for variable in features:
        median = data[variable].median()
        mad = np.median(np.abs(data[variable] - median))
        robust[f"{variable}_robust_z"] = (
            0.6745 * (data[variable] - median) / mad if mad > 0 else 0.0
        )
    z_columns = [column for column in robust if column.endswith("_robust_z")]
    robust["극단값_변수수"] = (robust[z_columns].abs() > 4).sum(axis=1)
    # Two-variable extremes can be physically plausible mountain/coastal climates
    # (for example, high wind and precipitation at Misiryeong). Requiring three
    # extreme dimensions limits automatic exclusion to internally unusual cells.
    robust["품질검토대상"] = robust["극단값_변수수"] >= 3
    robust.to_csv(
        table_dir / "S1-04_quality_flag_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    data = data.merge(
        robust[["기상셀ID", "극단값_변수수", "품질검토대상"]],
        on="기상셀ID",
        how="left",
        validate="one_to_one",
    )
    clean = data.loc[~data["품질검토대상"]].copy()

    score_records: list[dict] = []
    center_records: list[pd.DataFrame] = []
    assignments: list[pd.DataFrame] = []
    fitted: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, KMeans, StandardScaler]] = {}
    for dataset_name, frame in [("전체", data), ("품질검토대상제외", clean)]:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(frame[features])
        for k in [2, 3, 4]:
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = model.fit_predict(scaled)
            sizes = pd.Series(labels).value_counts().sort_index()
            score_records.append(
                {
                    "데이터": dataset_name,
                    "K": k,
                    "N_셀": len(frame),
                    "Silhouette": silhouette_score(scaled, labels),
                    "최소군집크기": sizes.min(),
                    "군집크기": "|".join(map(str, sizes.tolist())),
                    "기존3유형_ARI": (
                        adjusted_rand_score(frame["기후지형유형"], labels)
                        if k == 3
                        else np.nan
                    ),
                }
            )
            centers = pd.DataFrame(
                scaler.inverse_transform(model.cluster_centers_),
                columns=features,
            )
            centers.insert(0, "Cluster", range(k))
            centers.insert(0, "K", k)
            centers.insert(0, "데이터", dataset_name)
            center_records.append(centers)
            assignment = frame[
                ["기상셀ID", "기후지형유형", "품질검토대상"]
            ].copy()
            assignment["데이터"] = dataset_name
            assignment["K"] = k
            assignment["Cluster"] = labels
            assignments.append(assignment)
            fitted[(dataset_name, k)] = (scaled, labels, model, scaler)

    scores = pd.DataFrame(score_records)
    centers = pd.concat(center_records, ignore_index=True)
    assignment_table = pd.concat(assignments, ignore_index=True)
    scores.to_csv(
        table_dir / "S1-04_cluster_sensitivity_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    centers.to_csv(
        table_dir / "S1-04_cluster_centers.csv",
        index=False,
        encoding="utf-8-sig",
    )
    assignment_table.to_csv(
        table_dir / "S1-04_cluster_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    clean_k3 = assignment_table[
        (assignment_table["데이터"] == "품질검토대상제외")
        & (assignment_table["K"] == 3)
    ]
    crosstab_k3 = pd.crosstab(
        clean_k3["기후지형유형"],
        clean_k3["Cluster"],
    )
    crosstab_k3.to_csv(
        table_dir / "S1-04_crosstab_k3_quality_clean.csv",
        encoding="utf-8-sig",
    )

    clean_scaled, base_labels, _, _ = fitted[("품질검토대상제외", 3)]
    stability_records = []
    for seed in range(20):
        labels = KMeans(n_clusters=3, random_state=seed, n_init=20).fit_predict(
            clean_scaled
        )
        stability_records.append(
            {
                "random_state": seed,
                "Silhouette": silhouette_score(clean_scaled, labels),
                "ARI_vs_seed42": adjusted_rand_score(base_labels, labels),
                "군집크기": "|".join(
                    map(str, pd.Series(labels).value_counts().sort_index().tolist())
                ),
            }
        )
    stability = pd.DataFrame(stability_records)
    stability.to_csv(
        table_dir / "S1-04_k3_seed_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for k in [2, 3]:
        scaled, labels, _, _ = fitted[("품질검토대상제외", k)]
        map_values = clean[["기상셀ID"]].copy()
        map_values["Cluster"] = labels
        map_data = weather_grid.merge(
            map_values,
            on="기상셀ID",
            how="inner",
            validate="one_to_one",
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        map_data.plot(
            column="Cluster",
            categorical=True,
            legend=True,
            cmap="Set2",
            edgecolor="white",
            linewidth=0.5,
            ax=ax,
        )
        ax.set_title(f"S1-04 품질검토대상 제외 K={k} 공간 분포")
        ax.set_axis_off()
        _save(fig, plot_dir / f"S1-04_cluster_map_k{k}_quality_clean.png")

        sample_values = silhouette_samples(scaled, labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        y_lower = 10
        for cluster in range(k):
            values = np.sort(sample_values[labels == cluster])
            y_upper = y_lower + len(values)
            color = cm.nipy_spectral(float(cluster) / k)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax.text(-0.08, y_lower + len(values) / 2, str(cluster))
            y_lower = y_upper + 10
        mean_score = silhouette_score(scaled, labels)
        ax.axvline(mean_score, color="red", linestyle="--")
        ax.set_xlim(-0.15, 1)
        ax.set_title(f"S1-04 품질검토대상 제외 K={k} silhouette")
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("Cluster")
        _save(fig, plot_dir / f"S1-04_silhouette_k{k}_quality_clean.png")

    return {
        "data": data,
        "clean": clean,
        "quality_audit": robust,
        "scores": scores,
        "centers": centers,
        "stability": stability,
        "crosstab_k3": crosstab_k3,
        "quality_flag_ids": data.loc[
            data["품질검토대상"], "기상셀ID"
        ].tolist(),
    }


def run_s105(
    full_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    table_dir: Path,
    plot_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Validate descriptive separation while keeping construction circularity explicit."""
    table_dir, plot_dir = _prepare_dirs(table_dir, plot_dir)
    variables = [
        "DEM_고도_m",
        "평균기온_C",
        "평균_일최저기온_C",
        "평균_일교차_C",
        "평균습도_pct",
        "평균풍속_m_s",
        "연평균강수량_mm",
    ]

    global_records: list[dict] = []
    for dataset_name, frame in [("전체", full_data), ("품질검토대상제외", clean_data)]:
        for variable in variables:
            groups = [
                frame.loc[frame["기후지형유형"] == group, variable].dropna()
                for group in TYPE_ORDER
            ]
            h_stat, p_value = stats.kruskal(*groups)
            global_records.append(
                {
                    "데이터": dataset_name,
                    "변수": variable,
                    "N_셀": sum(map(len, groups)),
                    "H_stat": h_stat,
                    "p_value": p_value,
                    "epsilon_squared": _epsilon_squared(
                        h_stat,
                        sum(map(len, groups)),
                    ),
                }
            )
    global_tests = pd.DataFrame(global_records)
    global_tests["q_FDR"] = global_tests.groupby("데이터")["p_value"].transform(_fdr)
    global_tests.to_csv(
        table_dir / "S1-05_kruskal_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    descriptive = (
        clean_data.groupby("기후지형유형", observed=True)[variables]
        .agg(["count", "mean", "std", "median"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(
            columns={
                "level_1": "변수",
                "count": "N_셀",
                "mean": "평균",
                "std": "표준편차",
                "median": "중앙값",
            }
        )
    )
    descriptive.to_csv(
        table_dir / "S1-05_type_descriptive_quality_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_records = []
    for variable in variables:
        for group_a, group_b in combinations(TYPE_ORDER, 2):
            a = clean_data.loc[
                clean_data["기후지형유형"] == group_a,
                variable,
            ].dropna()
            b = clean_data.loc[
                clean_data["기후지형유형"] == group_b,
                variable,
            ].dropna()
            test = stats.mannwhitneyu(a, b, alternative="two-sided")
            pair_records.append(
                {
                    "변수": variable,
                    "그룹A": group_a,
                    "그룹B": group_b,
                    "A_N": len(a),
                    "B_N": len(b),
                    "A_중앙값": a.median(),
                    "B_중앙값": b.median(),
                    "MannWhitney_U": test.statistic,
                    "p_value": test.pvalue,
                    "Cliff_delta_A_minus_B": _cliffs_delta(a, b),
                    "Cohen_d_A_minus_B": _cohen_d_independent(a, b),
                }
            )
    pairwise = pd.DataFrame(pair_records)
    pairwise["q_FDR_global"] = _fdr(pairwise["p_value"])
    pairwise.to_csv(
        table_dir / "S1-05_pairwise_quality_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for variable in variables:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            data=clean_data,
            x="기후지형유형",
            y=variable,
            order=TYPE_ORDER,
            hue="기후지형유형",
            palette=TYPE_COLORS,
            legend=False,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=clean_data,
            x="기후지형유형",
            y=variable,
            order=TYPE_ORDER,
            color="black",
            alpha=0.55,
            size=4,
            jitter=0.16,
            ax=ax,
        )
        ax.set_title(f"S1-05 기후지형유형별 {variable}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)
        _save(fig, plot_dir / f"S1-05_boxplot_{variable}.png")

    standardized = clean_data[variables].apply(
        lambda column: (column - column.mean()) / column.std(ddof=0)
    )
    standardized["기후지형유형"] = clean_data["기후지형유형"].values
    profile = standardized.groupby("기후지형유형")[variables].mean().reindex(TYPE_ORDER)
    profile.to_csv(
        table_dir / "S1-05_standardized_profile_quality_clean.csv",
        encoding="utf-8-sig",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        profile,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("S1-05 셀 단위 표준화 후 기후지형유형 평균 프로필")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, plot_dir / "S1-05_standardized_profile_quality_clean.png")

    return {
        "global_tests": global_tests,
        "descriptive": descriptive,
        "pairwise": pairwise,
        "profile": profile,
    }


def run_s106(
    hourly_weather: pd.DataFrame,
    climate_type: pd.DataFrame,
    table_dir: Path,
    plot_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build equal-cell monthly summaries and correctly name hourly wind maxima."""
    table_dir, plot_dir = _prepare_dirs(table_dir, plot_dir)
    climate = climate_type[["기상셀ID", "기후지형유형"]]
    hourly = hourly_weather.copy()
    hourly["날짜"] = hourly["일시"].dt.normalize()
    hourly["월"] = hourly["일시"].dt.month
    hourly["서풍계열_여부"] = (
        (hourly["풍향_deg"] >= 225) & (hourly["풍향_deg"] <= 315)
    ).astype(int)
    daily = (
        hourly.groupby(["기상셀ID", "날짜", "월"], as_index=False)
        .agg(
            평균기온_C=("기온_C", "mean"),
            최고기온_C=("기온_C", "max"),
            최저기온_C=("기온_C", "min"),
            평균상대습도_pct=("습도_pct", "mean"),
            일강수량_mm=("강수량_mm", "sum"),
            일최대시간풍속_m_s=("풍속_m_s", "max"),
            서풍계열_비율=("서풍계열_여부", "mean"),
        )
        .merge(climate, on="기상셀ID", how="left", validate="many_to_one")
    )
    daily["일교차_C"] = daily["최고기온_C"] - daily["최저기온_C"]
    variables = [
        "평균기온_C",
        "일교차_C",
        "평균상대습도_pct",
        "일최대시간풍속_m_s",
        "일강수량_mm",
        "서풍계열_비율",
    ]

    cell_month = (
        daily.groupby(["기상셀ID", "기후지형유형", "월"], as_index=False)[variables]
        .mean()
    )
    monthly_summary = (
        cell_month.groupby(["기후지형유형", "월"], observed=True)[variables]
        .agg(["count", "mean", "std", "median"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(
            columns={
                "level_2": "변수",
                "count": "N_셀",
                "mean": "평균",
                "std": "표준편차",
                "median": "중앙값",
            }
        )
    )
    monthly_summary.to_csv(
        table_dir / "S1-06_monthly_climate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_records = []
    for month in range(1, 13):
        month_data = cell_month[cell_month["월"] == month]
        for variable in variables:
            groups = [
                month_data.loc[
                    month_data["기후지형유형"] == group,
                    variable,
                ].dropna()
                for group in TYPE_ORDER
            ]
            h_stat, p_value = stats.kruskal(*groups)
            test_records.append(
                {
                    "월": month,
                    "변수": variable,
                    "N_셀": sum(map(len, groups)),
                    "H_stat": h_stat,
                    "p_value": p_value,
                    "epsilon_squared": _epsilon_squared(
                        h_stat,
                        sum(map(len, groups)),
                    ),
                }
            )
    monthly_tests = pd.DataFrame(test_records)
    monthly_tests["q_FDR_global"] = _fdr(monthly_tests["p_value"])
    monthly_tests.to_csv(
        table_dir / "S1-06_monthly_type_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_month = (
        cell_month.groupby(["기후지형유형", "월"], observed=True)[variables]
        .agg(["mean", "std", "count"])
    )
    for variable in variables:
        fig, ax = plt.subplots(figsize=(9, 6))
        for group in TYPE_ORDER:
            subset = group_month.loc[group]
            mean = subset[(variable, "mean")]
            se = subset[(variable, "std")] / np.sqrt(subset[(variable, "count")])
            ax.plot(
                mean.index,
                mean.values,
                marker="o",
                color=TYPE_COLORS[group],
                label=group,
            )
            ax.fill_between(
                mean.index,
                mean - 1.96 * se,
                mean + 1.96 * se,
                color=TYPE_COLORS[group],
                alpha=0.15,
            )
        ax.set_xticks(range(1, 13))
        ax.set_title(f"S1-06 월별 {variable} (셀 평균과 95% CI)")
        ax.set_xlabel("월")
        ax.set_ylabel(variable)
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, plot_dir / f"S1-06_monthly_trend_{variable}.png")

        pivot = (
            cell_month.groupby(["기후지형유형", "월"], observed=True)[variable]
            .mean()
            .unstack("월")
            .reindex(TYPE_ORDER)
        )
        z_values = (pivot - pivot.stack().mean()) / pivot.stack().std(ddof=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            z_values,
            annot=True,
            fmt=".1f",
            cmap="coolwarm",
            center=0,
            ax=ax,
        )
        ax.set_title(f"S1-06 월별 {variable} 표준화 평균")
        ax.set_xlabel("월")
        ax.set_ylabel("")
        _save(fig, plot_dir / f"S1-06_monthly_heatmap_{variable}.png")

    wind = hourly[["기상셀ID", "풍향_deg"]].merge(
        climate,
        on="기상셀ID",
        how="left",
        validate="many_to_one",
    )
    bins = np.arange(0, 361, 22.5)
    wind["풍향구간"] = pd.cut(
        wind["풍향_deg"] % 360,
        bins=bins,
        include_lowest=True,
        right=False,
    )
    wind_frequency = (
        wind.groupby(["기후지형유형", "풍향구간"], observed=True)
        .size()
        .rename("건수")
        .reset_index()
    )
    wind_frequency["비율"] = wind_frequency.groupby("기후지형유형")["건수"].transform(
        lambda values: values / values.sum()
    )
    wind_frequency.to_csv(
        table_dir / "S1-06_wind_direction_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    theta = np.deg2rad(np.arange(11.25, 360, 22.5))
    for group in TYPE_ORDER:
        values = (
            wind_frequency[wind_frequency["기후지형유형"] == group]
            .sort_values("풍향구간")["비율"]
            .to_numpy()
        )
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        ax.bar(
            theta,
            values,
            width=np.deg2rad(22.5),
            color=TYPE_COLORS[group],
            alpha=0.8,
            edgecolor="white",
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"S1-06 {group} 풍향 빈도")
        _save(fig, plot_dir / f"S1-06_wind_direction_{group}.png")

    return {
        "daily": daily,
        "cell_month": cell_month,
        "monthly_summary": monthly_summary,
        "monthly_tests": monthly_tests,
        "wind_frequency": wind_frequency,
    }


def run_s107(
    full_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    table_dir: Path,
    plot_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Report overall, sensitivity, and within-type altitude associations."""
    table_dir, plot_dir = _prepare_dirs(table_dir, plot_dir)
    variables = [
        "평균기온_C",
        "평균_일최저기온_C",
        "평균_일교차_C",
        "평균습도_pct",
        "평균풍속_m_s",
        "연평균강수량_mm",
    ]
    records = []
    for dataset_name, frame in [("전체", full_data), ("품질검토대상제외", clean_data)]:
        group_frames = [("전체권역", frame)]
        if dataset_name == "품질검토대상제외":
            group_frames.extend(
                (group, frame[frame["기후지형유형"] == group])
                for group in TYPE_ORDER
            )
        for group_name, group_frame in group_frames:
            for variable in variables:
                rho, p_value = stats.spearmanr(
                    group_frame["DEM_고도_m"],
                    group_frame[variable],
                )
                records.append(
                    {
                        "데이터": dataset_name,
                        "층": group_name,
                        "변수": variable,
                        "N_셀": len(group_frame),
                        "Spearman_rho": rho,
                        "p_value": p_value,
                    }
                )
    correlations = pd.DataFrame(records)
    correlations["q_FDR_global"] = _fdr(correlations["p_value"])
    correlations.to_csv(
        table_dir / "S1-07_altitude_spearman_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    max_altitude = int(np.ceil(clean_data["DEM_고도_m"].max() / 200) * 200)
    clean_data = clean_data.copy()
    clean_data["고도구간"] = pd.cut(
        clean_data["DEM_고도_m"],
        bins=np.arange(0, max_altitude + 201, 200),
        include_lowest=True,
    )
    altitude_bins = (
        clean_data.groupby("고도구간", observed=True)[variables]
        .agg(["count", "mean", "std", "median"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(
            columns={
                "level_1": "변수",
                "count": "N_셀",
                "mean": "평균",
                "std": "표준편차",
                "median": "중앙값",
            }
        )
    )
    altitude_bins.to_csv(
        table_dir / "S1-07_altitude_bins_200m_quality_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nonlinear_records = []
    x = clean_data["DEM_고도_m"].to_numpy(dtype=float)
    for variable in variables:
        y = clean_data[variable].to_numpy(dtype=float)
        linear_coef = np.polyfit(x, y, deg=1)
        quadratic_coef = np.polyfit(x, y, deg=2)
        linear_pred = np.polyval(linear_coef, x)
        quadratic_pred = np.polyval(quadratic_coef, x)
        ss_total = np.sum((y - y.mean()) ** 2)
        linear_r2 = 1 - np.sum((y - linear_pred) ** 2) / ss_total
        quadratic_r2 = 1 - np.sum((y - quadratic_pred) ** 2) / ss_total
        nonlinear_records.append(
            {
                "변수": variable,
                "선형_R2": linear_r2,
                "이차_R2": quadratic_r2,
                "이차_minus_선형_R2": quadratic_r2 - linear_r2,
                "이차항계수": quadratic_coef[0],
            }
        )
    nonlinear = pd.DataFrame(nonlinear_records)
    nonlinear.to_csv(
        table_dir / "S1-07_nonlinearity_summary_quality_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for variable in variables:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=clean_data,
            x="DEM_고도_m",
            y=variable,
            hue="기후지형유형",
            hue_order=TYPE_ORDER,
            palette=TYPE_COLORS,
            s=42,
            alpha=0.75,
            ax=ax,
        )
        smooth = lowess(
            clean_data[variable],
            clean_data["DEM_고도_m"],
            frac=0.45,
            return_sorted=True,
        )
        ax.plot(
            smooth[:, 0],
            smooth[:, 1],
            color="black",
            linestyle="--",
            linewidth=2,
            label="LOWESS",
        )
        ax.set_title(f"S1-07 고도와 {variable}")
        ax.set_xlabel("중심점 DEM 고도 (m)")
        ax.legend()
        _save(fig, plot_dir / f"S1-07_altitude_{variable}.png")

    return {
        "correlations": correlations,
        "altitude_bins": altitude_bins,
        "nonlinear": nonlinear,
    }


def run_s108(
    canadian_indices: pd.DataFrame,
    climate_type: pd.DataFrame,
    table_dir: Path,
    plot_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Use cells, not repeated cell-days, as the independent regional units."""
    table_dir, plot_dir = _prepare_dirs(table_dir, plot_dir)
    data = _merge_climate_type(canadian_indices, climate_type)
    data["날짜"] = pd.to_datetime(data["날짜"])
    data["월"] = data["날짜"].dt.month
    data["계절"] = _season_from_month(data["월"])
    variables = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
    data = data.dropna(subset=variables)

    cell_overall = (
        data.groupby(["기상셀ID", "기후지형유형"], as_index=False)[variables]
        .mean()
    )
    test_records = []
    pair_records = []
    for variable in variables:
        groups = [
            cell_overall.loc[
                cell_overall["기후지형유형"] == group,
                variable,
            ].dropna()
            for group in TYPE_ORDER
        ]
        h_stat, p_value = stats.kruskal(*groups)
        test_records.append(
            {
                "FWI지수": variable,
                "N_셀": sum(map(len, groups)),
                "H_stat": h_stat,
                "p_value": p_value,
                "epsilon_squared": _epsilon_squared(
                    h_stat,
                    sum(map(len, groups)),
                ),
            }
        )
        for group_a, group_b in combinations(TYPE_ORDER, 2):
            a = cell_overall.loc[
                cell_overall["기후지형유형"] == group_a,
                variable,
            ]
            b = cell_overall.loc[
                cell_overall["기후지형유형"] == group_b,
                variable,
            ]
            test = stats.mannwhitneyu(a, b, alternative="two-sided")
            pair_records.append(
                {
                    "FWI지수": variable,
                    "그룹A": group_a,
                    "그룹B": group_b,
                    "A_N": len(a),
                    "B_N": len(b),
                    "MannWhitney_U": test.statistic,
                    "p_value": test.pvalue,
                    "Cliff_delta_A_minus_B": _cliffs_delta(a, b),
                }
            )
    regional_tests = pd.DataFrame(test_records)
    regional_tests["q_FDR"] = _fdr(regional_tests["p_value"])
    regional_tests.to_csv(
        table_dir / "S1-08_fwi_cell_level_kruskal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairwise = pd.DataFrame(pair_records)
    pairwise["q_FDR_global"] = _fdr(pairwise["p_value"])
    pairwise.to_csv(
        table_dir / "S1-08_fwi_cell_level_pairwise.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cell_overall.to_csv(
        table_dir / "S1-08_fwi_cell_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cell_season = (
        data.groupby(
            ["기상셀ID", "기후지형유형", "계절"],
            as_index=False,
            observed=True,
        )[variables]
        .mean()
    )
    season_summary = (
        cell_season.groupby(["기후지형유형", "계절"], observed=True)[variables]
        .agg(["count", "mean", "std", "median"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(
            columns={
                "level_2": "FWI지수",
                "count": "N_셀",
                "mean": "평균",
                "std": "표준편차",
                "median": "중앙값",
            }
        )
    )
    season_summary.to_csv(
        table_dir / "S1-08_fwi_season_cell_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    season_tests = []
    for group in TYPE_ORDER:
        group_data = cell_season[cell_season["기후지형유형"] == group]
        for variable in variables:
            pivot = group_data.pivot(
                index="기상셀ID",
                columns="계절",
                values=variable,
            )[SEASON_ORDER].dropna()
            test = stats.friedmanchisquare(
                *[pivot[season] for season in SEASON_ORDER]
            )
            season_tests.append(
                {
                    "기후지형유형": group,
                    "FWI지수": variable,
                    "N_셀": len(pivot),
                    "Friedman_chi2": test.statistic,
                    "p_value": test.pvalue,
                    "Kendall_W": test.statistic
                    / (len(pivot) * (len(SEASON_ORDER) - 1)),
                }
            )
    season_test_df = pd.DataFrame(season_tests)
    season_test_df["q_FDR_global"] = _fdr(season_test_df["p_value"])
    season_test_df.to_csv(
        table_dir / "S1-08_fwi_season_friedman.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cell_month = (
        data.groupby(
            ["기상셀ID", "기후지형유형", "월"],
            as_index=False,
            observed=True,
        )[variables]
        .mean()
    )
    monthly_summary = (
        cell_month.groupby(["기후지형유형", "월"], observed=True)[variables]
        .mean()
        .reset_index()
    )
    monthly_summary.to_csv(
        table_dir / "S1-08_fwi_monthly_cell_mean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for variable in variables:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(
            data=cell_overall,
            x="기후지형유형",
            y=variable,
            order=TYPE_ORDER,
            hue="기후지형유형",
            palette=TYPE_COLORS,
            legend=False,
            inner="quart",
            ax=ax,
        )
        sns.stripplot(
            data=cell_overall,
            x="기후지형유형",
            y=variable,
            order=TYPE_ORDER,
            color="black",
            alpha=0.45,
            size=3.5,
            ax=ax,
        )
        ax.set_title(f"S1-08 셀 평균 {variable} 권역 분포")
        ax.set_xlabel("")
        _save(fig, plot_dir / f"S1-08_violin_cell_mean_{variable}.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.ecdfplot(
            data=cell_overall,
            x=variable,
            hue="기후지형유형",
            hue_order=TYPE_ORDER,
            palette=TYPE_COLORS,
            linewidth=2,
            ax=ax,
        )
        ax.set_title(f"S1-08 셀 평균 {variable} ECDF")
        _save(fig, plot_dir / f"S1-08_ecdf_cell_mean_{variable}.png")

        pivot = monthly_summary.pivot(
            index="기후지형유형",
            columns="월",
            values=variable,
        ).reindex(TYPE_ORDER)
        z_values = (pivot - pivot.stack().mean()) / pivot.stack().std(ddof=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            z_values,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            center=0,
            ax=ax,
        )
        ax.set_title(f"S1-08 월별 셀 평균 {variable} 표준화값")
        ax.set_ylabel("")
        _save(fig, plot_dir / f"S1-08_monthly_heatmap_{variable}.png")

    return {
        "data": data,
        "cell_overall": cell_overall,
        "regional_tests": regional_tests,
        "pairwise": pairwise,
        "cell_season": cell_season,
        "season_summary": season_summary,
        "season_tests": season_test_df,
        "monthly_summary": monthly_summary,
    }


def run_s109(
    canadian_data: pd.DataFrame,
    weather_grid: gpd.GeoDataFrame,
    table_dir: Path,
    plot_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Describe the paper-reference Indexed_FFMC stages without calibration claims."""
    table_dir, plot_dir = _prepare_dirs(table_dir, plot_dir)
    data = canadian_data.copy()
    stage_labels = {
        1: "1단계",
        2: "2단계",
        3: "3단계",
        4: "4단계",
    }
    data["선행연구단계"] = data["Indexed_FFMC"].map(stage_labels)
    crosstab = (
        pd.crosstab(
            [data["기후지형유형"], data["계절"]],
            data["선행연구단계"],
            normalize="index",
        )
        * 100
    )
    crosstab.to_csv(
        table_dir / "S1-09_indexed_ffmc_reference_stage_pct.csv",
        encoding="utf-8-sig",
    )

    ffmc10_cell_season = (
        data.groupby(
            ["기상셀ID", "기후지형유형", "계절"],
            as_index=False,
            observed=True,
        )["FFMC_10일평균"]
        .mean()
    )
    ffmc10_summary = (
        ffmc10_cell_season.groupby(
            ["기후지형유형", "계절"],
            observed=True,
        )["FFMC_10일평균"]
        .agg(N_셀="count", 평균="mean", 표준편차="std", 중앙값="median")
        .reset_index()
    )
    ffmc10_summary.to_csv(
        table_dir / "S1-09_ffmc10_season_cell_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cell_high_stage = (
        data.assign(상위단계=data["Indexed_FFMC"].ge(3))
        .groupby("기상셀ID", as_index=False)["상위단계"]
        .mean()
        .rename(columns={"상위단계": "Indexed_FFMC_3_4단계_비율"})
    )
    cell_high_stage["Indexed_FFMC_3_4단계_비율_pct"] = (
        100 * cell_high_stage["Indexed_FFMC_3_4단계_비율"]
    )
    cell_high_stage.to_csv(
        table_dir / "S1-09_cell_high_stage_ratio.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    crosstab.plot(
        kind="bar",
        stacked=True,
        color=sns.color_palette("YlOrRd", 4),
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_title("S1-09 권역·계절별 Indexed_FFMC 선행연구 단계 비율")
    ax.set_ylabel("비율 (%)")
    ax.set_xlabel("기후지형유형·계절")
    ax.legend(title="선행연구 단계", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(
        [f"{group}\n({season})" for group, season in crosstab.index],
        rotation=0,
    )
    _save(fig, plot_dir / "S1-09_indexed_ffmc_reference_stages.png")

    map_data = weather_grid[["기상셀ID", "geometry"]].merge(
        cell_high_stage,
        on="기상셀ID",
        how="left",
        validate="one_to_one",
    )
    fig, ax = plt.subplots(figsize=(9, 9))
    map_data.plot(
        column="Indexed_FFMC_3_4단계_비율_pct",
        cmap="Reds",
        linewidth=0.15,
        edgecolor="white",
        legend=True,
        legend_kwds={
            "label": "Indexed_FFMC 3~4단계 비율 (%)",
            "orientation": "horizontal",
        },
        ax=ax,
    )
    map_data.dissolve().boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title("S1-09 선행연구 Indexed_FFMC 3~4단계 공간 비율")
    ax.set_axis_off()
    _save(fig, plot_dir / "S1-09_high_reference_stage_map.png")

    return {
        "crosstab": crosstab,
        "ffmc10_summary": ffmc10_summary,
        "cell_high_stage": cell_high_stage,
    }


def run_s110(
    canadian_data: pd.DataFrame,
    table_dir: Path,
    plot_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Check construction consistency and expose seasonal/type stratification."""
    table_dir, plot_dir = _prepare_dirs(table_dir, plot_dir)
    data = canadian_data.copy()
    weather_variables = ["기온_C", "습도_pct", "풍속_km_h", "강수량_mm"]
    fwi_variables = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
    cross_corr = data[weather_variables + fwi_variables].corr(
        method="spearman"
    ).loc[weather_variables, fwi_variables]
    cross_corr.to_csv(
        table_dir / "S1-10_weather_fwi_spearman_corr.csv",
        encoding="utf-8-sig",
    )

    stratified_records = []
    for (group, season), subset in data.groupby(
        ["기후지형유형", "계절"],
        observed=True,
    ):
        for weather_variable in weather_variables:
            for fwi_variable in fwi_variables:
                rho, p_value = stats.spearmanr(
                    subset[weather_variable],
                    subset[fwi_variable],
                )
                stratified_records.append(
                    {
                        "기후지형유형": group,
                        "계절": season,
                        "기상변수": weather_variable,
                        "FWI지수": fwi_variable,
                        "N_셀일": len(subset),
                        "Spearman_rho": rho,
                        "p_value": p_value,
                    }
                )
    stratified = pd.DataFrame(stratified_records)
    stratified["q_FDR_global"] = _fdr(stratified["p_value"])
    stratified.to_csv(
        table_dir / "S1-10_weather_fwi_stratified_spearman.csv",
        index=False,
        encoding="utf-8-sig",
    )

    construction = pd.DataFrame(
        [
            {
                "지수": "FFMC",
                "직접입력": "기온, 습도, 풍속, 직전24시간 강수",
                "해석": "산식 입력과 출력의 방향성 점검",
            },
            {
                "지수": "DMC",
                "직접입력": "기온, 습도, 직전24시간 강수, 월",
                "해석": "산식 입력과 출력의 방향성 점검",
            },
            {
                "지수": "DC",
                "직접입력": "기온, 직전24시간 강수, 월",
                "해석": "산식 입력과 출력의 방향성 점검",
            },
            {
                "지수": "ISI",
                "직접입력": "FFMC, 풍속",
                "해석": "구성요소 관계 점검",
            },
            {
                "지수": "BUI",
                "직접입력": "DMC, DC",
                "해석": "구성요소 관계 점검",
            },
            {
                "지수": "FWI",
                "직접입력": "ISI, BUI",
                "해석": "구성요소 관계 점검",
            },
        ]
    )
    construction.to_csv(
        table_dir / "S1-10_index_construction_map.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        cross_corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("S1-10 기상 입력과 캐나다 지수 Spearman 상관")
    _save(fig, plot_dir / "S1-10_weather_fwi_heatmap.png")

    sample = data.sample(n=min(10_000, len(data)), random_state=42)
    scatter_specs = [
        ("습도_pct", "FFMC"),
        ("풍속_km_h", "ISI"),
        ("강수량_mm", "FWI"),
    ]
    for x_variable, y_variable in scatter_specs:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.scatterplot(
            data=sample,
            x=x_variable,
            y=y_variable,
            alpha=0.25,
            s=12,
            color="#4C78A8",
            ax=ax,
        )
        rho = cross_corr.loc[x_variable, y_variable]
        ax.set_title(
            f"S1-10 {x_variable} vs {y_variable} (전체 rho={rho:.2f})"
        )
        _save(fig, plot_dir / f"S1-10_scatter_{x_variable}_{y_variable}.png")

    return {
        "cross_corr": cross_corr,
        "stratified": stratified,
        "construction": construction,
    }


def run_s1a01(
    hourly_weather: pd.DataFrame,
    full_data: pd.DataFrame,
    quality_flag_ids: list[str],
    table_dir: Path,
) -> pd.DataFrame:
    """Create an explicit quality audit; do not silently drop flagged cells."""
    table_dir = Path(table_dir)
    hourly = hourly_weather.copy()
    hourly["연도"] = hourly["일시"].dt.year
    annual = (
        hourly.groupby(["기상셀ID", "연도"], as_index=False)
        .agg(
            연평균기온_C=("기온_C", "mean"),
            연평균습도_pct=("습도_pct", "mean"),
            연평균풍속_m_s=("풍속_m_s", "mean"),
            연강수량_mm=("강수량_mm", "sum"),
        )
    )
    annual_mean = annual.groupby("기상셀ID", as_index=False).mean(numeric_only=True)
    variables = [
        "연평균기온_C",
        "연평균습도_pct",
        "연평균풍속_m_s",
        "연강수량_mm",
    ]
    for variable in variables:
        annual_mean[f"{variable}_백분위"] = annual_mean[variable].rank(pct=True)
    audit = annual_mean[annual_mean["기상셀ID"].isin(quality_flag_ids)].copy()
    audit = audit.merge(
        full_data[
            [
                "기상셀ID",
                "기후지형유형",
                "대표지점명",
                "DEM_고도_m",
                "극단값_변수수",
            ]
        ],
        on="기상셀ID",
        how="left",
        validate="one_to_one",
    )
    audit["처리원칙"] = (
        "원천 보정 전 자동 제거 금지; 전체 포함 결과와 품질검토대상 제외 민감도 결과 병기"
    )
    audit.to_csv(
        table_dir / "S1-A01_flagged_cell_source_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return audit
