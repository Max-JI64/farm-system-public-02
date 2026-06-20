from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage12_interpret_feature_set import load_modeling_frame
from stage13_logistic_or_inference import build_design_matrix, drop_zero_variance_and_collinear


warnings.filterwarnings("ignore")


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "jsw" / "Analysis" / "logistic").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
LOGISTIC_DIR = ROOT / "jsw" / "Analysis" / "logistic"
OUTPUT_DIR = LOGISTIC_DIR / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"
SPLIT_DIR = OUTPUT_DIR / "splits"

FEATURE_SET_PATH = FEATURE_DIR / "stage12_interpret_feature_sets.json"
MAPPING_PATH = TABLE_DIR / "stage12_feature_mapping.csv"
OUTER_PATH = SPLIT_DIR / "outer_cv_manifest.csv"

TARGET = "Target"
ID_COL = "샘플ID"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    for directory in [TABLE_DIR, PLOT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_stage12() -> tuple[dict[str, list[str]], set[str], pd.DataFrame]:
    payload = json.loads(FEATURE_SET_PATH.read_text(encoding="utf-8"))
    feature_sets = payload["feature_sets"]
    categorical = set(payload["categorical_features"])
    mapping = pd.read_csv(MAPPING_PATH, encoding="utf-8-sig")
    return feature_sets, categorical, mapping


def stable_no_landcover_features(full_features: list[str], mapping: pd.DataFrame) -> list[str]:
    group = mapping.set_index("feature")["concept_group"].to_dict()
    landcover = {f for f in full_features if group.get(f) == "토지피복"}
    landcover_interactions = {"시가화_x_도로10m", "초지_x_dry0p1"}
    return [f for f in full_features if f not in (landcover | landcover_interactions)]


def diagnostic_feature_list(dev: pd.DataFrame, mapping: pd.DataFrame, feature_sets: dict[str, list[str]]) -> list[str]:
    full = list(feature_sets["INTERPRET_EDA_INTERACTIONS"])
    extra = [
        "직전24h_평균습도",
        "D-1_최소습도_pct",
        "시점_습도_pct",
        "직전24h_강수량합",
        "D-1_강수량합_mm",
        "dry_spell_h_0p1",
        "dry_spell_h_5p0",
        "wind_mean_6h",
        "직전24h_최대풍속",
        "D1_DC",
        "D1_BUI",
        "D1_FFMC_10일평균",
    ]
    categorical = set(mapping.loc[mapping["feature_type"].eq("categorical"), "feature"])
    excluded_roles = {"excluded", "excluded_artifact"}
    role = mapping.set_index("feature")["role"].to_dict()
    candidates = []
    for feature in full + extra:
        if feature in candidates:
            continue
        if feature not in dev.columns:
            continue
        if feature in categorical:
            continue
        if role.get(feature) in excluded_roles:
            continue
        numeric = pd.to_numeric(dev[feature], errors="coerce")
        if numeric.notna().sum() > 0 and numeric.nunique(dropna=True) > 1:
            candidates.append(feature)
    return candidates


def make_correlations(dev: pd.DataFrame, features: list[str], mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = dev[features].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    group = mapping.set_index("feature")["concept_group"].to_dict()
    rows = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1 :]:
            rows.append(
                {
                    "feature_1": f1,
                    "feature_2": f2,
                    "group_1": group.get(f1, ""),
                    "group_2": group.get(f2, ""),
                    "pearson": float(pearson.loc[f1, f2]),
                    "spearman": float(spearman.loc[f1, f2]),
                    "abs_pearson": abs(float(pearson.loc[f1, f2])),
                    "abs_spearman": abs(float(spearman.loc[f1, f2])),
                }
            )
    corr_long = pd.DataFrame(rows).sort_values(["abs_spearman", "abs_pearson"], ascending=False)
    return corr_long, spearman


def compute_vif(design: pd.DataFrame, term_info: pd.DataFrame) -> pd.DataFrame:
    X = design.drop(columns=["const"], errors="ignore").copy()
    rows = []
    values = X.to_numpy(dtype=float)
    columns = X.columns.tolist()
    for idx, col in enumerate(columns):
        y = values[:, idx]
        others = np.delete(values, idx, axis=1)
        if others.shape[1] == 0:
            vif = np.nan
            r2 = np.nan
        else:
            others = np.column_stack([np.ones(len(others)), others])
            coef, *_ = np.linalg.lstsq(others, y, rcond=None)
            fitted = others @ coef
            sst = float(np.sum((y - y.mean()) ** 2))
            sse = float(np.sum((y - fitted) ** 2))
            r2 = 1 - sse / sst if sst > 0 else np.nan
            if r2 is np.nan or np.isnan(r2):
                vif = np.nan
            elif r2 >= 0.999999:
                vif = np.inf
            else:
                vif = 1 / (1 - r2)
        rows.append({"term_id": col, "r2": r2, "vif": vif})
    vif = pd.DataFrame(rows)
    return vif.merge(term_info, on="term_id", how="left").sort_values("vif", ascending=False)


def make_vif(dev: pd.DataFrame, features: list[str], categorical: set[str], mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    design, term_info = build_design_matrix(dev, features, categorical, mapping)
    design, term_info, dropped = drop_zero_variance_and_collinear(design, term_info)
    vif = compute_vif(design, term_info)
    dropped.insert(0, "diagnostic_model", "STABLE_NO_LANDCOVER")
    return vif, dropped


def landcover_diagnostics(dev: pd.DataFrame, outer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = dev.merge(outer[[ID_COL, "outer_fold"]], on=ID_COL, how="left", validate="one_to_one")
    rows = []
    for feature in ["토지피복_L1_NAME", "토지피복_산림유형", "토지피복_L2_NAME"]:
        if feature not in data.columns:
            continue
        temp = data[[feature, TARGET, "outer_fold", "샘플유형"]].copy()
        temp[feature] = temp[feature].fillna("결측").astype(str)
        for level, group in temp.groupby(feature, dropna=False):
            fold_counts = group.groupby("outer_fold")[TARGET].agg(["count", "sum"]).reset_index()
            zero_pos_folds = int((fold_counts["sum"] == 0).sum())
            zero_neg_folds = int(((fold_counts["count"] - fold_counts["sum"]) == 0).sum())
            n = int(len(group))
            pos = int(group[TARGET].sum())
            neg = n - pos
            rows.append(
                {
                    "feature": feature,
                    "level": level,
                    "n": n,
                    "target1_n": pos,
                    "target0_n": neg,
                    "target1_rate": float(pos / n) if n else np.nan,
                    "folds_present": int(group["outer_fold"].nunique()),
                    "zero_target1_folds": zero_pos_folds,
                    "zero_target0_folds": zero_neg_folds,
                    "rare_n_lt_100": n < 100,
                    "rare_target1_lt_10": pos < 10,
                    "rare_target0_lt_10": neg < 10,
                    "sample_types": ", ".join(sorted(group["샘플유형"].dropna().astype(str).unique())),
                }
            )
    diag = pd.DataFrame(rows).sort_values(["feature", "n"], ascending=[True, False])

    if {"토지피복_L1_NAME", "토지피복_산림유형"}.issubset(data.columns):
        crosstab = pd.crosstab(
            data["토지피복_L1_NAME"].fillna("결측").astype(str),
            data["토지피복_산림유형"].fillna("결측").astype(str),
        )
        crosstab = crosstab.reset_index().rename(columns={"토지피복_L1_NAME": "토지피복_L1_NAME"})
    else:
        crosstab = pd.DataFrame()
    return diag, crosstab


def odds_from_rate(rate: float) -> float:
    rate = min(max(rate, 1e-8), 1 - 1e-8)
    return rate / (1 - rate)


def binned_effects(dev: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "직전24h_최소습도": {
            "source": "직전24h_최소습도",
            "bins": [-np.inf, 20, 30, 40, 60, np.inf],
            "labels": ["<=20", "20-30", "30-40", "40-60", ">60"],
        },
        "도로_최단거리_m": {
            "source": "도로_최단거리_m",
            "bins": [-np.inf, 10, 30, 100, np.inf],
            "labels": ["<=10m", "10-30m", "30-100m", ">100m"],
        },
        "wind_max_6h": {
            "source": "wind_max_6h",
            "bins": [-np.inf, 2, 3, 5, np.inf],
            "labels": ["<=2", "2-3", "3-5", ">5"],
        },
        "D1_FWI": {
            "source": "D1_FWI",
            "bins": [-np.inf, 5, 10, 20, np.inf],
            "labels": ["<=5", "5-10", "10-20", ">20"],
        },
        "D1_ISI": {
            "source": "D1_ISI",
            "bins": [-np.inf, 3, 6, 10, np.inf],
            "labels": ["<=3", "3-6", "6-10", ">10"],
        },
        "D1_FFMC": {
            "source": "D1_FFMC",
            "bins": [-np.inf, 70, 85, 90, np.inf],
            "labels": ["<=70", "70-85", "85-90", ">90"],
        },
    }
    base_rate = float(dev[TARGET].mean())
    base_odds = odds_from_rate(base_rate)
    rows = []
    for variable, spec in specs.items():
        source = spec["source"]
        if source not in dev.columns:
            continue
        values = pd.to_numeric(dev[source], errors="coerce")
        bins = pd.cut(values, bins=spec["bins"], labels=spec["labels"], include_lowest=True)
        temp = pd.DataFrame({"bin": bins.astype(str), TARGET: dev[TARGET].astype(int), "value": values})
        for order, label in enumerate(spec["labels"], start=1):
            group = temp.loc[temp["bin"].eq(label)]
            n = int(len(group))
            pos = int(group[TARGET].sum()) if n else 0
            rate = float(pos / n) if n else np.nan
            odds = odds_from_rate(rate) if n else np.nan
            rows.append(
                {
                    "variable": variable,
                    "source": source,
                    "bin_order": order,
                    "bin": label,
                    "n": n,
                    "target1_n": pos,
                    "target0_n": n - pos,
                    "target1_rate": rate,
                    "lift_vs_base_rate": rate / base_rate if n and base_rate else np.nan,
                    "odds_ratio_vs_base_rate": odds / base_odds if n and base_odds else np.nan,
                    "value_min": float(group["value"].min()) if n else np.nan,
                    "value_median": float(group["value"].median()) if n else np.nan,
                    "value_max": float(group["value"].max()) if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_corr_heatmap(spearman: pd.DataFrame) -> Path:
    features = spearman.columns.tolist()
    fig_size = max(8, min(18, len(features) * 0.35))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    image = ax.imshow(spearman.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title("Step15 Spearman correlation heatmap")
    fig.colorbar(image, ax=ax, shrink=0.75)
    path = PLOT_DIR / "stage15_corr_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_vif(vif: pd.DataFrame) -> Path:
    plot_df = vif.loc[vif["term_id"].ne("const")].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.sort_values("vif", ascending=False).head(25).sort_values("vif")
    labels = plot_df["term_label"].fillna(plot_df["term_id"]).astype(str)
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.35)), constrained_layout=True)
    ax.barh(labels, plot_df["vif"], color="#2563eb", alpha=0.85)
    ax.axvline(5, color="#f97316", linestyle="--", linewidth=1, label="VIF=5")
    ax.axvline(10, color="#ef4444", linestyle="--", linewidth=1, label="VIF=10")
    ax.set_xlabel("VIF")
    ax.set_title("Step15 VIF 상위 항")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    path = PLOT_DIR / "stage15_vif_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_binned(binned: pd.DataFrame, variable: str, filename: str, title: str) -> Path:
    data = binned.loc[binned["variable"].eq(variable)].sort_values("bin_order")
    fig, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax1.bar(data["bin"], data["target1_rate"], color="#2563eb", alpha=0.85)
    ax1.set_ylabel("Target 1 rate")
    ax1.set_title(title)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(data["bin"], data["n"], color="#f97316", marker="o")
    ax2.set_ylabel("n")
    for idx, row in data.iterrows():
        ax1.text(row["bin_order"] - 1, row["target1_rate"], f"{row['target1_rate']:.3f}", ha="center", va="bottom", fontsize=8)
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(
    corr_long: pd.DataFrame,
    high_corr: pd.DataFrame,
    vif: pd.DataFrame,
    vif_dropped: pd.DataFrame,
    landcover_diag: pd.DataFrame,
    landcover_cross: pd.DataFrame,
    binned: pd.DataFrame,
    spearman: pd.DataFrame,
    dev: pd.DataFrame,
    stable_features: list[str],
) -> None:
    corr_path = TABLE_DIR / "stage15_correlations.csv"
    high_corr_path = TABLE_DIR / "stage15_high_correlations.csv"
    vif_path = TABLE_DIR / "stage15_vif.csv"
    vif_dropped_path = TABLE_DIR / "stage15_vif_dropped_terms.csv"
    landcover_path = TABLE_DIR / "stage15_landcover_category_diagnostics.csv"
    landcover_cross_path = TABLE_DIR / "stage15_landcover_dependency_crosstab.csv"
    binned_path = TABLE_DIR / "stage15_binned_effects.csv"
    summary_path = OUTPUT_DIR / "stage15_collinearity_nonlinearity_summary.md"

    corr_long.to_csv(corr_path, index=False, encoding="utf-8-sig")
    high_corr.to_csv(high_corr_path, index=False, encoding="utf-8-sig")
    vif.to_csv(vif_path, index=False, encoding="utf-8-sig")
    vif_dropped.to_csv(vif_dropped_path, index=False, encoding="utf-8-sig")
    landcover_diag.to_csv(landcover_path, index=False, encoding="utf-8-sig")
    landcover_cross.to_csv(landcover_cross_path, index=False, encoding="utf-8-sig")
    binned.to_csv(binned_path, index=False, encoding="utf-8-sig")

    corr_plot = plot_corr_heatmap(spearman)
    vif_plot = plot_vif(vif)
    binned_plots = [
        plot_binned(binned, "직전24h_최소습도", "stage15_binned_humidity.png", "직전24h 최소습도 bin별 Target 1 비율"),
        plot_binned(binned, "도로_최단거리_m", "stage15_binned_road_distance.png", "도로 최단거리 bin별 Target 1 비율"),
        plot_binned(binned, "wind_max_6h", "stage15_binned_wind.png", "6시간 최대풍속 bin별 Target 1 비율"),
        plot_binned(binned, "D1_FWI", "stage15_binned_fwi.png", "D-1 FWI bin별 Target 1 비율"),
        plot_binned(binned, "D1_ISI", "stage15_binned_isi.png", "D-1 ISI bin별 Target 1 비율"),
        plot_binned(binned, "D1_FFMC", "stage15_binned_ffmc.png", "D-1 FFMC bin별 Target 1 비율"),
    ]

    def md_table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_csv(index=False)

    high_corr_show = high_corr.head(25).copy()
    high_corr_show[high_corr_show.select_dtypes(include="number").columns] = high_corr_show.select_dtypes(include="number").round(4)

    vif_show = vif.head(25).copy()
    vif_show[vif_show.select_dtypes(include="number").columns] = vif_show.select_dtypes(include="number").round(3)

    canada_pairs = corr_long.loc[
        corr_long["feature_1"].str.startswith("D1_") & corr_long["feature_2"].str.startswith("D1_")
    ].head(20).copy()
    canada_pairs[canada_pairs.select_dtypes(include="number").columns] = canada_pairs.select_dtypes(include="number").round(4)

    rare_landcover = landcover_diag.loc[
        landcover_diag["rare_n_lt_100"] | landcover_diag["rare_target1_lt_10"] | landcover_diag["rare_target0_lt_10"]
    ].copy()
    rare_landcover = rare_landcover.sort_values(["feature", "n"]).head(40)
    rare_landcover[rare_landcover.select_dtypes(include="number").columns] = rare_landcover.select_dtypes(include="number").round(4)

    binned_show = binned.copy()
    binned_show[binned_show.select_dtypes(include="number").columns] = binned_show.select_dtypes(include="number").round(4)

    vif_gt_10 = int((pd.to_numeric(vif["vif"], errors="coerce") >= 10).sum())
    vif_gt_5 = int((pd.to_numeric(vif["vif"], errors="coerce") >= 5).sum())
    high_corr_n = int(len(high_corr))
    rare_n = int(len(rare_landcover))

    lines = [
        "# Stage 15 공선성·희소범주·비선형성 진단",
        "",
        "## 1. 목적",
        "",
        "- Step13에서 나타난 캐나다지수 부호 역전 가능성을 확인한다.",
        "- Step14에서 확인된 토지피복 범주 투입 시 OOF 성능 악화를 진단한다.",
        "- 습도·강수·풍속·공간 변수의 중복성과 비선형성을 점검한다.",
        "- lockbox는 사용하지 않고 strict development 표본만 사용했다.",
        "",
        "## 2. 입력과 기준",
        "",
        f"- development 표본: {len(dev):,}행",
        f"- Target 1: {int(dev[TARGET].sum()):,}건",
        f"- Target 0: {int((1 - dev[TARGET]).sum()):,}건",
        f"- VIF 기준 변수셋: 토지피복 제거 안정 모델, {len(stable_features)}개 원변수",
        "",
        "## 3. 고상관 변수쌍",
        "",
        f"- |Spearman| >= 0.80 또는 |Pearson| >= 0.80인 변수쌍: {high_corr_n}개",
        "",
        md_table(high_corr_show),
        "",
        "### 캐나다지수 간 상관",
        "",
        md_table(canada_pairs),
        "",
        "## 4. VIF 진단",
        "",
        f"- VIF >= 5 항 수: {vif_gt_5}",
        f"- VIF >= 10 항 수: {vif_gt_10}",
        "",
        md_table(vif_show[["term_label", "feature", "concept_group", "role", "vif", "r2"]]),
        "",
        "## 5. 토지피복 범주 희소성/분리 위험",
        "",
        f"- 희소 또는 quasi-separation 후보 범주 수: {rare_n}",
        "",
        md_table(
            rare_landcover[
                [
                    "feature",
                    "level",
                    "n",
                    "target1_n",
                    "target0_n",
                    "target1_rate",
                    "folds_present",
                    "zero_target1_folds",
                    "zero_target0_folds",
                ]
            ]
        ),
        "",
        "## 6. 주요 변수 bin 효과",
        "",
        md_table(
            binned_show[
                [
                    "variable",
                    "bin",
                    "n",
                    "target1_n",
                    "target1_rate",
                    "lift_vs_base_rate",
                    "odds_ratio_vs_base_rate",
                ]
            ]
        ),
        "",
        "## 7. 판단",
        "",
        "- 캐나다지수는 서로 강하게 중복될 가능성이 높다. Step13의 `D1_FWI` 양의 OR과 `D1_ISI` 음의 OR은 독립 효과라기보다 공선성/억제효과로 볼 가능성이 크다.",
        "- 토지피복 세부 범주는 희소범주와 선형종속 위험이 있어 unregularized GLM에 그대로 넣기 어렵다.",
        "- 최종 해석 모델에서는 토지피복을 제거하거나, 미상/수역/희소범주를 통합한 단순 플래그로 축약하는 방향이 안전하다.",
        "- 비선형성은 bin 효과를 보고 판단한다. 선형 OR은 보고서 핵심으로 쓰되, bin 그림은 임계구간 후보 설명에 사용한다.",
        "",
        "## 8. 산출물",
        "",
        f"- `{corr_path.relative_to(ROOT)}`",
        f"- `{high_corr_path.relative_to(ROOT)}`",
        f"- `{vif_path.relative_to(ROOT)}`",
        f"- `{vif_dropped_path.relative_to(ROOT)}`",
        f"- `{landcover_path.relative_to(ROOT)}`",
        f"- `{landcover_cross_path.relative_to(ROOT)}`",
        f"- `{binned_path.relative_to(ROOT)}`",
        f"- `{corr_plot.relative_to(ROOT)}`",
        f"- `{vif_plot.relative_to(ROOT)}`",
    ]
    lines.extend([f"- `{path.relative_to(ROOT)}`" for path in binned_plots])
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    feature_sets, categorical, mapping = load_stage12()
    _, dev = load_modeling_frame()
    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    outer = outer.loc[outer[ID_COL].isin(set(dev[ID_COL]))].copy()

    full_features = feature_sets["INTERPRET_EDA_INTERACTIONS"]
    stable_features = stable_no_landcover_features(full_features, mapping)
    diagnostic_features = diagnostic_feature_list(dev, mapping, feature_sets)

    corr_long, spearman = make_correlations(dev, diagnostic_features, mapping)
    high_corr = corr_long.loc[
        corr_long["abs_spearman"].ge(0.80) | corr_long["abs_pearson"].ge(0.80)
    ].copy()

    vif, vif_dropped = make_vif(dev, stable_features, categorical, mapping)
    landcover_diag, landcover_cross = landcover_diagnostics(dev, outer)
    binned = binned_effects(dev)

    write_outputs(
        corr_long,
        high_corr,
        vif,
        vif_dropped,
        landcover_diag,
        landcover_cross,
        binned,
        spearman,
        dev,
        stable_features,
    )
    print("Stage15 완료")
    print(f"요약: {OUTPUT_DIR / 'stage15_collinearity_nonlinearity_summary.md'}")
    print(f"VIF: {TABLE_DIR / 'stage15_vif.csv'}")


if __name__ == "__main__":
    main()
