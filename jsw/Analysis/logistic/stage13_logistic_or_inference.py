from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from stage12_interpret_feature_set import load_modeling_frame


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

FEATURE_SET_PATH = FEATURE_DIR / "stage12_interpret_feature_sets.json"
MAPPING_PATH = TABLE_DIR / "stage12_feature_mapping.csv"

TARGET = "Target"
PRIMARY_COV_TYPE = "cluster_date"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class TermInfo:
    term_id: str
    feature: str
    term_label: str
    concept_group: str
    role: str
    feature_type: str
    unit_for_or: str
    transform_sign: float
    transform_scale: float
    original_sd: float | None
    reference: str | None = None
    level: str | None = None


REFERENCE_LEVELS = {
    "기후지형유형": "영서 내륙형",
    "토지피복_L1_NAME": "산림지역",
    "토지피복_산림유형": "비산림",
}


NUMERIC_TRANSFORMS = {
    "직전24h_최소습도": (-1.0, 5.0, "5%p 감소"),
    "직전48h_강수량합": (-1.0, 5.0, "5mm 감소"),
    "wind_max_6h": (1.0, 1.0, "1m/s 증가"),
    "기압변동_3h": (-1.0, 1.0, "1hPa 하강"),
    "log1p_도로거리_m": (1.0, 1.0, "log1p 거리 1 증가"),
    "log1p_시가화거리_m": (1.0, 1.0, "log1p 거리 1 증가"),
    "log1p_산림거리_m": (1.0, 1.0, "log1p 거리 1 증가"),
    "고도(m)": (1.0, 100.0, "100m 증가"),
    "경사도(도)": (1.0, 5.0, "5도 증가"),
    "TPI(지형위치지수)": (1.0, 1.0, "1 단위 증가"),
    "D1_FFMC": (1.0, 5.0, "5점 증가"),
    "D1_DMC": (1.0, 10.0, "10점 증가"),
    "D1_ISI": (1.0, 1.0, "1점 증가"),
    "D1_FWI": (1.0, 5.0, "5점 증가"),
    "영동_x_wind_max_6h": (1.0, 1.0, "영동 해안형에서 1m/s 증가"),
    "월_sin": (1.0, 1.0, "1 단위 증가"),
    "월_cos": (1.0, 1.0, "1 단위 증가"),
    "시간_sin": (1.0, 1.0, "1 단위 증가"),
    "시간_cos": (1.0, 1.0, "1 단위 증가"),
}


def ensure_dirs() -> None:
    for directory in [TABLE_DIR, PLOT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_stage12_config() -> tuple[dict[str, list[str]], set[str], pd.DataFrame]:
    payload = json.loads(FEATURE_SET_PATH.read_text(encoding="utf-8"))
    feature_sets = payload["feature_sets"]
    categorical = set(payload["categorical_features"])
    mapping = pd.read_csv(MAPPING_PATH, encoding="utf-8-sig")
    return feature_sets, categorical, mapping


def fdr_bh(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return q
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q.loc[order] = adjusted
    return q


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def metadata_lookup(mapping: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return mapping.set_index("feature").to_dict(orient="index")


def add_numeric_term(
    parts: list[pd.Series],
    terms: list[TermInfo],
    df: pd.DataFrame,
    feature: str,
    meta: dict[str, Any],
    term_number: int,
) -> int:
    raw = safe_numeric(df[feature])
    if raw.isna().any():
        raw = raw.fillna(raw.median())
    sign, scale, default_unit = NUMERIC_TRANSFORMS.get(feature, (1.0, 1.0, meta.get("unit_for_or", "1 단위")))
    value = sign * raw / scale
    term_id = f"x{term_number:03d}"
    parts.append(pd.Series(value.to_numpy(dtype=float), index=df.index, name=term_id))
    original_sd = float(raw.std()) if raw.notna().any() else np.nan
    unit = str(meta.get("unit_for_or") or default_unit)
    if feature in NUMERIC_TRANSFORMS:
        unit = default_unit
    terms.append(
        TermInfo(
            term_id=term_id,
            feature=feature,
            term_label=f"{feature} ({unit})",
            concept_group=str(meta.get("concept_group", "")),
            role=str(meta.get("role", "")),
            feature_type=str(meta.get("feature_type", "numeric")),
            unit_for_or=unit,
            transform_sign=sign,
            transform_scale=scale,
            original_sd=original_sd,
        )
    )
    return term_number + 1


def add_categorical_terms(
    parts: list[pd.Series],
    terms: list[TermInfo],
    df: pd.DataFrame,
    feature: str,
    meta: dict[str, Any],
    term_number: int,
) -> int:
    values = df[feature].fillna("미상").astype(str)
    observed = sorted(values.unique().tolist())
    reference = REFERENCE_LEVELS.get(feature)
    if reference not in observed:
        reference = values.value_counts().index[0]
    levels = [level for level in observed if level != reference]
    for level in levels:
        term_id = f"x{term_number:03d}"
        value = values.eq(level).astype(float)
        parts.append(pd.Series(value.to_numpy(dtype=float), index=df.index, name=term_id))
        terms.append(
            TermInfo(
                term_id=term_id,
                feature=feature,
                term_label=f"{feature}={level} vs {reference}",
                concept_group=str(meta.get("concept_group", "")),
                role=str(meta.get("role", "")),
                feature_type="categorical",
                unit_for_or=f"{level} vs {reference}",
                transform_sign=1.0,
                transform_scale=1.0,
                original_sd=None,
                reference=reference,
                level=level,
            )
        )
        term_number += 1
    return term_number


def build_design_matrix(
    df: pd.DataFrame,
    features: list[str],
    categorical: set[str],
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_by_feature = metadata_lookup(mapping)
    parts: list[pd.Series] = []
    terms: list[TermInfo] = []
    term_number = 1

    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"데이터에 없는 변수: {feature}")
        meta = meta_by_feature.get(feature, {})
        if feature in categorical:
            term_number = add_categorical_terms(parts, terms, df, feature, meta, term_number)
        else:
            term_number = add_numeric_term(parts, terms, df, feature, meta, term_number)

    X = pd.concat(parts, axis=1)
    X.insert(0, "const", 1.0)

    term_rows = [
        {
            "term_id": "const",
            "feature": "Intercept",
            "term_label": "Intercept",
            "concept_group": "상수항",
            "role": "intercept",
            "feature_type": "intercept",
            "unit_for_or": "",
            "transform_sign": 1.0,
            "transform_scale": 1.0,
            "original_sd": np.nan,
            "reference": "",
            "level": "",
        }
    ]
    term_rows.extend(term.__dict__ for term in terms)
    term_info = pd.DataFrame(term_rows)

    return X.astype(float), term_info


def drop_zero_variance_and_collinear(
    X: pd.DataFrame,
    term_info: pd.DataFrame,
    tol: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dropped_rows = []
    original_term_info = term_info.copy()

    std = X.drop(columns=["const"], errors="ignore").std(axis=0)
    zero_cols = std.index[std <= tol].tolist()
    if zero_cols:
        for col in zero_cols:
            dropped_rows.append({"term_id": col, "drop_reason": "zero_variance"})
        X = X.drop(columns=zero_cols)

    if X.shape[1] > 1:
        matrix = X.to_numpy(dtype=float)
        q, r, piv = qr(matrix, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r))
        threshold = tol * max(matrix.shape) * (diag.max() if len(diag) else 1.0)
        rank = int((diag > threshold).sum())
        keep_idx = sorted(piv[:rank])
        keep_cols = X.columns[keep_idx].tolist()
        dependent_cols = [col for col in X.columns if col not in keep_cols]
        if dependent_cols:
            for col in dependent_cols:
                dropped_rows.append({"term_id": col, "drop_reason": "linear_dependency"})
            X = X[keep_cols]

    term_info = term_info.loc[term_info["term_id"].isin(X.columns)].copy()
    dropped = pd.DataFrame(dropped_rows)
    if len(dropped):
        dropped = dropped.merge(
            original_term_info[["term_id", "feature", "term_label", "concept_group", "role"]],
            on="term_id",
            how="left",
        )
    else:
        dropped = pd.DataFrame(columns=["term_id", "drop_reason", "feature", "term_label", "concept_group", "role"])
    return X, term_info, dropped


def fit_glm(y: pd.Series, X: pd.DataFrame, groups: dict[str, pd.Series]) -> dict[str, Any]:
    model = sm.GLM(y.astype(float), X, family=sm.families.Binomial())
    fitted = {}
    fitted["nonrobust"] = model.fit(maxiter=300, disp=0)
    fitted["HC1"] = model.fit(maxiter=300, disp=0, cov_type="HC1")
    fitted["cluster_date"] = model.fit(
        maxiter=300,
        disp=0,
        cov_type="cluster",
        cov_kwds={"groups": groups["date"], "use_correction": True},
    )
    fitted["cluster_cell"] = model.fit(
        maxiter=300,
        disp=0,
        cov_type="cluster",
        cov_kwds={"groups": groups["cell"], "use_correction": True},
    )
    return fitted


def result_to_table(
    feature_set: str,
    cov_type: str,
    result: Any,
    term_info: pd.DataFrame,
) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    table = pd.DataFrame(
        {
            "feature_set": feature_set,
            "cov_type": cov_type,
            "term_id": params.index,
            "coef": params.to_numpy(dtype=float),
            "std_error": result.bse.reindex(params.index).to_numpy(dtype=float),
            "z_value": result.tvalues.reindex(params.index).to_numpy(dtype=float),
            "p_value": result.pvalues.reindex(params.index).to_numpy(dtype=float),
            "ci_low_logit": conf.loc[params.index, 0].to_numpy(dtype=float),
            "ci_high_logit": conf.loc[params.index, 1].to_numpy(dtype=float),
        }
    )
    table = table.merge(term_info, on="term_id", how="left")
    table["odds_ratio"] = np.exp(table["coef"])
    table["or_ci_low"] = np.exp(table["ci_low_logit"])
    table["or_ci_high"] = np.exp(table["ci_high_logit"])
    table["q_value"] = fdr_bh(table["p_value"].where(table["term_id"].ne("const")))

    factor = pd.to_numeric(table["original_sd"], errors="coerce") / pd.to_numeric(
        table["transform_scale"], errors="coerce"
    )
    table["std_or_expected_direction"] = np.exp(table["coef"] * factor)
    table["std_or_ci_low_expected_direction"] = np.exp(
        (table["coef"] - 1.96 * table["std_error"]) * factor
    )
    table["std_or_ci_high_expected_direction"] = np.exp(
        (table["coef"] + 1.96 * table["std_error"]) * factor
    )
    table.loc[table["feature_type"].isin(["categorical", "binary", "intercept"]), [
        "std_or_expected_direction",
        "std_or_ci_low_expected_direction",
        "std_or_ci_high_expected_direction",
    ]] = np.nan
    table["report_caution"] = ""
    table.loc[
        table["term_label"].astype(str).str.contains("=미상", regex=False),
        "report_caution",
    ] = "미상/매칭품질 범주이므로 실제 토지피복 효과로 과해석 금지"
    table.loc[
        table["feature"].astype(str).isin(["D1_FFMC", "D1_DMC", "D1_ISI", "D1_FWI"]),
        "report_caution",
    ] = table.loc[
        table["feature"].astype(str).isin(["D1_FFMC", "D1_DMC", "D1_ISI", "D1_FWI"]),
        "report_caution",
    ].mask(
        table.loc[
            table["feature"].astype(str).isin(["D1_FFMC", "D1_DMC", "D1_ISI", "D1_FWI"]),
            "report_caution",
        ].eq(""),
        "캐나다지수 간 공선성/억제효과 가능성, Step15에서 확인 필요",
    )
    table["report_candidate"] = (
        table["term_id"].ne("const")
        & table["q_value"].le(0.05)
        & table["report_caution"].eq("")
        & ~table["role"].isin(["control", "intercept"])
    )
    return table


def fit_metrics(feature_set: str, result: Any, y: pd.Series, X: pd.DataFrame, dropped: pd.DataFrame) -> dict[str, Any]:
    pred = np.asarray(result.predict(X), dtype=float)
    null_llf = getattr(result, "llnull", np.nan)
    llf = float(result.llf)
    return {
        "feature_set": feature_set,
        "n": int(len(y)),
        "positive_n": int(y.sum()),
        "positive_rate": float(y.mean()),
        "n_terms_including_intercept": int(X.shape[1]),
        "n_dropped_terms": int(len(dropped)),
        "converged": bool(result.converged),
        "log_likelihood": llf,
        "null_log_likelihood": float(null_llf) if not pd.isna(null_llf) else np.nan,
        "mcfadden_r2": float(1 - llf / null_llf) if null_llf and not pd.isna(null_llf) else np.nan,
        "aic": float(result.aic),
        "bic_llf": float(getattr(result, "bic_llf", np.nan)),
        "auprc_in_sample": float(average_precision_score(y, pred)),
        "auroc_in_sample": float(roc_auc_score(y, pred)),
        "brier_in_sample": float(brier_score_loss(y, pred)),
        "log_loss_in_sample": float(log_loss(y, pred, labels=[0, 1])),
    }


def prepare_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    date_group = pd.to_datetime(df["기준시각"]).dt.strftime("%Y-%m-%d")
    cell_group = df["기상셀ID"].fillna("unknown").astype(str)
    return {"date": date_group, "cell": cell_group}


def fit_all_models(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: set[str],
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].astype(int)
    groups = prepare_groups(df)
    all_or = []
    fit_rows = []
    dropped_rows = []

    for feature_set, features in feature_sets.items():
        print(f"Step13 적합: {feature_set} ({len(features)} features)")
        X, term_info = build_design_matrix(df, features, categorical, mapping)
        X, term_info, dropped = drop_zero_variance_and_collinear(X, term_info)
        if len(dropped):
            dropped.insert(0, "feature_set", feature_set)
            dropped_rows.append(dropped)

        fitted = fit_glm(y, X, groups)
        for cov_type, result in fitted.items():
            all_or.append(result_to_table(feature_set, cov_type, result, term_info))
        fit_rows.append(fit_metrics(feature_set, fitted["nonrobust"], y, X, dropped))

    odds = pd.concat(all_or, ignore_index=True)
    fit_summary = pd.DataFrame(fit_rows)
    dropped_summary = (
        pd.concat(dropped_rows, ignore_index=True)
        if dropped_rows
        else pd.DataFrame(columns=["feature_set", "term_id", "drop_reason", "feature", "term_label"])
    )
    return odds, fit_summary, dropped_summary


def make_primary_tables(odds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = odds.loc[odds["cov_type"].eq(PRIMARY_COV_TYPE) & odds["term_id"].ne("const")].copy()
    primary["significant_q05"] = primary["q_value"].le(0.05)
    primary["significant_p05"] = primary["p_value"].le(0.05)
    display_cols = [
        "feature_set",
        "concept_group",
        "role",
        "feature",
        "term_label",
        "unit_for_or",
        "coef",
        "std_error",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
        "p_value",
        "q_value",
        "std_or_expected_direction",
        "std_or_ci_low_expected_direction",
        "std_or_ci_high_expected_direction",
        "significant_q05",
        "report_caution",
        "report_candidate",
    ]
    primary_display = primary[[col for col in display_cols if col in primary.columns]].copy()
    significant = primary_display.loc[primary_display["significant_q05"].eq(True)].copy()
    significant = significant.sort_values(["feature_set", "q_value", "p_value"])
    return primary_display, significant


def make_forest_plot(primary: pd.DataFrame) -> Path:
    plot_df = primary.loc[
        primary["feature_set"].eq("INTERPRET_EDA_INTERACTIONS")
        & primary["role"].isin(
            [
                "core",
                "core_control",
                "canada_core",
                "interaction_core",
                "core_interaction",
            ]
        )
    ].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["odds_ratio", "or_ci_low", "or_ci_high"])
    plot_df = plot_df.loc[(plot_df["or_ci_low"] > 0) & (plot_df["or_ci_high"] < 100)]
    plot_df["abs_log_or"] = np.abs(np.log(plot_df["odds_ratio"]))
    plot_df = plot_df.sort_values("abs_log_or", ascending=False).head(24)
    plot_df = plot_df.sort_values("odds_ratio")

    labels = plot_df["term_label"].str.replace(" (", "\n(", regex=False)
    y_pos = np.arange(len(plot_df))
    fig_height = max(6, len(plot_df) * 0.35)
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    ax.errorbar(
        plot_df["odds_ratio"],
        y_pos,
        xerr=[
            plot_df["odds_ratio"] - plot_df["or_ci_low"],
            plot_df["or_ci_high"] - plot_df["odds_ratio"],
        ],
        fmt="o",
        color="#2563eb",
        ecolor="#94a3b8",
        capsize=3,
    )
    ax.axvline(1, color="#ef4444", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Odds ratio, date cluster-robust 95% CI")
    ax.set_title("Stage13 INTERPRET_EDA_INTERACTIONS 주요 OR")
    ax.grid(axis="x", alpha=0.25)
    out_path = PLOT_DIR / "stage13_or_forestplot.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_outputs(
    odds: pd.DataFrame,
    fit_summary: pd.DataFrame,
    dropped: pd.DataFrame,
    primary: pd.DataFrame,
    significant: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    odds_path = TABLE_DIR / "stage13_odds_ratios.csv"
    odds_md_path = TABLE_DIR / "stage13_odds_ratios.md"
    primary_path = TABLE_DIR / "stage13_primary_date_cluster_odds_ratios.csv"
    significant_path = TABLE_DIR / "stage13_significant_terms_q05.csv"
    report_candidate_path = TABLE_DIR / "stage13_report_candidate_terms.csv"
    fit_path = TABLE_DIR / "stage13_model_fit.csv"
    dropped_path = TABLE_DIR / "stage13_dropped_terms.csv"
    summary_path = OUTPUT_DIR / "stage13_logistic_or_inference_summary.md"

    odds.to_csv(odds_path, index=False, encoding="utf-8-sig")
    primary.to_csv(primary_path, index=False, encoding="utf-8-sig")
    significant.to_csv(significant_path, index=False, encoding="utf-8-sig")
    primary.loc[primary["report_candidate"].eq(True)].to_csv(
        report_candidate_path,
        index=False,
        encoding="utf-8-sig",
    )
    fit_summary.to_csv(fit_path, index=False, encoding="utf-8-sig")
    dropped.to_csv(dropped_path, index=False, encoding="utf-8-sig")

    primary_md = primary.copy()
    numeric_cols = primary_md.select_dtypes(include="number").columns
    primary_md[numeric_cols] = primary_md[numeric_cols].round(5)
    odds_md_path.write_text(md_table(primary_md), encoding="utf-8")

    forest_path = make_forest_plot(primary)

    key_fit = fit_summary.sort_values("aic").copy()
    key_fit_round = key_fit.copy()
    key_fit_round[key_fit_round.select_dtypes(include="number").columns] = key_fit_round.select_dtypes(
        include="number"
    ).round(5)

    sig_count = (
        primary.assign(q05=primary["q_value"].le(0.05))
        .groupby("feature_set")["q05"]
        .sum()
        .reset_index(name="q05_significant_terms")
    )
    sig_count["q05_significant_terms"] = sig_count["q05_significant_terms"].astype(int)

    representative = primary.loc[primary["feature_set"].eq("INTERPRET_EDA_INTERACTIONS")].copy()
    representative = representative.loc[
        representative["role"].isin(["core", "core_control", "canada_core", "interaction_core"])
    ].sort_values("q_value")
    representative_show = representative.head(18).copy()
    representative_show[representative_show.select_dtypes(include="number").columns] = representative_show.select_dtypes(
        include="number"
    ).round(5)

    sig_show = significant.head(30).copy()
    if not sig_show.empty:
        sig_show[sig_show.select_dtypes(include="number").columns] = sig_show.select_dtypes(include="number").round(5)

    date_groups = pd.to_datetime(df["기준시각"]).dt.date.nunique()
    cell_groups = df["기상셀ID"].nunique()

    lines = [
        "# Stage 13 로지스틱 오즈비 및 통계 검정",
        "",
        "## 1. 목적",
        "",
        "- Step12에서 확정한 해석용 변수셋을 이용해 로지스틱 GLM을 적합했다.",
        "- 성능 최고 모델의 ANOVA 선택 계수는 사용하지 않았다.",
        "- 요인점수는 제외했다.",
        "- lockbox는 열지 않고 strict development 표본만 사용했다.",
        "- primary 해석 표준오차는 날짜 기준 cluster-robust SE로 둔다.",
        "",
        "## 2. 표본과 군집",
        "",
        f"- development 표본: {len(df):,}행",
        f"- Target 1: {int(df[TARGET].sum()):,}건",
        f"- Target 0: {int((1 - df[TARGET]).sum()):,}건",
        f"- 양성 비율: {df[TARGET].mean():.4f}",
        f"- 날짜 cluster 수: {date_groups:,}",
        f"- 기상셀 cluster 수: {cell_groups:,}",
        "",
        "## 3. 모델 적합 요약",
        "",
        md_table(key_fit_round),
        "",
        "## 4. 날짜 cluster 기준 q<0.05 항 수",
        "",
        md_table(sig_count),
        "",
        "## 5. INTERPRET_EDA_INTERACTIONS 주요 변수 OR",
        "",
        md_table(
            representative_show[
                [
                    "concept_group",
                    "role",
                    "term_label",
                    "unit_for_or",
                    "odds_ratio",
                    "or_ci_low",
                    "or_ci_high",
                    "p_value",
                    "q_value",
                    "std_or_expected_direction",
                ]
            ]
        ),
        "",
        "## 6. FDR q<0.05 변수",
        "",
    ]
    if sig_show.empty:
        lines += ["- 날짜 cluster 기준 FDR q<0.05 변수는 없다.", ""]
    else:
        lines += [
            md_table(
                sig_show[
                    [
                        "feature_set",
                        "concept_group",
                        "role",
                        "term_label",
                        "unit_for_or",
                        "odds_ratio",
                        "or_ci_low",
                        "or_ci_high",
                        "p_value",
                        "q_value",
                    ]
                ]
            ),
            "",
        ]

    lines += [
        "## 7. 해석 주의사항",
        "",
        "- 이 결과는 development 표본 전체에 적합한 통계 해석 모델의 계수표이다. Stage11의 OOF 성능표와 목적이 다르다.",
        "- OR은 관찰자료의 조건부 연관성이지 인과 효과가 아니다.",
        "- 날짜 cluster-robust SE는 2월 집단 발생과 날짜 군집을 완화하기 위한 primary 기준이다.",
        "- 기상셀 cluster-robust 결과는 전체 CSV에 함께 저장했다.",
        "- 캐나다 지수는 D-1 정오 기준만 사용했다.",
        "- `비산림_WUI_접경후보`와 그 기반 상호작용은 Step12에서 제외했으므로 사용하지 않았다.",
        "- `토지피복_L1_NAME=미상`은 q<0.05로 나타나더라도 실제 피복 효과가 아니라 토지피복 매칭 품질/미상 범주의 효과일 수 있으므로 최종 해석 변수로 쓰지 않는다.",
        "- `D1_FWI`와 `D1_ISI`처럼 캐나다지수를 동시에 넣었을 때 부호가 엇갈리는 항은 지수 간 공선성/억제효과 가능성이 있으므로 Step15에서 확인해야 한다.",
        "- 범주형 변수 OR은 지정 기준범주 대비 효과이다.",
        "- 수치형 변수 OR은 `unit_for_or`에 적힌 실제 단위 기준이다. 예를 들어 습도는 5%p 감소 기준이다.",
        "",
        "## 8. 산출물",
        "",
        f"- `{odds_path.relative_to(ROOT)}`",
        f"- `{odds_md_path.relative_to(ROOT)}`",
        f"- `{primary_path.relative_to(ROOT)}`",
        f"- `{significant_path.relative_to(ROOT)}`",
        f"- `{report_candidate_path.relative_to(ROOT)}`",
        f"- `{fit_path.relative_to(ROOT)}`",
        f"- `{dropped_path.relative_to(ROOT)}`",
        f"- `{forest_path.relative_to(ROOT)}`",
        "",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    feature_sets, categorical, mapping = load_stage12_config()
    _, dev = load_modeling_frame()
    odds, fit_summary, dropped = fit_all_models(dev, feature_sets, categorical, mapping)
    primary, significant = make_primary_tables(odds)
    write_outputs(odds, fit_summary, dropped, primary, significant, dev)
    print("Stage13 완료")
    print(f"요약: {OUTPUT_DIR / 'stage13_logistic_or_inference_summary.md'}")
    print(f"OR 표: {TABLE_DIR / 'stage13_odds_ratios.csv'}")


if __name__ == "__main__":
    main()
