from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from stage12_interpret_feature_set import load_modeling_frame
from stage13_logistic_or_inference import (
    build_design_matrix,
    drop_zero_variance_and_collinear,
    fdr_bh,
    result_to_table,
)
from stage15_collinearity_nonlinearity import compute_vif


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
PREDICTION_DIR = OUTPUT_DIR / "predictions"
SPLIT_DIR = OUTPUT_DIR / "splits"

FEATURE_SET_PATH = FEATURE_DIR / "stage12_interpret_feature_sets.json"
MAPPING_PATH = TABLE_DIR / "stage12_feature_mapping.csv"
OUTER_PATH = SPLIT_DIR / "outer_cv_manifest.csv"
STAGE11_BENCHMARK_PATH = TABLE_DIR / "stage11_final_logistic_benchmark.csv"

TARGET = "Target"
ID_COL = "샘플ID"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    for directory in [TABLE_DIR, PLOT_DIR, PREDICTION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_stage12_config() -> tuple[set[str], pd.DataFrame]:
    payload = json.loads(FEATURE_SET_PATH.read_text(encoding="utf-8"))
    categorical = set(payload["categorical_features"])
    mapping = pd.read_csv(MAPPING_PATH, encoding="utf-8-sig")
    return categorical, mapping


def final_model_specs() -> dict[str, dict[str, Any]]:
    base = [
        "기후지형유형",
        "월_sin",
        "월_cos",
        "시간_sin",
        "시간_cos",
        "직전24h_최소습도",
        "rh_local_q05",
        "dry_spell_5p0_gt_240h",
        "wind_max_6h",
        "서풍계열_여부",
        "기압변동_3h",
        "log1p_도로거리_m",
        "고도(m)",
    ]

    return {
        "FINAL_REDUCED_WEATHER_SPACE": {
            "features": base,
            "description": "날씨·건조도·도로접근성 중심의 최소 해석 모델",
            "canada_policy": "none",
            "landcover_policy": "none",
        },
        "FINAL_REDUCED_WITH_FWI": {
            "features": base + ["D1_FWI"],
            "description": "최소 해석 모델에 D-1 FWI 단독 추가",
            "canada_policy": "D1_FWI only",
            "landcover_policy": "none",
        },
        "FINAL_REDUCED_WITH_FFMC_FWI": {
            "features": base + ["D1_FFMC", "D1_FWI"],
            "description": "최소 해석 모델에 D-1 FFMC와 D-1 FWI 추가",
            "canada_policy": "D1_FFMC + D1_FWI",
            "landcover_policy": "none",
        },
        "FINAL_REDUCED_WITH_SIMPLE_LANDCOVER_FLAGS": {
            "features": base
            + [
                "D1_FWI",
                "토지피복_산림지역",
                "토지피복_시가화건조지역",
                "토지피복_초지",
                "도로_10m_이내",
                "시가화_x_도로10m",
            ],
            "description": "FWI 포함 모델에 토지피복 세부범주 대신 단순 이진 플래그 추가",
            "canada_policy": "D1_FWI only",
            "landcover_policy": "simple binary flags",
        },
    }


def validate_specs(dev: pd.DataFrame, specs: dict[str, dict[str, Any]]) -> None:
    missing = {}
    for name, spec in specs.items():
        missing[name] = [feature for feature in spec["features"] if feature not in dev.columns]
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        raise KeyError(f"최종 축소 모델 변수 누락: {missing}")


def best_f1_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, pred)
    if len(thresholds) == 0:
        threshold = 0.5
    else:
        f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-15, None)
        threshold = float(thresholds[int(np.nanargmax(f1_values))])
    y_hat = (pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "best_f1_threshold": threshold,
        "best_f1_accuracy": float(accuracy_score(y_true, y_hat)),
        "best_f1_balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "best_f1_precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "best_f1_recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "best_f1_specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "best_f1_f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "best_f1_mcc": float(matthews_corrcoef(y_true, y_hat)),
        "best_f1_tn": int(tn),
        "best_f1_fp": int(fp),
        "best_f1_fn": int(fn),
        "best_f1_tp": int(tp),
    }


def probability_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    out = {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "auprc": float(average_precision_score(y_true, pred)),
        "auroc": float(roc_auc_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, pred)),
        "log_loss": float(log_loss(y_true, pred, labels=[0, 1])),
    }
    out.update(best_f1_metrics(y_true, pred))
    return out


def fit_train_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: set[str],
    mapping: pd.DataFrame,
) -> tuple[np.ndarray, int, int]:
    X_train_raw, term_info = build_design_matrix(train, features, categorical, mapping)
    X_test_raw, _ = build_design_matrix(test, features, categorical, mapping)
    X_train, term_info, dropped = drop_zero_variance_and_collinear(X_train_raw, term_info)
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0.0)
    y_train = train[TARGET].astype(float)
    result = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit(maxiter=300, disp=0)
    pred = np.asarray(result.predict(X_test), dtype=float)
    return np.clip(pred, 1e-8, 1 - 1e-8), int(X_train.shape[1]), int(len(dropped))


def run_oof(
    dev: pd.DataFrame,
    outer: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    categorical: set[str],
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = dev.merge(outer[[ID_COL, "outer_fold"]], on=ID_COL, how="left", validate="one_to_one")
    if data["outer_fold"].isna().any():
        raise ValueError("outer_fold 없는 행이 있습니다.")

    prediction_parts = []
    fold_rows = []
    for model_name, spec in specs.items():
        features = spec["features"]
        print(f"Step16 OOF: {model_name} ({len(features)} features)")
        pred_all = pd.Series(np.nan, index=data.index, dtype=float)
        for fold in sorted(data["outer_fold"].unique()):
            train = data.loc[data["outer_fold"].ne(fold)].copy()
            test = data.loc[data["outer_fold"].eq(fold)].copy()
            pred, n_terms, n_dropped = fit_train_predict(train, test, features, categorical, mapping)
            pred_all.loc[test.index] = pred
            fold_metric = probability_metrics(test[TARGET].to_numpy(dtype=int), pred)
            fold_metric.update(
                {
                    "model": model_name,
                    "outer_fold": int(fold),
                    "n_terms_including_intercept": n_terms,
                    "n_dropped_terms": n_dropped,
                }
            )
            fold_rows.append(fold_metric)
        if pred_all.isna().any():
            raise ValueError(f"OOF prediction missing: {model_name}")
        prediction_parts.append(
            pd.DataFrame(
                {
                    ID_COL: data[ID_COL],
                    "Target": data[TARGET].astype(int),
                    "샘플유형": data["샘플유형"],
                    "outer_fold": data["outer_fold"].astype(int),
                    "model": model_name,
                    "pred_prob": pred_all.to_numpy(dtype=float),
                }
            )
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    overall_rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        metrics = probability_metrics(group[TARGET].to_numpy(dtype=int), group["pred_prob"].to_numpy(dtype=float))
        metrics["model"] = model_name
        overall_rows.append(metrics)
    overall = pd.DataFrame(overall_rows).sort_values("auprc", ascending=False)
    return predictions, fold_metrics, overall


def fit_full_models(
    dev: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    categorical: set[str],
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = dev[TARGET].astype(float)
    date_groups = pd.to_datetime(dev["기준시각"]).dt.strftime("%Y-%m-%d")
    fit_rows = []
    or_parts = []
    vif_parts = []
    dropped_parts = []

    for model_name, spec in specs.items():
        features = spec["features"]
        X_raw, term_info = build_design_matrix(dev, features, categorical, mapping)
        X, term_info, dropped = drop_zero_variance_and_collinear(X_raw, term_info)
        result = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            maxiter=300,
            disp=0,
            cov_type="cluster",
            cov_kwds={"groups": date_groups, "use_correction": True},
        )
        pred = np.asarray(result.predict(X), dtype=float)
        or_table = result_to_table(model_name, "cluster_date", result, term_info)
        or_table["model"] = model_name
        or_parts.append(or_table)

        vif = compute_vif(X, term_info)
        vif.insert(0, "model", model_name)
        vif_parts.append(vif)

        if len(dropped):
            dropped = dropped.copy()
            dropped.insert(0, "model", model_name)
            dropped_parts.append(dropped)

        fit_rows.append(
            {
                "model": model_name,
                "n": int(len(y)),
                "positive_n": int(y.sum()),
                "positive_rate": float(y.mean()),
                "n_features": len(features),
                "n_terms_including_intercept": int(X.shape[1]),
                "n_dropped_terms": int(len(dropped)),
                "converged": bool(result.converged),
                "log_likelihood": float(result.llf),
                "null_log_likelihood": float(result.llnull),
                "mcfadden_r2": float(1 - result.llf / result.llnull),
                "aic": float(result.aic),
                "bic_llf": float(getattr(result, "bic_llf", np.nan)),
                "in_sample_auprc": float(average_precision_score(y, pred)),
                "in_sample_auroc": float(roc_auc_score(y, pred)),
                "in_sample_brier": float(brier_score_loss(y, pred)),
                "in_sample_log_loss": float(log_loss(y, np.clip(pred, 1e-8, 1 - 1e-8), labels=[0, 1])),
                "max_vif": float(pd.to_numeric(vif.loc[vif["term_id"].ne("const"), "vif"], errors="coerce").max()),
                "vif_ge_5_terms": int((pd.to_numeric(vif["vif"], errors="coerce") >= 5).sum()),
                "vif_ge_10_terms": int((pd.to_numeric(vif["vif"], errors="coerce") >= 10).sum()),
            }
        )

    fit_summary = pd.DataFrame(fit_rows)
    odds = pd.concat(or_parts, ignore_index=True)
    vifs = pd.concat(vif_parts, ignore_index=True)
    dropped_all = (
        pd.concat(dropped_parts, ignore_index=True)
        if dropped_parts
        else pd.DataFrame(columns=["model", "term_id", "drop_reason", "feature", "term_label"])
    )
    return odds, fit_summary, vifs, dropped_all


def make_report_or_table(odds: pd.DataFrame, recommended_model: str) -> pd.DataFrame:
    table = odds.loc[odds["model"].eq(recommended_model) & odds["term_id"].ne("const")].copy()
    table["q_value"] = fdr_bh(table["p_value"])
    table["significant_q05"] = table["q_value"].le(0.05)
    keep_roles = {
        "core",
        "core_control",
        "canada_core",
        "core_interaction",
        "interaction_core",
        "secondary",
    }
    table = table.loc[table["role"].isin(keep_roles)].copy()
    cols = [
        "model",
        "concept_group",
        "role",
        "feature",
        "term_label",
        "unit_for_or",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
        "p_value",
        "q_value",
        "significant_q05",
        "std_or_expected_direction",
        "report_caution",
    ]
    return table[[col for col in cols if col in table.columns]].sort_values(["q_value", "p_value"])


def choose_recommended(overall: pd.DataFrame, fit_summary: pd.DataFrame) -> str:
    merged = overall.merge(fit_summary[["model", "max_vif", "vif_ge_10_terms"]], on="model", how="left")
    stable = merged.loc[merged["vif_ge_10_terms"].eq(0)].copy()
    if stable.empty:
        stable = merged.copy()
    stable = stable.sort_values(["auprc", "brier"], ascending=[False, True])
    return str(stable.iloc[0]["model"])


def plot_metrics(overall: pd.DataFrame, recommended_model: str) -> Path:
    plot_df = overall.copy().sort_values("auprc", ascending=True)
    labels = plot_df["model"].str.replace("_", "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    metrics = [("auprc", "OOF AUPRC", "#2563eb"), ("auroc", "OOF ROC AUC", "#16a34a"), ("brier", "OOF Brier", "#f97316")]
    for ax, (metric, title, color) in zip(axes, metrics):
        bars = ax.barh(labels, plot_df[metric], color=color, alpha=0.85)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        for idx, (value, model) in enumerate(zip(plot_df[metric], plot_df["model"])):
            weight = "bold" if model == recommended_model else "normal"
            ax.text(value, idx, f" {value:.3f}", va="center", fontsize=8, fontweight=weight)
    path = PLOT_DIR / "stage16_final_model_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_or(report_or: pd.DataFrame, recommended_model: str) -> Path:
    plot_df = report_or.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["odds_ratio", "or_ci_low", "or_ci_high"])
    plot_df = plot_df.loc[(plot_df["or_ci_low"] > 0) & (plot_df["or_ci_high"] < 100)]
    plot_df["abs_log_or"] = np.abs(np.log(plot_df["odds_ratio"]))
    plot_df = plot_df.sort_values("abs_log_or", ascending=False).head(24).sort_values("odds_ratio")

    labels = plot_df["term_label"].astype(str).str.replace(" (", "\n(", regex=False)
    y_pos = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.38)), constrained_layout=True)
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
    ax.set_title(f"Stage16 recommended OR: {recommended_model}")
    ax.grid(axis="x", alpha=0.25)
    path = PLOT_DIR / "stage16_final_or_forestplot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_outputs(
    specs: dict[str, dict[str, Any]],
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    overall: pd.DataFrame,
    odds: pd.DataFrame,
    fit_summary: pd.DataFrame,
    vifs: pd.DataFrame,
    dropped: pd.DataFrame,
    recommended_model: str,
) -> None:
    design_path = TABLE_DIR / "stage16_final_model_design.csv"
    metrics_path = TABLE_DIR / "stage16_final_model_metrics.csv"
    fold_path = TABLE_DIR / "stage16_final_fold_metrics.csv"
    pred_path = PREDICTION_DIR / "stage16_final_oof_predictions.csv"
    odds_path = TABLE_DIR / "stage16_final_or_date_cluster.csv"
    report_or_path = TABLE_DIR / "stage16_final_report_or_table.csv"
    fit_path = TABLE_DIR / "stage16_final_glm_fit.csv"
    vif_path = TABLE_DIR / "stage16_final_vif.csv"
    dropped_path = TABLE_DIR / "stage16_final_dropped_terms.csv"
    summary_path = OUTPUT_DIR / "stage16_final_reduced_logistic_summary.md"

    design = pd.DataFrame(
        [
            {
                "model": name,
                "n_features": len(spec["features"]),
                "features": ", ".join(spec["features"]),
                "description": spec["description"],
                "canada_policy": spec["canada_policy"],
                "landcover_policy": spec["landcover_policy"],
            }
            for name, spec in specs.items()
        ]
    )
    design.to_csv(design_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(fold_path, index=False, encoding="utf-8-sig")

    metrics = overall.merge(
        fit_summary[
            [
                "model",
                "n_features",
                "n_terms_including_intercept",
                "n_dropped_terms",
                "mcfadden_r2",
                "aic",
                "bic_llf",
                "max_vif",
                "vif_ge_5_terms",
                "vif_ge_10_terms",
            ]
        ],
        on="model",
        how="left",
    ).merge(design[["model", "description", "canada_policy", "landcover_policy"]], on="model", how="left")
    metrics["recommended"] = metrics["model"].eq(recommended_model)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    odds.to_csv(odds_path, index=False, encoding="utf-8-sig")
    fit_summary.to_csv(fit_path, index=False, encoding="utf-8-sig")
    vifs.to_csv(vif_path, index=False, encoding="utf-8-sig")
    dropped.to_csv(dropped_path, index=False, encoding="utf-8-sig")

    report_or = make_report_or_table(odds, recommended_model)
    report_or.to_csv(report_or_path, index=False, encoding="utf-8-sig")

    metric_plot = plot_metrics(overall, recommended_model)
    or_plot = plot_or(report_or, recommended_model)

    metrics_show = metrics.copy().sort_values(["recommended", "auprc"], ascending=[False, False])
    metrics_show[metrics_show.select_dtypes(include="number").columns] = metrics_show.select_dtypes(include="number").round(5)
    metrics_show = metrics_show[
        [
            "model",
            "recommended",
            "auprc",
            "auroc",
            "brier",
            "log_loss",
            "best_f1_f1",
            "best_f1_precision",
            "best_f1_recall",
            "mcfadden_r2",
            "aic",
            "max_vif",
            "vif_ge_10_terms",
            "description",
        ]
    ]

    or_show = report_or.copy()
    or_show[or_show.select_dtypes(include="number").columns] = or_show.select_dtypes(include="number").round(5)
    or_show = or_show[
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
            "significant_q05",
        ]
    ]

    recommended_metrics = metrics.loc[metrics["model"].eq(recommended_model)].iloc[0]
    stage11_note = ""
    if STAGE11_BENCHMARK_PATH.exists():
        stage11 = pd.read_csv(STAGE11_BENCHMARK_PATH, encoding="utf-8-sig")
        rep = stage11.loc[stage11["feature_set"].eq("PLUS_LANDCOVER_RULES_ANOVA")]
        if not rep.empty:
            rep = rep.iloc[0]
            stage11_note = (
                f"- 참고: Stage11 대표 성능 모델 `PLUS_LANDCOVER_RULES_ANOVA`는 "
                f"AUPRC {rep.auprc:.4f}, ROC AUC {rep.auroc:.4f}, Brier {rep.brier:.5f}였다. "
                "Stage16 모델은 성능 최고 모델이 아니라 보고서용 해석 안정 모델이다."
            )

    lines = [
        "# Stage 16 최종 축소·안정 로지스틱 모델",
        "",
        "## 1. 목적",
        "",
        "- Step13~15 결과를 반영해 최종 보고서용 축소·안정 로지스틱 모델을 만들었다.",
        "- 캐나다지수 전체 동시 투입은 공선성 때문에 피했다.",
        "- 토지피복 세부 범주는 희소범주/선형종속 문제 때문에 사용하지 않았다.",
        "- 예측 성능은 strict outer fold OOF 기준으로 산출했다.",
        "- OR은 development 전체 적합 후 날짜 cluster-robust SE 기준으로 산출했다.",
        "- lockbox는 사용하지 않았다.",
        "",
        "## 2. 후보 모델",
        "",
        md_table(design[["model", "n_features", "canada_policy", "landcover_policy", "description"]]),
        "",
        "## 3. 최종 추천 모델",
        "",
        f"- 추천 모델: `{recommended_model}`",
        f"- OOF AUPRC: {recommended_metrics.auprc:.4f}",
        f"- OOF ROC AUC: {recommended_metrics.auroc:.4f}",
        f"- OOF Brier: {recommended_metrics.brier:.5f}",
        f"- OOF log loss: {recommended_metrics.log_loss:.5f}",
        f"- OOF best-F1: {recommended_metrics.best_f1_f1:.4f}",
        f"- max VIF: {recommended_metrics.max_vif:.2f}",
        "",
    ]
    if stage11_note:
        lines += [stage11_note, ""]

    lines += [
        "## 4. 후보 모델 성능 비교",
        "",
        md_table(metrics_show),
        "",
        "## 5. 추천 모델 날짜 cluster 기준 OR",
        "",
        md_table(or_show),
        "",
        "## 6. 해석",
        "",
        "- Step16 모델은 Stage11의 성능 최고 모델을 대체하는 목적이 아니다.",
        "- 목적은 최종 보고서에서 방어 가능한 OR 방향과 신뢰구간을 제시하는 것이다.",
        "- 캐나다지수는 `D1_FWI` 단독 또는 `D1_FFMC + D1_FWI` 정도로 축소했기 때문에 Step13의 FWI/ISI 부호 역전 문제를 줄였다.",
        "- 토지피복 세부 범주는 제외했으며, 단순 플래그 모델은 별도로 비교했다.",
        "- 최종 보고서에서는 Stage11을 성능 기준, Stage16을 해석 기준으로 분리해서 사용한다.",
        "",
        "## 7. 산출물",
        "",
        f"- `{design_path.relative_to(ROOT)}`",
        f"- `{metrics_path.relative_to(ROOT)}`",
        f"- `{fold_path.relative_to(ROOT)}`",
        f"- `{pred_path.relative_to(ROOT)}`",
        f"- `{odds_path.relative_to(ROOT)}`",
        f"- `{report_or_path.relative_to(ROOT)}`",
        f"- `{fit_path.relative_to(ROOT)}`",
        f"- `{vif_path.relative_to(ROOT)}`",
        f"- `{dropped_path.relative_to(ROOT)}`",
        f"- `{metric_plot.relative_to(ROOT)}`",
        f"- `{or_plot.relative_to(ROOT)}`",
        "",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    categorical, mapping = load_stage12_config()
    specs = final_model_specs()
    _, dev = load_modeling_frame()
    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    outer = outer.loc[outer[ID_COL].isin(set(dev[ID_COL]))].copy()
    validate_specs(dev, specs)

    predictions, fold_metrics, overall = run_oof(dev, outer, specs, categorical, mapping)
    odds, fit_summary, vifs, dropped = fit_full_models(dev, specs, categorical, mapping)
    recommended = choose_recommended(overall, fit_summary)
    write_outputs(specs, predictions, fold_metrics, overall, odds, fit_summary, vifs, dropped, recommended)

    print("Stage16 완료")
    print(f"추천 모델: {recommended}")
    print(f"요약: {OUTPUT_DIR / 'stage16_final_reduced_logistic_summary.md'}")


if __name__ == "__main__":
    main()
