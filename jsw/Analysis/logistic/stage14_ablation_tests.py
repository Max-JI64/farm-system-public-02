from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
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
)


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
FULL_SET_NAME = "FULL_INTERPRET_EDA_INTERACTIONS"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    for directory in [TABLE_DIR, PLOT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_stage12() -> tuple[list[str], set[str], pd.DataFrame]:
    payload = json.loads(FEATURE_SET_PATH.read_text(encoding="utf-8"))
    full_features = payload["feature_sets"]["INTERPRET_EDA_INTERACTIONS"]
    categorical = set(payload["categorical_features"])
    mapping = pd.read_csv(MAPPING_PATH, encoding="utf-8-sig")
    return full_features, categorical, mapping


def feature_group_map(mapping: pd.DataFrame) -> dict[str, str]:
    return mapping.set_index("feature")["concept_group"].to_dict()


def make_ablation_sets(full_features: list[str], mapping: pd.DataFrame) -> dict[str, dict[str, Any]]:
    group = feature_group_map(mapping)

    weather_features = {
        f
        for f in full_features
        if group.get(f) in {"습도/건조", "강수/무강수", "풍속/풍향", "기압/기온"}
    }
    weather_interactions = {
        "영동_x_wind_max_6h",
        "rh_local_q05_AND_wind_max_6h_ge_5",
        "rh_local_q05_AND_westerly_strong_max_6h",
        "초지_x_dry0p1",
        "영동_x_rh_q05_x_wind5",
    }

    space_features = {
        f
        for f in full_features
        if group.get(f) in {"공간/접근성", "지형"}
    }
    space_interactions = {"시가화_x_도로10m"}

    canada_features = {f for f in full_features if group.get(f) == "캐나다 산불지수"}
    landcover_features = {f for f in full_features if group.get(f) == "토지피복"}
    landcover_interactions = {"시가화_x_도로10m", "초지_x_dry0p1"}
    interaction_features = {f for f in full_features if group.get(f) == "상호작용"}
    topography_features = {f for f in full_features if group.get(f) == "지형"}

    def drop(remove: set[str]) -> list[str]:
        return [f for f in full_features if f not in remove]

    specs = {
        FULL_SET_NAME: {
            "features": full_features,
            "removed_group": "none",
            "description": "Step12의 전체 해석 모델",
            "is_full": True,
            "reference_family": "full",
        },
        "DROP_WEATHER": {
            "features": drop(weather_features | weather_interactions),
            "removed_group": "weather_main_and_weather_interactions",
            "description": "습도·강수·풍속·기압 및 날씨 기반 상호작용 제거",
            "is_full": False,
            "reference_family": "full",
        },
        "DROP_SPACE": {
            "features": drop(space_features | space_interactions),
            "removed_group": "space_accessibility_topography_and_space_interactions",
            "description": "공간접근성·지형 및 공간 기반 상호작용 제거",
            "is_full": False,
            "reference_family": "full",
        },
        "DROP_CANADA": {
            "features": drop(canada_features),
            "removed_group": "canadian_indices",
            "description": "D-1 캐나다 산불지수 제거",
            "is_full": False,
            "reference_family": "full",
        },
        "DROP_LANDCOVER": {
            "features": drop(landcover_features | landcover_interactions),
            "removed_group": "landcover_and_landcover_interactions",
            "description": "토지피복 범주 및 토지피복 기반 상호작용 제거. OOF 안정 기준모델로도 사용",
            "is_full": False,
            "reference_family": "full_and_stable",
        },
        "DROP_INTERACTIONS": {
            "features": drop(interaction_features),
            "removed_group": "eda_interactions",
            "description": "EDA 기반 상호작용만 제거",
            "is_full": False,
            "reference_family": "full",
        },
        "DROP_TOPOGRAPHY": {
            "features": drop(topography_features),
            "removed_group": "topography",
            "description": "고도·경사도·TPI 제거",
            "is_full": False,
            "reference_family": "full",
        },
        "STABLE_DROP_WEATHER": {
            "features": [
                f
                for f in drop(landcover_features | landcover_interactions)
                if f not in (weather_features | weather_interactions)
            ],
            "removed_group": "stable_weather_main_and_weather_interactions",
            "description": "토지피복 제거 안정 기준에서 습도·강수·풍속·기압 및 날씨 상호작용 제거",
            "is_full": False,
            "reference_family": "stable_no_landcover",
        },
        "STABLE_DROP_SPACE": {
            "features": [
                f
                for f in drop(landcover_features | landcover_interactions)
                if f not in (space_features | space_interactions)
            ],
            "removed_group": "stable_space_accessibility_topography",
            "description": "토지피복 제거 안정 기준에서 공간접근성·지형 제거",
            "is_full": False,
            "reference_family": "stable_no_landcover",
        },
        "STABLE_DROP_CANADA": {
            "features": [
                f
                for f in drop(landcover_features | landcover_interactions)
                if f not in canada_features
            ],
            "removed_group": "stable_canadian_indices",
            "description": "토지피복 제거 안정 기준에서 D-1 캐나다 산불지수 제거",
            "is_full": False,
            "reference_family": "stable_no_landcover",
        },
        "STABLE_DROP_INTERACTIONS": {
            "features": [
                f
                for f in drop(landcover_features | landcover_interactions)
                if f not in interaction_features
            ],
            "removed_group": "stable_eda_interactions",
            "description": "토지피복 제거 안정 기준에서 EDA 기반 상호작용 제거",
            "is_full": False,
            "reference_family": "stable_no_landcover",
        },
        "STABLE_DROP_TOPOGRAPHY": {
            "features": [
                f
                for f in drop(landcover_features | landcover_interactions)
                if f not in topography_features
            ],
            "removed_group": "stable_topography",
            "description": "토지피복 제거 안정 기준에서 고도·경사도·TPI 제거",
            "is_full": False,
            "reference_family": "stable_no_landcover",
        },
    }

    rows = []
    for name, spec in specs.items():
        removed = [f for f in full_features if f not in spec["features"]]
        spec["removed_features"] = removed
        rows.append(
            {
                "ablation_model": name,
                "n_features": len(spec["features"]),
                "n_removed_features": len(removed),
                "removed_group": spec["removed_group"],
                "description": spec["description"],
                "reference_family": spec["reference_family"],
                "removed_features": ", ".join(removed),
            }
        )
    spec_table = pd.DataFrame(rows)
    return specs, spec_table


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


def fit_glm_train_predict(
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
    return np.clip(pred, 1e-8, 1 - 1e-8), X_train.shape[1], len(dropped)


def run_oof(
    dev: pd.DataFrame,
    outer: pd.DataFrame,
    ablation_sets: dict[str, dict[str, Any]],
    categorical: set[str],
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = dev.merge(outer[[ID_COL, "outer_fold"]], on=ID_COL, how="left", validate="one_to_one")
    if data["outer_fold"].isna().any():
        raise ValueError("outer_fold가 없는 development 행이 있습니다.")

    prediction_rows = []
    fold_rows = []

    for model_name, spec in ablation_sets.items():
        features = spec["features"]
        print(f"Step14 OOF: {model_name} ({len(features)} features)")
        pred_all = pd.Series(np.nan, index=data.index, dtype=float)
        for fold in sorted(data["outer_fold"].unique()):
            train = data.loc[data["outer_fold"].ne(fold)].copy()
            test = data.loc[data["outer_fold"].eq(fold)].copy()
            pred, n_terms, n_dropped = fit_glm_train_predict(train, test, features, categorical, mapping)
            pred_all.loc[test.index] = pred
            fold_metric = probability_metrics(test[TARGET].to_numpy(dtype=int), pred)
            fold_metric.update(
                {
                    "ablation_model": model_name,
                    "outer_fold": int(fold),
                    "n_terms_including_intercept": int(n_terms),
                    "n_dropped_terms": int(n_dropped),
                }
            )
            fold_rows.append(fold_metric)

        if pred_all.isna().any():
            raise ValueError(f"OOF 예측 누락: {model_name}")
        prediction_rows.append(
            pd.DataFrame(
                {
                    ID_COL: data[ID_COL],
                    "Target": data[TARGET].astype(int),
                    "샘플유형": data["샘플유형"],
                    "outer_fold": data["outer_fold"].astype(int),
                    "ablation_model": model_name,
                    "pred_prob": pred_all.to_numpy(dtype=float),
                }
            )
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    return predictions, fold_metrics


def summarize_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("ablation_model", sort=False):
        metrics = probability_metrics(group["Target"].to_numpy(dtype=int), group["pred_prob"].to_numpy(dtype=float))
        metrics["ablation_model"] = model_name
        rows.append(metrics)
    overall = pd.DataFrame(rows)
    full = overall.loc[overall["ablation_model"].eq(FULL_SET_NAME)].iloc[0]
    stable = overall.loc[overall["ablation_model"].eq("DROP_LANDCOVER")].iloc[0]
    for metric in ["auprc", "auroc", "brier", "log_loss", "best_f1_f1", "best_f1_precision", "best_f1_recall"]:
        overall[f"delta_{metric}_vs_full"] = overall[metric] - full[metric]
        overall[f"delta_{metric}_vs_stable_no_landcover"] = overall[metric] - stable[metric]
    return overall.sort_values("auprc", ascending=False)


def fit_full_development_models(
    dev: pd.DataFrame,
    ablation_sets: dict[str, dict[str, Any]],
    categorical: set[str],
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = dev[TARGET].astype(float)
    fit_rows = []
    dropped_rows = []
    for model_name, spec in ablation_sets.items():
        features = spec["features"]
        X_raw, term_info = build_design_matrix(dev, features, categorical, mapping)
        X, term_info, dropped = drop_zero_variance_and_collinear(X_raw, term_info)
        result = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=300, disp=0)
        fit_rows.append(
            {
                "ablation_model": model_name,
                "n": int(len(y)),
                "positive_n": int(y.sum()),
                "n_features": len(features),
                "n_terms_including_intercept": int(X.shape[1]),
                "n_dropped_terms": int(len(dropped)),
                "converged": bool(result.converged),
                "log_likelihood": float(result.llf),
                "null_log_likelihood": float(result.llnull),
                "mcfadden_r2": float(1 - result.llf / result.llnull),
                "aic": float(result.aic),
                "bic_llf": float(getattr(result, "bic_llf", np.nan)),
            }
        )
        if len(dropped):
            dropped = dropped.copy()
            dropped.insert(0, "ablation_model", model_name)
            dropped_rows.append(dropped)

    fit_summary = pd.DataFrame(fit_rows)
    dropped_summary = (
        pd.concat(dropped_rows, ignore_index=True)
        if dropped_rows
        else pd.DataFrame(columns=["ablation_model", "term_id", "drop_reason", "feature", "term_label"])
    )
    return fit_summary, dropped_summary


def likelihood_ratio_tests(fit_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    comparisons = [
        (
            FULL_SET_NAME,
            fit_summary.loc[
                fit_summary["reference_family"].isin(["full", "full_and_stable"])
                & fit_summary["ablation_model"].ne(FULL_SET_NAME),
                "ablation_model",
            ].tolist(),
        ),
        (
            "DROP_LANDCOVER",
            fit_summary.loc[
                fit_summary["reference_family"].eq("stable_no_landcover"),
                "ablation_model",
            ].tolist(),
        ),
    ]

    for reference_name, candidates in comparisons:
        reference = fit_summary.loc[fit_summary["ablation_model"].eq(reference_name)].iloc[0]
        for candidate in candidates:
            row = fit_summary.loc[fit_summary["ablation_model"].eq(candidate)].iloc[0]
            df_diff = int(reference["n_terms_including_intercept"] - row["n_terms_including_intercept"])
            lr_stat = 2 * (float(reference["log_likelihood"]) - float(row["log_likelihood"]))
            p_value = float(chi2.sf(lr_stat, df_diff)) if df_diff > 0 else np.nan
            rows.append(
                {
                    "reference_model": reference_name,
                    "reduced_model": row["ablation_model"],
                    "removed_group": row.get("removed_group", ""),
                    "df_diff": df_diff,
                    "lr_stat": lr_stat,
                    "p_value": p_value,
                    "delta_aic_reduced_minus_reference": float(row["aic"] - reference["aic"]),
                    "delta_bic_reduced_minus_reference": float(row["bic_llf"] - reference["bic_llf"]),
                    "delta_log_likelihood_reduced_minus_reference": float(
                        row["log_likelihood"] - reference["log_likelihood"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["reference_model", "lr_stat"], ascending=[True, False])


def make_plots(overall: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = overall.copy()
    plot_df["label"] = plot_df["ablation_model"].str.replace("_", "\n")
    order = plot_df.sort_values("auprc", ascending=True)

    auprc_path = PLOT_DIR / "stage14_ablation_auprc.png"
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(order["label"], order["auprc"], color="#2563eb", alpha=0.85)
    ax.axvline(float(overall.loc[overall["ablation_model"].eq(FULL_SET_NAME), "auprc"].iloc[0]), color="#ef4444", linestyle="--")
    ax.set_xlabel("OOF AUPRC")
    ax.set_title("Step14 변수군 제거 실험: AUPRC")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(order["auprc"]):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=8)
    fig.savefig(auprc_path, dpi=180)
    plt.close(fig)

    brier_path = PLOT_DIR / "stage14_ablation_brier.png"
    order = plot_df.sort_values("brier", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(order["label"], order["brier"], color="#f97316", alpha=0.85)
    ax.axvline(float(overall.loc[overall["ablation_model"].eq(FULL_SET_NAME), "brier"].iloc[0]), color="#ef4444", linestyle="--")
    ax.set_xlabel("OOF Brier score")
    ax.set_title("Step14 변수군 제거 실험: Brier")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(order["brier"]):
        ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)
    fig.savefig(brier_path, dpi=180)
    plt.close(fig)
    return auprc_path, brier_path


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_outputs(
    spec_table: pd.DataFrame,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    overall: pd.DataFrame,
    fit_summary: pd.DataFrame,
    lrt: pd.DataFrame,
    dropped: pd.DataFrame,
) -> None:
    spec_path = TABLE_DIR / "stage14_ablation_design.csv"
    pred_path = OUTPUT_DIR / "predictions" / "stage14_ablation_oof_predictions.csv"
    fold_path = TABLE_DIR / "stage14_ablation_fold_metrics.csv"
    overall_path = TABLE_DIR / "stage14_ablation_metrics.csv"
    fit_path = TABLE_DIR / "stage14_ablation_glm_fit.csv"
    lrt_path = TABLE_DIR / "stage14_likelihood_ratio_tests.csv"
    dropped_path = TABLE_DIR / "stage14_ablation_dropped_terms.csv"
    summary_path = OUTPUT_DIR / "stage14_ablation_tests_summary.md"

    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    spec_table.to_csv(spec_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(fold_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    fit_summary.to_csv(fit_path, index=False, encoding="utf-8-sig")
    lrt.to_csv(lrt_path, index=False, encoding="utf-8-sig")
    dropped.to_csv(dropped_path, index=False, encoding="utf-8-sig")

    auprc_plot, brier_plot = make_plots(overall)

    full = overall.loc[overall["ablation_model"].eq(FULL_SET_NAME)].iloc[0]
    stable = overall.loc[overall["ablation_model"].eq("DROP_LANDCOVER")].iloc[0]
    ranking = overall.sort_values("delta_auprc_vs_full").copy()
    ranking["auprc_loss_vs_full"] = -ranking["delta_auprc_vs_full"]
    ranking_show = ranking[
        [
            "ablation_model",
            "auprc",
            "delta_auprc_vs_full",
            "auroc",
            "delta_auroc_vs_full",
            "brier",
            "delta_brier_vs_full",
            "log_loss",
            "delta_log_loss_vs_full",
            "best_f1_f1",
            "delta_best_f1_f1_vs_full",
        ]
    ].copy()
    ranking_show[ranking_show.select_dtypes(include="number").columns] = ranking_show.select_dtypes(
        include="number"
    ).round(5)

    stable_ranking = overall.loc[
        overall["ablation_model"].eq("DROP_LANDCOVER")
        | overall["ablation_model"].str.startswith("STABLE_")
    ].sort_values("delta_auprc_vs_stable_no_landcover")
    stable_show = stable_ranking[
        [
            "ablation_model",
            "auprc",
            "delta_auprc_vs_stable_no_landcover",
            "auroc",
            "delta_auroc_vs_stable_no_landcover",
            "brier",
            "delta_brier_vs_stable_no_landcover",
            "log_loss",
            "delta_log_loss_vs_stable_no_landcover",
            "best_f1_f1",
            "delta_best_f1_f1_vs_stable_no_landcover",
        ]
    ].copy()
    stable_show[stable_show.select_dtypes(include="number").columns] = stable_show.select_dtypes(
        include="number"
    ).round(5)

    if {"removed_group", "description"}.issubset(fit_summary.columns):
        fit_show = fit_summary.copy()
    else:
        fit_show = fit_summary.merge(
            spec_table[["ablation_model", "removed_group", "description"]],
            on="ablation_model",
            how="left",
        )
    fit_show = fit_show.sort_values("aic")
    fit_show[fit_show.select_dtypes(include="number").columns] = fit_show.select_dtypes(include="number").round(5)

    lrt_show = lrt.copy()
    lrt_show[lrt_show.select_dtypes(include="number").columns] = lrt_show.select_dtypes(include="number").round(6)

    lines = [
        "# Stage 14 변수군 제거 실험",
        "",
        "## 1. 목적",
        "",
        "- Step12/13의 전체 해석 모델에서 변수군을 하나씩 제거해 성능과 적합도 변화를 확인했다.",
        "- 예측 성능은 strict outer fold OOF 기준으로 계산했다.",
        "- 통계 적합도와 likelihood ratio test는 development 전체 GLM 적합 기준으로 계산했다.",
        "- lockbox는 사용하지 않았다.",
        "- 요인점수와 비통계적 이진분류 모델은 사용하지 않았다.",
        "",
        "## 2. 기준 모델",
        "",
        f"- 기준 모델: `{FULL_SET_NAME}`",
        f"- OOF AUPRC: {full.auprc:.4f}",
        f"- OOF ROC AUC: {full.auroc:.4f}",
        f"- OOF Brier: {full.brier:.5f}",
        f"- OOF log loss: {full.log_loss:.5f}",
        f"- OOF best-F1: {full.best_f1_f1:.4f}",
        "",
        "추가로, 전체 모델에서 토지피복 범주를 제거한 `DROP_LANDCOVER`가 OOF에서 더 안정적이었으므로 이를 안정 기준모델로도 따로 비교했다.",
        "",
        f"- 안정 기준모델: `DROP_LANDCOVER`",
        f"- OOF AUPRC: {stable.auprc:.4f}",
        f"- OOF ROC AUC: {stable.auroc:.4f}",
        f"- OOF Brier: {stable.brier:.5f}",
        f"- OOF log loss: {stable.log_loss:.5f}",
        f"- OOF best-F1: {stable.best_f1_f1:.4f}",
        "",
        "## 3. 변수군 제거 설계",
        "",
        md_table(
            spec_table[
                [
                    "ablation_model",
                    "n_features",
                    "n_removed_features",
                    "removed_group",
                    "reference_family",
                    "description",
                ]
            ]
        ),
        "",
        "## 4. 전체 모델 기준 OOF 성능 변화",
        "",
        md_table(ranking_show),
        "",
        "해석 기준:",
        "",
        "- `delta_auprc_vs_full`이 음수이면 해당 변수군 제거로 AUPRC가 하락했다는 뜻이다.",
        "- `delta_brier_vs_full`이 양수이면 해당 변수군 제거로 확률 오차가 악화됐다는 뜻이다.",
        "",
        "## 4-1. 토지피복 제거 안정 기준 OOF 성능 변화",
        "",
        md_table(stable_show),
        "",
        "해석 기준:",
        "",
        "- `delta_auprc_vs_stable_no_landcover`가 음수이면 안정 기준모델 대비 AUPRC가 하락했다는 뜻이다.",
        "- 이 표는 토지피복 범주 불안정성을 제거한 뒤 날씨, 공간, 캐나다지수, 상호작용 기여도를 다시 보기 위한 것이다.",
        "",
        "## 5. GLM 적합도",
        "",
        md_table(
            fit_show[
                [
                    "ablation_model",
                    "removed_group",
                    "n_features",
                    "n_terms_including_intercept",
                    "n_dropped_terms",
                    "mcfadden_r2",
                    "aic",
                    "bic_llf",
                    "description",
                ]
            ]
        ),
        "",
        "## 6. Likelihood ratio test",
        "",
        md_table(lrt_show),
        "",
        "## 7. 해석",
        "",
    ]

    full_family = overall.loc[
        overall["ablation_model"].isin(
            [
                "DROP_WEATHER",
                "DROP_SPACE",
                "DROP_CANADA",
                "DROP_LANDCOVER",
                "DROP_INTERACTIONS",
                "DROP_TOPOGRAPHY",
            ]
        )
    ].copy()
    most_auprc_loss = full_family.sort_values("delta_auprc_vs_full").iloc[0]
    most_brier_worse = full_family.sort_values(
        "delta_brier_vs_full", ascending=False
    ).iloc[0]
    stable_family = overall.loc[overall["ablation_model"].str.startswith("STABLE_")].copy()
    stable_auprc_loss = stable_family.sort_values("delta_auprc_vs_stable_no_landcover").iloc[0]
    stable_brier_worse = stable_family.sort_values(
        "delta_brier_vs_stable_no_landcover", ascending=False
    ).iloc[0]
    lines += [
        f"- OOF AUPRC 기준 가장 큰 하락은 `{most_auprc_loss.ablation_model}`에서 나타났다 "
        f"({most_auprc_loss.delta_auprc_vs_full:.4f}).",
        f"- Brier 기준 가장 큰 악화는 `{most_brier_worse.ablation_model}`에서 나타났다 "
        f"({most_brier_worse.delta_brier_vs_full:.5f}).",
        f"- 전체 모델에서는 `DROP_LANDCOVER`가 오히려 AUPRC를 {stable.delta_auprc_vs_full:+.4f} 높이고 Brier를 {stable.delta_brier_vs_full:+.5f} 낮췄다. 이는 토지피복 범주가 unregularized GLM의 OOF 일반화를 불안정하게 만든다는 신호이다.",
        f"- 토지피복 제거 안정 기준에서는 AUPRC 기준 가장 큰 하락이 `{stable_auprc_loss.ablation_model}`에서 나타났다 "
        f"({stable_auprc_loss.delta_auprc_vs_stable_no_landcover:.4f}).",
        f"- 토지피복 제거 안정 기준에서 Brier 기준 가장 큰 악화는 `{stable_brier_worse.ablation_model}`에서 나타났다 "
        f"({stable_brier_worse.delta_brier_vs_stable_no_landcover:.5f}).",
        "- LRT는 변수군이 전체 적합도에 기여하는지 보는 통계 검정이고, OOF 성능 변화는 예측 일반화 관점의 비교이다.",
        "- 두 기준이 항상 같은 결론을 주지는 않으므로, 최종 보고서에서는 둘을 구분해서 쓴다.",
        "- 최종 해석 모델은 토지피복 범주를 그대로 넣은 전체 모델보다, 토지피복을 제거하거나 더 큰 범주로 축약한 안정 모델을 우선 검토해야 한다.",
        "- 캐나다지수는 Step13에서 부호 역전 가능성이 있었으므로, Step15 공선성 확인 전까지 해석을 보류한다.",
        "",
        "## 8. 산출물",
        "",
        f"- `{spec_path.relative_to(ROOT)}`",
        f"- `{overall_path.relative_to(ROOT)}`",
        f"- `{fold_path.relative_to(ROOT)}`",
        f"- `{fit_path.relative_to(ROOT)}`",
        f"- `{lrt_path.relative_to(ROOT)}`",
        f"- `{dropped_path.relative_to(ROOT)}`",
        f"- `{pred_path.relative_to(ROOT)}`",
        f"- `{auprc_plot.relative_to(ROOT)}`",
        f"- `{brier_plot.relative_to(ROOT)}`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    full_features, categorical, mapping = load_stage12()
    ablation_sets, spec_table = make_ablation_sets(full_features, mapping)
    _, dev = load_modeling_frame()
    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    outer = outer.loc[outer[ID_COL].isin(set(dev[ID_COL]))].copy()

    predictions, fold_metrics = run_oof(dev, outer, ablation_sets, categorical, mapping)
    overall = summarize_oof(predictions)

    fit_summary, dropped = fit_full_development_models(dev, ablation_sets, categorical, mapping)
    fit_summary = fit_summary.merge(
        spec_table[["ablation_model", "removed_group", "description", "reference_family"]],
        on="ablation_model",
        how="left",
    )
    lrt = likelihood_ratio_tests(fit_summary)

    write_outputs(spec_table, predictions, fold_metrics, overall, fit_summary, lrt, dropped)
    print("Stage14 완료")
    print(f"요약: {OUTPUT_DIR / 'stage14_ablation_tests_summary.md'}")
    print(f"OOF 성능표: {TABLE_DIR / 'stage14_ablation_metrics.csv'}")


if __name__ == "__main__":
    main()
