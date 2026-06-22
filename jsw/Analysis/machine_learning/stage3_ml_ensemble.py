from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

RANDOM_STATE = 20260621
MAX_SINGLE_WEIGHT = 0.70
N_WEIGHT_TRIALS = 2500
TOP_CALIBRATION_ENSEMBLES = 3


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "학습데이터" / "학습데이터_로지스틱_D2D3.csv").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
LOGISTIC_DIR = ROOT / "jsw" / "Analysis" / "logistic"
LOGISTIC_OUTPUT_DIR = LOGISTIC_DIR / "outputs"
ML_DIR = ROOT / "jsw" / "Analysis" / "machine_learning"
OUTPUT_DIR = ML_DIR / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

for directory in [OUTPUT_DIR, METRIC_DIR, PREDICTION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

STAGE1_PREDICTION_PATH = PREDICTION_DIR / "ml_stage1_v2_oof_predictions.csv"
STAGE1_COMPARISON_PATH = METRIC_DIR / "ml_stage1_v2_all_model_comparison.csv"
STAGE1_TOP_RISK_PATH = METRIC_DIR / "ml_stage1_v2_top_risk_capture.csv"
STAGE1_SUBGROUP_PATH = METRIC_DIR / "ml_stage1_v2_subgroup_metrics.csv"
STAGE2_PREDICTION_PATH = PREDICTION_DIR / "ml_stage2_oof_predictions.csv"
STAGE2_COMPARISON_PATH = METRIC_DIR / "ml_stage2_model_comparison.csv"
LOGISTIC_METRICS_PATH = LOGISTIC_OUTPUT_DIR / "tables" / "stage17_ml_comparison_metrics.csv"
OUTER_PATH = LOGISTIC_OUTPUT_DIR / "splits" / "stage75_date_exposure_component_cv_outer_cv_manifest.csv"
INNER_PATH = LOGISTIC_OUTPUT_DIR / "splits" / "stage75_date_exposure_component_cv_inner_cv_manifest.csv"
LOCKBOX_PATH = LOGISTIC_OUTPUT_DIR / "splits" / "lockbox_manifest.csv"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    source: str
    feature_set: str
    feature_group: str
    model: str
    score_type: str
    role: str
    priority: int
    rationale: str


def clipped_probability(probability: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)


def logit(probability: np.ndarray | pd.Series) -> np.ndarray:
    probability = clipped_probability(probability)
    return np.log(probability / (1 - probability))


def sigmoid(score: np.ndarray | pd.Series) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    score = np.clip(score, -700, 700)
    return clipped_probability(1.0 / (1.0 + np.exp(-score)))


def probability_metrics(y_true, probability) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    probability = clipped_probability(probability)
    result: dict[str, float | int] = {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }
    result["auroc"] = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else np.nan
    return result


def threshold_curve(y_true, probability) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    probability = clipped_probability(probability)
    frame = pd.DataFrame({"y": y_true, "probability": probability})
    grouped = (
        frame.groupby("probability", as_index=False)
        .agg(n=("y", "size"), pos=("y", "sum"))
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    grouped["neg"] = grouped["n"] - grouped["pos"]
    grouped["tp"] = grouped["pos"].cumsum()
    grouped["fp"] = grouped["neg"].cumsum()
    total_pos = int(frame["y"].sum())
    total_neg = int((1 - frame["y"]).sum())
    grouped["fn"] = total_pos - grouped["tp"]
    grouped["tn"] = total_neg - grouped["fp"]
    grouped["selected_n"] = grouped["tp"] + grouped["fp"]
    grouped["precision"] = grouped["tp"] / grouped["selected_n"]
    grouped["recall"] = np.where(total_pos > 0, grouped["tp"] / total_pos, np.nan)
    grouped["specificity"] = np.where(total_neg > 0, grouped["tn"] / total_neg, np.nan)
    denominator = grouped["precision"] + grouped["recall"]
    grouped["f1"] = np.where(denominator > 0, 2 * grouped["precision"] * grouped["recall"] / denominator, 0.0)
    grouped["balanced_accuracy"] = (grouped["recall"] + grouped["specificity"]) / 2
    return grouped.rename(columns={"probability": "threshold"})


def best_f1_threshold(y_true, probability) -> float:
    curve = threshold_curve(y_true, probability).copy()
    curve["selected_rate"] = curve["selected_n"] / len(y_true)
    curve = curve.sort_values(
        ["f1", "balanced_accuracy", "selected_rate", "threshold"],
        ascending=[False, False, True, False],
    )
    return float(curve.iloc[0]["threshold"])


def threshold_for_recall(y_true, probability, target_recall: float) -> float:
    curve = threshold_curve(y_true, probability)
    candidates = curve.loc[curve["recall"] >= target_recall]
    if candidates.empty:
        return 0.0
    return float(candidates.iloc[0]["threshold"])


def classification_metrics_at_threshold(y_true, probability, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    pred = (clipped_probability(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "selected_n": int(pred.sum()),
        "selected_rate": float(pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
    }


def selected_config_summary(part: pd.DataFrame) -> str:
    values = sorted(part["selected_config_id"].dropna().astype(str).unique())
    return values[0] if len(values) == 1 else "|".join(values)


def prediction_long_frame(predictions: pd.DataFrame, include_calibrated_best: bool = True) -> pd.DataFrame:
    rows = []
    base_cols = [
        "샘플ID",
        "feature_set",
        "feature_group",
        "model",
        "outer_fold",
        "Target",
        "샘플유형",
        "기후지형유형",
        "selected_config_id",
        "run_status",
    ]
    for score_type, score_col in {"raw": "score_raw", "sigmoid": "score_sigmoid", "isotonic": "score_isotonic"}.items():
        if score_col not in predictions.columns:
            continue
        part = predictions.loc[predictions[score_col].notna(), base_cols + [score_col]].copy()
        if part.empty:
            continue
        part = part.rename(columns={score_col: "score"})
        part["score_type"] = score_type
        rows.append(part)
    if include_calibrated_best and "score_calibrated" in predictions.columns:
        part = predictions.loc[
            predictions["calibration_method"].ne("raw") & predictions["score_calibrated"].notna(),
            base_cols + ["score_calibrated"],
        ].copy()
        if not part.empty:
            part = part.rename(columns={"score_calibrated": "score"})
            part["score_type"] = "calibrated_best"
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def comparison_from_predictions(
    predictions: pd.DataFrame,
    logistic_auprc: float,
    logistic_brier: float,
    stage1_best_auprc: float,
) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        row = {
            "feature_set": feature_set,
            "feature_group": str(part["feature_group"].iloc[0]),
            "model": model,
            "score_type": score_type,
            "run_status": "OK",
            "selected_config_id": selected_config_summary(part),
            **probability_metrics(part["Target"], part["score"]),
        }
        threshold = best_f1_threshold(part["Target"], part["score"])
        row.update(
            {
                f"best_f1_{key}": value
                for key, value in classification_metrics_at_threshold(part["Target"], part["score"], threshold).items()
            }
        )
        row["delta_auprc_vs_logistic"] = float(row["auprc"] - logistic_auprc)
        row["delta_brier_vs_logistic"] = float(row["brier"] - logistic_brier)
        row["delta_auprc_vs_stage1_best"] = float(row["auprc"] - stage1_best_auprc)
        row["stage3_success_vs_stage1_plus_0p005"] = bool(row["delta_auprc_vs_stage1_best"] >= 0.005)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["auprc", "brier"], ascending=[False, True], na_position="last")


def make_threshold_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        thresholds = {
            "fixed_0.50": 0.5,
            "best_f1": best_f1_threshold(part["Target"], part["score"]),
            "recall_ge_0.50": threshold_for_recall(part["Target"], part["score"], 0.50),
            "recall_ge_0.70": threshold_for_recall(part["Target"], part["score"], 0.70),
            "recall_ge_0.90": threshold_for_recall(part["Target"], part["score"], 0.90),
        }
        for operating_point, threshold in thresholds.items():
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "operating_point": operating_point,
                    **classification_metrics_at_threshold(part["Target"], part["score"], threshold),
                }
            )
    return pd.DataFrame(rows)


def make_top_risk_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        part = part.sort_values("score", ascending=False).reset_index(drop=True)
        total_positive = int(part["Target"].sum())
        base_rate = float(part["Target"].mean())
        for pct in [0.05, 0.10, 0.20]:
            n = int(math.ceil(len(part) * pct))
            top = part.iloc[:n]
            precision = float(top["Target"].mean())
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "top_pct": pct,
                    "selected_n": n,
                    "selected_rate": float(n / len(part)),
                    "threshold_min": float(top["score"].min()),
                    "captured_positive_n": int(top["Target"].sum()),
                    "total_positive_n": total_positive,
                    "capture_rate_recall": float(top["Target"].sum() / total_positive) if total_positive else np.nan,
                    "precision": precision,
                    "base_positive_rate": base_rate,
                    "lift_vs_base": float(precision / base_rate) if base_rate else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_subgroup_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame(predictions)
    negative_types = [value for value in sorted(long["샘플유형"].dropna().unique()) if value != "Target_1"]
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        positives = part.loc[part["샘플유형"].eq("Target_1")]
        for negative_type in negative_types:
            subset = pd.concat([positives, part.loc[part["샘플유형"].eq(negative_type)]], ignore_index=True)
            if subset["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "subgroup_type": "negative_type",
                    "subgroup": negative_type,
                    **probability_metrics(subset["Target"], subset["score"]),
                }
            )
        for climate, subset in part.groupby("기후지형유형", observed=True):
            if subset["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "subgroup_type": "기후지형유형",
                    "subgroup": climate,
                    **probability_metrics(subset["Target"], subset["score"]),
                }
            )
    return pd.DataFrame(rows)


def candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            "s1_PL_HGB_raw",
            "stage1",
            "PLUS_LANDCOVER",
            "logistic_final_guided",
            "HistGradientBoosting",
            "raw",
            "primary",
            1,
            "1차 최고 모델, core anchor",
        ),
        CandidateSpec(
            "s1_RULES_HGB_raw",
            "stage1",
            "PLUS_LANDCOVER_RULES_ANOVA_PROXY",
            "logistic_final_guided",
            "HistGradientBoosting",
            "raw",
            "primary",
            2,
            "규칙 proxy 계열 대표",
        ),
        CandidateSpec(
            "s1_PL_LGBM_raw",
            "stage1",
            "PLUS_LANDCOVER",
            "logistic_final_guided",
            "LightGBM",
            "raw",
            "primary",
            3,
            "recall/ROC AUC 강점",
        ),
        CandidateSpec(
            "s1_RULES_RF_raw",
            "stage1",
            "PLUS_LANDCOVER_RULES_ANOVA_PROXY",
            "logistic_final_guided",
            "RandomForest",
            "raw",
            "primary",
            4,
            "bagging 다양성",
        ),
        CandidateSpec(
            "s1_RULES_XGB_raw",
            "stage1",
            "PLUS_LANDCOVER_RULES_ANOVA_PROXY",
            "logistic_final_guided",
            "XGBoost",
            "raw",
            "primary",
            5,
            "boosting 계열 다양성",
        ),
        CandidateSpec(
            "s1_PL_HGB_sigmoid",
            "stage1",
            "PLUS_LANDCOVER",
            "logistic_final_guided",
            "HistGradientBoosting",
            "sigmoid",
            "auxiliary",
            6,
            "1차 최고 모델의 sigmoid 보정 score",
        ),
        CandidateSpec(
            "s1_PL_LGBM_sigmoid",
            "stage1",
            "PLUS_LANDCOVER",
            "logistic_final_guided",
            "LightGBM",
            "sigmoid",
            "auxiliary",
            7,
            "LightGBM 보정 score",
        ),
        CandidateSpec(
            "s1_M1_RF_raw",
            "stage1",
            "M1",
            "diagnostic_m123",
            "RandomForest",
            "raw",
            "auxiliary",
            8,
            "M1 진단 기준선과 다른 피처 구조",
        ),
        CandidateSpec(
            "s1_STAGE7_HGB_raw",
            "stage1",
            "STAGE7_RECOMMENDED",
            "logistic_final_guided",
            "HistGradientBoosting",
            "raw",
            "auxiliary",
            9,
            "토지피복 제외 Stage7 구조",
        ),
        CandidateSpec(
            "s1_RULES_ET_raw",
            "stage1",
            "PLUS_LANDCOVER_RULES_ANOVA_PROXY",
            "logistic_final_guided",
            "ExtraTrees",
            "raw",
            "auxiliary",
            10,
            "ExtraTrees bagging 다양성",
        ),
        CandidateSpec(
            "s2_RULES_HGB_raw",
            "stage2",
            "PLUS_LANDCOVER_RULES_ANOVA_PROXY",
            "logistic_final_guided",
            "HistGradientBoosting",
            "raw",
            "diagnostic_only",
            11,
            "2차 Optuna 결과, 다양성 확인용",
        ),
        CandidateSpec(
            "s2_PL_XGB_raw",
            "stage2",
            "PLUS_LANDCOVER",
            "logistic_final_guided",
            "XGBoost",
            "raw",
            "diagnostic_only",
            12,
            "2차 Optuna 결과, 다양성 확인용",
        ),
    ]


def score_column(score_type: str) -> str:
    mapping = {
        "raw": "score_raw",
        "sigmoid": "score_sigmoid",
        "isotonic": "score_isotonic",
        "calibrated_best": "score_calibrated",
    }
    return mapping[score_type]


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
    float,
]:
    stage1_predictions = pd.read_csv(STAGE1_PREDICTION_PATH, encoding="utf-8-sig", low_memory=False)
    stage1_comparison = pd.read_csv(STAGE1_COMPARISON_PATH, encoding="utf-8-sig")
    stage2_predictions = pd.read_csv(STAGE2_PREDICTION_PATH, encoding="utf-8-sig", low_memory=False)
    stage2_comparison = pd.read_csv(STAGE2_COMPARISON_PATH, encoding="utf-8-sig")
    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    inner = pd.read_csv(INNER_PATH, encoding="utf-8-sig")
    lockbox = pd.read_csv(LOCKBOX_PATH, encoding="utf-8-sig")
    logistic = pd.read_csv(LOGISTIC_METRICS_PATH, encoding="utf-8-sig")

    logistic_baseline = logistic.loc[logistic["model"].eq("PLUS_LANDCOVER_RULES_ANOVA")].iloc[0]
    stage1_best = (
        stage1_comparison.loc[stage1_comparison["score_type"].eq("raw") & stage1_comparison["run_status"].eq("OK")]
        .sort_values(["auprc", "brier"], ascending=[False, True])
        .iloc[0]
    )
    return (
        stage1_predictions,
        stage1_comparison,
        stage2_predictions,
        stage2_comparison,
        outer,
        inner,
        lockbox,
        float(logistic_baseline["auprc"]),
        float(logistic_baseline["brier"]),
        float(stage1_best["auprc"]),
    )


def extract_candidate(
    spec: CandidateSpec,
    source_predictions: dict[str, pd.DataFrame],
    source_comparisons: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = source_predictions[spec.source]
    score_col = score_column(spec.score_type)
    mask = (
        predictions["feature_set"].eq(spec.feature_set)
        & predictions["model"].eq(spec.model)
        & predictions["run_status"].eq("OK")
        & predictions[score_col].notna()
    )
    part = predictions.loc[
        mask,
        [
            "샘플ID",
            "outer_fold",
            "Target",
            "샘플유형",
            "기후지형유형",
            "selected_config_id",
            score_col,
        ],
    ].copy()
    if part.empty:
        raise ValueError(f"후보 score를 찾지 못했습니다: {spec}")
    if part["샘플ID"].duplicated().any():
        raise ValueError(f"후보 샘플ID 중복: {spec.candidate_id}")
    part = part.rename(columns={score_col: "score"})
    part["score"] = clipped_probability(part["score"])
    metrics = probability_metrics(part["Target"], part["score"])

    comparison = source_comparisons[spec.source]
    comp_mask = (
        comparison["feature_set"].eq(spec.feature_set)
        & comparison["model"].eq(spec.model)
        & comparison["score_type"].eq(spec.score_type)
        & comparison["run_status"].eq("OK")
    )
    source_auprc = np.nan
    source_auroc = np.nan
    if comp_mask.any():
        source_auprc = float(comparison.loc[comp_mask, "auprc"].iloc[0])
        source_auroc = float(comparison.loc[comp_mask, "auroc"].iloc[0])

    registry_row = {
        "candidate_id": spec.candidate_id,
        "source": spec.source,
        "feature_set": spec.feature_set,
        "feature_group": spec.feature_group,
        "model": spec.model,
        "score_type": spec.score_type,
        "role": spec.role,
        "priority": spec.priority,
        "rationale": spec.rationale,
        "selected_config_id": selected_config_summary(part),
        "source_table_auprc": source_auprc,
        "source_table_auroc": source_auroc,
        **metrics,
    }
    return part, registry_row


def build_candidate_pool(
    stage1_predictions: pd.DataFrame,
    stage1_comparison: pd.DataFrame,
    stage2_predictions: pd.DataFrame,
    stage2_comparison: pd.DataFrame,
    outer: pd.DataFrame,
    lockbox: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_predictions = {"stage1": stage1_predictions, "stage2": stage2_predictions}
    source_comparisons = {"stage1": stage1_comparison, "stage2": stage2_comparison}
    metadata: pd.DataFrame | None = None
    score_columns: dict[str, pd.Series] = {}
    registry_rows = []
    development_ids = set(lockbox.loc[lockbox["split"].eq("development"), "샘플ID"])
    lockbox_ids = set(lockbox.loc[lockbox["split"].eq("lockbox_test"), "샘플ID"])

    for spec in candidate_specs():
        part, registry_row = extract_candidate(spec, source_predictions, source_comparisons)
        ids = set(part["샘플ID"])
        if ids != set(outer["샘플ID"]):
            raise ValueError(f"outer manifest와 후보 샘플ID 불일치: {spec.candidate_id}")
        if ids != development_ids:
            raise ValueError(f"development manifest와 후보 샘플ID 불일치: {spec.candidate_id}")
        if ids & lockbox_ids:
            raise ValueError(f"lockbox 샘플이 후보에 포함됨: {spec.candidate_id}")

        ordered = part.sort_values(["outer_fold", "샘플ID"]).reset_index(drop=True)
        if metadata is None:
            metadata = ordered[["샘플ID", "outer_fold", "Target", "샘플유형", "기후지형유형"]].copy()
        else:
            check_cols = ["샘플ID", "outer_fold", "Target", "샘플유형", "기후지형유형"]
            if not metadata[check_cols].equals(ordered[check_cols]):
                raise ValueError(f"후보 metadata 정합성 불일치: {spec.candidate_id}")
        score_columns[spec.candidate_id] = ordered["score"].reset_index(drop=True)
        registry_rows.append(registry_row)

    if metadata is None:
        raise ValueError("candidate pool이 비어 있습니다.")
    score_matrix = pd.DataFrame(score_columns)
    if score_matrix.isna().any().any() or not np.isfinite(score_matrix.to_numpy()).all():
        raise ValueError("candidate score에 NaN/inf가 있습니다.")
    if (score_matrix.lt(0) | score_matrix.gt(1)).any().any():
        raise ValueError("candidate score가 [0, 1] 범위를 벗어났습니다.")

    registry = pd.DataFrame(registry_rows).sort_values(["priority", "candidate_id"])
    return metadata, score_matrix, registry


def candidate_correlations(score_matrix: pd.DataFrame) -> pd.DataFrame:
    pearson = score_matrix.corr(method="pearson")
    spearman = score_matrix.corr(method="spearman")
    rows = []
    for i, left in enumerate(score_matrix.columns):
        for right in score_matrix.columns[i + 1 :]:
            rows.append(
                {
                    "candidate_1": left,
                    "candidate_2": right,
                    "pearson": float(pearson.loc[left, right]),
                    "spearman": float(spearman.loc[left, right]),
                }
            )
    return pd.DataFrame(rows).sort_values(["pearson", "spearman"], ascending=[False, False])


def candidate_fold_metrics(metadata: pd.DataFrame, score_matrix: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    role_by_candidate = registry.set_index("candidate_id")["role"].to_dict()
    for candidate_id in score_matrix.columns:
        for outer_fold, index in metadata.groupby("outer_fold", observed=True).groups.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "role": role_by_candidate.get(candidate_id, ""),
                    "outer_fold": int(outer_fold),
                    **probability_metrics(metadata.loc[index, "Target"], score_matrix.loc[index, candidate_id]),
                }
            )
    return pd.DataFrame(rows)


def candidate_subgroup_metrics(metadata: pd.DataFrame, score_matrix: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    role_by_candidate = registry.set_index("candidate_id")["role"].to_dict()
    negative_types = [value for value in sorted(metadata["샘플유형"].dropna().unique()) if value != "Target_1"]
    for candidate_id in score_matrix.columns:
        candidate = pd.DataFrame({"score": score_matrix[candidate_id], **metadata.to_dict(orient="series")})
        positives = candidate.loc[candidate["샘플유형"].eq("Target_1")]
        for negative_type in negative_types:
            subset = pd.concat([positives, candidate.loc[candidate["샘플유형"].eq(negative_type)]], ignore_index=True)
            if subset["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "role": role_by_candidate.get(candidate_id, ""),
                    "subgroup_type": "negative_type",
                    "subgroup": negative_type,
                    **probability_metrics(subset["Target"], subset["score"]),
                }
            )
        for climate, subset in candidate.groupby("기후지형유형", observed=True):
            if subset["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "role": role_by_candidate.get(candidate_id, ""),
                    "subgroup_type": "기후지형유형",
                    "subgroup": climate,
                    **probability_metrics(subset["Target"], subset["score"]),
                }
            )
    return pd.DataFrame(rows)


def pct_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def combine_scores(score_matrix: pd.DataFrame, candidates: list[str], method: str, weights: np.ndarray | None = None) -> np.ndarray:
    scores = score_matrix[candidates].to_numpy(dtype=float)
    if method == "average":
        combined = scores.mean(axis=1)
    elif method == "weighted":
        if weights is None:
            raise ValueError("weighted method에는 weights가 필요합니다.")
        combined = np.average(scores, axis=1, weights=weights)
    elif method == "geometric_mean":
        combined = np.exp(np.log(clipped_probability(scores)).mean(axis=1))
    elif method == "logit_average":
        combined = sigmoid(logit(scores).mean(axis=1))
    elif method == "rank_average":
        rank_frame = score_matrix[candidates].apply(pct_rank, axis=0)
        combined = rank_frame.mean(axis=1).to_numpy(dtype=float)
    else:
        raise KeyError(method)
    return clipped_probability(combined)


def make_prediction_rows(
    metadata: pd.DataFrame,
    ensemble_kind: str,
    ensemble_id: str,
    selected_config_id: str,
    score: np.ndarray,
) -> pd.DataFrame:
    rows = metadata.copy()
    rows.insert(1, "feature_set", ensemble_kind)
    rows.insert(2, "feature_group", "stage3_ensemble")
    rows.insert(3, "model", ensemble_id)
    rows["selected_config_id"] = selected_config_id
    rows["score_raw"] = clipped_probability(score)
    rows["score_sigmoid"] = np.nan
    rows["score_isotonic"] = np.nan
    rows["score_calibrated"] = rows["score_raw"]
    rows["calibration_method"] = "raw"
    rows["run_status"] = "OK"
    return rows[
        [
            "샘플ID",
            "feature_set",
            "feature_group",
            "model",
            "outer_fold",
            "Target",
            "샘플유형",
            "기후지형유형",
            "selected_config_id",
            "score_raw",
            "score_sigmoid",
            "score_isotonic",
            "score_calibrated",
            "calibration_method",
            "run_status",
        ]
    ]


def performance_weights(candidates: list[str], registry: pd.DataFrame, mode: str = "linear") -> np.ndarray:
    metric = registry.set_index("candidate_id").loc[candidates, "auprc"].to_numpy(dtype=float)
    if mode == "sqrt_delta":
        baseline = float(registry["auprc"].min())
        metric = np.sqrt(np.maximum(metric - baseline, 1e-6))
    metric = np.maximum(metric, 1e-6)
    weights = metric / metric.sum()
    if weights.max() > MAX_SINGLE_WEIGHT:
        weights = np.minimum(weights, MAX_SINGLE_WEIGHT)
        weights = weights / weights.sum()
    return weights


def simple_ensembles(metadata: pd.DataFrame, score_matrix: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recipes: list[dict[str, Any]] = [
        {
            "ensemble_id": "simple_top2_PL_HGB_LGBM_avg",
            "method": "average",
            "candidates": ["s1_PL_HGB_raw", "s1_PL_LGBM_raw"],
        },
        {
            "ensemble_id": "simple_top2_PL_HGB_RULES_HGB_avg",
            "method": "average",
            "candidates": ["s1_PL_HGB_raw", "s1_RULES_HGB_raw"],
        },
        {
            "ensemble_id": "simple_top3_primary_avg",
            "method": "average",
            "candidates": ["s1_PL_HGB_raw", "s1_RULES_HGB_raw", "s1_PL_LGBM_raw"],
        },
        {
            "ensemble_id": "simple_top5_primary_avg",
            "method": "average",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
            ],
        },
        {
            "ensemble_id": "simple_top5_rank_avg",
            "method": "rank_average",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
            ],
        },
        {
            "ensemble_id": "simple_top5_logit_avg",
            "method": "logit_average",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
            ],
        },
        {
            "ensemble_id": "simple_top5_geometric_mean",
            "method": "geometric_mean",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
            ],
        },
        {
            "ensemble_id": "simple_diverse_stage7_m1_avg",
            "method": "average",
            "candidates": ["s1_PL_HGB_raw", "s1_PL_LGBM_raw", "s1_STAGE7_HGB_raw", "s1_M1_RF_raw"],
        },
        {
            "ensemble_id": "simple_top5_plus_stage2_diag_avg",
            "method": "average",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_XGB_raw",
                "s2_RULES_HGB_raw",
            ],
        },
    ]
    fixed_weight_recipes = [
        {
            "ensemble_id": "fixed_anchor70_LGBM30",
            "method": "weighted",
            "candidates": ["s1_PL_HGB_raw", "s1_PL_LGBM_raw"],
            "weights": np.array([0.70, 0.30]),
        },
        {
            "ensemble_id": "fixed_anchor70_RULES30",
            "method": "weighted",
            "candidates": ["s1_PL_HGB_raw", "s1_RULES_HGB_raw"],
            "weights": np.array([0.70, 0.30]),
        },
        {
            "ensemble_id": "fixed_anchor50_LGBM30_XGB20",
            "method": "weighted",
            "candidates": ["s1_PL_HGB_raw", "s1_PL_LGBM_raw", "s1_RULES_XGB_raw"],
            "weights": np.array([0.50, 0.30, 0.20]),
        },
        {
            "ensemble_id": "fixed_top5_perf_weighted",
            "method": "weighted",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
            ],
            "weights": None,
            "weight_mode": "linear",
        },
        {
            "ensemble_id": "fixed_top5_sqrt_delta_weighted",
            "method": "weighted",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
            ],
            "weights": None,
            "weight_mode": "sqrt_delta",
        },
    ]
    predictions = []
    recipe_rows = []
    for recipe in [*recipes, *fixed_weight_recipes]:
        candidates = recipe["candidates"]
        weights = recipe.get("weights")
        if recipe["method"] == "weighted" and weights is None:
            weights = performance_weights(candidates, registry, recipe.get("weight_mode", "linear"))
        score = combine_scores(score_matrix, candidates, recipe["method"], weights)
        config = {
            "method": recipe["method"],
            "candidates": candidates,
            "weights": None if weights is None else {candidate: float(weight) for candidate, weight in zip(candidates, weights)},
        }
        predictions.append(
            make_prediction_rows(
                metadata,
                "ENSEMBLE_SIMPLE",
                recipe["ensemble_id"],
                json.dumps(config, ensure_ascii=False, sort_keys=True),
                score,
            )
        )
        recipe_rows.append({"ensemble_id": recipe["ensemble_id"], **config})
    recipe_table = pd.DataFrame(recipe_rows)
    return pd.concat(predictions, ignore_index=True), recipe_table


def top10_capture(y_true: pd.Series | np.ndarray, score: np.ndarray) -> float:
    frame = pd.DataFrame({"Target": np.asarray(y_true).astype(int), "score": clipped_probability(score)})
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    n = int(math.ceil(len(frame) * 0.10))
    positive_n = int(frame["Target"].sum())
    return float(frame.iloc[:n]["Target"].sum() / positive_n) if positive_n else np.nan


def subgroup_0a_auprc(metadata: pd.DataFrame, score: np.ndarray, index: np.ndarray | list[int]) -> float:
    subset = metadata.iloc[index].copy()
    subset["score"] = score
    part = subset.loc[subset["샘플유형"].isin(["Target_1", "Target_0A"])]
    if part["Target"].nunique() < 2:
        return np.nan
    return float(average_precision_score(part["Target"], part["score"]))


def random_weight_candidates(n_candidates: int, rng: np.random.Generator, n_trials: int) -> list[np.ndarray]:
    weights = [np.full(n_candidates, 1.0 / n_candidates)]
    for anchor in range(n_candidates):
        weight = np.full(n_candidates, (1.0 - MAX_SINGLE_WEIGHT) / (n_candidates - 1))
        weight[anchor] = MAX_SINGLE_WEIGHT
        weights.append(weight)
    while len(weights) < n_trials:
        batch = rng.dirichlet(np.ones(n_candidates), size=n_trials)
        batch = batch[batch.max(axis=1) <= MAX_SINGLE_WEIGHT + 1e-12]
        for row in batch:
            weights.append(row)
            if len(weights) >= n_trials:
                break
    return weights[:n_trials]


def search_weights_for_fold(
    metadata: pd.DataFrame,
    score_matrix: pd.DataFrame,
    candidates: list[str],
    train_index: np.ndarray,
    outer_fold: int,
    ensemble_id: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE + int(outer_fold) * 1000 + len(candidates))
    x_train = score_matrix.loc[train_index, candidates].to_numpy(dtype=float)
    y_train = metadata.loc[train_index, "Target"].to_numpy(dtype=int)
    weight_candidates = random_weight_candidates(len(candidates), rng, N_WEIGHT_TRIALS)
    rows = []
    for trial, weight in enumerate(weight_candidates):
        score = clipped_probability(x_train @ weight)
        row = {
            "ensemble_id": ensemble_id,
            "outer_fold": int(outer_fold),
            "trial": int(trial),
            "auprc": float(average_precision_score(y_train, score)),
            "brier": float(brier_score_loss(y_train, score)),
            "top10_capture": top10_capture(y_train, score),
            "subgroup_0A_auprc": subgroup_0a_auprc(metadata, score, train_index),
        }
        for candidate, value in zip(candidates, weight):
            row[f"w__{candidate}"] = float(value)
        rows.append(row)
    trials = pd.DataFrame(rows)
    best = trials.sort_values(
        ["auprc", "subgroup_0A_auprc", "top10_capture", "brier"],
        ascending=[False, False, False, True],
        na_position="last",
    ).iloc[0]
    weights = np.array([best[f"w__{candidate}"] for candidate in candidates], dtype=float)
    return weights, trials


def weighted_ensembles(metadata: pd.DataFrame, score_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = [
        {
            "ensemble_id": "weighted_top3_primary",
            "candidates": ["s1_PL_HGB_raw", "s1_RULES_HGB_raw", "s1_PL_LGBM_raw"],
        },
        {
            "ensemble_id": "weighted_top5_primary",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
            ],
        },
        {
            "ensemble_id": "weighted_top5_plus_stage7",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
                "s1_STAGE7_HGB_raw",
            ],
        },
        {
            "ensemble_id": "weighted_diverse_m1_stage7",
            "candidates": ["s1_PL_HGB_raw", "s1_PL_LGBM_raw", "s1_RULES_XGB_raw", "s1_STAGE7_HGB_raw", "s1_M1_RF_raw"],
        },
    ]
    predictions = []
    selected_rows = []
    trial_tables = []
    for group in groups:
        ensemble_id = group["ensemble_id"]
        candidates = group["candidates"]
        oof_score = np.full(len(metadata), np.nan, dtype=float)
        for outer_fold, val_index in metadata.groupby("outer_fold", observed=True).groups.items():
            val_index = np.asarray(list(val_index), dtype=int)
            train_index = np.asarray(metadata.index.difference(val_index), dtype=int)
            weights, trials = search_weights_for_fold(metadata, score_matrix, candidates, train_index, int(outer_fold), ensemble_id)
            trial_tables.append(trials)
            x_val = score_matrix.loc[val_index, candidates].to_numpy(dtype=float)
            oof_score[val_index] = clipped_probability(x_val @ weights)
            selected = {
                "ensemble_id": ensemble_id,
                "outer_fold": int(outer_fold),
                "candidates": "|".join(candidates),
            }
            for candidate, value in zip(candidates, weights):
                selected[f"w__{candidate}"] = float(value)
            selected_rows.append(selected)
        if np.isnan(oof_score).any():
            raise ValueError(f"weighted ensemble OOF 결측: {ensemble_id}")
        predictions.append(
            make_prediction_rows(
                metadata,
                "ENSEMBLE_WEIGHTED",
                ensemble_id,
                json.dumps({"method": "outer_train_weight_search", "candidates": candidates}, ensure_ascii=False, sort_keys=True),
                oof_score,
            )
        )
    return pd.concat(predictions, ignore_index=True), pd.concat(trial_tables, ignore_index=True), pd.DataFrame(selected_rows)


def transformed_features(score_matrix: pd.DataFrame, candidates: list[str], transform: str) -> pd.DataFrame:
    part = score_matrix[candidates].copy()
    if transform == "raw":
        return part
    if transform == "logit":
        return pd.DataFrame(logit(part.to_numpy(dtype=float)), columns=candidates, index=part.index)
    if transform == "rank":
        return part.apply(pct_rank, axis=0)
    raise KeyError(transform)


def meta_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for transform in ["raw", "logit", "rank"]:
        for class_weight in [None, "balanced"]:
            for c_value in [0.03, 0.1, 0.3, 1.0, 3.0]:
                configs.append(
                    {
                        "config_id": f"logreg_l2_{transform}_cw{class_weight or 'none'}_C{c_value}",
                        "model": "LogisticRegression_L2",
                        "transform": transform,
                        "class_weight": class_weight,
                        "C": c_value,
                    }
                )
    for transform in ["raw", "logit", "rank"]:
        for c_value, l1_ratio in [(0.1, 0.25), (0.3, 0.50), (1.0, 0.50)]:
            configs.append(
                {
                    "config_id": f"logreg_elastic_{transform}_C{c_value}_l1{l1_ratio}",
                    "model": "LogisticRegression_ElasticNet",
                    "transform": transform,
                    "class_weight": None,
                    "C": c_value,
                    "l1_ratio": l1_ratio,
                }
            )
    for transform in ["raw", "logit", "rank"]:
        for alpha in [0.1, 1.0, 10.0]:
            configs.append(
                {
                    "config_id": f"ridge_{transform}_alpha{alpha}",
                    "model": "RidgeClassifier",
                    "transform": transform,
                    "alpha": alpha,
                }
            )
    return configs


def build_meta_model(config: dict[str, Any]) -> Pipeline:
    if config["model"] == "LogisticRegression_L2":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=float(config["C"]),
                        solver="lbfgs",
                        max_iter=2000,
                        class_weight=config["class_weight"],
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if config["model"] == "LogisticRegression_ElasticNet":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        C=float(config["C"]),
                        l1_ratio=float(config["l1_ratio"]),
                        solver="saga",
                        max_iter=5000,
                        class_weight=config["class_weight"],
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if config["model"] == "RidgeClassifier":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeClassifier(alpha=float(config["alpha"]), class_weight="balanced")),
            ]
        )
    raise KeyError(config["model"])


def predict_meta_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        return clipped_probability(model.predict_proba(x)[:, 1])
    return sigmoid(model.decision_function(x))


def tune_stacking_config(
    metadata: pd.DataFrame,
    score_matrix: pd.DataFrame,
    inner: pd.DataFrame,
    outer_fold: int,
    candidates: list[str],
    configs: list[dict[str, Any]],
    ensemble_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    inner_part = inner.loc[inner["outer_fold"].eq(outer_fold)].set_index("샘플ID")
    id_to_index = metadata.reset_index().set_index("샘플ID")["index"].to_dict()
    rows = []
    y = metadata["Target"].to_numpy(dtype=int)
    for config in configs:
        x_all = transformed_features(score_matrix, candidates, config["transform"])
        scores = []
        for inner_fold in sorted(inner_part["inner_fold"].unique()):
            inner_val_ids = inner_part.index[inner_part["inner_fold"].eq(inner_fold)].tolist()
            inner_train_ids = inner_part.index[~inner_part["inner_fold"].eq(inner_fold)].tolist()
            train_index = [id_to_index[sample_id] for sample_id in inner_train_ids]
            val_index = [id_to_index[sample_id] for sample_id in inner_val_ids]
            model = build_meta_model(config)
            model.fit(x_all.iloc[train_index], y[train_index])
            prob = predict_meta_probability(model, x_all.iloc[val_index])
            score = float(average_precision_score(y[val_index], prob))
            scores.append(score)
            rows.append(
                {
                    "ensemble_id": ensemble_id,
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "config_id": config["config_id"],
                    "meta_model": config["model"],
                    "transform": config["transform"],
                    "auprc": score,
                }
            )
    tuning = pd.DataFrame(rows)
    summary = (
        tuning.groupby("config_id", as_index=False)
        .agg(mean_auprc=("auprc", "mean"), std_auprc=("auprc", "std"))
        .sort_values(["mean_auprc", "std_auprc", "config_id"], ascending=[False, True, True])
    )
    best_id = str(summary.iloc[0]["config_id"])
    best_config = [config for config in configs if config["config_id"] == best_id][0]
    return best_config, tuning


def stacking_ensembles(metadata: pd.DataFrame, score_matrix: pd.DataFrame, inner: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stack_groups = [
        {
            "ensemble_id": "stack_top5_primary",
            "candidates": [
                "s1_PL_HGB_raw",
                "s1_RULES_HGB_raw",
                "s1_PL_LGBM_raw",
                "s1_RULES_RF_raw",
                "s1_RULES_XGB_raw",
            ],
        },
        {
            "ensemble_id": "stack_top3_primary",
            "candidates": ["s1_PL_HGB_raw", "s1_RULES_HGB_raw", "s1_PL_LGBM_raw"],
        },
    ]
    predictions = []
    tuning_tables = []
    selected_rows = []
    configs = meta_configs()
    y = metadata["Target"].to_numpy(dtype=int)
    for group in stack_groups:
        ensemble_id = group["ensemble_id"]
        candidates = group["candidates"]
        oof_score = np.full(len(metadata), np.nan, dtype=float)
        for outer_fold, val_index in metadata.groupby("outer_fold", observed=True).groups.items():
            val_index = np.asarray(list(val_index), dtype=int)
            train_index = np.asarray(metadata.index.difference(val_index), dtype=int)
            best_config, tuning = tune_stacking_config(
                metadata,
                score_matrix,
                inner,
                int(outer_fold),
                candidates,
                configs,
                ensemble_id,
            )
            tuning_tables.append(tuning)
            x_all = transformed_features(score_matrix, candidates, best_config["transform"])
            model = build_meta_model(best_config)
            model.fit(x_all.iloc[train_index], y[train_index])
            oof_score[val_index] = predict_meta_probability(model, x_all.iloc[val_index])
            selected_rows.append(
                {
                    "ensemble_id": ensemble_id,
                    "outer_fold": int(outer_fold),
                    "candidates": "|".join(candidates),
                    **best_config,
                }
            )
        if np.isnan(oof_score).any():
            raise ValueError(f"stacking OOF 결측: {ensemble_id}")
        predictions.append(
            make_prediction_rows(
                metadata,
                "ENSEMBLE_STACKING",
                ensemble_id,
                json.dumps({"method": "outer_train_stacking", "candidates": candidates}, ensure_ascii=False, sort_keys=True),
                oof_score,
            )
        )
    return pd.concat(predictions, ignore_index=True), pd.concat(tuning_tables, ignore_index=True), pd.DataFrame(selected_rows)


def calibrate_top_ensembles(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = predictions.copy()
    top = comparison.loc[comparison["score_type"].eq("raw")].head(TOP_CALIBRATION_ENSEMBLES)
    calibration_rows = []
    id_to_index = metadata.reset_index().set_index("샘플ID")["index"].to_dict()
    y = metadata["Target"].to_numpy(dtype=int)
    for row in top.itertuples(index=False):
        feature_set = str(row.feature_set)
        model_name = str(row.model)
        mask_model = predictions["feature_set"].eq(feature_set) & predictions["model"].eq(model_name)
        model_part = predictions.loc[mask_model, ["샘플ID", "outer_fold", "score_raw"]].copy()
        raw_by_sample = model_part.set_index("샘플ID")["score_raw"]
        sigmoid_score = pd.Series(index=model_part["샘플ID"], dtype=float)
        isotonic_score = pd.Series(index=model_part["샘플ID"], dtype=float)
        for outer_fold in sorted(metadata["outer_fold"].unique()):
            val_ids = metadata.loc[metadata["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
            train_ids = metadata.loc[~metadata["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
            train_index = [id_to_index[sample_id] for sample_id in train_ids]
            raw_train = clipped_probability(raw_by_sample.loc[train_ids].to_numpy())
            raw_val = clipped_probability(raw_by_sample.loc[val_ids].to_numpy())
            sigmoid_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
            sigmoid_model.fit(raw_train.reshape(-1, 1), y[train_index])
            isotonic_model = IsotonicRegression(out_of_bounds="clip")
            isotonic_model.fit(raw_train, y[train_index])
            sigmoid_score.loc[val_ids] = clipped_probability(sigmoid_model.predict_proba(raw_val.reshape(-1, 1))[:, 1])
            isotonic_score.loc[val_ids] = clipped_probability(isotonic_model.predict(raw_val))

        predictions.loc[mask_model, "score_sigmoid"] = predictions.loc[mask_model, "샘플ID"].map(sigmoid_score)
        predictions.loc[mask_model, "score_isotonic"] = predictions.loc[mask_model, "샘플ID"].map(isotonic_score)
        y_ordered = predictions.loc[mask_model, "Target"]
        sigmoid_metrics = probability_metrics(y_ordered, predictions.loc[mask_model, "score_sigmoid"])
        isotonic_metrics = probability_metrics(y_ordered, predictions.loc[mask_model, "score_isotonic"])
        if (isotonic_metrics["brier"], isotonic_metrics["log_loss"]) < (sigmoid_metrics["brier"], sigmoid_metrics["log_loss"]):
            best_method = "isotonic"
            best_metrics = isotonic_metrics
        else:
            best_method = "sigmoid"
            best_metrics = sigmoid_metrics
        predictions.loc[mask_model, "score_calibrated"] = predictions.loc[mask_model, f"score_{best_method}"]
        predictions.loc[mask_model, "calibration_method"] = best_method
        calibration_rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "best_calibration_method": best_method,
                "sigmoid_brier": sigmoid_metrics["brier"],
                "sigmoid_log_loss": sigmoid_metrics["log_loss"],
                "sigmoid_auprc": sigmoid_metrics["auprc"],
                "isotonic_brier": isotonic_metrics["brier"],
                "isotonic_log_loss": isotonic_metrics["log_loss"],
                "isotonic_auprc": isotonic_metrics["auprc"],
                "best_calibration_brier": best_metrics["brier"],
                "best_calibration_log_loss": best_metrics["log_loss"],
            }
        )
    return predictions, pd.DataFrame(calibration_rows)


def validation_checks(predictions: pd.DataFrame, metadata: pd.DataFrame, lockbox: pd.DataFrame) -> pd.DataFrame:
    checks = []
    lockbox_ids = set(lockbox.loc[lockbox["split"].eq("lockbox_test"), "샘플ID"])
    development_ids = set(lockbox.loc[lockbox["split"].eq("development"), "샘플ID"])
    base_ids = set(metadata["샘플ID"])
    checks.append({"check": "development_row_count", "value": len(metadata), "expected": 13632, "passed": len(metadata) == 13632})
    checks.append({"check": "development_positive_n", "value": int(metadata["Target"].sum()), "expected": 1242, "passed": int(metadata["Target"].sum()) == 1242})
    checks.append({"check": "sample_id_missing", "value": int(metadata["샘플ID"].isna().sum()), "expected": 0, "passed": metadata["샘플ID"].isna().sum() == 0})
    checks.append({"check": "sample_id_duplicates", "value": int(metadata["샘플ID"].duplicated().sum()), "expected": 0, "passed": metadata["샘플ID"].duplicated().sum() == 0})
    checks.append({"check": "lockbox_overlap", "value": len(base_ids & lockbox_ids), "expected": 0, "passed": len(base_ids & lockbox_ids) == 0})
    checks.append(
        {
            "check": "development_manifest_match",
            "value": len(base_ids ^ development_ids),
            "expected": 0,
            "passed": len(base_ids ^ development_ids) == 0,
        }
    )
    raw = predictions.loc[predictions["score_raw"].notna()]
    checks.append({"check": "ensemble_raw_nan", "value": int(raw["score_raw"].isna().sum()), "expected": 0, "passed": raw["score_raw"].isna().sum() == 0})
    checks.append(
        {
            "check": "ensemble_raw_inf",
            "value": int((~np.isfinite(raw["score_raw"].to_numpy(dtype=float))).sum()),
            "expected": 0,
            "passed": np.isfinite(raw["score_raw"].to_numpy(dtype=float)).all(),
        }
    )
    checks.append(
        {
            "check": "ensemble_raw_range_0_1",
            "value": int((raw["score_raw"].lt(0) | raw["score_raw"].gt(1)).sum()),
            "expected": 0,
            "passed": not (raw["score_raw"].lt(0) | raw["score_raw"].gt(1)).any(),
        }
    )
    for (feature_set, model), part in raw.groupby(["feature_set", "model"], observed=True):
        checks.append(
            {
                "check": f"oof_rows::{feature_set}/{model}",
                "value": len(part),
                "expected": 13632,
                "passed": len(part) == 13632,
            }
        )
        checks.append(
            {
                "check": f"oof_folds::{feature_set}/{model}",
                "value": int(part["outer_fold"].nunique()),
                "expected": 5,
                "passed": int(part["outer_fold"].nunique()) == 5,
            }
        )
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        raise ValueError("validation check 실패:\n" + result.loc[~result["passed"]].to_string(index=False))
    return result


def fold_metrics_for_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame(predictions)
    for (feature_set, model, score_type, outer_fold), part in long.groupby(
        ["feature_set", "model", "score_type", "outer_fold"], observed=True
    ):
        rows.append(
            {
                "feature_set": feature_set,
                "feature_group": str(part["feature_group"].iloc[0]),
                "model": model,
                "score_type": score_type,
                "outer_fold": int(outer_fold),
                **probability_metrics(part["Target"], part["score"]),
            }
        )
    return pd.DataFrame(rows)


def write_final_comparison(
    stage1_comparison: pd.DataFrame,
    stage2_comparison: pd.DataFrame,
    stage3_comparison: pd.DataFrame,
    logistic_auprc: float,
    logistic_brier: float,
    stage1_best_auprc: float,
) -> pd.DataFrame:
    logistic = pd.read_csv(LOGISTIC_METRICS_PATH, encoding="utf-8-sig")
    logistic_rows = []
    for row in logistic.itertuples(index=False):
        logistic_rows.append(
            {
                "model_family": "logistic",
                "stage": "stage17",
                "feature_set": getattr(row, "model"),
                "feature_group": "logistic_stage17",
                "model": getattr(row, "model"),
                "score_type": "oof",
                "run_status": "OK",
                "n": getattr(row, "n"),
                "positive_n": getattr(row, "positive_n"),
                "positive_rate": getattr(row, "positive_rate"),
                "auprc": getattr(row, "auprc"),
                "auroc": getattr(row, "auroc"),
                "brier": getattr(row, "brier"),
                "log_loss": getattr(row, "log_loss"),
                "best_f1_f1": getattr(row, "best_f1_f1"),
                "best_f1_precision": getattr(row, "best_f1_precision"),
                "best_f1_recall": getattr(row, "best_f1_recall"),
                "delta_auprc_vs_logistic": float(getattr(row, "auprc") - logistic_auprc),
                "delta_brier_vs_logistic": float(getattr(row, "brier") - logistic_brier),
                "delta_auprc_vs_stage1_best": float(getattr(row, "auprc") - stage1_best_auprc),
            }
        )
    stage1 = stage1_comparison.copy()
    stage1.insert(0, "stage", "stage1_v2")
    stage1.insert(0, "model_family", "machine_learning")
    stage2 = stage2_comparison.copy()
    stage2.insert(0, "stage", "stage2_optuna")
    stage2.insert(0, "model_family", "machine_learning")
    stage3 = stage3_comparison.copy()
    stage3.insert(0, "stage", "stage3_ensemble")
    stage3.insert(0, "model_family", "machine_learning")
    combined = pd.concat([pd.DataFrame(logistic_rows), stage1, stage2, stage3], ignore_index=True, sort=False)
    combined = combined.sort_values(["auprc", "brier"], ascending=[False, True], na_position="last")
    combined.to_csv(METRIC_DIR / "ml_stage3_final_comparison_with_stage1_stage2_logistic.csv", index=False, encoding="utf-8-sig")
    return combined


def to_markdown(df: pd.DataFrame, columns: list[str] | None = None, n: int | None = None) -> str:
    part = df.copy()
    if columns is not None:
        part = part[columns]
    if n is not None:
        part = part.head(n)
    return part.round(5).to_markdown(index=False)


def write_summary_and_logs(
    stage3_comparison: pd.DataFrame,
    simple_comparison: pd.DataFrame,
    weighted_comparison: pd.DataFrame,
    stacking_comparison: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
    top_risk: pd.DataFrame,
    subgroup: pd.DataFrame,
    validation: pd.DataFrame,
    logistic_auprc: float,
    stage1_best_auprc: float,
) -> None:
    raw = stage3_comparison.loc[stage3_comparison["score_type"].eq("raw")].copy()
    best = raw.iloc[0]
    probability_ready = raw.loc[
        raw["delta_auprc_vs_stage1_best"].ge(0.005) & raw["brier"].le(0.10) & raw["log_loss"].le(0.30)
    ].copy()
    report_candidate = probability_ready.iloc[0] if not probability_ready.empty else best
    best_top = top_risk.loc[
        top_risk["feature_set"].eq(best["feature_set"])
        & top_risk["model"].eq(best["model"])
        & top_risk["score_type"].eq("raw")
    ]
    best_subgroup = subgroup.loc[
        subgroup["feature_set"].eq(best["feature_set"])
        & subgroup["model"].eq(best["model"])
        & subgroup["score_type"].eq("raw")
        & subgroup["subgroup_type"].eq("negative_type")
    ]
    stage1_delta = float(best["auprc"] - stage1_best_auprc)
    logistic_delta = float(best["auprc"] - logistic_auprc)
    if stage1_delta >= 0.005:
        decision = "3차 앙상블 후보를 채택할 수 있다."
    elif stage1_delta > 0:
        decision = "1차 최고 대비 개선은 있으나 +0.005 기준에는 못 미치므로 보조 후보로만 유지한다."
    else:
        decision = "1차 최고를 넘지 못했으므로 현재 최종 후보는 1차 단독 모델로 유지한다."
    report_note = (
        f"`{report_candidate['feature_set']} / {report_candidate['model']} / raw`: "
        f"AUPRC {float(report_candidate['auprc']):.4f}, "
        f"Brier {float(report_candidate['brier']):.5f}, "
        f"log loss {float(report_candidate['log_loss']):.5f}, "
        f"1차 대비 ΔAUPRC {float(report_candidate['delta_auprc_vs_stage1_best']):+.4f}."
    )

    display_cols = [
        "feature_set",
        "model",
        "score_type",
        "auprc",
        "auroc",
        "brier",
        "log_loss",
        "best_f1_f1",
        "best_f1_precision",
        "best_f1_recall",
        "delta_auprc_vs_stage1_best",
        "delta_auprc_vs_logistic",
    ]
    lines = [
        "# 머신러닝 3차 앙상블 결과",
        "",
        "## 1. 실행 목적",
        "",
        "- 1차 strict OOF 최고 모델보다 높은 예측 성능을 만들기 위해 OOF 기반 앙상블을 수행했다.",
        "- 2차 Optuna 결과는 성능이 낮았으므로 주 후보가 아니라 다양성 진단 후보로만 사용했다.",
        "- lockbox test는 사용하지 않았다.",
        "",
        "## 2. 기준선",
        "",
        f"- 로지스틱 Stage17 기준선 AUPRC: {logistic_auprc:.4f}",
        f"- 1차 ML 최고 AUPRC: {stage1_best_auprc:.4f}",
        "",
        "## 3. 3차 최고 결과",
        "",
        f"- 최고 앙상블: `{best['feature_set']} / {best['model']} / raw`",
        f"- AUPRC {float(best['auprc']):.4f}, ROC AUC {float(best['auroc']):.4f}, Brier {float(best['brier']):.5f}, log loss {float(best['log_loss']):.5f}",
        f"- 1차 최고 대비 ΔAUPRC {stage1_delta:+.4f}, 로지스틱 대비 ΔAUPRC {logistic_delta:+.4f}",
        f"- 판단: {decision}",
        "",
        "주의: `simple_top5_rank_avg`는 rank-normalized score라 순위 지표(AUPRC/ROC/top-risk)에는 유효하지만 raw Brier/log loss를 확률 품질로 해석하기 어렵다. 확률 점수와 운영 threshold까지 같이 고려하면 다음 후보가 보고서용 1순위다.",
        "",
        f"- 보고서용 확률 후보: {report_note}",
        "",
        "## 4. 전체 앙상블 상위 결과",
        "",
        to_markdown(stage3_comparison, display_cols, 20),
        "",
        "## 5. 단순 앙상블 결과",
        "",
        to_markdown(simple_comparison, display_cols, 20),
        "",
        "## 6. 제한 weight search 결과",
        "",
        to_markdown(weighted_comparison, display_cols, 20),
        "",
        "## 7. stacking 결과",
        "",
        to_markdown(stacking_comparison, display_cols, 20),
        "",
        "## 8. calibration 결과",
        "",
        calibration_comparison.round(5).to_markdown(index=False) if not calibration_comparison.empty else "calibration 대상 없음",
        "",
        "## 9. 최고 앙상블 top-risk capture",
        "",
        best_top.round(5).to_markdown(index=False),
        "",
        "## 10. 최고 앙상블 hard-negative subgroup",
        "",
        best_subgroup[["feature_set", "model", "score_type", "subgroup", "auprc", "auroc", "brier", "log_loss"]]
        .round(5)
        .to_markdown(index=False),
        "",
        "## 11. 검증",
        "",
        validation.to_markdown(index=False),
        "",
        "## 12. 산출물",
        "",
        "- `outputs/metrics/ml_stage3_ensemble_candidate_registry.csv`",
        "- `outputs/metrics/ml_stage3_candidate_correlation.csv`",
        "- `outputs/metrics/ml_stage3_candidate_fold_metrics.csv`",
        "- `outputs/metrics/ml_stage3_simple_ensemble_comparison.csv`",
        "- `outputs/metrics/ml_stage3_weighted_ensemble_comparison.csv`",
        "- `outputs/metrics/ml_stage3_stacking_comparison.csv`",
        "- `outputs/metrics/ml_stage3_calibration_comparison.csv`",
        "- `outputs/metrics/ml_stage3_thresholds.csv`",
        "- `outputs/metrics/ml_stage3_top_risk_capture.csv`",
        "- `outputs/metrics/ml_stage3_subgroup_metrics.csv`",
        "- `outputs/metrics/ml_stage3_final_comparison_with_stage1_stage2_logistic.csv`",
        "- `outputs/predictions/ml_stage3_ensemble_oof_predictions.csv`",
    ]
    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "ml_stage3_ensemble_summary.md").write_text(summary, encoding="utf-8")
    (ML_DIR / "머신러닝_3차_앙상블_진행_결과.md").write_text(summary, encoding="utf-8")

    log_path = ML_DIR / "LOG.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# 머신러닝 모델링 진행 로그\n"
    log_entry = "\n".join(
        [
            "",
            "## 2026-06-21",
            "",
            "### 3차 앙상블 실행",
            "",
            "- 1차 OOF 예측을 주 재료로 단순 평균, 제한 weight search, stacking, calibration을 수행했다.",
            "- 2차 Optuna 후보는 성능이 낮아 주 후보가 아니라 진단 후보로만 사용했다.",
            "- lockbox test는 사용하지 않았다.",
            "",
            "### 3차 최고 결과",
            "",
            f"- `{best['feature_set']} / {best['model']} / raw`",
            f"- AUPRC {float(best['auprc']):.4f}, ROC AUC {float(best['auroc']):.4f}, Brier {float(best['brier']):.5f}, log loss {float(best['log_loss']):.5f}",
            f"- 1차 최고 대비 ΔAUPRC {stage1_delta:+.4f}, 로지스틱 대비 ΔAUPRC {logistic_delta:+.4f}",
            f"- 판단: {decision}",
            f"- 단, rank 평균은 확률 calibration 지표가 나쁘므로 보고서용 확률 후보는 {report_note}",
            "",
            "### 산출물",
            "",
            "- `outputs/ml_stage3_ensemble_summary.md`",
            "- `머신러닝_3차_앙상블_진행_결과.md`",
            "- `outputs/metrics/ml_stage3_final_comparison_with_stage1_stage2_logistic.csv`",
            "- `outputs/predictions/ml_stage3_ensemble_oof_predictions.csv`",
            "",
        ]
    )
    log_path.write_text(existing.rstrip() + "\n" + log_entry, encoding="utf-8")

    overall_path = ML_DIR / "머신러닝_전체_진행_결과.md"
    if overall_path.exists():
        overall_existing = overall_path.read_text(encoding="utf-8").rstrip()
        overall_path.write_text(overall_existing + "\n\n---\n\n" + summary, encoding="utf-8")
    else:
        overall_path.write_text(summary, encoding="utf-8")


def main() -> None:
    print("ML Stage3: load inputs")
    (
        stage1_predictions,
        stage1_comparison,
        stage2_predictions,
        stage2_comparison,
        outer,
        inner,
        lockbox,
        logistic_auprc,
        logistic_brier,
        stage1_best_auprc,
    ) = load_inputs()

    print("ML Stage3: candidate pool")
    metadata, score_matrix, registry = build_candidate_pool(
        stage1_predictions,
        stage1_comparison,
        stage2_predictions,
        stage2_comparison,
        outer,
        lockbox,
    )
    registry["delta_auprc_vs_stage1_best"] = registry["auprc"] - stage1_best_auprc
    registry["delta_auprc_vs_logistic"] = registry["auprc"] - logistic_auprc
    registry.to_csv(METRIC_DIR / "ml_stage3_ensemble_candidate_registry.csv", index=False, encoding="utf-8-sig")
    candidate_correlations(score_matrix).to_csv(METRIC_DIR / "ml_stage3_candidate_correlation.csv", index=False, encoding="utf-8-sig")
    candidate_fold_metrics(metadata, score_matrix, registry).to_csv(METRIC_DIR / "ml_stage3_candidate_fold_metrics.csv", index=False, encoding="utf-8-sig")
    candidate_subgroup_metrics(metadata, score_matrix, registry).to_csv(
        METRIC_DIR / "ml_stage3_candidate_subgroup_metrics.csv", index=False, encoding="utf-8-sig"
    )

    print("ML Stage3: simple ensembles")
    simple_predictions, simple_recipes = simple_ensembles(metadata, score_matrix, registry)
    simple_recipes.to_csv(METRIC_DIR / "ml_stage3_simple_ensemble_recipes.csv", index=False, encoding="utf-8-sig")
    simple_predictions.to_csv(PREDICTION_DIR / "ml_stage3_simple_ensemble_oof_predictions.csv", index=False, encoding="utf-8-sig")
    simple_comparison = comparison_from_predictions(simple_predictions, logistic_auprc, logistic_brier, stage1_best_auprc)
    simple_comparison.to_csv(METRIC_DIR / "ml_stage3_simple_ensemble_comparison.csv", index=False, encoding="utf-8-sig")

    print("ML Stage3: weighted ensembles")
    weighted_predictions, weight_trials, selected_weights = weighted_ensembles(metadata, score_matrix)
    weighted_predictions.to_csv(PREDICTION_DIR / "ml_stage3_weighted_ensemble_oof_predictions.csv", index=False, encoding="utf-8-sig")
    weight_trials.to_csv(METRIC_DIR / "ml_stage3_weight_search_trials.csv", index=False, encoding="utf-8-sig")
    selected_weights.to_csv(METRIC_DIR / "ml_stage3_selected_weights.csv", index=False, encoding="utf-8-sig")
    weighted_comparison = comparison_from_predictions(weighted_predictions, logistic_auprc, logistic_brier, stage1_best_auprc)
    weighted_comparison.to_csv(METRIC_DIR / "ml_stage3_weighted_ensemble_comparison.csv", index=False, encoding="utf-8-sig")

    print("ML Stage3: stacking ensembles")
    stacking_predictions, stacking_tuning, stacking_selected = stacking_ensembles(metadata, score_matrix, inner)
    stacking_predictions.to_csv(PREDICTION_DIR / "ml_stage3_stacking_oof_predictions.csv", index=False, encoding="utf-8-sig")
    stacking_tuning.to_csv(METRIC_DIR / "ml_stage3_stacking_inner_tuning.csv", index=False, encoding="utf-8-sig")
    stacking_selected.to_csv(METRIC_DIR / "ml_stage3_stacking_selected_configs.csv", index=False, encoding="utf-8-sig")
    stacking_comparison = comparison_from_predictions(stacking_predictions, logistic_auprc, logistic_brier, stage1_best_auprc)
    stacking_comparison.to_csv(METRIC_DIR / "ml_stage3_stacking_comparison.csv", index=False, encoding="utf-8-sig")

    print("ML Stage3: calibration")
    combined_predictions = pd.concat([simple_predictions, weighted_predictions, stacking_predictions], ignore_index=True)
    pre_calibration_comparison = comparison_from_predictions(combined_predictions, logistic_auprc, logistic_brier, stage1_best_auprc)
    combined_predictions, calibration_comparison = calibrate_top_ensembles(
        combined_predictions,
        pre_calibration_comparison,
        metadata,
    )
    calibration_comparison.to_csv(METRIC_DIR / "ml_stage3_calibration_comparison.csv", index=False, encoding="utf-8-sig")

    print("ML Stage3: final metrics")
    combined_predictions.to_csv(PREDICTION_DIR / "ml_stage3_ensemble_oof_predictions.csv", index=False, encoding="utf-8-sig")
    validation = validation_checks(combined_predictions, metadata, lockbox)
    validation.to_csv(METRIC_DIR / "ml_stage3_validation_checks.csv", index=False, encoding="utf-8-sig")
    stage3_comparison = comparison_from_predictions(combined_predictions, logistic_auprc, logistic_brier, stage1_best_auprc)
    thresholds = make_threshold_table(combined_predictions)
    top_risk = make_top_risk_table(combined_predictions)
    subgroup = make_subgroup_table(combined_predictions)
    fold_metrics_for_predictions(combined_predictions).to_csv(METRIC_DIR / "ml_stage3_ensemble_fold_metrics.csv", index=False, encoding="utf-8-sig")
    stage3_comparison.to_csv(METRIC_DIR / "ml_stage3_ensemble_model_comparison.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(METRIC_DIR / "ml_stage3_thresholds.csv", index=False, encoding="utf-8-sig")
    top_risk.to_csv(METRIC_DIR / "ml_stage3_top_risk_capture.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(METRIC_DIR / "ml_stage3_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    write_final_comparison(
        stage1_comparison,
        stage2_comparison,
        stage3_comparison,
        logistic_auprc,
        logistic_brier,
        stage1_best_auprc,
    )
    write_summary_and_logs(
        stage3_comparison,
        simple_comparison,
        weighted_comparison,
        stacking_comparison,
        calibration_comparison,
        top_risk,
        subgroup,
        validation,
        logistic_auprc,
        stage1_best_auprc,
    )

    best = stage3_comparison.loc[stage3_comparison["score_type"].eq("raw")].iloc[0]
    print(
        "ML Stage3 완료: "
        f"{best['feature_set']} / {best['model']} / raw "
        f"AUPRC={float(best['auprc']):.4f}, "
        f"Δstage1={float(best['delta_auprc_vs_stage1_best']):+.4f}, "
        f"Δlogistic={float(best['delta_auprc_vs_logistic']):+.4f}"
    )


if __name__ == "__main__":
    main()
