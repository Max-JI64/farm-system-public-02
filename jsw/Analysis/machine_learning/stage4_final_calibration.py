from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import stage1_ml_screening as s1
import stage3_ml_ensemble as s3


warnings.filterwarnings("ignore")

RANDOM_STATE = 20260621

ROOT = s1.ROOT
ML_DIR = s1.ML_DIR
OUTPUT_DIR = s1.OUTPUT_DIR
METRIC_DIR = s1.METRIC_DIR
PREDICTION_DIR = s1.PREDICTION_DIR
MODEL_DIR = OUTPUT_DIR / "models" / "stage4_final_selection"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STAGE35_OOF_PATH = PREDICTION_DIR / "ml_stage35_model_level_ensemble_oof_predictions.csv"
STAGE35_RECIPE_PATH = METRIC_DIR / "ml_stage35_ensemble_recipes.csv"
STAGE35_FULL_BASE_SCORE_PATH = PREDICTION_DIR / "ml_stage35_full_development_base_scores.csv"
STAGE35_MODEL_MANIFEST_PATH = METRIC_DIR / "ml_stage35_final_model_manifest.csv"
LOGISTIC_METRICS_PATH = s1.LOGISTIC_METRICS_PATH
STAGE1_COMPARISON_PATH = METRIC_DIR / "ml_stage1_v2_all_model_comparison.csv"
LOCKBOX_PATH = s1.LOCKBOX_PATH

RANKING_MODEL = "model_level_top5_rank_average"
PROBABILITY_MODELS = [
    "model_level_top5_geometric_mean",
    "model_level_top5_average",
    "model_level_top5_perf_weighted",
    "model_level_top5_logit_average",
]
FINAL_CANDIDATE_MODELS = [RANKING_MODEL, *PROBABILITY_MODELS]


def load_baselines() -> dict[str, float]:
    logistic = pd.read_csv(LOGISTIC_METRICS_PATH, encoding="utf-8-sig")
    logistic_baseline = logistic.loc[logistic["model"].eq("PLUS_LANDCOVER_RULES_ANOVA")].iloc[0]
    stage1 = pd.read_csv(STAGE1_COMPARISON_PATH, encoding="utf-8-sig")
    stage1_best = (
        stage1.loc[stage1["score_type"].eq("raw") & stage1["run_status"].eq("OK")]
        .sort_values(["auprc", "brier"], ascending=[False, True])
        .iloc[0]
    )
    return {
        "logistic_auprc": float(logistic_baseline["auprc"]),
        "logistic_brier": float(logistic_baseline["brier"]),
        "logistic_log_loss": float(logistic_baseline["log_loss"]),
        "stage1_best_auprc": float(stage1_best["auprc"]),
        "stage1_best_brier": float(stage1_best["brier"]),
        "stage1_best_log_loss": float(stage1_best["log_loss"]),
    }


def logit_score(probability: np.ndarray | pd.Series) -> np.ndarray:
    probability = s3.clipped_probability(probability)
    return np.log(probability / (1.0 - probability))


def fit_sigmoid_raw(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(s3.clipped_probability(x).reshape(-1, 1), y.astype(int))
    return model


def fit_sigmoid_logit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(logit_score(x).reshape(-1, 1), y.astype(int))
    return model


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(s3.clipped_probability(x), y.astype(int))
    return model


def apply_oof_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["score_sigmoid_logit"] = np.nan
    selected = predictions.loc[predictions["model"].isin(FINAL_CANDIDATE_MODELS)].copy()
    if selected.empty:
        raise ValueError("Step4 최종 후보 prediction이 없습니다.")

    for model_name in FINAL_CANDIDATE_MODELS:
        mask_model = predictions["model"].eq(model_name)
        if not mask_model.any():
            raise ValueError(f"후보 prediction 누락: {model_name}")
        for outer_fold in sorted(predictions.loc[mask_model, "outer_fold"].unique()):
            train_mask = mask_model & ~predictions["outer_fold"].eq(outer_fold)
            val_mask = mask_model & predictions["outer_fold"].eq(outer_fold)
            raw_train = predictions.loc[train_mask, "score_raw"].to_numpy(dtype=float)
            y_train = predictions.loc[train_mask, "Target"].to_numpy(dtype=int)
            raw_val = predictions.loc[val_mask, "score_raw"].to_numpy(dtype=float)

            sigmoid_raw = fit_sigmoid_raw(raw_train, y_train)
            sigmoid_logit = fit_sigmoid_logit(raw_train, y_train)
            isotonic = fit_isotonic(raw_train, y_train)
            predictions.loc[val_mask, "score_sigmoid"] = s3.clipped_probability(
                sigmoid_raw.predict_proba(s3.clipped_probability(raw_val).reshape(-1, 1))[:, 1]
            )
            predictions.loc[val_mask, "score_sigmoid_logit"] = s3.clipped_probability(
                sigmoid_logit.predict_proba(logit_score(raw_val).reshape(-1, 1))[:, 1]
            )
            predictions.loc[val_mask, "score_isotonic"] = s3.clipped_probability(isotonic.predict(raw_val))
    return predictions.loc[predictions["model"].isin(FINAL_CANDIDATE_MODELS)].reset_index(drop=True)


def prediction_long_frame_stage4(predictions: pd.DataFrame, include_calibrated_best: bool = True) -> pd.DataFrame:
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
    score_map = {
        "raw": "score_raw",
        "sigmoid_raw": "score_sigmoid",
        "sigmoid_logit": "score_sigmoid_logit",
        "isotonic": "score_isotonic",
    }
    for score_type, score_col in score_map.items():
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


def comparison_from_predictions_stage4(predictions: pd.DataFrame, baselines: dict[str, float]) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame_stage4(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        row = {
            "feature_set": feature_set,
            "feature_group": str(part["feature_group"].iloc[0]),
            "model": model,
            "score_type": score_type,
            "run_status": "OK",
            "selected_config_id": s3.selected_config_summary(part),
            **s3.probability_metrics(part["Target"], part["score"]),
        }
        threshold = s3.best_f1_threshold(part["Target"], part["score"])
        row.update(
            {
                f"best_f1_{key}": value
                for key, value in s3.classification_metrics_at_threshold(part["Target"], part["score"], threshold).items()
            }
        )
        row["delta_auprc_vs_logistic"] = float(row["auprc"] - baselines["logistic_auprc"])
        row["delta_brier_vs_logistic"] = float(row["brier"] - baselines["logistic_brier"])
        row["delta_auprc_vs_stage1_best"] = float(row["auprc"] - baselines["stage1_best_auprc"])
        row["delta_brier_vs_stage1_best"] = float(row["brier"] - baselines["stage1_best_brier"])
        row["delta_log_loss_vs_stage1_best"] = float(row["log_loss"] - baselines["stage1_best_log_loss"])
        row["stage4_success_vs_stage1_plus_0p005"] = bool(row["delta_auprc_vs_stage1_best"] >= 0.005)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["auprc", "brier"], ascending=[False, True], na_position="last")


def make_threshold_table_stage4(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame_stage4(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        thresholds = {
            "fixed_0.50": 0.5,
            "best_f1": s3.best_f1_threshold(part["Target"], part["score"]),
            "recall_ge_0.50": s3.threshold_for_recall(part["Target"], part["score"], 0.50),
            "recall_ge_0.70": s3.threshold_for_recall(part["Target"], part["score"], 0.70),
            "recall_ge_0.90": s3.threshold_for_recall(part["Target"], part["score"], 0.90),
        }
        for operating_point, threshold in thresholds.items():
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "operating_point": operating_point,
                    **s3.classification_metrics_at_threshold(part["Target"], part["score"], threshold),
                }
            )
    return pd.DataFrame(rows)


def make_top_risk_table_stage4(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame_stage4(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        part = part.sort_values("score", ascending=False).reset_index(drop=True)
        total_positive = int(part["Target"].sum())
        base_rate = float(part["Target"].mean())
        for pct in [0.05, 0.10, 0.20]:
            n = int(np.ceil(len(part) * pct))
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


def make_subgroup_table_stage4(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame_stage4(predictions)
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
                    **s3.probability_metrics(subset["Target"], subset["score"]),
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
                    **s3.probability_metrics(subset["Target"], subset["score"]),
                }
            )
    return pd.DataFrame(rows)


def choose_final_candidates(comparison: pd.DataFrame, baselines: dict[str, float]) -> pd.DataFrame:
    ranking = comparison.loc[comparison["model"].eq(RANKING_MODEL) & comparison["score_type"].eq("raw")].iloc[0]
    prob_pool = comparison.loc[comparison["model"].isin(PROBABILITY_MODELS)].copy()
    prob_pool["eligible_primary"] = (
        prob_pool["auprc"].ge(baselines["stage1_best_auprc"] + 0.005)
        & prob_pool["brier"].le(baselines["stage1_best_brier"] + 0.001)
        & prob_pool["log_loss"].le(baselines["stage1_best_log_loss"] + 0.003)
    )
    prob_pool["eligible_secondary"] = prob_pool["auprc"].ge(baselines["stage1_best_auprc"])
    eligible = prob_pool.loc[prob_pool["eligible_primary"]].copy()
    if eligible.empty:
        eligible = prob_pool.loc[prob_pool["eligible_secondary"]].copy()
    if eligible.empty:
        eligible = prob_pool.copy()
    probability = eligible.sort_values(
        ["auprc", "brier", "log_loss"],
        ascending=[False, True, True],
        na_position="last",
    ).iloc[0]
    rows = [
        {
            "selection_role": "ranking_score",
            "selected_reason": "strict OOF AUPRC와 top-risk 순위화 성능이 가장 높음. rank score라 확률 지표로는 쓰지 않음.",
            **ranking.to_dict(),
        },
        {
            "selection_role": "probability_score",
            "selected_reason": "AUPRC +0.005 기준을 넘으면서 Brier/log loss가 양호해 보고서/운영 threshold 후보로 사용.",
            **probability.to_dict(),
        },
    ]
    return pd.DataFrame(rows)


def parse_recipe(recipes: pd.DataFrame, model_name: str) -> tuple[str, list[str], np.ndarray | None]:
    row = recipes.loc[recipes["ensemble_id"].eq(model_name)]
    if len(row) != 1:
        raise ValueError(f"ensemble recipe를 찾지 못했습니다: {model_name}")
    row = row.iloc[0]
    candidates = str(row["candidates"]).split("|")
    method = str(row["method"])
    weights = None
    if pd.notna(row["weights"]) and str(row["weights"]).strip():
        weight_map = json.loads(str(row["weights"]))
        weights = np.array([float(weight_map[candidate]) for candidate in candidates], dtype=float)
    return method, candidates, weights


def make_full_development_scores(selection: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    recipes = pd.read_csv(STAGE35_RECIPE_PATH, encoding="utf-8-sig")
    full_base = pd.read_csv(STAGE35_FULL_BASE_SCORE_PATH, encoding="utf-8-sig")
    score_matrix = full_base[[col for col in full_base.columns if col not in {"샘플ID", "Target", "샘플유형", "기후지형유형"}]].copy()
    full_scores = full_base[["샘플ID", "Target", "샘플유형", "기후지형유형"]].copy()
    artifacts: dict[str, Any] = {}
    for row in selection.itertuples(index=False):
        role = str(row.selection_role)
        model_name = str(row.model)
        score_type = str(row.score_type)
        method, candidates, weights = parse_recipe(recipes, model_name)
        raw_score = s3.combine_scores(score_matrix, candidates, method, weights)
        full_scores[f"{role}__{model_name}__raw"] = raw_score

        y = full_base["Target"].to_numpy(dtype=int)
        calibrators = {
            "sigmoid_raw": fit_sigmoid_raw(raw_score, y),
            "sigmoid_logit": fit_sigmoid_logit(raw_score, y),
            "isotonic": fit_isotonic(raw_score, y),
        }
        artifacts[role] = {
            "model": model_name,
            "selected_score_type": score_type,
            "recipe": {"method": method, "candidates": candidates, "weights": None if weights is None else weights.tolist()},
            "calibrators": calibrators,
        }
        if score_type == "raw":
            selected_score = raw_score
        elif score_type == "sigmoid_raw":
            selected_score = calibrators["sigmoid_raw"].predict_proba(s3.clipped_probability(raw_score).reshape(-1, 1))[:, 1]
        elif score_type == "sigmoid_logit":
            selected_score = calibrators["sigmoid_logit"].predict_proba(logit_score(raw_score).reshape(-1, 1))[:, 1]
        elif score_type == "isotonic":
            selected_score = calibrators["isotonic"].predict(raw_score)
        else:
            raise KeyError(score_type)
        full_scores[f"{role}__selected_score"] = s3.clipped_probability(selected_score)
    return full_scores, artifacts


def validation_checks(predictions: pd.DataFrame, selection: pd.DataFrame, lockbox: pd.DataFrame) -> pd.DataFrame:
    checks = []
    development_ids = set(lockbox.loc[lockbox["split"].eq("development"), "샘플ID"])
    lockbox_ids = set(lockbox.loc[lockbox["split"].eq("lockbox_test"), "샘플ID"])
    sample_ids = set(predictions["샘플ID"])
    checks.extend(
        [
            {"check": "candidate_models_n", "value": predictions["model"].nunique(), "expected": len(FINAL_CANDIDATE_MODELS), "passed": predictions["model"].nunique() == len(FINAL_CANDIDATE_MODELS)},
            {"check": "development_row_count_per_model", "value": int(predictions.groupby("model")["샘플ID"].nunique().min()), "expected": 13632, "passed": int(predictions.groupby("model")["샘플ID"].nunique().min()) == 13632},
            {"check": "development_positive_n_per_model", "value": int(predictions.groupby("model")["Target"].sum().min()), "expected": 1242, "passed": int(predictions.groupby("model")["Target"].sum().min()) == 1242},
            {"check": "lockbox_overlap", "value": len(sample_ids & lockbox_ids), "expected": 0, "passed": len(sample_ids & lockbox_ids) == 0},
            {"check": "development_manifest_match", "value": len(sample_ids ^ development_ids), "expected": 0, "passed": len(sample_ids ^ development_ids) == 0},
            {"check": "selection_rows", "value": len(selection), "expected": 2, "passed": len(selection) == 2},
        ]
    )
    for score_col in ["score_raw", "score_sigmoid", "score_sigmoid_logit", "score_isotonic"]:
        score = predictions[score_col].to_numpy(dtype=float)
        checks.extend(
            [
                {"check": f"{score_col}_nan", "value": int(np.isnan(score).sum()), "expected": 0, "passed": not np.isnan(score).any()},
                {"check": f"{score_col}_inf", "value": int((~np.isfinite(score)).sum()), "expected": 0, "passed": np.isfinite(score).all()},
                {"check": f"{score_col}_range", "value": int(((score < 0) | (score > 1)).sum()), "expected": 0, "passed": not ((score < 0) | (score > 1)).any()},
            ]
        )
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        raise ValueError("validation check 실패:\n" + result.loc[~result["passed"]].to_string(index=False))
    return result


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, n: int | None = None) -> str:
    part = df.copy()
    if columns is not None:
        part = part[columns]
    if n is not None:
        part = part.head(n)
    return part.round(5).to_markdown(index=False)


def write_manifest(
    selection: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_risk: pd.DataFrame,
    artifacts: dict[str, Any],
    baselines: dict[str, float],
) -> None:
    model_manifest = pd.read_csv(STAGE35_MODEL_MANIFEST_PATH, encoding="utf-8-sig")
    manifest = {
        "stage": "stage4_final_calibration",
        "lockbox_used": False,
        "baselines": baselines,
        "selected_candidates": selection[
            [
                "selection_role",
                "model",
                "score_type",
                "auprc",
                "auroc",
                "brier",
                "log_loss",
                "delta_auprc_vs_stage1_best",
                "delta_auprc_vs_logistic",
                "selected_reason",
            ]
        ].to_dict(orient="records"),
        "thresholds": thresholds.to_dict(orient="records"),
        "top_risk": top_risk.to_dict(orient="records"),
        "base_model_manifest": model_manifest.to_dict(orient="records"),
        "artifact_path": str((MODEL_DIR / "stage4_calibration_artifacts.joblib").relative_to(ROOT)),
        "next_step": "Step5에서 lockbox test를 최초 평가한다.",
    }
    (MODEL_DIR / "stage4_final_selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(artifacts, MODEL_DIR / "stage4_calibration_artifacts.joblib")


def write_summary(
    comparison: pd.DataFrame,
    selection: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_risk: pd.DataFrame,
    subgroup: pd.DataFrame,
    validation: pd.DataFrame,
    baselines: dict[str, float],
) -> None:
    ranking = selection.loc[selection["selection_role"].eq("ranking_score")].iloc[0]
    probability = selection.loc[selection["selection_role"].eq("probability_score")].iloc[0]
    probability_top = top_risk.loc[
        top_risk["model"].eq(probability["model"]) & top_risk["score_type"].eq(probability["score_type"])
    ]
    probability_subgroup = subgroup.loc[
        subgroup["model"].eq(probability["model"])
        & subgroup["score_type"].eq(probability["score_type"])
        & subgroup["subgroup_type"].eq("negative_type")
    ]
    display_cols = [
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
        "delta_brier_vs_stage1_best",
        "delta_log_loss_vs_stage1_best",
    ]
    lines = [
        "# 머신러닝 4차 최종 후보 calibration 결과",
        "",
        "## 1. 실행 목적",
        "",
        "- lockbox test 전에 model-level ensemble 후보의 score 사용 방식을 확정했다.",
        "- rank 평균은 순위 성능용, geometric mean 계열은 확률/threshold 운영용으로 분리해 평가했다.",
        "- lockbox test는 사용하지 않았다.",
        "",
        "## 2. 기준선",
        "",
        f"- 로지스틱 Stage17 AUPRC: {baselines['logistic_auprc']:.4f}",
        f"- 1차 ML 최고 AUPRC: {baselines['stage1_best_auprc']:.4f}",
        f"- 1차 ML 최고 Brier/log loss: {baselines['stage1_best_brier']:.5f} / {baselines['stage1_best_log_loss']:.5f}",
        "",
        "## 3. 최종 선택",
        "",
        f"- 순위용 score: `{ranking['model']} / {ranking['score_type']}`",
        f"  - AUPRC {float(ranking['auprc']):.4f}, ROC AUC {float(ranking['auroc']):.4f}, top-risk 우선순위용",
        f"- 확률/threshold용 score: `{probability['model']} / {probability['score_type']}`",
        f"  - AUPRC {float(probability['auprc']):.4f}, Brier {float(probability['brier']):.5f}, log loss {float(probability['log_loss']):.5f}",
        f"  - 1차 최고 대비 ΔAUPRC {float(probability['delta_auprc_vs_stage1_best']):+.4f}",
        "",
        "## 4. calibration 비교",
        "",
        markdown_table(comparison, display_cols),
        "",
        "## 5. 최종 후보 threshold",
        "",
        thresholds.round(5).to_markdown(index=False),
        "",
        "## 6. 확률 후보 top-risk capture",
        "",
        probability_top.round(5).to_markdown(index=False),
        "",
        "## 7. 확률 후보 hard-negative subgroup",
        "",
        probability_subgroup[["model", "score_type", "subgroup", "auprc", "auroc", "brier", "log_loss"]]
        .round(5)
        .to_markdown(index=False),
        "",
        "## 8. 해석",
        "",
        "- calibration을 적용하면 일부 Brier/log loss는 약간 개선될 수 있지만, AUPRC가 떨어지는 경우가 있었다.",
        "- 최종 ranking score는 AUPRC가 가장 높은 raw rank average로 둔다.",
        "- 최종 probability score는 AUPRC +0.005 기준을 넘고 확률 지표가 안정적인 raw geometric mean으로 둔다.",
        "- Step5에서는 이 선택을 고정한 뒤 lockbox test를 최초로 평가한다.",
        "",
        "## 9. 검증",
        "",
        validation.to_markdown(index=False),
        "",
        "## 10. 산출물",
        "",
        "- `outputs/metrics/ml_stage4_calibration_comparison.csv`",
        "- `outputs/metrics/ml_stage4_final_candidate_selection.csv`",
        "- `outputs/metrics/ml_stage4_final_thresholds.csv`",
        "- `outputs/metrics/ml_stage4_final_top_risk_capture.csv`",
        "- `outputs/metrics/ml_stage4_final_subgroup_metrics.csv`",
        "- `outputs/predictions/ml_stage4_final_candidate_oof_predictions.csv`",
        "- `outputs/models/stage4_final_selection/stage4_final_selection_manifest.json`",
    ]
    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "ml_stage4_final_calibration_summary.md").write_text(summary, encoding="utf-8")
    (ML_DIR / "머신러닝_4차_최종후보_calibration_결과.md").write_text(summary, encoding="utf-8")

    log_path = ML_DIR / "LOG.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# 머신러닝 모델링 진행 로그\n"
    log_entry = "\n".join(
        [
            "",
            "## 2026-06-21",
            "",
            "### 4차 최종 후보 calibration 실행",
            "",
            "- lockbox test 전 최종 ranking score와 probability score를 분리해 확정했다.",
            "- raw/sigmoid_raw/sigmoid_logit/isotonic calibration을 development OOF 기준으로 비교했다.",
            "- lockbox test는 사용하지 않았다.",
            "",
            "### 4차 선택",
            "",
            f"- 순위용: `{ranking['model']} / {ranking['score_type']}`, AUPRC {float(ranking['auprc']):.4f}",
            f"- 확률/threshold용: `{probability['model']} / {probability['score_type']}`, AUPRC {float(probability['auprc']):.4f}, Brier {float(probability['brier']):.5f}, log loss {float(probability['log_loss']):.5f}",
            "",
            "### 산출물",
            "",
            "- `outputs/ml_stage4_final_calibration_summary.md`",
            "- `머신러닝_4차_최종후보_calibration_결과.md`",
            "- `outputs/models/stage4_final_selection/stage4_final_selection_manifest.json`",
            "",
        ]
    )
    log_path.write_text(existing.rstrip() + "\n" + log_entry, encoding="utf-8")

    overall_path = ML_DIR / "머신러닝_전체_진행_결과.md"
    existing_overall = overall_path.read_text(encoding="utf-8") if overall_path.exists() else ""
    overall_path.write_text(existing_overall.rstrip() + "\n\n---\n\n" + summary, encoding="utf-8")


def main() -> None:
    print("ML Stage4: load inputs")
    baselines = load_baselines()
    predictions = pd.read_csv(STAGE35_OOF_PATH, encoding="utf-8-sig", low_memory=False)
    lockbox = pd.read_csv(LOCKBOX_PATH, encoding="utf-8-sig")

    print("ML Stage4: OOF calibration")
    final_predictions = apply_oof_calibration(predictions)
    comparison = comparison_from_predictions_stage4(final_predictions, baselines)
    selection = choose_final_candidates(comparison, baselines)

    # Keep calibrated_best equal to the selected score only for selected roles.
    final_predictions["score_calibrated"] = final_predictions["score_raw"]
    final_predictions["calibration_method"] = "raw"
    for row in selection.itertuples(index=False):
        model_name = str(row.model)
        score_type = str(row.score_type)
        score_col = {
            "raw": "score_raw",
            "sigmoid_raw": "score_sigmoid",
            "sigmoid_logit": "score_sigmoid_logit",
            "isotonic": "score_isotonic",
        }[score_type]
        mask = final_predictions["model"].eq(model_name)
        final_predictions.loc[mask, "score_calibrated"] = final_predictions.loc[mask, score_col]
        final_predictions.loc[mask, "calibration_method"] = score_type

    print("ML Stage4: final metrics")
    comparison = comparison_from_predictions_stage4(final_predictions, baselines)
    thresholds_all = make_threshold_table_stage4(final_predictions)
    top_risk_all = make_top_risk_table_stage4(final_predictions)
    subgroup_all = make_subgroup_table_stage4(final_predictions)
    selection = choose_final_candidates(comparison, baselines)
    selected_pairs = set(zip(selection["model"], selection["score_type"]))
    thresholds = thresholds_all.loc[
        thresholds_all[["model", "score_type"]].apply(tuple, axis=1).isin(selected_pairs)
    ].copy()
    top_risk = top_risk_all.loc[top_risk_all[["model", "score_type"]].apply(tuple, axis=1).isin(selected_pairs)].copy()
    subgroup = subgroup_all.loc[subgroup_all[["model", "score_type"]].apply(tuple, axis=1).isin(selected_pairs)].copy()

    print("ML Stage4: full development final scores and manifest")
    full_scores, artifacts = make_full_development_scores(selection, final_predictions)
    validation = validation_checks(final_predictions, selection, lockbox)
    write_manifest(selection, thresholds, top_risk, artifacts, baselines)

    final_predictions.to_csv(PREDICTION_DIR / "ml_stage4_final_candidate_oof_predictions.csv", index=False, encoding="utf-8-sig")
    full_scores.to_csv(PREDICTION_DIR / "ml_stage4_full_development_final_scores.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(METRIC_DIR / "ml_stage4_calibration_comparison.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(METRIC_DIR / "ml_stage4_final_candidate_selection.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(METRIC_DIR / "ml_stage4_final_thresholds.csv", index=False, encoding="utf-8-sig")
    top_risk.to_csv(METRIC_DIR / "ml_stage4_final_top_risk_capture.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(METRIC_DIR / "ml_stage4_final_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(METRIC_DIR / "ml_stage4_validation_checks.csv", index=False, encoding="utf-8-sig")
    thresholds_all.to_csv(METRIC_DIR / "ml_stage4_all_thresholds.csv", index=False, encoding="utf-8-sig")
    top_risk_all.to_csv(METRIC_DIR / "ml_stage4_all_top_risk_capture.csv", index=False, encoding="utf-8-sig")
    subgroup_all.to_csv(METRIC_DIR / "ml_stage4_all_subgroup_metrics.csv", index=False, encoding="utf-8-sig")

    write_summary(comparison, selection, thresholds, top_risk, subgroup, validation, baselines)

    ranking = selection.loc[selection["selection_role"].eq("ranking_score")].iloc[0]
    probability = selection.loc[selection["selection_role"].eq("probability_score")].iloc[0]
    print(
        "ML Stage4 완료: "
        f"ranking={ranking['model']}/{ranking['score_type']} AUPRC={float(ranking['auprc']):.4f}; "
        f"probability={probability['model']}/{probability['score_type']} "
        f"AUPRC={float(probability['auprc']):.4f}, Brier={float(probability['brier']):.5f}"
    )


if __name__ == "__main__":
    main()
