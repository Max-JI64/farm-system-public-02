from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import stage1_ml_screening as s1
import stage3_ml_ensemble as s3


warnings.filterwarnings("ignore")

RANDOM_STATE = 20260621

ROOT = s1.ROOT
ML_DIR = s1.ML_DIR
OUTPUT_DIR = s1.OUTPUT_DIR
METRIC_DIR = s1.METRIC_DIR
PREDICTION_DIR = s1.PREDICTION_DIR
MODEL_DIR = OUTPUT_DIR / "models" / "stage35_model_level_ensemble"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CONFIG_PATH = METRIC_DIR / "ml_stage1_v2_selected_configs.csv"
STAGE1_PREDICTION_PATH = PREDICTION_DIR / "ml_stage1_v2_oof_predictions.csv"
STAGE1_COMPARISON_PATH = METRIC_DIR / "ml_stage1_v2_all_model_comparison.csv"
STAGE2_COMPARISON_PATH = METRIC_DIR / "ml_stage2_model_comparison.csv"
LOGISTIC_METRICS_PATH = s1.LOGISTIC_METRICS_PATH
LOCKBOX_PATH = s1.LOCKBOX_PATH


@dataclass(frozen=True)
class BaseCandidate:
    candidate_id: str
    feature_set: str
    model: str
    role: str
    rationale: str


BASE_CANDIDATES = [
    BaseCandidate("PL_HGB", "PLUS_LANDCOVER", "HistGradientBoosting", "anchor", "1차 최고 anchor"),
    BaseCandidate("RULES_HGB", "PLUS_LANDCOVER_RULES_ANOVA_PROXY", "HistGradientBoosting", "primary", "규칙 proxy 계열"),
    BaseCandidate("PL_LGBM", "PLUS_LANDCOVER", "LightGBM", "primary", "ROC/recall 보조"),
    BaseCandidate("RULES_RF", "PLUS_LANDCOVER_RULES_ANOVA_PROXY", "RandomForest", "primary", "bagging 다양성"),
    BaseCandidate("RULES_XGB", "PLUS_LANDCOVER_RULES_ANOVA_PROXY", "XGBoost", "primary", "boosting 다양성"),
]

ENSEMBLE_RECIPES: list[dict[str, Any]] = [
    {
        "ensemble_id": "model_level_top2_hgb_lgbm_average",
        "method": "average",
        "candidates": ["PL_HGB", "PL_LGBM"],
    },
    {
        "ensemble_id": "model_level_top5_average",
        "method": "average",
        "candidates": ["PL_HGB", "RULES_HGB", "PL_LGBM", "RULES_RF", "RULES_XGB"],
    },
    {
        "ensemble_id": "model_level_top5_geometric_mean",
        "method": "geometric_mean",
        "candidates": ["PL_HGB", "RULES_HGB", "PL_LGBM", "RULES_RF", "RULES_XGB"],
    },
    {
        "ensemble_id": "model_level_top5_logit_average",
        "method": "logit_average",
        "candidates": ["PL_HGB", "RULES_HGB", "PL_LGBM", "RULES_RF", "RULES_XGB"],
    },
    {
        "ensemble_id": "model_level_top5_rank_average",
        "method": "rank_average",
        "candidates": ["PL_HGB", "RULES_HGB", "PL_LGBM", "RULES_RF", "RULES_XGB"],
    },
    {
        "ensemble_id": "model_level_anchor50_lgbm30_xgb20",
        "method": "weighted",
        "candidates": ["PL_HGB", "PL_LGBM", "RULES_XGB"],
        "weights": {"PL_HGB": 0.50, "PL_LGBM": 0.30, "RULES_XGB": 0.20},
    },
]


def load_baselines() -> tuple[float, float, float]:
    logistic = pd.read_csv(LOGISTIC_METRICS_PATH, encoding="utf-8-sig")
    logistic_baseline = logistic.loc[logistic["model"].eq("PLUS_LANDCOVER_RULES_ANOVA")].iloc[0]
    stage1 = pd.read_csv(STAGE1_COMPARISON_PATH, encoding="utf-8-sig")
    stage1_best = (
        stage1.loc[stage1["score_type"].eq("raw") & stage1["run_status"].eq("OK")]
        .sort_values(["auprc", "brier"], ascending=[False, True])
        .iloc[0]
    )
    return float(logistic_baseline["auprc"]), float(logistic_baseline["brier"]), float(stage1_best["auprc"])


def selected_config_for_fold(selected: pd.DataFrame, candidate: BaseCandidate, outer_fold: int) -> str:
    part = selected.loc[
        selected["feature_set"].eq(candidate.feature_set)
        & selected["model"].eq(candidate.model)
        & selected["outer_fold"].eq(outer_fold)
    ]
    if len(part) != 1:
        raise ValueError(f"selected config를 찾지 못했습니다: {candidate} outer={outer_fold}")
    return str(part.iloc[0]["selected_config_id"])


def feature_group_for(candidate: BaseCandidate) -> str:
    return s1.FEATURE_SET_META[candidate.feature_set]["feature_group"]


def train_base_oof(
    data: pd.DataFrame,
    outer: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(SELECTED_CONFIG_PATH, encoding="utf-8-sig")
    config_by_id = {config.config_id: config for config in s1.model_configs()}
    indexed = data.set_index("샘플ID", drop=False)
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []

    for candidate in BASE_CANDIDATES:
        features = feature_sets[candidate.feature_set]
        feature_group = feature_group_for(candidate)
        print(f"Stage3.5 base OOF: {candidate.candidate_id} ({candidate.feature_set} / {candidate.model})")
        for outer_fold in sorted(outer["outer_fold"].unique()):
            train_ids = outer.loc[~outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
            val_ids = outer.loc[outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
            config_id = selected_config_for_fold(selected, candidate, int(outer_fold))
            config = config_by_id[config_id]

            pipeline = s1.make_pipeline(features, categorical, config, indexed.loc[train_ids, "Target"])
            pipeline.fit(indexed.loc[train_ids, features], indexed.loc[train_ids, "Target"])
            val_score = s1.predict_probability(pipeline, indexed.loc[val_ids, features])

            fold_metric = {
                "candidate_id": candidate.candidate_id,
                "feature_set": candidate.feature_set,
                "feature_group": feature_group,
                "model": candidate.model,
                "outer_fold": int(outer_fold),
                "selected_config_id": config_id,
                **s3.probability_metrics(indexed.loc[val_ids, "Target"], val_score),
            }
            fold_rows.append(fold_metric)

            for sample_id, score in zip(val_ids, val_score):
                row = indexed.loc[sample_id]
                rows.append(
                    {
                        "샘플ID": sample_id,
                        "candidate_id": candidate.candidate_id,
                        "feature_set": candidate.feature_set,
                        "feature_group": feature_group,
                        "model": candidate.model,
                        "outer_fold": int(outer_fold),
                        "Target": int(row["Target"]),
                        "샘플유형": row["샘플유형"],
                        "기후지형유형": row["기후지형유형"],
                        "selected_config_id": config_id,
                        "score_raw": float(score),
                        "score_sigmoid": np.nan,
                        "score_isotonic": np.nan,
                        "score_calibrated": float(score),
                        "calibration_method": "raw",
                        "run_status": "OK",
                    }
                )

        candidate_predictions = pd.DataFrame([row for row in rows if row["candidate_id"] == candidate.candidate_id])
        registry_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "feature_set": candidate.feature_set,
                "feature_group": feature_group,
                "model": candidate.model,
                "role": candidate.role,
                "rationale": candidate.rationale,
                "selected_config_id": "|".join(sorted(candidate_predictions["selected_config_id"].unique())),
                **s3.probability_metrics(candidate_predictions["Target"], candidate_predictions["score_raw"]),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(fold_rows), pd.DataFrame(registry_rows)


def metadata_and_score_matrix(base_predictions: pd.DataFrame, outer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_ids = outer.sort_values(["outer_fold", "샘플ID"])["샘플ID"].tolist()
    metadata_source = (
        base_predictions.loc[base_predictions["candidate_id"].eq(BASE_CANDIDATES[0].candidate_id)]
        .set_index("샘플ID")
        .loc[ordered_ids]
        .reset_index()
    )
    metadata = metadata_source[["샘플ID", "outer_fold", "Target", "샘플유형", "기후지형유형"]].copy()
    matrix = pd.DataFrame(index=metadata.index)
    for candidate in BASE_CANDIDATES:
        part = base_predictions.loc[base_predictions["candidate_id"].eq(candidate.candidate_id)].set_index("샘플ID").loc[ordered_ids]
        check = part[["outer_fold", "Target", "샘플유형", "기후지형유형"]].reset_index(drop=True)
        if not check.equals(metadata[["outer_fold", "Target", "샘플유형", "기후지형유형"]]):
            raise ValueError(f"base candidate metadata 불일치: {candidate.candidate_id}")
        matrix[candidate.candidate_id] = s3.clipped_probability(part["score_raw"].to_numpy())
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("base model score matrix에 NaN/inf가 있습니다.")
    return metadata, matrix


def weights_from_recipe(recipe: dict[str, Any]) -> np.ndarray | None:
    if recipe["method"] != "weighted":
        return None
    weights = recipe["weights"]
    return np.array([float(weights[candidate]) for candidate in recipe["candidates"]], dtype=float)


def add_performance_weight_recipes(base_registry: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = ["PL_HGB", "RULES_HGB", "PL_LGBM", "RULES_RF", "RULES_XGB"]
    metric = base_registry.set_index("candidate_id").loc[candidates, "auprc"].to_numpy(dtype=float)
    linear = metric / metric.sum()
    delta = np.sqrt(np.maximum(metric - metric.min(), 1e-6))
    sqrt_delta = delta / delta.sum()
    return [
        {
            "ensemble_id": "model_level_top5_perf_weighted",
            "method": "weighted",
            "candidates": candidates,
            "weights": {candidate: float(weight) for candidate, weight in zip(candidates, linear)},
        },
        {
            "ensemble_id": "model_level_top5_sqrt_delta_weighted",
            "method": "weighted",
            "candidates": candidates,
            "weights": {candidate: float(weight) for candidate, weight in zip(candidates, sqrt_delta)},
        },
    ]


def build_ensemble_predictions(
    metadata: pd.DataFrame,
    score_matrix: pd.DataFrame,
    base_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recipes = [*ENSEMBLE_RECIPES, *add_performance_weight_recipes(base_registry)]
    prediction_frames = []
    recipe_rows = []
    for recipe in recipes:
        candidates = recipe["candidates"]
        weights = weights_from_recipe(recipe)
        score = s3.combine_scores(score_matrix, candidates, recipe["method"], weights)
        selected_config_id = json.dumps(
            {
                "method": recipe["method"],
                "candidates": candidates,
                "weights": None if weights is None else {candidate: float(weight) for candidate, weight in zip(candidates, weights)},
                "base_model_source": "stage35_retrained_outer_fold_models",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        prediction_frames.append(
            s3.make_prediction_rows(
                metadata,
                "ENSEMBLE_MODEL_LEVEL",
                recipe["ensemble_id"],
                selected_config_id,
                score,
            )
        )
        recipe_rows.append(
            {
                "ensemble_id": recipe["ensemble_id"],
                "method": recipe["method"],
                "candidates": "|".join(candidates),
                "weights": "" if weights is None else json.dumps({candidate: float(weight) for candidate, weight in zip(candidates, weights)}, sort_keys=True),
            }
        )
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(recipe_rows)


def reconstruction_check(base_predictions: pd.DataFrame) -> pd.DataFrame:
    stage1_predictions = pd.read_csv(STAGE1_PREDICTION_PATH, encoding="utf-8-sig", low_memory=False)
    rows = []
    for candidate in BASE_CANDIDATES:
        recreated = (
            base_predictions.loc[base_predictions["candidate_id"].eq(candidate.candidate_id), ["샘플ID", "score_raw"]]
            .rename(columns={"score_raw": "score_stage35"})
            .set_index("샘플ID")
        )
        original = (
            stage1_predictions.loc[
                stage1_predictions["feature_set"].eq(candidate.feature_set)
                & stage1_predictions["model"].eq(candidate.model)
                & stage1_predictions["run_status"].eq("OK"),
                ["샘플ID", "score_raw"],
            ]
            .rename(columns={"score_raw": "score_stage1"})
            .set_index("샘플ID")
        )
        joined = recreated.join(original, how="inner")
        diff = np.abs(joined["score_stage35"].to_numpy(dtype=float) - joined["score_stage1"].to_numpy(dtype=float))
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "feature_set": candidate.feature_set,
                "model": candidate.model,
                "matched_n": len(joined),
                "max_abs_diff_vs_stage1_oof": float(diff.max()),
                "mean_abs_diff_vs_stage1_oof": float(diff.mean()),
                "corr_vs_stage1_oof": float(joined["score_stage35"].corr(joined["score_stage1"])),
            }
        )
    return pd.DataFrame(rows)


def validation_checks(
    metadata: pd.DataFrame,
    base_predictions: pd.DataFrame,
    ensemble_predictions: pd.DataFrame,
    lockbox: pd.DataFrame,
) -> pd.DataFrame:
    checks = []
    lockbox_ids = set(lockbox.loc[lockbox["split"].eq("lockbox_test"), "샘플ID"])
    development_ids = set(lockbox.loc[lockbox["split"].eq("development"), "샘플ID"])
    base_ids = set(metadata["샘플ID"])
    checks.extend(
        [
            {"check": "development_row_count", "value": len(metadata), "expected": 13632, "passed": len(metadata) == 13632},
            {"check": "development_positive_n", "value": int(metadata["Target"].sum()), "expected": 1242, "passed": int(metadata["Target"].sum()) == 1242},
            {"check": "sample_id_missing", "value": int(metadata["샘플ID"].isna().sum()), "expected": 0, "passed": metadata["샘플ID"].isna().sum() == 0},
            {"check": "sample_id_duplicates", "value": int(metadata["샘플ID"].duplicated().sum()), "expected": 0, "passed": metadata["샘플ID"].duplicated().sum() == 0},
            {"check": "lockbox_overlap", "value": len(base_ids & lockbox_ids), "expected": 0, "passed": len(base_ids & lockbox_ids) == 0},
            {"check": "development_manifest_match", "value": len(base_ids ^ development_ids), "expected": 0, "passed": len(base_ids ^ development_ids) == 0},
        ]
    )
    for label, predictions, group_col in [
        ("base", base_predictions, "candidate_id"),
        ("ensemble", ensemble_predictions, "model"),
    ]:
        for group, part in predictions.groupby(group_col, observed=True):
            score = part["score_raw"].to_numpy(dtype=float)
            checks.extend(
                [
                    {"check": f"{label}_rows::{group}", "value": len(part), "expected": 13632, "passed": len(part) == 13632},
                    {"check": f"{label}_folds::{group}", "value": int(part["outer_fold"].nunique()), "expected": 5, "passed": int(part["outer_fold"].nunique()) == 5},
                    {"check": f"{label}_nan::{group}", "value": int(np.isnan(score).sum()), "expected": 0, "passed": not np.isnan(score).any()},
                    {"check": f"{label}_inf::{group}", "value": int((~np.isfinite(score)).sum()), "expected": 0, "passed": np.isfinite(score).all()},
                    {"check": f"{label}_range::{group}", "value": int(((score < 0) | (score > 1)).sum()), "expected": 0, "passed": not ((score < 0) | (score > 1)).any()},
                ]
            )
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        raise ValueError("validation check 실패:\n" + result.loc[~result["passed"]].to_string(index=False))
    return result


def choose_full_config(selected: pd.DataFrame, candidate: BaseCandidate) -> str:
    part = selected.loc[selected["feature_set"].eq(candidate.feature_set) & selected["model"].eq(candidate.model)].copy()
    summary = (
        part.groupby("selected_config_id", as_index=False)
        .agg(n=("selected_config_id", "size"), mean_inner_auprc=("inner_mean_auprc", "mean"))
        .sort_values(["n", "mean_inner_auprc", "selected_config_id"], ascending=[False, False, True])
    )
    return str(summary.iloc[0]["selected_config_id"])


def train_full_development_models(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
    best_report_ensemble: str,
) -> pd.DataFrame:
    selected = pd.read_csv(SELECTED_CONFIG_PATH, encoding="utf-8-sig")
    config_by_id = {config.config_id: config for config in s1.model_configs()}
    indexed = data.set_index("샘플ID", drop=False)
    manifest_rows = []
    full_score_frame = indexed[["샘플ID", "Target", "샘플유형", "기후지형유형"]].copy()

    for candidate in BASE_CANDIDATES:
        features = feature_sets[candidate.feature_set]
        config_id = choose_full_config(selected, candidate)
        config = config_by_id[config_id]
        pipeline = s1.make_pipeline(features, categorical, config, indexed["Target"])
        pipeline.fit(indexed[features], indexed["Target"])
        model_path = MODEL_DIR / f"{candidate.candidate_id}_{candidate.feature_set}_{candidate.model}_{config_id}.joblib"
        joblib.dump(pipeline, model_path)
        full_score_frame[candidate.candidate_id] = s1.predict_probability(pipeline, indexed[features])
        manifest_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "feature_set": candidate.feature_set,
                "feature_group": feature_group_for(candidate),
                "model": candidate.model,
                "selected_full_config_id": config_id,
                "n_features": len(features),
                "features": "|".join(features),
                "model_path": str(model_path.relative_to(ROOT)),
            }
        )

    full_score_frame.to_csv(PREDICTION_DIR / "ml_stage35_full_development_base_scores.csv", index=False, encoding="utf-8-sig")
    recipe = {
        "selected_report_ensemble": best_report_ensemble,
        "note": "lockbox 평가는 아직 수행하지 않았다. 이 recipe는 Step4/Step5 적용용이다.",
        "base_models": [row["candidate_id"] for row in manifest_rows],
        "recipes": ENSEMBLE_RECIPES,
    }
    (MODEL_DIR / "stage35_ensemble_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(METRIC_DIR / "ml_stage35_final_model_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, n: int | None = None) -> str:
    part = df.copy()
    if columns is not None:
        part = part[columns]
    if n is not None:
        part = part.head(n)
    return part.round(5).to_markdown(index=False)


def write_summary(
    base_registry: pd.DataFrame,
    ensemble_comparison: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_risk: pd.DataFrame,
    subgroup: pd.DataFrame,
    validation: pd.DataFrame,
    reconstruction: pd.DataFrame,
    manifest: pd.DataFrame,
    logistic_auprc: float,
    stage1_best_auprc: float,
) -> None:
    raw = ensemble_comparison.loc[ensemble_comparison["score_type"].eq("raw")].copy()
    best = raw.iloc[0]
    probability_ready = raw.loc[
        raw["delta_auprc_vs_stage1_best"].ge(0.005) & raw["brier"].le(0.10) & raw["log_loss"].le(0.30)
    ]
    report_candidate = probability_ready.iloc[0] if not probability_ready.empty else best
    best_top = top_risk.loc[
        top_risk["feature_set"].eq(report_candidate["feature_set"])
        & top_risk["model"].eq(report_candidate["model"])
        & top_risk["score_type"].eq("raw")
    ]
    best_subgroup = subgroup.loc[
        subgroup["feature_set"].eq(report_candidate["feature_set"])
        & subgroup["model"].eq(report_candidate["model"])
        & subgroup["score_type"].eq("raw")
        & subgroup["subgroup_type"].eq("negative_type")
    ]
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
        "# 머신러닝 3.5차 모델 레벨 앙상블 결과",
        "",
        "## 1. 실행 목적",
        "",
        "- 3차 OOF blending 진단을 실제 재학습 가능한 model-level ensemble 파이프라인으로 전환했다.",
        "- 각 outer fold에서 base model 5개를 다시 학습한 뒤, 해당 validation fold 예측을 합쳐 strict OOF 앙상블 성능을 계산했다.",
        "- lockbox test는 사용하지 않았다.",
        "",
        "## 2. 기준선",
        "",
        f"- 로지스틱 Stage17 기준선 AUPRC: {logistic_auprc:.4f}",
        f"- 1차 ML 최고 AUPRC: {stage1_best_auprc:.4f}",
        "",
        "## 3. 최고 결과",
        "",
        f"- AUPRC 최고: `{best['feature_set']} / {best['model']} / raw`",
        f"- AUPRC {float(best['auprc']):.4f}, ROC AUC {float(best['auroc']):.4f}, Brier {float(best['brier']):.5f}, log loss {float(best['log_loss']):.5f}",
        f"- 1차 최고 대비 ΔAUPRC {float(best['delta_auprc_vs_stage1_best']):+.4f}, 로지스틱 대비 ΔAUPRC {float(best['delta_auprc_vs_logistic']):+.4f}",
        "",
        "rank average는 순위 성능 진단용이다. 확률 품질까지 고려한 보고서용 후보는 다음으로 둔다.",
        "",
        f"- 보고서용 후보: `{report_candidate['feature_set']} / {report_candidate['model']} / raw`",
        f"- AUPRC {float(report_candidate['auprc']):.4f}, ROC AUC {float(report_candidate['auroc']):.4f}, Brier {float(report_candidate['brier']):.5f}, log loss {float(report_candidate['log_loss']):.5f}",
        f"- 1차 최고 대비 ΔAUPRC {float(report_candidate['delta_auprc_vs_stage1_best']):+.4f}",
        "",
        "## 4. base model 재학습 OOF 성능",
        "",
        markdown_table(base_registry, ["candidate_id", "feature_set", "model", "auprc", "auroc", "brier", "log_loss"]),
        "",
        "## 5. model-level ensemble 성능",
        "",
        markdown_table(ensemble_comparison, display_cols),
        "",
        "## 6. 1차 저장 OOF와 재학습 OOF 비교",
        "",
        reconstruction.round(8).to_markdown(index=False),
        "",
        "## 7. 보고서용 후보 top-risk capture",
        "",
        best_top.round(5).to_markdown(index=False),
        "",
        "## 8. 보고서용 후보 hard-negative subgroup",
        "",
        best_subgroup[["feature_set", "model", "score_type", "subgroup", "auprc", "auroc", "brier", "log_loss"]]
        .round(5)
        .to_markdown(index=False),
        "",
        "## 9. full development 재학습 artifact",
        "",
        manifest[["candidate_id", "feature_set", "model", "selected_full_config_id", "model_path"]].to_markdown(index=False),
        "",
        "## 10. 검증",
        "",
        validation.to_markdown(index=False),
        "",
        "## 11. 산출물",
        "",
        "- `outputs/metrics/ml_stage35_base_model_oof_metrics.csv`",
        "- `outputs/metrics/ml_stage35_ensemble_model_comparison.csv`",
        "- `outputs/metrics/ml_stage35_thresholds.csv`",
        "- `outputs/metrics/ml_stage35_top_risk_capture.csv`",
        "- `outputs/metrics/ml_stage35_subgroup_metrics.csv`",
        "- `outputs/predictions/ml_stage35_base_model_oof_predictions.csv`",
        "- `outputs/predictions/ml_stage35_model_level_ensemble_oof_predictions.csv`",
        "- `outputs/models/stage35_model_level_ensemble/`",
    ]
    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "ml_stage35_model_level_ensemble_summary.md").write_text(summary, encoding="utf-8")
    (ML_DIR / "머신러닝_3_5차_모델레벨_앙상블_진행_결과.md").write_text(summary, encoding="utf-8")

    log_path = ML_DIR / "LOG.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# 머신러닝 모델링 진행 로그\n"
    log_entry = "\n".join(
        [
            "",
            "## 2026-06-21",
            "",
            "### 3.5차 모델 레벨 앙상블 실행",
            "",
            "- 3차 OOF blending을 재학습 가능한 model-level ensemble 파이프라인으로 전환했다.",
            "- 각 outer fold에서 base model 5개를 실제로 다시 학습하고 validation 예측을 결합했다.",
            "- full development 재학습 model artifact도 저장했다. lockbox 평가는 아직 하지 않았다.",
            "",
            "### 3.5차 결과",
            "",
            f"- AUPRC 최고: `{best['feature_set']} / {best['model']} / raw`, AUPRC {float(best['auprc']):.4f}, Δstage1 {float(best['delta_auprc_vs_stage1_best']):+.4f}",
            f"- 보고서용 후보: `{report_candidate['feature_set']} / {report_candidate['model']} / raw`, AUPRC {float(report_candidate['auprc']):.4f}, Brier {float(report_candidate['brier']):.5f}, log loss {float(report_candidate['log_loss']):.5f}",
            "",
            "### 산출물",
            "",
            "- `outputs/ml_stage35_model_level_ensemble_summary.md`",
            "- `머신러닝_3_5차_모델레벨_앙상블_진행_결과.md`",
            "- `outputs/metrics/ml_stage35_ensemble_model_comparison.csv`",
            "- `outputs/models/stage35_model_level_ensemble/`",
            "",
        ]
    )
    log_path.write_text(existing.rstrip() + "\n" + log_entry, encoding="utf-8")

    overall_path = ML_DIR / "머신러닝_전체_진행_결과.md"
    existing_overall = overall_path.read_text(encoding="utf-8") if overall_path.exists() else ""
    overall_path.write_text(existing_overall.rstrip() + "\n\n---\n\n" + summary, encoding="utf-8")


def main() -> None:
    print("ML Stage3.5: data prepare")
    data, outer, _inner, feature_sets, categorical = s1.prepare_data()
    lockbox = pd.read_csv(LOCKBOX_PATH, encoding="utf-8-sig")
    logistic_auprc, logistic_brier, stage1_best_auprc = load_baselines()

    print("ML Stage3.5: train base models by outer fold")
    base_predictions, base_fold_metrics, base_registry = train_base_oof(data, outer, feature_sets, categorical)
    base_predictions.to_csv(PREDICTION_DIR / "ml_stage35_base_model_oof_predictions.csv", index=False, encoding="utf-8-sig")
    base_fold_metrics.to_csv(METRIC_DIR / "ml_stage35_base_model_fold_metrics.csv", index=False, encoding="utf-8-sig")
    base_registry.to_csv(METRIC_DIR / "ml_stage35_base_model_oof_metrics.csv", index=False, encoding="utf-8-sig")

    print("ML Stage3.5: base reconstruction check")
    reconstruction = reconstruction_check(base_predictions)
    reconstruction.to_csv(METRIC_DIR / "ml_stage35_base_reconstruction_check.csv", index=False, encoding="utf-8-sig")

    print("ML Stage3.5: build ensembles")
    metadata, score_matrix = metadata_and_score_matrix(base_predictions, outer)
    ensemble_predictions, recipe_table = build_ensemble_predictions(metadata, score_matrix, base_registry)
    ensemble_predictions.to_csv(PREDICTION_DIR / "ml_stage35_model_level_ensemble_oof_predictions.csv", index=False, encoding="utf-8-sig")
    recipe_table.to_csv(METRIC_DIR / "ml_stage35_ensemble_recipes.csv", index=False, encoding="utf-8-sig")

    print("ML Stage3.5: metrics")
    validation = validation_checks(metadata, base_predictions, ensemble_predictions, lockbox)
    validation.to_csv(METRIC_DIR / "ml_stage35_validation_checks.csv", index=False, encoding="utf-8-sig")
    ensemble_comparison = s3.comparison_from_predictions(ensemble_predictions, logistic_auprc, logistic_brier, stage1_best_auprc)
    thresholds = s3.make_threshold_table(ensemble_predictions)
    top_risk = s3.make_top_risk_table(ensemble_predictions)
    subgroup = s3.make_subgroup_table(ensemble_predictions)
    fold_metrics = s3.fold_metrics_for_predictions(ensemble_predictions)
    ensemble_comparison.to_csv(METRIC_DIR / "ml_stage35_ensemble_model_comparison.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(METRIC_DIR / "ml_stage35_thresholds.csv", index=False, encoding="utf-8-sig")
    top_risk.to_csv(METRIC_DIR / "ml_stage35_top_risk_capture.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(METRIC_DIR / "ml_stage35_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(METRIC_DIR / "ml_stage35_ensemble_fold_metrics.csv", index=False, encoding="utf-8-sig")

    raw = ensemble_comparison.loc[ensemble_comparison["score_type"].eq("raw")].copy()
    probability_ready = raw.loc[
        raw["delta_auprc_vs_stage1_best"].ge(0.005) & raw["brier"].le(0.10) & raw["log_loss"].le(0.30)
    ]
    report_candidate = probability_ready.iloc[0] if not probability_ready.empty else raw.iloc[0]

    print("ML Stage3.5: train full development artifacts")
    manifest = train_full_development_models(data, feature_sets, categorical, str(report_candidate["model"]))

    write_summary(
        base_registry,
        ensemble_comparison,
        thresholds,
        top_risk,
        subgroup,
        validation,
        reconstruction,
        manifest,
        logistic_auprc,
        stage1_best_auprc,
    )

    best = raw.iloc[0]
    print(
        "ML Stage3.5 완료: "
        f"{best['feature_set']} / {best['model']} / raw "
        f"AUPRC={float(best['auprc']):.4f}, "
        f"Δstage1={float(best['delta_auprc_vs_stage1_best']):+.4f}; "
        f"report={report_candidate['model']} AUPRC={float(report_candidate['auprc']):.4f}"
    )


if __name__ == "__main__":
    main()
