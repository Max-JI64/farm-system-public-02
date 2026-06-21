from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import stage1_ml_screening as s1
import stage3_ml_ensemble as s3


warnings.filterwarnings("ignore")

ROOT = s1.ROOT
ML_DIR = s1.ML_DIR
OUTPUT_DIR = s1.OUTPUT_DIR
METRIC_DIR = s1.METRIC_DIR
PREDICTION_DIR = s1.PREDICTION_DIR

STAGE4_MANIFEST_PATH = OUTPUT_DIR / "models" / "stage4_final_selection" / "stage4_final_selection_manifest.json"
STAGE4_SELECTION_PATH = METRIC_DIR / "ml_stage4_final_candidate_selection.csv"
STAGE4_THRESHOLD_PATH = METRIC_DIR / "ml_stage4_final_thresholds.csv"
STAGE35_MODEL_MANIFEST_PATH = METRIC_DIR / "ml_stage35_final_model_manifest.csv"
STAGE35_FULL_BASE_SCORE_PATH = PREDICTION_DIR / "ml_stage35_full_development_base_scores.csv"
LOCKBOX_PATH = s1.LOCKBOX_PATH


def load_baselines() -> tuple[float, float, float]:
    manifest = json.loads(STAGE4_MANIFEST_PATH.read_text(encoding="utf-8"))
    baselines = manifest["baselines"]
    return (
        float(baselines["logistic_auprc"]),
        float(baselines["logistic_brier"]),
        float(baselines["stage1_best_auprc"]),
    )


def prepare_full_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[str]]:
    data = pd.read_csv(s1.DATA_PATH, encoding="utf-8-sig", parse_dates=["기준시각"], low_memory=False)
    engineered = pd.read_csv(s1.ENGINEERED_PATH, encoding="utf-8-sig")
    engineered = engineered[[col for col in engineered.columns if not col.endswith(".1")]].copy()
    engineered_cols = [col for col in engineered.columns if col not in {"Target", "샘플유형"}]
    data = data.merge(engineered[engineered_cols], on="샘플ID", how="left", validate="one_to_one")
    data = s1.add_stage9_features(data)
    feature_sets = s1.load_feature_sets()
    all_features = sorted(set().union(*feature_sets.values()))
    missing = sorted(set(all_features) - set(data.columns))
    if missing:
        raise KeyError(f"데이터에 없는 피처: {missing}")

    categorical = [col for col in s1.BASE_CATEGORICAL + s1.LANDCOVER_CATEGORICAL if col in all_features]
    for col in all_features:
        if col in categorical:
            data[col] = data[col].fillna("미상").astype(str)
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    lockbox = pd.read_csv(LOCKBOX_PATH, encoding="utf-8-sig")
    if len(data) != 17045:
        raise ValueError(f"전체 데이터 행 수 불일치: {len(data)}")
    if data["샘플ID"].duplicated().any() or data["샘플ID"].isna().any():
        raise ValueError("샘플ID 중복 또는 결측이 있습니다.")
    if set(data["샘플ID"]) != set(lockbox["샘플ID"]):
        raise ValueError("데이터와 lockbox manifest 샘플ID가 다릅니다.")
    return data, lockbox, feature_sets, categorical


def predict_base_models(lockbox_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_manifest = pd.read_csv(STAGE35_MODEL_MANIFEST_PATH, encoding="utf-8-sig")
    indexed = lockbox_data.set_index("샘플ID", drop=False)
    metadata = lockbox_data[["샘플ID", "Target", "샘플유형", "기후지형유형"]].copy().reset_index(drop=True)
    score_matrix = pd.DataFrame(index=metadata.index)
    rows = []

    for row in model_manifest.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        features = str(row.features).split("|")
        model_path = ROOT / str(row.model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = joblib.load(model_path)
        score = s1.predict_probability(model, indexed[features])
        score_matrix[candidate_id] = score
        for sample_id, probability in zip(metadata["샘플ID"], score):
            source = indexed.loc[sample_id]
            rows.append(
                {
                    "샘플ID": sample_id,
                    "candidate_id": candidate_id,
                    "feature_set": row.feature_set,
                    "feature_group": row.feature_group,
                    "model": row.model,
                    "dataset": "lockbox_test",
                    "Target": int(source["Target"]),
                    "샘플유형": source["샘플유형"],
                    "기후지형유형": source["기후지형유형"],
                    "score": float(probability),
                }
            )
    if score_matrix.isna().any().any() or not np.isfinite(score_matrix.to_numpy()).all():
        raise ValueError("base lockbox score에 NaN/inf가 있습니다.")
    return pd.DataFrame(rows), score_matrix


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    return np.searchsorted(reference, values, side="right") / len(reference)


def combine_with_reference(
    lockbox_scores: pd.DataFrame,
    reference_scores: pd.DataFrame,
    method: str,
    candidates: list[str],
    weights: np.ndarray | None,
) -> np.ndarray:
    if method == "rank_average":
        rank_frame = pd.DataFrame(index=lockbox_scores.index)
        for candidate in candidates:
            rank_frame[candidate] = empirical_percentile(
                reference_scores[candidate].to_numpy(dtype=float),
                lockbox_scores[candidate].to_numpy(dtype=float),
            )
        return s3.clipped_probability(rank_frame.mean(axis=1).to_numpy(dtype=float))
    return s3.combine_scores(lockbox_scores, candidates, method, weights)


def parse_selected_config(config_text: str) -> tuple[str, list[str], np.ndarray | None]:
    config = json.loads(config_text)
    method = str(config["method"])
    candidates = list(config["candidates"])
    weights = None
    if config.get("weights"):
        weight_map = config["weights"]
        weights = np.array([float(weight_map[candidate]) for candidate in candidates], dtype=float)
    return method, candidates, weights


def build_final_predictions(
    lockbox_data: pd.DataFrame,
    base_score_matrix: pd.DataFrame,
    reference_scores: pd.DataFrame,
) -> pd.DataFrame:
    selection = pd.read_csv(STAGE4_SELECTION_PATH, encoding="utf-8-sig")
    metadata = lockbox_data[["샘플ID", "Target", "샘플유형", "기후지형유형"]].copy().reset_index(drop=True)
    frames = []
    for row in selection.itertuples(index=False):
        role = str(row.selection_role)
        model_name = str(row.model)
        score_type = str(row.score_type)
        method, candidates, weights = parse_selected_config(str(row.selected_config_id))
        raw_score = combine_with_reference(base_score_matrix, reference_scores, method, candidates, weights)
        if score_type != "raw":
            raise NotImplementedError(f"Step5 selected score_type은 현재 raw만 지원합니다: {score_type}")
        selected_config_id = json.dumps(
            {
                "selection_role": role,
                "model": model_name,
                "score_type": score_type,
                "method": method,
                "candidates": candidates,
                "weights": None if weights is None else {candidate: float(weight) for candidate, weight in zip(candidates, weights)},
                "rank_reference": "full_development_base_score_ecdf" if method == "rank_average" else "not_applicable",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        part = metadata.copy()
        part.insert(1, "feature_set", "LOCKBOX_FINAL")
        part.insert(2, "feature_group", "stage5_lockbox")
        part.insert(3, "model", model_name)
        part.insert(4, "selection_role", role)
        part["dataset"] = "lockbox_test"
        part["outer_fold"] = -1
        part["selected_config_id"] = selected_config_id
        part["score_raw"] = raw_score
        part["score_sigmoid"] = np.nan
        part["score_isotonic"] = np.nan
        part["score_calibrated"] = raw_score
        part["calibration_method"] = score_type
        part["run_status"] = "OK"
        frames.append(
            part[
                [
                    "샘플ID",
                    "feature_set",
                    "feature_group",
                    "model",
                    "selection_role",
                    "dataset",
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
        )
    predictions = pd.concat(frames, ignore_index=True)
    if predictions["score_raw"].isna().any() or not np.isfinite(predictions["score_raw"].to_numpy(dtype=float)).all():
        raise ValueError("final lockbox score에 NaN/inf가 있습니다.")
    if (predictions["score_raw"].lt(0) | predictions["score_raw"].gt(1)).any():
        raise ValueError("final lockbox score가 [0, 1] 범위를 벗어났습니다.")
    return predictions


def prediction_long_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.rename(columns={"score_raw": "score"}).assign(score_type="raw")


def model_comparison(predictions: pd.DataFrame, logistic_auprc: float, logistic_brier: float, stage1_best_auprc: float) -> pd.DataFrame:
    rows = []
    for (model, role), part in predictions.groupby(["model", "selection_role"], observed=True):
        row = {
            "dataset": "lockbox_test",
            "feature_set": str(part["feature_set"].iloc[0]),
            "feature_group": str(part["feature_group"].iloc[0]),
            "model": model,
            "selection_role": role,
            "score_type": "raw",
            "run_status": "OK",
            "selected_config_id": s3.selected_config_summary(part),
            **s3.probability_metrics(part["Target"], part["score_raw"]),
        }
        threshold = s3.best_f1_threshold(part["Target"], part["score_raw"])
        row.update(
            {
                f"best_f1_{key}": value
                for key, value in s3.classification_metrics_at_threshold(part["Target"], part["score_raw"], threshold).items()
            }
        )
        row["delta_auprc_vs_logistic_oof_baseline"] = float(row["auprc"] - logistic_auprc)
        row["delta_brier_vs_logistic_oof_baseline"] = float(row["brier"] - logistic_brier)
        row["delta_auprc_vs_stage1_oof_best"] = float(row["auprc"] - stage1_best_auprc)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["auprc", "brier"], ascending=[False, True])


def fixed_threshold_table(predictions: pd.DataFrame) -> pd.DataFrame:
    stage4_thresholds = pd.read_csv(STAGE4_THRESHOLD_PATH, encoding="utf-8-sig")
    rows = []
    for row in stage4_thresholds.itertuples(index=False):
        model_name = str(row.model)
        score_type = str(row.score_type)
        if score_type != "raw":
            continue
        part = predictions.loc[predictions["model"].eq(model_name)]
        if part.empty:
            continue
        rows.append(
            {
                "dataset": "lockbox_test",
                "feature_set": str(part["feature_set"].iloc[0]),
                "feature_group": str(part["feature_group"].iloc[0]),
                "model": model_name,
                "selection_role": str(part["selection_role"].iloc[0]),
                "score_type": score_type,
                "operating_point": str(row.operating_point),
                "source": "stage4_development_oof_fixed_threshold",
                **s3.classification_metrics_at_threshold(part["Target"], part["score_raw"], float(row.threshold)),
            }
        )
    return pd.DataFrame(rows)


def top_risk_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, role), part in predictions.groupby(["model", "selection_role"], observed=True):
        part = part.sort_values("score_raw", ascending=False).reset_index(drop=True)
        total_positive = int(part["Target"].sum())
        base_rate = float(part["Target"].mean())
        for pct in [0.05, 0.10, 0.20]:
            n = int(np.ceil(len(part) * pct))
            top = part.iloc[:n]
            precision = float(top["Target"].mean())
            rows.append(
                {
                    "dataset": "lockbox_test",
                    "feature_set": str(part["feature_set"].iloc[0]),
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "selection_role": role,
                    "score_type": "raw",
                    "top_pct": pct,
                    "selected_n": n,
                    "selected_rate": float(n / len(part)),
                    "threshold_min": float(top["score_raw"].min()),
                    "captured_positive_n": int(top["Target"].sum()),
                    "total_positive_n": total_positive,
                    "capture_rate_recall": float(top["Target"].sum() / total_positive) if total_positive else np.nan,
                    "precision": precision,
                    "base_positive_rate": base_rate,
                    "lift_vs_base": float(precision / base_rate) if base_rate else np.nan,
                }
            )
    return pd.DataFrame(rows)


def subgroup_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    negative_types = [value for value in sorted(predictions["샘플유형"].dropna().unique()) if value != "Target_1"]
    for (model, role), part in predictions.groupby(["model", "selection_role"], observed=True):
        positives = part.loc[part["샘플유형"].eq("Target_1")]
        for negative_type in negative_types:
            subset = pd.concat([positives, part.loc[part["샘플유형"].eq(negative_type)]], ignore_index=True)
            if subset["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "dataset": "lockbox_test",
                    "feature_set": str(part["feature_set"].iloc[0]),
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "selection_role": role,
                    "score_type": "raw",
                    "subgroup_type": "negative_type",
                    "subgroup": negative_type,
                    **s3.probability_metrics(subset["Target"], subset["score_raw"]),
                }
            )
        for climate, subset in part.groupby("기후지형유형", observed=True):
            if subset["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "dataset": "lockbox_test",
                    "feature_set": str(part["feature_set"].iloc[0]),
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "selection_role": role,
                    "score_type": "raw",
                    "subgroup_type": "기후지형유형",
                    "subgroup": climate,
                    **s3.probability_metrics(subset["Target"], subset["score_raw"]),
                }
            )
    return pd.DataFrame(rows)


def oof_vs_lockbox_comparison(lockbox_comparison: pd.DataFrame) -> pd.DataFrame:
    stage4_selection = pd.read_csv(STAGE4_SELECTION_PATH, encoding="utf-8-sig")
    rows = []
    metric_cols = ["auprc", "auroc", "brier", "log_loss", "best_f1_f1", "best_f1_precision", "best_f1_recall"]
    for row in stage4_selection.itertuples(index=False):
        model_name = str(row.model)
        role = str(row.selection_role)
        lockbox_row = lockbox_comparison.loc[lockbox_comparison["model"].eq(model_name)].iloc[0]
        for metric in metric_cols:
            rows.append(
                {
                    "selection_role": role,
                    "model": model_name,
                    "metric": metric,
                    "development_oof": float(getattr(row, metric)),
                    "lockbox_test": float(lockbox_row[metric]),
                    "delta_lockbox_minus_oof": float(lockbox_row[metric] - getattr(row, metric)),
                }
            )
    return pd.DataFrame(rows)


def validation_checks(
    data: pd.DataFrame,
    lockbox: pd.DataFrame,
    lockbox_data: pd.DataFrame,
    base_predictions: pd.DataFrame,
    final_predictions: pd.DataFrame,
) -> pd.DataFrame:
    development_ids = set(lockbox.loc[lockbox["split"].eq("development"), "샘플ID"])
    lockbox_ids = set(lockbox.loc[lockbox["split"].eq("lockbox_test"), "샘플ID"])
    pred_ids = set(final_predictions["샘플ID"])
    rows = [
        {"check": "full_data_n", "value": len(data), "expected": 17045, "passed": len(data) == 17045},
        {"check": "development_n", "value": len(development_ids), "expected": 13632, "passed": len(development_ids) == 13632},
        {"check": "lockbox_n", "value": len(lockbox_ids), "expected": 3413, "passed": len(lockbox_ids) == 3413},
        {
            "check": "lockbox_positive_n",
            "value": int(lockbox_data["Target"].sum()),
            "expected": int(lockbox.loc[lockbox["split"].eq("lockbox_test"), "Target"].sum()),
            "passed": int(lockbox_data["Target"].sum()) == int(lockbox.loc[lockbox["split"].eq("lockbox_test"), "Target"].sum()),
        },
        {"check": "prediction_ids_match_lockbox", "value": len(pred_ids ^ lockbox_ids), "expected": 0, "passed": len(pred_ids ^ lockbox_ids) == 0},
        {"check": "prediction_development_overlap", "value": len(pred_ids & development_ids), "expected": 0, "passed": len(pred_ids & development_ids) == 0},
        {"check": "final_models_n", "value": final_predictions["model"].nunique(), "expected": 2, "passed": final_predictions["model"].nunique() == 2},
        {"check": "base_models_n", "value": base_predictions["candidate_id"].nunique(), "expected": 5, "passed": base_predictions["candidate_id"].nunique() == 5},
    ]
    for label, frame, score_col, group_col in [
        ("base", base_predictions, "score", "candidate_id"),
        ("final", final_predictions, "score_raw", "model"),
    ]:
        for group, part in frame.groupby(group_col, observed=True):
            score = part[score_col].to_numpy(dtype=float)
            rows.extend(
                [
                    {"check": f"{label}_rows::{group}", "value": len(part), "expected": len(lockbox_ids), "passed": len(part) == len(lockbox_ids)},
                    {"check": f"{label}_nan::{group}", "value": int(np.isnan(score).sum()), "expected": 0, "passed": not np.isnan(score).any()},
                    {"check": f"{label}_inf::{group}", "value": int((~np.isfinite(score)).sum()), "expected": 0, "passed": np.isfinite(score).all()},
                    {"check": f"{label}_range::{group}", "value": int(((score < 0) | (score > 1)).sum()), "expected": 0, "passed": not ((score < 0) | (score > 1)).any()},
                ]
            )
    result = pd.DataFrame(rows)
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


def write_summary(
    lockbox_data: pd.DataFrame,
    comparison: pd.DataFrame,
    oof_vs_lockbox: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_risk: pd.DataFrame,
    subgroup: pd.DataFrame,
    validation: pd.DataFrame,
    logistic_auprc: float,
    stage1_best_auprc: float,
) -> None:
    ranking = comparison.loc[comparison["selection_role"].eq("ranking_score")].iloc[0]
    probability = comparison.loc[comparison["selection_role"].eq("probability_score")].iloc[0]
    probability_thresholds = thresholds.loc[thresholds["selection_role"].eq("probability_score")]
    probability_top = top_risk.loc[top_risk["selection_role"].eq("probability_score")]
    probability_subgroup = subgroup.loc[
        subgroup["selection_role"].eq("probability_score") & subgroup["subgroup_type"].eq("negative_type")
    ]
    display_cols = [
        "selection_role",
        "model",
        "auprc",
        "auroc",
        "brier",
        "log_loss",
        "best_f1_f1",
        "best_f1_precision",
        "best_f1_recall",
        "delta_auprc_vs_logistic_oof_baseline",
        "delta_auprc_vs_stage1_oof_best",
    ]
    lines = [
        "# 머신러닝 5차 lockbox test 평가 결과",
        "",
        "## 1. 실행 목적",
        "",
        "- Step4에서 고정한 최종 후보를 lockbox test에 최초 적용했다.",
        "- 새 후보 선택이나 추가 튜닝은 하지 않았다.",
        "- full-development로 학습된 Step3.5 base model artifact만 사용했다.",
        "",
        "## 2. lockbox 구성",
        "",
        f"- lockbox row: {len(lockbox_data):,}",
        f"- Target 1: {int(lockbox_data['Target'].sum()):,}",
        f"- positive rate: {float(lockbox_data['Target'].mean()):.4f}",
        "",
        "## 3. lockbox 최종 성능",
        "",
        markdown_table(comparison, display_cols),
        "",
        "## 4. 개발 OOF 대비 lockbox 변화",
        "",
        oof_vs_lockbox.round(5).to_markdown(index=False),
        "",
        "## 5. 해석",
        "",
        f"- ranking score `{ranking['model']}`는 lockbox AUPRC {float(ranking['auprc']):.4f}를 기록했다.",
        f"- probability score `{probability['model']}`는 lockbox AUPRC {float(probability['auprc']):.4f}, Brier {float(probability['brier']):.5f}, log loss {float(probability['log_loss']):.5f}를 기록했다.",
        f"- probability score는 로지스틱 OOF 기준선 AUPRC {logistic_auprc:.4f} 대비 {float(probability['auprc'] - logistic_auprc):+.4f}, 1차 ML OOF 최고 AUPRC {stage1_best_auprc:.4f} 대비 {float(probability['auprc'] - stage1_best_auprc):+.4f}이다.",
        "- lockbox 결과는 development OOF보다 낮아졌는지를 중심으로 해석해야 하며, 이번 결과 이후에는 lockbox를 이용한 재튜닝을 하지 않는다.",
        "",
        "## 6. probability score 고정 threshold 성능",
        "",
        probability_thresholds.round(5).to_markdown(index=False),
        "",
        "## 7. probability score top-risk capture",
        "",
        probability_top.round(5).to_markdown(index=False),
        "",
        "## 8. probability score hard-negative subgroup",
        "",
        probability_subgroup[["selection_role", "model", "subgroup", "auprc", "auroc", "brier", "log_loss"]]
        .round(5)
        .to_markdown(index=False),
        "",
        "## 9. 검증",
        "",
        validation.to_markdown(index=False),
        "",
        "## 10. 산출물",
        "",
        "- `outputs/metrics/ml_stage5_lockbox_model_comparison.csv`",
        "- `outputs/metrics/ml_stage5_oof_vs_lockbox_comparison.csv`",
        "- `outputs/metrics/ml_stage5_lockbox_fixed_thresholds.csv`",
        "- `outputs/metrics/ml_stage5_lockbox_top_risk_capture.csv`",
        "- `outputs/metrics/ml_stage5_lockbox_subgroup_metrics.csv`",
        "- `outputs/predictions/ml_stage5_lockbox_final_predictions.csv`",
        "- `outputs/predictions/ml_stage5_lockbox_base_model_predictions.csv`",
    ]
    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "ml_stage5_lockbox_summary.md").write_text(summary, encoding="utf-8")
    (ML_DIR / "머신러닝_5차_lockbox_평가_결과.md").write_text(summary, encoding="utf-8")

    log_path = ML_DIR / "LOG.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# 머신러닝 모델링 진행 로그\n"
    log_entry = "\n".join(
        [
            "",
            "## 2026-06-21",
            "",
            "### 5차 lockbox test 최초 평가",
            "",
            "- Step4에서 고정한 ranking/probability score를 lockbox test에 최초 적용했다.",
            "- lockbox를 이용한 추가 후보 선택이나 튜닝은 하지 않았다.",
            "",
            "### 5차 결과",
            "",
            f"- ranking score `{ranking['model']}`: AUPRC {float(ranking['auprc']):.4f}, ROC AUC {float(ranking['auroc']):.4f}",
            f"- probability score `{probability['model']}`: AUPRC {float(probability['auprc']):.4f}, Brier {float(probability['brier']):.5f}, log loss {float(probability['log_loss']):.5f}",
            "",
            "### 산출물",
            "",
            "- `outputs/ml_stage5_lockbox_summary.md`",
            "- `머신러닝_5차_lockbox_평가_결과.md`",
            "- `outputs/metrics/ml_stage5_lockbox_model_comparison.csv`",
            "- `outputs/predictions/ml_stage5_lockbox_final_predictions.csv`",
            "",
        ]
    )
    log_path.write_text(existing.rstrip() + "\n" + log_entry, encoding="utf-8")

    overall_path = ML_DIR / "머신러닝_전체_진행_결과.md"
    existing_overall = overall_path.read_text(encoding="utf-8") if overall_path.exists() else ""
    overall_path.write_text(existing_overall.rstrip() + "\n\n---\n\n" + summary, encoding="utf-8")


def main() -> None:
    print("ML Stage5: prepare full data and lockbox")
    data, lockbox, _feature_sets, _categorical = prepare_full_data()
    lockbox_ids = set(lockbox.loc[lockbox["split"].eq("lockbox_test"), "샘플ID"])
    lockbox_data = data.loc[data["샘플ID"].isin(lockbox_ids)].copy().sort_values("샘플ID").reset_index(drop=True)
    logistic_auprc, logistic_brier, stage1_best_auprc = load_baselines()

    print("ML Stage5: base model lockbox prediction")
    base_predictions, base_score_matrix = predict_base_models(lockbox_data)
    reference_scores = pd.read_csv(STAGE35_FULL_BASE_SCORE_PATH, encoding="utf-8-sig")
    base_model_metrics = []
    for candidate_id, part in base_predictions.groupby("candidate_id", observed=True):
        base_model_metrics.append(
            {
                "dataset": "lockbox_test",
                "candidate_id": candidate_id,
                "feature_set": str(part["feature_set"].iloc[0]),
                "model": str(part["model"].iloc[0]),
                **s3.probability_metrics(part["Target"], part["score"]),
            }
        )
    base_model_metrics = pd.DataFrame(base_model_metrics).sort_values(["auprc", "brier"], ascending=[False, True])

    print("ML Stage5: final ensemble lockbox prediction")
    final_predictions = build_final_predictions(lockbox_data, base_score_matrix, reference_scores)
    comparison = model_comparison(final_predictions, logistic_auprc, logistic_brier, stage1_best_auprc)
    thresholds = fixed_threshold_table(final_predictions)
    top_risk = top_risk_table(final_predictions)
    subgroup = subgroup_table(final_predictions)
    oof_vs_lockbox = oof_vs_lockbox_comparison(comparison)
    validation = validation_checks(data, lockbox, lockbox_data, base_predictions, final_predictions)

    print("ML Stage5: write outputs")
    base_predictions.to_csv(PREDICTION_DIR / "ml_stage5_lockbox_base_model_predictions.csv", index=False, encoding="utf-8-sig")
    final_predictions.to_csv(PREDICTION_DIR / "ml_stage5_lockbox_final_predictions.csv", index=False, encoding="utf-8-sig")
    base_model_metrics.to_csv(METRIC_DIR / "ml_stage5_lockbox_base_model_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(METRIC_DIR / "ml_stage5_lockbox_model_comparison.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(METRIC_DIR / "ml_stage5_lockbox_fixed_thresholds.csv", index=False, encoding="utf-8-sig")
    top_risk.to_csv(METRIC_DIR / "ml_stage5_lockbox_top_risk_capture.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(METRIC_DIR / "ml_stage5_lockbox_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    oof_vs_lockbox.to_csv(METRIC_DIR / "ml_stage5_oof_vs_lockbox_comparison.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(METRIC_DIR / "ml_stage5_validation_checks.csv", index=False, encoding="utf-8-sig")
    write_summary(
        lockbox_data,
        comparison,
        oof_vs_lockbox,
        thresholds,
        top_risk,
        subgroup,
        validation,
        logistic_auprc,
        stage1_best_auprc,
    )

    probability = comparison.loc[comparison["selection_role"].eq("probability_score")].iloc[0]
    print(
        "ML Stage5 완료: "
        f"probability={probability['model']} "
        f"AUPRC={float(probability['auprc']):.4f}, "
        f"ROC={float(probability['auroc']):.4f}, "
        f"Brier={float(probability['brier']):.5f}"
    )


if __name__ == "__main__":
    main()
