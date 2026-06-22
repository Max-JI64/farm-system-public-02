from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ML_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ML_DIR / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

FINAL_MODEL = "model_level_top5_geometric_mean"
FINAL_ROLE = "probability_score"

STAGE4_FINAL_OOF_PATH = PREDICTION_DIR / "ml_stage4_final_candidate_oof_predictions.csv"
STAGE5_FINAL_LOCKBOX_PATH = PREDICTION_DIR / "ml_stage5_lockbox_final_predictions.csv"
STAGE35_BASE_OOF_PATH = PREDICTION_DIR / "ml_stage35_base_model_oof_predictions.csv"
STAGE5_BASE_LOCKBOX_PATH = PREDICTION_DIR / "ml_stage5_lockbox_base_model_predictions.csv"
STAGE35_MANIFEST_PATH = METRIC_DIR / "ml_stage35_final_model_manifest.csv"
STAGE6_DECISION_PATH = METRIC_DIR / "ml_stage6_final_model_decision.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def md_table(df: pd.DataFrame, columns: list[str] | None = None, *, floatfmt: str = ".5f", max_rows: int | None = None) -> str:
    table = df.copy()
    if columns is not None:
        table = table[[col for col in columns if col in table.columns]]
    if max_rows is not None:
        table = table.head(max_rows)
    try:
        return table.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return "```\n" + table.to_string(index=False) + "\n```"


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def load_final_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = read_csv(STAGE4_FINAL_OOF_PATH)
    oof = oof[(oof["model"] == FINAL_MODEL) & (oof["run_status"] == "OK")].copy()
    oof = oof[oof["calibration_method"].eq("raw")].copy()
    oof = oof[["샘플ID", "Target", "샘플유형", "기후지형유형", "outer_fold", "score_raw"]].rename(columns={"score_raw": "score"})
    oof["dataset"] = "development_oof"

    lockbox = read_csv(STAGE5_FINAL_LOCKBOX_PATH)
    lockbox = lockbox[
        (lockbox["model"] == FINAL_MODEL)
        & (lockbox["selection_role"] == FINAL_ROLE)
        & (lockbox["run_status"] == "OK")
    ].copy()
    lockbox = lockbox[["샘플ID", "Target", "샘플유형", "기후지형유형", "outer_fold", "score_raw"]].rename(
        columns={"score_raw": "score"}
    )
    lockbox["dataset"] = "lockbox_test"
    return oof, lockbox


def decile_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in frame.groupby("dataset", sort=False):
        ordered = part.sort_values("score", ascending=False).reset_index(drop=True)
        ordered["risk_decile"] = np.ceil((np.arange(len(ordered)) + 1) / len(ordered) * 10).astype(int)
        ordered["risk_decile"] = ordered["risk_decile"].clip(1, 10)
        total_pos = int(ordered["Target"].sum())
        base_rate = float(ordered["Target"].mean())
        cumulative_selected = 0
        cumulative_positive = 0
        for decile, group in ordered.groupby("risk_decile", sort=True):
            n = len(group)
            pos = int(group["Target"].sum())
            cumulative_selected += n
            cumulative_positive += pos
            rate = pos / n if n else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "risk_decile": int(decile),
                    "risk_band": f"top_{(decile - 1) * 10:02d}_{decile * 10:02d}pct" if decile > 1 else "top_00_10pct",
                    "n": n,
                    "positive_n": pos,
                    "positive_rate": rate,
                    "score_min": float(group["score"].min()),
                    "score_median": float(group["score"].median()),
                    "score_max": float(group["score"].max()),
                    "base_positive_rate": base_rate,
                    "lift_vs_base": rate / base_rate if base_rate > 0 else float("nan"),
                    "cumulative_selected_n": cumulative_selected,
                    "cumulative_selected_rate": cumulative_selected / len(ordered),
                    "cumulative_positive_n": cumulative_positive,
                    "cumulative_capture_rate": cumulative_positive / total_pos if total_pos else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def add_outcome(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = frame.copy()
    out["predicted_positive"] = out["score"] >= threshold
    out["outcome"] = np.select(
        [
            (out["Target"].eq(1)) & (out["predicted_positive"]),
            (out["Target"].eq(0)) & (out["predicted_positive"]),
            (out["Target"].eq(1)) & (~out["predicted_positive"]),
            (out["Target"].eq(0)) & (~out["predicted_positive"]),
        ],
        ["TP", "FP", "FN", "TN"],
        default="unknown",
    )
    return out


def profile_by(frame: pd.DataFrame, group_cols: list[str], profile_type: str) -> pd.DataFrame:
    rows = []
    total_by_dataset = frame.groupby("dataset").size().to_dict()
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        dataset = row.get("dataset")
        rows.append(
            {
                "profile_type": profile_type,
                **row,
                "n": len(group),
                "share_in_dataset": len(group) / total_by_dataset.get(dataset, len(frame)),
                "positive_n": int(group["Target"].sum()),
                "positive_rate": float(group["Target"].mean()),
                "score_mean": float(group["score"].mean()),
                "score_median": float(group["score"].median()),
                "score_min": float(group["score"].min()),
                "score_max": float(group["score"].max()),
            }
        )
    return pd.DataFrame(rows)


def threshold_error_profile(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified = add_outcome(frame, threshold)
    profiles = [
        profile_by(classified, ["dataset", "outcome"], "outcome_summary"),
        profile_by(classified, ["dataset", "outcome", "샘플유형"], "outcome_by_sample_type"),
        profile_by(classified, ["dataset", "outcome", "기후지형유형"], "outcome_by_climate_type"),
    ]

    error_rows = []
    for dataset, part in classified.groupby("dataset", sort=False):
        fp = part[part["outcome"].eq("FP")].sort_values("score", ascending=False).head(30).copy()
        fp["error_rank_type"] = "false_positive_highest_score"
        fn_low = part[part["outcome"].eq("FN")].sort_values("score", ascending=True).head(30).copy()
        fn_low["error_rank_type"] = "false_negative_lowest_score"
        fn_high = part[part["outcome"].eq("FN")].sort_values("score", ascending=False).head(30).copy()
        fn_high["error_rank_type"] = "false_negative_near_threshold"
        error_rows.append(pd.concat([fp, fn_low, fn_high], ignore_index=True))
    top_errors = pd.concat(error_rows, ignore_index=True)
    top_errors = top_errors[
        ["dataset", "error_rank_type", "샘플ID", "Target", "샘플유형", "기후지형유형", "score", "outcome"]
    ]
    return pd.concat(profiles, ignore_index=True, sort=False), top_errors


def pivot_base_predictions(base_oof: pd.DataFrame, base_lockbox: pd.DataFrame) -> pd.DataFrame:
    oof = base_oof.copy()
    oof["dataset"] = "development_oof"
    oof["base_score"] = oof["score_raw"]
    oof_cols = ["dataset", "샘플ID", "Target", "샘플유형", "기후지형유형", "candidate_id", "base_score"]

    lockbox = base_lockbox.copy()
    lockbox["base_score"] = lockbox["score"]
    lock_cols = ["dataset", "샘플ID", "Target", "샘플유형", "기후지형유형", "candidate_id", "base_score"]

    base = pd.concat([oof[oof_cols], lockbox[lock_cols]], ignore_index=True)
    matrix = base.pivot_table(
        index=["dataset", "샘플ID", "Target", "샘플유형", "기후지형유형"],
        columns="candidate_id",
        values="base_score",
        aggfunc="first",
    ).reset_index()
    matrix.columns.name = None
    return matrix


def base_agreement(base_matrix: pd.DataFrame, final_predictions: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = [col for col in base_matrix.columns if col not in {"dataset", "샘플ID", "Target", "샘플유형", "기후지형유형"}]
    merged = base_matrix.merge(
        final_predictions[["dataset", "샘플ID", "score"]],
        on=["dataset", "샘플ID"],
        how="left",
        validate="one_to_one",
    )
    scores = merged[base_cols]
    merged["base_score_mean"] = scores.mean(axis=1)
    merged["base_score_std"] = scores.std(axis=1)
    merged["base_score_min"] = scores.min(axis=1)
    merged["base_score_max"] = scores.max(axis=1)
    merged["base_score_range"] = merged["base_score_max"] - merged["base_score_min"]
    merged["base_models_above_threshold"] = (scores >= threshold).sum(axis=1)
    merged["final_above_threshold"] = merged["score"] >= threshold

    rows = []
    for (dataset, above_n), group in merged.groupby(["dataset", "base_models_above_threshold"], sort=True):
        rows.append(
            {
                "dataset": dataset,
                "base_models_above_threshold": int(above_n),
                "n": len(group),
                "selected_by_final_n": int(group["final_above_threshold"].sum()),
                "positive_n": int(group["Target"].sum()),
                "positive_rate": float(group["Target"].mean()),
                "final_score_mean": float(group["score"].mean()),
                "final_score_median": float(group["score"].median()),
                "base_score_std_mean": float(group["base_score_std"].mean()),
                "base_score_range_mean": float(group["base_score_range"].mean()),
            }
        )
    agreement = pd.DataFrame(rows)

    corr_rows = []
    for dataset, part in merged.groupby("dataset", sort=False):
        corr = part[base_cols].corr(method="spearman")
        for left in base_cols:
            for right in base_cols:
                if left < right:
                    corr_rows.append(
                        {
                            "dataset": dataset,
                            "candidate_left": left,
                            "candidate_right": right,
                            "spearman_corr": float(corr.loc[left, right]),
                        }
                    )
    correlation = pd.DataFrame(corr_rows)
    return agreement, correlation


def original_feature_name(transformed_feature: str, original_features: list[str]) -> str:
    exact = [feature for feature in original_features if transformed_feature == feature]
    if exact:
        return exact[0]
    candidates = [feature for feature in original_features if transformed_feature.startswith(f"{feature}_")]
    if candidates:
        return max(candidates, key=len)
    return transformed_feature


def feature_domain(feature: str) -> str:
    if "_x_" in feature or "_AND_" in feature:
        return "interaction_rule"
    if feature.startswith("D1_"):
        return "canada_fwi"
    if feature.startswith(("월_", "시간_")):
        return "time"
    if feature.startswith(("고도", "경사도", "사면", "TPI", "log1p_")):
        return "spatial"
    if feature.startswith(("토지피복", "비산림", "산림지역", "침엽수림", "활엽수림", "혼효림", "초지", "시가화", "도로_")):
        return "landcover_rule"
    if feature.startswith(("rh_", "wind_", "dry_spell")):
        return "eda_weather_rule"
    if feature.startswith(("시점_", "직전", "풍향", "서풍", "기압", "D-")):
        return "weather"
    return "other"


def native_feature_importance() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = read_csv(STAGE35_MANIFEST_PATH)
    rows = []
    model_rows = []
    for item in manifest.itertuples(index=False):
        candidate_id = str(item.candidate_id)
        model_name = str(item.model)
        features = str(item.features).split("|")
        path = Path(str(item.model_path))
        if not path.is_absolute():
            path = ML_DIR.parents[2] / path
        try:
            pipeline = joblib.load(path)
            estimator = pipeline.named_steps["model"]
            if not hasattr(estimator, "feature_importances_"):
                model_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "model": model_name,
                        "status": "skipped_no_feature_importances",
                        "n_transformed_features": np.nan,
                        "raw_importance_sum": np.nan,
                    }
                )
                continue
            transformed_features = list(pipeline.named_steps["preprocess"].get_feature_names_out())
            importances = np.asarray(estimator.feature_importances_, dtype=float)
            if len(transformed_features) != len(importances):
                raise ValueError(f"feature length mismatch: {len(transformed_features)} vs {len(importances)}")
            total = float(importances.sum())
            if total <= 0:
                total = 1.0
            model_rows.append(
                {
                    "candidate_id": candidate_id,
                    "model": model_name,
                    "status": "OK",
                    "n_transformed_features": len(transformed_features),
                    "raw_importance_sum": float(importances.sum()),
                }
            )
            temp = pd.DataFrame(
                {
                    "candidate_id": candidate_id,
                    "model": model_name,
                    "transformed_feature": transformed_features,
                    "raw_importance": importances,
                }
            )
            temp["original_feature"] = temp["transformed_feature"].map(lambda name: original_feature_name(str(name), features))
            grouped = (
                temp.groupby(["candidate_id", "model", "original_feature"], as_index=False)
                .agg(raw_importance=("raw_importance", "sum"))
                .assign(normalized_importance=lambda x: x["raw_importance"] / total)
            )
            rows.append(grouped)
        except Exception as exc:
            model_rows.append(
                {
                    "candidate_id": candidate_id,
                    "model": model_name,
                    "status": f"error::{type(exc).__name__}: {exc}",
                    "n_transformed_features": np.nan,
                    "raw_importance_sum": np.nan,
                }
            )

    if rows:
        per_model = pd.concat(rows, ignore_index=True)
        aggregated = (
            per_model.groupby("original_feature", as_index=False)
            .agg(
                mean_normalized_importance=("normalized_importance", "mean"),
                max_normalized_importance=("normalized_importance", "max"),
                raw_importance_sum=("raw_importance", "sum"),
                n_models=("candidate_id", "nunique"),
                contributing_models=("candidate_id", lambda x: "|".join(sorted(set(map(str, x))))),
            )
            .sort_values(["mean_normalized_importance", "max_normalized_importance"], ascending=False)
        )
        aggregated["feature_domain"] = aggregated["original_feature"].map(feature_domain)
        aggregated = aggregated[
            [
                "original_feature",
                "feature_domain",
                "mean_normalized_importance",
                "max_normalized_importance",
                "raw_importance_sum",
                "n_models",
                "contributing_models",
            ]
        ]
    else:
        aggregated = pd.DataFrame(
            columns=[
                "original_feature",
                "feature_domain",
                "mean_normalized_importance",
                "max_normalized_importance",
                "raw_importance_sum",
                "n_models",
                "contributing_models",
            ]
        )
    return aggregated, pd.DataFrame(model_rows)


def build_report() -> None:
    required = [
        STAGE4_FINAL_OOF_PATH,
        STAGE5_FINAL_LOCKBOX_PATH,
        STAGE35_BASE_OOF_PATH,
        STAGE5_BASE_LOCKBOX_PATH,
        STAGE35_MANIFEST_PATH,
        STAGE6_DECISION_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 파일이 없습니다:\n" + "\n".join(missing))

    decision = read_csv(STAGE6_DECISION_PATH).iloc[0]
    threshold = safe_float(decision["recommended_threshold"])
    if math.isnan(threshold):
        raise ValueError("Step6 recommended_threshold를 읽지 못했습니다.")

    oof, lockbox = load_final_predictions()
    final_predictions = pd.concat([oof, lockbox], ignore_index=True, sort=False)
    final_predictions["score"] = pd.to_numeric(final_predictions["score"], errors="coerce")

    deciles = decile_analysis(final_predictions)
    deciles.to_csv(METRIC_DIR / "ml_stage7_score_decile_analysis.csv", index=False, encoding="utf-8-sig")

    error_profile, top_errors = threshold_error_profile(final_predictions, threshold)
    error_profile.to_csv(METRIC_DIR / "ml_stage7_threshold_error_profile.csv", index=False, encoding="utf-8-sig")
    top_errors.to_csv(METRIC_DIR / "ml_stage7_top_error_samples.csv", index=False, encoding="utf-8-sig")

    base_oof = read_csv(STAGE35_BASE_OOF_PATH)
    base_lockbox = read_csv(STAGE5_BASE_LOCKBOX_PATH)
    base_matrix = pivot_base_predictions(base_oof, base_lockbox)
    agreement, correlation = base_agreement(base_matrix, final_predictions, threshold)
    agreement.to_csv(METRIC_DIR / "ml_stage7_base_model_agreement.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(METRIC_DIR / "ml_stage7_base_model_correlation.csv", index=False, encoding="utf-8-sig")

    importance, importance_status = native_feature_importance()
    importance.to_csv(METRIC_DIR / "ml_stage7_native_feature_importance.csv", index=False, encoding="utf-8-sig")
    importance_status.to_csv(METRIC_DIR / "ml_stage7_native_feature_importance_status.csv", index=False, encoding="utf-8-sig")

    lockbox_decile = deciles[deciles["dataset"].eq("lockbox_test")].copy()
    oof_decile = deciles[deciles["dataset"].eq("development_oof")].copy()
    lockbox_top_decile = lockbox_decile[lockbox_decile["risk_decile"].eq(1)].iloc[0]
    lockbox_top3_decile = lockbox_decile[lockbox_decile["risk_decile"].le(3)].iloc[-1]
    lockbox_outcome = error_profile[
        (error_profile["dataset"].eq("lockbox_test")) & (error_profile["profile_type"].eq("outcome_summary"))
    ].copy()
    lockbox_agreement = agreement[agreement["dataset"].eq("lockbox_test")].copy()
    top_importance = importance.head(20).copy()

    validation = pd.DataFrame(
        [
            {"check": "development_oof_rows", "value": len(oof), "expected": 13632, "passed": len(oof) == 13632},
            {"check": "lockbox_rows", "value": len(lockbox), "expected": 3413, "passed": len(lockbox) == 3413},
            {
                "check": "final_score_nan_or_inf",
                "value": int(final_predictions["score"].replace([np.inf, -np.inf], np.nan).isna().sum()),
                "expected": 0,
                "passed": int(final_predictions["score"].replace([np.inf, -np.inf], np.nan).isna().sum()) == 0,
            },
            {
                "check": "base_matrix_missing_scores",
                "value": int(base_matrix.drop(columns=["dataset", "샘플ID", "Target", "샘플유형", "기후지형유형"]).isna().sum().sum()),
                "expected": 0,
                "passed": int(base_matrix.drop(columns=["dataset", "샘플ID", "Target", "샘플유형", "기후지형유형"]).isna().sum().sum()) == 0,
            },
            {
                "check": "recommended_threshold",
                "value": threshold,
                "expected": "finite",
                "passed": math.isfinite(threshold),
            },
            {
                "check": "native_feature_importance_available_models",
                "value": int(importance_status["status"].eq("OK").sum()),
                "expected": ">=1",
                "passed": int(importance_status["status"].eq("OK").sum()) >= 1,
            },
        ]
    )
    validation.to_csv(METRIC_DIR / "ml_stage7_validation_checks.csv", index=False, encoding="utf-8-sig")

    decile_cols = [
        "dataset",
        "risk_decile",
        "n",
        "positive_n",
        "positive_rate",
        "score_min",
        "score_median",
        "score_max",
        "lift_vs_base",
        "cumulative_capture_rate",
    ]
    outcome_cols = ["outcome", "n", "share_in_dataset", "positive_n", "positive_rate", "score_mean", "score_median"]
    agreement_cols = [
        "base_models_above_threshold",
        "n",
        "selected_by_final_n",
        "positive_n",
        "positive_rate",
        "final_score_mean",
        "base_score_std_mean",
        "base_score_range_mean",
    ]
    importance_cols = [
        "original_feature",
        "feature_domain",
        "mean_normalized_importance",
        "max_normalized_importance",
        "n_models",
        "contributing_models",
    ]

    report = f"""# 머신러닝 7차 최종모델 해석/오류분석 결과

## 1. 단계 요약

- 최종 모델 `{FINAL_MODEL} / raw`의 score decile, threshold 오류, base model agreement, native feature importance를 산출했다.
- Step6에서 정한 threshold `{threshold:.5f}`를 그대로 사용했다.
- 새 학습, 추가 튜닝, lockbox 기반 후보 재선택은 하지 않았다.

## 2. Lockbox score decile

lockbox top 10% decile은 positive rate {lockbox_top_decile["positive_rate"]:.4f}, lift {lockbox_top_decile["lift_vs_base"]:.2f}, cumulative capture {lockbox_top_decile["cumulative_capture_rate"]:.4f}를 기록했다. top 30%까지 누적하면 capture {lockbox_top3_decile["cumulative_capture_rate"]:.4f}이다.

{md_table(lockbox_decile, decile_cols)}

## 3. Development OOF score decile

{md_table(oof_decile, decile_cols)}

## 4. Threshold 오류분석

아래 표는 Step6 권장 threshold `{threshold:.5f}`를 적용한 lockbox TP/FP/FN/TN 요약이다.

{md_table(lockbox_outcome, outcome_cols)}

## 5. Base model agreement

5개 base model 중 threshold 이상을 낸 모델 수가 많을수록 최종 score와 양성률이 높아지는지 확인했다. 이 표는 앙상블이 소수 모델의 단독 판단인지, 여러 모델의 합의인지 확인하는 진단용이다.

{md_table(lockbox_agreement, agreement_cols)}

## 6. Native feature importance

아래 중요도는 `feature_importances_`를 제공하는 base model에서 산출한 내부 중요도이다. 인과 효과나 독립적인 변수 영향으로 해석하지 않고, 모델이 분할에 자주 사용한 신호로만 해석한다.

{md_table(top_importance, importance_cols, max_rows=20)}

## 7. 해석

- lockbox 최상위 decile의 양성률과 lift가 높아 top-risk screening에는 최종 score를 직접 사용할 수 있다.
- 권장 threshold는 recall 확보를 우선하는 운영점이므로 FP가 남는다. 보고서에서는 threshold 운영과 top-risk 운영을 분리해 설명하는 것이 적절하다.
- base model agreement가 높을수록 양성률이 높다면, 최종 앙상블은 단일 모델 우연이 아니라 여러 모델이 비슷하게 위험을 본 샘플에서 강하게 반응한다고 해석할 수 있다.
- native importance는 거리/기상/EDA 파생/토지피복 규칙이 함께 쓰였는지 확인하는 보조 근거다.

## 8. 검증

{md_table(validation)}

## 9. 산출물

- `outputs/metrics/ml_stage7_score_decile_analysis.csv`
- `outputs/metrics/ml_stage7_threshold_error_profile.csv`
- `outputs/metrics/ml_stage7_base_model_agreement.csv`
- `outputs/metrics/ml_stage7_base_model_correlation.csv`
- `outputs/metrics/ml_stage7_native_feature_importance.csv`
- `outputs/metrics/ml_stage7_top_error_samples.csv`
- `outputs/metrics/ml_stage7_validation_checks.csv`
- `outputs/ml_stage7_final_model_diagnostics.md`
- `머신러닝_7차_최종모델_해석_결과.md`
"""
    (OUTPUT_DIR / "ml_stage7_final_model_diagnostics.md").write_text(report, encoding="utf-8")
    (ML_DIR / "머신러닝_7차_최종모델_해석_결과.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_report()
