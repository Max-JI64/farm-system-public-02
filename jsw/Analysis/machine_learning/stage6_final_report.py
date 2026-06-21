from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ML_DIR = Path(__file__).resolve().parent
ROOT = ML_DIR.parents[2]
OUTPUT_DIR = ML_DIR / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"
LOGISTIC_METRIC_PATH = ML_DIR.parent / "logistic" / "outputs" / "tables" / "stage17_ml_comparison_metrics.csv"

FINAL_PROB_MODEL = "model_level_top5_geometric_mean"
RANK_DIAGNOSTIC_MODEL = "model_level_top5_rank_average"


INPUT_PATHS = {
    "logistic_stage17": LOGISTIC_METRIC_PATH,
    "stage1_model_comparison": METRIC_DIR / "ml_stage1_v2_all_model_comparison.csv",
    "stage2_model_comparison": METRIC_DIR / "ml_stage2_model_comparison.csv",
    "stage4_candidate_selection": METRIC_DIR / "ml_stage4_final_candidate_selection.csv",
    "stage4_thresholds": METRIC_DIR / "ml_stage4_final_thresholds.csv",
    "stage4_top_risk": METRIC_DIR / "ml_stage4_final_top_risk_capture.csv",
    "stage4_subgroup": METRIC_DIR / "ml_stage4_final_subgroup_metrics.csv",
    "stage5_lockbox_model_comparison": METRIC_DIR / "ml_stage5_lockbox_model_comparison.csv",
    "stage5_oof_vs_lockbox": METRIC_DIR / "ml_stage5_oof_vs_lockbox_comparison.csv",
    "stage5_thresholds": METRIC_DIR / "ml_stage5_lockbox_fixed_thresholds.csv",
    "stage5_top_risk": METRIC_DIR / "ml_stage5_lockbox_top_risk_capture.csv",
    "stage5_subgroup": METRIC_DIR / "ml_stage5_lockbox_subgroup_metrics.csv",
    "stage5_base_model_metrics": METRIC_DIR / "ml_stage5_lockbox_base_model_metrics.csv",
    "stage5_validation": METRIC_DIR / "ml_stage5_validation_checks.csv",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def ensure_inputs() -> None:
    missing = [str(path) for path in INPUT_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 파일이 없습니다:\n" + "\n".join(missing))


def finite_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isinf(number):
        return float("nan")
    return number


def pick_best(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.Series:
    candidate = df.copy()
    if filters:
        for col, value in filters.items():
            if col in candidate.columns:
                candidate = candidate[candidate[col] == value]
    if "run_status" in candidate.columns:
        candidate = candidate[candidate["run_status"] == "OK"]
    if candidate.empty:
        raise ValueError(f"조건에 맞는 후보가 없습니다: {filters}")
    return candidate.sort_values(["auprc", "auroc"], ascending=False).iloc[0]


def pick_one(df: pd.DataFrame, **filters: Any) -> pd.Series:
    candidate = df.copy()
    for col, value in filters.items():
        if col not in candidate.columns:
            raise KeyError(f"컬럼이 없습니다: {col}")
        candidate = candidate[candidate[col] == value]
    if candidate.empty:
        raise ValueError(f"조건에 맞는 행이 없습니다: {filters}")
    return candidate.iloc[0]


def metric_value(row: pd.Series, name: str) -> float:
    return finite_or_nan(row[name]) if name in row.index else float("nan")


def add_metric_row(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    source_stage: str,
    model_role: str,
    row: pd.Series,
    feature_set: str | None = None,
    model: str | None = None,
    score_type: str | None = None,
    selected_for_final: bool = False,
    note: str = "",
    logistic_auprc: float,
    stage1_best_auprc: float,
) -> None:
    model_name = model if model is not None else str(row.get("model", ""))
    feature_name = feature_set if feature_set is not None else str(row.get("feature_set", ""))
    score_name = score_type if score_type is not None else str(row.get("score_type", ""))
    auprc = metric_value(row, "auprc")
    rows.append(
        {
            "dataset": dataset,
            "source_stage": source_stage,
            "model_role": model_role,
            "feature_set": feature_name,
            "model": model_name,
            "score_type": score_name,
            "n": metric_value(row, "n"),
            "positive_n": metric_value(row, "positive_n"),
            "positive_rate": metric_value(row, "positive_rate"),
            "auprc": auprc,
            "auroc": metric_value(row, "auroc"),
            "brier": metric_value(row, "brier"),
            "log_loss": metric_value(row, "log_loss"),
            "best_f1_precision": metric_value(row, "best_f1_precision"),
            "best_f1_recall": metric_value(row, "best_f1_recall"),
            "best_f1_f1": metric_value(row, "best_f1_f1"),
            "delta_auprc_vs_logistic_oof_baseline": auprc - logistic_auprc,
            "delta_auprc_vs_stage1_single_ml_oof": auprc - stage1_best_auprc,
            "selected_for_final_report": selected_for_final,
            "note": note,
        }
    )


def normalize_stage4_detail(df: pd.DataFrame, selection: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    role_map = dict(zip(selection["model"], selection["selection_role"]))
    out.insert(0, "dataset", dataset)
    out.insert(3, "selection_role", out["model"].map(role_map).fillna(""))
    return out


def with_source_stage(df: pd.DataFrame, source_stage: str) -> pd.DataFrame:
    out = df.copy()
    if "source_stage" not in out.columns:
        out.insert(0, "source_stage", source_stage)
    return out


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


def bool_passed(series: pd.Series) -> bool:
    normalized = series.astype(str).str.lower().str.strip()
    return bool(normalized.isin(["true", "1", "yes"]).all())


def build_report() -> None:
    ensure_inputs()
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logistic = read_csv(INPUT_PATHS["logistic_stage17"])
    stage1 = read_csv(INPUT_PATHS["stage1_model_comparison"])
    stage2 = read_csv(INPUT_PATHS["stage2_model_comparison"])
    stage4_selection = read_csv(INPUT_PATHS["stage4_candidate_selection"])
    stage4_thresholds = read_csv(INPUT_PATHS["stage4_thresholds"])
    stage4_top_risk = read_csv(INPUT_PATHS["stage4_top_risk"])
    stage4_subgroup = read_csv(INPUT_PATHS["stage4_subgroup"])
    stage5_models = read_csv(INPUT_PATHS["stage5_lockbox_model_comparison"])
    stage5_oof_lockbox = read_csv(INPUT_PATHS["stage5_oof_vs_lockbox"])
    stage5_thresholds = read_csv(INPUT_PATHS["stage5_thresholds"])
    stage5_top_risk = read_csv(INPUT_PATHS["stage5_top_risk"])
    stage5_subgroup = read_csv(INPUT_PATHS["stage5_subgroup"])
    stage5_base = read_csv(INPUT_PATHS["stage5_base_model_metrics"])
    stage5_validation = read_csv(INPUT_PATHS["stage5_validation"])

    logistic_baseline = pick_one(logistic, model="PLUS_LANDCOVER_RULES_ANOVA")
    logistic_interpret = pick_one(logistic, model="FINAL_REDUCED_WITH_FWI")
    stage1_best = pick_best(stage1, {"score_type": "raw"})
    stage2_best = pick_best(stage2)
    stage4_prob = pick_one(stage4_selection, selection_role="probability_score")
    stage4_rank = pick_one(stage4_selection, selection_role="ranking_score")
    stage5_prob = pick_one(stage5_models, selection_role="probability_score", model=FINAL_PROB_MODEL)
    stage5_rank = pick_one(stage5_models, selection_role="ranking_score", model=RANK_DIAGNOSTIC_MODEL)
    stage5_best_base = pick_best(stage5_base)

    logistic_auprc = metric_value(logistic_baseline, "auprc")
    stage1_best_auprc = metric_value(stage1_best, "auprc")

    rows: list[dict[str, Any]] = []
    add_metric_row(
        rows,
        dataset="development_oof",
        source_stage="logistic_stage17",
        model_role="logistic_oof_baseline_for_ml",
        row=logistic_baseline,
        score_type="probability",
        note="ML 비교 기준선으로 사용한 로지스틱 OOF 성능",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )
    add_metric_row(
        rows,
        dataset="development_oof",
        source_stage="logistic_stage17",
        model_role="logistic_interpretation_model",
        row=logistic_interpret,
        score_type="probability",
        note="최종 해석용 로지스틱 모델. ML 비교 기준선은 아님",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )
    add_metric_row(
        rows,
        dataset="development_oof",
        source_stage="stage1_screening",
        model_role="best_single_ml_oof",
        row=stage1_best,
        note="1차 단일 ML 최고 성능",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )
    add_metric_row(
        rows,
        dataset="development_oof",
        source_stage="stage2_optuna",
        model_role="best_stage2_optuna_oof_not_adopted",
        row=stage2_best,
        note="Optuna 튜닝 최고 후보지만 1차 최고보다 낮아 최종 후보에서 제외",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )
    add_metric_row(
        rows,
        dataset="development_oof",
        source_stage="stage4_final_selection",
        model_role="final_probability_score_oof",
        row=stage4_prob,
        selected_for_final=True,
        note="최종 보고서/운영 threshold용 probability score",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )
    add_metric_row(
        rows,
        dataset="development_oof",
        source_stage="stage4_final_selection",
        model_role="rank_diagnostic_oof",
        row=stage4_rank,
        note="순위화 진단용 score. 확률 지표로 해석하지 않음",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )
    add_metric_row(
        rows,
        dataset="lockbox_test",
        source_stage="stage5_lockbox",
        model_role="final_probability_score_lockbox",
        row=stage5_prob,
        selected_for_final=True,
        note="Step4에서 고정한 probability score의 lockbox 최초 평가",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )
    add_metric_row(
        rows,
        dataset="lockbox_test",
        source_stage="stage5_lockbox",
        model_role="rank_diagnostic_lockbox",
        row=stage5_rank,
        note="Step4에서 고정한 ranking score의 lockbox 최초 평가",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )
    add_metric_row(
        rows,
        dataset="lockbox_test",
        source_stage="stage5_lockbox_base",
        model_role="best_single_base_model_lockbox_reference",
        row=stage5_best_base,
        feature_set=str(stage5_best_base["feature_set"]),
        model=f"{stage5_best_base['candidate_id']}::{stage5_best_base['model']}",
        score_type="raw",
        note="앙상블 구성 base model 중 lockbox AUPRC 최고. 최종 선택 기준은 아님",
        logistic_auprc=logistic_auprc,
        stage1_best_auprc=stage1_best_auprc,
    )

    final_comparison = pd.DataFrame(rows)
    final_comparison.to_csv(METRIC_DIR / "ml_stage6_final_model_comparison.csv", index=False, encoding="utf-8-sig")

    stage4_thresholds_norm = with_source_stage(
        normalize_stage4_detail(stage4_thresholds, stage4_selection, "development_oof"),
        "stage4_final_selection",
    )
    stage5_thresholds_norm = with_source_stage(stage5_thresholds, "stage5_lockbox")
    operating_points = pd.concat([stage4_thresholds_norm, stage5_thresholds_norm], ignore_index=True, sort=False)
    operating_points.to_csv(METRIC_DIR / "ml_stage6_final_operating_points.csv", index=False, encoding="utf-8-sig")

    stage4_top_risk_norm = with_source_stage(
        normalize_stage4_detail(stage4_top_risk, stage4_selection, "development_oof"),
        "stage4_final_selection",
    )
    stage5_top_risk_norm = with_source_stage(stage5_top_risk, "stage5_lockbox")
    top_risk = pd.concat([stage4_top_risk_norm, stage5_top_risk_norm], ignore_index=True, sort=False)
    top_risk.to_csv(METRIC_DIR / "ml_stage6_final_top_risk_capture.csv", index=False, encoding="utf-8-sig")

    stage4_subgroup_norm = with_source_stage(
        normalize_stage4_detail(stage4_subgroup, stage4_selection, "development_oof"),
        "stage4_final_selection",
    )
    stage5_subgroup_norm = with_source_stage(stage5_subgroup, "stage5_lockbox")
    subgroup = pd.concat([stage4_subgroup_norm, stage5_subgroup_norm], ignore_index=True, sort=False)
    subgroup.to_csv(METRIC_DIR / "ml_stage6_final_subgroup_metrics.csv", index=False, encoding="utf-8-sig")

    lockbox_recall70 = pick_one(
        stage5_thresholds,
        model=FINAL_PROB_MODEL,
        selection_role="probability_score",
        operating_point="recall_ge_0.70",
    )
    lockbox_top10 = pick_one(
        stage5_top_risk,
        model=FINAL_PROB_MODEL,
        selection_role="probability_score",
        top_pct=0.1,
    )

    decision = pd.DataFrame(
        [
            {
                "final_model": FINAL_PROB_MODEL,
                "final_score_type": "raw",
                "final_role": "probability_score",
                "development_oof_auprc": metric_value(stage4_prob, "auprc"),
                "development_oof_auroc": metric_value(stage4_prob, "auroc"),
                "development_oof_brier": metric_value(stage4_prob, "brier"),
                "development_oof_log_loss": metric_value(stage4_prob, "log_loss"),
                "lockbox_auprc": metric_value(stage5_prob, "auprc"),
                "lockbox_auroc": metric_value(stage5_prob, "auroc"),
                "lockbox_brier": metric_value(stage5_prob, "brier"),
                "lockbox_log_loss": metric_value(stage5_prob, "log_loss"),
                "delta_oof_auprc_vs_logistic_baseline": metric_value(stage4_prob, "auprc") - logistic_auprc,
                "delta_oof_auprc_vs_stage1_best": metric_value(stage4_prob, "auprc") - stage1_best_auprc,
                "delta_lockbox_auprc_vs_logistic_oof_baseline": metric_value(stage5_prob, "auprc") - logistic_auprc,
                "recommended_operating_point": "recall_ge_0.70",
                "recommended_threshold": metric_value(lockbox_recall70, "threshold"),
                "recommended_precision": metric_value(lockbox_recall70, "precision"),
                "recommended_recall": metric_value(lockbox_recall70, "recall"),
                "recommended_selected_rate": metric_value(lockbox_recall70, "selected_rate"),
                "top10_capture_rate": metric_value(lockbox_top10, "capture_rate_recall"),
                "top10_precision": metric_value(lockbox_top10, "precision"),
                "ranking_diagnostic_model": RANK_DIAGNOSTIC_MODEL,
                "selection_statement": "최종 보고서/운영 threshold 모델은 probability score를 사용하고, rank-average는 순위 진단 보조로만 유지한다.",
            }
        ]
    )
    decision.to_csv(METRIC_DIR / "ml_stage6_final_model_decision.csv", index=False, encoding="utf-8-sig")

    required_final = final_comparison[
        (final_comparison["model"] == FINAL_PROB_MODEL) & (final_comparison["selected_for_final_report"])
    ]
    required_rank = final_comparison[final_comparison["model"] == RANK_DIAGNOSTIC_MODEL]
    core_metric_cols = ["auprc", "auroc", "brier", "log_loss"]
    core_metrics = final_comparison[core_metric_cols].replace([np.inf, -np.inf], np.nan)
    validation = pd.DataFrame(
        [
            {"check": "required_input_files", "value": len(INPUT_PATHS), "expected": len(INPUT_PATHS), "passed": True},
            {
                "check": "final_probability_rows_development_and_lockbox",
                "value": len(required_final),
                "expected": 2,
                "passed": len(required_final) == 2,
            },
            {
                "check": "rank_diagnostic_rows_development_and_lockbox",
                "value": len(required_rank),
                "expected": 2,
                "passed": len(required_rank) == 2,
            },
            {
                "check": "stage5_validation_all_passed",
                "value": int(bool_passed(stage5_validation["passed"])),
                "expected": 1,
                "passed": bool_passed(stage5_validation["passed"]),
            },
            {
                "check": "core_metric_nan_or_inf",
                "value": int(core_metrics.isna().sum().sum()),
                "expected": 0,
                "passed": int(core_metrics.isna().sum().sum()) == 0,
            },
            {
                "check": "recommended_threshold_exists",
                "value": metric_value(lockbox_recall70, "threshold"),
                "expected": "finite",
                "passed": not math.isnan(metric_value(lockbox_recall70, "threshold")),
            },
        ]
    )
    validation.to_csv(METRIC_DIR / "ml_stage6_validation_checks.csv", index=False, encoding="utf-8-sig")

    prob_operating = operating_points[
        (operating_points["model"] == FINAL_PROB_MODEL)
        & (operating_points["selection_role"] == "probability_score")
        & (operating_points["dataset"] == "lockbox_test")
    ].copy()
    prob_top_risk = top_risk[
        (top_risk["model"] == FINAL_PROB_MODEL)
        & (top_risk["selection_role"] == "probability_score")
        & (top_risk["dataset"] == "lockbox_test")
    ].copy()
    prob_hard_negative = subgroup[
        (subgroup["model"] == FINAL_PROB_MODEL)
        & (subgroup["selection_role"] == "probability_score")
        & (subgroup["dataset"] == "lockbox_test")
        & (subgroup["subgroup_type"] == "negative_type")
    ].copy()

    comparison_cols = [
        "dataset",
        "source_stage",
        "model_role",
        "model",
        "auprc",
        "auroc",
        "brier",
        "log_loss",
        "best_f1_precision",
        "best_f1_recall",
        "best_f1_f1",
        "delta_auprc_vs_logistic_oof_baseline",
        "delta_auprc_vs_stage1_single_ml_oof",
    ]
    operating_cols = [
        "operating_point",
        "threshold",
        "selected_n",
        "selected_rate",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
        "mcc",
    ]
    top_risk_cols = [
        "top_pct",
        "selected_n",
        "selected_rate",
        "captured_positive_n",
        "total_positive_n",
        "capture_rate_recall",
        "precision",
        "lift_vs_base",
    ]
    subgroup_cols = ["subgroup", "n", "positive_n", "positive_rate", "auprc", "auroc", "brier", "log_loss"]

    report = f"""# 머신러닝 6차 최종보고서 정리 결과

## 1. 단계 요약

- Step6는 새 모델을 학습하지 않고, Step1~5에서 산출한 성능표를 최종 보고서용으로 정리했다.
- 최종 보고서/운영 threshold 모델은 `{FINAL_PROB_MODEL} / raw`로 고정한다.
- `{RANK_DIAGNOSTIC_MODEL} / raw`는 순위화 진단 보조 결과로만 유지한다.
- lockbox test는 Step5에서 이미 최초 평가했고, Step6에서 추가 선택이나 튜닝에 사용하지 않았다.

## 2. 최종 비교표

{md_table(final_comparison, comparison_cols)}

## 3. 최종 선택

최종 모델은 `{FINAL_PROB_MODEL} / raw`이다. development OOF에서는 AUPRC {metric_value(stage4_prob, "auprc"):.4f}, ROC AUC {metric_value(stage4_prob, "auroc"):.4f}, Brier {metric_value(stage4_prob, "brier"):.5f}, log loss {metric_value(stage4_prob, "log_loss"):.5f}를 기록했다. 이는 로지스틱 OOF 기준선 AUPRC {logistic_auprc:.4f} 대비 +{metric_value(stage4_prob, "auprc") - logistic_auprc:.4f}, 1차 단일 ML 최고 AUPRC {stage1_best_auprc:.4f} 대비 +{metric_value(stage4_prob, "auprc") - stage1_best_auprc:.4f}이다.

lockbox에서는 AUPRC {metric_value(stage5_prob, "auprc"):.4f}, ROC AUC {metric_value(stage5_prob, "auroc"):.4f}, Brier {metric_value(stage5_prob, "brier"):.5f}, log loss {metric_value(stage5_prob, "log_loss"):.5f}를 기록했다. lockbox 성능이 development OOF보다 높게 나왔기 때문에, 이 결과는 고정 후보가 lockbox에서 강한 성능을 유지했다는 의미로만 해석하고 lockbox 재튜닝 근거로 쓰지 않는다.

## 4. 운영 threshold 후보

화재위험 스크리닝 목적에서는 `recall_ge_0.70` 운영점을 기본 후보로 둔다. lockbox 기준 threshold {metric_value(lockbox_recall70, "threshold"):.5f}, precision {metric_value(lockbox_recall70, "precision"):.4f}, recall {metric_value(lockbox_recall70, "recall"):.4f}, selected rate {metric_value(lockbox_recall70, "selected_rate"):.4f}이다.

{md_table(prob_operating, operating_cols)}

## 5. Top-risk capture

최종 probability score는 lockbox top 10%에서 양성 {int(metric_value(lockbox_top10, "captured_positive_n"))}/{int(metric_value(lockbox_top10, "total_positive_n"))}개를 포착했고, capture rate {metric_value(lockbox_top10, "capture_rate_recall"):.4f}, precision {metric_value(lockbox_top10, "precision"):.4f}를 기록했다.

{md_table(prob_top_risk, top_risk_cols)}

## 6. Hard-negative subgroup

{md_table(prob_hard_negative, subgroup_cols)}

## 7. OOF와 lockbox 비교

{md_table(stage5_oof_lockbox)}

## 8. 보고서 작성 시 주의점

- 로지스틱 기준선은 development OOF 기준이고, lockbox 로지스틱 성능표가 아니므로 lockbox ML 성능과 직접 동일 테스트셋 비교로 표현하지 않는다.
- 2차 Optuna 최고 후보는 1차 단일 ML 최고보다 낮았으므로 최종 후보로 채택하지 않았다.
- Step3 OOF blending은 진단 단계였고, Step3.5에서 실제 model-level ensemble을 재학습 가능한 형태로 구성했다.
- lockbox 결과는 최초 평가 결과이며, 이후 threshold 해석은 가능하지만 후보 재선택이나 하이퍼파라미터 튜닝에는 사용하지 않는다.

## 9. 검증

{md_table(validation)}

## 10. 산출물

- `outputs/metrics/ml_stage6_final_model_comparison.csv`
- `outputs/metrics/ml_stage6_final_operating_points.csv`
- `outputs/metrics/ml_stage6_final_top_risk_capture.csv`
- `outputs/metrics/ml_stage6_final_subgroup_metrics.csv`
- `outputs/metrics/ml_stage6_final_model_decision.csv`
- `outputs/metrics/ml_stage6_validation_checks.csv`
- `outputs/ml_stage6_final_report.md`
- `머신러닝_6차_최종보고서_정리_결과.md`
"""

    (OUTPUT_DIR / "ml_stage6_final_report.md").write_text(report, encoding="utf-8")
    (ML_DIR / "머신러닝_6차_최종보고서_정리_결과.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_report()
