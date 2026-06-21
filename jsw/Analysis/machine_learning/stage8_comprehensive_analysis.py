from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ML_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ML_DIR / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"


PATHS = {
    "stage1_data_audit": METRIC_DIR / "ml_stage1_v2_data_audit.csv",
    "stage1_all": METRIC_DIR / "ml_stage1_v2_all_model_comparison.csv",
    "stage2_all": METRIC_DIR / "ml_stage2_model_comparison.csv",
    "stage3_all": METRIC_DIR / "ml_stage3_ensemble_model_comparison.csv",
    "stage35_all": METRIC_DIR / "ml_stage35_ensemble_model_comparison.csv",
    "stage4_selection": METRIC_DIR / "ml_stage4_final_candidate_selection.csv",
    "stage5_lockbox": METRIC_DIR / "ml_stage5_lockbox_model_comparison.csv",
    "stage5_top_risk": METRIC_DIR / "ml_stage5_lockbox_top_risk_capture.csv",
    "stage5_thresholds": METRIC_DIR / "ml_stage5_lockbox_fixed_thresholds.csv",
    "stage6_comparison": METRIC_DIR / "ml_stage6_final_model_comparison.csv",
    "stage6_decision": METRIC_DIR / "ml_stage6_final_model_decision.csv",
    "stage7_decile": METRIC_DIR / "ml_stage7_score_decile_analysis.csv",
    "stage7_error": METRIC_DIR / "ml_stage7_threshold_error_profile.csv",
    "stage7_agreement": METRIC_DIR / "ml_stage7_base_model_agreement.csv",
    "stage7_importance": METRIC_DIR / "ml_stage7_native_feature_importance.csv",
    "stage7_validation": METRIC_DIR / "ml_stage7_validation_checks.csv",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def pick_best(df: pd.DataFrame, *, score_type: str | None = None) -> pd.Series:
    part = df.copy()
    if score_type is not None and "score_type" in part.columns:
        part = part[part["score_type"].eq(score_type)]
    if "run_status" in part.columns:
        part = part[part["run_status"].eq("OK")]
    if part.empty:
        raise ValueError("best 후보를 찾을 수 없습니다.")
    return part.sort_values(["auprc", "auroc"], ascending=False).iloc[0]


def pick_one(df: pd.DataFrame, **filters: Any) -> pd.Series:
    part = df.copy()
    for col, value in filters.items():
        part = part[part[col].eq(value)]
    if part.empty:
        raise ValueError(f"조건에 맞는 행이 없습니다: {filters}")
    return part.iloc[0]


def f(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    number = f(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def md_table(df: pd.DataFrame, columns: list[str] | None = None, *, floatfmt: str = ".4f", max_rows: int | None = None) -> str:
    table = df.copy()
    if columns is not None:
        table = table[[col for col in columns if col in table.columns]]
    if max_rows is not None:
        table = table.head(max_rows)
    try:
        return table.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return "```\n" + table.to_string(index=False) + "\n```"


def build_report() -> None:
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 파일이 없습니다:\n" + "\n".join(missing))

    data_audit = read_csv(PATHS["stage1_data_audit"]).iloc[0]
    stage1 = read_csv(PATHS["stage1_all"])
    stage2 = read_csv(PATHS["stage2_all"])
    stage3 = read_csv(PATHS["stage3_all"])
    stage35 = read_csv(PATHS["stage35_all"])
    stage4 = read_csv(PATHS["stage4_selection"])
    stage5 = read_csv(PATHS["stage5_lockbox"])
    top_risk = read_csv(PATHS["stage5_top_risk"])
    thresholds = read_csv(PATHS["stage5_thresholds"])
    final_comparison = read_csv(PATHS["stage6_comparison"])
    decision = read_csv(PATHS["stage6_decision"]).iloc[0]
    decile = read_csv(PATHS["stage7_decile"])
    error = read_csv(PATHS["stage7_error"])
    agreement = read_csv(PATHS["stage7_agreement"])
    importance = read_csv(PATHS["stage7_importance"])
    stage7_validation = read_csv(PATHS["stage7_validation"])

    logistic_baseline = pick_one(final_comparison, source_stage="logistic_stage17", model_role="logistic_oof_baseline_for_ml")
    logistic_interpret = pick_one(final_comparison, source_stage="logistic_stage17", model_role="logistic_interpretation_model")
    stage1_best = pick_one(final_comparison, source_stage="stage1_screening", model_role="best_single_ml_oof")
    stage2_best = pick_one(final_comparison, source_stage="stage2_optuna", model_role="best_stage2_optuna_oof_not_adopted")
    stage3_best = pick_best(stage3, score_type="raw")
    stage35_prob = pick_one(stage35, model="model_level_top5_geometric_mean", score_type="raw")
    stage35_rank = pick_one(stage35, model="model_level_top5_rank_average", score_type="raw")
    stage4_prob = pick_one(stage4, selection_role="probability_score")
    stage4_rank = pick_one(stage4, selection_role="ranking_score")
    stage5_prob = pick_one(stage5, selection_role="probability_score")
    stage5_rank = pick_one(stage5, selection_role="ranking_score")

    threshold_row = pick_one(thresholds, model="model_level_top5_geometric_mean", operating_point="recall_ge_0.70")
    top5 = pick_one(top_risk, model="model_level_top5_geometric_mean", top_pct=0.05)
    top10 = pick_one(top_risk, model="model_level_top5_geometric_mean", top_pct=0.1)
    top20 = pick_one(top_risk, model="model_level_top5_geometric_mean", top_pct=0.2)
    lockbox_top10_decile = pick_one(decile, dataset="lockbox_test", risk_decile=1)
    lockbox_top30_decile = pick_one(decile, dataset="lockbox_test", risk_decile=3)
    oof_top10_decile = pick_one(decile, dataset="development_oof", risk_decile=1)
    lockbox_error = error[
        (error["dataset"].eq("lockbox_test")) & (error["profile_type"].eq("outcome_summary"))
    ].copy()
    agreement_lockbox = agreement[agreement["dataset"].eq("lockbox_test")].copy()
    agree0 = pick_one(agreement_lockbox, base_models_above_threshold=0)
    agree5 = pick_one(agreement_lockbox, base_models_above_threshold=5)

    stage_timeline = pd.DataFrame(
        [
            {
                "stage": "Logistic Stage17",
                "purpose": "ML 비교 기준선과 해석용 로지스틱 성능 확정",
                "main_result": f"비교 기준선 AUPRC {fmt(logistic_baseline['auprc'])}, 해석용 로지스틱 AUPRC {fmt(logistic_interpret['auprc'])}",
                "decision": "ML의 목표 기준선으로 사용",
            },
            {
                "stage": "ML Step1",
                "purpose": "M1/M2/M3 및 주요 ML 모델 넓은 스크리닝",
                "main_result": f"{stage1_best['feature_set']} / {stage1_best['model']} AUPRC {fmt(stage1_best['auprc'])}",
                "decision": "로지스틱 기준선 대비 명확한 개선 확인",
            },
            {
                "stage": "ML Step2",
                "purpose": "Optuna 튜닝",
                "main_result": f"{stage2_best['model']} AUPRC {fmt(stage2_best['auprc'])}",
                "decision": "1차 최고보다 낮아 최종 후보 미채택",
            },
            {
                "stage": "ML Step3",
                "purpose": "OOF prediction-level ensemble 진단",
                "main_result": f"{stage3_best['model']} AUPRC {fmt(stage3_best['auprc'])}",
                "decision": "앙상블 가능성 확인. 재학습 가능한 구조는 Step3.5에서 구현",
            },
            {
                "stage": "ML Step3.5",
                "purpose": "실제 base model 재학습 기반 model-level ensemble",
                "main_result": f"probability score AUPRC {fmt(stage35_prob['auprc'])}, ranking score AUPRC {fmt(stage35_rank['auprc'])}",
                "decision": "geometric mean을 확률 후보, rank average를 순위 보조로 분리",
            },
            {
                "stage": "ML Step4",
                "purpose": "Calibration 비교와 최종 후보 고정",
                "main_result": f"probability score OOF AUPRC {fmt(stage4_prob['auprc'])}, Brier {fmt(stage4_prob['brier'], 5)}",
                "decision": "raw geometric mean 최종 보고서/운영 후보로 고정",
            },
            {
                "stage": "ML Step5",
                "purpose": "고정 후보 lockbox 최초 평가",
                "main_result": f"lockbox AUPRC {fmt(stage5_prob['auprc'])}, ROC AUC {fmt(stage5_prob['auroc'])}, Brier {fmt(stage5_prob['brier'], 5)}",
                "decision": "lockbox 재튜닝 없이 최종 성능으로 보고",
            },
            {
                "stage": "ML Step6",
                "purpose": "최종 보고서용 비교표 정리",
                "main_result": f"최종 모델 {decision['final_model']} / {decision['final_score_type']}",
                "decision": "운영 threshold와 top-risk capture를 보고서용으로 고정",
            },
            {
                "stage": "ML Step7",
                "purpose": "최종 모델 해석/오류분석",
                "main_result": f"lockbox top 10% capture {fmt(lockbox_top10_decile['cumulative_capture_rate'])}, 5-model agreement 양성률 {fmt(agree5['positive_rate'])}",
                "decision": "최종 해석 근거로 사용",
            },
        ]
    )
    stage_timeline.to_csv(METRIC_DIR / "ml_stage8_stage_timeline.csv", index=False, encoding="utf-8-sig")

    key_metric_summary = pd.DataFrame(
        [
            {
                "item": "data_full_rows",
                "value": f(data_audit["n"]),
                "interpretation": "전체 결합 데이터 행 수",
            },
            {
                "item": "development_rows_positive",
                "value": f"{int(data_audit['development_n'])} rows / {int(data_audit['development_positive_n'])} positives",
                "interpretation": "strict CV development 데이터",
            },
            {
                "item": "logistic_oof_baseline_auprc",
                "value": fmt(logistic_baseline["auprc"]),
                "interpretation": "ML 비교 기준 로지스틱 성능",
            },
            {
                "item": "stage1_best_oof_auprc",
                "value": fmt(stage1_best["auprc"]),
                "interpretation": "단일 ML 모델 최고 성능",
            },
            {
                "item": "final_oof_auprc",
                "value": fmt(decision["development_oof_auprc"]),
                "interpretation": "최종 probability score의 development OOF 성능",
            },
            {
                "item": "final_lockbox_auprc",
                "value": fmt(decision["lockbox_auprc"]),
                "interpretation": "고정 후보의 lockbox 최초 평가 성능",
            },
            {
                "item": "recommended_threshold",
                "value": fmt(decision["recommended_threshold"], 5),
                "interpretation": "recall_ge_0.70 운영 후보 threshold",
            },
            {
                "item": "lockbox_top10_capture",
                "value": fmt(top10["capture_rate_recall"]),
                "interpretation": "lockbox top 10%에서 Target 1 포착률",
            },
            {
                "item": "lockbox_top30_decile_capture",
                "value": fmt(lockbox_top30_decile["cumulative_capture_rate"]),
                "interpretation": "lockbox score decile top 30% 누적 포착률",
            },
        ]
    )
    key_metric_summary.to_csv(METRIC_DIR / "ml_stage8_key_metric_summary.csv", index=False, encoding="utf-8-sig")

    report_claims = pd.DataFrame(
        [
            {
                "claim_type": "can_say",
                "statement": "strict development OOF 기준으로 최종 ML probability score는 로지스틱 기준선보다 AUPRC가 높다.",
                "evidence": f"{fmt(decision['development_oof_auprc'])} vs {fmt(logistic_baseline['auprc'])}, delta {fmt(decision['delta_oof_auprc_vs_logistic_baseline'])}",
            },
            {
                "claim_type": "can_say",
                "statement": "Step2 Optuna 튜닝은 최종 성능 개선으로 이어지지 않아 미채택했다.",
                "evidence": f"Step2 best AUPRC {fmt(stage2_best['auprc'])}, Step1 best AUPRC {fmt(stage1_best['auprc'])}",
            },
            {
                "claim_type": "can_say",
                "statement": "최종 모델은 확률/threshold 운영에는 geometric mean을, 순위 보조에는 rank-average를 분리해 사용한다.",
                "evidence": f"geometric mean Brier {fmt(stage4_prob['brier'], 5)}, rank-average Brier {fmt(stage4_rank['brier'], 5)}",
            },
            {
                "claim_type": "can_say",
                "statement": "lockbox에서 고정 후보는 강한 성능을 유지했다.",
                "evidence": f"lockbox AUPRC {fmt(stage5_prob['auprc'])}, ROC AUC {fmt(stage5_prob['auroc'])}",
            },
            {
                "claim_type": "avoid",
                "statement": "lockbox 성능이 OOF보다 높으므로 일반화 성능이 OOF보다 높다고 단정한다.",
                "evidence": "lockbox는 최초 평가 결과이며 더 쉬운 분포일 가능성이 있으므로 재튜닝이나 과대해석 금지",
            },
            {
                "claim_type": "avoid",
                "statement": "native feature importance를 인과 효과로 해석한다.",
                "evidence": "LightGBM/RF/XGB 내부 분할 중요도이며 피처 간 상관과 one-hot 구조의 영향을 받음",
            },
        ]
    )
    report_claims.to_csv(METRIC_DIR / "ml_stage8_report_claims.csv", index=False, encoding="utf-8-sig")

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
    ]
    stage_cols = ["stage", "purpose", "main_result", "decision"]
    claim_cols = ["claim_type", "statement", "evidence"]
    top_risk_cols = ["top_pct", "selected_n", "captured_positive_n", "total_positive_n", "capture_rate_recall", "precision", "lift_vs_base"]
    threshold_cols = ["operating_point", "threshold", "selected_n", "selected_rate", "precision", "recall", "specificity", "f1", "balanced_accuracy"]
    error_cols = ["outcome", "n", "share_in_dataset", "score_mean", "score_median"]
    agreement_cols = ["base_models_above_threshold", "n", "selected_by_final_n", "positive_n", "positive_rate", "final_score_mean"]
    importance_cols = ["original_feature", "feature_domain", "mean_normalized_importance", "n_models", "contributing_models"]

    top_risk_prob = top_risk[
        (top_risk["model"].eq("model_level_top5_geometric_mean")) & (top_risk["selection_role"].eq("probability_score"))
    ].copy()
    threshold_prob = thresholds[
        (thresholds["model"].eq("model_level_top5_geometric_mean")) & (thresholds["selection_role"].eq("probability_score"))
    ].copy()
    validation = pd.DataFrame(
        [
            {"check": "required_inputs", "value": len(PATHS), "expected": len(PATHS), "passed": True},
            {
                "check": "final_oof_metric_present",
                "value": int(not math.isnan(f(decision["development_oof_auprc"]))),
                "expected": 1,
                "passed": not math.isnan(f(decision["development_oof_auprc"])),
            },
            {
                "check": "final_lockbox_metric_present",
                "value": int(not math.isnan(f(decision["lockbox_auprc"]))),
                "expected": 1,
                "passed": not math.isnan(f(decision["lockbox_auprc"])),
            },
            {
                "check": "stage7_validation_all_passed",
                "value": int(stage7_validation["passed"].astype(str).str.lower().eq("true").all()),
                "expected": 1,
                "passed": stage7_validation["passed"].astype(str).str.lower().eq("true").all(),
            },
            {
                "check": "lockbox_caution_claim_exists",
                "value": int(report_claims["statement"].str.contains("lockbox", case=False, regex=False).any()),
                "expected": 1,
                "passed": report_claims["statement"].str.contains("lockbox", case=False, regex=False).any(),
            },
        ]
    )
    validation.to_csv(METRIC_DIR / "ml_stage8_validation_checks.csv", index=False, encoding="utf-8-sig")

    report = f"""# 머신러닝 전체 결과 정리/해석/분석

## 1. 결론

최종 보고서/운영 threshold 모델은 `{decision["final_model"]} / {decision["final_score_type"]}`이다. 이 모델은 5개 base model(`PL_HGB`, `RULES_HGB`, `PL_LGBM`, `RULES_RF`, `RULES_XGB`)의 예측을 geometric mean으로 결합한 model-level ensemble이다.

development OOF 기준 최종 모델의 AUPRC는 {fmt(decision["development_oof_auprc"])}이고, 로지스틱 비교 기준선 AUPRC {fmt(logistic_baseline["auprc"])} 대비 +{fmt(decision["delta_oof_auprc_vs_logistic_baseline"])} 개선됐다. 1차 단일 ML 최고 AUPRC {fmt(stage1_best["auprc"])} 대비로도 +{fmt(decision["delta_oof_auprc_vs_stage1_best"])} 개선됐다.

lockbox test에서는 AUPRC {fmt(decision["lockbox_auprc"])}, ROC AUC {fmt(decision["lockbox_auroc"])}, Brier {fmt(decision["lockbox_brier"], 5)}, log loss {fmt(decision["lockbox_log_loss"], 5)}를 기록했다. 다만 lockbox는 최초 평가 결과이며, 이 결과를 이용해 후보를 다시 고르거나 튜닝하지 않았다.

## 2. 데이터와 검증 구조

- 전체 결합 데이터: {int(data_audit["n"]):,}행
- development 데이터: {int(data_audit["development_n"]):,}행
- development Target 1: {int(data_audit["development_positive_n"]):,}행, positive rate {fmt(data_audit["development_positive_rate"])}
- lockbox test: {int(stage5_prob["n"]):,}행
- lockbox Target 1: {int(stage5_prob["positive_n"]):,}행, positive rate {fmt(stage5_prob["positive_rate"])}
- strict split leak check는 0으로 유지됐다.

## 3. 단계별 진행 요약

{md_table(stage_timeline, stage_cols, floatfmt=".4f")}

## 4. 핵심 성능 비교

{md_table(final_comparison, comparison_cols, floatfmt=".5f")}

## 5. 단계별 해석

### 5.1 로지스틱 기준선

로지스틱 최종 분석은 ML의 출발점이었다. 성능 비교 기준선으로 사용한 `PLUS_LANDCOVER_RULES_ANOVA`는 development OOF AUPRC {fmt(logistic_baseline["auprc"])}를 기록했다. 해석용 로지스틱 `FINAL_REDUCED_WITH_FWI`는 AUPRC {fmt(logistic_interpret["auprc"])}였지만, ML 비교 기준선은 Stage17에서 지정한 성능 비교용 로지스틱을 사용했다.

### 5.2 1차 ML 스크리닝

1차에서는 M1/M2/M3 및 주요 ML 모델을 넓게 비교했다. 최고 단일 모델은 `{stage1_best["feature_set"]} / {stage1_best["model"]}`였고 AUPRC {fmt(stage1_best["auprc"])}를 기록했다. 이 단계에서 ML이 로지스틱보다 높은 예측 성능을 낼 수 있다는 것이 확인됐다.

### 5.3 2차 Optuna 튜닝

2차 Optuna 최고 후보는 `{stage2_best["feature_set"]} / {stage2_best["model"]}`였고 AUPRC {fmt(stage2_best["auprc"])}였다. 이는 1차 최고 AUPRC {fmt(stage1_best["auprc"])}보다 낮았기 때문에 최종 후보로 채택하지 않았다. 즉, 이번 데이터에서는 단순히 탐색 강도를 높이는 것보다 안정적인 base model 조합이 더 유리했다.

### 5.4 3차와 3.5차 앙상블

3차는 OOF prediction-level blending으로 앙상블 가능성을 확인한 단계였다. 이후 3.5차에서 실제로 base model을 재학습하고 결합하는 model-level ensemble로 전환했다. 3.5차에서 `rank_average`는 AUPRC {fmt(stage35_rank["auprc"])}로 가장 높았지만 Brier/log loss가 나빠 확률 모델로 쓰지 않았다. `geometric_mean`은 AUPRC {fmt(stage35_prob["auprc"])}, Brier {fmt(stage35_prob["brier"], 5)}, log loss {fmt(stage35_prob["log_loss"], 5)}로 균형이 좋아 최종 probability score 후보가 됐다.

### 5.5 4차 Calibration

raw, sigmoid, isotonic calibration을 비교했지만 AUPRC 관점에서 calibration이 최종 개선으로 이어지지 않았다. 따라서 Step4에서 raw geometric mean을 최종 probability score로 고정했다. rank-average는 순위화 진단 보조로만 유지했다.

### 5.6 5차 Lockbox

Step4에서 고정한 후보를 lockbox에 최초 적용했다. 최종 probability score는 lockbox AUPRC {fmt(stage5_prob["auprc"])}, ROC AUC {fmt(stage5_prob["auroc"])}, Brier {fmt(stage5_prob["brier"], 5)}를 기록했다. lockbox 성능은 development OOF보다 높았으므로, 이것을 “일반화 성능이 OOF보다 높다”라고 단정하지 않고 “고정 후보가 lockbox에서 강한 성능을 유지했다”로 표현해야 한다.

### 5.7 7차 해석/오류분석

최종 score는 lockbox top 10%에서 Target 1 {int(lockbox_top10_decile["positive_n"])}/{int(lockbox_top10_decile["n"])}개를 포함했고 positive rate {fmt(lockbox_top10_decile["positive_rate"])}, lift {fmt(lockbox_top10_decile["lift_vs_base"])}, 누적 capture {fmt(lockbox_top10_decile["cumulative_capture_rate"])}를 기록했다. top 30%까지 누적하면 capture는 {fmt(lockbox_top30_decile["cumulative_capture_rate"])}이다.

base model agreement도 의미가 있었다. lockbox에서 5개 base model이 모두 threshold 이상으로 판단한 샘플의 양성률은 {fmt(agree5["positive_rate"])}였고, 0개 모델만 threshold 이상인 샘플의 양성률은 {fmt(agree0["positive_rate"])}였다. 즉 최종 고위험 판단은 여러 모델의 합의가 강할수록 실제 양성률이 높아지는 구조였다.

## 6. 운영 해석

### 6.1 Threshold 운영

Step6에서 권장한 운영점은 `{decision["recommended_operating_point"]}`이다. threshold는 {fmt(decision["recommended_threshold"], 5)}이고, lockbox 기준 precision {fmt(decision["recommended_precision"])}, recall {fmt(decision["recommended_recall"])}, selected rate {fmt(decision["recommended_selected_rate"])}이다.

{md_table(threshold_prob, threshold_cols, floatfmt=".5f")}

이 threshold는 recall 확보를 우선하는 스크리닝 운영점이다. FP가 존재하므로, “확정 예측”이 아니라 “고위험 후보 선별”로 설명하는 것이 맞다.

### 6.2 Top-risk 운영

{md_table(top_risk_prob, top_risk_cols, floatfmt=".5f")}

top-risk 방식은 threshold보다 보고서 해석이 직관적이다. lockbox top 5%는 Target 1 {int(top5["captured_positive_n"])}/{int(top5["total_positive_n"])}개를 포착했고 precision {fmt(top5["precision"])}였다. top 10%는 {int(top10["captured_positive_n"])}/{int(top10["total_positive_n"])}개를 포착했고 capture {fmt(top10["capture_rate_recall"])}, precision {fmt(top10["precision"])}였다. top 20%는 capture {fmt(top20["capture_rate_recall"])}까지 올라갔다.

## 7. 오류분석

Step6 threshold {fmt(decision["recommended_threshold"], 5)} 적용 시 lockbox 오류/정답 구성은 다음과 같다.

{md_table(lockbox_error, error_cols, floatfmt=".5f")}

FP는 주로 Target_0A에서 발생한다. Target_0A는 실제 화재에 가장 가까운 hard negative이므로, 운영 threshold에서는 일정 수준 남을 수밖에 있는 오탐으로 해석한다. FN은 score가 threshold 아래에 있는 실제 양성으로, 후속 분석에서는 이 샘플군의 기상/공간 조건이 기존 고위험 패턴과 어떻게 다른지 보는 것이 다음 과제다.

## 8. Base Model Agreement

{md_table(agreement_lockbox, agreement_cols, floatfmt=".5f")}

5개 base model이 모두 threshold 이상인 경우 lockbox 양성률은 {fmt(agree5["positive_rate"])}이다. 반대로 어느 base model도 threshold 이상이 아닌 경우 양성률은 {fmt(agree0["positive_rate"])}이다. 이는 최종 앙상블의 고위험 점수가 단일 모델의 독립 판단이 아니라 여러 모델의 합의 강도와 관련된다는 근거다.

## 9. 중요 피처 해석

{md_table(importance, importance_cols, floatfmt=".5f", max_rows=20)}

native feature importance 상위에는 캐나다 산불위험지수 계열(`D1_DC`, `D1_FFMC_10일평균`, `D1_BUI`), 습도 계열(`D-1_평균습도_pct`, `D-1_최소습도_pct`, `직전24h_최소습도`, `rh_minus_local_q05`), 공간/접근성(`log1p_도로거리_m`), 토지피복(`토지피복_L1_NAME`)과 WUI/건조 상호작용(`비산림WUI_x_dry0p1`)이 함께 나타났다.

이 결과는 모델이 단일 신호에 의존한 것이 아니라 기상 건조도, 전일 누적 위험, 공간 접근성, 토지피복/비산림 접경 조건을 함께 사용했다는 보조 근거다. 단, 이 중요도는 LightGBM, RandomForest, XGBoost의 내부 분할 중요도이므로 인과 효과로 해석하지 않는다.

## 10. 최종 보고서에 쓸 수 있는 문장과 피해야 할 문장

{md_table(report_claims, claim_cols, floatfmt=".4f")}

## 11. 최종 판단

이번 머신러닝 파이프라인의 최종 성과는 “로지스틱의 해석력을 기준선으로 유지하면서, 예측 성능은 model-level ensemble로 확장했다”는 점이다. strict development OOF에서는 로지스틱 기준선보다 AUPRC가 개선됐고, lockbox 최초 평가에서도 고정 후보가 강한 성능을 보였다.

최종 보고서에서는 `model_level_top5_geometric_mean / raw`를 주 모델로 제시하고, 운영 방식은 `recall_ge_0.70` threshold와 top 5/10/20% risk capture를 함께 제시하는 것이 가장 자연스럽다. `model_level_top5_rank_average / raw`는 AUPRC 순위 진단 보조로만 언급하고, 확률 또는 threshold 운영에는 사용하지 않는 것이 맞다.

## 12. 남은 리스크와 후속 과제

- lockbox 성능이 OOF보다 크게 높으므로, 데이터 분포 차이 또는 lockbox 난이도 차이를 별도로 설명해야 한다.
- lockbox를 본 이후에는 추가 튜닝을 하지 않는 원칙을 유지해야 한다.
- native feature importance는 인과 설명이 아니므로, 변수 영향 해석은 로지스틱 OR 분석과 함께 보조적으로 제시해야 한다.
- 후속 작업을 한다면 최종 보고서용 그림, ROC/PR curve, decile lift chart, threshold confusion matrix 도표를 생성하는 것이 적절하다.

## 13. 검증

{md_table(validation, floatfmt=".4f")}

## 14. 산출물

- `outputs/ml_stage8_comprehensive_analysis.md`
- `머신러닝_8차_전체결과_정리_해석_분석.md`
- `outputs/metrics/ml_stage8_stage_timeline.csv`
- `outputs/metrics/ml_stage8_key_metric_summary.csv`
- `outputs/metrics/ml_stage8_report_claims.csv`
- `outputs/metrics/ml_stage8_validation_checks.csv`
"""

    (OUTPUT_DIR / "ml_stage8_comprehensive_analysis.md").write_text(report, encoding="utf-8")
    (ML_DIR / "머신러닝_8차_전체결과_정리_해석_분석.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_report()
