from __future__ import annotations

from pathlib import Path

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


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
METRICS = OUTPUT / "metrics"
PREDICTIONS = OUTPUT / "predictions"


def probability_metrics(y_true: pd.Series, probability: pd.Series) -> dict:
    y = y_true.to_numpy(dtype=int)
    p = probability.to_numpy(dtype=float)
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    calibration = sm.GLM(
        y,
        sm.add_constant(logit),
        family=sm.families.Binomial(),
    ).fit()
    return {
        "n": len(y),
        "positive_n": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": float(calibration.params[0]),
        "calibration_slope": float(calibration.params[1]),
    }


def choose_thresholds(y_true: pd.Series, probability: pd.Series) -> dict:
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probability,
    )
    f1_values = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    )
    best_idx = int(np.nanargmax(f1_values))
    recall90_candidates = np.flatnonzero(recall[:-1] >= 0.90)
    recall90_idx = (
        int(recall90_candidates[-1])
        if len(recall90_candidates)
        else 0
    )
    return {
        "fixed_0.5": 0.5,
        "oof_best_f1": float(thresholds[best_idx]),
        "oof_recall90": float(thresholds[recall90_idx]),
    }


def threshold_metrics(
    y_true: pd.Series,
    probability: pd.Series,
    threshold: float,
) -> dict:
    y = y_true.to_numpy(dtype=int)
    pred = (probability.to_numpy(dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_threshold_table(
    predictions: pd.DataFrame,
    model_column: str,
) -> pd.DataFrame:
    rows = []
    for model_name, part in predictions.groupby(model_column, observed=True):
        thresholds = choose_thresholds(part["Target"], part["probability"])
        for threshold_type, threshold in thresholds.items():
            rows.append(
                {
                    model_column: model_name,
                    "threshold_type": threshold_type,
                    **threshold_metrics(
                        part["Target"],
                        part["probability"],
                        threshold,
                    ),
                }
            )
    return pd.DataFrame(rows)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def threshold_table_md(
    table: pd.DataFrame,
    model_column: str,
    model_name: str,
) -> list[str]:
    labels = {
        "fixed_0.5": "고정 0.5",
        "oof_best_f1": "OOF F1 최적",
        "oof_recall90": "OOF 재현율 90%",
    }
    selected = table.loc[table[model_column].eq(model_name)].copy()
    selected["order"] = selected["threshold_type"].map(
        {"fixed_0.5": 0, "oof_best_f1": 1, "oof_recall90": 2}
    )
    selected = selected.sort_values("order")
    lines = [
        "| 기준 | threshold | 정확도 | 균형정확도 | 정밀도 | 재현율 | 특이도 | F1 | MCC | TN / FP / FN / TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[row.threshold_type],
                    num(row.threshold),
                    num(row.accuracy),
                    num(row.balanced_accuracy),
                    num(row.precision),
                    num(row.recall),
                    num(row.specificity),
                    num(row.f1),
                    num(row.mcc),
                    f"{row.tn} / {row.fp} / {row.fn} / {row.tp}",
                ]
            )
            + " |"
        )
    return lines


def top_risk_metrics(part: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prevalence = part["Target"].mean()
    for fraction in [0.05, 0.10, 0.20]:
        n = max(1, int(len(part) * fraction))
        top = part.nlargest(n, "probability")
        rows.append(
            {
                "top_fraction": fraction,
                "n": n,
                "positive_rate": top["Target"].mean(),
                "lift": top["Target"].mean() / prevalence,
                "recall_capture": top["Target"].sum()
                / part["Target"].sum(),
            }
        )
    return pd.DataFrame(rows)


def refresh_stage5() -> None:
    overall = pd.read_csv(
        METRICS / "stage5_overall_metrics.csv",
        encoding="utf-8-sig",
    )
    thresholds = pd.read_csv(
        METRICS / "stage5_threshold_metrics.csv",
        encoding="utf-8-sig",
    )
    negative = pd.read_csv(
        METRICS / "stage5_negative_type_metrics.csv",
        encoding="utf-8-sig",
    )
    fold_summary = pd.read_csv(
        METRICS / "stage5_fold_summary.csv",
        encoding="utf-8-sig",
    )
    indexed = overall.set_index("model")
    negative_indexed = negative.set_index(["model", "negative_type"])
    m2 = indexed.loc["M2"]
    m3 = indexed.loc["M3"]
    null_accuracy = 1 - m2["positive_rate"]

    lines = [
        "# 5단계 L2 로지스틱 기준 모델 상세 결과",
        "",
        "## 1. 이번 단계에서 비교한 모델",
        "",
        "- `M1`: 날씨·공간 + FA 요인점수 + D-1 캐나다 산불지수",
        "- `M2`: 날씨·공간 + D-1 캐나다 산불지수",
        "- `M3`: 날씨·공간",
        "- `Dummy`: 양성률만 사용하는 무정보 기준모델",
        "- 평가 표본은 development 13,632행이며 Target 1은 1,242행, 양성률은 9.11%이다.",
        "- 모든 성능은 동일한 grouped outer 5-fold의 OOF 예측으로 계산했으며 lockbox test는 사용하지 않았다.",
        "",
        "## 2. 왜 AUPRC를 주 지표로 사용했는가",
        "",
        f"- 이 자료에서 모두 음성으로 예측해도 정확도는 {pct(null_accuracy)}이다.",
        "- 따라서 정확도만 보면 산불을 한 건도 찾지 못하는 모델도 성능이 높아 보인다.",
        "- AUPRC는 산불 양성을 높은 순위에 얼마나 잘 모으는지와 그 과정의 정밀도·재현율을 함께 반영하므로 불균형 자료의 모델 선택에 적합하다.",
        "- ROC AUC도 함께 보고하지만 음성이 90.89%인 자료에서는 많은 음성을 잘 구분하는 효과가 크게 반영될 수 있어 보조 지표로 둔다.",
        "- 정확도·정밀도·재현율·F1은 임계값에 따라 달라지므로 아래에서 threshold 0.5, F1 최적, 재현율 90% 기준을 모두 제시한다.",
        "",
        "## 3. 임계값을 정하지 않은 전체 OOF 성능",
        "",
        "| 모델 | AUPRC | ROC AUC | Brier | Log loss | Calibration intercept | Calibration slope |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["Dummy", "M1", "M2", "M3"]:
        row = indexed.loc[model]
        lines.append(
            f"| {model} | {num(row.auprc)} | {num(row.auroc)} | "
            f"{num(row.brier, 5)} | {num(row.log_loss, 5)} | "
            f"{num(row.calibration_intercept, 3)} | "
            f"{num(row.calibration_slope, 3)} |"
        )
    lines += [
        "",
        "해석:",
        "",
        f"- M1과 M2의 AUPRC 차이는 {m2.auprc - indexed.loc['M1'].auprc:+.6f}으로 사실상 0이다. FA 요인점수는 추가 예측력을 만들지 못했다.",
        f"- M2는 M3보다 AUPRC가 {m2.auprc - m3.auprc:+.4f}, ROC AUC가 {m2.auroc - m3.auroc:+.4f} 높고 Brier는 {m2.brier - m3.brier:+.5f} 낮다.",
        "- M2 calibration slope 0.968은 1에 가깝고 intercept -0.060은 0에 가까워 개발 OOF 범위에서는 확률 보정이 비교적 양호하다.",
        "",
        "## 4. M2 임계값별 분류 성능",
        "",
    ]
    lines += threshold_table_md(thresholds, "model", "M2")
    fixed = thresholds.set_index(["model", "threshold_type"]).loc[
        ("M2", "fixed_0.5")
    ]
    best = thresholds.set_index(["model", "threshold_type"]).loc[
        ("M2", "oof_best_f1")
    ]
    recall90 = thresholds.set_index(["model", "threshold_type"]).loc[
        ("M2", "oof_recall90")
    ]
    lines += [
        "",
        "해석:",
        "",
        f"- threshold 0.5에서는 정확도가 {pct(fixed.accuracy)}지만 재현율이 {pct(fixed.recall)}라서 실제 산불 1,242개 중 {int(fixed.tp)}개만 찾는다.",
        f"- OOF에서 F1을 최대화한 threshold {best.threshold:.4f}에서는 정밀도 {pct(best.precision)}, 재현율 {pct(best.recall)}, F1 {best.f1:.3f}이다.",
        f"- 산불의 90%를 찾도록 threshold를 {recall90.threshold:.4f}까지 낮추면 재현율은 {pct(recall90.recall)}지만 정밀도는 {pct(recall90.precision)}이고 false positive가 {int(recall90.fp):,}건이다.",
        "- 즉 정확도와 F1은 모델 고유의 단일 값이 아니라 운영 threshold를 어떻게 정하느냐에 따라 달라진다.",
        "",
        "## 5. 대조군 유형별 난이도",
        "",
        "| 비교 대상 | M2 AUPRC | M2 ROC AUC | 의미 |",
        "|---|---:|---:|---|",
    ]
    descriptions = {
        "Target_0A": "같은 산불 위치의 다른 시각: 가장 어려운 시간 구분",
        "Target_0B1": "같은 기상셀·공간층의 배경점",
        "Target_0B2": "더 쉬운 전역 배경점",
    }
    for negative_type in ["Target_0A", "Target_0B1", "Target_0B2"]:
        row = negative_indexed.loc[("M2", negative_type)]
        lines.append(
            f"| {negative_type} | {num(row.auprc)} | "
            f"{num(row.auroc)} | {descriptions[negative_type]} |"
        )
    lines += [
        "",
        "- 전체 점수에는 쉬운 B2 대조군의 영향이 포함된다. 실제 시간적 위험 구분 능력을 보려면 0-A 성능을 반드시 함께 봐야 한다.",
        f"- M2의 0-A AUPRC는 {negative_indexed.loc[('M2', 'Target_0A'), 'auprc']:.4f}이고 M3보다 약 +0.0596 높다.",
        "",
        "## 6. Fold 안정성과 bias-variance 진단",
        "",
    ]
    m2_fold = fold_summary.loc[fold_summary["model"].eq("M2")].iloc[0]
    lines += [
        f"- M2 outer-fold AUPRC 평균±SD는 {m2_fold['auprc_mean']:.4f}±{m2_fold['auprc_sd']:.4f}이다.",
        f"- ROC AUC 평균±SD는 {m2_fold['auroc_mean']:.4f}±{m2_fold['auroc_sd']:.4f}, Brier 평균±SD는 {m2_fold['brier_mean']:.5f}±{m2_fold['brier_sd']:.5f}이다.",
        "- train-validation 격차가 전반적으로 작아 5단계에서는 심한 분산 과적합이 관찰되지 않았다.",
        "- 다만 현재 분할은 동일 날짜와 일부 동일 기상노출이 여러 fold에 나뉠 수 있으므로 미래 기간 일반화 성능으로 단정할 수 없다.",
        "",
        "## 7. 변수와 오즈비 해석",
        "",
        "- M1/M2/M3 모두 L2 정규화 예측모델이다. 정규화 계수는 예측에 기여한 조건부 방향을 보여주지만 전통적 유의확률이나 인과 오즈비가 아니다.",
        "- FWI, ISI, BUI와 습도·풍속 시간창 변수 사이의 강한 다중공선성 때문에 큰 양수와 음수 계수가 함께 나타났다.",
        "- 따라서 5단계 계수의 `exp(coefficient)`를 개별 변수의 최종 오즈비로 보고하지 않는다. 오즈비는 상관 블록을 축소한 별도 설명모델에서 산출한다.",
        "",
        "## 8. 5단계 결론",
        "",
        "- FA 요인점수의 추가 성능은 사실상 0이다.",
        "- D-1 캐나다 지수는 날씨·공간 변수만 사용한 모델보다 AUPRC와 0-A 성능을 분명히 개선했다.",
        "- 기준모델은 M2로 정하고 6단계에서 L1/L2/Elastic Net, class weight, 캐나다 지수 조합을 비교한다.",
        "- lockbox test는 아직 열지 않는다.",
    ]
    (OUTPUT / "stage5_result_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def refresh_stage6() -> None:
    regularization = pd.read_csv(
        METRICS / "stage6_regularization_overall_metrics.csv",
        encoding="utf-8-sig",
    )
    subset = pd.read_csv(
        METRICS / "stage6_canadian_subset_overall_metrics.csv",
        encoding="utf-8-sig",
    )
    negative = pd.read_csv(
        METRICS / "stage6_canadian_subset_negative_type_metrics.csv",
        encoding="utf-8-sig",
    )
    regularization_predictions = pd.read_csv(
        PREDICTIONS / "stage6_regularization_oof_predictions.csv",
        encoding="utf-8-sig",
    )
    subset_predictions = pd.read_csv(
        PREDICTIONS / "stage6_canadian_subset_oof_predictions.csv",
        encoding="utf-8-sig",
    )
    regularization_thresholds = make_threshold_table(
        regularization_predictions,
        "variant",
    )
    subset_thresholds = make_threshold_table(
        subset_predictions,
        "subset",
    )
    regularization_thresholds.to_csv(
        METRICS / "stage6_regularization_threshold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subset_thresholds.to_csv(
        METRICS / "stage6_canadian_subset_threshold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reg_indexed = regularization.set_index("variant")
    subset_indexed = subset.set_index("subset")
    negative_indexed = negative.set_index(["subset", "negative_type"])

    lines = [
        "# 6단계 정규화·class weight·캐나다 지수 축소 상세 결과",
        "",
        "## 1. 이번 단계의 목적",
        "",
        "- 5단계에서 제외 근거가 확보된 FA 요인점수는 사용하지 않았다.",
        "- 동일한 grouped nested CV에서 L2, L1, Elastic Net과 `class_weight=None/balanced`를 비교했다.",
        "- 정규화 승자를 고른 뒤 D-1 캐나다 지수 7개 전체, 일부 조합, FWI 단독, 미사용을 비교했다.",
        "- 개발 OOF 예측만 사용했고 lockbox test는 사용하지 않았다.",
        "",
        "## 2. 평가 지표 해석 원칙",
        "",
        "- 모델 선택의 주 지표는 불균형 자료에 적합한 AUPRC이다.",
        "- ROC AUC는 전체 양성-음성 순위 구분, Brier와 Log loss는 확률 오차, calibration은 예측확률의 신뢰성을 평가한다.",
        "- 정확도·정밀도·재현율·F1은 threshold 의존 지표이므로 아래에 임계값별로 따로 제시한다.",
        "- 정확도는 모두 음성으로 예측해도 90.89%가 되므로 단독 선택 기준으로 사용하지 않는다.",
        "",
        "## 3. 정규화와 class weight 비교",
        "",
        "| 후보 | AUPRC | ROC AUC | Brier | Log loss | Calibration slope |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = [
        "L2_none",
        "ElasticNet_none",
        "L1_none",
        "L1_balanced",
        "ElasticNet_balanced",
        "L2_balanced",
    ]
    for name in order:
        row = reg_indexed.loc[name]
        lines.append(
            f"| {name} | {row.auprc:.4f} | {row.auroc:.4f} | "
            f"{row.brier:.5f} | {row.log_loss:.5f} | "
            f"{row.calibration_slope:.3f} |"
        )
    lines += [
        "",
        "해석:",
        "",
        "- `L2_none`이 AUPRC와 Brier 기준 승자이다.",
        "- L1과 Elastic Net의 AUPRC 차이는 약 0.0015 이내로 작고 실질적인 변수 제거도 만들지 못했다.",
        "- balanced 모델은 ROC AUC가 약 0.004 높지만 AUPRC가 약 0.055 낮고 Brier가 0.1915로 크게 악화됐다.",
        "- balanced weight는 분류 경계를 이동시키지만 현재 샘플링 구조에서는 확률을 과대평가해 경보 확률모델로 부적합하다.",
        "",
        "## 4. 정규화 승자 L2_none의 임계값별 성능",
        "",
    ]
    lines += threshold_table_md(
        regularization_thresholds,
        "variant",
        "L2_none",
    )
    lines += [
        "",
        "- threshold 0.5에서는 정밀도는 높지만 재현율이 낮아 대부분의 산불을 놓친다.",
        "- F1 최적 threshold는 개발 OOF에서 약 0.221이며, 이 값도 최종 운영 임계값이 아니라 후보값이다.",
        "- 최종 threshold는 비용 기준을 정한 뒤 validation에서 고정하고 lockbox에 한 번 적용해야 한다.",
        "",
        "## 5. 캐나다 산불지수 조합 비교",
        "",
        "| 지수 구성 | AUPRC | ROC AUC | Brier | Log loss | 0-A AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    subset_order = [
        "CAN_ALL",
        "CAN_FWI_ONLY",
        "NO_CANADA",
        "CAN_FFMC_ISI_BUI",
        "CAN_FFMC_DMC_DC",
    ]
    for name in subset_order:
        row = subset_indexed.loc[name]
        zero_a = negative_indexed.loc[(name, "Target_0A")]
        lines.append(
            f"| {name} | {row.auprc:.4f} | {row.auroc:.4f} | "
            f"{row.brier:.5f} | {row.log_loss:.5f} | "
            f"{zero_a.auprc:.4f} |"
        )
    all_row = subset_indexed.loc["CAN_ALL"]
    no_canada = subset_indexed.loc["NO_CANADA"]
    lines += [
        "",
        f"- 전체 7개 지수는 캐나다 지수 미사용보다 AUPRC가 {all_row.auprc - no_canada.auprc:+.4f}, 0-A AUPRC가 {negative_indexed.loc[('CAN_ALL', 'Target_0A'), 'auprc'] - negative_indexed.loc[('NO_CANADA', 'Target_0A'), 'auprc']:+.4f} 높다.",
        "- FWI 단독과 일부 지수 조합은 캐나다 지수 미사용 모델과 거의 비슷하다.",
        "- 이 결과는 7개 지수가 각각 독립적으로 유의하다는 뜻이 아니라, 공선성이 강한 지수 묶음 전체가 예측에 추가 정보를 제공했다는 뜻이다.",
        "",
        "## 6. CAN_ALL의 임계값별 성능",
        "",
    ]
    lines += threshold_table_md(
        subset_thresholds,
        "subset",
        "CAN_ALL",
    )
    lines += [
        "",
        "## 7. 변수 선택과 오즈비 해석",
        "",
        "- L1_none과 ElasticNet_none은 49개 변수를 모든 outer fold에서 선택했다. 선택된 C가 10~1000으로 규제가 약해 희소화가 발생하지 않았다.",
        "- CAN_ALL에서는 FWI가 큰 양수, ISI와 BUI가 큰 음수로 나타나는 등 구성지수 사이의 억제효과가 유지됐다.",
        "- 따라서 예측모델의 개별 정규화 계수는 통계적 유의성이나 독립 오즈비로 해석하지 않는다.",
        "- 8단계에서는 습도·풍속·강수·캐나다지수 상관 블록에서 대표변수를 선택한 별도 축소 설명모델을 사용해야 한다.",
        "",
        "## 8. Bias-variance 판단",
        "",
        "- L1/L2/Elastic Net 사이의 성능 차이가 매우 작고 강한 규제보다 약한 규제가 선택됐다.",
        "- 이는 현재 성능 한계가 정규화 방식보다는 비선형성, 권역별 이질성, 부족한 피처에서 올 가능성이 더 크다는 뜻이다.",
        "- class weight를 바꾸는 것만으로는 성능이 개선되지 않았으며 확률 보정만 악화됐다.",
        "- 따라서 Step 6 정규화 탐색을 반복하기보다 EDA 기반 피처를 추가하는 Step 7로 진행하는 것이 타당하다.",
        "",
        "## 9. 6단계 결론",
        "",
        "- FA: 제외",
        "- 정규화: L2",
        "- class weight: 없음",
        "- 캐나다 지수: D-1 전체 7개 유지",
        "- 이후 모델은 `날씨·공간 + D-1 캐나다 전체 지수`를 기준으로 한다.",
        "- lockbox test는 계속 보존한다.",
    ]
    (OUTPUT / "stage6_result_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def refresh_stage7() -> None:
    predictions = pd.read_csv(
        PREDICTIONS / "stage7_feature_set_oof_predictions.csv",
        encoding="utf-8-sig",
    )
    thresholds = make_threshold_table(predictions, "feature_set")
    thresholds.to_csv(
        METRICS / "stage7_feature_set_threshold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    probability_rows = []
    for feature_set, part in predictions.groupby(
        "feature_set",
        observed=True,
    ):
        probability_rows.append(
            {
                "feature_set": feature_set,
                **probability_metrics(part["Target"], part["probability"]),
            }
        )
    probability = pd.DataFrame(probability_rows).sort_values(
        "auprc",
        ascending=False,
    )
    probability.to_csv(
        METRICS / "stage7_feature_set_overall_metrics_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    negative = pd.read_csv(
        METRICS / "stage7_feature_set_negative_type_metrics.csv",
        encoding="utf-8-sig",
    )
    climate = pd.read_csv(
        METRICS / "stage7_feature_set_climate_metrics.csv",
        encoding="utf-8-sig",
    )
    fold = pd.read_csv(
        METRICS / "stage7_feature_set_fold_metrics.csv",
        encoding="utf-8-sig",
    )
    prevalence = pd.read_csv(
        METRICS / "stage7_candidate_feature_prevalence.csv",
        encoding="utf-8-sig",
    )
    probability_indexed = probability.set_index("feature_set")
    negative_indexed = negative.set_index(
        ["feature_set", "negative_type"]
    )
    climate_indexed = climate.set_index(
        ["feature_set", "기후지형유형"]
    )
    recommended = "PLUS_CONTINUOUS_FLAGS"
    metric_best = "PLUS_ALL_INTERACTIONS"
    base = "BASE_CAN_ALL"
    recommended_row = probability_indexed.loc[recommended]
    base_row = probability_indexed.loc[base]
    best_row = probability_indexed.loc[metric_best]
    recommended_predictions = predictions.loc[
        predictions["feature_set"].eq(recommended)
    ]
    risk = top_risk_metrics(recommended_predictions)
    risk.to_csv(
        METRICS / "stage7_recommended_top_risk_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 7단계 EDA 피처 확장 상세 결과",
        "",
        "## 1. 이번 단계의 목적과 피처",
        "",
        "- 6단계 승자인 `L2 + class_weight 없음 + D-1 캐나다 지수 7개`를 기준모델로 사용했다.",
        "- FA 요인점수는 사용하지 않았다.",
        "- 기준시각 이전 6시간 평균·최대풍속, 0.1mm/5mm 기준 무강수 지속시간, 기상셀×월×시간대 국지 하위 5% 습도, EDA 복합 플래그를 추가했다.",
        "- FFMC/ISI 결합 플래그는 미래 누수를 피하기 위해 D-1 정오 지수를 사용했다.",
        "- 모든 결과는 development grouped OOF 예측이며 lockbox test는 사용하지 않았다.",
        "",
        "## 2. 평가 지표를 읽는 방법",
        "",
        "- AUPRC는 양성률 9.11%인 불균형 자료에서 산불을 상위 위험순위에 모으는 능력을 평가하므로 모델 선택의 주 지표이다.",
        "- ROC AUC도 함께 보고하며, Brier·Log loss·calibration은 예측확률 자체의 품질을 평가한다.",
        "- F1·정확도·정밀도·재현율은 threshold에 따라 변하므로 모델 자체의 고정 성능값이 아니다.",
        "- 아래에 threshold 0.5, OOF F1 최적, 재현율 90% 기준을 모두 보고한다.",
        "",
        "## 3. 피처 세트별 임계값 비의존 성능",
        "",
        "| 피처 세트 | AUPRC | ROC AUC | Brier | Log loss | Calibration intercept | Calibration slope |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    feature_order = [
        "BASE_CAN_ALL",
        "PLUS_CONTINUOUS",
        "PLUS_FLAGS",
        "PLUS_CONTINUOUS_FLAGS",
        "PLUS_ALL_INTERACTIONS",
    ]
    for name in feature_order:
        row = probability_indexed.loc[name]
        lines.append(
            f"| {name} | {row.auprc:.4f} | {row.auroc:.4f} | "
            f"{row.brier:.5f} | {row.log_loss:.5f} | "
            f"{row.calibration_intercept:.3f} | "
            f"{row.calibration_slope:.3f} |"
        )
    lines += [
        "",
        "해석:",
        "",
        f"- 추천 세트는 기준모델보다 AUPRC가 {recommended_row.auprc - base_row.auprc:+.4f}, ROC AUC가 {recommended_row.auroc - base_row.auroc:+.4f} 높다.",
        f"- Brier는 {recommended_row.brier - base_row.brier:+.5f} 낮아져 확률 오차도 개선됐다.",
        f"- 수치상 1위인 전체 상호작용 세트와 추천 세트의 AUPRC 차이는 {best_row.auprc - recommended_row.auprc:+.4f}뿐이다.",
        "- 사전 단순성 기준 0.005보다 차이가 작으므로 `PLUS_CONTINUOUS_FLAGS`를 후속 주 모델로 선택한다.",
        "",
        "## 4. 추천 모델의 임계값별 분류 성능",
        "",
    ]
    lines += threshold_table_md(
        thresholds,
        "feature_set",
        recommended,
    )
    threshold_indexed = thresholds.set_index(
        ["feature_set", "threshold_type"]
    )
    fixed = threshold_indexed.loc[(recommended, "fixed_0.5")]
    best_threshold = threshold_indexed.loc[
        (recommended, "oof_best_f1")
    ]
    recall90 = threshold_indexed.loc[
        (recommended, "oof_recall90")
    ]
    lines += [
        "",
        "해석:",
        "",
        f"- threshold 0.5의 정확도는 {pct(fixed.accuracy)}지만 재현율은 {pct(fixed.recall)}이다. 높은 정확도만 보고 좋은 모델이라고 판단하면 안 된다.",
        f"- F1 최적 threshold는 {best_threshold.threshold:.4f}이고 정밀도 {pct(best_threshold.precision)}, 재현율 {pct(best_threshold.recall)}, F1 {best_threshold.f1:.3f}이다.",
        f"- 재현율 90% threshold는 {recall90.threshold:.4f}이며 정밀도는 {pct(recall90.precision)}이다. 조기경보처럼 누락을 줄이면 오경보가 크게 늘어나는 trade-off가 있다.",
        "- OOF에서 고른 threshold는 탐색값이다. 최종 threshold는 비용 기준을 정한 뒤 별도 validation 정책으로 고정해야 한다.",
        "",
        "## 5. Stage 6 기준모델과 F1 비교",
        "",
    ]
    base_best = threshold_indexed.loc[(base, "oof_best_f1")]
    lines += [
        "| 모델 | F1 최적 threshold | 정확도 | 정밀도 | 재현율 | F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| BASE_CAN_ALL | {base_best.threshold:.4f} | {base_best.accuracy:.4f} | {base_best.precision:.4f} | {base_best.recall:.4f} | {base_best.f1:.4f} |",
        f"| PLUS_CONTINUOUS_FLAGS | {best_threshold.threshold:.4f} | {best_threshold.accuracy:.4f} | {best_threshold.precision:.4f} | {best_threshold.recall:.4f} | {best_threshold.f1:.4f} |",
        "",
        f"- F1은 {base_best.f1:.4f}에서 {best_threshold.f1:.4f}로 약 {best_threshold.f1 - base_best.f1:+.4f} 개선됐다.",
        "- F1 개선 폭은 AUPRC 개선보다 작다. 이는 F1이 특정 threshold 한 점만 평가하고 AUPRC는 가능한 모든 threshold의 정밀도-재현율 관계를 평가하기 때문이다.",
        "",
        "## 6. 상위 위험군 집중도",
        "",
        "| 예측위험 상위 구간 | 표본 수 | 실제 양성률 | 전체 대비 lift | 전체 산불 포착률 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in risk.itertuples(index=False):
        lines.append(
            f"| 상위 {int(row.top_fraction * 100)}% | "
            f"{row.n:,} | {pct(row.positive_rate)} | "
            f"{row.lift:.2f}배 | {pct(row.recall_capture)} |"
        )
    lines += [
        "",
        "- 전체 양성률은 9.11%지만 예측위험 상위 5%의 실제 양성률은 약 49.2%이다.",
        "- 따라서 현재 모델은 확정 진단보다 예찰·점검 우선순위를 정하는 위험순위 모델로 더 적합하다.",
        "",
        "## 7. 대조군 유형별 성능",
        "",
        "| 비교 대상 | BASE AUPRC | 추천 모델 AUPRC | 변화 | 추천 ROC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for negative_type in ["Target_0A", "Target_0B1", "Target_0B2"]:
        base_part = negative_indexed.loc[(base, negative_type)]
        new_part = negative_indexed.loc[(recommended, negative_type)]
        lines.append(
            f"| {negative_type} | {base_part.auprc:.4f} | "
            f"{new_part.auprc:.4f} | "
            f"{new_part.auprc - base_part.auprc:+.4f} | "
            f"{new_part.auroc:.4f} |"
        )
    lines += [
        "",
        "- 핵심 hard-negative인 0-A AUPRC가 약 +0.046 개선됐다.",
        "- B2는 소폭 낮아졌지만 B2는 상대적으로 쉬운 전역 배경 대조군이다. 실제 시간적 위험 구분에 가까운 0-A 개선을 더 중요하게 본다.",
        "",
        "## 8. 기후지형유형별 성능",
        "",
        "| 유형 | BASE AUPRC | 추천 모델 AUPRC | 변화 | 추천 ROC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for climate_type in ["영동 해안형", "영서 내륙형", "고지·산간형"]:
        base_part = climate_indexed.loc[(base, climate_type)]
        new_part = climate_indexed.loc[(recommended, climate_type)]
        lines.append(
            f"| {climate_type} | {base_part.auprc:.4f} | "
            f"{new_part.auprc:.4f} | "
            f"{new_part.auprc - base_part.auprc:+.4f} | "
            f"{new_part.auroc:.4f} |"
        )
    lines += [
        "",
        "- 전체 성능 개선은 주로 영동 해안형에서 발생했다.",
        "- 영서 내륙형과 고지·산간형은 개선이 없거나 소폭 악화됐다. 강원도 전체에 동일한 기상관계를 강제하는 단일 선형모델의 한계일 수 있다.",
        "- 후속 로지스틱 계열 확장 또는 권역별 상호작용 검토에서 영서·고지 성능을 별도로 확인해야 한다.",
        "",
        "## 9. Fold 안정성과 bias-variance",
        "",
    ]
    validation = fold.loc[
        fold["feature_set"].eq(recommended)
        & fold["dataset"].eq("validation")
    ]
    training = fold.loc[
        fold["feature_set"].eq(recommended)
        & fold["dataset"].eq("train"),
        ["outer_fold", "auprc"],
    ].rename(columns={"auprc": "train_auprc"})
    gap = validation.merge(training, on="outer_fold")
    gap["gap"] = gap["train_auprc"] - gap["auprc"]
    lines += [
        f"- 추천 모델 validation AUPRC는 fold별 {validation.auprc.min():.4f}~{validation.auprc.max():.4f}, 평균±SD {validation.auprc.mean():.4f}±{validation.auprc.std():.4f}이다.",
        f"- 평균 train-validation AUPRC 차이는 {gap.gap.mean():.4f}, 최대 절대 차이는 {gap.gap.abs().max():.4f}이다.",
        "- 현재 결과에서는 심한 분산 과적합보다 영서·고지에서 관계를 충분히 표현하지 못하는 bias가 더 큰 문제로 보인다.",
        "",
        "## 10. EDA 위험조건의 발생률",
        "",
        "| 조건 | Target 1 | Target 0-A | 단순 방향 |",
        "|---|---:|---:|---|",
    ]
    for feature in [
        "rh_local_q05",
        "dry_spell_0p1_gt_24h",
        "dry_spell_5p0_gt_240h",
        "rh_local_q05_AND_ffmc_ge_90",
        "rh_local_q05_AND_isi_ge_10",
        "rh_local_q05_AND_wind_max_6h_ge_5",
    ]:
        part = prevalence.loc[prevalence["feature"].eq(feature)].set_index(
            "샘플유형"
        )
        lines.append(
            f"| `{feature}` | {pct(part.loc['Target_1', 'rate'])} | "
            f"{pct(part.loc['Target_0A', 'rate'])} | "
            "산불에서 더 높음 |"
        )
    lines += [
        "",
        "- 위 표는 보정되지 않은 발생률 비교이다. 변수 간 상관과 날짜 군집을 통제한 독립 오즈비가 아니다.",
        "- 예측모델에서는 연속형과 플래그가 동시에 들어가므로 일부 조건부 계수는 단변량 방향과 반대로 나타날 수 있다.",
        "",
        "## 11. 오즈비와 통계적 유의성에 대한 현재 상태",
        "",
        "- Stage 7의 L2 계수는 수치형과 이진형 모두 StandardScaler를 거친 정규화 계수이다.",
        "- 따라서 현재 계수의 지수값을 `조건 충족 여부의 오즈비`로 직접 해석할 수 없다.",
        "- 캐나다 지수와 습도·풍속·강수 시간창 사이에 높은 상관이 있어 full model의 부호는 억제효과 영향을 받는다.",
        "- 아직 cluster-robust 95% CI와 p/q-value를 계산하지 않았으므로 '통계적으로 유의한 독립 변수'를 확정할 단계가 아니다.",
        "- Step 8에서는 상관 블록 대표변수를 사용한 축소 비정규화 로지스틱을 별도로 적합해 표준화 OR, 95% CI, 날짜 또는 그룹 cluster-robust p-value, FDR q-value를 보고해야 한다.",
        "",
        "## 12. 중요한 검증 한계",
        "",
        "- 현재 grouped CV는 모델링 그룹 ID는 분리하지만 동일 `기상셀ID×기준시각` 노출과 집단발생 날짜 전체를 하나의 fold로 묶지는 않았다.",
        "- 사후 점검에서 동일 기상노출 키 694개, 관련 1,853행과 양성 419행이 여러 fold에 걸친 것을 확인했다.",
        "- 2021년 2월 1~3일 같은 대규모 집단발생일도 여러 fold에 분산됐다.",
        "- 따라서 현재 점수는 그룹 중복은 막았지만 새로운 날짜·새로운 기상사건에 대한 일반화 성능보다 낙관적일 가능성이 있다.",
        "- lockbox를 열기 전에 날짜·기상노출을 함께 고려한 더 엄격한 CV에서 기준모델과 추천모델을 재검증해야 한다.",
        "",
        "## 13. 7단계 최종 판단",
        "",
        "- 현재 추천 예측모델: `PLUS_CONTINUOUS_FLAGS`",
        f"- 전체 AUPRC/ROC AUC: {recommended_row.auprc:.4f}/{recommended_row.auroc:.4f}",
        f"- F1 최적 운영점의 정밀도/재현율/F1: {best_threshold.precision:.4f}/{best_threshold.recall:.4f}/{best_threshold.f1:.4f}",
        f"- 가장 어려운 0-A AUPRC: {negative_indexed.loc[(recommended, 'Target_0A'), 'auprc']:.4f}",
        "- 성능 수준은 무작위보다 분명히 높고 위험순위화에는 활용 가능성이 있으나, 산불 발생을 확정 판정하거나 인과관계를 주장할 수준은 아니다.",
        "- 다음 순서는 더 엄격한 분할 재검증 후 Step 8 오즈비 분석이며, 그 다음 동일 분할로 로지스틱 계열 통계모델을 확장 비교한다.",
    ]
    (OUTPUT / "stage7_result_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    refresh_stage5()
    refresh_stage6()
    refresh_stage7()
    print("Refreshed stage 5-7 summaries and threshold metric tables.")


if __name__ == "__main__":
    main()
