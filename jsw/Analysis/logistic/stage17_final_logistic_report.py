from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"
PRED_DIR = OUTPUT_DIR / "predictions"

STAGE11_BENCHMARK = TABLE_DIR / "stage11_final_logistic_benchmark.csv"
STAGE10_PRED = PRED_DIR / "stage10_logistic_stat_extensions_oof_predictions.csv"
STAGE16_METRICS = TABLE_DIR / "stage16_final_model_metrics.csv"
STAGE16_OR = TABLE_DIR / "stage16_final_report_or_table.csv"
STAGE16_PRED = PRED_DIR / "stage16_final_oof_predictions.csv"

SUMMARY_PATH = OUTPUT_DIR / "stage17_final_logistic_report_summary.md"
ROLE_TABLE_PATH = TABLE_DIR / "stage17_model_role_summary.csv"
METRICS_TABLE_PATH = TABLE_DIR / "stage17_ml_comparison_metrics.csv"
OPERATING_TABLE_PATH = TABLE_DIR / "stage17_operating_thresholds.csv"
TOP_RISK_TABLE_PATH = TABLE_DIR / "stage17_top_risk_capture.csv"
OR_REPORT_TABLE_PATH = TABLE_DIR / "stage17_final_or_report_table.csv"
EDA_LINK_TABLE_PATH = TABLE_DIR / "stage17_eda_linkage_table.csv"

PERFORMANCE_MODEL = "PLUS_LANDCOVER_RULES_ANOVA"
F1_AUX_MODEL = "PLUS_LANDCOVER"
INTERPRET_MODEL = "FINAL_REDUCED_WITH_FWI"
REJECTED_LANDCOVER_MODEL = "FINAL_REDUCED_WITH_SIMPLE_LANDCOVER_FLAGS"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)


def fmt(x: float | int | str | None, digits: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def to_float(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": float(threshold),
        "selected_n": int(pred.sum()),
        "selected_rate": float(pred.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def threshold_curve(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.astype(int), "prob": prob.astype(float)})
    grouped = (
        df.groupby("prob", as_index=False)
        .agg(n=("y", "size"), pos=("y", "sum"))
        .sort_values("prob", ascending=False)
        .reset_index(drop=True)
    )
    grouped["neg"] = grouped["n"] - grouped["pos"]
    grouped["tp"] = grouped["pos"].cumsum()
    grouped["fp"] = grouped["neg"].cumsum()
    total_pos = int(df["y"].sum())
    total_neg = int((1 - df["y"]).sum())
    grouped["fn"] = total_pos - grouped["tp"]
    grouped["tn"] = total_neg - grouped["fp"]
    grouped["selected_n"] = grouped["tp"] + grouped["fp"]
    grouped["precision"] = grouped["tp"] / grouped["selected_n"]
    grouped["recall"] = np.where(total_pos > 0, grouped["tp"] / total_pos, np.nan)
    grouped["specificity"] = np.where(total_neg > 0, grouped["tn"] / total_neg, np.nan)
    denom = grouped["precision"] + grouped["recall"]
    grouped["f1"] = np.where(denom > 0, 2 * grouped["precision"] * grouped["recall"] / denom, 0.0)
    grouped["balanced_accuracy"] = (grouped["recall"] + grouped["specificity"]) / 2
    grouped = grouped.rename(columns={"prob": "threshold"})
    return grouped


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    curve = threshold_curve(y_true, prob).copy()
    curve["selected_rate"] = curve["selected_n"] / len(y_true)
    curve = curve.sort_values(
        ["f1", "balanced_accuracy", "selected_rate", "threshold"],
        ascending=[False, False, True, False],
    )
    return float(curve.iloc[0]["threshold"])


def threshold_for_recall(y_true: np.ndarray, prob: np.ndarray, target_recall: float) -> float:
    curve = threshold_curve(y_true, prob)
    candidates = curve.loc[curve["recall"] >= target_recall]
    if candidates.empty:
        return 0.0
    # Curve is sorted by threshold descending, so the first row is the smallest alert volume
    # that satisfies the requested recall.
    return float(candidates.iloc[0]["threshold"])


def operating_points_for_model(model_role: str, model_name: str, y_true: np.ndarray, prob: np.ndarray) -> list[dict]:
    points = []
    requested = [
        ("fixed_0.50", 0.5, "고정 0.50 threshold. 불균형 자료에서는 보통 너무 보수적이다."),
        ("best_f1", best_f1_threshold(y_true, prob), "F1 최대 threshold. 성능 비교표의 기본 운영점이다."),
        ("recall_ge_0.50", threshold_for_recall(y_true, prob, 0.50), "recall 0.50 이상을 만족하는 가장 높은 threshold."),
        ("recall_ge_0.70", threshold_for_recall(y_true, prob, 0.70), "recall 0.70 이상을 만족하는 가장 높은 threshold."),
        ("recall_ge_0.90", threshold_for_recall(y_true, prob, 0.90), "recall 0.90 이상을 만족하는 가장 높은 threshold."),
    ]
    for point_name, thr, description in requested:
        row = binary_metrics(y_true, prob, thr)
        row.update(
            {
                "model_role": model_role,
                "model": model_name,
                "operating_point": point_name,
                "description": description,
            }
        )
        points.append(row)
    return points


def top_risk_capture_for_model(
    model_role: str,
    model_name: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    top_pcts: Iterable[float] = (0.05, 0.10, 0.20),
) -> list[dict]:
    order = np.argsort(-prob)
    y_sorted = y_true[order]
    p_sorted = prob[order]
    n = len(y_true)
    positive_n = int(y_true.sum())
    positive_rate = positive_n / n
    rows = []
    for pct in top_pcts:
        k = int(math.ceil(n * pct))
        selected_y = y_sorted[:k]
        selected_pos = int(selected_y.sum())
        precision = selected_pos / k if k else np.nan
        capture_rate = selected_pos / positive_n if positive_n else np.nan
        rows.append(
            {
                "model_role": model_role,
                "model": model_name,
                "top_pct": pct,
                "selected_n": k,
                "selected_rate": k / n,
                "threshold_min": float(p_sorted[k - 1]) if k else np.nan,
                "captured_positive_n": selected_pos,
                "total_positive_n": positive_n,
                "capture_rate_recall": capture_rate,
                "precision": precision,
                "base_positive_rate": positive_rate,
                "lift_vs_base": precision / positive_rate if positive_rate else np.nan,
            }
        )
    return rows


def full_probability_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    best_thr = best_f1_threshold(y_true, prob)
    best = binary_metrics(y_true, prob, best_thr)
    return {
        "n": len(y_true),
        "positive_n": int(y_true.sum()),
        "positive_rate": float(np.mean(y_true)),
        "auprc": float(average_precision_score(y_true, prob)),
        "auroc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "best_f1_threshold": best["threshold"],
        "best_f1_accuracy": best["accuracy"],
        "best_f1_balanced_accuracy": best["balanced_accuracy"],
        "best_f1_precision": best["precision"],
        "best_f1_recall": best["recall"],
        "best_f1_specificity": best["specificity"],
        "best_f1_f1": best["f1"],
    }


def load_prediction_series() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stage10 = pd.read_csv(STAGE10_PRED, encoding="utf-8-sig")
    stage16 = pd.read_csv(STAGE16_PRED, encoding="utf-8-sig")

    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in [PERFORMANCE_MODEL, F1_AUX_MODEL]:
        sub = stage10.loc[stage10["feature_set"] == model].copy()
        if sub.empty:
            raise ValueError(f"Stage10 predictions missing model: {model}")
        output[model] = (sub["Target"].to_numpy(dtype=int), sub["probability"].to_numpy(dtype=float))

    for model in [INTERPRET_MODEL]:
        sub = stage16.loc[stage16["model"] == model].copy()
        if sub.empty:
            raise ValueError(f"Stage16 predictions missing model: {model}")
        output[model] = (sub["Target"].to_numpy(dtype=int), sub["pred_prob"].to_numpy(dtype=float))

    return output


def build_role_table() -> pd.DataFrame:
    rows = [
        {
            "model_role": "성능 비교용 대표 로지스틱",
            "source_stage": "Stage11",
            "model": PERFORMANCE_MODEL,
            "use_in_report": "ML 성능 비교표",
            "decision": "유지",
            "reason": "기존 로지스틱 후보 중 성능 기준 대표 모델. 계수 해석용으로는 사용하지 않음.",
        },
        {
            "model_role": "F1 운영점 보조 로지스틱",
            "source_stage": "Stage11",
            "model": F1_AUX_MODEL,
            "use_in_report": "threshold 보조 비교",
            "decision": "보조 유지",
            "reason": "F1이 근소하게 높아 운영 threshold 비교에만 보조로 사용.",
        },
        {
            "model_role": "최종 해석용 로지스틱",
            "source_stage": "Stage16",
            "model": INTERPRET_MODEL,
            "use_in_report": "OR·EDA 연결 해석",
            "decision": "최종 추천",
            "reason": "AUPRC가 높고 max VIF 2.27로 낮아 날짜 cluster-robust OR 해석이 안정적.",
        },
        {
            "model_role": "토지피복 플래그 비교 모델",
            "source_stage": "Stage16",
            "model": REJECTED_LANDCOVER_MODEL,
            "use_in_report": "채택하지 않는 비교 근거",
            "decision": "해석 모델 제외",
            "reason": "AUPRC는 0.2600으로 근소하게 높지만 VIF가 50 이상이라 계수 해석에 부적합.",
        },
    ]
    return pd.DataFrame(rows)


def build_metrics_table(predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    stage11 = pd.read_csv(STAGE11_BENCHMARK, encoding="utf-8-sig")
    stage16 = pd.read_csv(STAGE16_METRICS, encoding="utf-8-sig")

    role_map = {
        PERFORMANCE_MODEL: "성능 비교용 대표 로지스틱",
        F1_AUX_MODEL: "F1 운영점 보조 로지스틱",
        INTERPRET_MODEL: "최종 해석용 로지스틱",
        REJECTED_LANDCOVER_MODEL: "토지피복 플래그 비교 모델",
    }
    source_map = {
        PERFORMANCE_MODEL: "Stage11",
        F1_AUX_MODEL: "Stage11",
        INTERPRET_MODEL: "Stage16",
        REJECTED_LANDCOVER_MODEL: "Stage16",
    }

    rows = []
    for model in [PERFORMANCE_MODEL, F1_AUX_MODEL]:
        row = stage11.loc[stage11["feature_set"] == model].iloc[0].to_dict()
        rows.append(
            {
                "source_stage": source_map[model],
                "model_role": role_map[model],
                "model": model,
                "n": np.nan,
                "positive_n": np.nan,
                "positive_rate": np.nan,
                "auprc": row["auprc"],
                "auroc": row["auroc"],
                "brier": row["brier"],
                "log_loss": row["log_loss"],
                "best_f1_threshold": row["best_f1_threshold"],
                "best_f1_accuracy": row["best_f1_accuracy"],
                "best_f1_balanced_accuracy": row["best_f1_balanced_accuracy"],
                "best_f1_precision": row["best_f1_precision"],
                "best_f1_recall": row["best_f1_recall"],
                "best_f1_specificity": row["best_f1_specificity"],
                "best_f1_f1": row["best_f1_f1"],
                "max_vif": np.nan,
                "interpretation_use": "성능 비교" if model == PERFORMANCE_MODEL else "운영점 보조",
            }
        )

    for model in [INTERPRET_MODEL, REJECTED_LANDCOVER_MODEL]:
        row = stage16.loc[stage16["model"] == model].iloc[0].to_dict()
        rows.append(
            {
                "source_stage": source_map[model],
                "model_role": role_map[model],
                "model": model,
                "n": row["n"],
                "positive_n": row["positive_n"],
                "positive_rate": row["positive_rate"],
                "auprc": row["auprc"],
                "auroc": row["auroc"],
                "brier": row["brier"],
                "log_loss": row["log_loss"],
                "best_f1_threshold": row["best_f1_threshold"],
                "best_f1_accuracy": row["best_f1_accuracy"],
                "best_f1_balanced_accuracy": row["best_f1_balanced_accuracy"],
                "best_f1_precision": row["best_f1_precision"],
                "best_f1_recall": row["best_f1_recall"],
                "best_f1_specificity": row["best_f1_specificity"],
                "best_f1_f1": row["best_f1_f1"],
                "max_vif": row["max_vif"],
                "interpretation_use": "OR 해석" if model == INTERPRET_MODEL else "공선성 때문에 제외",
            }
        )

    # Add recomputed support from predictions for models available as OOF predictions.
    support_rows = []
    for model, (y, p) in predictions.items():
        row = full_probability_metrics(y, p)
        row["model"] = model
        support_rows.append(row)
    support = pd.DataFrame(support_rows)
    df = pd.DataFrame(rows).merge(
        support[["model", "n", "positive_n", "positive_rate"]].rename(
            columns={"n": "n_pred", "positive_n": "positive_n_pred", "positive_rate": "positive_rate_pred"}
        ),
        on="model",
        how="left",
    )
    for col in ["n", "positive_n", "positive_rate"]:
        pred_col = f"{col}_pred"
        df[col] = df[col].where(df[col].notna(), df[pred_col])
        df = df.drop(columns=[pred_col])
    front = ["source_stage", "model_role", "model", "interpretation_use", "n", "positive_n", "positive_rate"]
    return df[front + [c for c in df.columns if c not in front]]


def build_operating_tables(predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_map = {
        PERFORMANCE_MODEL: "성능 비교용 대표 로지스틱",
        F1_AUX_MODEL: "F1 운영점 보조 로지스틱",
        INTERPRET_MODEL: "최종 해석용 로지스틱",
    }
    operating_rows = []
    top_rows = []
    for model in [PERFORMANCE_MODEL, F1_AUX_MODEL, INTERPRET_MODEL]:
        y, p = predictions[model]
        operating_rows.extend(operating_points_for_model(role_map[model], model, y, p))
        top_rows.extend(top_risk_capture_for_model(role_map[model], model, y, p))
    operating = pd.DataFrame(operating_rows)
    top_risk = pd.DataFrame(top_rows)
    return operating, top_risk


def build_or_report_table() -> pd.DataFrame:
    odds = pd.read_csv(STAGE16_OR, encoding="utf-8-sig")
    odds = odds.loc[odds["model"] == INTERPRET_MODEL].copy()
    odds["or_95ci"] = odds.apply(
        lambda r: f"{float(r['odds_ratio']):.3f} ({float(r['or_ci_low']):.3f}~{float(r['or_ci_high']):.3f})",
        axis=1,
    )
    interpretation = {
        "log1p_도로거리_m": "도로에서 멀수록 산불 odds가 감소한다. EDA의 도로 초접경 집중과 일치한다.",
        "고도(m)": "다른 변수 통제 후 고도 증가가 odds 감소 방향으로 남는다. 저고도·생활권 접경 편향과 연결된다.",
        "기후지형유형": "기후지형유형 더미는 조건부 층화 효과이다. 영동 더미 OR만으로 영동 위험을 단순 판단하지 않는다.",
        "직전24h_최소습도": "직전 24시간 최소습도 하강은 가장 일관적인 건조 신호 중 하나이다.",
        "dry_spell_5p0_gt_240h": "5mm 이상 젖힘 강수 이후 10일 이상 경과한 상태는 무강수 지속 신호이다.",
        "rh_local_q05": "해당 셀·시간대 평상시 대비 하위 5% 저습은 강한 양의 신호이다.",
        "D1_FWI": "캐나다 FWI는 다른 캐나다지수와 동시 투입하지 않을 때 안정적인 종합 위험 지수로 남는다.",
        "기압변동_3h": "단기 기압 하강은 보조적인 종관 변화 프록시로 해석한다.",
        "wind_max_6h": "풍속 상승은 습도보다 약하지만 독립 양의 신호로 남는다.",
        "서풍계열_여부": "서풍계열 단독은 유의하지 않다. 영동·저습·풍속 결합 해석이 필요하다.",
    }
    odds["report_interpretation"] = odds["feature"].map(interpretation).fillna(
        "조건부 로지스틱 모델의 보조 항으로 해석한다."
    )
    columns = [
        "model",
        "concept_group",
        "role",
        "feature",
        "term_label",
        "unit_for_or",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
        "or_95ci",
        "p_value",
        "q_value",
        "significant_q05",
        "report_interpretation",
    ]
    return odds[columns]


def build_eda_linkage_table() -> pd.DataFrame:
    rows = [
        {
            "topic": "습도·국지건조",
            "eda_result": "EDA에서 직전 24~48시간 저습과 국지 상대건조가 가장 안정적인 신호였다.",
            "logistic_result": "`직전24h_최소습도` 5%p 감소 OR 1.074, `rh_local_q05` OR 2.477로 q<0.05.",
            "report_use": "습도는 강원도 산불 발생 노출의 핵심 조건부 신호로 쓴다.",
        },
        {
            "topic": "무강수 지속",
            "eda_result": "강수량 자체보다 5mm 이상 젖힘 강수 후 장기 무강수 지속이 중요했다.",
            "logistic_result": "`dry_spell_5p0_gt_240h` OR 1.667, q=0.0073.",
            "report_use": "무강수 지속은 습도와 함께 건조 누적 조건으로 해석한다.",
        },
        {
            "topic": "풍속",
            "eda_result": "전체 평균 풍속 신호는 약하지만 영동 해안형에서 강풍·저습 결합이 두드러졌다.",
            "logistic_result": "`wind_max_6h` OR 1.113, q=0.0461. `서풍계열_여부` 단독은 비유의.",
            "report_use": "풍속은 단독 원인보다 저습과 결합되는 보조 트리거로 설명한다.",
        },
        {
            "topic": "도로 접근성",
            "eda_result": "WUI 발생지의 약 72%가 도로 10m 이내에 몰렸고 도로 초접경 편향이 가장 강했다.",
            "logistic_result": "`log1p_도로거리_m` OR 0.758, q≈0.",
            "report_use": "도로에서 가까울수록 발생 odds가 높다는 공간 결론을 로지스틱으로 보강한다.",
        },
        {
            "topic": "캐나다 산불지수",
            "eda_result": "FFMC, ISI, FWI 등은 위험 구간이 보였지만 서로 공선성이 컸다.",
            "logistic_result": "`D1_FWI` 5점 증가 OR 1.155, q=0.0172. 전체 지수 동시 투입은 제외.",
            "report_use": "FWI는 해석 가능한 보조 종합지수로 쓰고, 여러 캐나다지수의 동시 OR 해석은 피한다.",
        },
        {
            "topic": "토지피복",
            "eda_result": "토지피복은 WUI/도로접근성과 강하게 얽히고 희소 범주가 있었다.",
            "logistic_result": "단순 토지피복 플래그 모델은 AUPRC 0.2600이지만 max VIF 50.62.",
            "report_use": "토지피복 세부 범주는 최종 해석 모델에서 제외하고 EDA 보조 설명으로만 둔다.",
        },
    ]
    return pd.DataFrame(rows)


def plot_metric_comparison(metrics: pd.DataFrame) -> None:
    plot_df = metrics.loc[
        metrics["model"].isin([PERFORMANCE_MODEL, F1_AUX_MODEL, INTERPRET_MODEL, REJECTED_LANDCOVER_MODEL])
    ].copy()
    plot_df["label"] = plot_df["model"].map(
        {
            PERFORMANCE_MODEL: "Stage11\nperformance",
            F1_AUX_MODEL: "Stage11\nF1 aux",
            INTERPRET_MODEL: "Stage16\ninterpret",
            REJECTED_LANDCOVER_MODEL: "Stage16\nlandcover flags",
        }
    )
    metric_cols = [("auprc", "AUPRC"), ("auroc", "ROC AUC"), ("best_f1_f1", "Best-F1"), ("brier", "Brier")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (col, title) in zip(axes, metric_cols):
        ax.barh(plot_df["label"], plot_df[col].astype(float), color="#4c78a8")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        for y, v in enumerate(plot_df[col].astype(float)):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=9)
    fig.suptitle("Step17 logistic role-based metric comparison")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stage17_metric_role_comparison.png", dpi=180)
    plt.close(fig)


def plot_top_risk(top_risk: pd.DataFrame) -> None:
    plot_df = top_risk.loc[top_risk["model"].isin([PERFORMANCE_MODEL, INTERPRET_MODEL])].copy()
    plot_df["top_pct_label"] = (plot_df["top_pct"] * 100).astype(int).astype(str) + "%"
    plot_df["model_label"] = plot_df["model"].map(
        {
            PERFORMANCE_MODEL: "Stage11 performance",
            INTERPRET_MODEL: "Stage16 interpret",
        }
    )
    pivot = plot_df.pivot(index="top_pct_label", columns="model_label", values="capture_rate_recall")
    pivot = pivot.reindex(["5%", "10%", "20%"])
    ax = pivot.plot(kind="bar", figsize=(8, 5), color=["#4c78a8", "#f58518"])
    ax.set_title("Positive capture rate in top-risk bands")
    ax.set_xlabel("Top predicted-risk band")
    ax.set_ylabel("Captured positives / all positives")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stage17_top_risk_capture.png", dpi=180)
    plt.close()


def write_summary(
    role_table: pd.DataFrame,
    metrics: pd.DataFrame,
    operating: pd.DataFrame,
    top_risk: pd.DataFrame,
    or_table: pd.DataFrame,
    eda_link: pd.DataFrame,
) -> None:
    metrics_display = metrics.copy()
    metric_cols = [
        "auprc",
        "auroc",
        "brier",
        "log_loss",
        "best_f1_f1",
        "best_f1_precision",
        "best_f1_recall",
        "best_f1_accuracy",
        "best_f1_balanced_accuracy",
        "max_vif",
    ]
    for col in metric_cols:
        if col in metrics_display.columns:
            metrics_display[col] = metrics_display[col].map(lambda x: fmt(x, 4))

    op_display = operating.loc[
        operating["model"].isin([PERFORMANCE_MODEL, INTERPRET_MODEL])
        & operating["operating_point"].isin(["best_f1", "recall_ge_0.70", "recall_ge_0.90"])
    ].copy()
    for col in ["threshold", "selected_rate", "precision", "recall", "specificity", "f1", "accuracy", "balanced_accuracy"]:
        op_display[col] = op_display[col].map(lambda x: fmt(x, 4))

    top_display = top_risk.loc[top_risk["model"].isin([PERFORMANCE_MODEL, INTERPRET_MODEL])].copy()
    for col in ["top_pct", "selected_rate", "threshold_min", "capture_rate_recall", "precision", "lift_vs_base"]:
        top_display[col] = top_display[col].map(lambda x: fmt(x, 4))

    or_display = or_table.copy()
    or_display["odds_ratio"] = or_display["odds_ratio"].map(lambda x: fmt(x, 3))
    or_display["or_ci_low"] = or_display["or_ci_low"].map(lambda x: fmt(x, 3))
    or_display["or_ci_high"] = or_display["or_ci_high"].map(lambda x: fmt(x, 3))
    or_display["p_value"] = or_display["p_value"].map(lambda x: fmt(x, 4))
    or_display["q_value"] = or_display["q_value"].map(lambda x: fmt(x, 4))

    perf = metrics.loc[metrics["model"] == PERFORMANCE_MODEL].iloc[0]
    interp = metrics.loc[metrics["model"] == INTERPRET_MODEL].iloc[0]
    landcover = metrics.loc[metrics["model"] == REJECTED_LANDCOVER_MODEL].iloc[0]

    md = f"""# Step17 최종 로지스틱 결과 통합 정리

## 1. 결론

- Step18은 별도 단계로 분리하지 않고 Step17에 통합했다.
- 새 모델을 추가 학습하지 않고, Stage11·Stage16 산출물을 최종 보고서용으로 정리했다.
- 성능 비교용 로지스틱은 `{PERFORMANCE_MODEL}`로 유지한다.
- 해석·오즈비용 로지스틱은 `{INTERPRET_MODEL}`로 확정한다.
- `{REJECTED_LANDCOVER_MODEL}`는 AUPRC가 {float(landcover['auprc']):.4f}로 근소하게 높지만 max VIF가 {float(landcover['max_vif']):.2f}라 최종 해석 모델에서 제외한다.

## 2. 모델 역할 정리

{role_table.to_markdown(index=False)}

## 3. 머신러닝 비교용 성능표

{metrics_display[['source_stage','model_role','model','interpretation_use','positive_rate','auprc','auroc','brier','log_loss','best_f1_f1','best_f1_precision','best_f1_recall','best_f1_accuracy','best_f1_balanced_accuracy','max_vif']].to_markdown(index=False)}

해석:

- `{PERFORMANCE_MODEL}`는 AUPRC {float(perf['auprc']):.4f}, ROC AUC {float(perf['auroc']):.4f}, F1 {float(perf['best_f1_f1']):.4f}로 성능 비교 기준 모델로 사용한다.
- `{INTERPRET_MODEL}`는 AUPRC {float(interp['auprc']):.4f}, ROC AUC {float(interp['auroc']):.4f}, F1 {float(interp['best_f1_f1']):.4f}, max VIF {float(interp['max_vif']):.2f}로 해석 안정성이 높다.
- 불균형 비율이 약 {float(interp['positive_rate']):.1%}이므로 accuracy 단독 평가는 부적절하다. AUPRC, ROC AUC, F1, precision, recall을 함께 제시한다.

## 4. threshold/운영 성능 요약

{op_display[['model_role','model','operating_point','threshold','selected_n','selected_rate','precision','recall','specificity','f1','accuracy','balanced_accuracy']].to_markdown(index=False)}

해석:

- 0.50 고정 threshold는 불균형 자료에서 너무 보수적일 수 있으므로 최종 비교의 중심으로 쓰지 않는다.
- best-F1 threshold는 모델 간 비교용으로 쓴다.
- recall 0.70 또는 0.90 threshold는 운영적으로 산불을 더 많이 잡고 싶을 때 필요한 경보량 증가를 보여주는 보조 표로만 사용한다.

## 5. 상위 위험도 구간 포착률

{top_display[['model_role','model','top_pct','selected_n','threshold_min','captured_positive_n','total_positive_n','capture_rate_recall','precision','lift_vs_base']].to_markdown(index=False)}

해석:

- 상위 5%, 10%, 20% 위험도 구간은 실제 운영에서 “몇 개를 우선 점검하면 산불 노출을 얼마나 포착하는가”를 설명한다.
- 이 표는 threshold 하나를 고정하는 것보다 불균형 자료에서 직관적이다.

## 6. 최종 해석용 OR 표

{or_display[['concept_group','term_label','unit_for_or','or_95ci','p_value','q_value','significant_q05','report_interpretation']].to_markdown(index=False)}

## 7. EDA와 로지스틱 연결

{eda_link.to_markdown(index=False)}

## 8. 보고서 작성 기준

- 성능 비교 문단에서는 Stage11 `{PERFORMANCE_MODEL}`와 Stage16 `{INTERPRET_MODEL}`를 함께 보여준다.
- 오즈비 해석 문단에서는 Stage16 `{INTERPRET_MODEL}`만 사용한다.
- 토지피복 세부 범주와 단순 토지피복 플래그 모델은 성능이 조금 좋아도 공선성 때문에 계수 해석에 쓰지 않는다.
- 요인점수는 최종 로지스틱 주 모델에서 제외한다.
- 최종 표현은 “산불 발생을 완벽히 예측했다”가 아니라 “발생 노출에서 반복적으로 관찰되는 조건부 연관성을 정량화했다”로 쓴다.

## 9. 산출물

- `outputs/tables/stage17_model_role_summary.csv`
- `outputs/tables/stage17_ml_comparison_metrics.csv`
- `outputs/tables/stage17_operating_thresholds.csv`
- `outputs/tables/stage17_top_risk_capture.csv`
- `outputs/tables/stage17_final_or_report_table.csv`
- `outputs/tables/stage17_eda_linkage_table.csv`
- `outputs/plots/stage17_metric_role_comparison.png`
- `outputs/plots/stage17_top_risk_capture.png`
"""
    SUMMARY_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    predictions = load_prediction_series()
    role_table = build_role_table()
    metrics = build_metrics_table(predictions)
    operating, top_risk = build_operating_tables(predictions)
    or_table = build_or_report_table()
    eda_link = build_eda_linkage_table()

    role_table.to_csv(ROLE_TABLE_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(METRICS_TABLE_PATH, index=False, encoding="utf-8-sig")
    operating.to_csv(OPERATING_TABLE_PATH, index=False, encoding="utf-8-sig")
    top_risk.to_csv(TOP_RISK_TABLE_PATH, index=False, encoding="utf-8-sig")
    or_table.to_csv(OR_REPORT_TABLE_PATH, index=False, encoding="utf-8-sig")
    eda_link.to_csv(EDA_LINK_TABLE_PATH, index=False, encoding="utf-8-sig")

    plot_metric_comparison(metrics)
    plot_top_risk(top_risk)
    write_summary(role_table, metrics, operating, top_risk, or_table, eda_link)

    print("Stage17 완료")
    print(f"요약: {SUMMARY_PATH}")
    print(f"성능표: {METRICS_TABLE_PATH}")
    print(f"운영 threshold: {OPERATING_TABLE_PATH}")


if __name__ == "__main__":
    main()
