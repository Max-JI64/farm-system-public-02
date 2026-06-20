from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "jsw" / "Analysis" / "logistic").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
LOGISTIC_DIR = ROOT / "jsw" / "Analysis" / "logistic"
OUTPUT_DIR = LOGISTIC_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

BENCHMARK_PATH = OUTPUT_DIR / "logistic_benchmark_for_model_comparison.csv"

REPRESENTATIVE_MODEL = "PLUS_LANDCOVER_RULES_ANOVA"
F1_AUX_MODEL = "PLUS_LANDCOVER"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def read_benchmark() -> pd.DataFrame:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"기존 benchmark CSV가 없습니다: {BENCHMARK_PATH}\n"
            "`make_logistic_benchmark_tables.py`를 먼저 실행해야 합니다."
        )
    return pd.read_csv(BENCHMARK_PATH, encoding="utf-8-sig")


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    columns = set(columns)
    return [col for col in candidates if col in columns]


def add_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    role = []
    decision = []
    reason = []

    for feature_set in df["feature_set"]:
        if feature_set == REPRESENTATIVE_MODEL:
            role.append("대표 성능 모델")
            decision.append("최종 성능 비교의 기본 로지스틱 기준선")
            reason.append("AUPRC, ROC AUC, Brier, 0-A AUPRC가 가장 안정적")
        elif feature_set == F1_AUX_MODEL:
            role.append("F1 운영점 보조 모델")
            decision.append("F1-score 중심 비교 시 함께 제시")
            reason.append("best-F1이 가장 높고 대표 모델과 성능 차이가 작음")
        elif "BALANCED" in feature_set or "WEIGHT" in feature_set:
            role.append("진단용 참고 모델")
            decision.append("최종 대표 모델로는 사용하지 않음")
            reason.append("recall 또는 특정 대조군 성능은 높일 수 있으나 Brier/log loss가 악화됨")
        elif "ELASTICNET" in feature_set:
            role.append("정규화 참고 모델")
            decision.append("해석 안정성 비교용 참고")
            reason.append("성능은 유지되지만 대표 모델보다 AUPRC가 낮음")
        else:
            role.append("참고 모델")
            decision.append("최종 대표 모델로는 사용하지 않음")
            reason.append("대표 모델 대비 전체 성능 개선이 없음")

    df.insert(1, "stage11_role", role)
    df.insert(2, "stage11_decision", decision)
    df.insert(3, "stage11_reason", reason)
    return df


def metric_leaders(df: pd.DataFrame) -> dict[str, pd.Series]:
    required = ["auprc", "auroc", "brier", "log_loss", "best_f1_f1", "Target_0A_auprc"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"benchmark CSV에 필요한 컬럼이 없습니다: {missing}")

    return {
        "AUPRC 최고": df.sort_values("auprc", ascending=False).iloc[0],
        "ROC AUC 최고": df.sort_values("auroc", ascending=False).iloc[0],
        "Brier 최저": df.sort_values("brier", ascending=True).iloc[0],
        "log loss 최저": df.sort_values("log_loss", ascending=True).iloc[0],
        "best-F1 최고": df.sort_values("best_f1_f1", ascending=False).iloc[0],
        "0-A AUPRC 최고": df.sort_values("Target_0A_auprc", ascending=False).iloc[0],
    }


def select_compact_columns(df: pd.DataFrame) -> list[str]:
    return first_existing(
        df.columns,
        [
            "feature_set",
            "stage11_role",
            "auprc",
            "auroc",
            "brier",
            "log_loss",
            "best_f1_threshold",
            "best_f1_accuracy",
            "best_f1_balanced_accuracy",
            "best_f1_precision",
            "best_f1_recall",
            "best_f1_specificity",
            "best_f1_f1",
            "best_f1_mcc",
            "fixed05_accuracy",
            "fixed05_precision",
            "fixed05_recall",
            "fixed05_f1",
            "recall90_threshold",
            "recall90_accuracy",
            "recall90_precision",
            "recall90_recall",
            "recall90_f1",
            "Target_0A_auprc",
            "Target_0A_auroc",
            "Target_0A_brier",
            "Target_0A_log_loss",
            "Target_0B1_auprc",
            "Target_0B1_auroc",
            "Target_0B2_auprc",
            "Target_0B2_auroc",
            "stage11_decision",
            "stage11_reason",
        ],
    )


def round_for_report(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(5)
    return out


def to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def make_key_model_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        REPRESENTATIVE_MODEL,
        F1_AUX_MODEL,
        "LOGIT_LIGHT_INTERACTIONS_L2_C1",
        "LOGIT_ELASTICNET_C1_L1R05",
        "LOGIT_BALANCED_L2_C1",
        "LOGIT_0A_WEIGHT_X4_L2_C1",
    ]
    compact_cols = first_existing(
        df.columns,
        [
            "feature_set",
            "stage11_role",
            "auprc",
            "auroc",
            "brier",
            "log_loss",
            "best_f1_accuracy",
            "best_f1_precision",
            "best_f1_recall",
            "best_f1_f1",
            "best_f1_mcc",
            "Target_0A_auprc",
            "Target_0B1_auprc",
            "Target_0B2_auprc",
            "stage11_decision",
        ],
    )
    return df.loc[df["feature_set"].isin(keep), compact_cols].copy()


def make_threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.loc[df["feature_set"].isin([REPRESENTATIVE_MODEL, F1_AUX_MODEL])].copy()
    cols = first_existing(
        rows.columns,
        [
            "feature_set",
            "stage11_role",
            "best_f1_threshold",
            "best_f1_accuracy",
            "best_f1_balanced_accuracy",
            "best_f1_precision",
            "best_f1_recall",
            "best_f1_specificity",
            "best_f1_f1",
            "best_f1_mcc",
            "fixed05_accuracy",
            "fixed05_precision",
            "fixed05_recall",
            "fixed05_f1",
            "recall90_threshold",
            "recall90_accuracy",
            "recall90_precision",
            "recall90_recall",
            "recall90_f1",
        ],
    )
    return rows[cols].copy()


def maybe_make_plot(df: pd.DataFrame) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    key = make_key_model_table(df)
    key = key.sort_values("auprc", ascending=True)
    labels = key["feature_set"].tolist()
    metrics = ["auprc", "auroc", "best_f1_f1", "Target_0A_auprc"]
    colors = ["#2563eb", "#16a34a", "#f97316", "#9333ea"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), constrained_layout=True)
    for ax, metric, color in zip(axes, metrics, colors):
        ax.barh(labels, key[metric], color=color, alpha=0.85)
        ax.set_title(metric)
        ax.set_xlim(0, max(key[metric].max() * 1.15, 0.05))
        ax.grid(axis="x", alpha=0.25)
        for i, value in enumerate(key[metric]):
            ax.text(value, i, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("Stage11 final logistic benchmark candidates", fontsize=13)

    out_path = PLOT_DIR / "stage11_final_logistic_benchmark_metrics.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_outputs(df: pd.DataFrame) -> None:
    ensure_dirs()

    final_cols = select_compact_columns(df)
    final_table = df[final_cols].copy()
    final_table = final_table.sort_values(["auprc", "Target_0A_auprc"], ascending=False)

    final_csv = TABLE_DIR / "stage11_final_logistic_benchmark.csv"
    final_md = TABLE_DIR / "stage11_final_logistic_benchmark.md"
    decision_csv = TABLE_DIR / "stage11_final_model_decision.csv"
    threshold_csv = TABLE_DIR / "stage11_threshold_operating_points.csv"
    summary_md = OUTPUT_DIR / "stage11_final_logistic_benchmark_summary.md"

    final_table.to_csv(final_csv, index=False, encoding="utf-8-sig")
    final_md.write_text(to_markdown(round_for_report(final_table)) + "\n", encoding="utf-8")

    decision_cols = first_existing(
        df.columns,
        [
            "feature_set",
            "stage11_role",
            "stage11_decision",
            "stage11_reason",
            "auprc",
            "auroc",
            "brier",
            "log_loss",
            "best_f1_f1",
            "Target_0A_auprc",
        ],
    )
    decisions = df.loc[df["feature_set"].isin([REPRESENTATIVE_MODEL, F1_AUX_MODEL]), decision_cols].copy()
    decisions.to_csv(decision_csv, index=False, encoding="utf-8-sig")

    threshold_table = make_threshold_table(df)
    threshold_table.to_csv(threshold_csv, index=False, encoding="utf-8-sig")

    plot_path = maybe_make_plot(df)

    leaders = metric_leaders(df)
    rep = df.loc[df["feature_set"].eq(REPRESENTATIVE_MODEL)].iloc[0]
    aux = df.loc[df["feature_set"].eq(F1_AUX_MODEL)].iloc[0]
    positive_rate = float(rep["positive_rate"]) if "positive_rate" in rep else float("nan")
    auprc_lift = rep["auprc"] / positive_rate if positive_rate and positive_rate > 0 else float("nan")

    key_table = make_key_model_table(df)

    lines = [
        "# Stage 11 최종 로지스틱 성능 기준 모델 확정",
        "",
        "## 1. 목적",
        "",
        "- 다른 모델 결과와 비교할 로지스틱 기준선을 고정했다.",
        "- strict `date_exposure_component_cv` 기준 development OOF 결과만 사용했다.",
        "- lockbox는 사용하지 않았다.",
        "- 요인점수는 제외한 로지스틱 흐름을 유지했다.",
        "- 비통계적 이진분류 모델은 포함하지 않았다.",
        "",
        "## 2. 최종 모델 판단",
        "",
        f"- 대표 성능 모델: `{REPRESENTATIVE_MODEL}`",
        f"  - AUPRC {rep.auprc:.4f}, ROC AUC {rep.auroc:.4f}, Brier {rep.brier:.5f}, log loss {rep.log_loss:.5f}",
        f"  - best-F1 운영점: threshold {rep.best_f1_threshold:.4f}, Accuracy {rep.best_f1_accuracy:.4f}, Precision {rep.best_f1_precision:.4f}, Recall {rep.best_f1_recall:.4f}, F1 {rep.best_f1_f1:.4f}, MCC {rep.best_f1_mcc:.4f}",
        f"  - 0-A AUPRC {rep.Target_0A_auprc:.4f}",
        f"- F1 운영점 보조 모델: `{F1_AUX_MODEL}`",
        f"  - AUPRC {aux.auprc:.4f}, ROC AUC {aux.auroc:.4f}, Brier {aux.brier:.5f}, log loss {aux.log_loss:.5f}",
        f"  - best-F1 운영점: threshold {aux.best_f1_threshold:.4f}, Accuracy {aux.best_f1_accuracy:.4f}, Precision {aux.best_f1_precision:.4f}, Recall {aux.best_f1_recall:.4f}, F1 {aux.best_f1_f1:.4f}, MCC {aux.best_f1_mcc:.4f}",
        "",
        "## 3. 성능 해석",
        "",
        f"- development OOF 표본의 양성 비율은 {positive_rate:.4f}이다.",
        f"- 대표 모델의 AUPRC는 {rep.auprc:.4f}로, 무작위 기준 양성 비율 대비 약 {auprc_lift:.2f}배이다.",
        f"- ROC AUC는 {rep.auroc:.4f}로 순위화 성능은 중간 이상이지만, best-F1 precision은 {rep.best_f1_precision:.4f} 수준이라 실제 양성 예측의 정밀도는 높지 않다.",
        f"- 이는 산불 발생 표본 비율이 낮고 0-A 대조군 구분이 어렵기 때문에, 로지스틱을 고정 threshold 예측기보다 위험도 순위화 모델로 보는 것이 더 적절하다는 의미이다.",
        f"- fixed 0.5 threshold에서는 대표 모델 recall이 {rep.fixed05_recall:.4f}로 매우 낮다. 따라서 0.5 고정 threshold는 운영 기준으로 쓰지 않는다.",
        "",
        "## 4. 주요 후보 비교",
        "",
        to_markdown(round_for_report(key_table)),
        "",
        "## 5. 지표별 최고 모델",
        "",
    ]

    for metric_name, row in leaders.items():
        if metric_name == "Brier 최저":
            value = row["brier"]
            value_text = f"{value:.5f}"
        elif metric_name == "log loss 최저":
            value = row["log_loss"]
            value_text = f"{value:.5f}"
        elif metric_name == "best-F1 최고":
            value = row["best_f1_f1"]
            value_text = f"{value:.4f}"
        elif metric_name == "0-A AUPRC 최고":
            value = row["Target_0A_auprc"]
            value_text = f"{value:.4f}"
        elif metric_name == "ROC AUC 최고":
            value = row["auroc"]
            value_text = f"{value:.4f}"
        else:
            value = row["auprc"]
            value_text = f"{value:.4f}"
        lines.append(f"- {metric_name}: `{row['feature_set']}` / {value_text}")

    lines += [
        "",
        "## 6. threshold 운영점 비교",
        "",
        to_markdown(round_for_report(threshold_table)),
        "",
        "## 7. 최종 사용 방침",
        "",
        f"- 다른 모델과 대표 성능을 비교할 때는 `{REPRESENTATIVE_MODEL}`를 사용한다.",
        f"- F1-score 중심 비교에서는 `{F1_AUX_MODEL}`를 보조로 같이 제시한다.",
        "- 확률 보정과 전체 성능까지 고려하면 class-weight 또는 0-A 가중 모델은 최종 대표 모델로 쓰지 않는다.",
        "- 다음 Step 12부터는 예측 성능 모델과 별도로 해석용 로지스틱 모델을 구성한다.",
        "- 성능 모델의 ANOVA 선택 계수를 그대로 오즈비 해석에 사용하지 않는다.",
        "",
        "## 8. 산출물",
        "",
        f"- `{final_csv.relative_to(ROOT)}`",
        f"- `{final_md.relative_to(ROOT)}`",
        f"- `{decision_csv.relative_to(ROOT)}`",
        f"- `{threshold_csv.relative_to(ROOT)}`",
    ]
    if plot_path is not None:
        lines.append(f"- `{plot_path.relative_to(ROOT)}`")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    benchmark = read_benchmark()
    benchmark = add_decision_columns(benchmark)
    write_outputs(benchmark)
    print("Stage11 완료")
    print(f"대표 성능 모델: {REPRESENTATIVE_MODEL}")
    print(f"F1 운영점 보조 모델: {F1_AUX_MODEL}")
    print(f"요약: {OUTPUT_DIR / 'stage11_final_logistic_benchmark_summary.md'}")


if __name__ == "__main__":
    main()
