from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "jsw" / "Analysis" / "logistic" / "outputs").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
OUTPUT_DIR = ROOT / "jsw" / "Analysis" / "logistic" / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(METRIC_DIR / name, encoding="utf-8-sig")


def dedupe_keep_best(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["_rank"] = rows.groupby("feature_set")["auprc"].rank(method="first", ascending=False)
    rows = rows.loc[rows["_rank"].eq(1)].drop(columns="_rank")
    return rows


def main() -> None:
    overall = pd.concat(
        [
            read_csv("stage9_logistic_enhancement_overall_metrics.csv"),
            read_csv("stage10_logistic_stat_extensions_overall_metrics.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    overall = dedupe_keep_best(overall)

    thresholds = pd.concat(
        [
            read_csv("stage9_logistic_enhancement_threshold_metrics.csv"),
            read_csv("stage10_logistic_stat_extensions_threshold_metrics.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    thresholds = thresholds.drop_duplicates(["feature_set", "threshold_type"], keep="last")

    best_f1 = thresholds.loc[thresholds["threshold_type"].eq("best_f1_oof")].copy()
    best_f1 = best_f1.add_prefix("best_f1_").rename(columns={"best_f1_feature_set": "feature_set"})

    fixed_05 = thresholds.loc[thresholds["threshold_type"].eq("fixed_0.5")].copy()
    fixed_05 = fixed_05.add_prefix("fixed05_").rename(columns={"fixed05_feature_set": "feature_set"})

    recall90 = thresholds.loc[thresholds["threshold_type"].eq("recall90_oof")].copy()
    recall90 = recall90.add_prefix("recall90_").rename(columns={"recall90_feature_set": "feature_set"})

    negative = pd.concat(
        [
            read_csv("stage9_logistic_enhancement_sample_type_metrics.csv"),
            read_csv("stage10_logistic_stat_extensions_sample_type_metrics.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    negative = negative.drop_duplicates(["feature_set", "negative_type"], keep="last")
    negative_cols = ["auprc", "auroc", "brier", "log_loss"]
    negative_wide = negative.pivot(index="feature_set", columns="negative_type", values=negative_cols)
    negative_wide.columns = [f"{negative_type}_{metric}" for metric, negative_type in negative_wide.columns]
    negative_wide = negative_wide.reset_index()

    table = (
        overall.merge(best_f1, on="feature_set", how="left")
        .merge(fixed_05, on="feature_set", how="left")
        .merge(recall90, on="feature_set", how="left")
        .merge(negative_wide, on="feature_set", how="left")
    )

    preferred_order = [
        "feature_set",
        "n",
        "positive_n",
        "positive_rate",
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
        "best_f1_tn",
        "best_f1_fp",
        "best_f1_fn",
        "best_f1_tp",
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
        "Target_0B1_brier",
        "Target_0B1_log_loss",
        "Target_0B2_auprc",
        "Target_0B2_auroc",
        "Target_0B2_brier",
        "Target_0B2_log_loss",
    ]
    existing_order = [col for col in preferred_order if col in table.columns]
    remaining = [col for col in table.columns if col not in existing_order]
    table = table[existing_order + remaining].sort_values(["auprc", "Target_0A_auprc"], ascending=False)

    csv_path = OUTPUT_DIR / "logistic_benchmark_for_model_comparison.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    key_models = [
        "PLUS_LANDCOVER_RULES_ANOVA",
        "PLUS_LANDCOVER",
        "LOGIT_LIGHT_INTERACTIONS_L2_C1",
        "LOGIT_ELASTICNET_C1_L1R05",
        "LOGIT_BALANCED_L2_C1",
        "LOGIT_0A_WEIGHT_X4_L2_C1",
    ]
    compact = table.loc[table["feature_set"].isin(key_models)].copy()
    compact_cols = [
        "feature_set",
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
    ]
    compact = compact[[col for col in compact_cols if col in compact.columns]]

    best = table.iloc[0]
    best_f1_row = table.sort_values("best_f1_f1", ascending=False).iloc[0]
    best_0a = table.sort_values("Target_0A_auprc", ascending=False).iloc[0]
    best_brier = table.sort_values("brier", ascending=True).iloc[0]

    summary_lines = [
        "# 로지스틱 모델 ML 비교용 성능표",
        "",
        "## 1. 목적",
        "",
        "- 다른 모델 결과와 같은 열 구조로 비교하기 위한 로지스틱 성능표이다.",
        "- strict `date_exposure_component_cv` 기준 development OOF 결과만 사용했다.",
        "- lockbox는 사용하지 않았다.",
        "",
        "## 2. 포함 지표",
        "",
        "- 전체 순위 성능: AUPRC, ROC AUC",
        "- 확률 오차: Brier score, log loss",
        "- best-F1 운영점: threshold, accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, TN/FP/FN/TP",
        "- fixed 0.5 운영점: accuracy, precision, recall, F1",
        "- recall 90% 운영점: threshold, accuracy, precision, recall, F1",
        "- 대조군별 성능: Target_0A, Target_0B1, Target_0B2 각각 AUPRC, ROC AUC, Brier, log loss",
        "",
        "## 3. 주요 로지스틱 후보 요약",
        "",
        compact.round(5).to_markdown(index=False),
        "",
        "## 4. 현재 최고값",
        "",
        f"- 전체 AUPRC 최고: `{best['feature_set']}` / AUPRC {best.auprc:.4f}",
        f"- best-F1 최고: `{best_f1_row['feature_set']}` / F1 {best_f1_row.best_f1_f1:.4f}",
        f"- 0-A AUPRC 최고: `{best_0a['feature_set']}` / 0-A AUPRC {best_0a.Target_0A_auprc:.4f}",
        f"- Brier 최저: `{best_brier['feature_set']}` / Brier {best_brier.brier:.5f}",
        "",
        "## 5. 산출물",
        "",
        f"- CSV: `{csv_path.relative_to(ROOT)}`",
    ]
    (OUTPUT_DIR / "logistic_benchmark_for_model_comparison.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
