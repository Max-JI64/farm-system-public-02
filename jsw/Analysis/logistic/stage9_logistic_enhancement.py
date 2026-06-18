from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stage7_feature_extension import probability_metrics
from stage8_d2d3_logistic_analysis import (
    LANDCOVER_FEATURES,
    classification_metrics_at_threshold,
    make_threshold_table,
    negative_type_metrics,
    recall_threshold,
    subgroup_metrics,
    top_risk_metrics,
)


warnings.filterwarnings("ignore")
RANDOM_STATE = 20260618


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "학습데이터" / "학습데이터_로지스틱_D2D3.csv").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "학습데이터"
LOGISTIC_DIR = ROOT / "jsw" / "Analysis" / "logistic"
OUTPUT_DIR = LOGISTIC_DIR / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
PLOT_DIR = OUTPUT_DIR / "plots"
COEFFICIENT_DIR = OUTPUT_DIR / "coefficients"
SPLIT_DIR = OUTPUT_DIR / "splits"
for directory in [FEATURE_DIR, METRIC_DIR, PREDICTION_DIR, PLOT_DIR, COEFFICIENT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "학습데이터_로지스틱_D2D3.csv"
ENGINEERED_PATH = FEATURE_DIR / "stage7_engineered_features.csv"
RECOMMENDED_PATH = FEATURE_DIR / "stage7_recommended_feature_set.json"
OUTER_PATH = SPLIT_DIR / "stage75_date_exposure_component_cv_outer_cv_manifest.csv"
INNER_PATH = SPLIT_DIR / "stage75_date_exposure_component_cv_inner_cv_manifest.csv"
STAGE75_PREDICTION_PATH = PREDICTION_DIR / "stage75_strict_oof_predictions.csv"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


BASE_CATEGORICAL = [
    "기후지형유형",
    "토지피복_L1_NAME",
    "토지피복_L2_NAME",
    "토지피복_매칭방식",
    "토지피복_산림유형",
]


RULE_FEATURES = [
    "도로_10m_이내",
    "도로_30m_이내",
    "시가화_10m_이내",
    "시가화_30m_이내",
    "산림_10m_이내",
    "비산림WUI_x_도로10m",
    "비산림WUI_x_rh_q05",
    "비산림WUI_x_dry0p1",
    "비산림WUI_x_wind5",
    "산림지역_x_rh_q05",
    "침엽수림_x_rh_q05",
    "활엽수림_x_rh_q05",
    "혼효림_x_rh_q05",
    "침엽수림_x_dry5",
    "초지_x_dry0p1",
    "시가화_x_도로10m",
    "영동_x_토지피복도로",
    "영동_x_침엽수림",
    "영동_x_rh_q05_x_wind5",
    "영서_x_비산림WUI_x_dry0p1",
]


BIN_FEATURES = [
    "직전24h_최소습도_bin",
    "wind_max_6h_bin",
    "dry_spell_5p0_bin",
    "도로거리_bin",
    "D1_FWI_bin",
    "D1_ISI_bin",
]


def add_stage9_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["영동_여부"] = data["기후지형유형"].eq("영동 해안형").astype(np.int8)
    data["영서_여부"] = data["기후지형유형"].eq("영서 내륙형").astype(np.int8)
    data["고지산간_여부"] = data["기후지형유형"].eq("고지·산간형").astype(np.int8)

    data["도로_10m_이내"] = data["도로_최단거리_m"].le(10).astype(np.int8)
    data["도로_30m_이내"] = data["도로_최단거리_m"].le(30).astype(np.int8)
    data["시가화_10m_이내"] = data["시가화_최단거리_m"].le(10).astype(np.int8)
    data["시가화_30m_이내"] = data["시가화_최단거리_m"].le(30).astype(np.int8)
    data["산림_10m_이내"] = data["산림_최단거리_m"].le(10).astype(np.int8)

    data["비산림WUI_x_도로10m"] = data["비산림_WUI_접경후보"] * data["도로_10m_이내"]
    data["비산림WUI_x_rh_q05"] = data["비산림_WUI_접경후보"] * data["rh_local_q05"]
    data["비산림WUI_x_dry0p1"] = data["비산림_WUI_접경후보"] * data["dry_spell_0p1_gt_24h"]
    data["비산림WUI_x_wind5"] = data["비산림_WUI_접경후보"] * data["wind_max_6h_ge_5"]
    data["산림지역_x_rh_q05"] = data["토지피복_산림지역"] * data["rh_local_q05"]
    data["침엽수림_x_rh_q05"] = data["토지피복_침엽수림"] * data["rh_local_q05"]
    data["활엽수림_x_rh_q05"] = data["토지피복_활엽수림"] * data["rh_local_q05"]
    data["혼효림_x_rh_q05"] = data["토지피복_혼효림"] * data["rh_local_q05"]
    data["침엽수림_x_dry5"] = data["토지피복_침엽수림"] * data["dry_spell_5p0_gt_240h"]
    data["초지_x_dry0p1"] = data["토지피복_초지"] * data["dry_spell_0p1_gt_24h"]
    data["시가화_x_도로10m"] = data["토지피복_시가화건조지역"] * data["도로_10m_이내"]
    data["영동_x_토지피복도로"] = data["영동_여부"] * data["토지피복_도로"]
    data["영동_x_침엽수림"] = data["영동_여부"] * data["토지피복_침엽수림"]
    data["영동_x_rh_q05_x_wind5"] = data["영동_여부"] * data["rh_local_q05"] * data["wind_max_6h_ge_5"]
    data["영서_x_비산림WUI_x_dry0p1"] = (
        data["영서_여부"] * data["비산림_WUI_접경후보"] * data["dry_spell_0p1_gt_24h"]
    )

    data["직전24h_최소습도_bin"] = pd.cut(
        data["직전24h_최소습도"],
        bins=[-np.inf, 20, 30, 40, 60, np.inf],
        labels=["<=20", "20-30", "30-40", "40-60", ">60"],
    ).astype(str)
    data["wind_max_6h_bin"] = pd.cut(
        data["wind_max_6h"],
        bins=[-np.inf, 2, 3, 5, np.inf],
        labels=["<=2", "2-3", "3-5", ">5"],
    ).astype(str)
    data["dry_spell_5p0_bin"] = pd.cut(
        data["dry_spell_h_5p0"],
        bins=[-np.inf, 24, 72, 240, np.inf],
        labels=["<=24h", "24-72h", "72-240h", ">240h"],
    ).astype(str)
    data["도로거리_bin"] = pd.cut(
        data["도로_최단거리_m"],
        bins=[-np.inf, 10, 30, 100, np.inf],
        labels=["<=10m", "10-30m", "30-100m", ">100m"],
    ).astype(str)
    data["D1_FWI_bin"] = pd.cut(
        data["D1_FWI"],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=["<=5", "5-10", "10-20", ">20"],
    ).astype(str)
    data["D1_ISI_bin"] = pd.cut(
        data["D1_ISI"],
        bins=[-np.inf, 3, 6, 10, np.inf],
        labels=["<=3", "3-6", "6-10", ">10"],
    ).astype(str)
    return data


def make_pipeline(
    features: list[str],
    categorical: list[str],
    c_value: float,
    penalty: str = "l2",
    selector_k: int | str | None = None,
) -> Pipeline:
    categorical_used = [c for c in categorical if c in features]
    numeric = [c for c in features if c not in categorical_used]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                categorical_used,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps = [("preprocess", preprocessor)]
    if selector_k is not None:
        steps.append(("select", SelectKBest(score_func=f_classif, k=selector_k)))
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = LogisticRegression(
        penalty=penalty,
        C=float(c_value),
        solver=solver,
        class_weight=None,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    steps.append(("model", model))
    return Pipeline(steps)


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if "select" in pipeline.named_steps:
        selector = pipeline.named_steps["select"]
        try:
            names = names[selector.get_support()]
        except Exception:
            pass
    return names


def nested_oof(
    data: pd.DataFrame,
    feature_set: str,
    features: list[str],
    categorical: list[str],
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    configs: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = data.set_index("샘플ID", drop=False)
    oof_rows = []
    fold_rows = []
    tuning_rows = []
    coefficient_rows = []

    for outer_fold in range(5):
        val_ids = outer.loc[outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
        train_ids = outer.loc[~outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
        inner_fold_map = inner.loc[inner["outer_fold"].eq(outer_fold)].set_index("샘플ID")

        config_scores = []
        for config_id, config in enumerate(configs):
            scores = []
            for inner_fold in range(4):
                inner_val_ids = inner_fold_map.index[inner_fold_map["inner_fold"].eq(inner_fold)].tolist()
                inner_train_ids = inner_fold_map.index[~inner_fold_map["inner_fold"].eq(inner_fold)].tolist()
                pipeline = make_pipeline(
                    features=features,
                    categorical=categorical,
                    c_value=config["C"],
                    penalty=config["penalty"],
                    selector_k=config.get("selector_k"),
                )
                pipeline.fit(indexed.loc[inner_train_ids, features], indexed.loc[inner_train_ids, "Target"])
                probability = pipeline.predict_proba(indexed.loc[inner_val_ids, features])[:, 1]
                score = average_precision_score(indexed.loc[inner_val_ids, "Target"], probability)
                scores.append(float(score))
                tuning_rows.append(
                    {
                        "feature_set": feature_set,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "config_id": config_id,
                        **config,
                        "auprc": float(score),
                    }
                )
            config_scores.append(
                {
                    "config_id": config_id,
                    "mean_auprc": float(np.mean(scores)),
                    **config,
                }
            )
        best = sorted(config_scores, key=lambda row: (-row["mean_auprc"], row["config_id"]))[0]
        pipeline = make_pipeline(
            features=features,
            categorical=categorical,
            c_value=best["C"],
            penalty=best["penalty"],
            selector_k=best.get("selector_k"),
        )
        pipeline.fit(indexed.loc[train_ids, features], indexed.loc[train_ids, "Target"])
        train_probability = pipeline.predict_proba(indexed.loc[train_ids, features])[:, 1]
        val_probability = pipeline.predict_proba(indexed.loc[val_ids, features])[:, 1]

        for sample_id, probability in zip(val_ids, val_probability):
            row = indexed.loc[sample_id]
            oof_rows.append(
                {
                    "샘플ID": sample_id,
                    "feature_set": feature_set,
                    "outer_fold": outer_fold,
                    "Target": int(row["Target"]),
                    "샘플유형": row["샘플유형"],
                    "기후지형유형": row["기후지형유형"],
                    "probability": float(probability),
                    "selected_config_id": int(best["config_id"]),
                    "selected_C": float(best["C"]),
                    "selected_penalty": best["penalty"],
                    "selected_selector_k": str(best.get("selector_k", "")),
                }
            )
        for dataset_name, ids, probability in [
            ("train", train_ids, train_probability),
            ("validation", val_ids, val_probability),
        ]:
            fold_rows.append(
                {
                    "feature_set": feature_set,
                    "outer_fold": outer_fold,
                    "dataset": dataset_name,
                    "selected_config_id": int(best["config_id"]),
                    "selected_C": float(best["C"]),
                    "selected_penalty": best["penalty"],
                    "selected_selector_k": str(best.get("selector_k", "")),
                    **probability_metrics(indexed.loc[ids, "Target"], probability),
                }
            )

        names = get_feature_names(pipeline)
        coefs = pipeline.named_steps["model"].coef_[0]
        for name, coef in zip(names, coefs):
            coefficient_rows.append(
                {
                    "feature_set": feature_set,
                    "outer_fold": outer_fold,
                    "feature": name,
                    "coefficient": float(coef),
                }
            )
        print(
            f"{feature_set} outer={outer_fold}: "
            f"config={best['config_id']}, C={best['C']}, penalty={best['penalty']}, "
            f"k={best.get('selector_k', '')}, AUPRC={average_precision_score(indexed.loc[val_ids, 'Target'], val_probability):.4f}"
        )

    return (
        pd.DataFrame(oof_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficient_rows),
    )


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[str]]:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["기준시각"], low_memory=False)
    engineered = pd.read_csv(ENGINEERED_PATH, encoding="utf-8-sig")
    engineered_cols = [c for c in engineered.columns if c not in {"Target", "샘플유형"}]
    data = data.merge(engineered[engineered_cols], on="샘플ID", how="left", validate="one_to_one")
    data = add_stage9_features(data)
    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    inner = pd.read_csv(INNER_PATH, encoding="utf-8-sig")
    data = data.loc[data["샘플ID"].isin(set(outer["샘플ID"]))].copy()
    if len(data) != len(outer):
        raise ValueError("development 행 수가 strict outer manifest와 다릅니다.")

    with RECOMMENDED_PATH.open("r", encoding="utf-8") as file:
        recommended = json.load(file)
    base_features = list(recommended["features"]) + LANDCOVER_FEATURES
    feature_sets = {
        "PLUS_LANDCOVER_RULES_L2": base_features + RULE_FEATURES,
        "PLUS_LANDCOVER_RULES_ANOVA": base_features + RULE_FEATURES,
        "PLUS_LANDCOVER_RULES_L1": base_features + RULE_FEATURES,
        "PLUS_LANDCOVER_RULES_BINS_L2": base_features + RULE_FEATURES + BIN_FEATURES,
        "PLUS_LANDCOVER_RULES_BINS_ANOVA": base_features + RULE_FEATURES + BIN_FEATURES,
    }
    categorical = [c for c in BASE_CATEGORICAL + BIN_FEATURES if c in data.columns]

    all_features = sorted(set(sum(feature_sets.values(), [])))
    for col in all_features:
        if col in categorical:
            data[col] = data[col].fillna("미상").astype(str)
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    missing = data[all_features].isna().sum()
    if int(missing.sum()) != 0:
        raise ValueError("피처 결측 발생:\n" + missing[missing > 0].to_string())
    return data, outer, inner, feature_sets, categorical


def configs_for(feature_set: str) -> list[dict]:
    if feature_set.endswith("_ANOVA"):
        return [
            {"penalty": "l2", "C": c, "selector_k": k}
            for c in [0.1, 1.0, 10.0]
            for k in [40, 80, "all"]
        ]
    if feature_set.endswith("_L1"):
        return [{"penalty": "l1", "C": c} for c in [0.1, 1.0, 10.0]]
    return [{"penalty": "l2", "C": c} for c in [0.1, 1.0, 10.0, 100.0]]


def summarize(predictions: pd.DataFrame, baseline_predictions: pd.DataFrame) -> None:
    all_predictions = pd.concat([baseline_predictions, predictions], ignore_index=True)
    all_predictions.to_csv(
        PREDICTION_DIR / "stage9_logistic_enhancement_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_rows = []
    for feature_set, part in all_predictions.groupby("feature_set", observed=True):
        overall_rows.append({"feature_set": feature_set, **probability_metrics(part["Target"], part["probability"])})
    overall = pd.DataFrame(overall_rows).sort_values("auprc", ascending=False)
    baseline = overall.set_index("feature_set").loc["PLUS_LANDCOVER"]
    overall["delta_auprc_vs_plus_landcover"] = overall["auprc"] - baseline["auprc"]
    overall["delta_auroc_vs_plus_landcover"] = overall["auroc"] - baseline["auroc"]
    overall["delta_brier_vs_plus_landcover"] = overall["brier"] - baseline["brier"]
    overall.to_csv(METRIC_DIR / "stage9_logistic_enhancement_overall_metrics.csv", index=False, encoding="utf-8-sig")

    thresholds = make_threshold_table(all_predictions)
    thresholds.to_csv(METRIC_DIR / "stage9_logistic_enhancement_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    negative = negative_type_metrics(all_predictions)
    negative.to_csv(METRIC_DIR / "stage9_logistic_enhancement_sample_type_metrics.csv", index=False, encoding="utf-8-sig")
    climate = subgroup_metrics(all_predictions, "기후지형유형")
    climate.to_csv(METRIC_DIR / "stage9_logistic_enhancement_climate_metrics.csv", index=False, encoding="utf-8-sig")
    top_risk = pd.concat([top_risk_metrics(all_predictions, fs) for fs in overall["feature_set"]], ignore_index=True)
    top_risk.to_csv(METRIC_DIR / "stage9_logistic_enhancement_top_risk_metrics.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(11, 5))
    sns.barplot(data=overall, x="feature_set", y="auprc", color="#4C78A8")
    plt.axhline(float(baseline["auprc"]), color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Stage9 로지스틱 개선 후보 AUPRC")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stage9_01_logistic_enhancement_auprc.png", dpi=180)
    plt.close()

    best = overall.iloc[0]
    best_threshold = thresholds.loc[
        thresholds["feature_set"].eq(best["feature_set"]) & thresholds["threshold_type"].eq("best_f1_oof")
    ].iloc[0]
    baseline_threshold = thresholds.loc[
        thresholds["feature_set"].eq("PLUS_LANDCOVER") & thresholds["threshold_type"].eq("best_f1_oof")
    ].iloc[0]
    neg_idx = negative.set_index(["feature_set", "negative_type"])
    best_0a = neg_idx.loc[(best["feature_set"], "Target_0A")]
    base_0a = neg_idx.loc[("PLUS_LANDCOVER", "Target_0A")]

    summary = [
        "# Stage 9 로지스틱 추가 개선 실험",
        "",
        "## 1. 목적",
        "",
        "- Step 7.5의 가장 엄격한 `date_exposure_component_cv`를 그대로 사용했다.",
        "- lockbox는 열지 않았다.",
        "- 로지스틱 안에서 할 수 있는 추가 개선으로 EDA 기반 상호작용, 고정 bin 비선형화, ANOVA F-test 변수선택, L1 정규화를 비교했다.",
        "",
        "## 2. 전체 성능",
        "",
        overall[[
            "feature_set",
            "auprc",
            "auroc",
            "brier",
            "log_loss",
            "delta_auprc_vs_plus_landcover",
            "delta_auroc_vs_plus_landcover",
            "delta_brier_vs_plus_landcover",
        ]].round(5).to_markdown(index=False),
        "",
        "## 3. 최고 후보",
        "",
        f"- 최고 모델: `{best['feature_set']}`",
        f"- AUPRC: {best.auprc:.4f}",
        f"- ROC AUC: {best.auroc:.4f}",
        f"- Brier: {best.brier:.5f}",
        f"- `PLUS_LANDCOVER` 대비 AUPRC 변화: {best.delta_auprc_vs_plus_landcover:+.4f}",
        "",
        "## 4. F1 운영점",
        "",
        "| 모델 | threshold | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PLUS_LANDCOVER | {baseline_threshold.threshold:.4f} | {baseline_threshold.accuracy:.4f} | {baseline_threshold.precision:.4f} | {baseline_threshold.recall:.4f} | {baseline_threshold.f1:.4f} |",
        f"| {best['feature_set']} | {best_threshold.threshold:.4f} | {best_threshold.accuracy:.4f} | {best_threshold.precision:.4f} | {best_threshold.recall:.4f} | {best_threshold.f1:.4f} |",
        "",
        "## 5. 0-A 성능",
        "",
        f"- PLUS_LANDCOVER 0-A AUPRC: {base_0a.auprc:.4f}",
        f"- 최고 후보 0-A AUPRC: {best_0a.auprc:.4f}",
        f"- 변화: {best_0a.auprc - base_0a.auprc:+.4f}",
        "",
        "## 6. 판단",
        "",
        "- ANOVA 변수선택은 nested CV 내부에서 검증했으므로 leakage는 없다.",
        "- 단순 L1/L2/ANOVA 수준의 개선폭은 제한적이다.",
        "- 다음 단계는 같은 strict split 기준에서 로지스틱 계열 통계모델을 더 확장하는 것이다.",
    ]
    (OUTPUT_DIR / "stage9_logistic_enhancement_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("Stage 9 로지스틱 개선 실험 시작")
    data, outer, inner, feature_sets, categorical = prepare_data()
    baseline = pd.read_csv(STAGE75_PREDICTION_PATH, encoding="utf-8-sig")
    baseline = baseline.loc[
        baseline["strategy"].eq("date_exposure_component_cv")
        & baseline["feature_set"].isin(["STAGE7_RECOMMENDED", "PLUS_LANDCOVER"])
    ].copy()
    baseline = baseline.drop(columns=["strategy"], errors="ignore")

    oof_parts = []
    fold_parts = []
    tuning_parts = []
    coefficient_parts = []
    for feature_set, features in feature_sets.items():
        oof, fold, tuning, coef = nested_oof(
            data=data,
            feature_set=feature_set,
            features=features,
            categorical=categorical,
            outer=outer,
            inner=inner,
            configs=configs_for(feature_set),
        )
        oof_parts.append(oof)
        fold_parts.append(fold)
        tuning_parts.append(tuning)
        coefficient_parts.append(coef)

    predictions = pd.concat(oof_parts, ignore_index=True)
    predictions.to_csv(PREDICTION_DIR / "stage9_new_candidate_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(fold_parts, ignore_index=True).to_csv(
        METRIC_DIR / "stage9_logistic_enhancement_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(tuning_parts, ignore_index=True).to_csv(
        METRIC_DIR / "stage9_logistic_enhancement_inner_tuning.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(coefficient_parts, ignore_index=True).to_csv(
        COEFFICIENT_DIR / "stage9_logistic_enhancement_fold_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize(predictions, baseline)
    print("Stage 9 로지스틱 개선 실험 완료")


if __name__ == "__main__":
    main()
