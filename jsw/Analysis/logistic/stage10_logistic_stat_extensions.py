from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from stage7_feature_extension import probability_metrics
from stage8_d2d3_logistic_analysis import (
    LANDCOVER_FEATURES,
    make_threshold_table,
    negative_type_metrics,
    subgroup_metrics,
    top_risk_metrics,
)
from stage9_logistic_enhancement import BASE_CATEGORICAL, RULE_FEATURES, add_stage9_features


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
for directory in [METRIC_DIR, PREDICTION_DIR, PLOT_DIR, COEFFICIENT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


DATA_PATH = DATA_DIR / "학습데이터_로지스틱_D2D3.csv"
ENGINEERED_PATH = FEATURE_DIR / "stage7_engineered_features.csv"
RECOMMENDED_PATH = FEATURE_DIR / "stage7_recommended_feature_set.json"
OUTER_PATH = SPLIT_DIR / "stage75_date_exposure_component_cv_outer_cv_manifest.csv"
STAGE9_PREDICTION_PATH = PREDICTION_DIR / "stage9_logistic_enhancement_oof_predictions.csv"


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


LIGHT_INTERACTION_FEATURES = [
    "습도24최소_x_풍속6최대",
    "습도24최소_x_무강수0p1",
    "D1_DC_x_습도24최소",
    "D1_FWI_x_국지저습",
    "도로거리_x_비산림WUI",
    "영동_x_풍속6최대",
    "영동_x_D1_ISI",
]


MODEL_SPECS = [
    {
        "feature_set": "LOGIT_L2_RULES_C1",
        "penalty": "l2",
        "C": 1.0,
        "features": "rules",
    },
    {
        "feature_set": "LOGIT_BALANCED_L2_C1",
        "penalty": "l2",
        "C": 1.0,
        "features": "rules",
        "sample_weight_strategy": "balanced",
    },
    {
        "feature_set": "LOGIT_0A_WEIGHT_X2_L2_C1",
        "penalty": "l2",
        "C": 1.0,
        "features": "rules",
        "sample_weight_strategy": "0a_x2",
    },
    {
        "feature_set": "LOGIT_0A_WEIGHT_X4_L2_C1",
        "penalty": "l2",
        "C": 1.0,
        "features": "rules",
        "sample_weight_strategy": "0a_x4",
    },
    {
        "feature_set": "LOGIT_LIGHT_INTERACTIONS_L2_C1",
        "penalty": "l2",
        "C": 1.0,
        "features": "interactions",
    },
    {
        "feature_set": "LOGIT_ELASTICNET_C01_L1R05",
        "penalty": "elasticnet",
        "C": 0.1,
        "l1_ratio": 0.5,
        "features": "rules",
    },
    {
        "feature_set": "LOGIT_ELASTICNET_C1_L1R05",
        "penalty": "elasticnet",
        "C": 1.0,
        "l1_ratio": 0.5,
        "features": "rules",
    },
]


def add_light_interactions(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["습도24최소_x_풍속6최대"] = data["직전24h_최소습도"] * data["wind_max_6h"]
    data["습도24최소_x_무강수0p1"] = data["직전24h_최소습도"] * np.log1p(data["dry_spell_h_0p1"])
    data["D1_DC_x_습도24최소"] = data["D1_DC"] * data["직전24h_최소습도"]
    data["D1_FWI_x_국지저습"] = data["D1_FWI"] * data["rh_local_q05"]
    data["도로거리_x_비산림WUI"] = data["log1p_도로거리_m"] * data["비산림_WUI_접경후보"]
    data["영동_x_풍속6최대"] = data["기후지형유형"].eq("영동 해안형").astype(int) * data["wind_max_6h"]
    data["영동_x_D1_ISI"] = data["기후지형유형"].eq("영동 해안형").astype(int) * data["D1_ISI"]
    return data


def make_pipeline(features: list[str], categorical: list[str], spec: dict) -> Pipeline:
    categorical_used = [c for c in categorical if c in features]
    numeric = [c for c in features if c not in categorical_used]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="미상")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
                    ]
                ),
                categorical_used,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    penalty = spec["penalty"]
    solver = "saga" if penalty == "elasticnet" else "lbfgs"
    model = LogisticRegression(
        penalty=penalty,
        C=float(spec["C"]),
        l1_ratio=spec.get("l1_ratio"),
        solver=solver,
        max_iter=2500,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def sample_weight_for(indexed: pd.DataFrame, ids: list[str], strategy: str | None) -> np.ndarray | None:
    if strategy is None:
        return None
    part = indexed.loc[ids]
    weight = compute_sample_weight(class_weight="balanced", y=part["Target"].astype(int)).astype(float)
    if strategy == "balanced":
        return weight
    if strategy == "0a_x2":
        return weight * np.where(part["샘플유형"].eq("Target_0A"), 2.0, 1.0)
    if strategy == "0a_x4":
        return weight * np.where(part["샘플유형"].eq("Target_0A"), 4.0, 1.0)
    raise ValueError(f"알 수 없는 sample weight 전략: {strategy}")


def fit_pipeline(pipeline: Pipeline, indexed: pd.DataFrame, ids: list[str], features: list[str], spec: dict) -> Pipeline:
    weight = sample_weight_for(indexed, ids, spec.get("sample_weight_strategy"))
    if weight is None:
        pipeline.fit(indexed.loc[ids, features], indexed.loc[ids, "Target"].astype(int))
    else:
        pipeline.fit(indexed.loc[ids, features], indexed.loc[ids, "Target"].astype(int), model__sample_weight=weight)
    return pipeline


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["preprocess"].get_feature_names_out()


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[str]]:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["기준시각"], low_memory=False)
    engineered = pd.read_csv(ENGINEERED_PATH, encoding="utf-8-sig")
    engineered_cols = [c for c in engineered.columns if c not in {"Target", "샘플유형"}]
    data = data.merge(engineered[engineered_cols], on="샘플ID", how="left", validate="one_to_one")
    data = add_stage9_features(data)
    data = add_light_interactions(data)

    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    data = data.loc[data["샘플ID"].isin(set(outer["샘플ID"]))].copy()
    if len(data) != len(outer):
        raise ValueError(f"development 행 수 불일치: data={len(data)}, outer={len(outer)}")

    with RECOMMENDED_PATH.open("r", encoding="utf-8") as file:
        recommended = json.load(file)
    rules = list(dict.fromkeys(list(recommended["features"]) + LANDCOVER_FEATURES + RULE_FEATURES))
    interactions = list(dict.fromkeys(rules + LIGHT_INTERACTION_FEATURES))
    feature_sets = {
        "rules": [c for c in rules if c in data.columns],
        "interactions": [c for c in interactions if c in data.columns],
    }
    categorical = [c for c in BASE_CATEGORICAL if c in data.columns]
    all_features = sorted(set(sum(feature_sets.values(), [])))
    for col in all_features:
        if col in categorical:
            data[col] = data[col].fillna("미상").astype(str)
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data, outer, feature_sets, categorical


def run_oof(
    data: pd.DataFrame,
    outer: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = data.set_index("샘플ID", drop=False)
    prediction_rows = []
    fold_rows = []
    coefficient_rows = []

    for spec in MODEL_SPECS:
        feature_set_name = spec["feature_set"]
        features = feature_sets[spec["features"]]
        for outer_fold in sorted(outer["outer_fold"].unique()):
            val_ids = outer.loc[outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
            train_ids = outer.loc[~outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
            pipeline = make_pipeline(features, categorical, spec)
            fit_pipeline(pipeline, indexed, train_ids, features, spec)
            train_probability = pipeline.predict_proba(indexed.loc[train_ids, features])[:, 1]
            val_probability = pipeline.predict_proba(indexed.loc[val_ids, features])[:, 1]

            for sample_id, probability in zip(val_ids, val_probability):
                row = indexed.loc[sample_id]
                prediction_rows.append(
                    {
                        "샘플ID": sample_id,
                        "feature_set": feature_set_name,
                        "outer_fold": int(outer_fold),
                        "Target": int(row["Target"]),
                        "샘플유형": row["샘플유형"],
                        "기후지형유형": row["기후지형유형"],
                        "probability": float(probability),
                    }
                )
            for dataset_name, ids, probability in [
                ("train", train_ids, train_probability),
                ("validation", val_ids, val_probability),
            ]:
                fold_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "outer_fold": int(outer_fold),
                        "dataset": dataset_name,
                        **probability_metrics(indexed.loc[ids, "Target"], probability),
                    }
                )
            names = get_feature_names(pipeline)
            coefs = pipeline.named_steps["model"].coef_[0]
            for feature, coef in zip(names, coefs):
                coefficient_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "outer_fold": int(outer_fold),
                        "feature": str(feature),
                        "coefficient": float(coef),
                    }
                )
            fold_ap = probability_metrics(indexed.loc[val_ids, "Target"], val_probability)["auprc"]
            print(f"{feature_set_name} outer={outer_fold}: AUPRC={fold_ap:.4f}")
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def load_baselines() -> pd.DataFrame:
    baseline = pd.read_csv(STAGE9_PREDICTION_PATH, encoding="utf-8-sig")
    keep = ["PLUS_LANDCOVER", "PLUS_LANDCOVER_RULES_ANOVA"]
    return baseline.loc[baseline["feature_set"].isin(keep)].copy()


def summarize(predictions: pd.DataFrame, fold: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    baseline = load_baselines()
    all_predictions = pd.concat(
        [
            baseline[["샘플ID", "feature_set", "outer_fold", "Target", "샘플유형", "기후지형유형", "probability"]],
            predictions,
        ],
        ignore_index=True,
    )
    all_predictions.to_csv(
        PREDICTION_DIR / "stage10_logistic_stat_extensions_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_rows = []
    for feature_set, part in all_predictions.groupby("feature_set", observed=True):
        overall_rows.append({"feature_set": feature_set, **probability_metrics(part["Target"], part["probability"])})
    overall = pd.DataFrame(overall_rows).sort_values("auprc", ascending=False)
    stage9_best = overall.set_index("feature_set").loc["PLUS_LANDCOVER_RULES_ANOVA"]
    overall["delta_auprc_vs_stage9_best"] = overall["auprc"] - stage9_best["auprc"]
    overall["delta_brier_vs_stage9_best"] = overall["brier"] - stage9_best["brier"]
    overall.to_csv(METRIC_DIR / "stage10_logistic_stat_extensions_overall_metrics.csv", index=False, encoding="utf-8-sig")

    thresholds = make_threshold_table(all_predictions)
    thresholds.to_csv(METRIC_DIR / "stage10_logistic_stat_extensions_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    negative = negative_type_metrics(all_predictions)
    negative.to_csv(METRIC_DIR / "stage10_logistic_stat_extensions_sample_type_metrics.csv", index=False, encoding="utf-8-sig")
    climate = subgroup_metrics(all_predictions, "기후지형유형")
    climate.to_csv(METRIC_DIR / "stage10_logistic_stat_extensions_climate_metrics.csv", index=False, encoding="utf-8-sig")
    top_risk = pd.concat([top_risk_metrics(all_predictions, fs) for fs in overall["feature_set"]], ignore_index=True)
    top_risk.to_csv(METRIC_DIR / "stage10_logistic_stat_extensions_top_risk_metrics.csv", index=False, encoding="utf-8-sig")
    fold.to_csv(METRIC_DIR / "stage10_logistic_stat_extensions_fold_metrics.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(
        COEFFICIENT_DIR / "stage10_logistic_stat_extensions_fold_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold_best = thresholds.loc[thresholds["threshold_type"].eq("best_f1_oof")].copy()
    neg_0a = negative.loc[negative["negative_type"].eq("Target_0A")].copy()

    plt.figure(figsize=(11, 5))
    sns.barplot(data=overall, x="feature_set", y="auprc", color="#4C78A8")
    plt.axhline(float(stage9_best["auprc"]), color="black", linestyle="--", linewidth=1)
    plt.title("Stage10 경량 로지스틱 후보 AUPRC")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stage10_logistic_01_auprc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.barplot(data=threshold_best.sort_values("f1", ascending=False), x="feature_set", y="f1", color="#F58518")
    plt.title("Stage10 경량 로지스틱 후보 best-F1")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stage10_logistic_02_best_f1.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.barplot(data=neg_0a.sort_values("auprc", ascending=False), x="feature_set", y="auprc", color="#54A24B")
    plt.axhline(
        float(neg_0a.loc[neg_0a["feature_set"].eq("PLUS_LANDCOVER_RULES_ANOVA"), "auprc"].iloc[0]),
        color="black",
        linestyle="--",
        linewidth=1,
    )
    plt.title("Stage10 경량 로지스틱 후보 0-A AUPRC")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stage10_logistic_03_0a_auprc.png", dpi=180)
    plt.close()

    best = overall.iloc[0]
    best_f1 = threshold_best.sort_values("f1", ascending=False).iloc[0]
    best_0a = neg_0a.sort_values("auprc", ascending=False).iloc[0]
    best_brier = overall.sort_values("brier").iloc[0]

    summary = [
        "# Stage 10 경량 로지스틱 계열 통계모델 확장",
        "",
        "## 1. 목적",
        "",
        "- strict `date_exposure_component_cv`를 그대로 사용했다.",
        "- lockbox는 열지 않았다.",
        "- 비통계적 이진분류 모델은 제외했다.",
        "- 오래 걸리는 spline/GAM은 제외하고, 고정 후보의 빠른 outer OOF 스크리닝만 수행했다.",
        "",
        "## 2. 전체 성능",
        "",
        overall[
            ["feature_set", "auprc", "auroc", "brier", "log_loss", "delta_auprc_vs_stage9_best", "delta_brier_vs_stage9_best"]
        ].round(5).to_markdown(index=False),
        "",
        "## 3. F1 운영점",
        "",
        threshold_best[
            ["feature_set", "threshold", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc"]
        ].sort_values("f1", ascending=False).round(5).to_markdown(index=False),
        "",
        "## 4. 0-A 성능",
        "",
        neg_0a[["feature_set", "auprc", "auroc", "brier", "log_loss"]]
        .sort_values("auprc", ascending=False)
        .round(5)
        .to_markdown(index=False),
        "",
        "## 5. 후보별 역할",
        "",
        f"- 전체 AUPRC 최고 후보: `{best['feature_set']}` ({best.auprc:.4f})",
        f"- F1 최고 후보: `{best_f1['feature_set']}` ({best_f1.f1:.4f})",
        f"- 0-A AUPRC 최고 후보: `{best_0a['feature_set']}` ({best_0a.auprc:.4f})",
        f"- Brier 최고 후보: `{best_brier['feature_set']}` ({best_brier.brier:.5f})",
        "",
        "## 6. 해석",
        "",
        "- class-weight 또는 0-A 가중 모델은 운영 threshold 후보로만 보고, 오즈비 해석은 비가중 모델을 우선한다.",
        "- Elastic Net이 성능을 유지하면서 계수 수를 줄이면 해석 모델 후보로 볼 수 있다.",
        "- 이 스크리닝에서 개선폭이 작으면 로지스틱 성능 개선은 사실상 한계에 가깝다고 판단한다.",
    ]
    (OUTPUT_DIR / "stage10_logistic_stat_extensions_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("Stage 10 경량 로지스틱 계열 확장 시작")
    data, outer, feature_sets, categorical = prepare_data()
    predictions, fold, coefficients = run_oof(data, outer, feature_sets, categorical)
    summarize(predictions, fold, coefficients)
    print("Stage 10 경량 로지스틱 계열 확장 완료")


if __name__ == "__main__":
    main()
