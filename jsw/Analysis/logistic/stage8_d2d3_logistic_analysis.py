from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

from stage7_feature_extension import nested_oof_for_feature_set, probability_metrics


warnings.filterwarnings("ignore")


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
LOCKBOX_PATH = SPLIT_DIR / "lockbox_manifest.csv"
OUTER_PATH = SPLIT_DIR / "outer_cv_manifest.csv"
INNER_PATH = SPLIT_DIR / "inner_cv_manifest.csv"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


LANDCOVER_FEATURES = [
    "토지피복_L1_NAME",
    "토지피복_L2_NAME",
    "토지피복_매칭방식",
    "토지피복_산림유형",
    "토지피복_산림지역",
    "토지피복_시가화건조지역",
    "토지피복_농업지역",
    "토지피복_초지",
    "토지피복_나지",
    "토지피복_도로",
    "토지피복_활엽수림",
    "토지피복_침엽수림",
    "토지피복_혼효림",
    "비산림_WUI_접경후보",
]

FOREST_FEATURES = [
    "임상도_출처",
    "임상구분",
    "수종",
    "경급",
    "영급",
    "소밀도",
    "임상_수종_대분류",
    "임상도_매칭여부",
    "임상_영급_숫자",
    "임상_경급_숫자",
    "임상_소밀도_순서",
    "임상_산림여부",
    "임상_침엽수림",
    "임상_활엽수림",
    "임상_혼효림",
    "임상_소나무류",
    "임상_침엽수_수종",
]

CATEGORICAL_EXTRA = [
    "토지피복_L1_NAME",
    "토지피복_L2_NAME",
    "토지피복_매칭방식",
    "토지피복_산림유형",
    "임상도_출처",
    "임상구분",
    "수종",
    "경급",
    "영급",
    "소밀도",
    "임상_수종_대분류",
]


def classification_metrics_at_threshold(y_true, probability, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true, probability) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def recall_threshold(y_true, probability, target_recall: float = 0.9) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return float(thresholds[np.argmax(recall[:-1])])
    # 조건을 만족하는 threshold 중 가장 높은 값을 선택해 오경보를 줄인다.
    return float(thresholds[valid[-1]])


def make_threshold_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set, part in predictions.groupby("feature_set", observed=True):
        y = part["Target"].to_numpy()
        p = part["probability"].to_numpy()
        thresholds = {
            "fixed_0.5": 0.5,
            "best_f1_oof": best_f1_threshold(y, p),
            "recall90_oof": recall_threshold(y, p, 0.9),
        }
        for name, threshold in thresholds.items():
            rows.append(
                {
                    "feature_set": feature_set,
                    "threshold_type": name,
                    **classification_metrics_at_threshold(y, p, threshold),
                }
            )
    return pd.DataFrame(rows)


def subgroup_metrics(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for keys, part in predictions.groupby(["feature_set", group_col], observed=True):
        feature_set, group_value = keys
        if part["Target"].nunique() < 2:
            continue
        rows.append(
            {
                "feature_set": feature_set,
                group_col: group_value,
                **probability_metrics(part["Target"], part["probability"]),
            }
        )
    return pd.DataFrame(rows)


def negative_type_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    negative_types = [
        value
        for value in sorted(predictions["샘플유형"].dropna().unique())
        if value != "Target_1"
    ]
    for feature_set, feature_part in predictions.groupby("feature_set", observed=True):
        positives = feature_part.loc[feature_part["샘플유형"].eq("Target_1")]
        for negative_type in negative_types:
            negatives = feature_part.loc[feature_part["샘플유형"].eq(negative_type)]
            part = pd.concat([positives, negatives], ignore_index=True)
            if part["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "negative_type": negative_type,
                    **probability_metrics(part["Target"], part["probability"]),
                }
            )
    return pd.DataFrame(rows)


def top_risk_metrics(predictions: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    part = predictions.loc[predictions["feature_set"].eq(feature_set)].copy()
    part = part.sort_values("probability", ascending=False).reset_index(drop=True)
    total_pos = part["Target"].sum()
    base_rate = part["Target"].mean()
    rows = []
    for frac in [0.05, 0.10, 0.20]:
        n = max(1, int(np.ceil(len(part) * frac)))
        top = part.iloc[:n]
        pos_rate = top["Target"].mean()
        rows.append(
            {
                "feature_set": feature_set,
                "top_fraction": frac,
                "n": int(n),
                "positive_rate": float(pos_rate),
                "lift": float(pos_rate / base_rate) if base_rate else np.nan,
                "positive_capture_rate": float(top["Target"].sum() / total_pos) if total_pos else np.nan,
            }
        )
    return pd.DataFrame(rows)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    print("8단계: D2D3 데이터와 Stage7 피처 로드")
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["기준시각"], low_memory=False)
    engineered = pd.read_csv(ENGINEERED_PATH, encoding="utf-8-sig")
    engineered_cols = [c for c in engineered.columns if c not in {"Target", "샘플유형"}]
    data = data.merge(
        engineered[engineered_cols],
        on="샘플ID",
        how="left",
        validate="one_to_one",
    )
    lockbox = pd.read_csv(LOCKBOX_PATH, encoding="utf-8-sig")
    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    inner = pd.read_csv(INNER_PATH, encoding="utf-8-sig")
    development_ids = set(lockbox.loc[lockbox["split"].eq("development"), "샘플ID"])
    data = data.loc[data["샘플ID"].isin(development_ids)].copy()
    if len(data) != len(outer):
        raise ValueError(f"development 행 수 불일치: data={len(data)}, outer={len(outer)}")

    with RECOMMENDED_PATH.open("r", encoding="utf-8") as file:
        recommended = json.load(file)
    base_features = list(recommended["features"])

    categorical = ["기후지형유형"] + CATEGORICAL_EXTRA
    categorical = [c for c in categorical if c in data.columns]
    all_features = sorted(set(base_features + LANDCOVER_FEATURES + FOREST_FEATURES))
    for col in all_features:
        if col in categorical:
            data[col] = data[col].fillna("미상").astype(str)
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    missing = data[all_features].isna().sum()
    if int(missing.sum()) != 0:
        raise ValueError("피처 결측이 남아 있습니다:\n" + missing[missing > 0].to_string())
    return data, outer, inner, base_features, categorical


def main() -> None:
    data, outer, inner, base_features, categorical = prepare_data()

    feature_sets = {
        "STAGE7_RECOMMENDED": base_features,
        "PLUS_LANDCOVER": base_features + LANDCOVER_FEATURES,
        "PLUS_FOREST_STAND": base_features + FOREST_FEATURES,
        "PLUS_LANDCOVER_FOREST": base_features + LANDCOVER_FEATURES + FOREST_FEATURES,
    }
    (FEATURE_DIR / "stage8_d2d3_feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    oof_parts = []
    fold_parts = []
    tuning_parts = []
    coefficient_parts = []
    for name, features in feature_sets.items():
        print(f"8단계 모델 학습: {name} ({len(features)} features)")
        oof, folds, tuning, coefficients = nested_oof_for_feature_set(
            data=data,
            feature_set_name=name,
            features=features,
            categorical=categorical,
            outer_manifest=outer,
            inner_manifest=inner,
        )
        oof_parts.append(oof)
        fold_parts.append(folds)
        tuning_parts.append(tuning)
        coefficient_parts.append(coefficients)

    predictions = pd.concat(oof_parts, ignore_index=True)
    fold_metrics = pd.concat(fold_parts, ignore_index=True)
    tuning = pd.concat(tuning_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)

    predictions.to_csv(PREDICTION_DIR / "stage8_d2d3_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(METRIC_DIR / "stage8_d2d3_fold_metrics.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(METRIC_DIR / "stage8_d2d3_inner_tuning.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(COEFFICIENT_DIR / "stage8_d2d3_fold_coefficients.csv", index=False, encoding="utf-8-sig")

    overall_rows = []
    for feature_set, part in predictions.groupby("feature_set", observed=True):
        overall_rows.append(
            {
                "feature_set": feature_set,
                **probability_metrics(part["Target"], part["probability"]),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values("auprc", ascending=False)
    base = overall.set_index("feature_set").loc["STAGE7_RECOMMENDED"]
    overall["delta_auprc_vs_stage7"] = overall["auprc"] - base["auprc"]
    overall["delta_auroc_vs_stage7"] = overall["auroc"] - base["auroc"]
    overall["delta_brier_vs_stage7"] = overall["brier"] - base["brier"]
    overall.to_csv(METRIC_DIR / "stage8_d2d3_overall_metrics.csv", index=False, encoding="utf-8-sig")

    threshold = make_threshold_table(predictions)
    threshold.to_csv(METRIC_DIR / "stage8_d2d3_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    negative = negative_type_metrics(predictions)
    negative.to_csv(METRIC_DIR / "stage8_d2d3_sample_type_metrics.csv", index=False, encoding="utf-8-sig")
    climate = subgroup_metrics(predictions, "기후지형유형")
    climate.to_csv(METRIC_DIR / "stage8_d2d3_climate_metrics.csv", index=False, encoding="utf-8-sig")
    top_risk = pd.concat(
        [top_risk_metrics(predictions, fs) for fs in overall["feature_set"]],
        ignore_index=True,
    )
    top_risk.to_csv(METRIC_DIR / "stage8_d2d3_top_risk_metrics.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=overall, x="feature_set", y="auprc", color="#4C78A8")
    plt.xticks(rotation=20, ha="right")
    plt.title("Stage8 D2D3 피처 세트별 AUPRC")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stage8_d2d3_01_auprc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    delta_plot = overall.sort_values("delta_auprc_vs_stage7", ascending=False)
    sns.barplot(data=delta_plot, x="feature_set", y="delta_auprc_vs_stage7", color="#F58518")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=20, ha="right")
    plt.title("Stage7 추천모델 대비 AUPRC 변화")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stage8_d2d3_02_delta_auprc.png", dpi=180)
    plt.close()

    best_name = str(overall.iloc[0]["feature_set"])
    best_threshold = threshold.loc[
        threshold["feature_set"].eq(best_name) & threshold["threshold_type"].eq("best_f1_oof")
    ].iloc[0]
    forest_summary = json.loads((FEATURE_DIR / "d2d3_dataset_summary.json").read_text(encoding="utf-8"))
    landcover_match = forest_summary["landcover_match"]
    forest_match = forest_summary["forest_match"]

    summary_lines = [
        "# 8단계 D2D3 토지피복·임상도 보강 로지스틱 결과",
        "",
        "## 1. 데이터 생성 결과",
        "",
        f"- 입력 데이터: `학습데이터_로지스틱_D1.csv`",
        f"- 출력 데이터: `학습데이터_로지스틱_D2D3.csv`",
        f"- 행/열: {forest_summary['rows']:,}행 × {forest_summary['columns']:,}열",
        f"- Target 분포: {forest_summary['target_counts']}",
        f"- 토지피복 매칭: {landcover_match}",
        f"- 임상도 매칭: {forest_match}",
        "",
        "해석:",
        "",
        "- 토지피복 D2는 대부분의 행에 정상 부여됐다.",
        "- 2020 수종별 임상도 zip 기반 D3는 매칭이 28건뿐이어서 예측 변수로서 정보량이 매우 작다.",
        "- 따라서 이번 성능 변화는 사실상 토지피복 보강 효과로 해석해야 한다.",
        "",
        "## 2. 피처 세트별 성능",
        "",
        overall[[
            "feature_set",
            "auprc",
            "auroc",
            "brier",
            "log_loss",
            "delta_auprc_vs_stage7",
            "delta_auroc_vs_stage7",
            "delta_brier_vs_stage7",
        ]].round(5).to_markdown(index=False),
        "",
        f"최고 AUPRC 모델은 `{best_name}`이다.",
        "",
        "## 3. 최고 모델의 F1 운영점",
        "",
        f"- threshold: {best_threshold.threshold:.4f}",
        f"- Accuracy: {best_threshold.accuracy:.4f}",
        f"- Precision: {best_threshold.precision:.4f}",
        f"- Recall: {best_threshold.recall:.4f}",
        f"- F1: {best_threshold.f1:.4f}",
        "",
        "## 4. 대조군 유형별 성능",
        "",
        negative[[
            "feature_set",
            "negative_type",
            "auprc",
            "auroc",
            "brier",
        ]].round(5).to_markdown(index=False),
        "",
        "## 5. 기후지형유형별 성능",
        "",
        climate[[
            "feature_set",
            "기후지형유형",
            "auprc",
            "auroc",
            "brier",
        ]].round(5).to_markdown(index=False),
        "",
        "## 6. 주의할 해석",
        "",
        "- 토지피복·임상도는 시간에 따라 변하지 않는 정적 공간 변수다.",
        "- 따라서 같은 발생 위치의 다른 시간대를 비교하는 0-A 구분에는 제한적일 수 있다.",
        "- D3 임상도는 현재 원천 zip 매칭률이 낮아 최종 변수로 강하게 주장하기 어렵다.",
        "- 최종 결론 전에는 Stage 7.5의 날짜·기상노출 엄격 검증이 여전히 필요하다.",
    ]
    (OUTPUT_DIR / "stage8_d2d3_result_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print(f"8단계 완료: 최고 모델 {best_name}, AUPRC={overall.iloc[0]['auprc']:.4f}")


if __name__ == "__main__":
    main()
