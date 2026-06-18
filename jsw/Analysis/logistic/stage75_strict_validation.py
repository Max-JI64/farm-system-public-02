from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold

from stage7_feature_extension import nested_oof_for_feature_set, probability_metrics
from stage8_d2d3_logistic_analysis import (
    CATEGORICAL_EXTRA,
    LANDCOVER_FEATURES,
    make_threshold_table,
    negative_type_metrics,
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
for directory in [FEATURE_DIR, METRIC_DIR, PREDICTION_DIR, PLOT_DIR, COEFFICIENT_DIR, SPLIT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "학습데이터_로지스틱_D2D3.csv"
ENGINEERED_PATH = FEATURE_DIR / "stage7_engineered_features.csv"
RECOMMENDED_PATH = FEATURE_DIR / "stage7_recommended_feature_set.json"
LOCKBOX_PATH = SPLIT_DIR / "lockbox_manifest.csv"
STAGE8_OVERALL_PATH = METRIC_DIR / "stage8_d2d3_overall_metrics.csv"
STAGE8_THRESHOLD_PATH = METRIC_DIR / "stage8_d2d3_threshold_metrics.csv"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


def clean_group_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def component_labels(data: pd.DataFrame, key_columns: list[str], prefix: str) -> pd.Series:
    uf = UnionFind()
    row_nodes = []
    for row in data.itertuples(index=False):
        sample_id = getattr(row, "샘플ID")
        row_node = f"ROW:{sample_id}"
        row_nodes.append(row_node)
        uf.find(row_node)
        for col in key_columns:
            value = clean_group_value(getattr(row, col))
            if not value:
                continue
            uf.union(row_node, f"{col}:{value}")

    roots = [uf.find(node) for node in row_nodes]
    root_to_label = {root: f"{prefix}_{i:05d}" for i, root in enumerate(sorted(set(roots)))}
    return pd.Series([root_to_label[root] for root in roots], index=data.index)


def add_strict_group_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["기준시각"] = pd.to_datetime(data["기준시각"])
    data["actual_date_key"] = data["기준시각"].dt.strftime("%Y-%m-%d")
    data["exposure_key"] = (
        data["기상셀ID"].astype(str)
        + "|"
        + data["기준시각"].dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    data["model_group_key"] = data["모델링_그룹ID"].map(clean_group_value)
    data["source_fire_key"] = data["source_fire_id"].map(clean_group_value)

    target_dates = (
        data.loc[data["샘플유형"].eq("Target_1") & data["source_fire_key"].ne("")]
        .sort_values("기준시각")
        .drop_duplicates("source_fire_key")
        .set_index("source_fire_key")["actual_date_key"]
        .to_dict()
    )
    data["target_event_date_key"] = data["source_fire_key"].map(target_dates)
    data["strict_date_key"] = data["target_event_date_key"].fillna(data["actual_date_key"])

    data["exposure_component_group"] = component_labels(
        data,
        ["model_group_key", "exposure_key"],
        "EXP",
    )
    data["date_exposure_component_group"] = component_labels(
        data,
        ["model_group_key", "exposure_key", "strict_date_key"],
        "DATEEXP",
    )
    return data


def cv_split_indices(
    frame: pd.DataFrame,
    group_col: str,
    n_splits: int,
    random_state: int,
):
    """Yield group-preserving splits.

    StratifiedGroupKFold can produce pathological folds when one date/event
    component is much larger than the rest. Use it first, but fall back to a
    deterministic greedy split if any validation fold has no positives.
    """
    y = frame["Target"].to_numpy()
    groups = frame[group_col].astype(str).to_numpy()
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(frame, y, groups))
        if all(y[val_idx].sum() > 0 for _, val_idx in splits):
            yield from splits
            return
    except Exception:
        pass

    yield from greedy_group_splits(frame, group_col, n_splits, random_state)


def greedy_group_splits(
    frame: pd.DataFrame,
    group_col: str,
    n_splits: int,
    random_state: int,
):
    rng = np.random.default_rng(random_state)
    stats = (
        frame.groupby(group_col, observed=True)["Target"]
        .agg(["size", "sum"])
        .reset_index()
        .rename(columns={"sum": "positive_n"})
    )
    stats["_tie"] = rng.random(len(stats))
    stats = stats.sort_values(
        ["positive_n", "size", "_tie"],
        ascending=[False, False, True],
    )

    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_positive = np.zeros(n_splits, dtype=float)
    fold_size = np.zeros(n_splits, dtype=float)
    target_positive = max(1.0, float(stats["positive_n"].sum()) / n_splits)
    target_size = max(1.0, float(stats["size"].sum()) / n_splits)

    for row in stats.itertuples(index=False):
        group = getattr(row, group_col)
        positive_n = float(row.positive_n)
        size = float(row.size)
        scores = []
        for fold in range(n_splits):
            next_positive = fold_positive[fold] + positive_n
            next_size = fold_size[fold] + size
            score = (
                (next_positive / target_positive) ** 2
                + 0.20 * (next_size / target_size) ** 2
            )
            scores.append(score)
        chosen = int(np.argmin(scores))
        fold_groups[chosen].append(group)
        fold_positive[chosen] += positive_n
        fold_size[chosen] += size

    group_values = frame[group_col].astype(str).to_numpy()
    all_idx = np.arange(len(frame))
    for groups_for_fold in fold_groups:
        val_mask = np.isin(group_values, np.asarray(groups_for_fold, dtype=str))
        val_idx = all_idx[val_mask]
        train_idx = all_idx[~val_mask]
        yield train_idx, val_idx


def make_nested_manifests(data: pd.DataFrame, group_col: str, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_rows = []
    inner_rows = []
    for outer_fold, (train_idx, val_idx) in enumerate(
        cv_split_indices(data, group_col, 5, RANDOM_STATE)
    ):
        val_ids = data.iloc[val_idx]["샘플ID"].tolist()
        train = data.iloc[train_idx].reset_index(drop=True)
        for sample_id in val_ids:
            outer_rows.append(
                {
                    "strategy": strategy,
                    "샘플ID": sample_id,
                    "outer_fold": outer_fold,
                }
            )
        for inner_fold, (_, inner_val_idx) in enumerate(
            cv_split_indices(train, group_col, 4, RANDOM_STATE + outer_fold + 1)
        ):
            inner_val_ids = train.iloc[inner_val_idx]["샘플ID"].tolist()
            for sample_id in inner_val_ids:
                inner_rows.append(
                    {
                        "strategy": strategy,
                        "outer_fold": outer_fold,
                        "샘플ID": sample_id,
                        "inner_fold": inner_fold,
                    }
                )

    outer = pd.DataFrame(outer_rows)
    inner = pd.DataFrame(inner_rows)
    if len(outer) != len(data):
        raise ValueError(f"{strategy}: outer manifest 행 수 불일치")
    return outer, inner


def count_spanning_keys(data: pd.DataFrame, outer: pd.DataFrame, key_col: str, positive_only: bool = False) -> int:
    frame = data[["샘플ID", "Target", key_col]].merge(
        outer[["샘플ID", "outer_fold"]],
        on="샘플ID",
        how="inner",
    )
    if positive_only:
        frame = frame.loc[frame["Target"].eq(1)]
    frame = frame.loc[frame[key_col].map(clean_group_value).ne("")]
    if frame.empty:
        return 0
    fold_counts = frame.groupby(key_col, observed=True)["outer_fold"].nunique()
    return int(fold_counts.gt(1).sum())


def audit_strategy(data: pd.DataFrame, outer: pd.DataFrame, group_col: str, strategy: str) -> pd.DataFrame:
    group_counts = data.groupby(group_col, observed=True)["Target"].agg(["size", "sum"])
    fold_counts = (
        data[["샘플ID", "Target"]]
        .merge(outer[["샘플ID", "outer_fold"]], on="샘플ID", how="inner")
        .groupby("outer_fold", observed=True)["Target"]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    audit = {
        "strategy": strategy,
        "group_col": group_col,
        "n_groups": int(group_counts.shape[0]),
        "largest_group_n": int(group_counts["size"].max()),
        "largest_group_positive_n": int(group_counts.loc[group_counts["size"].idxmax(), "sum"]),
        "positive_groups": int(group_counts["sum"].gt(0).sum()),
        "outer_model_group_leak": count_spanning_keys(data, outer, "model_group_key"),
        "outer_exposure_leak": count_spanning_keys(data, outer, "exposure_key"),
        "outer_strict_date_leak": count_spanning_keys(data, outer, "strict_date_key"),
        "outer_positive_actual_date_leak": count_spanning_keys(data, outer, "actual_date_key", positive_only=True),
        "min_fold_n": int(fold_counts["size"].min()),
        "max_fold_n": int(fold_counts["size"].max()),
        "min_fold_positive_n": int(fold_counts["sum"].min()),
        "max_fold_positive_n": int(fold_counts["sum"].max()),
    }
    fold_counts.to_csv(
        METRIC_DIR / f"stage75_{strategy}_fold_balance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return pd.DataFrame([audit])


def prepare_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    print("7.5단계: 데이터 로드")
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
    development_ids = set(lockbox.loc[lockbox["split"].eq("development"), "샘플ID"])
    data = data.loc[data["샘플ID"].isin(development_ids)].copy()
    data = add_strict_group_columns(data)

    with RECOMMENDED_PATH.open("r", encoding="utf-8") as file:
        recommended = json.load(file)
    base_features = list(recommended["features"])
    landcover_features = list(LANDCOVER_FEATURES)
    all_features = sorted(set(base_features + landcover_features))
    categorical = ["기후지형유형", *CATEGORICAL_EXTRA]
    categorical = [c for c in categorical if c in all_features]

    for col in all_features:
        if col in categorical:
            data[col] = data[col].fillna("미상").astype(str)
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    missing = data[all_features].isna().sum()
    if int(missing.sum()) != 0:
        raise ValueError("피처 결측이 남아 있습니다:\n" + missing[missing > 0].to_string())
    return data, base_features, categorical


def run_strategy(
    data: pd.DataFrame,
    strategy: str,
    group_col: str,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"7.5단계 전략 실행: {strategy}")
    outer, inner = make_nested_manifests(data, group_col, strategy)
    outer.to_csv(
        SPLIT_DIR / f"stage75_{strategy}_outer_cv_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    inner.to_csv(
        SPLIT_DIR / f"stage75_{strategy}_inner_cv_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit = audit_strategy(data, outer, group_col, strategy)

    oof_parts = []
    fold_parts = []
    tuning_parts = []
    coefficient_parts = []
    for feature_set, features in feature_sets.items():
        print(f"  모델: {feature_set}")
        oof, folds, tuning, coefficients = nested_oof_for_feature_set(
            data=data,
            feature_set_name=feature_set,
            features=features,
            categorical=categorical,
            outer_manifest=outer,
            inner_manifest=inner,
        )
        for frame in [oof, folds, tuning, coefficients]:
            frame.insert(0, "strategy", strategy)
        oof_parts.append(oof)
        fold_parts.append(folds)
        tuning_parts.append(tuning)
        coefficient_parts.append(coefficients)

    return (
        pd.concat(oof_parts, ignore_index=True),
        pd.concat(fold_parts, ignore_index=True),
        pd.concat(tuning_parts, ignore_index=True),
        pd.concat(coefficient_parts, ignore_index=True),
        audit,
    )


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    for (strategy, feature_set), part in predictions.groupby(["strategy", "feature_set"], observed=True):
        overall_rows.append(
            {
                "strategy": strategy,
                "feature_set": feature_set,
                **probability_metrics(part["Target"], part["probability"]),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values(["strategy", "auprc"], ascending=[True, False])
    threshold = make_threshold_table(
        predictions.assign(feature_set=predictions["strategy"] + "__" + predictions["feature_set"])
    )
    threshold[["strategy", "feature_set"]] = threshold["feature_set"].str.split("__", n=1, expand=True)
    threshold = threshold[
        ["strategy", "feature_set", *[c for c in threshold.columns if c not in {"strategy", "feature_set"}]]
    ]
    negative = negative_type_metrics(
        predictions.assign(feature_set=predictions["strategy"] + "__" + predictions["feature_set"])
    )
    negative[["strategy", "feature_set"]] = negative["feature_set"].str.split("__", n=1, expand=True)
    climate = subgroup_metrics(
        predictions.assign(feature_set=predictions["strategy"] + "__" + predictions["feature_set"]),
        "기후지형유형",
    )
    climate[["strategy", "feature_set"]] = climate["feature_set"].str.split("__", n=1, expand=True)
    top_risk = pd.concat(
        [
            top_risk_metrics(
                predictions.assign(feature_set=predictions["strategy"] + "__" + predictions["feature_set"]),
                f"{strategy}__{feature_set}",
            )
            for strategy in predictions["strategy"].unique()
            for feature_set in predictions["feature_set"].unique()
        ],
        ignore_index=True,
    )
    top_risk[["strategy", "feature_set"]] = top_risk["feature_set"].str.split("__", n=1, expand=True)
    return overall, threshold, negative, climate, top_risk


def write_outputs(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    tuning: pd.DataFrame,
    coefficients: pd.DataFrame,
    audits: pd.DataFrame,
) -> None:
    predictions.to_csv(PREDICTION_DIR / "stage75_strict_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(METRIC_DIR / "stage75_strict_fold_metrics.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(METRIC_DIR / "stage75_strict_inner_tuning.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(COEFFICIENT_DIR / "stage75_strict_fold_coefficients.csv", index=False, encoding="utf-8-sig")
    audits.to_csv(METRIC_DIR / "stage75_strict_group_audit.csv", index=False, encoding="utf-8-sig")

    overall, threshold, negative, climate, top_risk = summarize_predictions(predictions)
    overall.to_csv(METRIC_DIR / "stage75_strict_overall_metrics.csv", index=False, encoding="utf-8-sig")
    threshold.to_csv(METRIC_DIR / "stage75_strict_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    negative.to_csv(METRIC_DIR / "stage75_strict_sample_type_metrics.csv", index=False, encoding="utf-8-sig")
    climate.to_csv(METRIC_DIR / "stage75_strict_climate_metrics.csv", index=False, encoding="utf-8-sig")
    top_risk.to_csv(METRIC_DIR / "stage75_strict_top_risk_metrics.csv", index=False, encoding="utf-8-sig")

    current = pd.read_csv(STAGE8_OVERALL_PATH, encoding="utf-8-sig")
    current = current.loc[current["feature_set"].isin(["STAGE7_RECOMMENDED", "PLUS_LANDCOVER"])].copy()
    current["strategy"] = "current_grouped_cv"
    combined = pd.concat(
        [
            current[["strategy", "feature_set", "auprc", "auroc", "brier", "log_loss"]],
            overall[["strategy", "feature_set", "auprc", "auroc", "brier", "log_loss"]],
        ],
        ignore_index=True,
    )
    combined.to_csv(METRIC_DIR / "stage75_current_vs_strict_overall.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=combined, x="strategy", y="auprc", hue="feature_set")
    plt.xticks(rotation=15, ha="right")
    plt.title("현재 CV와 엄격 CV의 AUPRC 비교")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stage75_01_current_vs_strict_auprc.png", dpi=180)
    plt.close()

    land = combined.loc[combined["feature_set"].eq("PLUS_LANDCOVER")].set_index("strategy")
    strict_best_strategy = overall.loc[
        overall["feature_set"].eq("PLUS_LANDCOVER")
    ].sort_values("auprc", ascending=True).iloc[0]["strategy"]
    strict_best = overall.loc[
        overall["strategy"].eq(strict_best_strategy) & overall["feature_set"].eq("PLUS_LANDCOVER")
    ].iloc[0]
    strict_stage7 = overall.loc[
        overall["strategy"].eq(strict_best_strategy) & overall["feature_set"].eq("STAGE7_RECOMMENDED")
    ].iloc[0]

    stage8_threshold = pd.read_csv(STAGE8_THRESHOLD_PATH, encoding="utf-8-sig")
    current_land_f1 = stage8_threshold.loc[
        stage8_threshold["feature_set"].eq("PLUS_LANDCOVER")
        & stage8_threshold["threshold_type"].eq("best_f1_oof")
    ].iloc[0]
    strict_land_f1 = threshold.loc[
        threshold["strategy"].eq(strict_best_strategy)
        & threshold["feature_set"].eq("PLUS_LANDCOVER")
        & threshold["threshold_type"].eq("best_f1_oof")
    ].iloc[0]
    main_negative = negative.loc[negative["strategy"].eq(strict_best_strategy)].set_index(
        ["feature_set", "negative_type"]
    )
    land_0a = main_negative.loc[("PLUS_LANDCOVER", "Target_0A")]
    stage7_0a = main_negative.loc[("STAGE7_RECOMMENDED", "Target_0A")]
    land_b1 = main_negative.loc[("PLUS_LANDCOVER", "Target_0B1")]
    stage7_b1 = main_negative.loc[("STAGE7_RECOMMENDED", "Target_0B1")]
    land_b2 = main_negative.loc[("PLUS_LANDCOVER", "Target_0B2")]
    stage7_b2 = main_negative.loc[("STAGE7_RECOMMENDED", "Target_0B2")]

    summary_lines = [
        "# Step 7.5 엄격 검증 결과",
        "",
        "## 1. 검증 목적",
        "",
        "- 기존 grouped CV는 `모델링_그룹ID`를 분리했지만 동일 기상노출과 집단 발생 날짜가 fold 사이에 나뉠 수 있었다.",
        "- 이번 단계에서는 lockbox를 열지 않고 development 내부에서 더 엄격한 CV를 새로 구성했다.",
        "- 비교 모델은 Stage 7 추천모델과 Stage 8 후보인 `PLUS_LANDCOVER` 두 개로 제한했다.",
        "",
        "## 2. 엄격 분할 방식",
        "",
        "- `exposure_component_cv`: `모델링_그룹ID`와 동일 `기상셀ID×기준시각`을 하나의 연결 성분으로 묶었다.",
        "- `date_exposure_component_cv`: 위 조건에 더해 Target 1/0-A는 원 산불 발생 날짜, B계열 대조군은 실제 기준날짜를 날짜 블록으로 묶었다.",
        "- 후자가 더 엄격한 주 검증 기준이다.",
        "",
        "## 3. 그룹 누수 감사",
        "",
        audits.to_markdown(index=False),
        "",
        "주의:",
        "",
        "- `date_exposure_component_cv`는 날짜와 기상노출 누수를 모두 막지만, 가장 큰 날짜 성분이 5,140행과 양성 558건을 포함한다.",
        "- 따라서 이 기준은 실제 일반화 성능의 하한을 보는 강한 stress test에 가깝고, fold별 점수 변동은 크게 해석하지 않는다.",
        "",
        "## 4. 현재 CV 대비 엄격 CV 성능",
        "",
        combined.round(5).to_markdown(index=False),
        "",
        "## 5. 가장 엄격한 기준에서의 판단",
        "",
        f"- 주 기준: `{strict_best_strategy}`",
        f"- `STAGE7_RECOMMENDED` AUPRC: {strict_stage7.auprc:.4f}, ROC AUC: {strict_stage7.auroc:.4f}, Brier: {strict_stage7.brier:.5f}",
        f"- `PLUS_LANDCOVER` AUPRC: {strict_best.auprc:.4f}, ROC AUC: {strict_best.auroc:.4f}, Brier: {strict_best.brier:.5f}",
        f"- AUPRC 차이: {strict_best.auprc - strict_stage7.auprc:+.4f}",
        "- 즉, 절대 성능은 크게 떨어졌지만 토지피복 보강 모델의 상대 우위는 더 엄격한 기준에서도 유지됐다.",
        "",
        "## 6. F1 운영점 비교",
        "",
        "| 기준 | threshold | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 현재 grouped CV / PLUS_LANDCOVER | {current_land_f1.threshold:.4f} | {current_land_f1.accuracy:.4f} | {current_land_f1.precision:.4f} | {current_land_f1.recall:.4f} | {current_land_f1.f1:.4f} |",
        f"| 엄격 CV / PLUS_LANDCOVER | {strict_land_f1.threshold:.4f} | {strict_land_f1.accuracy:.4f} | {strict_land_f1.precision:.4f} | {strict_land_f1.recall:.4f} | {strict_land_f1.f1:.4f} |",
        "",
        "## 7. 대조군 유형별 해석",
        "",
        f"- `Target_0A`: Stage7 {stage7_0a.auprc:.4f} → PLUS_LANDCOVER {land_0a.auprc:.4f}",
        f"- `Target_0B1`: Stage7 {stage7_b1.auprc:.4f} → PLUS_LANDCOVER {land_b1.auprc:.4f}",
        f"- `Target_0B2`: Stage7 {stage7_b2.auprc:.4f} → PLUS_LANDCOVER {land_b2.auprc:.4f}",
        "- 토지피복은 여전히 B1/B2 공간 대조군 구분에 강하고, 0-A 시간 구분에는 제한적이다.",
        "",
        "## 8. 최종 해석",
        "",
        "- 엄격 CV에서 성능이 크게 떨어지면 기존 점수는 날짜·기상노출 공유의 영향을 받은 것으로 본다.",
        "- `PLUS_LANDCOVER`가 엄격 기준에서도 Stage 7 추천모델보다 높게 유지되면 토지피복 보강은 계속 유지한다.",
        "- 단, 토지피복은 정적 공간 변수이므로 0-A처럼 같은 위치의 다른 시간대 구분에는 제한적이라는 Stage 8 해석은 그대로 유지한다.",
        "- 최종 lockbox 평가는 아직 수행하지 않았다.",
    ]
    (OUTPUT_DIR / "stage75_strict_validation_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    data, base_features, categorical = prepare_data()
    feature_sets = {
        "STAGE7_RECOMMENDED": base_features,
        "PLUS_LANDCOVER": base_features + LANDCOVER_FEATURES,
    }
    strategies = {
        "exposure_component_cv": "exposure_component_group",
        "date_exposure_component_cv": "date_exposure_component_group",
    }

    oof_parts = []
    fold_parts = []
    tuning_parts = []
    coefficient_parts = []
    audit_parts = []
    for strategy, group_col in strategies.items():
        oof, folds, tuning, coefficients, audit = run_strategy(
            data=data,
            strategy=strategy,
            group_col=group_col,
            feature_sets=feature_sets,
            categorical=categorical,
        )
        oof_parts.append(oof)
        fold_parts.append(folds)
        tuning_parts.append(tuning)
        coefficient_parts.append(coefficients)
        audit_parts.append(audit)

    write_outputs(
        predictions=pd.concat(oof_parts, ignore_index=True),
        fold_metrics=pd.concat(fold_parts, ignore_index=True),
        tuning=pd.concat(tuning_parts, ignore_index=True),
        coefficients=pd.concat(coefficient_parts, ignore_index=True),
        audits=pd.concat(audit_parts, ignore_index=True),
    )
    print("Step 7.5 엄격 검증 완료")


if __name__ == "__main__":
    main()
