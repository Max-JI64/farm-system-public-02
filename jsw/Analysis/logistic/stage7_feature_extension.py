from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")
RANDOM_STATE = 20260618
C_GRID = [10.0, 100.0, 1000.0]


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (
            candidate
            / "data"
            / "학습데이터"
            / "학습데이터_로지스틱_D1.csv"
        ).exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "학습데이터"
LOGISTIC_DIR = ROOT / "jsw" / "Analysis" / "logistic"
OUTPUT_DIR = LOGISTIC_DIR / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
PLOT_DIR = OUTPUT_DIR / "plots"
COEFFICIENT_DIR = OUTPUT_DIR / "coefficients"
FEATURE_DIR = OUTPUT_DIR / "features"
for directory in [
    METRIC_DIR,
    PREDICTION_DIR,
    PLOT_DIR,
    COEFFICIENT_DIR,
    FEATURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

MODELING_PATH = DATA_DIR / "학습데이터_로지스틱_D1.csv"
RAW_WEATHER_PATH = (
    ROOT / "data" / "강원도_날씨데이터" / "강원도날씨_격자_시간단위.csv"
)
FEATURE_MANIFEST_PATH = OUTPUT_DIR / "manifests" / "feature_manifest.json"
LOCKBOX_PATH = OUTPUT_DIR / "splits" / "lockbox_manifest.csv"
OUTER_PATH = OUTPUT_DIR / "splits" / "outer_cv_manifest.csv"
INNER_PATH = OUTPUT_DIR / "splits" / "inner_cv_manifest.csv"
STAGE6_OOF_PATH = (
    PREDICTION_DIR / "stage6_canadian_subset_oof_predictions.csv"
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


def probability_metrics(y_true, probability) -> dict:
    y_true = np.asarray(y_true)
    probability = np.clip(np.asarray(probability), 1e-12, 1 - 1e-12)
    return {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "auprc": float(average_precision_score(y_true, probability)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def make_pipeline(features: list[str], categorical: list[str], c_value: float):
    categorical_used = [c for c in categorical if c in features]
    numeric = [c for c in features if c not in categorical_used]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
                categorical_used,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(
        penalty="l2",
        C=float(c_value),
        solver="lbfgs",
        class_weight=None,
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def extract_weather_features(
    samples: pd.DataFrame,
    raw_weather: pd.DataFrame,
) -> pd.DataFrame:
    raw_weather = raw_weather.sort_values(
        ["기상셀ID", "일시"]
    ).reset_index(drop=True)
    raw_weather["month"] = raw_weather["일시"].dt.month
    raw_weather["hour"] = raw_weather["일시"].dt.hour
    raw_weather["westerly"] = (
        raw_weather["풍향_deg"].between(202.5, 292.5, inclusive="both")
    ).astype(np.int8)

    quantiles = (
        raw_weather.groupby(["기상셀ID", "month", "hour"], observed=True)[
            "습도_pct"
        ]
        .quantile(0.05)
        .rename("rh_q05")
        .reset_index()
    )
    raw_weather = raw_weather.merge(
        quantiles,
        on=["기상셀ID", "month", "hour"],
        how="left",
        validate="many_to_one",
    )

    weather_groups = {
        cell: group.reset_index(drop=True)
        for cell, group in raw_weather.groupby("기상셀ID", sort=False)
    }
    rows = []
    for cell, sample_group in samples.groupby("기상셀ID", sort=False):
        weather = weather_groups.get(cell)
        if weather is None:
            raise KeyError(f"원시 기상자료에 없는 기상셀: {cell}")

        times = weather["일시"].to_numpy(dtype="datetime64[ns]")
        wind = weather["풍속_m_s"].to_numpy(dtype=float)
        rain = weather["강수량_mm"].to_numpy(dtype=float)
        westerly = weather["westerly"].to_numpy(dtype=np.int8)
        event_0p1 = times[np.nan_to_num(rain, nan=0.0) >= 0.1]
        event_5p0 = times[np.nan_to_num(rain, nan=0.0) >= 5.0]

        current_lookup = weather.set_index("일시")[
            ["습도_pct", "rh_q05"]
        ]
        for sample in sample_group.itertuples(index=False):
            timestamp = pd.Timestamp(sample.기준시각)
            t_np = np.datetime64(timestamp, "ns")
            end_idx = int(np.searchsorted(times, t_np, side="left"))
            start_6h = int(
                np.searchsorted(
                    times,
                    t_np - np.timedelta64(6, "h"),
                    side="left",
                )
            )
            wind_window = wind[start_6h:end_idx]
            west_window = westerly[start_6h:end_idx]
            complete_6h = len(wind_window) == 6 and np.isfinite(
                wind_window
            ).all()
            if complete_6h:
                wind_mean_6h = float(np.mean(wind_window))
                max_position = int(np.argmax(wind_window))
                wind_max_6h = float(wind_window[max_position])
                west_at_max = int(west_window[max_position])
            else:
                wind_mean_6h = np.nan
                wind_max_6h = np.nan
                west_at_max = np.nan

            def hours_since_event(event_times: np.ndarray) -> float:
                event_idx = int(
                    np.searchsorted(event_times, t_np, side="left")
                ) - 1
                if event_idx < 0:
                    return 720.0
                elapsed = (t_np - event_times[event_idx]) / np.timedelta64(
                    1, "h"
                )
                return float(min(float(elapsed), 720.0))

            if timestamp not in current_lookup.index:
                current_rh = np.nan
                rh_q05 = np.nan
            else:
                current = current_lookup.loc[timestamp]
                if isinstance(current, pd.DataFrame):
                    current = current.iloc[0]
                current_rh = float(current["습도_pct"])
                rh_q05 = float(current["rh_q05"])

            rows.append(
                {
                    "샘플ID": sample.샘플ID,
                    "rh_q05_threshold": rh_q05,
                    "rh_minus_local_q05": current_rh - rh_q05,
                    "wind_mean_6h": wind_mean_6h,
                    "wind_max_6h": wind_max_6h,
                    "westerly_at_wind_max_6h": west_at_max,
                    "dry_spell_h_0p1": hours_since_event(event_0p1),
                    "dry_spell_h_5p0": hours_since_event(event_5p0),
                }
            )
    return pd.DataFrame(rows)


def add_candidate_flags(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["rh_local_q05"] = (
        data["rh_minus_local_q05"] <= 0
    ).astype(np.int8)
    data["wind_max_6h_ge_5"] = (
        data["wind_max_6h"] >= 5.0
    ).astype(np.int8)
    data["dry_spell_0p1_gt_24h"] = (
        data["dry_spell_h_0p1"] > 24.0
    ).astype(np.int8)
    data["dry_spell_5p0_gt_240h"] = (
        data["dry_spell_h_5p0"] > 240.0
    ).astype(np.int8)
    data["rh_local_q05_AND_ffmc_ge_90"] = (
        data["rh_local_q05"].eq(1) & data["D1_FFMC"].ge(90.0)
    ).astype(np.int8)
    data["rh_local_q05_AND_isi_ge_10"] = (
        data["rh_local_q05"].eq(1) & data["D1_ISI"].ge(10.0)
    ).astype(np.int8)
    data["rh_local_q05_AND_wind_max_6h_ge_5"] = (
        data["rh_local_q05"].eq(1) & data["wind_max_6h"].ge(5.0)
    ).astype(np.int8)
    data["rh_local_q05_AND_westerly_strong_max_6h"] = (
        data["rh_local_q05"].eq(1)
        & data["wind_max_6h"].ge(5.0)
        & data["westerly_at_wind_max_6h"].eq(1)
    ).astype(np.int8)

    data["영동_여부"] = data["기후지형유형"].eq("영동 해안형").astype(
        np.int8
    )
    data["영서_여부"] = data["기후지형유형"].eq("영서 내륙형").astype(
        np.int8
    )
    data["고지산간_여부"] = data["기후지형유형"].eq(
        "고지·산간형"
    ).astype(np.int8)
    data["영동_x_wind_mean_6h"] = (
        data["영동_여부"] * data["wind_mean_6h"]
    )
    data["영동_x_wind_max_6h"] = (
        data["영동_여부"] * data["wind_max_6h"]
    )
    data["영동_x_westerly_at_max6h"] = (
        data["영동_여부"] * data["westerly_at_wind_max_6h"]
    )
    data["영동_x_rh_local_q05"] = (
        data["영동_여부"] * data["rh_local_q05"]
    )
    data["영서_x_24h건조도"] = data["영서_여부"] * (
        100.0 - data["직전24h_최소습도"]
    )
    data["고지산간_x_24h건조도"] = data["고지산간_여부"] * (
        100.0 - data["직전24h_최소습도"]
    )
    data["24h건조도_x_wind_max_6h"] = (
        100.0 - data["직전24h_최소습도"]
    ) * data["wind_max_6h"]
    return data


def nested_oof_for_feature_set(
    data: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
    categorical: list[str],
    outer_manifest: pd.DataFrame,
    inner_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof_rows = []
    fold_rows = []
    tuning_rows = []
    coefficient_rows = []

    indexed = data.set_index("샘플ID", drop=False)
    for outer_fold in range(5):
        val_ids = outer_manifest.loc[
            outer_manifest["outer_fold"].eq(outer_fold), "샘플ID"
        ].tolist()
        train_ids = outer_manifest.loc[
            ~outer_manifest["outer_fold"].eq(outer_fold), "샘플ID"
        ].tolist()
        inner = inner_manifest.loc[
            inner_manifest["outer_fold"].eq(outer_fold)
        ].set_index("샘플ID")

        config_scores = []
        for c_value in C_GRID:
            inner_scores = []
            for inner_fold in range(4):
                inner_val_ids = inner.index[
                    inner["inner_fold"].eq(inner_fold)
                ].tolist()
                inner_train_ids = inner.index[
                    ~inner["inner_fold"].eq(inner_fold)
                ].tolist()
                pipeline = make_pipeline(
                    features,
                    categorical,
                    c_value,
                )
                pipeline.fit(
                    indexed.loc[inner_train_ids, features],
                    indexed.loc[inner_train_ids, "Target"],
                )
                probability = pipeline.predict_proba(
                    indexed.loc[inner_val_ids, features]
                )[:, 1]
                score = average_precision_score(
                    indexed.loc[inner_val_ids, "Target"],
                    probability,
                )
                inner_scores.append(float(score))
                tuning_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "C": c_value,
                        "auprc": float(score),
                    }
                )
            config_scores.append(
                {
                    "C": c_value,
                    "mean_auprc": float(np.mean(inner_scores)),
                }
            )

        best = sorted(
            config_scores,
            key=lambda row: (-row["mean_auprc"], row["C"]),
        )[0]
        pipeline = make_pipeline(features, categorical, best["C"])
        pipeline.fit(
            indexed.loc[train_ids, features],
            indexed.loc[train_ids, "Target"],
        )
        train_probability = pipeline.predict_proba(
            indexed.loc[train_ids, features]
        )[:, 1]
        val_probability = pipeline.predict_proba(
            indexed.loc[val_ids, features]
        )[:, 1]

        for sample_id, probability in zip(val_ids, val_probability):
            row = indexed.loc[sample_id]
            oof_rows.append(
                {
                    "샘플ID": sample_id,
                    "feature_set": feature_set_name,
                    "outer_fold": outer_fold,
                    "Target": int(row["Target"]),
                    "샘플유형": row["샘플유형"],
                    "기후지형유형": row["기후지형유형"],
                    "probability": float(probability),
                    "selected_C": best["C"],
                }
            )

        for dataset_name, ids, probability in [
            ("train", train_ids, train_probability),
            ("validation", val_ids, val_probability),
        ]:
            fold_rows.append(
                {
                    "feature_set": feature_set_name,
                    "outer_fold": outer_fold,
                    "dataset": dataset_name,
                    "selected_C": best["C"],
                    **probability_metrics(
                        indexed.loc[ids, "Target"],
                        probability,
                    ),
                }
            )

        feature_names = pipeline.named_steps[
            "preprocess"
        ].get_feature_names_out()
        coefficients = pipeline.named_steps["model"].coef_[0]
        for feature, coefficient in zip(feature_names, coefficients):
            coefficient_rows.append(
                {
                    "feature_set": feature_set_name,
                    "outer_fold": outer_fold,
                    "feature": feature,
                    "coefficient": float(coefficient),
                }
            )
        print(
            f"{feature_set_name} outer={outer_fold}: "
            f"C={best['C']:g}, "
            f"AUPRC={average_precision_score(indexed.loc[val_ids, 'Target'], val_probability):.4f}"
        )

    return (
        pd.DataFrame(oof_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficient_rows),
    )


def main() -> None:
    print("7단계: 데이터 로드")
    data = pd.read_csv(
        MODELING_PATH,
        encoding="utf-8-sig",
        parse_dates=["기준시각"],
        low_memory=False,
    )
    with FEATURE_MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    lockbox = pd.read_csv(LOCKBOX_PATH, encoding="utf-8-sig")
    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    inner = pd.read_csv(INNER_PATH, encoding="utf-8-sig")
    development_ids = set(
        lockbox.loc[lockbox["split"].eq("development"), "샘플ID"]
    )
    data = data.loc[data["샘플ID"].isin(development_ids)].copy()
    assert len(data) == len(outer)

    print("7단계: 원시 시간기상에서 신규 피처 계산")
    raw_weather = pd.read_csv(
        RAW_WEATHER_PATH,
        encoding="utf-8-sig",
        usecols=[
            "기상셀ID",
            "일시",
            "풍속_m_s",
            "풍향_deg",
            "강수량_mm",
            "습도_pct",
        ],
        parse_dates=["일시"],
    )
    engineered = extract_weather_features(
        data[["샘플ID", "기상셀ID", "기준시각"]],
        raw_weather,
    )
    assert engineered["샘플ID"].is_unique
    data = data.merge(
        engineered,
        on="샘플ID",
        how="left",
        validate="one_to_one",
    )
    data = add_candidate_flags(data)

    engineered_columns = [
        column
        for column in data.columns
        if column
        in set(engineered.columns)
        | {
            "rh_local_q05",
            "wind_max_6h_ge_5",
            "dry_spell_0p1_gt_24h",
            "dry_spell_5p0_gt_240h",
            "rh_local_q05_AND_ffmc_ge_90",
            "rh_local_q05_AND_isi_ge_10",
            "rh_local_q05_AND_wind_max_6h_ge_5",
            "rh_local_q05_AND_westerly_strong_max_6h",
            "영동_x_wind_mean_6h",
            "영동_x_wind_max_6h",
            "영동_x_westerly_at_max6h",
            "영동_x_rh_local_q05",
            "영서_x_24h건조도",
            "고지산간_x_24h건조도",
            "24h건조도_x_wind_max_6h",
        }
    ]
    feature_audit = data[
        ["샘플ID", "Target", "샘플유형", *engineered_columns]
    ].copy()
    feature_audit.to_csv(
        FEATURE_DIR / "stage7_engineered_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing = data[engineered_columns].isna().sum()
    if int(missing.sum()) != 0:
        raise ValueError(
            "신규 피처 결측 발생:\n"
            + missing[missing > 0].to_string()
        )

    baseline_features = manifest["model_sets"]["M2"]
    categorical = manifest["categorical_features"]
    continuous_features = [
        "rh_minus_local_q05",
        "wind_mean_6h",
        "wind_max_6h",
        "dry_spell_h_0p1",
        "dry_spell_h_5p0",
    ]
    threshold_features = [
        "rh_local_q05",
        "wind_max_6h_ge_5",
        "dry_spell_0p1_gt_24h",
        "dry_spell_5p0_gt_240h",
        "rh_local_q05_AND_ffmc_ge_90",
        "rh_local_q05_AND_isi_ge_10",
        "rh_local_q05_AND_wind_max_6h_ge_5",
        "rh_local_q05_AND_westerly_strong_max_6h",
    ]
    interaction_features = [
        "영동_x_wind_mean_6h",
        "영동_x_wind_max_6h",
        "영동_x_westerly_at_max6h",
        "영동_x_rh_local_q05",
        "영서_x_24h건조도",
        "고지산간_x_24h건조도",
        "24h건조도_x_wind_max_6h",
    ]
    feature_sets = {
        "BASE_CAN_ALL": baseline_features,
        "PLUS_CONTINUOUS": baseline_features + continuous_features,
        "PLUS_FLAGS": baseline_features + threshold_features,
        "PLUS_CONTINUOUS_FLAGS": (
            baseline_features + continuous_features + threshold_features
        ),
        "PLUS_ALL_INTERACTIONS": (
            baseline_features
            + continuous_features
            + threshold_features
            + interaction_features
        ),
    }
    feature_set_manifest = {
        name: features for name, features in feature_sets.items()
    }
    (
        FEATURE_DIR / "stage7_feature_sets.json"
    ).write_text(
        json.dumps(
            feature_set_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 기존 6단계 baseline OOF를 재사용한다.
    stage6_oof = pd.read_csv(
        STAGE6_OOF_PATH,
        encoding="utf-8-sig",
    )
    baseline_oof = stage6_oof.loc[
        stage6_oof["subset"].eq("CAN_ALL")
    ].copy()
    baseline_oof["feature_set"] = "BASE_CAN_ALL"
    baseline_oof = baseline_oof[
        [
            "샘플ID",
            "feature_set",
            "outer_fold",
            "Target",
            "샘플유형",
            "기후지형유형",
            "probability",
            "selected_C",
        ]
    ]

    oof_parts = [baseline_oof]
    fold_parts = []
    tuning_parts = []
    coefficient_parts = []
    for name, features in feature_sets.items():
        if name == "BASE_CAN_ALL":
            continue
        oof, folds, tuning, coefficients = nested_oof_for_feature_set(
            data,
            name,
            features,
            categorical,
            outer,
            inner,
        )
        oof_parts.append(oof)
        fold_parts.append(folds)
        tuning_parts.append(tuning)
        coefficient_parts.append(coefficients)

    oof_all = pd.concat(oof_parts, ignore_index=True)
    fold_all = pd.concat(fold_parts, ignore_index=True)
    tuning_all = pd.concat(tuning_parts, ignore_index=True)
    coefficients_all = pd.concat(coefficient_parts, ignore_index=True)
    oof_all.to_csv(
        PREDICTION_DIR / "stage7_feature_set_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_all.to_csv(
        METRIC_DIR / "stage7_feature_set_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tuning_all.to_csv(
        METRIC_DIR / "stage7_feature_set_inner_tuning.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coefficients_all.to_csv(
        COEFFICIENT_DIR / "stage7_feature_set_fold_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_rows = []
    negative_rows = []
    climate_rows = []
    for name, part in oof_all.groupby("feature_set", observed=True):
        overall_rows.append(
            {"feature_set": name, **probability_metrics(
                part["Target"], part["probability"]
            )}
        )
        for negative_type in ["Target_0A", "Target_0B1", "Target_0B2"]:
            subset = part.loc[
                part["Target"].eq(1)
                | part["샘플유형"].eq(negative_type)
            ]
            negative_rows.append(
                {
                    "feature_set": name,
                    "negative_type": negative_type,
                    **probability_metrics(
                        subset["Target"], subset["probability"]
                    ),
                }
            )
        for climate, subset in part.groupby(
            "기후지형유형", observed=True
        ):
            climate_rows.append(
                {
                    "feature_set": name,
                    "기후지형유형": climate,
                    **probability_metrics(
                        subset["Target"], subset["probability"]
                    ),
                }
            )

    overall = pd.DataFrame(overall_rows).sort_values(
        ["auprc", "brier"],
        ascending=[False, True],
    )
    negative = pd.DataFrame(negative_rows)
    climate = pd.DataFrame(climate_rows)
    overall.to_csv(
        METRIC_DIR / "stage7_feature_set_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    negative.to_csv(
        METRIC_DIR / "stage7_feature_set_negative_type_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    climate.to_csv(
        METRIC_DIR / "stage7_feature_set_climate_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_overall = overall.set_index("feature_set").loc[
        "BASE_CAN_ALL"
    ]
    baseline_0a = negative.set_index(
        ["feature_set", "negative_type"]
    ).loc[("BASE_CAN_ALL", "Target_0A")]
    comparison = overall.copy()
    comparison["delta_auprc_vs_base"] = (
        comparison["auprc"] - baseline_overall["auprc"]
    )
    comparison["delta_brier_vs_base"] = (
        comparison["brier"] - baseline_overall["brier"]
    )
    zero_a = negative.loc[
        negative["negative_type"].eq("Target_0A"),
        ["feature_set", "auprc"],
    ].rename(columns={"auprc": "auprc_0A"})
    comparison = comparison.merge(zero_a, on="feature_set", how="left")
    comparison["delta_0A_auprc_vs_base"] = (
        comparison["auprc_0A"] - baseline_0a["auprc"]
    )
    comparison.to_csv(
        METRIC_DIR / "stage7_feature_set_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 후보 피처 발생률 감사
    flag_columns = threshold_features
    prevalence_rows = []
    for flag in flag_columns:
        for sample_type, subset in data.groupby(
            "샘플유형", observed=True
        ):
            prevalence_rows.append(
                {
                    "feature": flag,
                    "샘플유형": sample_type,
                    "n": len(subset),
                    "rate": float(subset[flag].mean()),
                    "positive_n": int(subset[flag].sum()),
                }
            )
    prevalence = pd.DataFrame(prevalence_rows)
    prevalence.to_csv(
        METRIC_DIR / "stage7_candidate_feature_prevalence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 플롯 1: 전체와 0-A
    plot_data = comparison[
        ["feature_set", "auprc", "auprc_0A"]
    ].melt(
        id_vars="feature_set",
        var_name="평가",
        value_name="AUPRC",
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=plot_data,
        x="feature_set",
        y="AUPRC",
        hue="평가",
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=25)
    ax.set(
        title="7단계 신규 피처 블록별 전체 및 0-A AUPRC",
        xlabel="",
        ylabel="AUPRC",
    )
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "stage7_01_feature_sets_auprc.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 플롯 2: baseline 대비 차이
    delta_plot = comparison.melt(
        id_vars="feature_set",
        value_vars=[
            "delta_auprc_vs_base",
            "delta_0A_auprc_vs_base",
        ],
        var_name="평가",
        value_name="AUPRC 차이",
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=delta_plot,
        x="feature_set",
        y="AUPRC 차이",
        hue="평가",
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", alpha=0.6)
    ax.tick_params(axis="x", rotation=25)
    ax.set(
        title="BASE_CAN_ALL 대비 AUPRC 변화",
        xlabel="",
        ylabel="ΔAUPRC",
    )
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "stage7_02_delta_vs_baseline.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 플롯 3: 기후지형유형
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=climate,
        x="기후지형유형",
        y="auprc",
        hue="feature_set",
        ax=ax,
    )
    ax.set(
        title="기후지형유형별 신규 피처 세트 AUPRC",
        xlabel="",
        ylabel="AUPRC",
    )
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "stage7_03_climate_auprc.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 플롯 4: 후보 플래그 발생률
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        data=prevalence,
        x="rate",
        y="feature",
        hue="샘플유형",
        ax=ax,
    )
    ax.set(
        title="EDA 후보 피처의 샘플유형별 충족률",
        xlabel="충족률",
        ylabel="피처",
    )
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "stage7_04_candidate_prevalence.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    best_set = str(overall.iloc[0]["feature_set"])
    comparison_indexed = comparison.set_index("feature_set")
    best_row = comparison_indexed.loc[best_set]
    complexity_order = [
        "BASE_CAN_ALL",
        "PLUS_CONTINUOUS",
        "PLUS_FLAGS",
        "PLUS_CONTINUOUS_FLAGS",
        "PLUS_ALL_INTERACTIONS",
    ]
    near_best = comparison.loc[
        comparison["auprc"].ge(float(best_row["auprc"]) - 0.005)
        & comparison["auprc_0A"].ge(
            float(best_row["auprc_0A"]) - 0.005
        )
        & comparison["brier"].le(float(best_row["brier"]) + 0.0005)
    ]
    recommended_set = next(
        name
        for name in complexity_order
        if name in set(near_best["feature_set"])
    )
    recommended_row = comparison_indexed.loc[recommended_set]
    recommendation = {
        "metric_best_feature_set": best_set,
        "recommended_feature_set": recommended_set,
        "selection_rule": (
            "최고 모델 대비 전체·0-A AUPRC 0.005 이내, "
            "Brier 0.0005 이내 후보 중 가장 단순한 세트"
        ),
        "features": feature_sets[recommended_set],
    }
    (
        FEATURE_DIR / "stage7_recommended_feature_set.json"
    ).write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_lines = [
        "# 7단계 EDA 피처 확장 결과 요약",
        "",
        "## 피처 계산 정책",
        "",
        "- 기준시각 이전 6시간 풍속만 사용했다.",
        "- 무강수 지속시간은 기준시각 이전 마지막 0.1mm/5.0mm 이상 시간강수로부터 계산하고 720시간에서 상한 처리했다.",
        "- 국지 하위 5% 습도 기준은 기상셀×월×시간대의 2020~2021 고정 기후분포를 사용했다.",
        "- FFMC/ISI 결합 플래그는 미래 누수를 피하기 위해 D-1 정오 지수를 사용했다.",
        "- FA 요인점수는 사용하지 않았다.",
        "",
        "## 결과",
        "",
        f"- 수치상 최상위 피처 세트: {best_set}",
        f"- 전체 AUPRC: {best_row['auprc']:.4f}",
        f"- Target 0-A AUPRC: {best_row['auprc_0A']:.4f}",
        f"- BASE 대비 전체 ΔAUPRC: {best_row['delta_auprc_vs_base']:+.4f}",
        f"- BASE 대비 0-A ΔAUPRC: {best_row['delta_0A_auprc_vs_base']:+.4f}",
        f"- BASE 대비 Brier 변화: {best_row['delta_brier_vs_base']:+.5f}",
        f"- 복잡도 반영 최종 추천 세트: {recommended_set}",
        f"- 추천 세트 전체/0-A AUPRC: {recommended_row['auprc']:.4f} / {recommended_row['auprc_0A']:.4f}",
        f"- 수치상 1위 대비 추천 세트 전체 AUPRC 차이: {recommended_row['auprc'] - best_row['auprc']:+.4f}",
        "",
        "## 판단 기준",
        "",
        "- 전체 및 0-A AUPRC가 모두 개선되고 Brier가 악화되지 않을 때 신규 피처를 유지한다.",
        "- 개선 폭이 0.005 미만이면 복잡성을 고려해 기존 BASE 모델을 우선한다.",
        "- 상호작용 전체 세트의 추가 이득이 0.005 미만이므로, 후속 모델은 연속형+EDA 플래그 세트를 우선한다.",
        "- lockbox test는 사용하지 않았다.",
    ]
    summary_path = OUTPUT_DIR / "stage7_result_summary.md"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )
    print("\n".join(summary_lines))
    print("\n7단계 비교표")
    print(comparison.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
