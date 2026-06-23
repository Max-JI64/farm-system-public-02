from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import step2_tuned_single_models as step2


RANDOM_STATE = 20260622
FINAL_THRESHOLD = float("nan")
FINAL_CANDIDATE_ID = "TUNE_LGBM_ALL_ALL_LC_NONE"
TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
ERROR_GROUPS = ("TP", "FN", "FP", "TN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="new_machine_learning Step4: tuned LightGBM interpretation at F2 threshold",
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--shap-batch-size",
        type=int,
        default=1024,
        help="TreeSHAP validation 계산 batch 크기",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="기존 Step 4 CSV를 읽어 플롯만 다시 생성",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_plot_style() -> None:
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if not font_path.exists():
        raise FileNotFoundError(f"한글 폰트 파일이 없습니다: {font_path}")
    fm.fontManager.addfont(str(font_path))
    font_name = fm.FontProperties(fname=str(font_path)).get_name()
    sns.set_theme(
        style="whitegrid",
        font=font_name,
        rc={
            "font.family": font_name,
            "font.sans-serif": [font_name],
            "axes.unicode_minus": False,
        },
    )
    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.max_columns", None)
    log(f"matplotlib font.family={plt.rcParams['font.family']}")
    log(f"matplotlib font.sans-serif={plt.rcParams['font.sans-serif'][:3]}")


def final_features() -> list[str]:
    return list(
        step2.build_feature_sets()["WS_ALL__CANADA_ALL__LC_USED"].features
    )


def load_params(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("selected_params_by_outer_fold")
    if not isinstance(params, dict) or len(params) != 5:
        raise ValueError(f"outer fold별 parameter가 완전하지 않습니다: {path}")
    return params


def build_outer_splits(
    data: pd.DataFrame,
    manifest_path: Path,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    fold_by_id = manifest.set_index(step2.SAMPLE_ID_COL)["outer_fold"]
    folds = data[step2.SAMPLE_ID_COL].map(fold_by_id)
    if folds.isna().any():
        raise RuntimeError("split manifest와 매칭되지 않는 표본이 있습니다.")
    folds = folds.astype(int).to_numpy()
    positions = np.arange(len(data))
    return [
        (fold, positions[folds != fold], positions[folds == fold])
        for fold in sorted(np.unique(folds))
    ]


def transformed_to_original(
    transformed_names: list[str],
    features: list[str],
) -> list[str]:
    categorical = [
        "기후지형유형",
        "토지피복_L1_NAME",
        "토지피복_L2_NAME",
        "토지피복_매칭방식",
        "토지피복_산림유형",
    ]
    result: list[str] = []
    for name in transformed_names:
        if name in features:
            result.append(name)
            continue
        matched = next(
            (
                feature
                for feature in categorical
                if feature in features and name.startswith(f"{feature}_")
            ),
            None,
        )
        if matched is None:
            raise KeyError(f"변환 피처를 원변수로 매핑할 수 없습니다: {name}")
        result.append(matched)
    return result


def positive_shap_values(explainer: Any, transformed: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(transformed, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
    if values.ndim != 2:
        raise ValueError(f"지원하지 않는 SHAP shape: {values.shape}")
    return values.astype(np.float64, copy=False)


def positive_expected_value(explainer: Any) -> float:
    value = np.asarray(explainer.expected_value)
    if value.ndim == 0:
        return float(value)
    return float(value.reshape(-1)[-1])


def aggregate_original_shap(
    transformed_shap: np.ndarray,
    original_names: list[str],
    features: list[str],
) -> np.ndarray:
    result = np.zeros((len(transformed_shap), len(features)), dtype=np.float32)
    feature_index = {feature: index for index, feature in enumerate(features)}
    for transformed_index, original_name in enumerate(original_names):
        result[:, feature_index[original_name]] += transformed_shap[
            :, transformed_index
        ].astype(np.float32)
    return result


def fit_oof_shap(
    data: pd.DataFrame,
    oof: pd.DataFrame,
    splits: list[tuple[int, np.ndarray, np.ndarray]],
    params_by_fold: dict[str, dict[str, Any]],
    features: list[str],
    *,
    n_jobs: int,
    batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = step2.build_candidates()[FINAL_CANDIDATE_ID]
    shap_matrix = np.full(
        (len(data), len(features)),
        np.nan,
        dtype=np.float32,
    )
    validation_rows: list[dict[str, Any]] = []
    oof_score = oof.set_index("sample_id")["y_prob"]

    for outer_fold, train_idx, valid_idx in splits:
        log(f"outer fold {outer_fold}: LightGBM fit")
        y_train = data.iloc[train_idx][step2.TARGET_COL].astype(int)
        model = step2.make_pipeline(
            candidate=candidate,
            features=features,
            y_train=y_train,
            params=params_by_fold[str(outer_fold)],
            random_state=RANDOM_STATE + outer_fold,
            n_jobs=n_jobs,
        )
        model.fit(data.iloc[train_idx][features], y_train)

        valid_frame = data.iloc[valid_idx]
        x_valid = valid_frame[features]
        probability = step2.predict_probability(model, x_valid)
        expected_probability = oof_score.loc[
            valid_frame[step2.SAMPLE_ID_COL].astype(str)
        ].to_numpy(dtype=float)
        prediction_diff = np.abs(probability - expected_probability)

        preprocessor = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        transformed = np.asarray(preprocessor.transform(x_valid), dtype=np.float64)
        transformed_names = [
            str(name) for name in preprocessor.get_feature_names_out()
        ]
        original_names = transformed_to_original(transformed_names, features)
        explainer = shap.TreeExplainer(estimator)
        expected_value = positive_expected_value(explainer)

        fold_shap_parts: list[np.ndarray] = []
        raw_margin_parts: list[np.ndarray] = []
        for start in range(0, len(transformed), batch_size):
            end = min(start + batch_size, len(transformed))
            batch = transformed[start:end]
            batch_shap = positive_shap_values(explainer, batch)
            fold_shap_parts.append(batch_shap)
            raw_margin_parts.append(
                np.asarray(estimator.predict(batch, raw_score=True), dtype=float)
            )
        transformed_shap = np.vstack(fold_shap_parts)
        raw_margin = np.concatenate(raw_margin_parts)
        reconstructed_margin = expected_value + transformed_shap.sum(axis=1)
        additivity_error = np.abs(reconstructed_margin - raw_margin)

        original_shap = aggregate_original_shap(
            transformed_shap,
            original_names,
            features,
        )
        shap_matrix[valid_idx] = original_shap
        validation_rows.append(
            {
                "outer_fold": outer_fold,
                "train_n": len(train_idx),
                "valid_n": len(valid_idx),
                "original_feature_n": len(features),
                "transformed_feature_n": transformed.shape[1],
                "prediction_max_abs_diff_vs_saved_oof": float(
                    prediction_diff.max()
                ),
                "prediction_mean_abs_diff_vs_saved_oof": float(
                    prediction_diff.mean()
                ),
                "shap_additivity_max_abs_error_raw_margin": float(
                    additivity_error.max()
                ),
                "shap_additivity_mean_abs_error_raw_margin": float(
                    additivity_error.mean()
                ),
                "shap_nan_n": int(np.isnan(original_shap).sum()),
            }
        )
        log(
            f"outer fold {outer_fold}: SHAP 완료 | "
            f"n={len(valid_idx):,} | OOF max diff={prediction_diff.max():.3e}"
        )

    if np.isnan(shap_matrix).any():
        raise RuntimeError("OOF SHAP 값에 누락이 있습니다.")
    return shap_matrix, pd.DataFrame(validation_rows)


def add_analysis_columns(data: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    metadata = oof[
        [
            "sample_id",
            "outer_fold",
            "y_true",
            "y_prob",
            "sample_type",
            "climate_type",
            "group_id",
        ]
    ].copy()
    metadata = metadata.rename(
        columns={
            "sample_id": step2.SAMPLE_ID_COL,
            "y_prob": "risk_score",
        }
    )
    frame = data.merge(
        metadata[
            [
                step2.SAMPLE_ID_COL,
                "outer_fold",
                "risk_score",
            ]
        ],
        on=step2.SAMPLE_ID_COL,
        how="left",
        validate="one_to_one",
    )
    if frame["risk_score"].isna().any():
        raise RuntimeError("OOF score가 원자료와 매칭되지 않았습니다.")
    frame["pred_f2"] = (frame["risk_score"] >= FINAL_THRESHOLD).astype(int)
    frame["error_group"] = np.select(
        [
            (frame[step2.TARGET_COL] == 1) & (frame["pred_f2"] == 1),
            (frame[step2.TARGET_COL] == 1) & (frame["pred_f2"] == 0),
            (frame[step2.TARGET_COL] == 0) & (frame["pred_f2"] == 1),
        ],
        ["TP", "FN", "FP"],
        default="TN",
    )
    frame["risk_rank"] = frame["risk_score"].rank(
        method="first",
        ascending=False,
    )
    frame["top_fraction"] = frame["risk_rank"] / len(frame)
    return frame


def importance_table(
    shap_matrix: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": np.mean(np.abs(shap_matrix), axis=0),
            "median_abs_shap": np.median(np.abs(shap_matrix), axis=0),
            "mean_shap": np.mean(shap_matrix, axis=0),
            "positive_contribution_share": np.mean(shap_matrix > 0, axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    frame["importance_share"] = (
        frame["mean_abs_shap"] / frame["mean_abs_shap"].sum()
    )
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame.reset_index(drop=True)


def subgroup_importance(
    analysis_data: pd.DataFrame,
    shap_matrix: np.ndarray,
    features: list[str],
    group_column: str,
    output_group_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, positions in analysis_data.groupby(group_column).groups.items():
        indices = np.asarray(list(positions), dtype=int)
        values = shap_matrix[indices]
        mean_abs = np.mean(np.abs(values), axis=0)
        total = float(mean_abs.sum())
        for feature_index, feature in enumerate(features):
            rows.append(
                {
                    output_group_name: group_value,
                    "n": len(indices),
                    "feature": feature,
                    "mean_abs_shap": float(mean_abs[feature_index]),
                    "importance_share": (
                        float(mean_abs[feature_index] / total) if total else 0.0
                    ),
                    "mean_shap": float(np.mean(values[:, feature_index])),
                }
            )
    frame = pd.DataFrame(rows)
    frame["rank_within_group"] = (
        frame.groupby(output_group_name)["mean_abs_shap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return frame


def direction_table(
    analysis_data: pd.DataFrame,
    shap_matrix: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        numeric = pd.to_numeric(analysis_data[feature], errors="coerce").to_numpy()
        valid = np.isfinite(numeric)
        if valid.sum() < 20:
            continue
        correlation, p_value = spearmanr(
            numeric[valid],
            shap_matrix[valid, feature_index],
        )
        q25, q75 = np.nanquantile(numeric[valid], [0.25, 0.75])
        low = valid & (numeric <= q25)
        high = valid & (numeric >= q75)
        rows.append(
            {
                "feature": feature,
                "n": int(valid.sum()),
                "spearman_feature_value_vs_shap": float(correlation),
                "spearman_p_value": float(p_value),
                "q25_value": float(q25),
                "q75_value": float(q75),
                "mean_shap_low_quartile": float(
                    np.mean(shap_matrix[low, feature_index])
                ),
                "mean_shap_high_quartile": float(
                    np.mean(shap_matrix[high, feature_index])
                ),
                "high_minus_low_mean_shap": float(
                    np.mean(shap_matrix[high, feature_index])
                    - np.mean(shap_matrix[low, feature_index])
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_subgroup_metrics(analysis_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for negative_type in ("Target_0A", "Target_0B1", "Target_0B2"):
        subset = analysis_data.loc[
            analysis_data[step2.SAMPLE_TYPE_COL].isin(["Target_1", negative_type])
        ]
        y = subset[step2.TARGET_COL].to_numpy(dtype=int)
        pred = subset["pred_f2"].to_numpy(dtype=int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        rows.append(
            {
                "subgroup_type": "hard_negative_pair",
                "subgroup": f"Target_1_vs_{negative_type}",
                "n": len(subset),
                "positive_n": int(y.sum()),
                "negative_n": int((1 - y).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": tp / (tp + fn),
                "precision": tp / (tp + fp),
                "specificity": tn / (tn + fp),
                "selected_rate": float(pred.mean()),
                "fp_rate_within_negative": fp / max(1, int((1 - y).sum())),
            }
        )

    for climate_type, subset in analysis_data.groupby(step2.CLIMATE_COL):
        y = subset[step2.TARGET_COL].to_numpy(dtype=int)
        pred = subset["pred_f2"].to_numpy(dtype=int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        rows.append(
            {
                "subgroup_type": "climate_type",
                "subgroup": climate_type,
                "n": len(subset),
                "positive_n": int(y.sum()),
                "negative_n": int((1 - y).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": tp / max(1, tp + fn),
                "precision": tp / max(1, tp + fp),
                "specificity": tn / max(1, tn + fp),
                "selected_rate": float(pred.mean()),
                "fp_rate_within_negative": fp / max(1, int((1 - y).sum())),
            }
        )
    return pd.DataFrame(rows)


def control_fp_summary(analysis_data: pd.DataFrame) -> pd.DataFrame:
    controls = analysis_data.loc[analysis_data[step2.TARGET_COL] == 0]
    total_fp = int((controls["error_group"] == "FP").sum())
    rows: list[dict[str, Any]] = []
    for dimension, column in (
        ("sample_type", step2.SAMPLE_TYPE_COL),
        ("climate_type", step2.CLIMATE_COL),
    ):
        for group, subset in controls.groupby(column):
            fp_n = int((subset["error_group"] == "FP").sum())
            rows.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "control_n": len(subset),
                    "fp_n": fp_n,
                    "fp_rate": fp_n / len(subset),
                    "share_of_all_fp": fp_n / total_fp,
                    "score_median": float(subset["risk_score"].median()),
                    "score_q90": float(subset["risk_score"].quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def local_contributors(
    analysis_data: pd.DataFrame,
    shap_matrix: np.ndarray,
    features: list[str],
    selected_mask: np.ndarray,
    *,
    top_n: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    indices = np.flatnonzero(selected_mask)
    for row_index in indices:
        values = shap_matrix[row_index]
        positive = [index for index in np.argsort(values)[::-1] if values[index] > 0][
            :top_n
        ]
        negative = [index for index in np.argsort(values) if values[index] < 0][
            :top_n
        ]
        row: dict[str, Any] = {
            "sample_id": analysis_data.at[row_index, step2.SAMPLE_ID_COL],
            "outer_fold": int(analysis_data.at[row_index, "outer_fold"]),
            "target": int(analysis_data.at[row_index, step2.TARGET_COL]),
            "sample_type": analysis_data.at[row_index, step2.SAMPLE_TYPE_COL],
            "climate_type": analysis_data.at[row_index, step2.CLIMATE_COL],
            "group_id": analysis_data.at[row_index, step2.GROUP_COL],
            "error_group": analysis_data.at[row_index, "error_group"],
            "risk_score": float(analysis_data.at[row_index, "risk_score"]),
            "distance_from_threshold": float(
                analysis_data.at[row_index, "risk_score"] - FINAL_THRESHOLD
            ),
            "risk_rank": int(analysis_data.at[row_index, "risk_rank"]),
        }
        for rank, feature_index in enumerate(positive, start=1):
            feature = features[feature_index]
            row[f"positive_feature_{rank}"] = feature
            row[f"positive_shap_{rank}"] = float(values[feature_index])
            row[f"positive_value_{rank}"] = analysis_data.at[row_index, feature]
        for rank, feature_index in enumerate(negative, start=1):
            feature = features[feature_index]
            row[f"negative_feature_{rank}"] = feature
            row[f"negative_shap_{rank}"] = float(values[feature_index])
            row[f"negative_value_{rank}"] = analysis_data.at[row_index, feature]
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_distribution_by_group(
    analysis_data: pd.DataFrame,
    features: list[str],
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        numeric = pd.to_numeric(analysis_data[feature], errors="coerce")
        if numeric.notna().sum() < 20:
            continue
        for group, positions in analysis_data.groupby(group_column).groups.items():
            values = numeric.loc[list(positions)].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "q10": float(values.quantile(0.10)),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                    "q90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def top_risk_bands(analysis_data: pd.DataFrame) -> pd.Series:
    fraction = analysis_data["top_fraction"]
    return pd.Series(
        np.select(
            [
                fraction <= 0.05,
                fraction <= 0.10,
                fraction <= 0.20,
                fraction <= 0.30,
            ],
            ["top_0_5", "top_5_10", "top_10_20", "top_20_30"],
            default="bottom_70",
        ),
        index=analysis_data.index,
        name="risk_band",
    )


def fire_distribution_comparison(
    analysis_data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    fires = analysis_data.loc[analysis_data[step2.TARGET_COL] == 1].copy()
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        selected = fires["top_fraction"] <= fraction
        for feature in features:
            numeric = pd.to_numeric(fires[feature], errors="coerce")
            high = numeric.loc[selected].dropna()
            other = numeric.loc[~selected].dropna()
            if len(high) < 5 or len(other) < 5:
                continue
            overall_iqr = float(numeric.quantile(0.75) - numeric.quantile(0.25))
            median_difference = float(high.median() - other.median())
            rows.append(
                {
                    "top_fraction": fraction,
                    "feature": feature,
                    "top_risk_fire_n": len(high),
                    "other_fire_n": len(other),
                    "top_risk_fire_median": float(high.median()),
                    "other_fire_median": float(other.median()),
                    "median_difference": median_difference,
                    "standardized_median_difference_iqr": (
                        median_difference / overall_iqr if overall_iqr > 0 else np.nan
                    ),
                    "top_risk_fire_q25": float(high.quantile(0.25)),
                    "top_risk_fire_q75": float(high.quantile(0.75)),
                    "other_fire_q25": float(other.quantile(0.25)),
                    "other_fire_q75": float(other.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def categorical_distribution(
    analysis_data: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    categorical_features = [step2.CLIMATE_COL]
    for feature in categorical_features:
        for group, subset in analysis_data.groupby(group_column):
            counts = subset[feature].fillna("__missing__").value_counts()
            for category, count in counts.items():
                rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "category": category,
                        "n": int(count),
                        "share": float(count / len(subset)),
                    }
                )
    return pd.DataFrame(rows)


def save_shap_values(
    output_dir: Path,
    analysis_data: pd.DataFrame,
    shap_matrix: np.ndarray,
    features: list[str],
) -> None:
    metadata = analysis_data[
        [
            step2.SAMPLE_ID_COL,
            "outer_fold",
            step2.TARGET_COL,
            step2.SAMPLE_TYPE_COL,
            step2.CLIMATE_COL,
            step2.GROUP_COL,
            "risk_score",
            "pred_f2",
            "error_group",
            "risk_rank",
            "top_fraction",
        ]
    ].copy()
    shap_frame = pd.DataFrame(
        shap_matrix,
        columns=[f"shap__{feature}" for feature in features],
    )
    pd.concat([metadata.reset_index(drop=True), shap_frame], axis=1).to_csv(
        output_dir / "oof_shap_values_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_plots(
    output_dir: Path,
    global_importance: pd.DataFrame,
    climate_importance: pd.DataFrame,
    error_importance: pd.DataFrame,
    control_summary: pd.DataFrame,
    fire_comparison: pd.DataFrame,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    top20 = global_importance.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["mean_abs_shap"], color="#c44e52")
    ax.set_title("최종 LightGBM 전체 OOF SHAP 중요도")
    ax.set_xlabel("mean |SHAP| (raw margin)")
    fig.tight_layout()
    fig.savefig(plot_dir / "shap_global_top20.png", dpi=180)
    plt.close(fig)

    top15_features = global_importance.head(15)["feature"].tolist()
    climate_heat = climate_importance.loc[
        climate_importance["feature"].isin(top15_features)
    ].pivot(
        index="feature",
        columns="climate_type",
        values="importance_share",
    )
    climate_heat = climate_heat.reindex(top15_features)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(climate_heat, cmap="YlOrRd", annot=True, fmt=".3f", ax=ax)
    ax.set_title("기후지형유형별 SHAP 중요도 비중")
    ax.set_xlabel("기후지형유형")
    ax.set_ylabel("변수")
    fig.tight_layout()
    fig.savefig(plot_dir / "shap_by_climate_heatmap.png", dpi=180)
    plt.close(fig)

    error_heat = error_importance.loc[
        error_importance["feature"].isin(top15_features)
    ].pivot(
        index="feature",
        columns="error_group",
        values="mean_abs_shap",
    )
    error_heat = error_heat.reindex(top15_features)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(error_heat, cmap="Blues", annot=True, fmt=".3f", ax=ax)
    ax.set_title("TP·FN·FP·TN별 mean |SHAP|")
    ax.set_xlabel("오류 그룹")
    ax.set_ylabel("변수")
    fig.tight_layout()
    fig.savefig(plot_dir / "shap_by_error_group_heatmap.png", dpi=180)
    plt.close(fig)

    control = control_summary.loc[
        control_summary["dimension"] == "sample_type"
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=control, x="group", y="fp_rate", color="#dd8452", ax=ax)
    ax.set_title("F2 threshold 운영점의 대조군별 FP 비율")
    ax.set_xlabel("대조군 유형")
    ax.set_ylabel("FP rate")
    fig.tight_layout()
    fig.savefig(plot_dir / "fp_rate_by_control_type.png", dpi=180)
    plt.close(fig)

    top10_comparison = fire_comparison.loc[
        fire_comparison["top_fraction"] == 0.10
    ].copy()
    top10_comparison = top10_comparison.assign(
        abs_standardized=top10_comparison[
            "standardized_median_difference_iqr"
        ].abs()
    ).nlargest(15, "abs_standardized")
    top10_comparison = top10_comparison.sort_values(
        "standardized_median_difference_iqr"
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = np.where(
        top10_comparison["standardized_median_difference_iqr"] >= 0,
        "#c44e52",
        "#4c72b0",
    )
    ax.barh(
        top10_comparison["feature"],
        top10_comparison["standardized_median_difference_iqr"],
        color=colors,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("top 10% 산불 vs 나머지 산불 변수 중앙값 차이")
    ax.set_xlabel("median difference / 전체 산불 IQR")
    fig.tight_layout()
    fig.savefig(plot_dir / "top10_fire_distribution_difference.png", dpi=180)
    plt.close(fig)


def regenerate_plots_from_outputs(output_dir: Path) -> None:
    required = {
        "global_importance": output_dir / "shap_importance_global.csv",
        "climate_importance": output_dir / "shap_importance_by_climate.csv",
        "error_importance": output_dir / "shap_importance_by_error_group.csv",
        "control_summary": output_dir / "f2_false_positive_controls.csv",
        "fire_comparison": output_dir / "top_risk_fire_vs_other_fire_distribution.csv",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(f"플롯 재생성에 필요한 결과 파일이 없습니다: {path}")
    frames = {
        name: pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        for name, path in required.items()
    }
    save_plots(
        output_dir,
        frames["global_importance"],
        frames["climate_importance"],
        frames["error_importance"],
        frames["control_summary"],
        frames["fire_comparison"],
    )
    log(f"한글 폰트 적용 플롯 재생성 완료: {output_dir / 'plots'}")


def load_final_context(
    analysis_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, Path, Path]:
    step3_dir = analysis_dir / "outputs" / "step3_f2_threshold"
    step4_dir = analysis_dir / "outputs" / "step4_model_interpretation"
    threshold_path = step3_dir / "final_thresholds.json"
    final_oof_path = step3_dir / "final_oof__lightgbm_raw_score.csv"
    for path in (threshold_path, final_oof_path):
        if not path.exists():
            raise FileNotFoundError(path)
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    if thresholds.get("candidate_id") != FINAL_CANDIDATE_ID:
        raise ValueError("STEP 3 최종 모델이 tuned LightGBM 원형이 아닙니다.")
    if thresholds.get("calibration_applied") is not False:
        raise ValueError("STEP 4는 calibration이 없는 raw score만 해석합니다.")
    final_oof = pd.read_csv(final_oof_path, encoding="utf-8-sig", low_memory=False)
    selection = {
        "candidate_kind": "single",
        "streams": ["lgbm_baseline"],
    }
    return selection, thresholds, final_oof, step3_dir, step4_dir


def normalized_score_oof(final_oof: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "outer_fold",
        "y_true",
        "sample_type",
        "climate_type",
        "group_id",
        "risk_score",
    }
    missing = sorted(required - set(final_oof.columns))
    if missing:
        raise KeyError(f"STEP 4 최종 OOF 필수 열 누락: {missing}")
    result = final_oof[
        [
            "sample_id",
            "outer_fold",
            "y_true",
            "sample_type",
            "climate_type",
            "group_id",
            "risk_score",
        ]
    ].copy()
    return result.rename(columns={"risk_score": "y_prob"})


def base_model_performance(
    matrix: pd.DataFrame,
    streams: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    hard_mask = matrix["sample_type"].isin(["Target_1", "Target_0A"])
    for stream in streams:
        rows.append(
            {
                "stream": stream,
                "auprc": step2.safe_average_precision(
                    matrix["y_true"],
                    matrix[stream],
                ),
                "target0a_auprc": step2.safe_average_precision(
                    matrix.loc[hard_mask, "y_true"],
                    matrix.loc[hard_mask, stream],
                ),
                "score_mean": float(matrix[stream].mean()),
                "score_std": float(matrix[stream].std()),
            }
        )
    return pd.DataFrame(rows).sort_values("auprc", ascending=False)


def build_fold_scorer(
    selection: dict[str, Any],
    *,
    outer_fold: int,
    inner_oof: pd.DataFrame,
    selected_weights: pd.DataFrame,
):
    candidate_kind = str(selection["candidate_kind"])
    method = str(selection["method"])
    streams = tuple(str(value) for value in selection["streams"])
    ensemble_id = str(selection["selected_ensemble_id"])

    if candidate_kind in {"single", "fixed"}:
        return lambda frame: step3.combine_scores(frame, streams, method)

    if candidate_kind == "weighted":
        row = selected_weights.loc[
            (selected_weights["ensemble_id"] == ensemble_id)
            & (selected_weights["outer_fold"] == outer_fold)
        ]
        if len(row) != 1:
            raise RuntimeError(
                f"{ensemble_id} outer_fold={outer_fold} weight 행이 유일하지 않습니다."
            )
        weights = np.asarray(
            [float(row.iloc[0][f"weight__{stream}"]) for stream in streams]
        )
        return lambda frame: step3.combine_scores(
            frame,
            streams,
            method,
            weights=weights,
        )

    if candidate_kind == "stacking":
        transform = str(selection.get("transform", "logit"))
        train = inner_oof.loc[inner_oof["outer_fold"] == outer_fold]
        x_train = step3.transformed_matrix(train, streams, transform)
        y_train = train["y_true"].to_numpy(dtype=int)
        model = step3.build_stacking_model()
        model.fit(x_train, y_train)
        return lambda frame: step3.clipped_probability(
            model.predict_proba(
                step3.transformed_matrix(frame, streams, transform)
            )[:, 1]
        )

    raise ValueError(f"지원하지 않는 최종 후보 유형: {candidate_kind}")


def permutation_importance_by_outer_fold(
    matrix: pd.DataFrame,
    selection: dict[str, Any],
    step3_dir: Path,
    *,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner_path = step3_dir / "inner_base_oof.csv"
    weight_path = step3_dir / "selected_weights_by_outer_fold.csv"
    inner_oof = (
        pd.read_csv(inner_path, encoding="utf-8-sig", low_memory=False)
        if inner_path.exists()
        else pd.DataFrame()
    )
    selected_weights = (
        pd.read_csv(weight_path, encoding="utf-8-sig", low_memory=False)
        if weight_path.exists()
        else pd.DataFrame()
    )
    streams = [str(value) for value in selection["streams"]]
    rng = np.random.default_rng(RANDOM_STATE)
    importance_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    selected_oof_path = (
        step3_dir / f"oof__{selection['selected_ensemble_id']}.csv"
    )
    selected_oof = pd.read_csv(
        selected_oof_path,
        encoding="utf-8-sig",
        low_memory=False,
    ).set_index("sample_id")["y_score"]

    for outer_fold, valid in matrix.groupby("outer_fold", sort=True):
        valid = valid.copy().reset_index(drop=True)
        scorer = build_fold_scorer(
            selection,
            outer_fold=int(outer_fold),
            inner_oof=inner_oof,
            selected_weights=selected_weights,
        )
        baseline_score = scorer(valid)
        stored_score = valid["sample_id"].map(selected_oof).to_numpy(dtype=float)
        baseline_auprc = average_precision_score(valid["y_true"], baseline_score)
        reproduction_rows.append(
            {
                "outer_fold": int(outer_fold),
                "n": int(len(valid)),
                "baseline_auprc": float(baseline_auprc),
                "score_max_abs_difference": float(
                    np.max(np.abs(baseline_score - stored_score))
                ),
            }
        )
        for stream in streams:
            deltas: list[float] = []
            for repeat in range(repeats):
                permuted = valid.copy()
                permuted[stream] = rng.permutation(
                    permuted[stream].to_numpy(dtype=float)
                )
                permuted_auprc = average_precision_score(
                    permuted["y_true"],
                    scorer(permuted),
                )
                deltas.append(float(baseline_auprc - permuted_auprc))
            importance_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "stream": stream,
                    "repeats": int(repeats),
                    "auprc_drop_mean": float(np.mean(deltas)),
                    "auprc_drop_std": float(np.std(deltas, ddof=1)),
                    "auprc_drop_min": float(np.min(deltas)),
                    "auprc_drop_max": float(np.max(deltas)),
                }
            )

    by_fold = pd.DataFrame(importance_rows)
    aggregate = (
        by_fold.groupby("stream", as_index=False)
        .agg(
            auprc_drop_mean=("auprc_drop_mean", "mean"),
            auprc_drop_std_across_folds=("auprc_drop_mean", "std"),
            fold_positive_count=("auprc_drop_mean", lambda value: int((value > 0).sum())),
        )
        .sort_values("auprc_drop_mean", ascending=False)
    )
    aggregate["rank"] = np.arange(1, len(aggregate) + 1)
    aggregate = aggregate.merge(
        by_fold.groupby("stream", as_index=False).agg(
            repeat_std_mean=("auprc_drop_std", "mean")
        ),
        on="stream",
        how="left",
    )
    return aggregate, pd.DataFrame(reproduction_rows)


def save_ensemble_plots(
    output_dir: Path,
    base_performance: pd.DataFrame,
    permutation: pd.DataFrame,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=base_performance, x="auprc", y="stream", ax=ax, color="#4c78a8")
    ax.set_title("Base model OOF AUPRC")
    fig.tight_layout()
    fig.savefig(plot_dir / "base_model_auprc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=permutation,
        x="auprc_drop_mean",
        y="stream",
        ax=ax,
        color="#e45756",
    )
    ax.set_title("Outer-validation base-score permutation importance")
    ax.set_xlabel("AUPRC decrease after permutation")
    fig.tight_layout()
    fig.savefig(
        plot_dir / "ensemble_permutation_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_ensemble_interpretation(
    *,
    data: pd.DataFrame,
    score_oof: pd.DataFrame,
    selection: dict[str, Any],
    step3_dir: Path,
    output_dir: Path,
    permutation_repeats: int,
) -> dict[str, Any]:
    matrix = pd.read_csv(
        step3_dir / "ensemble_input_oof_matrix.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    streams = [str(value) for value in selection["streams"]]
    analysis_data = add_analysis_columns(data, score_oof)
    analysis_data["risk_band"] = top_risk_bands(analysis_data)
    base_performance = base_model_performance(matrix, streams)
    permutation, reproduction = permutation_importance_by_outer_fold(
        matrix,
        selection,
        step3_dir,
        repeats=permutation_repeats,
    )
    threshold_subgroups = threshold_subgroup_metrics(analysis_data)
    control_summary = control_fp_summary(analysis_data)
    features = final_features()
    error_distributions = numeric_distribution_by_group(
        analysis_data,
        features,
        "error_group",
    )
    fire_comparison = fire_distribution_comparison(analysis_data, features)
    fn_cases = analysis_data.loc[analysis_data["error_group"] == "FN"].copy()
    fp_cases = analysis_data.loc[analysis_data["error_group"] == "FP"].copy()
    target0a_fp = fp_cases.loc[
        fp_cases[step2.SAMPLE_TYPE_COL] == "Target_0A"
    ].copy()

    base_performance.to_csv(
        output_dir / "base_model_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    permutation.to_csv(
        output_dir / "outer_validation_permutation_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reproduction.to_csv(
        output_dir / "ensemble_score_reproduction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_subgroups.to_csv(
        output_dir / "f2_subgroup_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    control_summary.to_csv(
        output_dir / "f2_false_positive_controls.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fn_cases.to_csv(
        output_dir / "fn_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fp_cases.to_csv(
        output_dir / "all_fp_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    target0a_fp.to_csv(
        output_dir / "target0a_fp_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    error_distributions.to_csv(
        output_dir / "feature_distribution_by_error_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fire_comparison.to_csv(
        output_dir / "top_risk_fire_vs_other_fire_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analysis_data.to_csv(
        output_dir / "prediction_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    details_path = None
    if selection["candidate_kind"] == "stacking":
        details_path = step3_dir / "stacking_coefficients.csv"
    elif selection["candidate_kind"] == "weighted":
        details_path = step3_dir / "selected_weights_by_outer_fold.csv"
    if details_path is not None and details_path.exists():
        details = pd.read_csv(details_path, encoding="utf-8-sig")
        details.loc[
            details["ensemble_id"] == selection["selected_ensemble_id"]
        ].to_csv(
            output_dir / "selected_ensemble_parameters.csv",
            index=False,
            encoding="utf-8-sig",
        )

    confusion_counts = (
        analysis_data["error_group"].value_counts().reindex(ERROR_GROUPS, fill_value=0)
    )
    validations = pd.DataFrame(
        [
            {
                "check": "prediction_group_total",
                "value": int(confusion_counts.sum()),
                "expected": int(len(analysis_data)),
                "passed": int(confusion_counts.sum()) == int(len(analysis_data)),
            },
            {
                "check": "ensemble_reproduction_max_abs_difference",
                "value": float(reproduction["score_max_abs_difference"].max()),
                "expected": 1e-10,
                "passed": float(reproduction["score_max_abs_difference"].max()) <= 1e-10,
            },
        ]
    )
    validations.to_csv(
        output_dir / "validation_checks__step5.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_ensemble_plots(output_dir, base_performance, permutation)
    return {
        "interpretation_mode": "ensemble",
        "streams": streams,
        "tp": int(confusion_counts["TP"]),
        "fp": int(confusion_counts["FP"]),
        "fn": int(confusion_counts["FN"]),
        "tn": int(confusion_counts["TN"]),
        "permutation_repeats": int(permutation_repeats),
        "validation_passed": bool(validations["passed"].all()),
    }


def main() -> None:
    global FINAL_THRESHOLD
    args = parse_args()
    setup_plot_style()
    root = step2.find_project_root()
    analysis_dir = root / "jsw" / "Analysis" / "new_machine_learning"
    data_path = args.data or (
        root / "data" / "학습데이터" / "최종_머신러닝_학습데이터.csv"
    )
    output_dir = args.output_dir or (
        analysis_dir / "outputs" / "step4_model_interpretation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        regenerate_plots_from_outputs(output_dir)
        return

    selection, thresholds, final_oof, step3_dir, step4_dir = load_final_context(
        analysis_dir
    )
    FINAL_THRESHOLD = float(
        thresholds["operating_points"]["best_f2"]["threshold"]
    )
    score_oof = normalized_score_oof(final_oof)
    data = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)

    started = time.perf_counter()
    if not (
        selection["candidate_kind"] == "single"
        and selection["streams"] == ["lgbm_baseline"]
    ):
        raise RuntimeError("STEP 4는 tuned LightGBM 단일 모델만 해석합니다.")

    step1_dir = analysis_dir / "outputs" / "step1_single_models"
    step2_dir = analysis_dir / "outputs" / "step2_tuned_single_models"
    oof_path = step2_dir / "oof__TUNE_LGBM_ALL_ALL_LC_NONE.csv"
    params_path = (
        step2_dir / "selected_params__TUNE_LGBM_ALL_ALL_LC_NONE.json"
    )
    split_path = step1_dir / "split_manifest_outer_cv.csv"
    for path in (data_path, oof_path, params_path, split_path):
        if not path.exists():
            raise FileNotFoundError(path)

    oof = pd.read_csv(oof_path, encoding="utf-8-sig", low_memory=False)
    features = final_features()
    params_by_fold = load_params(params_path)
    splits = build_outer_splits(data, split_path)

    shap_matrix, validation = fit_oof_shap(
        data,
        oof,
        splits,
        params_by_fold,
        features,
        n_jobs=args.n_jobs,
        batch_size=args.shap_batch_size,
    )
    analysis_data = add_analysis_columns(data, score_oof)
    analysis_data["risk_band"] = top_risk_bands(analysis_data)

    global_importance = importance_table(shap_matrix, features)
    fold_importance = subgroup_importance(
        analysis_data,
        shap_matrix,
        features,
        "outer_fold",
        "outer_fold",
    )
    climate_importance = subgroup_importance(
        analysis_data,
        shap_matrix,
        features,
        step2.CLIMATE_COL,
        "climate_type",
    )
    error_importance = subgroup_importance(
        analysis_data,
        shap_matrix,
        features,
        "error_group",
        "error_group",
    )
    direction = direction_table(analysis_data, shap_matrix, features)
    threshold_subgroups = threshold_subgroup_metrics(analysis_data)
    control_summary = control_fp_summary(analysis_data)
    fn_cases = local_contributors(
        analysis_data,
        shap_matrix,
        features,
        analysis_data["error_group"].to_numpy() == "FN",
    )
    target0a_fp_cases = local_contributors(
        analysis_data,
        shap_matrix,
        features,
        (
            (analysis_data["error_group"].to_numpy() == "FP")
            & (
                analysis_data[step2.SAMPLE_TYPE_COL].to_numpy()
                == "Target_0A"
            )
        ),
    )
    all_fp_cases = local_contributors(
        analysis_data,
        shap_matrix,
        features,
        analysis_data["error_group"].to_numpy() == "FP",
    )
    error_distributions = numeric_distribution_by_group(
        analysis_data,
        features,
        "error_group",
    )
    risk_band_distributions = numeric_distribution_by_group(
        analysis_data,
        features,
        "risk_band",
    )
    fire_comparison = fire_distribution_comparison(analysis_data, features)
    risk_band_categories = categorical_distribution(
        analysis_data,
        "risk_band",
    )

    global_importance.to_csv(
        output_dir / "shap_importance_global.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_importance.to_csv(
        output_dir / "shap_importance_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    climate_importance.to_csv(
        output_dir / "shap_importance_by_climate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    error_importance.to_csv(
        output_dir / "shap_importance_by_error_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direction.to_csv(
        output_dir / "shap_direction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_subgroups.to_csv(
        output_dir / "f2_subgroup_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    control_summary.to_csv(
        output_dir / "f2_false_positive_controls.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fn_cases.to_csv(
        output_dir / "fn_cases_with_shap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    target0a_fp_cases.to_csv(
        output_dir / "target0a_fp_cases_with_shap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_fp_cases.to_csv(
        output_dir / "all_fp_cases_with_shap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    error_distributions.to_csv(
        output_dir / "feature_distribution_by_error_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    risk_band_distributions.to_csv(
        output_dir / "feature_distribution_by_risk_band.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fire_comparison.to_csv(
        output_dir / "top_risk_fire_vs_other_fire_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    risk_band_categories.to_csv(
        output_dir / "categorical_distribution_by_risk_band.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation.to_csv(
        output_dir / "validation_checks__step4.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analysis_data[
        [
            step2.SAMPLE_ID_COL,
            "outer_fold",
            step2.TARGET_COL,
            step2.SAMPLE_TYPE_COL,
            step2.CLIMATE_COL,
            step2.GROUP_COL,
            "risk_score",
            "pred_f2",
            "error_group",
            "risk_rank",
            "top_fraction",
            "risk_band",
        ]
    ].to_csv(
        output_dir / "prediction_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_shap_values(output_dir, analysis_data, shap_matrix, features)
    save_plots(
        output_dir,
        global_importance,
        climate_importance,
        error_importance,
        control_summary,
        fire_comparison,
    )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "started_at": timestamp(),
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "shap_version": shap.__version__,
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "oof_path": str(oof_path.resolve()),
        "n_rows": len(data),
        "positive_n": int(data[step2.TARGET_COL].sum()),
        "feature_n": len(features),
        "outer_fold_n": len(splits),
        "candidate_id": FINAL_CANDIDATE_ID,
        "score_type": "raw_oof_model_score",
        "calibration_applied": False,
        "final_threshold": FINAL_THRESHOLD,
        "primary_operating_point": "best_f2",
        "shap_output": "LightGBM raw margin contribution",
        "interpretation": "predictive contribution, not causal effect",
    }
    write_json(output_dir / "run_manifest__step4.json", manifest)
    log(
        f"Step 4 완료 | mode=single_lightgbm | output={output_dir} | "
        f"elapsed={step2.format_seconds(manifest['elapsed_seconds'])}"
    )


if __name__ == "__main__":
    main()
