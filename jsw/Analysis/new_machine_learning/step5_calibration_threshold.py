from __future__ import annotations

import argparse
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import step2_tuned_single_models as step2


RANDOM_STATE = 20260622
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
CLIP_EPSILON = 1e-6
N_CALIBRATION_BINS = 10

STACK_ID = "stack__drop_isi__lgbm_rf_et__logit"
GEOMEAN_ID = "fixed__drop_isi__lgbm_rf_et__geomean"
SINGLE_ID = "single__lgbm_drop_isi"

DEFAULT_CANDIDATE_IDS = [STACK_ID, GEOMEAN_ID, SINGLE_ID]
DEFAULT_CALIBRATION_METHODS = ["raw", "sigmoid", "isotonic"]
METADATA_COLUMNS = [
    "sample_id",
    "outer_fold",
    "y_true",
    "sample_type",
    "climate_type",
    "group_id",
]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    role: str
    input_filename: str
    description: str


@dataclass
class FittedCalibrator:
    calibration_id: str
    model: Any | None
    feature_transform: str


CANDIDATE_SPECS = {
    STACK_ID: CandidateSpec(
        candidate_id=STACK_ID,
        role="primary_probability_candidate",
        input_filename=f"oof__{STACK_ID}.csv",
        description="Step4 official selected stacking score",
    ),
    GEOMEAN_ID: CandidateSpec(
        candidate_id=GEOMEAN_ID,
        role="simple_challenger",
        input_filename=f"oof__{GEOMEAN_ID}.csv",
        description="Step4 fixed geometric mean challenger",
    ),
    SINGLE_ID: CandidateSpec(
        candidate_id=SINGLE_ID,
        role="single_model_reference",
        input_filename=f"oof__{SINGLE_ID}.csv",
        description="Step3/4 best single LightGBM reference",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="new_machine_learning Step5: calibration and final threshold selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--candidate-ids",
        nargs="*",
        default=["all"],
        help="쉼표 또는 공백으로 구분. all이면 Step5 기본 3개 후보 사용",
    )
    parser.add_argument(
        "--calibration-methods",
        nargs="*",
        default=["all"],
        help="쉼표 또는 공백으로 구분. all이면 raw,sigmoid,isotonic",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def select_from_csv_argument(
    values: list[str],
    allowed: list[str],
    *,
    argument_name: str,
) -> list[str]:
    if not values or values == ["all"]:
        return allowed
    selected: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if token not in allowed:
                raise ValueError(
                    f"{argument_name} 값이 잘못됐습니다: {token}. 허용값: {allowed}"
                )
            selected.append(token)
    return list(dict.fromkeys(selected))


def project_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = step2.find_project_root()
    input_dir = (
        Path(args.input_dir)
        if args.input_dir
        else root / "jsw" / "Analysis" / "new_machine_learning" / "outputs" / "step4_ensemble"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "jsw"
        / "Analysis"
        / "new_machine_learning"
        / "outputs"
        / "step5_calibration_threshold"
    )
    return input_dir, output_dir


def clipped_probability(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), CLIP_EPSILON, 1.0 - CLIP_EPSILON)


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    probability = clipped_probability(values)
    return np.log(probability / (1.0 - probability))


def calibrator_features(scores: np.ndarray | pd.Series, feature_transform: str) -> np.ndarray:
    if feature_transform == "logit":
        values = logit(scores)
    elif feature_transform == "raw":
        values = clipped_probability(scores)
    else:
        raise ValueError(f"알 수 없는 calibration feature transform: {feature_transform}")
    return values.reshape(-1, 1)


def fit_calibrator(
    y_true: np.ndarray | pd.Series,
    raw_score: np.ndarray | pd.Series,
    calibration_id: str,
) -> FittedCalibrator:
    y = np.asarray(y_true, dtype=int)
    score = clipped_probability(raw_score)
    if calibration_id == "raw":
        return FittedCalibrator(calibration_id="raw", model=None, feature_transform="raw")
    if len(np.unique(y)) < 2:
        raise ValueError(f"{calibration_id} calibration train set에 양/음성이 모두 필요합니다.")
    if calibration_id == "sigmoid":
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        model.fit(calibrator_features(score, "logit"), y)
        return FittedCalibrator(calibration_id="sigmoid", model=model, feature_transform="logit")
    if calibration_id == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        model.fit(score, y)
        return FittedCalibrator(calibration_id="isotonic", model=model, feature_transform="raw")
    raise ValueError(f"알 수 없는 calibration_id: {calibration_id}")


def apply_calibrator(calibrator: FittedCalibrator, raw_score: np.ndarray | pd.Series) -> np.ndarray:
    score = clipped_probability(raw_score)
    if calibrator.calibration_id == "raw":
        return score
    if calibrator.calibration_id == "sigmoid":
        return clipped_probability(
            calibrator.model.predict_proba(
                calibrator_features(score, calibrator.feature_transform)
            )[:, 1]
        )
    if calibrator.calibration_id == "isotonic":
        return clipped_probability(calibrator.model.predict(score))
    raise ValueError(f"알 수 없는 calibration_id: {calibrator.calibration_id}")


def load_candidate_oof(input_dir: Path, spec: CandidateSpec) -> pd.DataFrame:
    path = input_dir / spec.input_filename
    if not path.exists():
        raise FileNotFoundError(f"Step5 입력 OOF 파일이 없습니다: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = set(METADATA_COLUMNS + ["y_score"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} 필수 열 누락: {missing}")
    result = frame[METADATA_COLUMNS + ["y_score"]].copy()
    result["sample_id"] = result["sample_id"].astype(str)
    result["outer_fold"] = result["outer_fold"].astype(int)
    result["y_true"] = result["y_true"].astype(int)
    result["sample_type"] = result["sample_type"].astype(str)
    result["climate_type"] = result["climate_type"].astype(str)
    result["group_id"] = result["group_id"].astype(str)
    result["raw_score"] = clipped_probability(result["y_score"])
    result = result.drop(columns=["y_score"])
    result["candidate_id"] = spec.candidate_id
    result["candidate_role"] = spec.role
    result["input_path"] = str(path)
    return result


def load_candidates(input_dir: Path, candidate_ids: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for candidate_id in candidate_ids:
        frames[candidate_id] = load_candidate_oof(input_dir, CANDIDATE_SPECS[candidate_id])
    return frames


def candidate_input_manifest(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, frame in frames.items():
        first = frame.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_role": first["candidate_role"],
                "input_path": first["input_path"],
                "n": int(len(frame)),
                "positive_n": int(frame["y_true"].sum()),
                "positive_rate": float(frame["y_true"].mean()),
                "outer_fold_count": int(frame["outer_fold"].nunique()),
                "raw_score_min": float(frame["raw_score"].min()),
                "raw_score_max": float(frame["raw_score"].max()),
                "description": CANDIDATE_SPECS[candidate_id].description,
            }
        )
    return pd.DataFrame(rows)


def validate_candidate_inputs(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not frames:
        return pd.DataFrame()
    first_id = next(iter(frames))
    reference = frames[first_id][METADATA_COLUMNS].sort_values("sample_id").reset_index(drop=True)
    expected_n = len(reference)
    for candidate_id, frame in frames.items():
        part = frame.sort_values("sample_id").reset_index(drop=True)
        score = part["raw_score"].to_numpy(dtype=float)
        metadata = part[METADATA_COLUMNS]
        aligned = (
            len(part) == expected_n
            and metadata[["sample_id", "outer_fold", "y_true"]].equals(
                reference[["sample_id", "outer_fold", "y_true"]]
            )
        )
        input_path = str(part.iloc[0]["input_path"]).lower()
        lockbox_path_detected = any(
            token in input_path for token in ["lockbox", "holdout", "\\test", "/test"]
        )
        row = {
            "candidate_id": candidate_id,
            "check_scope": "input_oof",
            "expected_n": int(expected_n),
            "prediction_n": int(len(part)),
            "duplicate_sample_id_n": int(part["sample_id"].duplicated().sum()),
            "metadata_alignment_passed": bool(aligned),
            "nan_score_n": int(np.isnan(score).sum()),
            "inf_score_n": int(np.isinf(score).sum()),
            "score_below_zero_n": int((score < 0).sum()),
            "score_above_one_n": int((score > 1).sum()),
            "outer_fold_count": int(part["outer_fold"].nunique()),
            "folds_without_positive_n": int(
                sum(int(fold["y_true"].sum()) == 0 for _, fold in part.groupby("outer_fold"))
            ),
            "lockbox_path_detected": bool(lockbox_path_detected),
        }
        row["passed"] = (
            row["prediction_n"] == row["expected_n"]
            and row["duplicate_sample_id_n"] == 0
            and row["metadata_alignment_passed"]
            and row["nan_score_n"] == 0
            and row["inf_score_n"] == 0
            and row["score_below_zero_n"] == 0
            and row["score_above_one_n"] == 0
            and row["folds_without_positive_n"] == 0
            and not row["lockbox_path_detected"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_curve(y_true: np.ndarray | pd.Series, probability: np.ndarray | pd.Series) -> pd.DataFrame:
    return step2.threshold_curve(y_true, probability)


def select_thresholds_step5(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
) -> dict[str, float]:
    curve = threshold_curve(y_true, probability)
    thresholds: dict[str, float] = {"fixed_0.50": 0.5}
    if curve.empty:
        thresholds.update({"best_f1": 0.5, "best_f2": 0.5})
        for target in step2.RECALL_TARGETS:
            thresholds[f"recall_ge_{target:.2f}"] = 0.0
        return thresholds

    best_f1 = curve.sort_values(
        ["f1", "balanced_accuracy", "selected_rate", "threshold"],
        ascending=[False, False, True, False],
    ).iloc[0]
    best_f2 = curve.sort_values(
        ["f2", "recall", "selected_rate", "threshold"],
        ascending=[False, False, True, False],
    ).iloc[0]
    thresholds["best_f1"] = float(best_f1["threshold"])
    thresholds["best_f2"] = float(best_f2["threshold"])

    for target in step2.RECALL_TARGETS:
        candidates = curve.loc[curve["recall"] >= target]
        if candidates.empty:
            thresholds[f"recall_ge_{target:.2f}"] = 0.0
            continue
        selected = candidates.sort_values(
            ["precision", "selected_rate", "threshold"],
            ascending=[False, True, False],
        ).iloc[0]
        thresholds[f"recall_ge_{target:.2f}"] = float(selected["threshold"])
    return thresholds


def threshold_rows(
    predictions: pd.DataFrame,
    *,
    score_col: str,
    threshold_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, calibration_id), part in predictions.groupby(
        ["candidate_id", "calibration_id"],
        sort=False,
    ):
        thresholds = select_thresholds_step5(part["y_true"], part[score_col])
        for operating_point, threshold in thresholds.items():
            row = {
                "candidate_id": candidate_id,
                "calibration_id": calibration_id,
                "operating_point": operating_point,
                "threshold_source": threshold_source,
            }
            row.update(
                step2.classification_metrics_at_threshold(
                    part["y_true"],
                    part[score_col],
                    threshold,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def top_risk_rows(predictions: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, calibration_id), part in predictions.groupby(
        ["candidate_id", "calibration_id"],
        sort=False,
    ):
        sorted_part = part.sort_values(score_col, ascending=False).reset_index(drop=True)
        total_positive = int(sorted_part["y_true"].sum())
        base_rate = float(sorted_part["y_true"].mean())
        for fraction in step2.TOP_RISK_FRACTIONS:
            selected_n = max(1, int(math.ceil(len(sorted_part) * fraction)))
            selected = sorted_part.head(selected_n)
            captured = int(selected["y_true"].sum())
            precision = float(selected["y_true"].mean())
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "calibration_id": calibration_id,
                    "top_fraction": fraction,
                    "selected_n": selected_n,
                    "positive_captured_n": captured,
                    "total_positive_n": total_positive,
                    "capture_rate": captured / total_positive if total_positive else float("nan"),
                    "precision": precision,
                    "lift": precision / base_rate if base_rate > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def subgroup_rows(predictions: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, calibration_id), part in predictions.groupby(
        ["candidate_id", "calibration_id"],
        sort=False,
    ):
        for negative_type in ["Target_0A", "Target_0B1", "Target_0B2"]:
            subset = part.loc[part["sample_type"].isin(["Target_1", negative_type])]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "calibration_id": calibration_id,
                    "subgroup_type": "sample_type_pair",
                    "subgroup_value": f"Target_1_vs_{negative_type}",
                    **step2.probability_metrics(subset["y_true"], subset[score_col]),
                }
            )
        for climate_type, subset in part.groupby("climate_type"):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "calibration_id": calibration_id,
                    "subgroup_type": "climate_type",
                    "subgroup_value": str(climate_type),
                    **step2.probability_metrics(subset["y_true"], subset[score_col]),
                }
            )
        for outer_fold, subset in part.groupby("outer_fold"):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "calibration_id": calibration_id,
                    "subgroup_type": "outer_fold",
                    "subgroup_value": str(outer_fold),
                    **step2.probability_metrics(subset["y_true"], subset[score_col]),
                }
            )
    return pd.DataFrame(rows)


def fold_metric_rows(predictions: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, calibration_id, outer_fold), part in predictions.groupby(
        ["candidate_id", "calibration_id", "outer_fold"],
        sort=False,
    ):
        rows.append(
            {
                "candidate_id": candidate_id,
                "calibration_id": calibration_id,
                "outer_fold": int(outer_fold),
                **step2.probability_metrics(part["y_true"], part[score_col]),
            }
        )
    return pd.DataFrame(rows)


def calibration_bin_rows(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    *,
    n_bins: int,
    bin_type: str,
) -> list[dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    prob = clipped_probability(probability)
    n = len(prob)
    if n == 0:
        return []
    if bin_type == "equal_width":
        bin_index = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    elif bin_type == "quantile":
        order = np.argsort(prob, kind="stable")
        bin_index = np.zeros(n, dtype=int)
        bin_index[order] = np.minimum(np.arange(n) * n_bins // n, n_bins - 1)
    else:
        raise ValueError(f"알 수 없는 bin_type: {bin_type}")
    rows: list[dict[str, Any]] = []
    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        if not mask.any():
            lower = bin_id / n_bins if bin_type == "equal_width" else float("nan")
            upper = (bin_id + 1) / n_bins if bin_type == "equal_width" else float("nan")
            rows.append(
                {
                    "bin_type": bin_type,
                    "bin_id": bin_id,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "n": 0,
                    "positive_n": 0,
                    "mean_score": float("nan"),
                    "observed_rate": float("nan"),
                    "abs_error": float("nan"),
                }
            )
            continue
        selected_prob = prob[mask]
        selected_y = y[mask]
        rows.append(
            {
                "bin_type": bin_type,
                "bin_id": bin_id,
                "bin_lower": float(selected_prob.min()),
                "bin_upper": float(selected_prob.max()),
                "n": int(mask.sum()),
                "positive_n": int(selected_y.sum()),
                "mean_score": float(selected_prob.mean()),
                "observed_rate": float(selected_y.mean()),
                "abs_error": float(abs(selected_prob.mean() - selected_y.mean())),
            }
        )
    return rows


def calibration_error_from_bins(bin_frame: pd.DataFrame) -> tuple[float, float]:
    nonempty = bin_frame.loc[bin_frame["n"] > 0]
    if nonempty.empty:
        return float("nan"), float("nan")
    total_n = float(nonempty["n"].sum())
    ece = float((nonempty["n"] / total_n * nonempty["abs_error"]).sum())
    mce = float(nonempty["abs_error"].max())
    return ece, mce


def calibration_intercept_slope(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    try:
        model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1000)
        model.fit(logit(probability).reshape(-1, 1), y)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return float("nan"), float("nan")


def calibration_bins_for_predictions(
    predictions: pd.DataFrame,
    *,
    score_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, calibration_id), part in predictions.groupby(
        ["candidate_id", "calibration_id"],
        sort=False,
    ):
        for bin_type in ["equal_width", "quantile"]:
            for row in calibration_bin_rows(
                part["y_true"],
                part[score_col],
                n_bins=N_CALIBRATION_BINS,
                bin_type=bin_type,
            ):
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "calibration_id": calibration_id,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def summary_rows(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    *,
    score_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fold_stats = (
        fold_metrics.groupby(["candidate_id", "calibration_id"], as_index=False)
        .agg(
            fold_auprc_mean=("auprc", "mean"),
            fold_auprc_std=("auprc", "std"),
        )
    )
    for (candidate_id, calibration_id), part in predictions.groupby(
        ["candidate_id", "calibration_id"],
        sort=False,
    ):
        metrics = step2.probability_metrics(part["y_true"], part[score_col])
        equal_bins = calibration_bins.loc[
            (calibration_bins["candidate_id"] == candidate_id)
            & (calibration_bins["calibration_id"] == calibration_id)
            & (calibration_bins["bin_type"] == "equal_width")
        ]
        quantile_bins = calibration_bins.loc[
            (calibration_bins["candidate_id"] == candidate_id)
            & (calibration_bins["calibration_id"] == calibration_id)
            & (calibration_bins["bin_type"] == "quantile")
        ]
        ece_equal, mce_equal = calibration_error_from_bins(equal_bins)
        ece_quantile, mce_quantile = calibration_error_from_bins(quantile_bins)
        intercept, slope = calibration_intercept_slope(part["y_true"], part[score_col])
        hard_mask = part["sample_type"].isin(["Target_1", "Target_0A"])
        row = {
            "candidate_id": candidate_id,
            "candidate_role": str(part.iloc[0]["candidate_role"]),
            "calibration_id": calibration_id,
            **metrics,
            "target0a_auprc": step2.safe_average_precision(
                part.loc[hard_mask, "y_true"],
                part.loc[hard_mask, score_col],
            ),
            "ece_equal_width": ece_equal,
            "mce_equal_width": mce_equal,
            "ece_quantile": ece_quantile,
            "mce_quantile": mce_quantile,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.merge(fold_stats, on=["candidate_id", "calibration_id"], how="left")
    return summary


def add_operating_metrics(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_risk: pd.DataFrame,
) -> pd.DataFrame:
    result = summary.copy()
    for operating_point, prefix in [
        ("recall_ge_0.70", "recall70"),
        ("recall_ge_0.90", "recall90"),
        ("recall_ge_0.95", "recall95"),
    ]:
        subset = thresholds.loc[
            thresholds["operating_point"] == operating_point,
            [
                "candidate_id",
                "calibration_id",
                "threshold",
                "recall",
                "precision",
                "selected_rate",
                "fn",
            ],
        ].rename(
            columns={
                "threshold": f"{prefix}_threshold",
                "recall": f"{prefix}_recall",
                "precision": f"{prefix}_precision",
                "selected_rate": f"{prefix}_selected_rate",
                "fn": f"{prefix}_fn",
            }
        )
        result = result.merge(subset, on=["candidate_id", "calibration_id"], how="left")
    for fraction, prefix in [(0.10, "top10"), (0.20, "top20"), (0.30, "top30")]:
        subset = top_risk.loc[
            np.isclose(top_risk["top_fraction"], fraction),
            ["candidate_id", "calibration_id", "capture_rate", "precision"],
        ].rename(
            columns={
                "capture_rate": f"{prefix}_capture",
                "precision": f"{prefix}_precision",
            }
        )
        result = result.merge(subset, on=["candidate_id", "calibration_id"], how="left")
    return result


def cross_calibrate_candidate(
    frame: pd.DataFrame,
    *,
    calibration_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibrated_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    candidate_id = str(frame.iloc[0]["candidate_id"])
    role = str(frame.iloc[0]["candidate_role"])
    for outer_fold in sorted(frame["outer_fold"].unique()):
        train = frame.loc[frame["outer_fold"] != outer_fold].copy()
        valid = frame.loc[frame["outer_fold"] == outer_fold].copy()
        calibrator = fit_calibrator(train["y_true"], train["raw_score"], calibration_id)
        train_score = apply_calibrator(calibrator, train["raw_score"])
        valid_score = apply_calibrator(calibrator, valid["raw_score"])

        fold_fit = {
            "candidate_id": candidate_id,
            "calibration_id": calibration_id,
            "outer_fold": int(outer_fold),
            "train_n": int(len(train)),
            "valid_n": int(len(valid)),
            "train_positive_n": int(train["y_true"].sum()),
            "valid_positive_n": int(valid["y_true"].sum()),
        }
        if calibration_id == "sigmoid":
            fold_fit["sigmoid_intercept"] = float(calibrator.model.intercept_[0])
            fold_fit["sigmoid_coef"] = float(calibrator.model.coef_[0][0])
        elif calibration_id == "isotonic":
            fold_fit["isotonic_threshold_count"] = int(len(calibrator.model.X_thresholds_))
        fit_rows.append(fold_fit)

        thresholds = select_thresholds_step5(train["y_true"], train_score)
        for operating_point, threshold in thresholds.items():
            row = {
                "candidate_id": candidate_id,
                "calibration_id": calibration_id,
                "outer_fold": int(outer_fold),
                "operating_point": operating_point,
                "threshold_source": "train_folds_cross_calibrated",
            }
            row.update(step2.classification_metrics_at_threshold(valid["y_true"], valid_score, threshold))
            audit_rows.append(row)

        valid = valid.copy()
        valid["calibration_id"] = calibration_id
        valid["calibrated_score"] = valid_score
        valid["score_source"] = "cross_calibrated_oof"
        valid["candidate_role"] = role
        calibrated_parts.append(valid)

    calibrated = pd.concat(calibrated_parts, ignore_index=True)
    calibrated = calibrated.sort_values("sample_id").reset_index(drop=True)
    return calibrated, pd.DataFrame(audit_rows), pd.DataFrame(fit_rows)


def cross_calibrate_all(
    frames: dict[str, pd.DataFrame],
    calibration_methods: list[str],
    *,
    progress_bar: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = [(candidate_id, method) for candidate_id in frames for method in calibration_methods]
    iterator = tasks
    if progress_bar and step2.tqdm is not None:
        iterator = step2.tqdm(tasks, desc="Step5 calibration", unit="candidate")
    calibrated_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    for candidate_id, method in iterator:
        calibrated, audit, fit = cross_calibrate_candidate(frames[candidate_id], calibration_id=method)
        calibrated_frames.append(calibrated)
        audit_frames.append(audit)
        fit_frames.append(fit)
    return (
        pd.concat(calibrated_frames, ignore_index=True),
        pd.concat(audit_frames, ignore_index=True),
        pd.concat(fit_frames, ignore_index=True),
    )


def validation_rows_for_calibrated(predictions: pd.DataFrame, expected_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, calibration_id), part in predictions.groupby(
        ["candidate_id", "calibration_id"],
        sort=False,
    ):
        score = part["calibrated_score"].to_numpy(dtype=float)
        row = {
            "candidate_id": candidate_id,
            "calibration_id": calibration_id,
            "check_scope": "cross_calibrated_oof",
            "expected_n": int(expected_n),
            "prediction_n": int(len(part)),
            "duplicate_sample_id_n": int(part["sample_id"].duplicated().sum()),
            "nan_score_n": int(np.isnan(score).sum()),
            "inf_score_n": int(np.isinf(score).sum()),
            "score_below_zero_n": int((score < 0).sum()),
            "score_above_one_n": int((score > 1).sum()),
            "outer_fold_count": int(part["outer_fold"].nunique()),
            "folds_without_positive_n": int(
                sum(int(fold["y_true"].sum()) == 0 for _, fold in part.groupby("outer_fold"))
            ),
            "lockbox_accessed": False,
        }
        row["passed"] = (
            row["prediction_n"] == row["expected_n"]
            and row["duplicate_sample_id_n"] == 0
            and row["nan_score_n"] == 0
            and row["inf_score_n"] == 0
            and row["score_below_zero_n"] == 0
            and row["score_above_one_n"] == 0
            and row["folds_without_positive_n"] == 0
            and not row["lockbox_accessed"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_calibration_selection_flags(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["raw_brier"] = np.nan
    result["raw_log_loss"] = np.nan
    result["raw_auprc"] = np.nan
    result["raw_target0a_auprc"] = np.nan
    result["raw_top20_capture"] = np.nan
    result["raw_recall70_precision"] = np.nan
    result["raw_recall70_selected_rate"] = np.nan
    result["brier_improvement_vs_raw"] = np.nan
    result["log_loss_improvement_vs_raw"] = np.nan
    result["ece_improvement_vs_raw"] = np.nan
    result["eligible_calibration"] = False
    result["calibration_reject_reason"] = ""
    result["selected_for_candidate"] = False

    for candidate_id, part in result.groupby("candidate_id", sort=False):
        raw_rows = part.loc[part["calibration_id"] == "raw"]
        if raw_rows.empty:
            continue
        raw = raw_rows.iloc[0]
        idx = part.index
        result.loc[idx, "raw_brier"] = raw["brier"]
        result.loc[idx, "raw_log_loss"] = raw["log_loss"]
        result.loc[idx, "raw_auprc"] = raw["auprc"]
        result.loc[idx, "raw_target0a_auprc"] = raw["target0a_auprc"]
        result.loc[idx, "raw_top20_capture"] = raw["top20_capture"]
        result.loc[idx, "raw_recall70_precision"] = raw["recall70_precision"]
        result.loc[idx, "raw_recall70_selected_rate"] = raw["recall70_selected_rate"]
        result.loc[idx, "brier_improvement_vs_raw"] = raw["brier"] - result.loc[idx, "brier"]
        result.loc[idx, "log_loss_improvement_vs_raw"] = raw["log_loss"] - result.loc[idx, "log_loss"]
        result.loc[idx, "ece_improvement_vs_raw"] = raw["ece_equal_width"] - result.loc[idx, "ece_equal_width"]

        eligible_indices: list[int] = []
        for row_idx in idx:
            row = result.loc[row_idx]
            reasons: list[str] = []
            if row["calibration_id"] != "raw":
                if raw["auprc"] - row["auprc"] > 0.001:
                    reasons.append("auprc_drop_gt_0.001")
                if raw["target0a_auprc"] - row["target0a_auprc"] > 0.002:
                    reasons.append("target0a_drop_gt_0.002")
                if raw["top20_capture"] - row["top20_capture"] > 0.005:
                    reasons.append("top20_capture_drop_gt_0.5pp")
                if (
                    row["recall70_selected_rate"] - raw["recall70_selected_rate"] > 0.005
                    and row["recall70_precision"] <= raw["recall70_precision"]
                ):
                    reasons.append("recall70_more_selected_without_precision_gain")
            if reasons:
                result.loc[row_idx, "eligible_calibration"] = False
                result.loc[row_idx, "calibration_reject_reason"] = "|".join(reasons)
            else:
                result.loc[row_idx, "eligible_calibration"] = True
                result.loc[row_idx, "calibration_reject_reason"] = ""
                eligible_indices.append(row_idx)

        eligible = result.loc[eligible_indices].copy()
        nonraw_both_improved = eligible.loc[
            (eligible["calibration_id"] != "raw")
            & (eligible["brier_improvement_vs_raw"] > 0)
            & (eligible["log_loss_improvement_vs_raw"] > 0)
        ]
        nonraw_one_improved = eligible.loc[
            (eligible["calibration_id"] != "raw")
            & (
                (eligible["brier_improvement_vs_raw"] > 0)
                | (eligible["log_loss_improvement_vs_raw"] > 0)
            )
            & (
                (eligible["ece_improvement_vs_raw"] > 0)
                | (
                    (eligible["calibration_slope"] - 1.0).abs()
                    <= abs(float(raw["calibration_slope"]) - 1.0)
                )
            )
        ]
        if not nonraw_both_improved.empty:
            pool = nonraw_both_improved
        elif not nonraw_one_improved.empty:
            pool = nonraw_one_improved
        else:
            pool = eligible.loc[eligible["calibration_id"] == "raw"]

        method_priority = {"sigmoid": 0, "isotonic": 1, "raw": 2}
        selected = (
            pool.assign(method_priority=pool["calibration_id"].map(method_priority).fillna(9))
            .sort_values(
                [
                    "brier_improvement_vs_raw",
                    "log_loss_improvement_vs_raw",
                    "ece_improvement_vs_raw",
                    "auprc",
                    "method_priority",
                ],
                ascending=[False, False, False, False, True],
            )
            .iloc[0]
        )
        result.loc[selected.name, "selected_for_candidate"] = True
    return result


def choose_final_score(summary: pd.DataFrame) -> tuple[pd.Series, str]:
    selected = summary.loc[summary["selected_for_candidate"]].copy()
    selected = selected.loc[selected["eligible_calibration"]].copy()
    if selected.empty:
        raise RuntimeError("선택 가능한 calibrated score가 없습니다.")
    by_candidate = selected.set_index("candidate_id", drop=False)
    if STACK_ID not in by_candidate.index:
        final = selected.sort_values("auprc", ascending=False).iloc[0]
        return final, "fallback_best_auprc_no_stack_candidate"

    stack = by_candidate.loc[STACK_ID]
    final = stack
    reason = "default_step4_primary_stack"
    if GEOMEAN_ID in by_candidate.index:
        geo = by_candidate.loc[GEOMEAN_ID]
        auprc_within = float(geo["auprc"]) >= float(stack["auprc"]) - 0.002
        threshold_better = (
            float(geo["recall70_selected_rate"]) <= float(stack["recall70_selected_rate"]) - 0.003
            or float(geo["recall70_precision"]) >= float(stack["recall70_precision"]) + 0.005
        )
        probability_better = (
            float(geo["brier"]) <= float(stack["brier"]) - 0.0005
            or float(geo["log_loss"]) <= float(stack["log_loss"]) - 0.002
        )
        top10_condition = (
            float(geo["top10_capture"]) > float(stack["top10_capture"])
            and float(geo["top20_capture"]) >= float(stack["top20_capture"]) - 0.005
            and float(geo["target0a_auprc"]) >= float(stack["target0a_auprc"]) - 0.002
        )
        if auprc_within and (threshold_better or probability_better or top10_condition):
            final = geo
            reason = "simple_geomean_challenger_passed_step5_gate"
    return final, reason


def bootstrap_against_reference(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    reference_candidate_id: str,
    reference_calibration_id: str,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    if iterations <= 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    combos = summary[["candidate_id", "calibration_id"]].drop_duplicates()
    ref_part = predictions.loc[
        (predictions["candidate_id"] == reference_candidate_id)
        & (predictions["calibration_id"] == reference_calibration_id),
        METADATA_COLUMNS + ["calibrated_score"],
    ].rename(columns={"calibrated_score": "reference_score"})
    if ref_part.empty:
        return pd.DataFrame()
    group_codes, groups = pd.factorize(ref_part["group_id"], sort=False)
    n_groups = len(groups)
    y_true = ref_part["y_true"].to_numpy(dtype=int)
    reference_score = ref_part["reference_score"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    for _, combo in combos.iterrows():
        candidate_id = str(combo["candidate_id"])
        calibration_id = str(combo["calibration_id"])
        part = predictions.loc[
            (predictions["candidate_id"] == candidate_id)
            & (predictions["calibration_id"] == calibration_id),
            ["sample_id", "calibrated_score"],
        ].rename(columns={"calibrated_score": "candidate_score"})
        merged = ref_part[["sample_id"]].merge(part, on="sample_id", how="inner", validate="one_to_one")
        if len(merged) != len(ref_part):
            continue
        candidate_score = merged["candidate_score"].to_numpy(dtype=float)
        deltas = np.empty(iterations, dtype=float)
        for iteration in range(iterations):
            counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
            sample_weight = counts[group_codes]
            deltas[iteration] = (
                step2.safe_average_precision(y_true, candidate_score)
                if sample_weight.sum() == 0
                else average_precision_score(
                    y_true,
                    candidate_score,
                    sample_weight=sample_weight,
                )
            ) - average_precision_score(
                y_true,
                reference_score,
                sample_weight=sample_weight,
            )
        rows.append(
            {
                "candidate_id": candidate_id,
                "calibration_id": calibration_id,
                "reference_candidate_id": reference_candidate_id,
                "reference_calibration_id": reference_calibration_id,
                "iterations": int(iterations),
                "delta_auprc_mean": float(deltas.mean()),
                "delta_auprc_ci_low": float(np.quantile(deltas, 0.025)),
                "delta_auprc_ci_high": float(np.quantile(deltas, 0.975)),
                "probability_delta_gt_zero": float((deltas > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def fit_full_selected_calibrator(
    frame: pd.DataFrame,
    *,
    calibration_id: str,
) -> tuple[FittedCalibrator, pd.DataFrame]:
    calibrator = fit_calibrator(frame["y_true"], frame["raw_score"], calibration_id)
    final_score = apply_calibrator(calibrator, frame["raw_score"])
    final_oof = frame.copy()
    final_oof["calibration_id"] = calibration_id
    final_oof["final_calibrated_score"] = final_score
    final_oof["score_source"] = "full_oof_final_calibrator"
    return calibrator, final_oof


def threshold_json_payload(
    final_oof: pd.DataFrame,
    *,
    score_col: str,
    threshold_source: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    thresholds = select_thresholds_step5(final_oof["y_true"], final_oof[score_col])
    payload: dict[str, Any] = {
        "threshold_source": threshold_source,
        "candidate_id": str(final_oof.iloc[0]["candidate_id"]),
        "calibration_id": str(final_oof.iloc[0]["calibration_id"]),
        "score_col": score_col,
        "operating_points": {},
    }
    for operating_point, threshold in thresholds.items():
        metrics = step2.classification_metrics_at_threshold(
            final_oof["y_true"],
            final_oof[score_col],
            threshold,
        )
        row = {
            "candidate_id": payload["candidate_id"],
            "calibration_id": payload["calibration_id"],
            "operating_point": operating_point,
            "threshold_source": threshold_source,
            **metrics,
        }
        rows.append(row)
        payload["operating_points"][operating_point] = {
            key: (int(value) if isinstance(value, (np.integer,)) else float(value) if isinstance(value, (np.floating,)) else value)
            for key, value in row.items()
            if key not in {"candidate_id", "calibration_id", "operating_point", "threshold_source"}
        }
    return payload, pd.DataFrame(rows)


def write_individual_oof(predictions: pd.DataFrame, output_dir: Path) -> None:
    for (candidate_id, calibration_id), part in predictions.groupby(
        ["candidate_id", "calibration_id"],
        sort=False,
    ):
        path = output_dir / f"oof__{candidate_id}__{calibration_id}.csv"
        part[
            METADATA_COLUMNS
            + [
                "candidate_id",
                "candidate_role",
                "calibration_id",
                "raw_score",
                "calibrated_score",
                "score_source",
            ]
        ].to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    candidate_ids = select_from_csv_argument(
        args.candidate_ids,
        DEFAULT_CANDIDATE_IDS,
        argument_name="--candidate-ids",
    )
    calibration_methods = select_from_csv_argument(
        args.calibration_methods,
        DEFAULT_CALIBRATION_METHODS,
        argument_name="--calibration-methods",
    )
    input_dir, output_dir = project_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_candidates(input_dir, candidate_ids)
    input_manifest = candidate_input_manifest(frames)
    input_validation = validate_candidate_inputs(frames)
    expected_n = int(input_manifest["n"].iloc[0]) if not input_manifest.empty else 0

    run_manifest: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "started_at": step2.timestamp(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "candidate_ids": candidate_ids,
        "calibration_methods": calibration_methods,
        "expected_n": expected_n,
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_seed": args.random_seed,
        "lockbox_accessed": False,
    }
    input_manifest.to_csv(output_dir / "candidate_input_manifest.csv", index=False, encoding="utf-8-sig")
    input_validation.to_csv(output_dir / "validation_checks__step5_input.csv", index=False, encoding="utf-8-sig")
    step2.write_json(
        output_dir / ("run_manifest__check_config.json" if args.check_config else "run_manifest__step5.json"),
        run_manifest,
    )
    step2.log(
        f"Step5 구성: candidates={len(candidate_ids)}, calibration_methods={calibration_methods}, "
        f"n={expected_n:,}, lockbox_accessed=False"
    )
    if not bool(input_validation["passed"].all()):
        failed = input_validation.loc[~input_validation["passed"]]
        raise RuntimeError(
            "Step5 입력 검증 실패: "
            + failed.to_json(orient="records", force_ascii=False)
        )
    if args.check_config:
        step2.log("--check-config 지정: calibration 없이 종료합니다.")
        return

    predictions, threshold_audit, calibration_fit_metrics = cross_calibrate_all(
        frames,
        calibration_methods,
        progress_bar=(not args.no_progress_bar and step2.tqdm is not None),
    )
    calibrated_validation = validation_rows_for_calibrated(predictions, expected_n)
    validation_checks = pd.concat([input_validation, calibrated_validation], ignore_index=True)
    if not bool(validation_checks["passed"].all()):
        failed = validation_checks.loc[~validation_checks["passed"]]
        raise RuntimeError(
            "Step5 validation 실패: "
            + failed.to_json(orient="records", force_ascii=False)
        )

    folds = fold_metric_rows(predictions, score_col="calibrated_score")
    thresholds = threshold_rows(
        predictions,
        score_col="calibrated_score",
        threshold_source="cross_calibrated_oof",
    )
    top_risk = top_risk_rows(predictions, score_col="calibrated_score")
    subgroups = subgroup_rows(predictions, score_col="calibrated_score")
    calibration_bins = calibration_bins_for_predictions(predictions, score_col="calibrated_score")
    summary = summary_rows(
        predictions,
        folds,
        calibration_bins,
        score_col="calibrated_score",
    )
    summary = add_operating_metrics(summary, thresholds, top_risk)
    summary = add_calibration_selection_flags(summary)
    final_row, selection_reason = choose_final_score(summary)
    summary["selected_probability_score"] = (
        (summary["candidate_id"] == final_row["candidate_id"])
        & (summary["calibration_id"] == final_row["calibration_id"])
    )

    bootstrap = bootstrap_against_reference(
        predictions,
        summary,
        reference_candidate_id=str(final_row["candidate_id"]),
        reference_calibration_id=str(final_row["calibration_id"]),
        iterations=args.bootstrap_iterations,
        random_state=args.random_seed,
    )

    selected_frame = frames[str(final_row["candidate_id"])]
    selected_calibrator, final_oof = fit_full_selected_calibrator(
        selected_frame,
        calibration_id=str(final_row["calibration_id"]),
    )
    final_threshold_payload, final_threshold_rows = threshold_json_payload(
        final_oof,
        score_col="final_calibrated_score",
        threshold_source="full_oof_final_calibrator",
    )

    calibrator_artifact_path = output_dir / "calibrator__selected_probability_score.pkl"
    with calibrator_artifact_path.open("wb") as file:
        pickle.dump(
            {
                "candidate_id": str(final_row["candidate_id"]),
                "calibration_id": str(final_row["calibration_id"]),
                "calibrator_model": selected_calibrator.model,
                "feature_transform": selected_calibrator.feature_transform,
                "clip_epsilon": CLIP_EPSILON,
                "score_input": "raw_score_from_step4_candidate",
            },
            file,
        )

    selected_payload = {
        "selected_candidate_id": str(final_row["candidate_id"]),
        "selected_calibration_id": str(final_row["calibration_id"]),
        "selection_reason": selection_reason,
        "score_source_for_selection": "cross_calibrated_oof",
        "final_calibrator_source": "fit_on_full_development_oof",
        "calibrator_artifact_path": str(calibrator_artifact_path),
        "metrics": {
            "auprc": float(final_row["auprc"]),
            "auroc": float(final_row["auroc"]),
            "brier": float(final_row["brier"]),
            "log_loss": float(final_row["log_loss"]),
            "target0a_auprc": float(final_row["target0a_auprc"]),
            "ece_equal_width": float(final_row["ece_equal_width"]),
            "calibration_intercept": float(final_row["calibration_intercept"]),
            "calibration_slope": float(final_row["calibration_slope"]),
            "recall70_precision": float(final_row["recall70_precision"]),
            "recall70_selected_rate": float(final_row["recall70_selected_rate"]),
            "top10_capture": float(final_row["top10_capture"]),
            "top20_capture": float(final_row["top20_capture"]),
        },
        "lockbox_accessed": False,
        "next_stage": "Step6 lockbox first evaluation",
    }

    final_candidate_manifest = {
        "selected_probability_score_path": str(output_dir / "selected_probability_score.json"),
        "final_thresholds_path": str(output_dir / "final_thresholds.json"),
        "calibrator_artifact_path": str(calibrator_artifact_path),
        "selected_candidate_id": selected_payload["selected_candidate_id"],
        "selected_calibration_id": selected_payload["selected_calibration_id"],
        "step4_oof_input_file": str(
            input_dir / CANDIDATE_SPECS[selected_payload["selected_candidate_id"]].input_filename
        ),
        "threshold_source": final_threshold_payload["threshold_source"],
        "do_not_reselect_or_retune_on_lockbox": True,
        "lockbox_accessed_in_step5": False,
    }

    calibration_manifest = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "calibration_id": method,
                "fit_scope": "outer_train_folds_for_cross_calibrated_oof",
                "final_refit_scope": (
                    "full_development_oof"
                    if candidate_id == selected_payload["selected_candidate_id"]
                    and method == selected_payload["selected_calibration_id"]
                    else ""
                ),
                "clip_epsilon": CLIP_EPSILON,
                "lockbox_accessed": False,
            }
            for candidate_id in candidate_ids
            for method in calibration_methods
        ]
    )

    write_individual_oof(predictions, output_dir)
    predictions.to_csv(output_dir / "all_calibrated_oof_predictions.csv", index=False, encoding="utf-8-sig")
    calibration_manifest.to_csv(output_dir / "calibration_manifest.csv", index=False, encoding="utf-8-sig")
    calibration_fit_metrics.to_csv(output_dir / "calibration_fit_metrics__step5.csv", index=False, encoding="utf-8-sig")
    summary.sort_values(
        ["selected_probability_score", "selected_for_candidate", "auprc"],
        ascending=[False, False, False],
    ).to_csv(output_dir / "summary__step5_calibration.csv", index=False, encoding="utf-8-sig")
    calibration_bins.to_csv(output_dir / "calibration_bins__step5.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(output_dir / "threshold_metrics__step5.csv", index=False, encoding="utf-8-sig")
    threshold_audit.to_csv(
        output_dir / "threshold_audit_by_outer_fold__step5.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_threshold_rows.to_csv(
        output_dir / "threshold_metrics__selected_final_calibrator_step5.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds.to_csv(output_dir / "fold_metrics__step5_calibration.csv", index=False, encoding="utf-8-sig")
    top_risk.to_csv(output_dir / "top_risk_metrics__step5.csv", index=False, encoding="utf-8-sig")
    subgroups.to_csv(output_dir / "subgroup_metrics__step5.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_dir / "bootstrap__step5_calibration.csv", index=False, encoding="utf-8-sig")
    validation_checks.to_csv(output_dir / "validation_checks__step5.csv", index=False, encoding="utf-8-sig")
    final_oof.to_csv(output_dir / "final_oof__selected_probability_score.csv", index=False, encoding="utf-8-sig")
    step2.write_json(output_dir / "selected_probability_score.json", selected_payload)
    step2.write_json(output_dir / "final_thresholds.json", final_threshold_payload)
    step2.write_json(output_dir / "final_candidate_manifest.json", final_candidate_manifest)

    run_manifest.update(
        {
            "finished_at": step2.timestamp(),
            "elapsed_seconds": time.perf_counter() - started,
            "selected_candidate_id": selected_payload["selected_candidate_id"],
            "selected_calibration_id": selected_payload["selected_calibration_id"],
            "final_thresholds_path": str(output_dir / "final_thresholds.json"),
            "actual_candidate_calibration_count": int(summary.shape[0]),
        }
    )
    step2.write_json(output_dir / "run_manifest__step5.json", run_manifest)
    step2.log(
        "STEP5 DONE | "
        f"selected={selected_payload['selected_candidate_id']} / "
        f"{selected_payload['selected_calibration_id']} | "
        f"thresholds={output_dir / 'final_thresholds.json'} | "
        f"elapsed={step2.format_seconds(run_manifest['elapsed_seconds'])}"
    )


if __name__ == "__main__":
    main()
