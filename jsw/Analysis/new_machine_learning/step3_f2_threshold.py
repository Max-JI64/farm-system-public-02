from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import step2_tuned_single_models as step2


FINAL_CANDIDATE_ID = "TUNE_LGBM_ALL_ALL_LC_NONE"
REPORTED_RECALL_TARGETS = (0.70, 0.80, 0.90)
RECALL_GRID = tuple(np.round(np.arange(0.50, 1.00, 0.05), 2))
EXPECTED_ROWS = 17_045


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "new_machine_learning Step3: tuned LightGBM raw OOF score의 "
            "F2 threshold 및 recall 민감도 분석"
        ),
    )
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(native(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {
        "sample_id",
        "outer_fold",
        "y_true",
        "y_prob",
        "candidate_id",
        "sample_type",
        "climate_type",
        "group_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"LightGBM OOF 필수 열 누락: {missing}")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["y_prob"] = pd.to_numeric(frame["y_prob"], errors="coerce")
    candidate_ids = set(frame["candidate_id"].astype(str))
    if candidate_ids != {FINAL_CANDIDATE_ID}:
        raise ValueError(f"예상하지 않은 candidate_id: {sorted(candidate_ids)}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def select_recall_threshold(curve: pd.DataFrame, target: float) -> float:
    candidates = curve.loc[curve["recall"] >= target]
    if candidates.empty:
        return 0.0
    selected = candidates.sort_values(
        ["precision", "selected_rate", "threshold"],
        ascending=[False, True, False],
    ).iloc[0]
    return float(selected["threshold"])


def select_operating_thresholds(
    y_true: pd.Series,
    score: pd.Series,
) -> dict[str, float]:
    curve = step2.threshold_curve(y_true, score)
    if curve.empty:
        raise RuntimeError("threshold curve가 비어 있습니다.")
    best_f2 = curve.sort_values(
        ["f2", "recall", "selected_rate", "threshold"],
        ascending=[False, False, True, False],
    ).iloc[0]
    thresholds = {"best_f2": float(best_f2["threshold"])}
    for target in REPORTED_RECALL_TARGETS:
        thresholds[f"recall_ge_{target:.2f}"] = select_recall_threshold(
            curve,
            target,
        )
    return thresholds


def operating_point_table(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    threshold_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for operating_point, threshold in thresholds.items():
        rows.append(
            {
                "candidate_id": FINAL_CANDIDATE_ID,
                "operating_point": operating_point,
                "threshold_source": threshold_source,
                **step2.classification_metrics_at_threshold(
                    frame["y_true"],
                    frame["y_prob"],
                    threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def recall_grid_table(frame: pd.DataFrame) -> pd.DataFrame:
    curve = step2.threshold_curve(frame["y_true"], frame["y_prob"])
    rows: list[dict[str, Any]] = []
    for target in RECALL_GRID:
        threshold = select_recall_threshold(curve, float(target))
        rows.append(
            {
                "recall_target": float(target),
                **step2.classification_metrics_at_threshold(
                    frame["y_true"],
                    frame["y_prob"],
                    threshold,
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("recall_target").reset_index(drop=True)
    result["additional_selected_n_vs_previous"] = result["selected_n"].diff()
    result["additional_tp_vs_previous"] = result["tp"].diff()
    result["additional_fp_vs_previous"] = result["fp"].diff()
    result["additional_fn_reduced_vs_previous"] = -result["fn"].diff()
    result["additional_fp_per_additional_tp"] = np.where(
        result["additional_tp_vs_previous"] > 0,
        result["additional_fp_vs_previous"]
        / result["additional_tp_vs_previous"],
        np.nan,
    )
    return result


def metrics_from_counts(part: pd.DataFrame) -> dict[str, float | int]:
    tp = int(part["tp"].sum())
    fp = int(part["fp"].sum())
    fn = int(part["fn"].sum())
    tn = int(part["tn"].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f2_denominator = 4 * precision + recall
    f2 = 5 * precision * recall / f2_denominator if f2_denominator > 0 else 0.0
    total = tp + fp + fn + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "f2": float(f2),
        "selected_n": tp + fp,
        "selected_rate": float((tp + fp) / total) if total else float("nan"),
        "fp_per_tp": float(fp / tp) if tp else float("inf"),
    }


def cross_fold_threshold_audit(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for outer_fold in sorted(frame["outer_fold"].unique()):
        train = frame.loc[frame["outer_fold"] != outer_fold]
        valid = frame.loc[frame["outer_fold"] == outer_fold]
        thresholds = select_operating_thresholds(train["y_true"], train["y_prob"])
        for operating_point, threshold in thresholds.items():
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "operating_point": operating_point,
                    "threshold_source": "other_outer_folds",
                    "train_n": int(len(train)),
                    "valid_n": int(len(valid)),
                    "train_valid_overlap_n": 0,
                    **step2.classification_metrics_at_threshold(
                        valid["y_true"],
                        valid["y_prob"],
                        threshold,
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    summary: dict[str, dict[str, Any]] = {}
    for operating_point, part in audit.groupby("operating_point", sort=False):
        summary[str(operating_point)] = {
            "fold_count": int(part["outer_fold"].nunique()),
            "threshold_min": float(part["threshold"].min()),
            "threshold_max": float(part["threshold"].max()),
            "fold_recall_min": float(part["recall"].min()),
            "fold_recall_max": float(part["recall"].max()),
            **metrics_from_counts(part),
        }
    return audit, summary


def subgroup_table(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for negative_type in ("Target_0A", "Target_0B1", "Target_0B2"):
        groups.append(
            (
                "sample_type_pair",
                f"Target_1_vs_{negative_type}",
                frame.loc[
                    frame["sample_type"].isin(["Target_1", negative_type])
                ],
            )
        )
    groups.extend(
        ("climate_type", str(climate_type), subset)
        for climate_type, subset in frame.groupby("climate_type")
    )
    for group_type, group_value, subset in groups:
        rows.append(
            {
                "subgroup_type": group_type,
                "subgroup_value": group_value,
                **step2.probability_metrics(subset["y_true"], subset["y_prob"]),
                **{
                    f"f2_threshold_{key}": value
                    for key, value in step2.classification_metrics_at_threshold(
                        subset["y_true"],
                        subset["y_prob"],
                        threshold,
                    ).items()
                },
            }
        )
    return pd.DataFrame(rows)


def validation_table(
    frame: pd.DataFrame,
    operating_points: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    confusion_total = int(
        operating_points.loc[
            operating_points["operating_point"] == "best_f2",
            ["tp", "fp", "fn", "tn"],
        ]
        .sum(axis=1)
        .iloc[0]
    )
    checks = [
        ("row_count", int(len(frame)), EXPECTED_ROWS),
        ("unique_sample_id", int(frame["sample_id"].nunique()), EXPECTED_ROWS),
        ("duplicate_sample_id", int(frame["sample_id"].duplicated().sum()), 0),
        ("nan_score_n", int(frame["y_prob"].isna().sum()), 0),
        ("outer_fold_n", int(frame["outer_fold"].nunique()), 5),
        (
            "score_out_of_bounds_n",
            int(((frame["y_prob"] < 0) | (frame["y_prob"] > 1)).sum()),
            0,
        ),
        ("full_oof_confusion_total", confusion_total, EXPECTED_ROWS),
        (
            "audit_train_valid_overlap_n",
            int(audit["train_valid_overlap_n"].sum()),
            0,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check": check,
                "observed": observed,
                "expected": expected,
                "passed": observed == expected,
            }
            for check, observed, expected in checks
        ]
    )


def final_oof_table(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = frame.copy()
    result["risk_score"] = result["y_prob"]
    result["pred_f2"] = (result["risk_score"] >= threshold).astype(int)
    result["error_group"] = np.select(
        [
            (result["y_true"] == 1) & (result["pred_f2"] == 1),
            (result["y_true"] == 1) & (result["pred_f2"] == 0),
            (result["y_true"] == 0) & (result["pred_f2"] == 1),
        ],
        ["TP", "FN", "FP"],
        default="TN",
    )
    return result


def write_threshold_plots(
    curve: pd.DataFrame,
    operating_points: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_f2 = operating_points.loc[
        operating_points["operating_point"] == "best_f2"
    ].iloc[0]
    recall_70 = operating_points.loc[
        operating_points["operating_point"] == "recall_ge_0.70"
    ].iloc[0]
    recall_80 = operating_points.loc[
        operating_points["operating_point"] == "recall_ge_0.80"
    ].iloc[0]
    recall_90 = operating_points.loc[
        operating_points["operating_point"] == "recall_ge_0.90"
    ].iloc[0]

    curve_by_selected = curve.sort_values("selected_rate").reset_index(drop=True)
    created: list[str] = []

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(
        curve_by_selected["selected_rate"],
        curve_by_selected["f2"],
        color="#1f77b4",
        linewidth=2.5,
        label="F2",
    )
    ax.plot(
        curve_by_selected["selected_rate"],
        curve_by_selected["recall"],
        color="#2ca02c",
        linewidth=2.0,
        alpha=0.9,
        label="Recall",
    )
    ax.plot(
        curve_by_selected["selected_rate"],
        curve_by_selected["precision"],
        color="#ff7f0e",
        linewidth=2.0,
        alpha=0.9,
        label="Precision",
    )
    ax.axvline(
        float(best_f2["selected_rate"]),
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label="Best F2 point",
    )
    ax.scatter(
        [float(best_f2["selected_rate"])],
        [float(best_f2["f2"])],
        color="#d62728",
        s=55,
        zorder=5,
    )
    ax.annotate(
        f"Best F2\nselected {float(best_f2['selected_rate']):.1%}\nthreshold {float(best_f2['threshold']):.4f}",
        xy=(float(best_f2["selected_rate"]), float(best_f2["f2"])),
        xytext=(0.215, 0.805),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2},
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d62728", "alpha": 0.9},
    )
    for row, label, color in [
        (recall_70, "R70", "#9467bd"),
        (recall_80, "R80", "#8c564b"),
        (recall_90, "R90", "#7f7f7f"),
    ]:
        ax.scatter(
            [float(row["selected_rate"])],
            [float(row["f2"])],
            color=color,
            s=40,
            zorder=5,
        )
        ax.text(
            float(row["selected_rate"]),
            float(row["f2"]) + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )
    ax.set_title("Threshold trade-off by selected rate (tuned LightGBM OOF)")
    ax.set_xlabel("Selected rate")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0, min(0.50, max(0.36, float(curve_by_selected["selected_rate"].max()))))
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.legend(loc="lower right", frameon=True)
    selected_rate_path = plot_dir / "threshold_tradeoff_by_selected_rate__step3.png"
    fig.savefig(selected_rate_path, dpi=300)
    plt.close(fig)
    created.append(str(selected_rate_path))

    positive_threshold_curve = curve.loc[curve["threshold"] > 0].sort_values(
        "threshold"
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(
        positive_threshold_curve["threshold"],
        positive_threshold_curve["f2"],
        color="#1f77b4",
        linewidth=2.5,
        label="F2",
    )
    ax.axvline(
        float(best_f2["threshold"]),
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label="Best F2 threshold",
    )
    ax.scatter(
        [float(best_f2["threshold"])],
        [float(best_f2["f2"])],
        color="#d62728",
        s=55,
        zorder=5,
    )
    for row, label, color in [
        (recall_70, "Recall 70%", "#9467bd"),
        (recall_80, "Recall 80%", "#8c564b"),
        (recall_90, "Recall 90%", "#7f7f7f"),
    ]:
        ax.scatter(
            [float(row["threshold"])],
            [float(row["f2"])],
            color=color,
            s=40,
            zorder=5,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_title("F2 curve by raw score threshold (log scale)")
    ax.set_xlabel("Raw LightGBM score threshold")
    ax.set_ylabel("F2")
    ax.set_ylim(0, min(0.72, float(positive_threshold_curve["f2"].max()) + 0.06))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, which="both", color="#dddddd", linewidth=0.8)
    ax.legend(loc="lower center", ncol=2, frameon=True)
    threshold_path = plot_dir / "f2_by_raw_threshold_log_scale__step3.png"
    fig.savefig(threshold_path, dpi=300)
    plt.close(fig)
    created.append(str(threshold_path))

    return created


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    root = step2.find_project_root()
    analysis_dir = root / "jsw" / "Analysis" / "new_machine_learning"
    oof_path = args.oof or (
        analysis_dir
        / "outputs"
        / "step2_tuned_single_models"
        / "oof__TUNE_LGBM_ALL_ALL_LC_NONE.csv"
    )
    output_dir = args.output_dir or (
        analysis_dir / "outputs" / "step3_f2_threshold"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_oof(oof_path)
    thresholds = select_operating_thresholds(frame["y_true"], frame["y_prob"])
    operating_points = operating_point_table(
        frame,
        thresholds,
        threshold_source="full_development_oof",
    )
    recall_grid = recall_grid_table(frame)
    audit, audit_summary = cross_fold_threshold_audit(frame)
    best_f2_threshold = thresholds["best_f2"]
    subgroups = subgroup_table(frame, best_f2_threshold)
    top_risk = pd.DataFrame(step2.make_top_risk_rows(frame))
    validations = validation_table(frame, operating_points, audit)
    final_oof = final_oof_table(frame, best_f2_threshold)

    threshold_curve = step2.threshold_curve(frame["y_true"], frame["y_prob"])
    threshold_curve.to_csv(
        output_dir / "threshold_curve__step3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_paths = write_threshold_plots(threshold_curve, operating_points, output_dir)
    operating_points.to_csv(
        output_dir / "threshold_metrics__step3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recall_grid.to_csv(
        output_dir / "recall_target_comparison__step3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        output_dir / "threshold_audit_by_outer_fold__step3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subgroups.to_csv(
        output_dir / "subgroup_metrics__step3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top_risk.to_csv(
        output_dir / "top_risk_metrics__step3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validations.to_csv(
        output_dir / "validation_checks__step3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_oof.to_csv(
        output_dir / "final_oof__lightgbm_raw_score.csv",
        index=False,
        encoding="utf-8-sig",
    )

    operating_payload = {
        row["operating_point"]: {
            key: value
            for key, value in row.items()
            if key not in {"candidate_id", "operating_point", "threshold_source"}
        }
        for row in operating_points.to_dict(orient="records")
    }
    write_json(
        output_dir / "final_thresholds.json",
        {
            "candidate_id": FINAL_CANDIDATE_ID,
            "model": "tuned LightGBM",
            "score": "raw OOF model score",
            "calibration_applied": False,
            "probability_interpretation_allowed": False,
            "primary_operating_point": "best_f2",
            "sensitivity_operating_points": [
                "recall_ge_0.70",
                "recall_ge_0.80",
                "recall_ge_0.90",
            ],
            "operating_points": operating_payload,
            "cross_validation_audit": audit_summary,
            "top_risk_capture": {
                f"top_{int(round(row['top_fraction'] * 100))}pct": {
                    "selected_n": int(row["selected_n"]),
                    "positive_captured_n": int(row["positive_captured_n"]),
                    "total_positive_n": int(row["total_positive_n"]),
                    "capture_rate": float(row["capture_rate"]),
                    "precision": float(row["precision"]),
                    "lift": float(row["lift"]),
                }
                for row in top_risk.to_dict(orient="records")
            },
            "selection_note": (
                "F2 최대점을 기본 운영점으로 사용한다. recall 70/80/90%는 "
                "precision, selected rate, FP, FN과 경보 부담을 비교하는 민감도 분석이다."
            ),
            "score_warning": (
                "calibration을 수행하지 않았으므로 risk_score를 실제 산불 발생확률로 "
                "해석하지 않는다."
            ),
            "plots": [str(Path(path).resolve()) for path in plot_paths],
        },
    )
    write_json(
        output_dir / "run_manifest__step3.json",
        {
            "script": str(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "oof_path": str(oof_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "n_rows": int(len(frame)),
            "positive_n": int(frame["y_true"].sum()),
            "outer_fold_n": int(frame["outer_fold"].nunique()),
            "elapsed_seconds": time.perf_counter() - started,
            "validation_passed": bool(validations["passed"].all()),
        },
    )
    if not validations["passed"].all():
        failed = validations.loc[~validations["passed"], "check"].tolist()
        raise RuntimeError(f"STEP 3 검증 실패: {failed}")
    step2.log(
        "Step 3 완료 | "
        f"best_f2_threshold={best_f2_threshold:.8f} | "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
