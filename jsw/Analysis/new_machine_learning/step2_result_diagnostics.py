from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
METADATA_COLUMNS = [
    "sample_id",
    "outer_fold",
    "y_true",
    "sample_type",
    "climate_type",
    "group_id",
]


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


def parse_args() -> argparse.Namespace:
    root = find_project_root()
    default_output = root / "jsw" / "Analysis" / "new_machine_learning" / "outputs" / "step2_tuned_single_models"
    parser = argparse.ArgumentParser(description="Step2 OOF diversity and ensemble-base diagnostics")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def load_oof_matrix(output_dir: Path, candidate_ids: list[str]) -> pd.DataFrame:
    matrix: pd.DataFrame | None = None
    for candidate_id in candidate_ids:
        path = output_dir / f"oof__{candidate_id}.csv"
        oof = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = set(METADATA_COLUMNS + ["y_prob"])
        missing = sorted(required - set(oof.columns))
        if missing:
            raise KeyError(f"{path.name} 필수 열 누락: {missing}")
        current = oof[METADATA_COLUMNS + ["y_prob"]].copy()
        current = current.rename(columns={"y_prob": f"pred_{candidate_id}"})
        if matrix is None:
            matrix = current
            continue
        matrix = matrix.merge(
            current,
            on=METADATA_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    if matrix is None:
        raise ValueError("OOF 후보가 없습니다.")
    if matrix["sample_id"].duplicated().any():
        raise ValueError("통합 OOF matrix에 중복 sample_id가 있습니다.")
    return matrix.sort_values("sample_id").reset_index(drop=True)


def make_correlation_rows(matrix: pd.DataFrame, candidate_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate_a, candidate_b in itertools.combinations(candidate_ids, 2):
        pred_a = matrix[f"pred_{candidate_a}"]
        pred_b = matrix[f"pred_{candidate_b}"]
        rows.append(
            {
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "pearson": float(pred_a.corr(pred_b, method="pearson")),
                "spearman": float(pred_a.corr(pred_b, method="spearman")),
            }
        )
    return rows


def top_mask(values: pd.Series, fraction: float) -> np.ndarray:
    selected_n = max(1, int(np.ceil(len(values) * fraction)))
    order = np.argsort(-values.to_numpy(dtype=float), kind="stable")
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:selected_n]] = True
    return mask


def make_overlap_rows(matrix: pd.DataFrame, candidate_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_true = matrix["y_true"].to_numpy(dtype=int)
    for fraction in TOP_FRACTIONS:
        masks = {
            candidate_id: top_mask(matrix[f"pred_{candidate_id}"], fraction)
            for candidate_id in candidate_ids
        }
        for candidate_a, candidate_b in itertools.combinations(candidate_ids, 2):
            mask_a = masks[candidate_a]
            mask_b = masks[candidate_b]
            intersection = mask_a & mask_b
            union = mask_a | mask_b
            positive_a = mask_a & (y_true == 1)
            positive_b = mask_b & (y_true == 1)
            positive_union = positive_a | positive_b
            rows.append(
                {
                    "top_fraction": fraction,
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "selected_intersection_n": int(intersection.sum()),
                    "selected_union_n": int(union.sum()),
                    "selected_jaccard": float(intersection.sum() / union.sum()),
                    "positive_capture_a_n": int(positive_a.sum()),
                    "positive_capture_b_n": int(positive_b.sum()),
                    "positive_capture_union_n": int(positive_union.sum()),
                    "positive_union_gain_vs_best_n": int(
                        positive_union.sum() - max(positive_a.sum(), positive_b.sum())
                    ),
                }
            )
    return rows


def make_champion_complement_rows(
    matrix: pd.DataFrame,
    candidate_ids: list[str],
    champion_id: str,
    threshold_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_true = matrix["y_true"].to_numpy(dtype=int)
    champion_prob = matrix[f"pred_{champion_id}"].to_numpy(dtype=float)

    for fraction in (0.10, 0.20):
        champion_mask = top_mask(pd.Series(champion_prob), fraction)
        for candidate_id in candidate_ids:
            candidate_mask = top_mask(matrix[f"pred_{candidate_id}"], fraction)
            rows.append(
                {
                    "comparison_type": f"top_{int(fraction * 100)}pct",
                    "candidate_id": candidate_id,
                    "champion_id": champion_id,
                    "candidate_positive_capture_n": int((candidate_mask & (y_true == 1)).sum()),
                    "champion_positive_capture_n": int((champion_mask & (y_true == 1)).sum()),
                    "candidate_additional_positive_n": int(
                        (candidate_mask & ~champion_mask & (y_true == 1)).sum()
                    ),
                    "champion_additional_positive_n": int(
                        (champion_mask & ~candidate_mask & (y_true == 1)).sum()
                    ),
                }
            )

    recall70 = threshold_frame.loc[threshold_frame["operating_point"] == "recall_ge_0.70"].set_index("candidate_id")
    champion_threshold = float(recall70.loc[champion_id, "threshold"])
    champion_mask = champion_prob >= champion_threshold
    for candidate_id in candidate_ids:
        candidate_threshold = float(recall70.loc[candidate_id, "threshold"])
        candidate_mask = matrix[f"pred_{candidate_id}"].to_numpy(dtype=float) >= candidate_threshold
        rows.append(
            {
                "comparison_type": "recall_ge_0.70",
                "candidate_id": candidate_id,
                "champion_id": champion_id,
                "candidate_positive_capture_n": int((candidate_mask & (y_true == 1)).sum()),
                "champion_positive_capture_n": int((champion_mask & (y_true == 1)).sum()),
                "candidate_additional_positive_n": int(
                    (candidate_mask & ~champion_mask & (y_true == 1)).sum()
                ),
                "champion_additional_positive_n": int(
                    (champion_mask & ~candidate_mask & (y_true == 1)).sum()
                ),
            }
        )
    return rows


def make_candidate_diagnostics(
    output_dir: Path,
    summary: pd.DataFrame,
    matrix: pd.DataFrame,
    candidate_ids: list[str],
    champion_id: str,
) -> pd.DataFrame:
    threshold = pd.read_csv(
        output_dir / "threshold_metrics__step2_tuned_candidates.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    top_risk = pd.read_csv(
        output_dir / "top_risk_metrics__step2_tuned_candidates.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    subgroup = pd.read_csv(
        output_dir / "subgroup_metrics__step2_tuned_candidates.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )

    recall70 = threshold.loc[
        threshold["operating_point"] == "recall_ge_0.70",
        ["candidate_id", "threshold", "recall", "precision", "selected_rate", "fn"],
    ].rename(
        columns={
            "threshold": "recall70_threshold",
            "recall": "recall70_recall",
            "precision": "recall70_precision",
            "selected_rate": "recall70_selected_rate",
            "fn": "recall70_fn",
        }
    )
    recall90 = threshold.loc[
        threshold["operating_point"] == "recall_ge_0.90",
        ["candidate_id", "threshold", "recall", "precision", "selected_rate", "fn"],
    ].rename(
        columns={
            "threshold": "recall90_threshold",
            "recall": "recall90_recall",
            "precision": "recall90_precision",
            "selected_rate": "recall90_selected_rate",
            "fn": "recall90_fn",
        }
    )
    top10 = top_risk.loc[
        np.isclose(top_risk["top_fraction"], 0.10),
        ["candidate_id", "capture_rate", "precision", "lift"],
    ].rename(
        columns={
            "capture_rate": "top10_capture_rate",
            "precision": "top10_precision",
            "lift": "top10_lift",
        }
    )
    top20 = top_risk.loc[
        np.isclose(top_risk["top_fraction"], 0.20),
        ["candidate_id", "capture_rate", "precision", "lift"],
    ].rename(
        columns={
            "capture_rate": "top20_capture_rate",
            "precision": "top20_precision",
            "lift": "top20_lift",
        }
    )
    hard_negative = subgroup.loc[
        (subgroup["subgroup_type"] == "sample_type_pair")
        & (subgroup["subgroup_value"] == "Target_1_vs_Target_0A"),
        ["candidate_id", "auprc", "auroc"],
    ].rename(columns={"auprc": "target0a_auprc", "auroc": "target0a_auroc"})

    diagnostics = summary.merge(recall70, on="candidate_id", how="left")
    diagnostics = diagnostics.merge(recall90, on="candidate_id", how="left")
    diagnostics = diagnostics.merge(top10, on="candidate_id", how="left")
    diagnostics = diagnostics.merge(top20, on="candidate_id", how="left")
    diagnostics = diagnostics.merge(hard_negative, on="candidate_id", how="left")
    champion_prob = matrix[f"pred_{champion_id}"]
    diagnostics["pearson_vs_champion"] = [
        float(matrix[f"pred_{candidate_id}"].corr(champion_prob, method="pearson"))
        for candidate_id in diagnostics["candidate_id"]
    ]
    diagnostics["spearman_vs_champion"] = [
        float(matrix[f"pred_{candidate_id}"].corr(champion_prob, method="spearman"))
        for candidate_id in diagnostics["candidate_id"]
    ]
    return diagnostics.sort_values("auprc", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    summary = pd.read_csv(
        output_dir / "summary__step2_tuned_candidates.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    summary = summary.loc[summary["status"] == "OK"].copy()
    candidate_ids = summary["candidate_id"].astype(str).tolist()
    if not candidate_ids:
        raise ValueError("성공한 Step2 후보가 없습니다.")
    champion_id = str(summary.sort_values("auprc", ascending=False).iloc[0]["candidate_id"])

    matrix = load_oof_matrix(output_dir, candidate_ids)
    matrix.to_csv(output_dir / "oof_matrix__step2_base_candidates.csv", index=False, encoding="utf-8-sig")

    correlation = pd.DataFrame(make_correlation_rows(matrix, candidate_ids))
    correlation.to_csv(
        output_dir / "diversity_correlation__step2_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overlap = pd.DataFrame(make_overlap_rows(matrix, candidate_ids))
    overlap.to_csv(
        output_dir / "diversity_top_risk_overlap__step2_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold = pd.read_csv(
        output_dir / "threshold_metrics__step2_tuned_candidates.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    complement = pd.DataFrame(
        make_champion_complement_rows(
            matrix,
            candidate_ids,
            champion_id,
            threshold,
        )
    )
    complement.to_csv(
        output_dir / "diversity_unique_capture__step2_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics = make_candidate_diagnostics(
        output_dir,
        summary,
        matrix,
        candidate_ids,
        champion_id,
    )
    diagnostics.to_csv(
        output_dir / "ensemble_base_candidate_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "champion_id": champion_id,
        "row_count": int(len(matrix)),
        "positive_n": int(matrix["y_true"].sum()),
        "duplicate_sample_id_n": int(matrix["sample_id"].duplicated().sum()),
        "nan_prediction_n": int(
            matrix[[f"pred_{candidate_id}" for candidate_id in candidate_ids]].isna().sum().sum()
        ),
    }
    (output_dir / "step2_diagnostics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
