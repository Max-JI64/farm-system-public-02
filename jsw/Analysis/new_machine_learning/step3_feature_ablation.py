from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import step1_single_models as step1
import step2_tuned_single_models as step2


DEFAULT_BASE_CANDIDATE_ID = "TUNE_LGBM_ALL_ALL_LC_NONE"
STEP1_RF_CANDIDATE_ID = "STEP1_RF_CORE_ALL_LC_BAL"

CORE_ABLATION_IDS = [
    "DROP_CANADA_ALL",
    "DROP_LANDCOVER",
    "DROP_WEATHER",
    "DROP_TERRAIN_DISTANCE",
    "DROP_SEASON_TIME",
]
CANADA_SENSITIVITY_IDS = [
    "CANADA_FWI_ONLY",
    "CANADA_FFMC_FWI",
    "CANADA_CORE_EDA",
    "CANADA_LONG_TERM_SENSITIVITY",
    "CANADA_DROP_FFMC",
    "CANADA_DROP_FFMC_10D_MEAN",
    "CANADA_DROP_DMC",
    "CANADA_DROP_DC",
    "CANADA_DROP_ISI",
    "CANADA_DROP_BUI",
    "CANADA_DROP_FWI",
]
ALL_ABLATION_IDS = CORE_ABLATION_IDS + CANADA_SENSITIVITY_IDS

CANADA_COLUMNS = ["FFMC", "FFMC_10일평균", "DMC", "DC", "ISI", "BUI", "FWI"]
LANDCOVER_COLUMNS = [
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
WEATHER_COLUMNS = [
    "시점_습도_pct",
    "직전24h_평균습도",
    "직전24h_최소습도",
    "직전48h_평균습도",
    "직전48h_최소습도",
    "D-1_최소습도_pct",
    "D-1_평균습도_pct",
    "D-2_최소습도_pct",
    "D-3_최소습도_pct",
    "직전24h_강수량합",
    "직전48h_강수량합",
    "D-1_강수량합_mm",
    "시점_기온_C",
    "시점_풍속_m_s",
    "직전24h_평균풍속",
    "직전24h_최대풍속",
    "직전48h_평균풍속",
    "직전48h_최대풍속",
    "직전24h_평균기온_C",
    "풍향_sin",
    "풍향_cos",
    "서풍계열_여부",
    "시점_현지기압_hPa",
    "시점_해면기압_hPa",
    "기압변동_3h",
]
TERRAIN_DISTANCE_COLUMNS = [
    "log1p_도로_최단거리_m",
    "log1p_시가화거리_m",
    "log1p_산림지역_최단거리_m",
    "고도(m)",
    "경사도(도)",
    "TPI(지형위치지수)",
    "log1p_농업거리_m",
    "log1p_임도_최단거리_m",
    "log1p_등산로거리_m",
    "사면방향_sin",
    "사면방향_cos",
]
SEASON_TIME_COLUMNS = ["월_sin", "월_cos", "시간_sin", "시간_cos"]


@dataclass(frozen=True)
class BaseModelSpec:
    base_candidate_id: str
    source_stage: str
    model_name: str
    model_label: str
    feature_set: str
    imbalance_option: str
    baseline_oof_path: Path
    selected_params_path: Path | None


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    features: list[str]
    removed_features: list[str]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="new_machine_learning Step3: selected-model feature ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="", help="입력 CSV 경로")
    parser.add_argument("--step1-output-dir", type=str, default="", help="Step1 결과 폴더")
    parser.add_argument("--step2-output-dir", type=str, default="", help="Step2 결과 폴더")
    parser.add_argument("--output-dir", type=str, default="", help="Step3 결과 폴더")
    parser.add_argument(
        "--base-candidate-ids",
        nargs="*",
        default=[DEFAULT_BASE_CANDIDATE_ID],
        help="ablation할 base candidate. 기본은 tuned LightGBM",
    )
    parser.add_argument(
        "--ablation-ids",
        nargs="*",
        default=["all"],
        help="all, core, canada 또는 개별 ablation_id",
    )
    parser.add_argument("--max-outer-folds", type=int, default=0, help="검증용 outer fold 제한. 0이면 전체 5-fold")
    parser.add_argument("--check-config", action="store_true", help="구성과 예상 fit 수만 점검하고 종료")
    parser.add_argument("--overwrite", action="store_true", help="완료된 ablation OOF 파일도 다시 계산")
    parser.add_argument("--no-progress-bar", action="store_true", help="tqdm 진행 막대를 사용하지 않음")
    parser.add_argument("--random-state", type=int, default=20260622)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def parse_selection(values: list[str], allowed: list[str], *, argument_name: str) -> list[str]:
    selected: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in allowed:
                raise ValueError(f"{argument_name} 값이 잘못됐습니다: {token}. 허용값: {allowed}")
            selected.append(token)
    return step2.dedupe(selected)


def parse_ablation_ids(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(token.strip() for token in value.split(",") if token.strip())
    if not tokens or tokens == ["all"]:
        return ALL_ABLATION_IDS
    expanded: list[str] = []
    for token in tokens:
        if token == "all":
            expanded.extend(ALL_ABLATION_IDS)
        elif token == "core":
            expanded.extend(CORE_ABLATION_IDS)
        elif token == "canada":
            expanded.extend(CANADA_SENSITIVITY_IDS)
        elif token in ALL_ABLATION_IDS:
            expanded.append(token)
        else:
            raise ValueError(
                f"--ablation-ids 값이 잘못됐습니다: {token}. "
                f"허용값: all, core, canada, {ALL_ABLATION_IDS}"
            )
    return step2.dedupe(expanded)


def build_base_model_specs(
    *,
    step1_output_dir: Path,
    step2_output_dir: Path,
) -> dict[str, BaseModelSpec]:
    specs: dict[str, BaseModelSpec] = {}
    for candidate in step2.build_candidates().values():
        specs[candidate.candidate_id] = BaseModelSpec(
            base_candidate_id=candidate.candidate_id,
            source_stage="step2",
            model_name=candidate.model_name,
            model_label=candidate.model_label,
            feature_set=candidate.feature_set,
            imbalance_option=candidate.imbalance_option,
            baseline_oof_path=step2_output_dir / f"oof__{candidate.candidate_id}.csv",
            selected_params_path=step2_output_dir / f"selected_params__{candidate.candidate_id}.json",
        )

    specs[STEP1_RF_CANDIDATE_ID] = BaseModelSpec(
        base_candidate_id=STEP1_RF_CANDIDATE_ID,
        source_stage="step1",
        model_name="random_forest",
        model_label="RandomForest",
        feature_set="WS_CORE__CANADA_ALL__LC_USED",
        imbalance_option="balanced",
        baseline_oof_path=(
            step1_output_dir
            / "oof__random_forest__WS_CORE__CANADA_ALL__LC_USED__balanced.csv"
        ),
        selected_params_path=None,
    )
    return specs


def remove_features(base_features: list[str], columns: list[str]) -> tuple[list[str], list[str]]:
    remove_set = set(columns)
    removed = [feature for feature in base_features if feature in remove_set]
    retained = [feature for feature in base_features if feature not in remove_set]
    return retained, removed


def replace_canada_features(
    base_features: list[str],
    selected_canada: list[str],
) -> tuple[list[str], list[str]]:
    retained, removed = remove_features(base_features, CANADA_COLUMNS)
    return step2.dedupe(retained + selected_canada), [
        feature for feature in removed if feature not in selected_canada
    ]


def build_ablation_specs(
    base_features: list[str],
    selected_ablation_ids: list[str],
) -> dict[str, AblationSpec]:
    specs: dict[str, AblationSpec] = {}

    direct_drops = {
        "DROP_CANADA_ALL": (CANADA_COLUMNS, "캐나다 산불지수 전체 제거"),
        "DROP_LANDCOVER": (LANDCOVER_COLUMNS, "토지피복 파생변수 전체 제거"),
        "DROP_WEATHER": (WEATHER_COLUMNS, "습도·강수·기온·바람·기압 변수 제거"),
        "DROP_TERRAIN_DISTANCE": (TERRAIN_DISTANCE_COLUMNS, "지형·거리·접근성 변수 제거"),
        "DROP_SEASON_TIME": (SEASON_TIME_COLUMNS, "월·시간 주기 변수 제거"),
    }
    for ablation_id, (columns, description) in direct_drops.items():
        features, removed = remove_features(base_features, columns)
        specs[ablation_id] = AblationSpec(ablation_id, features, removed, description)

    canada_variants = {
        "CANADA_FWI_ONLY": (["FWI"], "캐나다지수는 FWI만 유지"),
        "CANADA_FFMC_FWI": (["FFMC", "FWI"], "캐나다지수는 FFMC와 FWI만 유지"),
        "CANADA_CORE_EDA": (["FFMC", "ISI", "FWI"], "캐나다지수는 FFMC·ISI·FWI만 유지"),
        "CANADA_LONG_TERM_SENSITIVITY": (
            ["DMC", "DC", "BUI"],
            "캐나다지수는 장기 건조축 DMC·DC·BUI만 유지",
        ),
    }
    for ablation_id, (selected_canada, description) in canada_variants.items():
        features, removed = replace_canada_features(base_features, selected_canada)
        specs[ablation_id] = AblationSpec(ablation_id, features, removed, description)

    leave_one_out = {
        "CANADA_DROP_FFMC": "FFMC",
        "CANADA_DROP_FFMC_10D_MEAN": "FFMC_10일평균",
        "CANADA_DROP_DMC": "DMC",
        "CANADA_DROP_DC": "DC",
        "CANADA_DROP_ISI": "ISI",
        "CANADA_DROP_BUI": "BUI",
        "CANADA_DROP_FWI": "FWI",
    }
    for ablation_id, removed_canada in leave_one_out.items():
        features, removed = remove_features(base_features, [removed_canada])
        specs[ablation_id] = AblationSpec(
            ablation_id,
            features,
            removed,
            f"캐나다지수 leave-one-out: {removed_canada} 제거",
        )

    selected_specs = {ablation_id: specs[ablation_id] for ablation_id in selected_ablation_ids}
    for ablation_id, spec in selected_specs.items():
        if not spec.removed_features:
            raise ValueError(
                f"{ablation_id}: base feature set에서 제거되거나 교체되는 피처가 없습니다."
            )
        if not spec.features:
            raise ValueError(f"{ablation_id}: 남은 피처가 없습니다.")
    return selected_specs


def load_selected_params(spec: BaseModelSpec) -> dict[str, dict[str, Any]]:
    if spec.source_stage == "step1":
        return {}
    if spec.selected_params_path is None or not spec.selected_params_path.exists():
        raise FileNotFoundError(f"selected parameter 파일이 없습니다: {spec.selected_params_path}")
    payload = json.loads(spec.selected_params_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_params_by_outer_fold")
    if not isinstance(selected, dict) or not selected:
        raise ValueError(f"outer fold별 parameter가 없습니다: {spec.selected_params_path}")
    return selected


def normalize_baseline_oof(oof: pd.DataFrame, spec: BaseModelSpec) -> pd.DataFrame:
    required = {
        "sample_id",
        "outer_fold",
        "y_true",
        "y_prob",
        "sample_type",
        "climate_type",
        "group_id",
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise KeyError(f"{spec.baseline_oof_path.name} 필수 열 누락: {missing}")
    normalized = oof[list(required)].copy()
    normalized["base_candidate_id"] = spec.base_candidate_id
    normalized["source_stage"] = spec.source_stage
    normalized["ablation_id"] = "BASELINE"
    normalized["model_name"] = spec.model_name
    normalized["model_label"] = spec.model_label
    normalized["original_feature_set"] = spec.feature_set
    normalized["feature_count"] = np.nan
    return normalized


def make_ablation_oof(
    data: pd.DataFrame,
    *,
    predicted_idx: np.ndarray,
    probability: np.ndarray,
    outer_fold: np.ndarray,
    base_spec: BaseModelSpec,
    ablation_spec: AblationSpec,
) -> pd.DataFrame:
    subset = data.iloc[predicted_idx]
    return pd.DataFrame(
        {
            "sample_id": subset[step2.SAMPLE_ID_COL].astype(str).to_numpy(),
            "outer_fold": outer_fold.astype(int),
            "y_true": subset[step2.TARGET_COL].astype(int).to_numpy(),
            "y_prob": step2.clipped_probability(probability),
            "base_candidate_id": base_spec.base_candidate_id,
            "source_stage": base_spec.source_stage,
            "ablation_id": ablation_spec.ablation_id,
            "model_name": base_spec.model_name,
            "model_label": base_spec.model_label,
            "original_feature_set": base_spec.feature_set,
            "feature_count": len(ablation_spec.features),
            "sample_type": subset[step2.SAMPLE_TYPE_COL].astype(str).to_numpy(),
            "climate_type": subset[step2.CLIMATE_COL].astype(str).to_numpy(),
            "group_id": subset[step2.GROUP_COL].astype(str).to_numpy(),
        }
    )


def fit_ablation(
    data: pd.DataFrame,
    *,
    base_spec: BaseModelSpec,
    ablation_spec: AblationSpec,
    outer_splits: list[tuple[int, np.ndarray, np.ndarray]],
    selected_params: dict[str, dict[str, Any]],
    random_state: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    predicted_indices: list[np.ndarray] = []
    predicted_probabilities: list[np.ndarray] = []
    predicted_folds: list[np.ndarray] = []

    for outer_fold, train_idx, valid_idx in outer_splits:
        step2.log(
            f"ABLATION {base_spec.base_candidate_id} | {ablation_spec.ablation_id} | "
            f"outer_fold={outer_fold} fit start"
        )
        y_train = data.iloc[train_idx][step2.TARGET_COL].astype(int)
        if base_spec.source_stage == "step2":
            params = selected_params.get(str(outer_fold))
            if params is None:
                raise KeyError(
                    f"{base_spec.base_candidate_id}: outer_fold={outer_fold} parameter가 없습니다."
                )
            candidate = step2.CandidateSpec(
                candidate_id=base_spec.base_candidate_id,
                tuning_group="ablation",
                model_name=base_spec.model_name,
                model_label=base_spec.model_label,
                feature_set=ablation_spec.ablation_id,
                imbalance_option=base_spec.imbalance_option,
                role="Step3 feature ablation",
            )
            model = step2.make_pipeline(
                candidate=candidate,
                features=ablation_spec.features,
                y_train=y_train,
                params=params,
                random_state=random_state + outer_fold,
                n_jobs=n_jobs,
            )
        else:
            model = step1.make_pipeline(
                base_spec.model_name,
                ablation_spec.features,
                y_train,
                imbalance_option=base_spec.imbalance_option,
                random_state=random_state + outer_fold,
                n_jobs=n_jobs,
            )

        model.fit(
            data.iloc[train_idx][ablation_spec.features],
            y_train,
        )
        probability = step2.predict_probability(
            model,
            data.iloc[valid_idx][ablation_spec.features],
        )
        predicted_indices.append(valid_idx)
        predicted_probabilities.append(probability)
        predicted_folds.append(np.full(len(valid_idx), outer_fold, dtype=int))
        step2.log(
            f"ABLATION {base_spec.base_candidate_id} | {ablation_spec.ablation_id} | "
            f"outer_fold={outer_fold} prediction done"
        )

    predicted_idx = np.concatenate(predicted_indices)
    probability = np.concatenate(predicted_probabilities)
    folds = np.concatenate(predicted_folds)
    order = np.argsort(predicted_idx)
    oof = make_ablation_oof(
        data,
        predicted_idx=predicted_idx[order],
        probability=probability[order],
        outer_fold=folds[order],
        base_spec=base_spec,
        ablation_spec=ablation_spec,
    )
    return oof, time.perf_counter() - start


def summary_row(
    oof: pd.DataFrame,
    *,
    base_spec: BaseModelSpec,
    ablation_id: str,
    source_feature_count: int,
    removed_feature_count: int,
    elapsed_seconds: float,
    baseline_metrics: dict[str, float | int],
) -> dict[str, Any]:
    metrics = step2.probability_metrics(oof["y_true"], oof["y_prob"])
    fold_auprcs = [
        step2.safe_average_precision(fold_df["y_true"], fold_df["y_prob"])
        for _, fold_df in oof.groupby("outer_fold")
    ]
    row: dict[str, Any] = {
        "base_candidate_id": base_spec.base_candidate_id,
        "source_stage": base_spec.source_stage,
        "model_name": base_spec.model_name,
        "model_label": base_spec.model_label,
        "original_feature_set": base_spec.feature_set,
        "ablation_id": ablation_id,
        "is_baseline": ablation_id == "BASELINE",
        "source_feature_count": source_feature_count,
        "feature_count": source_feature_count - removed_feature_count,
        "removed_feature_count": removed_feature_count,
        "outer_fold_count": int(oof["outer_fold"].nunique()),
        "elapsed_seconds": elapsed_seconds,
        **metrics,
        "fold_auprc_mean": float(np.nanmean(fold_auprcs)),
        "fold_auprc_std": (
            float(np.nanstd(fold_auprcs, ddof=1)) if len(fold_auprcs) > 1 else float("nan")
        ),
    }
    for metric_name in ["auprc", "auroc", "brier", "log_loss"]:
        baseline_value = float(baseline_metrics[metric_name])
        row[f"baseline_{metric_name}"] = baseline_value
        row[f"delta_{metric_name}"] = float(row[metric_name]) - baseline_value
    return row


def fold_rows(oof: pd.DataFrame, base_spec: BaseModelSpec, ablation_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_fold, fold_df in oof.groupby("outer_fold"):
        rows.append(
            {
                "base_candidate_id": base_spec.base_candidate_id,
                "source_stage": base_spec.source_stage,
                "model_name": base_spec.model_name,
                "ablation_id": ablation_id,
                "outer_fold": int(outer_fold),
                **step2.probability_metrics(fold_df["y_true"], fold_df["y_prob"]),
            }
        )
    return rows


def threshold_rows(oof: pd.DataFrame, base_spec: BaseModelSpec, ablation_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operating_point, threshold in step2.select_thresholds(oof["y_true"], oof["y_prob"]).items():
        rows.append(
            {
                "base_candidate_id": base_spec.base_candidate_id,
                "source_stage": base_spec.source_stage,
                "model_name": base_spec.model_name,
                "ablation_id": ablation_id,
                "threshold_source": "full_oof_diagnostic",
                "operating_point": operating_point,
                **step2.classification_metrics_at_threshold(
                    oof["y_true"],
                    oof["y_prob"],
                    threshold,
                ),
            }
        )
    return rows


def subgroup_rows(oof: pd.DataFrame, base_spec: BaseModelSpec, ablation_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "base_candidate_id": base_spec.base_candidate_id,
        "source_stage": base_spec.source_stage,
        "model_name": base_spec.model_name,
        "ablation_id": ablation_id,
    }
    for negative_type in ["Target_0A", "Target_0B1", "Target_0B2"]:
        subset = oof.loc[oof["sample_type"].isin(["Target_1", negative_type])]
        rows.append(
            {
                **base,
                "subgroup_type": "sample_type_pair",
                "subgroup_value": f"Target_1_vs_{negative_type}",
                **step2.probability_metrics(subset["y_true"], subset["y_prob"]),
            }
        )
    for climate_type, subset in oof.groupby("climate_type"):
        rows.append(
            {
                **base,
                "subgroup_type": "climate_type",
                "subgroup_value": str(climate_type),
                **step2.probability_metrics(subset["y_true"], subset["y_prob"]),
            }
        )
    return rows


def top_risk_rows(oof: pd.DataFrame, base_spec: BaseModelSpec, ablation_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sorted_oof = oof.sort_values("y_prob", ascending=False).reset_index(drop=True)
    total_positive = int(sorted_oof["y_true"].sum())
    base_rate = float(sorted_oof["y_true"].mean())
    for fraction in step2.TOP_RISK_FRACTIONS:
        selected_n = max(1, int(np.ceil(len(sorted_oof) * fraction)))
        selected = sorted_oof.head(selected_n)
        positive_captured = int(selected["y_true"].sum())
        precision = float(selected["y_true"].mean())
        rows.append(
            {
                "base_candidate_id": base_spec.base_candidate_id,
                "source_stage": base_spec.source_stage,
                "model_name": base_spec.model_name,
                "ablation_id": ablation_id,
                "top_fraction": fraction,
                "selected_n": selected_n,
                "positive_captured_n": positive_captured,
                "total_positive_n": total_positive,
                "capture_rate": positive_captured / total_positive,
                "precision": precision,
                "lift": precision / base_rate,
            }
        )
    return rows


def validation_row(
    oof: pd.DataFrame,
    *,
    base_spec: BaseModelSpec,
    ablation_id: str,
    expected_n: int,
    outer_group_leakage_n: int,
) -> dict[str, Any]:
    probability = oof["y_prob"].to_numpy(dtype=float)
    return {
        "base_candidate_id": base_spec.base_candidate_id,
        "source_stage": base_spec.source_stage,
        "model_name": base_spec.model_name,
        "ablation_id": ablation_id,
        "expected_n": expected_n,
        "prediction_n": int(len(oof)),
        "missing_prediction_n": int(max(0, expected_n - len(oof))),
        "duplicate_sample_id_n": int(oof["sample_id"].duplicated().sum()),
        "nan_probability_n": int(np.isnan(probability).sum()),
        "inf_probability_n": int(np.isinf(probability).sum()),
        "outer_group_leakage_n": outer_group_leakage_n,
        "folds_without_positive_n": int(
            sum(int(fold_df["y_true"].sum()) == 0 for _, fold_df in oof.groupby("outer_fold"))
        ),
    }


def write_aggregate_outputs(
    *,
    output_dir: Path,
    data: pd.DataFrame,
    selected_base_specs: list[BaseModelSpec],
    ablation_specs_by_base: dict[str, dict[str, AblationSpec]],
    outer_splits: list[tuple[int, np.ndarray, np.ndarray]],
    elapsed_by_experiment: dict[tuple[str, str], float],
    outer_group_leakage_n: int,
) -> None:
    selected_folds = {outer_fold for outer_fold, _, _ in outer_splits}
    expected_n = sum(len(valid_idx) for _, _, valid_idx in outer_splits)
    summary: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    subgroups: list[dict[str, Any]] = []
    top_risks: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []

    feature_sets = step2.build_feature_sets()
    for base_spec in selected_base_specs:
        source_features = feature_sets[base_spec.feature_set].features
        baseline = normalize_baseline_oof(
            pd.read_csv(base_spec.baseline_oof_path, encoding="utf-8-sig", low_memory=False),
            base_spec,
        )
        baseline = baseline.loc[baseline["outer_fold"].isin(selected_folds)].copy()
        baseline["feature_count"] = len(source_features)
        baseline_metrics = step2.probability_metrics(baseline["y_true"], baseline["y_prob"])
        experiment_oofs: list[tuple[str, pd.DataFrame, int, float]] = [
            ("BASELINE", baseline, 0, 0.0)
        ]

        for ablation_id, ablation_spec in ablation_specs_by_base[base_spec.base_candidate_id].items():
            path = output_dir / f"oof__{base_spec.base_candidate_id}__{ablation_id}.csv"
            if not path.exists():
                continue
            oof = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            experiment_oofs.append(
                (
                    ablation_id,
                    oof,
                    len(ablation_spec.removed_features),
                    elapsed_by_experiment.get((base_spec.base_candidate_id, ablation_id), float("nan")),
                )
            )

        for ablation_id, oof, removed_count, elapsed_seconds in experiment_oofs:
            summary.append(
                summary_row(
                    oof,
                    base_spec=base_spec,
                    ablation_id=ablation_id,
                    source_feature_count=len(source_features),
                    removed_feature_count=removed_count,
                    elapsed_seconds=elapsed_seconds,
                    baseline_metrics=baseline_metrics,
                )
            )
            folds.extend(fold_rows(oof, base_spec, ablation_id))
            thresholds.extend(threshold_rows(oof, base_spec, ablation_id))
            subgroups.extend(subgroup_rows(oof, base_spec, ablation_id))
            top_risks.extend(top_risk_rows(oof, base_spec, ablation_id))
            validations.append(
                validation_row(
                    oof,
                    base_spec=base_spec,
                    ablation_id=ablation_id,
                    expected_n=expected_n,
                    outer_group_leakage_n=outer_group_leakage_n,
                )
            )

    step2.write_csv(output_dir / "summary__step3_feature_ablation.csv", summary)
    step2.write_csv(output_dir / "fold_metrics__step3_feature_ablation.csv", folds)
    step2.write_csv(output_dir / "threshold_metrics__step3_feature_ablation.csv", thresholds)
    step2.write_csv(output_dir / "subgroup_metrics__step3_feature_ablation.csv", subgroups)
    step2.write_csv(output_dir / "top_risk_metrics__step3_feature_ablation.csv", top_risks)
    step2.write_csv(output_dir / "validation_checks__step3_feature_ablation.csv", validations)


def main() -> None:
    args = parse_args()
    root = step2.find_project_root()
    data_path = (
        Path(args.data)
        if args.data
        else root / "data" / "학습데이터" / "최종_머신러닝_학습데이터.csv"
    )
    step1_output_dir = (
        Path(args.step1_output_dir)
        if args.step1_output_dir
        else root / "jsw" / "Analysis" / "new_machine_learning" / "outputs" / "step1_single_models"
    )
    step2_output_dir = (
        Path(args.step2_output_dir)
        if args.step2_output_dir
        else root / "jsw" / "Analysis" / "new_machine_learning" / "outputs" / "step2_tuned_single_models"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "jsw" / "Analysis" / "new_machine_learning" / "outputs" / "step3_feature_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    step2.log(f"입력 데이터 로드: {data_path}")
    data = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    data[step2.TARGET_COL] = data[step2.TARGET_COL].astype(int)

    base_specs = build_base_model_specs(
        step1_output_dir=step1_output_dir,
        step2_output_dir=step2_output_dir,
    )
    selected_base_ids = parse_selection(
        args.base_candidate_ids,
        list(base_specs),
        argument_name="--base-candidate-ids",
    )
    selected_base_specs = [base_specs[candidate_id] for candidate_id in selected_base_ids]
    selected_ablation_ids = parse_ablation_ids(args.ablation_ids)
    feature_sets = step2.build_feature_sets()

    for base_spec in selected_base_specs:
        if not base_spec.baseline_oof_path.exists():
            raise FileNotFoundError(f"baseline OOF 파일이 없습니다: {base_spec.baseline_oof_path}")
        if base_spec.feature_set not in feature_sets:
            raise KeyError(f"알 수 없는 feature set: {base_spec.feature_set}")
        if base_spec.source_stage == "step2":
            load_selected_params(base_spec)

    outer_manifest = step2.load_or_create_outer_manifest(
        data,
        step1_output_dir=step1_output_dir,
        output_dir=output_dir,
        n_splits=5,
        random_state=args.random_state,
    )
    outer_splits = step2.outer_splits_from_manifest(
        outer_manifest,
        max_outer_folds=args.max_outer_folds,
    )
    outer_group_leakage_n = step2.check_group_leakage(data, outer_splits)

    ablation_specs_by_base: dict[str, dict[str, AblationSpec]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for base_spec in selected_base_specs:
        source_features = feature_sets[base_spec.feature_set].features
        missing_source = [feature for feature in source_features if feature not in data.columns]
        if missing_source:
            raise KeyError(f"{base_spec.base_candidate_id} 원본 피처 누락: {missing_source}")
        ablation_specs = build_ablation_specs(source_features, selected_ablation_ids)
        ablation_specs_by_base[base_spec.base_candidate_id] = ablation_specs
        for ablation_spec in ablation_specs.values():
            missing = [feature for feature in ablation_spec.features if feature not in data.columns]
            if missing:
                raise KeyError(
                    f"{base_spec.base_candidate_id}/{ablation_spec.ablation_id} 피처 누락: {missing}"
                )
            output_path = (
                output_dir
                / f"oof__{base_spec.base_candidate_id}__{ablation_spec.ablation_id}.csv"
            )
            manifest_rows.append(
                {
                    "base_candidate_id": base_spec.base_candidate_id,
                    "source_stage": base_spec.source_stage,
                    "model_name": base_spec.model_name,
                    "model_label": base_spec.model_label,
                    "original_feature_set": base_spec.feature_set,
                    "imbalance_option": base_spec.imbalance_option,
                    "ablation_id": ablation_spec.ablation_id,
                    "description": ablation_spec.description,
                    "source_feature_count": len(source_features),
                    "feature_count": len(ablation_spec.features),
                    "removed_feature_count": len(ablation_spec.removed_features),
                    "removed_features": "|".join(ablation_spec.removed_features),
                    "output_path": str(output_path),
                    "existing_complete": output_path.exists(),
                }
            )
    step2.write_csv(output_dir / "ablation_manifest.csv", manifest_rows)

    planned_experiments = sum(len(specs) for specs in ablation_specs_by_base.values())
    planned_fits = planned_experiments * len(outer_splits)
    run_config = {
        "script": str(Path(__file__).resolve()),
        "started_at": step2.timestamp(),
        "data_path": str(data_path),
        "step1_output_dir": str(step1_output_dir),
        "step2_output_dir": str(step2_output_dir),
        "output_dir": str(output_dir),
        "n_rows": int(len(data)),
        "positive_n": int(data[step2.TARGET_COL].sum()),
        "base_candidate_ids": selected_base_ids,
        "ablation_ids": selected_ablation_ids,
        "experiment_count": planned_experiments,
        "outer_fold_count": len(outer_splits),
        "planned_fits": planned_fits,
        "max_outer_folds": args.max_outer_folds,
        "overwrite": bool(args.overwrite),
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "outer_group_leakage_n": outer_group_leakage_n,
    }
    step2.write_json(
        output_dir / ("run_manifest__check_config.json" if args.check_config else "run_manifest__all.json"),
        run_config,
    )
    step2.log(
        f"Step3 계획: base_candidates={len(selected_base_specs)}, "
        f"experiments={planned_experiments}, outer_folds={len(outer_splits)}, "
        f"planned_fits={planned_fits}"
    )
    step2.log("Optuna 재튜닝 없음: Step2 fold별 parameter 또는 Step1 고정 preset 재사용")
    step2.log(f"결과 폴더: {output_dir}")

    if args.check_config:
        step2.log("--check-config 지정: 학습 없이 종료합니다.")
        return

    elapsed_by_experiment: dict[tuple[str, str], float] = {}
    completed_count = 0
    skipped_count = 0
    experiment_bar = None
    if not args.no_progress_bar and step2.tqdm is not None:
        experiment_bar = step2.tqdm(
            total=planned_experiments,
            desc="STEP3 ablations",
            unit="experiment",
            dynamic_ncols=True,
        )

    try:
        for base_spec in selected_base_specs:
            selected_params = load_selected_params(base_spec)
            for ablation_spec in ablation_specs_by_base[base_spec.base_candidate_id].values():
                output_path = (
                    output_dir
                    / f"oof__{base_spec.base_candidate_id}__{ablation_spec.ablation_id}.csv"
                )
                if output_path.exists() and not args.overwrite:
                    existing = pd.read_csv(output_path, encoding="utf-8-sig", low_memory=False)
                    expected_n = sum(len(valid_idx) for _, _, valid_idx in outer_splits)
                    if (
                        len(existing) == expected_n
                        and existing["sample_id"].duplicated().sum() == 0
                        and existing["y_prob"].isna().sum() == 0
                    ):
                        step2.log(
                            f"ABLATION SKIP {base_spec.base_candidate_id} | "
                            f"{ablation_spec.ablation_id} | completed OOF exists"
                        )
                        skipped_count += 1
                        if experiment_bar is not None:
                            experiment_bar.update(1)
                        continue

                step2.log(
                    f"ABLATION START {base_spec.base_candidate_id} | "
                    f"{ablation_spec.ablation_id} | features={len(ablation_spec.features)}"
                )
                oof, elapsed = fit_ablation(
                    data,
                    base_spec=base_spec,
                    ablation_spec=ablation_spec,
                    outer_splits=outer_splits,
                    selected_params=selected_params,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                )
                oof.to_csv(output_path, index=False, encoding="utf-8-sig")
                elapsed_by_experiment[
                    (base_spec.base_candidate_id, ablation_spec.ablation_id)
                ] = elapsed
                completed_count += 1
                metrics = step2.probability_metrics(oof["y_true"], oof["y_prob"])
                step2.log(
                    f"ABLATION DONE {base_spec.base_candidate_id} | "
                    f"{ablation_spec.ablation_id} | auprc={metrics['auprc']:.4f} | "
                    f"elapsed={step2.format_seconds(elapsed)}"
                )
                write_aggregate_outputs(
                    output_dir=output_dir,
                    data=data,
                    selected_base_specs=selected_base_specs,
                    ablation_specs_by_base=ablation_specs_by_base,
                    outer_splits=outer_splits,
                    elapsed_by_experiment=elapsed_by_experiment,
                    outer_group_leakage_n=outer_group_leakage_n,
                )
                if experiment_bar is not None:
                    experiment_bar.update(1)
                    experiment_bar.set_postfix(
                        {"last": ablation_spec.ablation_id},
                        refresh=False,
                    )
    finally:
        if experiment_bar is not None:
            experiment_bar.close()

    write_aggregate_outputs(
        output_dir=output_dir,
        data=data,
        selected_base_specs=selected_base_specs,
        ablation_specs_by_base=ablation_specs_by_base,
        outer_splits=outer_splits,
        elapsed_by_experiment=elapsed_by_experiment,
        outer_group_leakage_n=outer_group_leakage_n,
    )

    final_manifest = dict(run_config)
    final_manifest.update(
        {
            "finished_at": step2.timestamp(),
            "completed_count_this_run": completed_count,
            "skipped_existing_count": skipped_count,
        }
    )
    step2.write_json(output_dir / "run_manifest__all.json", final_manifest)
    step2.log(
        f"STEP3 DONE | completed_this_run={completed_count} | "
        f"skipped_existing={skipped_count} | output={output_dir}"
    )


if __name__ == "__main__":
    main()
