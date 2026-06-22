from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import step1_single_models as step1
import step2_tuned_single_models as step2


RANDOM_STATE = 20260622
DEFAULT_INNER_SPLITS = 4
DEFAULT_WEIGHT_TRIALS = 300
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
WEIGHT_TIE_TOLERANCE = 0.0005
CLIP_EPSILON = 1e-12

STREAM_LGBM = "lgbm_baseline"
STREAM_LGBM_DROP_ISI = "lgbm_drop_isi"
STREAM_RF = "rf_step1"
STREAM_ET = "et_step1"
STREAM_XGB = "xgb_tuned"
STREAM_HGB = "hgb_tuned"

INNER_STREAMS = [
    STREAM_LGBM,
    STREAM_LGBM_DROP_ISI,
    STREAM_RF,
    STREAM_ET,
    STREAM_XGB,
]
OUTER_STREAMS = INNER_STREAMS + [STREAM_HGB]

METADATA_COLUMNS = [
    "sample_id",
    "outer_fold",
    "y_true",
    "sample_type",
    "climate_type",
    "group_id",
]


@dataclass(frozen=True)
class StreamSpec:
    stream_id: str
    source_stage: str
    model_name: str
    model_label: str
    feature_set: str
    imbalance_option: str
    oof_path: Path
    selected_params_path: Path | None
    drop_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnsembleRecipe:
    ensemble_id: str
    candidate_kind: str
    method: str
    streams: tuple[str, ...]
    lgbm_variant: str
    selection_role: str
    transform: str = "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="new_machine_learning Step4: fixed and nested ensemble selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="", help="입력 CSV 경로")
    parser.add_argument("--step1-output-dir", type=str, default="", help="Step1 결과 폴더")
    parser.add_argument("--step2-output-dir", type=str, default="", help="Step2 결과 폴더")
    parser.add_argument("--step3-output-dir", type=str, default="", help="Step3 결과 폴더")
    parser.add_argument("--output-dir", type=str, default="", help="Step4 결과 폴더")
    parser.add_argument("--inner-splits", type=int, default=DEFAULT_INNER_SPLITS)
    parser.add_argument("--weight-trials", type=int, default=DEFAULT_WEIGHT_TRIALS)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--max-outer-folds", type=int, default=0, help="검증용 outer fold 제한")
    parser.add_argument("--check-config", action="store_true", help="구성과 입력만 검사하고 종료")
    parser.add_argument("--resume-existing", action="store_true", help="완전한 inner OOF가 있으면 재사용")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def clipped_probability(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), CLIP_EPSILON, 1.0 - CLIP_EPSILON)


def pct_rank(values: np.ndarray | pd.Series) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy()


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    probability = clipped_probability(values)
    return np.log(probability / (1.0 - probability))


def top_capture(y_true: np.ndarray, score: np.ndarray, fraction: float) -> float:
    selected_n = max(1, int(math.ceil(len(y_true) * fraction)))
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    positive_n = int(np.asarray(y_true, dtype=int).sum())
    if positive_n == 0:
        return float("nan")
    return float(np.asarray(y_true, dtype=int)[order[:selected_n]].sum() / positive_n)


def target0a_auprc(metadata: pd.DataFrame, score: np.ndarray) -> float:
    mask = metadata["sample_type"].isin(["Target_1", "Target_0A"]).to_numpy()
    return step2.safe_average_precision(
        metadata.loc[mask, "y_true"],
        np.asarray(score, dtype=float)[mask],
    )


def build_stream_specs(
    *,
    step1_output_dir: Path,
    step2_output_dir: Path,
    step3_output_dir: Path,
) -> dict[str, StreamSpec]:
    return {
        STREAM_LGBM: StreamSpec(
            stream_id=STREAM_LGBM,
            source_stage="step2",
            model_name="lightgbm",
            model_label="LightGBM",
            feature_set="WS_ALL__CANADA_ALL__LC_USED",
            imbalance_option="none",
            oof_path=step2_output_dir / "oof__TUNE_LGBM_ALL_ALL_LC_NONE.csv",
            selected_params_path=step2_output_dir / "selected_params__TUNE_LGBM_ALL_ALL_LC_NONE.json",
        ),
        STREAM_LGBM_DROP_ISI: StreamSpec(
            stream_id=STREAM_LGBM_DROP_ISI,
            source_stage="step3",
            model_name="lightgbm",
            model_label="LightGBM",
            feature_set="WS_ALL__CANADA_ALL__LC_USED",
            imbalance_option="none",
            oof_path=(
                step3_output_dir
                / "oof__TUNE_LGBM_ALL_ALL_LC_NONE__CANADA_DROP_ISI.csv"
            ),
            selected_params_path=step2_output_dir / "selected_params__TUNE_LGBM_ALL_ALL_LC_NONE.json",
            drop_features=("ISI",),
        ),
        STREAM_RF: StreamSpec(
            stream_id=STREAM_RF,
            source_stage="step1",
            model_name="random_forest",
            model_label="RandomForest",
            feature_set="WS_CORE__CANADA_ALL__LC_USED",
            imbalance_option="balanced",
            oof_path=(
                step1_output_dir
                / "oof__random_forest__WS_CORE__CANADA_ALL__LC_USED__balanced.csv"
            ),
            selected_params_path=None,
        ),
        STREAM_ET: StreamSpec(
            stream_id=STREAM_ET,
            source_stage="step1",
            model_name="extra_trees",
            model_label="ExtraTrees",
            feature_set="WS_ALL__CANADA_ALL__LC_NONE",
            imbalance_option="balanced",
            oof_path=(
                step1_output_dir
                / "oof__extra_trees__WS_ALL__CANADA_ALL__LC_NONE__balanced.csv"
            ),
            selected_params_path=None,
        ),
        STREAM_XGB: StreamSpec(
            stream_id=STREAM_XGB,
            source_stage="step2",
            model_name="xgboost",
            model_label="XGBoost",
            feature_set="WS_ALL__CANADA_ALL__LC_USED",
            imbalance_option="none",
            oof_path=step2_output_dir / "oof__RESCUE_XGB_ALL_ALL_LC_NONE.csv",
            selected_params_path=step2_output_dir / "selected_params__RESCUE_XGB_ALL_ALL_LC_NONE.json",
        ),
        STREAM_HGB: StreamSpec(
            stream_id=STREAM_HGB,
            source_stage="step2",
            model_name="hist_gradient_boosting",
            model_label="HistGradientBoosting",
            feature_set="WS_ALL__CANADA_ALL__LC_USED",
            imbalance_option="none",
            oof_path=step2_output_dir / "oof__TUNE_HGB_ALL_ALL_LC_NONE.csv",
            selected_params_path=step2_output_dir / "selected_params__TUNE_HGB_ALL_ALL_LC_NONE.json",
        ),
    }


def stream_features(
    stream: StreamSpec,
    feature_sets: dict[str, step2.FeatureSetSpec],
) -> list[str]:
    features = feature_sets[stream.feature_set].features
    drop_set = set(stream.drop_features)
    return [feature for feature in features if feature not in drop_set]


def load_selected_params(stream: StreamSpec) -> dict[str, dict[str, Any]]:
    if stream.selected_params_path is None:
        return {}
    payload = json.loads(stream.selected_params_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_params_by_outer_fold")
    if not isinstance(selected, dict) or not selected:
        raise ValueError(f"outer fold별 parameter가 없습니다: {stream.selected_params_path}")
    return selected


def load_outer_matrix(streams: dict[str, StreamSpec]) -> pd.DataFrame:
    matrix: pd.DataFrame | None = None
    for stream_id in OUTER_STREAMS:
        stream = streams[stream_id]
        if not stream.oof_path.exists():
            raise FileNotFoundError(f"OOF 파일이 없습니다: {stream.oof_path}")
        oof = pd.read_csv(stream.oof_path, encoding="utf-8-sig", low_memory=False)
        required = set(METADATA_COLUMNS + ["y_prob"])
        missing = sorted(required - set(oof.columns))
        if missing:
            raise KeyError(f"{stream.oof_path.name} 필수 열 누락: {missing}")
        current = oof[METADATA_COLUMNS + ["y_prob"]].copy()
        current = current.rename(columns={"y_prob": stream_id})
        if matrix is None:
            matrix = current
        else:
            matrix = matrix.merge(
                current,
                on=METADATA_COLUMNS,
                how="inner",
                validate="one_to_one",
            )
    if matrix is None:
        raise ValueError("OOF stream이 없습니다.")
    if matrix["sample_id"].duplicated().any():
        raise ValueError("ensemble input에 중복 sample_id가 있습니다.")
    if matrix[OUTER_STREAMS].isna().any().any():
        raise ValueError("ensemble input에 결측 base prediction이 있습니다.")
    return matrix.sort_values("sample_id").reset_index(drop=True)


def make_inner_manifest(
    data: pd.DataFrame,
    outer_splits: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    inner_splits: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outer_fold, train_idx, _ in outer_splits:
        outer_train = data.iloc[train_idx].reset_index()
        y_train = outer_train[step2.TARGET_COL].astype(int).to_numpy()
        groups = outer_train[step2.GROUP_COL].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=random_state + 1000 + outer_fold,
        )
        assigned = np.full(len(outer_train), -1, dtype=int)
        for inner_fold, (_, inner_valid_local) in enumerate(
            splitter.split(outer_train, y_train, groups)
        ):
            assigned[inner_valid_local] = inner_fold
        if (assigned < 0).any():
            raise RuntimeError(f"outer_fold={outer_fold}: inner fold 미배정 표본이 있습니다.")
        for local_index, inner_fold in enumerate(assigned):
            row = outer_train.iloc[local_index]
            rows.append(
                {
                    "sample_id": str(row[step2.SAMPLE_ID_COL]),
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "y_true": int(row[step2.TARGET_COL]),
                    "sample_type": str(row[step2.SAMPLE_TYPE_COL]),
                    "climate_type": str(row[step2.CLIMATE_COL]),
                    "group_id": str(row[step2.GROUP_COL]),
                    "data_index": int(row["index"]),
                }
            )
    manifest = pd.DataFrame(rows)
    duplicate_n = int(
        manifest.duplicated(subset=["sample_id", "outer_fold"]).sum()
    )
    if duplicate_n:
        raise RuntimeError(f"inner manifest 중복: {duplicate_n}")
    return manifest


def validate_inner_group_leakage(manifest: pd.DataFrame) -> int:
    leakage = 0
    for outer_fold, outer_part in manifest.groupby("outer_fold"):
        for inner_fold in sorted(outer_part["inner_fold"].unique()):
            valid_groups = set(
                outer_part.loc[
                    outer_part["inner_fold"] == inner_fold,
                    "group_id",
                ]
            )
            train_groups = set(
                outer_part.loc[
                    outer_part["inner_fold"] != inner_fold,
                    "group_id",
                ]
            )
            leakage += len(valid_groups.intersection(train_groups))
    return leakage


def build_stream_model(
    *,
    stream: StreamSpec,
    features: list[str],
    y_train: pd.Series,
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
) -> Any:
    if stream.source_stage == "step1":
        return step1.make_pipeline(
            stream.model_name,
            features,
            y_train,
            imbalance_option=stream.imbalance_option,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    candidate = step2.CandidateSpec(
        candidate_id=stream.stream_id,
        tuning_group="step4_inner",
        model_name=stream.model_name,
        model_label=stream.model_label,
        feature_set=stream.feature_set,
        imbalance_option=stream.imbalance_option,
        role="Step4 nested ensemble inner OOF",
    )
    return step2.make_pipeline(
        candidate=candidate,
        features=features,
        y_train=y_train,
        params=params,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def generate_inner_base_oof(
    data: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    *,
    streams: dict[str, StreamSpec],
    feature_sets: dict[str, step2.FeatureSetSpec],
    random_state: int,
    n_jobs: int,
    progress_bar: bool,
) -> pd.DataFrame:
    result = inner_manifest.copy()
    total_fits = (
        result[["outer_fold", "inner_fold"]].drop_duplicates().shape[0]
        * len(INNER_STREAMS)
    )
    bar = None
    if progress_bar and step2.tqdm is not None:
        bar = step2.tqdm(
            total=total_fits,
            desc="STEP4 inner base fits",
            unit="fit",
            dynamic_ncols=True,
        )

    selected_params = {
        stream_id: load_selected_params(streams[stream_id])
        for stream_id in INNER_STREAMS
    }
    try:
        for stream_id in INNER_STREAMS:
            stream = streams[stream_id]
            features = stream_features(stream, feature_sets)
            result[stream_id] = np.nan
            for outer_fold, outer_part in result.groupby("outer_fold"):
                params = (
                    selected_params[stream_id].get(str(int(outer_fold)), {})
                    if stream.source_stage != "step1"
                    else {}
                )
                if stream.source_stage != "step1" and not params:
                    raise KeyError(
                        f"{stream_id}: outer_fold={outer_fold} parameter가 없습니다."
                    )
                for inner_fold in sorted(outer_part["inner_fold"].unique()):
                    inner_valid_rows = outer_part.index[
                        outer_part["inner_fold"] == inner_fold
                    ]
                    inner_train_rows = outer_part.index[
                        outer_part["inner_fold"] != inner_fold
                    ]
                    train_idx = result.loc[inner_train_rows, "data_index"].to_numpy(
                        dtype=int
                    )
                    valid_idx = result.loc[inner_valid_rows, "data_index"].to_numpy(
                        dtype=int
                    )
                    y_train = data.iloc[train_idx][step2.TARGET_COL].astype(int)
                    model = build_stream_model(
                        stream=stream,
                        features=features,
                        y_train=y_train,
                        params=params,
                        random_state=(
                            random_state
                            + int(outer_fold) * 100
                            + int(inner_fold)
                        ),
                        n_jobs=n_jobs,
                    )
                    model.fit(data.iloc[train_idx][features], y_train)
                    probability = step2.predict_probability(
                        model,
                        data.iloc[valid_idx][features],
                    )
                    result.loc[inner_valid_rows, stream_id] = probability
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix(
                            {
                                "stream": stream_id,
                                "outer": int(outer_fold),
                                "inner": int(inner_fold),
                            },
                            refresh=False,
                        )
    finally:
        if bar is not None:
            bar.close()

    if result[INNER_STREAMS].isna().any().any():
        raise RuntimeError("inner base OOF 예측 누락")
    return result


def complete_inner_oof(
    path: Path,
    inner_manifest: pd.DataFrame,
) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return False
    if len(frame) != len(inner_manifest):
        return False
    required = set(METADATA_COLUMNS + ["inner_fold", "data_index"] + INNER_STREAMS)
    if not required.issubset(frame.columns):
        return False
    if frame[INNER_STREAMS].isna().any().any():
        return False
    if frame.duplicated(subset=["sample_id", "outer_fold"]).any():
        return False
    return True


def simple_recipe_definitions() -> list[EnsembleRecipe]:
    recipes: list[EnsembleRecipe] = []
    variant_stream = {
        "baseline": STREAM_LGBM,
        "drop_isi": STREAM_LGBM_DROP_ISI,
    }
    groups = [
        ("lgbm_rf", (STREAM_RF,), ("mean", "geomean")),
        (
            "lgbm_rf_et",
            (STREAM_RF, STREAM_ET),
            ("mean", "geomean", "rank_average", "max"),
        ),
        (
            "lgbm_rf_xgb",
            (STREAM_RF, STREAM_XGB),
            ("mean", "geomean", "rank_average", "max"),
        ),
        (
            "lgbm_rf_et_xgb",
            (STREAM_RF, STREAM_ET, STREAM_XGB),
            ("mean", "geomean"),
        ),
        (
            "lgbm_rf_hgb",
            (STREAM_RF, STREAM_HGB),
            ("mean", "geomean", "max"),
        ),
    ]
    for variant, lgbm_stream in variant_stream.items():
        for group_name, other_streams, methods in groups:
            streams = (lgbm_stream, *other_streams)
            for method in methods:
                role = (
                    "ranking_only"
                    if method == "rank_average"
                    else "recall_diagnostic"
                    if method == "max"
                    else "probability_candidate"
                )
                recipes.append(
                    EnsembleRecipe(
                        ensemble_id=f"fixed__{variant}__{group_name}__{method}",
                        candidate_kind="fixed",
                        method=method,
                        streams=streams,
                        lgbm_variant=variant,
                        selection_role=role,
                    )
                )
    return recipes


def weighted_recipe_definitions() -> list[EnsembleRecipe]:
    recipes: list[EnsembleRecipe] = []
    variant_stream = {
        "baseline": STREAM_LGBM,
        "drop_isi": STREAM_LGBM_DROP_ISI,
    }
    groups = [
        ("lgbm_rf_et", (STREAM_RF, STREAM_ET)),
        ("lgbm_rf_xgb", (STREAM_RF, STREAM_XGB)),
        ("lgbm_rf_et_xgb", (STREAM_RF, STREAM_ET, STREAM_XGB)),
    ]
    for variant, lgbm_stream in variant_stream.items():
        for group_name, other_streams in groups:
            streams = (lgbm_stream, *other_streams)
            for method in ["weighted_mean", "weighted_geomean"]:
                recipes.append(
                    EnsembleRecipe(
                        ensemble_id=f"weighted__{variant}__{group_name}__{method}",
                        candidate_kind="weighted",
                        method=method,
                        streams=streams,
                        lgbm_variant=variant,
                        selection_role="probability_candidate",
                    )
                )
    return recipes


def stacking_recipe_definitions() -> list[EnsembleRecipe]:
    recipes: list[EnsembleRecipe] = []
    variant_stream = {
        "baseline": STREAM_LGBM,
        "drop_isi": STREAM_LGBM_DROP_ISI,
    }
    groups = [
        ("lgbm_rf_et", (STREAM_RF, STREAM_ET)),
        ("lgbm_rf_xgb", (STREAM_RF, STREAM_XGB)),
    ]
    for variant, lgbm_stream in variant_stream.items():
        for group_name, other_streams in groups:
            streams = (lgbm_stream, *other_streams)
            for transform in ["raw", "logit"]:
                recipes.append(
                    EnsembleRecipe(
                        ensemble_id=f"stack__{variant}__{group_name}__{transform}",
                        candidate_kind="stacking",
                        method="logistic_l2",
                        streams=streams,
                        lgbm_variant=variant,
                        selection_role="probability_candidate",
                        transform=transform,
                    )
                )
    return recipes


def single_recipe_definitions() -> list[EnsembleRecipe]:
    return [
        EnsembleRecipe(
            ensemble_id=f"single__{stream_id}",
            candidate_kind="single",
            method="single",
            streams=(stream_id,),
            lgbm_variant=(
                "drop_isi"
                if stream_id == STREAM_LGBM_DROP_ISI
                else "baseline"
                if stream_id == STREAM_LGBM
                else "not_applicable"
            ),
            selection_role="probability_reference",
        )
        for stream_id in OUTER_STREAMS
    ]


def combine_scores(
    matrix: pd.DataFrame,
    streams: tuple[str, ...],
    method: str,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    values = clipped_probability(matrix[list(streams)].to_numpy(dtype=float))
    if method == "single":
        return values[:, 0]
    if method == "mean":
        return clipped_probability(values.mean(axis=1))
    if method == "geomean":
        return clipped_probability(np.exp(np.log(values).mean(axis=1)))
    if method == "rank_average":
        ranks = np.column_stack(
            [pct_rank(values[:, column]) for column in range(values.shape[1])]
        )
        return np.asarray(ranks.mean(axis=1), dtype=float)
    if method == "max":
        return clipped_probability(values.max(axis=1))
    if weights is None:
        raise ValueError(f"{method}: weight가 필요합니다.")
    weights = np.asarray(weights, dtype=float)
    if method == "weighted_mean":
        return clipped_probability(values @ weights)
    if method == "weighted_geomean":
        return clipped_probability(np.exp(np.log(values) @ weights))
    raise KeyError(method)


def make_prediction_frame(
    metadata: pd.DataFrame,
    recipe: EnsembleRecipe,
    score: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{column: metadata[column].to_numpy() for column in METADATA_COLUMNS},
            "ensemble_id": recipe.ensemble_id,
            "candidate_kind": recipe.candidate_kind,
            "method": recipe.method,
            "selection_role": recipe.selection_role,
            "lgbm_variant": recipe.lgbm_variant,
            "streams": "|".join(recipe.streams),
            "y_score": np.asarray(score, dtype=float),
        }
    )


def make_fixed_predictions(
    outer_matrix: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for recipe in single_recipe_definitions() + simple_recipe_definitions():
        score = combine_scores(
            outer_matrix,
            recipe.streams,
            recipe.method,
        )
        predictions.append(make_prediction_frame(outer_matrix, recipe, score))
        manifest_rows.append(recipe_row(recipe))
    return predictions, manifest_rows


def recipe_row(recipe: EnsembleRecipe) -> dict[str, Any]:
    return {
        "ensemble_id": recipe.ensemble_id,
        "candidate_kind": recipe.candidate_kind,
        "method": recipe.method,
        "transform": recipe.transform,
        "selection_role": recipe.selection_role,
        "lgbm_variant": recipe.lgbm_variant,
        "streams": "|".join(recipe.streams),
        "n_models": len(recipe.streams),
    }


def enqueue_weight_seeds(
    study: optuna.Study,
    streams: tuple[str, ...],
) -> None:
    study.enqueue_trial({f"raw__{stream}": 1.0 for stream in streams})
    for anchor in streams:
        study.enqueue_trial(
            {
                f"raw__{stream}": 1.0 if stream == anchor else 0.0
                for stream in streams
            }
        )


def select_weight_trial(
    trials: list[optuna.trial.FrozenTrial],
) -> optuna.trial.FrozenTrial:
    completed = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    if not completed:
        raise RuntimeError("완료된 weight trial이 없습니다.")
    best_auprc = max(float(trial.value) for trial in completed)
    near_best = [
        trial
        for trial in completed
        if float(trial.value) >= best_auprc - WEIGHT_TIE_TOLERANCE
    ]
    near_best.sort(
        key=lambda trial: (
            -float(trial.user_attrs["target0a_auprc"]),
            -float(trial.user_attrs["top10_capture"]),
            float(trial.user_attrs["brier"]),
            float(trial.user_attrs["max_weight"]),
            trial.number,
        )
    )
    return near_best[0]


def optimize_weights(
    train_frame: pd.DataFrame,
    recipe: EnsembleRecipe,
    *,
    n_trials: int,
    random_state: int,
    outer_fold: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    y_true = train_frame["y_true"].to_numpy(dtype=int)

    def objective(trial: optuna.Trial) -> float:
        raw = np.array(
            [
                trial.suggest_float(f"raw__{stream}", 0.0, 1.0)
                for stream in recipe.streams
            ],
            dtype=float,
        )
        if raw.sum() <= 0:
            weights = np.full(len(raw), 1.0 / len(raw))
        else:
            weights = raw / raw.sum()
        score = combine_scores(
            train_frame,
            recipe.streams,
            recipe.method,
            weights=weights,
        )
        auprc = step2.safe_average_precision(y_true, score)
        trial.set_user_attr("target0a_auprc", target0a_auprc(train_frame, score))
        trial.set_user_attr("top10_capture", top_capture(y_true, score, 0.10))
        trial.set_user_attr(
            "brier",
            float(step2.probability_metrics(y_true, score)["brier"]),
        )
        trial.set_user_attr("max_weight", float(weights.max()))
        for stream, weight in zip(recipe.streams, weights):
            trial.set_user_attr(f"weight__{stream}", float(weight))
        return float(auprc)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=random_state + outer_fold * 1000 + len(recipe.streams)
        ),
    )
    enqueue_weight_seeds(study, recipe.streams)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=False,
    )
    selected = select_weight_trial(study.trials)
    weights = np.array(
        [
            float(selected.user_attrs[f"weight__{stream}"])
            for stream in recipe.streams
        ],
        dtype=float,
    )
    trial_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        row: dict[str, Any] = {
            "ensemble_id": recipe.ensemble_id,
            "outer_fold": int(outer_fold),
            "trial": int(trial.number),
            "auprc": float(trial.value),
            "target0a_auprc": float(trial.user_attrs["target0a_auprc"]),
            "top10_capture": float(trial.user_attrs["top10_capture"]),
            "brier": float(trial.user_attrs["brier"]),
            "max_weight": float(trial.user_attrs["max_weight"]),
            "selected": trial.number == selected.number,
        }
        for stream in recipe.streams:
            row[f"weight__{stream}"] = float(
                trial.user_attrs[f"weight__{stream}"]
            )
        trial_rows.append(row)
    return weights, pd.DataFrame(trial_rows)


def make_weighted_predictions(
    outer_matrix: pd.DataFrame,
    inner_oof: pd.DataFrame,
    recipes: list[EnsembleRecipe],
    *,
    n_trials: int,
    random_state: int,
    progress_bar: bool,
) -> tuple[
    list[pd.DataFrame],
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
]:
    predictions: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    bar = None
    total = len(recipes) * outer_matrix["outer_fold"].nunique()
    if progress_bar and step2.tqdm is not None:
        bar = step2.tqdm(
            total=total,
            desc="STEP4 weight search",
            unit="fold",
            dynamic_ncols=True,
        )
    try:
        for recipe in recipes:
            score = np.full(len(outer_matrix), np.nan, dtype=float)
            for outer_fold in sorted(outer_matrix["outer_fold"].unique()):
                train_frame = inner_oof.loc[
                    inner_oof["outer_fold"] == outer_fold
                ].reset_index(drop=True)
                weights, trials = optimize_weights(
                    train_frame,
                    recipe,
                    n_trials=n_trials,
                    random_state=random_state,
                    outer_fold=int(outer_fold),
                )
                trial_frames.append(trials)
                valid_mask = (
                    outer_matrix["outer_fold"].to_numpy() == outer_fold
                )
                valid_frame = outer_matrix.loc[valid_mask]
                score[valid_mask] = combine_scores(
                    valid_frame,
                    recipe.streams,
                    recipe.method,
                    weights=weights,
                )
                selected_row: dict[str, Any] = {
                    "ensemble_id": recipe.ensemble_id,
                    "outer_fold": int(outer_fold),
                    "streams": "|".join(recipe.streams),
                    "weight_sum": float(weights.sum()),
                    "max_weight": float(weights.max()),
                }
                for stream, weight in zip(recipe.streams, weights):
                    selected_row[f"weight__{stream}"] = float(weight)
                selected_rows.append(selected_row)
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(
                        {
                            "ensemble": recipe.ensemble_id,
                            "outer": int(outer_fold),
                        },
                        refresh=False,
                    )
            if np.isnan(score).any():
                raise RuntimeError(f"{recipe.ensemble_id}: weighted OOF 누락")
            predictions.append(
                make_prediction_frame(outer_matrix, recipe, score)
            )
            manifest_rows.append(recipe_row(recipe))
    finally:
        if bar is not None:
            bar.close()
    return (
        predictions,
        manifest_rows,
        pd.concat(trial_frames, ignore_index=True),
        pd.DataFrame(selected_rows),
    )


def transformed_matrix(
    frame: pd.DataFrame,
    streams: tuple[str, ...],
    transform: str,
) -> np.ndarray:
    values = clipped_probability(frame[list(streams)].to_numpy(dtype=float))
    if transform == "raw":
        return values
    if transform == "logit":
        return logit(values)
    raise KeyError(transform)


def build_stacking_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_stacking_predictions(
    outer_matrix: pd.DataFrame,
    inner_oof: pd.DataFrame,
    recipes: list[EnsembleRecipe],
    *,
    progress_bar: bool,
) -> tuple[
    list[pd.DataFrame],
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
]:
    predictions: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fit_metric_rows: list[dict[str, Any]] = []
    bar = None
    total = len(recipes) * outer_matrix["outer_fold"].nunique()
    if progress_bar and step2.tqdm is not None:
        bar = step2.tqdm(
            total=total,
            desc="STEP4 stacking",
            unit="fold",
            dynamic_ncols=True,
        )
    try:
        for recipe in recipes:
            score = np.full(len(outer_matrix), np.nan, dtype=float)
            for outer_fold in sorted(outer_matrix["outer_fold"].unique()):
                train_frame = inner_oof.loc[
                    inner_oof["outer_fold"] == outer_fold
                ].reset_index(drop=True)
                valid_mask = (
                    outer_matrix["outer_fold"].to_numpy() == outer_fold
                )
                valid_frame = outer_matrix.loc[valid_mask]
                x_train = transformed_matrix(
                    train_frame,
                    recipe.streams,
                    recipe.transform,
                )
                y_train = train_frame["y_true"].to_numpy(dtype=int)
                x_valid = transformed_matrix(
                    valid_frame,
                    recipe.streams,
                    recipe.transform,
                )
                model = build_stacking_model()
                model.fit(x_train, y_train)
                train_probability = clipped_probability(
                    model.predict_proba(x_train)[:, 1]
                )
                score[valid_mask] = clipped_probability(
                    model.predict_proba(x_valid)[:, 1]
                )
                fit_metrics = step2.probability_metrics(
                    y_train,
                    train_probability,
                )
                fit_metric_rows.append(
                    {
                        "ensemble_id": recipe.ensemble_id,
                        "outer_fold": int(outer_fold),
                        "transform": recipe.transform,
                        **fit_metrics,
                    }
                )
                estimator = model.named_steps["model"]
                coefficient_row: dict[str, Any] = {
                    "ensemble_id": recipe.ensemble_id,
                    "outer_fold": int(outer_fold),
                    "transform": recipe.transform,
                    "intercept": float(estimator.intercept_[0]),
                }
                for stream, coefficient in zip(
                    recipe.streams,
                    estimator.coef_[0],
                ):
                    coefficient_row[f"coef__{stream}"] = float(coefficient)
                coefficient_rows.append(coefficient_row)
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(
                        {
                            "ensemble": recipe.ensemble_id,
                            "outer": int(outer_fold),
                        },
                        refresh=False,
                    )
            if np.isnan(score).any():
                raise RuntimeError(f"{recipe.ensemble_id}: stacking OOF 누락")
            predictions.append(
                make_prediction_frame(outer_matrix, recipe, score)
            )
            manifest_rows.append(recipe_row(recipe))
    finally:
        if bar is not None:
            bar.close()
    return (
        predictions,
        manifest_rows,
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(fit_metric_rows),
    )


def summary_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ensemble_id, part in predictions.groupby("ensemble_id", sort=False):
        first = part.iloc[0]
        metrics = step2.probability_metrics(part["y_true"], part["y_score"])
        fold_auprcs = [
            step2.safe_average_precision(fold["y_true"], fold["y_score"])
            for _, fold in part.groupby("outer_fold")
        ]
        row = {
            "ensemble_id": ensemble_id,
            "candidate_kind": first["candidate_kind"],
            "method": first["method"],
            "selection_role": first["selection_role"],
            "lgbm_variant": first["lgbm_variant"],
            "streams": first["streams"],
            "n_models": len(str(first["streams"]).split("|")),
            **metrics,
            "fold_auprc_mean": float(np.nanmean(fold_auprcs)),
            "fold_auprc_std": (
                float(np.nanstd(fold_auprcs, ddof=1))
                if len(fold_auprcs) > 1
                else float("nan")
            ),
        }
        hard_mask = part["sample_type"].isin(["Target_1", "Target_0A"])
        row["target0a_auprc"] = step2.safe_average_precision(
            part.loc[hard_mask, "y_true"],
            part.loc[hard_mask, "y_score"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fold_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ensemble_id, outer_fold), part in predictions.groupby(
        ["ensemble_id", "outer_fold"],
        sort=False,
    ):
        rows.append(
            {
                "ensemble_id": ensemble_id,
                "outer_fold": int(outer_fold),
                **step2.probability_metrics(
                    part["y_true"],
                    part["y_score"],
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ensemble_id, part in predictions.groupby("ensemble_id", sort=False):
        first = part.iloc[0]
        thresholds = step2.select_thresholds(
            part["y_true"],
            part["y_score"],
        )
        for operating_point, threshold in thresholds.items():
            rows.append(
                {
                    "ensemble_id": ensemble_id,
                    "selection_role": first["selection_role"],
                    "operating_point": operating_point,
                    "threshold_source": "full_oof_diagnostic",
                    **step2.classification_metrics_at_threshold(
                        part["y_true"],
                        part["y_score"],
                        threshold,
                    ),
                }
            )
    return pd.DataFrame(rows)


def top_risk_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ensemble_id, part in predictions.groupby("ensemble_id", sort=False):
        sorted_part = part.sort_values(
            "y_score",
            ascending=False,
        ).reset_index(drop=True)
        total_positive = int(sorted_part["y_true"].sum())
        base_rate = float(sorted_part["y_true"].mean())
        for fraction in step2.TOP_RISK_FRACTIONS:
            selected_n = max(1, int(math.ceil(len(sorted_part) * fraction)))
            selected = sorted_part.head(selected_n)
            captured = int(selected["y_true"].sum())
            precision = float(selected["y_true"].mean())
            rows.append(
                {
                    "ensemble_id": ensemble_id,
                    "top_fraction": fraction,
                    "selected_n": selected_n,
                    "positive_captured_n": captured,
                    "total_positive_n": total_positive,
                    "capture_rate": captured / total_positive,
                    "precision": precision,
                    "lift": precision / base_rate,
                }
            )
    return pd.DataFrame(rows)


def subgroup_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ensemble_id, part in predictions.groupby("ensemble_id", sort=False):
        for negative_type in ["Target_0A", "Target_0B1", "Target_0B2"]:
            subset = part.loc[
                part["sample_type"].isin(["Target_1", negative_type])
            ]
            rows.append(
                {
                    "ensemble_id": ensemble_id,
                    "subgroup_type": "sample_type_pair",
                    "subgroup_value": f"Target_1_vs_{negative_type}",
                    **step2.probability_metrics(
                        subset["y_true"],
                        subset["y_score"],
                    ),
                }
            )
        for climate_type, subset in part.groupby("climate_type"):
            rows.append(
                {
                    "ensemble_id": ensemble_id,
                    "subgroup_type": "climate_type",
                    "subgroup_value": str(climate_type),
                    **step2.probability_metrics(
                        subset["y_true"],
                        subset["y_score"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def validation_rows(
    predictions: pd.DataFrame,
    *,
    expected_n: int,
    outer_group_leakage_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ensemble_id, part in predictions.groupby("ensemble_id", sort=False):
        score = part["y_score"].to_numpy(dtype=float)
        rows.append(
            {
                "ensemble_id": ensemble_id,
                "expected_n": expected_n,
                "prediction_n": int(len(part)),
                "missing_prediction_n": int(max(0, expected_n - len(part))),
                "duplicate_sample_id_n": int(
                    part["sample_id"].duplicated().sum()
                ),
                "nan_score_n": int(np.isnan(score).sum()),
                "inf_score_n": int(np.isinf(score).sum()),
                "score_below_zero_n": int((score < 0).sum()),
                "score_above_one_n": int((score > 1).sum()),
                "outer_group_leakage_n": int(outer_group_leakage_n),
                "folds_without_positive_n": int(
                    sum(
                        int(fold["y_true"].sum()) == 0
                        for _, fold in part.groupby("outer_fold")
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def structural_validation_rows(
    *,
    outer_matrix: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    inner_oof: pd.DataFrame,
    selected_weights: pd.DataFrame,
    outer_group_leakage_n: int,
    inner_group_leakage_n: int,
) -> pd.DataFrame:
    checks = [
        {
            "check": "outer_input_duplicate_sample_id",
            "value": int(outer_matrix["sample_id"].duplicated().sum()),
            "expected": 0,
        },
        {
            "check": "outer_input_nan_base_prediction",
            "value": int(outer_matrix[OUTER_STREAMS].isna().sum().sum()),
            "expected": 0,
        },
        {
            "check": "inner_manifest_duplicate_sample_outer",
            "value": int(
                inner_manifest.duplicated(
                    subset=["sample_id", "outer_fold"]
                ).sum()
            ),
            "expected": 0,
        },
        {
            "check": "inner_oof_duplicate_sample_outer",
            "value": int(
                inner_oof.duplicated(
                    subset=["sample_id", "outer_fold"]
                ).sum()
            ),
            "expected": 0,
        },
        {
            "check": "inner_oof_nan_base_prediction",
            "value": int(inner_oof[INNER_STREAMS].isna().sum().sum()),
            "expected": 0,
        },
        {
            "check": "outer_group_leakage",
            "value": int(outer_group_leakage_n),
            "expected": 0,
        },
        {
            "check": "inner_group_leakage",
            "value": int(inner_group_leakage_n),
            "expected": 0,
        },
        {
            "check": "selected_weight_sum_violation",
            "value": int(
                (~np.isclose(selected_weights["weight_sum"], 1.0)).sum()
            )
            if not selected_weights.empty
            else 0,
            "expected": 0,
        },
    ]
    frame = pd.DataFrame(checks)
    frame["passed"] = frame["value"] == frame["expected"]
    return frame


def fixed_regression_checks(
    summary: pd.DataFrame,
    *,
    full_outer_folds: bool,
) -> pd.DataFrame:
    checks = [
        {
            "check": "fixed_baseline_lgbm_rf_mean_auprc",
            "ensemble_id": "fixed__baseline__lgbm_rf__mean",
            "expected": 0.696426,
            "tolerance": 0.001,
        },
        {
            "check": "fixed_dropisi_lgbm_rf_et_geomean_auprc",
            "ensemble_id": "fixed__drop_isi__lgbm_rf_et__geomean",
            "expected": 0.700652,
            "tolerance": 0.001,
        },
    ]
    rows: list[dict[str, Any]] = []
    by_id = summary.set_index("ensemble_id")
    for check in checks:
        actual = (
            float(by_id.loc[check["ensemble_id"], "auprc"])
            if check["ensemble_id"] in by_id.index
            else float("nan")
        )
        passed = (
            abs(actual - check["expected"]) <= check["tolerance"]
            if full_outer_folds and np.isfinite(actual)
            else True
        )
        rows.append(
            {
                **check,
                "actual": actual,
                "full_outer_folds_required": True,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows)


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
                "ensemble_id",
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
        result = result.merge(subset, on="ensemble_id", how="left")
    for fraction, prefix in [(0.10, "top10"), (0.20, "top20")]:
        subset = top_risk.loc[
            np.isclose(top_risk["top_fraction"], fraction),
            ["ensemble_id", "capture_rate", "precision"],
        ].rename(
            columns={
                "capture_rate": f"{prefix}_capture",
                "precision": f"{prefix}_precision",
            }
        )
        result = result.merge(subset, on="ensemble_id", how="left")
    return result


def add_fold_comparison(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    reference_id: str,
) -> pd.DataFrame:
    result = summary.copy()
    reference = fold_metrics.loc[
        fold_metrics["ensemble_id"] == reference_id,
        ["outer_fold", "auprc"],
    ].rename(columns={"auprc": "reference_fold_auprc"})
    rows: list[dict[str, Any]] = []
    for ensemble_id, part in fold_metrics.groupby("ensemble_id"):
        merged = part.merge(reference, on="outer_fold", how="left")
        delta = merged["auprc"] - merged["reference_fold_auprc"]
        rows.append(
            {
                "ensemble_id": ensemble_id,
                "fold_improvement_count": int((delta > 0).sum()),
                "fold_worst_delta": float(delta.min()),
                "fold_mean_delta": float(delta.mean()),
            }
        )
    return result.merge(pd.DataFrame(rows), on="ensemble_id", how="left")


def bootstrap_candidates(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    reference_id: str,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    probability = summary.loc[
        summary["selection_role"].isin(
            ["probability_candidate", "probability_reference"]
        )
    ].sort_values("auprc", ascending=False)
    candidate_ids = probability.head(10)["ensemble_id"].tolist()
    if reference_id not in candidate_ids:
        candidate_ids.append(reference_id)
    wide = None
    metadata = None
    for ensemble_id in candidate_ids:
        part = predictions.loc[
            predictions["ensemble_id"] == ensemble_id,
            METADATA_COLUMNS + ["y_score"],
        ].rename(columns={"y_score": ensemble_id})
        if wide is None:
            wide = part.copy()
            metadata = part[METADATA_COLUMNS].copy()
        else:
            wide = wide.merge(
                part[["sample_id", ensemble_id]],
                on="sample_id",
                how="inner",
                validate="one_to_one",
            )
    if wide is None or metadata is None:
        return pd.DataFrame()
    group_codes, groups = pd.factorize(wide["group_id"], sort=False)
    n_groups = len(groups)
    rng = np.random.default_rng(random_state)
    y_true = wide["y_true"].to_numpy(dtype=int)
    reference_score = wide[reference_id].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for ensemble_id in candidate_ids:
        score = wide[ensemble_id].to_numpy(dtype=float)
        deltas = np.empty(iterations, dtype=float)
        for iteration in range(iterations):
            counts = rng.multinomial(
                n_groups,
                np.full(n_groups, 1.0 / n_groups),
            )
            sample_weight = counts[group_codes]
            deltas[iteration] = (
                average_precision_score(
                    y_true,
                    score,
                    sample_weight=sample_weight,
                )
                - average_precision_score(
                    y_true,
                    reference_score,
                    sample_weight=sample_weight,
                )
            )
        rows.append(
            {
                "ensemble_id": ensemble_id,
                "reference_id": reference_id,
                "iterations": iterations,
                "delta_auprc_mean": float(deltas.mean()),
                "delta_auprc_ci_low": float(np.quantile(deltas, 0.025)),
                "delta_auprc_ci_high": float(np.quantile(deltas, 0.975)),
                "probability_delta_gt_zero": float((deltas > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def add_weight_collapse(
    summary: pd.DataFrame,
    selected_weights: pd.DataFrame,
) -> pd.DataFrame:
    result = summary.copy()
    if selected_weights.empty:
        result["weight_collapse_fold_count"] = 0
        result["weight_collapse"] = False
        return result
    collapsed = (
        selected_weights.assign(
            collapsed=selected_weights["max_weight"] >= 0.95
        )
        .groupby("ensemble_id", as_index=False)
        .agg(weight_collapse_fold_count=("collapsed", "sum"))
    )
    result = result.merge(collapsed, on="ensemble_id", how="left")
    result["weight_collapse_fold_count"] = (
        result["weight_collapse_fold_count"].fillna(0).astype(int)
    )
    result["weight_collapse"] = (
        result["weight_collapse_fold_count"] >= 3
    )
    return result


def counterpart_id(ensemble_id: str) -> str:
    return ensemble_id.replace("__drop_isi__", "__baseline__")


def add_dropisi_gate(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    result = summary.copy()
    result["dropisi_gate_pass"] = True
    by_id = result.set_index("ensemble_id")
    for index, row in result.loc[
        result["lgbm_variant"] == "drop_isi"
    ].iterrows():
        baseline_id = counterpart_id(str(row["ensemble_id"]))
        if baseline_id not in by_id.index:
            result.loc[index, "dropisi_gate_pass"] = False
            continue
        baseline_auprc = float(by_id.loc[baseline_id, "auprc"])
        drop_auprc = float(row["auprc"])
        drop_fold = fold_metrics.loc[
            fold_metrics["ensemble_id"] == row["ensemble_id"],
            ["outer_fold", "auprc"],
        ].rename(columns={"auprc": "drop_auprc"})
        baseline_fold = fold_metrics.loc[
            fold_metrics["ensemble_id"] == baseline_id,
            ["outer_fold", "auprc"],
        ].rename(columns={"auprc": "baseline_auprc"})
        paired = drop_fold.merge(baseline_fold, on="outer_fold", how="inner")
        fold_wins = int(
            (paired["drop_auprc"] > paired["baseline_auprc"]).sum()
        )
        result.loc[index, "dropisi_gate_pass"] = (
            drop_auprc > baseline_auprc and fold_wins >= 3
        )
    return result


def select_final_candidate(
    summary: pd.DataFrame,
    *,
    reference_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = summary.copy()
    reference = result.loc[result["ensemble_id"] == reference_id].iloc[0]
    result["delta_auprc_vs_best_single"] = (
        result["auprc"] - float(reference["auprc"])
    )
    result["delta_target0a_vs_best_single"] = (
        result["target0a_auprc"] - float(reference["target0a_auprc"])
    )
    result["delta_top10_vs_best_single"] = (
        result["top10_capture"] - float(reference["top10_capture"])
    )
    result["delta_recall70_selected_vs_best_single"] = (
        result["recall70_selected_rate"]
        - float(reference["recall70_selected_rate"])
    )
    probability_candidate = result["selection_role"] == "probability_candidate"
    improvement = result["delta_auprc_vs_best_single"] >= 0.002
    operational_improvement = (
        (result["delta_auprc_vs_best_single"] > 0)
        & (
            (
                -result["delta_recall70_selected_vs_best_single"]
                >= 0.005
            )
            | (result["delta_top10_vs_best_single"] >= 0.005)
        )
    )
    fold_ok = (
        (result["fold_improvement_count"] >= 3)
        & (result["fold_worst_delta"] >= -0.01)
    )
    target0a_ok = result["delta_target0a_vs_best_single"] >= -0.002
    brier_ok = result["brier"] <= float(reference["brier"]) * 1.05
    log_loss_ok = (
        result["log_loss"] <= float(reference["log_loss"]) * 1.05
    )
    weight_ok = ~result["weight_collapse"]
    dropisi_ok = result["dropisi_gate_pass"]
    result["acceptance_pass"] = (
        probability_candidate
        & (improvement | operational_improvement)
        & fold_ok
        & target0a_ok
        & brier_ok
        & log_loss_ok
        & weight_ok
        & dropisi_ok
    )
    result["acceptance_reason"] = ""
    checks = [
        (probability_candidate, "probability_candidate"),
        (improvement | operational_improvement, "performance"),
        (fold_ok, "fold"),
        (target0a_ok, "target0a"),
        (brier_ok, "brier"),
        (log_loss_ok, "log_loss"),
        (weight_ok, "weight_collapse"),
        (dropisi_ok, "dropisi_gate"),
    ]
    for index in result.index:
        failed = [name for mask, name in checks if not bool(mask.loc[index])]
        result.loc[index, "acceptance_reason"] = (
            "PASS" if not failed else "FAIL:" + "|".join(failed)
        )

    eligible = result.loc[result["acceptance_pass"]].copy()
    if eligible.empty:
        selected = result.loc[
            result["ensemble_id"] == f"single__{STREAM_LGBM}"
        ].iloc[0]
        selection_mode = "fallback_step2_baseline"
    else:
        eligible = eligible.sort_values(
            ["auprc", "target0a_auprc", "top10_capture", "brier"],
            ascending=[False, False, False, True],
        )
        top = eligible.iloc[0]
        comparable = eligible.loc[
            (float(top["auprc"]) - eligible["auprc"] < 0.001)
            & (
                (float(top["top10_capture"]) - eligible["top10_capture"]).abs()
                < 0.005
            )
            & (
                (
                    float(top["recall70_selected_rate"])
                    - eligible["recall70_selected_rate"]
                ).abs()
                < 0.005
            )
        ].copy()
        complexity = {"fixed": 1, "weighted": 2, "stacking": 3}
        comparable["complexity_rank"] = comparable["candidate_kind"].map(
            complexity
        ).fillna(9)
        selected = comparable.sort_values(
            ["complexity_rank", "n_models", "auprc"],
            ascending=[True, True, False],
        ).iloc[0]
        selection_mode = "accepted_ensemble"

    result["selected_for_step5"] = (
        result["ensemble_id"] == selected["ensemble_id"]
    )
    payload = {
        "selected_ensemble_id": str(selected["ensemble_id"]),
        "selection_mode": selection_mode,
        "candidate_kind": str(selected["candidate_kind"]),
        "method": str(selected["method"]),
        "selection_role": str(selected["selection_role"]),
        "lgbm_variant": str(selected["lgbm_variant"]),
        "streams": str(selected["streams"]).split("|"),
        "n_models": int(selected["n_models"]),
        "metrics": {
            "auprc": float(selected["auprc"]),
            "auroc": float(selected["auroc"]),
            "brier": float(selected["brier"]),
            "log_loss": float(selected["log_loss"]),
            "target0a_auprc": float(selected["target0a_auprc"]),
            "top10_capture": float(selected["top10_capture"]),
            "top20_capture": float(selected["top20_capture"]),
            "recall70_precision": float(selected["recall70_precision"]),
            "recall70_selected_rate": float(
                selected["recall70_selected_rate"]
            ),
        },
        "reference_best_single_id": reference_id,
        "delta_auprc_vs_best_single": float(
            selected["delta_auprc_vs_best_single"]
        ),
        "calibration_applied": False,
        "next_stage": "Step5 calibration and final threshold",
    }
    return result, payload


def write_individual_oof(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    for ensemble_id, part in predictions.groupby("ensemble_id", sort=False):
        part.to_csv(
            output_dir / f"oof__{ensemble_id}.csv",
            index=False,
            encoding="utf-8-sig",
        )


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
        else root
        / "jsw"
        / "Analysis"
        / "new_machine_learning"
        / "outputs"
        / "step1_single_models"
    )
    step2_output_dir = (
        Path(args.step2_output_dir)
        if args.step2_output_dir
        else root
        / "jsw"
        / "Analysis"
        / "new_machine_learning"
        / "outputs"
        / "step2_tuned_single_models"
    )
    step3_output_dir = (
        Path(args.step3_output_dir)
        if args.step3_output_dir
        else root
        / "jsw"
        / "Analysis"
        / "new_machine_learning"
        / "outputs"
        / "step3_feature_ablation"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "jsw"
        / "Analysis"
        / "new_machine_learning"
        / "outputs"
        / "step4_ensemble"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    step2.log(f"입력 데이터 로드: {data_path}")
    data = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    data[step2.TARGET_COL] = data[step2.TARGET_COL].astype(int)
    streams = build_stream_specs(
        step1_output_dir=step1_output_dir,
        step2_output_dir=step2_output_dir,
        step3_output_dir=step3_output_dir,
    )
    feature_sets = step2.build_feature_sets()
    for stream in streams.values():
        if not stream.oof_path.exists():
            raise FileNotFoundError(stream.oof_path)
        features = stream_features(stream, feature_sets)
        missing = [feature for feature in features if feature not in data.columns]
        if missing:
            raise KeyError(f"{stream.stream_id} 피처 누락: {missing}")
        if stream.selected_params_path is not None:
            load_selected_params(stream)

    outer_matrix = load_outer_matrix(streams)
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
    selected_outer_folds = [fold for fold, _, _ in outer_splits]
    outer_matrix = outer_matrix.loc[
        outer_matrix["outer_fold"].isin(selected_outer_folds)
    ].reset_index(drop=True)
    outer_group_leakage_n = step2.check_group_leakage(
        data,
        outer_splits,
    )
    inner_manifest = make_inner_manifest(
        data,
        outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
    )
    inner_group_leakage_n = validate_inner_group_leakage(inner_manifest)
    if inner_group_leakage_n:
        raise RuntimeError(f"inner group leakage={inner_group_leakage_n}")

    fixed_recipes = single_recipe_definitions() + simple_recipe_definitions()
    weighted_recipes = weighted_recipe_definitions()
    stacking_recipes = stacking_recipe_definitions()
    planned_base_fits = (
        len(INNER_STREAMS)
        * len(outer_splits)
        * args.inner_splits
    )
    run_config = {
        "script": str(Path(__file__).resolve()),
        "started_at": step2.timestamp(),
        "data_path": str(data_path),
        "step1_output_dir": str(step1_output_dir),
        "step2_output_dir": str(step2_output_dir),
        "step3_output_dir": str(step3_output_dir),
        "output_dir": str(output_dir),
        "n_rows": int(len(data)),
        "positive_n": int(data[step2.TARGET_COL].sum()),
        "outer_fold_count": len(outer_splits),
        "inner_splits": args.inner_splits,
        "inner_stream_count": len(INNER_STREAMS),
        "planned_base_fits": planned_base_fits,
        "fixed_candidate_count": len(fixed_recipes),
        "weighted_candidate_count": len(weighted_recipes),
        "stacking_candidate_count": len(stacking_recipes),
        "weight_trials_per_outer": args.weight_trials,
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "outer_group_leakage_n": outer_group_leakage_n,
        "inner_group_leakage_n": inner_group_leakage_n,
        "lockbox_accessed": False,
        "calibration_applied": False,
    }
    step2.write_json(
        output_dir
        / (
            "run_manifest__check_config.json"
            if args.check_config
            else "run_manifest__all.json"
        ),
        run_config,
    )
    outer_matrix.to_csv(
        output_dir / "ensemble_input_oof_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    inner_manifest.to_csv(
        output_dir / "inner_split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    step2.log(
        f"Step4 계획: outer_folds={len(outer_splits)}, "
        f"inner={args.inner_splits}, base_streams={len(INNER_STREAMS)}, "
        f"planned_base_fits={planned_base_fits}"
    )
    step2.log(
        f"후보: fixed={len(fixed_recipes)}, weighted={len(weighted_recipes)}, "
        f"stacking={len(stacking_recipes)}"
    )
    step2.log("lockbox 접근 없음, calibration 수행 없음")
    if args.check_config:
        step2.log("--check-config 지정: 학습 없이 종료합니다.")
        return

    fixed_predictions, manifest_rows = make_fixed_predictions(outer_matrix)
    inner_oof_path = output_dir / "inner_base_oof.csv"
    if args.resume_existing and complete_inner_oof(
        inner_oof_path,
        inner_manifest,
    ):
        step2.log("기존 inner base OOF 재사용")
        inner_oof = pd.read_csv(
            inner_oof_path,
            encoding="utf-8-sig",
            low_memory=False,
        )
    else:
        inner_oof = generate_inner_base_oof(
            data,
            inner_manifest,
            streams=streams,
            feature_sets=feature_sets,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            progress_bar=(
                not args.no_progress_bar and step2.tqdm is not None
            ),
        )
        inner_oof.to_csv(
            inner_oof_path,
            index=False,
            encoding="utf-8-sig",
        )

    (
        weighted_predictions,
        weighted_manifest,
        weight_trials,
        selected_weights,
    ) = make_weighted_predictions(
        outer_matrix,
        inner_oof,
        weighted_recipes,
        n_trials=args.weight_trials,
        random_state=args.random_state,
        progress_bar=(
            not args.no_progress_bar and step2.tqdm is not None
        ),
    )
    (
        stacking_predictions,
        stacking_manifest,
        stacking_coefficients,
        stacking_fit_metrics,
    ) = make_stacking_predictions(
        outer_matrix,
        inner_oof,
        stacking_recipes,
        progress_bar=(
            not args.no_progress_bar and step2.tqdm is not None
        ),
    )
    all_predictions = pd.concat(
        fixed_predictions + weighted_predictions + stacking_predictions,
        ignore_index=True,
    )
    all_manifest = pd.DataFrame(
        manifest_rows + weighted_manifest + stacking_manifest
    )

    summary = summary_rows(all_predictions)
    folds = fold_metric_rows(all_predictions)
    thresholds = threshold_metric_rows(all_predictions)
    top_risk = top_risk_rows(all_predictions)
    subgroups = subgroup_rows(all_predictions)
    validations = validation_rows(
        all_predictions,
        expected_n=len(outer_matrix),
        outer_group_leakage_n=outer_group_leakage_n,
    )
    structural_validations = structural_validation_rows(
        outer_matrix=outer_matrix,
        inner_manifest=inner_manifest,
        inner_oof=inner_oof,
        selected_weights=selected_weights,
        outer_group_leakage_n=outer_group_leakage_n,
        inner_group_leakage_n=inner_group_leakage_n,
    )
    summary = add_operating_metrics(summary, thresholds, top_risk)
    best_single_id = str(
        summary.loc[
            summary["candidate_kind"] == "single"
        ].sort_values("auprc", ascending=False).iloc[0]["ensemble_id"]
    )
    summary = add_fold_comparison(
        summary,
        folds,
        reference_id=best_single_id,
    )
    summary = add_weight_collapse(summary, selected_weights)
    summary = add_dropisi_gate(summary, folds)
    regression_checks = fixed_regression_checks(
        summary,
        full_outer_folds=len(outer_splits) == 5,
    )
    if not bool(structural_validations["passed"].all()):
        failed = structural_validations.loc[
            ~structural_validations["passed"]
        ]
        raise RuntimeError(
            "Step4 구조 검증 실패: "
            + failed[["check", "value", "expected"]].to_json(
                orient="records",
                force_ascii=False,
            )
        )
    if not bool(regression_checks["passed"].all()):
        failed = regression_checks.loc[~regression_checks["passed"]]
        raise RuntimeError(
            "Step4 고정 조합 회귀검사 실패: "
            + failed[
                ["check", "actual", "expected", "tolerance"]
            ].to_json(orient="records", force_ascii=False)
        )
    bootstrap = bootstrap_candidates(
        all_predictions,
        summary,
        reference_id=best_single_id,
        iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    summary, selected_payload = select_final_candidate(
        summary,
        reference_id=best_single_id,
    )

    all_manifest.to_csv(
        output_dir / "ensemble_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_manifest.loc[
        all_manifest["candidate_kind"] == "fixed"
    ].to_csv(
        output_dir / "fixed_ensemble_recipes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weight_trials.to_csv(
        output_dir / "optuna_weight_trials.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_weights.to_csv(
        output_dir / "selected_weights_by_outer_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stacking_coefficients.to_csv(
        output_dir / "stacking_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stacking_fit_metrics.to_csv(
        output_dir / "stacking_fit_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_predictions.to_csv(
        output_dir / "all_ensemble_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_individual_oof(all_predictions, output_dir)
    summary.sort_values(
        ["selected_for_step5", "auprc"],
        ascending=[False, False],
    ).to_csv(
        output_dir / "summary__step4_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds.to_csv(
        output_dir / "fold_metrics__step4_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )
    thresholds.to_csv(
        output_dir / "threshold_metrics__step4_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top_risk.to_csv(
        output_dir / "top_risk_metrics__step4_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subgroups.to_csv(
        output_dir / "subgroup_metrics__step4_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        output_dir / "bootstrap__step4_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validations.to_csv(
        output_dir / "validation_checks__step4_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )
    structural_validations.to_csv(
        output_dir / "validation_checks__step4_structure.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regression_checks.to_csv(
        output_dir / "regression_checks__step4_fixed_ensembles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    step2.write_json(
        output_dir / "selected_ensemble_candidate.json",
        selected_payload,
    )

    final_manifest = dict(run_config)
    final_manifest.update(
        {
            "finished_at": step2.timestamp(),
            "elapsed_seconds": time.perf_counter() - started,
            "actual_candidate_count": int(summary.shape[0]),
            "best_single_id": best_single_id,
            "selected_ensemble_id": selected_payload[
                "selected_ensemble_id"
            ],
            "selected_for_step5": True,
        }
    )
    step2.write_json(output_dir / "run_manifest__all.json", final_manifest)
    step2.log(
        f"STEP4 DONE | candidates={len(summary)} | "
        f"best_single={best_single_id} | "
        f"selected={selected_payload['selected_ensemble_id']} | "
        f"elapsed={step2.format_seconds(final_manifest['elapsed_seconds'])}"
    )


if __name__ == "__main__":
    main()
