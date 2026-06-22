from __future__ import annotations

import argparse
import builtins
import functools
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import stage1_ml_screening as s1


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
print = functools.partial(builtins.print, flush=True)

RANDOM_STATE = 20260620
STAGE1_BEST_AUPRC = 0.2983967715401457
STAGE2_SUCCESS_AUPRC = STAGE1_BEST_AUPRC + 0.005

OPTUNA_DIR = s1.OUTPUT_DIR / "optuna"
INTERPRET_DIR = s1.OUTPUT_DIR / "interpretation"
for directory in [OPTUNA_DIR, INTERPRET_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

OPTUNA_DB = OPTUNA_DIR / "ml_stage2_optuna_studies.sqlite3"
RUN_LABEL = "full"


@dataclass(frozen=True)
class Stage2Candidate:
    feature_set: str
    model: str
    role: str
    priority: int
    stage1_auprc: float
    reason: str

    @property
    def candidate_id(self) -> str:
        return f"{self.feature_set}__{self.model}".replace(" ", "_")


MAIN_CANDIDATES = [
    Stage2Candidate("PLUS_LANDCOVER", "HistGradientBoosting", "main", 1, 0.2983967715, "1차 전체 최고"),
    Stage2Candidate("PLUS_LANDCOVER", "LightGBM", "main", 2, 0.287786, "AUROC와 recall 강점"),
    Stage2Candidate("PLUS_LANDCOVER_RULES_ANOVA_PROXY", "HistGradientBoosting", "main", 3, 0.290154, "규칙/ANOVA proxy 1위권"),
    Stage2Candidate("PLUS_LANDCOVER_RULES_ANOVA_PROXY", "XGBoost", "main", 4, 0.280135, "XGBoost 대표 후보"),
    Stage2Candidate("PLUS_LANDCOVER_RULES_ANOVA_PROXY", "RandomForest", "main", 5, 0.282546, "안정적 bagging 후보"),
]

AUXILIARY_CANDIDATES = [
    Stage2Candidate("M1", "RandomForest", "auxiliary", 6, 0.260139, "M1/M2/M3 진단 기준선"),
    Stage2Candidate("PLUS_LANDCOVER", "XGBoost", "auxiliary", 7, 0.199495, "XGBoost feature-set 민감도 확인"),
]

ALL_STAGE2_CANDIDATES = MAIN_CANDIDATES + AUXILIARY_CANDIDATES


def storage_url() -> str:
    return f"sqlite:///{OPTUNA_DB.as_posix()}"


def categorical_for_features(features: list[str], categorical: list[str]) -> list[str]:
    return [col for col in categorical if col in features]


def make_stage2_pipeline(
    features: list[str],
    categorical: list[str],
    model_name: str,
    params: dict[str, Any],
    y_train: pd.Series | np.ndarray,
    seed_offset: int = 0,
) -> Pipeline:
    categorical_used = categorical_for_features(features, categorical)
    numeric = [col for col in features if col not in categorical_used]
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_used),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    estimator = build_stage2_estimator(model_name, params, y_train, seed_offset)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def build_stage2_estimator(
    model_name: str,
    params: dict[str, Any],
    y_train: pd.Series | np.ndarray,
    seed_offset: int = 0,
) -> Any:
    random_state = RANDOM_STATE + seed_offset

    if model_name == "HistGradientBoosting":
        max_depth = params.get("max_depth")
        if max_depth == "none":
            max_depth = None
        return HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            max_depth=max_depth,
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            max_bins=int(params["max_bins"]),
            class_weight=params["class_weight"],
            early_stopping=bool(params["early_stopping"]),
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )

    if model_name == "LightGBM":
        if s1.LGBMClassifier is None:
            raise RuntimeError(f"LightGBM unavailable: {s1.LGBM_IMPORT_ERROR}")
        max_depth = int(params["max_depth"])
        return s1.LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            max_depth=max_depth,
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            min_split_gain=float(params["min_split_gain"]),
            class_weight=params["class_weight"],
            objective="binary",
            n_jobs=4,
            random_state=random_state,
            verbose=-1,
        )

    if model_name == "XGBoost":
        if s1.XGBClassifier is None:
            raise RuntimeError(f"XGBoost unavailable: {s1.XGB_IMPORT_ERROR}")
        scale_pos_weight = resolve_scale_pos_weight(params["scale_pos_weight_mode"], y_train)
        return s1.XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            gamma=float(params["gamma"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            max_delta_step=float(params["max_delta_step"]),
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=4,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )

    if model_name == "RandomForest":
        bootstrap = bool(params["bootstrap"])
        max_samples = float(params["max_samples"]) if bootstrap else None
        max_depth = params["max_depth"]
        if max_depth == "none":
            max_depth = None
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=max_depth,
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            bootstrap=bootstrap,
            max_samples=max_samples,
            class_weight=params["class_weight"],
            n_jobs=4,
            random_state=random_state,
        )

    raise KeyError(model_name)


def resolve_scale_pos_weight(mode: str, y_train: pd.Series | np.ndarray) -> float:
    base = s1.positive_scale_weight(y_train)
    if mode == "none":
        return 1.0
    if mode == "auto_0p75":
        return max(1.0, base * 0.75)
    if mode == "auto":
        return base
    if mode == "auto_1p25":
        return base * 1.25
    return float(mode)


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "HistGradientBoosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 240, step=20),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 47, step=4),
            "max_depth": trial.suggest_categorical("max_depth", ["none", 3, 4, 5, 6, 8]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 16, 120, step=4),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 3.0, log=True),
            "max_bins": trial.suggest_categorical("max_bins", [64, 128, 255]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "early_stopping": trial.suggest_categorical("early_stopping", [False, True]),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 260, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 79, step=4),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 8]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 140, step=5),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 15.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 260, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 25.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 25.0, log=True),
            "scale_pos_weight_mode": trial.suggest_categorical(
                "scale_pos_weight_mode", ["none", "auto_0p75", "auto", "auto_1p25"]
            ),
            "max_delta_step": trial.suggest_categorical("max_delta_step", [0, 1, 3, 5]),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 140, 280, step=20),
            "max_depth": trial.suggest_categorical("max_depth", ["none", 8, 12, 16, 24]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 24),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.35, 0.50, 0.75]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_samples": trial.suggest_float("max_samples", 0.55, 1.0),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        }
    raise KeyError(model_name)


def objective_factory(
    candidate: Stage2Candidate,
    indexed: pd.DataFrame,
    inner_part: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    outer_fold: int,
):
    inner_part = inner_part.set_index("샘플ID")
    inner_folds = sorted(inner_part["inner_fold"].unique())

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, candidate.model)
        scores = []
        for inner_fold in inner_folds:
            inner_val_ids = inner_part.index[inner_part["inner_fold"].eq(inner_fold)].tolist()
            inner_train_ids = inner_part.index[~inner_part["inner_fold"].eq(inner_fold)].tolist()
            pipeline = make_stage2_pipeline(
                features,
                categorical,
                candidate.model,
                params,
                indexed.loc[inner_train_ids, "Target"],
                seed_offset=int(outer_fold * 10000 + trial.number),
            )
            pipeline.fit(indexed.loc[inner_train_ids, features], indexed.loc[inner_train_ids, "Target"])
            probability = s1.predict_probability(pipeline, indexed.loc[inner_val_ids, features])
            scores.append(float(average_precision_score(indexed.loc[inner_val_ids, "Target"], probability)))
        trial.set_user_attr("inner_scores", json.dumps(scores))
        trial.set_user_attr("inner_mean_auprc", float(np.mean(scores)))
        trial.set_user_attr("inner_std_auprc", float(np.std(scores, ddof=0)))
        return float(np.mean(scores))

    return objective


def study_name(candidate: Stage2Candidate, outer_fold: int) -> str:
    return f"stage2_{RUN_LABEL}__{candidate.candidate_id}__outer{outer_fold}"


def tune_study(
    candidate: Stage2Candidate,
    indexed: pd.DataFrame,
    inner: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    outer_fold: int,
    target_total_trials: int,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE + outer_fold + candidate.priority * 100)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name(candidate, outer_fold),
        storage=storage_url(),
        load_if_exists=True,
        sampler=sampler,
    )
    completed = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, target_total_trials - completed)
    if remaining > 0:
        inner_part = inner.loc[inner["outer_fold"].eq(outer_fold)].copy()
        objective = objective_factory(candidate, indexed, inner_part, features, categorical, outer_fold)
        study.optimize(objective, n_trials=remaining, show_progress_bar=False, gc_after_trial=True)
    return study


def run_tuning_phase(
    phase: str,
    candidates: list[Stage2Candidate],
    indexed: pd.DataFrame,
    inner: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
    outer_folds: list[int],
    target_total_trials: int,
) -> dict[tuple[str, int], optuna.Study]:
    studies = {}
    for candidate in candidates:
        features = feature_sets[candidate.feature_set]
        for outer_fold in outer_folds:
            print(
                f"Stage2 {phase}: {candidate.feature_set} / {candidate.model} / outer {outer_fold} "
                f"target_trials={target_total_trials}"
            )
            studies[(candidate.candidate_id, outer_fold)] = tune_study(
                candidate,
                indexed,
                inner,
                features,
                categorical,
                outer_fold,
                target_total_trials,
            )
    return studies


def candidate_inner_summary(candidates: list[Stage2Candidate], outer_folds: list[int]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        values = []
        for outer_fold in outer_folds:
            study = optuna.load_study(study_name=study_name(candidate, outer_fold), storage=storage_url())
            values.append(float(study.best_value))
        rows.append(
            {
                "feature_set": candidate.feature_set,
                "model": candidate.model,
                "role": candidate.role,
                "priority": candidate.priority,
                "stage1_auprc": candidate.stage1_auprc,
                "inner_best_mean_auprc": float(np.mean(values)),
                "inner_best_std_auprc": float(np.std(values, ddof=0)),
                "outer_fold_best_values": "|".join(f"{value:.6f}" for value in values),
            }
        )
    return pd.DataFrame(rows).sort_values(["inner_best_mean_auprc", "stage1_auprc"], ascending=[False, False])


def export_trials(candidates: list[Stage2Candidate], outer_folds: list[int]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        for outer_fold in outer_folds:
            study = optuna.load_study(study_name=study_name(candidate, outer_fold), storage=storage_url())
            for trial in study.trials:
                rows.append(
                    {
                        "feature_set": candidate.feature_set,
                        "model": candidate.model,
                        "role": candidate.role,
                        "outer_fold": outer_fold,
                        "trial_number": trial.number,
                        "state": trial.state.name,
                        "value": trial.value,
                        "inner_mean_auprc": trial.user_attrs.get("inner_mean_auprc", np.nan),
                        "inner_std_auprc": trial.user_attrs.get("inner_std_auprc", np.nan),
                        "params_json": json.dumps(trial.params, ensure_ascii=False, sort_keys=True),
                    }
                )
    return pd.DataFrame(rows)


def selected_params_table(candidates: list[Stage2Candidate], outer_folds: list[int]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        for outer_fold in outer_folds:
            study = optuna.load_study(study_name=study_name(candidate, outer_fold), storage=storage_url())
            rows.append(
                {
                    "feature_set": candidate.feature_set,
                    "feature_group": s1.FEATURE_SET_META[candidate.feature_set]["feature_group"],
                    "model": candidate.model,
                    "role": candidate.role,
                    "outer_fold": outer_fold,
                    "selected_config_id": f"stage2_outer{outer_fold}_trial{study.best_trial.number}",
                    "best_trial_number": study.best_trial.number,
                    "inner_best_auprc": float(study.best_value),
                    "inner_std_auprc": study.best_trial.user_attrs.get("inner_std_auprc", np.nan),
                    "params_json": json.dumps(study.best_params, ensure_ascii=False, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def train_stage2_oof(
    data: pd.DataFrame,
    outer: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = data.set_index("샘플ID", drop=False)
    oof_rows = []
    fold_rows = []
    importance_rows = []

    for row in selected.itertuples(index=False):
        feature_set = str(row.feature_set)
        model_name = str(row.model)
        outer_fold = int(row.outer_fold)
        selected_config_id = str(row.selected_config_id)
        params = json.loads(row.params_json)
        features = feature_sets[feature_set]
        train_ids = outer.loc[~outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
        val_ids = outer.loc[outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()

        pipeline = make_stage2_pipeline(
            features,
            categorical,
            model_name,
            params,
            indexed.loc[train_ids, "Target"],
            seed_offset=outer_fold * 1000,
        )
        pipeline.fit(indexed.loc[train_ids, features], indexed.loc[train_ids, "Target"])
        train_probability = s1.predict_probability(pipeline, indexed.loc[train_ids, features])
        val_probability = s1.predict_probability(pipeline, indexed.loc[val_ids, features])

        for dataset_name, ids, probability in [
            ("train", train_ids, train_probability),
            ("validation", val_ids, val_probability),
        ]:
            fold_rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": s1.FEATURE_SET_META[feature_set]["feature_group"],
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "dataset": dataset_name,
                    "score_type": "raw",
                    "selected_config_id": selected_config_id,
                    **s1.probability_metrics(indexed.loc[ids, "Target"], probability),
                }
            )

        for sample_id, probability in zip(val_ids, val_probability):
            sample = indexed.loc[sample_id]
            oof_rows.append(
                {
                    "샘플ID": sample_id,
                    "feature_set": feature_set,
                    "feature_group": s1.FEATURE_SET_META[feature_set]["feature_group"],
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "Target": int(sample["Target"]),
                    "샘플유형": sample["샘플유형"],
                    "기후지형유형": sample["기후지형유형"],
                    "selected_config_id": selected_config_id,
                    "score_raw": float(probability),
                    "score_sigmoid": np.nan,
                    "score_isotonic": np.nan,
                    "score_calibrated": float(probability),
                    "calibration_method": "raw",
                    "run_status": "OK",
                }
            )

        importance_rows.extend(extract_importance_rows(pipeline, feature_set, model_name, outer_fold, selected_config_id))
        print(
            f"Stage2 OOF: {feature_set} / {model_name} / outer {outer_fold}, "
            f"val AUPRC={average_precision_score(indexed.loc[val_ids, 'Target'], val_probability):.4f}"
        )

    return pd.DataFrame(oof_rows), pd.DataFrame(fold_rows), pd.DataFrame(importance_rows)


def extract_importance_rows(
    pipeline: Pipeline,
    feature_set: str,
    model_name: str,
    outer_fold: int,
    selected_config_id: str,
) -> list[dict[str, Any]]:
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return [
            {
                "feature_set": feature_set,
                "model": model_name,
                "outer_fold": outer_fold,
                "selected_config_id": selected_config_id,
                "importance_type": "not_available",
                "feature": "",
                "importance": np.nan,
            }
        ]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = np.asarray(estimator.feature_importances_, dtype=float)
    total = values.sum()
    normalized = values / total if total > 0 else values
    rows = []
    for feature, value, norm in zip(names, values, normalized):
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "outer_fold": outer_fold,
                "selected_config_id": selected_config_id,
                "importance_type": "built_in",
                "feature": feature,
                "importance": float(value),
                "importance_normalized": float(norm),
            }
        )
    return rows


def fit_calibrators(raw_probability: np.ndarray, y_true: np.ndarray) -> tuple[LogisticRegression, IsotonicRegression]:
    raw_probability = s1.clipped_probability(raw_probability)
    y_true = np.asarray(y_true).astype(int)
    sigmoid = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    sigmoid.fit(raw_probability.reshape(-1, 1), y_true)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_probability, y_true)
    return sigmoid, isotonic


def apply_stage2_calibration(
    data: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    raw_comparison: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    indexed = data.set_index("샘플ID", drop=False)
    predictions = predictions.copy()
    raw = raw_comparison.loc[raw_comparison["score_type"].eq("raw") & raw_comparison["run_status"].eq("OK")].head(top_n)
    selected_lookup = selected.set_index(["feature_set", "model", "outer_fold"])

    for candidate in raw.itertuples(index=False):
        feature_set = str(candidate.feature_set)
        model_name = str(candidate.model)
        features = feature_sets[feature_set]
        print(f"Stage2 calibration: {feature_set} / {model_name}")
        for outer_fold in sorted(outer["outer_fold"].unique()):
            params = json.loads(selected_lookup.loc[(feature_set, model_name, outer_fold), "params_json"])
            inner_part = inner.loc[inner["outer_fold"].eq(outer_fold)].set_index("샘플ID")
            train_ids = inner_part.index.tolist()
            inner_oof = pd.Series(index=train_ids, dtype=float)
            for inner_fold in sorted(inner_part["inner_fold"].unique()):
                inner_val_ids = inner_part.index[inner_part["inner_fold"].eq(inner_fold)].tolist()
                inner_train_ids = inner_part.index[~inner_part["inner_fold"].eq(inner_fold)].tolist()
                pipeline = make_stage2_pipeline(
                    features,
                    categorical,
                    model_name,
                    params,
                    indexed.loc[inner_train_ids, "Target"],
                    seed_offset=outer_fold * 1000 + int(inner_fold),
                )
                pipeline.fit(indexed.loc[inner_train_ids, features], indexed.loc[inner_train_ids, "Target"])
                inner_oof.loc[inner_val_ids] = s1.predict_probability(pipeline, indexed.loc[inner_val_ids, features])
            sigmoid, isotonic = fit_calibrators(inner_oof.to_numpy(), indexed.loc[train_ids, "Target"].to_numpy())
            mask = (
                predictions["feature_set"].eq(feature_set)
                & predictions["model"].eq(model_name)
                & predictions["outer_fold"].eq(outer_fold)
            )
            raw_val = predictions.loc[mask, "score_raw"].to_numpy()
            predictions.loc[mask, "score_sigmoid"] = s1.clipped_probability(
                sigmoid.predict_proba(raw_val.reshape(-1, 1))[:, 1]
            )
            predictions.loc[mask, "score_isotonic"] = s1.clipped_probability(isotonic.predict(raw_val))

    comparison = s1.model_comparison(predictions)
    calibration_rows = []
    for (feature_set, model), part in comparison.loc[comparison["score_type"].isin(["sigmoid", "isotonic"])].groupby(
        ["feature_set", "model"], observed=True
    ):
        best = part.sort_values(["brier", "log_loss"], ascending=[True, True]).iloc[0]
        calibration_rows.append(
            {
                "feature_set": feature_set,
                "feature_group": best["feature_group"],
                "model": model,
                "best_calibration_method": best["score_type"],
                "best_calibration_brier": best["brier"],
                "best_calibration_log_loss": best["log_loss"],
                "raw_auprc": raw_comparison.loc[
                    raw_comparison["feature_set"].eq(feature_set)
                    & raw_comparison["model"].eq(model)
                    & raw_comparison["score_type"].eq("raw"),
                    "auprc",
                ].iloc[0],
                "calibrated_auprc": best["auprc"],
            }
        )
        mask = predictions["feature_set"].eq(feature_set) & predictions["model"].eq(model)
        predictions.loc[mask, "score_calibrated"] = predictions.loc[mask, f"score_{best['score_type']}"]
        predictions.loc[mask, "calibration_method"] = str(best["score_type"])

    pd.DataFrame(calibration_rows).to_csv(
        s1.METRIC_DIR / "ml_stage2_calibration_comparison.csv", index=False, encoding="utf-8-sig"
    )
    return predictions


def write_data_audit(data: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "dataset": "stage2_development",
            "development_n": int(len(data)),
            "development_positive_n": int(data["Target"].sum()),
            "development_positive_rate": float(data["Target"].mean()),
            "outer_manifest_n": int(len(outer)),
            "inner_manifest_n": int(len(inner)),
            "outer_fold_n": int(outer["outer_fold"].nunique()),
            "inner_fold_n": int(inner["inner_fold"].nunique()),
            "sample_id_duplicate_n": int(data["샘플ID"].duplicated().sum()),
            "sample_id_missing_n": int(data["샘플ID"].isna().sum()),
            "strict_leak_audit": "passed_by_stage1_prepare_data",
            "lockbox_used": False,
        }
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(s1.METRIC_DIR / "ml_stage2_data_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def write_candidate_registry(
    feature_sets: dict[str, list[str]],
    main_trials: int,
    aux_trials: int,
    focused_trials: int,
    focused_top_n: int,
) -> pd.DataFrame:
    tuned_keys = {(candidate.feature_set, candidate.model): candidate for candidate in ALL_STAGE2_CANDIDATES}
    stage1_comparison = pd.read_csv(s1.METRIC_DIR / "ml_stage1_v2_all_model_comparison.csv", encoding="utf-8-sig")
    raw_stage1 = stage1_comparison.loc[stage1_comparison["score_type"].eq("raw")].copy()
    rows = []
    for row in raw_stage1.itertuples(index=False):
        key = (str(row.feature_set), str(row.model))
        candidate = tuned_keys.get(key)
        if candidate is None:
            status = "NOT_TUNED_STAGE2"
            reason = "1차에서 2차 집중 튜닝 근거가 약해 제외"
            role = "excluded"
            priority = np.nan
            trial_budget = 0
        else:
            status = "TUNED_STAGE2"
            reason = candidate.reason
            role = candidate.role
            priority = candidate.priority
            trial_budget = main_trials if candidate.role == "main" else aux_trials
            if candidate.role == "main":
                trial_budget += focused_trials if candidate.priority <= focused_top_n else 0
        rows.append(
            {
                "feature_set": key[0],
                "feature_group": s1.FEATURE_SET_META.get(key[0], {}).get("feature_group", ""),
                "model": key[1],
                "stage2_status": status,
                "stage2_role": role,
                "priority": priority,
                "planned_trial_budget_per_outer": trial_budget,
                "stage1_auprc": getattr(row, "auprc", np.nan),
                "reason": reason,
                "n_features": len(feature_sets[key[0]]) if key[0] in feature_sets else np.nan,
            }
        )
    registry = pd.DataFrame(rows)
    registry.to_csv(s1.METRIC_DIR / "ml_stage2_candidate_registry.csv", index=False, encoding="utf-8-sig")
    return registry


def make_fold_stability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    val = fold_metrics.loc[fold_metrics["dataset"].eq("validation")].copy()
    rows = []
    for (feature_set, model), part in val.groupby(["feature_set", "model"], observed=True):
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "fold_auprc_mean": float(part["auprc"].mean()),
                "fold_auprc_std": float(part["auprc"].std(ddof=0)),
                "fold_auprc_min": float(part["auprc"].min()),
                "fold_auprc_max": float(part["auprc"].max()),
                "fold_auprc_values": "|".join(f"{value:.6f}" for value in part.sort_values("outer_fold")["auprc"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold_auprc_mean", "fold_auprc_std"], ascending=[False, True])


def write_combined_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    logistic = pd.read_csv(s1.LOGISTIC_METRICS_PATH, encoding="utf-8-sig")
    stage1 = pd.read_csv(s1.METRIC_DIR / "ml_stage1_v2_all_model_comparison.csv", encoding="utf-8-sig")

    logistic_rows = []
    for row in logistic.itertuples(index=False):
        logistic_rows.append(
            {
                "source": "logistic_stage17",
                "feature_set": getattr(row, "model"),
                "feature_group": "logistic_stage17",
                "model": getattr(row, "model"),
                "score_type": "oof",
                "auprc": getattr(row, "auprc"),
                "auroc": getattr(row, "auroc"),
                "brier": getattr(row, "brier"),
                "log_loss": getattr(row, "log_loss"),
                "best_f1_f1": getattr(row, "best_f1_f1"),
                "best_f1_precision": getattr(row, "best_f1_precision"),
                "best_f1_recall": getattr(row, "best_f1_recall"),
            }
        )

    stage1_rows = stage1.copy()
    stage1_rows.insert(0, "source", "ml_stage1_v2")
    stage2_rows = comparison.copy()
    stage2_rows.insert(0, "source", "ml_stage2")
    common = sorted(set(pd.DataFrame(logistic_rows).columns) | set(stage1_rows.columns) | set(stage2_rows.columns))
    combined = pd.concat(
        [
            pd.DataFrame(logistic_rows).reindex(columns=common),
            stage1_rows.reindex(columns=common),
            stage2_rows.reindex(columns=common),
        ],
        ignore_index=True,
    ).sort_values(["auprc", "brier"], ascending=[False, True], na_position="last")
    combined.to_csv(s1.METRIC_DIR / "ml_stage2_with_stage1_logistic_comparison.csv", index=False, encoding="utf-8-sig")
    return combined


def make_summary(
    args: argparse.Namespace,
    comparison: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_risk: pd.DataFrame,
    subgroup: pd.DataFrame,
    selected: pd.DataFrame,
    inner_summary: pd.DataFrame,
    fold_stability: pd.DataFrame,
) -> str:
    raw = comparison.loc[comparison["score_type"].eq("raw") & comparison["run_status"].eq("OK")].copy()
    best = raw.iloc[0]
    best_thresholds = thresholds.loc[
        thresholds["feature_set"].eq(best["feature_set"])
        & thresholds["model"].eq(best["model"])
        & thresholds["score_type"].eq(best["score_type"])
    ]
    best_top = top_risk.loc[
        top_risk["feature_set"].eq(best["feature_set"])
        & top_risk["model"].eq(best["model"])
        & top_risk["score_type"].eq(best["score_type"])
    ]
    best_subgroup = subgroup.loc[
        subgroup["feature_set"].eq(best["feature_set"])
        & subgroup["model"].eq(best["model"])
        & subgroup["score_type"].eq(best["score_type"])
        & subgroup["subgroup_type"].eq("negative_type")
    ]
    selected_display = selected[
        ["feature_set", "model", "role", "outer_fold", "inner_best_auprc", "inner_std_auprc", "best_trial_number"]
    ].copy()
    display_cols = [
        "feature_set",
        "feature_group",
        "model",
        "score_type",
        "auprc",
        "auroc",
        "brier",
        "log_loss",
        "best_f1_f1",
        "best_f1_precision",
        "best_f1_recall",
        "delta_auprc_vs_logistic",
        "delta_auprc_vs_stage1_best",
    ]
    success_text = "성공선 도달" if float(best["auprc"]) >= STAGE2_SUCCESS_AUPRC else "성공선 미달"
    lines = [
        "# 머신러닝 2차 결과",
        "",
        "## 1. 실행 목적",
        "",
        "- 1차에서 성능 근거가 확인된 후보를 Optuna로 집중 튜닝했다.",
        "- lockbox test는 사용하지 않았고, 1차와 같은 strict development OOF 기준으로 평가했다.",
        "- 주 지표는 raw AUPRC이며, hard-negative 0A, top-risk capture, Brier/log loss를 보조 판단축으로 사용했다.",
        "",
        "## 2. 실행 설정",
        "",
        f"- main 후보 pilot trials per outer: {args.main_trials}",
        f"- auxiliary 후보 trials per outer: {args.aux_trials}",
        f"- focused top-N: {args.focused_top}",
        f"- focused 추가 trials per outer: {args.focused_trials}",
        f"- calibration top-N: {args.calibration_top}",
        "",
        "## 3. 기준선",
        "",
        f"- 로지스틱 기준선 AUPRC: 0.2398",
        f"- 1차 최고 AUPRC: {STAGE1_BEST_AUPRC:.4f}",
        f"- 2차 성공선 AUPRC: {STAGE2_SUCCESS_AUPRC:.4f}",
        "",
        "## 4. 2차 최고 후보",
        "",
        f"- `{best['feature_set']} / {best['model']} / {best['score_type']}`",
        f"- AUPRC {float(best['auprc']):.4f}, ROC AUC {float(best['auroc']):.4f}, Brier {float(best['brier']):.5f}, log loss {float(best['log_loss']):.5f}",
        f"- 로지스틱 대비 ΔAUPRC {float(best['delta_auprc_vs_logistic']):+.4f}",
        f"- 1차 최고 대비 ΔAUPRC {float(best['delta_auprc_vs_stage1_best']):+.4f} ({success_text})",
        "",
        "## 5. 후보별 성능",
        "",
        raw[display_cols].round(5).to_markdown(index=False),
        "",
        "## 6. Inner tuning 요약",
        "",
        inner_summary.round(5).to_markdown(index=False),
        "",
        "## 7. Fold 안정성",
        "",
        fold_stability.round(5).to_markdown(index=False),
        "",
        "## 8. 최고 후보 운영점",
        "",
        best_thresholds.round(5).to_markdown(index=False),
        "",
        "## 9. 최고 후보 top-risk capture",
        "",
        best_top.round(5).to_markdown(index=False),
        "",
        "## 10. 최고 후보 hard-negative 성능",
        "",
        best_subgroup[["feature_set", "model", "score_type", "subgroup", "auprc", "auroc", "brier", "log_loss"]]
        .round(5)
        .to_markdown(index=False),
        "",
        "## 11. 선택된 파라미터",
        "",
        selected_display.round(5).to_markdown(index=False),
        "",
        "## 12. 결과 해석",
        "",
        stage2_interpretation(best, raw, best_subgroup, best_top),
        "",
        "## 13. 산출물",
        "",
        "- `outputs/optuna/ml_stage2_optuna_studies.sqlite3`",
        "- `outputs/metrics/ml_stage2_candidate_registry.csv`",
        "- `outputs/metrics/ml_stage2_optuna_trials.csv`",
        "- `outputs/metrics/ml_stage2_selected_params.csv`",
        "- `outputs/metrics/ml_stage2_model_comparison.csv`",
        "- `outputs/metrics/ml_stage2_fold_metrics.csv`",
        "- `outputs/metrics/ml_stage2_thresholds.csv`",
        "- `outputs/metrics/ml_stage2_top_risk_capture.csv`",
        "- `outputs/metrics/ml_stage2_subgroup_metrics.csv`",
        "- `outputs/metrics/ml_stage2_with_stage1_logistic_comparison.csv`",
        "- `outputs/interpretation/ml_stage2_feature_importance.csv`",
        "- `outputs/predictions/ml_stage2_oof_predictions.csv`",
    ]
    return "\n".join(lines) + "\n"


def stage2_interpretation(best: pd.Series, raw: pd.DataFrame, best_subgroup: pd.DataFrame, best_top: pd.DataFrame) -> str:
    best_auprc = float(best["auprc"])
    delta_stage1 = float(best["delta_auprc_vs_stage1_best"])
    delta_logistic = float(best["delta_auprc_vs_logistic"])
    top10 = best_top.loc[best_top["top_pct"].eq(0.10)].iloc[0]
    subgroup_0a = best_subgroup.loc[best_subgroup["subgroup"].eq("Target_0A")]
    subgroup_text = ""
    if not subgroup_0a.empty:
        subgroup_text = (
            f" 0A hard-negative AUPRC는 {float(subgroup_0a.iloc[0]['auprc']):.4f}로, "
            "이 값은 쉬운 배경 음성 분리에만 의존하지 않는지 판단하는 핵심 보조 지표다."
        )
    second = raw.iloc[1] if len(raw) > 1 else None
    second_text = ""
    if second is not None:
        second_text = (
            f" 2위 후보는 `{second['feature_set']} / {second['model']}`이며 "
            f"AUPRC {float(second['auprc']):.4f}로 1위와의 차이는 {best_auprc - float(second['auprc']):.4f}이다."
        )
    success = (
        "2차 성공선인 1차 최고 대비 +0.005 이상을 넘었다."
        if best_auprc >= STAGE2_SUCCESS_AUPRC
        else "2차 성공선인 1차 최고 대비 +0.005 이상에는 아직 도달하지 못했다."
    )
    return (
        f"2차 최고 후보는 `{best['feature_set']} / {best['model']}`이다. "
        f"로지스틱 기준선 대비 AUPRC 변화는 {delta_logistic:+.4f}이고, "
        f"1차 최고 대비 변화는 {delta_stage1:+.4f}이다. {success}"
        f"{second_text}\n\n"
        f"Top 10% 위험군에서는 전체 양성의 {float(top10['capture_rate_recall']):.1%}를 포착했고, "
        f"precision은 {float(top10['precision']):.1%}, lift는 {float(top10['lift_vs_base']):.2f}배였다."
        f"{subgroup_text}\n\n"
        "3차로 넘길 때는 raw AUPRC 1위만 보지 않고, 1위와 0.005 이내인 후보 중 0A 성능이나 top-risk capture가 더 좋은 모델도 함께 남기는 것이 맞다. "
        "Calibration은 이번 단계에서 보조 비교로만 해석하고, 최종 확률 보정은 3차 lockbox 평가 직전에 확정한다."
    )


def write_summary_files(summary: str) -> None:
    (s1.OUTPUT_DIR / "ml_stage2_summary.md").write_text(summary, encoding="utf-8")
    (s1.ML_DIR / "머신러닝_2차_진행_결과.md").write_text(summary, encoding="utf-8")


def append_log(comparison: pd.DataFrame) -> None:
    raw = comparison.loc[comparison["score_type"].eq("raw") & comparison["run_status"].eq("OK")].copy()
    best = raw.iloc[0]
    log_text = s1.ML_DIR.joinpath("LOG.md").read_text(encoding="utf-8")
    marker = "### 2차 실행 결과"
    if marker in log_text:
        log_text = log_text.split(marker)[0].rstrip() + "\n\n"
    lines = [
        marker,
        "",
        f"- 전체 최고: `{best['feature_set']} / {best['model']} / raw`",
        f"- AUPRC {float(best['auprc']):.4f}, ROC AUC {float(best['auroc']):.4f}, Brier {float(best['brier']):.5f}, log loss {float(best['log_loss']):.5f}",
        f"- 로지스틱 대비 ΔAUPRC {float(best['delta_auprc_vs_logistic']):+.4f}",
        f"- 1차 최고 대비 ΔAUPRC {float(best['delta_auprc_vs_stage1_best']):+.4f}",
        "",
        "2차 후보별 raw AUPRC:",
        "",
        raw[["feature_set", "model", "auprc", "auroc", "brier", "log_loss", "delta_auprc_vs_stage1_best"]]
        .round(5)
        .to_markdown(index=False),
        "",
        "산출물:",
        "",
        "- `outputs/ml_stage2_summary.md`",
        "- `머신러닝_2차_진행_결과.md`",
        "- `outputs/metrics/ml_stage2_model_comparison.csv`",
        "- `outputs/predictions/ml_stage2_oof_predictions.csv`",
    ]
    s1.ML_DIR.joinpath("LOG.md").write_text(log_text.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(predictions: pd.DataFrame, selected: pd.DataFrame, candidates: list[Stage2Candidate]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        part = predictions.loc[
            predictions["feature_set"].eq(candidate.feature_set) & predictions["model"].eq(candidate.model)
        ]
        rows.append(
            {
                "feature_set": candidate.feature_set,
                "model": candidate.model,
                "oof_n": int(len(part)),
                "positive_n": int(part["Target"].sum()) if not part.empty else 0,
                "score_raw_min": float(part["score_raw"].min()) if not part.empty else np.nan,
                "score_raw_max": float(part["score_raw"].max()) if not part.empty else np.nan,
                "score_raw_nan_n": int(part["score_raw"].isna().sum()) if not part.empty else 0,
                "score_raw_inf_n": int(np.isinf(part["score_raw"]).sum()) if not part.empty else 0,
                "selected_outer_folds": int(
                    selected.loc[
                        selected["feature_set"].eq(candidate.feature_set) & selected["model"].eq(candidate.model),
                        "outer_fold",
                    ].nunique()
                ),
            }
        )
    validation = pd.DataFrame(rows)
    validation.to_csv(s1.METRIC_DIR / "ml_stage2_validation_checks.csv", index=False, encoding="utf-8-sig")
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML Stage2 Optuna tuning")
    parser.add_argument("--run-label", default="full")
    parser.add_argument("--main-trials", type=int, default=4)
    parser.add_argument("--aux-trials", type=int, default=2)
    parser.add_argument("--focused-top", type=int, default=2)
    parser.add_argument("--focused-trials", type=int, default=4)
    parser.add_argument("--calibration-top", type=int, default=3)
    parser.add_argument("--skip-focused", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    global RUN_LABEL
    args = parse_args()
    RUN_LABEL = str(args.run_label).strip() or "full"
    started_path = s1.OUTPUT_DIR / f"ml_stage2_{RUN_LABEL}_started.txt"
    done_path = s1.OUTPUT_DIR / f"ml_stage2_{RUN_LABEL}_done.txt"
    started_path.write_text(f"started={pd.Timestamp.now().isoformat()}\n", encoding="utf-8")
    if done_path.exists():
        done_path.unlink()
    print("ML Stage2: data prepare")
    data, outer, inner, feature_sets, categorical = s1.prepare_data()
    write_data_audit(data, outer, inner)
    write_candidate_registry(feature_sets, args.main_trials, args.aux_trials, args.focused_trials, args.focused_top)
    indexed = data.set_index("샘플ID", drop=False)
    outer_folds = sorted(int(value) for value in outer["outer_fold"].unique())

    print("ML Stage2: Optuna pilot")
    run_tuning_phase(
        "2A_pilot_main",
        MAIN_CANDIDATES,
        indexed,
        inner,
        feature_sets,
        categorical,
        outer_folds,
        args.main_trials,
    )
    run_tuning_phase(
        "2A_pilot_aux",
        AUXILIARY_CANDIDATES,
        indexed,
        inner,
        feature_sets,
        categorical,
        outer_folds,
        args.aux_trials,
    )

    inner_summary = candidate_inner_summary(ALL_STAGE2_CANDIDATES, outer_folds)
    if not args.skip_focused and args.focused_trials > 0:
        focused_keys = set(
            inner_summary.loc[inner_summary["role"].eq("main")]
            .head(args.focused_top)[["feature_set", "model"]]
            .itertuples(index=False, name=None)
        )
        focused_candidates = [
            candidate for candidate in MAIN_CANDIDATES if (candidate.feature_set, candidate.model) in focused_keys
        ]
        print("ML Stage2: Optuna focused")
        run_tuning_phase(
            "2B_focused",
            focused_candidates,
            indexed,
            inner,
            feature_sets,
            categorical,
            outer_folds,
            args.main_trials + args.focused_trials,
        )

    inner_summary = candidate_inner_summary(ALL_STAGE2_CANDIDATES, outer_folds)
    trials = export_trials(ALL_STAGE2_CANDIDATES, outer_folds)
    selected = selected_params_table(ALL_STAGE2_CANDIDATES, outer_folds)
    trials.to_csv(s1.METRIC_DIR / "ml_stage2_optuna_trials.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(s1.METRIC_DIR / "ml_stage2_selected_params.csv", index=False, encoding="utf-8-sig")
    inner_summary.to_csv(s1.METRIC_DIR / "ml_stage2_inner_summary.csv", index=False, encoding="utf-8-sig")

    print("ML Stage2: OOF train")
    predictions, fold_metrics, importance = train_stage2_oof(data, outer, feature_sets, categorical, selected)
    raw_comparison = s1.model_comparison(predictions)

    if args.skip_calibration:
        calibrated_predictions = predictions
    else:
        print("ML Stage2: calibration")
        calibrated_predictions = apply_stage2_calibration(
            data,
            outer,
            inner,
            feature_sets,
            categorical,
            predictions,
            selected,
            raw_comparison,
            args.calibration_top,
        )

    calibrated_predictions.to_csv(
        s1.PREDICTION_DIR / "ml_stage2_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fold_metrics.to_csv(s1.METRIC_DIR / "ml_stage2_fold_metrics.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(INTERPRET_DIR / "ml_stage2_feature_importance.csv", index=False, encoding="utf-8-sig")

    comparison = s1.model_comparison(calibrated_predictions)
    comparison["delta_auprc_vs_stage1_best"] = comparison["auprc"] - STAGE1_BEST_AUPRC
    comparison["stage2_success_vs_stage1_plus_0p005"] = comparison["auprc"] >= STAGE2_SUCCESS_AUPRC
    thresholds = s1.make_threshold_table(calibrated_predictions)
    top_risk = s1.make_top_risk_table(calibrated_predictions)
    subgroup = s1.make_subgroup_table(calibrated_predictions)
    fold_stability = make_fold_stability(fold_metrics)
    validation = validate_outputs(calibrated_predictions, selected, ALL_STAGE2_CANDIDATES)

    comparison.to_csv(s1.METRIC_DIR / "ml_stage2_model_comparison.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(s1.METRIC_DIR / "ml_stage2_thresholds.csv", index=False, encoding="utf-8-sig")
    top_risk.to_csv(s1.METRIC_DIR / "ml_stage2_top_risk_capture.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(s1.METRIC_DIR / "ml_stage2_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    fold_stability.to_csv(s1.METRIC_DIR / "ml_stage2_fold_stability.csv", index=False, encoding="utf-8-sig")
    write_combined_comparison(comparison)

    summary = make_summary(args, comparison, thresholds, top_risk, subgroup, selected, inner_summary, fold_stability)
    write_summary_files(summary)
    append_log(comparison)

    best = comparison.loc[comparison["score_type"].eq("raw") & comparison["run_status"].eq("OK")].iloc[0]
    if not (validation["oof_n"].eq(13632).all() and validation["positive_n"].eq(1242).all()):
        raise ValueError("OOF validation row/positive count check failed.")
    if validation["score_raw_nan_n"].sum() or validation["score_raw_inf_n"].sum():
        raise ValueError("OOF score contains NaN or inf.")
    print(
        "ML Stage2 완료: "
        f"{best['feature_set']} / {best['model']} AUPRC={best['auprc']:.4f}, "
        f"Δstage1={best['delta_auprc_vs_stage1_best']:+.4f}, "
        f"Δlogistic={best['delta_auprc_vs_logistic']:+.4f}"
    )
    done_path.write_text(
        "\n".join(
            [
                f"finished={pd.Timestamp.now().isoformat()}",
                "exit_code=0",
                f"best={best['feature_set']} / {best['model']}",
                f"auprc={best['auprc']:.6f}",
                f"delta_stage1={best['delta_auprc_vs_stage1_best']:+.6f}",
                f"delta_logistic={best['delta_auprc_vs_logistic']:+.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        label = RUN_LABEL or "full"
        (s1.OUTPUT_DIR / f"ml_stage2_{label}_done.txt").write_text(
            "\n".join(
                [
                    f"finished={pd.Timestamp.now().isoformat()}",
                    "exit_code=1",
                    f"error={type(exc).__name__}: {exc}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        raise
