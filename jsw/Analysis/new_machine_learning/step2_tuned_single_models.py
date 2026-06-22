from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover - 실행 환경 의존
    LGBMClassifier = None
    LGBM_IMPORT_ERROR = repr(exc)
else:
    LGBM_IMPORT_ERROR = ""

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - 실행 환경 의존
    XGBClassifier = None
    XGB_IMPORT_ERROR = repr(exc)
else:
    XGB_IMPORT_ERROR = ""

try:
    from catboost import CatBoostClassifier
except Exception as exc:  # pragma: no cover - 실행 환경 의존
    CatBoostClassifier = None
    CATBOOST_IMPORT_ERROR = repr(exc)
else:
    CATBOOST_IMPORT_ERROR = ""

try:
    from tqdm.auto import tqdm
except Exception as exc:  # pragma: no cover - 실행 환경 의존
    tqdm = None
    TQDM_IMPORT_ERROR = repr(exc)
else:
    TQDM_IMPORT_ERROR = ""


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


TARGET_COL = "Target"
SAMPLE_ID_COL = "샘플ID"
SAMPLE_TYPE_COL = "샘플유형"
GROUP_COL = "모델링_그룹ID"
CLIMATE_COL = "기후지형유형"

PRIMARY_CANDIDATE_IDS = [
    "TUNE_RF_CORE_ALL_LC_BAL",
    "TUNE_RF_ALL_ALL_LC_BAL",
    "TUNE_LGBM_ALL_ALL_LC_NONE",
    "TUNE_ET_ALL_ALL_NO_LC_BAL",
    "TUNE_HGB_ALL_ALL_LC_NONE",
]
RESCUE_CANDIDATE_IDS = [
    "RESCUE_XGB_ALL_ALL_LC_NONE",
    "RESCUE_CAT_ALL_ALL_LC_NONE",
]
ALL_CANDIDATE_IDS = PRIMARY_CANDIDATE_IDS + RESCUE_CANDIDATE_IDS

RECALL_TARGETS = [0.70, 0.90, 0.95]
TOP_RISK_FRACTIONS = [0.05, 0.10, 0.20, 0.30]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    tuning_group: str
    model_name: str
    model_label: str
    feature_set: str
    imbalance_option: str
    role: str


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    weather_space_axis: str
    canada_axis: str
    landcover_axis: str
    features: list[str]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    if tqdm is not None:
        try:
            tqdm.write(f"[{timestamp()}] {message}")
            return
        except Exception:
            pass
    print(f"[{timestamp()}] {message}", flush=True)


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, second = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{second:02d}"


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else start
    for candidate in [start, *start.parents]:
        expected = candidate / "data" / "학습데이터" / "최종_머신러닝_학습데이터.csv"
        if expected.exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다. D:\\farm-system-public-02 아래에서 실행하세요.")


def dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def build_feature_sets() -> dict[str, FeatureSetSpec]:
    time_climate = [
        "기후지형유형",
        "월_sin",
        "월_cos",
        "시간_sin",
        "시간_cos",
    ]
    weather_core = [
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
    ]
    weather_extra = [
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
    space_core = [
        "log1p_도로_최단거리_m",
        "log1p_시가화거리_m",
        "log1p_산림지역_최단거리_m",
        "고도(m)",
        "경사도(도)",
        "TPI(지형위치지수)",
    ]
    space_extra = [
        "log1p_농업거리_m",
        "log1p_임도_최단거리_m",
        "log1p_등산로거리_m",
        "사면방향_sin",
        "사면방향_cos",
    ]
    landcover_binary = [
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

    weather_space_axes = {
        "WS_CORE": dedupe(time_climate + weather_core + space_core),
        "WS_ALL": dedupe(time_climate + weather_core + weather_extra + space_core + space_extra),
    }
    canada_axes = {
        "CANADA_NONE": [],
        "CANADA_CORE": ["FFMC", "ISI", "FWI"],
        "CANADA_LONG": ["DMC", "DC", "BUI"],
        "CANADA_ALL": ["FFMC", "FFMC_10일평균", "DMC", "DC", "ISI", "BUI", "FWI"],
    }
    landcover_axes = {
        "LC_NONE": [],
        "LC_USED": landcover_binary,
    }
    ordered_names = [
        "WS_CORE__CANADA_NONE__LC_NONE",
        "WS_CORE__CANADA_CORE__LC_NONE",
        "WS_CORE__CANADA_LONG__LC_NONE",
        "WS_CORE__CANADA_ALL__LC_NONE",
        "WS_CORE__CANADA_NONE__LC_USED",
        "WS_CORE__CANADA_CORE__LC_USED",
        "WS_CORE__CANADA_LONG__LC_USED",
        "WS_CORE__CANADA_ALL__LC_USED",
        "WS_ALL__CANADA_NONE__LC_NONE",
        "WS_ALL__CANADA_CORE__LC_NONE",
        "WS_ALL__CANADA_LONG__LC_NONE",
        "WS_ALL__CANADA_ALL__LC_NONE",
        "WS_ALL__CANADA_NONE__LC_USED",
        "WS_ALL__CANADA_CORE__LC_USED",
        "WS_ALL__CANADA_LONG__LC_USED",
        "WS_ALL__CANADA_ALL__LC_USED",
    ]
    specs: dict[str, FeatureSetSpec] = {}
    for ws_name, ws_features in weather_space_axes.items():
        for ca_name, ca_features in canada_axes.items():
            for lc_name, lc_features in landcover_axes.items():
                name = f"{ws_name}__{ca_name}__{lc_name}"
                specs[name] = FeatureSetSpec(
                    name=name,
                    weather_space_axis=ws_name,
                    canada_axis=ca_name,
                    landcover_axis=lc_name,
                    features=dedupe(ws_features + ca_features + lc_features),
                )
    return {name: specs[name] for name in ordered_names}


def build_candidates() -> dict[str, CandidateSpec]:
    candidates = [
        CandidateSpec(
            "TUNE_RF_CORE_ALL_LC_BAL",
            "primary",
            "random_forest",
            "RandomForest",
            "WS_CORE__CANADA_ALL__LC_USED",
            "balanced",
            "Step 1 전체 1위",
        ),
        CandidateSpec(
            "TUNE_RF_ALL_ALL_LC_BAL",
            "primary",
            "random_forest",
            "RandomForest",
            "WS_ALL__CANADA_ALL__LC_USED",
            "balanced",
            "RandomForest WS_ALL 민감도",
        ),
        CandidateSpec(
            "TUNE_LGBM_ALL_ALL_LC_NONE",
            "primary",
            "lightgbm",
            "LightGBM",
            "WS_ALL__CANADA_ALL__LC_USED",
            "none",
            "확률 품질 최상위 후보",
        ),
        CandidateSpec(
            "TUNE_ET_ALL_ALL_NO_LC_BAL",
            "primary",
            "extra_trees",
            "ExtraTrees",
            "WS_ALL__CANADA_ALL__LC_NONE",
            "balanced",
            "bagging 대안 후보",
        ),
        CandidateSpec(
            "TUNE_HGB_ALL_ALL_LC_NONE",
            "primary",
            "HistGradientBoosting".lower(),
            "HistGradientBoosting",
            "WS_ALL__CANADA_ALL__LC_USED",
            "none",
            "sklearn boosting 기준 후보",
        ),
        CandidateSpec(
            "RESCUE_XGB_ALL_ALL_LC_NONE",
            "rescue",
            "xgboost",
            "XGBoost",
            "WS_ALL__CANADA_ALL__LC_USED",
            "none",
            "XGBoost 구제 튜닝",
        ),
        CandidateSpec(
            "RESCUE_CAT_ALL_ALL_LC_NONE",
            "rescue",
            "catboost",
            "CatBoost",
            "WS_ALL__CANADA_ALL__LC_USED",
            "none",
            "CatBoost 구제 튜닝",
        ),
    ]
    # 실수 방지: lower()로 만든 sklearn 이름을 실제 slug로 보정한다.
    fixed: list[CandidateSpec] = []
    for candidate in candidates:
        model_name = "hist_gradient_boosting" if candidate.candidate_id == "TUNE_HGB_ALL_ALL_LC_NONE" else candidate.model_name
        fixed.append(
            CandidateSpec(
                candidate.candidate_id,
                candidate.tuning_group,
                model_name,
                candidate.model_label,
                candidate.feature_set,
                candidate.imbalance_option,
                candidate.role,
            )
        )
    return {candidate.candidate_id: candidate for candidate in fixed}


def dependency_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"package": "sklearn", "available": True, "error": ""},
            {"package": "optuna", "available": True, "error": ""},
            {"package": "lightgbm", "available": LGBMClassifier is not None, "error": LGBM_IMPORT_ERROR},
            {"package": "xgboost", "available": XGBClassifier is not None, "error": XGB_IMPORT_ERROR},
            {"package": "catboost", "available": CatBoostClassifier is not None, "error": CATBOOST_IMPORT_ERROR},
            {"package": "tqdm", "available": tqdm is not None, "error": TQDM_IMPORT_ERROR},
        ]
    )


def model_available(model_name: str) -> tuple[bool, str]:
    if model_name == "lightgbm" and LGBMClassifier is None:
        return False, LGBM_IMPORT_ERROR
    if model_name == "xgboost" and XGBClassifier is None:
        return False, XGB_IMPORT_ERROR
    if model_name == "catboost" and CatBoostClassifier is None:
        return False, CATBOOST_IMPORT_ERROR
    return True, ""


def positive_scale_weight(y_train: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_train, dtype=int)
    positive_n = max(1, int(y.sum()))
    negative_n = max(1, int((1 - y).sum()))
    return float(negative_n / positive_n)


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 420, step=40),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 14, 18, 24]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.35, 0.5, 0.75]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 160, 500, step=40),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 14, 18, 24]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.35, 0.5, 0.75]),
            "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 80, 360, step=40),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-5, 1.0, log=True),
            "max_bins": trial.suggest_categorical("max_bins", [64, 128, 255]),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 9, 12]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 8.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 3.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500, step=50),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        }
    raise KeyError(model_name)


def build_estimator(
    model_name: str,
    imbalance_option: str,
    y_train: pd.Series | np.ndarray,
    params: dict[str, Any],
    *,
    random_state: int,
    n_jobs: int,
) -> Any:
    class_weight = "balanced" if imbalance_option == "balanced" else None
    if model_name == "random_forest":
        return RandomForestClassifier(
            **params,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            **params,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            **params,
            class_weight=class_weight,
            random_state=random_state,
        )
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError(f"LightGBM import 실패: {LGBM_IMPORT_ERROR}")
        return LGBMClassifier(
            **params,
            objective="binary",
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError(f"XGBoost import 실패: {XGB_IMPORT_ERROR}")
        scale_pos_weight = positive_scale_weight(y_train) if imbalance_option == "balanced" else 1.0
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError(f"CatBoost import 실패: {CATBOOST_IMPORT_ERROR}")
        auto_class_weights = "Balanced" if imbalance_option == "balanced" else None
        return CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights=auto_class_weights,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
        )
    raise KeyError(model_name)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - 구버전 sklearn 대응
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(
    *,
    candidate: CandidateSpec,
    features: list[str],
    y_train: pd.Series | np.ndarray,
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
) -> Pipeline:
    categorical_candidates = [
        "기후지형유형",
        "토지피복_L1_NAME",
        "토지피복_L2_NAME",
        "토지피복_매칭방식",
        "토지피복_산림유형",
    ]
    categorical_features = [column for column in categorical_candidates if column in features]
    numeric_features = [column for column in features if column not in categorical_features]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    estimator = build_estimator(
        candidate.model_name,
        candidate.imbalance_option,
        y_train,
        params,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def clipped_probability(probability: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)


def predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return clipped_probability(model.predict_proba(x)[:, 1])
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return clipped_probability(1.0 / (1.0 + np.exp(-score)))
    return clipped_probability(model.predict(x))


def safe_average_precision(y_true: np.ndarray | pd.Series, probability: np.ndarray | pd.Series) -> float:
    if len(y_true) == 0:
        return float("nan")
    try:
        return float(average_precision_score(y_true, clipped_probability(probability)))
    except Exception:
        return float("nan")


def safe_auroc(y_true: np.ndarray | pd.Series, probability: np.ndarray | pd.Series) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, clipped_probability(probability)))
    except Exception:
        return float("nan")


def safe_log_loss(y_true: np.ndarray | pd.Series, probability: np.ndarray | pd.Series) -> float:
    try:
        return float(log_loss(np.asarray(y_true, dtype=int), clipped_probability(probability), labels=[0, 1]))
    except Exception:
        return float("nan")


def probability_metrics(y_true: np.ndarray | pd.Series, probability: np.ndarray | pd.Series) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    prob = clipped_probability(probability)
    positive_n = int(y.sum())
    return {
        "n": int(len(y)),
        "positive_n": positive_n,
        "negative_n": int(len(y) - positive_n),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "auprc": safe_average_precision(y, prob),
        "auroc": safe_auroc(y, prob),
        "brier": float(brier_score_loss(y, prob)) if len(y) else float("nan"),
        "log_loss": safe_log_loss(y, prob),
    }


def threshold_curve(y_true: np.ndarray | pd.Series, probability: np.ndarray | pd.Series) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    prob = clipped_probability(probability)
    frame = pd.DataFrame({"y_true": y, "threshold": prob})
    grouped = (
        frame.groupby("threshold", as_index=False)
        .agg(selected_at_threshold=("y_true", "size"), positives_at_threshold=("y_true", "sum"))
        .sort_values("threshold", ascending=False)
        .reset_index(drop=True)
    )
    grouped["negatives_at_threshold"] = grouped["selected_at_threshold"] - grouped["positives_at_threshold"]
    grouped["tp"] = grouped["positives_at_threshold"].cumsum()
    grouped["fp"] = grouped["negatives_at_threshold"].cumsum()
    total_pos = int(y.sum())
    total_neg = int(len(y) - total_pos)
    grouped["fn"] = total_pos - grouped["tp"]
    grouped["tn"] = total_neg - grouped["fp"]
    grouped["selected_n"] = grouped["tp"] + grouped["fp"]
    grouped["selected_rate"] = grouped["selected_n"] / max(1, len(y))
    grouped["precision"] = np.where(grouped["selected_n"] > 0, grouped["tp"] / grouped["selected_n"], 0.0)
    grouped["recall"] = np.where(total_pos > 0, grouped["tp"] / total_pos, np.nan)
    grouped["specificity"] = np.where(total_neg > 0, grouped["tn"] / total_neg, np.nan)
    grouped["false_positive_rate"] = 1.0 - grouped["specificity"]
    f1_denominator = grouped["precision"] + grouped["recall"]
    grouped["f1"] = np.where(f1_denominator > 0, 2 * grouped["precision"] * grouped["recall"] / f1_denominator, 0.0)
    beta2 = 4.0
    f2_denominator = beta2 * grouped["precision"] + grouped["recall"]
    grouped["f2"] = np.where(
        f2_denominator > 0,
        (1.0 + beta2) * grouped["precision"] * grouped["recall"] / f2_denominator,
        0.0,
    )
    grouped["balanced_accuracy"] = (grouped["recall"] + grouped["specificity"]) / 2.0
    return grouped


def select_thresholds(y_true: np.ndarray | pd.Series, probability: np.ndarray | pd.Series) -> dict[str, float]:
    curve = threshold_curve(y_true, probability)
    thresholds: dict[str, float] = {"fixed_0.50": 0.5}
    if curve.empty:
        thresholds.update({"best_f1": 0.5, "best_f2": 0.5})
        for target in RECALL_TARGETS:
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
    for target in RECALL_TARGETS:
        candidates = curve.loc[curve["recall"] >= target]
        thresholds[f"recall_ge_{target:.2f}"] = float(candidates.iloc[0]["threshold"]) if not candidates.empty else 0.0
    return thresholds


def classification_metrics_at_threshold(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    pred = (clipped_probability(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    f2_denominator = 4 * precision + recall
    f2 = float((5 * precision * recall / f2_denominator) if f2_denominator > 0 else 0.0)
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "false_positive_rate": float(1.0 - specificity) if np.isfinite(specificity) else float("nan"),
        "f1": f1,
        "f2": f2,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "selected_n": int(pred.sum()),
        "selected_rate": float(pred.mean()) if len(pred) else float("nan"),
        "fp_per_tp": float(fp / tp) if tp > 0 else float("inf"),
    }


def load_or_create_outer_manifest(
    data: pd.DataFrame,
    *,
    step1_output_dir: Path,
    output_dir: Path,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    step1_manifest_path = step1_output_dir / "split_manifest_outer_cv.csv"
    if step1_manifest_path.exists():
        manifest = pd.read_csv(step1_manifest_path, encoding="utf-8-sig")
        required = {SAMPLE_ID_COL, "outer_fold"}
        if not required.issubset(manifest.columns):
            raise KeyError(f"Step1 split manifest에 필요한 컬럼이 없습니다: {required}")
        manifest = data[[SAMPLE_ID_COL, GROUP_COL, TARGET_COL, SAMPLE_TYPE_COL, CLIMATE_COL]].merge(
            manifest[[SAMPLE_ID_COL, "outer_fold"]],
            on=SAMPLE_ID_COL,
            how="left",
            validate="one_to_one",
        )
        if manifest["outer_fold"].isna().any():
            raise RuntimeError("Step1 split manifest와 매칭되지 않는 샘플이 있습니다.")
        manifest["outer_fold"] = manifest["outer_fold"].astype(int)
        manifest.to_csv(output_dir / "split_manifest_outer_cv.csv", index=False, encoding="utf-8-sig")
        return manifest

    y = data[TARGET_COL].astype(int).to_numpy()
    groups = data[GROUP_COL].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    manifest = data[[SAMPLE_ID_COL, GROUP_COL, TARGET_COL, SAMPLE_TYPE_COL, CLIMATE_COL]].copy()
    manifest["outer_fold"] = -1
    for fold, (_, valid_idx) in enumerate(splitter.split(data, y, groups)):
        manifest.loc[valid_idx, "outer_fold"] = fold
    if (manifest["outer_fold"] < 0).any():
        raise RuntimeError("outer fold가 배정되지 않은 표본이 있습니다.")
    manifest.to_csv(output_dir / "split_manifest_outer_cv.csv", index=False, encoding="utf-8-sig")
    return manifest


def outer_splits_from_manifest(manifest: pd.DataFrame, max_outer_folds: int = 0) -> list[tuple[int, np.ndarray, np.ndarray]]:
    folds = sorted(int(fold) for fold in manifest["outer_fold"].dropna().unique())
    if max_outer_folds and max_outer_folds > 0:
        folds = folds[:max_outer_folds]
    index = np.arange(len(manifest))
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold in folds:
        valid_mask = manifest["outer_fold"].to_numpy() == fold
        valid_idx = index[valid_mask]
        train_idx = index[~valid_mask]
        splits.append((fold, train_idx, valid_idx))
    return splits


def check_group_leakage(data: pd.DataFrame, splits: list[tuple[int, np.ndarray, np.ndarray]]) -> int:
    groups = data[GROUP_COL].astype(str).to_numpy()
    leakage = 0
    for _, train_idx, valid_idx in splits:
        leakage += len(set(groups[train_idx]).intersection(set(groups[valid_idx])))
    return leakage


def tune_candidate_for_outer_fold(
    data: pd.DataFrame,
    candidate: CandidateSpec,
    feature_spec: FeatureSetSpec,
    train_idx: np.ndarray,
    *,
    outer_fold: int,
    inner_splits: int,
    n_trials: int,
    random_state: int,
    n_jobs: int,
    progress_bar: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_data = data.iloc[train_idx].reset_index(drop=True)
    y_train = train_data[TARGET_COL].astype(int).to_numpy()
    groups_train = train_data[GROUP_COL].astype(str).to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, candidate.model_name)
        inner_probability = np.full(len(train_data), np.nan, dtype=float)
        splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + 1000 + outer_fold)
        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(splitter.split(train_data, y_train, groups_train)):
            model = make_pipeline(
                candidate=candidate,
                features=feature_spec.features,
                y_train=train_data.iloc[inner_train_idx][TARGET_COL].astype(int),
                params=params,
                random_state=random_state + outer_fold * 100 + inner_fold,
                n_jobs=n_jobs,
            )
            model.fit(
                train_data.iloc[inner_train_idx][feature_spec.features],
                train_data.iloc[inner_train_idx][TARGET_COL].astype(int),
            )
            inner_probability[inner_valid_idx] = predict_probability(
                model,
                train_data.iloc[inner_valid_idx][feature_spec.features],
            )
        if np.isnan(inner_probability).any():
            raise RuntimeError("inner OOF 예측 누락")
        return safe_average_precision(y_train, inner_probability)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state + outer_fold))
    bar = None
    if progress_bar and tqdm is not None:
        bar = tqdm(
            total=n_trials,
            desc=f"{candidate.candidate_id} outer{outer_fold} Optuna",
            unit="trial",
            leave=True,
            dynamic_ncols=True,
        )

    def callback(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        if bar is not None:
            best = study.best_value if study.best_trial is not None else float("nan")
            bar.update(1)
            bar.set_postfix({"best_auprc": f"{best:.4f}"}, refresh=False)

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], gc_after_trial=True, catch=(Exception,))
    finally:
        if bar is not None:
            bar.close()

    completed_trials = [trial for trial in study.trials if trial.value is not None and trial.state.name == "COMPLETE"]
    if not completed_trials:
        raise RuntimeError(f"{candidate.candidate_id} outer{outer_fold}: 성공한 Optuna trial이 없습니다.")

    trials_frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    trials_frame.insert(0, "outer_fold", outer_fold)
    trials_frame.insert(0, "candidate_id", candidate.candidate_id)
    return dict(study.best_params), trials_frame


def make_oof_frame(
    data: pd.DataFrame,
    *,
    predicted_idx: np.ndarray,
    probability: np.ndarray,
    outer_fold: np.ndarray,
    candidate: CandidateSpec,
) -> pd.DataFrame:
    subset = data.iloc[predicted_idx].copy()
    return pd.DataFrame(
        {
            "sample_id": subset[SAMPLE_ID_COL].astype(str).to_numpy(),
            "outer_fold": outer_fold.astype(int),
            "y_true": subset[TARGET_COL].astype(int).to_numpy(),
            "y_prob": clipped_probability(probability),
            "candidate_id": candidate.candidate_id,
            "tuning_group": candidate.tuning_group,
            "model_name": candidate.model_name,
            "model_label": candidate.model_label,
            "feature_set": candidate.feature_set,
            "imbalance_option": candidate.imbalance_option,
            "sample_type": subset[SAMPLE_TYPE_COL].astype(str).to_numpy(),
            "climate_type": subset[CLIMATE_COL].astype(str).to_numpy(),
            "group_id": subset[GROUP_COL].astype(str).to_numpy(),
        }
    )


def make_summary_row(
    oof: pd.DataFrame,
    *,
    candidate: CandidateSpec,
    feature_spec: FeatureSetSpec,
    elapsed_seconds: float,
    n_trials_per_outer: int,
    outer_fold_count: int,
    status: str,
    error: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "tuning_group": candidate.tuning_group,
        "model_name": candidate.model_name,
        "model_label": candidate.model_label,
        "feature_set": candidate.feature_set,
        "imbalance_option": candidate.imbalance_option,
        "weather_space_axis": feature_spec.weather_space_axis,
        "canada_axis": feature_spec.canada_axis,
        "landcover_axis": feature_spec.landcover_axis,
        "feature_count": len(feature_spec.features),
        "n_trials_per_outer": n_trials_per_outer,
        "outer_fold_count": outer_fold_count,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
    }
    if oof.empty:
        row.update({key: np.nan for key in ["n", "positive_n", "negative_n", "positive_rate", "auprc", "auroc", "brier", "log_loss"]})
        row["fold_auprc_mean"] = np.nan
        row["fold_auprc_std"] = np.nan
        return row
    row.update(probability_metrics(oof["y_true"], oof["y_prob"]))
    fold_auprcs = [
        safe_average_precision(fold_df["y_true"], fold_df["y_prob"])
        for _, fold_df in oof.groupby("outer_fold")
    ]
    row["fold_auprc_mean"] = float(np.nanmean(fold_auprcs)) if fold_auprcs else float("nan")
    row["fold_auprc_std"] = float(np.nanstd(fold_auprcs, ddof=1)) if len(fold_auprcs) > 1 else float("nan")
    return row


def make_fold_metric_rows(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if oof.empty:
        return rows
    for fold, fold_df in oof.groupby("outer_fold"):
        row = {
            "candidate_id": oof["candidate_id"].iloc[0],
            "tuning_group": oof["tuning_group"].iloc[0],
            "model_name": oof["model_name"].iloc[0],
            "model_label": oof["model_label"].iloc[0],
            "feature_set": oof["feature_set"].iloc[0],
            "imbalance_option": oof["imbalance_option"].iloc[0],
            "outer_fold": int(fold),
        }
        row.update(probability_metrics(fold_df["y_true"], fold_df["y_prob"]))
        rows.append(row)
    return rows


def make_threshold_rows(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if oof.empty:
        return rows
    thresholds = select_thresholds(oof["y_true"], oof["y_prob"])
    for operating_point, threshold in thresholds.items():
        row = {
            "candidate_id": oof["candidate_id"].iloc[0],
            "tuning_group": oof["tuning_group"].iloc[0],
            "model_name": oof["model_name"].iloc[0],
            "model_label": oof["model_label"].iloc[0],
            "feature_set": oof["feature_set"].iloc[0],
            "imbalance_option": oof["imbalance_option"].iloc[0],
            "threshold_source": "tuned_oof",
            "operating_point": operating_point,
        }
        row.update(classification_metrics_at_threshold(oof["y_true"], oof["y_prob"], threshold))
        rows.append(row)
    return rows


def make_subgroup_rows(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if oof.empty:
        return rows
    base = {
        "candidate_id": oof["candidate_id"].iloc[0],
        "tuning_group": oof["tuning_group"].iloc[0],
        "model_name": oof["model_name"].iloc[0],
        "model_label": oof["model_label"].iloc[0],
        "feature_set": oof["feature_set"].iloc[0],
        "imbalance_option": oof["imbalance_option"].iloc[0],
    }
    for negative_type in ["Target_0A", "Target_0B1", "Target_0B2"]:
        subgroup = oof.loc[oof["sample_type"].isin(["Target_1", negative_type])]
        if subgroup.empty:
            continue
        row = {**base, "subgroup_type": "sample_type_pair", "subgroup_value": f"Target_1_vs_{negative_type}"}
        row.update(probability_metrics(subgroup["y_true"], subgroup["y_prob"]))
        rows.append(row)
    for climate_type, subgroup in oof.groupby("climate_type"):
        row = {**base, "subgroup_type": "climate_type", "subgroup_value": str(climate_type)}
        row.update(probability_metrics(subgroup["y_true"], subgroup["y_prob"]))
        rows.append(row)
    return rows


def make_top_risk_rows(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if oof.empty:
        return rows
    sorted_oof = oof.sort_values("y_prob", ascending=False).reset_index(drop=True)
    total_positive = int(sorted_oof["y_true"].sum())
    base_rate = float(sorted_oof["y_true"].mean()) if len(sorted_oof) else float("nan")
    for fraction in TOP_RISK_FRACTIONS:
        selected_n = max(1, int(math.ceil(len(sorted_oof) * fraction)))
        selected = sorted_oof.head(selected_n)
        captured_positive = int(selected["y_true"].sum())
        precision = float(selected["y_true"].mean()) if len(selected) else float("nan")
        rows.append(
            {
                "candidate_id": oof["candidate_id"].iloc[0],
                "tuning_group": oof["tuning_group"].iloc[0],
                "model_name": oof["model_name"].iloc[0],
                "model_label": oof["model_label"].iloc[0],
                "feature_set": oof["feature_set"].iloc[0],
                "imbalance_option": oof["imbalance_option"].iloc[0],
                "top_fraction": fraction,
                "selected_n": selected_n,
                "positive_captured_n": captured_positive,
                "total_positive_n": total_positive,
                "capture_rate": float(captured_positive / total_positive) if total_positive else float("nan"),
                "precision": precision,
                "lift": float(precision / base_rate) if base_rate and np.isfinite(base_rate) else float("nan"),
            }
        )
    return rows


def make_validation_row(
    oof: pd.DataFrame,
    *,
    candidate: CandidateSpec,
    expected_n: int,
    outer_group_leakage_n: int,
    status: str,
    error: str,
) -> dict[str, Any]:
    probability = oof["y_prob"].to_numpy(dtype=float) if not oof.empty else np.array([])
    duplicate_sample_id_n = int(oof["sample_id"].duplicated().sum()) if not oof.empty else 0
    folds_without_positive_n = 0
    if not oof.empty:
        for _, fold_df in oof.groupby("outer_fold"):
            if int(fold_df["y_true"].sum()) == 0:
                folds_without_positive_n += 1
    return {
        "candidate_id": candidate.candidate_id,
        "tuning_group": candidate.tuning_group,
        "model_name": candidate.model_name,
        "model_label": candidate.model_label,
        "feature_set": candidate.feature_set,
        "imbalance_option": candidate.imbalance_option,
        "status": status,
        "expected_n": expected_n,
        "prediction_n": int(len(oof)),
        "missing_prediction_n": int(max(0, expected_n - len(oof))),
        "duplicate_sample_id_n": duplicate_sample_id_n,
        "nan_probability_n": int(np.isnan(probability).sum()) if len(probability) else 0,
        "inf_probability_n": int(np.isinf(probability).sum()) if len(probability) else 0,
        "outer_group_leakage_n": int(outer_group_leakage_n),
        "folds_without_positive_n": int(folds_without_positive_n),
        "error": error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_from_csv_argument(values: list[str], allowed: list[str], *, argument_name: str) -> list[str]:
    if not values or values == ["all"]:
        return allowed
    selected: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in allowed:
                raise ValueError(f"{argument_name} 값이 잘못됐습니다: {token}. 허용값: {allowed}")
            selected.append(token)
    return dedupe(selected)


def run_candidate(
    data: pd.DataFrame,
    candidate: CandidateSpec,
    feature_spec: FeatureSetSpec,
    outer_splits: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    output_dir: Path,
    n_trials: int,
    inner_splits: int,
    random_state: int,
    n_jobs: int,
    progress_bar: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    predicted_indices: list[np.ndarray] = []
    predicted_probabilities: list[np.ndarray] = []
    predicted_folds: list[np.ndarray] = []
    selected_params_by_fold: dict[str, Any] = {}
    trial_frames: list[pd.DataFrame] = []

    for outer_fold, train_idx, valid_idx in outer_splits:
        log(f"CANDIDATE {candidate.candidate_id} | outer_fold={outer_fold} | tuning start | trials={n_trials}")
        best_params, trials_frame = tune_candidate_for_outer_fold(
            data,
            candidate,
            feature_spec,
            train_idx,
            outer_fold=outer_fold,
            inner_splits=inner_splits,
            n_trials=n_trials,
            random_state=random_state,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )
        trial_frames.append(trials_frame)
        selected_params_by_fold[str(outer_fold)] = best_params

        model = make_pipeline(
            candidate=candidate,
            features=feature_spec.features,
            y_train=data.iloc[train_idx][TARGET_COL].astype(int),
            params=best_params,
            random_state=random_state + outer_fold,
            n_jobs=n_jobs,
        )
        model.fit(data.iloc[train_idx][feature_spec.features], data.iloc[train_idx][TARGET_COL].astype(int))
        probability = predict_probability(model, data.iloc[valid_idx][feature_spec.features])
        predicted_indices.append(valid_idx)
        predicted_probabilities.append(probability)
        predicted_folds.append(np.full(len(valid_idx), outer_fold, dtype=int))
        log(
            f"CANDIDATE {candidate.candidate_id} | outer_fold={outer_fold} | "
            f"best_inner_auprc={trials_frame['value'].max():.4f} | outer prediction done"
        )

    predicted_idx = np.concatenate(predicted_indices)
    probability = np.concatenate(predicted_probabilities)
    outer_fold_values = np.concatenate(predicted_folds)
    order = np.argsort(predicted_idx)
    oof = make_oof_frame(
        data,
        predicted_idx=predicted_idx[order],
        probability=probability[order],
        outer_fold=outer_fold_values[order],
        candidate=candidate,
    )

    oof.to_csv(output_dir / f"oof__{candidate.candidate_id}.csv", index=False, encoding="utf-8-sig")
    trials_all = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    trials_all.to_csv(output_dir / f"optuna_trials__{candidate.candidate_id}.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_dir / f"selected_params__{candidate.candidate_id}.json",
        {
            "candidate_id": candidate.candidate_id,
            "model_name": candidate.model_name,
            "feature_set": candidate.feature_set,
            "imbalance_option": candidate.imbalance_option,
            "n_trials_per_outer": n_trials,
            "selected_params_by_outer_fold": selected_params_by_fold,
        },
    )

    elapsed = time.perf_counter() - start
    return {
        "oof": oof,
        "summary_row": make_summary_row(
            oof,
            candidate=candidate,
            feature_spec=feature_spec,
            elapsed_seconds=elapsed,
            n_trials_per_outer=n_trials,
            outer_fold_count=len(outer_splits),
            status="OK",
            error="",
        ),
        "fold_rows": make_fold_metric_rows(oof),
        "threshold_rows": make_threshold_rows(oof),
        "subgroup_rows": make_subgroup_rows(oof),
        "top_risk_rows": make_top_risk_rows(oof),
        "validation_row": make_validation_row(
            oof,
            candidate=candidate,
            expected_n=sum(len(valid_idx) for _, _, valid_idx in outer_splits),
            outer_group_leakage_n=0,
            status="OK",
            error="",
        ),
    }


def failed_candidate_result(
    candidate: CandidateSpec,
    feature_spec: FeatureSetSpec,
    *,
    expected_n: int,
    error: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    oof = pd.DataFrame()
    return {
        "oof": oof,
        "summary_row": make_summary_row(
            oof,
            candidate=candidate,
            feature_spec=feature_spec,
            elapsed_seconds=elapsed_seconds,
            n_trials_per_outer=0,
            outer_fold_count=0,
            status="FAIL",
            error=error,
        ),
        "fold_rows": [],
        "threshold_rows": [],
        "subgroup_rows": [],
        "top_risk_rows": [],
        "validation_row": make_validation_row(
            oof,
            candidate=candidate,
            expected_n=expected_n,
            outer_group_leakage_n=0,
            status="FAIL",
            error=error,
        ),
    }


def build_step1_comparison(output_dir: Path, step1_output_dir: Path, step2_summary: pd.DataFrame) -> None:
    step1_summary_path = step1_output_dir / "all_single_models_summary.csv"
    if not step1_summary_path.exists() or step2_summary.empty:
        return
    step1 = pd.read_csv(step1_summary_path, encoding="utf-8-sig")
    join_cols = ["model_name", "feature_set", "imbalance_option"]
    base_cols = join_cols + ["auprc", "auroc", "brier", "log_loss", "fold_auprc_std"]
    step1_base = step1[base_cols].rename(
        columns={
            "auprc": "step1_auprc",
            "auroc": "step1_auroc",
            "brier": "step1_brier",
            "log_loss": "step1_log_loss",
            "fold_auprc_std": "step1_fold_auprc_std",
        }
    )
    step2_base = step2_summary[
        ["candidate_id", "tuning_group", *join_cols, "auprc", "auroc", "brier", "log_loss", "fold_auprc_std"]
    ].rename(
        columns={
            "auprc": "step2_auprc",
            "auroc": "step2_auroc",
            "brier": "step2_brier",
            "log_loss": "step2_log_loss",
            "fold_auprc_std": "step2_fold_auprc_std",
        }
    )
    comparison = step2_base.merge(step1_base, on=join_cols, how="left")
    comparison["delta_auprc"] = comparison["step2_auprc"] - comparison["step1_auprc"]
    comparison["delta_auroc"] = comparison["step2_auroc"] - comparison["step1_auroc"]
    comparison["delta_brier"] = comparison["step2_brier"] - comparison["step1_brier"]
    comparison["delta_log_loss"] = comparison["step2_log_loss"] - comparison["step1_log_loss"]
    comparison.to_csv(output_dir / "summary__step2_vs_step1.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="new_machine_learning Step2: Optuna tuned single models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="", help="입력 CSV 경로")
    parser.add_argument("--step1-output-dir", type=str, default="", help="Step1 결과 폴더")
    parser.add_argument("--output-dir", type=str, default="", help="Step2 결과 폴더")
    parser.add_argument("--candidate-ids", nargs="*", default=["all"], help="실행할 candidate_id. all 또는 쉼표 구분 가능")
    parser.add_argument("--primary-trials", type=int, default=40, help="주 튜닝 후보 outer fold별 Optuna trial 수")
    parser.add_argument("--rescue-trials", type=int, default=15, help="구제 튜닝 후보 outer fold별 Optuna trial 수")
    parser.add_argument("--trial-override", type=int, default=0, help="모든 후보 trial 수를 이 값으로 강제. 0이면 group별 trial 사용")
    parser.add_argument("--outer-splits", type=int, default=5, help="Step1 manifest가 없을 때 생성할 outer split 수")
    parser.add_argument("--inner-splits", type=int, default=4, help="outer-train 내부 inner CV 수")
    parser.add_argument("--max-outer-folds", type=int, default=0, help="디버그용 outer fold 제한. 0이면 전체")
    parser.add_argument("--check-config", action="store_true", help="데이터/후보/의존성만 점검하고 튜닝하지 않음")
    parser.add_argument("--no-progress-bar", action="store_true", help="tqdm 막대바를 사용하지 않음")
    parser.add_argument("--random-state", type=int, default=20260622, help="분할, Optuna, 모델 seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="지원 모델 병렬 작업 수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()
    data_path = Path(args.data) if args.data else root / "data" / "학습데이터" / "최종_머신러닝_학습데이터.csv"
    step1_output_dir = (
        Path(args.step1_output_dir)
        if args.step1_output_dir
        else root / "jsw" / "Analysis" / "new_machine_learning" / "outputs" / "step1_single_models"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "jsw" / "Analysis" / "new_machine_learning" / "outputs" / "step2_tuned_single_models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"입력 데이터 로드: {data_path}")
    data = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    required_columns = [TARGET_COL, SAMPLE_ID_COL, SAMPLE_TYPE_COL, GROUP_COL, CLIMATE_COL]
    missing_required = [column for column in required_columns if column not in data.columns]
    if missing_required:
        raise KeyError(f"필수 컬럼이 없습니다: {missing_required}")
    data[TARGET_COL] = data[TARGET_COL].astype(int)

    feature_sets = build_feature_sets()
    candidates = build_candidates()
    selected_candidate_ids = select_from_csv_argument(args.candidate_ids, ALL_CANDIDATE_IDS, argument_name="--candidate-ids")
    selected_candidates = [candidates[candidate_id] for candidate_id in selected_candidate_ids]

    dependency = dependency_audit()
    dependency.to_csv(output_dir / "dependency_audit.csv", index=False, encoding="utf-8-sig")
    candidate_manifest = pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "tuning_group": candidate.tuning_group,
                "model_name": candidate.model_name,
                "model_label": candidate.model_label,
                "feature_set": candidate.feature_set,
                "imbalance_option": candidate.imbalance_option,
                "role": candidate.role,
                "feature_count": len(feature_sets[candidate.feature_set].features),
                "features": "|".join(feature_sets[candidate.feature_set].features),
            }
            for candidate in candidates.values()
        ]
    )
    candidate_manifest.to_csv(output_dir / "candidate_manifest.csv", index=False, encoding="utf-8-sig")

    outer_manifest = load_or_create_outer_manifest(
        data,
        step1_output_dir=step1_output_dir,
        output_dir=output_dir,
        n_splits=args.outer_splits,
        random_state=args.random_state,
    )
    outer_splits = outer_splits_from_manifest(outer_manifest, max_outer_folds=args.max_outer_folds)
    outer_group_leakage_n = check_group_leakage(data, outer_splits)

    run_config = {
        "script": str(Path(__file__).resolve()),
        "started_at": timestamp(),
        "data_path": str(data_path),
        "step1_output_dir": str(step1_output_dir),
        "output_dir": str(output_dir),
        "n_rows": int(len(data)),
        "positive_n": int(data[TARGET_COL].sum()),
        "positive_rate": float(data[TARGET_COL].mean()),
        "candidate_ids": selected_candidate_ids,
        "primary_trials": args.primary_trials,
        "rescue_trials": args.rescue_trials,
        "trial_override": args.trial_override,
        "outer_fold_count": len(outer_splits),
        "inner_splits": args.inner_splits,
        "max_outer_folds": args.max_outer_folds,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "progress_bar": bool((not args.no_progress_bar) and tqdm is not None),
        "outer_group_leakage_n": int(outer_group_leakage_n),
        "dependency_audit": dependency.to_dict(orient="records"),
    }
    write_json(output_dir / ("run_manifest__check_config.json" if args.check_config else "run_manifest__all.json"), run_config)

    log(
        f"데이터 행 수={len(data):,}, positive={int(data[TARGET_COL].sum()):,}, "
        f"positive_rate={data[TARGET_COL].mean():.4f}"
    )
    log(
        f"Step2 계획: candidates={len(selected_candidates)}, outer_folds={len(outer_splits)}, "
        f"inner={args.inner_splits}, primary_trials={args.primary_trials}, rescue_trials={args.rescue_trials}"
    )
    log(f"결과 폴더: {output_dir}")

    missing_features_by_candidate: dict[str, list[str]] = {}
    for candidate in selected_candidates:
        missing = [column for column in feature_sets[candidate.feature_set].features if column not in data.columns]
        if missing:
            missing_features_by_candidate[candidate.candidate_id] = missing
    if missing_features_by_candidate:
        raise KeyError(f"후보 피처셋에 누락 컬럼이 있습니다: {missing_features_by_candidate}")

    if args.check_config:
        log("--check-config 지정: 튜닝 없이 종료합니다.")
        return

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    top_risk_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    candidate_bar = None
    if (not args.no_progress_bar) and tqdm is not None:
        candidate_bar = tqdm(total=len(selected_candidates), desc="STEP2 candidates", unit="candidate", dynamic_ncols=True)

    try:
        for candidate in selected_candidates:
            candidate_start = time.perf_counter()
            feature_spec = feature_sets[candidate.feature_set]
            available, dependency_error = model_available(candidate.model_name)
            n_trials = args.trial_override if args.trial_override > 0 else (
                args.primary_trials if candidate.tuning_group == "primary" else args.rescue_trials
            )
            log(
                f"CANDIDATE START {candidate.candidate_id} | model={candidate.model_name} | "
                f"feature_set={candidate.feature_set} | imbalance={candidate.imbalance_option} | trials={n_trials}"
            )
            if not available:
                result = failed_candidate_result(
                    candidate,
                    feature_spec,
                    expected_n=sum(len(valid_idx) for _, _, valid_idx in outer_splits),
                    error=f"dependency_missing: {dependency_error}",
                    elapsed_seconds=time.perf_counter() - candidate_start,
                )
            else:
                try:
                    result = run_candidate(
                        data,
                        candidate,
                        feature_spec,
                        outer_splits,
                        output_dir=output_dir,
                        n_trials=n_trials,
                        inner_splits=args.inner_splits,
                        random_state=args.random_state,
                        n_jobs=args.n_jobs,
                        progress_bar=(not args.no_progress_bar) and tqdm is not None,
                    )
                except Exception as exc:
                    result = failed_candidate_result(
                        candidate,
                        feature_spec,
                        expected_n=sum(len(valid_idx) for _, _, valid_idx in outer_splits),
                        error=repr(exc),
                        elapsed_seconds=time.perf_counter() - candidate_start,
                    )
                    log(f"CANDIDATE FAIL {candidate.candidate_id} | error={repr(exc)}")

            summary_rows.append(result["summary_row"])
            fold_rows.extend(result["fold_rows"])
            threshold_rows.extend(result["threshold_rows"])
            subgroup_rows.extend(result["subgroup_rows"])
            top_risk_rows.extend(result["top_risk_rows"])
            validation_rows.append(result["validation_row"])

            write_csv(output_dir / "summary__step2_tuned_candidates.csv", summary_rows)
            write_csv(output_dir / "fold_metrics__step2_tuned_candidates.csv", fold_rows)
            write_csv(output_dir / "threshold_metrics__step2_tuned_candidates.csv", threshold_rows)
            write_csv(output_dir / "subgroup_metrics__step2_tuned_candidates.csv", subgroup_rows)
            write_csv(output_dir / "top_risk_metrics__step2_tuned_candidates.csv", top_risk_rows)
            write_csv(output_dir / "validation_checks__step2_tuned_candidates.csv", validation_rows)

            elapsed = time.perf_counter() - candidate_start
            log(f"CANDIDATE DONE {candidate.candidate_id} | elapsed={format_seconds(elapsed)} | status={result['summary_row']['status']}")
            if candidate_bar is not None:
                candidate_bar.update(1)
                candidate_bar.set_postfix({"last": candidate.candidate_id}, refresh=False)
    finally:
        if candidate_bar is not None:
            candidate_bar.close()

    summary_frame = pd.DataFrame(summary_rows)
    build_step1_comparison(output_dir, step1_output_dir, summary_frame)

    final_manifest = dict(run_config)
    final_manifest.update(
        {
            "finished_at": timestamp(),
            "success_count": int((summary_frame["status"] == "OK").sum()) if not summary_frame.empty else 0,
            "failure_count": int((summary_frame["status"] == "FAIL").sum()) if not summary_frame.empty else 0,
            "actual_summary_rows": int(len(summary_frame)),
        }
    )
    write_json(output_dir / "run_manifest__all.json", final_manifest)
    log(
        f"STEP2 DONE | success={final_manifest['success_count']} | "
        f"failure={final_manifest['failure_count']} | output={output_dir}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("사용자 중단 KeyboardInterrupt")
        sys.exit(130)
