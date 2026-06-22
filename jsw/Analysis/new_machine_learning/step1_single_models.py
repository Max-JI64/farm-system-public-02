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
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - 실행 환경 의존
    XGBClassifier = None
    XGB_IMPORT_ERROR = repr(exc)
else:
    XGB_IMPORT_ERROR = ""

try:
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover - 실행 환경 의존
    LGBMClassifier = None
    LGBM_IMPORT_ERROR = repr(exc)
else:
    LGBM_IMPORT_ERROR = ""

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


TARGET_COL = "Target"
SAMPLE_ID_COL = "샘플ID"
SAMPLE_TYPE_COL = "샘플유형"
GROUP_COL = "모델링_그룹ID"
CLIMATE_COL = "기후지형유형"

MODEL_ORDER = [
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "extra_trees",
]

IMBALANCE_OPTIONS = ["none", "balanced"]
RECALL_TARGETS = [0.70, 0.90, 0.95]
TOP_RISK_FRACTIONS = [0.05, 0.10, 0.20, 0.30]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    family: str


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    weather_space_axis: str
    canada_axis: str
    landcover_axis: str
    features: list[str]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else start
    for candidate in [start, *start.parents]:
        expected = candidate / "data" / "학습데이터" / "최종_머신러닝_학습데이터.csv"
        if expected.exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다. D:\\farm-system-public-02 아래에서 실행하세요.")


def dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def build_feature_sets() -> list[FeatureSetSpec]:
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

    specs: list[FeatureSetSpec] = []
    for ws_name in ["WS_CORE", "WS_ALL"]:
        for lc_name in ["LC_NONE", "LC_USED"]:
            for ca_name in ["CANADA_NONE", "CANADA_CORE", "CANADA_LONG", "CANADA_ALL"]:
                name = f"{ws_name}__{ca_name}__{lc_name}"
                specs.append(
                    FeatureSetSpec(
                        name=name,
                        weather_space_axis=ws_name,
                        canada_axis=ca_name,
                        landcover_axis=lc_name,
                        features=dedupe(weather_space_axes[ws_name] + canada_axes[ca_name] + landcover_axes[lc_name]),
                    )
                )

    # 문서의 번호 순서와 맞추기 위해 LC 축을 안쪽이 아니라 바깥쪽 순서로 다시 정렬한다.
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
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in ordered_names]


def model_specs() -> dict[str, ModelSpec]:
    return {
        "hist_gradient_boosting": ModelSpec("hist_gradient_boosting", "HistGradientBoosting", "sklearn_hgb"),
        "lightgbm": ModelSpec("lightgbm", "LightGBM", "lightgbm"),
        "xgboost": ModelSpec("xgboost", "XGBoost", "xgboost"),
        "catboost": ModelSpec("catboost", "CatBoost", "catboost"),
        "random_forest": ModelSpec("random_forest", "RandomForest", "sklearn_forest"),
        "extra_trees": ModelSpec("extra_trees", "ExtraTrees", "sklearn_forest"),
    }


def dependency_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"package": "sklearn", "available": True, "error": ""},
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


def build_estimator(
    model_name: str,
    imbalance_option: str,
    y_train: pd.Series | np.ndarray,
    *,
    random_state: int,
    n_jobs: int,
) -> Any:
    if imbalance_option not in IMBALANCE_OPTIONS:
        raise ValueError(f"지원하지 않는 imbalance_option: {imbalance_option}")

    class_weight = "balanced" if imbalance_option == "balanced" else None

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.01,
            class_weight=class_weight,
            random_state=random_state,
        )

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError(f"LightGBM import 실패: {LGBM_IMPORT_ERROR}")
        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.055,
            num_leaves=31,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
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
            n_estimators=180,
            max_depth=3,
            learning_rate=0.055,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_lambda=2.0,
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
            iterations=160,
            depth=5,
            learning_rate=0.055,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights=auto_class_weights,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
        )

    raise KeyError(f"지원하지 않는 모델: {model_name}")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - 구버전 sklearn 대응
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(
    model_name: str,
    features: list[str],
    y_train: pd.Series | np.ndarray,
    *,
    imbalance_option: str,
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

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    estimator = build_estimator(
        model_name,
        imbalance_option,
        y_train,
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
    prediction = model.predict(x)
    return clipped_probability(np.asarray(prediction, dtype=float))


def safe_average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    try:
        return float(average_precision_score(y_true, probability))
    except Exception:
        return float("nan")


def safe_auroc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, probability))
    except Exception:
        return float("nan")


def safe_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, clipped_probability(probability), labels=[0, 1]))
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
        "brier": float(brier_score_loss(y, prob)) if len(np.unique(y)) >= 1 else float("nan"),
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
        if candidates.empty:
            thresholds[f"recall_ge_{target:.2f}"] = 0.0
        else:
            thresholds[f"recall_ge_{target:.2f}"] = float(candidates.iloc[0]["threshold"])
    return thresholds


def classification_metrics_at_threshold(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    prob = clipped_probability(probability)
    pred = (prob >= threshold).astype(int)
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


def make_outer_splits(
    data: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, list[tuple[int, np.ndarray, np.ndarray]], int]:
    y = data[TARGET_COL].astype(int).to_numpy()
    groups = data[GROUP_COL].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    split_manifest = data[[SAMPLE_ID_COL, GROUP_COL, TARGET_COL, SAMPLE_TYPE_COL, CLIMATE_COL]].copy()
    split_manifest["outer_fold"] = -1
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    leakage_count = 0

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(data, y, groups)):
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])
        leakage_count += len(train_groups.intersection(valid_groups))
        split_manifest.loc[valid_idx, "outer_fold"] = fold
        splits.append((fold, train_idx, valid_idx))

    if (split_manifest["outer_fold"] < 0).any():
        raise RuntimeError("outer fold가 배정되지 않은 표본이 있습니다.")
    return split_manifest, splits, leakage_count


def planned_fit_count_per_combo(n_outer_splits: int, n_inner_splits: int, honest_thresholds: bool) -> int:
    if honest_thresholds:
        return n_outer_splits * (n_inner_splits + 1)
    return n_outer_splits


class ProgressTracker:
    def __init__(
        self,
        *,
        total_combos: int,
        total_fits: int,
        model_fit_plan: dict[str, int],
        model_combo_plan: dict[str, int],
        use_progress_bar: bool,
    ) -> None:
        self.total_combos = max(1, total_combos)
        self.total_fits = max(1, total_fits)
        self.model_fit_plan = {key: max(1, value) for key, value in model_fit_plan.items()}
        self.model_combo_plan = {key: max(1, value) for key, value in model_combo_plan.items()}
        self.use_progress_bar = bool(use_progress_bar and tqdm is not None)
        self.start_time = time.perf_counter()
        self.model_start_time: dict[str, float] = {}
        self.completed_fits = 0
        self.completed_combos = 0
        self.model_completed_fits = {key: 0 for key in model_fit_plan}
        self.model_completed_combos = {key: 0 for key in model_combo_plan}
        self.overall_bar: Any | None = None
        self.model_bar: Any | None = None
        if self.use_progress_bar:
            self.overall_bar = tqdm(
                total=self.total_fits,
                desc="STEP1 total fits",
                unit="fit",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )

    def _eta(self, completed: int, total: int, elapsed: float) -> float | None:
        if completed <= 0:
            return None
        rate = completed / max(elapsed, 1e-9)
        return (total - completed) / max(rate, 1e-9)

    def _write(self, message: str) -> None:
        if self.use_progress_bar and tqdm is not None:
            tqdm.write(f"[{timestamp()}] {message}")
        else:
            log(message)

    def start_model(self, model_name: str, model_position: int, model_total: int) -> None:
        self.model_start_time[model_name] = time.perf_counter()
        if self.use_progress_bar:
            if self.model_bar is not None:
                self.model_bar.close()
            self.model_bar = tqdm(
                total=self.model_fit_plan.get(model_name, 0),
                desc=f"{model_name} fits",
                unit="fit",
                position=1,
                leave=True,
                dynamic_ncols=True,
            )
        self._write(
            f"MODEL START {model_name} ({model_position}/{model_total}) | "
            f"planned_combos={self.model_combo_plan.get(model_name, 0)} | "
            f"planned_fits={self.model_fit_plan.get(model_name, 0)}"
        )

    def start_combo(self, model_name: str, combo_label: str) -> None:
        elapsed_total = time.perf_counter() - self.start_time
        total_pct = 100.0 * self.completed_fits / self.total_fits
        total_eta = self._eta(self.completed_fits, self.total_fits, elapsed_total)
        model_elapsed = time.perf_counter() - self.model_start_time.get(model_name, time.perf_counter())
        model_done = self.model_completed_fits.get(model_name, 0)
        model_total = self.model_fit_plan.get(model_name, 1)
        model_pct = 100.0 * model_done / model_total
        model_eta = self._eta(model_done, model_total, model_elapsed)
        if self.use_progress_bar:
            self._write(
                f"COMBO START model={model_name} | {combo_label} | "
                f"model={model_done}/{model_total} ({model_pct:.1f}%, "
                f"elapsed={format_seconds(model_elapsed)}, eta={format_seconds(model_eta)}) | "
                f"total={self.completed_fits}/{self.total_fits} ({total_pct:.1f}%, eta={format_seconds(total_eta)})"
            )
        else:
            log(
                f"COMBO START model={model_name} | {combo_label} | "
                f"model_fits={model_done}/{model_total} ({model_pct:.1f}%) "
                f"model_elapsed={format_seconds(model_elapsed)} model_eta={format_seconds(model_eta)} | "
                f"total_fits={self.completed_fits}/{self.total_fits} ({total_pct:.1f}%) "
                f"total_eta={format_seconds(total_eta)}"
            )

    def finish_fit(self, model_name: str, fit_label: str) -> None:
        self.completed_fits += 1
        self.model_completed_fits[model_name] = self.model_completed_fits.get(model_name, 0) + 1

        elapsed_total = time.perf_counter() - self.start_time
        total_pct = 100.0 * self.completed_fits / self.total_fits
        total_eta = self._eta(self.completed_fits, self.total_fits, elapsed_total)

        model_elapsed = time.perf_counter() - self.model_start_time.get(model_name, self.start_time)
        model_done = self.model_completed_fits.get(model_name, 0)
        model_total = self.model_fit_plan.get(model_name, 1)
        model_pct = 100.0 * model_done / model_total
        model_eta = self._eta(model_done, model_total, model_elapsed)

        if self.use_progress_bar:
            if self.overall_bar is not None:
                self.overall_bar.update(1)
                self.overall_bar.set_postfix(
                    {
                        "model": model_name,
                        "total": f"{total_pct:.1f}%",
                        "eta": format_seconds(total_eta),
                    },
                    refresh=False,
                )
            if self.model_bar is not None:
                self.model_bar.update(1)
                self.model_bar.set_postfix(
                    {
                        "progress": f"{model_pct:.1f}%",
                        "elapsed": format_seconds(model_elapsed),
                        "eta": format_seconds(model_eta),
                    },
                    refresh=False,
                )
        else:
            log(
                f"FIT DONE model={model_name} | {fit_label} | "
                f"model_fits={model_done}/{model_total} ({model_pct:.1f}%) "
                f"model_elapsed={format_seconds(model_elapsed)} model_eta={format_seconds(model_eta)} | "
                f"total_fits={self.completed_fits}/{self.total_fits} ({total_pct:.1f}%) "
                f"total_elapsed={format_seconds(elapsed_total)} total_eta={format_seconds(total_eta)}"
            )

    def finish_combo(self, model_name: str, combo_label: str) -> None:
        self.completed_combos += 1
        self.model_completed_combos[model_name] = self.model_completed_combos.get(model_name, 0) + 1
        self._write(
            f"COMBO DONE model={model_name} | {combo_label} | "
            f"model_combos={self.model_completed_combos[model_name]}/{self.model_combo_plan.get(model_name, 1)} | "
            f"total_combos={self.completed_combos}/{self.total_combos}"
        )

    def close(self) -> None:
        if self.model_bar is not None:
            self.model_bar.close()
        if self.overall_bar is not None:
            self.overall_bar.close()


def output_path_for_oof(output_dir: Path, model_name: str, feature_set: str, imbalance_option: str) -> Path:
    return output_dir / f"oof__{model_name}__{feature_set}__{imbalance_option}.csv"


def make_oof_frame(
    data: pd.DataFrame,
    *,
    y_prob: np.ndarray,
    outer_fold: np.ndarray,
    model_name: str,
    model_label: str,
    feature_set: str,
    imbalance_option: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": data[SAMPLE_ID_COL].astype(str),
            "outer_fold": outer_fold.astype(int),
            "y_true": data[TARGET_COL].astype(int),
            "y_prob": clipped_probability(y_prob),
            "feature_set": feature_set,
            "model_name": model_name,
            "model_label": model_label,
            "imbalance_option": imbalance_option,
            "sample_type": data[SAMPLE_TYPE_COL].astype(str),
            "climate_type": data[CLIMATE_COL].astype(str),
            "group_id": data[GROUP_COL].astype(str),
        }
    )


def summarize_oof(
    oof: pd.DataFrame,
    *,
    feature_count: int,
    weather_space_axis: str,
    canada_axis: str,
    landcover_axis: str,
    status: str,
    elapsed_seconds: float,
    error: str = "",
) -> dict[str, Any]:
    metrics = probability_metrics(oof["y_true"], oof["y_prob"])
    fold_auprcs: list[float] = []
    for _, fold_df in oof.groupby("outer_fold"):
        fold_auprcs.append(safe_average_precision(fold_df["y_true"].to_numpy(), fold_df["y_prob"].to_numpy()))

    row: dict[str, Any] = {
        "model_name": oof["model_name"].iloc[0],
        "model_label": oof["model_label"].iloc[0],
        "feature_set": oof["feature_set"].iloc[0],
        "imbalance_option": oof["imbalance_option"].iloc[0],
        "weather_space_axis": weather_space_axis,
        "canada_axis": canada_axis,
        "landcover_axis": landcover_axis,
        "feature_count": feature_count,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
    }
    row.update(metrics)
    row["fold_auprc_mean"] = float(np.nanmean(fold_auprcs)) if fold_auprcs else float("nan")
    row["fold_auprc_std"] = float(np.nanstd(fold_auprcs, ddof=1)) if len(fold_auprcs) > 1 else float("nan")
    return row


def make_fold_metric_rows(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, fold_df in oof.groupby("outer_fold"):
        row = {
            "model_name": oof["model_name"].iloc[0],
            "model_label": oof["model_label"].iloc[0],
            "feature_set": oof["feature_set"].iloc[0],
            "imbalance_option": oof["imbalance_option"].iloc[0],
            "outer_fold": int(fold),
        }
        row.update(probability_metrics(fold_df["y_true"], fold_df["y_prob"]))
        rows.append(row)
    return rows


def make_threshold_rows(oof: pd.DataFrame, *, threshold_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = select_thresholds(oof["y_true"], oof["y_prob"])
    for operating_point, threshold in thresholds.items():
        row = {
            "model_name": oof["model_name"].iloc[0],
            "model_label": oof["model_label"].iloc[0],
            "feature_set": oof["feature_set"].iloc[0],
            "imbalance_option": oof["imbalance_option"].iloc[0],
            "threshold_source": threshold_source,
            "operating_point": operating_point,
        }
        row.update(classification_metrics_at_threshold(oof["y_true"], oof["y_prob"], threshold))
        rows.append(row)
    return rows


def make_subgroup_rows(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_info = {
        "model_name": oof["model_name"].iloc[0],
        "model_label": oof["model_label"].iloc[0],
        "feature_set": oof["feature_set"].iloc[0],
        "imbalance_option": oof["imbalance_option"].iloc[0],
    }

    for negative_type in ["Target_0A", "Target_0B1", "Target_0B2"]:
        subgroup = oof.loc[oof["sample_type"].isin(["Target_1", negative_type])].copy()
        if subgroup.empty:
            continue
        row = {
            **model_info,
            "subgroup_type": "sample_type_pair",
            "subgroup_value": f"Target_1_vs_{negative_type}",
        }
        row.update(probability_metrics(subgroup["y_true"], subgroup["y_prob"]))
        rows.append(row)

    for climate_type, subgroup in oof.groupby("climate_type"):
        row = {
            **model_info,
            "subgroup_type": "climate_type",
            "subgroup_value": str(climate_type),
        }
        row.update(probability_metrics(subgroup["y_true"], subgroup["y_prob"]))
        rows.append(row)

    return rows


def make_top_risk_rows(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    expected_n: int,
    outer_group_leakage_n: int,
    missing_feature_n: int,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    probability = oof["y_prob"].to_numpy(dtype=float) if not oof.empty else np.array([])
    duplicate_sample_id_n = int(oof["sample_id"].duplicated().sum()) if "sample_id" in oof else 0
    folds_without_positive_n = 0
    if "outer_fold" in oof and "y_true" in oof:
        for _, fold_df in oof.groupby("outer_fold"):
            if int(fold_df["y_true"].sum()) == 0:
                folds_without_positive_n += 1
    return {
        "model_name": oof["model_name"].iloc[0] if not oof.empty else "",
        "model_label": oof["model_label"].iloc[0] if not oof.empty else "",
        "feature_set": oof["feature_set"].iloc[0] if not oof.empty else "",
        "imbalance_option": oof["imbalance_option"].iloc[0] if not oof.empty else "",
        "status": status,
        "expected_n": expected_n,
        "prediction_n": int(len(oof)),
        "missing_prediction_n": int(max(0, expected_n - len(oof))),
        "duplicate_sample_id_n": duplicate_sample_id_n,
        "nan_probability_n": int(np.isnan(probability).sum()) if len(probability) else 0,
        "inf_probability_n": int(np.isinf(probability).sum()) if len(probability) else 0,
        "outer_group_leakage_n": int(outer_group_leakage_n),
        "folds_without_positive_n": int(folds_without_positive_n),
        "missing_feature_n": int(missing_feature_n),
        "error": error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    if isinstance(rows, pd.DataFrame):
        frame = rows
    else:
        frame = pd.DataFrame(rows)
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


def run_inner_oof_thresholds(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    feature_spec: FeatureSetSpec,
    *,
    model_spec: ModelSpec,
    imbalance_option: str,
    outer_fold: int,
    inner_splits: int,
    random_state: int,
    n_jobs: int,
    progress: ProgressTracker,
) -> dict[str, float]:
    train_data = data.iloc[train_idx]
    y_train = train_data[TARGET_COL].astype(int).to_numpy()
    groups_train = train_data[GROUP_COL].astype(str).to_numpy()
    inner_probability = np.full(len(train_idx), np.nan, dtype=float)

    splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + 1000 + outer_fold)
    for inner_fold, (inner_train_local, inner_valid_local) in enumerate(splitter.split(train_data, y_train, groups_train)):
        actual_train_idx = train_idx[inner_train_local]
        actual_valid_idx = train_idx[inner_valid_local]
        inner_model = make_pipeline(
            model_spec.name,
            feature_spec.features,
            data.iloc[actual_train_idx][TARGET_COL].astype(int),
            imbalance_option=imbalance_option,
            random_state=random_state + outer_fold * 100 + inner_fold,
            n_jobs=n_jobs,
        )
        inner_model.fit(data.iloc[actual_train_idx][feature_spec.features], data.iloc[actual_train_idx][TARGET_COL].astype(int))
        inner_probability[inner_valid_local] = predict_probability(inner_model, data.iloc[actual_valid_idx][feature_spec.features])
        progress.finish_fit(
            model_spec.name,
            f"{feature_spec.name}/{imbalance_option}/outer{outer_fold}/inner{inner_fold}",
        )

    if np.isnan(inner_probability).any():
        raise RuntimeError("inner OOF 예측 누락이 있습니다.")
    return select_thresholds(y_train, inner_probability)


def run_combo(
    data: pd.DataFrame,
    feature_spec: FeatureSetSpec,
    model_spec: ModelSpec,
    imbalance_option: str,
    outer_splits: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    output_dir: Path,
    outer_group_leakage_n: int,
    honest_thresholds: bool,
    inner_splits: int,
    random_state: int,
    n_jobs: int,
    progress: ProgressTracker,
) -> dict[str, Any]:
    start = time.perf_counter()
    missing_features = [column for column in feature_spec.features if column not in data.columns]
    if missing_features:
        raise KeyError(f"피처셋 {feature_spec.name}에 데이터에 없는 컬럼이 있습니다: {missing_features}")

    oof_probability = np.full(len(data), np.nan, dtype=float)
    oof_outer_fold = np.full(len(data), -1, dtype=int)
    honest_threshold_rows: list[dict[str, Any]] = []

    for outer_fold, train_idx, valid_idx in outer_splits:
        thresholds_from_inner: dict[str, float] = {}
        if honest_thresholds:
            thresholds_from_inner = run_inner_oof_thresholds(
                data,
                train_idx,
                feature_spec,
                model_spec=model_spec,
                imbalance_option=imbalance_option,
                outer_fold=outer_fold,
                inner_splits=inner_splits,
                random_state=random_state,
                n_jobs=n_jobs,
                progress=progress,
            )

        y_train = data.iloc[train_idx][TARGET_COL].astype(int)
        model = make_pipeline(
            model_spec.name,
            feature_spec.features,
            y_train,
            imbalance_option=imbalance_option,
            random_state=random_state + outer_fold,
            n_jobs=n_jobs,
        )
        model.fit(data.iloc[train_idx][feature_spec.features], y_train)
        valid_probability = predict_probability(model, data.iloc[valid_idx][feature_spec.features])
        oof_probability[valid_idx] = valid_probability
        oof_outer_fold[valid_idx] = outer_fold
        progress.finish_fit(model_spec.name, f"{feature_spec.name}/{imbalance_option}/outer{outer_fold}")

        if honest_thresholds:
            y_valid = data.iloc[valid_idx][TARGET_COL].astype(int)
            for operating_point, threshold in thresholds_from_inner.items():
                row = {
                    "model_name": model_spec.name,
                    "model_label": model_spec.label,
                    "feature_set": feature_spec.name,
                    "imbalance_option": imbalance_option,
                    "outer_fold": outer_fold,
                    "threshold_source": "inner_oof",
                    "operating_point": operating_point,
                }
                row.update(classification_metrics_at_threshold(y_valid, valid_probability, threshold))
                honest_threshold_rows.append(row)

    if np.isnan(oof_probability).any() or (oof_outer_fold < 0).any():
        raise RuntimeError("outer OOF 예측 누락이 있습니다.")

    oof = make_oof_frame(
        data,
        y_prob=oof_probability,
        outer_fold=oof_outer_fold,
        model_name=model_spec.name,
        model_label=model_spec.label,
        feature_set=feature_spec.name,
        imbalance_option=imbalance_option,
    )
    oof_path = output_path_for_oof(output_dir, model_spec.name, feature_spec.name, imbalance_option)
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")

    elapsed = time.perf_counter() - start
    return metrics_from_oof(
        oof,
        feature_spec=feature_spec,
        outer_group_leakage_n=outer_group_leakage_n,
        status="OK",
        elapsed_seconds=elapsed,
        error="",
        honest_threshold_rows=honest_threshold_rows,
    )


def metrics_from_oof(
    oof: pd.DataFrame,
    *,
    feature_spec: FeatureSetSpec,
    outer_group_leakage_n: int,
    status: str,
    elapsed_seconds: float,
    error: str,
    honest_threshold_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if honest_threshold_rows is None:
        honest_threshold_rows = []
    summary_row = summarize_oof(
        oof,
        feature_count=len(feature_spec.features),
        weather_space_axis=feature_spec.weather_space_axis,
        canada_axis=feature_spec.canada_axis,
        landcover_axis=feature_spec.landcover_axis,
        status=status,
        elapsed_seconds=elapsed_seconds,
        error=error,
    )
    fold_rows = make_fold_metric_rows(oof)
    threshold_rows = make_threshold_rows(oof, threshold_source="full_oof")
    if honest_threshold_rows:
        threshold_rows.extend(honest_threshold_rows)
    subgroup_rows = make_subgroup_rows(oof)
    top_risk_rows = make_top_risk_rows(oof)
    validation_row = make_validation_row(
        oof,
        expected_n=len(oof),
        outer_group_leakage_n=outer_group_leakage_n,
        missing_feature_n=0,
        status=status,
        error=error,
    )
    return {
        "summary_row": summary_row,
        "fold_rows": fold_rows,
        "threshold_rows": threshold_rows,
        "subgroup_rows": subgroup_rows,
        "top_risk_rows": top_risk_rows,
        "validation_row": validation_row,
    }


def failed_combo_rows(
    *,
    model_spec: ModelSpec,
    feature_spec: FeatureSetSpec,
    imbalance_option: str,
    expected_n: int,
    error: str,
    elapsed_seconds: float,
    missing_feature_n: int = 0,
) -> dict[str, Any]:
    summary_row = {
        "model_name": model_spec.name,
        "model_label": model_spec.label,
        "feature_set": feature_spec.name,
        "imbalance_option": imbalance_option,
        "weather_space_axis": feature_spec.weather_space_axis,
        "canada_axis": feature_spec.canada_axis,
        "landcover_axis": feature_spec.landcover_axis,
        "feature_count": len(feature_spec.features),
        "status": "FAIL",
        "elapsed_seconds": elapsed_seconds,
        "error": error,
        "n": expected_n,
        "positive_n": np.nan,
        "negative_n": np.nan,
        "positive_rate": np.nan,
        "auprc": np.nan,
        "auroc": np.nan,
        "brier": np.nan,
        "log_loss": np.nan,
        "fold_auprc_mean": np.nan,
        "fold_auprc_std": np.nan,
    }
    validation_row = {
        "model_name": model_spec.name,
        "model_label": model_spec.label,
        "feature_set": feature_spec.name,
        "imbalance_option": imbalance_option,
        "status": "FAIL",
        "expected_n": expected_n,
        "prediction_n": 0,
        "missing_prediction_n": expected_n,
        "duplicate_sample_id_n": 0,
        "nan_probability_n": 0,
        "inf_probability_n": 0,
        "outer_group_leakage_n": np.nan,
        "folds_without_positive_n": np.nan,
        "missing_feature_n": missing_feature_n,
        "error": error,
    }
    return {
        "summary_row": summary_row,
        "fold_rows": [],
        "threshold_rows": [],
        "subgroup_rows": [],
        "top_risk_rows": [],
        "validation_row": validation_row,
    }


def flush_model_outputs(
    output_dir: Path,
    model_name: str,
    *,
    summary_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    subgroup_rows: list[dict[str, Any]],
    top_risk_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / f"summary__{model_name}.csv", summary_rows)
    write_csv(output_dir / f"fold_metrics__{model_name}.csv", fold_rows)
    write_csv(output_dir / f"threshold_metrics__{model_name}.csv", threshold_rows)
    write_csv(output_dir / f"subgroup_metrics__{model_name}.csv", subgroup_rows)
    write_csv(output_dir / f"top_risk_metrics__{model_name}.csv", top_risk_rows)
    write_csv(output_dir / f"validation_checks__{model_name}.csv", validation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="new_machine_learning Step1: 여러 단일 모델 피처셋 screening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="", help="입력 CSV 경로. 비우면 data/학습데이터/최종_머신러닝_학습데이터.csv 사용")
    parser.add_argument("--output-dir", type=str, default="", help="결과 저장 폴더. 비우면 outputs/step1_single_models 사용")
    parser.add_argument("--models", nargs="*", default=["all"], help="실행할 모델 slug. all 또는 쉼표 구분 가능")
    parser.add_argument("--feature-sets", nargs="*", default=["all"], help="실행할 feature_set. all 또는 쉼표 구분 가능")
    parser.add_argument("--imbalance-options", nargs="*", default=["none", "balanced"], help="none, balanced 중 선택")
    parser.add_argument("--outer-splits", type=int, default=5, help="outer StratifiedGroupKFold 개수")
    parser.add_argument("--inner-splits", type=int, default=4, help="threshold 선택용 inner StratifiedGroupKFold 개수")
    parser.add_argument("--skip-inner-thresholds", action="store_true", help="inner OOF 기반 honest threshold 계산을 생략")
    parser.add_argument("--resume", action="store_true", help="OOF 파일이 이미 있으면 재학습하지 않고 기존 OOF로 지표 재계산")
    parser.add_argument("--check-config", action="store_true", help="데이터/피처셋/의존성만 점검하고 학습하지 않음")
    parser.add_argument("--no-progress-bar", action="store_true", help="tqdm 막대바 대신 기존 텍스트 로그를 사용")
    parser.add_argument("--random-state", type=int, default=20260622, help="분할과 모델 seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="지원 모델의 병렬 작업 수")
    parser.add_argument("--max-combos", type=int, default=0, help="디버그용 최대 조합 수. 0이면 제한 없음")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()
    data_path = Path(args.data) if args.data else root / "data" / "학습데이터" / "최종_머신러닝_학습데이터.csv"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "jsw" / "Analysis" / "new_machine_learning" / "outputs" / "step1_single_models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_feature_specs = build_feature_sets()
    feature_spec_by_name = {spec.name: spec for spec in all_feature_specs}
    selected_feature_names = select_from_csv_argument(
        args.feature_sets,
        list(feature_spec_by_name.keys()),
        argument_name="--feature-sets",
    )
    selected_feature_specs = [feature_spec_by_name[name] for name in selected_feature_names]

    all_models = model_specs()
    selected_model_names = select_from_csv_argument(args.models, MODEL_ORDER, argument_name="--models")
    selected_imbalance_options = select_from_csv_argument(
        args.imbalance_options,
        IMBALANCE_OPTIONS,
        argument_name="--imbalance-options",
    )

    log(f"입력 데이터 로드: {data_path}")
    data = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    required_columns = [TARGET_COL, SAMPLE_ID_COL, SAMPLE_TYPE_COL, GROUP_COL, CLIMATE_COL]
    missing_required = [column for column in required_columns if column not in data.columns]
    if missing_required:
        raise KeyError(f"필수 컬럼이 없습니다: {missing_required}")
    data[TARGET_COL] = data[TARGET_COL].astype(int)

    dependency = dependency_audit()
    dependency.to_csv(output_dir / "dependency_audit.csv", index=False, encoding="utf-8-sig")

    feature_manifest_rows = []
    for spec in all_feature_specs:
        missing = [column for column in spec.features if column not in data.columns]
        feature_manifest_rows.append(
            {
                "feature_set": spec.name,
                "weather_space_axis": spec.weather_space_axis,
                "canada_axis": spec.canada_axis,
                "landcover_axis": spec.landcover_axis,
                "feature_count": len(spec.features),
                "missing_feature_count": len(missing),
                "missing_features": "|".join(missing),
                "features": "|".join(spec.features),
            }
        )
    write_csv(output_dir / "feature_set_manifest.csv", feature_manifest_rows)
    write_json(
        output_dir / "feature_set_manifest.json",
        {
            spec.name: {
                "weather_space_axis": spec.weather_space_axis,
                "canada_axis": spec.canada_axis,
                "landcover_axis": spec.landcover_axis,
                "feature_count": len(spec.features),
                "features": spec.features,
            }
            for spec in all_feature_specs
        },
    )

    split_manifest, outer_splits, outer_group_leakage_n = make_outer_splits(
        data,
        n_splits=args.outer_splits,
        random_state=args.random_state,
    )
    split_manifest.to_csv(output_dir / "split_manifest_outer_cv.csv", index=False, encoding="utf-8-sig")

    honest_thresholds = not args.skip_inner_thresholds
    selected_combos: list[tuple[ModelSpec, FeatureSetSpec, str]] = []
    for model_name in selected_model_names:
        for feature_spec in selected_feature_specs:
            for imbalance_option in selected_imbalance_options:
                selected_combos.append((all_models[model_name], feature_spec, imbalance_option))
    if args.max_combos and args.max_combos > 0:
        selected_combos = selected_combos[: args.max_combos]

    fit_per_combo = planned_fit_count_per_combo(args.outer_splits, args.inner_splits, honest_thresholds)
    model_combo_plan = {model_name: 0 for model_name in selected_model_names}
    model_fit_plan = {model_name: 0 for model_name in selected_model_names}
    for model_spec, _, _ in selected_combos:
        model_combo_plan[model_spec.name] += 1
        model_fit_plan[model_spec.name] += fit_per_combo

    run_config = {
        "script": str(Path(__file__).resolve()),
        "started_at": timestamp(),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "n_rows": int(len(data)),
        "positive_n": int(data[TARGET_COL].sum()),
        "positive_rate": float(data[TARGET_COL].mean()),
        "models": selected_model_names,
        "feature_sets": selected_feature_names,
        "imbalance_options": selected_imbalance_options,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "honest_thresholds": honest_thresholds,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "progress_bar": bool((not args.no_progress_bar) and tqdm is not None),
        "progress_bar_package": "tqdm",
        "progress_bar_available": bool(tqdm is not None),
        "progress_bar_error": TQDM_IMPORT_ERROR,
        "planned_combos": len(selected_combos),
        "planned_fits": len(selected_combos) * fit_per_combo,
        "outer_group_leakage_n": int(outer_group_leakage_n),
        "dependency_audit": dependency.to_dict(orient="records"),
    }
    run_manifest_name = "run_manifest__check_config.json" if args.check_config else "run_manifest__all.json"
    write_json(output_dir / run_manifest_name, run_config)

    log(
        f"데이터 행 수={len(data):,}, positive={int(data[TARGET_COL].sum()):,}, "
        f"positive_rate={data[TARGET_COL].mean():.4f}"
    )
    log(
        f"실행 계획: combos={len(selected_combos):,}, planned_fits={len(selected_combos) * fit_per_combo:,}, "
        f"outer={args.outer_splits}, inner={args.inner_splits}, honest_thresholds={honest_thresholds}"
    )
    log(f"결과 폴더: {output_dir}")

    if args.check_config:
        log("--check-config 지정: 학습 없이 종료합니다.")
        return

    progress = ProgressTracker(
        total_combos=len(selected_combos),
        total_fits=len(selected_combos) * fit_per_combo,
        model_fit_plan=model_fit_plan,
        model_combo_plan=model_combo_plan,
        use_progress_bar=(not args.no_progress_bar) and tqdm is not None,
    )

    all_summary_rows: list[dict[str, Any]] = []
    model_position_lookup = {model_name: idx + 1 for idx, model_name in enumerate(selected_model_names)}
    active_model_name = ""

    for model_name in selected_model_names:
        active_model_name = model_name
        model_spec = all_models[model_name]
        progress.start_model(model_name, model_position_lookup[model_name], len(selected_model_names))
        model_start = time.perf_counter()

        summary_rows: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        subgroup_rows: list[dict[str, Any]] = []
        top_risk_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []

        available, dependency_error = model_available(model_name)
        model_combos = [combo for combo in selected_combos if combo[0].name == model_name]

        for _, feature_spec, imbalance_option in model_combos:
            combo_label = f"feature_set={feature_spec.name} | imbalance={imbalance_option}"
            combo_start = time.perf_counter()
            progress.start_combo(model_name, combo_label)

            if not available:
                result = failed_combo_rows(
                    model_spec=model_spec,
                    feature_spec=feature_spec,
                    imbalance_option=imbalance_option,
                    expected_n=len(data),
                    error=f"dependency_missing: {dependency_error}",
                    elapsed_seconds=time.perf_counter() - combo_start,
                )
                log(f"COMBO SKIP model={model_name} | {combo_label} | dependency_missing")
            else:
                missing_features = [column for column in feature_spec.features if column not in data.columns]
                if missing_features:
                    result = failed_combo_rows(
                        model_spec=model_spec,
                        feature_spec=feature_spec,
                        imbalance_option=imbalance_option,
                        expected_n=len(data),
                        error=f"missing_features: {missing_features}",
                        elapsed_seconds=time.perf_counter() - combo_start,
                        missing_feature_n=len(missing_features),
                    )
                    log(f"COMBO FAIL model={model_name} | {combo_label} | missing_features={missing_features}")
                else:
                    oof_path = output_path_for_oof(output_dir, model_name, feature_spec.name, imbalance_option)
                    try:
                        if args.resume and oof_path.exists():
                            oof = pd.read_csv(oof_path, encoding="utf-8-sig")
                            result = metrics_from_oof(
                                oof,
                                feature_spec=feature_spec,
                                outer_group_leakage_n=outer_group_leakage_n,
                                status="RESUMED_FROM_OOF",
                                elapsed_seconds=time.perf_counter() - combo_start,
                                error="",
                            )
                            log(f"COMBO RESUME model={model_name} | {combo_label} | {oof_path.name}")
                        else:
                            result = run_combo(
                                data,
                                feature_spec,
                                model_spec,
                                imbalance_option,
                                outer_splits,
                                output_dir=output_dir,
                                outer_group_leakage_n=outer_group_leakage_n,
                                honest_thresholds=honest_thresholds,
                                inner_splits=args.inner_splits,
                                random_state=args.random_state,
                                n_jobs=args.n_jobs,
                                progress=progress,
                            )
                    except Exception as exc:
                        result = failed_combo_rows(
                            model_spec=model_spec,
                            feature_spec=feature_spec,
                            imbalance_option=imbalance_option,
                            expected_n=len(data),
                            error=repr(exc),
                            elapsed_seconds=time.perf_counter() - combo_start,
                        )
                        log(f"COMBO FAIL model={model_name} | {combo_label} | error={repr(exc)}")

            summary_rows.append(result["summary_row"])
            fold_rows.extend(result["fold_rows"])
            threshold_rows.extend(result["threshold_rows"])
            subgroup_rows.extend(result["subgroup_rows"])
            top_risk_rows.extend(result["top_risk_rows"])
            validation_rows.append(result["validation_row"])
            all_summary_rows.append(result["summary_row"])

            flush_model_outputs(
                output_dir,
                model_name,
                summary_rows=summary_rows,
                fold_rows=fold_rows,
                threshold_rows=threshold_rows,
                subgroup_rows=subgroup_rows,
                top_risk_rows=top_risk_rows,
                validation_rows=validation_rows,
            )
            write_csv(output_dir / "all_single_models_summary.csv", all_summary_rows)
            progress.finish_combo(model_name, combo_label)

        model_elapsed = time.perf_counter() - model_start
        model_manifest = {
            **run_config,
            "model_name": model_name,
            "model_label": model_spec.label,
            "model_family": model_spec.family,
            "model_started_at": timestamp(),
            "model_elapsed_seconds": model_elapsed,
            "model_combo_count": len(model_combos),
            "success_count": int(sum(1 for row in summary_rows if row.get("status") in {"OK", "RESUMED_FROM_OOF"})),
            "failure_count": int(sum(1 for row in summary_rows if row.get("status") == "FAIL")),
        }
        write_json(output_dir / f"run_manifest__{model_name}.json", model_manifest)
        log(f"MODEL DONE {model_name} | elapsed={format_seconds(model_elapsed)}")

    progress.close()

    final_manifest = dict(run_config)
    final_manifest.update(
        {
            "finished_at": timestamp(),
            "final_active_model": active_model_name,
            "actual_summary_rows": len(all_summary_rows),
            "success_count": int(sum(1 for row in all_summary_rows if row.get("status") in {"OK", "RESUMED_FROM_OOF"})),
            "failure_count": int(sum(1 for row in all_summary_rows if row.get("status") == "FAIL")),
        }
    )
    write_json(output_dir / "run_manifest__all.json", final_manifest)
    write_csv(output_dir / "all_single_models_summary.csv", all_summary_rows)
    log(
        f"STEP1 DONE | success={final_manifest['success_count']} | failure={final_manifest['failure_count']} | "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("사용자 중단 KeyboardInterrupt")
        sys.exit(130)
