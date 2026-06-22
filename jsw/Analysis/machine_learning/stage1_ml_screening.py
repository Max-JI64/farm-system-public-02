from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - dependency audit records this.
    XGBClassifier = None
    XGB_IMPORT_ERROR = str(exc)
else:
    XGB_IMPORT_ERROR = ""

try:
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover - dependency audit records this.
    LGBMClassifier = None
    LGBM_IMPORT_ERROR = str(exc)
else:
    LGBM_IMPORT_ERROR = ""

try:
    from catboost import CatBoostClassifier
except Exception as exc:  # pragma: no cover - dependency audit records this.
    CatBoostClassifier = None
    CATBOOST_IMPORT_ERROR = str(exc)
else:
    CATBOOST_IMPORT_ERROR = ""


warnings.filterwarnings("ignore")

RANDOM_STATE = 20260620
TOP_CALIBRATION_CANDIDATES = 3

ALL_MODEL_NAMES = [
    "RandomForest",
    "ExtraTrees",
    "HistGradientBoosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "학습데이터" / "학습데이터_로지스틱_D2D3.csv").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "학습데이터"
LOGISTIC_DIR = ROOT / "jsw" / "Analysis" / "logistic"
LOGISTIC_OUTPUT_DIR = LOGISTIC_DIR / "outputs"
ML_DIR = ROOT / "jsw" / "Analysis" / "machine_learning"
OUTPUT_DIR = ML_DIR / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
FEATURE_DIR = OUTPUT_DIR / "features"
for directory in [OUTPUT_DIR, METRIC_DIR, PREDICTION_DIR, FEATURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "학습데이터_로지스틱_D2D3.csv"
ENGINEERED_PATH = LOGISTIC_OUTPUT_DIR / "features" / "stage7_engineered_features.csv"
FEATURE_MANIFEST_PATH = LOGISTIC_OUTPUT_DIR / "manifests" / "feature_manifest.json"
STAGE7_RECOMMENDED_PATH = LOGISTIC_OUTPUT_DIR / "features" / "stage7_recommended_feature_set.json"
STAGE8_FEATURE_SETS_PATH = LOGISTIC_OUTPUT_DIR / "features" / "stage8_d2d3_feature_sets.json"
LOCKBOX_PATH = LOGISTIC_OUTPUT_DIR / "splits" / "lockbox_manifest.csv"
OUTER_PATH = LOGISTIC_OUTPUT_DIR / "splits" / "stage75_date_exposure_component_cv_outer_cv_manifest.csv"
INNER_PATH = LOGISTIC_OUTPUT_DIR / "splits" / "stage75_date_exposure_component_cv_inner_cv_manifest.csv"
STRICT_AUDIT_PATH = LOGISTIC_OUTPUT_DIR / "metrics" / "stage75_strict_group_audit.csv"
LOGISTIC_METRICS_PATH = LOGISTIC_OUTPUT_DIR / "tables" / "stage17_ml_comparison_metrics.csv"
LOGISTIC_THRESHOLDS_PATH = LOGISTIC_OUTPUT_DIR / "tables" / "stage17_operating_thresholds.csv"
LOGISTIC_TOP_RISK_PATH = LOGISTIC_OUTPUT_DIR / "tables" / "stage17_top_risk_capture.csv"

BASE_CATEGORICAL = ["기후지형유형"]
LANDCOVER_CATEGORICAL = [
    "토지피복_L1_NAME",
    "토지피복_L2_NAME",
    "토지피복_매칭방식",
    "토지피복_산림유형",
]
LEAKAGE_OR_ID_COLUMNS = {
    "Target",
    "샘플유형",
    "실험안",
    "시간샘플링방식",
    "공간층",
    "샘플가중치",
    "샘플ID",
    "기준시각",
    "위도",
    "경도",
    "기상셀ID",
    "fire_id",
    "원본_fire_id",
    "후보점ID",
    "source_fire_id",
    "모델링_그룹ID",
    "기준날짜",
    "D1_지수기준날짜",
    "캐나다지수_정책",
}

RULE_FEATURES = [
    "도로_10m_이내",
    "도로_30m_이내",
    "시가화_10m_이내",
    "시가화_30m_이내",
    "산림_10m_이내",
    "비산림WUI_x_도로10m",
    "비산림WUI_x_rh_q05",
    "비산림WUI_x_dry0p1",
    "비산림WUI_x_wind5",
    "산림지역_x_rh_q05",
    "침엽수림_x_rh_q05",
    "활엽수림_x_rh_q05",
    "혼효림_x_rh_q05",
    "침엽수림_x_dry5",
    "초지_x_dry0p1",
    "시가화_x_도로10m",
    "영동_x_토지피복도로",
    "영동_x_침엽수림",
    "영동_x_rh_q05_x_wind5",
    "영서_x_비산림WUI_x_dry0p1",
]

FINAL_REDUCED_WITH_FWI_FEATURES = [
    "기후지형유형",
    "월_sin",
    "월_cos",
    "시간_sin",
    "시간_cos",
    "직전24h_최소습도",
    "rh_local_q05",
    "dry_spell_5p0_gt_240h",
    "wind_max_6h",
    "서풍계열_여부",
    "기압변동_3h",
    "log1p_도로거리_m",
    "고도(m)",
    "D1_FWI",
]


@dataclass(frozen=True)
class ModelConfig:
    model: str
    config_id: str
    description: str
    params: dict[str, Any]


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    group: str
    description: str
    features: list[str]


FEATURE_SET_META: dict[str, dict[str, str]] = {}


def positive_scale_weight(y: pd.Series | np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    positives = max(1, int(y.sum()))
    negatives = max(1, int((1 - y).sum()))
    return float(negatives / positives)


def dependency_audit() -> pd.DataFrame:
    rows = [
        {"package": "sklearn", "available": True, "error": ""},
        {"package": "xgboost", "available": XGBClassifier is not None, "error": XGB_IMPORT_ERROR},
        {"package": "lightgbm", "available": LGBMClassifier is not None, "error": LGBM_IMPORT_ERROR},
        {"package": "catboost", "available": CatBoostClassifier is not None, "error": CATBOOST_IMPORT_ERROR},
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(METRIC_DIR / "ml_stage1_v2_dependency_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def model_configs() -> list[ModelConfig]:
    configs = [
        ModelConfig("RandomForest", "rf_base", "RF 기본", {"class_weight": None}),
        ModelConfig("RandomForest", "rf_balanced_subsample", "RF balanced_subsample", {"class_weight": "balanced_subsample"}),
        ModelConfig("ExtraTrees", "et_base", "ExtraTrees 기본", {"class_weight": None}),
        ModelConfig("ExtraTrees", "et_balanced", "ExtraTrees balanced", {"class_weight": "balanced"}),
        ModelConfig("HistGradientBoosting", "hgb_base", "HGB 기본", {"class_weight": None}),
        ModelConfig("HistGradientBoosting", "hgb_balanced", "HGB balanced", {"class_weight": "balanced"}),
    ]
    if XGBClassifier is not None:
        configs.extend(
            [
                ModelConfig("XGBoost", "xgb_base", "XGBoost 기본", {"scale_pos_weight": 1.0}),
                ModelConfig("XGBoost", "xgb_weighted", "XGBoost scale_pos_weight", {"scale_pos_weight": "auto"}),
            ]
        )
    if LGBMClassifier is not None:
        configs.extend(
            [
                ModelConfig("LightGBM", "lgbm_base", "LightGBM 기본", {"class_weight": None}),
                ModelConfig("LightGBM", "lgbm_balanced", "LightGBM balanced", {"class_weight": "balanced"}),
            ]
        )
    if CatBoostClassifier is not None:
        configs.extend(
            [
                ModelConfig("CatBoost", "cat_base", "CatBoost 기본", {"auto_class_weights": None}),
                ModelConfig("CatBoost", "cat_balanced", "CatBoost Balanced", {"auto_class_weights": "Balanced"}),
            ]
        )
    return configs


def build_estimator(config: ModelConfig, y_train: pd.Series | np.ndarray) -> Any:
    if config.model == "RandomForest":
        return RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight=config.params["class_weight"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if config.model == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight=config.params["class_weight"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if config.model == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.01,
            class_weight=config.params["class_weight"],
            random_state=RANDOM_STATE,
        )
    if config.model == "XGBoost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost를 사용할 수 없습니다.")
        scale_pos_weight = config.params["scale_pos_weight"]
        if scale_pos_weight == "auto":
            scale_pos_weight = positive_scale_weight(y_train)
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
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        )
    if config.model == "LightGBM":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm을 사용할 수 없습니다.")
        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.055,
            num_leaves=31,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary",
            class_weight=config.params["class_weight"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    if config.model == "CatBoost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost를 사용할 수 없습니다.")
        return CatBoostClassifier(
            iterations=160,
            depth=5,
            learning_rate=0.055,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights=config.params["auto_class_weights"],
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    raise KeyError(config.model)


def make_pipeline(features: list[str], categorical: list[str], config: ModelConfig, y_train) -> Pipeline:
    categorical_used = [col for col in categorical if col in features]
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
    return Pipeline([("preprocess", preprocessor), ("model", build_estimator(config, y_train))])


def clipped_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)


def predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return clipped_probability(model.predict_proba(x)[:, 1])
    score = model.decision_function(x)
    return clipped_probability(1.0 / (1.0 + np.exp(-score)))


def probability_metrics(y_true, probability) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    probability = clipped_probability(probability)
    return {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "auprc": float(average_precision_score(y_true, probability)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def threshold_curve(y_true, probability) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    probability = clipped_probability(probability)
    frame = pd.DataFrame({"y": y_true, "probability": probability})
    grouped = (
        frame.groupby("probability", as_index=False)
        .agg(n=("y", "size"), pos=("y", "sum"))
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    grouped["neg"] = grouped["n"] - grouped["pos"]
    grouped["tp"] = grouped["pos"].cumsum()
    grouped["fp"] = grouped["neg"].cumsum()
    total_pos = int(frame["y"].sum())
    total_neg = int((1 - frame["y"]).sum())
    grouped["fn"] = total_pos - grouped["tp"]
    grouped["tn"] = total_neg - grouped["fp"]
    grouped["selected_n"] = grouped["tp"] + grouped["fp"]
    grouped["precision"] = grouped["tp"] / grouped["selected_n"]
    grouped["recall"] = np.where(total_pos > 0, grouped["tp"] / total_pos, np.nan)
    grouped["specificity"] = np.where(total_neg > 0, grouped["tn"] / total_neg, np.nan)
    denominator = grouped["precision"] + grouped["recall"]
    grouped["f1"] = np.where(denominator > 0, 2 * grouped["precision"] * grouped["recall"] / denominator, 0.0)
    grouped["balanced_accuracy"] = (grouped["recall"] + grouped["specificity"]) / 2
    return grouped.rename(columns={"probability": "threshold"})


def best_f1_threshold(y_true, probability) -> float:
    curve = threshold_curve(y_true, probability).copy()
    curve["selected_rate"] = curve["selected_n"] / len(y_true)
    curve = curve.sort_values(
        ["f1", "balanced_accuracy", "selected_rate", "threshold"],
        ascending=[False, False, True, False],
    )
    return float(curve.iloc[0]["threshold"])


def threshold_for_recall(y_true, probability, target_recall: float) -> float:
    curve = threshold_curve(y_true, probability)
    candidates = curve.loc[curve["recall"] >= target_recall]
    if candidates.empty:
        return 0.0
    return float(candidates.iloc[0]["threshold"])


def classification_metrics_at_threshold(y_true, probability, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    pred = (clipped_probability(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "selected_n": int(pred.sum()),
        "selected_rate": float(pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
    }


def add_stage9_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["영동_여부"] = data["기후지형유형"].eq("영동 해안형").astype(np.int8)
    data["영서_여부"] = data["기후지형유형"].eq("영서 내륙형").astype(np.int8)
    data["도로_10m_이내"] = data["도로_최단거리_m"].le(10).astype(np.int8)
    data["도로_30m_이내"] = data["도로_최단거리_m"].le(30).astype(np.int8)
    data["시가화_10m_이내"] = data["시가화_최단거리_m"].le(10).astype(np.int8)
    data["시가화_30m_이내"] = data["시가화_최단거리_m"].le(30).astype(np.int8)
    data["산림_10m_이내"] = data["산림_최단거리_m"].le(10).astype(np.int8)
    data["비산림WUI_x_도로10m"] = data["비산림_WUI_접경후보"] * data["도로_10m_이내"]
    data["비산림WUI_x_rh_q05"] = data["비산림_WUI_접경후보"] * data["rh_local_q05"]
    data["비산림WUI_x_dry0p1"] = data["비산림_WUI_접경후보"] * data["dry_spell_0p1_gt_24h"]
    data["비산림WUI_x_wind5"] = data["비산림_WUI_접경후보"] * data["wind_max_6h_ge_5"]
    data["산림지역_x_rh_q05"] = data["토지피복_산림지역"] * data["rh_local_q05"]
    data["침엽수림_x_rh_q05"] = data["토지피복_침엽수림"] * data["rh_local_q05"]
    data["활엽수림_x_rh_q05"] = data["토지피복_활엽수림"] * data["rh_local_q05"]
    data["혼효림_x_rh_q05"] = data["토지피복_혼효림"] * data["rh_local_q05"]
    data["침엽수림_x_dry5"] = data["토지피복_침엽수림"] * data["dry_spell_5p0_gt_240h"]
    data["초지_x_dry0p1"] = data["토지피복_초지"] * data["dry_spell_0p1_gt_24h"]
    data["시가화_x_도로10m"] = data["토지피복_시가화건조지역"] * data["도로_10m_이내"]
    data["영동_x_토지피복도로"] = data["영동_여부"] * data["토지피복_도로"]
    data["영동_x_침엽수림"] = data["영동_여부"] * data["토지피복_침엽수림"]
    data["영동_x_rh_q05_x_wind5"] = data["영동_여부"] * data["rh_local_q05"] * data["wind_max_6h_ge_5"]
    data["영서_x_비산림WUI_x_dry0p1"] = (
        data["영서_여부"] * data["비산림_WUI_접경후보"] * data["dry_spell_0p1_gt_24h"]
    )
    return data


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def load_feature_sets() -> dict[str, list[str]]:
    global FEATURE_SET_META

    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    stage7 = json.loads(STAGE7_RECOMMENDED_PATH.read_text(encoding="utf-8"))
    stage8 = json.loads(STAGE8_FEATURE_SETS_PATH.read_text(encoding="utf-8"))

    specs = [
        FeatureSetSpec("M1", "diagnostic_m123", "날씨·공간 + FA + D-1 캐나다 지수", manifest["model_sets"]["M1"]),
        FeatureSetSpec("M2", "diagnostic_m123", "날씨·공간 + D-1 캐나다 지수", manifest["model_sets"]["M2"]),
        FeatureSetSpec("M3", "diagnostic_m123", "날씨·공간", manifest["model_sets"]["M3"]),
        FeatureSetSpec(
            "STAGE7_RECOMMENDED",
            "logistic_final_guided",
            "로지스틱 Stage7 추천 EDA 파생 피처",
            stage7["features"],
        ),
        FeatureSetSpec(
            "PLUS_LANDCOVER",
            "logistic_final_guided",
            "Stage7 추천 피처 + 토지피복",
            stage8["PLUS_LANDCOVER"],
        ),
        FeatureSetSpec(
            "PLUS_LANDCOVER_RULES_ANOVA_PROXY",
            "logistic_final_guided",
            "PLUS_LANDCOVER + Stage9 규칙 피처",
            dedupe(stage8["PLUS_LANDCOVER"] + RULE_FEATURES),
        ),
        FeatureSetSpec(
            "FINAL_REDUCED_WITH_FWI_PROXY",
            "logistic_final_guided",
            "Stage16 최종 해석 모델 FINAL_REDUCED_WITH_FWI 피처",
            FINAL_REDUCED_WITH_FWI_FEATURES,
        ),
    ]
    feature_sets = {spec.name: dedupe(spec.features) for spec in specs}
    FEATURE_SET_META = {
        spec.name: {"feature_group": spec.group, "feature_description": spec.description} for spec in specs
    }
    for name, features in feature_sets.items():
        bad = sorted(set(features) & LEAKAGE_OR_ID_COLUMNS)
        if bad:
            raise ValueError(f"{name}에 누수/식별자 변수가 포함되어 있습니다: {bad}")

    payload = {
        name: {**FEATURE_SET_META[name], "n_features": len(features), "features": features}
        for name, features in feature_sets.items()
    }
    (FEATURE_DIR / "ml_stage1_v2_feature_sets.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "feature_set": name,
                **FEATURE_SET_META[name],
                "n_features": len(features),
                "features": "|".join(features),
            }
            for name, features in feature_sets.items()
        ]
    ).to_csv(METRIC_DIR / "ml_stage1_v2_feature_set_metadata.csv", index=False, encoding="utf-8-sig")
    return feature_sets


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[str]]:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["기준시각"], low_memory=False)
    engineered = pd.read_csv(ENGINEERED_PATH, encoding="utf-8-sig")
    engineered = engineered[[col for col in engineered.columns if not col.endswith(".1")]].copy()
    engineered_cols = [col for col in engineered.columns if col not in {"Target", "샘플유형"}]
    data = data.merge(engineered[engineered_cols], on="샘플ID", how="left", validate="one_to_one")
    data = add_stage9_features(data)
    feature_sets = load_feature_sets()
    all_features = sorted(set().union(*feature_sets.values()))
    missing = sorted(set(all_features) - set(data.columns))
    if missing:
        raise KeyError(f"데이터에 없는 피처: {missing}")

    categorical = [col for col in BASE_CATEGORICAL + LANDCOVER_CATEGORICAL if col in all_features]
    for col in all_features:
        if col in categorical:
            data[col] = data[col].fillna("미상").astype(str)
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    lockbox = pd.read_csv(LOCKBOX_PATH, encoding="utf-8-sig")
    outer = pd.read_csv(OUTER_PATH, encoding="utf-8-sig")
    inner = pd.read_csv(INNER_PATH, encoding="utf-8-sig")
    development_ids = set(lockbox.loc[lockbox["split"].eq("development"), "샘플ID"])
    lockbox_ids = set(lockbox.loc[lockbox["split"].eq("lockbox_test"), "샘플ID"])
    data = data.loc[data["샘플ID"].isin(development_ids)].copy()

    if len(pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=["샘플ID"])) != 17045:
        raise ValueError("전체 행 수가 17,045가 아닙니다.")
    if len(data) != 13632:
        raise ValueError(f"development 행 수 불일치: {len(data)}")
    if int(data["Target"].sum()) != 1242:
        raise ValueError(f"development Target 1 수 불일치: {int(data['Target'].sum())}")
    if data["샘플ID"].duplicated().any() or data["샘플ID"].isna().any():
        raise ValueError("샘플ID 중복 또는 결측이 있습니다.")
    if set(data["샘플ID"]) != set(outer["샘플ID"]):
        raise ValueError("development 데이터와 strict outer manifest 샘플ID가 다릅니다.")
    if set(outer["샘플ID"]) & lockbox_ids or set(inner["샘플ID"]) & lockbox_ids:
        raise ValueError("strict CV manifest에 lockbox 샘플이 포함되어 있습니다.")

    strict_audit = pd.read_csv(STRICT_AUDIT_PATH, encoding="utf-8-sig")
    date_audit = strict_audit.loc[strict_audit["strategy"].eq("date_exposure_component_cv")].iloc[0]
    leak_cols = ["outer_model_group_leak", "outer_exposure_leak", "outer_strict_date_leak", "outer_positive_actual_date_leak"]
    for col in leak_cols:
        if int(date_audit[col]) != 0:
            raise ValueError(f"strict split audit 실패: {col}={date_audit[col]}")

    pd.DataFrame(
        [
            {
                "dataset": "full_after_join",
                "n": 17045,
                "actual_joined_n": int(len(pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=["샘플ID"]))),
                "development_n": int(len(data)),
                "development_positive_n": int(data["Target"].sum()),
                "development_positive_rate": float(data["Target"].mean()),
                "outer_manifest_n": int(len(outer)),
                "inner_manifest_n": int(len(inner)),
                **{col: int(date_audit[col]) for col in leak_cols},
            }
        ]
    ).to_csv(METRIC_DIR / "ml_stage1_v2_data_audit.csv", index=False, encoding="utf-8-sig")
    return data, outer, inner, feature_sets, categorical


def train_and_tune(
    data: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = data.set_index("샘플ID", drop=False)
    configs = model_configs()
    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for feature_set, features in feature_sets.items():
        for model_name in ALL_MODEL_NAMES:
            model_configs_for_name = [cfg for cfg in configs if cfg.model == model_name]
            if not model_configs_for_name:
                continue
            for outer_fold in sorted(outer["outer_fold"].unique()):
                val_ids = outer.loc[outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
                train_ids = outer.loc[~outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
                inner_part = inner.loc[inner["outer_fold"].eq(outer_fold)].set_index("샘플ID")

                config_scores = []
                for config in model_configs_for_name:
                    scores = []
                    for inner_fold in sorted(inner_part["inner_fold"].unique()):
                        inner_val_ids = inner_part.index[inner_part["inner_fold"].eq(inner_fold)].tolist()
                        inner_train_ids = inner_part.index[~inner_part["inner_fold"].eq(inner_fold)].tolist()
                        pipeline = make_pipeline(features, categorical, config, indexed.loc[inner_train_ids, "Target"])
                        pipeline.fit(indexed.loc[inner_train_ids, features], indexed.loc[inner_train_ids, "Target"])
                        probability = predict_probability(pipeline, indexed.loc[inner_val_ids, features])
                        score = average_precision_score(indexed.loc[inner_val_ids, "Target"], probability)
                        scores.append(float(score))
                        tuning_rows.append(
                            {
                                "feature_set": feature_set,
                                "feature_group": FEATURE_SET_META[feature_set]["feature_group"],
                                "model": model_name,
                                "outer_fold": int(outer_fold),
                                "inner_fold": int(inner_fold),
                                "config_id": config.config_id,
                                "description": config.description,
                                "auprc": float(score),
                            }
                        )
                    config_scores.append(
                        {
                            "config": config,
                            "mean_auprc": float(np.mean(scores)),
                            "std_auprc": float(np.std(scores, ddof=0)),
                        }
                    )

                best = sorted(config_scores, key=lambda row: (-row["mean_auprc"], row["config"].config_id))[0]
                config = best["config"]
                selected_rows.append(
                    {
                        "feature_set": feature_set,
                        "feature_group": FEATURE_SET_META[feature_set]["feature_group"],
                        "model": model_name,
                        "outer_fold": int(outer_fold),
                        "selected_config_id": config.config_id,
                        "selected_description": config.description,
                        "inner_mean_auprc": best["mean_auprc"],
                        "inner_std_auprc": best["std_auprc"],
                    }
                )

                pipeline = make_pipeline(features, categorical, config, indexed.loc[train_ids, "Target"])
                pipeline.fit(indexed.loc[train_ids, features], indexed.loc[train_ids, "Target"])
                train_probability = predict_probability(pipeline, indexed.loc[train_ids, features])
                val_probability = predict_probability(pipeline, indexed.loc[val_ids, features])

                for dataset_name, ids, probability in [
                    ("train", train_ids, train_probability),
                    ("validation", val_ids, val_probability),
                ]:
                    fold_rows.append(
                        {
                            "feature_set": feature_set,
                            "feature_group": FEATURE_SET_META[feature_set]["feature_group"],
                            "model": model_name,
                            "outer_fold": int(outer_fold),
                            "dataset": dataset_name,
                            "score_type": "raw",
                            "selected_config_id": config.config_id,
                            **probability_metrics(indexed.loc[ids, "Target"], probability),
                        }
                    )

                for sample_id, probability in zip(val_ids, val_probability):
                    row = indexed.loc[sample_id]
                    oof_rows.append(
                        {
                            "샘플ID": sample_id,
                            "feature_set": feature_set,
                            "feature_group": FEATURE_SET_META[feature_set]["feature_group"],
                            "model": model_name,
                            "outer_fold": int(outer_fold),
                            "Target": int(row["Target"]),
                            "샘플유형": row["샘플유형"],
                            "기후지형유형": row["기후지형유형"],
                            "selected_config_id": config.config_id,
                            "score_raw": float(probability),
                            "score_sigmoid": np.nan,
                            "score_isotonic": np.nan,
                            "score_calibrated": float(probability),
                            "calibration_method": "raw",
                            "run_status": "OK",
                        }
                    )
                print(
                    f"{feature_set} / {model_name} / outer {outer_fold}: "
                    f"{config.config_id}, val AUPRC={average_precision_score(indexed.loc[val_ids, 'Target'], val_probability):.4f}"
                )

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(METRIC_DIR / "ml_stage1_v2_selected_configs.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(oof_rows), pd.DataFrame(fold_rows), pd.DataFrame(tuning_rows)


def prediction_long_frame(predictions: pd.DataFrame, include_calibrated_best: bool = True) -> pd.DataFrame:
    rows = []
    base_cols = [
        "샘플ID",
        "feature_set",
        "feature_group",
        "model",
        "outer_fold",
        "Target",
        "샘플유형",
        "기후지형유형",
        "selected_config_id",
        "run_status",
    ]
    for score_type, score_col in {"raw": "score_raw", "sigmoid": "score_sigmoid", "isotonic": "score_isotonic"}.items():
        part = predictions.loc[predictions[score_col].notna(), base_cols + [score_col]].copy()
        if part.empty:
            continue
        part = part.rename(columns={score_col: "score"})
        part["score_type"] = score_type
        rows.append(part)
    if include_calibrated_best:
        part = predictions.loc[
            predictions["calibration_method"].ne("raw") & predictions["score_calibrated"].notna(),
            base_cols + ["score_calibrated"],
        ].copy()
        if not part.empty:
            part = part.rename(columns={"score_calibrated": "score"})
            part["score_type"] = "calibrated_best"
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def selected_config_summary(part: pd.DataFrame) -> str:
    values = sorted(part["selected_config_id"].dropna().astype(str).unique())
    return values[0] if len(values) == 1 else "|".join(values)


def model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    logistic = pd.read_csv(LOGISTIC_METRICS_PATH, encoding="utf-8-sig")
    logistic_baseline = logistic.loc[logistic["model"].eq("PLUS_LANDCOVER_RULES_ANOVA")].iloc[0]
    rows = []
    long = prediction_long_frame(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        row = {
            "feature_set": feature_set,
            "feature_group": str(part["feature_group"].iloc[0]),
            "feature_description": FEATURE_SET_META.get(feature_set, {}).get("feature_description", ""),
            "model": model,
            "score_type": score_type,
            "run_status": "OK",
            "selected_config_id": selected_config_summary(part),
            **probability_metrics(part["Target"], part["score"]),
        }
        threshold = best_f1_threshold(part["Target"], part["score"])
        row.update({f"best_f1_{key}": value for key, value in classification_metrics_at_threshold(part["Target"], part["score"], threshold).items()})
        row["max_vif"] = "not_applicable"
        row["delta_auprc_vs_logistic"] = float(row["auprc"] - logistic_baseline["auprc"])
        row["delta_brier_vs_logistic"] = float(row["brier"] - logistic_baseline["brier"])
        rows.append(row)

    comparison = pd.DataFrame(rows)
    missing_models = [name for name in ALL_MODEL_NAMES if name not in {cfg.model for cfg in model_configs()}]
    for feature_set in FEATURE_SET_META:
        for model_name in missing_models:
            comparison = pd.concat(
                [
                    comparison,
                    pd.DataFrame(
                        [
                            {
                                "feature_set": feature_set,
                                "feature_group": FEATURE_SET_META[feature_set]["feature_group"],
                                "feature_description": FEATURE_SET_META[feature_set]["feature_description"],
                                "model": model_name,
                                "score_type": "raw",
                                "run_status": "NOT_RUN",
                                "selected_config_id": "",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    comparison = comparison.sort_values(["auprc", "brier"], ascending=[False, True], na_position="last")
    return comparison


def fit_calibrators(raw_probability: np.ndarray, y_true: np.ndarray) -> tuple[LogisticRegression, IsotonicRegression]:
    raw_probability = clipped_probability(raw_probability)
    y_true = np.asarray(y_true).astype(int)
    sigmoid = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    sigmoid.fit(raw_probability.reshape(-1, 1), y_true)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_probability, y_true)
    return sigmoid, isotonic


def select_calibration_candidates(raw_comparison: pd.DataFrame) -> pd.DataFrame:
    raw = raw_comparison.loc[raw_comparison["score_type"].eq("raw") & raw_comparison["run_status"].eq("OK")].copy()
    candidates = [raw.head(TOP_CALIBRATION_CANDIDATES)]
    for feature_set in ["M1", "M2", "M3"]:
        part = raw.loc[raw["feature_set"].eq(feature_set)].head(1)
        if not part.empty:
            candidates.append(part)
    guided = raw.loc[raw["feature_group"].eq("logistic_final_guided")].head(1)
    if not guided.empty:
        candidates.append(guided)
    return pd.concat(candidates, ignore_index=True).drop_duplicates(["feature_set", "model"]).reset_index(drop=True)


def apply_calibration(
    data: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
    predictions: pd.DataFrame,
    top_candidates: pd.DataFrame,
) -> pd.DataFrame:
    indexed = data.set_index("샘플ID", drop=False)
    selected = pd.read_csv(METRIC_DIR / "ml_stage1_v2_selected_configs.csv", encoding="utf-8-sig")
    config_by_id = {cfg.config_id: cfg for cfg in model_configs()}
    predictions = predictions.copy()

    for candidate in top_candidates.itertuples(index=False):
        feature_set = str(candidate.feature_set)
        model_name = str(candidate.model)
        features = feature_sets[feature_set]
        print(f"Calibration: {feature_set} / {model_name}")
        for outer_fold in sorted(outer["outer_fold"].unique()):
            val_ids = outer.loc[outer["outer_fold"].eq(outer_fold), "샘플ID"].tolist()
            inner_part = inner.loc[inner["outer_fold"].eq(outer_fold)].set_index("샘플ID")
            train_ids = inner_part.index.tolist()
            selected_row = selected.loc[
                selected["feature_set"].eq(feature_set)
                & selected["model"].eq(model_name)
                & selected["outer_fold"].eq(outer_fold)
            ].iloc[0]
            config = config_by_id[str(selected_row["selected_config_id"])]
            inner_oof = pd.Series(index=train_ids, dtype=float)
            for inner_fold in sorted(inner_part["inner_fold"].unique()):
                inner_val_ids = inner_part.index[inner_part["inner_fold"].eq(inner_fold)].tolist()
                inner_train_ids = inner_part.index[~inner_part["inner_fold"].eq(inner_fold)].tolist()
                pipeline = make_pipeline(features, categorical, config, indexed.loc[inner_train_ids, "Target"])
                pipeline.fit(indexed.loc[inner_train_ids, features], indexed.loc[inner_train_ids, "Target"])
                inner_oof.loc[inner_val_ids] = predict_probability(pipeline, indexed.loc[inner_val_ids, features])
            sigmoid, isotonic = fit_calibrators(inner_oof.to_numpy(), indexed.loc[train_ids, "Target"].to_numpy())
            mask = (
                predictions["feature_set"].eq(feature_set)
                & predictions["model"].eq(model_name)
                & predictions["outer_fold"].eq(outer_fold)
            )
            raw_val = predictions.loc[mask, "score_raw"].to_numpy()
            predictions.loc[mask, "score_sigmoid"] = clipped_probability(sigmoid.predict_proba(raw_val.reshape(-1, 1))[:, 1])
            predictions.loc[mask, "score_isotonic"] = clipped_probability(isotonic.predict(raw_val))

    comparison = model_comparison(predictions)
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
            }
        )
        mask = predictions["feature_set"].eq(feature_set) & predictions["model"].eq(model)
        predictions.loc[mask, "score_calibrated"] = predictions.loc[mask, f"score_{best['score_type']}"]
        predictions.loc[mask, "calibration_method"] = str(best["score_type"])
    pd.DataFrame(calibration_rows).to_csv(METRIC_DIR / "ml_stage1_v2_calibration_selection.csv", index=False, encoding="utf-8-sig")
    return predictions


def make_threshold_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        thresholds = {
            "fixed_0.50": 0.5,
            "best_f1": best_f1_threshold(part["Target"], part["score"]),
            "recall_ge_0.50": threshold_for_recall(part["Target"], part["score"], 0.50),
            "recall_ge_0.70": threshold_for_recall(part["Target"], part["score"], 0.70),
            "recall_ge_0.90": threshold_for_recall(part["Target"], part["score"], 0.90),
        }
        for operating_point, threshold in thresholds.items():
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "operating_point": operating_point,
                    **classification_metrics_at_threshold(part["Target"], part["score"], threshold),
                }
            )
    return pd.DataFrame(rows)


def make_top_risk_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame(predictions)
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        part = part.sort_values("score", ascending=False).reset_index(drop=True)
        total_positive = int(part["Target"].sum())
        base_rate = float(part["Target"].mean())
        for pct in [0.05, 0.10, 0.20]:
            n = int(math.ceil(len(part) * pct))
            top = part.iloc[:n]
            precision = float(top["Target"].mean())
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "top_pct": pct,
                    "selected_n": n,
                    "selected_rate": float(n / len(part)),
                    "threshold_min": float(top["score"].min()),
                    "captured_positive_n": int(top["Target"].sum()),
                    "total_positive_n": total_positive,
                    "capture_rate_recall": float(top["Target"].sum() / total_positive) if total_positive else np.nan,
                    "precision": precision,
                    "base_positive_rate": base_rate,
                    "lift_vs_base": float(precision / base_rate) if base_rate else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_subgroup_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = prediction_long_frame(predictions)
    negative_types = [value for value in sorted(long["샘플유형"].dropna().unique()) if value != "Target_1"]
    for (feature_set, model, score_type), part in long.groupby(["feature_set", "model", "score_type"], observed=True):
        positives = part.loc[part["샘플유형"].eq("Target_1")]
        for negative_type in negative_types:
            subset = pd.concat([positives, part.loc[part["샘플유형"].eq(negative_type)]], ignore_index=True)
            if subset["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "subgroup_type": "negative_type",
                    "subgroup": negative_type,
                    **probability_metrics(subset["Target"], subset["score"]),
                }
            )
        for climate, subset in part.groupby("기후지형유형", observed=True):
            if subset["Target"].nunique() < 2:
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_group": str(part["feature_group"].iloc[0]),
                    "model": model,
                    "score_type": score_type,
                    "subgroup_type": "기후지형유형",
                    "subgroup": climate,
                    **probability_metrics(subset["Target"], subset["score"]),
                }
            )
    return pd.DataFrame(rows)


def make_rank_tables(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = comparison.loc[comparison["score_type"].eq("raw") & comparison["run_status"].eq("OK")].copy()
    raw["rank_in_feature_set"] = raw.groupby("feature_set")["auprc"].rank(method="first", ascending=False).astype(int)
    rank = raw.sort_values(["feature_set", "rank_in_feature_set"])
    groups = []
    for group_name, part in [
        ("diagnostic_m123", raw.loc[raw["feature_group"].eq("diagnostic_m123")]),
        ("logistic_final_guided", raw.loc[raw["feature_group"].eq("logistic_final_guided")]),
        ("overall", raw),
    ]:
        if part.empty:
            continue
        best = part.sort_values(["auprc", "brier"], ascending=[False, True]).iloc[0].to_dict()
        best["result_group"] = group_name
        groups.append(best)
    return rank, pd.DataFrame(groups)


def write_logistic_combined_table(comparison: pd.DataFrame) -> pd.DataFrame:
    logistic = pd.read_csv(LOGISTIC_METRICS_PATH, encoding="utf-8-sig")
    logistic_rows = []
    for row in logistic.itertuples(index=False):
        logistic_rows.append(
            {
                "model_family": "logistic",
                "feature_set": getattr(row, "model"),
                "feature_group": "logistic_stage17",
                "model": getattr(row, "model"),
                "score_type": "oof",
                "run_status": "OK",
                "n": getattr(row, "n"),
                "positive_n": getattr(row, "positive_n"),
                "positive_rate": getattr(row, "positive_rate"),
                "auprc": getattr(row, "auprc"),
                "auroc": getattr(row, "auroc"),
                "brier": getattr(row, "brier"),
                "log_loss": getattr(row, "log_loss"),
                "best_f1_f1": getattr(row, "best_f1_f1"),
                "best_f1_precision": getattr(row, "best_f1_precision"),
                "best_f1_recall": getattr(row, "best_f1_recall"),
            }
        )
    ml_rows = comparison.copy()
    ml_rows.insert(0, "model_family", "machine_learning")
    common = sorted(set(pd.DataFrame(logistic_rows).columns) | set(ml_rows.columns))
    combined = pd.concat(
        [pd.DataFrame(logistic_rows).reindex(columns=common), ml_rows.reindex(columns=common)],
        ignore_index=True,
    ).sort_values(["auprc", "brier"], ascending=[False, True], na_position="last")
    combined.to_csv(METRIC_DIR / "ml_stage1_v2_with_logistic_comparison.csv", index=False, encoding="utf-8-sig")
    return combined


def write_summary(
    comparison: pd.DataFrame,
    rank: pd.DataFrame,
    best_by_group: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_risk: pd.DataFrame,
    subgroup: pd.DataFrame,
) -> None:
    logistic = pd.read_csv(LOGISTIC_METRICS_PATH, encoding="utf-8-sig")
    baseline = logistic.loc[logistic["model"].eq("PLUS_LANDCOVER_RULES_ANOVA")].iloc[0]
    raw = comparison.loc[comparison["score_type"].eq("raw") & comparison["run_status"].eq("OK")].copy()
    best = raw.iloc[0]
    best_thresholds = thresholds.loc[
        thresholds["feature_set"].eq(best["feature_set"])
        & thresholds["model"].eq(best["model"])
        & thresholds["score_type"].eq("raw")
    ]
    best_top = top_risk.loc[
        top_risk["feature_set"].eq(best["feature_set"])
        & top_risk["model"].eq(best["model"])
        & top_risk["score_type"].eq("raw")
    ]
    best_subgroup = subgroup.loc[
        subgroup["feature_set"].eq(best["feature_set"])
        & subgroup["model"].eq(best["model"])
        & subgroup["score_type"].eq("raw")
        & subgroup["subgroup_type"].eq("negative_type")
    ]
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
    ]
    rank_cols = [
        "feature_set",
        "rank_in_feature_set",
        "model",
        "auprc",
        "auroc",
        "brier",
        "log_loss",
        "best_f1_f1",
        "delta_auprc_vs_logistic",
    ]
    lines = [
        "# 머신러닝 1차 v2 결과",
        "",
        "## 1. 실행 목적",
        "",
        "- 로지스틱 Step17 최종 결과를 기반으로 strict OOF에서 더 높은 예측 성능을 내는 ML 후보를 탐색했다.",
        "- `M1/M2/M3`는 확인용으로 다시 비교했고, 주 실험은 Stage7/토지피복/Stage9 규칙/Stage16 축소 피처 기반으로 구성했다.",
        "- Optuna는 사용하지 않았다. 1차는 넓은 screening이며, 2차에서 상위 후보를 튜닝한다.",
        "- lockbox test는 사용하지 않았다.",
        "",
        "## 2. 기준 로지스틱",
        "",
        f"- `PLUS_LANDCOVER_RULES_ANOVA`: AUPRC {float(baseline['auprc']):.4f}, ROC AUC {float(baseline['auroc']):.4f}, Brier {float(baseline['brier']):.5f}, log loss {float(baseline['log_loss']):.5f}, best-F1 {float(baseline['best_f1_f1']):.4f}.",
        "",
        "## 3. 전체 최고 후보",
        "",
        f"- `{best['feature_set']} / {best['model']} / raw`: AUPRC {float(best['auprc']):.4f}, ROC AUC {float(best['auroc']):.4f}, Brier {float(best['brier']):.5f}, ΔAUPRC {float(best['delta_auprc_vs_logistic']):+.4f}.",
        "",
        "## 4. 그룹별 최고 후보",
        "",
        best_by_group[["result_group", *display_cols]].round(5).to_markdown(index=False),
        "",
        "## 5. 상위 20개 전체 조합",
        "",
        comparison.loc[comparison["run_status"].eq("OK"), display_cols].head(20).round(5).to_markdown(index=False),
        "",
        "## 6. 피처셋별 모델 순위",
        "",
        rank[rank_cols].round(5).to_markdown(index=False),
        "",
        "## 7. 최고 후보 운영점",
        "",
        best_thresholds.round(5).to_markdown(index=False),
        "",
        "## 8. 최고 후보 top-risk capture",
        "",
        best_top.round(5).to_markdown(index=False),
        "",
        "## 9. 최고 후보 hard-negative 성능",
        "",
        best_subgroup[["feature_set", "model", "score_type", "subgroup", "auprc", "auroc", "brier", "log_loss"]]
        .round(5)
        .to_markdown(index=False),
        "",
        "## 10. 산출물",
        "",
        "- `outputs/metrics/ml_stage1_v2_all_model_comparison.csv`",
        "- `outputs/metrics/ml_stage1_v2_model_rank_by_feature_set.csv`",
        "- `outputs/metrics/ml_stage1_v2_best_by_feature_group.csv`",
        "- `outputs/metrics/ml_stage1_v2_thresholds.csv`",
        "- `outputs/metrics/ml_stage1_v2_top_risk_capture.csv`",
        "- `outputs/metrics/ml_stage1_v2_subgroup_metrics.csv`",
        "- `outputs/predictions/ml_stage1_v2_oof_predictions.csv`",
    ]
    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "ml_stage1_v2_summary.md").write_text(summary, encoding="utf-8")

    log_lines = [
        "# 머신러닝 모델링 진행 로그",
        "",
        "## 2026-06-20",
        "",
        "### 1차 v2 실행",
        "",
        "- 로지스틱 Step17 최종 결과 기반으로 1차 ML screening을 재구성했다.",
        "- Optuna는 사용하지 않았다.",
        "- 모든 피처셋에 모든 사용 가능 모델을 적용했다.",
        "- 로지스틱 Stage17과 같은 metric/threshold/top-risk/subgroup 지표를 산출했다.",
        "",
        "### 최고 결과",
        "",
        f"- 전체 최고: `{best['feature_set']} / {best['model']} / raw`",
        f"- AUPRC {float(best['auprc']):.4f}, ROC AUC {float(best['auroc']):.4f}, Brier {float(best['brier']):.5f}, ΔAUPRC {float(best['delta_auprc_vs_logistic']):+.4f}",
        "",
        "### 그룹별 최고",
        "",
        best_by_group[["result_group", "feature_set", "model", "auprc", "auroc", "brier", "log_loss", "delta_auprc_vs_logistic"]]
        .round(5)
        .to_markdown(index=False),
        "",
        "### 산출물",
        "",
        "- `outputs/ml_stage1_v2_summary.md`",
        "- `outputs/metrics/ml_stage1_v2_all_model_comparison.csv`",
        "- `outputs/metrics/ml_stage1_v2_model_rank_by_feature_set.csv`",
        "- `outputs/predictions/ml_stage1_v2_oof_predictions.csv`",
    ]
    log_text = "\n".join(log_lines) + "\n"
    (ML_DIR / "LOG.md").write_text(log_text, encoding="utf-8")
    (ML_DIR / "머신러닝_전체_진행_결과.md").write_text(summary, encoding="utf-8")


def main() -> None:
    print("ML Stage1 v2: dependency audit")
    dependency_audit()
    print("ML Stage1 v2: data prepare")
    data, outer, inner, feature_sets, categorical = prepare_data()
    print("ML Stage1 v2: screening")
    predictions, fold_metrics, tuning = train_and_tune(data, outer, inner, feature_sets, categorical)
    raw_comparison = model_comparison(predictions)
    candidates = select_calibration_candidates(raw_comparison)
    print("ML Stage1 v2: calibration")
    predictions = apply_calibration(data, outer, inner, feature_sets, categorical, predictions, candidates)
    predictions.to_csv(PREDICTION_DIR / "ml_stage1_v2_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(METRIC_DIR / "ml_stage1_v2_fold_metrics.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(METRIC_DIR / "ml_stage1_v2_inner_tuning.csv", index=False, encoding="utf-8-sig")

    comparison = model_comparison(predictions)
    thresholds = make_threshold_table(predictions)
    top_risk = make_top_risk_table(predictions)
    subgroup = make_subgroup_table(predictions)
    rank, best_by_group = make_rank_tables(comparison)
    comparison.to_csv(METRIC_DIR / "ml_stage1_v2_all_model_comparison.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(METRIC_DIR / "ml_stage1_v2_thresholds.csv", index=False, encoding="utf-8-sig")
    top_risk.to_csv(METRIC_DIR / "ml_stage1_v2_top_risk_capture.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(METRIC_DIR / "ml_stage1_v2_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(METRIC_DIR / "ml_stage1_v2_model_rank_by_feature_set.csv", index=False, encoding="utf-8-sig")
    best_by_group.to_csv(METRIC_DIR / "ml_stage1_v2_best_by_feature_group.csv", index=False, encoding="utf-8-sig")
    write_logistic_combined_table(comparison)
    write_summary(comparison, rank, best_by_group, thresholds, top_risk, subgroup)
    best = comparison.loc[comparison["score_type"].eq("raw") & comparison["run_status"].eq("OK")].iloc[0]
    print(
        "ML Stage1 v2 완료: "
        f"{best['feature_set']} / {best['model']} AUPRC={best['auprc']:.4f}, "
        f"Δlogistic={best['delta_auprc_vs_logistic']:+.4f}"
    )


if __name__ == "__main__":
    main()
