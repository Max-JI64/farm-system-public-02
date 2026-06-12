from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


DATA_FILE = "학습데이터_최종_캐나다지수.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.30
THRESHOLD = 0.50

FEATURE_COLUMNS = ["Indexed_FFMC"]
TARGET_COLUMN = "Target"
PAPER_MODEL_NAME = "논문식_FFMC"
RETRAINED_MODEL_NAME = "재학습_FFMC_Logistic"
L1_CV_MODEL_NAME = "규제튜닝_L1_LogisticCV"
L2_CV_MODEL_NAME = "규제튜닝_L2_LogisticCV"
TRAINED_MODEL_NAMES = [
    RETRAINED_MODEL_NAME,
    L1_CV_MODEL_NAME,
    L2_CV_MODEL_NAME,
]
MODEL_ORDER = [PAPER_MODEL_NAME, *TRAINED_MODEL_NAMES]
CV_CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

REQUIRED_COLUMNS = [
    "샘플ID",
    "기준시각",
    "기상셀ID",
    "샘플유형",
    TARGET_COLUMN,
    "FFMC",
    "FFMC_10일평균",
    "Indexed_FFMC",
    "FFMC_논문식_발생확률",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI",
]


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "data" / "학습데이터").exists():
            return candidate
    raise FileNotFoundError("Repository root with data/학습데이터 was not found.")


def setup_visual_style() -> None:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()

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
    print("현재 matplotlib font.family:", plt.rcParams["font.family"])
    print("현재 matplotlib font.sans-serif:", plt.rcParams["font.sans-serif"][:3])


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    repo_root = find_repo_root()
    path = data_path or repo_root / "data" / "학습데이터" / DATA_FILE
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["기준시각"] = pd.to_datetime(df["기준시각"], errors="coerce")
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["기준시각"].isna().any():
        raise ValueError("기준시각 contains invalid datetime values.")

    target_values = set(df[TARGET_COLUMN].dropna().unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(f"Target must contain only 0 and 1. Found: {sorted(target_values)}")

    canadian_columns = [
        "FFMC",
        "FFMC_10일평균",
        "Indexed_FFMC",
        "FFMC_논문식_발생확률",
        "DMC",
        "DC",
        "ISI",
        "BUI",
        "FWI",
    ]
    missing = df[canadian_columns].isna().sum()
    if int(missing.sum()) > 0:
        raise ValueError(f"Canadian index columns contain missing values: {missing.to_dict()}")

    if not df["Indexed_FFMC"].between(1, 4).all():
        raise ValueError("Indexed_FFMC must be in the 1..4 range.")

    if not df["FFMC_논문식_발생확률"].between(0, 1).all():
        raise ValueError("FFMC_논문식_발생확률 must be in the 0..1 range.")

    if not np.isfinite(df[canadian_columns].to_numpy(dtype=float)).all():
        raise ValueError("Canadian index columns contain non-finite values.")


def make_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    train = df.loc[train_idx].copy().reset_index(drop=True)
    test = df.loc[test_idx].copy().reset_index(drop=True)
    return train, test


def paper_ffmc_probability(df: pd.DataFrame) -> pd.Series:
    logit = -0.529 + 0.422 * df["Indexed_FFMC"].astype(float)
    return pd.Series(1.0 / (1.0 + np.exp(-logit)), index=df.index, name=PAPER_MODEL_NAME)


def train_reestimated_logistic(train: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(train[FEATURE_COLUMNS].astype(float), train[TARGET_COLUMN].astype(int))
    return model


def train_regularized_logistic_models(train: pd.DataFrame) -> dict[str, LogisticRegression | LogisticRegressionCV]:
    x = train[FEATURE_COLUMNS].astype(float)
    y = train[TARGET_COLUMN].astype(int)
    models: dict[str, LogisticRegression | LogisticRegressionCV] = {
        RETRAINED_MODEL_NAME: train_reestimated_logistic(train),
        L1_CV_MODEL_NAME: LogisticRegressionCV(
            Cs=CV_CS,
            cv=5,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            scoring="average_precision",
            max_iter=5000,
            random_state=RANDOM_STATE,
            refit=True,
        ),
        L2_CV_MODEL_NAME: LogisticRegressionCV(
            Cs=CV_CS,
            cv=5,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            scoring="average_precision",
            max_iter=5000,
            random_state=RANDOM_STATE,
            refit=True,
        ),
    }
    for model_name, model in models.items():
        if model_name == RETRAINED_MODEL_NAME:
            continue
        model.fit(x, y)
    return models


def predict_reestimated_logistic(model: LogisticRegression, df: pd.DataFrame) -> pd.Series:
    probability = model.predict_proba(df[FEATURE_COLUMNS].astype(float))[:, 1]
    return pd.Series(probability, index=df.index, name=RETRAINED_MODEL_NAME)


def predict_logistic_model(
    model_name: str,
    model: LogisticRegression | LogisticRegressionCV,
    df: pd.DataFrame,
) -> pd.Series:
    probability = model.predict_proba(df[FEATURE_COLUMNS].astype(float))[:, 1]
    return pd.Series(probability, index=df.index, name=model_name)


def evaluate_predictions(model_name: str, y_true: pd.Series, y_prob: pd.Series) -> dict[str, float | int | str]:
    y_label = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_label, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "mean_predicted_probability": float(y_prob.mean()),
        "threshold": THRESHOLD,
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_label)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_label)),
        "precision": float(precision_score(y_true, y_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_label, zero_division=0)),
        "F1": float(f1_score(y_true, y_label, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def build_predictions_table(test: pd.DataFrame, probability_by_model: dict[str, pd.Series]) -> pd.DataFrame:
    predictions = test[
        [
            "샘플ID",
            "기준시각",
            "기상셀ID",
            "샘플유형",
            TARGET_COLUMN,
            "FFMC",
            "FFMC_10일평균",
            "Indexed_FFMC",
        ]
    ].copy()
    for model_name in MODEL_ORDER:
        probability = probability_by_model[model_name].to_numpy()
        predictions[f"{model_name}_prob"] = probability
        predictions[f"{model_name}_pred"] = (probability >= THRESHOLD).astype(int)
    return predictions


def _model_penalty_info(model_name: str, model: LogisticRegression | LogisticRegressionCV) -> dict[str, float | str | None]:
    penalty = str(model.penalty)
    c_value = float(model.C_[0]) if isinstance(model, LogisticRegressionCV) else float(model.C)
    l1_ratio = None
    ratio_values = getattr(model, "l1_ratio_", None)
    if ratio_values is not None and ratio_values[0] is not None:
        l1_ratio = float(ratio_values[0])
    return {
        "penalty": penalty,
        "C": c_value,
        "l1_ratio": l1_ratio,
        "cv_scoring": "average_precision" if isinstance(model, LogisticRegressionCV) else None,
    }


def build_coefficients_table(
    trained_models: dict[str, LogisticRegression | LogisticRegressionCV],
) -> pd.DataFrame:
    rows = [
        {
            "model": PAPER_MODEL_NAME,
            "feature": "intercept",
            "coefficient": -0.529,
            "trained_on_current_data": False,
            "penalty": "논문 고정식",
            "C": None,
            "l1_ratio": None,
            "cv_scoring": None,
        },
        {
            "model": PAPER_MODEL_NAME,
            "feature": "Indexed_FFMC",
            "coefficient": 0.422,
            "trained_on_current_data": False,
            "penalty": "논문 고정식",
            "C": None,
            "l1_ratio": None,
            "cv_scoring": None,
        },
    ]
    for model_name in TRAINED_MODEL_NAMES:
        model = trained_models[model_name]
        info = _model_penalty_info(model_name, model)
        rows.extend(
            [
                {
                    "model": model_name,
                    "feature": "intercept",
                    "coefficient": float(model.intercept_[0]),
                    "trained_on_current_data": True,
                    **info,
                },
                {
                    "model": model_name,
                    "feature": "Indexed_FFMC",
                    "coefficient": float(model.coef_[0][0]),
                    "trained_on_current_data": True,
                    **info,
                },
            ]
        )
    return pd.DataFrame(rows)


def summarize_by_sample_type(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_type, group in predictions.groupby("샘플유형", sort=True):
        row = {
            "샘플유형": sample_type,
            "n": int(len(group)),
            "target_rate": float(group[TARGET_COLUMN].mean()),
        }
        for model_name in MODEL_ORDER:
            row[f"{model_name}_mean_prob"] = float(group[f"{model_name}_prob"].mean())
            row[f"{model_name}_pred_positive_rate"] = float(group[f"{model_name}_pred"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def make_roc_pr_plots(predictions: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Figure]:
    y_true = predictions[TARGET_COLUMN].astype(int)
    model_prob_cols = [(model_name, f"{model_name}_prob") for model_name in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for model_name, prob_col in model_prob_cols:
        fpr, tpr, _ = roc_curve(y_true, predictions[prob_col])
        auc = metrics.loc[metrics["model"] == model_name, "AUROC"].iloc[0]
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_fig = fig

    fig, ax = plt.subplots(figsize=(7.5, 6))
    base_rate = y_true.mean()
    ax.axhline(base_rate, linestyle="--", color="0.45", linewidth=1, label=f"Base rate={base_rate:.3f}")
    for model_name, prob_col in model_prob_cols:
        precision, recall, _ = precision_recall_curve(y_true, predictions[prob_col])
        auprc = metrics.loc[metrics["model"] == model_name, "AUPRC"].iloc[0]
        ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AUPRC={auprc:.3f})")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return {"roc_curve": roc_fig, "precision_recall_curve": fig}


def make_confusion_matrix_plot(predictions: pd.DataFrame) -> Figure:
    y_true = predictions[TARGET_COLUMN].astype(int)
    model_pred_cols = [(model_name, f"{model_name}_pred") for model_name in MODEL_ORDER]
    ncols = 2
    nrows = int(np.ceil(len(model_pred_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.2 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (model_name, pred_col) in zip(axes_flat, model_pred_cols):
        cm = confusion_matrix(y_true, predictions[pred_col], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["예측 0", "예측 1"],
            yticklabels=["실제 0", "실제 1"],
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes_flat[len(model_pred_cols) :]:
        ax.axis("off")
    return fig


def make_probability_distribution_plot(predictions: pd.DataFrame) -> Figure:
    long = predictions.melt(
        id_vars=[TARGET_COLUMN, "샘플유형"],
        value_vars=[f"{model_name}_prob" for model_name in MODEL_ORDER],
        var_name="model",
        value_name="예측확률",
    )
    long["model"] = long["model"].str.replace("_prob", "", regex=False)
    long["Target"] = long[TARGET_COLUMN].map({0: "Target 0", 1: "Target 1"})
    g = sns.displot(
        data=long,
        x="예측확률",
        hue="Target",
        col="model",
        col_wrap=2,
        kind="hist",
        bins=24,
        stat="density",
        common_norm=False,
        height=4.2,
        aspect=1.1,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("예측확률", "Density")
    g.figure.tight_layout()
    return g.figure


def make_sample_type_plot(predictions: pd.DataFrame) -> Figure:
    long = predictions.melt(
        id_vars=["샘플유형", TARGET_COLUMN],
        value_vars=[f"{model_name}_prob" for model_name in MODEL_ORDER],
        var_name="model",
        value_name="예측확률",
    )
    long["model"] = long["model"].str.replace("_prob", "", regex=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.boxplot(data=long, x="샘플유형", y="예측확률", hue="model", ax=ax)
    ax.set_title("샘플유형별 예측확률 분포")
    ax.set_xlabel("")
    ax.set_ylabel("예측확률")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="")
    fig.tight_layout()
    return fig


def make_metrics_plot(metrics: pd.DataFrame) -> Figure:
    long = metrics.melt(
        id_vars=["model"],
        value_vars=["AUROC", "AUPRC", "balanced_accuracy", "F1"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(data=long, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("모델 성능 비교")
    ax.set_xlabel("")
    ax.set_ylabel("score")
    ax.legend(title="")
    fig.tight_layout()
    return fig


def make_figures(predictions: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Figure]:
    figures = {}
    figures.update(make_roc_pr_plots(predictions, metrics))
    figures["confusion_matrix"] = make_confusion_matrix_plot(predictions)
    figures["probability_distribution"] = make_probability_distribution_plot(predictions)
    figures["probability_by_sample_type"] = make_sample_type_plot(predictions)
    figures["metrics_comparison"] = make_metrics_plot(metrics)
    return figures


def main() -> None:
    setup_visual_style()
    df = load_dataset()
    train, test = make_train_test_split(df)
    paper_prob = paper_ffmc_probability(test)
    trained_models = train_regularized_logistic_models(train)
    probability_by_model = {PAPER_MODEL_NAME: paper_prob}
    for model_name in TRAINED_MODEL_NAMES:
        probability_by_model[model_name] = predict_logistic_model(model_name, trained_models[model_name], test)
    metrics = pd.DataFrame(
        [evaluate_predictions(model_name, test[TARGET_COLUMN], probability_by_model[model_name]) for model_name in MODEL_ORDER]
    )
    coefficients = build_coefficients_table(trained_models)
    print("학습/평가 완료")
    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
    print(metrics.round(6).to_string(index=False))
    print(coefficients.round(6).to_string(index=False))
    print("외부 파일 저장 없음: 표와 그림은 모델학습.ipynb 출력에서 생성한다.")


if __name__ == "__main__":
    main()
