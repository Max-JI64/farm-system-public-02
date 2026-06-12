from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = PROJECT_ROOT / "data" / "학습데이터" / "학습데이터_최종_캐나다지수.csv"
NOTEBOOK_PATH = ANALYSIS_DIR / "FWI지수학습.ipynb"
README_PATH = ANALYSIS_DIR / "README.md"

FWI_COLUMNS = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
REQUIRED_COLUMNS = ["Target", "샘플유형", "기후지형유형", "기준시각", *FWI_COLUMNS]
RANDOM_STATE = 42
TEST_SIZE = 0.3


def fmt(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def markdown_table(records: list[dict], columns: list[str] | None = None) -> str:
    if not records:
        return ""
    columns = columns or list(records[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(fmt(record.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def to_records(frame: pd.DataFrame) -> list[dict]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["기준시각"] = pd.to_datetime(df["기준시각"], errors="coerce")
    df["월"] = df["기준시각"].dt.month
    df["연도"] = df["기준시각"].dt.year
    for column in FWI_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Target"] = pd.to_numeric(df["Target"], errors="coerce").astype("int64")
    return df


def validation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in FWI_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        finite_values = values[np.isfinite(values)]
        rows.append(
            {
                "지수": column,
                "결측": int(values.isna().sum()),
                "비유한값": int((~np.isfinite(values.dropna())).sum()),
                "음수": int((finite_values < 0).sum()),
                "최솟값": float(finite_values.min()),
                "최댓값": float(finite_values.max()),
                "평균": float(finite_values.mean()),
                "중앙값": float(finite_values.median()),
            }
        )
    return pd.DataFrame(rows).round(6)


def target_compare_table(df: pd.DataFrame) -> pd.DataFrame:
    mean_by_target = df.groupby("Target")[FWI_COLUMNS].mean()
    median_by_target = df.groupby("Target")[FWI_COLUMNS].median()
    q75_by_target = df.groupby("Target")[FWI_COLUMNS].quantile(0.75)
    rows = []
    for column in FWI_COLUMNS:
        target0_mean = float(mean_by_target.loc[0, column])
        target1_mean = float(mean_by_target.loc[1, column])
        target0_median = float(median_by_target.loc[0, column])
        target1_median = float(median_by_target.loc[1, column])
        rows.append(
            {
                "지수": column,
                "Target0_평균": target0_mean,
                "Target1_평균": target1_mean,
                "평균차": target1_mean - target0_mean,
                "평균비": target1_mean / target0_mean if target0_mean else np.nan,
                "Target0_중앙값": target0_median,
                "Target1_중앙값": target1_median,
                "중앙값차": target1_median - target0_median,
                "Target0_Q75": float(q75_by_target.loc[0, column]),
                "Target1_Q75": float(q75_by_target.loc[1, column]),
            }
        )
    return pd.DataFrame(rows).round(6)


def score_decile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in FWI_COLUMNS:
        ranked = df[column].rank(method="first")
        decile = pd.qcut(ranked, 10, labels=range(1, 11)).astype(int)
        grouped = df.assign(점수분위=decile).groupby("점수분위", observed=True)
        for bucket, part in grouped:
            rows.append(
                {
                    "지수": column,
                    "점수분위": int(bucket),
                    "건수": int(part.shape[0]),
                    "Target1건수": int(part["Target"].sum()),
                    "Target1비율": float(part["Target"].mean()),
                    "점수최솟값": float(part[column].min()),
                    "점수최댓값": float(part[column].max()),
                }
            )
    return pd.DataFrame(rows).round(6)


def decile_summary_table(deciles: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    rows = []
    for column, part in deciles.groupby("지수"):
        ordered = part.sort_values("점수분위")
        bottom = float(ordered.iloc[0]["Target1비율"])
        top = float(ordered.iloc[-1]["Target1비율"])
        corr = float(ordered["점수분위"].corr(ordered["Target1비율"], method="spearman"))
        rows.append(
            {
                "지수": column,
                "하위10%_Target1비율": bottom,
                "상위10%_Target1비율": top,
                "상위10%_Lift": top / base_rate if base_rate else np.nan,
                "상하위_비율차": top - bottom,
                "분위-발생률_Spearman": corr,
            }
        )
    return pd.DataFrame(rows).round(6).sort_values("상위10%_Lift", ascending=False)


def top_quantile_table(df: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    rows = []
    total_positive = float(df["Target"].sum())
    for column in FWI_COLUMNS:
        for share in [0.10, 0.20, 0.30]:
            threshold = float(df[column].quantile(1 - share))
            selected = df[df[column] >= threshold]
            positive = float(selected["Target"].sum())
            positive_rate = float(selected["Target"].mean())
            rows.append(
                {
                    "지수": column,
                    "상위비율": share,
                    "임계값": threshold,
                    "선택건수": int(selected.shape[0]),
                    "Target1비율": positive_rate,
                    "Lift": positive_rate / base_rate if base_rate else np.nan,
                    "Target1포착률": positive / total_positive if total_positive else np.nan,
                }
            )
    return pd.DataFrame(rows).round(6)


def youden_threshold(y_true: pd.Series, score: pd.Series) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, score)
    finite_mask = np.isfinite(thresholds)
    finite_fpr = fpr[finite_mask]
    finite_tpr = tpr[finite_mask]
    finite_thresholds = thresholds[finite_mask]
    idx = int(np.argmax(finite_tpr - finite_fpr))
    return float(finite_thresholds[idx])


def evaluate_threshold(y_true: pd.Series, score: pd.Series, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "임계값": float(threshold),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "예측양성비율": float(pred.mean()),
    }


def prediction_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Target"],
    )
    y_train = train["Target"]
    y_test = test["Target"]
    base_rate = float(y_test.mean())

    score_rows = []
    threshold_rows = []
    for column in FWI_COLUMNS:
        threshold = youden_threshold(y_train, train[column])
        auroc = float(roc_auc_score(y_test, test[column]))
        auprc = float(average_precision_score(y_test, test[column]))
        score_rows.append(
            {
                "지수": column,
                "AUROC": auroc,
                "AUPRC": auprc,
                "Base_Rate": base_rate,
                "AUPRC_Base대비": auprc / base_rate if base_rate else np.nan,
            }
        )
        threshold_rows.append({"지수": column, **evaluate_threshold(y_test, test[column], threshold)})

    split = {
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": base_rate,
    }
    return pd.DataFrame(score_rows).round(6), pd.DataFrame(threshold_rows).round(6), split


def compute_results() -> dict:
    df = load_dataset()
    source_columns = len(pd.read_csv(DATA_PATH, encoding="utf-8-sig", nrows=0).columns)
    base_rate = float(df["Target"].mean())
    deciles = score_decile_table(df)
    score_metrics, threshold_metrics, split = prediction_tables(df)

    target_counts = (
        df["Target"].value_counts().sort_index().rename_axis("Target").reset_index(name="건수")
    )
    target_counts["비율"] = target_counts["건수"] / len(df)
    sample_counts = (
        df["샘플유형"].value_counts().rename_axis("샘플유형").reset_index(name="건수")
    )
    sample_counts["비율"] = sample_counts["건수"] / len(df)

    monthly = (
        df.groupby("월")
        .agg(건수=("Target", "size"), Target1건수=("Target", "sum"), Target1비율=("Target", "mean"), FWI평균=("FWI", "mean"), DC평균=("DC", "mean"), BUI평균=("BUI", "mean"))
        .reset_index()
        .round(6)
    )

    sample_means = df.groupby("샘플유형")[FWI_COLUMNS].mean().round(6).reset_index()
    climate_means = df.groupby("기후지형유형")[FWI_COLUMNS].mean().round(6).reset_index()
    climate_target = (
        df.groupby(["기후지형유형", "Target"])[["DC", "BUI", "FWI"]]
        .mean()
        .round(6)
        .reset_index()
    )
    correlation = df[FWI_COLUMNS].corr().round(6).reset_index().rename(columns={"index": "지수"})

    decile_summary = decile_summary_table(deciles, base_rate)
    top_quantiles = top_quantile_table(df, base_rate)
    top10 = top_quantiles[top_quantiles["상위비율"] == 0.10].sort_values("Lift", ascending=False)

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(source_columns)},
        "analysis_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "time_range": {"min": str(df["기준시각"].min()), "max": str(df["기준시각"].max())},
        "base_rate": base_rate,
        "target_counts": to_records(target_counts.round(6)),
        "sample_counts": to_records(sample_counts.round(6)),
        "validation": to_records(validation_table(df)),
        "target_compare": to_records(target_compare_table(df)),
        "sample_means": to_records(sample_means),
        "climate_means": to_records(climate_means),
        "climate_target": to_records(climate_target),
        "monthly": to_records(monthly),
        "correlation": to_records(correlation),
        "decile_summary": to_records(decile_summary),
        "top10": to_records(top10),
        "score_metrics": to_records(score_metrics),
        "threshold_metrics": to_records(threshold_metrics),
        "split": split,
    }


def build_readme(results: dict) -> None:
    best_auroc = max(results["score_metrics"], key=lambda row: row["AUROC"])
    best_auprc = max(results["score_metrics"], key=lambda row: row["AUPRC"])
    best_balanced = max(results["threshold_metrics"], key=lambda row: row["Balanced_Accuracy"])
    best_top10 = max(results["top10"], key=lambda row: row["Lift"])
    best_decile = max(results["decile_summary"], key=lambda row: row["상위10%_Lift"])

    lines = [
        "# 캐나다 FWI 지수 기반 산불위험 평가",
        "",
        "이 문서는 `FWI지수학습.ipynb`의 분석 결과를 요약한 것이다. FWI 트랙은 로지스틱 회귀나 머신러닝 모델을 새로 학습하지 않고, `FFMC`, `DMC`, `DC`, `ISI`, `BUI`, `FWI`를 산불위험 점수로 보고 평가한다.",
        "",
        "## 분석 원칙",
        "",
        "- 입력 데이터: `data/학습데이터/학습데이터_최종_캐나다지수.csv`",
        f"- 사용 지수: `{ '`, `'.join(FWI_COLUMNS) }`",
        "- `FWI_발생확률` 같은 자체 확률 컬럼은 만들지 않는다.",
        "- 결과표와 플롯은 `FWI지수학습.ipynb` 코드셀 출력으로만 생성한다.",
        "",
        "## 1단계 결과: 데이터 로드 및 검증",
        "",
        f"- 데이터 크기: {results['shape']['rows']:,}행 × {results['shape']['columns']:,}열",
        f"- 기준시각 범위: {results['time_range']['min']} ~ {results['time_range']['max']}",
        f"- Target 1 비율: {results['base_rate']:.6f}",
        "",
        "### Target 분포",
        "",
        markdown_table(results["target_counts"], ["Target", "건수", "비율"]),
        "",
        "### 샘플유형 분포",
        "",
        markdown_table(results["sample_counts"], ["샘플유형", "건수", "비율"]),
        "",
        "### FWI 지수 검증",
        "",
        markdown_table(results["validation"], ["지수", "결측", "비유한값", "음수", "최솟값", "최댓값", "평균", "중앙값"]),
        "",
        "해석: FWI 계열 6개 지수는 결측, 비유한값, 음수가 모두 없어 지수 기반 분석에 바로 사용할 수 있다. Target 1 비율은 약 9.11%로 낮기 때문에 Accuracy만으로 성능을 판단하면 안 된다.",
        "",
        "## 2단계 결과: 확장 EDA",
        "",
        "### Target별 지수 차이",
        "",
        markdown_table(results["target_compare"], ["지수", "Target0_평균", "Target1_평균", "평균차", "평균비", "Target0_중앙값", "Target1_중앙값", "중앙값차", "Target0_Q75", "Target1_Q75"]),
        "",
        "### 상위 10% 위험구간 집중도",
        "",
        markdown_table(results["top10"], ["지수", "상위비율", "임계값", "선택건수", "Target1비율", "Lift", "Target1포착률"]),
        "",
        "### 분위별 발생률 요약",
        "",
        markdown_table(results["decile_summary"], ["지수", "하위10%_Target1비율", "상위10%_Target1비율", "상위10%_Lift", "상하위_비율차", "분위-발생률_Spearman"]),
        "",
        "### 샘플유형별 평균",
        "",
        markdown_table(results["sample_means"], ["샘플유형", *FWI_COLUMNS]),
        "",
        "### 기후지형유형별 평균",
        "",
        markdown_table(results["climate_means"], ["기후지형유형", *FWI_COLUMNS]),
        "",
        "### 월별 FWI 핵심 지수",
        "",
        markdown_table(results["monthly"], ["월", "건수", "Target1건수", "Target1비율", "FWI평균", "DC평균", "BUI평균"]),
        "",
        "해석: 모든 지수에서 Target 1 평균이 Target 0보다 높다. 평균비 기준으로는 `DC`가 가장 크게 벌어지고, 상위 10% 집중도 기준으로는 `ISI`가 가장 강하게 Target 1 비율을 끌어올린다. 다만 상위 10%에서도 Target 1 비율이 절대적으로 매우 높지는 않으므로, 지수 단독으로 확정 예측을 하기보다는 위험구간을 우선순위화하는 용도가 더 적절하다.",
        "",
        f"- 상위 10% Lift 최고: `{best_top10['지수']}` = {best_top10['Lift']:.6f}",
        f"- 분위별 상위 10% Lift 최고: `{best_decile['지수']}` = {best_decile['상위10%_Lift']:.6f}",
        "",
        "## 3단계 결과: 지수 기반 예측 성능 평가",
        "",
        f"- Train/Test split: train {results['split']['train_rows']:,}행, test {results['split']['test_rows']:,}행",
        f"- Test Target 1 비율: {results['split']['test_positive_rate']:.6f}",
        "",
        "### 위험 점수 성능",
        "",
        markdown_table(results["score_metrics"], ["지수", "AUROC", "AUPRC", "Base_Rate", "AUPRC_Base대비"]),
        "",
        "### Youden J 임계값 기반 Test 성능",
        "",
        markdown_table(results["threshold_metrics"], ["지수", "임계값", "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP", "예측양성비율"]),
        "",
        "## 최종 분석",
        "",
        f"- AUROC 최고 지수: `{best_auroc['지수']}` = {best_auroc['AUROC']:.6f}",
        f"- AUPRC 최고 지수: `{best_auprc['지수']}` = {best_auprc['AUPRC']:.6f}",
        f"- 임계값 기반 Balanced Accuracy 최고 지수: `{best_balanced['지수']}` = {best_balanced['Balanced_Accuracy']:.6f}",
        "",
        "결론: FWI 계열 지수는 Target 1에서 일관되게 높은 값을 보이고, 분위수 분석에서도 고위험 구간의 Target 1 비율이 상승한다. 따라서 지수는 산불 위험을 정렬하는 신호로는 유용하다. 그러나 AUPRC와 Precision이 낮아 단독 산불 예측 모델로 사용하기에는 한계가 크다.",
        "",
        "따라서 FWI는 `산불 발생확률`을 직접 산출하는 모델이 아니라, 산불 위험 기상 조건을 선별하는 보조 지표로 쓰는 것이 타당하다. 실제 예측 성능을 높이려면 공간·지형·인위 요인과 결합한 별도 모델 비교가 필요하다.",
        "",
    ]
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> None:
    setup_code = r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

font_path = "C:/Windows/Fonts/malgun.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

sns.set_theme(
    style="whitegrid",
    font=font_name,
    rc={
        "font.family": font_name,
        "font.sans-serif": [font_name],
        "axes.unicode_minus": False
    }
)

plt.rcParams["font.family"] = font_name
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False

pd.set_option("display.max_columns", None)
print("현재 matplotlib font.family:", plt.rcParams["font.family"])
print("현재 matplotlib font.sans-serif:", plt.rcParams["font.sans-serif"][:3])

PROJECT_ROOT = Path.cwd()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "data" / "학습데이터" / "학습데이터_최종_캐나다지수.csv").exists():
        PROJECT_ROOT = candidate
        break

DATA_PATH = PROJECT_ROOT / "data" / "학습데이터" / "학습데이터_최종_캐나다지수.csv"
FWI_COLUMNS = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
RANDOM_STATE = 42
TEST_SIZE = 0.3
'''

    load_code = r'''
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
source_shape = df.shape
df["기준시각"] = pd.to_datetime(df["기준시각"], errors="coerce")
df["월"] = df["기준시각"].dt.month
df["연도"] = df["기준시각"].dt.year
for column in FWI_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")
df["Target"] = pd.to_numeric(df["Target"], errors="coerce").astype("int64")

required_columns = ["Target", "샘플유형", "기후지형유형", "기준시각", *FWI_COLUMNS]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"필수 컬럼 누락: {missing_columns}")

print("원본 데이터 크기:", source_shape)
print("분석용 데이터 크기(월, 연도 파생 컬럼 추가 후):", df.shape)
print("기준시각 범위:", df["기준시각"].min(), "~", df["기준시각"].max())
print("Target 1 비율:", round(df["Target"].mean(), 6))

target_distribution = df["Target"].value_counts().sort_index().rename("건수").to_frame()
target_distribution["비율"] = target_distribution["건수"] / len(df)
display(target_distribution)

sample_distribution = df["샘플유형"].value_counts().rename("건수").to_frame()
sample_distribution["비율"] = sample_distribution["건수"] / len(df)
display(sample_distribution)

validation_rows = []
for column in FWI_COLUMNS:
    values = df[column]
    finite_values = values[np.isfinite(values)]
    validation_rows.append({
        "지수": column,
        "결측": int(values.isna().sum()),
        "비유한값": int((~np.isfinite(values.dropna())).sum()),
        "음수": int((finite_values < 0).sum()),
        "최솟값": float(finite_values.min()),
        "최댓값": float(finite_values.max()),
        "평균": float(finite_values.mean()),
        "중앙값": float(finite_values.median()),
    })
validation = pd.DataFrame(validation_rows)
display(validation.round(6))
'''

    target_eda_code = r'''
target_summary = df.groupby("Target")[FWI_COLUMNS].agg(["mean", "median", "std", "min", "max"])
display(target_summary.round(6))

mean_by_target = df.groupby("Target")[FWI_COLUMNS].mean()
median_by_target = df.groupby("Target")[FWI_COLUMNS].median()
q75_by_target = df.groupby("Target")[FWI_COLUMNS].quantile(0.75)
target_compare = []
for column in FWI_COLUMNS:
    target0_mean = mean_by_target.loc[0, column]
    target1_mean = mean_by_target.loc[1, column]
    target_compare.append({
        "지수": column,
        "Target0_평균": target0_mean,
        "Target1_평균": target1_mean,
        "평균차": target1_mean - target0_mean,
        "평균비": target1_mean / target0_mean,
        "Target0_중앙값": median_by_target.loc[0, column],
        "Target1_중앙값": median_by_target.loc[1, column],
        "중앙값차": median_by_target.loc[1, column] - median_by_target.loc[0, column],
        "Target0_Q75": q75_by_target.loc[0, column],
        "Target1_Q75": q75_by_target.loc[1, column],
    })
target_compare = pd.DataFrame(target_compare).round(6)
display(target_compare)

plot_df = df[["Target", *FWI_COLUMNS]].melt(id_vars="Target", var_name="지수", value_name="지수값")
plot_df["Target"] = plot_df["Target"].map({0: "Target 0", 1: "Target 1"})
plot_df["log1p_지수값"] = np.log1p(plot_df["지수값"])

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=plot_df, x="지수", y="log1p_지수값", hue="Target", ax=ax)
ax.set_title("Target별 FWI 계열 지수 분포 (log1p)")
ax.set_xlabel("")
ax.set_ylabel("log1p(지수값)")
ax.legend(title="")
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=plot_df, x="지수", y="log1p_지수값", hue="Target", cut=0, inner="quartile", ax=ax)
ax.set_title("Target별 FWI 계열 지수 밀도 (log1p)")
ax.set_xlabel("")
ax.set_ylabel("log1p(지수값)")
ax.legend(title="")
plt.show()
'''

    expanded_eda_code = r'''
base_rate = df["Target"].mean()

def score_decile_table(data, score_columns):
    rows = []
    for column in score_columns:
        ranked = data[column].rank(method="first")
        decile = pd.qcut(ranked, 10, labels=range(1, 11)).astype(int)
        grouped = data.assign(점수분위=decile).groupby("점수분위", observed=True)
        for bucket, part in grouped:
            rows.append({
                "지수": column,
                "점수분위": int(bucket),
                "건수": int(part.shape[0]),
                "Target1건수": int(part["Target"].sum()),
                "Target1비율": part["Target"].mean(),
                "Lift": part["Target"].mean() / base_rate,
                "점수최솟값": part[column].min(),
                "점수최댓값": part[column].max(),
            })
    return pd.DataFrame(rows)

decile_rates = score_decile_table(df, FWI_COLUMNS)
display(decile_rates.round(6))

decile_summary = []
for column, part in decile_rates.groupby("지수"):
    ordered = part.sort_values("점수분위")
    decile_summary.append({
        "지수": column,
        "하위10%_Target1비율": ordered.iloc[0]["Target1비율"],
        "상위10%_Target1비율": ordered.iloc[-1]["Target1비율"],
        "상위10%_Lift": ordered.iloc[-1]["Lift"],
        "상하위_비율차": ordered.iloc[-1]["Target1비율"] - ordered.iloc[0]["Target1비율"],
        "분위-발생률_Spearman": ordered["점수분위"].corr(ordered["Target1비율"], method="spearman"),
    })
decile_summary = pd.DataFrame(decile_summary).sort_values("상위10%_Lift", ascending=False)
display(decile_summary.round(6))

top_quantile_rows = []
total_positive = df["Target"].sum()
for column in FWI_COLUMNS:
    for share in [0.10, 0.20, 0.30]:
        threshold = df[column].quantile(1 - share)
        selected = df[df[column] >= threshold]
        top_quantile_rows.append({
            "지수": column,
            "상위비율": share,
            "임계값": threshold,
            "선택건수": len(selected),
            "Target1비율": selected["Target"].mean(),
            "Lift": selected["Target"].mean() / base_rate,
            "Target1포착률": selected["Target"].sum() / total_positive,
        })
top_quantiles = pd.DataFrame(top_quantile_rows)
display(top_quantiles.round(6))

fig, ax = plt.subplots(figsize=(11, 6))
sns.lineplot(data=decile_rates, x="점수분위", y="Target1비율", hue="지수", marker="o", ax=ax)
ax.axhline(base_rate, color="gray", linestyle="--", linewidth=1, label=f"전체 Target1 비율={base_rate:.3f}")
ax.set_title("지수 점수 분위별 Target 1 비율")
ax.set_xlabel("점수 분위: 1=하위 10%, 10=상위 10%")
ax.set_ylabel("Target 1 비율")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=top_quantiles[top_quantiles["상위비율"] == 0.10], x="지수", y="Lift", ax=ax)
ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.set_title("상위 10% 위험구간 Target 1 Lift")
ax.set_xlabel("")
ax.set_ylabel("Lift")
plt.show()
'''

    stratified_eda_code = r'''
sample_means = df.groupby("샘플유형")[FWI_COLUMNS].mean().round(6)
climate_means = df.groupby("기후지형유형")[FWI_COLUMNS].mean().round(6)
climate_target_means = df.groupby(["기후지형유형", "Target"])[["DC", "BUI", "FWI"]].mean().round(6)
monthly_summary = (
    df.groupby("월")
    .agg(
        건수=("Target", "size"),
        Target1건수=("Target", "sum"),
        Target1비율=("Target", "mean"),
        FWI평균=("FWI", "mean"),
        DC평균=("DC", "mean"),
        BUI평균=("BUI", "mean"),
    )
    .round(6)
)

display(sample_means)
display(climate_means)
display(climate_target_means)
display(monthly_summary)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=df, x="샘플유형", y="FWI", hue="Target", errorbar=None, ax=axes[0])
axes[0].set_title("샘플유형별 FWI 평균")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=20)
sns.barplot(data=df, x="기후지형유형", y="FWI", hue="Target", errorbar=None, ax=axes[1])
axes[1].set_title("기후지형유형별 FWI 평균")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 5))
monthly_summary[["FWI평균", "DC평균", "BUI평균"]].plot(ax=ax1, marker="o")
ax1.set_title("월별 FWI 핵심 지수 평균")
ax1.set_xlabel("월")
ax1.set_ylabel("지수 평균")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=monthly_summary.reset_index(), x="월", y="Target1비율", ax=ax)
ax.axhline(df["Target"].mean(), color="gray", linestyle="--", linewidth=1)
ax.set_title("월별 Target 1 비율")
ax.set_ylabel("Target 1 비율")
plt.show()
'''

    correlation_code = r'''
pearson_corr = df[FWI_COLUMNS].corr(method="pearson").round(3)
spearman_corr = df[FWI_COLUMNS].corr(method="spearman").round(3)
display(pearson_corr)
display(spearman_corr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=axes[0])
axes[0].set_title("Pearson 상관")
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=axes[1])
axes[1].set_title("Spearman 상관")
plt.tight_layout()
plt.show()
'''

    prediction_code = r'''
train, test = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["Target"],
)
y_train = train["Target"]
y_test = test["Target"]
test_base_rate = y_test.mean()

def youden_threshold(y_true, score):
    fpr, tpr, thresholds = roc_curve(y_true, score)
    finite_mask = np.isfinite(thresholds)
    finite_fpr = fpr[finite_mask]
    finite_tpr = tpr[finite_mask]
    finite_thresholds = thresholds[finite_mask]
    idx = int(np.argmax(finite_tpr - finite_fpr))
    return float(finite_thresholds[idx])

def evaluate_threshold(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "임계값": threshold,
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "예측양성비율": pred.mean(),
    }

score_rows = []
threshold_rows = []
thresholds = {}
for column in FWI_COLUMNS:
    threshold = youden_threshold(y_train, train[column])
    thresholds[column] = threshold
    auroc = roc_auc_score(y_test, test[column])
    auprc = average_precision_score(y_test, test[column])
    score_rows.append({
        "지수": column,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Base_Rate": test_base_rate,
        "AUPRC_Base대비": auprc / test_base_rate,
    })
    threshold_rows.append({"지수": column, **evaluate_threshold(y_test, test[column], threshold)})

score_metrics = pd.DataFrame(score_rows).round(6)
threshold_metrics = pd.DataFrame(threshold_rows).round(6)
display(score_metrics)
display(threshold_metrics)

fig, ax = plt.subplots(figsize=(8, 6))
for column in FWI_COLUMNS:
    fpr, tpr, _ = roc_curve(y_test, test[column])
    auc = roc_auc_score(y_test, test[column])
    ax.plot(fpr, tpr, label=f"{column} (AUROC={auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
ax.set_title("FWI 계열 지수 ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", fontsize=9)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for column in FWI_COLUMNS:
    precision, recall, _ = precision_recall_curve(y_test, test[column])
    ap = average_precision_score(y_test, test[column])
    ax.plot(recall, precision, label=f"{column} (AUPRC={ap:.3f})")
ax.axhline(test_base_rate, color="gray", linestyle="--", linewidth=1, label=f"Base rate={test_base_rate:.3f}")
ax.set_title("FWI 계열 지수 Precision-Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(loc="upper right", fontsize=9)
plt.show()

metric_plot = threshold_metrics.melt(
    id_vars="지수",
    value_vars=["Balanced_Accuracy", "Precision", "Recall", "F1"],
    var_name="성능지표",
    value_name="값",
)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=metric_plot, x="지수", y="값", hue="성능지표", ax=ax)
ax.set_ylim(0, 1)
ax.set_title("Youden J 임계값 기반 Test 성능")
ax.set_xlabel("")
ax.set_ylabel("성능")
ax.legend(title="")
plt.show()

best_balanced_index = threshold_metrics.sort_values("Balanced_Accuracy", ascending=False).iloc[0]["지수"]
best_threshold = thresholds[best_balanced_index]
best_pred = (test[best_balanced_index] >= best_threshold).astype(int)
matrix = confusion_matrix(y_test, best_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["예측 0", "예측 1"],
    yticklabels=["실제 0", "실제 1"],
    ax=ax,
)
ax.set_title(f"최고 Balanced Accuracy 지수: {best_balanced_index}")
plt.show()
'''

    interpretation_code = r'''
best_auroc = score_metrics.sort_values("AUROC", ascending=False).iloc[0]
best_auprc = score_metrics.sort_values("AUPRC", ascending=False).iloc[0]
best_balanced = threshold_metrics.sort_values("Balanced_Accuracy", ascending=False).iloc[0]
best_top10 = top_quantiles[top_quantiles["상위비율"] == 0.10].sort_values("Lift", ascending=False).iloc[0]

print(f"AUROC 최고 지수: {best_auroc['지수']} = {best_auroc['AUROC']:.6f}")
print(f"AUPRC 최고 지수: {best_auprc['지수']} = {best_auprc['AUPRC']:.6f}")
print(f"Balanced Accuracy 최고 지수: {best_balanced['지수']} = {best_balanced['Balanced_Accuracy']:.6f}")
print(f"상위 10% Lift 최고 지수: {best_top10['지수']} = {best_top10['Lift']:.6f}")

if best_auroc["AUROC"] < 0.6:
    conclusion = "단독 예측 모델로 쓰기에는 성능이 약하다."
elif best_auroc["AUROC"] < 0.7:
    conclusion = "위험도 정렬 신호는 있으나 단독 예측 모델로는 제한적이다."
else:
    conclusion = "단독 위험도 점수로도 일정 수준의 구분력이 있다."

print("최종 판단:", conclusion)
print("해석: FWI 계열 지수는 확률이 아니라 위험 점수다. 고위험 구간 선별에는 유용하지만, 단독 산불 예측 모델로 쓰기에는 Precision과 AUPRC가 낮다.")
'''

    cells = [
        markdown_cell("# FWI 지수 기반 산불위험 평가\n\nFWI는 학습 모델이 아니라 기상 기반 위험 지수다. 이 노트북은 모든 결과표와 플롯을 노트북 내부 출력으로 생성한다."),
        markdown_cell("## 0. 환경 설정"),
        code_cell(setup_code.strip()),
        markdown_cell("## 1단계. 데이터 로드 및 검증"),
        code_cell(load_code.strip()),
        markdown_cell("## 2단계. Target별 지수 분포 EDA"),
        code_cell(target_eda_code.strip()),
        markdown_cell("## 2-1단계. 점수 분위와 상위 위험구간 EDA"),
        code_cell(expanded_eda_code.strip()),
        markdown_cell("## 2-2단계. 샘플유형, 기후지형유형, 월별 EDA"),
        code_cell(stratified_eda_code.strip()),
        markdown_cell("## 2-3단계. 지수 간 상관과 중복성 확인"),
        code_cell(correlation_code.strip()),
        markdown_cell("## 3단계. 지수 기반 예측 성능 평가"),
        code_cell(prediction_code.strip()),
        markdown_cell("## 최종 해석"),
        code_cell(interpretation_code.strip()),
        markdown_cell("`FWI`, `BUI`, `ISI` 등은 0~1 확률이 아니라 위험 점수다. 따라서 결과는 산불 발생확률 모델이 아니라, 위험 점수의 정렬력과 임계값 기반 분류 성능으로 해석한다."),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def execute_notebook() -> None:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)


def main() -> None:
    results = compute_results()
    build_readme(results)
    build_notebook()
    execute_notebook()
    print(f"README 작성 완료: {README_PATH}")
    print(f"노트북 작성 및 실행 완료: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
