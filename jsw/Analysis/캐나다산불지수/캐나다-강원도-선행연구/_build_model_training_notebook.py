from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_NAME = "모델학습.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """
# 캐나다 FFMC 선행연구 모델 학습

`학습데이터_최종_캐나다지수.csv`를 사용해 강원도 선행연구의 FFMC 기반 로지스틱 발생확률 모델을 현재 데이터에서 재현하고 평가한다.

- `논문식_FFMC`: `Indexed_FFMC`에 논문 고정 계수 `-0.529 + 0.422 x Indexed_FFMC` 적용
- `재학습_FFMC_Logistic`: 같은 `Indexed_FFMC` 1개 변수로 현재 데이터에서 로지스틱 회귀 재학습

모든 표와 그림은 이 노트북 출력 안에서 생성한다. 외부 CSV/PNG 산출물 디렉터리는 만들지 않는다.
"""
        ),
        markdown_cell("## 1. 환경 설정"),
        code_cell(
            """
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

NOTEBOOK_DIR = Path(r"D:/farm-system-public-02/jsw/Analysis/캐나다산불지수/캐나다-강원도-선행연구")
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

import model_training as mt

mt.setup_visual_style()
"""
        ),
        markdown_cell("## 2. 데이터 로드 및 기본 검증"),
        code_cell(
            """
df = mt.load_dataset()

print("데이터 크기:", df.shape)
print("Target 분포:")
display(df["Target"].value_counts().sort_index().rename("count").to_frame())
print("샘플유형 분포:")
display(df["샘플유형"].value_counts().rename("count").to_frame())

canadian_columns = [
    "FFMC", "FFMC_10일평균", "Indexed_FFMC", "FFMC_논문식_발생확률",
    "DMC", "DC", "ISI", "BUI", "FWI",
]
display(df[canadian_columns].describe().T)
"""
        ),
        markdown_cell("## 3. Train/Test 분할"),
        code_cell(
            """
train, test = mt.make_train_test_split(df)

split_summary = pd.DataFrame(
    {
        "split": ["train", "test"],
        "n": [len(train), len(test)],
        "target_1": [int(train["Target"].sum()), int(test["Target"].sum())],
        "target_rate": [train["Target"].mean(), test["Target"].mean()],
    }
)
display(split_summary)
"""
        ),
        markdown_cell("## 4. 논문식 FFMC 기준선 예측"),
        code_cell(
            """
paper_prob = mt.paper_ffmc_probability(test)
paper_check_max_abs_diff = (paper_prob - test["FFMC_논문식_발생확률"]).abs().max()

print("논문식 재계산값과 저장 컬럼의 최대 절대 차이:", paper_check_max_abs_diff)
display(
    test[["샘플ID", "Target", "Indexed_FFMC", "FFMC_논문식_발생확률"]]
    .assign(논문식_재계산확률=paper_prob.values)
    .head(10)
)
"""
        ),
        markdown_cell("## 5. 현재 데이터 재학습 로지스틱"),
        code_cell(
            """
model = mt.train_reestimated_logistic(train)
retrained_prob = mt.predict_reestimated_logistic(model, test)
coefficients = mt.build_coefficients_table(model)

display(coefficients)
"""
        ),
        markdown_cell("## 6. 예측 결과 및 성능 측정"),
        code_cell(
            """
metrics = pd.DataFrame(
    [
        mt.evaluate_predictions(mt.PAPER_MODEL_NAME, test[mt.TARGET_COLUMN], paper_prob),
        mt.evaluate_predictions(mt.RETRAINED_MODEL_NAME, test[mt.TARGET_COLUMN], retrained_prob),
    ]
)
predictions = mt.build_predictions_table(test, paper_prob, retrained_prob)
sample_type_summary = mt.summarize_by_sample_type(predictions)

display(metrics.round(6))
display(sample_type_summary.round(6))
display(predictions.head(10))
"""
        ),
        markdown_cell(
            """
## 7. 노트북 내부 시각화

아래 셀은 파일을 저장하지 않고 그래프를 노트북 출력으로만 생성한다.
"""
        ),
        code_cell(
            """
figures = mt.make_figures(predictions, metrics)

for name, fig in figures.items():
    print(name)
    display(fig)
"""
        ),
        markdown_cell(
            """
## 8. 해석 메모

- 두 모델 모두 입력 변수는 `Indexed_FFMC` 1개다.
- `논문식_FFMC`는 원 논문의 고정 계수이므로 현재 데이터로 학습하지 않는다.
- `재학습_FFMC_Logistic`은 같은 변수 구조를 유지하면서 현재 학습데이터의 70% train split으로 계수를 다시 추정한다.
- 성능 지표와 그림은 외부 파일로 저장하지 않고 이 노트북 출력에 남긴다.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.1",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook_path = Path(__file__).resolve().parent / NOTEBOOK_NAME
    notebook = build_notebook()
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {notebook_path}")


if __name__ == "__main__":
    main()
