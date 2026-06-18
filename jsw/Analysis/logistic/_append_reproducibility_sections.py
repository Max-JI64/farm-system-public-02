from pathlib import Path

import nbformat


HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "26.06.18_로지스틱_모델링.ipynb"
START_HEADING = "## 7.5~10단계. 최종 로지스틱 재현 실행 블록"


def source_text(cell) -> str:
    return "".join(cell.get("source", []))


nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

start_index = next(
    (
        index
        for index, cell in enumerate(nb.cells)
        if START_HEADING in source_text(cell)
    ),
    None,
)
if start_index is not None:
    nb.cells = nb.cells[:start_index]

new_cells = [
    nbformat.v4.new_markdown_cell(
        """## 7.5~10단계. 최종 로지스틱 재현 실행 블록

이 블록은 현재 분석에서 최종적으로 남긴 **로지스틱 계열 통계모델** 결과를 재현한다.

실행 순서:

1. Stage 7 EDA 핵심 피처 확장
2. D2 토지피복/D3 임상도 보조 데이터 생성
3. Stage 8 D2/D3 로지스틱 비교
4. Stage 7.5 날짜·기상노출 strict CV 재검증
5. Stage 9 로지스틱 추가 개선 실험
6. Stage 10 경량 로지스틱 계열 확장
7. ML 비교용 로지스틱 통합 성능표 생성

주의:

- lockbox는 열지 않는다.
- 요인점수는 이후 주 모델에서 제외한다.
- 비통계적 이진분류 모델은 포함하지 않는다.
- 전체 재실행은 수 분 이상 걸릴 수 있다."""
    ),
    nbformat.v4.new_code_cell(
        """from pathlib import Path
import os
import runpy
import time

analysis_dir = Path.cwd()
if not (analysis_dir / "stage7_feature_extension.py").exists():
    analysis_dir = Path(r"D:\\farm-system-public-02\\jsw\\Analysis\\logistic")
os.chdir(analysis_dir)
print("작업 디렉터리:", Path.cwd())

# Run All로 완전 재현하려면 True를 유지한다.
# 이미 산출된 결과만 빠르게 확인하려면 False로 바꾼다.
RUN_REPRODUCE_STAGE7_TO_STAGE10 = True

reproduce_scripts = [
    ("Stage 7 EDA 핵심 피처 확장", "stage7_feature_extension.py"),
    ("D2/D3 보조 데이터 생성", "build_d2d3_dataset.py"),
    ("Stage 8 D2/D3 로지스틱 비교", "stage8_d2d3_logistic_analysis.py"),
    ("Stage 7.5 strict CV 재검증", "stage75_strict_validation.py"),
    ("Stage 9 로지스틱 추가 개선", "stage9_logistic_enhancement.py"),
    ("Stage 10 경량 로지스틱 계열 확장", "stage10_logistic_stat_extensions.py"),
    ("로지스틱 통합 성능표 생성", "make_logistic_benchmark_tables.py"),
]

if RUN_REPRODUCE_STAGE7_TO_STAGE10:
    for label, script_name in reproduce_scripts:
        script_path = Path.cwd() / script_name
        if not script_path.exists():
            raise FileNotFoundError(script_path)
        started = time.time()
        print(f"\\n▶ {label}: {script_name}")
        runpy.run_path(str(script_path), run_name="__main__")
        print(f"✓ 완료: {label} ({time.time() - started:.1f}초)")
else:
    print("재실행을 건너뜁니다. 기존 outputs를 사용합니다.")"""
    ),
    nbformat.v4.new_markdown_cell(
        """## 11단계. 최종 로지스틱 결과 확인

아래 셀은 재현 실행 후 생성된 핵심 요약문과 비교표를 표시한다."""
    ),
    nbformat.v4.new_code_cell(
        """from pathlib import Path
import pandas as pd
from IPython.display import Markdown, display

analysis_dir = Path.cwd()
output_dir = analysis_dir / "outputs"
metric_dir = output_dir / "metrics"

summary_files = [
    "stage75_strict_validation_summary.md",
    "stage8_d2d3_result_summary.md",
    "stage9_logistic_enhancement_summary.md",
    "stage10_logistic_stat_extensions_summary.md",
    "logistic_benchmark_for_model_comparison.md",
]

for filename in summary_files:
    path = output_dir / filename
    if path.exists():
        display(Markdown(f"---\\n\\n# {filename}\\n\\n"))
        display(Markdown(path.read_text(encoding="utf-8")))
    else:
        print("없음:", path)

benchmark_path = output_dir / "logistic_benchmark_for_model_comparison.csv"
benchmark = pd.read_csv(benchmark_path, encoding="utf-8-sig")
display(
    benchmark[
        [
            "feature_set",
            "auprc",
            "auroc",
            "brier",
            "log_loss",
            "best_f1_accuracy",
            "best_f1_precision",
            "best_f1_recall",
            "best_f1_f1",
            "Target_0A_auprc",
            "Target_0B1_auprc",
            "Target_0B2_auprc",
        ]
    ].round(5)
)"""
    ),
    nbformat.v4.new_markdown_cell(
        """## 12단계. 재현 산출물 체크리스트

Run All 이후 최소한 아래 파일이 존재해야 한다."""
    ),
    nbformat.v4.new_code_cell(
        """required_outputs = [
    output_dir / "stage75_strict_validation_summary.md",
    output_dir / "stage8_d2d3_result_summary.md",
    output_dir / "stage9_logistic_enhancement_summary.md",
    output_dir / "stage10_logistic_stat_extensions_summary.md",
    output_dir / "logistic_benchmark_for_model_comparison.csv",
    metric_dir / "stage10_logistic_stat_extensions_overall_metrics.csv",
    metric_dir / "stage10_logistic_stat_extensions_threshold_metrics.csv",
    metric_dir / "stage10_logistic_stat_extensions_sample_type_metrics.csv",
]

check = pd.DataFrame(
    [
        {
            "path": str(path.relative_to(analysis_dir)),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        }
        for path in required_outputs
    ]
)
display(check)
assert check["exists"].all(), "필수 산출물이 누락되었습니다."
print("재현 체크 완료")"""
    ),
]

nb.cells.extend(new_cells)
nbformat.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
