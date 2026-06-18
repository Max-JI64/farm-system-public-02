from pathlib import Path

import nbformat
from nbclient import NotebookClient


HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "26.06.18_로지스틱_모델링.ipynb"
HEADING = "## 7단계. EDA 핵심 피처 확장"

nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
existing_text = "\n".join(
    "".join(cell.get("source", [])) for cell in nb.cells
)
heading_index = next(
    (
        index
        for index, cell in enumerate(nb.cells)
        if HEADING in "".join(cell.get("source", []))
    ),
    None,
)

new_cells = [
    nbformat.v4.new_markdown_cell(
        """## 7단계. EDA 핵심 피처 확장

6단계 최종 후보인 `L2 + class_weight 없음 + D-1 캐나다 전체 지수`에 국지 저습, 무강수 지속, 6시간 풍속과 권역 상호작용을 추가한다. FA 요인점수는 사용하지 않는다."""
    ),
    nbformat.v4.new_code_cell(
        """from pathlib import Path
import runpy

RUN_STAGE7 = False
stage7_script = Path.cwd() / "stage7_feature_extension.py"
if RUN_STAGE7:
    runpy.run_path(str(stage7_script), run_name="__main__")
else:
    print("기존 stage7 산출물을 사용합니다.")
    print("재계산하려면 RUN_STAGE7 = True로 변경하세요.")"""
    ),
    nbformat.v4.new_code_cell(
        """from pathlib import Path
import pandas as pd
from IPython.display import Markdown, display

analysis_dir = Path.cwd()
output_dir = analysis_dir / "outputs"
metric_dir = output_dir / "metrics"

display(Markdown((output_dir / "stage7_result_summary.md").read_text(encoding="utf-8")))
comparison = pd.read_csv(metric_dir / "stage7_feature_set_comparison.csv", encoding="utf-8-sig")
negative = pd.read_csv(metric_dir / "stage7_feature_set_negative_type_metrics.csv", encoding="utf-8-sig")
climate = pd.read_csv(metric_dir / "stage7_feature_set_climate_metrics.csv", encoding="utf-8-sig")

print("피처 세트 비교")
display(comparison.round(4))
print("대조군 유형별 성능")
display(negative[["feature_set", "negative_type", "auprc", "auroc", "brier"]].round(4))
print("기후지형유형별 성능")
display(climate[["feature_set", "기후지형유형", "auprc", "auroc", "brier"]].round(4))"""
    ),
]

temp = nbformat.v4.new_notebook(
    cells=[cell.copy() for cell in new_cells],
    metadata=nb.metadata,
)
client = NotebookClient(
    temp,
    timeout=120,
    kernel_name="python3",
    resources={"metadata": {"path": str(HERE)}},
)
client.execute()

if heading_index is None:
    nb.cells.extend(temp.cells)
else:
    nb.cells[heading_index : heading_index + 3] = temp.cells
nbformat.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
