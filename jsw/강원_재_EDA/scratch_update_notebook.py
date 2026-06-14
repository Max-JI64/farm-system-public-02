import json
from pathlib import Path

notebook_path = Path("Step2_강원도_산불발생_날씨지수_대조군재분석.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Clean previous S2-05 cells if any
ids_to_remove = {"s2_05_header", "s2_05_stats_code", "s2_05_plots_header", "s2_05_plots_code", "s2_05_audit_header", "s2_05_audit_code"}
nb["cells"] = [c for c in nb["cells"] if c.get("id") not in ids_to_remove]

# 2. Read codes from files
with open("s2_05_stats_code.py", "r", encoding="utf-8") as f:
    stats_code_lines = f.readlines()

with open("s2_05_plots_code.py", "r", encoding="utf-8") as f:
    plots_code_lines = f.readlines()

with open("s2_05_audit_code.py", "r", encoding="utf-8") as f:
    audit_code_lines = f.readlines()

# 3. Create cells
new_cells = [
    {
        "cell_type": "markdown",
        "id": "s2_05_header",
        "metadata": {},
        "source": [
            "## S2-05. 산불 대 전체 비발생 기상 탐색 비교\n",
            "\n",
            "### S2-05-1. 분석 범위 및 통계 비교"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "s2_05_stats_code",
        "metadata": {},
        "outputs": [],
        "source": stats_code_lines
    },
    {
        "cell_type": "markdown",
        "id": "s2_05_plots_header",
        "metadata": {},
        "source": [
            "### S2-05-2. 핵심 플롯 저장"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "s2_05_plots_code",
        "metadata": {},
        "outputs": [],
        "source": plots_code_lines
    },
    {
        "cell_type": "markdown",
        "id": "s2_05_audit_header",
        "metadata": {},
        "source": [
            "### S2-05-3. 산출물 목록과 최종 감사 통과 확인"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "s2_05_audit_code",
        "metadata": {},
        "outputs": [],
        "source": audit_code_lines
    }
]

# Append cells
nb["cells"].extend(new_cells)

# Save notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully with S2-05 cells from files!")
