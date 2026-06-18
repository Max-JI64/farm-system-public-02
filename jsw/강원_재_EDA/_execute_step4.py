from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "Step4_산불발생_공간지형및대조군_분석.ipynb"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1_800,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
        allow_errors=False,
    )
    client.execute(cwd=str(HERE))
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"executed: {NOTEBOOK_PATH.name}")


if __name__ == "__main__":
    main()
