from pathlib import Path
from collections.abc import Sequence
from typing import Any, TypedDict
import pandas as pd


class ReportFiles(TypedDict):
    pdf_path: str
    csv_path: str


def create_report(
    results: pd.DataFrame,
    images: Sequence[str | Path],
    graph_info: dict[str, Any],
    output_path: str | Path,
    db_results: pd.DataFrame | None = None,
) -> ReportFiles:
    """Returns: {"pdf_path": str, "csv_path": str}."""
    raise NotImplementedError

    ## 테스트
