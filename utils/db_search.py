from pathlib import Path
from typing import Any
import pandas as pd


def search_database(
    db_path: str | Path,
    filters: dict[str, Any],
    sort_by: str | None = None,
    ascending: bool = True,
    calculate_selectivity: bool = False,
) -> pd.DataFrame:
    """Returns: 검색 및 정렬된 DB DataFrame."""
    raise NotImplementedError
