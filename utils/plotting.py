from pathlib import Path
import pandas as pd


def plot_adsorption(
    results: pd.DataFrame,
    output_path: str | Path,
    db_results: pd.DataFrame | None = None,
) -> str:
    """Returns: 그래프 이미지 저장 경로 (str)."""
    raise NotImplementedError
