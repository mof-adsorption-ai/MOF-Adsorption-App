from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
import pandas as pd
from .preprocessing import GraphData


def predict_adsorption(
    graph_data: GraphData,
    mof_id: str,
    pressures: Sequence[float],
    gases: Sequence[str],
    checkpoint_path: str | Path,
) -> pd.DataFrame:
    """Returns: DataFrame[mof_id, gas, pressure, adsorption]."""
    raise NotImplementedError
