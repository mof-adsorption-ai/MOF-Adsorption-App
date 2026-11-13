from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
from itertools import product
from math import isfinite, log1p
from typing import TYPE_CHECKING

import pandas as pd
import torch

from models.cgcnn_scratch_model import load_cgcnn

if TYPE_CHECKING:
    from .preprocessing import GraphData


def predict_adsorption(
    graph_data: GraphData,
    mof_id: str,
    pressures: Sequence[float],
    gases: Sequence[str],
    checkpoint_path: str | Path,
) -> pd.DataFrame:
    """Predict adsorption for every gas/pressure pair.

    Returns a DataFrame with columns ``mof_id``, ``gas``, ``pressure``, and
    ``adsorption``. Checkpoint normalization statistics are stored in its
    ``preprocessor_state``.
    """
    pressure_values = list(pressures)
    gas_values = list(gases)
    supported_gases = {"CO2": 0, "N2": 1}
    unsupported = [gas for gas in gas_values if gas not in supported_gases]
    if unsupported:
        raise ValueError(f"Unsupported gas(s): {', '.join(map(str, unsupported))}")

    columns = ["mof_id", "gas", "pressure", "adsorption"]
    if not pressure_values or not gas_values:
        return pd.DataFrame(columns=columns)

    try:
        numeric_pressures = [float(pressure) for pressure in pressure_values]
    except (TypeError, ValueError) as exc:
        raise ValueError("Pressures must be numeric values") from exc
    if not all(isfinite(pressure) for pressure in numeric_pressures):
        raise ValueError("Pressures must be finite values")

    model, preprocessor_state, _ = load_cgcnn(checkpoint_path, device="cpu")
    pressure_mean = float(preprocessor_state["pressure_mean"])
    pressure_scale = float(preprocessor_state["pressure_scale"])
    if not isfinite(pressure_mean) or not isfinite(pressure_scale) or pressure_scale == 0:
        raise ValueError("Checkpoint pressure statistics must be finite with a non-zero scale")

    # The model pools atoms for one crystal, then evaluates each gas/pressure
    # pair against that shared crystal representation.
    pairs = list(product(gas_values, numeric_pressures))
    pressure_tensor = torch.tensor(
        [[(log1p(max(p, 0.0)) - pressure_mean) / pressure_scale]
         for _, p in pairs],
        dtype=torch.float32,
    )
    gas_index = torch.tensor([supported_gases[gas] for gas, _ in pairs], dtype=torch.long)
    crystal_idx = [torch.arange(graph_data["n_atoms"], dtype=torch.long)]
    sample_to_crystal = torch.zeros(len(pairs), dtype=torch.long)

    with torch.inference_mode():
        predictions = model(
            graph_data["atom"],
            graph_data["nbr"],
            graph_data["nbr_idx"],
            crystal_idx,
            pressure_tensor,
            gas_index,
            sample_to_crystal,
        ).cpu().tolist()

    y_stats = preprocessor_state["y_stats"]
    rows = []
    for (gas, pressure), scaled_prediction in zip(pairs, predictions):
        # The project checkpoint stores each gas statistic as (mean, scale).
        mean, scale = map(float, y_stats[gas])
        if not isfinite(mean) or not isfinite(scale):
            raise ValueError(f"Checkpoint output statistics for {gas} must be finite")
        rows.append({
            "mof_id": mof_id,
            "gas": gas,
            "pressure": pressure,
            "adsorption": float(scaled_prediction) * scale + mean,
        })
    return pd.DataFrame(rows, columns=columns)
