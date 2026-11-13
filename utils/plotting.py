from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


_COLUMNS = ["mof_id", "gas", "pressure", "adsorption"]
_GASES = ("CO2", "N2")


def _prepare_data(data: pd.DataFrame, name: str, *, database: bool = False) -> pd.DataFrame:
    """Copy and validate data without modifying the caller's DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not data.columns.is_unique:
        raise ValueError(f"{name} must have unique column names")
    if data.empty:
        raise ValueError(f"{name} must contain at least one adsorption value")

    frame = data.copy()
    temperature_columns = [
        column for column in frame.columns
        if str(column).casefold() in {"temperature", "temperature_k"}
    ]
    if database:
        # Match the long/wide adsorption layouts supported by db_search.py.
        aliases = {str(column).casefold(): column for column in frame.columns}
        rename = {}
        for target, candidates in {
            "mof_id": ("mof_id",),
            "pressure": ("pressure",),
            "gas": ("gas", "component"),
            "adsorption": ("adsorption", "uptake", "loading", "capacity"),
        }.items():
            for candidate in candidates:
                if candidate in aliases:
                    rename[aliases[candidate]] = target
                    break
        frame = frame.rename(columns=rename)
        if "gas" not in frame.columns:
            gas_columns = {}
            for gas in _GASES:
                candidates = [gas.casefold()]
                for value in ("adsorption", "uptake", "loading", "capacity"):
                    candidates.extend((f"{gas}_{value}".casefold(), f"{value}_{gas}".casefold()))
                for candidate in candidates:
                    if candidate in aliases:
                        gas_columns[aliases[candidate]] = gas
                        break
            if gas_columns and {"mof_id", "pressure"}.issubset(frame.columns):
                frame = frame.melt(
                    id_vars=["mof_id", "pressure"] + temperature_columns,
                    value_vars=list(gas_columns),
                    var_name="gas", value_name="adsorption",
                )
                frame["gas"] = frame["gas"].map(gas_columns)

    missing = set(_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(sorted(missing))}")
    # Even disjoint pressure grids must not connect different temperatures.
    frame = frame[_COLUMNS + temperature_columns].copy()
    for column in ("mof_id", "gas"):
        if frame[column].isna().any():
            raise ValueError(f"{name}.{column} must not contain missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{name}.{column} must not contain empty values")
    frame["gas"] = frame["gas"].str.upper().str.replace("₂", "2", regex=False)
    if not frame["gas"].isin(_GASES).all():
        raise ValueError(f"{name}.gas must contain only CO2 or N2")
    for column in temperature_columns:
        if (frame.groupby(["mof_id", "gas"])[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"{name} must be filtered to a single temperature per MOF and gas")
    for column in ("pressure", "adsorption"):
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}.{column} must contain finite numeric values") from error
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} must contain finite numeric values")
        frame[column] = values
    if (frame["pressure"] < 0).any():
        raise ValueError(f"{name}.pressure must be non-negative")
    if frame.duplicated(["mof_id", "gas", "pressure"]).any():
        raise ValueError(
            f"{name} has multiple values for the same MOF, gas and pressure. "
            "Filter to a single measurement condition (e.g. temperature) first."
        )
    return frame[_COLUMNS].sort_values(["mof_id", "gas", "pressure"], kind="stable")


def plot_adsorption(
    results: pd.DataFrame,
    output_path: str | Path,
    db_results: pd.DataFrame | None = None,
) -> str:
    """Save pressure/adsorption curves and return the image file path.

    ``results`` requires mof_id, gas (CO2/N2), pressure and adsorption columns.
    ``db_results`` accepts the same long layout (also component and uptake/
    loading/capacity aliases), or a wide layout with mof_id, pressure and gas
    columns such as CO2_adsorption/N2_adsorption or CO2/N2. None or an empty
    database skips comparison. Filter DB data to one measurement condition
    before calling; duplicate MOF/gas/pressure points are rejected.

    Each gas has its own panel. MOF colors are shared across panels; solid
    circles show predictions and dashed squares show DB values. Values are
    plotted as supplied, without normalization, interpolation or clipping.
    Both sources must use the same pressure and adsorption units. Units are
    intentionally unspecified until the project's data units are confirmed.

    output_path must include a PNG, JPG/JPEG, SVG or PDF extension. Parent
    directories are created as needed. Rendering uses a private Agg canvas
    suitable for Streamlit and environments without a display.

    Dependencies: numpy, pandas, matplotlib (already in requirements.txt).
    Raises TypeError/ValueError for invalid input; file errors propagate.
    """
    predictions = _prepare_data(results, "results")
    database = None
    if db_results is not None:
        if not isinstance(db_results, pd.DataFrame):
            raise TypeError("db_results must be a pandas DataFrame or None")
        if not db_results.empty:
            database = _prepare_data(db_results, "db_results", database=True)

    path = Path(output_path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        raise ValueError("output_path must end in .png, .jpg, .jpeg, .svg or .pdf")

    sources = [(predictions, "Prediction", "-", "o")]
    if database is not None:
        sources.append((database, "DB", "--", "s"))
    combined = pd.concat([source[0] for source in sources], ignore_index=True)
    gases = [gas for gas in _GASES if gas in set(combined["gas"])]
    mof_ids = sorted(combined["mof_id"].unique())
    # An explicit palette keeps output stable regardless of global plot styles.
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#666666")
    colors = {mof: palette[index % len(palette)] for index, mof in enumerate(mof_ids)}
    figure = Figure(figsize=(6.5 * len(gases), 4.8), facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    try:
        axes = figure.subplots(1, len(gases), squeeze=False)[0]
        for axis, gas in zip(axes, gases):
            for frame, label, linestyle, marker in sources:
                for mof_id, group in frame.loc[frame["gas"].eq(gas)].groupby("mof_id", sort=True):
                    axis.plot(
                        group["pressure"], group["adsorption"],
                        color=colors[mof_id], linestyle=linestyle, marker=marker,
                        linewidth=1.8, markersize=4,
                        label=f"{mof_id} ({label})",
                    )
            axis.set_title(gas.replace("2", "₂"), fontsize=13)
            axis.set_xlabel("Pressure")
            axis.set_ylabel("Adsorption")
            axis.set_facecolor("white")
            axis.grid(True, alpha=0.2)
            axis.set_axisbelow(True)
            axis.margins(x=0.05)
            axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        figure.suptitle("MOF adsorption" + (" — DB comparison" if database is not None else ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format=path.suffix[1:].lower(), dpi=180, bbox_inches="tight")
    finally:
        figure.clear()
    return str(path)
