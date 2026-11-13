from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any
import pandas as pd


_ADSORPTION_NAMES = ("adsorption", "uptake", "loading", "capacity")


def _read_database(db_path: str | Path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            if _looks_like_git_lfs_pointer(path):
                raise ValueError(
                    f"{path} is a Git LFS pointer, not the real Parquet file. "
                    "Download the LFS data before searching the database."
                ) from exc
            raise
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=suffix in {".jsonl", ".ndjson"})
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(
        f"Unsupported database format '{suffix}'. "
        "Use parquet, csv, tsv, json, xlsx, or pickle."
    )


def _looks_like_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(64).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def _resolve_column(df: pd.DataFrame, column: str) -> str:
    if column in df.columns:
        return column

    normalized = column.casefold()
    matches = [existing for existing in df.columns if str(existing).casefold() == normalized]
    if matches:
        return matches[0]

    raise KeyError(f"Column '{column}' not found. Available columns: {list(df.columns)}")


def _as_iterable(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _is_empty_filter(condition: Any) -> bool:
    return condition is None or (isinstance(condition, str) and condition == "")


def _apply_single_filter(df: pd.DataFrame, column: str, condition: Any) -> pd.DataFrame:
    resolved_column = _resolve_column(df, column)
    series = df[resolved_column]

    if callable(condition):
        mask = condition(series)
        return df[mask]

    if isinstance(condition, dict):
        mask = pd.Series(True, index=df.index)
        for operator, value in condition.items():
            op = str(operator).casefold()
            if op in {"min", "gte", "ge", ">="}:
                mask &= pd.to_numeric(series, errors="coerce") >= value
            elif op in {"max", "lte", "le", "<="}:
                mask &= pd.to_numeric(series, errors="coerce") <= value
            elif op in {"gt", ">"}:
                mask &= pd.to_numeric(series, errors="coerce") > value
            elif op in {"lt", "<"}:
                mask &= pd.to_numeric(series, errors="coerce") < value
            elif op == "between":
                low, high = value
                mask &= pd.to_numeric(series, errors="coerce").between(low, high)
            elif op in {"eq", "equals", "=="}:
                mask &= series == value
            elif op in {"ne", "not", "!="}:
                mask &= series != value
            elif op in {"in", "isin"}:
                mask &= series.isin(_as_iterable(value))
            elif op in {"not_in", "notin"}:
                mask &= ~series.isin(_as_iterable(value))
            elif op == "contains":
                mask &= series.astype(str).str.contains(str(value), case=False, na=False)
            elif op == "startswith":
                mask &= series.astype(str).str.startswith(str(value), na=False)
            elif op == "endswith":
                mask &= series.astype(str).str.endswith(str(value), na=False)
            elif op == "regex":
                mask &= series.astype(str).str.contains(str(value), regex=True, na=False)
            elif op in {"isnull", "is_na"}:
                mask &= series.isna() if bool(value) else series.notna()
            elif op in {"notnull", "not_na"}:
                mask &= series.notna() if bool(value) else series.isna()
            else:
                raise ValueError(f"Unsupported filter operator '{operator}' for '{column}'")
        return df[mask]

    if isinstance(condition, (list, tuple, set, frozenset)):
        return df[series.isin(condition)]

    return df[series == condition]


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    if not filters:
        return df

    filtered = df
    for column, condition in filters.items():
        if _is_empty_filter(condition):
            continue
        filtered = _apply_single_filter(filtered, column, condition)
    return filtered


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized_columns = {str(column).casefold(): column for column in df.columns}
    for candidate in candidates:
        match = normalized_columns.get(candidate.casefold())
        if match is not None:
            return match
    return None


def _find_gas_value_column(df: pd.DataFrame, gas: str) -> str | None:
    gas_token = gas.casefold()
    exact_candidates = [
        gas,
        f"{gas}_adsorption",
        f"{gas}_uptake",
        f"{gas}_loading",
        f"adsorption_{gas}",
        f"uptake_{gas}",
        f"loading_{gas}",
    ]
    exact_match = _find_column(df, exact_candidates)
    if exact_match is not None:
        return exact_match

    for column in df.columns:
        name = str(column).casefold()
        if gas_token in name and any(keyword in name for keyword in _ADSORPTION_NAMES):
            return column
    return None


def _calculate_wide_selectivity(df: pd.DataFrame) -> pd.DataFrame | None:
    co2_column = _find_gas_value_column(df, "CO2")
    n2_column = _find_gas_value_column(df, "N2")
    if co2_column is None or n2_column is None:
        return None

    result = df.copy()
    co2 = pd.to_numeric(result[co2_column], errors="coerce")
    n2 = pd.to_numeric(result[n2_column], errors="coerce")
    result["selectivity_CO2_N2"] = co2 / n2.mask(n2 == 0)
    return result


def _calculate_long_selectivity(df: pd.DataFrame) -> pd.DataFrame | None:
    gas_column = _find_column(df, ["gas", "Gas", "component"])
    value_column = _find_column(df, ["adsorption", "uptake", "loading", "capacity"])
    if gas_column is None or value_column is None:
        return None

    gas_values = df[gas_column].astype(str).str.upper()
    gas_subset = df[gas_values.isin({"CO2", "N2"})].copy()
    if gas_subset.empty:
        return None

    key_columns = [
        column
        for column in df.columns
        if column not in {gas_column, value_column, "selectivity_CO2_N2"}
    ]
    if not key_columns:
        return None

    gas_subset[gas_column] = gas_subset[gas_column].astype(str).str.upper()
    gas_subset[value_column] = pd.to_numeric(gas_subset[value_column], errors="coerce")

    pivot = gas_subset.pivot_table(
        index=key_columns,
        columns=gas_column,
        values=value_column,
        aggfunc="first",
    )
    if "CO2" not in pivot.columns or "N2" not in pivot.columns:
        return None

    selectivity = (pivot["CO2"] / pivot["N2"].mask(pivot["N2"] == 0)).rename(
        "selectivity_CO2_N2"
    )
    return df.merge(selectivity.reset_index(), on=key_columns, how="left")


def _add_selectivity(df: pd.DataFrame) -> pd.DataFrame:
    wide_result = _calculate_wide_selectivity(df)
    if wide_result is not None:
        return wide_result

    long_result = _calculate_long_selectivity(df)
    if long_result is not None:
        return long_result

    raise KeyError(
        "Cannot calculate CO2/N2 selectivity. Expected either wide columns like "
        "'CO2_adsorption' and 'N2_adsorption', or long columns like 'gas' and "
        "'adsorption'."
    )


def _sort_results(
    df: pd.DataFrame,
    sort_by: str | Sequence[str] | None,
    ascending: bool | Sequence[bool],
) -> pd.DataFrame:
    if sort_by is None:
        return df

    sort_columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)
    resolved_columns: list[str] = []
    resolved_ascending: list[bool] = []

    if isinstance(ascending, bool):
        ascending_values = [ascending] * len(sort_columns)
    else:
        ascending_values = list(ascending)
        if len(ascending_values) != len(sort_columns):
            raise ValueError("Length of 'ascending' must match length of 'sort_by'.")

    for column, direction in zip(sort_columns, ascending_values, strict=True):
        if isinstance(column, str) and column.startswith("-"):
            resolved_columns.append(_resolve_column(df, column[1:]))
            resolved_ascending.append(False)
        else:
            resolved_columns.append(_resolve_column(df, str(column)))
            resolved_ascending.append(bool(direction))

    return df.sort_values(by=resolved_columns, ascending=resolved_ascending).reset_index(
        drop=True
    )


def search_database(
    db_path: str | Path,
    filters: dict[str, Any],
    sort_by: str | None = None,
    ascending: bool = True,
    calculate_selectivity: bool = False,
) -> pd.DataFrame:
    """Search, sort, and optionally add CO2/N2 selectivity to a MOF database."""
    df = _read_database(db_path)
    result = _apply_filters(df, filters).copy()

    if calculate_selectivity:
        result = _add_selectivity(result)

    result = _sort_results(result, sort_by, ascending)
    return result.reset_index(drop=True)
