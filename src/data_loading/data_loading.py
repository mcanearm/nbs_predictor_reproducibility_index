import datetime as dt
import re
from typing import List, Union

import pandas as pd
import xarray as xr

from src.constants import DATA_DIR

__all__ = ["load_data"]

lake_order = ["sup", "mic_hur", "eri", "ont"]
name_remap = {
    "Erie": "eri",
    "Ontario": "ont",
    "Superior": "sup",
    "Huron": "hur",
    "Michigan": "mic",
}  # mic_hur is combined inside read_cfsr_files / read_cfs_file


# ---------------------------------------------------------------------------
# Low-level reader functions
# ---------------------------------------------------------------------------


def read_historical_files(path, date_format="%Y%m%d") -> xr.DataArray:
    """
    Read a GLCC CSV file (Date index, one column per lake).
    Returns a DataArray with dims (Date, lake).
    """
    df = pd.read_csv(path, index_col="Date")[lake_order]
    df.index = pd.to_datetime(df.index.astype(str), format=date_format)
    df.index.name = "Date"
    return xr.DataArray(
        df,
        dims=["Date", "lake"],
        coords={"Date": df.index, "lake": df.columns},
    )


def read_cfsr_files(path) -> xr.DataArray:
    """
    Read a CFSR reanalysis CSV file (year/month columns, {Type}{Lake} columns).
    Michigan and Huron are averaged into mic_hur.
    Returns a DataArray with dims (Date, lake, type).
    """
    df = pd.read_csv(path)

    date_index = pd.Index(
        [dt.datetime(year, month, 1) for year, month in zip(df["year"], df["month"])],
        name="Date",
    )

    df = df.drop(["year", "month"], axis=1).set_index(date_index)
    new_columns = pd.MultiIndex.from_tuples(
        [tuple(re.findall("[A-Z][^A-Z]*", x)) for x in df.columns],
        names=["type", "lake"],
    )
    df.columns = new_columns
    df = df.rename(columns=name_remap, level=1)

    output_array = (
        df.melt(value_name="value", ignore_index=False)
        .set_index(["lake", "type"], append=True)
        .to_xarray()
        .to_array()
        .squeeze()
        .drop_vars("variable")
    )

    mic_hur = (
        output_array.sel(lake=["mic", "hur"])
        .mean(dim="lake")
        .expand_dims(dim={"lake": ["mic_hur"]})
    )
    cur_forecasts = output_array.sel(lake=["eri", "sup", "ont"])
    return (
        xr.concat([mic_hur, cur_forecasts], dim="lake")
        .sel(lake=lake_order)
        .transpose("Date", "lake", "type")
    )


def read_cfs_file(path) -> xr.DataArray:
    """
    Read a CFS forecast CSV file (year, month, cfsrun, {Type}{Lake} columns).
    Michigan and Huron are averaged into mic_hur.
    Returns a DataArray with dims (Date, months_ahead, lake, type).
    """
    input_csv = pd.read_csv(path)
    forecast_date = pd.to_datetime(
        input_csv.pop("year").astype(str) + input_csv.pop("month").astype(str),
        format="%Y%m",
    )

    input_csv = input_csv.assign(
        cfsrun=pd.to_datetime(input_csv["cfsrun"], format="%Y%m%d%H"),
        forecast_date=forecast_date,
    ).set_index(["cfsrun", "forecast_date"])

    new_columns = [tuple(re.findall("[A-Z][^A-Z]*", x)) for x in input_csv.columns]
    input_csv.columns = pd.MultiIndex.from_tuples(new_columns, names=["type", "lake"])
    input_csv = input_csv.rename(columns=name_remap)

    input_csv["months_ahead"] = (
        input_csv.sort_values(["cfsrun", "forecast_date"]).groupby("cfsrun").cumcount()
    )
    input_csv = input_csv.reset_index(level=1, drop=True).set_index(
        "months_ahead", append=True
    )
    input_csv.index.names = ["Date", "months_ahead"]

    output_array = (
        input_csv.melt(ignore_index=False)
        .set_index(["lake", "type"], append=True)
        .to_xarray()
        .to_array()
        .squeeze()
        .drop_vars("variable")
    )

    mic_hur = (
        output_array.sel(lake=["mic", "hur"])
        .mean(dim="lake")
        .expand_dims(dim={"lake": ["mic_hur"]})
    )
    cur_forecasts = output_array.sel(lake=["eri", "sup", "ont"])
    return (
        xr.concat([mic_hur, cur_forecasts], dim="lake")
        .sel(lake=lake_order)
        .transpose("Date", "months_ahead", "lake", "type")
    )


# ---------------------------------------------------------------------------
# Per-series loaders — each closes over its path and any format quirks
# ---------------------------------------------------------------------------


def _load_rnbs():
    return read_historical_files(DATA_DIR / "GLCC" / "rnbs_glcc.csv")


def _load_precip():
    glcc = read_historical_files(
        DATA_DIR / "GLCC" / "pcp_glerl_lakes_mic_hur_combined.csv"
    )
    glcc = glcc.expand_dims({"type": ["Thiessen"]})
    cfsr = read_cfsr_files(DATA_DIR / "CFSR" / "CFSR_APCP_Basin_Avgs.csv")
    return xr.concat([glcc, cfsr], dim="type", join="outer")


def _load_temp():
    return read_cfsr_files(DATA_DIR / "CFSR" / "CFSR_TMP_Basin_Avgs.csv")


def _load_evap():
    glcc = read_historical_files(
        DATA_DIR / "GLCC" / "evap_glerl_lakes_mic_hur_combined.csv"
    )
    glcc = glcc.expand_dims({"type": ["Thiessen"]})
    cfsr = read_cfsr_files(DATA_DIR / "CFSR" / "CFSR_EVAP_Basin_Avgs.csv")
    return xr.concat([glcc, cfsr], dim="type", join="outer")


def _load_runoff():
    # runoff uses YYYYMM date format, not YYYYMMDD
    return read_historical_files(
        DATA_DIR / "GLCC" / "runoff_glerl_mic_hur_combined.csv",
        date_format="%Y%m",
    )


def _load_water_level():
    return read_historical_files(DATA_DIR / "GLCC" / "wl_glcc.csv")


def _load_lhfx():
    return read_cfsr_files(DATA_DIR / "CFSR" / "CFSR_LHFX_Basin_Avgs.csv")


def _forecast_precip():
    return read_cfs_file(DATA_DIR / "CFS" / "CFS_APCP_Basin_Avgs.csv")


def _forecast_temp():
    return read_cfs_file(DATA_DIR / "CFS" / "CFS_TMP_Basin_Avgs.csv")


def _forecast_evap():
    return read_cfs_file(DATA_DIR / "CFS" / "CFS_EVAP_Basin_Avgs.csv")


_input_loaders = {
    "rnbs": _load_rnbs,
    "precip": _load_precip,
    "temp": _load_temp,
    "evap": _load_evap,
    "runoff": _load_runoff,
    "water_level": _load_water_level,
    "lhfx": _load_lhfx,
}

_forecast_loaders = {
    "precip": _forecast_precip,
    "temp": _forecast_temp,
    "evap": _forecast_evap,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(series: Union[str, List[str]], data_type: str = "inputs"):
    """
    Load one or more data series by name.

    Args:
        series: Series name or list of names.
                Inputs:    "rnbs", "precip", "temp", "evap", "runoff",
                           "water_level", "lhfx"
                Forecasts: "precip", "temp", "evap"
        data_type: "inputs" (default) or "forecasts"

    Returns:
        xr.DataArray for a single series, xr.Dataset for a list.
    """
    assert data_type in ("inputs", "forecasts"), (
        f"data_type must be 'inputs' or 'forecasts', got {data_type!r}"
    )
    loaders = _input_loaders if data_type == "inputs" else _forecast_loaders

    if isinstance(series, list):
        return xr.merge(
            [load_data(s, data_type=data_type).rename(s) for s in series],
            join="outer",
        ).transpose("Date", "lake", ...)

    if series not in loaders:
        raise ValueError(
            f"Unknown series {series!r} for data_type={data_type!r}. "
            f"Options: {list(loaders)}"
        )

    return loaders[series]()
