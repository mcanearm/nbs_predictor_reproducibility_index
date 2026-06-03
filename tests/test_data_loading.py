import pytest
import xarray as xr

from src.data_loading.data_loading import load_data, _input_loaders, _forecast_loaders


@pytest.mark.parametrize("series", _input_loaders.keys())
def test_single_series(series):
    df = load_data(series)
    assert isinstance(df, xr.DataArray)


@pytest.mark.parametrize(
    "loaders, data_type", [(_input_loaders, "inputs"), (_forecast_loaders, "forecasts")]
)
def test_multi_series(loaders, data_type):
    series_list = list(loaders.keys())
    covars = load_data(series_list, data_type=data_type)

    assert list(covars.data_vars) == series_list
    assert isinstance(covars, xr.Dataset)
    assert list(covars["precip"].dims)[:2] == ["Date", "lake"]
