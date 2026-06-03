"""
Script 0: One-Time Data Export

Reads raw GLCC/CFSR source files via load_data() and writes two NetCDF files
that the remaining scripts consume directly — no further knowledge of the
source CSVs or DATA_DIR is needed after this runs.

Outputs:
  - data/lake_data.nc   (Date x lake x variable, type=Basin, all four variables)
  - data/covar_data.nc  (Date x type x variable, lake=Superior, covariates only)

Run once from the project root (requires raw data files in DATA_DIR):
    python scripts/00_save_data.py
"""

from pathlib import Path
from src.data_loading.data_loading import load_data

(DATA_DIR := Path("data")).mkdir(exist_ok=True, parents=True)

# --- lake_data: Basin-averaged, all four variables, NaN rows dropped ---
lake_data = (
    load_data(["rnbs", "precip", "temp", "evap"])
    .sel(type="Basin")
    .dropna("Date")
    .to_array()
    .transpose("Date", "lake", ...)
)

lake_data.name = "lake_data"
lake_data.to_netcdf(DATA_DIR / "lake_data.nc")
print(f"Saved {DATA_DIR / 'lake_data.nc'}  shape={dict(lake_data.sizes)}")

# -- covar_data: Lake Superior only, covariates only, NaN rows dropped ---
# This data is strictly for plotting purposes - the analysis data is stored in lake_data.nc

covar_data = (
    load_data(["precip", "temp", "evap"])
    .sel(lake="sup")
    .drop_vars(["lake"])
    .to_array()
    .transpose("Date", "type", "variable")
)

covar_data.name = "covar_data"
covar_data.to_netcdf(DATA_DIR / "covar_data.nc")
print(f"Saved {DATA_DIR / 'covar_data.nc'}  shape={dict(covar_data.sizes)}")
