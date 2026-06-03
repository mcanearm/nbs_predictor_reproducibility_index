"""
Script 0: One-Time Data Export

Reads raw GLCC/CFSR source files via load_data() and writes two NetCDF files
that the remaining scripts consume directly — no further knowledge of the
source CSVs or DATA_DIR is needed after this runs.

Outputs:
  - data/lake_data.nc   (Date × lake × variable, type=Basin, all four variables)
  - data/covar_data.nc  (Date × type × variable, lake=Superior, covariates only)

Run once from the project root (requires raw data files in DATA_DIR):
    python scripts/00_save_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading.data_loading import load_data

DATA_DIR = Path("data")

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

# --- covar_data: all spatial types, Lake Superior only, covariates only ---
covar_data = (
    load_data(["precip", "temp", "evap"])
    .to_array()
    .sel(lake="sup")
    .dropna("Date")
    .transpose("Date", ...)
)
covar_data.name = "covar_data"
covar_data.to_netcdf(DATA_DIR / "covar_data.nc")
print(f"Saved {DATA_DIR / 'covar_data.nc'}  shape={dict(covar_data.sizes)}")
