"""
Script 1: Data Loading and Exploratory Plots

Loads the raw GLCC/CFSR data and generates exploratory figures used in the paper:
  - Raw RNBS time series (images/rnbs_raw.pdf)
  - Covariate distributions for Lake Superior (images/covars_raw.pdf)

Run from the project root:
    python scripts/01_preprocess_data.py
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import xarray as xr

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

LAKE_ORDER = ["sup", "mic_hur", "eri", "ont"]
LAKE_LABELS = {
    "sup": "Superior",
    "mic_hur": "Michigan/Huron",
    "eri": "Erie",
    "ont": "Ontario",
}


lake_data = xr.open_dataarray("data/lake_data.nc")

# --- Figure: Raw RNBS time series ---
rnbs_data = (
    lake_data.sel(variable="rnbs")
    .drop_vars(["variable", "type"])
    .to_dataframe(name="rnbs")
    .reset_index()
)

fig, axs = plt.subplots(2, 2, figsize=(8, 4), sharex=True, sharey=False)
axs = axs.flatten()

for i, lake in enumerate(LAKE_ORDER):
    ax = axs[i]
    df = rnbs_data[rnbs_data["lake"] == lake].sort_values("Date")
    ax.plot(df["Date"], df["rnbs"], lw=0.8)
    ax.set_title(LAKE_LABELS[lake])
    ax.xaxis.set_major_locator(mdates.YearLocator(20))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

fig.text(0.04, 0.5, r"RNBS $(m^3/s)$", va="center", rotation="vertical")
fig.tight_layout(rect=[0.05, 0, 1, 1])
fig.savefig(IMAGES_DIR / "rnbs_raw.pdf")
plt.close(fig)
print(f"Saved {IMAGES_DIR / 'rnbs_raw.pdf'}")

# --- Figure: Covariate distributions (Lake Superior) ---
covar_data = xr.open_dataarray("data/covar_data.nc")

VARIABLE_ORDER = ["precip", "temp", "evap"]
VARIABLE_LABELS = {
    "precip": "Precipitation (mm)",
    "temp": "Temperature (K)",
    "evap": "Evaporation (mm)",
}
TYPE_ORDER = ["Basin", "Land", "Water"]

fig, axs = plt.subplots(
    len(VARIABLE_ORDER), len(TYPE_ORDER), figsize=(9, 5), sharex=False, sharey=False
)

for row, var in enumerate(VARIABLE_ORDER):
    for col, typ in enumerate(TYPE_ORDER):
        ax = axs[row, col]
        subset = covar_data.sel(variable=var, type=typ).sortby("Date")
        ax.plot(subset["Date"], subset, lw=0.8)
        if row == 0:
            ax.set_title(typ)
        if col == 0:
            ax.set_ylabel(VARIABLE_LABELS[var], fontsize=8)

        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

fig.tight_layout()
fig.savefig(IMAGES_DIR / "covars_raw.pdf")
plt.close(fig)
print(f"Saved {IMAGES_DIR / 'covars_raw.pdf'}")
