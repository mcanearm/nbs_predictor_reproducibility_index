"""
Script 2: Model Fitting (Cross-Validation)

Fits all candidate models via time-series cross-validation and saves per-model
per-split prediction CSVs and fitted model artifacts to disk.

Outputs:
  - data/model_results/{model}_{split}.csv   (one per model × CV split)
  - data/models/{model}_{split}.nc           (Bayesian models, ArviZ NetCDF)
  - data/models/{model}_{split}.pkl          (non-Bayesian models, pickle)
  - data/y_scaler.pkl                        (fitted XArrayStandardScaler for y)

Run from the project root:
    python scripts/02_fit_models.py

Set overwrite=True (below) to re-fit models even when saved results exist.
"""

import logging
import os
import pickle as pkl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpyro
import torch
import xarray as xr
from sklearn.gaussian_process import kernels
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.modeling.ensemble import BoostedRegressor, DefaultEnsemble, RandomForest
from src.modeling.gaussian_process import MultitaskGP, SklearnGPModel
from src.modeling.lm import LinearModel
from src.modeling.multivariate import LakeMVT
from src.modeling.var_models import VARX
from src.preprocessing.preprocessing import (
    SeasonalFeatures,
    XArrayFeatureUnion,
    XArrayStandardScaler,
)
from src.utils import flatten_array

logging.basicConfig(level=logging.INFO)

device = "cpu"
torch.set_default_device(device)
os.environ["JAX_PLATFORM_NAME"] = device
numpyro.set_platform(device)
numpyro.set_host_device_count(4)

RESULTS_DIR = Path("data/model_results")
MODEL_DIR = Path("data/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

overwrite = False

# --- Data loading ---
lake_data = xr.open_dataarray("data/lake_data.nc")

num_splits = 10
forecast_steps = 13

test_data = lake_data.sel(Date=slice("2006", "2010"))
train_data = lake_data.sel(Date=slice(None, "2005"))

splitter = TimeSeriesSplit(n_splits=num_splits, test_size=forecast_steps, gap=0)
splits = list(splitter.split(train_data))

X, y = (
    train_data.sel(variable=["precip", "temp", "evap"]).drop_vars("type"),
    train_data.sel(variable=["rnbs"]).squeeze().drop_vars(["type", "variable"]),
)
test_x, test_y = (
    test_data.sel(variable=["precip", "temp", "evap"]).drop_vars("type"),
    test_data.sel(variable=["rnbs"]).squeeze().drop_vars(["type", "variable"]),
)

y_scaler = XArrayStandardScaler()
y = y_scaler.fit_transform(y)
test_y = y_scaler.transform(test_y)

with open(Path("data") / "y_scaler.pkl", "wb") as f:
    pkl.dump(y_scaler, f)
logging.info("Saved y_scaler to data/y_scaler.pkl")

# --- Model definitions ---
preprocessor = XArrayFeatureUnion(
    [
        (
            "preprocess",
            Pipeline(
                steps=[
                    ("scale", XArrayStandardScaler()),
                    ("flatten", FunctionTransformer(flatten_array)),
                ]
            ),
        ),
        ("seasonal", SeasonalFeatures()),
    ]
)

gp_models = {
    "GP_Matern": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                SklearnGPModel(
                    1.0 * kernels.Matern(nu=1.5) * kernels.RationalQuadratic()
                ),
            ),
        ]
    ),
    "MultitaskGP": Pipeline(
        steps=[("preprocess", preprocessor), ("model", MultitaskGP(epochs=100, rank=1))]
    ),
}

simple_models = {
    "SimpleLM": Pipeline(
        steps=[("preprocess", preprocessor), ("model", LinearModel())]
    ),
    "RF": Pipeline(steps=[("preprocess", preprocessor), ("model", RandomForest())]),
    "BoostedTrees": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", BoostedRegressor(n_estimators=500, learning_rate=0.1)),
        ]
    ),
    "MVT": Pipeline(steps=[("preprocess", preprocessor), ("model", LakeMVT())]),
}

varx_models = {
    "VARX": Pipeline(
        steps=[
            ("preprocess", XArrayStandardScaler()),
            (
                "model",
                VARX(
                    lags={"y": 1, "precip": 0, "temp": 0, "evap": 0},
                    num_warmup=500,
                    num_chains=4,
                    num_samples=500,
                ),
            ),
        ]
    ),
    "VARX_lag2": Pipeline(
        steps=[
            ("preprocess", XArrayStandardScaler()),
            (
                "model",
                VARX(
                    lags={"y": 2, "precip": 0, "temp": 0, "evap": 0},
                    num_warmup=500,
                    num_chains=4,
                    num_samples=500,
                ),
            ),
        ]
    ),
    "VARX_lag3": Pipeline(
        steps=[
            ("preprocess", XArrayStandardScaler()),
            (
                "model",
                VARX(
                    lags={"y": 3, "precip": 0, "temp": 0, "evap": 0},
                    num_warmup=500,
                    num_chains=4,
                    num_samples=500,
                ),
            ),
        ]
    ),
}

all_models = {
    "Default": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", DefaultEnsemble())]
    ),
    **simple_models,
    **gp_models,
    **varx_models,
}

# --- Cross-validation ---
import pandas as pd

results = []
for name, model in all_models.items():
    logging.info(f"Fitting {name} model...")
    for i, (train_id, test_id) in enumerate(splits):
        prediction_file = RESULTS_DIR / f"{name}_{i}.csv"
        if prediction_file.exists() and not overwrite:
            logging.info(f"Loading {name} model (split {i + 1}/{num_splits})")
            predictions = pd.read_csv(prediction_file).assign(split=i + 1)
            predictions["Date"] = pd.to_datetime(predictions["Date"])
        else:
            logging.info(f"Fitting {name} model (split {i + 1}/{num_splits})")
            model.fit(X[train_id], y[train_id])

            try:
                az.to_netcdf(
                    model.named_steps["model"].trace, MODEL_DIR / f"{name}_{i}.nc"
                )
            except AttributeError:
                with open(MODEL_DIR / f"{name}_{i}.pkl", "wb") as f:
                    pkl.dump(model, f)
            except Exception as e:
                logging.warning(
                    f"Failed to save model {name} (split {i + 1}/{num_splits}): {e}"
                )
                continue

            preds = model.predict(
                X[: max(test_id) + 1],
                y=y[: max(test_id) + 1],
                forecast_steps=forecast_steps,
            )

            array = xr.concat(
                [preds, y[test_id].expand_dims({"value": ["true"]}, axis=-1)],
                dim="value",
            )

            predictions = pd.concat(
                [
                    arr.to_pandas().assign(
                        months_ahead=list(range(1, forecast_steps + 1)), model=name
                    )
                    for arr in array.transpose("lake", ...)
                ],
                axis=0,
                keys=preds.lake.values,
                names=["lake"],
            ).assign(split=i + 1)
            predictions.to_csv(prediction_file)
        results.append(predictions)

cv_results = pd.concat(results, axis=0)
logging.info(f"Cross-validation complete. {len(cv_results)} rows saved to {RESULTS_DIR}")
