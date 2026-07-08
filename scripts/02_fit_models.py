"""
Script 2: Model Fitting (Cross-Validation)

Fits all candidate models via time-series cross-validation and saves per-model
per-split prediction CSVs and fitted model artifacts to disk.

Outputs:
  - data/model_results/{model}_{split}.csv   (one per model x CV split)
  - data/models/{model}_{split}.nc           (Bayesian models, ArviZ NetCDF)
  - data/models/{model}_{split}.pkl          (non-Bayesian models, pickle)
  - data/y_scaler.pkl                        (fitted XArrayStandardScaler for y)

Run from the project root:
    python scripts/02_fit_models.py

Set overwrite=True (below) to re-fit models even when saved results exist.
"""

import logging
import os
from pathlib import Path

import arviz as az
import dill as pkl
import numpy as np
import numpyro
import pandas as pd
import torch
import xarray as xr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm

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

# set global seed for scikit-learn models
np.random.seed(20250607)

logging.basicConfig(level=logging.WARNING)

device = "cpu"
torch.set_default_device(device)
os.environ["JAX_PLATFORM_NAME"] = device
numpyro.set_platform(device)
numpyro.set_host_device_count(8)

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
                    GaussianProcessRegressor(
                        kernel=1.0
                        * kernels.Matern(nu=1.5)
                        * kernels.RationalQuadratic()
                    )
                ),
            ),
        ]
    ),
    "MultitaskGP": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                MultitaskGP(epochs=100, kernel_args={"rank": 1}),
            ),
        ]
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
            (
                "model",
                BoostedRegressor(
                    base_regressor=GradientBoostingRegressor(loss="quantile")
                ),
            ),
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
                    lags={"y": 1, "x": 0},
                    num_warmup=2500,
                    num_chains=4,
                    num_samples=500,
                    progress_bar=True,
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
                    lags={"y": 2, "x": 0},
                    num_warmup=2500,
                    num_chains=4,
                    num_samples=500,
                    progress_bar=True,
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
                    lags={"y": 3, "x": 0},
                    num_warmup=2500,
                    num_chains=4,
                    num_samples=500,
                    progress_bar=True,
                ),
            ),
        ]
    ),
    "NARX_lag1": Pipeline(
        steps=[
            ("preprocess", XArrayStandardScaler()),
            (
                "model",
                VARX(
                    lags={"y": 1, "x": 0},
                    num_warmup=2500,
                    num_chains=4,
                    num_samples=500,
                    progress_bar=True,
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

results = []
model_bar = tqdm(all_models.items(), desc="Models", unit="model", position=0)
for name, model in model_bar:
    model_bar.set_postfix_str(name)
    split_bar = tqdm(
        enumerate(splits),
        total=num_splits,
        desc=name,
        unit="split",
        leave=False,
        position=1,
    )
    for i, (train_id, test_id) in split_bar:
        prediction_file = RESULTS_DIR / f"{name}_{i}.csv"
        if prediction_file.exists() and not overwrite:
            split_bar.set_postfix_str(f"split {i + 1} (cached)")
            predictions = pd.read_csv(prediction_file).assign(split=i + 1)
            predictions["Date"] = pd.to_datetime(predictions["Date"])
        else:
            split_bar.set_postfix_str(f"split {i + 1} (fitting)")
            model.fit(X[train_id], y[train_id])
            for bar in tqdm._instances:
                if bar.pos > 1:
                    bar.close()

            try:
                az.to_netcdf(
                    model.named_steps["model"].trace, MODEL_DIR / f"{name}_{i}.nc"
                )
            except AttributeError:
                try:
                    with open(MODEL_DIR / f"{name}_{i}.pkl", "wb") as f:
                        pkl.dump(model, f)
                except Exception as e:
                    tqdm.write(
                        f"Warning: could not save {name} split {i + 1} artifact: {e}"
                    )
            except Exception as e:
                tqdm.write(
                    f"Warning: could not save {name} split {i + 1} artifact: {e}"
                )

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
tqdm.write(f"Cross-validation complete. {len(cv_results)} rows saved to {RESULTS_DIR}")
