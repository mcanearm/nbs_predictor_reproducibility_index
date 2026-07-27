from abc import ABC, abstractmethod

import arviz as az
import numpy as np
import xarray as xr
import jax
from jax import numpy as jnp
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.postprocessing import output_forecast_results

__all__ = [
    # Classes
    "ModelBase",
    "NumpyroModel",
    "ScaledTarget",
    # Functions
    "split_data",
    "train_model",
    "evaluate_model",
]


class ScaledTarget:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        # compute per-lake mean/std on the TRAIN y only, keep as xarray-friendly
        self.y_mean_ = y.mean(dim="Date")
        self.y_std_ = y.std(dim="Date")
        y_scaled = (y - self.y_mean_) / self.y_std_  # still xarray, indexes intact
        self.model.fit(X, y_scaled, **kwargs)
        return self

    def predict(self, X, y, return_original_scale=False, **kwargs):
        y_scaled = (y - self.y_mean_) / self.y_std_
        preds_scaled = self.model.predict(X=X, y=y_scaled, **kwargs)

        # by default, return the scaled predictions, but if requested, return the original scale
        if not return_original_scale:
            return preds_scaled
        else:
            unscaled_preds = self.y_mean_ + self.y_std_ * preds_scaled.sel(
                value=["mean", "lower", "upper"]
            )
            unscaled_sd = self.y_std_ * preds_scaled.sel(value="std")

            return xr.concat([unscaled_preds, unscaled_sd], dim="value").transpose(
                "Date", "lake", "value"
            )


class ModelBase(BaseEstimator, ABC):
    def __init__(self):
        super().__init__()

    def __sklearn_is_fitted__(self) -> bool:
        """Report fitted status based on the sklearn convention of trailing-underscore
        attributes. Subclasses should name all fitted state with a trailing underscore
        (e.g. self.coef_) so this check works automatically."""
        return any(v.endswith("_") and not v.startswith("__") for v in vars(self))

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, y=None, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        """
        Abstract method for running the predictions
        Args:
            X:
            y: the input series
            forecast_steps: how many time steps from the end of the input series are actually forecast steps. This
            is implemented so that we can consider the covariates

            *args: other arguments
            **kwargs: keyword arguments

        Returns: A nx4x3 XArray DataArray where n is the number of periods forward that you are forecasting,
        for 4 lakes, as well as a low and high estimate value
        """
        pass


class NumpyroModel(ModelBase):
    def __init__(
        self,
        lags=None,
        num_chains=4,
        num_samples=1000,
        num_warmup=1000,
        progress_bar=True,
        chain_method="parallel",
    ):
        super().__init__()
        self.lags = lags or {}
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.progress_bar = progress_bar
        self.chain_method = chain_method
        self.predictive_fn_ = None  # updated during kernel fitting
        self.trace_ = None
        self.mcmc_ = None
        self.lakes_ = None
        self._is_fitted = False
        self.cov_names_ = None

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    @abstractmethod
    def coords(self) -> dict[str, list[str]]:
        pass

    @property
    @abstractmethod
    def dims(self) -> dict[str, list[str]]:
        pass

    def fit(self, X: xr.DataArray, y: xr.DataArray, rng_key=None, *args, **kwargs):
        """
        Args:
            y: the time series to predict forward
            y_index: The datetime index for the series
            X: Extra X variables that can be passed in as predictors at fitting time
        Returns:
            fitted kernel object
        """
        self.lakes_, y_index = y.indexes["lake"], y.indexes["Date"]

        if rng_key is None:
            rng_key = self.get_rng_key()

        y_arr = jnp.array(y)
        months = jnp.array(y_index.month - 1)
        if "variable" in getattr(X, "dims", ()):
            self.cov_names_ = list(X.coords["variable"].values)
            covariates = jnp.stack(
                [jnp.array(X.sel(variable=v).values) for v in self.cov_names_], axis=-1
            )
        else:
            covariates = X

        kernel = NUTS(self.model)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
            chain_method=self.chain_method,
        )
        mcmc.run(rng_key, y=y_arr, months=months, lags=self.lags, covariates=covariates)
        samples = mcmc.get_samples()

        self.predictive_fn_ = Predictive(
            self.model, samples, return_sites=["y", "y_forecast"]
        )

        self.trace_ = az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)
        self._is_fitted = True

    @staticmethod
    @abstractmethod
    def model(y, months, lags, covariates, future=0):
        """
        Model method. Must be static to work with Numpyro MCMC.
        Args:
            y:
            y_index:
            lags:
            covariates:
            future:

        Returns:

        """
        pass

    def predict(self, X, y, forecast_steps=12, rng_key=None, *args, **kwargs):
        """

        Args:
            X:
            y: The time series to predict forward
            forecast_steps:

        Returns:

        """
        y_index = y.indexes["Date"]
        if rng_key is None:
            rng_key = self.get_rng_key()

        months = jnp.array(y_index.month - 1)
        if hasattr(self, "cov_names_"):
            covariates = jnp.stack(
                [jnp.array(X.sel(variable=v).values) for v in self.cov_names_], axis=-1
            )
        else:
            covariates = X

        forecast_marginal = self.predictive_fn_(
            rng_key,
            y=jnp.array(y),
            months=months,
            covariates=covariates,
            lags=self.lags,
            future=forecast_steps,
        )["y_forecast"]

        mean = jnp.mean(forecast_marginal, axis=0)
        std = jnp.std(forecast_marginal, axis=0)
        low, high = hpdi(forecast_marginal)

        forecasts = jnp.stack([mean, low, high, std], axis=2)

        results = output_forecast_results(forecasts, y_index[-forecast_steps:])
        return results

    def get_rng_key(self):
        return jax.random.key(np.random.randint(1e6))


def split_data(data, target_column):
    """
    Split data into train and test sets.

    Args:
    - data (pd.DataFrame): Input data
    - target_column (str): Name of the target column

    Returns:
    - tuple: (X_train, X_test, y_train, y_test)
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(model: ModelBase, X_train, y_train, *args, **kwargs):
    """
    Train a Random Forest Classifier.

    Args:
    - X_train (pd.DataFrame): Training features
    - y_train (pd.Series): Training target

    Returns:
    - RandomForestClassifier: Trained kernel
    """
    model.fit(X=X_train, y=y_train, *args, **kwargs)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a kernel using accuracy score.

    Args:
    - kernel: Trained kernel
    - X_test (pd.DataFrame): Test features
    - y_test (pd.Series): Test target

    Returns:
    - float: Accuracy score
    """
    y_pred = model.predict(None, X_test)
    return accuracy_score(y_test, y_pred)
