import numpy as np
import xarray as xr
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

from src.modeling.modeling import ModelBase
from src.postprocessing.postprocessing import output_forecast_results

__all__ = [
    # Classes
    "DefaultEnsemble",
    "BaggedXArrayRegressor",
]


class DefaultEnsemble(ModelBase):
    @property
    def name(self):
        return "DefaultEnsemble"

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.month_df_ = None
        self.lakes = ["sup", "mic_hur", "eri", "ont"]

    def fit(self, X, y, *args, **kwargs):
        quantiles = (
            y.groupby("Date.month")
            .quantile(
                q=[self.alpha / 2, 1 - self.alpha / 2],
            )
            .assign_coords({"quantile": ["lower", "upper"]})
            .rename({"quantile": "variable"})
        )
        means = y.groupby("Date.month").mean().expand_dims(variable=["mean"])
        std = y.groupby("Date.month").std().expand_dims(variable=["std"])

        self.month_df_ = xr.concat([means, quantiles, std], dim="variable")
        return self

    def predict(self, X, y=None, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        # convert to DatetimeIndex type just in case for .month syntax

        forecast_index = X.indexes["Date"][-forecast_steps:]
        forecasts = (
            self.month_df_.sel(month=forecast_index.month)
            .rename(month="Date")
            .assign_coords(Date=forecast_index)
            .transpose("Date", "lake", "variable")
        )

        formatted_output = output_forecast_results(forecasts, forecast_index)
        return formatted_output

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


class BaggedXArrayRegressor(ModelBase):
    """
    Take any given Sklearn regressor model and wrap it into a BaggedRegressor to
    get a simple empirical estimate of standard deviation and quantile of predictions
    """

    def __init__(self, sklearn_bagging_regressor=None):
        super().__init__()
        if sklearn_bagging_regressor is None:
            self.sklearn_bagging_regressor_ = BaggingRegressor(
                estimator=LinearRegression(), n_estimators=250, n_jobs=-1
            )
        else:
            self.sklearn_bagging_regressor_ = sklearn_bagging_regressor
        # self.model = BaggingRegressor(estimator=self.regressor_, **bagging_kwargs)

    @property
    def name(self):
        return f"BaggedXarrayRegressor({self.sklearn_bagging_regressor_.__repr__()})"

    def fit(self, X, y, **kwargs):
        self.sklearn_bagging_regressor_.fit(X, y)

    def predict(
        self, X, y=None, forecast_steps=12, alpha=0.05, *args, **kwargs
    ) -> xr.DataArray:
        predictions = np.stack(
            [m.predict(X) for m in self.sklearn_bagging_regressor_.estimators_], axis=-1
        )

        output_array = np.stack(
            [
                predictions.mean(axis=-1),
                *np.quantile(predictions, q=[alpha / 2, 1 - alpha / 2], axis=-1),
                predictions.std(axis=-1),
            ],
            axis=-1,
        )[-forecast_steps:]

        return output_forecast_results(
            output_array, forecast_labels=X.indexes["Date"][-forecast_steps:]
        )

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


class RandomForest(ModelBase):
    def __init__(self, rf_model=None):
        super().__init__()
        self.rf_model_ = rf_model or RandomForestRegressor(n_estimators=100)

    @property
    def name(self):
        return "RandomForest"

    def fit(self, X, y, *args, **kwargs):
        self.rf_model_.fit(X, y)

    def predict(
        self, X, y=None, forecast_steps=12, alpha=0.05, *args, **kwargs
    ) -> xr.DataArray:
        tree_estimates = np.stack(
            [t.predict(X)[-forecast_steps:] for t in self.rf_model_.estimators_]
        )
        results = np.stack(
            [
                tree_estimates.mean(axis=0),
                np.quantile(tree_estimates, q=alpha / 2, axis=0),
                np.quantile(tree_estimates, q=1 - (alpha / 2), axis=0),
                tree_estimates.std(axis=0),
            ],
            axis=2,
        )
        return output_forecast_results(results, X.indexes["Date"][-forecast_steps:])

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


class BoostedRegressor(ModelBase):
    def __init__(self, alpha=0.05, base_regressor=None):
        super().__init__()
        self.alpha = alpha
        if base_regressor is not None:
            self.base_regressor = base_regressor
        else:
            self.base_regressor = GradientBoostingRegressor(loss="quantile")
        self.models_ = None

    @property
    def name(self):
        return "GradientBoostingRegressor"

    def fit(self, X, y, *args, **kwargs):
        assert y.shape[1] == 4, "y value must have 4 columns, one for each lake"

        # for consistency across the various models, we use the same arguments. This may or may not be optimal.
        # Create 4 copies of the models, one for each lake
        low_alpha, high_alpha = self.alpha / 2, 1 - (self.alpha / 2)
        self.models_ = [
            {
                "median": clone(self.base_regressor).set_params(alpha=0.5),
                "low": clone(self.base_regressor).set_params(alpha=low_alpha),
                "high": clone(self.base_regressor).set_params(alpha=high_alpha),
            }
            for _ in range(4)  # one for each lake!
        ]
        for i, model in enumerate(self.models_):
            for q_model in model.values():
                q_model.fit(X, y[:, i])

    def predict(self, X, y=None, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        # note: the normal "standard deviation" is not the same under quantile loss
        results = np.stack(
            [
                np.stack(
                    [
                        submodel.predict(X)[-forecast_steps:]
                        for submodel in model.values()
                    ],
                    axis=1,
                )
                for model in self.models_
            ]
        ).transpose(1, 0, 2)

        stddev_est = (
            np.abs(results[:, :, 0] - results[:, :, 1]) / 4
            + np.abs(results[:, :, 0] - results[:, :, 2]) / 4
        )

        output_results = np.append(results, np.expand_dims(stddev_est, axis=-1), axis=2)
        return output_forecast_results(
            output_results, X.indexes["Date"][-forecast_steps:]
        )

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
