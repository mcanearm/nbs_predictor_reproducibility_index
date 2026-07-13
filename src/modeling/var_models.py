import calendar
from functools import reduce
from typing import Union

import jax
import numpy as np
import numpyro
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import LocScaleReparam

from src.modeling.modeling import NumpyroModel
from src.utils import flatten_array, lag_array

__all__ = [
    # Classes
    "VARX",
    "NARX",
    "VAR",
]


class VAR(NumpyroModel):
    @property
    def name(self):
        return "VAR"

    def __init__(
        self,
        lags: Union[None, dict] = None,
        num_chains=4,
        num_samples=1000,
        num_warmup=1000,
        lakes=("sup", "mic_hur", "eri", "ont"),
        progress_bar=True,
    ):
        super().__init__()
        if lags is None:
            self.lags = {"y": 3, "precip": 0}
        else:
            self.lags = lags
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.lakes = list(lakes)
        self.progress_bar = progress_bar
        self.predictive_fn_ = None  # updated during model fitting
        self.trace_ = None
        self.mcmc_ = None

    @property
    def coords(self):
        return {
            "series": self.lakes,
            "lakes": self.lakes,
            "month": list(calendar.month_abbr)[1:],
            "lag": range(1, self.lags["y"] + 1),
            "covariate": self.cov_names_,
        }

    @property
    def dims(self):
        return {
            "intercept": ["month", "lakes"],
            "sigma": ["lakes"],
            "l_omega": ["series", "lakes"],
            "corr": ["series", "lakes"],
            "theta": ["lakes"],
            "covar_alphas": ["covariate", "series", "lakes"],
        }

    @staticmethod
    @numpyro.handlers.reparam(config={"intercept": LocScaleReparam(0)})
    def model(y, months, lags, covariates, future=0):
        """
        Autoregressive process.
        Args:
            y: the time series to fit
            covariates: An XArray of covariates for use in the model. Leading index should be date, second index should be lake, and the third index is the actual covariate values.
            lags: A dictionary indicating which covariates have which lags
            future: How many periods to run into the future.

        Returns:
            None - samples

        """
        global_mu = numpyro.sample("global_mu", dist.Normal(0, 1))
        nu = numpyro.sample("nu", dist.HalfNormal(10.0))

        # this effectively removes first first entries from the covariates so that
        # the total length of the covariates is the same as the length of the y.
        ar_lag = max_lag = lags.get("y")
        theta = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))

        intercept_sigma = numpyro.sample("intercept_sigma", dist.HalfNormal(1))
        with numpyro.plate("lakes", size=4):
            with numpyro.plate("months", size=12):
                intercept = numpyro.sample(
                    "intercept", dist.Normal(global_mu, intercept_sigma)
                )
            with numpyro.plate("lags", size=max_lag):
                lag_sigma = numpyro.sample("lag_sigma", dist.HalfNormal(1))
                lag_beta = numpyro.sample("lag_terms", dist.Normal(0, lag_sigma))

        # correlation stucture for a multivariate T distribution; note that
        # are assuming a t-distribution so we need a "nu" parameter as well.
        l_omega = numpyro.sample("l_omega", dist.LKJCholesky(4, concentration=0.5))
        numpyro.deterministic("corr", l_omega @ l_omega.T)
        sigma = jnp.sqrt(theta)
        L_Omega = sigma[..., None] * l_omega

        def transition_fn(carry, covars):
            prev_y = carry
            month_t = covars

            m_t = intercept[month_t] + jnp.einsum("tl,tl -> l", lag_beta, prev_y)
            # generate an empty sum, one for each lake
            y_t = numpyro.sample(
                "y", dist.MultivariateStudentT(df=nu, loc=m_t, scale_tril=L_Omega)
            )

            if ar_lag > 1:
                new_vals = jnp.append(prev_y[1:], y_t.reshape(1, -1), axis=0)
            else:
                new_vals = y_t.reshape(1, -1)
            return new_vals, y_t

        prev = y[:max_lag][-ar_lag:]
        initial_values = prev

        covar_tuple = months[max_lag:]

        if future > 0:
            y_fit = y[:-future]
        else:
            y_fit = y

        # The conditioning here is extremely important, as it tells us
        # what the actual values are. We start with the first value in the series
        # for which we have a lagged value and predict forward from there
        with numpyro.handlers.condition(data={"y": y_fit[max_lag:]}):
            _, ys = scan(transition_fn, initial_values, covar_tuple)

        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])


class VARX(NumpyroModel):
    @property
    def name(self):
        return "VARX"

    def __init__(
        self,
        lags: Union[None, dict] = None,
        num_chains=4,
        num_samples=1000,
        num_warmup=1000,
        lakes=("sup", "mic_hur", "eri", "ont"),
    ):
        super().__init__()
        if lags is None:
            self.lags = {"y": 3, "precip": 0, "evap": 0, "temp": 0}
        else:
            self.lags = lags
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.lakes = list(lakes)
        self.predictive_fn = None  # updated during model fitting
        self.trace_ = None
        self.mcmc_ = None

    @property
    def coords(self):
        return {
            "series": self.lakes,
            "lakes": self.lakes,
            "month": list(calendar.month_abbr)[1:],
            "lag": range(1, self.lags["y"] + 1),
            "covariate": self.cov_names_,
        }

    @property
    def dims(self):
        return {
            "intercept": ["month", "lakes"],
            "sigma": ["lakes"],
            "corr": ["series", "lakes"],
            "theta": ["lakes"],
            "covar_alphas": ["covariate", "series", "lakes"],
            "y_alphas": ["lag", "series", "lakes"],
        }

    @classmethod
    def load(cls, path):
        pass

    def save(self, path):
        pass

    @staticmethod
    @numpyro.handlers.reparam(config={"intercept": LocScaleReparam(0)})
    def model(y, months, lags, covariates, future=0):
        """
        Autoregressive process.
        Args:
            y: the time series to fit
            months: An array of month indices for each time step.
            covariates: An XArray of covariates for use in the model. Leading index should be date, second index should be lake, and the third index is the actual covariate values.
            lags: A dictionary indicating which covariates have which lags
            future: How many periods to run into the future.

        Returns:
            None - samples

        """
        global_mu = numpyro.sample("global_mu", dist.Normal(0, 1))
        nu = numpyro.sample("nu", dist.HalfNormal(10.0))

        ar_lag = max_lag = lags.get("y")

        # remove all lagging for covariates
        lagged_covars = covariates[ar_lag:]

        n_covs = covariates.shape[-1]

        # n_covs + 1 because we are including the lagged y term as a covariate
        with numpyro.plate("covariate", n_covs, dim=-3):
            with numpyro.plate("in_lake", 4, dim=-2):
                with numpyro.plate("out_lake", 4, dim=-1):
                    covar_alphas = numpyro.sample("covar_alphas", dist.Normal(0, 0.5))

        with numpyro.plate("lag", ar_lag, dim=-3):
            with numpyro.plate("y_in_lake", 4, dim=-2):
                with numpyro.plate("y_out_lake", 4, dim=-1):
                    y_alphas = numpyro.sample("y_alphas", dist.Normal(0, 0.5))

        with numpyro.plate("lakes", size=4):
            with numpyro.plate("months", size=12):
                intercept = numpyro.sample("intercept", dist.Normal(global_mu, 1))

        theta = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))
        l_omega = numpyro.sample("l_omega", dist.LKJCholesky(4, concentration=0.5))
        numpyro.deterministic("corr", l_omega @ l_omega.T)
        sigma = jnp.sqrt(theta)
        L_Omega = sigma[..., None] * l_omega

        def transition_fn(carry, covars):
            prev_y = carry
            month_t = covars[0]
            covar_values = covars[1]

            m = jnp.zeros((4,))

            for i in jnp.arange(n_covs):
                alphas = covar_alphas[i]
                dataset = covar_values[:, i]

                # first, the covariates
                # if lags are included, do each one separately; this looks buggy though
                if len(alphas.shape) > 2:
                    for j in jnp.arange(alphas.shape[-1]):
                        m += jnp.matmul(alphas[:, :, j], dataset[j, :])
                else:
                    # This is the thing that SHOULD be happening every time when X is provided with no lags.
                    m += jnp.matmul(alphas, dataset)

            # add the lagged y terms to the mean
            for i in jnp.arange(ar_lag):
                alphas = y_alphas[i]
                dataset = prev_y[i, :]
                m += jnp.matmul(alphas, dataset)

            # now add the intercepts
            m_t = intercept[month_t, :] + m
            y_t = numpyro.sample(
                "y", dist.MultivariateStudentT(df=nu, loc=m_t, scale_tril=L_Omega)
            )

            # if we have more than one lag, we need to keep all previous values except the first one
            # If we only have one lag, we can just use the current value.
            if ar_lag > 1:
                new_vals = jnp.append(prev_y[1:], y_t.reshape(1, -1), axis=0)
            else:
                new_vals = y_t.reshape(1, -1)
            return new_vals, y_t

        prev = y[:max_lag][-ar_lag:]
        # need to subtract one because indexing starts at 0.
        initial_values = prev

        covars = (months[max_lag:], lagged_covars)

        if future > 0:
            y_fit = y[:-future]
        else:
            y_fit = y

        with numpyro.handlers.condition(data={"y": y_fit[max_lag:]}):
            _, ys = scan(transition_fn, initial_values, covars)

        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])


class NARX(NumpyroModel):
    def __init__(self, lags=None, num_chains=4, num_samples=1000, num_warmup=1000):
        super().__init__(lags, num_chains, num_samples, num_warmup)
        self.lags = lags or {"y": 3, "evap": 2, "precip": 2}

    @property
    def name(self):
        return "NARX"

    @property
    def coords(self):
        pass

    @property
    def dims(self):
        pass

    @staticmethod
    def model(y, months, lags, covariates, future=0):
        """
        Autoregressive process.
        Args:
            y: the time series to fit
            months: jnp array of 0-indexed months (pre-computed from y_index)
            lags: A dictionary indicating which covariates have which lags
            future: How many periods to run into the future.

        Returns:
            None - samples

        """
        nu = numpyro.sample("nu", dist.HalfNormal(10.0))

        ar_lag = max_lag = lags.get("y")

        # align with VARX, just assume no covariate lags for now
        theta = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))

        l_omega = numpyro.sample("l_omega", dist.LKJCholesky(4, concentration=0.5))
        numpyro.deterministic("corr", l_omega @ l_omega.T)
        sigma = jnp.sqrt(theta)
        L_Omega = sigma[..., None] * l_omega

        input_dim = 4 * lags["y"] + covariates.shape[-1]
        h1 = 8
        output_dim = 4  # 4 lakes

        # first layer of the neural network
        w1 = numpyro.sample(
            "w1", dist.Normal(jnp.zeros((input_dim, h1)), jnp.ones((input_dim, h1)))
        )
        b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(h1), jnp.ones(h1)))

        # output layer of the neural network
        w2 = numpyro.sample(
            "w2",
            dist.Normal(jnp.zeros((h1, output_dim)), jnp.ones((h1, output_dim))),
        )
        b2 = numpyro.sample(
            "b2", dist.Normal(jnp.zeros(output_dim), jnp.ones(output_dim))
        )

        def transition_fn(carry, covars):
            """
            Function for predicting the next value in the time series given the previous values and covariates.
            Args:
                carry: The previous prediction of the time series
                covars: The covariates for this time step
            Returns:
                A tuple containing the next values to carry forward (y value plus the lag) and the
                predicted value for this time step.
            """
            prev_y = carry

            input_vals = jnp.concatenate([carry.reshape(-1), covars], axis=-1)
            z1 = jax.nn.relu(jnp.matmul(input_vals, w1) + b1)
            z2 = jnp.matmul(z1, w2) + b2

            y_t = numpyro.sample(
                "y", dist.MultivariateStudentT(df=nu, loc=z2, scale_tril=L_Omega)
            )

            if ar_lag > 1:
                new_vals = jnp.append(prev_y[1:], y_t.reshape(1, -1), axis=0)
            else:
                new_vals = y_t.reshape(1, -1)
            return new_vals, y_t

        initial_values = jnp.array(y[:max_lag][-ar_lag:])

        if future > 0:
            y_fit = jnp.array(y[:-future])
        else:
            y_fit = jnp.array(y)

        with numpyro.handlers.condition(data={"y": y_fit[max_lag:]}):
            _, ys = scan(transition_fn, initial_values, covariates[ar_lag:])

        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])
