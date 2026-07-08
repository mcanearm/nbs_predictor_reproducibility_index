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
        }

    @property
    def dims(self):
        return {
            "intercept": ["month", "lakes"],
            "sigma": ["lakes"],
            "l_omega": ["series", "lakes"],
            "corr": ["series", "lakes"],
            "theta": ["lakes"],
            **{
                f"{k}_alpha": (
                    ["series", "lakes", "lag"] if k == "y" else ["series", "lakes"]
                )
                for k in self.lags.keys()
            },
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


class VARX(VAR):
    @property
    def name(self):
        return "VARX"

    @property
    def coords(self):
        return {
            "series": self.lakes,
            "lakes": self.lakes,
            "month": list(calendar.month_abbr)[1:],
            "lag": range(1, self.lags["y"] + 1),
        }

    @property
    def dims(self):
        return {
            "intercept": ["month", "lakes"],
            "l_omega": ["series", "lakes"],
            "corr": ["series", "lakes"],
            "theta": ["lakes"],
            "y_alpha": ["series", "lakes", "lag"],
            "x_alpha": ["variable", "series", "lakes"],
        }

    def load(cls, path):
        pass

    def save(self, path):
        pass

    @staticmethod
    @numpyro.handlers.reparam(config={"intercept": LocScaleReparam(0)})
    def model(y, months, lags, covariates, future=0):
        """
        lags: {"y": ar_lag, "x": x_lag}
          x_lag=0 → concurrent: use x[t] to predict y[t]
          x_lag=k → lagged: use x[t-k], …, x[t-1]
        covariates: dict of jnp arrays, each (T, 4).

        Shapes:
          prev_y carry:  (ar_lag, 4)
          y_alpha:       (out_lake=4, in_lake=4, ar_lag)
          x_alpha:       (n_covs, 4)            when x_lag=0
                         (n_covs, 4, x_lag)     when x_lag>0
          x_for_scan:    (T-ar_lag, n_covs, 4)  when x_lag=0
                         (T-ar_lag, x_lag, n_covs, 4) when x_lag>0
        """
        ar_lag = lags["y"]
        x_lag = lags.get("x", 0)

        x_arr = covariates  # (T, n_covs, 4), pre-stacked in caller-controlled order
        T, n_covs = x_arr.shape[0], x_arr.shape[1]

        # Pre-slice x aligned with prediction steps [ar_lag, T).
        # At step t we predict y[ar_lag+t]; x context depends on x_lag.
        if x_lag == 0:
            x_for_scan = x_arr[ar_lag:]  # (T-ar_lag, n_covs, 4)
        else:
            # oldest-to-most-recent lag window; j=x_lag gives x[ar_lag+t - x_lag]
            x_for_scan = jnp.stack(
                [x_arr[ar_lag - j : T - j] for j in range(x_lag, 0, -1)], axis=1
            )  # (T-ar_lag, x_lag, n_covs, 4)

        global_mu = numpyro.sample("global_mu", dist.Normal(0, 1))
        nu = numpyro.sample("nu", dist.HalfNormal(10.0))
        covar_sigma = numpyro.sample("covar_sigma", dist.HalfNormal(1))

        # y AR weights: full cross-lake VAR matrix per lag step
        y_alpha = numpyro.sample(
            "y_alpha", dist.Normal(0, covar_sigma), sample_shape=(4, 4, ar_lag)
        )  # (out_lake, in_lake, ar_lag)

        # x weights: diagonal lake assumption — covariate v at lake o → output lake o
        if x_lag == 0:
            x_alpha = numpyro.sample(
                "x_alpha", dist.Normal(0, covar_sigma), sample_shape=(n_covs, 4, 4)
            )  # (n_covs, in_lake, out_lake)
        else:
            x_alpha = numpyro.sample(
                "x_alpha", dist.Normal(0, covar_sigma), sample_shape=(n_covs, 4, 4, x_lag)
            )  # (n_covs, in_lake, out_lake, x_lag)

        theta = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))
        intercept_sigma = numpyro.sample("intercept_sigma", dist.HalfNormal(1))
        with numpyro.plate("lakes", size=4):
            with numpyro.plate("months", size=12):
                intercept = numpyro.sample(
                    "intercept", dist.Normal(global_mu, intercept_sigma)
                )  # (12, 4)

        l_omega = numpyro.sample("l_omega", dist.LKJCholesky(4, concentration=0.5))
        numpyro.deterministic("corr", l_omega @ l_omega.T)
        L_Omega = jnp.sqrt(theta)[..., None] * l_omega

        def transition_fn(carry, scan_input):
            prev_y = carry  # (ar_lag, 4)
            month_t = scan_input[0]  # scalar ∈ [0, 11]
            x_t = scan_input[1]  # (n_covs, 4) or (x_lag, n_covs, 4)

            # y AR: (out, in, lag) × (lag, in) → (out,)
            m = jnp.einsum("oij,ji->o", y_alpha.reshape(4, 4, ar_lag), prev_y)

            # x term: full cross-lake — covariate v at input lake i contributes to output lake o
            if x_lag == 0:
                # (n_covs, in, out) × (n_covs, in) → (out,)
                m = m + jnp.einsum("vio,vi->o", x_alpha, x_t)
            else:
                # (n_covs, in, out, lag) × (lag, n_covs, in) → (out,)
                m = m + jnp.einsum("vioj,jvi->o", x_alpha, x_t)

            y_t = numpyro.sample(
                "y",
                dist.MultivariateStudentT(
                    df=nu, loc=intercept[month_t, :] + m, scale_tril=L_Omega
                ),
            )

            if ar_lag > 1:
                new_carry = jnp.append(prev_y[1:], y_t.reshape(1, -1), axis=0)
            else:
                new_carry = y_t.reshape(1, -1)
            return new_carry, y_t

        y_fit = y[:-future] if future > 0 else y
        with numpyro.handlers.condition(data={"y": y_fit[ar_lag:]}):
            _, ys = scan(transition_fn, y[:ar_lag], (months[ar_lag:], x_for_scan))

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

        lagged_covars = [covariates[covar][max_lag:] for covar in lags if covar != "y"]
        covars = jnp.concatenate(lagged_covars, axis=-1)

        theta = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))

        l_omega = numpyro.sample("l_omega", dist.LKJCholesky(4, concentration=0.5))
        numpyro.deterministic("corr", l_omega @ l_omega.T)
        sigma = jnp.sqrt(theta)
        L_Omega = sigma[..., None] * l_omega

        input_dim = 4 * lags["y"] + covars.shape[-1]
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
            _, ys = scan(transition_fn, initial_values, covars)

        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])
