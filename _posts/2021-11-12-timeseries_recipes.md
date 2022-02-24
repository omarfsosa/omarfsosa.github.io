---
title:  "Time series recipes in Numpyro"
date:   2022-02-24
---

# A catalogue of time-series models written in Numpyro
I've compiled here a bunch of numpyro timeseries models that I find myself
coming back to. Maybe they'll be of use to others.

These are the required import statments

```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import sample, plate
from numpyro.contrib.control_flow import scan
from numpyro.infer import Predictive, NUTS, MCMC

rng = jax.random.PRNGKey(0)
```

## Auto regressive models

```python
def ar2(num_timesteps, y0, y1, y=None):
    """
    An auto regressive (K=2) model.
    
    Parameters
    ----------
    num_timesteps: int, positive
        The total number of timesteps to model
    
    y: ndarray, shape (num_timesteps,)
        The observed values beyond y0 and y1
    
    y0, y1: floats
        The initial values of the process
    """
    a1 = sample("a1", dist.Normal())
    a2 = sample("a2", dist.Normal())
    const = sample("const", dist.Normal())
    sigma = sample("sigma", dist.Exponential())

    def transition(carry, _):
        y_prev, y_prev_prev = carry
        m_t = const + a1 * y_prev + a2 * y_prev_prev
        y_t = sample("y", dist.Normal(m_t, sigma))
        carry = (y_t, y_prev)
        return carry, None

    timesteps = jnp.arange(num_timesteps)
    init = (y0, y1)
    with numpyro.handlers.condition(data={"y": y}):
        scan(transition, init, timesteps)
```

_Prior simulation_
```python
num_timesteps = 40
y0, y1 = 0.3, -0.1

prior = Predictive(ar2, num_samples=10)
prior_samples = prior(rng, num_timesteps, y0, y1)
```

_Fitting_
```python
y0, y1, *y = y_true
y = np.array(y)
num_timesteps = len(y)
mcmc = MCMC(NUTS(ar2), **mcmc_settings)
mcmc.run(rng, num_timesteps, y0, y1, y)
```

_Forecast_
```python
num_forecast = 10
y0, y1 = y_true[-2:]
forecaster = Predictive(ar2, posterior_samples=mcmc.get_samples())
forecast_samples = forecaster(rng, num_forecast, y0, y1)
```

## Exponential smoothing models

```python
def exponential_smoothing(n_timesteps, y=None):
    alpha = sample("alpha", dist.Beta(5, 5))
    sigma = sample("sigma", dist.LogNormal())
    l0 = sample("l0", dist.Normal())

    def transition(carry, _):
        l_prev = carry
        y_t = numpyro.sample("y", dist.Normal(l_prev, sigma))
        e_t = y_t - l_prev
        l_t = l_prev + alpha * e_t
        return l_t, y_t

    timesteps = jnp.arange(n_timesteps)
    init = l0
    with numpyro.handlers.condition(data={"y": y}):
        scan(transition, init, timesteps)
```

```python
def holt(n_timesteps, y=None):
    alpha = sample("alpha", dist.Beta(5, 5))
    beta = sample("beta", dist.Beta(5, 5))
    sigma = sample("sigma", dist.LogNormal())
    l0 = sample("l0", dist.Normal())
    b0 = sample("b0", dist.Normal())

    def transition(carry, _):
        l_prev, b_prev = carry
        y_t = numpyro.sample("y", dist.Normal(l_prev + b_prev, sigma))
        e_t = y_t - l_prev - b_prev
        l_t = l_prev + b_prev + alpha * e_t
        b_t = b_prev + alpha * beta * e_t
        carry = (l_t, b_t)
        return carry, y_t

    timesteps = jnp.arange(n_timesteps)
    init = (l0, b0)
    with numpyro.handlers.condition(data={"y": y}):
        scan(transition, init, timesteps)
```

```python
def holt_winters(n_timesteps, seasonality, y=None):
    alpha = sample("alpha", dist.Beta(5, 5))
    beta = sample("beta", dist.Beta(5, 5))
    gamma = sample("gamma", dist.Beta(5, 5))

    l0 = sample("l0", dist.Normal(0, 1))
    b0 = sample("b0", dist.Normal(0, 1))
    with plate("plate_period", seasonality):
        s0 = sample("s0", dist.Normal(0, 1))

    sigma = sample("sigma", dist.LogNormal())

    def transition(carry, _):
        l_prev, b_prev, *s_prev = carry
        m_t = l_prev + b_prev + s_prev[0]
        y_t = sample("y", dist.Normal(m_t, sigma))
        e_t = y_t - m_t
        l_t = l_prev + b_prev + alpha * e_t
        b_t = b_prev + alpha * beta * e_t
        s_t = s_prev[0] + γ * (1 - alpha) * e_t
        s_new = jnp.concatenate((jnp.array(s_prev[1:]), jnp.array([s_t])))
        carry = (l_t, b_t, s_new)
        return carry, y_t

    timesteps = jnp.arange(n_timesteps)
    init = (l0, b0, s0)
    with numpyro.handlers.condition(data={"y": y}):
        scan(transition, init, timesteps)
```

## Hierarchical time series

WIP

## Gaussian Process

WIP