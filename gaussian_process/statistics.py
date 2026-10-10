"""Stable Gaussian density and probability-band helpers (variance arguments)."""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr


def normal_logpdf(y, mu, var):
    """Log N(y | mu, var). All inputs broadcast, var is variance."""
    var = jnp.maximum(var, 1e-12)
    return -0.5 * (jnp.log(2. * jnp.pi * var) + (y-mu)**2 / var)


def _log1mexp(x):
    """Stable log(1-exp(x)), x<=0; safe inactive JAX branch."""
    x = jnp.minimum(x, 0.)
    return jnp.where(x < -jnp.log(2.), jnp.log1p(-jnp.exp(x)),
                     jnp.log(-jnp.expm1(x)))


def _logdiffexp(a, b):
    return a + _log1mexp(jnp.minimum(b-a, 0.))


def normal_interval_logprob(x_min, x_max, mu=0., var=1.):
    """Log P(x_min <= N(mu,var) <= x_max), stable in both tails.

    Returns -inf when x_max <= x_min. Argument is a *variance*, not sigma.
    """
    sigma = jnp.sqrt(jnp.maximum(var, 1e-12))
    lo, hi = (x_min-mu)/sigma, (x_max-mu)/sigma
    cdf = _logdiffexp(log_ndtr(hi), log_ndtr(lo))
    sf = _logdiffexp(log_ndtr(-lo), log_ndtr(-hi))
    return jnp.where(x_max > x_min, jnp.where(lo >= 0, sf, cdf), -jnp.inf)


def normal_interval_prob_stable(x_min, x_max, mu=0., var=1.):
    return jnp.exp(normal_interval_logprob(x_min, x_max, mu, var))
