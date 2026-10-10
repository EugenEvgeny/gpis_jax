"""Matérn 1/2 (exponential) kernel."""

from functools import partial

import jax
import jax.numpy as jnp

from .base_kernel import BaseKernel


class Matern12Kernel(BaseKernel):
    """Matérn 1/2 covariance. Not differentiable at x=y.

    Mixed and higher derivatives at coinciding points are mathematically
    undefined; do not use this kernel for gradient-observation GP models or KSD.
    """

    def __init__(self, sig_f=1.0, l_f=1.0, dim=2):
        self.dim = int(dim)
        super().__init__([sig_f, l_f])

    @property
    def name(self):
        return "matern12"

    @property
    def sig_f(self):
        return self.params[0]

    @property
    def l_f(self):
        return self.params[1]

    def __repr__(self):
        return f"Matern12Kernel(sig_f={self.sig_f}, l_f={self.l_f})"

    @partial(jax.jit, static_argnums=(0,))
    def k_fun(self, x, y, sig_f=1.0, l_f=1.0):
        r = (x - y) / l_f
        # Safeguard the sqrt backward pass without shifting the diagonal.
        r2 = jnp.dot(r, r)
        norm = jnp.sqrt(jnp.maximum(r2, jnp.finfo(r2.dtype).tiny))
        return sig_f**2 * jnp.exp(-jnp.where(r2 == 0, 0.0, norm))
