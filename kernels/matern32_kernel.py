"""Matérn 3/2 kernel with analytic first and mixed derivatives."""

from functools import partial

import jax
import jax.numpy as jnp

from .base_kernel import BaseKernel


class Matern32Kernel(BaseKernel):
    def __init__(self, sig_f=1.0, l_f=1.0, dim=2):
        self.dim = int(dim)
        super().__init__([sig_f, l_f])

    @property
    def name(self):
        return "matern32"

    @property
    def sig_f(self):
        return self.params[0]

    @property
    def l_f(self):
        return self.params[1]

    def __repr__(self):
        return f"Matern32Kernel(sig_f={self.sig_f}, l_f={self.l_f})"

    @partial(jax.jit, static_argnums=(0,))
    def k_fun(self, x, y, sig_f=1.0, l_f=1.0):
        d = x - y
        r2 = jnp.dot(d, d)
        r = jnp.sqrt(jnp.maximum(r2, jnp.finfo(r2.dtype).tiny))
        a = jnp.sqrt(3.0) / l_f
        return sig_f**2 * (1.0 + a * r) * jnp.exp(-a * r)

    @partial(jax.jit, static_argnums=(0,))
    def dk_fun(self, x, y, sig_f=1.0, l_f=1.0):
        diff = x - y
        a = jnp.sqrt(3.0) / l_f
        r = jnp.linalg.norm(diff)
        return -sig_f**2 * a**2 * jnp.exp(-a * r) * diff

    @partial(jax.jit, static_argnums=(0,))
    def dk_y_fun(self, x, y, sig_f=1.0, l_f=1.0):
        return -self.dk_fun(x, y, sig_f, l_f)

    @partial(jax.jit, static_argnums=(0,))
    def ddk_fun(self, x, y, sig_f=1.0, l_f=1.0):
        diff = x - y
        r = jnp.linalg.norm(diff)
        a = jnp.sqrt(3.0) / l_f
        # Use a masked denominator for the continuous r->0 limit.
        rr = jnp.outer(diff, diff)
        factor = a / jnp.maximum(r, jnp.finfo(r.dtype).tiny)
        return sig_f**2 * a**2 * jnp.exp(-a * r) * (
            jnp.eye(x.shape[0], dtype=x.dtype) - factor * rr
        )

    @partial(jax.jit, static_argnums=(0,))
    def ddkxx_fun(self, x, y, sig_f=1.0, l_f=1.0):
        return -self.ddk_fun(x, y, sig_f, l_f)
