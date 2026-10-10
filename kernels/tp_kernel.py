"""Compactly supported third-order polynomial kernel."""

import jax.numpy as jnp

from .base_kernel import BaseKernel


class TPSKernel(BaseKernel):
    """k(r) = 2r^3 - 3Cr^2 + C^3 for r<=C, else zero."""

    def __init__(self, C0=1.0, dim=3):
        self.dim = int(dim)
        super().__init__([C0])

    def k_fun(self, x, y, C0):
        r = jnp.linalg.norm(x - y)
        base = 2.0 * r**3 - 3.0 * C0 * r**2 + C0**3
        return jnp.where(r <= C0, base, 0.0)

    def dk_fun(self, x, y, C0):
        diff = x - y
        r = jnp.linalg.norm(diff)
        return jnp.where(r <= C0, 6.0 * (r - C0) * diff, jnp.zeros_like(diff))

    def dk_y_fun(self, x, y, C0):
        return -self.dk_fun(x, y, C0)

    def ddk_fun(self, x, y, C0):
        diff = x - y
        r = jnp.linalg.norm(diff)
        safe_r = jnp.maximum(r, jnp.finfo(r.dtype).tiny)
        dxy = 6.0 * (C0 - r) * jnp.eye(x.shape[0], dtype=x.dtype) - 6.0 * jnp.outer(diff, diff) / safe_r
        return jnp.where(r <= C0, dxy, jnp.zeros_like(dxy))

    def ddkxx_fun(self, x, y, C0):
        return -self.ddk_fun(x, y, C0)

    def __repr__(self):
        return f"TPSKernel(C0={self.params[0]}, dim={self.dim})"
