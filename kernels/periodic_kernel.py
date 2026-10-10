"""Periodic kernel on angles."""

from functools import partial

import jax
import jax.numpy as jnp

from .base_kernel import BaseKernel


class TotalPeriodicKernel(BaseKernel):
    """k(a,b) = sig_f² exp(-2 sin²(d(a-b)/2)/l_f²) + sig_r²."""

    def __init__(self, sig_f=1.0, l_f=1.0, sig_r=0.0, d=1, dim=1):
        self.dim = int(dim)
        self.d = d
        super().__init__([sig_f, l_f, sig_r])

    @property
    def name(self):
        return "PeriodicKernel"

    @property
    def sig_f(self):
        return self.params[0]

    @property
    def l_f(self):
        return self.params[1]

    def __repr__(self):
        return f"TotalPeriodicKernel(sig_f={self.sig_f}, l_f={self.l_f}, d={self.d})"

    @partial(jax.jit, static_argnums=(0,))
    def k_fun(self, phi_x, phi_y, sig_f=1.0, l_f=1.0, sig_r=0.0):
        delta = jnp.squeeze(phi_x - phi_y)
        sin_sq = jnp.sin(self.d * delta / 2.0) ** 2
        return sig_f**2 * jnp.exp(-2.0 * sin_sq / l_f**2) + sig_r**2

    @partial(jax.jit, static_argnums=(0,))
    def dk_fun(self, phi_x, phi_y, sig_f=1.0, l_f=1.0, sig_r=0.0):
        delta = jnp.squeeze(phi_x - phi_y)
        varying = self.k_fun(phi_x, phi_y, sig_f, l_f, sig_r) - sig_r**2
        return jnp.atleast_1d(-self.d * jnp.sin(self.d * delta) * varying / l_f**2)

    @partial(jax.jit, static_argnums=(0,))
    def dk_y_fun(self, phi_x, phi_y, sig_f=1.0, l_f=1.0, sig_r=0.0):
        return -self.dk_fun(phi_x, phi_y, sig_f, l_f, sig_r)

    @partial(jax.jit, static_argnums=(0,))
    def ddk_fun(self, phi_x, phi_y, sig_f=1.0, l_f=1.0, sig_r=0.0):
        delta = jnp.squeeze(phi_x - phi_y)
        varying = self.k_fun(phi_x, phi_y, sig_f, l_f, sig_r) - sig_r**2
        d2 = self.d**2
        val = varying * d2 * (
            jnp.cos(self.d * delta) / l_f**2
            - jnp.sin(self.d * delta) ** 2 / l_f**4
        )
        return jnp.reshape(val, (1, 1))

    @partial(jax.jit, static_argnums=(0,))
    def ddkxx_fun(self, phi_x, phi_y, sig_f=1.0, l_f=1.0, sig_r=0.0):
        return -self.ddk_fun(phi_x, phi_y, sig_f, l_f, sig_r)
