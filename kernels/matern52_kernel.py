"""Matérn 5/2 kernel with analytic gradient and mixed Hessian."""

from functools import partial

import jax
import jax.numpy as jnp

from .base_kernel import BaseKernel


class Matern52Kernel(BaseKernel):
    def __init__(self, sig_f=1.0, l_f=1.0, dim=3):
        self.dim = int(dim)
        super().__init__([sig_f, l_f])

    @property
    def name(self):
        return "matern52"

    @property
    def sig_f(self):
        return self.params[0]

    @property
    def l_f(self):
        return self.params[1]

    def __repr__(self):
        return f"Matern52Kernel(sig_f={self.sig_f}, l_f={self.l_f})"

    @partial(jax.jit, static_argnums=(0,))
    def k_fun(self, x, y, sig_f=1.0, l_f=1.0):
        diff = (x - y) / l_f
        r2 = jnp.dot(diff, diff)
        scaled = jnp.sqrt(5.0 * jnp.maximum(r2, jnp.finfo(r2.dtype).tiny))
        return sig_f**2 * (1.0 + scaled + (5.0/3.0) * r2) * jnp.exp(-scaled)

    @partial(jax.jit, static_argnums=(0,))
    def dk_fun(self, x, y, sig_f=1.0, l_f=1.0):
        diff = x - y
        scaled = jnp.sqrt(5.0) * jnp.linalg.norm(diff) / l_f
        return -(5.0 * sig_f**2 / (3.0 * l_f**2)) * (
            (1.0 + scaled) * jnp.exp(-scaled)
        ) * diff

    @partial(jax.jit, static_argnums=(0,))
    def dk_y_fun(self, x, y, sig_f=1.0, l_f=1.0):
        return -self.dk_fun(x, y, sig_f, l_f)

    @partial(jax.jit, static_argnums=(0,))
    def ddk_fun(self, x, y, sig_f=1.0, l_f=1.0):
        diff = x - y
        scaled = jnp.sqrt(5.0) * jnp.linalg.norm(diff) / l_f
        coeff = (5.0 * sig_f**2 / (3.0 * l_f**2)) * jnp.exp(-scaled)
        return coeff * (
            (1.0 + scaled) * jnp.eye(x.shape[0], dtype=x.dtype)
            - (5.0 / l_f**2) * jnp.outer(diff, diff)
        )

    @partial(jax.jit, static_argnums=(0,))
    def ddkxx_fun(self, x, y, sig_f=1.0, l_f=1.0):
        return -self.ddk_fun(x, y, sig_f, l_f)

    @partial(jax.jit, static_argnums=(0,))
    def dddkxxx_fun(self, x, y, sig_f=1.0, l_f=1.0):
        """Negative third x derivative, including its zero limit at x=y.

        This avoids NaNs caused by repeatedly autodifferentiating ``norm`` at
        coincident inputs. Matérn 5/2 has a well-defined third derivative there.
        """
        d = x - y
        r = jnp.linalg.norm(d)
        a = jnp.sqrt(5.0) / l_f
        eye = jnp.eye(x.shape[0], dtype=x.dtype)
        sym = (jnp.einsum("ij,k->ijk", eye, d)
               + jnp.einsum("ik,j->ijk", eye, d)
               + jnp.einsum("jk,i->ijk", eye, d))
        inverse_r = jnp.where(r > 0, 1.0 / jnp.where(r > 0, r, 1.0), 0.0)
        cubic = jnp.einsum("i,j,k->ijk", d, d, d)
        return -(sig_f**2 * a**4 / 3.0) * jnp.exp(-a*r) * (sym - a*inverse_r*cubic)
