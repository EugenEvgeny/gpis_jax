"""Squared-exponential (Gaussian/RBF) kernel and analytic derivatives."""

from functools import partial

import jax
import jax.numpy as jnp

from .base_kernel import BaseKernel


class SEKernel(BaseKernel):
    """k(x,y) = sig_f**2 exp(-||x-y||**2/(2*l_f**2))."""

    def __init__(self, sig_f=1.0, l_f=1.0, dim=2):
        self.dim = int(dim)
        super().__init__([sig_f, l_f])

    @property
    def name(self):
        return "squared_exponential"

    @property
    def sig_f(self):
        return self.params[0]

    @property
    def l_f(self):
        return self.params[1]

    def __repr__(self):
        return f"SEKernel(sig_f={self.sig_f}, l_f={self.l_f})"

    @partial(jax.jit, static_argnums=(0,))
    def k_fun(self, x, y, sig_f, l_f):
        r = x - y
        return sig_f**2 * jnp.exp(-0.5 * jnp.dot(r, r) / l_f**2)

    @partial(jax.jit, static_argnums=(0,))
    def dk_fun(self, x, y, sig_f, l_f):
        return -(x - y) * (self.k_fun(x, y, sig_f, l_f) / l_f**2)

    @partial(jax.jit, static_argnums=(0,))
    def dk_y_fun(self, x, y, sig_f, l_f):
        return -self.dk_fun(x, y, sig_f, l_f)

    @partial(jax.jit, static_argnums=(0,))
    def ddk_fun(self, x, y, sig_f, l_f):
        r = x - y
        inv_l2 = 1.0 / l_f**2
        k = self.k_fun(x, y, sig_f, l_f)
        return (inv_l2 * jnp.eye(x.shape[0], dtype=x.dtype)
                - inv_l2**2 * jnp.outer(r, r)) * k

    @partial(jax.jit, static_argnums=(0,))
    def ddkxx_fun(self, x, y, sig_f, l_f):
        return -self.ddk_fun(x, y, sig_f, l_f)

    @partial(jax.jit, static_argnums=(0,))
    def dddkxxx_fun(self, x, y, sig_f, l_f):
        """Negative third x derivative (legacy BaseKernel convention)."""
        r = x - y
        inv_l2 = 1.0 / l_f**2
        eye = jnp.eye(x.shape[0], dtype=x.dtype)
        sym = (jnp.einsum("ij,k->ijk", eye, r)
               + jnp.einsum("ik,j->ijk", eye, r)
               + jnp.einsum("jk,i->ijk", eye, r))
        return self.k_fun(x, y, sig_f, l_f) * (
            inv_l2**3 * jnp.einsum("i,j,k->ijk", r, r, r)
            - inv_l2**2 * sym
        )

    def K_mat_single(self, x, y, sig_f=None, l_f=None):
        """(1+d)x(1+d) Hermite covariance, matching K_mat_comb layout."""
        sig_f = self.sig_f if sig_f is None else sig_f
        l_f = self.l_f if l_f is None else l_f
        x, y = jnp.asarray(x).reshape(-1), jnp.asarray(y).reshape(-1)
        k = self.k_fun(x, y, sig_f, l_f)
        grad_x = self.dk_fun(x, y, sig_f, l_f)
        grad_y = -grad_x
        mixed = self.ddk_fun(x, y, sig_f, l_f)
        return jnp.concatenate(
            [jnp.concatenate([k.reshape(1, 1), grad_y[None, :]], axis=1),
             jnp.concatenate([grad_x[:, None], mixed], axis=1)], axis=0
        )


RBFKernel = SEKernel
