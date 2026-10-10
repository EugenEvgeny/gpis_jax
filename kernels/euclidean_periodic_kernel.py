"""Periodic covariance between directions represented by vectors in R²."""

from functools import partial

import jax
import jax.numpy as jnp

from .base_kernel import BaseKernel


class EuclideanPeriodicKernel(BaseKernel):
    """Angle-based periodic covariance; undefined derivatives at zero vectors."""

    def __init__(self, sig_f=1.0, l_f=1.0, sig_r=0.0, d=1, dim=2):
        if dim != 2:
            raise ValueError("EuclideanPeriodicKernel requires dim=2")
        self.dim = 2
        self.d = d
        self.psi_x = 0
        self.psi_y = 0
        super().__init__([sig_f, l_f, sig_r])

    @property
    def name(self):
        return "EuclideanPeriodicKernel"

    @property
    def sig_f(self):
        return self.params[0]

    @property
    def l_f(self):
        return self.params[1]

    def __repr__(self):
        return f"EuclideanPeriodicKernel(sig_f={self.sig_f}, l_f={self.l_f}, d={self.d})"

    def set_psi_x(self, psi_x):
        self.psi_x = psi_x  # backwards-compatible, unused by the kernel

    def set_psi_y(self, psi_y):
        self.psi_y = psi_y

    @partial(jax.jit, static_argnums=(0,))
    def k_fun(self, x, y, sig_f=1.0, l_f=1.0, sig_r=0.0):
        delta = jnp.arctan2(x[1], x[0]) - jnp.arctan2(y[1], y[0])
        return sig_f**2 * jnp.exp(-2.0 * jnp.sin(self.d * delta / 2.0)**2 / l_f**2) + sig_r**2


@jax.jit
def u(phi):
    """Unit vector at angle phi (shape (2,1), for legacy callers)."""
    return jnp.stack([jnp.cos(phi), jnp.sin(phi)]).reshape(2, -1)


@jax.jit
def jax_angle(x, y):
    """Clockwise angular distance (original [0,2π] convention)."""
    x, y = jnp.asarray(x).reshape(2), jnp.asarray(y).reshape(2)
    angle = jnp.arctan2(x[0]*y[1] - x[1]*y[0], jnp.dot(x, y))
    return jnp.mod(-angle, 2*jnp.pi)


@jax.jit
def k_fun(x, y, d, sig_f=1.0, l_f=1.0, sig_r=0.0):
    delta = jnp.arctan2(x[1], x[0]) - jnp.arctan2(y[1], y[0])
    return sig_f**2 * jnp.exp(-2.0 * jnp.sin(d*delta/2.0)**2 / l_f**2) + sig_r**2


def k_grad_hess_analytic(x, y, d, sig_f=1.0, l_f=1.0, sig_r=0.0, eps=1e-12):
    """Return k, grad_x k, Hess_x k for nonzero 2D input vectors."""
    delta = jnp.arctan2(x[1], x[0]) - jnp.arctan2(y[1], y[0])
    sin_d, cos_d = jnp.sin(d*delta), jnp.cos(d*delta)
    varying = sig_f**2 * jnp.exp(-2.0 * jnp.sin(d*delta/2)**2 / l_f**2)
    k = varying + sig_r**2
    k_phi = -d * sin_d * varying / l_f**2
    k_phiphi = d**2 * varying * (sin_d**2 / l_f**4 - cos_d / l_f**2)
    x1, x2 = x
    r2 = jnp.maximum(jnp.dot(x, x), eps)
    grad_phi = jnp.stack([-x2, x1]) / r2
    hess_phi = jnp.array([[2*x1*x2, x2*x2-x1*x1],
                          [x2*x2-x1*x1, -2*x1*x2]]) / r2**2
    return k, k_phi*grad_phi, k_phiphi*jnp.outer(grad_phi, grad_phi) + k_phi*hess_phi


def k_grad_hess_autodiff(x, y, d, sig_f=1.0, l_f=1.0, sig_r=0.0):
    f = lambda xi: k_fun(xi, y, d, sig_f, l_f, sig_r)
    return f(x), jax.grad(f)(x), jax.hessian(f)(x)
