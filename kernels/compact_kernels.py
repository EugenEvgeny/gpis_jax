"""Compactly supported tensor-product cubic B-spline kernels."""

import jax
import jax.numpy as jnp
from jax import vmap


def _cubic_bspline(r):
    """Centered cubic B-spline, exactly zero at |r|>=2."""
    a = jnp.abs(r)
    central = (4.0 - 6.0*a**2 + 3.0*a**3) / 6.0
    rim = (2.0-a)**3 / 6.0
    return jnp.where(a < 1.0, central, jnp.where(a < 2.0, rim, 0.0))


def _scaled_product(x, y, bandwidth):
    diff = (x - y) / bandwidth
    return jnp.prod(_cubic_bspline(diff), axis=-1) / bandwidth**x.shape[-1]


@jax.jit
def c_kernel(X1, X2, sigma, length_scale=None):
    """Return the (n,m) tensor-product compact Gram matrix.

    ``length_scale`` (if provided) is used as the bandwidth; otherwise ``sigma``
    is used. Inputs must have shape (n,d), (m,d); bandwidth must be positive.
    """
    bandwidth = sigma if length_scale is None else length_scale
    return vmap(lambda x: vmap(lambda y: _scaled_product(x, y, bandwidth))(X2))(X1)


@jax.jit
def c_kernel_vec(X1, X2, sigma, dim=None):
    """Paired compact kernel, evaluated for corresponding rows of X1 and X2.

    ``dim`` is accepted for compatibility; the input dimension is inferred.
    """
    del dim
    return vmap(lambda x, y: _scaled_product(x, y, sigma))(X1, X2)


def c_kernel_pykeops(X1, X2, sigma, length_scale=None,
                      backend="torch", dtype="float32"):
    """Optional PyKeOps backend (dependencies imported only on use).

    The PyKeOps logarithmic product uses a small floor to support its channel
    reduction; tiny values outside the compact support can therefore remain.
    Use ``c_kernel`` for mathematically exact support and JAX gradients.
    """
    import numpy as np
    if backend not in ("torch", "numpy"):
        raise ValueError("backend must be 'torch' or 'numpy'")
    bandwidth = np.asarray(sigma if length_scale is None else length_scale, dtype=dtype)
    x = np.asarray(X1, dtype=dtype)
    y = np.asarray(X2, dtype=dtype)
    if backend == "torch":
        try:
            import torch
            from pykeops.torch import LazyTensor
        except ImportError as exc:
            raise ImportError("c_kernel_pykeops(backend='torch') requires torch and pykeops") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.as_tensor(x, device=device)
        y = torch.as_tensor(y, device=device)
        bandwidth = torch.as_tensor(bandwidth, device=device)
    else:
        try:
            from pykeops.numpy import LazyTensor
        except ImportError as exc:
            raise ImportError("c_kernel_pykeops(backend='numpy') requires pykeops") from exc

    r = (LazyTensor(x[:, None, :]) - LazyTensor(y[None, :, :])) / bandwidth
    a = r.abs()
    basis = ((a < 1) * (4 - 6*a**2 + 3*a**3)
             + ((a >= 1)*(a < 2))*(2-a)**3) / 6.0
    return (basis + 1e-12).log().sum(2).exp() / bandwidth**x.shape[1]
