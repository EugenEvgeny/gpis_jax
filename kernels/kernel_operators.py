"""RKHS statistics, mean embeddings and kernel Stein discrepancy.

The public free functions are also installed as methods on ``BaseKernel`` for
backwards compatibility. All matrix work stays JAX-native and differentiable.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, vmap

from ._linalg import as_2d, regularized_solve
from .base_kernel import BaseKernel


def _nonnegative(value: Array, tolerance: float = 1e-10) -> Array:
    del tolerance
    return jnp.maximum(value, 0.0)


def amcmd2(X1, Y1, X2, Y2, kernel_x: BaseKernel, kernel_y: BaseKernel,
           *, weighting_x=None, regularization_reference=1e-3,
           regularization_comparison=None):
    """Squared average maximum conditional mean discrepancy (AMCMD²)."""
    X1, X2 = as_2d(X1), as_2d(X2)
    x0 = X1 if weighting_x is None else as_2d(weighting_x)
    reg2 = regularization_reference if regularization_comparison is None else regularization_comparison
    a = regularized_solve(kernel_x(X1, X1), kernel_x(X1, x0), regularization_reference)
    b = regularized_solve(kernel_x(X2, X2), kernel_x(X2, x0), reg2)
    l11 = kernel_y(Y1, Y1)
    l22 = kernel_y(Y2, Y2)
    l12 = kernel_y(Y1, Y2)
    value = (jnp.sum(a * (l11 @ a)) + jnp.sum(b * (l22 @ b))
             - 2.0 * jnp.sum(a * (l12 @ b))) / len(x0)
    return _nonnegative(value)


def amcmd(*args: Any, **kwargs: Any) -> Array:
    return jnp.sqrt(amcmd2(*args, **kwargs))


def mcmd2(X1, Y1, X2, Y2, query_x, kernel_x: BaseKernel, kernel_y: BaseKernel,
          *, regularization_reference=1e-3, regularization_comparison=None):
    """Squared conditional-mean discrepancy at each query location."""
    X1, X2, query_x = as_2d(X1), as_2d(X2), as_2d(query_x)
    reg2 = regularization_reference if regularization_comparison is None else regularization_comparison
    a = regularized_solve(kernel_x(X1, X1), kernel_x(X1, query_x), regularization_reference)
    b = regularized_solve(kernel_x(X2, X2), kernel_x(X2, query_x), reg2)
    l11, l22, l12 = kernel_y(Y1, Y1), kernel_y(Y2, Y2), kernel_y(Y1, Y2)
    return _nonnegative(
        jnp.sum(a * (l11 @ a), axis=0) + jnp.sum(b * (l22 @ b), axis=0)
        - 2.0 * jnp.sum(a * (l12 @ b), axis=0)
    )


def mcmd(*args: Any, **kwargs: Any) -> Array:
    return jnp.sqrt(mcmd2(*args, **kwargs))


def jmmd2(X1, Y1, X2, Y2, kernel_x: BaseKernel, kernel_y: BaseKernel):
    """Biased empirical squared joint maximum mean discrepancy."""
    k11, k22, k12 = kernel_x(X1, X1), kernel_x(X2, X2), kernel_x(X1, X2)
    l11, l22, l12 = kernel_y(Y1, Y1), kernel_y(Y2, Y2), kernel_y(Y1, Y2)
    return _nonnegative((k11 * l11).mean() + (k22 * l22).mean() - 2.0 * (k12 * l12).mean())


def jmmd(*args: Any, **kwargs: Any) -> Array:
    return jnp.sqrt(jmmd2(*args, **kwargs))


def embed_pdf(self, pdf_fn, X_grid, X_obs, params=None, cutoff=None):
    """Approximate a 1D PDF's kernel mean embedding on a uniform grid."""
    grid = jnp.asarray(X_grid).reshape(-1)
    if grid.shape[0] < 2:
        raise ValueError("X_grid must contain at least two points")
    pvals = jnp.asarray(pdf_fn(grid)).reshape(-1)
    step = (grid[-1] - grid[0]) / (grid.shape[0] - 1)
    return self.K(as_2d(X_obs), grid[:, None], params, cutoff=cutoff) @ pvals * step


def kme_eval(self, X, Z, weights=None, params=None, cutoff=None):
    """Empirical kernel mean embedding of X evaluated at Z."""
    Kxz = self.K(X, Z, params, cutoff=cutoff)
    return Kxz.mean(axis=0) if weights is None else jnp.asarray(weights) @ Kxz


def kme_inner(self, X, Y, wX=None, wY=None, params=None, cutoff=None):
    """RKHS inner product of two (optionally weighted) empirical means."""
    Kxy = self.K(X, Y, params, cutoff=cutoff)
    n, m = Kxy.shape
    if wX is None and wY is None:
        return Kxy.mean()
    wx = jnp.full((n,), 1.0/n) if wX is None else jnp.asarray(wX)
    wy = jnp.full((m,), 1.0/m) if wY is None else jnp.asarray(wY)
    return wx @ Kxy @ wy


def mmd2(self, X, Y, params=None, cutoff=None, unbiased=False, wX=None, wY=None):
    """MMD²; unbiased U-statistic or biased/weighted V-statistic."""
    if not unbiased:
        return (self.kme_inner(X, X, wX, wX, params, cutoff)
                + self.kme_inner(Y, Y, wY, wY, params, cutoff)
                - 2.0 * self.kme_inner(X, Y, wX, wY, params, cutoff))
    if wX is not None or wY is not None:
        raise ValueError("Unbiased MMD requires uniform weights")
    Kxx = self.K(X, X, params, cutoff=cutoff)
    Kyy = self.K(Y, Y, params, cutoff=cutoff)
    Kxy = self.K(X, Y, params, cutoff=cutoff)
    n, m = Kxx.shape[0], Kyy.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Unbiased MMD requires at least two samples from each distribution")
    return ((Kxx.sum() - jnp.trace(Kxx)) / (n*(n-1))
            + (Kyy.sum() - jnp.trace(Kyy)) / (m*(m-1))
            - 2.0*Kxy.mean())


def mmd(self, X, Y, params=None, cutoff=None, unbiased=True, wX=None, wY=None):
    return jnp.sqrt(_nonnegative(self.mmd2(X, Y, params, cutoff, unbiased, wX, wY)))


def cross_covariance(self, X, Y, kernel_y=None, params=None, params_y=None,
                     cutoff=None, centered=True, return_grams=False):
    """Squared Hilbert-Schmidt norm of empirical RKHS cross-covariance.

    Uses O(n²) row/column mean centering instead of O(n³) H@K@H products.
    """
    kernel_y = self if kernel_y is None else kernel_y
    K = self.K(X, X, params, cutoff=cutoff)
    L = kernel_y.K(Y, Y, params_y, cutoff=cutoff)
    if K.shape != L.shape:
        raise ValueError("X and Y must have the same number of paired rows")
    if centered:
        K = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
        L = L - L.mean(axis=0, keepdims=True) - L.mean(axis=1, keepdims=True) + L.mean()
    hs2 = jnp.mean(K * L)
    return (hs2, K, L) if return_grams else hs2


def stein_fn(self, X, Y, scores_x, scores_y, params=None,
             assume_symmetric=False, cutoff=None):
    """Pairwise Langevin Stein kernel matrix; accepts rectangular X and Y.

    Only the trace of the mixed Hessian is computed; there is no n*m*d*d
    intermediate array. ``scores_x``/``scores_y`` have shapes (n,d)/(m,d).
    """
    X, Y = self._points(X, Y)
    scores_x = jnp.asarray(scores_x).reshape(-1, self.dim)
    scores_y = jnp.asarray(scores_y).reshape(-1, self.dim)
    params = self._resolve_params(X, params)
    gram = self.K(X, Y, params, cutoff=cutoff)
    gx = self.dK(X, Y, params, cutoff=cutoff)
    gy = self.dK_y(X, Y, params, assume_symmetric=assume_symmetric, cutoff=cutoff)
    mixed_trace = vmap(lambda x: vmap(lambda y: jnp.trace(self.ddk_fun(x, y, *params)))(Y))(X)
    if cutoff is not None:
        mixed_trace = jnp.where(self._cutoff_mask(X, Y, cutoff), mixed_trace, 0.0)
    return ((scores_x @ scores_y.T) * gram
            + jnp.einsum("id,ijd->ij", scores_x, gy)
            + jnp.einsum("jd,ijd->ij", scores_y, gx)
            + mixed_trace)


def ksd_matrix(self, X, Y, score_fn, biased=True, assume_symmetric=True, params=None):
    """Return the Stein kernel matrix; ``biased`` is kept for API compatibility."""
    del biased
    return self.stein_fn(X, Y, score_fn(X), score_fn(Y), params,
                         assume_symmetric=assume_symmetric)


def ksd2(self, X, score, params=None, cutoff=None, unbiased=True):
    """Kernel Stein discrepancy; ``score`` maps one point to one score vector."""
    X, _ = self._points(X)
    n = X.shape[0]
    if unbiased and n < 2:
        raise ValueError("Unbiased KSD requires at least two samples")
    scores = vmap(score)(X)
    U = self.stein_fn(X, X, scores, scores, params, cutoff=cutoff)
    return ((U.sum()-jnp.trace(U)) / (n*(n-1))) if unbiased else U.mean()


def ksd2_blocked(self, X, scores, block=512, biased=True,
                 assume_symmetric=True, params=None):
    """Compute scalar KSD² using O(block*n) memory and a device-side loop.

    ``scores`` must be an (n,d) array. Padded tail rows are masked, including
    their diagonal contributions, so n need not be divisible by ``block``.
    """
    X, _ = self._points(X)
    scores = jnp.asarray(scores).reshape(-1, self.dim)
    if scores.shape != X.shape:
        raise ValueError("scores must have shape (n, dim), matching X")
    if block < 1:
        raise ValueError("block must be positive")
    if X.shape[0] == 0 or (not biased and X.shape[0] < 2):
        raise ValueError("KSD requires >=1 sample (>=2 for unbiased)")
    p = self._resolve_params(X, params)
    return _ksd2_blocked_impl(self, X, scores, p, block, biased, assume_symmetric)


@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def _ksd2_blocked_impl(kernel, X, scores, params, block, biased, assume_symmetric):
    n, d = X.shape
    pad = (-n) % block
    padded_x = jnp.pad(X, ((0, pad), (0, 0)))
    padded_s = jnp.pad(scores, ((0, pad), (0, 0)))
    num_blocks = (n + pad)//block

    def run(idx, carry):
        total, diagonal = carry
        start = idx * block
        Xi = jax.lax.dynamic_slice(padded_x, (start, 0), (block, d))
        Si = jax.lax.dynamic_slice(padded_s, (start, 0), (block, d))
        U = kernel.stein_fn(Xi, X, Si, scores, params, assume_symmetric=assume_symmetric)
        active = start + jnp.arange(block) < n
        total = total + jnp.sum(jnp.where(active[:, None], U, 0))
        cols = jnp.minimum(start + jnp.arange(block), n-1)
        diagonal = diagonal + jnp.sum(jnp.where(active, U[jnp.arange(block), cols], 0))
        return total, diagonal

    zeros = (jnp.zeros((), dtype=X.dtype), jnp.zeros((), dtype=X.dtype))
    total, diagonal = jax.lax.fori_loop(0, num_blocks, run, zeros)
    return total/(n*n) if biased else (total-diagonal)/(n*(n-1))


# API compatibility: install free functions on BaseKernel as methods.
for _method in (embed_pdf, kme_eval, kme_inner, mmd2, mmd,
                cross_covariance, stein_fn, ksd_matrix, ksd2, ksd2_blocked):
    setattr(BaseKernel, _method.__name__, _method)
