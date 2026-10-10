"""Median-of-distances bandwidth heuristic shared by median kernels."""

import jax
import jax.numpy as jnp


@jax.jit
def median_bandwidth(X, default_lengthscale):
    """h = median(||x_i-x_j||)/(2 log n), using X only.

    This preserves the original code's bandwidth convention, but uses the
    supplied default if n<2 or all distances vanish. It is treated as a
    stop-gradient hyperparameter rather than differentiated through X.
    """
    n = X.shape[0]
    if n < 2:
        return jnp.asarray(default_lengthscale)
    sq = jnp.maximum(
        jnp.sum(X * X, axis=1, keepdims=True)
        + jnp.sum(X * X, axis=1)[None, :]
        - 2.0 * (X @ X.T),
        0.0,
    )
    h = jnp.median(jnp.sqrt(sq)) / (2.0 * jnp.log(n))
    fallback = jnp.asarray(default_lengthscale)
    return jax.lax.stop_gradient(jnp.where(h > 0, h, fallback))


@jax.jit
def pairwise_l2(x, y=None):
    """Euclidean distances between rows (legacy median-kernel helper)."""
    y = x if y is None else y
    sq = (jnp.sum(x*x, axis=1, keepdims=True)
          + jnp.sum(y*y, axis=1)[None, :] - 2.0*(x @ y.T))
    return jnp.sqrt(jnp.maximum(sq, 0.0))


@jax.jit
def pairwise_group_dist(X1, X2):
    """Squared log-map distances for vectorized Lie-group PyTrees.

    Kept for compatibility with callers importing it from the two median
    kernel modules. Requires ``inverse``, group multiplication and ``log``.
    """
    def single(x, y):
        tangent = (x.inverse() @ y).log()
        return jnp.vdot(tangent, tangent)
    return jax.vmap(lambda x: jax.vmap(lambda y: single(x,y))(X2))(X1)
