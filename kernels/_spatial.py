"""Small neighbour-list helpers used by the sparse kernel interfaces.

These avoid requiring the parent project's ``utils.spatial_utils`` module.
The padded variant runs on JAX devices; the KD-tree variant runs on the host.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(3,))
def radius_pairs_padded(x, y, cutoff, k_max):
    """Return flattened (row, column, valid) for at most ``k_max`` neighbours/row.

    If a row has more than ``k_max`` neighbours, the nearest ones are retained.
    Padding coordinates are set to zero and must be masked when forming BCOO.
    """
    if k_max < 1:
        raise ValueError("k_max must be positive")
    if y.shape[0] == 0:
        raise ValueError("Y must be nonempty for padded neighbour search")

    count = min(k_max, y.shape[0])
    sqdist = jnp.maximum(
        jnp.sum(x * x, axis=1, keepdims=True)
        + jnp.sum(y * y, axis=1)[None, :]
        - 2.0 * (x @ y.T),
        0.0,
    )
    # top_k selects the nearest points without sorting every row of the full
    # distance matrix; its output width is statically known to the compiler.
    _, cols = jax.lax.top_k(-sqdist, count)
    valid = (cutoff >= 0) & (jnp.take_along_axis(sqdist, cols, axis=1) <= cutoff**2)
    if count < k_max:
        cols = jnp.pad(cols, ((0, 0), (0, k_max - count)))
        valid = jnp.pad(valid, ((0, 0), (0, k_max - count)))
    cols = jnp.where(valid, cols, 0)
    rows = jnp.broadcast_to(jnp.arange(x.shape[0])[:, None], cols.shape)
    return rows.reshape(-1), cols.reshape(-1), valid.reshape(-1)


def radius_pairs_kdtree(x, y, cutoff):
    """Exact radius pairs, dynamically sized on the host (SciPy required)."""
    from scipy.spatial import cKDTree

    x_np, y_np = np.asarray(x), np.asarray(y)
    matches = cKDTree(y_np).query_ball_point(x_np, float(cutoff))
    rows = np.repeat(np.arange(len(matches)), [len(js) for js in matches])
    cols = np.fromiter((j for js in matches for j in js), dtype=np.int32)
    return jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(cols, dtype=jnp.int32)
