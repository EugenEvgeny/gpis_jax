"""JAX-native linear algebra operations shared by kernel estimators."""

import jax.numpy as jnp


def as_2d(value):
    """Represent scalar, vector or matrix data as an (n, d) matrix."""
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value[:, None]
    if value.ndim == 2:
        return value
    raise ValueError("Expected at most two dimensions")


def regularized_solve(gram, rhs, regularization=1e-3):
    """Solve ``(gram + regularization * I) @ solution = rhs``.

    ``gram`` is assumed positive semidefinite. Using ``solve`` instead of an
    explicit inverse is both faster and more numerically stable.
    """
    eye = jnp.eye(gram.shape[0], dtype=gram.dtype)
    return jnp.linalg.solve(gram + jnp.asarray(regularization) * eye, rhs)
