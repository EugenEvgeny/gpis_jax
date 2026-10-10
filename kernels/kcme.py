from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import jax.numpy as jnp
from jax import Array

from ._linalg import as_2d, regularized_solve


Kernel: TypeAlias = Callable[[Array, Array], Array]


def kcme_coefficients(
    x_train: Array,
    x_query: Array,
    feature_kernel: Kernel,
    *,
    regularization: float = 1e-3,
) -> Array:
    """Return KCME coefficients with shape ``(n_train, n_query)``."""
    x_train = as_2d(x_train)
    x_query = as_2d(x_query)

    gram = feature_kernel(x_train, x_train)
    cross = feature_kernel(x_train, x_query)

    return regularized_solve(gram, cross, regularization)


def conditional_expectation(
    x_train: Array,
    y_train: Array,
    x_query: Array,
    response_values: Array,
    feature_kernel: Kernel,
    *,
    regularization: float = 1e-3,
) -> Array:
    """Estimate ``E[h(Y) | X=x]`` using the empirical KCME.

    ``response_values`` supplies ``h(y_i)`` at each training response and may
    contain one or several test functions as columns. ``y_train`` is accepted
    to make call sites explicit and to validate row alignment.
    """
    x_train = as_2d(x_train)
    y_train = as_2d(y_train)
    values = as_2d(response_values)

    if x_train.shape[0] != y_train.shape[0] or values.shape[0] != y_train.shape[0]:
        raise ValueError(
            "x_train, y_train, and response_values must have equal row counts"
        )

    coefficients = kcme_coefficients(
        x_train,
        x_query,
        feature_kernel,
        regularization=regularization,
    )

    return coefficients.T @ values.astype(coefficients.dtype)


def conditional_probabilities(
    x_train: Array,
    labels: Array,
    x_query: Array,
    feature_kernel: Kernel,
    *,
    classes: Array | None = None,
    regularization: float = 1e-3,
    normalize: bool = True,
    eps: float = 1e-12,
) -> tuple[Array, Array]:
    """Estimate class probabilities using indicator response test functions.

    Negative finite-sample KCME estimates are clipped and each row is
    renormalized by default.
    """
    labels = as_2d(labels)

    if labels.shape[1] != 1:
        raise ValueError("classification labels must be scalar")

    flat = labels[:, 0]

    if classes is None:
        classes = jnp.unique(flat)

    classes = jnp.asarray(classes, dtype=flat.dtype)

    one_hot = (flat[:, None] == classes[None, :]).astype(x_train.dtype)

    probabilities = conditional_expectation(
        x_train,
        labels,
        x_query,
        one_hot,
        feature_kernel,
        regularization=regularization,
    )

    if normalize:
        probabilities = jnp.maximum(probabilities, 0)

        row_sum = probabilities.sum(axis=1, keepdims=True)
        denominator = jnp.maximum(row_sum, eps)

        uniform = jnp.full_like(
            probabilities,
            1.0 / probabilities.shape[1],
        )

        probabilities = jnp.where(
            row_sum > eps,
            probabilities / denominator,
            uniform,
        )

    return probabilities, classes
