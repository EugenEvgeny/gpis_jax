"""Star-shaped implicit surfaces built from user-supplied radial functions."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .base_geometric_prior import (BaseGeometricPrior, _points, _rotation,
                                   _parameters, _split_pose, _finite_bounds)


class RadialPrior(BaseGeometricPrior):
    """f(x) = ||Rᵀ(x-mean)|| - radial_fn(Rᵀ(x-mean)).

    ``radial_fn`` is a JAX-compatible function mapping one local (d,) point
    to a scalar radial length; optionally ``cov_fn`` returns radial variance.
    Center and Euler angles are optimizable, the callable is not.
    """
    def __init__(self, radial_fn, cov_fn=None, theta=None, mean=None,
                 dim=2, radius_epsilon=1e-8):
        super().__init__(dim=dim, mean=mean, theta=theta)
        self.radial_fn = radial_fn
        self.cov_fn = cov_fn
        self.radius_epsilon = float(radius_epsilon)
        if self.radius_epsilon <= 0:
            raise ValueError('radius_epsilon must be positive')

    def get_params(self):
        return jnp.concatenate([self.mean, self.theta])

    def set_params(self, params):
        p = _parameters(params, self.dim+(1 if self.dim==2 else 3), 'RadialPrior')
        self.mean, self.theta, _ = _split_pose(p, self.dim)
        return self

    def parameter_bounds(self):
        return _finite_bounds(len(self.get_params()))

    def _local(self, X, params):
        X = _points(X, self.dim)
        mean, theta, _ = _split_pose(params, self.dim)
        R = _rotation(theta, self.dim)
        return (X-mean) @ R, R

    def prior_f_params(self, X, params):
        local, _ = self._local(X, params)
        radius = jnp.sqrt(jnp.sum(local*local, axis=-1) + self.radius_epsilon**2)
        return radius - jax.vmap(self.radial_fn)(local).reshape(-1)

    def prior_grad_f_params(self, X, params):
        local, R = self._local(X, params)
        radius = jnp.sqrt(jnp.sum(local**2, axis=-1, keepdims=True) + self.radius_epsilon**2)
        grad_radius_fn = jax.vmap(jax.grad(lambda z: jnp.asarray(self.radial_fn(z)).reshape(())))(local)
        return (local/radius - grad_radius_fn) @ R.T

    def prior_f_cov(self, X):
        X = _points(X, self.dim)
        if self.cov_fn is None:
            return jnp.zeros((len(X),), dtype=X.dtype)
        return jax.vmap(self.cov_fn)(self.trans_loc(X)).reshape(-1)
