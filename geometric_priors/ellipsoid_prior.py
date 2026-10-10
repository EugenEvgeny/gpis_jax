"""Quadratic, oriented ellipsoid implicit-function prior."""
import jax.numpy as jnp

from .base_geometric_prior import (BaseGeometricPrior, _points, _parameters,
                                   _rotation, _split_pose, _finite_bounds)


class EllipsoidPrior(BaseGeometricPrior):
    """f(x)=h/2 * (sum_i (Rᵀ(x-mean))_i² / axis_i² - 1).

    The surface f=0 is an ellipsoid, but f is *not* a true signed distance.
    Unlike the source implementation, 3D rotations belong to the fit vector.
    """
    def __init__(self, h=2., a=1., b=1., c=1., theta=None, mean=None, dim=2):
        super().__init__(dim=dim, mean=mean, theta=theta)
        self.h = jnp.asarray((a + b + (c if dim == 3 else 0)) / dim if h is None else h)
        self.a, self.b, self.c = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)

    @property
    def ellipsoid_params(self):
        return jnp.stack([self.a, self.b] if self.dim == 2 else [self.a, self.b, self.c])

    @property
    def A_s(self):
        return self.rot_mat @ jnp.diag(1. / self.ellipsoid_params**2) @ self.rot_mat.T

    def get_params(self):
        return jnp.concatenate([self.mean, self.theta, self.h.reshape(1), self.ellipsoid_params])

    def set_params(self, params):
        p = _parameters(params, self.dim + (1 if self.dim == 2 else 3) + 1 + self.dim,
                        'EllipsoidPrior')
        self.mean, self.theta, rest = _split_pose(p, self.dim)
        self.h, self.a, self.b = rest[:3]
        if self.dim == 3:
            self.c = rest[3]
        return self

    def parameter_bounds(self):
        n = len(self.get_params())
        start = self.dim + (1 if self.dim == 2 else 3)
        return _finite_bounds(n, range(start, n))

    def prior_f_params(self, X, params):
        X = _points(X, self.dim)
        mean, theta, rest = _split_pose(params, self.dim)
        h, axes = rest[0], rest[1:]
        local = (X - mean) @ _rotation(theta, self.dim)
        return .5*h*(jnp.sum((local / axes)**2, axis=-1) - 1.)

    def prior_grad_f_params(self, X, params):
        X = _points(X, self.dim)
        mean, theta, rest = _split_pose(params, self.dim)
        h, axes = rest[0], rest[1:]
        R = _rotation(theta, self.dim)
        local = (X - mean) @ R
        return h * (local / axes**2) @ R.T
