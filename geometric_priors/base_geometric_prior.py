"""Pure, differentiable mean functions for JAX Gaussian-process regression.

The *_params methods are side-effect free and safe to call inside a JAX
marginal-likelihood objective. ``set_params`` is deliberately *not* jitted.
Rotations map local coordinates to world coordinates; 3D uses XYZ Euler angles
with R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np


def _points(X, dim):
    X = jnp.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or (dim is not None and X.shape[1] != dim):
        raise ValueError(f"Expected points with shape (n, {dim or 'dim'}), got {X.shape}")
    return X


def _parameters(params, size, name):
    p = jnp.asarray(params).reshape(-1)
    if p.size != size:
        raise ValueError(f"{name} needs {size} parameters, got {p.size}")
    return p


def _rotation(theta, dim):
    """Local-to-world rotation, differentiable w.r.t. theta."""
    if dim == 2:
        a = jnp.asarray(theta).reshape(())
        c, s = jnp.cos(a), jnp.sin(a)
        return jnp.stack([jnp.stack([c, -s]), jnp.stack([s, c])])
    if dim != 3:
        raise ValueError('Rotations currently support dim=2 or dim=3')
    roll, pitch, yaw = jnp.asarray(theta).reshape(3)
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([
        jnp.stack([cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr]),
        jnp.stack([sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr]),
        jnp.stack([-sp, cp*sr, cp*cr]),
    ])


def _split_pose(params, dim):
    p = jnp.asarray(params)
    r = 1 if dim == 2 else 3
    return p[:dim], p[dim:dim+r], p[dim+r:]


def _finite_bounds(n, positive_indices=()):
    lower = np.full((n,), -np.inf)
    upper = np.full((n,), np.inf)
    for i in positive_indices:
        lower[i] = 1e-5
    return lower, upper


class BaseGeometricPrior(ABC):
    """Base protocol for a deterministic scalar GP/GPIS prior mean.

    ``prior_f(X)`` returns (n,); ``prior_grad_f(X)`` returns (n,dim).
    Scalar GPIS training optionally calls ``prior_f_params(X,p)`` and
    ``prior_grad_f_params(X,p)`` while differentiating through p.
    ``parameter_bounds()`` defines per-parameter optimization bounds and
    respects unrestricted translation, orientation, and bias parameters.
    """

    def __init__(self, dim=2, mean=None, theta=None):
        self.dim = int(dim)
        if self.dim not in (2, 3):
            raise ValueError('Geometric priors support 2D and 3D')
        self.mean = (jnp.zeros(self.dim) if mean is None
                     else jnp.asarray(mean).reshape(self.dim))
        angle_size = 1 if self.dim == 2 else 3
        self.theta = (jnp.zeros((angle_size,)) if theta is None
                      else jnp.asarray(theta).reshape(angle_size))

    @property
    def name(self):
        return type(self).__name__

    def __repr__(self):
        return f'{type(self).__name__}(dim={self.dim})'

    def __call__(self, X):
        return self.prior_f(X)

    @property
    def location(self):
        return self.mean

    @property
    def rot_angle(self):
        return self.theta[0] if self.dim == 2 else self.theta

    @property
    def rot_mat(self):
        return _rotation(self.theta, self.dim)

    @property
    def pose(self):
        """Optional jaxlie SE(2)/SE(3) object (not used for GP inference)."""
        try:
            from jaxlie import SE2, SE3, SO2, SO3
        except ImportError as exc:
            raise ImportError('Install jaxlie to access the pose property') from exc
        if self.dim == 2:
            return SE2.from_rotation_and_translation(
                SO2.from_radians(self.theta[0]), self.mean)
        return SE3.from_rotation_and_translation(SO3.from_matrix(self.rot_mat), self.mean)

    def trans_loc(self, X):
        return (_points(X, self.dim) - self.mean) @ self.rot_mat

    def set_rot(self, theta):
        self.theta = jnp.asarray(theta).reshape(1 if self.dim == 2 else 3)
        return self

    def set_translation(self, mean):
        self.mean = jnp.asarray(mean).reshape(self.dim)
        return self

    @abstractmethod
    def get_params(self):
        """Flattened JAX parameter vector."""

    @abstractmethod
    def set_params(self, params):
        """Mutate parameters after optimization, never inside JIT."""

    @abstractmethod
    def prior_f_params(self, X, params):
        """Side-effect-free batched scalar mean, differentiable over params."""

    def prior_grad_f_params(self, X, params):
        X = _points(X, self.dim)
        scalar = lambda x: self.prior_f_params(x[None, :], params)[0]
        return jax.vmap(jax.grad(scalar))(X)

    def prior_f(self, X):
        return self.prior_f_params(X, self.get_params())

    def prior_grad_f(self, X):
        return self.prior_grad_f_params(X, self.get_params())

    def prior_mean(self, X):
        X = _points(X, self.dim)
        return jnp.column_stack([self.prior_f(X), self.prior_grad_f(X)])

    def jax_f(self, X):
        """Legacy batched Jacobian (n,n,d); prefer prior_grad_f (n,d)."""
        X = _points(X, self.dim)
        return jax.jacfwd(self.prior_f)(X)

    def parameter_bounds(self):
        return _finite_bounds(len(self.get_params()))

    def cache_signature(self):
        """Simple immutable state signature for GPIS cache invalidation."""
        return (id(self), np.asarray(self.get_params()).tobytes())

    def prior_f_cov(self, X):
        return jnp.zeros((len(_points(X, self.dim)),), dtype=jnp.asarray(X).dtype)

    def prior_grad_f_cov(self, X):
        return jnp.zeros_like(_points(X, self.dim))

    def prior_intensity(self, X, sensor_location):
        """Cosine incidence and inverse-square distance (finite at zero)."""
        X = _points(X, self.dim)
        direction = jnp.asarray(sensor_location) - X
        distance = jnp.linalg.norm(direction, axis=-1)
        normals = self.prior_grad_f(X)
        denominator = jnp.maximum(distance * jnp.linalg.norm(normals, axis=-1), 1e-12)
        cosine = jnp.sum(direction * normals, axis=-1) / denominator
        return cosine, distance, cosine / jnp.maximum(distance**2, 1e-12)

    def plot_prior_f(self, max_r=2., grid_size=100, vmin=None, vmax=None):
        """Optional 2D contour plot. No plotting dependency at import time."""
        if self.dim != 2:
            raise ValueError('For 3D priors use plot_sdf_plotly')
        import matplotlib.pyplot as plt
        grid = np.linspace(-max_r, max_r, grid_size)
        xx, yy = np.meshgrid(grid + float(self.mean[0]), grid + float(self.mean[1]))
        pts = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=-1))
        vals = np.asarray(self.prior_f(pts)).reshape(xx.shape)
        fig, ax = plt.subplots()
        img = ax.pcolormesh(xx, yy, vals, shading='auto', vmin=vmin, vmax=vmax, cmap='RdBu')
        ax.contour(xx, yy, vals, levels=[0.], colors='black')
        fig.colorbar(img, ax=ax)
        return fig, ax

    def plot_sdf_plotly(self, grid_resolution=30, grid_margin=1., show=False,
                       points=None, **volume_kwargs):
        """Optional Plotly 3D volume display, valid for any 3D prior."""
        if self.dim != 3:
            raise ValueError('Plotly 3D volume needs dim=3')
        import plotly.graph_objects as go
        p = np.asarray(self.mean)
        g = np.linspace(-grid_margin, grid_margin, grid_resolution)
        xx, yy, zz = np.meshgrid(*(g + p[i] for i in range(3)), indexing='ij')
        pts = jnp.asarray(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
        vals = np.asarray(self.prior_f(pts))
        kwargs = {'opacity': .15, 'surface_count': 10, 'colorscale': 'RdBu'}
        kwargs.update(volume_kwargs)
        fig = go.Figure(go.Volume(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), value=vals, **kwargs))
        if points is not None:
            points = np.asarray(points)
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                       mode='markers'))
        if show:
            fig.show()
        return fig
