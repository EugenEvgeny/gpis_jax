"""Differentiable, bounded plane/line-segment and multi-patch GP priors.

The original PlanePatchPrior's ``prior_f`` and ``prior_grad_f`` represented
*different* fields. Here they are exact derivatives of the same smooth windowed
signed-plane function (not an exact distance field):

    f(x) = h * w * exp(-sqrt(w²+eps²)/cutoff) * product_j gate_j(u_j),
    gate_j(u) = sigmoid((a_j²-u²)/(2*a_j*edge_width)).

Local coordinate w is normal to the supporting plane, u are tangential. The
smooth gate removes jumps at the finite-patch boundary; gradients near the
patch interior's zero set align with its normal. The value is zero outside the
patch on the infinite supporting plane, a limitation of an orientation-aware
zero-thickness surface prior (not a closed-surface SDF).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid

from .base_geometric_prior import (BaseGeometricPrior, _points, _parameters,
                                   _rotation, _split_pose, _finite_bounds)


class PlanePatchPrior(BaseGeometricPrior):
    def __init__(self, h=.5, a=1., b=1., c=1., cutoff_dist=1., theta=None,
                 mean=None, dim=2, edge_width=.05):
        super().__init__(dim=dim, mean=mean, theta=theta)
        if cutoff_dist <= 0 or edge_width <= 0:
            raise ValueError('cutoff_dist and edge_width must be positive')
        self.h = jnp.asarray((a + b + c)/dim if h is None else h)
        self.a, self.b = jnp.asarray(a), jnp.asarray(b)
        self.c = jnp.asarray(c)  # signature compatibility; unused in geometry
        self.cutoff_dist = float(cutoff_dist)
        self.edge_width = float(edge_width)

    @property
    def half_extents(self):
        return jnp.stack([self.a] if self.dim == 2 else [self.a, self.b])

    @property
    def plane_box(self):
        return self.half_extents

    @property
    def local_corners(self):
        if self.dim == 2:
            return jnp.stack([jnp.stack([-self.a, jnp.zeros_like(self.a)]),
                              jnp.stack([self.a, jnp.zeros_like(self.a)])])
        return jnp.stack([
            jnp.stack([-self.a,-self.b,jnp.zeros_like(self.a)]),
            jnp.stack([self.a,-self.b,jnp.zeros_like(self.a)]),
            jnp.stack([self.a,self.b,jnp.zeros_like(self.a)]),
            jnp.stack([-self.a,self.b,jnp.zeros_like(self.a)])
        ])

    @property
    def corner_locations(self):
        return self.local_corners @ self.rot_mat.T + self.mean

    def get_params(self):
        return jnp.concatenate([self.mean, self.theta, self.h.reshape(1), self.half_extents])

    def set_params(self, params):
        size = self.dim + (1 if self.dim == 2 else 3) + self.dim
        p = _parameters(params, size, 'PlanePatchPrior')
        self.mean, self.theta, rest = _split_pose(p, self.dim)
        self.h, self.a = rest[0], rest[1]
        if self.dim == 3:
            self.b = rest[2]
        return self

    def parameter_bounds(self):
        start = self.dim + (1 if self.dim == 2 else 3)
        return _finite_bounds(len(self.get_params()), range(start, len(self.get_params())))

    def _values_and_gradients(self, X, params):
        X = _points(X, self.dim)
        mean, theta, rest = _split_pose(params, self.dim)
        h, extents = rest[0], rest[1:]
        R = _rotation(theta, self.dim)
        local = (X - mean) @ R
        u, w = local[:, :-1], local[:, -1]
        gates = sigmoid((extents*extents - u*u)/(2*extents*self.edge_width))
        weight = jnp.prod(gates, axis=-1)
        radius = jnp.sqrt(w*w+1e-12)
        envelope = jnp.exp(-radius/self.cutoff_dist)
        f = h * w * envelope * weight
        d_tangent = f[:, None] * (-(1-gates) * u / (extents*self.edge_width))
        d_normal = h * envelope * weight * (1 - (w*w)/(self.cutoff_dist*radius))
        grad_local = jnp.concatenate([d_tangent, d_normal[:, None]], axis=-1)
        return f, grad_local @ R.T

    def prior_f_params(self, X, params):
        return self._values_and_gradients(X, params)[0]

    def prior_grad_f_params(self, X, params):
        return self._values_and_gradients(X, params)[1]

    def plot_surface(self, ax=None, points=None, color='steelblue', alpha=.4):
        """Optional 2D segment / 3D rectangular-patch Matplotlib plot."""
        import matplotlib.pyplot as plt
        corners = np.asarray(self.corner_locations)
        if self.dim == 2:
            if ax is None:
                _, ax = plt.subplots()
            ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=3)
            ax.set_aspect('equal')
            if points is not None:
                p = np.asarray(points)
                ax.scatter(p[:, 0], p[:, 1], s=6)
        else:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            if ax is None:
                ax = plt.figure().add_subplot(projection='3d')
            ax.add_collection3d(Poly3DCollection([corners], alpha=alpha,
                                               facecolors=color, edgecolors='black'))
            ax.auto_scale_xyz(corners[:,0], corners[:,1], corners[:,2])
            if points is not None:
                p = np.asarray(points)
                ax.scatter(p[:,0],p[:,1],p[:,2],s=6)
        return ax

    @classmethod
    def from_points(cls, points, dim=3, h=1., viewpoint=None,
                    robust_percentile=None, *, cutoff_dist=1., edge_width=.05):
        """Estimate a line/rectangular patch via PCA in 2D/3D.

        ``viewpoint`` selects the positive normal orientation. The 3D rotation
        is extracted analytically (no iterative nonlinear pose optimizer).
        """
        P = np.asarray(points, dtype=float)
        if P.ndim != 2 or P.shape[1] != dim or len(P) < dim:
            raise ValueError(f'points must have shape (n, {dim}), n >= {dim}')
        if robust_percentile is not None and not 0 <= robust_percentile < 50:
            raise ValueError('robust_percentile must be in [0,50)')
        center = P.mean(axis=0)
        Q = P-center
        _, vectors = np.linalg.eigh(Q.T @ Q / len(P))
        if dim == 2:
            along = vectors[:, 1]
            R = np.column_stack([along, np.array([-along[1], along[0]])])
        else:
            x, y = vectors[:, 2], vectors[:, 1]
            R = np.column_stack([x, y, np.cross(x, y)])
        if viewpoint is not None and (R[:, -1] @ (np.asarray(viewpoint)-center)) < 0:
            # Flip tangent and normal together to remain a proper rotation.
            R[:, 0] *= -1
            R[:, -1] *= -1
        local = Q @ R
        planar = local[:, :-1]
        if robust_percentile is None:
            lo, hi = planar.min(axis=0), planar.max(axis=0)
        else:
            p = float(robust_percentile)
            lo, hi = np.percentile(planar, p, axis=0), np.percentile(planar, 100-p, axis=0)
        extents = np.maximum((hi-lo)/2, 1e-5)
        center = center + R[:, :-1] @ ((hi+lo)/2)
        theta = cls._theta_for_rotation(R, dim)
        return cls(h=h, a=float(extents[0]), b=(float(extents[1]) if dim == 3 else 1.),
                   mean=center, theta=theta, dim=dim, cutoff_dist=cutoff_dist,
                   edge_width=edge_width)

    @staticmethod
    def _theta_for_rotation(R, dim, x0=None):
        R = np.asarray(R)
        if dim == 2:
            return np.arctan2(R[1, 0], R[0, 0])
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(R).as_euler('xyz')


class MultiPlanePatchPrior(BaseGeometricPrior):
    """Weighted mean of finite-patch mean functions.

    Does *not* form the union/nearest patch SDF. All child parameters can be
    optimized as a concatenated vector; weights are fixed. ``mean``/``theta``
    arguments are accepted for compatibility but are not applied to children.
    """
    def __init__(self, plane_priors, mean=None, theta=None, weights=None):
        if not plane_priors:
            raise ValueError('At least one patch is required')
        self.plane_priors = tuple(plane_priors)
        dim = self.plane_priors[0].dim
        if any(p.dim != dim for p in self.plane_priors):
            raise ValueError('All patches must have the same input dimension')
        super().__init__(dim=dim, mean=mean, theta=theta)
        self.num_priors = len(plane_priors)
        weights = jnp.ones((self.num_priors,)) if weights is None else jnp.asarray(weights)
        if weights.shape != (self.num_priors,) or np.any(np.asarray(weights) < 0) or not bool(jnp.sum(weights) > 0):
            raise ValueError('weights must be nonnegative and contain a positive entry')
        self.weights = weights / jnp.sum(weights)

    def get_params(self):
        return jnp.concatenate([p.get_params() for p in self.plane_priors])

    def set_params(self, params):
        p = _parameters(params, len(self.get_params()), 'MultiPlanePatchPrior')
        offset = 0
        for patch in self.plane_priors:
            n = len(patch.get_params())
            patch.set_params(p[offset:offset+n])
            offset += n
        return self

    def parameter_bounds(self):
        bnds = [p.parameter_bounds() for p in self.plane_priors]
        return tuple(np.concatenate([b[i] for b in bnds]) for i in (0, 1))

    def _weighted(self, X, params, name):
        offset = 0
        terms = []
        for patch in self.plane_priors:
            n = len(patch.get_params())
            terms.append(getattr(patch, name)(X, params[offset:offset+n]))
            offset += n
        return jnp.tensordot(self.weights, jnp.stack(terms, axis=0), axes=1)

    def prior_f_params(self, X, params):
        return self._weighted(X, params, 'prior_f_params')

    def prior_grad_f_params(self, X, params):
        return self._weighted(X, params, 'prior_grad_f_params')

    @property
    def corner_locations(self):
        return jnp.concatenate([p.corner_locations for p in self.plane_priors])

    def cache_signature(self):
        return (super().cache_signature(), np.asarray(self.weights).tobytes())


def points_in_bbox_from_corners(points, corners):
    """Axis-aligned inclusion test, for legacy callers (2D and 3D)."""
    pts, cs = jnp.asarray(points), jnp.asarray(corners)
    return jnp.all((pts >= cs.min(axis=0)) & (pts <= cs.max(axis=0)), axis=-1)
