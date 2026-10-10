"""Finite-basis Gaussian-process radial mean prior, without explicit inverses."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .base_geometric_prior import (BaseGeometricPrior, _points, _parameters,
                                   _rotation, _split_pose, _finite_bounds)


def _solve_factor(L, Y):
    return solve_triangular(L.T, solve_triangular(L, Y, lower=True), lower=False)


class GPRadialPrior(BaseGeometricPrior):
    """Prior mean h*(radius - GP-predicted radius) for star-shaped objects.

    The *inner* radial GP is fitted to centers ``u_f`` and radii ``x_f``.
    The inner kernel is independent of the outer GPIS covariance kernel.
    Its Gram Cholesky factor is cached; changes to the inner kernel's
    parameters refresh the cache on the next mean/gradient evaluation.

    ``get_params`` order: [center (d), angles (1 or 3), height (1), radii (m)].
    The scale ``h`` occurs exactly once, unlike the uploaded code. Inner
    posterior variance can be queried separately using ``prior_f_cov``; it is
    *not* automatically included in GPIS's outer covariance model.
    """
    def __init__(self, kernel, x_f, u_f, h=1., theta=None, mean=None,
                 dim=2, jitter=1e-6, radius_epsilon=1e-8):
        super().__init__(dim=dim, mean=mean, theta=theta)
        self.kernel = kernel
        self.u_f = _points(u_f, dim)
        self.x_f = jnp.asarray(x_f).reshape(-1)
        if len(self.x_f) != len(self.u_f) or not len(self.u_f):
            raise ValueError('x_f must contain one radius per inducing point u_f')
        if hasattr(kernel, 'dim') and kernel.dim != dim:
            raise ValueError('Inner radial kernel dimensionality mismatch')
        if jitter <= 0 or radius_epsilon <= 0:
            raise ValueError('jitter and radius_epsilon must be positive')
        self.h = jnp.asarray(h)
        self.jitter = float(jitter)
        self.radius_epsilon = float(radius_epsilon)
        self._basis_key = None
        self._basis_L = None
        self._ensure_basis()

    def _kernel_signature(self):
        return (np.asarray(self.kernel.params).ravel().tobytes(),
                np.asarray(self.u_f).tobytes(), self.jitter)

    def _ensure_basis(self):
        key = self._kernel_signature()
        if self._basis_key != key:
            K = self.kernel.K(self.u_f, self.u_f)
            K = (K + K.T)*.5
            L = jnp.linalg.cholesky(K + self.jitter*jnp.eye(len(K), dtype=K.dtype))
            if not bool(jnp.all(jnp.isfinite(L))):
                raise np.linalg.LinAlgError('Radial basis Cholesky failed')
            self._basis_L = L
            self._basis_key = key

    @property
    def K_uu(self):
        return self.kernel.K(self.u_f, self.u_f)

    @property
    def K_uu_inv(self):
        """Legacy compatibility only; do not use this for inference."""
        self._ensure_basis()
        return _solve_factor(self._basis_L, jnp.eye(len(self.u_f), dtype=self.u_f.dtype))

    @property
    def state(self):
        return jnp.concatenate([self.mean, self.theta, self.x_f])

    def set_state(self, new_state):
        p = _parameters(new_state, len(self.state), 'GPRadialPrior.state')
        a = self.dim + (1 if self.dim == 2 else 3)
        self.mean, self.theta, self.x_f = p[:self.dim], p[self.dim:a], p[a:]
        return self

    def set_x_f(self, new_x_f):
        self.x_f = jnp.asarray(new_x_f).reshape(len(self.u_f))
        return self

    def get_params(self):
        return jnp.concatenate([self.mean, self.theta, self.h.reshape(1), self.x_f])

    def set_params(self, params):
        n = self.dim+(1 if self.dim == 2 else 3)+1+len(self.u_f)
        p = _parameters(params, n, 'GPRadialPrior')
        self.mean, self.theta, rest = _split_pose(p, self.dim)
        self.h, self.x_f = rest[0], rest[1:]
        return self

    def parameter_bounds(self):
        start = self.dim + (1 if self.dim == 2 else 3)
        return _finite_bounds(len(self.get_params()), [start])

    def cache_signature(self):
        self._ensure_basis()
        return (super().cache_signature(), self._basis_key)

    def _local_and_weights(self, X, params):
        self._ensure_basis()
        X = _points(X, self.dim)
        center, angle, rest = _split_pose(params, self.dim)
        h, x_f = rest[0], rest[1:]
        R = _rotation(angle, self.dim)
        local = (X-center) @ R
        alpha = _solve_factor(self._basis_L, x_f)
        return local, R, h, alpha

    def radial_fn(self, X_L):
        self._ensure_basis()
        X_L = _points(X_L, self.dim)
        return self.kernel.K(X_L, self.u_f) @ _solve_factor(self._basis_L, self.x_f)

    def prior_f_params(self, X, params):
        local, _, h, alpha = self._local_and_weights(X, params)
        radius = jnp.sqrt(jnp.sum(local*local, axis=-1) + self.radius_epsilon**2)
        return h*(radius - self.kernel.K(local, self.u_f) @ alpha)

    def prior_grad_f_params(self, X, params):
        local, R, h, alpha = self._local_and_weights(X, params)
        radius = jnp.sqrt(jnp.sum(local*local, axis=-1, keepdims=True) + self.radius_epsilon**2)
        grad_radial = jnp.einsum('nmd,m->nd', self.kernel.dK(local, self.u_f), alpha)
        return h*(local/radius - grad_radial) @ R.T

    def pred_f_aug(self, Z_L, x_f=None):
        self._ensure_basis()
        Z_L = _points(Z_L, self.dim)
        alpha = _solve_factor(self._basis_L, self.x_f if x_f is None else x_f)
        radius = jnp.sqrt(jnp.sum(Z_L**2, axis=-1) + self.radius_epsilon**2)
        return self.h*(radius-self.kernel.K(Z_L, self.u_f)@alpha)

    def pred_aug_N(self, Z_L, x_f=None):
        self._ensure_basis()
        Z_L = _points(Z_L, self.dim)
        alpha = _solve_factor(self._basis_L, self.x_f if x_f is None else x_f)
        radius = jnp.sqrt(jnp.sum(Z_L**2, axis=-1, keepdims=True) + self.radius_epsilon**2)
        return self.h*(Z_L/radius-jnp.einsum('nmd,m->nd', self.kernel.dK(Z_L,self.u_f),alpha))

    def pred_aug_vec(self, Z_L, x_f=None):
        return jnp.concatenate([self.pred_f_aug(Z_L,x_f),
                                self.pred_aug_N(Z_L,x_f).T.reshape(-1)])

    def pred_f_vals(self, Z, mean, psi, x_f):
        """Legacy unscaled radial residual evaluated with explicit pose/radii."""
        Z = _points(Z, self.dim)
        self._ensure_basis()
        local = (Z-mean) @ _rotation(psi, self.dim)
        radius = jnp.sqrt(jnp.sum(local**2, axis=-1) + self.radius_epsilon**2)
        return radius - self.kernel.K(local, self.u_f) @ _solve_factor(self._basis_L, x_f)

    def _H_mean(self, Z):
        return jax.jacfwd(lambda mean: self.pred_f_vals(Z,mean,self.theta,self.x_f))(self.mean)

    def _H_theta(self, Z):
        return jax.jacfwd(lambda theta: self.pred_f_vals(Z,self.mean,theta,self.x_f))(self.theta)

    def _H_aug(self, Z):
        return jax.jacfwd(lambda radii: self.pred_f_vals(Z,self.mean,self.theta,radii))(self.x_f)

    def _H_state(self, Z):
        return jnp.concatenate([self._H_mean(Z), self._H_theta(Z), self._H_aug(Z)], axis=1)

    def prior_f_cov(self, X):
        """Separate inner radial-GP variance; not part of outer GPIS variance."""
        self._ensure_basis()
        local = self.trans_loc(X)
        k = self.kernel.K(local, self.u_f)
        z = solve_triangular(self._basis_L, k.T, lower=True)
        return jnp.maximum(self.h**2 * (self.kernel.K_vec(local, local) - jnp.sum(z*z,axis=0)), 0.)

    def prior_grad_f_cov(self, X):
        """Marginal inner radial gradient variance in world coordinates."""
        self._ensure_basis()
        local = self.trans_loc(X)
        kdx = self.kernel.dK(local, self.u_f)  # (n,inducing,dim)
        n, m, d = kdx.shape
        z = solve_triangular(self._basis_L, kdx.transpose(1,0,2).reshape(m,n*d), lower=True)
        z = z.T.reshape(n,d,m)
        p = jnp.asarray(self.kernel.params)
        H = jax.vmap(lambda x: self.kernel.ddk_fun(x,x,*p))(local)
        C = H - jnp.einsum('ndm,nem->nde', z, z)
        R = self.rot_mat
        C_world = R @ C @ R.T
        return jnp.maximum(self.h**2 * jnp.diagonal(C_world,axis1=1,axis2=2), 0.)
