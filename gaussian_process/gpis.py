"""GP implicit surfaces: scalar and normal-derivative observations.

Geometry, normal estimation, registration, and surface-specific statistics are
kept here. Inference, numerically stable Cholesky conditioning, posterior
covariances, and kernel-hyperparameter optimization are inherited from GP.
Optional visualization and geometry packages are imported only when needed.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .gp import GP, Observations, _as_points, _broadcast_variance, _chol_solve
from .priors import ZeroPrior
from .statistics import normal_logpdf, normal_interval_logprob


def estimate_normals(points, k=12):
    """PCA normals oriented away from the sample centroid.

    The orientation heuristic is appropriate mainly for star-shaped surfaces;
    callers should supply consistently oriented normals for other geometries.
    """
    P = _as_points(points)
    n, d = P.shape
    if n < d+1:
        raise ValueError(f"At least {d+1} points are needed for PCA normals")
    k = min(max(int(k), d+1), n)
    distances = jnp.sum((P[:, None, :] - P[None, :, :])**2, axis=-1)
    ids = jnp.argsort(distances, axis=-1)[:, :k]
    neighbors = P[ids]
    centered = neighbors - jnp.mean(neighbors, axis=1, keepdims=True)
    covariance = jnp.einsum('nki,nkj->nij', centered, centered) / k
    _, eigvecs = jnp.linalg.eigh(covariance)
    normals = eigvecs[:, :, 0]
    direction = jnp.sum(normals * (P - jnp.mean(P, axis=0)), axis=1)
    normals = jnp.where(direction[:, None] < 0, -normals, normals)
    return normals / jnp.maximum(jnp.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


class GPIS(GP):
    """Gaussian-process implicit surface with derivative constraints.

    Surface points constrain f(x)=0; supplied normals constrain ∇f(x)=n(x).
    Optionally, displaced points constrain f(x ± d*n(x))=±f_aug_val.
    This is a GP with *linear derivative observations*, NOT a separate
    approximate inference model. All observations enter a single covariance.

    ``sig_x`` and ``sig_n`` are observation noise variances (not std. devs).
    ``geo_prior`` must expose prior_f(X), prior_grad_f(X). ``ZeroPrior`` is
    used by default. Priors with tunable parameters may additionally implement
    prior_f_params(X, params) and prior_grad_f_params(X, params).
    """

    def __init__(self, kernel, X_train, N_train=None, geo_prior=None, *,
                 sig_x=1e-2, sig_n=1e-2, grid_size=50, d=0.05, X_obs=None,
                 augment=True, f_aug_val=None, normal_neighbors=12,
                 jitter=1e-6):
        X_train = _as_points(X_train)
        n, dim = X_train.shape
        if dim not in (2, 3):
            raise ValueError("GPIS presently supports 2D and 3D point sets")
        super().__init__(kernel, X_train, jnp.zeros((n,), dtype=X_train.dtype),
                         noise=sig_x, jitter=jitter)
        self.N_train = (estimate_normals(X_train, normal_neighbors) if N_train is None
                        else _as_points(N_train, self.dim))
        if self.N_train.shape != X_train.shape:
            raise ValueError("N_train must match X_train shape")
        self.sig_n = sig_n
        self.geo_prior = geo_prior if geo_prior is not None else ZeroPrior(dim=dim)
        if getattr(self.geo_prior, 'dim', dim) not in (None, dim):
            raise ValueError('Geometric prior dimensionality does not match GPIS')
        self.augment = bool(augment)
        self.f_aug_d = float(d)
        self._auto_aug_value = f_aug_val is None
        # For unit SDF normals, f(x +/- d*n) should approximately be +/-d,
        # not +/-1. Override f_aug_val to recover the original convention.
        self.f_aug_val = self.f_aug_d if f_aug_val is None else float(f_aug_val)
        # ``d`` is a backward-compatible property for augmentation spacing.
        if self.f_aug_d <= 0:
            raise ValueError("Augmentation offset d must be positive")
        self.grid_size = int(grid_size)
        self.grid_shape = (self.grid_size, self.grid_size) if dim == 2 else None
        self.X_obs = None if X_obs is None else _as_points(X_obs, dim)
        self.X_obs_new = None
        self.X_obs_var = None
        self._X_surface_obs = jnp.empty((0, dim), dtype=X_train.dtype)
        self._y_surface_obs = jnp.empty((0,), dtype=X_train.dtype)
        self._noise_surface_obs = jnp.empty((0,), dtype=X_train.dtype)
        self.mesh = None
        self.X_grid = None
        self.grid_corners = None
        self.pose = None
        self._kdtree = None

    @property
    def sig_x(self):
        return self.noise

    @sig_x.setter
    def sig_x(self, value):
        self.noise = value
        self.invalidate()

    @property
    def d(self):
        return self.f_aug_d

    @d.setter
    def d(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError('Augmentation distance d must be positive')
        self.f_aug_d = value
        if getattr(self, '_auto_aug_value', False):
            self.f_aug_val = value
        self.invalidate()

    @property
    def X_a(self):
        return jnp.concatenate([self.X_train + self.f_aug_d * self.N_train,
                                self.X_train - self.f_aug_d * self.N_train])

    @property
    def Y_a(self):
        n = self.n_train
        return jnp.concatenate([jnp.full((n,), self.f_aug_val),
                                jnp.full((n,), -self.f_aug_val)])

    @property
    def Y_f(self):
        self._ensure_conditioned()
        return self._obs.y[:len(self._obs.X_value)].reshape(-1, 1)

    @property
    def Y_N(self):
        return self.N_train.reshape(-1, 1)

    @property
    def Y_D(self):
        self._ensure_conditioned()
        return self._obs.y

    @property
    def alpha_ff(self):
        return self.alpha

    @property
    def Sig_f(self):
        self._ensure_conditioned()
        nf = len(self._obs.X_value)
        return jnp.diag(self._obs.noise[:nf])

    @property
    def Sig_N(self):
        self._ensure_conditioned()
        nf = len(self._obs.X_value)
        return jnp.diag(self._obs.noise[nf:])

    @property
    def Sig(self):
        """Legacy dense noise matrix; conditioning uses only its diagonal."""
        self._ensure_conditioned()
        return jnp.diag(self._obs.noise)

    @property
    def Kff_inv(self):
        """Expensive legacy adapter; avoid this and use Cholesky solves."""
        self._ensure_conditioned()
        return _chol_solve(self._L, jnp.eye(len(self._obs.y), dtype=self._L.dtype))

    def _extra_signature(self):
        normal_noise = np.asarray(self.sig_n).ravel().tobytes()
        if hasattr(self.geo_prior, 'cache_signature'):
            p = self.geo_prior.cache_signature()
        else:
            vals = self.geo_prior.get_params() if hasattr(self.geo_prior, 'get_params') else ()
            p = (id(self.geo_prior), np.asarray(vals).ravel().tobytes())
        return (normal_noise, p, self.augment, self.f_aug_val, self.f_aug_d)

    def _mean_values(self, X, prior_params=None):
        if prior_params is not None:
            if not hasattr(self.geo_prior, 'prior_f_params'):
                raise ValueError("To optimize geo_prior, implement prior_f_params(X, params)")
            values = jnp.asarray(self.geo_prior.prior_f_params(X, prior_params))
        else:
            values = jnp.asarray(self.geo_prior.prior_f(X))
        if values.shape != (len(X),):
            raise ValueError('geo_prior must return shape (n,) for scalar GPIS; '
                             f'got {values.shape}. Select an anchor channel if needed.')
        return values

    def _mean_gradient(self, X, prior_params=None):
        if prior_params is not None:
            if not hasattr(self.geo_prior, 'prior_grad_f_params'):
                raise ValueError("To optimize geo_prior, implement prior_grad_f_params(X, params)")
            values = self.geo_prior.prior_grad_f_params(X, prior_params)
        else:
            values = self.geo_prior.prior_grad_f(X)
        values = jnp.asarray(values)
        if values.shape != X.shape:
            raise ValueError("geo_prior.prior_grad_f must have shape (n, dim)")
        return values

    def _observations(self, *, noise=None, prior_params=None):
        n, d = self.n_train, self.dim
        if noise is None:
            noise_f, noise_n = self.sig_x, self.sig_n
        else:
            noise = jnp.asarray(noise).reshape(-1)
            if noise.size != 2:
                raise ValueError("GPIS noise override must be [sig_x, sig_n] variances")
            noise_f, noise_n = noise[0], noise[1]
        X_value = self.X_train
        y_value = jnp.zeros((n,), dtype=X_value.dtype)
        if self.augment:
            X_value = jnp.concatenate([X_value, self.X_a])
            y_value = jnp.concatenate([y_value, self.Y_a])
        if len(self._X_surface_obs):
            X_value = jnp.concatenate([X_value, self._X_surface_obs])
            y_value = jnp.concatenate([y_value, self._y_surface_obs])
        nbase = len(X_value)-len(self._X_surface_obs)
        noises = jnp.concatenate([
            _broadcast_variance(noise_f, nbase, "sig_x"), self._noise_surface_obs,
            jnp.tile(_broadcast_variance(noise_n, n, "sig_n"), d),
        ])
        grad_target = self.N_train.T.reshape(-1)   # coordinate-major
        X_grad = self.X_train
        prior = jnp.concatenate([self._mean_values(X_value, prior_params),
                                 self._mean_gradient(X_grad, prior_params).T.reshape(-1)])
        return Observations(X_value, X_grad,
                            jnp.concatenate([y_value, grad_target]),
                            prior, noises)

    def _train_covariance(self, obs, params):
        return self.kernel.K_mat_partial(obs.X_value, obs.X_grad, params=params)

    def _cross_covariance(self, X_test, obs, params):
        kf = self.kernel.K(X_test, obs.X_value, params=params)
        ky = self.kernel.dK_y(X_test, obs.X_grad, params=params)
        return jnp.concatenate([kf] + [ky[..., j] for j in range(self.dim)], axis=1)

    def _grad_cross_covariance(self, X_test, obs, params):
        gx = self.kernel.dK(X_test, obs.X_value, params=params)
        hxy = self.kernel.ddK(X_test, obs.X_grad, params=params)
        return jnp.concatenate(
            [jnp.concatenate([gx[..., i]] + [hxy[..., i, j] for j in range(self.dim)], axis=1)
             for i in range(self.dim)], axis=0,
        )

    def prior_mean(self, X_f, X_N):
        return jnp.concatenate([self._mean_values(_as_points(X_f, self.dim)),
                                self._mean_gradient(_as_points(X_N, self.dim)).T.reshape(-1)])

    def set_data(self, X_train, N_train=None):
        X = _as_points(X_train, self.dim)
        N = estimate_normals(X) if N_train is None else _as_points(N_train, self.dim)
        if X.shape != N.shape:
            raise ValueError("N_train must match X_train shape")
        self.X_train, self.N_train, self.n_train = X, N, len(X)
        self.y_train = jnp.zeros((len(X),), dtype=X.dtype)
        self._kdtree = None
        self.invalidate()
        return self

    def add_surface_observations(self, X, y=0., noise=None, *, replace=False):
        """Add optional scalar implicit-field observations and recondition lazily."""
        X = _as_points(X, self.dim)
        y = _broadcast_variance(y, len(X), 'y')
        noise = _broadcast_variance(self.sig_x if noise is None else noise, len(X), 'noise')
        if replace:
            self._X_surface_obs, self._y_surface_obs, self._noise_surface_obs = X, y, noise
        else:
            self._X_surface_obs = jnp.concatenate([self._X_surface_obs, X])
            self._y_surface_obs = jnp.concatenate([self._y_surface_obs, y])
            self._noise_surface_obs = jnp.concatenate([self._noise_surface_obs, noise])
        self.invalidate()
        return self

    def cache_model_input(self, augment=None, use_X_obs=False, use_N_obs=False):
        """Legacy GPIS method. Constructing observations is deferred to condition()."""
        if use_N_obs:
            raise NotImplementedError("Unlabeled observed normals are not supported")
        if augment is not None:
            self.augment = bool(augment)
        if use_X_obs and self.X_obs is not None:
            self.add_surface_observations(self.X_obs, replace=True)
        self.invalidate()
        return self

    def cache_kerne_mat(self):
        """Legacy spelling retained; computes Cholesky and posterior alpha."""
        return self.condition()

    def prepare_model_input(self, X_train=None, N_train=None, augment=None,
                            use_X_obs=False, use_N_obs=False):
        """Legacy tuple (X_f, X_N, noise matrix, mean, target)."""
        if X_train is not None or N_train is not None:
            raise ValueError("Use set_data(X_train, N_train) before preparing inputs")
        if augment is not None:
            self.augment = bool(augment)
        if use_N_obs:
            raise NotImplementedError("Observed normals need explicit labels")
        obs = self._observations()
        return obs.X_value, obs.X_grad, jnp.diag(obs.noise), obs.mean, obs.y

    def prepare_noise_mat(self, X, sig_x=None, sig_n=None):
        n = len(X)
        noise_f = self.sig_x if sig_x is None else sig_x
        noise_n = self.sig_n if sig_n is None else sig_n
        return jnp.diag(jnp.concatenate([
            _broadcast_variance(noise_f, n, 'sig_x'),
            jnp.tile(_broadcast_variance(noise_n, n, 'sig_n'), self.dim)]))

    def _noise_parameters(self):
        a = jnp.asarray(self.sig_x)
        b = jnp.asarray(self.sig_n)
        if a.ndim != 0 or b.ndim != 0:
            raise ValueError("Only scalar sig_x and sig_n may be optimized")
        return jnp.stack([a, b])

    def _assign_noise_parameters(self, params):
        self.noise, self.sig_n = params[0], params[1]

    def _prior_parameters(self):
        if not all(hasattr(self.geo_prior, a) for a in
                   ('get_params', 'set_params', 'prior_f_params', 'prior_grad_f_params')):
            raise ValueError("Prior fitting needs get_params/set_params and pure *_params methods")
        params = jnp.asarray(self.geo_prior.get_params()).reshape(-1)
        if not params.size:
            raise ValueError("Prior has no trainable parameters")
        return params

    def _assign_prior_parameters(self, params):
        self.geo_prior.set_params(params)

    def _prior_bounds(self):
        if hasattr(self.geo_prior, 'parameter_bounds'):
            return self.geo_prior.parameter_bounds()
        n = len(self._prior_parameters())
        return np.full(n, -np.inf), np.full(n, np.inf)

    def nll(self, theta, fit_noise=True, fit_geo_prior=False):
        """Legacy hyperparameter-vector negative log marginal likelihood."""
        nk = len(self.kernel.params)
        theta = jnp.asarray(theta)
        nv = theta[nk:nk+2] if fit_noise else None
        pv = theta[nk+(2 if fit_noise else 0):] if fit_geo_prior else None
        return self.negative_log_marginal_likelihood(theta[:nk], noise=nv, prior_params=pv)

    def fit(self, fit_noise=True, fit_geo_prior=False, *,
            upper_bound_sig=50., upper_bound_kparams=50., upper_bound_geo=None,
            lower_bound_kparams=1e-4, lower_bound_sig=1e-5,
            lower_bound_geo=None, maxiter=300, tol=1e-9):
        """Legacy GPIS API: maximize marginal likelihood of mixed observations.

        Unlike generic GP.fit(), GPIS.fit() optimizes hyperparameters as the
        original implementation did. Use condition() for conditioning only.
        """
        res = self.fit_hyperparameters(
            fit_noise=fit_noise, fit_prior=fit_geo_prior,
            upper_bound_kparams=upper_bound_kparams,
            lower_bound_kparams=lower_bound_kparams,
            lower_bound_noise=lower_bound_sig, upper_bound_noise=upper_bound_sig,
            lower_bound_prior=lower_bound_geo, upper_bound_prior=upper_bound_geo,
            maxiter=maxiter, tol=tol,
        )
        return jnp.asarray(res.x)   # legacy GPIS.fit() returned parameters

    def pred_f(self, X_test=None, chol=True, mean_only=False, sig_x_input=None,
               rot=None, *, full_cov=False, batch_size=None, **_unused):
        """Compatibility prediction: (mean, pointwise variance) by default."""
        X = self._default_points(X_test)
        if mean_only:
            return self.predict_mean(X, batch_size=batch_size)
        if sig_x_input is not None:
            return self.predict_uncertain(X, sig_x_input, input_jacobian=rot,
                                          return_mean=True, batch_size=batch_size)
        return self.predict(X, full_cov=full_cov, batch_size=batch_size)

    def _default_points(self, X_test):
        if X_test is not None:
            return _as_points(X_test, self.dim)
        if self.X_grid is None:
            if self.dim == 2:
                self.init_grid()
            else:
                raise ValueError("Supply X_test for 3D GPIS predictions")
        return self.X_grid

    def pred_f_fast(self, X_test):
        return self.predict_mean(_as_points(X_test, self.dim))

    def pred_f_single(self, X_i):
        return self.pred_f_fast(jnp.asarray(X_i).reshape(1, self.dim))

    def pred_f_seq(self, X_test, batch_size=1000):
        """Evaluate large point clouds without materializing all cross-covariances."""
        return self.predict_mean(X_test, batch_size=batch_size)

    def pred_cov_f(self, X_test=None, sig_x_input=None, diag=True, prior_jacobian=None,
                   *, batch_size=None, uncertainty_order=1, **_unused):
        X = self._default_points(X_test)
        if sig_x_input is not None:
            if not diag:
                raise NotImplementedError("Input-uncertain cross-covariances are not implemented")
            return self.predict_uncertain(X, sig_x_input, input_jacobian=prior_jacobian,
                                          order=uncertainty_order, return_mean=False,
                                          batch_size=batch_size)
        return self.predict(X, full_cov=not diag, batch_size=batch_size)[1]

    def pred_N(self, X_test=None, mean_only=False, *, full_cov=False, **_unused):
        return self.predict_gradient(self._default_points(X_test),
                                     return_var=not mean_only, full_cov=full_cov)

    def get_pred_f_grad(self, x):
        return self.predict_gradient(_as_points(x, self.dim), return_var=False)

    def predict_uncertain(self, X_test, cov_x, *, input_jacobian=None,
                          order=1, return_mean=True, batch_size=None):
        """Approximate marginal variance under Gaussian input uncertainty.

        1st-order: V[f(x+e)] ≈ V[f(x)] + gᵀΣg + tr(Σ Cov[∇f(x)]).
        2nd-order: adds 1/2 tr(Σ Hessian(V[f(x)])), computed using autodiff;
        considerably more costly and requires sufficiently smooth kernels.
        No expensive training inverse or full test-test covariance is formed.
        """
        X = _as_points(X_test, self.dim)
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if batch_size is not None:
            C_in = jnp.asarray(cov_x)
            J_in = None if input_jacobian is None else jnp.asarray(input_jacobian)
            result = [self.predict_uncertain(
                      X[i:i+batch_size],
                      C_in[i:i+batch_size] if C_in.ndim == 3 else C_in,
                      input_jacobian=(J_in[i:i+batch_size] if J_in is not None and J_in.ndim == 3 else J_in),
                      order=order, return_mean=return_mean)
                      for i in range(0, len(X), batch_size)]
            if not result:
                empty = jnp.empty((0,), dtype=X.dtype)
                return (empty, empty) if return_mean else empty
            if return_mean:
                return tuple(jnp.concatenate([r[j] for r in result]) for j in (0, 1))
            return jnp.concatenate(result)

        self._ensure_conditioned()
        params = jnp.asarray(self.kernel.params)
        m, d = X.shape
        C = jnp.asarray(cov_x)
        if C.shape not in ((d, d), (m, d, d)):
            raise ValueError(f"Input covariance shape must be ({d},{d}) or ({m},{d},{d})")
        C = jnp.broadcast_to(C, (m, d, d))
        if input_jacobian is not None:
            J = jnp.asarray(input_jacobian)
            if J.shape not in ((d, d), (m, d, d)):
                raise ValueError("input_jacobian shape must be (d,d) or (n,d,d)")
            J = jnp.broadcast_to(J, (m,d,d))
            C = J @ C @ J.transpose(0, 2, 1)

        mu, var = self._predict_core(X, self._obs, self._L, self._alpha, params,
                                     return_var=True, full_cov=False,
                                     observation_noise=0.)
        cross = self._grad_cross_covariance(X, self._obs, params)
        mean_grad = self._mean_gradient(X) + (cross @ self._alpha).reshape(d, m).T
        z = solve_triangular(self._L, cross.T, lower=True)
        z = z.T.reshape(d, m, -1).transpose(1, 0, 2)
        H0 = jax.vmap(lambda x: self.kernel.ddk_fun(x, x, *params))(X)
        cov_grad = H0 - jnp.einsum('mij,mkj->mik', z, z)
        var = var + jnp.einsum('mi,mij,mj->m', mean_grad, C, mean_grad)
        var = var + jnp.einsum('mij,mji->m', C, cov_grad)
        if order == 2:
            def scalar_var(x):
                return self._predict_core(x[None, :], self._obs, self._L,
                    self._alpha, params, return_var=True, full_cov=False,
                    observation_noise=0.)[1][0]
            Hessian = jax.vmap(jax.hessian(scalar_var))(X)
            var = var + 0.5 * jnp.einsum('mij,mji->m', C, Hessian)
        var = jnp.maximum(var, 0.)
        return (mu, var) if return_mean else var

    def pred_cov_f_up(self, X_test, cov_x_input, *legacy_args, jac=None,
                      R_t=None, order=2):
        """Legacy uncertain-input entry point; prefer ``predict_uncertain``.

        Older code passed X_f, X_N, K_inv, alpha, jac positionally. These
        matrices are intentionally ignored: the current Cholesky posterior
        is used. A positional jac is honored to preserve the frame mapping.
        """
        if legacy_args:
            if len(legacy_args) != 5:
                raise TypeError("Expected five legacy matrix arguments or none")
            if jac is None:
                jac = legacy_args[-1]
        if R_t is not None:
            jac = R_t
        return self.predict_uncertain(X_test, cov_x_input,
                     input_jacobian=jac, order=order, return_mean=False)

    def make_pred_cov_fn(self):
        """Returns a callable for the uncertainty variance (modern API)."""
        return lambda X_test, cov_x_input: self.pred_cov_f_up(X_test, cov_x_input)

    def get_surface_pdf(self, X_test, f_val=0., sig_x_input=None):
        return jnp.exp(self.log_surface_pdf(X_test, f_val, sig_x_input))

    def log_surface_pdf(self, X_test, f_val=0., sig_x_input=None):
        X = _as_points(X_test, self.dim)
        mean, var = self.pred_f(X, sig_x_input=sig_x_input)
        return normal_logpdf(f_val, mean, var)

    def get_surface_cdf(self, X_test, min_f, max_f, sig_x_input=None):
        return jnp.exp(self.log_surface_cdf(X_test, min_f, max_f, sig_x_input))

    def log_surface_cdf(self, X_test, x_min=-0.05, x_max=0.05, sig_x_input=None):
        mean, var = self.pred_f(X_test, sig_x_input=sig_x_input)
        return normal_interval_logprob(x_min, x_max, mean, var)

    def log_surface_pdf_grad(self, X_test, f_val=0., sig_x_input=None):
        X = _as_points(X_test, self.dim)
        scalar = lambda x: self.log_surface_pdf(x[None, :], f_val, sig_x_input)[0]
        return jax.vmap(jax.grad(scalar))(X)

    def log_surface_cdf_grad(self, X_test, x_min=-0.02, x_max=0.02,
                             sig_x_input=None):
        X = _as_points(X_test, self.dim)
        scalar = lambda x: self.log_surface_cdf(x[None, :], x_min, x_max,
                                                 sig_x_input)[0]
        return jax.vmap(jax.grad(scalar))(X)

    def get_surface_likelihood_batched(self, Z_t, batch_size=128, cov_x=None):
        """Surface log-likelihoods for (n_clouds, n_points, d)."""
        self._ensure_conditioned()
        Z = jnp.asarray(Z_t)
        if Z.ndim != 3 or Z.shape[-1] != self.dim:
            raise ValueError("Z_t must have shape (n_clouds, n_points, dim)")
        results = [jax.vmap(lambda x: self.log_surface_pdf(x, sig_x_input=cov_x))(
                   Z[i:i+batch_size]) for i in range(0, len(Z), batch_size)]
        return jnp.concatenate(results, axis=0) if results else jnp.empty(Z.shape[:2])

    def get_surface_likelihood_chunked(self, Z_t, R, batch_size=128):
        return self.get_surface_likelihood_batched(Z_t, batch_size=batch_size, cov_x=R)

    def knn_search(self, queries=None, k=10):
        """Nearest neighbours via SciPy cKDTree; no optional jaxkd dependency."""
        from scipy.spatial import cKDTree
        if self._kdtree is None:
            self._kdtree = cKDTree(np.asarray(self.X_train))
        queries = self.X_train if queries is None else _as_points(queries, self.dim)
        dist, ind = self._kdtree.query(np.asarray(queries), k=min(int(k), self.n_train))
        return jnp.asarray(ind), jnp.asarray(dist)

    def match_pcds(self, Y, T_init=None, *, n_iter=50, lam=1e-3, tol=1e-8,
                   damping_mode: Literal['lm', 'additive']='lm', update_mat=False,
                   augment=None, use_normals=True, verbose=False):
        """Gauss-Newton point-to-implicit-surface registration (requires jaxlie).

        Uses the analytic GP gradient and SE(2)/SE(3) left perturbation
        Jacobian. A backtracking line search prevents objective increases.
        ``use_normals=False`` uses only scalar-value constraints.
        """
        try:
            from jaxlie import SE2, SE3
        except ImportError as exc:
            raise ImportError('match_pcds requires `pip install jaxlie`') from exc
        if damping_mode not in ('lm', 'additive'):
            raise ValueError("damping_mode must be 'lm' or 'additive'")
        if n_iter < 1 or lam < 0:
            raise ValueError("Require n_iter>=1 and lam>=0")
        if update_mat:
            self.cache_model_input(augment=augment)
            self.condition()
        self._ensure_conditioned()
        G = SE3 if self.dim == 3 else SE2
        T = G.identity() if T_init is None else T_init
        Y = _as_points(Y, self.dim)
        if use_normals:
            posterior_mean = self.pred_f_fast
            posterior_grad = self.get_pred_f_grad
        else:
            params = jnp.asarray(self.kernel.params)
            nf = len(self._obs.X_value)
            Kff = self.kernel.K(self._obs.X_value, self._obs.X_value, params=params)
            L = jnp.linalg.cholesky(Kff + jnp.diag(self._obs.noise[:nf] + self.jitter))
            coeff = _chol_solve(L, self._obs.y[:nf] - self._obs.mean[:nf])
            posterior_mean = lambda P: self._mean_values(P) + self.kernel.K(P, self._obs.X_value, params=params) @ coeff
            posterior_grad = lambda P: (self._mean_gradient(P) + jnp.einsum('mnd,n->md',
                   self.kernel.dK(P, self._obs.X_value, params=params), coeff))
        history = []
        previous = np.inf
        ndof = 6 if self.dim == 3 else 3
        for it in range(n_iter):
            P = T.apply(Y)
            r = posterior_mean(P)
            grad = posterior_grad(P)
            if self.dim == 2:
                rot = jnp.sum(grad * jnp.stack([-P[:,1], P[:,0]], axis=1), axis=1)[:,None]
            else:
                rot = jnp.cross(P, grad)
            J = jnp.concatenate([grad, rot], axis=1)
            A = J.T @ J
            D = jnp.diag(jnp.maximum(jnp.diag(A), 1e-8)) if damping_mode == 'lm' else jnp.eye(ndof)
            step = -jnp.linalg.solve(A + lam*D + 1e-8*jnp.eye(ndof), J.T @ r)
            old_E = float(0.5*jnp.sum(r*r))
            new_T, new_E, used = T, old_E, 0.
            for scale in (1., 0.5, 0.25, 0.125, 0.0625):
                proposed = G.exp(scale * step) @ T
                E = float(0.5*jnp.sum(posterior_mean(proposed.apply(Y))**2))
                if np.isfinite(E) and E < new_E:
                    new_T, new_E, used = proposed, E, scale
                    break
            step_norm = float(jnp.linalg.norm(used * step))
            history.append((new_E, step_norm))
            if verbose:
                print(f'iter {it:03d}: E={new_E:.7g}, |delta|={step_norm:.3g}')
            T = new_T
            if step_norm <= tol or abs(old_E - new_E) <= tol:
                break
            previous = new_E
        self.pose = T
        return T, {'history': history, 'iters': len(history)}

    def pose_score(self, T, Y):
        values = self.pred_f_fast(T.apply(_as_points(Y, self.dim)))
        return 0.5*jnp.dot(values, values)

    def init_grid(self, X=None, N=None, mult=1.3, grid_size=None):
        """Construct a 2D world-coordinate grid (no unexpected recentering)."""
        if self.dim != 2:
            raise ValueError('init_grid is a 2D-only convenience function')
        P = np.asarray(self.X_train if X is None else X)
        n = self.grid_size if grid_size is None else int(grid_size)
        center = 0.5*(P.max(0)+P.min(0))
        half = np.maximum(0.5*(P.max(0)-P.min(0))*mult, 1e-3)
        xs = np.linspace(center[0]-half[0], center[0]+half[0], n)
        ys = np.linspace(center[1]-half[1], center[1]+half[1], n)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        self.X_grid = jnp.asarray(np.stack([gx.ravel(), gy.ravel()], axis=1))
        self.grid_shape = (n, n)
        self.grid_size = n
        self.grid_corners = np.array([[xs[0],ys[0]], [xs[-1],ys[0]],
                                      [xs[0],ys[-1]], [xs[-1],ys[-1]]])
        return self.X_grid

    def extract_contour(self, res=250, level=0., mult=1.5, use_X_obs=False):
        """Extract 2D level-set polylines (list of arrays in world coordinates)."""
        from skimage.measure import find_contours
        if self.dim != 2:
            raise ValueError('extract_contour requires a 2D GPIS')
        self.init_grid(grid_size=res, mult=mult)
        f = np.asarray(self.pred_f_seq(self.X_grid)).reshape(self.grid_shape)
        xs = np.asarray(self.X_grid[:, 0]).reshape(self.grid_shape)[:, 0]
        ys = np.asarray(self.X_grid[:, 1]).reshape(self.grid_shape)[0, :]
        contours = []
        for c in find_contours(f, level):
            contours.append(np.stack([np.interp(c[:, 0], np.arange(len(xs)), xs),
                                      np.interp(c[:, 1], np.arange(len(ys)), ys)], axis=1))
        return contours

    def _mesh_samples(self, grid_res, bbox=None, padding=0.1):
        if self.dim != 3:
            raise ValueError('3D mesh extraction requires dim=3')
        if not isinstance(grid_res, int) or grid_res < 2:
            raise ValueError('grid_res must be an integer >=2')
        P = np.asarray(self.X_train)
        if bbox is None:
            lo, hi = P.min(0)-padding, P.max(0)+padding
        else:
            lo, hi = map(np.asarray, bbox)
        axes = [np.linspace(lo[k],hi[k],grid_res) for k in range(3)]
        xyz = np.meshgrid(*axes, indexing='ij')
        grid = jnp.asarray(np.stack([x.ravel() for x in xyz], axis=1))
        return grid, lo, hi

    def extract_3d_mesh(self, X_train=None, grid_res=75, bbox=None, mult=0.1,
                        level=0., use_X_obs=False, color_variance=True, **_unused):
        """Extract 3D isosurface as an Open3D TriangleMesh (optional extras)."""
        from skimage.measure import marching_cubes
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError('extract_3d_mesh requires `pip install open3d`') from exc
        grid, lo, hi = self._mesh_samples(grid_res, bbox, mult)
        samples = np.asarray(self.pred_f_seq(grid)).reshape([grid_res]*3)
        if not samples.min() <= level <= samples.max():
            raise ValueError('Requested level is outside the sampled implicit field')
        spacing = (hi-lo)/(grid_res-1)
        verts, faces, _, _ = marching_cubes(samples, level=level,
                                            spacing=tuple(spacing))
        verts += lo
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
        mesh.compute_vertex_normals()
        if color_variance:
            std = np.sqrt(np.asarray(self.pred_cov_f(verts, batch_size=2048)))
            vmax = max(float(np.max(std)), 1e-12)
            from matplotlib import colormaps
            colors = colormaps['viridis'](std/vmax)[:,:3]
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.mesh = mesh
        return mesh

    def extract_prior_mesh(self, grid_res=60, bbox=((-1,-1,-1), (1,1,1)), level=0.):
        """Open3D mesh for the geometric prior mean."""
        from skimage.measure import marching_cubes
        import open3d as o3d
        grid, lo, hi = self._mesh_samples(grid_res, bbox)
        values = np.asarray(self._mean_values(grid)).reshape([grid_res]*3)
        verts, faces, _, _ = marching_cubes(values, level=level,
                                           spacing=tuple((hi-lo)/(grid_res-1)))
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector((verts+lo).astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        return mesh

    def _plot_map(self, kind='mean', *, grid_size=None, mult=1.4, colorbar=True,
                  plot_train_data=True, vmin=None, vmax=None, cmap=None,
                  plot_contour=False, **_unused):
        import matplotlib.pyplot as plt
        X = self.init_grid(grid_size=grid_size, mult=mult)
        mean, var = self.predict(X, batch_size=1024)
        if kind == 'mean':
            values = mean
            cmap = cmap or 'RdBu_r'
        elif kind == 'prior':
            values = self._mean_values(X)
            cmap = cmap or 'RdBu_r'
        elif kind == 'var':
            values = jnp.sqrt(var)
            cmap = cmap or 'plasma'
        elif kind == 'pdf':
            values = jnp.exp(normal_logpdf(0., mean, var))
            cmap = cmap or 'viridis'
        elif kind == 'cdf':
            eps = _unused.get('eps', 0.01)
            values = jnp.exp(normal_interval_logprob(-eps, eps, mean, var))
            cmap = cmap or 'viridis'
        else:
            raise ValueError(f'Unknown map type {kind!r}')
        grid = np.asarray(X).reshape(*self.grid_shape, 2)
        ax = plt.gca()
        artist = ax.pcolormesh(grid[:,:,0], grid[:,:,1],
                               np.asarray(values).reshape(self.grid_shape),
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        if colorbar:
            plt.colorbar(artist, ax=ax)
        if plot_contour or kind=='mean':
            ax.contour(grid[:,:,0],grid[:,:,1],np.asarray(mean).reshape(self.grid_shape),
                       levels=[0.], colors='k')
        if plot_train_data:
            P = np.asarray(self.X_train)
            ax.scatter(P[:,0],P[:,1], s=7, color='black')
        ax.set_aspect('equal')
        return ax

    def plot_mean_map(self, **kwargs):
        return self._plot_map('mean', **kwargs)

    def plot_var_map(self, **kwargs):
        return self._plot_map('var', **kwargs)

    def plot_prior_mean(self, **kwargs):
        return self._plot_map('prior', **kwargs)

    def plot_likelihood_map(self, **kwargs):
        return self._plot_map('pdf', **kwargs)

    def plot_cdf_map(self, **kwargs):
        return self._plot_map('cdf', **kwargs)

    def plot_contour(self, levels=10, filled=False, surface=True, colorbar=False):
        return self._plot_map('mean', colorbar=colorbar, plot_contour=surface)

    def plot_sdf_plotly(self, grid_resolution=30, grid_margin=1.,
                        isomin=None, isomax=None, surface_count=10, opacity=.15,
                        colorscale='RdBu', show_patch=True, show=True,
                        slice_axis=None, slice_value=None):
        """3D Plotly volume of the implicit mean. Optional Plotly dependency."""
        import plotly.graph_objects as go
        grid, lo, hi = self._mesh_samples(grid_resolution, padding=grid_margin)
        fields = np.asarray(self.pred_f_seq(grid)).reshape(-1)
        X, Y, Z = np.meshgrid(*[np.linspace(lo[i], hi[i], grid_resolution)
                                 for i in range(3)], indexing='ij')
        trace = {"x": X.ravel(),"y": Y.ravel(),"z": Z.ravel(),"value": fields,
                     "colorscale": colorscale,"opacity": opacity,"surface_count": surface_count,
                     "caps": {"x_show": False, "y_show": False,"z_show": False,}}
        if isomin is not None: trace['isomin'] = isomin
        if isomax is not None: trace['isomax'] = isomax
        if slice_axis is not None:
            if slice_axis not in ('x','y','z'):
                raise ValueError('slice_axis must be x, y, or z')
            i = 'xyz'.index(slice_axis)
            value = (lo[i]+hi[i])/2 if slice_value is None else slice_value
            trace['slices']={slice_axis:dict(show=True,locations=[float(value)])}
        fig = go.Figure([go.Volume(**trace)])
        if show_patch:
            P=np.asarray(self.X_train)
            fig.add_trace(go.Scatter3d(x=P[:,0],y=P[:,1],z=P[:,2],mode='markers',
                                       marker=dict(size=3,color='black')))
        if show:
            fig.show()
        return fig

    def save_model(self, save_dir='saved_GPIS_model', file_name='gpis_model', overwrite=True):
        """Serialize numerical GPIS state without compiled functions/Cholesky."""
        dest = Path(save_dir)/file_name
        if not str(dest).endswith('.pkl'):
            dest = dest.with_suffix('.pkl')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            raise FileExistsError(dest)
        prior = self.geo_prior
        prior_config = {
            'class': f'{type(prior).__module__}.{type(prior).__qualname__}',
            'params': np.asarray(prior.get_params()) if hasattr(prior, 'get_params') else None,
            'channel_id': getattr(prior, 'channel_id', None),
            'static': {key: getattr(prior, key) for key in
                       ('cutoff_dist', 'edge_width', 'distance_epsilon',
                        'radius_epsilon', 'jitter') if hasattr(prior, key)},
        }
        if hasattr(prior, 'u_f'):
            prior_config['u_f'] = np.asarray(prior.u_f)
        if hasattr(prior, 'kernel'):
            prior_config['inner_kernel_params'] = np.asarray(prior.kernel.params)
        config = {"geo_prior": prior_config,
                      "X_train": np.asarray(self.X_train), "N_train": np.asarray(self.N_train),
                      "kern_params": np.asarray(self.kernel.params),
                      "sig_x": np.asarray(self.sig_x), "sig_n": np.asarray(self.sig_n),
                      "augment": self.augment, "f_aug_val": self.f_aug_val, "d": self.f_aug_d,
                      "auto_aug_value": self._auto_aug_value,
                      "X_obs": np.asarray(self.X_obs) if self.X_obs is not None else None,
                      "X_surface_obs": np.asarray(self._X_surface_obs),
                      "y_surface_obs": np.asarray(self._y_surface_obs),
                      "noise_surface_obs": np.asarray(self._noise_surface_obs),
                      "fit_time": self.fit_time}
        with dest.open('wb') as f:
            pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
        return dest

    def load_gpis(self, save_dir='saved_GPIS_model', file_name='gpis_model'):
        path = Path(save_dir)/file_name
        if not str(path).endswith('.pkl'):
            path = path.with_suffix('.pkl')
        if not path.exists():
            old_path = path.with_name(f"{path.stem}_{self.dim}.pkl")
            if old_path.exists():
                path = old_path
        with path.open('rb') as f:
            state = pickle.load(f)
        stored_prior = state.get('geo_prior')
        if stored_prior is not None:
            prior = self.geo_prior
            current_type = f'{type(prior).__module__}.{type(prior).__qualname__}'
            if stored_prior['class'] != current_type:
                raise ValueError('Saved model uses prior ' + stored_prior['class'] +
                                 '; initialize GPIS with the matching geometric prior')
            if stored_prior.get('channel_id') != getattr(prior, 'channel_id', None):
                raise ValueError('Saved AnchorPrior channel selection does not match')
            for attr, expected in stored_prior.get('static', {}).items():
                if not np.allclose(np.asarray(getattr(prior, attr)), expected):
                    raise ValueError(f'Saved prior requires {attr}={expected}')
            if 'u_f' in stored_prior and not np.array_equal(
                    np.asarray(prior.u_f), stored_prior['u_f']):
                raise ValueError('Saved radial prior uses different inducing points u_f')
            if 'inner_kernel_params' in stored_prior:
                prior.kernel.set_params(jnp.asarray(stored_prior['inner_kernel_params']))
            if stored_prior['params'] is not None:
                if not hasattr(prior, 'set_params'):
                    raise ValueError('Prior cannot restore saved parameters')
                prior.set_params(jnp.asarray(stored_prior['params']))
        self.set_data(state['X_train'], state['N_train'])
        self.kernel.set_params(jnp.asarray(state['kern_params']))
        self.noise = jnp.asarray(state['sig_x'])
        self.sig_n = jnp.asarray(state['sig_n'])
        self.augment = bool(state.get('augment', False))
        self._auto_aug_value = bool(state.get('auto_aug_value', False))
        self.f_aug_val = float(state.get('f_aug_val', 1.))
        self.d = float(state.get('d', .05))
        self.X_obs = state.get('X_obs')
        self._X_surface_obs = jnp.asarray(state.get('X_surface_obs', np.empty((0,self.dim))))
        self._y_surface_obs = jnp.asarray(state.get('y_surface_obs', np.empty((0,))))
        self._noise_surface_obs = jnp.asarray(state.get('noise_surface_obs', np.empty((0,))))
        stored_fit_time = state.get('fit_time')
        self.invalidate()
        self.condition()
        self.fit_time = stored_fit_time
        return self
