"""Reusable exact, scalar-output Gaussian-process regression with JAX kernels.

Conventions
-----------
Inputs: (n, d), outputs: (n,); noise hyperparameters are VARIANCES.
All posterior factorizations use a lower-triangular Cholesky factor, never an
explicit inverse. Hyperparameter optimization is differentiable and side-effect
free; Python-side cache updates are performed only after optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular


@dataclass(frozen=True)
class Observations:
    """Values followed by coordinate-major partial derivatives.

    y, mean and noise are ordered as [f values, ∂_0 f, ..., ∂_(d-1) f].
    A vanilla GP has X_grad.shape == (0, d).
    """

    X_value: jax.Array
    X_grad: jax.Array
    y: jax.Array
    mean: jax.Array
    noise: jax.Array


def _as_points(X, dim=None):
    X = jnp.asarray(X)
    if X.ndim != 2 or (dim is not None and X.shape[1] != dim):
        raise ValueError(f"Points must have shape (n, {dim or 'd'}); got {X.shape}")
    return X


def _as_targets(y, n):
    y = jnp.asarray(y).reshape(-1)
    if y.size != n:
        raise ValueError(f"Expected {n} scalar training targets, got {y.size}")
    return y


def _broadcast_variance(value, size, name):
    value = jnp.asarray(value)
    if value.ndim > 1:
        raise ValueError(f"{name} must be scalar or a 1-D vector")
    try:
        value = jnp.broadcast_to(value, (size,))
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or have length {size}") from exc
    return value


def _chol_solve(L, rhs):
    z = solve_triangular(L, rhs, lower=True)
    return solve_triangular(L.T, z, lower=False)


class GP:
    """Exact scalar Gaussian process with optional nonzero prior mean.

    The kernel must provide K, K_vec, dK, dK_y, ddK and set_params as in
    the supplied ``kernels`` package. GPIS overrides training-observation and
    test-covariance hooks, reusing the *same* posterior conditioning routines.

    ``fit()`` conditions on observations. ``fit_hyperparameters()`` optimizes
    the exact log marginal likelihood. For large training datasets, exact GP
    conditioning remains O(n^3) time and O(n^2) memory.
    """

    def __init__(
        self,
        kernel,
        X_train,
        y_train,
        *,
        noise: float | jax.Array = 1e-4,
        mean_fn: Callable | None = None,
        jitter: float = 1e-6,
    ):
        self.kernel = kernel
        self.X_train = _as_points(X_train)
        self.n_train, self.dim = self.X_train.shape
        if self.n_train == 0:
            raise ValueError("X_train must not be empty")
        if hasattr(kernel, "dim") and kernel.dim != self.dim:
            raise ValueError("Kernel dimension does not match X_train")
        self.y_train = _as_targets(y_train, self.n_train)
        self.noise = noise
        self.mean_fn = mean_fn
        if jitter <= 0:
            raise ValueError("jitter must be strictly positive")
        self.jitter = float(jitter)
        self._obs: Observations | None = None
        self._L = None
        self._alpha = None
        self._cache_key = None
        self.fit_time = None
        self.condition_time = None
        self.optimization_result = None

    def __repr__(self):
        return f"{type(self).__name__}(kernel={self.kernel!r}, n={self.n_train})"

    def invalidate(self):
        """Invalidate the factorization after external edits to training state."""
        self._obs = self._L = self._alpha = self._cache_key = None

    def set_data(self, X_train, y_train):
        X = _as_points(X_train, self.dim)
        if len(X) == 0:
            raise ValueError("Training data cannot be empty")
        self.X_train = X
        self.y_train = _as_targets(y_train, len(X))
        self.n_train = len(X)
        self.invalidate()
        return self

    def _mean_values(self, X, prior_params=None):
        if self.mean_fn is None:
            if prior_params is not None:
                raise ValueError("The GP mean has no trainable prior parameters")
            return jnp.zeros((len(X),), dtype=X.dtype)
        if prior_params is not None:
            if not hasattr(self.mean_fn, 'prior_f_params'):
                raise ValueError("mean_fn must implement prior_f_params for prior fitting")
            result = jnp.asarray(self.mean_fn.prior_f_params(X, prior_params))
        else:
            result = jnp.asarray(self.mean_fn(X))
        if result.shape == (len(X), 1):
            result = result[:, 0]
        if result.shape != (len(X),):
            raise ValueError("mean_fn must return one scalar per query point; "
                             f"got {result.shape}. Select a channel for multi-output priors.")
        return result

    def _mean_gradient(self, X, prior_params=None):
        if self.mean_fn is None:
            return jnp.zeros_like(X)
        if prior_params is not None and hasattr(self.mean_fn, 'prior_grad_f_params'):
            grad = self.mean_fn.prior_grad_f_params(X, prior_params)
        elif prior_params is None and hasattr(self.mean_fn, 'prior_grad_f'):
            grad = self.mean_fn.prior_grad_f(X)
        else:
            grad = jax.vmap(jax.grad(lambda x: self._mean_values(x[None], prior_params)[0]))(X)
        grad = jnp.asarray(grad)
        if grad.shape != X.shape:
            raise ValueError(f"mean_fn gradient must have shape {X.shape}, got {grad.shape}")
        return grad

    def _observations(self, *, noise=None, prior_params=None):
        empty = jnp.empty((0, self.dim), dtype=self.X_train.dtype)
        noise = self.noise if noise is None else noise
        return Observations(
            self.X_train,
            empty,
            self.y_train,
            self._mean_values(self.X_train, prior_params),
            _broadcast_variance(noise, self.n_train, "noise"),
        )

    def _train_covariance(self, obs, params):
        return self.kernel.K(obs.X_value, obs.X_value, params=params)

    def _cross_covariance(self, X_test, obs, params):
        return self.kernel.K(X_test, obs.X_value, params=params)

    def _grad_cross_covariance(self, X_test, obs, params):
        """Shape (d*n_test, n_observations); test derivatives coordinate-major."""
        dx = self.kernel.dK(X_test, obs.X_value, params=params)
        return jnp.concatenate([dx[..., i] for i in range(self.dim)], axis=0)

    def _gradient_diagonal(self, X, params):
        # Computing only n diagonal Hessians avoids a full (n,n,d,d) tensor.
        H = jax.vmap(lambda x: self.kernel.ddk_fun(x, x, *params))(X)
        return jnp.swapaxes(jnp.diagonal(H, axis1=1, axis2=2), 0, 1).reshape(-1)

    def _covariance_factor(self, obs, params):
        K = self._train_covariance(obs, params)
        n = K.shape[0]
        if obs.y.size != n or obs.noise.size != n or obs.mean.size != n:
            raise ValueError("Training covariance and observation dimensions differ")
        K = 0.5 * (K + K.T)
        return jnp.linalg.cholesky(K + jnp.diag(obs.noise + self.jitter))

    def _condition_for(self, params, *, noise=None, prior_params=None):
        obs = self._observations(noise=noise, prior_params=prior_params)
        L = self._covariance_factor(obs, params)
        alpha = _chol_solve(L, obs.y - obs.mean)
        return obs, L, alpha

    def _signature(self):
        params = np.asarray(self.kernel.params).ravel()
        noise = np.asarray(self.noise).ravel()
        return (params.tobytes(), noise.tobytes(), self._extra_signature())

    def _extra_signature(self):
        # Bare Python functions are assumed stateless; prior objects declare
        # mutable parameter signatures to ensure the Cholesky cache is fresh.
        if hasattr(self.mean_fn, 'cache_signature'):
            return self.mean_fn.cache_signature()
        return ()

    def _ensure_conditioned(self):
        if self._L is None or self._signature() != self._cache_key:
            self.condition()

    def condition(self):
        """Factorize the training covariance and cache posterior coefficients."""
        start = perf_counter()
        params = jnp.asarray(self.kernel.params)
        obs = self._observations()
        if not bool(jnp.all(jnp.isfinite(obs.noise)) & jnp.all(obs.noise >= 0)):
            raise ValueError("Observation noise variances must be nonnegative and finite")
        obs, L, alpha = self._condition_for(params)
        if not bool(jnp.all(jnp.isfinite(L)) & jnp.all(jnp.isfinite(alpha))):
            raise np.linalg.LinAlgError(
                "GP covariance factorization failed; increase jitter or check the kernel/noise"
            )
        self._obs, self._L, self._alpha = obs, L, alpha
        self._cache_key = self._signature()
        self.condition_time = perf_counter() - start
        return self

    def fit(self, X_train=None, y_train=None):
        """Standard GP fit: optionally set new data, then condition."""
        if (X_train is None) != (y_train is None):
            raise ValueError("Pass both X_train and y_train, or neither")
        if X_train is not None:
            self.set_data(X_train, y_train)
        return self.condition()

    @property
    def alpha(self):
        self._ensure_conditioned()
        return self._alpha

    @property
    def L_ff(self):
        self._ensure_conditioned()
        return self._L

    @property
    def X_f(self):
        self._ensure_conditioned()
        return self._obs.X_value

    @property
    def X_N(self):
        self._ensure_conditioned()
        return self._obs.X_grad

    def _predict_mean_core(self, X, obs, alpha, params):
        return self._mean_values(X) + self._cross_covariance(X, obs, params) @ alpha

    def _predict_core(self, X, obs, L, alpha, params, *, return_var, full_cov, observation_noise):
        cross = self._cross_covariance(X, obs, params)
        mu = self._mean_values(X) + cross @ alpha
        if not return_var:
            return mu
        z = solve_triangular(L, cross.T, lower=True)
        if full_cov:
            Ktt = self.kernel.K(X, X, params=params)
            cov = Ktt - z.T @ z
            cov = 0.5 * (cov + cov.T)
            cov = cov + jnp.eye(len(X), dtype=X.dtype) * observation_noise
            return mu, cov
        prior_variance = self.kernel.K_vec(X, X, params=params)
        variance = jnp.maximum(prior_variance - jnp.sum(z*z, axis=0), 0.0)
        variance = variance + observation_noise
        return mu, variance

    def predict(self, X_test, *, return_var=True, full_cov=False, observation_noise=0.0,
                batch_size=None):
        """Posterior latent mean and diagonal variance (or full covariance).

        ``observation_noise`` is a scalar *variance* added only to predictions.
        ``batch_size`` works with mean and diagonal variance, not full covariance.
        """
        X = _as_points(X_test, self.dim)
        if full_cov and (batch_size is not None):
            raise ValueError("full_cov cannot be requested with batching")
        self._ensure_conditioned()
        params = jnp.asarray(self.kernel.params)
        if batch_size is not None:
            if int(batch_size) < 1:
                raise ValueError("batch_size must be positive")
            parts = [self._predict_core(X[i:i+batch_size], self._obs, self._L,
                                        self._alpha, params, return_var=return_var,
                                        full_cov=False, observation_noise=observation_noise)
                     for i in range(0, len(X), batch_size)]
            if not parts:
                return (jnp.empty((0,)), jnp.empty((0,))) if return_var else jnp.empty((0,))
            if return_var:
                return tuple(jnp.concatenate([p[k] for p in parts]) for k in (0, 1))
            return jnp.concatenate(parts)
        return self._predict_core(X, self._obs, self._L, self._alpha, params,
                                  return_var=return_var, full_cov=full_cov,
                                  observation_noise=observation_noise)

    def compile_predict(self, *, return_var=True, full_cov=False,
                        observation_noise=0.0):
        """Compile a *snapshot* of the currently conditioned posterior.

        Unlike wrapping ``jax.jit(model.predict)``, this excludes Python-side
        cache validation from tracing. It works for GPIS geometric priors as
        well as the base GP and is convenient for repeated GPU queries.

        IMPORTANT: the returned function captures the current model state.
        Call ``compile_predict()`` again after modifying training data, the
        kernel, noise, or prior parameters. Large exact GP training matrices
        may increase compilation time; call once and reuse the predictor.
        """
        self._ensure_conditioned()
        obs, L, alpha = self._obs, self._L, self._alpha
        params = jnp.asarray(self.kernel.params)
        dim = self.dim

        @jax.jit
        def predictor(X_test):
            X_test = _as_points(X_test, dim)
            return self._predict_core(X_test, obs, L, alpha, params,
                                      return_var=return_var,
                                      full_cov=full_cov,
                                      observation_noise=observation_noise)

        return predictor

    def predict_mean(self, X_test, *, batch_size=None):
        return self.predict(X_test, return_var=False, batch_size=batch_size)

    def predict_gradient(self, X_test, *, return_var=True, full_cov=False):
        """Posterior gradient, shape (n,d). Variance shape (n,d).

        Full covariance uses coordinate-major ordering, shape (n*d,n*d).
        """
        X = _as_points(X_test, self.dim)
        self._ensure_conditioned()
        params = jnp.asarray(self.kernel.params)
        cross = self._grad_cross_covariance(X, self._obs, params)
        mu = self._mean_gradient(X) + (cross @ self._alpha).reshape(self.dim, len(X)).T
        if not return_var:
            return mu
        z = solve_triangular(self._L, cross.T, lower=True)
        if full_cov:
            H = self.kernel.ddK(X, X, params=params)
            Ktt = H.transpose(2, 0, 3, 1).reshape(self.dim * len(X), -1)
            cov = Ktt - z.T @ z
            return mu, (cov + cov.T)/2
        diag = self._gradient_diagonal(X, params)
        var = jnp.maximum(diag - jnp.sum(z*z, axis=0), 0.0)
        return mu, var.reshape(self.dim, len(X)).T

    def predict_std(self, X_test, *, batch_size=None, observation_noise=0.):
        """Posterior standard deviation (sqrt of latent variance by default)."""
        return jnp.sqrt(self.predict(X_test, observation_noise=observation_noise,
                                     batch_size=batch_size)[1])

    def sample_posterior(self, key, X_test, *, n_samples=1, observation_noise=0.):
        """Draw joint posterior samples, shape (n_samples, n_test).

        Uses the full covariance, so this is intended for moderate query sets.
        """
        X = _as_points(X_test, self.dim)
        if n_samples < 1:
            raise ValueError('n_samples must be positive')
        mu, cov = self.predict(X, full_cov=True, observation_noise=observation_noise)
        L = jnp.linalg.cholesky(cov + self.jitter*jnp.eye(len(X), dtype=X.dtype))
        z = jax.random.normal(key, (n_samples, len(X)), dtype=X.dtype)
        return mu[None, :] + z @ L.T

    def negative_log_marginal_likelihood(self, params=None, *, noise=None, prior_params=None):
        params = jnp.asarray(self.kernel.params if params is None else params)
        obs, L, alpha = self._condition_for(params, noise=noise, prior_params=prior_params)
        residual = obs.y - obs.mean
        nll = (0.5 * (residual @ alpha)
               + jnp.sum(jnp.log(jnp.diag(L)))
               + 0.5 * residual.size * jnp.log(2.0 * jnp.pi))
        return jnp.where(jnp.isfinite(nll), nll, 1e30)

    def log_marginal_likelihood(self):
        self._ensure_conditioned()
        residual = self._obs.y - self._obs.mean
        return -(0.5 * (residual @ self._alpha)
                 + jnp.log(jnp.diag(self._L)).sum()
                 + 0.5 * residual.size * jnp.log(2.0*jnp.pi))

    def _noise_parameters(self):
        """Scalar noise hyperparameters to fit. A heteroscedastic GP has none."""
        noise = jnp.asarray(self.noise)
        if noise.ndim != 0:
            raise ValueError("Optimizing heteroscedastic noise requires a user-defined parameterization")
        return noise.reshape(1)

    def _assign_noise_parameters(self, params):
        self.noise = params[0]

    def _prior_parameters(self):
        prior = self.mean_fn
        if not all(hasattr(prior, name) for name in
                   ('get_params', 'set_params', 'prior_f_params', 'prior_grad_f_params')):
            raise ValueError('mean_fn does not implement the trainable prior protocol')
        parameters = jnp.asarray(prior.get_params()).reshape(-1)
        if not parameters.size:
            raise ValueError('mean_fn has no trainable prior parameters')
        return parameters

    def _assign_prior_parameters(self, params):
        if self.mean_fn is None or not hasattr(self.mean_fn, 'set_params'):
            raise ValueError('mean_fn has no trainable prior')
        self.mean_fn.set_params(params)

    def _prior_bounds(self):
        if self.mean_fn is not None and hasattr(self.mean_fn, 'parameter_bounds'):
            return self.mean_fn.parameter_bounds()
        # For custom trainable mean functions, parameters are unrestricted
        # unless the user supplies explicit optimizer bounds.
        n = len(self._prior_parameters())
        return np.full(n, -np.inf), np.full(n, np.inf)

    def fit_hyperparameters(self, *, fit_noise=True, fit_prior=False,
                            lower_bound_kparams=1e-4, upper_bound_kparams=50.,
                            lower_bound_noise=1e-8, upper_bound_noise=50.,
                            lower_bound_prior=None, upper_bound_prior=None,
                            maxiter=300, tol=1e-9):
        """Optimize kernel/noise/prior by exact marginal likelihood with L-BFGS-B.

        Uses SciPy for the optimizer and JAX value_and_grad for derivatives.
        Prior optimization requires a subclass with differentiable pure prior
        evaluation and _prior_parameters/_assign_prior_parameters support.
        """
        from scipy.optimize import minimize

        k0 = jnp.asarray(self.kernel.params).reshape(-1)
        noise0 = self._noise_parameters() if fit_noise else jnp.empty((0,), dtype=k0.dtype)
        prior0 = self._prior_parameters() if fit_prior else jnp.empty((0,), dtype=k0.dtype)
        starts = jnp.concatenate([k0, noise0, prior0])
        n_k, n_n = len(k0), len(noise0)
        if fit_prior:
            if lower_bound_prior is None or upper_bound_prior is None:
                auto_lower, auto_upper = self._prior_bounds()
            else:
                auto_lower, auto_upper = None, None
            if lower_bound_prior is None:
                lower_bound_prior = auto_lower
            if upper_bound_prior is None:
                upper_bound_prior = auto_upper
        else:
            lower_bound_prior, upper_bound_prior = [], []
        lb = np.concatenate([np.broadcast_to(lower_bound_kparams, k0.shape),
                             np.broadcast_to(lower_bound_noise, noise0.shape),
                             np.broadcast_to(lower_bound_prior, prior0.shape)])
        ub = np.concatenate([np.broadcast_to(upper_bound_kparams, k0.shape),
                             np.broadcast_to(upper_bound_noise, noise0.shape),
                             np.broadcast_to(upper_bound_prior, prior0.shape)])
        if np.any(lb >= ub):
            raise ValueError("Upper parameter bounds must exceed lower bounds")

        def objective(theta):
            kp = theta[:n_k]
            nv = theta[n_k:n_k+n_n] if fit_noise else None
            pv = theta[n_k+n_n:] if fit_prior else None
            return self.negative_log_marginal_likelihood(kp,
                                                          noise=nv[0] if fit_noise and n_n == 1 else nv,
                                                          prior_params=pv)

        value_grad = jax.jit(jax.value_and_grad(objective))
        def fun(theta):
            value, grad = value_grad(jnp.asarray(theta, dtype=starts.dtype))
            return float(value), np.asarray(grad, dtype=np.float64)

        t0 = perf_counter()
        res = minimize(fun, np.asarray(starts, dtype=np.float64), jac=True,
                       method="L-BFGS-B", bounds=list(zip(lb, ub)),
                       options={"maxiter": int(maxiter), "ftol": float(tol)})
        elapsed_optimization = perf_counter() - t0
        if not np.isfinite(res.fun):
            raise RuntimeError("Hyperparameter optimization produced nonfinite likelihood")
        self.kernel.set_params(jnp.asarray(res.x[:n_k], dtype=starts.dtype))
        if fit_noise:
            self._assign_noise_parameters(jnp.asarray(res.x[n_k:n_k+n_n], dtype=starts.dtype))
        if fit_prior:
            self._assign_prior_parameters(jnp.asarray(res.x[n_k+n_n:], dtype=starts.dtype))
        self.invalidate()
        self.condition()
        self.fit_time = elapsed_optimization
        self.optimization_result = res
        return res


GaussianProcess = GP
