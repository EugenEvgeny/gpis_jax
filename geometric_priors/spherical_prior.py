"""Sphere-shaped quadratic implicit function, retaining legacy method names."""
import jax.numpy as jnp

from .base_geometric_prior import (BaseGeometricPrior, _points, _parameters, _finite_bounds)


class SphericalPrior(BaseGeometricPrior):
    """f(x)=c/(2r) * (||x-center||²-r²). Gradient=(c/r)(x-center)."""
    def __init__(self, r=1., c=1., mean=None, dim=2):
        super().__init__(dim=dim, mean=mean)
        self.r = jnp.asarray(r)
        self.c = jnp.asarray(c)

    def get_params(self):
        return jnp.concatenate([self.mean, self.r.reshape(1), self.c.reshape(1)])

    def set_params(self, params):
        p = _parameters(params, self.dim+2, 'SphericalPrior')
        self.mean, self.r, self.c = p[:self.dim], p[-2], p[-1]
        return self

    def parameter_bounds(self):
        return _finite_bounds(self.dim+2, [self.dim, self.dim+1])

    def prior_f_params(self, X, params):
        X = _points(X, self.dim)
        mean, r, c = params[:self.dim], params[-2], params[-1]
        squared = jnp.sum((X-mean)**2, axis=-1)
        return c/(2.*r) * (squared-r*r)

    def prior_grad_f_params(self, X, params):
        X = _points(X, self.dim)
        return params[-1] / params[-2] * (X-params[:self.dim])

    def spherical_prior(self, X, r=None, c=None):
        params = self.get_params()
        params = params.at[-2].set(self.r if r is None else r)
        params = params.at[-1].set(self.c if c is None else c)
        return self.prior_f_params(X, params)

    def spherical_prior_diff(self, X, r=None, c=None):
        params = self.get_params()
        params = params.at[-2].set(self.r if r is None else r)
        params = params.at[-1].set(self.c if c is None else c)
        grad = self.prior_grad_f_params(X, params)
        return grad[0] if jnp.asarray(X).ndim == 1 else grad
