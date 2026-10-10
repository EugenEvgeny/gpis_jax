"""Zero mean prior with no trainable parameters."""
import jax.numpy as jnp

from .base_geometric_prior import BaseGeometricPrior, _points, _parameters


class ZeroPrior(BaseGeometricPrior):
    def __init__(self, dim=None, **kwargs):
        # Legacy ZeroPrior allowed omitting dim; infer it from the query if needed.
        if dim is None:
            self.dim = None
            self.mean = None
            self.theta = None
        else:
            super().__init__(dim=dim, **kwargs)

    def get_params(self):
        return jnp.empty((0,))

    def set_params(self, params):
        _parameters(params, 0, 'ZeroPrior')
        return self

    def prior_f_params(self, X, params):
        X = _points(X, self.dim)
        return jnp.zeros((len(X),), dtype=X.dtype)

    def prior_grad_f_params(self, X, params):
        return jnp.zeros_like(_points(X, self.dim))
