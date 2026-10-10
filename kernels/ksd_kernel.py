"""Score-conditioned Stein kernel wrapper around a BaseKernel."""

import jax.numpy as jnp

from .base_kernel import BaseKernel
from .kernel_operators import ksd2_blocked, stein_fn


class KSDKerne:
    """Pairwise Langevin Stein kernel.

    ``score`` is a batched callable accepting (n,dim) and returning (n,dim).
    The historical class name ``KSDKerne`` is retained for compatibility.
    """

    def __init__(self, kernel: BaseKernel, score, dim=None):
        self.kernel = kernel
        self.score = score
        self.dim = kernel.dim if dim is None else int(dim)

    def __repr__(self):
        return f"KSDKernel(kernel={self.kernel!r})"

    @property
    def params(self):
        return self.kernel.params

    def stein_kernel_matrix(self, X, Y, scores_x, scores_y, params=None,
                            assume_symmetric=True):
        return stein_fn(self.kernel, X, Y, scores_x, scores_y, params,
                        assume_symmetric=assume_symmetric)

    def ksd_matrix(self, X, Y, biased=True, assume_symmetric=True, params=None):
        """Return (n,m) Stein Gram values. ``biased`` retained for compatibility."""
        del biased
        return self.stein_kernel_matrix(X, Y, self.score(X), self.score(Y),
                                        params, assume_symmetric)

    def ksd2_blocked(self, X, scores=None, block=512, biased=True,
                     assume_symmetric=True, params=None):
        if scores is None:
            scores = self.score(X)
        return ksd2_blocked(self.kernel, X, scores, block, biased,
                            assume_symmetric, params)

    def ksd_fun(self, x1, x2):
        """Stein kernel evaluated at two points (scalar)."""
        X = jnp.asarray(x1).reshape(1, self.dim)
        Y = jnp.asarray(x2).reshape(1, self.dim)
        return self.ksd_matrix(X, Y)[0, 0]


KSDKernel = KSDKerne
