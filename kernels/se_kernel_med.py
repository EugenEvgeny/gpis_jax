"""RBF kernel with a dataset-adaptive median bandwidth."""

from ._median import median_bandwidth, pairwise_l2, pairwise_group_dist
from .se_kernel import SEKernel


class SEKernelMed(SEKernel):
    """RBF kernel, setting l_f from the first point set on each call.

    The default ``l_f`` acts as a fallback for degenerate/singleton sets.
    The same effective bandwidth is used for values and all derivatives.
    """

    def _resolve_params(self, X1, params):
        if params is not None:
            # Explicit parameters bypass the automatic heuristic; this also
            # keeps one bandwidth across every block of a Hermite matrix.
            return params
        return (self.sig_f, median_bandwidth(X1, self.l_f))

    def __repr__(self):
        return f"SEKernelMed(sig_f={self.sig_f}, l_f=median)"


RBFKernelMed = SEKernelMed
