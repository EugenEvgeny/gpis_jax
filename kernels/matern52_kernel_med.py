"""Matérn 5/2 kernel with a dataset-adaptive median bandwidth."""

from ._median import median_bandwidth, pairwise_l2, pairwise_group_dist
from .matern52_kernel import Matern52Kernel


class Matern52KernelMed(Matern52Kernel):
    """Like Matern52Kernel, with l_f chosen via a median heuristic on X1."""

    def _resolve_params(self, X1, params):
        if params is not None:
            # Explicit parameters bypass the automatic heuristic; this also
            # keeps one bandwidth across every block of a Hermite matrix.
            return params
        return (self.sig_f, median_bandwidth(X1, self.l_f))

    def __repr__(self):
        return f"Matern52KernelMed(sig_f={self.sig_f}, l_f=median)"
