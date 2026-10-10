"""JAX Gaussian-process regression and geometric implicit surfaces."""
from .gp import GP, GaussianProcess
from .gpis import GPIS
from .priors import (BaseGeometricPrior, ZeroPrior, EllipsoidPrior, SphericalPrior,
                     PlanePatchPrior, MultiPlanePatchPrior, AnchorPrior, RadialPrior,
                     GPRadialPrior)
from .statistics import normal_interval_logprob, normal_interval_prob_stable

__all__ = ['GP', 'GaussianProcess', 'GPIS', 'BaseGeometricPrior', 'ZeroPrior',
           'EllipsoidPrior', 'SphericalPrior', 'PlanePatchPrior',
           'MultiPlanePatchPrior', 'AnchorPrior', 'RadialPrior', 'GPRadialPrior',
           'normal_interval_logprob', 'normal_interval_prob_stable',
           'get_device', 'BackendUnavailableError']


def __getattr__(name):
    # Keep `python -m gaussian_process.device` free of runpy's eager-import warning.
    if name in ("get_device", "BackendUnavailableError"):
        from .device import get_device, BackendUnavailableError
        return {"get_device": get_device,
                "BackendUnavailableError": BackendUnavailableError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
