"""Compatibility entry point for the supplied legacy SciPy fitting utilities.

Independent of GP/GPIS inference. Optional matplotlib and SciPy requirements
are imported only if this module is explicitly requested.
"""
from .legacy.fit_geom_shape import *  # noqa: F401,F403
