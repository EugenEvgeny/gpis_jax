"""JAX covariance kernels, derivatives, sparse kernels and RKHS operators."""

from .base_kernel import BaseKernel
from .euclidean_periodic_kernel import EuclideanPeriodicKernel
from .inverse_log_kernel import InverseLogKernel
from .iqm_kernel import IMQKernel
from .ksd_kernel import KSDKerne, KSDKernel
from .matern12_kernel import Matern12Kernel
from .matern32_kernel import Matern32Kernel
from .matern52_kernel import Matern52Kernel
from .matern52_kernel_med import Matern52KernelMed
from .periodic_kernel import TotalPeriodicKernel
from .se_kernel import RBFKernel, SEKernel
from .se_kernel_med import RBFKernelMed, SEKernelMed
from .tp_kernel import TPSKernel

# Importing also installs RKHS statistics such as kernel.mmd2 on BaseKernel.
from . import kernel_operators

__all__ = [
    "BaseKernel", "EuclideanPeriodicKernel", "InverseLogKernel", "IMQKernel",
    "KSDKerne", "KSDKernel", "Matern12Kernel", "Matern32Kernel",
    "Matern52Kernel", "Matern52KernelMed", "TotalPeriodicKernel",
    "SEKernel", "SEKernelMed", "RBFKernel", "RBFKernelMed", "TPSKernel",
]

# The upstream project includes additional Lie-group kernels, but the uploaded
# archive does not contain the ``lie/`` source directory. Import them only when
# available, rather than breaking the standard Euclidean kernel package.
try:
    from .lie.lie_kernels import (
        LieGeodesicRBF, LieGeodesicRBFMed, LieTraceKernel, LieTraceKernelMed,
        LieTraceKernelMod, LieTraceKernelNorm, LieTraceKernelNormMed,
        LocalTangentGaussianKernel, SE2DistanceKernel, SE3DistanceKernel,
    )
except ModuleNotFoundError as exc:
    if exc.name not in (f"{__name__}.lie", f"{__name__}.lie.lie_kernels"):
        raise
else:
    from .lie.lie_se2_distance_matern_kernel import SE2DistanceMatern32Kernel, SE2DistanceMatern52Kernel
    from .lie.lie_se2_distance_rbf_kernel import SE2DistanceRBFKernel
    from .lie.lie_se3_distance_matern_kernel import SE3DistanceMatern32Kernel, SE3DistanceMatern52Kernel
    from .lie.lie_se_kernel import LieSEKernel
    from .lie.lie_se_kernel_med import LieSEKernelMed
    from .lie.lie_se_matern52 import LieSEMatern52Kernel
    __all__ += [
        "LieGeodesicRBF", "LieGeodesicRBFMed", "LieTraceKernel", "LieTraceKernelMed",
        "LieTraceKernelMod", "LieTraceKernelNorm", "LieTraceKernelNormMed",
        "LocalTangentGaussianKernel", "SE2DistanceKernel", "SE3DistanceKernel",
        "SE2DistanceMatern32Kernel", "SE2DistanceMatern52Kernel", "SE2DistanceRBFKernel",
        "SE3DistanceMatern32Kernel", "SE3DistanceMatern52Kernel", "LieSEKernel",
        "LieSEKernelMed", "LieSEMatern52Kernel",
    ]
