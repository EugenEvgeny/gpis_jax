"""Backend selection and execution checks for the JAX GP/GPIS package.

CUDA support is selected when INSTALLING JAX (via ``uv sync --extra cuda13``
for example), not by importing a separate GPU implementation of the models.
Use :func:`get_device` and ``jax.default_device(...)`` when explicit device
selection is needed; otherwise JAX will choose its normal default backend.

``gp-jax-device --backend cuda --smoke-test`` exits nonzero rather than
silently running on CPU if a usable GPU is absent.
"""

from __future__ import annotations

import argparse
import sys

import jax


class BackendUnavailableError(RuntimeError):
    """Raised when the requested accelerator is not usable by JAX."""


def get_device(backend: str = "auto", index: int = 0):
    """Return a JAX device, or fail explicitly for unavailable CUDA.

    Parameters
    ----------
    backend:
        ``'auto'`` uses JAX's default device; ``'cpu'`` selects the CPU;
        ``'gpu'`` or ``'cuda'`` requests a GPU and *never* falls back to CPU.
    index:
        Index within the devices exposed by the selected backend.

    Use it as ``with jax.default_device(get_device('cuda')): ...`` and create
    model inputs inside that context for consistent device placement.
    """
    backend = backend.lower()
    if backend not in ("auto", "cpu", "gpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', 'gpu', or 'cuda'")
    if index < 0:
        raise ValueError("index must be nonnegative")
    try:
        if backend == "auto":
            devices = jax.devices()
        else:
            devices = jax.devices("cpu" if backend == "cpu" else "gpu")
    except (RuntimeError, ValueError) as exc:
        if backend in ("gpu", "cuda"):
            raise BackendUnavailableError(
                "No usable JAX GPU backend. On Linux with an NVIDIA GPU, "
                "install `uv sync --extra cuda13` (or cuda12 for an older "
                "driver/GPU), check `nvidia-smi`, and verify CUDA/driver "
                "compatibility. See docs/INSTALL_UV_CUDA.md."
            ) from exc
        raise BackendUnavailableError(f"JAX cannot initialize backend {backend!r}: {exc}") from exc
    if index >= len(devices):
        raise BackendUnavailableError(
            f"No device {index} for backend {backend!r}; found {len(devices)}"
        )
    chosen = devices[index]
    if backend in ("gpu", "cuda") and chosen.platform not in ("gpu", "cuda"):
        raise BackendUnavailableError(f"Requested GPU but received {chosen.platform!r}")
    return chosen


def smoke_test(device):
    """Run a JIT-compiled numerical GP and GPIS prediction on ``device``.

    Creates all inputs inside a default-device context and checks that the
    prediction outputs reside on the requested device. Raises on mismatch.
    Returns the GP and GPIS mean arrays for optional numerical inspection.
    """
    import jax.numpy as jnp

    from kernels import SEKernel
    from .gp import GP
    from .gpis import GPIS

    with jax.default_device(device):
        x = jnp.linspace(-1.0, 1.0, 7)[:, None]
        y = jnp.sin(x[:, 0])
        q = jnp.array([[-0.3], [0.1], [0.7]])
        gp = GP(SEKernel(dim=1), x, y, noise=0.01)
        gp.condition()
        # Cache validation stays outside tracing via the public snapshot API.
        gp_mean, gp_variance = gp.compile_predict()(q)
        theta = jnp.arange(8) * (2.0 * jnp.pi / 8.0)
        circle = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
        gpis = GPIS(SEKernel(dim=2), circle, circle, augment=False,
                    sig_x=0.01, sig_n=0.02)
        gpis.condition()
        gpis_mean, gpis_variance = gpis.compile_predict()(
            jnp.array([[0.25, 0.25], [1.0, 0.0]])
        )
        outputs = (gp_mean, gp_variance, gpis_mean, gpis_variance)
        for array in outputs:
            array.block_until_ready()
            if device not in array.devices():
                raise RuntimeError(
                    f"Computation executed on {array.devices()}, expected {device}"
                )
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in outputs):
            raise RuntimeError("Non-finite GP/GPIS output during device smoke test")
    return gp_mean, gpis_mean


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check JAX backend / optional CUDA installation for GP/GPIS"
    )
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu", "cuda"),
                        default="auto", help="Select device (GPU never falls back to CPU)")
    parser.add_argument("--index", type=int, default=0,
                        help="Device index within selected backend")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Fail if the chosen JAX device is not a GPU")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a GP/GPIS JIT prediction on the selected device")
    args = parser.parse_args(argv)
    try:
        dev = get_device(args.backend, args.index)
        if args.require_gpu and dev.platform not in ("gpu", "cuda"):
            raise BackendUnavailableError(
                f"GPU required but selected device is {dev.platform!r}. "
                "Install a CUDA extra and check the NVIDIA driver."
            )
        print(f"JAX {jax.__version__}; selected backend={dev.platform}; device={dev}")
        if args.smoke_test:
            gp_mean, gpis_mean = smoke_test(dev)
            print(f"GP + GPIS device smoke test passed on {dev.platform} "
                  f"({gp_mean.size} + {gpis_mean.size} predictions)")
    except (BackendUnavailableError, RuntimeError, ValueError) as exc:
        print(f"JAX backend check FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
