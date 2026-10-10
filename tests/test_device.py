"""CPU execution and CUDA failure-mode tests.

A CUDA smoke test is available through `gp-jax-device --backend cuda --smoke-test`.
We cannot assert CUDA execution unless a physical CUDA device is present.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gaussian_process import get_device, BackendUnavailableError
from gaussian_process.device import smoke_test, main


def test_cpu_backend_and_jit_smoke():
    cpu = get_device("cpu")
    assert cpu.platform == "cpu"
    gp, gpis = smoke_test(cpu)
    assert gp.shape == (3,)
    assert gpis.shape == (2,)
    assert cpu in gp.devices() and cpu in gpis.devices()
    np.testing.assert_array_less(np.abs(np.asarray(gp)), 2.)


def test_auto_backend_is_available():
    auto = get_device("auto")
    assert auto in jax.devices()


def test_invalid_device_requests():
    with pytest.raises(ValueError, match="backend"):
        get_device("tpu-other")
    with pytest.raises(ValueError, match="index"):
        get_device("cpu", -1)
    with pytest.raises(BackendUnavailableError, match="No device"):
        get_device("cpu", 1_000_000)


def test_gpu_backend_never_silently_falls_back_to_cpu(monkeypatch):
    real_devices = jax.devices

    def no_gpu(backend=None):
        if backend == "gpu":
            raise RuntimeError("No GPU backend")
        return real_devices(backend)

    monkeypatch.setattr(jax, "devices", no_gpu)
    with pytest.raises(BackendUnavailableError, match="No usable JAX GPU"):
        get_device("cuda")
    assert main(["--backend", "cuda", "--require-gpu"]) == 2


def test_checker_cli_cpu(capsys):
    assert main(["--backend", "cpu", "--smoke-test"]) == 0
    result = capsys.readouterr()
    assert "selected backend=cpu" in result.out
    assert "device smoke test passed" in result.out


def test_cuda_device_when_available():
    try:
        gpu = get_device("cuda")
    except BackendUnavailableError:
        pytest.skip("No usable JAX CUDA backend on this machine")
    assert gpu.platform in ("gpu", "cuda")
    result, _ = smoke_test(gpu)
    assert gpu in result.devices()


def test_compiled_posterior_matches_eager_cpu():
    from gaussian_process import GP, GPIS
    from kernels import SEKernel

    with jax.default_device(get_device("cpu")):
        X = jnp.array([[-1.], [-.2], [.4], [1.]])
        y = jnp.sin(X[:, 0])
        qp = jnp.array([[0.13], [0.58]])
        gp = GP(SEKernel(dim=1), X, y, noise=0.02)
        mu1, var1 = gp.predict(qp)
        mu2, var2 = gp.compile_predict()(qp)
        np.testing.assert_allclose(mu1, mu2, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(var1, var2, rtol=1e-7, atol=1e-7)

        theta = jnp.arange(8) * 2 * jnp.pi / 8
        pts = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
        gp2 = GPIS(SEKernel(dim=2), pts, pts, augment=False)
        q = jnp.array([[0.5, 0.1], [1.0, 0.0]])
        m0, v0 = gp2.predict(q)
        m1, v1 = gp2.compile_predict()(q)
        np.testing.assert_allclose(m0, m1, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(v0, v1, atol=1e-7, rtol=1e-7)
