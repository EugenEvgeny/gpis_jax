"""Reproduce CPU centering benchmark without command-line arguments."""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

N = 512
REPETITIONS = 15


@jax.jit
def original_center(K):
    n = K.shape[0]
    H = jnp.eye(n, dtype=K.dtype) - jnp.ones((n, n), dtype=K.dtype) / n
    return H @ K @ H


@jax.jit
def optimized_center(K):
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def median_ms(fn, matrix):
    fn(matrix).block_until_ready()  # compile/warm up
    measurements = []
    for _ in range(REPETITIONS):
        start = perf_counter()
        fn(matrix).block_until_ready()
        measurements.append((perf_counter() - start) * 1000)
    return float(np.median(measurements))


def main():
    key = jax.random.key(5)
    A = jax.random.normal(key, (N, N))
    Gram = A @ A.T / N
    old = original_center(Gram).block_until_ready()
    new = optimized_center(Gram).block_until_ready()
    assert jnp.allclose(old, new, rtol=1e-4, atol=1e-5)
    t_old = median_ms(original_center, Gram)
    t_new = median_ms(optimized_center, Gram)
    print(f"JAX {jax.__version__}, {jax.devices()[0]}, dtype={Gram.dtype}, n={N}")
    print(f"H @ K @ H:      {t_old:.3f} ms")
    print(f"Mean centering: {t_new:.3f} ms")
    print(f"Speedup:        {t_old/t_new:.1f}x")


if __name__ == "__main__":
    main()
