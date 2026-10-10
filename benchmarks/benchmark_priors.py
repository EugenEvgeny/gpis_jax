"""Compare optimized analytic geometric-prior gradients with reverse-mode autodiff.

Run: python -m benchmarks.benchmark_priors
Times exclude compilation, JIT cache warmup and host-to-device transfer.
This is a microbenchmark, not representative of full GP inference cost.
"""
from time import perf_counter

import jax
import jax.numpy as jnp
from geometric_priors import EllipsoidPrior, PlanePatchPrior


def compare(prior, X, repeats=20):
    p = prior.get_params()
    analytic = jax.jit(lambda x, pv: prior.prior_grad_f_params(x, pv))
    autodiff = jax.jit(lambda x, pv: jax.vmap(jax.grad(
        lambda q: prior.prior_f_params(q[None, :], pv)[0]))(x))
    a = analytic(X, p).block_until_ready()
    b = autodiff(X, p).block_until_ready()
    worst_error = float(jnp.max(jnp.abs(a-b)))

    def timed(fn):
        sample = []
        for _ in range(repeats):
            t0 = perf_counter()
            fn(X, p).block_until_ready()
            sample.append((perf_counter()-t0)*1000)
        return sorted(sample)[len(sample)//2]

    t_a = timed(analytic)
    t_b = timed(autodiff)
    print(f'{type(prior).__name__:18s}  analytic {t_a:8.3f} ms   '
          f'autodiff {t_b:8.3f} ms   speedup {t_b/t_a:5.2f}x   '
          f'max |error| {worst_error:.2e}')


if __name__=='__main__':
    X = jax.random.normal(jax.random.key(15), (1000,3))
    compare(EllipsoidPrior(dim=3, theta=[.2,-.3,.4],a=.8,b=1.,c=1.2), X)
    compare(PlanePatchPrior(dim=3, theta=[-.3,.2,.1],a=1.,b=1.1), X)
