"""Compare posterior diagonal calculation with full inverse/matrix products.

This benchmark isolates prediction arithmetic and excludes JIT compilation,
training covariance construction and Cholesky factorization. The baseline is
mathematically equivalent but materializes an m-by-m matrix. Results depend on
hardware/JAX version, so no constant speedup should be assumed.
"""
import time
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def benchmark(n_train=96, n_query=600, n_repeats=10):
    key = jax.random.key(0)
    k1,k2 = jax.random.split(key)
    X = jax.random.normal(k1, (n_train,2))
    Q = jax.random.normal(k2, (n_query,2))
    # RBF kernel, with small positive noise for good conditioning
    def gram(A, B):
        diff = A[:,None,:]-B[None,:,:]
        return jnp.exp(-.5 * jnp.sum(diff*diff,axis=-1))
    K = gram(X,X) + .01*jnp.eye(n_train)
    cross = gram(Q,X)
    diag = jnp.ones((n_query,))
    L = jnp.linalg.cholesky(K)
    inv = jnp.linalg.inv(K)

    @jax.jit
    def old(C, K_inv, k_diag):
        return k_diag-jnp.diag(C @ K_inv @ C.T)

    @jax.jit
    def new(C, L_chol, k_diag):
        v = jax.scipy.linalg.solve_triangular(L_chol,C.T,lower=True)
        return k_diag-jnp.sum(v*v,axis=0)

    a,b=old(cross,inv,diag),new(cross,L,diag)
    a.block_until_ready(); b.block_until_ready()
    err=float(jnp.max(jnp.abs(a-b)))

    def measure(fn):
        samples=[]
        for _ in range(n_repeats):
            t=time.perf_counter()
            fn().block_until_ready()
            samples.append((time.perf_counter()-t)*1000)
        return sorted(samples)[len(samples)//2]
    old_ms,new_ms=measure(lambda:old(cross,inv,diag)),measure(lambda:new(cross,L,diag))
    print(f'Prediction diagonal: n_train={n_train}, n_query={n_query}')
    print(f'max difference: {err:.3e}')
    print(f'old (cached inverse + dense query gram): {old_ms:.3f} ms')
    print(f'new (Cholesky triangular solve):        {new_ms:.3f} ms')
    print(f'ratio old/new: {old_ms/new_ms:.2f}x')
    return old_ms,new_ms,err


if __name__=='__main__':
    benchmark()
