"""Regression tests for analytic kernels and dense/sparse interfaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kernels import (
    SEKernel, SEKernelMed, Matern12Kernel, Matern32Kernel,
    Matern52Kernel, Matern52KernelMed, TotalPeriodicKernel,
    EuclideanPeriodicKernel, TPSKernel, KSDKernel, IMQKernel,
)
from kernels.compact_kernels import c_kernel, c_kernel_vec
from kernels.kcme import kcme_coefficients, conditional_probabilities


@pytest.mark.parametrize("kernel", [
    SEKernel(1.5, 0.8, 2), Matern32Kernel(1.5, 0.8, 2),
    Matern52Kernel(1.5, 0.8, 2), IMQKernel(dim=2),
    TPSKernel(2.0, 2), EuclideanPeriodicKernel(dim=2, d=2),
])
def test_analytic_derivatives(kernel):
    x, y = jnp.array([0.4, -0.3]), jnp.array([-0.7, 0.1])
    params = kernel.params
    f = lambda x_, y_: kernel.k_fun(x_, y_, *params)
    np.testing.assert_allclose(kernel.dk_fun(x, y, *params), jax.grad(f, argnums=0)(x, y), rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(kernel.dk_y_fun(x, y, *params), jax.grad(f, argnums=1)(x, y), rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(kernel.ddk_fun(x, y, *params), jax.jacfwd(jax.grad(f, argnums=0), argnums=1)(x, y), rtol=5e-5, atol=2e-6)
    np.testing.assert_allclose(kernel.ddkxx_fun(x, y, *params), jax.hessian(lambda x_: f(x_, y))(x), rtol=5e-5, atol=2e-6)


def test_periodic_derivatives():
    k = TotalPeriodicKernel(sig_f=1.3, l_f=0.7, sig_r=0.5, d=3)
    x, y = jnp.array([0.2]), jnp.array([2.4])
    args = k.params
    np.testing.assert_allclose(k.dk_fun(x, y, *args), jax.grad(lambda x_: k.k_fun(x_, y, *args))(x), rtol=1e-6)
    np.testing.assert_allclose(k.ddk_fun(x, y, *args), jax.jacfwd(jax.grad(lambda x_: k.k_fun(x_, y, *args)))(x) * -1, rtol=1e-6)


def test_diagonal_and_third_derivative():
    x = jnp.array([0.2, -0.6])
    for kernel in (SEKernel(dim=2), Matern12Kernel(dim=2), Matern32Kernel(dim=2), Matern52Kernel(dim=2)):
        np.testing.assert_allclose(kernel.k_fun(x, x, *kernel.params), kernel.params[0] ** 2, atol=1e-12)
    for kernel in (Matern32Kernel(dim=2), Matern52Kernel(dim=2)):
        assert jnp.all(jnp.isfinite(kernel.ddk_fun(x, x, *kernel.params)))
        np.testing.assert_allclose(kernel.dk_fun(x, x, *kernel.params), 0.0)
    k = SEKernel(dim=2)
    f = lambda x_: k.k_fun(x_, jnp.array([-0.4, 0.3]), *k.params)
    true_negative = -jax.jacfwd(jax.hessian(f))(x)
    np.testing.assert_allclose(k.dddkxxx_fun(x, jnp.array([-0.4, 0.3]), *k.params), true_negative, atol=1e-7)


def test_set_params_after_jit_and_explicit_params():
    k = SEKernel(dim=2)
    X = jnp.array([[0., 0.], [0.3, 0.9], [1., 2.]])
    k_before = k.K(X)
    k.set_params([2.0, 0.4])
    k_after = k.K(X)
    assert not jnp.allclose(k_before, k_after)
    np.testing.assert_allclose(k_after[0, 0], 4.0)
    assert not jnp.allclose(k.K(X, params=[1., 1.]), k_after)
    np.testing.assert_allclose(k(X), k_after)
    np.testing.assert_allclose(k.K_vec(X, X), jnp.diag(k_after))
    assert jnp.all(k.K_vec(X, X, cutoff=0.01) > 0)
    assert k.dK(X, params=[1., 1.]).shape == (3, 3, 2)
    np.testing.assert_allclose(k.K_mat_single(X[0], X[1]), k.K_mat_comb(X[0:1], X[1:2]))


def test_dense_cutoff_and_derivative_blocks():
    k = SEKernel(dim=2)
    X = jnp.array([[0., 0.], [0.2, 0.1], [3., 0.]])
    cutoff = 0.5
    assert k.K(X, cutoff=cutoff)[0, 2] == 0
    assert k.dK(X, cutoff=cutoff)[0, 2, 0] == 0
    assert k.ddK(X, cutoff=cutoff)[0, 2, 0, 0] == 0
    np.testing.assert_allclose(k.K_mat_comb(X, X), k.K_mat_comb(X, X).T, atol=1e-12)
    np.testing.assert_allclose(k.K_mat_partial(X[:2], X[2:]), k.K_mat_partial(X[:2], X[2:]).T, atol=1e-12)
    assert k.K_mat_mixed(X[:2], X[:3], X[2:]).shape == (4, 5)
    np.testing.assert_allclose(k.K(X, X)[0], k.K_mat_comb_x(X, X)[:3, :3][0])


def test_sparse_indices_and_knn():
    k = SEKernel(dim=2)
    X = jnp.array([[0., 0.], [0.3, 0.5], [0.1, 1.]])
    i = jnp.array([0, 0, 1, 2])
    j = jnp.array([0, 1, 1, 2])
    reference = np.zeros((3, 3))
    reference[np.asarray(i), np.asarray(j)] = np.asarray(k.K(X))[np.asarray(i), np.asarray(j)]
    np.testing.assert_allclose(k.K_sparse(X, X, (i, j)).todense(), reference)
    np.testing.assert_allclose(k.K_sparse(X, X, jnp.stack([i, j], axis=-1)).todense(), reference)
    np.testing.assert_allclose(k.K_sparse(X, X, (i, j), sym=True).todense(), reference + reference.T-np.diag(np.diag(reference)))
    assert k.dK_sparse(X, X, (i, j)).shape == (3, 3, 2)
    assert k.ddK_sparse(X, X, (i, j)).shape == (3, 3, 2, 2)
    knn = jnp.array([[1, 2], [0, 2], [0, 1]])
    np.testing.assert_allclose(k.K_knn_sparse(X, knn).todense(), k.K(X), atol=1e-12)
    np.testing.assert_allclose(k.K_mat_comb_knn_sparse(X, knn).todense(), k.K_mat_comb(X, X), atol=1e-12)


def test_radius_sparse_padded_and_host():
    k = SEKernel(dim=2)
    X = jnp.array([[0., 0.], [0.2, 0.4], [3., 0.]])
    reference = k.K(X, cutoff=0.6)
    for backend in ('padded', 'kdtree'):
        kwargs = {'k_max': 3} if backend == 'padded' else {}
        np.testing.assert_allclose(k.K_radius_sparse(X, X, cutoff=0.6, backend=backend, **kwargs).todense(), reference)
        np.testing.assert_allclose(k.dK_radius_sparse(X, X, cutoff=0.6, backend=backend, **kwargs).todense(), k.dK(X, cutoff=0.6))
        np.testing.assert_allclose(k.ddK_radius_sparse(X, X, cutoff=0.6, backend=backend, **kwargs).todense(), k.ddK(X, cutoff=0.6))


def test_median_cross_covariance_and_derivatives():
    X = jnp.array([[0., 0.], [0.3, 0.5], [1.1, 2.1], [2., 1.]])
    Y = jnp.array([[0.2, 0.3], [0.8, 0.6]])
    for k in (SEKernelMed(dim=2), Matern52KernelMed(dim=2)):
        assert k.K(X, Y).shape == (4, 2)
        assert k.dK(X, Y).shape == (4, 2, 2)
        assert k.dK_y(X, Y).shape == (4, 2, 2)
        assert k.ddK(X, Y).shape == (4, 2, 2, 2)
        assert k.K_mat_comb(X, Y).shape == (12, 6)
        assert jnp.all(jnp.isfinite(k.K(X[:1], Y)))
        assert jnp.all(jnp.isfinite(k.K(jnp.repeat(X[:1], 4, axis=0), Y)))
        explicit = k.K(X, Y, params=[1.0, 3.0])
        np.testing.assert_allclose(explicit[0,0], k.k_fun(X[0], Y[0], 1.0, 3.0))


def test_ksd_blocked_nonmultiple():
    X = jnp.array([[0.1, 0.2], [0.4, -0.6], [-1.2, 0.3], [0.8, 1.1], [0.9, -0.9]])
    kernel = SEKernel(dim=2)
    score = lambda samples: -samples
    wrapper = KSDKernel(kernel, score)
    U = wrapper.ksd_matrix(X, X)
    np.testing.assert_allclose(wrapper.ksd_fun(X[0], X[1]), U[0, 1])
    np.testing.assert_allclose(kernel.ksd2(X, lambda x: -x, unbiased=False), U.mean(), atol=1e-10)
    np.testing.assert_allclose(wrapper.ksd2_blocked(X, block=2, biased=True), U.mean(), atol=1e-10)
    np.testing.assert_allclose(wrapper.ksd2_blocked(X, block=2, biased=False),
                               (U.sum()-jnp.trace(U))/(len(X)*(len(X)-1)), atol=1e-10)


def test_mmd_hsic_kcme_and_compact():
    X = jnp.array([[0.1], [0.4], [0.9], [1.4]])
    Y = jnp.array([[1.4], [0.4], [0.9], [1.4]])
    k = SEKernel(dim=1)
    assert jnp.abs(k.mmd2(X, X)) < 1e-10
    # U-statistics remove same-sample diagonals and can be negative, even
    # when comparing a finite dataset with itself.
    assert k.mmd2(X, X, unbiased=True) < 0
    K, L = k.K(X), k.K(Y)
    n = len(X)
    H = jnp.eye(n)-jnp.ones((n,n))/n
    expected = jnp.sum((H@K@H)*(H@L@H))/(n*n)
    np.testing.assert_allclose(k.cross_covariance(X,Y), expected, atol=1e-12)
    assert kcme_coefficients(X, Y, k).shape == (n, n)
    P, classes = conditional_probabilities(X, jnp.array([0,1,0,1]),Y,k)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert classes.shape == (2,)
    np.testing.assert_allclose(c_kernel(X, X, 1.0), c_kernel(X, X, 1.0).T)
    np.testing.assert_allclose(c_kernel_vec(X, X, 1.0), jnp.diag(c_kernel(X, X, 1.0)))
    assert c_kernel(X[:1], Y[-1:], 0.1)[0,0] == 0


def test_negative_cutoff_and_shapes():
    k = SEKernel(dim=2)
    X = jnp.array([[0.,0.], [1., 0.]])
    np.testing.assert_allclose(k.K(X, cutoff=-0.2), 0)
    np.testing.assert_allclose(k.K_vec(X,X,cutoff=-0.2), 0)
    np.testing.assert_allclose(k.K_radius_sparse(X,X,cutoff=-0.2,k_max=2).todense(), 0)
    with pytest.raises(ValueError):
        k.K_radius_sparse(X,X,cutoff=0.5,k_max=None)


def test_second_argument_sparse_and_function_gradient_matrix():
    k = EuclideanPeriodicKernel()
    X = jnp.array([[0.9,0.4],[0.2,1.0],[0.8,-0.7]])
    idx = (jnp.repeat(jnp.arange(3), 3), jnp.tile(jnp.arange(3), 3))
    np.testing.assert_allclose(k.dK_y_sparse(X,X,idx).todense(),k.dK_y(X,X),atol=1e-10)
    np.testing.assert_allclose(k.dKx_sparse(X,X,idx).todense(),k.K_mat_comb_x(X,X),atol=1e-10)
    value = k.dKf_dx_mat(X[:1], X, X)
    reference = jax.jacfwd(lambda x: k.K_mat_comb_x(x[None,:],X).ravel())(X[0]).T
    assert value.shape == (1,2,3+3*2)
    np.testing.assert_allclose(value[0],reference,atol=1e-7)


def test_parameter_gradient_after_compilation():
    kernel = SEKernel(dim=1)
    X = jnp.array([[0.2],[0.5],[0.7]])
    result = jax.grad(lambda l: kernel.K(X,params=[1.0,l]).sum())(jnp.array(0.8))
    assert jnp.isfinite(result) and result > 0


def test_legacy_median_helpers_and_amcmd():
    from kernels.se_kernel_med import pairwise_l2
    from kernels.kernel_operators import amcmd2, mcmd2, jmmd2
    x = jnp.array([[0.], [1.], [2.]])
    y = jnp.array([[0.], [1.]])
    np.testing.assert_allclose(pairwise_l2(x, y), jnp.abs(x-y.T))
    kernel = SEKernel(dim=1)
    assert jnp.isfinite(amcmd2(x, x, y, y, kernel, kernel))
    assert mcmd2(x, x, y, y, x, kernel, kernel).shape == (3,)
    assert jnp.isfinite(jmmd2(x,x,y,y,kernel,kernel))


def test_matern52_third_derivative_at_diagonal():
    k = Matern52Kernel(sig_f=1.1, l_f=0.9, dim=2)
    x = jnp.array([0.3, -0.4])
    y = jnp.array([-0.2, 1.3])
    expected = -jax.jacfwd(jax.hessian(lambda x_: k.k_fun(x_, y, *k.params)))(x)
    np.testing.assert_allclose(k.dddkxxx_fun(x,y,*k.params),expected,atol=2e-6,rtol=1e-6)
    np.testing.assert_allclose(k.dddkxxx_fun(x,x,*k.params),jnp.zeros((2,2,2)),atol=1e-12)
    assert jnp.all(jnp.isfinite(k.dddKxxx(x[None,:],x[None,:])))
