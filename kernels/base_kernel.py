import jax
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial

from abc import ABC
from ._spatial import radius_pairs_padded, radius_pairs_kdtree
from jax.experimental import sparse as jsparse


class BaseKernel(ABC):
    """
    Base class for kernels k(x, y).

    Distance cutoff
    ---------------
    All dense methods (K, dK, dK_y, ddK, ddKxx, dddKxxx) and the combined
    matrix constructors (K_mat_comb, K_mat_comb_x, K_mat_mixed, K_mat_partial,
    K_tri) accept an optional ``cutoff: float | None`` keyword.  When given,
    entries with ||X1[i] - X2[j]|| > cutoff are zeroed out via ``jnp.where``.

    * For compactly-supported kernels (e.g. TPS, Wendland) where the analytic
      form already vanishes outside the support, the cutoff is *exact* and
      ``C^1`` at the boundary.
    * For full-support kernels (e.g. RBF), the cutoff turns the kernel into
      an approximate compact-support variant.  The mask is constant w.r.t.
      kernel parameters, so it does not interfere with autodiff over params,
      but the analytic ``dk_fun`` / ``ddk_fun`` will disagree with the
      autodiff of the masked value at the boundary unless they are also
      zeroed beyond ``cutoff``.

    Each ``cutoff=None`` and ``cutoff=<float>`` call triggers a separate jit
    trace; the ``cutoff=None`` graph contains no distance computation.

    Sparse cutoff
    -------------
    ``K_radius_sparse``, ``dK_radius_sparse``, ``ddK_radius_sparse``
    build a sparse BCOO output using fixed-size padded neighbour lists
    (JAX; still O(n*m) distance memory) or a host-side SciPy KD-tree
    (no dense distance matrix). Choose k_max high enough for all neighbours
    if truncation must be avoided.
    """

    def __init__(self, params):
        self.params = params

        self.analytic_dk = callable(getattr(self, "dk_fun", None))
        self.analytic_ddk = callable(getattr(self, "ddk_fun", None))
        self.analytic_ddkxx = callable(getattr(self, "ddkxx_fun", None))
        self.analytic_dddkxxx = callable(getattr(self, "dddkxxx_fun", None))

        if not self.analytic_dk:
            self.dk_fun = jit(jax.grad(self.k_fun, argnums=0))

        # Gradient w.r.t. second kernel argument (needed e.g. for KSD)
        self.analytic_dk_y = callable(getattr(self, "dk_y_fun", None))
        if not self.analytic_dk_y:
            self.dk_y_fun = jit(jax.grad(self.k_fun, argnums=1))

        if not self.analytic_ddk:
            self.ddk_fun = self.ddk_fun_autodiff

        if not self.analytic_ddkxx:
            self.ddkxx_fun = self.ddkxx_fun_autodiff

        if not self.analytic_dddkxxx:
            self.dddkxxx_fun = self.dddkxxx_fun_autodiff

    def __repr__(self):
        return f"{type(self).__name__}(params={self.params})"

    def set_params(self, params):
        """Update parameters without invalidating compiled computations.

        Public evaluation methods resolve ``params`` *outside* their JIT-compiled
        implementations. In particular, ``set_params`` remains effective even
        after the first compiled call.
        """
        self.params = params

    def _resolve_params(self, X1, params):
        """Effective parameters; median-heuristic kernels override this hook."""
        return self.params if params is None else params

    def _points(self, X1, X2=None):
        X1 = jnp.asarray(X1).reshape(-1, self.dim)
        X2 = X1 if X2 is None else jnp.asarray(X2).reshape(-1, self.dim)
        return X1, X2

    def __call__(self, X1, X2=None, params=None, cutoff=None):
        return self.K(X1, X2, params=params, cutoff=cutoff)

    @staticmethod
    def _sparse_indices(idx):
        """Accept (row, col) arrays or an (nnz, 2) COO index matrix."""
        if isinstance(idx, (tuple, list)) and len(idx) == 2:
            a, b = jnp.asarray(idx[0]), jnp.asarray(idx[1])
            if a.ndim == 1 and b.ndim == 1:
                return a, b
        idx = jnp.asarray(idx)
        if idx.ndim != 2 or idx.shape[1] != 2:
            raise ValueError("idx must be (rows, cols) or an (nnz, 2) array")
        return idx[:, 0], idx[:, 1]

    @staticmethod
    def _sparse_diag(mat):
        """Select the diagonal without densifying a sparse BCOO matrix."""
        equal = mat.indices[:, 0] == mat.indices[:, 1]
        equal = equal.reshape((-1,) + (1,) * (mat.data.ndim - 1))
        return jsparse.BCOO((jnp.where(equal, mat.data, 0), mat.indices), shape=mat.shape)

    # =========================================================================
    # Distance helper
    # =========================================================================
    @partial(jax.jit, static_argnums=(0,))
    def _pairwise_squared_dist(self, X1, X2):
        """Squared distances using O(n*m) temporary storage, not O(n*m*d)."""
        sq = (jnp.sum(X1 * X1, axis=1)[:, None]
              + jnp.sum(X2 * X2, axis=1)[None, :]
              - 2.0 * (X1 @ X2.T))
        return jnp.maximum(sq, 0.0)

    def _pairwise_dist(self, X1, X2):
        return jnp.sqrt(self._pairwise_squared_dist(X1, X2))

    def _cutoff_mask(self, X1, X2, cutoff):
        return (cutoff >= 0) & (self._pairwise_squared_dist(X1, X2) <= cutoff**2)

    # =========================================================================
    # Combined / blocked matrix constructors (thread cutoff through)
    # =========================================================================
    def K_mat_comb(self, X1, X2, params=None, cutoff=None):
        """Hermite block matrix with values then coordinate-major gradients."""
        X1, X2 = self._points(X1, X2)
        p = self._resolve_params(X1, params)
        gram = self.K(X1, X2, p, cutoff=cutoff)
        grad_x = self.dK(X1, X2, p, cutoff=cutoff)
        grad_y = self.dK_y(X1, X2, p, cutoff=cutoff)
        mixed = self.ddK(X1, X2, p, cutoff=cutoff)
        return jnp.concatenate(
            [jnp.concatenate([gram] + [grad_y[..., i] for i in range(self.dim)], axis=1)]
            + [jnp.concatenate([grad_x[..., j]] + [mixed[..., j, i] for i in range(self.dim)], axis=1)
               for j in range(self.dim)], axis=0,
        )

    def K(self, X1, X2=None, params=None, cutoff=None):
        X1, X2 = self._points(X1, X2)
        return self._K_impl(X1, X2, self._resolve_params(X1, params), cutoff)

    @partial(jax.jit, static_argnums=(0,))
    def _K_impl(self, X1, X2, params, cutoff):
        out = vmap(lambda x: vmap(lambda y: self.k_fun(x, y, *params))(X2))(X1)
        if cutoff is not None:
            out = jnp.where(self._cutoff_mask(X1, X2, cutoff), out, 0.0)
        return out

    def dK(self, X1, X2=None, params=None, cutoff=None):
        X1, X2 = self._points(X1, X2)
        return self._dK_impl(X1, X2, self._resolve_params(X1, params), cutoff)

    @partial(jax.jit, static_argnums=(0,))
    def _dK_impl(self, X1, X2, params, cutoff):
        out = vmap(lambda x: vmap(lambda y: self.dk_fun(x, y, *params))(X2))(X1)
        if cutoff is not None:
            out = jnp.where(self._cutoff_mask(X1, X2, cutoff)[..., None], out, 0.0)
        return out

    dK_x = dK

    def ddK(self, X1, X2=None, params=None, cutoff=None):
        X1, X2 = self._points(X1, X2)
        return self._ddK_impl(X1, X2, self._resolve_params(X1, params), cutoff)

    @partial(jax.jit, static_argnums=(0,))
    def _ddK_impl(self, X1, X2, params, cutoff):
        out = vmap(lambda x: vmap(lambda y: self.ddk_fun(x, y, *params))(X2))(X1)
        if cutoff is not None:
            out = jnp.where(self._cutoff_mask(X1, X2, cutoff)[..., None, None], out, 0.0)
        return out

    ddK_x = ddK

    def K_mat_comb_x(self, X1, X2, params=None, cutoff=None):
        X1, X2 = self._points(X1, X2)
        params = self._resolve_params(X1, params)
        gram = self.K(X1, X2, params, cutoff=cutoff)
        grad_y = self.dK_y(X1, X2, params, cutoff=cutoff)
        return jnp.concatenate([gram] + [grad_y[..., i] for i in range(self.dim)], axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def ddk_fun_autodiff(self, x, y, *params):
        def grad_wrt_x(x1, y1, *params):
            return jax.grad(self.k_fun, argnums=0)(x1, y1, *params)

        val = jax.jacfwd(grad_wrt_x, argnums=1)(x, y, *params)
        return val

    @partial(jax.jit, static_argnums=(0,))
    def ddkxx_fun_autodiff(self, x, y, *params):
        # The true Hessian wrt x; -d²/dxdy only holds for stationary kernels.
        return jax.hessian(lambda x_: self.k_fun(x_, y, *params))(x)

    @partial(jax.jit, static_argnums=(0,))
    def dddkxxx_fun_autodiff(self, x, y, *params):
        def k_x(x_):
            return self.k_fun(x_, y, *params)

        # grad: R^d -> R^d
        g_x = jax.grad(k_x)
        # Hessian: R^d -> R^{dxd}
        H_x = jax.jacrev(g_x)
        # Third derivative tensor: R^d -> R^{dxdxd}
        T_x = jax.jacrev(H_x)
        return -T_x(x)

    def K_tri(self, X1, params=None, cutoff=None):
        X1, _ = self._points(X1)
        n = len(X1)
        tri_idx = jnp.triu_indices(n)
        params = self._resolve_params(X1, params)
        Kxx_tri = vmap(lambda i, j: self.k_fun(X1[i], X1[j], *params))(*tri_idx)
        if cutoff is not None:
            r2 = jnp.sum((X1[tri_idx[0]] - X1[tri_idx[1]]) ** 2, axis=-1)
            Kxx_tri = jnp.where((cutoff >= 0) & (r2 <= cutoff**2), Kxx_tri, 0.0)
        return Kxx_tri

    # =========================================================================
    # Sparse interface  (caller-supplied indices)
    # =========================================================================
    def K_sparse(self, X1, X2, idx, sym=False, params=None):
        """Sparse kernel with caller-provided indices (rows, cols) or (nnz,2)."""
        X1, X2 = self._points(X1, X2)
        i, j = self._sparse_indices(idx)
        coords = jnp.stack((i, j), axis=-1)
        values = self.K_vals(X1, X2, (i, j), params)
        mat = jsparse.BCOO((values, coords), shape=(len(X1), len(X2)))
        return mat + mat.T - self._sparse_diag(mat) if sym else mat

    def dK_sparse(self, X1, X2, idx, sym=False, params=None):
        X1, X2 = self._points(X1, X2)
        i, j = self._sparse_indices(idx)
        coords = jnp.stack((i, j), axis=-1)
        values = self.dK_vals(X1, X2, (i, j), params)
        mat = jsparse.BCOO((values, coords), shape=(len(X1), len(X2), self.dim))
        return mat - mat.transpose((1, 0, 2)) if sym else mat

    def ddK_sparse(self, X1, X2, idx, sym=False, params=None):
        X1, X2 = self._points(X1, X2)
        i, j = self._sparse_indices(idx)
        coords = jnp.stack((i, j), axis=-1)
        values = self.ddK_vals(X1, X2, (i, j), params)
        mat = jsparse.BCOO((values, coords), shape=(len(X1), len(X2), self.dim, self.dim))
        return mat + mat.transpose((1, 0, 2, 3)) - self._sparse_diag(mat) if sym else mat

    # =========================================================================
    # Radius-cutoff sparse interface  (auto-compute indices from a cutoff)
    # =========================================================================
    def _radius_idx(self, X1, X2, cutoff, k_max=None, backend="padded"):
        """
        Build (i_idx, j_idx, mask) for pairs within ``cutoff``.

        backend:
            "padded"  : JIT-friendly, fixed-size, requires ``k_max``.
                        Returns mask to drop padding slots.
            "kdtree"  : host-side scipy KDTree, dynamic length, mask is None.
        """
        X1 = X1.reshape(-1, self.dim)
        X2 = X2.reshape(-1, self.dim)
        if backend == "padded":
            if k_max is None:
                raise ValueError("padded backend needs k_max")
            i, j, m = radius_pairs_padded(X1, X2, cutoff, k_max)
            return i, j, m
        elif backend == "kdtree":
            i, j = radius_pairs_kdtree(X1, X2, cutoff)
            return i, j, None
        else:
            raise ValueError(f"unknown backend: {backend}")

    def K_radius_sparse(self, X1, X2, cutoff, k_max=None, sym=False, params=None, backend="padded"):
        """K_sparse where indices are auto-computed from a distance cutoff."""
        X1, X2 = self._points(X1, X2)
        i, j, mask = self._radius_idx(X1, X2, cutoff, k_max, backend)
        s_idx = jnp.stack([i, j], axis=-1)
        vals = self.K_vals(X1, X2, (i, j), params)
        if mask is not None:
            vals = jnp.where(mask, vals, 0.0)
        sparse_mat = jsparse.BCOO((vals, s_idx), shape=(X1.shape[0], X2.shape[0]))
        if sym:
            sparse_mat = sparse_mat + sparse_mat.T - self._sparse_diag(sparse_mat)
        return sparse_mat

    def dK_radius_sparse(self, X1, X2, cutoff, k_max=None, sym=False, params=None, backend="padded"):
        """dK_sparse where indices are auto-computed from a distance cutoff."""
        X1, X2 = self._points(X1, X2)
        i, j, mask = self._radius_idx(X1, X2, cutoff, k_max, backend)
        s_idx = jnp.stack([i, j], axis=-1)
        vals = self.dK_vals(X1, X2, (i, j), params)  # (..., dim)
        if mask is not None:
            vals = jnp.where(mask[:, None], vals, 0.0)
        sparse_mat = jsparse.BCOO((vals, s_idx), shape=(X1.shape[0], X2.shape[0], self.dim))
        if sym:
            sparse_mat = sparse_mat - sparse_mat.transpose((1, 0, 2))
        return sparse_mat

    def ddK_radius_sparse(self, X1, X2, cutoff, k_max=None, sym=False, params=None, backend="padded"):
        """ddK_sparse where indices are auto-computed from a distance cutoff."""
        X1, X2 = self._points(X1, X2)
        i, j, mask = self._radius_idx(X1, X2, cutoff, k_max, backend)
        s_idx = jnp.stack([i, j], axis=-1)
        vals = self.ddK_vals(X1, X2, (i, j), params)  # (..., dim, dim)
        if mask is not None:
            vals = jnp.where(mask[:, None, None], vals, 0.0)
        sparse_mat = jsparse.BCOO(
            (vals, s_idx),
            shape=(X1.shape[0], X2.shape[0], self.dim, self.dim),
        )
        if sym:
            sparse_mat = sparse_mat + sparse_mat.transpose((1, 0, 2, 3)) - self._sparse_diag(sparse_mat)
        return sparse_mat

    # =========================================================================
    # Indexed (vmap over idx) value evaluators
    # =========================================================================
    def K_vals(self, X1, X2, idx, params=None):
        X1, X2 = self._points(X1, X2)
        i, j = self._sparse_indices(idx)
        return self._K_vals_impl(X1, X2, i, j, self._resolve_params(X1, params))

    @partial(jax.jit, static_argnums=(0,))
    def _K_vals_impl(self, X1, X2, i, j, params):
        return vmap(lambda a, b: self.k_fun(X1[a], X2[b], *params))(i, j)

    def dK_vals(self, X1, X2, idx, params=None):
        X1, X2 = self._points(X1, X2)
        i, j = self._sparse_indices(idx)
        return self._dK_vals_impl(X1, X2, i, j, self._resolve_params(X1, params))

    @partial(jax.jit, static_argnums=(0,))
    def _dK_vals_impl(self, X1, X2, i, j, params):
        return vmap(lambda a, b: self.dk_fun(X1[a], X2[b], *params))(i, j)

    def ddK_vals(self, X1, X2, idx, params=None):
        X1, X2 = self._points(X1, X2)
        i, j = self._sparse_indices(idx)
        return self._ddK_vals_impl(X1, X2, i, j, self._resolve_params(X1, params))

    @partial(jax.jit, static_argnums=(0,))
    def _ddK_vals_impl(self, X1, X2, i, j, params):
        return vmap(lambda a, b: self.ddk_fun(X1[a], X2[b], *params))(i, j)

    def dK_y_vals(self, X1, X2, idx, params=None):
        """Gradient wrt the second input on the supplied COO pattern."""
        X1, X2 = self._points(X1, X2)
        i, j = self._sparse_indices(idx)
        return self._dK_y_vals_impl(X1, X2, i, j, self._resolve_params(X1, params))

    @partial(jax.jit, static_argnums=(0,))
    def _dK_y_vals_impl(self, X1, X2, i, j, params):
        return vmap(lambda a, b: self.dk_y_fun(X1[a], X2[b], *params))(i, j)

    def dK_y_sparse(self, X1, X2, idx, params=None):
        """Sparse second-argument gradients, shape (n,m,dim)."""
        X1, X2 = self._points(X1, X2)
        i, j = self._sparse_indices(idx)
        values = self.dK_y_vals(X1, X2, (i, j), params)
        return jsparse.BCOO((values, jnp.stack([i, j], axis=-1)),
                            shape=(len(X1), len(X2), self.dim))

    # =========================================================================
    # Paired evaluators
    # =========================================================================
    def K_vec(self, X1, X2, params=None, cutoff=None):
        X1, X2 = self._points(X1, X2)
        return self._K_vec_impl(X1, X2, self._resolve_params(X1, params), cutoff)

    @partial(jax.jit, static_argnums=(0,))
    def _K_vec_impl(self, X1, X2, params, cutoff):
        values = vmap(lambda x, y: self.k_fun(x, y, *params))(X1, X2)
        if cutoff is not None:
            values = jnp.where((cutoff >= 0) & (jnp.sum((X1-X2)**2, axis=1) <= cutoff**2), values, 0)
        return values

    def dK_vec(self, X1, X2, params=None, cutoff=None):
        X1, X2 = self._points(X1, X2)
        return self._dK_vec_impl(X1, X2, self._resolve_params(X1, params), cutoff)

    @partial(jax.jit, static_argnums=(0,))
    def _dK_vec_impl(self, X1, X2, params, cutoff):
        values = vmap(lambda x, y: self.dk_fun(x, y, *params))(X1, X2)
        if cutoff is not None:
            valid = (cutoff >= 0) & (jnp.sum((X1-X2)**2, axis=1) <= cutoff**2)
            values = jnp.where(valid[:, None], values, 0)
        return values

    def ddKxx(self, X1, X2=None, params=None, cutoff=None):
        X1, X2 = self._points(X1, X2)
        return self._ddKxx_impl(X1, X2, self._resolve_params(X1, params), cutoff)

    @partial(jax.jit, static_argnums=(0,))
    def _ddKxx_impl(self, X1, X2, params, cutoff):
        val = vmap(lambda x: vmap(lambda y: self.ddkxx_fun(x, y, *params))(X2))(X1)
        if cutoff is not None:
            val = jnp.where(self._cutoff_mask(X1, X2, cutoff)[..., None, None], val, 0)
        return val

    def dddKxxx(self, X1, X2=None, params=None, cutoff=None):
        X1, X2 = self._points(X1, X2)
        return self._dddKxxx_impl(X1, X2, self._resolve_params(X1, params), cutoff)

    @partial(jax.jit, static_argnums=(0,))
    def _dddKxxx_impl(self, X1, X2, params, cutoff):
        out = vmap(lambda x: vmap(lambda y: self.dddkxxx_fun(x, y, *params))(X2))(X1)
        if cutoff is not None:
            out = jnp.where(self._cutoff_mask(X1, X2, cutoff)[..., None, None, None], out, 0)
        return out

    # =========================================================================
    # Mixed / partial block matrices
    # =========================================================================
    def K_mat_mixed(self, X1, X2, X3, params=None, cutoff=None):
        """Rectangular Hermite blocks: function rows X1, columns X2;
        gradient rows and columns X3 (ordering is coordinate-major).
        """
        X1, X2 = self._points(X1, X2)
        X3, _ = self._points(X3)
        params = self._resolve_params(X1, params)
        gram = self.K(X1, X2, params, cutoff=cutoff)
        gy = self.dK_y(X1, X3, params, cutoff=cutoff)
        gx = self.dK(X3, X2, params, cutoff=cutoff)
        hxy = self.ddK(X3, X3, params, cutoff=cutoff)
        return jnp.concatenate(
            [jnp.concatenate([gram] + [gy[..., j] for j in range(self.dim)], axis=1)]
            + [jnp.concatenate([gx[..., i]] + [hxy[..., i, j] for j in range(self.dim)], axis=1)
               for i in range(self.dim)], axis=0,
        )

    def K_mat_partial(self, X1, X2, params=None, cutoff=None):
        """Kernel at X1 combined with gradient observations at X2."""
        X1, X2 = self._points(X1, X2)
        params = self._resolve_params(X1, params)
        gram = self.K(X1, X1, params, cutoff=cutoff)
        gy = self.dK_y(X1, X2, params, cutoff=cutoff)
        gx = self.dK(X2, X1, params, cutoff=cutoff)
        hxy = self.ddK(X2, X2, params, cutoff=cutoff)
        return jnp.concatenate(
            [jnp.concatenate([gram] + [gy[..., j] for j in range(self.dim)], axis=1)]
            + [jnp.concatenate([gx[..., i]] + [hxy[..., i, j] for j in range(self.dim)], axis=1)
               for i in range(self.dim)], axis=0,
        )

    def dKx_sparse(self, X1, X2, s_idx, params=None):
        """Sparse [k, ∂k/∂y] matrix on an arbitrary COO pattern."""
        X1, X2 = self._points(X1, X2)
        params = self._resolve_params(X1, params)
        k = self.K_sparse(X1, X2, s_idx, params=params)
        gy = self.dK_y_sparse(X1, X2, s_idx, params=params)
        return jsparse.bcoo_concatenate(
            [k] + [gy[:, :, i] for i in range(self.dim)], dimension=1
        )

    def dKf_dx_mat(self, x, X_f, X_N, cutoff=None):
        """Gradient of [k(x,X_f), ∇_y k(x,X_N)] wrt x."""
        x, _ = self._points(x)
        gx = self.dK(x, X_f, cutoff=cutoff)
        hxy = self.ddK(x, X_N, cutoff=cutoff)
        return jnp.concatenate([gx] + [hxy[:, :, :, i] for i in range(self.dim)], axis=1).transpose(0, 2, 1)

    # =========================================================================
    # Sanity tests (analytic vs autodiff), optionally exercising cutoff branch
    # =========================================================================
    def test_dk(self, dim=None, n=100, m=200, cutoff=None):
        dim = self.dim if dim is None else dim
        key = jax.random.key(0)
        x = jax.random.normal(key, (dim,))
        y = jax.random.normal(jax.random.split(key, 2)[1], (dim,))
        k_fn = self.k_fun
        dk_autodiff = jax.grad(k_fn, argnums=(0))
        dk_val_autodiff = dk_autodiff(x, y, *self.params)

        dk_val = self.dK(x[None, :], y[None, :], cutoff=cutoff)[0, 0]
        print(f"test analytic dk against autodiff  (cutoff={cutoff})")
        # apply same mask to the autodiff target for a fair comparison
        if cutoff is not None:
            r = jnp.linalg.norm(x - y)
            dk_val_autodiff = jnp.where(r <= cutoff, dk_val_autodiff, 0.0)
        print("max absolute error:", jnp.max(jnp.abs(dk_val_autodiff - dk_val)))

    def test_ddk(self, dim=None, n=100, m=200, cutoff=None):
        dim = self.dim if dim is None else dim
        key = jax.random.key(0)
        x = jax.random.normal(key, (dim,))
        y = jax.random.normal(jax.random.split(key, 2)[1], (dim,))

        ddk_autodiff_val = self.ddk_fun_autodiff(x, y, *self.params)
        ddk_val = self.ddK(x[None, :], y[None, :], cutoff=cutoff)[0, 0]
        print(f"test analytic ddk against autodiff  (cutoff={cutoff})")
        if cutoff is not None:
            r = jnp.linalg.norm(x - y)
            ddk_autodiff_val = jnp.where(r <= cutoff, ddk_autodiff_val, 0.0)
        print("max absolute error:", jnp.max(jnp.abs(ddk_autodiff_val - ddk_val)))

    def test_ddkxx(self, dim=None, n=100, m=200, cutoff=None):
        dim = self.dim if dim is None else dim
        key = jax.random.key(0)
        x = jax.random.normal(key, (dim,))
        y = jax.random.normal(jax.random.split(key, 2)[1], (dim,))

        ddk_autodiff_val = self.ddkxx_fun_autodiff(x, y, *self.params)
        ddk_val = self.ddKxx(x[None, :], y[None, :], cutoff=cutoff)[0, 0]
        print(f"test analytic ddkxx against autodiff  (cutoff={cutoff})")
        if cutoff is not None:
            ddk_autodiff_val = jnp.where(jnp.linalg.norm(x-y) <= cutoff, ddk_autodiff_val, 0.0)
        print("max absolute error:", jnp.max(jnp.abs(ddk_autodiff_val - ddk_val)))

    text_ddkxx = test_ddkxx  # legacy typo retained for existing callers

    def test_third(self, dim=None, n=100, m=200, cutoff=None):
        dim = self.dim if dim is None else dim
        key = jax.random.key(0)
        x = jax.random.normal(key, (dim,))
        y = jax.random.normal(jax.random.split(key, 2)[1], (dim,))

        autodiff_val = self.dddkxxx_fun_autodiff(x, y, *self.params)
        analytic_val = self.dddkxxx_fun(x, y, *self.params)

        print(f"test analytic third derivative against autodiff  (cutoff={cutoff})")
        if cutoff is not None:
            r = jnp.linalg.norm(x - y)
            autodiff_val = jnp.where(r <= cutoff, autodiff_val, 0.0)
            analytic_val = jnp.where(r <= cutoff, analytic_val, 0.0)
        print("max absolute error:", jnp.max(jnp.abs(autodiff_val - analytic_val)))

    def dK_y(self, X1, X2=None, params=None, assume_symmetric=False, cutoff=None):
        """Gradient with respect to the *second* argument, shape (n,m,d).

        ``assume_symmetric`` is an optional optimization for symmetric kernels:
        d_y k(x,y) = d_x k(y,x). It is invalid for nonsymmetric kernels.
        """
        X1, X2 = self._points(X1, X2)
        params = self._resolve_params(X1, params)
        return self._dK_y_impl(X1, X2, params, assume_symmetric, cutoff)

    @partial(jax.jit, static_argnums=(0, 4))
    def _dK_y_impl(self, X1, X2, params, assume_symmetric, cutoff):
        if assume_symmetric:
            out = jnp.swapaxes(self._dK_impl(X2, X1, params, None), 0, 1)
        else:
            out = vmap(lambda x: vmap(lambda y: self.dk_y_fun(x, y, *params))(X2))(X1)
        if cutoff is not None:
            out = jnp.where(self._cutoff_mask(X1, X2, cutoff)[..., None], out, 0)
        return out

    # =========================================================================
    # k-NN sparse interface  (neighbour-list indices, e.g. PointCloud.knn_indices)
    # =========================================================================
    # Mirrors the radius-cutoff sparse interface, but the (i, j) pairs come from
    # a precomputed k-NN neighbour list of shape (n, k) -- typically the
    # `knn_indices` attribute of a point cloud -- instead of a distance cutoff.
    # The neighbour list is converted to flat COO coordinates and the existing
    # *_vals evaluators + BCOO assembly are reused (no new numerics).

    @staticmethod
    def _knn_to_coo(knn_idx, include_diagonal=False, mask=None):
        """Convert an (n, k) neighbour list to flat COO coordinates.

        Parameters
        ----------
        knn_idx : (n, k) int array
            Row i lists the column indices j of its k nearest neighbours.
        include_diagonal : bool
            If True, prepend the (i, i) self-pairs. Use this when ``knn_idx``
            *excludes* self (the usual convention) so the Gram gets a proper
            diagonal; leave False if self is already in the list (else the
            diagonal is summed twice by BCOO).
        mask : (n, k) bool array or None
            False marks padded / dropped slots; their values are forced to 0.

        Returns
        -------
        rows, cols, valid : each length nnz = n*k (+ n if include_diagonal)
        """
        knn_idx = jnp.asarray(knn_idx)
        n, k = knn_idx.shape
        rows = jnp.repeat(jnp.arange(n), k)
        cols = knn_idx.reshape(-1)
        valid = (cols >= 0) if mask is None else jnp.asarray(mask, dtype=bool).reshape(-1) & (cols >= 0)
        cols = jnp.where(valid, cols, 0)
        if include_diagonal:
            d = jnp.arange(n)
            rows = jnp.concatenate([d, rows])
            cols = jnp.concatenate([d, cols])
            valid = jnp.concatenate([jnp.ones((n,), dtype=bool), valid])
        return rows, cols, valid

    def K_knn_sparse(self, X1, knn_idx, X2=None, sym=False, params=None, include_diagonal=True, mask=None):
        """Sparse Gram matrix on a k-NN neighbour list. BCOO shape (n, m).

        ``knn_idx`` : (n, k) int (e.g. ``point_cloud.knn_indices``); entry
        k(X1[i], X2[knn_idx[i, a]]) is evaluated for every (i, a). Assumes the
        list *excludes* self -- ``include_diagonal=True`` then adds the (i, i)
        entries so the result has a Gram diagonal (meaningful when X2 is X1).

        ``sym=True`` uses the same ``M + Mᵀ − diag`` convention as the radius
        methods. Note k-NN is directed, so this DOUBLE-COUNTS mutually-paired
        neighbours; prefer the directed form (sym=False) unless the index set
        was made symmetric beforehand.
        """
        X1, X2 = self._points(X1, X2)
        params = self._resolve_params(X1, params)
        i, j, valid = self._knn_to_coo(knn_idx, include_diagonal, mask)
        n = X1.reshape(-1, self.dim).shape[0]
        m = X2.reshape(-1, self.dim).shape[0]
        vals = jnp.where(valid, self.K_vals(X1, X2, (i, j), params), 0.0)
        sparse_mat = jsparse.BCOO((vals, jnp.stack([i, j], axis=-1)), shape=(n, m))
        if sym:
            sparse_mat = sparse_mat + sparse_mat.T - self._sparse_diag(sparse_mat)
        return sparse_mat

    def dK_knn_sparse(self, X1, knn_idx, X2=None, sym=False, params=None, include_diagonal=True, mask=None):
        """Sparse ∂k/∂x on a k-NN neighbour list. BCOO shape (n, m, dim)."""
        X1, X2 = self._points(X1, X2)
        params = self._resolve_params(X1, params)
        i, j, valid = self._knn_to_coo(knn_idx, include_diagonal, mask)
        n = X1.reshape(-1, self.dim).shape[0]
        m = X2.reshape(-1, self.dim).shape[0]
        vals = jnp.where(valid[:, None], self.dK_vals(X1, X2, (i, j), params), 0.0)
        sparse_mat = jsparse.BCOO((vals, jnp.stack([i, j], axis=-1)), shape=(n, m, self.dim))
        if sym:
            sparse_mat = sparse_mat - sparse_mat.transpose((1, 0, 2))
        return sparse_mat

    def ddK_knn_sparse(self, X1, knn_idx, X2=None, sym=False, params=None, include_diagonal=True, mask=None):
        """Sparse ∂²k/∂x∂y on a k-NN neighbour list. BCOO shape (n, m, dim, dim)."""
        X1, X2 = self._points(X1, X2)
        params = self._resolve_params(X1, params)
        i, j, valid = self._knn_to_coo(knn_idx, include_diagonal, mask)
        n = X1.reshape(-1, self.dim).shape[0]
        m = X2.reshape(-1, self.dim).shape[0]
        vals = jnp.where(valid[:, None, None], self.ddK_vals(X1, X2, (i, j), params), 0.0)
        sparse_mat = jsparse.BCOO((vals, jnp.stack([i, j], axis=-1)), shape=(n, m, self.dim, self.dim))
        if sym:
            sparse_mat = sparse_mat + sparse_mat.transpose((1, 0, 2, 3)) - self._sparse_diag(sparse_mat)
        return sparse_mat

    def K_mat_comb_knn_sparse(self, X1, knn_idx, X2=None, knn_idx_y=None, params=None, include_diagonal=True, mask=None, mask_y=None):
        """Sparse version of ``K_mat_comb`` on a k-NN neighbour list.

        Builds the gradient-enhanced ("Hermite") block matrix of shape
        ((1+d)*n, (1+d)*m)::

            [ K            ∂_y k(...,0)   ...  ∂_y k(...,d-1) ]
            [ ∂_x k(0)     ∂²_xy k(0,0)   ...  ∂²_xy k(0,d-1) ]
            [   ...                                            ]
            [ ∂_x k(d-1)   ∂²_xy k(d-1,0) ...  ∂²_xy k(d-1,d-1)]

        identical in layout to the dense ``K_mat_comb``, but every block lives
        on the k-NN sparsity pattern and the blocks are stitched with
        ``bcoo_concatenate`` (as in ``dKx_sparse``). All values come from one
        pass each of ``K_vals`` / ``dK_vals`` / ``ddK_vals`` -- no dense
        ``(n, m, d)`` BCOO slicing or transpose.

        The ``∂_y`` (top-right) block is the transpose of a ``∂_x`` block, so
        for the self case (``X2 is None``) the result is symmetric and matches
        the dense ``K_mat_comb(X, X)`` restricted to the pattern. For
        rectangular ``X1 != X2`` pass ``knn_idx_y`` = the X2->X1 neighbour list
        (the X2-into-X1 k-NN); it defaults to ``knn_idx`` (correct only when
        X1 is X2).

        ``include_diagonal`` adds the (i, i) entries to every block (assumes the
        neighbour lists exclude self); see ``K_knn_sparse`` for the convention.
        """
        X1, X2 = self._points(X1, X2)
        knn_idx_y = knn_idx if knn_idx_y is None else knn_idx_y
        mask_y = mask if mask_y is None else mask_y
        params = self._resolve_params(X1, params)
        d = self.dim
        n = X1.reshape(-1, d).shape[0]
        m = X2.reshape(-1, d).shape[0]

        # forward pattern for K, dK_x, ddK (rows in X1, cols in X2)
        i, j, valid = self._knn_to_coo(knn_idx, include_diagonal, mask)
        coords = jnp.stack([i, j], axis=-1)
        # dK_y block sits on the transpose: dk(X2[iy], X1[jy]) placed at (jy, iy)
        iy, jy, valid_y = self._knn_to_coo(knn_idx_y, include_diagonal, mask_y)
        coords_yT = jnp.stack([jy, iy], axis=-1)

        k_vals = jnp.where(valid, self.K_vals(X1, X2, (i, j), params), 0.0)
        dkx = jnp.where(valid[:, None], self.dK_vals(X1, X2, (i, j), params), 0.0)
        ddk = jnp.where(valid[:, None, None], self.ddK_vals(X1, X2, (i, j), params), 0.0)
        dky = jnp.where(valid_y[:, None], self.dK_vals(X2, X1, (iy, jy), params), 0.0)

        def blk(v, c):
            return jsparse.BCOO((v, c), shape=(n, m))
        top = jsparse.bcoo_concatenate([blk(k_vals, coords)] + [blk(dky[:, c], coords_yT) for c in range(d)], dimension=1)
        block_rows = [top]
        for r in range(d):
            row = jsparse.bcoo_concatenate([blk(dkx[:, r], coords)] + [blk(ddk[:, r, c], coords) for c in range(d)], dimension=1)
            block_rows.append(row)
        return jsparse.bcoo_concatenate(block_rows, dimension=0)
