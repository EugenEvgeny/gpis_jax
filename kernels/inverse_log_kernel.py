import jax
import jax.numpy as jnp
from kernels.base_kernel import BaseKernel
from functools import partial

class InverseLogKernel(BaseKernel):
    """
    Inverse-log kernel: 
        k(x,y) = 1 / log(1 + c² + ||x-y||²)
        
    params = (c)
      c    > 0 : scale / offset
    """

    def __init__(self, dim=2, c=1.0, eps_stable: float = 1e-12):
        self.dim = int(dim)
        super().__init__(params=(c, eps_stable))

    def __repr__(self):
        c, _ = self.params
        return f"InverseLogKernel(dim={self.dim}, c={c})"

    @partial(jax.jit, static_argnums=(0,))
    def k_fun(self, x, y, c, eps_stable):
        r = x - y
        r2 = jnp.dot(r, r)
        log_base = jnp.log1p(c * c + r2 + eps_stable)
        k_val = 1.0 / (log_base + eps_stable)
        return k_val
