import jax.numpy as jnp
from kernels.base_kernel import BaseKernel

class IMQKernel(BaseKernel):
    """
    Inverse Multi-Quadratic (IMQ) kernel:
        k(x,y) = sig2 * (c^2 + ||x-y||^2)^(-beta)

    params = (sig2, c, beta)
      sig2 > 0 : amplitude
      c    > 0 : scale / offset
      beta > 0 : exponent
    """

    def __init__(self, dim=2, params=(1.0, 1.0, 0.5)):
        self.dim = int(dim)
        super().__init__(params=params)

    def __repr__(self):
        sig2, c, beta = self.params
        return f"IMQKernel(dim={self.dim}, sig2={sig2}, c={c}, beta={beta})"

    @staticmethod
    def k_fun(x, y, sig2, c, beta):
        r = x - y
        r2 = jnp.dot(r, r)
        u = c * c + r2
        return sig2 * (u ** (-beta))

    @staticmethod
    def dk_fun(x, y, sig2, c, beta):
        # ∇_x k(x,y)
        r = x - y
        r2 = jnp.dot(r, r)
        u = c * c + r2
        # d/du u^{-beta} = -beta u^{-beta-1}, du/dx = 2r
        coeff = -2.0 * beta * sig2 * (u ** (-beta - 1.0))
        return coeff * r

    @staticmethod
    def dk_y_fun(x, y, sig2, c, beta):
        # ∇_y k(x,y) = -∇_x k(x,y)
        return -IMQKernel.dk_fun(x, y, sig2, c, beta)

    @staticmethod
    def ddk_fun(x, y, sig2, c, beta):
        """
        Mixed partials: ∇_y ∇_x k(x,y)  (shape dxd)

        For k = sig2 * u^{-beta}, u = c^2 + ||x-y||^2, r = x-y:
          ∇_y∇_x k =  2β sig2 u^{-β-1} I  - 4β(β+1) sig2 u^{-β-2} (r r^T)
        """
        r = x - y
        r2 = jnp.dot(r, r)
        u = c * c + r2
        d = x.shape[0]
        I = jnp.eye(d, dtype=x.dtype)

        term_I = 2.0 * beta * sig2 * (u ** (-beta - 1.0)) * I
        term_rr = 4.0 * beta * (beta + 1.0) * sig2 * (u ** (-beta - 2.0)) * jnp.outer(r, r)
        return term_I - term_rr

    @staticmethod
    def ddkxx_fun(x, y, sig2, c, beta):
        """
        Matches your AbstractKernel.ddkxx_fun_autodiff convention, which returns:
            ddkxx_fun = - (∇_y ∇_x k)^T
        Since mixed partials are symmetric in this case, this is simply -ddk_fun(x,y).
        """
        return -IMQKernel.ddk_fun(x, y, sig2, c, beta)
