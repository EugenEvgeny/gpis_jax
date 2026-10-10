"""Self-contained examples for regular GP regression and GPIS.

Run: python examples/demo.py
"""
import jax
import jax.numpy as jnp
from kernels import SEKernel
from gaussian_process import GP, GPIS


def gp_example():
    x = jnp.linspace(-2, 2, 25)[:, None]
    y = jnp.sin(x[:, 0])
    gp = GP(SEKernel(dim=1, l_f=.8), x, y, noise=0.01)
    gp.condition()
    test = jnp.linspace(-3, 3, 60)[:, None]
    mean, var = gp.predict(test)
    print('Regular GP:', mean.shape, var.shape,
          'log marginal likelihood:', float(gp.log_marginal_likelihood()))
    return gp


def gpis_example():
    angles = jnp.linspace(0, 2*jnp.pi, 13)[:-1]
    points = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    gp = GPIS(SEKernel(dim=2, l_f=.7), points, points,
              sig_x=.005, sig_n=.01, d=.05, f_aug_val=.05, augment=True)
    gp.condition()
    q = jnp.array([[1.,0.], [0.,0.], [1.5,0.]])
    mean,var = gp.predict(q)
    normals,_ = gp.predict_gradient(q)
    print('GPIS:', mean, '\nvariance:',var,'\ngradient:',normals)
    return gp


if __name__ == '__main__':
    gp_example()
    gpis_example()
