import jax.numpy as jnp
from gaussian_process import GPIS, ZeroPrior
from kernels import SEKernel
import matplotlib.pyplot as plt

angles = jnp.linspace(0, 2 * jnp.pi, 65)[:-1]
points = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
normals = points  # already unit outward normals for this circle

gpis = GPIS(
    SEKernel(sig_f=1.0, l_f=0.7, dim=2),
    X_train=points,
    N_train=normals,
    geo_prior=ZeroPrior(dim=2),
    sig_x=1e-3,
    sig_n=1e-2,
    augment=True,
    d=0.05,  # f_aug_val defaults to d for SDF consistency
)

gpis.condition()  # GP conditioning, without optimization
q = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0]])
f_mean, f_var = gpis.predict(q)  # inherited GP interface
normals_mean, normals_var = gpis.predict_gradient(q)

# GPIS compatibility interfaces:
f_mean_fast = gpis.pred_f_fast(q)
mean, variance = gpis.pred_f(q)
normal_mean, normal_variance = gpis.pred_N(q)
log_surface_density = gpis.log_surface_pdf(q, f_val=0)
probability_near_surface = gpis.get_surface_cdf(q, -0.02, 0.02)

# Geometric-prior hyperparameters are optional; kernel and two noise
# variances are optimized jointly using exactly the same GP machinery:
optimal_parameter_vector = gpis.fit(fit_noise=True, maxiter=100)
optimization_details = gpis.optimization_result

# A posterior for uncertain point positions, first-order moment expansion:
input_covariance = jnp.diag(jnp.array([0.001, 0.002]))
mean, variance_with_input_uncertainty = gpis.pred_f(q, sig_x_input=input_covariance)

# Optional rendering; imports matplotlib only on use:
gpis.plot_mean_map(grid_size=150)
gpis.plot_var_map()
plt.show()