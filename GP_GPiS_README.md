# Gaussian processes and GP implicit surfaces (JAX)

This project integrates the **previously cleaned JAX kernels**, eight geometric-prior variants, and a refactored GPIS implementation. `GPIS` is a subclass of a **general-purpose scalar Gaussian process** (`GP`). It reuses the same exact GP conditioning, Cholesky solves, covariance predictions, likelihood, and hyperparameter optimizer. No visualization/point-cloud dependencies are needed for numerical inference.

## Install with uv (CPU or NVIDIA CUDA)

The package now supports [`uv`](https://docs.astral.sh/uv/) with mutually
exclusive JAX CUDA extras and a compiled-backend verification command:

```bash
# Default CPU installation and numerical tests
uv sync --extra dev
uv run gp-jax-device --backend cpu --smoke-test

# NVIDIA GPU: CUDA 13 (Linux, recent NVIDIA driver)
uv sync --extra cuda13 --extra dev
uv run --extra cuda13 gp-jax-device --backend cuda --require-gpu --smoke-test

# Or NVIDIA GPU: CUDA 12, for supported older drivers / hardware
uv sync --extra cuda12 --extra dev
uv run --extra cuda12 gp-jax-device --backend cuda --require-gpu --smoke-test
```

### Install with `uv pip install` (editable source installation)

If you prefer the `uv pip` workflow instead of `uv sync`, create a virtual
environment and activate it first. Run these commands from the directory
containing `pyproject.toml` (after extracting the release ZIP, this is
`jax_gp_gpis_uv_cuda/`):

```bash
uv venv .venv
source .venv/bin/activate  # Linux / WSL2
```

Install **one** of the following configurations:

```bash
# Standard CPU installation
uv pip install -e .

# NVIDIA CUDA 12 (instead of the CPU command; supported Linux GPUs)
uv pip install -e '.[cuda12]'

# NVIDIA CUDA 13 (instead of the CPU/CUDA 12 commands)
uv pip install -e '.[cuda13]'
```

`-e` is an editable install: source changes are picked up without reinstalling
the package. To install development tools or optional visualization and
registration dependencies together with a selected backend, use e.g.:

```bash
uv pip install -e '.[cuda12,dev,visualization,registration]'
```

**Do not install `cuda12` and `cuda13` together in the same environment.**
When switching CUDA variants, use a fresh virtual environment. The CUDA
extras install JAX's CUDA runtime packages; a compatible NVIDIA driver must
already be installed on the host.

After installation, verify the backend using the installed command:

```bash
gp-jax-device --backend cpu --smoke-test
# On a CUDA machine, require a real GPU instead of accepting CPU fallback:
gp-jax-device --backend cuda --require-gpu --smoke-test
```

You can also install a **locally built or downloaded wheel** without an
editable source checkout. From the directory containing the wheel:

```bash
uv pip install './jax_gp_gpis_refactor-0.3.0-py3-none-any.whl[cuda12]'
# Replace cuda12 with cuda13; omit [cuda12] for CPU-only installation.
```

The package is not assumed to be published on PyPI; the command above uses
the local wheel file, not a package-index name. If you prefer standard pip,
`python -m pip install -e '.[cuda12]'` works in a pip-enabled environment.
Unlike `uv sync`, `uv pip install` does not generate or maintain a `uv.lock`
file. With an activated virtual environment, call `python`, `pytest`, or
`gp-jax-device` directly to avoid an implicit `uv run` project synchronization.

JAX selects an available CUDA GPU automatically when the relevant backend is
installed. Use `JAX_PLATFORMS=cuda` before launching Python to **require**
CUDA and fail rather than silently fall back to CPU. The `gp-jax-device`
checker compiles and runs small GP and GPIS predictions and verifies output
device placement. For repeated GPU queries, both `GP` and `GPIS` now expose
`predictor = model.compile_predict()`; recreate the predictor after changing
model parameters or data. Actual GPU execution requires an NVIDIA device and
driver.

**Compatibility:** CUDA 13 requires NVIDIA driver >= 580 and compute capability
>= 7.5; CUDA 12 requires driver >= 525 and compute capability >= 5.2. Native
CUDA JAX wheels are available for Linux x86-64 and aarch64; WSL2 support is
experimental. The `cuda12-local` and `cuda13-local` extras are also available
for environments managing CUDA/cuDNN outside pip. Choose only one CUDA extra.

See **[docs/INSTALL_UV_CUDA.md](docs/INSTALL_UV_CUDA.md)** for installation,
GPU selection, diagnostics, examples, and building a wheel with `uv build`.

## Architecture

```text
kernels/                       # optimized covariance kernels and derivatives
  base_kernel.py               # kernel value/gradient/Hermite interfaces

gaussian_process/
  gp.py                        # GP: scalar regression, Cholesky, NLL, optimization
  gpis.py                      # GPIS(GP): normals, augmentation, surfaces, registration
  priors.py                    # Compatibility re-exports for all priors

geometric_priors/
  base_geometric_prior.py      # Pure, differentiable prior API, poses, bounds
  ellipsoid_prior.py           # Oriented quadratic implicit ellipsoid
  spherical_prior.py           # Quadratic sphere
  plate_prior.py               # Smooth windowed plane/line and patch mixture
  anchor_prior.py              # Multi-channel range prior, scalar selection
  radial_prior.py              # Callable radial profile
  gp_radial_prior.py           # Cholesky-factored finite-basis radial GP
  zero_geometric_prior.py      # Zero prior
  legacy/fit_geom_shape.py     # Original standalone SciPy shape fitters
  statistics.py                # Gaussian log-PDF and stable interval probabilities

GP.py, GPIS.py                 # backwards-compatible import shims
examples/demo.py               # running examples for each model
tests/                         # GPIS/GP and original kernel regression suites
benchmarks/                    # simple, reproducible microbenchmarks
```

### Mathematical model

Generic GP regression observes scalar values, $y_i = f(x_i) + \epsilon_i$, with $f\sim\mathcal{GP}(m,k)$ and a diagonal **noise variance** matrix $\Sigma$. Let $K$ denote the covariance of the observations and $L=\operatorname{chol}(K+\Sigma+\text{jitter} I)$. The conditioned mean and covariance are

$$
\begin{aligned}
\alpha &= L^{-T}L^{-1}(y-m(X)),\\
\mu_*(X_*) &= m(X_*) + K_{*D}\alpha,\\
\operatorname{Cov}[f(X_*)\mid D] &= K_{**} - V^TV,
\qquad V=L^{-1}K_{D*}.
\end{aligned}
$$

`GP` uses this algorithm for regular value observations. **GPIS adds linear derivative observations to the same model**. For surface points $X_f$ and normal points $X_N$:

$$
\underbrace{\begin{bmatrix}f(X_f)\\\nabla f(X_N)\end{bmatrix}}_y,
\qquad K_{DD} =
\begin{bmatrix}
K(X_f,X_f) & \partial_{y}K(X_f,X_N)\\
\partial_{x}K(X_N,X_f) & \partial_x\partial_y K(X_N,X_N)
\end{bmatrix}.
$$

Gradient observations are stored in **coordinate-major order**, i.e. all x-derivatives before all y-derivatives, etc., matching the included kernel block constructors. Function-value observations precede gradient observations. Training and predictions are ordinary exact GP inference on this combined system, not separately fitted GPs.

## Requirements

- Python 3.10+; `uv sync` installs JAX, NumPy, and SciPy. GPU installations use one CUDA extra as described above.
- `pytest` for numerical regression tests.
- Optional: `matplotlib` and `scikit-image` for 2D level sets; `open3d` for 3D meshes; `plotly` for 3D volume plots; `jaxlie` for point-cloud registration.
- Optional original kernel extras: `torch`, `pykeops` for the PyKeOps compact-kernel backend.

Install and run from this folder (the project is editable with `uv sync`):

```bash
uv sync --extra dev
uv run python examples/demo.py
# Optional geometry helpers:
uv sync --extra visualization --extra registration
```

`pip install -e '.[dev]'` remains supported for users without uv.

If you need double precision, configure JAX **before running any model** (e.g. `JAX_ENABLE_X64=1`). The test suite enables float64 for accurate finite-difference comparisons. You can work in float32 for smaller/more regularized problems.

## Standard Gaussian process example

```python
import jax
import jax.numpy as jnp
from gaussian_process import GP
from kernels import SEKernel

X = jnp.linspace(-2, 2, 40)[:, None]
y = jnp.sin(X[:, 0])

# noise is VARIANCE, not standard deviation
gp = GP(SEKernel(sig_f=1., l_f=.8, dim=1), X, y, noise=0.01)
gp.fit()                                  # condition without optimizing
X_test = jnp.linspace(-3, 3, 150)[:, None]
mean, var = gp.predict(X_test)            # (150,), (150,)
mean, covariance = gp.predict(X_test, full_cov=True)
dmean, dvar = gp.predict_gradient(X_test) # (150,1) arrays
std = gp.predict_std(X_test)
samples = gp.sample_posterior(jax.random.key(0), X_test, n_samples=4)
compiled_predictor = gp.compile_predict()  # efficient repeated JIT execution
mean_compiled, var_compiled = compiled_predictor(X_test)
log_evidence = gp.log_marginal_likelihood()

# Shared L-BFGS-B optimizer, autodiff through kernel/noise/Cholesky:
result = gp.fit_hyperparameters(fit_noise=True)
print(result.success, result.x, result.fun)

# For huge test grids, avoid the O(n_test^2) predictive covariance:
mean, variance = gp.predict(X_test, batch_size=64)
```

The vanilla GP supports arbitrary input dimensions, heteroscedastic training noise (scalar or vector), custom callable prior means, gradient predictions, full predictive covariance, and posterior sampling. Exact GP training still scales cubically in the **number of training observations**, not in the ambient dimension.

## GPIS example

```python
import jax.numpy as jnp
from gaussian_process import GPIS, ZeroPrior
from kernels import SEKernel

angles = jnp.linspace(0, 2*jnp.pi, 65)[:-1]
points = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
normals = points  # already unit outward normals for this circle

gpis = GPIS(
    SEKernel(sig_f=1., l_f=0.7, dim=2),
    X_train=points,
    N_train=normals,
    geo_prior=ZeroPrior(dim=2),
    sig_x=1e-3, sig_n=1e-2,
    augment=True, d=0.05,  # f_aug_val defaults to d for SDF consistency
)

gpis.condition()                         # GP conditioning, without optimization
q = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0]])
f_mean, f_var = gpis.predict(q)         # inherited GP interface
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
# gpis.plot_mean_map(grid_size=150)
```

When no normals are supplied, `GPIS` uses k-NN PCA and chooses their signs relative to the point-cloud centroid. This sign convention may be wrong for non-star-shaped objects or point clouds with inconsistent orientation. Normal observation *targets* are components of the supplied normals; they need not be unit-length, but consistent scaling is important. The **augmentation default is intentionally corrected**: with `d=0.05`, displaced constraints default to `f(x±0.05n)=±0.05`, consistent with `∇f≈n` for a signed-distance field. The pasted version used values `±1` at an offset of `±0.05`, implying a directional slope of 20 despite normal-gradient targets of magnitude one. To reproduce the old behavior explicitly, set `f_aug_val=1.0`.

### Prior objects

GPIS accepts geometric priors implementing the following two pure functions:

- `prior_f(X) -> (n,)` – scalar mean of the implicit field.
- `prior_grad_f(X) -> (n,d)` – mean gradient at each observation.

For prior-parameter optimization (with `fit_geo_prior=True`), also provide `get_params()`, `set_params(params)`, **`prior_f_params(X,params)`**, and **`prior_grad_f_params(X,params)`**. The latter two must be **pure/JAX-differentiable**. The previous approach mutated prior parameters inside a JIT-compiled objective and was not reliably differentiable. `parameter_bounds()` supplies physically meaningful per-parameter limits; translations, rotations, and biases are allowed to be negative. The default `ZeroPrior` has no tunable parameters.

## API migration from the pasted GPIS class

| Old usage | New usage |
|---|---|
| `GPIS(kernel, X_train, N_train, ..., augment=...)` | Same constructor; with optional noise/augmentation/prior arguments |
| `cache_model_input(); cache_kerne_mat()` | Prefer `condition()`; old methods still available |
| `pred_f_fast(X)` | Unchanged; or inherited `predict_mean(X)` |
| `pred_f(X)` | Unchanged default `(mean, variance)`; or inherited `predict(X)` |
| `pred_cov_f(X)` | Unchanged (variance by default), or `predict(X)[1]` |
| `pred_N(X, mean_only=True)` | Unchanged; or inherited `predict_gradient(X, return_var=False)` |
| `get_pred_f_grad(X)` | Unchanged |
| `fit(fit_noise=True, fit_geo_prior=False)` | Same GPIS call, returning optimized parameter vector; full SciPy result in `.optimization_result` |
| `GPIS.log_surface_pdf / get_surface_cdf` | Retained, with corrected variance/std conventions |
| `pred_cov_f_up(...)` | Compatibility wrapper; prefer `predict_uncertain(X, cov_x, order=1 or 2)` |
| `Kff_inv`, `alpha_ff` | `alpha_ff` aliases posterior coefficients; `Kff_inv` works but is expensive – avoid it |
| `load_gpis` / `save_model` | Retained; recompute factorization rather than saving inverse/cache |
| `match_pcds(Y, ...)` | Retained with analytic Jacobians and backtracking (optional `jaxlie`) |

The **base** `GP.fit()` conditions without optimizing hyperparameters, following usual GP regression usage. **Legacy** `GPIS.fit()` still optimizes, matching the previous API; to condition GPIS without optimization call `.condition()`. Both subclasses have `.fit_hyperparameters()` (which returns the full SciPy result).

The pasted class also contained `update_obs` and `get_extent_estimate`, which assumed additional *unsupplied tracking state* (`self.t`, `self.R`, `self.x_post_est`, `self.cov_post_est`, `self.X_post_hist`, etc.). These methods are intentionally **not** included in this standalone GPIS module: reproducing their behavior without the missing tracker would be misleading. Use `add_surface_observations(X, y, noise=...)` for actual independent scalar observations; adapt tracking logic separately when integrating into the parent project. Some project-specific plotting conveniences (including the original custom polygon collection) are replaced by simpler NumPy polylines and optional standard plotting APIs.

## Important numerical and modeling considerations

1. All observation and predictive noise inputs are **variances**. `sig_x` is scalar implicit-value noise variance; `sig_n` is gradient-component noise variance. `sig_x_input` is instead the positional covariance (a `(d,d)` or `(n,d,d)` matrix).
2. `GPIS.predict_uncertain` is an **approximation**, not exact uncertain-input GP inference. `order=1` includes mean-gradient and posterior gradient-covariance terms; `order=2` adds the curvature correction via autodiff and can be expensive. Use a sufficiently smooth kernel (e.g. SE or Matérn 5/2) for second-order derivatives.
3. Mixed derivative observations require differentiable kernels. Do **not** use Matérn 1/2 at coincident points for GPIS normal constraints; its gradient covariance is not defined. A positive jitter and noise variances help factorization stability.
4. Predictions default to **latent** variance, excluding test observation noise. To include noisy measurements use `GP.predict(..., observation_noise=...)`.
5. Kernel parameters are resolved explicitly for all covariance blocks so training/test evaluations use a consistent kernel, including median-heuristic variants. Direct external edits to a kernel's parameters automatically trigger reconditioning, but training data changes should go through `set_data(...)`.
6. Exact GPIS conditioning uses $(n_f + d n_N)^2$ memory and $(n_f + d n_N)^3$ training time. Batching reduces query memory but **does not** make exact GPIS sparse; scaling to millions of normals will require an inducing-point or sparse representation.
7. **Only load trusted `.pkl` files.** Python pickle executes arbitrary code when deserializing. Save/load files do not include the original geometric prior class; initialize with the correct prior first.

## Tests and benchmarks

```bash
JAX_ENABLE_X64=1 pytest -q tests/test_gp.py
JAX_ENABLE_X64=1 pytest -q tests/test_kernels.py
python benchmarks/benchmark_gp.py
```

For a CPU-only microbenchmark with JIT compilation excluded, evaluating only the posterior diagonal took 1.38 ms (inverse/dense intermediate) versus 0.62 ms (Cholesky) for 600 query points and 96 training points; for 2,500 query points and 128 training points, 17.41 ms versus 3.27 ms. Different hardware, matrix sizes, and dtypes can change these results. The optimized implementation also avoids building an O(n_query²) intermediate matrix.

The GP tests check numerical agreement with direct Gaussian conditioning, gradient finite differences and covariance PSD, normals/augmentation ordering, prior optimization, parameter changes, shape validation, uncertain-input batches, stable surface probabilities, likelihood gradients, and save/load. The bundled kernel suite checks the optimized kernel derivatives and sparse and RKHS interfaces. Run the suites independently on memory-limited hosts to reduce JAX compilation cache pressure.

## Integrated geometric priors

All eight concrete geometric-prior types are exported from both
`geometric_priors` and `gaussian_process`:

```python
from geometric_priors import (
    ZeroPrior, EllipsoidPrior, SphericalPrior, PlanePatchPrior,
    MultiPlanePatchPrior, AnchorPrior, RadialPrior, GPRadialPrior,
)
```

Use a prior as the nonzero mean for either model. The GP regression API is
unchanged; when its `mean_fn` is a prior object, it also supports prior-parameter
optimization and automatic cache invalidation:

```python
from gaussian_process import GP, GPIS
from geometric_priors import EllipsoidPrior
from kernels import SEKernel

prior = EllipsoidPrior(dim=2, mean=[-0.2, 0.3], theta=-0.3,
                       a=1., b=0.7, h=1.)

# X: (n,2), y: (n,), N: (n,2), Q: (m,2)
gp = GP(SEKernel(dim=2), X, y, mean_fn=prior, noise=0.02)
mean, variance = gp.predict(Q)
# gp.fit_hyperparameters(fit_prior=True)  # fits prior, kernel and noise

gpis = GPIS(SEKernel(dim=2), X, N_train=N, geo_prior=prior,
            augment=False, sig_x=0.02, sig_n=0.05)
field_mean, field_variance = gpis.predict(Q)
# gpis.fit(fit_noise=False, fit_geo_prior=True, maxiter=100)
```

Run a self-contained example with `python -m examples.geometric_priors_demo`.

### Geometric interpretation and important changes

| Class | Mathematical meaning | Trainable fields |
| --- | --- | --- |
| `ZeroPrior` | Zero implicit mean | None |
| `EllipsoidPrior` | `h/2 * (sum_i(local_i/axis_i)**2 - 1)` | Center, orientation, scale, half-axes |
| `SphericalPrior` | `c/(2r) * (||x-center||²-r²)` | Center, radius, scale |
| `PlanePatchPrior` | Smooth signed-plane field, attenuated outside a finite patch | Center, orientation, scale, extents |
| `MultiPlanePatchPrior` | Fixed weighted average of child plane fields | Child-patch fields (concatenated) |
| `AnchorPrior` | One range bias and constant bias per anchor | Anchor positions, range biases, constant biases |
| `RadialPrior` | Local distance minus a JAX radial function | Center, orientation |
| `GPRadialPrior` | Local distance minus inner GP-predicted radial extent | Center, orientation, scale, inducing-point radii |

**Coordinate conventions:** `mean` is the world-space geometric center.
`theta` is a single 2D angle, or `[roll,pitch,yaw]` in 3D, using
`R=Rz(yaw) Ry(pitch) Rx(roll)`; local points are `(X-mean)@R`.
Ellipsoid and sphere priors are quadratic implicit fields, *not exact signed
distance functions*. This matters when selecting the scale `h` and the
normal-derivative target magnitude.

**Finite patches:** the uploaded `PlanePatchPrior.prior_f()` computed a
hard-windowed signed plane field, but `prior_grad_f()` returned the gradient of
a different unsigned-distance function. This silently violated GPIS's
assumption that normal observations measure the derivative of the same mean.
The refactor uses an analytical, consistent smooth-windowed signed-plane mean:

\[ f(x) = h\, w\,\exp(-\sqrt{w^2+\varepsilon^2}/c)
   \prod_i \operatorname{sigmoid}((a_i^2-u_i^2)/(2a_i\,\delta)), \]

where `w` is the local normal coordinate, `u_i` are tangent coordinates,
`c=cutoff_dist` and `delta=edge_width`. `f=0` on the entire supporting plane,
including outside the footprint: **a bounded, zero-thickness patch is not a
closed signed-distance surface**. Do not use this field to infer inside/outside
of a solid. The values smoothly decay rather than jumping at patch edges.
`PlanePatchPrior.from_points()` uses PCA, an optional viewpoint to fix normal
orientation, and direct rotation conversion instead of nonlinear fitting.
`MultiPlanePatchPrior` preserves the source's weighted-mean semantics; it is
not a union or minimum-distance SDF. `mean` and `theta` on the mixture itself
are compatibility placeholders and do not transform the children.

**Anchor channels:** with multiple anchors, an unselected `AnchorPrior`
returns shape `(n, n_anchors)` and its gradient has shape `(n, n_anchors, d)`.
The scalar GP and GPIS intentionally reject this until a channel is selected:

```python
anchors = AnchorPrior([[0.,0.], [1.,0.]], channel_ids=[101, 202])
channel101 = anchors.for_channel(101)
gpis = GPIS(SEKernel(dim=2), X, N, geo_prior=channel101)
```

`for_channel()` creates an independent prior copy. You may also specify
`channel_id=101` at construction. Norms are regularized extremely close to the
anchor to avoid an undefined derivative at zero distance.

**Radial GP:** `GPRadialPrior` performs Cholesky solves instead of explicitly
inverting the inner Gram matrix. It caches the inner kernel factorization,
re-evaluating it when its kernel parameters change. Its analytic gradient
includes both rotation and the derivative of the learned radial profile; the
previous implementation returned zero and scaled the mean twice by `h`.
`prior_f_cov` and `prior_grad_f_cov` expose the uncertainty of the *inner*
radial GP. **This uncertainty is not automatically incorporated into GPIS's
outer covariance**: for the current hierarchical model, these priors are used
as deterministic posterior mean functions. Propagating their stochastic
uncertainty into the outer GP requires a separate, explicitly defined joint
covariance model. Use a different inner kernel object from the outer GPIS
kernel when optimizing the outer kernel hyperparameters.

### Parameter fitting and bounds

Every concrete prior implements
`get_params()` / `set_params()` / `prior_f_params(X,p)` /
`prior_grad_f_params(X,p)` / `parameter_bounds()`. Fitting is side-effect free
within JAX and commits fitted values only after L-BFGS-B completes.

- Ellipsoid 2D: `[center_x, center_y, theta, h, a, b]`.
- Ellipsoid 3D: `[center_x, center_y, center_z, roll, pitch, yaw, h, a, b, c]`.
- PlanePatch 2D: `[center_x, center_y, theta, h, a]`.
- PlanePatch 3D: `[center_xyz, roll, pitch, yaw, h, a, b]`.
- Sphere: `[center_xyz, r, c]`.
- Anchor: `[flatten(anchor_positions), range_bias, constant_bias]`.
- Radial: `[center, angles]`.
- GP radial: `[center, angles, h, x_f]`.
- MultiPlanePatch: concatenation of each child patch's parameter vector.

By default translations, angles, and offsets/biases have **unrestricted**
optimization bounds, while positive scales and radii have lower bounds of
`1e-5`. Override bounds with `lower_bound_prior` /
`upper_bound_prior` on `fit_hyperparameters()` or the legacy
`lower_bound_geo` / `upper_bound_geo` on `GPIS.fit()`. The old 3D ellipsoid
parameter vector *did not contain rotation*, and the old 2D patch vector
contained an unused `b`; parameter vectors from those source classes therefore
need migration before reuse.

### Integration and migration

The new modules preserve the source filenames under `geometric_priors/` and
provide imports from `gaussian_process.priors` for convenience. The numerical
`fit_geom_shape.py` is retained under `geometric_priors.legacy` and via a
compatibility entry point `geometric_priors.fit_geom_shape`; its SciPy and
Matplotlib dependencies are imported only when requested. The source-only
plotting utilities embedded in prior classes have been reduced to generic,
optional `plot_prior_f` (2D) and `plot_sdf_plotly` (3D) methods.

GPIS `save_model` now includes prior class, fitted parameter vector and
selected immutable configuration. `load_gpis` requires you to construct the
receiving `GPIS` with the **same prior type**; it applies saved trainable prior
parameters and rejects certain static configuration mismatches. Plain
Python `RadialPrior` callables are not serializable through that mechanism;
provide the same callable at restore time. Old pickles without prior metadata
are still accepted for backward compatibility.

Run tests separately if JAX's compiler cache becomes large during a complete
CPU test run:

```bash
uv run --extra dev python -m pytest tests/test_kernels.py -q
uv run --extra dev python -m pytest tests/test_geometric_priors.py -q
uv run --extra dev python -m pytest tests/test_gp.py -q
uv run --extra dev python -m pytest tests/test_device.py -q
```
