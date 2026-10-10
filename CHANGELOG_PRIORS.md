# Geometric-prior integration changelog

## Source archive integrated

The supplied `geometric_priors.zip` contained the prior classes `BaseGeometricPrior`, `ZeroPrior`, `EllipsoidPrior`, `SphericalPrior`, `PlanePatchPrior`, `MultiPlanePatchPrior`, `AnchorPrior`, `RadialPrior`, and `GPRadialPrior`, plus a standalone SciPy shape-fitting helper and a script-like test module.

All eight concrete prior variants are in `geometric_priors/`, with compatibility re-exports in `gaussian_process.priors`. The standalone shape fitter is under `geometric_priors/legacy/fit_geom_shape.py` and can also be imported as `geometric_priors.fit_geom_shape` without loading SciPy plotting dependencies into the core inference path.

## Behavior and numerical fixes

- **GP + GPIS:** scalar prior objects can serve as `GP(mean_fn=prior)` or `GPIS(geo_prior=prior)`; fitted prior parameters are optimized via pure, JAX-differentiable evaluation. Prior parameter changes invalidate conditioned caches.
- **Parameter fitting:** each prior supplies per-parameter bounds; signed translations/angles/biases are unrestricted, while radii and positive shape scales are constrained. Plain `GP` can now optimize trainable geometric-mean parameters.
- **Ellipsoid:** 3D Euler rotation is included in the parameter vector and the mean/gradient are vectorized; the Hessian-derived gradient is consistent with the implicit field.
- **Sphere:** now conforms to the shared prior interface, with `spherical_prior()` and `spherical_prior_diff()` aliases.
- **Finite plane patch:** original function and normal-gradient formulas were inconsistent; replaced with a smooth finite-window signed-plane field and exact analytical gradient. `from_points` uses PCA plus direct rotation extraction, avoiding an unnecessary nonlinear solve. `cutoff_dist` now actually controls the spatial attenuation.
- **MultiPlanePatch:** implements a fixed, normalized weighted *average* of patches with trainable child parameters; it is not a union SDF.
- **Anchor:** vectorized range/gradient evaluation and numerically defined gradient at coincident anchor positions. Multi-output priors explicitly require a scalar channel for GP/GPIS; the other channels remain available outside them.
- **Radial:** coherent pose transformation and Jacobian; optional radial covariance supported.
- **GP radial:** removes `K_uu` inverse in numerical paths; Cholesky factor caching with invalidation on radial-kernel changes; removes forced zero gradient, repairs pose rotation and double-application of scale `h`; exposes inner-GP radial mean and gradient variance separately.
- **Serialization:** saves prior class/parameters/config metadata and validates the destination prior class, rather than silently restoring numerical state with the wrong mean function.
- **Dependencies:** core numerical priors require JAX/NumPy. Optional plotting and jaxlie imports are lazy.

## Migration considerations

The following reflect intentional correctness changes and may alter results for models trained using the original code:

1. `PlanePatchPrior` has a changed signed implicit field and different parameter counts. It should be retrained when replacing the inconsistent original mean/gradient pair.
2. 3D ellipsoid fit parameters now contain roll, pitch and yaw (`[mean(3), angles(3), h, a, b, c]`). The original vector (`[mean(3), h, a, b, c]`) was missing rotation.
3. The 2D `PlanePatchPrior` fit vector is now `[mean(2), theta, h, a]`; the old vector included an unused `b`.
4. `GPRadialPrior` uses one factor of `h` and correct nonzero spatial gradients. The old code had `h**2` and zero gradients.
5. Optimized prior bounds default to physically meaningful limits, not the earlier positive-only uniform bound, which was invalid for negative translations and rotations.
6. `AnchorPrior` with multiple anchors returns multiple channels by default, but a scalar GP/GPIS requires `channel_id=` or `.for_channel(...)`.
7. Ordinary deterministic GP/GPIS posterior variance does **not** include the separately estimated inner-GP uncertainty from a `GPRadialPrior`.

## Validation

- Existing GP/GPIS suite: 26 tests passed (run in three independent batches because repeated large JAX compilations can be expensive in one CPU process).
- Existing kernel suite: 20 tests passed.
- Added geometric-prior suite: 24 tests passed (small batches); checks include 2D/3D value-gradient consistency, differentiable prior parameters, classical GP/GPIS fitting, posterior caching, radial GP variance, patch fitting, and prior-parameter serialization.
- `python -m examples.geometric_priors_demo` completed successfully.
- `python -m benchmarks.benchmark_priors` provides a reproducible comparison between analytical and autodiff gradients. On the available CPU, both were within a few microseconds; no general analytical-gradient performance speedup is claimed.
