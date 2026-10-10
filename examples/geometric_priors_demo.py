"""Geometric mean priors for both GPIS and an ordinary scalar GP.

Run from the project root: python -m examples.geometric_priors_demo
"""
import jax.numpy as jnp
from kernels import SEKernel
from gaussian_process import GP, GPIS
from geometric_priors import EllipsoidPrior, PlanePatchPrior, AnchorPrior


def main():
    angles = jnp.linspace(0, 2*jnp.pi, 13)[:-1]
    X = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    Q = jnp.array([[.7,.1],[.2,-.3],[1.5,.1]])

    # An implicit ellipsoid field acts as a differentiable GPIS prior mean.
    ellipsoid = EllipsoidPrior(dim=2, a=1.1, b=.8, h=.7,
                               mean=[-.2,.1], theta=-.2)
    gpis = GPIS(SEKernel(dim=2, l_f=.7), X, N_train=X,
                geo_prior=ellipsoid, augment=False, sig_x=.01, sig_n=.02)
    print('GPIS with ellipsoid prior:', gpis.predict(Q))
    # L-BFGS-B fits negative centers/orientations as well as positive axes.
    # result = gpis.fit_hyperparameters(fit_noise=False, fit_prior=True)

    # The same class works as the mean function in ordinary scalar GP regression.
    regression = GP(SEKernel(dim=2, l_f=.7), X, ellipsoid.prior_f(X),
                    mean_fn=ellipsoid, noise=.02)
    print('GP with ellipsoid mean:', regression.predict(Q))

    # Optional multichannel range prior: select the scalar channel for GPIS.
    anchors = AnchorPrior([[.0,.0],[2.,0.]], channel_ids=[101, 202],
                          range_bias=[1.,-.5], constant_bias=[-.1,.2])
    print('All anchor channels:', anchors.prior_f(Q))
    print('Only channel 101:', anchors.for_channel(101).prior_f(Q))

    # Patch prior from 2D points, with normal facing a chosen viewpoint.
    line = jnp.array([[x, .2*x+.1] for x in jnp.linspace(-1,1,10)])
    patch = PlanePatchPrior.from_points(line, dim=2, viewpoint=[0.,1.])
    print('Patch gradients:', patch.prior_grad_f(Q))


if __name__=='__main__':
    main()
