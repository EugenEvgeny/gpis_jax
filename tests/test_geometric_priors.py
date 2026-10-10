"""Consistency, gradient, parameter, covariance, and GP/GPIS integration tests."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from kernels import SEKernel
from gaussian_process import GP, GPIS
from geometric_priors import (BaseGeometricPrior, ZeroPrior, EllipsoidPrior,
                              SphericalPrior, PlanePatchPrior, MultiPlanePatchPrior,
                              AnchorPrior, RadialPrior, GPRadialPrior)
from geometric_priors.base_geometric_prior import _rotation


@pytest.fixture
def data2d():
    ang = jnp.arange(6)*2*jnp.pi/6
    X = jnp.stack([jnp.cos(ang),jnp.sin(ang)],axis=-1)
    Q = jnp.array([[.2,.3],[.4,-.7],[-.6,.2],[1.2,.4]])
    return X,Q


@pytest.fixture
def circle_prior_set():
    angles = jnp.arange(6)*2*jnp.pi/6
    centers = jnp.stack([jnp.cos(angles),jnp.sin(angles)],axis=-1)
    return [
        ZeroPrior(dim=2), EllipsoidPrior(h=.75,a=.8,b=1.2,mean=[-.2,.3],theta=.4),
        SphericalPrior(r=.8,c=1.4,mean=[.1,-.4]),
        PlanePatchPrior(h=.8,a=.9,mean=[-.2,.15],theta=.2,edge_width=.15),
        MultiPlanePatchPrior([PlanePatchPrior(h=.5,a=1.,theta=.4),
                              PlanePatchPrior(h=1.,a=.7,mean=[.4,0.])],weights=[1,2]),
        AnchorPrior(centers[:2],channel_id=0,range_bias=[.2,-.3],constant_bias=[.1,.2]),
        RadialPrior(lambda z: 1.+.1*jnp.cos(z[0]),mean=[.1,.2],theta=-.4),
        GPRadialPrior(SEKernel(dim=2,l_f=.8), x_f=jnp.full((6,),1.),u_f=centers,
                      h=.9,mean=[-.1,.2],theta=.3),
    ]


@pytest.mark.parametrize('index',range(8))
def test_consistent_value_and_gradient(index,circle_prior_set,data2d):
    _, Q = data2d
    p = circle_prior_set[index]
    assert isinstance(p,BaseGeometricPrior)
    m = p.prior_f(Q)
    g = p.prior_grad_f(Q)
    assert m.shape==(len(Q),)
    assert g.shape==Q.shape
    expected = jax.vmap(jax.grad(lambda x:p.prior_f(x[None,:])[0]))(Q)
    np.testing.assert_allclose(g,expected,rtol=1e-6,atol=2e-7)
    np.testing.assert_allclose(p.prior_mean(Q),jnp.column_stack([m,g]))
    assert np.isfinite(np.asarray(g)).all()
    if len(p.get_params()):
        target = jnp.sum(p.prior_f_params(Q,p.get_params())**2)
        dp = jax.grad(lambda param:jnp.sum(p.prior_f_params(Q,param)**2))(p.get_params())
        assert dp.shape == p.get_params().shape
        assert np.isfinite(np.asarray(dp)).all()
    lower, upper=p.parameter_bounds()
    assert lower.shape == upper.shape == p.get_params().shape


def test_3d_ellipsoid_rotation_is_fittable_and_grad_consistent():
    p=EllipsoidPrior(dim=3,h=2.,a=.5,b=1.,c=1.5,theta=[.1,-.3,.7],mean=[-.4,1.,.1])
    assert len(p.get_params())==10
    Q=jnp.array([[.5,.2,1.],[-.3,2.,-1.]])
    G=jax.vmap(jax.grad(lambda x:p.prior_f(x[None,:])[0]))(Q)
    np.testing.assert_allclose(G,p.prior_grad_f(Q),atol=1e-9)
    assert abs(float(jax.grad(lambda t: p.prior_f_params(Q,t)[0])(p.get_params())[5]))>1e-7
    p.set_params(p.get_params().at[5].set(-.5))
    np.testing.assert_allclose(p.prior_grad_f(Q),jax.vmap(jax.grad(lambda x:p.prior_f(x[None,:])[0]))(Q))


def test_3d_patch_and_from_points():
    p=PlanePatchPrior(dim=3,mean=[.2,-.3,.1],a=1,b=2,h=.8,theta=[.3,.1,-.2])
    Q=jnp.array([[.3,-.1,.5],[-.4,.5,.3],[-.2,.8,.6]])
    expected=jax.vmap(jax.grad(lambda x:p.prior_f(x[None,:])[0]))(Q)
    np.testing.assert_allclose(expected,p.prior_grad_f(Q),atol=2e-8)
    # Exact coplanar observations (translated and rotated); PCA fit stays planar.
    grid=np.array([[x,y,0] for x in np.linspace(-1.,1.,6) for y in np.linspace(-.5,.5,5)])
    rot=np.asarray(_rotation(jnp.array([.2,.4,-.3]),3))
    points=grid@rot.T+np.array([.3,-.5,.2])
    fitted=PlanePatchPrior.from_points(points,dim=3,h=1.,viewpoint=[.3,-.5,3.])
    np.testing.assert_allclose(fitted.prior_f(points),0,atol=3e-8)
    assert np.linalg.det(np.asarray(fitted.rot_mat))>0.999
    assert np.mean(np.asarray(fitted.prior_grad_f(points))[:,2])>0.


def test_plane_patch_zero_surface_normal():
    p=PlanePatchPrior(dim=2,mean=[0.,0.],a=1.,h=1.,edge_width=.08)
    X=jnp.array([[0.,0.],[.5,0.]])
    np.testing.assert_allclose(p.prior_f(X),0,atol=1e-9)
    np.testing.assert_allclose(p.prior_grad_f(X)[:,1],1.,atol=.012)
    np.testing.assert_allclose(p.prior_grad_f(X)[:,0],0,atol=1e-12)


def test_2d_patch_from_points_with_orientation():
    P=np.array([[x,2*x+.3] for x in np.linspace(-1.,1.,21)])
    p=PlanePatchPrior.from_points(P,dim=2,viewpoint=[0.,3.])
    assert p.local_corners.shape==(2,2)
    assert p.corner_locations.shape==(2,2)
    np.testing.assert_allclose(p.prior_f(P),0,atol=1e-7)
    assert np.linalg.det(np.asarray(p.rot_mat))>0.999


def test_anchor_multichannel_requires_selection(data2d):
    X,Q=data2d
    prior=AnchorPrior([[1,2],[-1,-2]],channel_ids=[100,200],range_bias=[1.,-2.])
    assert prior.prior_f(Q).shape==(len(Q),2)
    assert prior.prior_grad_f(Q).shape==(len(Q),2,2)
    assert prior.single_prior_f(Q,channels=jnp.array([1])).shape==(len(Q),1)
    model=GPIS(SEKernel(dim=2),X,X,geo_prior=prior,augment=False)
    with pytest.raises(ValueError,match='Select an anchor'):
        model.condition()
    p=prior.for_channel(200)
    np.testing.assert_allclose(p.prior_f(Q),prior.prior_f(Q,channel_id=200))
    assert p.prior_grad_f(Q).shape==(len(Q),2)
    model.geo_prior=p
    model.condition()
    assert len(model.alpha)==18
    with pytest.raises(ValueError,match='Unknown channel'):
        prior.channel_index(101)


def test_gpis_each_prior_inference_and_gradient(circle_prior_set,data2d):
    X,Q=data2d
    for prior in circle_prior_set:
        m=GPIS(SEKernel(dim=2,l_f=.9),X,X,geo_prior=prior,
               augment=False,sig_x=.03,sig_n=.1)
        mean,var=m.predict(Q)
        grad=m.predict_gradient(Q,return_var=False)
        assert mean.shape==(4,) and var.shape==(4,) and grad.shape==(4,2)
        ref=jax.vmap(jax.grad(lambda x:m.predict_mean(x[None,:])[0]))(Q)
        np.testing.assert_allclose(ref,grad,rtol=1e-5,atol=1e-6)
        assert np.isfinite(np.asarray(var)).all()


def test_gpis_prior_changes_invalidate_factor(data2d):
    X,Q=data2d
    p=EllipsoidPrior(h=.3,a=1.3,b=.7,mean=[-.5,.25])
    m=GPIS(SEKernel(dim=2),X,X,geo_prior=p,augment=False,sig_n=.2)
    a=m.predict_mean(Q)
    old_alpha=m.alpha
    p.set_params(p.get_params().at[0].set(.7))
    b=m.predict_mean(Q)
    assert not np.allclose(a,b,atol=1e-4)
    assert not np.allclose(old_alpha,m.alpha,atol=1e-4)


def test_general_gp_with_geometric_prior_is_trainable(data2d):
    X,Q=data2d
    p=SphericalPrior(r=1.,c=1.,mean=[-.2,.4])
    y=p.prior_f(X)+jnp.array([.03,-.02,.02,-.01,.01,-.03])
    m=GP(SEKernel(dim=2,l_f=.7),X,y,noise=.08,mean_fn=p)
    mu,_=m.predict(Q)
    assert np.isfinite(np.asarray(mu)).all()
    target=p.get_params().at[0].set(.4)
    assert np.isfinite(float(m.negative_log_marginal_likelihood(prior_params=target)))
    assert bool(jnp.all(jnp.isfinite(jax.grad(lambda pv:m.negative_log_marginal_likelihood(prior_params=pv))(p.get_params()))))
    previous=m.negative_log_marginal_likelihood()
    result=m.fit_hyperparameters(fit_noise=False,fit_prior=True,maxiter=8)
    assert np.isfinite(result.fun)
    assert np.isfinite(float(m.negative_log_marginal_likelihood()))
    assert float(m.negative_log_marginal_likelihood()) <= float(previous)+1e-4


def test_gp_fit_prior_negative_parameters_and_bounds(data2d):
    X,Q=data2d
    p=EllipsoidPrior(h=.6,a=1.,b=1.,mean=[-.8,.1],theta=-.3)
    model=GPIS(SEKernel(dim=2),X,X,geo_prior=p,augment=False,sig_x=.02,sig_n=.05)
    before=float(model.negative_log_marginal_likelihood())
    lb, ub = p.parameter_bounds()
    assert np.isneginf(lb[:3]).all() and np.isposinf(ub[:3]).all()
    assert np.all(lb[3:] > 0)
    theta0=jnp.concatenate([jnp.asarray(model.kernel.params),p.get_params()])
    g=jax.grad(lambda theta: model.negative_log_marginal_likelihood(theta[:2],
                               prior_params=theta[2:]))(theta0)
    assert np.isfinite(np.asarray(g)).all()
    res=model.fit_hyperparameters(fit_noise=False,fit_prior=True,maxiter=8)
    assert np.isfinite(res.fun)
    assert float(model.negative_log_marginal_likelihood()) <= before + 1e-3


def test_gpradial_covariance_and_factor_refresh(data2d):
    X,Q=data2d
    kernel=SEKernel(dim=2,l_f=.8)
    p=GPRadialPrior(kernel,jnp.ones((len(X),)),X,theta=.7)
    cov=p.prior_f_cov(Q)
    grads=p.prior_grad_f_cov(Q)
    assert cov.shape==(len(Q),)
    assert grads.shape==Q.shape
    assert np.all(np.asarray(cov)>=0) and np.all(np.asarray(grads)>=0)
    # Compare to explicit NumPy posterior covariance (reference only).
    z=np.asarray(p.trans_loc(Q))
    u=np.asarray(p.u_f)
    K=np.asarray(kernel.K(u,u)) + p.jitter*np.eye(len(u))
    k=np.asarray(kernel.K(z,u))
    ref=np.diag(np.asarray(kernel.K(z,z)) - k@np.linalg.solve(K,k.T))
    np.testing.assert_allclose(cov,ref,atol=1e-8)
    old=p.prior_f(Q)
    kernel.set_params([1.,.3])
    new=p.prior_f(Q)
    assert not np.allclose(old,new)
    assert np.isfinite(np.asarray(p.prior_grad_f(Q))).all()
    assert p._basis_key is not None


def test_gpradial_state_and_fixed_scale(data2d):
    X,Q=data2d
    p=GPRadialPrior(SEKernel(dim=2),jnp.full((len(X),),1.),X,h=.4)
    np.testing.assert_allclose(p.prior_f(Q),p.pred_f_aug(Q),atol=1e-8)
    s=p.state
    p.set_state(s.at[2].set(.3))
    assert len(p.state)==len(s) and p.state[2]==pytest.approx(.3)
    np.testing.assert_allclose(p.prior_grad_f(Q),jax.vmap(jax.grad(lambda q:p.prior_f(q[None,:])[0]))(Q),atol=1e-8)


def test_prior_variance_defaults(circle_prior_set,data2d):
    _,Q=data2d
    for p in circle_prior_set[:-1]:
        assert p.prior_f_cov(Q).shape == (len(Q),)
        if not isinstance(p,RadialPrior):
            np.testing.assert_allclose(p.prior_f_cov(Q),0.)


def test_basic_prior_serializable_state(data2d):
    _,Q=data2d
    p=MultiPlanePatchPrior([PlanePatchPrior(mean=[-1.,.3]),PlanePatchPrior(mean=[.8,-.2])])
    pp=p.get_params().copy()
    a=p.prior_f(Q)
    p.set_params(pp.at[0].set(1.3))
    assert not np.allclose(a,p.prior_f(Q))
    p.set_params(pp)
    np.testing.assert_allclose(a,p.prior_f(Q),atol=1e-8)


def test_gpis_serialization_preserves_fitted_prior(tmp_path,data2d):
    X,Q=data2d
    p=EllipsoidPrior(dim=2,mean=[-.2,.3],theta=-.4,h=.9,a=1.1,b=.8)
    m=GPIS(SEKernel(dim=2),X,X,geo_prior=p,augment=False)
    ref=m.predict(Q)
    m.save_model(tmp_path,'ellipsoid')
    receiving=GPIS(SEKernel(dim=2),X,X,
                   geo_prior=EllipsoidPrior(dim=2,h=1.3,a=.4,b=1.8),augment=False)
    receiving.load_gpis(tmp_path,'ellipsoid')
    pred=receiving.predict(Q)
    np.testing.assert_allclose(ref[0],pred[0],atol=1e-8)
    np.testing.assert_allclose(ref[1],pred[1],atol=1e-8)
    np.testing.assert_allclose(p.get_params(),receiving.geo_prior.get_params(),atol=1e-8)
    wrong=GPIS(SEKernel(dim=2),X,X,geo_prior=SphericalPrior(dim=2),augment=False)
    with pytest.raises(ValueError,match='matching geometric prior'):
        wrong.load_gpis(tmp_path,'ellipsoid')


def test_gp_cache_invalidation_after_geometric_mean_update(data2d):
    X,Q=data2d
    p=EllipsoidPrior(dim=2,mean=[.3,-.2],theta=.3,a=1.2,b=.7)
    m=GP(SEKernel(dim=2),X,p.prior_f(X)+.2,mean_fn=p)
    a=m.predict_mean(Q)
    p.set_params(p.get_params().at[0].set(-.8))
    b=m.predict_mean(Q)
    assert not np.allclose(a,b,atol=1e-5)


def test_gp_radial_3d_grads():
    coords=jnp.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.]])
    prior=GPRadialPrior(SEKernel(dim=3),jnp.ones(4),coords,dim=3,
                        theta=[.1,.2,-.3],mean=[-.2,.3,.1])
    Q=jnp.array([[.3,.1,.9],[.2,.3,-.4]])
    np.testing.assert_allclose(prior.prior_grad_f(Q),
        jax.vmap(jax.grad(lambda x:prior.prior_f(x[None,:])[0]))(Q),atol=2e-7)
    assert np.isfinite(np.asarray(prior.prior_grad_f_cov(Q))).all()
