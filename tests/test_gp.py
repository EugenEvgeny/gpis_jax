"""Numerical and interface tests for the generic GP and GPIS subclass."""
import os
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from kernels import SEKernel, Matern52Kernel
from gaussian_process import GP, GPIS, ZeroPrior
from gaussian_process.gp import _chol_solve
from gaussian_process.statistics import normal_interval_logprob, normal_logpdf


@pytest.fixture

def xy():
    X = jnp.array([[-1., 0.], [-.4, .2], [0., -.2], [.6, .3], [1., 0.]])
    y = jnp.sin(X[:, 0]) + .15*X[:, 1]
    return X, y


def test_is_subclass():
    assert issubclass(GPIS, GP)


def test_gp_predictions_match_numpy_reference(xy):
    X, y = xy
    k = SEKernel(sig_f=1.4, l_f=.75, dim=2)
    model = GP(k, X, y, noise=.02, jitter=1e-8).fit()
    q = jnp.array([[.2, .1], [.9, -.1]])
    mu, cov = model.predict(q, full_cov=True)
    K = np.asarray(k.K(X, X) + jnp.eye(len(X))*(.02 + model.jitter))
    KT = np.asarray(k.K(q, X))
    target_mu = KT @ np.linalg.solve(K, np.asarray(y))
    target_cov = np.asarray(k.K(q, q)) - KT@np.linalg.solve(K, KT.T)
    np.testing.assert_allclose(mu, target_mu, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(cov, target_cov, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(model.predict(q)[1], np.diag(cov), atol=1e-8)
    np.testing.assert_allclose(model.predict(q, observation_noise=.03)[1], np.diag(cov)+.03)
    assert model.log_marginal_likelihood() == pytest.approx(-float(model.negative_log_marginal_likelihood()))


def test_gp_mean_and_noise_heteroscedastic(xy):
    X, y = xy
    mu0 = lambda P: P[:, 0]*.5 + 2.*P[:, 1]
    model = GP(SEKernel(dim=2), X, y, noise=jnp.linspace(.01, .05, len(X)), mean_fn=mu0)
    q = jnp.array([[.2,.4],[.5,-.1]])
    mean, var = model.predict(q)
    K = np.asarray(model.kernel.K(X,X)) + np.diag(np.asarray(model.noise)+model.jitter)
    ref = np.asarray(mu0(q)) + np.asarray(model.kernel.K(q,X))@np.linalg.solve(K,np.asarray(y-mu0(X)))
    np.testing.assert_allclose(mean,ref,atol=1e-8)
    assert bool(jnp.all(var >= 0))
    grad = model.predict_gradient(q, return_var=False)
    expected = np.stack([np.asarray(jax.grad(lambda p: model.predict_mean(p[None])[0])(x)) for x in q])
    np.testing.assert_allclose(grad,expected,rtol=1e-7,atol=1e-7)
    with pytest.raises(ValueError):
        model.fit_hyperparameters(fit_noise=True)


def test_invalid_gp_inputs(xy):
    X, y = xy
    with pytest.raises(ValueError): GP(SEKernel(dim=3), X, y)
    with pytest.raises(ValueError): GP(SEKernel(dim=2), X, y[:-1])
    with pytest.raises(ValueError): GP(SEKernel(dim=2), X, y, jitter=0)
    model=GP(SEKernel(dim=2),X,y)
    with pytest.raises(ValueError): model.predict(jnp.ones((5,3)))
    with pytest.raises(ValueError): model.fit(y_train=y)
    with pytest.raises(ValueError): model.predict(X, full_cov=True, batch_size=2)
    with pytest.raises(ValueError): model.predict(X, batch_size=0)


def test_gp_gradient_cov_full_and_diagonal(xy):
    X,y=xy
    m=GP(SEKernel(dim=2),X,y,noise=.04)
    Q=jnp.array([[.35,.25],[-.1,.33]])
    g,cov=m.predict_gradient(Q,full_cov=True)
    _,v=m.predict_gradient(Q)
    assert g.shape==(2,2)
    assert cov.shape==(4,4)
    np.testing.assert_allclose(v, jnp.diag(cov).reshape(2,2).T,atol=1e-8)
    assert np.linalg.eigvalsh(np.asarray(cov)).min()>=-1e-7


def test_gp_batching(xy):
    X,y=xy
    m=GP(SEKernel(dim=2),X,y,noise=.05)
    Q=jnp.tile(X,(3,1))
    a=m.predict(Q)
    b=m.predict(Q,batch_size=4)
    np.testing.assert_allclose(a[0],b[0],atol=1e-8)
    np.testing.assert_allclose(a[1],b[1],atol=1e-8)
    np.testing.assert_allclose(m.predict_mean(Q,batch_size=4),a[0],atol=1e-8)
    assert m.predict(Q[:0],batch_size=4)[0].size==0


def test_change_kernel_parameter_after_conditioning(xy):
    X,y=xy
    k=SEKernel(dim=2)
    m=GP(k,X,y).condition()
    before=m.predict_mean(X)
    k.set_params([1.0,.2])
    after=m.predict_mean(X)
    assert not np.allclose(before,after,atol=1e-5)


def test_gp_hyperparameters_fit(xy):
    X,y=xy
    model=GP(SEKernel(sig_f=1.,l_f=2.,dim=2),X,y,noise=.1)
    before=float(model.negative_log_marginal_likelihood())
    res=model.fit_hyperparameters(maxiter=60)
    assert np.isfinite(res.fun)
    assert float(model.negative_log_marginal_likelihood()) <= before+1e-5
    np.testing.assert_allclose(model.predict(X)[0].shape, y.shape)
    assert len(model.alpha)==len(X)


def make_gpis(augment=False, kernel=None, prior=None):
    angle=jnp.linspace(0,2*jnp.pi, 9)[:-1]
    X=jnp.stack([jnp.cos(angle),jnp.sin(angle)],axis=1)
    return GPIS(kernel or SEKernel(dim=2,l_f=.8), X, X,
                augment=augment, geo_prior=prior, sig_x=.003, sig_n=.01,
                d=.04, jitter=1e-8)


def test_gpis_obs_and_augmentation():
    m=make_gpis(augment=False)
    obs=m._observations()
    assert obs.y.shape==(8+2*8,)
    np.testing.assert_allclose(obs.y[:8],jnp.zeros((8,)))
    np.testing.assert_allclose(obs.y[8:],m.N_train.T.reshape(-1))
    assert m._train_covariance(obs,jnp.asarray(m.kernel.params)).shape==(24,24)
    m.augment=True
    m.condition()
    assert m.Y_D.shape==(3*8+2*8,)
    np.testing.assert_allclose(m.Y_f.ravel()[8:],m.Y_a)
    np.testing.assert_allclose(m.X_f[8:],m.X_a)
    assert len(m.alpha)==len(m.Y_D)


def test_gpis_full_reference_gradient_mean():
    m=make_gpis(augment=False)
    X=m.X_train
    obs=m._observations()
    K=np.asarray(m.kernel.K_mat_partial(obs.X_value,obs.X_grad))
    K=K+np.diag(np.asarray(obs.noise)+m.jitter)
    alpha=np.linalg.solve(K,np.asarray(obs.y-obs.mean))
    P=jnp.array([[.4,.3],[-.9,-.3]])
    cross=np.asarray(m._cross_covariance(P, obs, jnp.asarray(m.kernel.params)))
    reference=np.asarray(m._mean_values(P))+cross@alpha
    np.testing.assert_allclose(m.pred_f_fast(P),reference,rtol=1e-8,atol=1e-8)
    gm, gv=m.pred_N(P)
    grad_ref=np.stack([np.asarray(jax.grad(lambda q: m.pred_f_fast(q[None])[0])(p)) for p in P])
    np.testing.assert_allclose(gm,grad_ref,atol=2e-7)
    assert gv.shape==(2,2)
    assert bool(jnp.all(gv>=0))


def test_gpis_normal_derivative_covariance_consistency():
    m=make_gpis(augment=False)
    obs=m._observations()
    K=np.asarray(m._train_covariance(obs,jnp.asarray(m.kernel.params)))
    assert K.shape==(24,24)
    np.testing.assert_allclose(K,K.T,rtol=1e-8,atol=1e-8)
    assert np.linalg.eigvalsh(K).min() > -1e-7
    Y=jnp.array([[-.6,.3],[.9,-.2]])
    Kxy=np.asarray(m._cross_covariance(Y, obs, jnp.asarray(m.kernel.params)))
    # Finite difference in the test location matches cross derivative function.
    eps=1e-5
    for a in range(2):
        perturb=jnp.zeros_like(Y).at[:,a].set(eps)
        fd=(m._cross_covariance(Y+perturb, obs, jnp.asarray(m.kernel.params)) -
            m._cross_covariance(Y-perturb, obs, jnp.asarray(m.kernel.params)))/(2*eps)
        grad=np.asarray(m._grad_cross_covariance(Y,obs,jnp.asarray(m.kernel.params)))
        np.testing.assert_allclose(fd,grad[a*len(Y):(a+1)*len(Y)],atol=1e-7)


def test_gpis_uncertainty_zero_cov_equals_exact():
    m=make_gpis(augment=False)
    P=jnp.array([[.2,.1],[-.8,.3]])
    a=m.pred_f(P)
    b=m.pred_f(P,sig_x_input=jnp.zeros((2,2)))
    np.testing.assert_allclose(a[0],b[0])
    np.testing.assert_allclose(a[1],b[1],rtol=1e-7,atol=1e-7)
    u1=m.pred_cov_f(P,sig_x_input=jnp.eye(2)*.005)
    u2=m.pred_cov_f(P,sig_x_input=jnp.eye(2)*.005,uncertainty_order=2)
    assert np.isfinite(np.asarray(u1)).all()
    assert np.isfinite(np.asarray(u2)).all()


def test_gpis_prior_and_cache_invalidation():
    class LinearPrior:
        def __init__(self): self.slope=jnp.array([.3,-.2])
        def prior_f(self,X):return X @ self.slope
        def prior_grad_f(self,X):return jnp.broadcast_to(self.slope,X.shape)
        def get_params(self):return self.slope
        def set_params(self,p):self.slope=jnp.asarray(p)
        def prior_f_params(self,X,p):return X@p
        def prior_grad_f_params(self,X,p):return jnp.broadcast_to(p,X.shape)
    prior=LinearPrior()
    m=make_gpis(prior=prior)
    P=jnp.array([[.2,.25]])
    a=m.pred_f_fast(P)
    prior.set_params(jnp.array([.7,.6]))
    b=m.pred_f_fast(P)
    assert not np.allclose(a,b,atol=1e-6)
    assert np.isfinite(float(m.nll(jnp.array([1., .8, .01, .01, .5, .5]),fit_geo_prior=True)))


def test_gpis_parameter_fit_and_conditional(xy):
    m=make_gpis(augment=False)
    before=float(m.negative_log_marginal_likelihood())
    optimal=m.fit(fit_noise=True, maxiter=12)
    assert np.isfinite(np.asarray(optimal)).all()
    assert np.isfinite(m.optimization_result.fun)
    assert float(m.negative_log_marginal_likelihood()) <= before+1e-4
    assert bool(jnp.isfinite(m.pred_f_fast(m.X_train)).all())


def test_gpis_surface_statistics_and_gradients():
    m=make_gpis(augment=False)
    P=jnp.array([[.1,.2],[-.4,.1]])
    var=m.pred_cov_f(P)
    mval=m.pred_f_fast(P)
    np.testing.assert_allclose(m.log_surface_pdf(P),normal_logpdf(0.,mval,var))
    np.testing.assert_allclose(m.get_surface_pdf(P),jnp.exp(m.log_surface_pdf(P)))
    np.testing.assert_allclose(m.get_surface_cdf(P,-.01,.01),jnp.exp(m.log_surface_cdf(P,-.01,.01)))
    assert m.get_surface_cdf(P,-.01,.01).shape==(2,)
    assert m.log_surface_pdf_grad(P).shape==(2,2)
    assert m.log_surface_cdf_grad(P).shape==(2,2)
    q=m.log_surface_pdf_grad(P)
    ref=jax.grad(lambda x:m.log_surface_pdf(x[None,:])[0])(P[0])
    np.testing.assert_allclose(q[0],ref,atol=1e-7)


def test_interval_probability_uses_variance():
    from scipy.stats import norm
    m=jnp.array([0., 8., -8.])
    var=jnp.array([4., .25, .25])
    p=np.asarray(jnp.exp(normal_interval_logprob(-1.,1.,m,var)))
    expected=norm.cdf((1.-np.asarray(m))/np.sqrt(np.asarray(var)))-norm.cdf((-1.-np.asarray(m))/np.sqrt(np.asarray(var)))
    np.testing.assert_allclose(p,expected,atol=1e-9)
    assert np.isneginf(float(normal_interval_logprob(1.,-1.)))
    assert np.isneginf(float(normal_interval_logprob(0.,0.)))


def test_gpis_surface_observation_and_legacy_api():
    m=make_gpis(augment=False)
    q=jnp.array([[.2,.3]])
    mu_before=float(m.pred_f_fast(q)[0])
    m.add_surface_observations(q, .75, noise=.001)
    mu_after=float(m.pred_f_fast(q)[0])
    assert abs(mu_after-.75)<abs(mu_before-.75)
    assert m.cache_kerne_mat() is m
    Xf,Xn,Sig,mean,Y=m.prepare_model_input()
    assert Sig.shape[0]==len(Y)
    assert Xf.shape[1]==Xn.shape[1]==2
    assert m.get_pred_f_grad(q).shape==(1,2)
    assert m.pred_f_single(q[0]).shape==(1,)
    assert m.pred_f_seq(q).shape==(1,)
    assert m.pred_N(q,mean_only=True).shape==(1,2)


def test_gpis_serialization_roundtrip(tmp_path):
    m=make_gpis(augment=True)
    p=jnp.array([[.2,.3],[-.1,.4]])
    a=m.pred_f(p)
    path=m.save_model(tmp_path,'model')
    assert path.exists()
    b=make_gpis(augment=False)
    b.load_gpis(tmp_path,'model')
    np.testing.assert_allclose(a[0],b.pred_f(p)[0],rtol=1e-7)
    np.testing.assert_allclose(a[1],b.pred_f(p)[1],rtol=1e-7)


def test_gpis_2d_grid_not_recentered():
    X=jnp.array([[5.,10.],[6.,10.],[6.,11.],[5.,11.]])
    m=GPIS(SEKernel(dim=2),X,jnp.array([[1.,0.]]*4),augment=False)
    grid=m.init_grid(mult=1.,grid_size=7)
    np.testing.assert_allclose(grid.mean(0),jnp.array([5.5,10.5]),atol=1e-7)
    assert m.pred_f()[0].shape==(49,), 'GPIS pred_f defaults to the grid and variance'


def test_optional_libraries_not_imported_eagerly():
    import sys
    assert 'open3d' not in sys.modules
    assert 'plotly' not in sys.modules



def test_generic_gp_posterior_sampling_and_std(xy):
    X,y=xy
    m=GP(SEKernel(dim=2),X,y,noise=.025)
    P=jnp.array([[.2,.4],[-.5,.1]])
    mu,cov=m.predict(P,full_cov=True)
    samples=m.sample_posterior(jax.random.PRNGKey(8),P,n_samples=2500)
    assert samples.shape==(2500,2)
    np.testing.assert_allclose(np.mean(samples,axis=0),mu,atol=.045)
    np.testing.assert_allclose(m.predict_std(P)**2,jnp.diag(cov),atol=1e-8)


def test_uncertainty_batching_variable_covariance():
    m=make_gpis()
    P=jnp.array([[.1,.2],[-.1,.2],[.3,.4]])
    cov=jnp.stack([jnp.eye(2)*v for v in (.001,.002,.003)])
    J=jnp.stack([jnp.eye(2)*v for v in (1.,1.3,0.7)])
    single=m.predict_uncertain(P,cov,input_jacobian=J)
    batched=m.predict_uncertain(P,cov,input_jacobian=J,batch_size=2)
    for a,b in zip(single,batched):
        np.testing.assert_allclose(a,b,atol=1e-9)


def test_gpis_cloud_log_likelihood_batching():
    m=make_gpis()
    cloud=jnp.array([[[.1,.2],[.3,.4]],[[.2,.3],[.2,.1]],
                     [[.3,.3],[.5,.7]]])
    a=m.get_surface_likelihood_batched(cloud,batch_size=1)
    b=m.get_surface_likelihood_chunked(cloud,jnp.eye(2)*.001,batch_size=2)
    c=jax.vmap(lambda x:m.log_surface_pdf(x))(cloud)
    np.testing.assert_allclose(a,c,atol=1e-8)
    assert b.shape==(3,2)
    assert np.isfinite(np.asarray(b)).all()


def test_gpis_fit_prior_pure_functions():
    class OffsetPrior:
        def __init__(self):self.params=jnp.array([0.2])
        def prior_f(self,X):return jnp.full((len(X),),self.params[0])
        def prior_grad_f(self,X):return jnp.zeros_like(X)
        def get_params(self):return self.params
        def set_params(self,p):self.params=jnp.asarray(p)
        def prior_f_params(self,X,p):return jnp.full((len(X),),p[0])
        def prior_grad_f_params(self,X,p):return jnp.zeros_like(X)
    m=make_gpis(prior=OffsetPrior())
    v0=float(m.negative_log_marginal_likelihood())
    params=m.fit(fit_noise=False,fit_geo_prior=True,
              lower_bound_geo=-1.,upper_bound_geo=1.,maxiter=12)
    assert np.isfinite(np.asarray(params)).all()
    assert float(m.negative_log_marginal_likelihood())<=v0+1e-5


def test_invalid_noise_is_rejected(xy):
    X,y=xy
    with pytest.raises(ValueError,match='noise variances'):
        GP(SEKernel(dim=2),X,y,noise=-0.1).condition()


def test_augmentation_scaling_consistent_with_unit_normals():
    m=make_gpis(augment=True)
    np.testing.assert_allclose(m.f_aug_val,m.f_aug_d)
    np.testing.assert_allclose(m.Y_a[:m.n_train],jnp.ones((m.n_train,))*m.f_aug_d)
    m.d=.1
    np.testing.assert_allclose(m.X_a[:m.n_train],m.X_train+.1*m.N_train)
    assert m.d==m.f_aug_d==m.f_aug_val==.1
