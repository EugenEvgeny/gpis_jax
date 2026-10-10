"""Vector-valued range priors and opt-in scalar channel selection for GPIS."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .base_geometric_prior import BaseGeometricPrior, _points, _parameters, _finite_bounds


class AnchorPrior(BaseGeometricPrior):
    """Per-channel geometric means: f_j(x) = s_j ||x-a_j|| + c_j.

    By default ``prior_f(X)`` returns (n,n_anchors), and
    ``prior_grad_f(X)`` returns (n,n_anchors,d). The general GP and GPIS
    are scalar-output, so select a channel using ``channel_id=...`` in the
    constructor, ``for_channel(id)``, or pass a single-anchor instance.

    The channel mapping is static Python metadata (not differentiable).
    Anchor positions, range biases, and constant biases are trainable.
    """
    def __init__(self, anchor_positions, channel_ids=None, range_bias=None,
                 constant_bias=None, theta=None, mean=None, dim=2,
                 channel_id=None, distance_epsilon=1e-8):
        super().__init__(dim=dim, theta=theta, mean=mean)
        anchors = _points(anchor_positions, dim)
        if len(anchors) < 1:
            raise ValueError('At least one anchor is required')
        self.anchor_positions = anchors
        self.n_anchors = len(anchors)
        self.range_bias = (jnp.ones(self.n_anchors, dtype=anchors.dtype) if range_bias is None
                           else jnp.asarray(range_bias).reshape(self.n_anchors))
        self.constant_bias = (jnp.zeros(self.n_anchors, dtype=anchors.dtype) if constant_bias is None
                              else jnp.asarray(constant_bias).reshape(self.n_anchors))
        if channel_ids is None:
            channel_ids = tuple(range(self.n_anchors))
        if len(channel_ids) != self.n_anchors or len(set(channel_ids)) != self.n_anchors:
            raise ValueError('channel_ids must have one distinct entry per anchor')
        self.channel_ids = tuple(int(v) for v in channel_ids)
        self._channel_id_to_index = {c: j for j, c in enumerate(self.channel_ids)}
        self.distance_epsilon = float(distance_epsilon)
        if self.distance_epsilon <= 0:
            raise ValueError('distance_epsilon must be strictly positive')
        self.channel_id = channel_id
        if channel_id is not None:
            self.channel_index(channel_id)

    @property
    def channels(self):
        return jnp.arange(self.n_anchors)

    @property
    def params(self):
        return self.get_params()

    @property
    def n_anchor_params(self):
        return self.n_anchors*self.dim

    @property
    def anchor_params(self):
        return self.anchor_positions.reshape(-1)

    @property
    def bias_params(self):
        return jnp.concatenate([self.range_bias, self.constant_bias])

    def channel_index(self, channel_id):
        try:
            return self._channel_id_to_index[int(channel_id)]
        except KeyError as exc:
            raise ValueError(f'Unknown channel ID {channel_id}; known: {self.channel_ids}') from exc

    def for_channel(self, channel_id):
        """Independent prior copy with one scalar output channel selected."""
        return AnchorPrior(self.anchor_positions, channel_ids=self.channel_ids,
                           range_bias=self.range_bias, constant_bias=self.constant_bias,
                           mean=self.mean, theta=self.theta, dim=self.dim,
                           channel_id=channel_id, distance_epsilon=self.distance_epsilon)

    def get_params(self):
        return jnp.concatenate([self.anchor_params, self.range_bias, self.constant_bias])

    def unpack_params(self, params):
        p = _parameters(params, self.n_anchors*(self.dim+2), 'AnchorPrior')
        a = p[:self.n_anchor_params].reshape(self.n_anchors, self.dim)
        s = p[self.n_anchor_params:self.n_anchor_params+self.n_anchors]
        c = p[self.n_anchor_params+self.n_anchors:]
        return a, s, c

    def set_anchors(self, anchors):
        self.anchor_positions = jnp.asarray(anchors).reshape(self.n_anchors, self.dim)
        return self

    def set_range_bias(self, bias):
        self.range_bias = jnp.asarray(bias).reshape(self.n_anchors)
        return self

    def set_constant_bias(self, bias):
        self.constant_bias = jnp.asarray(bias).reshape(self.n_anchors)
        return self

    def set_params(self, params):
        self.anchor_positions, self.range_bias, self.constant_bias = self.unpack_params(params)
        return self

    def parameter_bounds(self):
        return _finite_bounds(len(self.get_params()))

    def cache_signature(self):
        return (super().cache_signature(), self.channel_id)

    def _evaluate(self, X, params, channel_id=None):
        X = _points(X, self.dim)
        anchors, slope, bias = self.unpack_params(params)
        diff = X[:, None, :] - anchors[None, :, :]
        # Smooth norm: well-defined derivatives at the anchor itself.
        norm = jnp.sqrt(jnp.sum(diff**2, axis=-1) + self.distance_epsilon**2)
        values = slope[None, :]*norm + bias[None, :]
        gradient = slope[None, :, None] * (diff/norm[..., None])
        selected = self.channel_id if channel_id is None else channel_id
        if selected is not None:
            idx = self.channel_index(selected)
            return values[:, idx], gradient[:, idx, :]
        if self.n_anchors == 1:
            return values[:, 0], gradient[:, 0, :]
        return values, gradient

    def prior_f_params(self, X, params):
        return self._evaluate(X, params)[0]

    def prior_grad_f_params(self, X, params):
        return self._evaluate(X, params)[1]

    def prior_f(self, X, channel_id=None):
        return self._evaluate(X, self.get_params(), channel_id)[0]

    def prior_grad_f(self, X, channel_id=None):
        return self._evaluate(X, self.get_params(), channel_id)[1]

    def mean_fun(self, X, params):
        return self.single_prior_f(X, channels=self.channels, params=params)

    def single_prior_f(self, X, channels=None, params=None):
        X = _points(X, self.dim)
        anchors, slope, bias = self.unpack_params(self.get_params() if params is None else params)
        channels = self.channels if channels is None else jnp.asarray(channels, dtype=int)
        diff = X[:, None, :] - anchors[channels][None, :, :]
        norm = jnp.sqrt(jnp.sum(diff**2, axis=-1) + self.distance_epsilon**2)
        return slope[channels][None, :]*norm + bias[channels][None, :]
