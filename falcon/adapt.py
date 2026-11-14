"""Online residual adaptation from IMU-derived force samples.

The accelerometer measures specific force, so F_obs = m R a_imu - T R e3 isolates the
aerodynamic force up to sensor noise. Each update takes a clipped gradient step on the
mean squared error over a sliding window, plus decay toward the prior parameters so the
model forgets stale wind instead of drifting.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class AdaptCfg(NamedTuple):
    lr: float = 0.05
    window: int = 128
    every: int = 5  # control steps between updates
    decay: float = 1e-4
    clip: float = 1.0
    scale: float = 0.265  # force normalizer, about m g for a Crazyflie


class Buffer(NamedTuple):
    v: jnp.ndarray
    q: jnp.ndarray
    tr: jnp.ndarray
    F: jnp.ndarray
    n: jnp.ndarray


def empty(window):
    return Buffer(jnp.zeros((window, 3)), jnp.tile(jnp.array([1.0, 0, 0, 0]), (window, 1)),
                  jnp.ones(window), jnp.zeros((window, 3)), jnp.array(0))


def push(b: Buffer, v, q, tr, F):
    i = b.n % b.v.shape[0]
    return Buffer(b.v.at[i].set(v), b.q.at[i].set(q), b.tr.at[i].set(tr), b.F.at[i].set(F), b.n + 1)


def loss(params, apply, b: Buffer, prior, cfg: AdaptCfg):
    mask = (jnp.arange(b.v.shape[0]) < b.n).astype(float)
    err = jax.vmap(apply, (None, 0, 0, 0))(params, b.v, b.q, b.tr) - b.F
    mse = jnp.sum(mask * jnp.sum(err**2, 1)) / jnp.maximum(mask.sum(), 1.0) / cfg.scale**2
    drift = sum(jnp.sum((a - c) ** 2) for a, c in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(prior)))
    return mse + cfg.decay * drift


def update(params, apply, b: Buffer, prior, cfg: AdaptCfg):
    g = jax.grad(loss)(params, apply, b, prior, cfg)
    norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g)))
    k = cfg.lr * jnp.minimum(1.0, cfg.clip / (norm + 1e-12))
    return jax.tree_util.tree_map(lambda p, d: p - k * d, params, g)
