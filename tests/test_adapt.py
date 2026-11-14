import jax
import jax.numpy as jnp

import falcon  # noqa: F401
from falcon.adapt import AdaptCfg, empty, loss, push, update
from falcon.aero import drag_force
from falcon.residual import MODELS

WIND = jnp.array([4.0, -1.5, 0.0])


def dataset(key, n=128):
    k1, k2 = jax.random.split(key)
    v = 2.0 * jax.random.normal(k1, (n, 3))
    tilt = 0.2 * jax.random.normal(k2, (n, 3))
    q = jnp.concatenate([jnp.ones((n, 1)), 0.5 * tilt], 1)
    q = q / jnp.linalg.norm(q, axis=1, keepdims=True)
    F = jax.vmap(drag_force, (0, 0, None, None))(v, q, 1.0, WIND)
    b = empty(n)
    for i in range(n):
        b = push(b, v[i], q[i], 1.0, F[i])
    return b


def test_ring_buffer_wraps():
    b = empty(4)
    for i in range(6):
        b = push(b, jnp.full(3, float(i)), jnp.array([1.0, 0, 0, 0]), 1.0, jnp.zeros(3))
    assert int(b.n) == 6 and float(b.v[0, 0]) == 4.0 and float(b.v[3, 0]) == 3.0


def test_ph_adaptation_fits_wind_drag():
    b = dataset(jax.random.PRNGKey(0))
    init, apply = MODELS["ph"]
    p0 = init(jax.random.PRNGKey(1))
    cfg = AdaptCfg(lr=0.05, decay=0.0)
    step = jax.jit(lambda p: update(p, apply, b, p0, cfg))
    p = p0
    for _ in range(3000):
        p = step(p)
    assert loss(p, apply, b, p0, cfg) < 0.05 * loss(p0, apply, b, p0, cfg)
