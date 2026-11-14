import jax
import jax.numpy as jnp
import numpy as np

import falcon  # noqa: F401
from falcon.aero import aero, drag_force, make_wind, wind_at


def random_quat(key):
    q = jax.random.normal(key, (4,))
    return q / jnp.linalg.norm(q)


def test_force_is_dissipative_in_air_frame():
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    v = 10 * jax.random.normal(keys[0], (512, 3))
    w = 10 * jax.random.normal(keys[1], (512, 3))
    q = jax.vmap(random_quat)(jax.random.split(keys[2], 512))
    F = jax.vmap(drag_force, (0, 0, None, 0))(v, q, 1.3, w)
    assert jnp.all(jnp.sum((v - w) * F, axis=1) <= 1e-15)


def test_no_force_when_moving_with_the_air():
    w = jnp.array([3.0, -1.0, 0.5])
    np.testing.assert_allclose(drag_force(w, jnp.array([1.0, 0, 0, 0]), 1.0, w), 0.0, atol=1e-15)


def test_full_aero_leaves_the_dissipative_class_but_vanishes_with_the_air():
    keys = jax.random.split(jax.random.PRNGKey(4), 3)
    v, w = 5 * jax.random.normal(keys[0], (512, 3)), 5 * jax.random.normal(keys[1], (512, 3))
    q = jax.vmap(random_quat)(jax.random.split(keys[2], 512))
    F, tau = jax.vmap(aero, (0, 0, None, 0, None))(v, q, 1.0, w, 0.027)
    lift = F - jax.vmap(drag_force, (0, 0, None, 0))(v, q, 1.0, w)
    assert jnp.any(jnp.sum((v - w) * lift, axis=1) > 0)  # induced lift is not of the form -D v_rel
    assert jnp.all(jnp.linalg.norm(tau, axis=1)[jnp.linalg.norm(v - w, axis=1) > 1] > 0)
    F0, tau0 = aero(w[0], q[0], 1.0, w[0], 0.027)
    np.testing.assert_allclose(jnp.concatenate([F0, tau0]), 0.0, atol=1e-15)


def test_wind_is_deterministic_and_centred_on_mean():
    a = make_wind(jax.random.PRNGKey(7), mean=(4.0, 0, 0), gust=2.0)
    b = make_wind(jax.random.PRNGKey(7), mean=(4.0, 0, 0), gust=2.0)
    ts = jnp.linspace(0, 200, 20001)
    wa = jax.vmap(wind_at, (0, None))(ts, a)
    np.testing.assert_array_equal(wa, jax.vmap(wind_at, (0, None))(ts, b))
    np.testing.assert_allclose(wa.mean(0), [4.0, 0, 0], atol=0.15)
