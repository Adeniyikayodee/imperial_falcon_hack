import jax
import jax.numpy as jnp

import falcon  # noqa: F401
from falcon.residual import D_MAX, W_MAX, mlp_apply, mlp_init, ph_apply, ph_damping, ph_init, wind_hat

N = 256


def scramble(params, key, scale):
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(tree, [scale * jax.random.normal(k, l.shape) for k, l in zip(keys, leaves)])


def samples(key):
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k2, (N, 4))
    return 15 * jax.random.normal(k1, (N, 3)), q / jnp.linalg.norm(q, axis=1, keepdims=True), jax.random.uniform(k3, (N,), maxval=2.5)


def test_ph_is_passive_for_any_parameters():
    v, q, tr = samples(jax.random.PRNGKey(1))
    for i in range(20):
        p = scramble(ph_init(jax.random.PRNGKey(i)), jax.random.PRNGKey(100 + i), scale=5.0)
        F = jax.vmap(ph_apply, (None, 0, 0, 0))(p, v, q, tr)
        v_rel = v - wind_hat(p)
        assert jnp.all(jnp.sum(v_rel * F, axis=1) <= 0.0)
        D = jax.vmap(ph_damping, (None, 0, 0, 0))(p, v, q, tr)
        eig = jnp.linalg.eigvalsh(D)
        assert jnp.all(eig > 0) and jnp.all(eig < D_MAX)
        assert jnp.linalg.norm(wind_hat(p)) < W_MAX * jnp.sqrt(3)


def test_mlp_baseline_can_inject_energy():
    v, q, tr = samples(jax.random.PRNGKey(2))
    p = scramble(mlp_init(jax.random.PRNGKey(0)), jax.random.PRNGKey(3), scale=1.0)
    F = jax.vmap(mlp_apply, (None, 0, 0, 0))(p, v, q, tr)
    assert jnp.any(jnp.sum(v * F, axis=1) > 0)


def test_both_start_near_zero_force_at_rest():
    q = jnp.array([1.0, 0, 0, 0])
    assert jnp.linalg.norm(ph_apply(ph_init(jax.random.PRNGKey(0)), jnp.zeros(3), q, 1.0)) == 0
    assert jnp.linalg.norm(mlp_apply(mlp_init(jax.random.PRNGKey(0)), jnp.zeros(3), q, 1.0)) == 0


def test_gradients_finite_at_zero_relative_velocity():
    p = ph_init(jax.random.PRNGKey(0))
    g = jax.grad(lambda p: jnp.sum(ph_apply(p, jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), 1.0) ** 2))(p)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(g))
