import jax
import jax.numpy as jnp
import numpy as np

import falcon  # noqa: F401  enables x64
from falcon.dynamics import hover_state, quat_to_rot, step
from falcon.params import Quad, allocation

P = Quad()
DT = 0.002
ZERO = jnp.zeros(3)


def run(x, f, n, F=ZERO):
    body = lambda x, _: (step(x, f, F, P, DT), None)
    return jax.lax.scan(body, x, None, length=n)[0]


def test_hover_is_equilibrium():
    x0 = hover_state()
    x = run(x0, jnp.full(4, P.hover_thrust / 4), 2500)
    np.testing.assert_allclose(x, x0, atol=1e-9)


def test_free_fall_matches_closed_form():
    x = run(hover_state(), jnp.zeros(4), 500)  # 1 s
    np.testing.assert_allclose(x[2], 1.0 - 0.5 * P.g, atol=1e-9)
    np.testing.assert_allclose(x[5], -P.g, atol=1e-9)


def test_energy_conserved_without_thrust():
    x0 = hover_state().at[3:6].set(jnp.array([1.0, -0.5, 2.0])).at[10:13].set(jnp.array([3.0, -2.0, 5.0]))
    x = run(x0, jnp.zeros(4), 2000)

    def energy(x):
        w = x[10:13]
        return 0.5 * P.mass * x[3:6] @ x[3:6] + P.mass * P.g * x[2] + 0.5 * w @ P.J @ w

    np.testing.assert_allclose(energy(x), energy(x0), rtol=1e-8)


def test_quaternion_stays_unit_and_rotation_orthonormal():
    x = run(hover_state().at[10:13].set(jnp.array([4.0, 1.0, -7.0])), jnp.zeros(4), 1000)
    R = quat_to_rot(x[6:10])
    np.testing.assert_allclose(jnp.linalg.norm(x[6:10]), 1.0, atol=1e-12)
    np.testing.assert_allclose(R @ R.T, jnp.eye(3), atol=1e-10)


def test_allocation_signs():
    A = allocation(P)
    assert jnp.linalg.matrix_rank(A) == 4
    # more thrust on left motors (y > 0) rolls right side down: positive tau_x
    assert (A @ jnp.array([0, 0, 1.0, 1.0]))[1] > 0
    # more thrust on front motors pitches nose up: negative tau_y
    assert (A @ jnp.array([1.0, 0, 0, 1.0]))[2] < 0
