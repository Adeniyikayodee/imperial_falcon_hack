import jax
import jax.numpy as jnp
import numpy as np

import falcon  # noqa: F401
from falcon.control import control
from falcon.dynamics import hover_state, step
from falcon.params import Quad

P = Quad()
DT = 0.002
TARGET = (jnp.array([0.0, 0.0, 1.0]), jnp.zeros(3), jnp.zeros(3), 0.0)


def fly(x0, F_ext, F_hat, seconds=4.0):
    def body(x, _):
        f, _ = control(x, TARGET, F_hat, P)
        return step(x, f, F_ext, P, DT), None

    return jax.lax.scan(body, x0, None, length=int(seconds / DT))[0]


def test_recovers_from_offset_and_yaw():
    x0 = hover_state((0.4, -0.3, 0.6)).at[6:10].set(jnp.array([jnp.cos(0.4), 0, 0, jnp.sin(0.4)]))
    x = fly(x0, jnp.zeros(3), jnp.zeros(3))
    np.testing.assert_allclose(x[0:3], TARGET[0], atol=2e-3)
    np.testing.assert_allclose(x[6:10], [1, 0, 0, 0], atol=2e-3)


def test_perfect_feedforward_cancels_disturbance():
    F = jnp.array([0.05, -0.03, 0.0])
    with_ff = fly(hover_state(), F, F)
    without = fly(hover_state(), F, jnp.zeros(3))
    np.testing.assert_allclose(with_ff[0:3], TARGET[0], atol=1e-6)
    # P-D only: steady-state offset is F / (m * Kp)
    np.testing.assert_allclose(without[0:2], F[:2] / (P.mass * 16.0), rtol=0.02)
