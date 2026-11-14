"""Crazyflie 2.x physical constants (SI units). Body frame: x forward, y left, z up."""
from typing import NamedTuple

import jax.numpy as jnp


class Quad(NamedTuple):
    mass: float = 0.027
    inertia: tuple = (1.66e-5, 1.66e-5, 2.93e-5)
    arm: float = 0.0325  # motor offset along body x and y
    yaw_coef: float = 0.006  # reaction torque per newton of thrust
    f_max: float = 0.15  # per motor
    g: float = 9.81

    @property
    def J(self):
        return jnp.diag(jnp.array(self.inertia))

    @property
    def hover_thrust(self):
        return self.mass * self.g


def allocation(p: Quad):
    """Map motor thrusts f (4,) to [T, tau_x, tau_y, tau_z]. X layout: FR, BR, BL, FL."""
    a, c = p.arm, p.yaw_coef
    x = jnp.array([a, -a, -a, a])
    y = jnp.array([-a, -a, a, a])
    spin = jnp.array([-1.0, 1.0, -1.0, 1.0])
    return jnp.stack([jnp.ones(4), y, -x, spin * c])
