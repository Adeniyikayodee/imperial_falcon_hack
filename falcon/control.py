"""Geometric SE(3) tracking controller (Lee, Leok, McClamroch 2010) with integral action and force feedforward.

F_hat is the learned aerodynamic force estimate. It is subtracted from the desired force, so a perfect
estimate cancels the disturbance exactly. The integral of position error removes steady-state offset
from any constant force, so a learned model is only credited for what integral action cannot do.
"""
from typing import NamedTuple

import jax.numpy as jnp

from falcon.dynamics import E3, quat_to_rot, split
from falcon.params import Quad, allocation


class Gains(NamedTuple):
    pos: float = 16.0  # wn = 4 rad/s
    vel: float = 8.0  # zeta = 1
    pos_i: float = 16.0  # s^3 + 8 s^2 + 16 s + 16: poles -5.7, -1.2 +/- 1.2j (Routh limit 128)
    i_max: float = 0.5  # anti-windup clamp on the error integral, m s
    att_wn: tuple = (25.0, 25.0, 10.0)
    att_zeta: float = 0.9
    max_tilt: float = 0.6  # rad


def vee(M):
    return jnp.array([M[2, 1], M[0, 2], M[1, 0]])


def control(x, ref, F_hat, p: Quad, g: Gains = Gains(), e_int=jnp.zeros(3)):
    """ref = (p_d, v_d, a_d, yaw_d); e_int the clamped integral of p_d - p. Returns motor thrusts (4,) and collective thrust."""
    pos, v, q, w = split(x)
    p_d, v_d, a_d, yaw_d = ref
    R = quat_to_rot(q)

    F = p.mass * (a_d + g.pos * (p_d - pos) + g.vel * (v_d - v) + g.pos_i * e_int + p.g * E3) - F_hat
    # tilt limit: keep the horizontal component within tan(max_tilt) of the vertical one
    Fz = jnp.maximum(F[2], 0.2 * p.hover_thrust)
    Fxy = F[:2] * jnp.minimum(1.0, jnp.tan(g.max_tilt) * Fz / (jnp.linalg.norm(F[:2]) + 1e-9))
    F = jnp.concatenate([Fxy, Fz[None]])

    b3 = F / jnp.linalg.norm(F)
    b2 = jnp.cross(b3, jnp.array([jnp.cos(yaw_d), jnp.sin(yaw_d), 0.0]))
    b2 = b2 / jnp.linalg.norm(b2)
    Rd = jnp.stack([jnp.cross(b2, b3), b2, b3], axis=1)

    J = jnp.array(p.inertia)
    wn = jnp.array(g.att_wn)
    e_R = 0.5 * vee(Rd.T @ R - R.T @ Rd)
    tau = -J * wn**2 * e_R - 2 * g.att_zeta * wn * J * w + jnp.cross(w, J * w)

    T = F @ (R @ E3)
    f = jnp.linalg.solve(allocation(p), jnp.concatenate([T[None], tau]))
    return f, T
