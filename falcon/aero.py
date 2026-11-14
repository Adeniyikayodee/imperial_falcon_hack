"""Ground-truth wind field and aerodynamics. Hidden from the controller and the learned model.

Everything depends on air-relative velocity v_rel = v - wind and vanishes when the drone moves with the air.
- Drag (rotor drag, linear and scaling with thrust, plus quadratic body drag) dissipates energy in the
  air-mass frame: v_rel . F <= 0. This is the structure the ph residual assumes.
- Two effects outside that structure, so the residual is not tested on its own model class:
  induced thrust from in-plane airspeed along body z (Faessler, Franchi, Scaramuzza 2018, k_h term),
  which can do positive work, and rotor drag acting at the rotor plane `h` above the centre of mass,
  which pitches the vehicle (a torque no force model can represent).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from falcon.dynamics import quat_to_rot


class Wind(NamedTuple):
    mean: jnp.ndarray  # (3,) m/s
    amp: jnp.ndarray  # (K, 3) gust amplitude per mode
    freq: jnp.ndarray  # (K,) Hz
    phase: jnp.ndarray  # (K,) rad


class Drag(NamedTuple):
    rotor: tuple = (0.010, 0.010, 0.004)  # N per m/s at hover thrust, body axes
    quad: float = 0.0018  # 0.5 rho Cd A, N per (m/s)^2
    lift: float = 0.01  # induced thrust per unit mass per (m/s)^2 of in-plane airspeed (k_h), 1/m
    h: float = 0.01  # rotor plane above the centre of mass, m


def make_wind(key, mean=(0.0, 0.0, 0.0), gust=0.0, modes=6, fmax=2.0):
    """Sum-of-sines gust model. Deterministic for a given key."""
    k1, k2, k3 = jax.random.split(key, 3)
    amp = gust * jax.random.normal(k1, (modes, 3)) / jnp.sqrt(modes)
    freq = jax.random.uniform(k2, (modes,), minval=0.1, maxval=fmax)
    phase = jax.random.uniform(k3, (modes,), maxval=2 * jnp.pi)
    return Wind(jnp.asarray(mean, dtype=float), amp, freq, phase)


def wind_at(t, w: Wind):
    return w.mean + jnp.sin(2 * jnp.pi * w.freq * t + w.phase) @ w.amp


def drag_force(v, q, thrust_ratio, wind, d: Drag = Drag()):
    """Dissipative part: rotor drag (linear, scales with thrust) plus quadratic body drag, world frame."""
    v_rel = v - wind
    R = quat_to_rot(q)
    rotor = -R @ (jnp.array(d.rotor) * (R.T @ v_rel)) * thrust_ratio
    body = -d.quad * jnp.linalg.norm(v_rel) * v_rel
    return rotor + body


def aero(v, q, thrust_ratio, wind, mass, d: Drag = Drag()):
    """Full aerodynamics: world-frame force and body-frame torque."""
    R = quat_to_rot(q)
    vb = R.T @ (v - wind)
    rotor_b = -jnp.array(d.rotor) * vb * thrust_ratio
    lift = mass * d.lift * (vb[0] ** 2 + vb[1] ** 2) * R[:, 2]
    tau = jnp.cross(jnp.array([0.0, 0.0, d.h]), rotor_b)
    return drag_force(v, q, thrust_ratio, wind, d) + lift, tau
