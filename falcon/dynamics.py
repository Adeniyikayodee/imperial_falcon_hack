"""Rigid-body quadrotor dynamics and an RK4 integrator.

State x (13,): position p, velocity v (world), quaternion q (w, x, y, z, body to world),
body rate w. Input: motor thrusts f (4,), an external world-frame force F_ext (3,) and body torque tau_ext (3,).
"""
import jax.numpy as jnp

from falcon.params import Quad, allocation

E3 = jnp.array([0.0, 0.0, 1.0])


def split(x):
    return x[0:3], x[3:6], x[6:10], x[10:13]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_rot(q):
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def wrench(f, p: Quad):
    """Collective thrust and body torque from clipped motor thrusts."""
    u = allocation(p) @ jnp.clip(f, 0.0, p.f_max)
    return u[0], u[1:]


def deriv(x, f, F_ext, p: Quad, tau_ext=jnp.zeros(3)):
    _, v, q, w = split(x)
    T, tau = wrench(f, p)
    J = p.J
    acc = (quat_to_rot(q) @ (T * E3) + F_ext) / p.mass - p.g * E3
    qdot = 0.5 * quat_mul(q, jnp.concatenate([jnp.zeros(1), w]))
    wdot = jnp.linalg.solve(J, tau + tau_ext - jnp.cross(w, J @ w))
    return jnp.concatenate([v, acc, qdot, wdot])


def step(x, f, F_ext, p: Quad, dt: float, tau_ext=jnp.zeros(3)):
    """One RK4 step with inputs held constant (zero-order hold), then renormalize q."""
    k1 = deriv(x, f, F_ext, p, tau_ext)
    k2 = deriv(x + 0.5 * dt * k1, f, F_ext, p, tau_ext)
    k3 = deriv(x + 0.5 * dt * k2, f, F_ext, p, tau_ext)
    k4 = deriv(x + dt * k3, f, F_ext, p, tau_ext)
    x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    q = x[6:10] / jnp.linalg.norm(x[6:10])
    return x.at[6:10].set(q)


def hover_state(pos=(0.0, 0.0, 1.0)):
    return jnp.concatenate([jnp.array(pos), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3)])
