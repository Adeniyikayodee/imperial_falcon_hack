"""Reference trajectories as smooth position functions; derivatives come from forward-mode AD."""
import jax
import jax.numpy as jnp


def smoothstep(s):
    s = jnp.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def hover(center=(0.0, 0.0, 1.0)):
    c = jnp.array(center)
    return lambda t: c + 0.0 * t


def figure8(center=(0.0, 0.0, 1.0), radius=1.0, period=6.0, ramp=2.0):
    """Lemniscate whose amplitude eases in over `ramp` seconds so it starts at rest."""
    c, w = jnp.array(center), 2 * jnp.pi / period

    def pos(t):
        a = radius * smoothstep(t / ramp)
        return c + a * jnp.array([jnp.sin(w * t), 0.5 * jnp.sin(2 * w * t), 0.0])

    return pos


def lap(center=(0.0, 0.0, 1.0), rx=2.0, ry=1.2, period=10.0, ramp=2.0):
    """Elliptical lap that meets a steady wind as headwind, crosswind and tailwind in turn.

    Arc phase s(t) eases in over `ramp` seconds (C2 at the joint), so it starts at rest at (rx, 0).
    """
    c, w = jnp.array(center), 2 * jnp.pi / period

    def pos(t):
        early = t**3 / ramp**2 - t**4 / (2 * ramp**3)
        s = jnp.where(t < ramp, early, t - ramp / 2)
        return c + jnp.array([rx * jnp.cos(w * s), ry * jnp.sin(w * s), 0.0])

    return pos


def reference(pos, t, yaw=0.0):
    vel = lambda s: jax.jvp(pos, (s,), (1.0,))[1]
    p, v = jax.jvp(pos, (t,), (1.0,))
    a = jax.jvp(vel, (t,), (1.0,))[1]
    return p, v, a, yaw


def inspect(center=(0.0, -2.6, 20.0), rx=4.0, ry=0.6, period=14.0, ramp=2.0):
    """Power-line inspection pass: a long loop at conductor height (20 m) beside a line along x."""
    return lap(center, rx, ry, period, ramp)


def street(center=(0.0, 2.0, 8.0), rx=9.0, ry=0.5, period=20.0, ramp=2.0):
    """Facade inspection: a long loop at 8 m in an 8 m-wide street, 2 m from the north wall."""
    return lap(center, rx, ry, period, ramp)


def station(center=(0.0, -2.6, 20.0)):
    """Station-keeping beside the line at the inspection height: where drag is identified."""
    return hover(center)


TRAJECTORIES = {"station": station, "hover": hover, "figure8": figure8, "lap": lap, "inspect": inspect, "street": street}
