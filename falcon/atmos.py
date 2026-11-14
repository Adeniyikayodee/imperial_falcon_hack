"""Surface-layer wind: log profile with height and von Karman turbulence frozen into the air.

Mean wind: U(z) = U10 ln(z / z0) / ln(10 / z0), with U10 the speed at the standard 10 m height and
z0 the roughness length (0.03 m open grass, about 1 m city). Turbulence: friction velocity
u* = kappa U10 / ln(10 / z0) and standard deviations (2.4, 1.9, 1.25) u* along, across and
vertical (Panofsky and Dutton 1984). Each component is a sum of sines over log-spaced wavenumbers
weighted by the von Karman spectrum with its own length scale (vertical eddies are limited by the
ground to about the height), advected at the local mean speed, so eddies keep their size and pass
faster in stronger wind (Taylor's frozen turbulence).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

KAPPA = 0.4
SIGMA_PER_USTAR = (2.4, 1.9, 1.25)  # along, across, vertical


class Turb(NamedTuple):
    k: jnp.ndarray  # (K,) wavenumbers, rad/m
    a: jnp.ndarray  # (3, K) amplitudes, unit variance per component
    phase: jnp.ndarray  # (3, K) independent phases per component


def make_turbulence(key, modes=32, L=(150.0, 75.0, 20.0), smallest=1.5, largest=600.0):
    """L: von Karman length scales (along, across, vertical), m; vertical about the flight height."""
    k = jnp.geomspace(2 * jnp.pi / largest, 2 * jnp.pi / smallest, modes)
    Ls = jnp.asarray(L)[:, None]
    spectrum = Ls / (1 + (1.339 * Ls * k) ** 2) ** (5 / 6)
    a = jnp.sqrt(spectrum * jnp.gradient(k))
    a = a / jnp.sqrt(jnp.sum(a**2, axis=1, keepdims=True) / 2)  # sum of sines: variance is sum(a^2) / 2
    return Turb(k, a, jax.random.uniform(key, (3, modes), maxval=2 * jnp.pi))


def profile(z, z0):
    """Mean speed at height z relative to the 10 m reference."""
    return jnp.log(jnp.maximum(z, 2 * z0) / z0) / jnp.log(10.0 / z0)


def wind(p, t, mean10, turb: Turb, intensity, z0):
    """Mean plus turbulence at point p, time t. mean10 is the 10 m wind vector; intensity scales
    the turbulence (1 = standard surface layer, 0 = smooth)."""
    sp = jnp.linalg.norm(mean10[:2])
    u = jnp.where(sp > 0.2, mean10 / jnp.maximum(sp, 1e-9), jnp.array([1.0, 0.0, 0.0]))
    across = jnp.array([-u[1], u[0], 0.0])
    prof = profile(p[2], z0)
    xi = p @ u - jnp.maximum(sp * prof, 1.5) * t
    comp = jnp.sum(turb.a * jnp.sin(turb.k * xi + turb.phase), axis=1)  # (3,) unit-variance signals
    sigma = intensity * KAPPA * sp / jnp.log(10.0 / z0) * jnp.array(SIGMA_PER_USTAR) * comp
    return mean10 * prof + sigma[0] * u + sigma[1] * across + jnp.array([0.0, 0.0, sigma[2]])
