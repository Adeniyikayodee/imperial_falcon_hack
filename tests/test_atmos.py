import jax
import jax.numpy as jnp
import numpy as np

import falcon  # noqa: F401
from falcon.atmos import KAPPA, SIGMA_PER_USTAR, make_turbulence, profile, wind

TURB = make_turbulence(jax.random.PRNGKey(0))


def test_log_profile_is_one_at_10m_and_grows_with_height():
    assert abs(float(profile(10.0, 0.03)) - 1.0) < 1e-12
    np.testing.assert_allclose(profile(20.0, 0.03), np.log(20 / 0.03) / np.log(10 / 0.03), rtol=1e-12)
    assert profile(2.0, 0.03) < 1.0 < profile(40.0, 0.03)


def test_turbulence_has_the_surface_layer_standard_deviations():
    mean, z0 = jnp.array([6.0, 0.0, 0.0]), 0.03
    ts = jnp.arange(0.0, 3000.0, 0.25)
    w = jax.vmap(lambda t: wind(jnp.array([0.0, 0.0, 10.0]), t, mean, TURB, 1.0, z0))(ts)
    ustar = KAPPA * 6.0 / np.log(10 / z0)
    np.testing.assert_allclose(w.mean(0), [6.0, 0.0, 0.0], atol=0.35)
    np.testing.assert_allclose(w.std(0), np.array(SIGMA_PER_USTAR) * ustar, rtol=0.2)


def test_eddies_keep_their_size_and_pass_faster_in_stronger_wind():
    """Frozen turbulence: the along-wind signal at speed U equals the one at U/2 played twice as fast."""
    p = jnp.array([0.0, 0.0, 10.0])
    slow = wind(p, 2.0, jnp.array([4.0, 0.0, 0.0]), TURB, 1.0, 0.03)[0] / 4.0 - 1.0
    fast = wind(p, 1.0, jnp.array([8.0, 0.0, 0.0]), TURB, 1.0, 0.03)[0] / 8.0 - 1.0
    np.testing.assert_allclose(slow, fast, atol=1e-9)


def test_calm_air_is_calm():
    np.testing.assert_allclose(wind(jnp.array([3.0, 1.0, 20.0]), 7.0, jnp.zeros(3), TURB, 1.0, 0.03), 0.0, atol=1e-12)
