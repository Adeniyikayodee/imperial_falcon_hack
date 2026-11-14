import jax
import jax.numpy as jnp
import numpy as np

import falcon  # noqa: F401
from falcon.aero import make_wind
from falcon.sim import metrics, rollout
from falcon.traj import figure8, reference

KEY = jax.random.PRNGKey(0)
F8 = figure8()


def wind(mean, gust=0.0):
    return make_wind(jax.random.PRNGKey(1), mean=mean, gust=gust)


def test_reference_derivatives_match_finite_differences():
    t, h = 3.3, 1e-5
    p, v, a, _ = reference(F8, t)
    np.testing.assert_allclose(v, (F8(t + h) - F8(t - h)) / (2 * h), atol=1e-6)
    np.testing.assert_allclose(a, (F8(t + h) - 2 * p + F8(t - h)) / h**2, atol=1e-3)


def test_tracks_figure8_in_still_air():
    m = metrics(rollout("none", F8, wind((0.0, 0.0, 0.0)), KEY)[0])
    assert not m["crashed"] and m["rmse_m"] < 0.03


def test_integral_action_removes_steady_wind_offset():
    from falcon.traj import hover
    log = rollout("none", hover(), wind((3.0, 0.0, 0.0)), KEY)[0]
    assert float(jnp.linalg.norm(log["pos"][-1000:] - log["ref"][-1000:], axis=1).mean()) < 0.005


def test_ph_beats_an_integral_controller_in_wind():
    w = wind((4.0, 0.0, 0.0))
    base = metrics(rollout("none", F8, w, KEY)[0])
    ph = metrics(rollout("ph", F8, w, KEY)[0])
    assert not ph["crashed"] and ph["rmse_m"] < 0.5 * base["rmse_m"]


def test_fast_learning_ph_stays_bounded_but_integral_control_alone_is_better():
    from falcon.adapt import AdaptCfg
    from falcon.sim import SimCfg
    cfg, w = SimCfg(learn=AdaptCfg(lr=0.5)), wind((4.0, 0.0, 0.0), 1.5)
    r = {m: metrics(rollout(m, F8, w, KEY, cfg)[0])["rmse_m"] for m in ("none", "mlp", "ph")}
    assert r["ph"] < 0.7 * r["mlp"] and r["none"] < r["ph"]


def test_rollout_is_deterministic():
    a = rollout("ph", F8, wind((2.0, 1.0, 0.0), 1.0), KEY)[0]
    b = rollout("ph", F8, wind((2.0, 1.0, 0.0), 1.0), KEY)[0]
    np.testing.assert_array_equal(a["pos"], b["pos"])



def test_logged_motor_thrusts_stay_within_limits():
    m = rollout("ph", F8, wind((4.0, 0.0, 0.0), 1.5), KEY)[0]["motors"]
    assert float(m.min()) >= 0.0 and float(m.max()) <= 0.15


def test_lap_starts_at_rest_and_is_smooth_through_the_ramp():
    from falcon.traj import lap
    L = lap()
    p0, v0, a0, _ = reference(L, 0.0)
    np.testing.assert_allclose(p0, [2.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(v0, 0.0, atol=1e-12)
    np.testing.assert_allclose(a0, 0.0, atol=1e-12)
    for t in (2.0 - 1e-6, 2.0 + 1e-6):  # velocity and acceleration continuous at the joint
        np.testing.assert_allclose(reference(L, t)[1], reference(L, 2.0)[1], atol=1e-5)
        np.testing.assert_allclose(reference(L, t)[2], reference(L, 2.0)[2], atol=1e-4)


def test_ph_holds_the_lap_better_than_no_model_in_wind():
    from falcon.traj import lap
    w = wind((4.0, 0.0, 0.0), 1.0)
    base = metrics(rollout("none", lap(), w, KEY)[0])
    ph = metrics(rollout("ph", lap(), w, KEY)[0])
    assert not ph["crashed"] and ph["rmse_m"] < 0.6 * base["rmse_m"]


def test_ph_keeps_its_standoff_when_wind_blows_toward_the_line():
    from falcon.traj import inspect
    w = make_wind(jax.random.PRNGKey(1000), mean=(0.0, 5.0, 0.0), gust=1.0)
    closest = {m: float(rollout(m, inspect(), w, KEY)[0]["pos"][2500:, 1].max()) for m in ("none", "ph")}
    assert closest["ph"] < -1.97 and closest["ph"] < closest["none"]  # planned edge of the pass is y = -2.0
