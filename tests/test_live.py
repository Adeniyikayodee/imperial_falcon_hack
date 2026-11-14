import jax.numpy as jnp
import numpy as np

import falcon  # noqa: F401
from falcon.live import BURST_FROM, BURST_S, BURST_V, MAX_BURSTS, Live, field


def run(live, seconds):
    return [live.tick() for _ in range(int(seconds / (live.n * 0.002)))]


def test_still_air_holds_the_path():
    f = run(Live(), 6.0)[-1]
    for m in f["models"].values():
        assert not m["crashed"]
        assert np.linalg.norm(np.array(m["pos"]) - np.array(f["ref"])) < 0.1


def test_live_is_deterministic():
    a, b = Live(), Live()
    for live in (a, b):
        live.set_wind(mean=(3.0, 0.0), gust=1.0)
    np.testing.assert_array_equal(run(a, 2.0)[-1]["models"]["ph"]["pos"], run(b, 2.0)[-1]["models"]["ph"]["pos"])


def test_gust_front_travels_downwind():
    live = Live()
    turb, none = live.turb, jnp.zeros(3)
    t0, d, sp = jnp.full(MAX_BURSTS, -1e9).at[0].set(0.0), jnp.tile(jnp.array([1.0, 0, 0]), (MAX_BURSTS, 1)), jnp.zeros(MAX_BURSTS).at[0].set(6.0)
    ts = jnp.linspace(0, 4, 4001)

    def peak_time(x):
        w = jnp.stack([field(jnp.array([x, 0.0, 1.0]), t, none, turb, 0.0, t0, d, sp)[0] for t in ts[::10]])
        return float(ts[::10][jnp.argmax(w)]), float(w.max())

    (ta, wa), (tb, wb) = peak_time(-2.0), peak_time(2.0)
    np.testing.assert_allclose(tb - ta, 4.0 / BURST_V, atol=0.02)
    np.testing.assert_allclose([wa, wb], 6.0, atol=0.05)
    np.testing.assert_allclose(ta, (-2.0 + BURST_FROM) / BURST_V + BURST_S / 2, atol=0.02)


def test_burst_reaches_the_drones_then_passes():
    live = Live()
    run(live, 2.0)
    live.burst(speed=6.0)
    frames = run(live, (2 * BURST_FROM) / BURST_V + BURST_S)
    winds = np.array([f["models"]["none"]["wind"][0] for f in frames])
    assert winds.max() > 5.0 and abs(winds[-1]) < 1e-9
    assert len(frames[0]["field"]["bursts"]) == 1
    assert not run(live, 1.2)[-1]["field"]["bursts"]  # dropped once the front has left the scene


def test_reset_restarts_learning_on_the_path():
    live = Live()
    live.set_wind(mean=(4.0, 0.0), learning="aggressive")
    run(live, 3.0)
    live.reset()
    f = live.tick()
    for m in f["models"].values():
        assert np.linalg.norm(np.array(m["pos"]) - np.array(f["ref"])) < 0.05


def test_open_site_leaves_the_wind_alone():
    from falcon.live import SITES, site_gain
    np.testing.assert_allclose(site_gain(jnp.array([3.0, 1.0, 8.0]), SITES["open"]), 1.0)


def test_street_canyon_shelters_channels_and_jets():
    from falcon.live import SITES, site_gain
    city = SITES["city"]
    mid_block, at_gap, above_roof = (site_gain(jnp.array(p), city) for p in ([0.0, 0.0, 8.0], [6.0, 0.0, 8.0], [0.0, 0.0, 25.0]))
    np.testing.assert_allclose(mid_block[:2], [city.channel, city.shelter], atol=1e-3)
    np.testing.assert_allclose(at_gap[:2], [city.channel, city.jet], atol=1e-6)
    np.testing.assert_allclose(above_roof, 1.0)


def test_city_flight_meets_gap_jets():
    live = Live(traj="street", site="city")
    live.set_wind(mean=(0.0, 4.0))
    cross = [f["models"]["ph"]["wind"][1] for f in run(live, 12.0)]
    from falcon.atmos import profile
    jet = live.site.jet * float(profile(8.0, live.site.z0)) * 4.0  # mean cross wind at 8 m, through a gap
    assert min(cross) < 1.5 and max(cross) > 0.9 * jet  # sheltered mid-block, jetting at the cross streets
