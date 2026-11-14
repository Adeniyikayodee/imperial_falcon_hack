"""Live mode: the three models fly in real time while a presenter controls the wind.

The wind is a travelling surface-layer field (falcon.atmos): a log profile with height set by the
10 m wind and the site's roughness, plus von Karman turbulence frozen into the air. Each gust
burst is a sin^2 front that starts BURST_FROM metres upwind and sweeps through at BURST_V m/s. All drones share one sim
frame, so they meet the same air at the same time; the page draws their lanes side by side
across the wind and evaluates the same field for its streaks. Every control period goes
through sim.flight_step, so live flights obey exactly the batch physics, control and learning.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from falcon.adapt import AdaptCfg
from falcon import atmos
from falcon.params import Quad
from falcon.residual import MODELS
from falcon.sim import SimCfg, State, flight_step, init_state
from falcon.traj import TRAJECTORIES, reference

MAX_BURSTS = 4
BURST_S = 1.2  # pulse duration at a fixed point
BURST_V = 4.0  # front speed, m/s
BURST_FROM = 4.5  # fronts start this far upwind of the origin
LEARNING = {"gentle": 0.05, "aggressive": 0.5}
RECOVER_S = 2.0


class Site(NamedTuple):
    """Stylised street-canyon flow (Oke 1988; Britter and Hanna 2003). Street runs along x.

    Below roof height the along-street wind is channelled (x channel), the cross-street wind is
    sheltered by the buildings (x shelter), except at cross streets where it jets through the gap
    (x jet). Gaps are Gaussian bumps of half-width gap_w centred on `gaps`. city=0 disables it.
    """
    city: float = 0.0
    z0: float = 0.03  # roughness length, m: open grass
    roof: float = 18.0
    gaps: tuple = (-6.0, 6.0)
    gap_w: float = 2.0
    channel: float = 1.25
    shelter: float = 0.3
    jet: float = 1.6


SITES = {"open": Site(), "city": Site(city=1.0, z0=1.0)}


def site_gain(p, site: Site):
    """Per-axis multipliers (along x, across y) for the wind at point p."""
    below = site.city * jnp.clip((site.roof - p[2]) / 2.0, 0.0, 1.0)
    gap = jnp.minimum(1.0, jnp.sum(jnp.exp(-(((p[0] - jnp.asarray(site.gaps)) / site.gap_w) ** 2))))
    across = site.shelter + (site.jet - site.shelter) * gap
    return jnp.array([1.0 + below * (site.channel - 1.0), 1.0 + below * (across - 1.0), 1.0])


def field(p, t, mean, turb: atmos.Turb, gust, b_t0, b_dir, b_speed, site: Site = Site()):
    """Wind at world point p and time t; mean is the 10 m wind, gust the turbulence intensity
    (1 = standard surface layer). Mirrored line for line by the web pages."""
    tau = (t - b_t0) - (b_dir @ p + BURST_FROM) / BURST_V
    shape = jnp.where((tau >= 0) & (tau <= BURST_S), jnp.sin(jnp.pi * tau / BURST_S) ** 2, 0.0)
    return site_gain(p, site) * (atmos.wind(p, t, mean, turb, gust, site.z0) + (shape * b_speed) @ b_dir)


@partial(jax.jit, static_argnames=("model", "traj", "cfg", "quad", "n", "site"))
def advance(model, traj, cfg, quad, n, p0, k_noise, s: State, k0, mean, turb, gust, b_t0, b_dir, b_speed, site=Site()):
    def body(s, k):
        w = field(s.x[:3], k * cfg.dt, mean, turb, gust, b_t0, b_dir, b_speed, site)
        return flight_step(model, traj, cfg, quad, p0, k_noise, s, k, w)

    s, log = jax.lax.scan(body, s, k0 + jnp.arange(n))
    return s, jax.tree_util.tree_map(lambda a: a[-1], log)


@partial(jax.jit, static_argnames=("model", "traj", "cfg", "quad", "site"))
def rollout_field(model, traj, key, mean, turb, gust, b_t0, b_dir, b_speed, cfg=SimCfg(), quad=Quad(), site=Site()):
    """Batch flight from t = 0 through the same wind field the live demo flies; returns the full log."""
    init, _ = MODELS[model]
    k_init, k_noise = jax.random.split(key)
    p0 = init(k_init)

    def body(s, k):
        w = field(s.x[:3], k * cfg.dt, mean, turb, gust, b_t0, b_dir, b_speed, site)
        return flight_step(model, traj, cfg, quad, p0, k_noise, s, k, w)

    return jax.lax.scan(body, init_state(traj, p0, cfg), jnp.arange(int(cfg.duration / cfg.dt)))[1]


class Live:
    def __init__(self, traj="figure8", models=("none", "mlp", "ph"), seed=0, substeps=10, site="open"):
        self.traj, self.models, self.n, self.site = TRAJECTORIES[traj](), models, substeps, SITES[site]
        self.quad, self.k = Quad(), 0
        self.turb = atmos.make_turbulence(jax.random.PRNGKey(1000 + seed))
        self.mean, self.gust, self.learning = np.zeros(3), 0.0, "gentle"
        self.b_t0 = np.full(MAX_BURSTS, -1e9)
        self.b_dir, self.b_speed = np.tile([1.0, 0.0, 0.0], (MAX_BURSTS, 1)), np.zeros(MAX_BURSTS)
        self.p0, self.noise, self.state, self.crash_t = {}, {}, {}, {}
        for i, m in enumerate(models):
            k_init, self.noise[m] = jax.random.split(jax.random.PRNGKey(seed + i))
            self.p0[m] = MODELS[m][0](k_init)
        self.reset()

    @property
    def cfg(self):
        return SimCfg(learn=AdaptCfg(lr=LEARNING[self.learning]))

    @property
    def t(self):
        return self.k * self.cfg.dt

    def reset(self, model=None):
        for m in [model] if model else self.models:
            self.state[m] = init_state(self.traj, self.p0[m], self.cfg, self.t)
            self.crash_t[m] = None

    def set_wind(self, mean=None, gust=None, learning=None):
        if mean is not None:
            self.mean = np.array([mean[0], mean[1], 0.0], dtype=float)
        if gust is not None:
            self.gust = float(gust)
        if learning is not None:
            self.learning = learning

    def burst(self, speed=6.0, direction=(1.0, 0.0)):
        """Launch a gust front from upwind now, replacing the oldest slot."""
        i = int(np.argmin(self.b_t0))
        d = np.array([direction[0], direction[1], 0.0], dtype=float)
        self.b_t0[i], self.b_dir[i], self.b_speed[i] = self.t, d / max(np.linalg.norm(d), 1e-9), speed

    def field_spec(self) -> dict:
        """Everything the page needs to evaluate `field` itself."""
        live = self.b_t0 > self.t - (2 * BURST_FROM + 4) / BURST_V - BURST_S
        return {
            "mean": self.mean.tolist(), "gust": self.gust, "burst_s": BURST_S, "burst_v": BURST_V,
            "burst_from": BURST_FROM, "site": self.site._asdict(), "kappa": atmos.KAPPA, "sigma": atmos.SIGMA_PER_USTAR,
            "k": np.asarray(self.turb.k).tolist(), "a": np.asarray(self.turb.a).tolist(), "phase": np.asarray(self.turb.phase).tolist(),
            "bursts": [{"t0": float(self.b_t0[i]), "dir": self.b_dir[i].tolist(), "speed": float(self.b_speed[i])}
                       for i in np.flatnonzero(live)],
        }

    def tick(self) -> dict:
        """Advance one display frame (substeps control periods) and return what to draw."""
        cfg, frame = self.cfg, {"models": {}}
        for m in self.models:
            self.state[m], row = advance(m, self.traj, cfg, self.quad, self.n, self.p0[m], self.noise[m], self.state[m],
                                         self.k, self.mean, self.turb, self.gust, self.b_t0, self.b_dir, self.b_speed, self.site)
            row = jax.device_get(row)
            frame["models"][m] = {
                "pos": row["pos"].tolist(), "q": row["q"].tolist(), "motors": row["motors"].tolist(), "F_hat": row["F_hat"].tolist(), "F": row["F"].tolist(),
                "wind": row["wind"].tolist(), "crashed": bool(row["crashed"]),
            }
            if row["crashed"] and self.crash_t[m] is None:
                self.crash_t[m] = self.t
        self.k += self.n
        for m, t0 in self.crash_t.items():
            if t0 is not None and self.t - t0 > RECOVER_S:
                self.reset(m)
        frame.update(t=self.t, ref=jax.device_get(reference(self.traj, self.t)[0]).tolist(), field=self.field_spec(),
                     controls={"mean": self.mean[:2].tolist(), "gust": self.gust, "learning": self.learning},
                     f_max=self.quad.f_max)
        return frame
