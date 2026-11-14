"""One named experiment: a flight per seed for a model, trajectory and wind, averaged and logged.
Shared by the sweep script and the agent tools so both report identical numbers.

Every flight goes through the live demo's wind field (falcon.live.field): `wind` is the mean wind at the
10 m reference height, scaled with height by the surface-layer profile; `gust` is the turbulence
intensity (1 = standard surface layer, as in the demo); `front` sends one gust front of that speed along
the wind, reaching the route about FRONT_T seconds in, like the demo's Gust button."""
import math

import jax
import numpy as np

from falcon import atmos
from falcon.adapt import AdaptCfg
from falcon.live import MAX_BURSTS, SITES, rollout_field
from falcon.params import Quad
from falcon.sim import SimCfg, metrics
from falcon.track import log_run
from falcon.traj import TRAJECTORIES


FRONT_T = 10.0  # s, when a gust front is launched (it reaches the route about half a second later)


def t95(df):
    """Two-sided 95% Student t quantile (Cornish-Fisher expansion, within 1% for df >= 3)."""
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2) \
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)


def run_experiment(model="ph", traj="figure8", wind=(4.0, 0.0, 0.0), gust=0.0, lr=0.05,
                   seeds=10, duration=20.0, log=True, front=0.0, site="open") -> dict:
    cfg = SimCfg(duration=duration, learn=AdaptCfg(lr=lr))
    path = TRAJECTORIES[traj]()
    sp = math.hypot(wind[0], wind[1])
    d = [wind[0] / sp, wind[1] / sp, 0.0] if sp > 0.2 else [0.0, 1.0, 0.0]  # the demo's default direction
    b_t0, b_speed = np.full(MAX_BURSTS, -1e9), np.zeros(MAX_BURSTS)
    if front > 0:
        b_t0[0], b_speed[0] = FRONT_T, front
    mean = np.array([wind[0], wind[1], 0.0])
    runs = []
    for s in range(seeds):
        turb = atmos.make_turbulence(jax.random.PRNGKey(1000 + s))
        log_ = rollout_field(model, path, jax.random.PRNGKey(s), mean, turb, gust, b_t0, np.tile(d, (MAX_BURSTS, 1)),
                             b_speed, cfg, Quad(), SITES[site])
        runs.append(metrics(log_))
    out = {k: float(np.mean([r[k] for r in runs])) for k in ("rmse_m", "max_err_m", "force_rmse_n", "aero_force_xy_n", "aero_torque_nm")}
    rmse = np.array([r["rmse_m"] for r in runs])
    out["rmse_ci95_m"] = float(t95(seeds - 1) * rmse.std(ddof=1) / np.sqrt(seeds)) if seeds > 1 else 0.0
    out["crash_rate"] = sum(r["crashed"] for r in runs) / seeds
    params = dict(model=model, traj=traj, wind=list(wind), gust=gust, front=front, lr=lr, seeds=seeds,
                  duration=duration, site=site)
    name = f"{model}-{traj}-w{wind[0]:g},{wind[1]:g}-g{gust:g}-f{front:g}-lr{lr:g}"
    rid = log_run(name, params, out) if log else None
    return {"id": rid, "params": params, "metrics": out}
