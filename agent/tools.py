"""Flight-engineer tools: the single source of each tool's logic, limits and the docstring Claude sees.

Offline tools run seeded batch experiments. `Campaign` adds the flight-test rules a mission flies under:
build-up in wind, no return above a wind that crashed, and a report whose numbers are checked against
the run log (agent.verify). `Live` acts on the running demo through any stage with `control(dict)` and
`observe(seconds) -> dict` (scripts/live.py), within the same limits as the page.
"""
import json
import os
import urllib.request
import math
import time
from pathlib import Path

import falcon  # noqa: F401
from falcon.control import Gains
from falcon.experiment import run_experiment
from falcon.params import Quad
from falcon.residual import D_MAX, W_MAX
from falcon import track
from falcon.track import load_runs

from agent.verify import verify

LIMITS = {"buildup_mps": 2.0, "wind": 10.0, "gust": 4.0, "lr": (1e-3, 2.0), "seeds": (1, 8), "live_wind": 7.0, "watch_s": 20.0}


def fly(model: str, traj: str, wind_x: float, wind_y: float, gust: float, lr: float, seeds: int, front: float = 0.0) -> str:
    """Fly seeded offline missions and return the run id and averaged metrics: rmse_m with rmse_ci95_m,
    max_err_m, crash_rate, force_rmse_n (learned-force error) and aero_force_xy_n (mean horizontal
    aerodynamic force the accelerometer measured) and aero_torque_nm (mean roll/pitch aerodynamic torque).

    Args:
        model: Residual model: none, mlp or ph.
        traj: station (hold position at 20 m beside the line), inspect (the power-line pass at 20 m, line along x;
            +y blows toward it), or hover / figure8 (at 1 m, where the wind is about 40% weaker).
        wind_x: Mean wind along x at the 10 m reference height, m/s, as on the live demo; the surface-layer
            profile makes it about 12% stronger at 20 m.
        wind_y: Mean wind along y at 10 m, m/s.
        gust: Turbulence intensity, 0 to 4: 0 steady air, 1 the standard surface layer (what the live demo flies).
        lr: Online adaptation learning rate, 0.001 to 2. Default 0.05.
        seeds: Number of seeds to average, 1 to 8.
        front: Speed of one gust front sent along the wind about 10 s in, m/s, 0 to 4; the live Gust button sends 2.
    """
    if model not in ("none", "mlp", "ph") or traj not in ("station", "inspect", "hover", "figure8"):
        raise ValueError("model must be none|mlp|ph and traj station|inspect|hover|figure8")
    if math.hypot(wind_x, wind_y) > LIMITS["wind"] or not 0 <= gust <= LIMITS["gust"] or not 0 <= front <= LIMITS["gust"]:
        raise ValueError(f"mean wind must be <= {LIMITS['wind']} m/s, gust and front in [0, {LIMITS['gust']}]")
    if not LIMITS["lr"][0] <= lr <= LIMITS["lr"][1] or not LIMITS["seeds"][0] <= seeds <= LIMITS["seeds"][1]:
        raise ValueError(f"lr must be in {LIMITS['lr']} and seeds in {LIMITS['seeds']}")
    return json.dumps(run_experiment(model, traj, (wind_x, wind_y, 0.0), gust, lr, seeds, front=front))


def recent_runs(limit: int) -> str:
    """Return the most recent logged experiment results.

    Args:
        limit: How many runs to return, up to 50.
    """
    return json.dumps([{"name": r["name"], "metrics": r["metrics"]} for r in load_runs(max(1, min(limit, 50)))])


def vehicle_facts() -> str:
    """Return vehicle limits, controller gains and the ph model's safety bounds."""
    q, g = Quad(), Gains()
    return json.dumps({
        "mass_kg": q.mass,
        "max_total_thrust_n": 4 * q.f_max,
        "max_tilt_rad": g.max_tilt,
        "max_horizontal_force_n": round(q.hover_thrust * math.tan(g.max_tilt), 4),
        "attitude_stiffness_nm_per_rad": round(q.inertia[0] * g.att_wn[0] ** 2, 5),
        "controller_velocity_gain_n_per_mps": q.mass * g.vel,
        "controller_integral_gain_per_s3": g.pos_i,
        "ph_damping_bound_n_per_mps": D_MAX,
        "ph_wind_estimate_bound_mps": W_MAX,
        "guarantee": "ph residual force is dissipative relative to its wind estimate, with damping "
                     "eigenvalues in (0, d_max) and d_max below the controller velocity gain",
    })


def site_forecast(query: str) -> str:
    """Search the live web (Linkup) for the wind forecast at the mission site and return a sourced
    answer: {"answer": ..., "sources": [{"name", "url"}]}. Convert km/h to m/s (divide by 3.6) and cite
    the source URL in the report.

    Args:
        query: What to look up, e.g. "wind speed and gusts forecast tomorrow at <site>".
    """
    key = os.getenv("LINKUP_API_KEY")
    if not key:
        raise ValueError("no LINKUP_API_KEY; use the forecast in the brief")
    req = urllib.request.Request("https://api.linkup.so/v1/search", method="POST",
                                 data=json.dumps({"q": query, "depth": "standard", "outputType": "sourcedAnswer"}).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    return json.dumps({"answer": out.get("answer"), "sources": [{"name": x.get("name"), "url": x.get("url")}
                                                                 for x in out.get("sources", [])[:5]]})


class Campaign:
    """One mission's flight-test state. Envelope expansion as on a real vehicle: per configuration
    (model, trajectory, gust, front, learning rate) each step in wind may go at most `buildup_mps` past the
    highest wind cleared without a crash, and nothing is flown at or above a wind that crashed."""

    def __init__(self, reports=None):
        self.cleared, self.edge, self.flown, self.submissions, self.report = {}, {}, [], [], None
        self.reports = Path(reports) if reports else track.RUNS / "reports"

    def fly(self, model: str, traj: str, wind_x: float, wind_y: float, gust: float, lr: float, seeds: int,
            front: float = 0.0) -> str:
        """Fly seeded offline missions under the build-up rule and return the run id and averaged
        metrics: rmse_m with rmse_ci95_m, max_err_m, crash_rate, force_rmse_n (learned-force error) and
        aero_force_xy_n (mean horizontal aerodynamic force the accelerometer measured) and aero_torque_nm
        (mean roll/pitch aerodynamic torque, as gyro rates and motor commands reveal it). Each step in
        wind may go at most 2 m/s past the highest wind this configuration (model, traj, gust, front, lr) has
        cleared without a crash, starting from calm, and nothing is flown at or above a wind that crashed.

        Args:
            model: Residual model: none, mlp or ph.
            traj: station (hold position at 20 m beside the line), inspect (the power-line pass at 20 m, line along x;
                +y blows toward it), or hover / figure8 (at 1 m, where the wind is about 40% weaker).
            wind_x: Mean wind along x at the 10 m reference height, m/s, as on the live demo; the surface-layer
                profile makes it about 12% stronger at 20 m.
            wind_y: Mean wind along y at 10 m, m/s.
            gust: Turbulence intensity, 0 to 4: 0 steady air, 1 the standard surface layer (what the live demo flies).
            lr: Online adaptation learning rate, 0.001 to 2. Default 0.05.
            seeds: Number of seeds to average, 1 to 8.
            front: Speed of one gust front sent along the wind about 10 s in, m/s, 0 to 4; the live Gust button sends 2.
        """
        key, w = (model, traj, round(gust, 3), round(front, 3), round(lr, 4)), math.hypot(wind_x, wind_y)
        ok = self.cleared.get(key, 0.0)
        if w > ok + LIMITS["buildup_mps"] + 1e-9:
            raise ValueError(f"build-up: {key} has cleared {ok:g} m/s, so the next step may go to {ok + LIMITS['buildup_mps']:g} m/s")
        if key in self.edge and w >= self.edge[key] - 1e-9:
            raise ValueError(f"{key} crashed at {self.edge[key]:g} m/s; fly below it")
        out = json.loads(fly(model, traj, wind_x, wind_y, gust, lr, seeds, front))
        if out["metrics"]["crash_rate"] > 0:
            self.edge[key] = min(w, self.edge.get(key, math.inf))
        else:
            self.cleared[key] = max(ok, w)
        self.flown.append(out)
        return json.dumps(out)

    def submit_report(self, report: str) -> str:
        """Submit the flight readiness report (markdown). It is checked against the run log before it
        is accepted. Cite every measured quantity right after it: `2.3 cm [r1a2b3.rmse_m]` for a run
        metric, `0.6 rad [facts.max_tilt_rad]` for a vehicle fact, `7.6 m/s [pred]` for your own
        prediction. Numbers in cm, mm, N, mN, %, rad or deg without a citation, or that do not match
        the cited value at the precision written, are rejected with the reason; fix and resubmit.

        Args:
            report: The full report in markdown.
        """
        res = verify(report, json.loads(vehicle_facts()))
        self.submissions.append({"accepted": res["accepted"], "problems": res["problems"]})
        if res["accepted"]:
            self.reports.mkdir(parents=True, exist_ok=True)
            path = self.reports / f"report-{time.strftime('%Y%m%d-%H%M%S')}.md"
            path.write_text(report + "\n\n" + res["footer"])
            self.report = {"markdown": report, "claims": res["claims"], "path": str(path)}
            return json.dumps({"accepted": True, "verified_claims": len(res["claims"]), "saved": str(path)})
        return json.dumps({"accepted": False, "problems": res["problems"]})

    def all(self):
        return [self.fly, self.submit_report]


class Live:
    """Tools bound to the running demo. All three drones fly in the same air."""

    def __init__(self, stage):
        self.stage = stage

    def live_set(self, wind_mps: float, heading_deg: float, learning: str) -> str:
        """Change the live demo's conditions.

        Args:
            wind_mps: Mean wind at 10 m height, 0 to 7 m/s. Above about 5 m/s turbulence reaches the tilt limit.
            heading_deg: Direction the wind blows toward, degrees; 90 blows toward the power line.
            learning: gentle (rate 0.05) or aggressive (rate 0.5, ten times faster).
        """
        if not 0 <= wind_mps <= LIMITS["live_wind"] or learning not in ("gentle", "aggressive"):
            raise ValueError(f"wind must be in [0, {LIMITS['live_wind']}] m/s and learning gentle|aggressive")
        h = math.radians(heading_deg)
        self.stage.control({"mean": [wind_mps * math.cos(h), wind_mps * math.sin(h)],
                            "gust": 1.0 if wind_mps > 0 else 0.0, "learning": learning})
        return json.dumps({"ok": True})

    def live_gust(self) -> str:
        """Send a 2 m/s gust front through the live demo along the current wind."""
        mx, my = self.stage.observe(0.0)["mean_wind_xy"]
        sp = math.hypot(mx, my)
        self.stage.control({"burst": {"speed": 2.0, "dir": [mx / sp, my / sp] if sp > 0.2 else [0.0, 1.0]}})
        return json.dumps({"ok": True})

    def live_watch(self, seconds: float) -> str:
        """Let the live demo fly, then return each model's tracking error (rms and max, m) over that window.

        Args:
            seconds: How long to watch, 1 to 20.
        """
        s = min(max(seconds, 1.0), LIMITS["watch_s"])
        time.sleep(s)
        return json.dumps(self.stage.observe(s))

    def live_reset(self) -> str:
        """Put all three live drones back on their planned pass."""
        self.stage.control({"reset": True})
        return json.dumps({"ok": True})

    def all(self):
        return [self.live_set, self.live_gust, self.live_watch, self.live_reset]
