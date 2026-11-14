"""Closed-loop flight: controller, hidden aerodynamics, IMU force samples, online adaptation.

The vehicle sees what a real one would: position and velocity with estimator noise, motors that lag
their commands (first order), and an aerodynamic force sample built from the accelerometer minus the
commanded thrust, so motor lag shows up as error in what the model learns from.

`flight_step` is the single control period used by both the batch `rollout` and the live demo,
so the two cannot drift apart. Deterministic for a given key. A run counts as crashed once
tracking error exceeds `crash_dist`, after which the state is frozen so logs stay finite.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from falcon import adapt
from falcon.aero import Wind, aero, wind_at
from falcon.control import Gains, control
from falcon.dynamics import hover_state, quat_to_rot, step, wrench
from falcon.params import Quad
from falcon.residual import MODELS
from falcon.traj import reference


class SimCfg(NamedTuple):
    dt: float = 0.002
    duration: float = 20.0
    imu_noise: float = 0.3  # m/s^2, per axis
    pos_noise: float = 0.005  # m, estimator noise per axis (assumed, external positioning)
    vel_noise: float = 0.03  # m/s
    motor_tau: float = 0.03  # s, first-order motor time constant (assumed, brushed micro motors)
    crash_dist: float = 2.0
    learn: adapt.AdaptCfg = adapt.AdaptCfg()


class State(NamedTuple):
    x: jnp.ndarray
    params: dict
    buf: adapt.Buffer
    tr: jnp.ndarray  # commanded thrust ratio, what the vehicle knows
    crashed: jnp.ndarray
    f: jnp.ndarray  # actual motor thrusts (4,)
    e_int: jnp.ndarray  # clamped integral of position error (3,)


def init_state(traj, p0, cfg: SimCfg, t=0.0) -> State:
    """Start on the reference at time t, moving with it, with fresh learned parameters."""
    p, v, _, _ = reference(traj, t)
    x0 = hover_state(p).at[3:6].set(v)
    return State(x0, p0, adapt.empty(cfg.learn.window), jnp.array(1.0), jnp.array(False),
                 jnp.full(4, Quad().hover_thrust / 4), jnp.zeros(3))


def flight_step(model: str, traj, cfg: SimCfg, quad: Quad, p0, k_noise, s: State, k, w):
    """One control period at step k with world wind w. Returns the next state and a log row."""
    _, apply = MODELS[model]
    t = k * cfg.dt
    ref = reference(traj, t)
    n = jax.random.normal(jax.random.fold_in(k_noise, k), (9,))
    est = s.x.at[:6].add(jnp.concatenate([cfg.pos_noise * n[:3], cfg.vel_noise * n[3:6]]))
    v, q, g = est[3:6], s.x[6:10], Gains()

    F_hat = apply(s.params, v, q, s.tr)
    e_int = jnp.clip(s.e_int + (ref[0] - est[:3]) * cfg.dt, -g.i_max, g.i_max)
    f_cmd = jnp.clip(control(est, ref, F_hat, quad, g, e_int)[0], 0.0, quad.f_max)
    f = s.f + (1.0 - jnp.exp(-cfg.dt / cfg.motor_tau)) * (f_cmd - s.f)
    T, T_cmd = wrench(f, quad)[0], wrench(f_cmd, quad)[0]
    F, tau = aero(s.x[3:6], q, T / quad.hover_thrust, w, quad.mass)
    x_next = step(s.x, f, F, quad, cfg.dt, tau)

    # accelerometer: (R T e3 + F) / m; subtracting the commanded thrust leaves F, the thrust error and noise
    F_obs = F + (T - T_cmd) * quat_to_rot(q)[:, 2] + quad.mass * cfg.imu_noise * n[6:]
    tr = T_cmd / quad.hover_thrust
    buf = adapt.push(s.buf, v, q, tr, F_obs)
    learn = (k % cfg.learn.every == 0) & (k > 0) & ~s.crashed
    params = jax.lax.cond(learn, lambda p: adapt.update(p, apply, buf, p0, cfg.learn), lambda p: p, s.params)

    crashed = s.crashed | (jnp.linalg.norm(x_next[:3] - ref[0]) > cfg.crash_dist) | ~jnp.all(jnp.isfinite(x_next))
    x_next = jnp.where(crashed, s.x, x_next)
    log = dict(t=t, pos=s.x[:3], q=q, motors=f, ref=ref[0], F=F, F_obs=F_obs, F_hat=F_hat, tau=tau, wind=w, crashed=crashed)
    return State(x_next, params, buf, tr, crashed, f, e_int), log


@partial(jax.jit, static_argnames=("model", "traj", "cfg", "quad"))
def rollout(model: str, traj, wind: Wind, key, cfg: SimCfg = SimCfg(), quad: Quad = Quad()):
    init, _ = MODELS[model]
    k_init, k_noise = jax.random.split(key)
    p0 = init(k_init)

    def body(s, k):
        return flight_step(model, traj, cfg, quad, p0, k_noise, s, k, wind_at(k * cfg.dt, wind))

    s, log = jax.lax.scan(body, init_state(traj, p0, cfg), jnp.arange(int(cfg.duration / cfg.dt)))
    return log, s.params


def metrics(log, settle=5.0):
    """Tracking and force-estimation error after `settle` seconds, and the mean horizontal aerodynamic
    force the accelerometer measured (noise averages out; what a flight test would record), and the mean
    roll/pitch aerodynamic torque (what gyro rates and motor commands reveal on a real vehicle)."""
    m = log["t"] >= settle
    err = jnp.linalg.norm(log["pos"] - log["ref"], axis=1)[m]
    ferr = jnp.linalg.norm(log["F_hat"] - log["F"], axis=1)[m]
    return {
        "aero_force_xy_n": float(jnp.linalg.norm(jnp.mean(log["F_obs"][m, :2], axis=0))),
        "aero_torque_nm": float(jnp.linalg.norm(jnp.mean(log["tau"][m, :2], axis=0))),
        "rmse_m": float(jnp.sqrt(jnp.mean(err**2))),
        "max_err_m": float(err.max()),
        "force_rmse_n": float(jnp.sqrt(jnp.mean(ferr**2))),
        "crashed": bool(log["crashed"][-1]),
    }
