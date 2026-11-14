"""Demo figure: top-down figure-8 tracks per model, and the ph force estimate against the true aero force.
    uv run --extra viz python scripts/plot.py --wind 4 --gust 1.5 --out runs/demo.png
"""
import argparse
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import falcon  # noqa: E402,F401
from falcon.adapt import AdaptCfg  # noqa: E402
from falcon.aero import make_wind  # noqa: E402
from falcon.sim import SimCfg, metrics, rollout  # noqa: E402
from falcon.traj import figure8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", type=float, default=4.0)
    ap.add_argument("--gust", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="runs/demo.png")
    a = ap.parse_args()

    path, cfg = figure8(), SimCfg(learn=AdaptCfg(lr=a.lr))
    wind = make_wind(jax.random.PRNGKey(1000), mean=(a.wind, 0.0, 0.0), gust=a.gust)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5))
    for model, color in [("none", "#9aa0a6"), ("mlp", "#e8710a"), ("ph", "#1a73e8")]:
        log, _ = rollout(model, path, wind, jax.random.PRNGKey(0), cfg)
        m = metrics(log)
        ax.plot(log["pos"][:, 0], log["pos"][:, 1], color=color, lw=1.2,
                label=f"{model}: {100 * m['rmse_m']:.1f} cm rmse" + (" (crashed)" if m["crashed"] else ""))
        if model == "ph":
            bx.plot(log["t"], 1000 * log["F"][:, 0], color="#9aa0a6", lw=1, label="true aero force x")
            bx.plot(log["t"], 1000 * log["F_hat"][:, 0], color=color, lw=1.5, label="ph estimate x")
    ax.plot(log["ref"][:, 0], log["ref"][:, 1], "k--", lw=0.8, label="reference")
    ax.set(title=f"Figure-8, {a.wind:g} m/s wind, {a.gust:g} m/s gusts", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    bx.set(title="Online aero force estimate", xlabel="time (s)", ylabel="mN")
    ax.legend(loc="upper right", fontsize=8)
    bx.legend(fontsize=8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(a.out)


if __name__ == "__main__":
    main()
