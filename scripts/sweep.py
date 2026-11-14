"""Compare residual models across wind speeds, in the live demo's wind field. Example:
    uv run python scripts/sweep.py --traj inspect --axis y --winds 3 4 5 --gust 1 --front 2
"""
import argparse

import falcon  # noqa: F401
from falcon.experiment import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="figure8", choices=["station", "hover", "figure8", "inspect"])
    ap.add_argument("--winds", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    ap.add_argument("--gust", type=float, default=1.0, help="turbulence intensity, 1 = standard surface layer")
    ap.add_argument("--front", type=float, default=0.0, help="one gust front of this speed, m/s, as the demo's Gust button")
    ap.add_argument("--lr", type=float, nargs="+", default=[0.05])
    ap.add_argument("--models", nargs="+", default=["none", "mlp", "ph"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--axis", choices=["x", "y"], default="x", help="mean wind direction; y blows toward the line on inspect")
    a = ap.parse_args()

    print(f"{'wind':>5} {'lr':>5} " + " ".join(f"{m + ' rmse cm (95% CI), crash':>26}" for m in a.models))
    for lr in a.lr:
        for w in a.winds:
            cells = []
            for m in a.models:
                wind = (w, 0.0, 0.0) if a.axis == "x" else (0.0, w, 0.0)
                r = run_experiment(m, a.traj, wind, a.gust, lr, a.seeds, front=a.front)["metrics"]
                cells.append(f"{100 * r['rmse_m']:6.1f} +/- {100 * r['rmse_ci95_m']:4.1f} {100 * r['crash_rate']:4.0f}%")
            print(f"{w:5.1f} {lr:5.2f} " + " ".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
