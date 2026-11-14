# Falcon

**An autonomous flight-test engineer for inspection drones.**

Give Falcon a mission brief, such as "clear the drone for tomorrow's power-line inspection, 3 to 5 m/s
toward the line, gusty", and a Claude agent does what a test engineer does before signing off a
vehicle. It measures the drag in flight, predicts with code it writes where the drone runs out of
control authority, expands the flight envelope one safe step at a time, and files a GO / NO-GO report
in which every number has been checked against the flight log.

The drone under test learns the wind's force mid-flight and cancels it, with a model that can only
remove energy from the flight, never add it.

Built at the Imperial AI Hackathon 2026 for the Autonomous Agents track.

![Three drones fly the same gust beside a high-voltage line](docs/demo.jpg)

## Contents

- [Features](#features)
- [How the mission works](#how-the-mission-works)
- [Results](#results)
- [Getting started](#getting-started)
- [Usage](#usage)
- [How the drone learns](#how-the-drone-learns)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Built with](#built-with)
- [Limitations](#limitations)

## Features

- **Autonomous clearance mission.** It runs from a one-line brief to a signed-off report with
  operating limits.
- **Writes and runs its own analysis.** It fits a drag law with server-side code execution and
  predicts the envelope edge before testing it.
- **Build-up enforced in code.** The tools refuse any step more than 2 m/s past the last wind cleared,
  and anything at or above a wind that crashed.
- **No invented numbers.** Every measured value in a report must cite a run, like
  `2.1 cm [r8b3739.rmse_m]`. A checker reads the run log and rejects misquoted or uncited numbers,
  and the agent fixes the report and resubmits.
- **Live forecast.** It looks up the wind forecast for a named site with Linkup and cites the source.
- **Every flight in MLflow.** Runs are logged to MLflow on Databricks, tagged with the id the report
  cites.
- **Live 3D demo.** Three drones fly the same pass in the same air while you, or the agent, change the
  wind.

## How the mission works

| Step | What the agent does | What keeps it honest |
|---|---|---|
| 1 Identify | Holds position at 20 m in steady wind and reads the aerodynamic force and torque | Real sensor path: accelerometer minus commanded thrust, with noise and motor lag |
| 2 Predict | Writes and runs code to fit a drag law and predict where tilt runs out | Commits to the number before testing |
| 3 Expand | Flies the inspection pass for each controller, stepping up in wind with turbulence and a gust front | Build-up rule enforced by the tools |
| 4 Reconcile | Compares prediction with test and explains the gap | Drag alone predicts more margin than the drone has. Rotor drag acts above the centre of mass and pitches the drone back. Remove that torque from the simulator and the edge moves up by about 0.5 m/s |
| 5 Report | Files the readiness report with operating limits | Checked number by number against the run log |
| 6 Show | Flies the recommended limit live with a gust front | Same wind field, physics and controller as the batch runs |

## Results

Inspection pass at 20 m in the demo's wind: mean wind given at 10 m, standard surface-layer
turbulence, and one 2 m/s gust front. Tracking error in cm, mean ± 95% CI over 10 seeds.

| | No model | Neural net | Learner |
|---|---:|---:|---:|
| 3 m/s toward the wires | 3.2 ± 0.2 | 2.8 ± 0.3 | **1.0 ± 0.1** |
| 4 m/s toward the wires | 4.5 ± 1.1 | 3.8 ± 0.9 | **2.1 ± 1.1** |
| 4 m/s, learning 10× faster | **4.5 ± 1.1** | 30.5 ± 8.2 | 7.6 ± 2.7 |
| 5 m/s toward the wires | 1 in 10 crash | 1 in 10 crash | 1 in 10 crash |

- The learner tracks 2 to 3× tighter than the controller alone.
- Learning ten times too fast, the unconstrained neural net is nearly 7× worse than no model, while
  the learner stays within 1.7×.
- The envelope edge is between 4 and 5 m/s. On a typical mission the agent clears 3.5 m/s, and the
  live check at that limit holds.

## Getting started

### Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- A Claude API key, for the agent. The live demo runs without one.

### Install

```bash
git clone https://github.com/Adeniyikayodee/imperial_falcon_hack.git
cd imperial_falcon_hack
uv sync --extra agent
cp .env.example .env    # then add your keys
```

### Configure

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | The flight engineer (or run `ant auth login`) |
| `FALCON_MODEL` | Optional. Defaults to `claude-fable-5-2` |
| `LINKUP_API_KEY` | Optional. Site wind forecasts |
| `MLFLOW_TRACKING_URI=databricks`, `DATABRICKS_HOST`, `DATABRICKS_TOKEN` | Optional. Logging to MLflow on Databricks (`uv sync --extra track`) |

## Usage

**Live demo:**

```bash
uv run --extra agent python scripts/live.py
```

Open http://localhost:8765 and press **Clear for flight** to watch a mission. You can type a brief
first, for example one that names a site. **Run demo** plays a scripted gust sequence without Claude.
Use `--scene city` for a street canyon, or open `/stage` for the presenter view.

**Mission from the terminal:**

```bash
uv run --extra agent python -m agent.run --mission "Clear the drone for tomorrow's pass near South Kensington, London."
```

**One question:**

```bash
uv run --extra agent python -m agent.run "Is fast learning safe in a 4 m/s wind toward the line?"
```

**Sweep the envelope yourself:**

```bash
uv run python scripts/sweep.py --traj inspect --axis y --winds 3 4 5 --gust 1 --front 2
```

## How the drone learns

```
F_hat = -D (v - w_hat),   0 < eig(D) < d_max
```

The learned force always opposes motion through the estimated air, so it removes energy for any
weights. `d_max` sits below the controller's own damping, so the learned model never fights the
controller. Both properties are tested.

The simulator models a Crazyflie 2.x rigid body in JAX at 500 Hz, with geometric SE(3) control and
integral action, motor lag, sensor noise, and force estimates from the accelerometer. The wind is a
surface-layer profile with von Kármán turbulence and gust fronts. The air also produces induced lift
and rotor torque, which the learned model can't represent, so it never trains on its own equation.

## Project structure

```
falcon/   simulator, controller, wind field, learned models, online learning, experiments
agent/    Claude flight engineer: mission, tools, build-up rule, report checker (verify.py)
scripts/  live demo server, envelope sweep, plots
web/      demo pages; field.js mirrors the simulator's wind exactly
tests/    57 tests
docs/     Devpost write-up, video script, screenshot
```

## Testing

```bash
uv run --extra agent pytest
```

The tests cover the energy bound, controller stability, the wind field mirrored in the browser,
build-up enforcement, and the checker rejecting misquoted, uncited and invented numbers.

## Built with

- [Claude Fable 5.2](https://www.anthropic.com) through the Anthropic tool runner, with code execution
- [Linkup](https://www.linkup.so) for live web search
- [MLflow on Databricks](https://www.databricks.com) for run tracking
- [JAX](https://github.com/jax-ml/jax) for the simulator
- [three.js](https://threejs.org) for the live demo

## Limitations

- Everything is simulated; nothing has been flown on hardware.
- The energy bound covers the learned model, not the whole closed loop.
- Motor lag and sensor noise are assumed values, not measured ones.
