# Falcon

**An autonomous flight-test engineer for inspection drones.**

Give Falcon a mission brief, such as "clear the drone for tomorrow's power-line inspection near South
Kensington", and a Claude agent does what a test engineer does before signing off a vehicle. It
checks the forecast, measures the drag in flight, predicts with code it writes where the drone runs
out of control authority, expands the flight envelope one safe step at a time, and files a GO /
NO-GO report in which every number has been checked against the flight log.

The drone under test learns the wind's force mid-flight and cancels it, with a model that can only
remove energy from the flight, never add it.

Built at the Imperial AI Hackathon 2026 for the Autonomous Agents track.

![Three drones fly the same gust beside a high-voltage line](docs/demo.jpg)

## Contents

- [The problem](#the-problem)
- [What we built](#what-we-built)
- [How the mission works](#how-the-mission-works)
- [Results](#results)
- [Physics](#physics)
- [Getting started](#getting-started)
- [Usage](#usage)
- [APIs, models and datasets](#apis-models-and-datasets)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Hackathon timeline](#hackathon-timeline)
- [Limitations](#limitations)

## The problem

Drones inspect power lines, bridges and facades, often a metre or two from the structure, in wind
they can't see. Before such a vehicle flies a job, someone has to decide whether it's safe and under
what limits: how much wind, which controller, how fast it may adapt. Today a flight-test engineer
works that out over days of measurement, prediction and careful step-by-step testing.

Handing that judgement to an AI agent is only useful if the agent's answers can be trusted. It must
not skip safety steps, and it must not state a number it didn't measure.

## What we built

- **An autonomous clearance mission.** It runs from a one-line brief to a signed-off readiness report
  with operating limits.
- **Analysis it writes and runs itself.** It fits a drag law with server-side code execution and
  commits to a predicted envelope edge before testing it.
- **Build-up enforced in code.** The tools refuse any step more than 2 m/s past the last wind
  cleared, and anything at or above a wind that crashed.
- **No invented numbers.** Every measured value in a report cites a run, like
  `2.1 cm [r8b3739.rmse_m]`. A checker reads the run log and rejects misquoted or uncited numbers,
  and the agent fixes the report and resubmits.
- **Live forecast with sources.** It looks up the site forecast through Linkup and cites the source.
- **Every flight in MLflow.** Runs are logged to MLflow on Databricks, tagged with the id the report
  cites.
- **A live 3D demo.** Three drones fly the same pass in the same air. The agent flies its
  recommended limit there, and any recorded mission can be replayed offline.

## How the mission works

| Step | What the agent does | What keeps it honest |
|---|---|---|
| 1 Identify | Holds position at 20 m in steady wind and reads the aerodynamic force and torque | The vehicle's own sensor path: accelerometer minus commanded thrust, with noise and motor lag |
| 2 Predict | Writes and runs code to fit a drag law and predict where tilt runs out | Commits to the number before testing |
| 3 Expand | Flies the inspection pass for each controller, stepping up in wind with turbulence and a gust front | Build-up rule enforced by the tools |
| 4 Reconcile | Compares prediction with test and explains the gap | Physics it can measure: force, torque, attitude stiffness |
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

## Physics

Everything the drone, the learner and the agent experience comes from one simulator. The simulator
holds a ground truth that the learner cannot fully represent, so no result rests on a model being
tested on its own equation.

### Vehicle

A 27 g Crazyflie 2.x. It's a 13-state rigid body (position, velocity, unit quaternion, body rates)
integrated with fourth-order Runge-Kutta at 500 Hz. Four motors in an X layout give at most 0.15 N
each, and each follows its command through a 30 ms first-order lag.

### What the vehicle knows

The controller and the learner see what a real vehicle would. Position comes with 5 mm of noise and
velocity with 0.03 m/s, as from external positioning. Aerodynamic force is never given directly. It
is reconstructed the way it would be on hardware:

```
F_obs = m a_imu - T_cmd R e3        (accelerometer specific force minus commanded thrust)
```

with 0.3 m/s² of accelerometer noise per axis. Because the motors lag, the thrust error lands in
`F_obs`, so the learner trains on the same corrupted signal a real vehicle has.

### Control

A geometric SE(3) tracking controller (Lee, Leok and McClamroch, 2010) with integral action and
force feedforward. Gains:

- **Position loop:** ωn = 4 rad/s, ζ = 1.
- **Integral term:** places the closed-loop poles at -5.7 and -1.2 ± 1.2j, with an anti-windup clamp
  of 0.5 m·s.
- **Attitude loop:** ωn = 25 rad/s in roll and pitch, 10 rad/s in yaw.

The commanded tilt is limited to 0.6 rad. That caps the horizontal force the vehicle can produce at
hover to

```
F_max = m g tan(0.6) = 0.181 N.
```

The baseline already has integral action, which removes any steady offset by itself. A learned
model therefore gets credit only for what integral action cannot do: gusts, turbulence and
manoeuvres.

### Aerodynamics (ground truth, hidden from the learner)

All terms depend on the air-relative velocity `v_rel = v - w` and vanish when the drone moves with
the air.

- **Rotor drag**, linear and scaling with thrust, in body axes:
  `F = -R K Rᵀ v_rel (T / T_hover)`, with K = diag(0.010, 0.010, 0.004) N·s/m.
- **Body drag**, quadratic: `F = -c |v_rel| v_rel`, with c = 0.0018 N·s²/m².
- **Induced lift** from in-plane airspeed along the body z axis (Faessler, Franchi and Scaramuzza,
  2018). It can do positive work.
- **Rotor-plane torque.** Rotor drag acts 1 cm above the centre of mass, so it pitches the vehicle
  back. A force model cannot represent this torque.

The first two terms are dissipative in the air frame (`v_rel · F ≤ 0`), which is the class the
learner assumes. The last two lie outside it on purpose.

### Atmosphere

A surface-layer wind field, the same one in batch runs, the live demo and the browser:

- **Mean profile:** `U(z) = U10 ln(z/z0) / ln(10/z0)`, with roughness z0 = 0.03 m (open grass). At
  the 20 m pass the mean wind is 12% above its 10 m value.
- **Turbulence:** friction velocity `u* = κ U10 / ln(10/z0)`, and standard deviations
  (2.4, 1.9, 1.25) u* along, across and vertical (Panofsky and Dutton, 1984). That is a turbulence
  intensity of about 15% at 20 m.
- **Spectrum:** each component follows a von Kármán spectrum with its own length scale (150, 75 and
  20 m). The field is frozen into the air and advected at the local mean speed (Taylor's
  hypothesis), so eddies keep their size and pass faster in stronger wind.
- **Gust fronts:** a sin² pulse of 1.2 s that travels downwind at 4 m/s, so neighbouring drones meet
  it in turn.

### The learner, and why it is safe for any weights

With the kinetic energy relative to the estimated air mass, `H = ½ m |v - ŵ|²`, the learned force is

```
F_hat = -D(z) (v - ŵ),     D = R sat(L Lᵀ + diag(softplus(d)) + εI) Rᵀ,
sat(A) = d_max A (d_max I + A)⁻¹
```

where L and d come from a small network and ŵ is a learned wind estimate.

- **It only removes energy.** Its contribution to the energy rate is `-(v - ŵ)ᵀ D (v - ŵ) ≤ 0`, for
  any network weights.
- **It never fights the controller.** The saturation keeps every eigenvalue of D inside
  `(0, d_max)`. With d_max = 0.1 N·s/m, below the controller's own velocity gain m·Kv = 0.216 N·s/m,
  the learned damping can never exceed it.
- **The wind estimate is bounded:** |ŵᵢ| < 12 m/s.

The unconstrained neural-net baseline has none of these properties, and a test shows it injecting
energy.

### Online learning

The model learns during every flight from scratch. Every 10 ms it takes a clipped gradient step on
the squared force error over the last 128 accelerometer samples, plus a weak pull toward its starting
weights, so it forgets stale wind instead of drifting.

### What the agent reasons about

The envelope edge comes from three effects that compete for the same 0.6 rad of tilt:

1. **Mean drag**, `F(U) ≈ aU + bU²`. The agent identifies it by holding position in steady wind and
   compares it with F_max.
2. **Attitude offset.** The rotor-plane torque τ is resisted by the attitude loop with stiffness
   `k = J ωn² ≈ 0.0104 N·m/rad`. That leaves a steady error of about τ / k, about 0.06 rad near the
   edge, which is tilt the drone cannot use. Removing that torque in the simulator moves the steady
   edge up by about 0.5 m/s, which confirms the effect.
3. **Gusts and turbulence**, which need tilt above the mean on top of that.

Drag alone therefore over-predicts the edge. The agent is expected to measure the torque, account
for it, and say what remains unexplained.

### What the tests check

- **Rigid body:** free fall matches the closed form, energy is conserved without thrust, the
  quaternion stays unit length, and hover is an equilibrium.
- **Control:** perfect feedforward cancels a disturbance, and integral action removes a steady
  offset.
- **Learned models:** the learner is passive for random weights, the neural net can inject energy,
  and the full aerodynamics leave the dissipative class while vanishing when the drone moves with
  the air.
- **Wind:** turbulence has the surface-layer standard deviations, the log profile equals 1 at 10 m,
  gust fronts travel downwind, and the browser's wind matches the simulator's exactly.

## Getting started

### Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- A Claude API key, for the agent. The live demo and replay run without one.

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

Open http://localhost:8765. Press **Clear for flight** to watch a mission (about 10 minutes on
Fable). You can type a brief first, for example one that names a site. **Replay** plays the last
recorded mission at 20× speed, including its live steps, with no API or network needed. **Run demo**
plays a scripted gust sequence.

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

## APIs, models and datasets

**Models**

- **Claude Fable 5.2** (`claude-fable-5-2`), through the Anthropic Messages API. It is used with:
  - the Python SDK's tool runner;
  - server-side code execution, for the agent's own analysis;
  - refusal fallback, and progress updates between tool calls.
- **Claude Opus 5.5** (`claude-opus-5-5`), optional through `FALCON_MODEL`.
- **The drone's learned models**, the energy-bounded learner and the neural-net baseline, are ours.
  They start from a seeded random initialisation and learn online during each flight. No pretrained
  weights are used.

**APIs**

- **Anthropic API:** the flight-engineer agent.
- **Linkup Search API:** live site wind forecasts with sources.
- **Databricks:** MLflow experiment tracking (Free Edition).

**Datasets**

None. Every flight is generated by the simulator in this repository, and forecasts are fetched live.

**Open-source libraries**

| Library | Version | Used for |
|---|---|---|
| JAX | 0.10 | Simulator and learning |
| NumPy | 2.4 | Numerics |
| anthropic | 2.8 | Python SDK |
| MLflow | 3.18 | Run tracking |
| three.js | r128 | Demo page, with OrbitControls |
| uv, pytest | | Tooling and tests |

**References**

- Lee, Leok and McClamroch, 2010: geometric SE(3) control.
- Faessler, Franchi and Scaramuzza, 2018: rotor drag and induced lift.
- Panofsky and Dutton, 1984: surface-layer turbulence.
- von Kármán spectrum and Taylor's frozen turbulence.
- Oke, 1988, and Britter and Hanna, 2003: street-canyon flow, for the city scene.
- The learner's structure is inspired by port-Hamiltonian neural networks and by Neural-Fly
  (O'Connell et al., 2022).

## Project structure

```
falcon/   simulator, controller, wind field, learned models, online learning, experiments
agent/    Claude flight engineer: mission, tools, build-up rule, report checker (verify.py)
scripts/  live demo server with mission recording and replay, envelope sweep, plots
web/      demo pages; field.js mirrors the simulator's wind exactly
tests/    57 tests
docs/     Devpost write-up, presentation script, recorded sample mission, screenshot
```

## Testing

```bash
uv run --extra agent pytest
```

## Hackathon timeline

To be clear about what was made when: we prototyped the simulator core on 12 and 13 November,
before the event. That covers the dynamics, controller, aerodynamics, wind field, learned models
and a first live demo. This repository was created at the event, and its commit history doesn't
show those earlier dates.

Built during the hackathon (14 and 15 November):

- the autonomous mission agent, its build-up rule and the report checker;
- the Linkup forecast and the Databricks tracking;
- measured force and torque, and a single wind field for batch and live flights;
- the mission panel, mission recording and replay.

## Limitations

- **Simulation only.** Nothing has been flown on hardware, and motor lag and sensor noise are
  assumed values, not measured ones.
- **The energy bound covers the learned model, not the whole closed loop.** How D varies with
  velocity is unbounded, so the Jacobian of the learned force can still exceed d_max.
- **A single wind estimate.** The learner treats the air around the drone as uniform. Strong shear
  across the vehicle is outside that assumption.
- **Effects not modelled:** ground effect, rotor-to-rotor interaction, blade flapping beyond linear
  rotor drag, battery sag, and rain.
