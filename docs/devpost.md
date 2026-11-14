# Falcon: an autonomous flight-test engineer

Track: Autonomous Agents

## What it is

Before a drone inspects a live power line in the wind, someone has to decide whether it's safe and
under what limits. Today that takes a flight-test engineer days. Falcon gives the job to a Claude
agent and holds it to the same standard: measure first, predict, test in safe steps, and sign off
only on numbers that came from a flight.

## What it does

You give it a mission brief, for example "clear the drone for tomorrow's line inspection, 3 to 5 m/s
toward the line, gusty". Then the agent, on its own:

1. **Identifies** the vehicle. It hovers in steady wind and reads the aerodynamic force from the
   accelerometer.
2. **Predicts** the envelope edge. It writes and runs Python to fit a drag law and work out the
   wind at which the tilt limit runs out, and commits to that number before testing it.
3. **Expands** the envelope. It flies the inspection pass with three controllers (no model, an
   unconstrained neural net, and our energy-bounded learner), stepping up in wind.
4. **Reconciles** prediction with test. Drag alone predicts more margin than the drone has. The gap
   is rotor drag acting above the centre of mass, which pitches the vehicle back. We confirmed it:
   remove that torque from the simulator and the steady edge moves up by about 0.5 m/s.
5. **Reports** GO, GO WITH LIMITS or NO-GO with operating limits. Every number cites the flight it
   came from.
6. **Shows** the limit live in a 3D simulation of the power-line corridor.

## Why you can trust it

An agent is only useful if you can trust what it hands you. Falcon enforces that in code, not in
the prompt:

- **Build-up rule.** The tools refuse any step more than 2 m/s past the last wind a configuration
  cleared, and anything at or above a wind that crashed. The agent has to plan around this, the
  same way a test pilot does.
- **Report checker.** Every measured number must cite a run, like `4.2 cm [r8b3739.rmse_m]`. The
  checker reads the run log and rejects misquoted, uncited or invented numbers. The agent fixes the
  report and resubmits. Predictions have to be labelled as predictions.
- **The drone's learner is bounded too.** Its learned force is `F = -D (v - w_hat)` with
  `0 < D < d_max`, so for any weights it can only remove energy, and it never fights the
  controller's own damping. Learning ten times too fast makes the unconstrained neural net nearly
  7× worse than no model; the bounded learner stays within 1.7×.

## How we built it

- **Simulator:** a Crazyflie 2.x rigid body in JAX at 500 Hz, with geometric SE(3) control and
  integral action, motor lag, sensor noise and accelerometer force estimates. Wind is surface-layer
  with von Karman turbulence. The air also produces induced lift and rotor torque, which the learned
  model can't represent, so it never trains on its own equation.
- **Agent:** Claude Fable 5.2 through the Anthropic tool runner, with typed Python tools, server-side
  code execution for the physics fit, and progress updates streamed to the page.
- **Linkup:** the agent looks up the live wind forecast for the mission site and cites the source.
- **Databricks:** every flight is logged to MLflow on Databricks, tagged with the id the report
  cites, so each number in the report opens in the workspace.
- **Live demo:** a three.js page streams the simulation at 50 Hz. Its wind field is mirrored line for
  line from the simulator and checked by a test.
- **Tests:** 56, including the energy bound, build-up enforcement, and the checker rejecting
  misquoted, uncited and invented numbers.

## Results

Inspection pass at 20 m in the live demo's wind: mean wind given at 10 m, standard surface-layer
turbulence and one 2 m/s gust front. Tracking error (cm), mean ± 95% CI, 10 seeds:

| | No model | Neural net | Learner |
|---|---:|---:|---:|
| 3 m/s toward the wires | 3.2 ± 0.2 | 2.8 ± 0.3 | **1.0 ± 0.1** |
| 4 m/s toward the wires | 4.5 ± 1.1 | 3.8 ± 0.9 | **2.1 ± 1.1** |
| 4 m/s, learning 10× faster | **4.5 ± 1.1** | 30.5 ± 8.2 | 7.6 ± 2.7 |
| 5 m/s toward the wires | 1 in 10 crash | 1 in 10 crash | 1 in 10 crash |

Batch runs and the live demo fly the same wind field. An early mission caught them disagreeing: its
live check crashed at a limit the batch runs had cleared. We unified them, and the batch edge now
matches what the demo meets.

## Limits

Everything here is simulated, not flown on hardware. The energy bound covers the learned model, not
the whole closed loop. Motor lag and sensor noise are assumed values, not measured ones.
