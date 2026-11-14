# Falcon physics

Everything the drone, the learner and the agent experience comes from one simulator. The simulator
holds a ground truth that the learner cannot fully represent, so no result rests on a model being
tested on its own equation.

## Vehicle

A 27 g Crazyflie 2.x. It's a 13-state rigid body (position, velocity, unit quaternion, body rates)
integrated with fourth-order Runge-Kutta at 500 Hz. Four motors in an X layout give at most 0.15 N
each, and each follows its command through a 30 ms first-order lag.

## What the vehicle knows

The controller and the learner see what a real vehicle would. Position comes with 5 mm of noise and
velocity with 0.03 m/s, as from external positioning. Aerodynamic force is never given directly. It
is reconstructed the way it would be on hardware:

```
F_obs = m a_imu - T_cmd R e3        (accelerometer specific force minus commanded thrust)
```

with 0.3 m/s² of accelerometer noise per axis. Because the motors lag, the thrust error lands in
`F_obs`, so the learner trains on the same corrupted signal a real vehicle has.

## Control

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

## Aerodynamics (ground truth, hidden from the learner)

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

## Atmosphere

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

## The learner, and why it is safe for any weights

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

## Online learning

The model learns during every flight from scratch. Every 10 ms it takes a clipped gradient step on
the squared force error over the last 128 accelerometer samples, plus a weak pull toward its starting
weights, so it forgets stale wind instead of drifting.

## What the agent reasons about

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

## What the tests check

- **Rigid body:** free fall matches the closed form, energy is conserved without thrust, the
  quaternion stays unit length, and hover is an equilibrium.
- **Control:** perfect feedforward cancels a disturbance, and integral action removes a steady
  offset.
- **Learned models:** the learner is passive for random weights, the neural net can inject energy,
  and the full aerodynamics leave the dissipative class while vanishing when the drone moves with
  the air.
- **Wind:** turbulence has the surface-layer standard deviations, the log profile equals 1 at 10 m,
  gust fronts travel downwind, and the browser's wind matches the simulator's exactly.

## References

- Lee, Leok and McClamroch, 2010: geometric SE(3) control.
- Faessler, Franchi and Scaramuzza, 2018: rotor drag and induced lift.
- Panofsky and Dutton, 1984: surface-layer turbulence.
- von Kármán spectrum and Taylor's frozen turbulence.
- Oke, 1988, and Britter and Hanna, 2003: street-canyon flow, for the city scene.
- The learner's structure is inspired by port-Hamiltonian neural networks and by Neural-Fly
  (O'Connell et al., 2022).

## Limitations

- **Simulation only.** Nothing has been flown on hardware, and motor lag and sensor noise are
  assumed values, not measured ones.
- **The energy bound covers the learned model, not the whole closed loop.** How D varies with
  velocity is unbounded, so the Jacobian of the learned force can still exceed d_max.
- **A single wind estimate.** The learner treats the air around the drone as uniform. Strong shear
  across the vehicle is outside that assumption.
- **Effects not modelled:** ground effect, rotor-to-rotor interaction, blade flapping beyond linear
  rotor drag, battery sag, and rain.
