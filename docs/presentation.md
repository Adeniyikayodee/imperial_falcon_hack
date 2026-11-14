# Live presentation (about 3 minutes)

Before the judges arrive, start the server (`uv run --extra agent python scripts/live.py`) and open
http://localhost:8765. Then either:

- press **Clear for flight** about 10 minutes ahead, and present while the mission panel fills in; or
- if time or the network is short, press **Replay**. It plays a recorded real mission at 20× speed,
  including its live gust check, with no API needed.

**0:00 to 0:20. The problem.** Show the 3D corridor.

> "Drones inspect live power lines a metre from the wire, in wind they can't see. Someone has to
> clear them to fly. I built an agent that does that job, and I built it so you can trust it."

**0:20 to 0:50. The brief and the forecast.** Point to the brief and the Linkup result.

> "One sentence in. It checks the real forecast for the site, then holds position at 20 metres to
> measure its own drag, from the accelerometer, the way the vehicle would on hardware."

**0:50 to 1:20. Predict and expand.** Point to the code it wrote, then the envelope chart.

> "It writes code to fit a drag law and commits to a prediction before testing. Then it expands the
> envelope, never more than 2 m/s past what it has cleared. That rule is in the tools, not the
> prompt. Here are the crosses where all three drones lose control."

**1:20 to 1:50. The physics.**

> "Drag alone predicts more margin than the drone has. The agent measures the reason: rotor drag
> acts above the centre of mass and pitches the drone back, which costs it tilt. I checked by
> removing that torque in the simulator, and the edge moves up."

**1:50 to 2:20. The report.** Point to the green stamp and hover over a citation.

> "Every number cites the flight it came from. A checker reads the log and rejects anything misquoted
> or invented. Each of these opens as an MLflow run on Databricks."

**2:20 to 2:50. The live check.** The wind rises, a gust front arrives, and the three drones react.

> "Then it flies its own limit live. The blue learner holds within centimetres, because its learned
> force can only remove energy. With learning set ten times too fast, the unconstrained net
> (purple) is nearly seven times worse than no model at all."

**2:50 to 3:00.**

> "Falcon: an agent that can't skip a safety step, and can't make up a number."

## Likely questions

- **Is this real hardware?** No, it's simulation. The sensing path, motor lag and wind are modelled
  to be honest about what a real vehicle sees, and the README lists what is not modelled.
- **Why trust the simulator?** Its tests check the rigid body against closed forms, the turbulence
  against surface-layer statistics, and the learner's energy bound for random weights. The learner
  is never tested on its own equation, because the true air includes effects it can't represent.
- **What was built before the event?** A prototype of the flight simulator. The agent, the checker,
  the integrations and the unified wind I built here.
- **What does a mission cost?** About $3 to $6 on Fable 5.2, and about half that on Opus 5.5.
