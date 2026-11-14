"""Claude flight engineer: answers questions, or flies a whole flight-test mission, through tools.
    uv run --extra agent python -m agent.run "Is fast learning safe in a 4 m/s wind toward the line?"
    uv run --extra agent python -m agent.run --mission "Clear the drone for tomorrow's line inspection."

A mission identifies the drag in flight, predicts the envelope edge with code it writes and runs
(server-side code execution), expands the envelope under the build-up rule, reconciles prediction
with test, and files a flight readiness report that agent.verify checks number by number against
the run log before it is accepted.

Nothing here needs credentials until a question is asked. `status()` checks, without a network call,
whether the SDK can resolve ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile;
callers wait on it instead of failing. A profile is picked up on the next check; an exported key needs
the process restarted.
"""
import json
import os
import sys

from agent import tools

MODEL = os.getenv("FALCON_MODEL", "claude-fable-5-2")  # most capable for long agentic work; claude-opus-5-5 costs half
HOW = "run `ant auth login`, or export ANTHROPIC_API_KEY and restart"
BETAS = ["server-side-fallback-2026-07-01", "thinking-display-updates-2026-08-18"]
CODE = {"type": "code_execution_20260521", "name": "code_execution"}
SYSTEM = """You are the flight engineer for a Crazyflie quadrotor simulator that learns an aerodynamic
residual online. Models: none (integral control only), mlp (unconstrained net), ph (dissipative residual
with bounded damping). All share one controller with integral action. Ground every claim in tool output:
quote the numbers and say when a result is near the vehicle's tilt or thrust limit (vehicle_facts).
With live tools, change one thing at a time, watch long enough to measure (10 s or more), and restore
the conditions you found unless asked otherwise. Prefer the fewest runs that answer. Keep the answer short."""
MISSION = """You are the flight-test engineer clearing a 27 g Crazyflie quadrotor for the mission in the brief.
It flies a simulator with hidden aerodynamics; three controllers share one SE(3) controller with integral
action: none (no learned model), mlp (unconstrained learned force), ph (learned force that can only remove
energy, damping bounded below the controller's own). The deliverable is a flight readiness report a chief
pilot would sign: a decision (GO, GO WITH LIMITS, or NO-GO) and the operating limits that make it safe.

If the brief names a place, check its wind forecast with site_forecast first, and put the source URL in
the report. Plan for the worse of the brief and the forecast, and say if the forecast falls outside
what you can clear.

Work the way flight test is done on real aircraft:
Batch flights and the live demo fly the same air: a surface-layer wind given at the 10 m reference height
(stronger at the 20 m pass), turbulence at intensity gust, and optional gust fronts. The live check runs at
gust 1 with one 2 m/s gust front, so clear that condition in batch (gust 1, front 2) before you recommend
a limit.

1. Identify. Hold station at pass height in steady wind (traj station, model none, gust 0) at a few speeds
   and read aero_force_xy_n, the aerodynamic force the accelerometer measured, and aero_torque_nm, the
   aerodynamic torque. Get the vehicle's limits and attitude stiffness from vehicle_facts.
2. Predict. Write and run code to fit a drag law to those points and predict the wind at which the
   steady force uses up the horizontal force the tilt limit allows. Commit to that number before testing.
3. Expand. Fly the mission trajectory in turbulence for each controller, stepping up in wind. The
   harness enforces build-up (at most 2 m/s past the last wind a configuration cleared) and will not fly
   at or above a wind that crashed. Use enough seeds to separate the controllers (4 or more) and look at
   max error as well as rms: a pass beside a live conductor is limited by the worst moment.
4. Reconcile. Compare the prediction with the test. Where they disagree, say which physics your model
   left out and how you know. Check whether fast learning (lr 0.5) is safe at the recommended limit.
5. Report with submit_report, under 350 words: decision, operating limits (mean wind, gust, controller,
   learning rate, with margin below the edge), the evidence as a short table, prediction against test,
   and what this campaign does not cover. Cite every measured number right after it, one number per
   citation: `2.3 cm [r1a2b3.rmse_m]`, `0.6 rad [facts.max_tilt_rad]`, `7.6 m/s [pred]`. The checker
   rejects uncited or misquoted numbers; fix what it lists and resubmit until accepted.
6. If the live demo is attached, after the report is accepted fly the recommended limit live once
   (live_set, a gust, live_watch 10 s or more), say whether it held, then restore calm gentle conditions.

Between steps, write one short line on what you just learned and what you will do next. Spend runs like
flight hours: every flight should answer a question."""


def status() -> dict:
    """Whether a question can be sent now, and how to enable it if not."""
    try:
        import anthropic
    except ImportError:
        return {"ready": False, "model": MODEL, "how": "uv sync --extra agent"}
    c = anthropic.Anthropic()
    ready = any(x is not None for x in (c.api_key, c.auth_token, c.credentials))
    return {"ready": ready, "model": MODEL, "how": "" if ready else HOW}


def _tools(stage, campaign=None):
    from anthropic import beta_tool

    live = tools.Live(stage).all() if stage is not None else []
    if campaign is None:
        return [beta_tool(f) for f in [tools.fly, tools.recent_runs, tools.vehicle_facts, *live]]
    return [beta_tool(f) for f in [campaign.fly, tools.vehicle_facts, tools.site_forecast, campaign.submit_report, *live]] + [CODE]


def _events(message):
    for block in message.content:
        if block.type == "tool_use":
            yield {"type": "tool", "name": block.name, "input": block.input}
        elif block.type == "server_tool_use":
            yield {"type": "code", "input": block.input}
        elif block.type == "bash_code_execution_tool_result":
            r = block.content
            yield {"type": "code_result", "stdout": getattr(r, "stdout", ""), "stderr": getattr(r, "stderr", ""),
                   "error": getattr(r, "error_code", None)}
        elif block.type == "thinking" and block.thinking:
            yield {"type": "note", "text": block.thinking}
        elif block.type == "text":
            yield {"type": "text", "text": block.text}


def _run(system, prompt, tool_list, effort, after=lambda: []):
    """Drive the tool runner, yielding events; `after` drains events the tools produced in between."""
    import anthropic

    try:
        runner = anthropic.Anthropic().beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=system,
            tools=tool_list,
            thinking={"type": "adaptive", "display": "updates"},
            output_config={"effort": effort},
            betas=BETAS,
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        for message in runner:
            yield from after()
            yield from _events(message)
            if message.stop_reason == "refusal":
                yield {"type": "error", "text": f"declined: {getattr(message.stop_details, 'explanation', '')}"}
        yield from after()
    except anthropic.AuthenticationError:
        yield {"type": "error", "text": f"credentials rejected; {HOW}"}
    except anthropic.RateLimitError:
        yield {"type": "error", "text": "rate limited; try again shortly"}
    except anthropic.APIStatusError as e:
        yield {"type": "error", "text": f"API error {e.status_code}: {e.message}"}
    except anthropic.APIConnectionError:
        yield {"type": "error", "text": "no connection to the API"}


def engineer(prompt: str, stage=None):
    """Answer one question, yielding {"type": "tool"|"text"|"note"|"error", ...} events as they happen."""
    yield from _run(SYSTEM, prompt, _tools(stage), "medium")


def mission(brief: str, stage=None, campaign=None):
    """Fly a flight-test mission. Besides the engineer's events, yields {"type": "run"} for every
    flight, {"type": "rejected"} for each report the checker sends back, and {"type": "report"} once one
    is accepted."""
    campaign = campaign or tools.Campaign()
    seen = {"runs": 0, "subs": 0}

    def after():
        for r in campaign.flown[seen["runs"]:]:
            yield {"type": "run", **r}
        seen["runs"] = len(campaign.flown)
        for s in campaign.submissions[seen["subs"]:]:
            yield {"type": "rejected", "problems": s["problems"]} if not s["accepted"] else \
                  {"type": "report", **campaign.report}
        seen["subs"] = len(campaign.submissions)

    yield from _run(MISSION, brief, _tools(stage, campaign), "high", after)


def main(argv: list[str]) -> int:
    s = status()
    if not s["ready"]:
        print(f"flight engineer offline: {s['how']}", file=sys.stderr)
        return 2
    fly_mission = argv[:1] == ["--mission"]
    prompt = " ".join(argv[1:] if fly_mission else argv)
    events = mission(prompt or BRIEF) if fly_mission else engineer(prompt or "How much does the ph residual help on a figure-8 in a 4 m/s wind?")
    for e in events:
        t = e["type"]
        if t == "tool":
            print(f"> {e['name']}({json.dumps(e['input'])})", flush=True)
        elif t == "code":
            print(f"$ {e['input'].get('command') or e['input'].get('code', '')}", flush=True)
        elif t == "code_result":
            print(e["stdout"] or e["stderr"] or e["error"], flush=True)
        elif t == "run":
            m = e["metrics"]
            print(f"  {e['id']}: rms {100 * m['rmse_m']:.1f} cm, max {100 * m['max_err_m']:.1f} cm, crash {m['crash_rate']:.0%}", flush=True)
        elif t == "rejected":
            print("  report rejected:\n  - " + "\n  - ".join(e["problems"]), flush=True)
        elif t == "report":
            print(f"\n=== accepted, saved to {e['path']} ===\n{e['markdown']}", flush=True)
        else:
            print(e.get("text", ""), flush=True)
    return 0


BRIEF = ("Clear the drone for tomorrow's power-line inspection pass at 20 m. Forecast: 3 to 5 m/s toward "
         "the line, gusty (gust 1.0). Decide GO, GO WITH LIMITS or NO-GO, and set the operating limits.")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
