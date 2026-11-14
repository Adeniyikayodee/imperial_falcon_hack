"""Report checker: every measured number in a flight readiness report must cite the run log.

A claim is a number, an optional unit and a citation: `2.3 cm [r1a2b3.rmse_m]` (a run metric),
`0.6 rad [facts.max_tilt_rad]` (a vehicle fact) or `7.6 m/s [pred]` (the engineer's own prediction,
shown as such). A cited value must equal the logged one at the precision written. A number in a
result unit (cm, mm, N, mN, %, rad, deg) with no citation is rejected, so nothing reaches the report
that was not measured, looked up, or labelled a prediction.
"""
import math
import re

from falcon.track import find_run

UNITS = {"": 1.0, "m": 1.0, "cm": 0.01, "mm": 0.001, "N": 1.0, "mN": 0.001, "%": 0.01, "rad": 1.0,
         "deg": math.pi / 180, "°": math.pi / 180, "m/s": 1.0, "s": 1.0, "kg": 1.0, "g": 0.001,
         "N s/m": 1.0, "N/(m/s)": 1.0, "N per m/s": 1.0, "x": 1.0, "×": 1.0,
         "N·m": 1.0, "N m": 1.0, "Nm": 1.0, "mN·m": 0.001, "N·m/rad": 1.0, "N m/rad": 1.0}
NUM = r"(?<![\w.])(?P<num>-?\d+(?:\.\d+)?)"
CITED = re.compile(NUM + r"\s*(?P<unit>[^\[\]\d\n|]{0,12}?)\s*\[(?:(?P<src>r[0-9a-f]{6}|facts)\.(?P<key>\w+)|(?P<pred>pred))\]")
RESULT = re.compile(NUM + r"\s*(?P<unit>cm|mm|mN|N|%|rad|deg|°)(?![A-Za-z])(?!\s*(?:CI|confidence))")
MIN_RUNS = 3


def verify(report: str, facts: dict) -> dict:
    problems, claims, spans = [], [], []
    for m in CITED.finditer(report):
        spans.append(m.span())
        text, unit = m.group(0), m.group("unit").strip()
        if m.group("pred"):
            claims.append({"text": text, "source": "pred"})
            continue
        if unit not in UNITS:
            problems.append(f"'{text}': unknown unit '{unit}'")
            continue
        src, key = m.group("src"), m.group("key")
        if src == "facts":
            v = facts.get(key)
        else:
            run = find_run(src)
            v = run["metrics"].get(key) if run else None
            if run is None:
                problems.append(f"'{text}': no run {src} in the log")
                continue
        if not isinstance(v, (int, float)):
            problems.append(f"'{text}': {src} has no numeric '{key}'")
            continue
        shown, places = float(m.group("num")), len(m.group("num").partition(".")[2])
        want = v / UNITS[unit]
        if abs(shown - want) > 0.5 * 10**-places * 1.001 + 1e-12:
            problems.append(f"'{text}': logged {key} is {want:.{places + 2}f} {unit}".rstrip())
            continue
        claims.append({"text": text, "source": src, "key": key, "logged": v})
    for m in RESULT.finditer(report):
        if not any(a <= m.start() < b for a, b in spans):
            problems.append(f"'{m.group(0)}' has no citation; cite the run or fact, mark it [pred], or drop it")
    if sum(c["source"] not in ("facts", "pred") for c in claims) < MIN_RUNS:
        problems.append(f"cite at least {MIN_RUNS} measured run values")
    footer = "---\nVerified against the run log: " + f"{len(claims)} claims.\n\n" + "\n".join(
        f"- `{c['text']}`" + (f" = {c['logged']:.6g}" if "logged" in c else " (prediction)") for c in claims)
    return {"accepted": not problems, "problems": problems, "claims": claims, "footer": footer}
