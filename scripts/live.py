"""Live demo server. Steps the JAX simulator in real time and streams frames to the browser.
    uv run python scripts/live.py [--scene line|city]     then open http://localhost:8765

GET  /         the scene page: power-line inspection (web/simple.html) or city (web/city.html)
GET  /scene    {"scene": "line" | "city"}
GET  /stage    full 3D stage (web/live.html)
GET  /path     the reference route as [[x, y, z], ...], sampled from the simulator's trajectory
GET  /stream   server-sent events, one JSON frame per 20 ms
POST /control  {"mean": [x, y], "gust": g, "learning": "gentle"|"aggressive",
                "burst": {"speed": s, "dir": [x, y]}, "reset": true, "paused": bool}
GET  /agent    {"ready": bool, "model": ..., "how": ...}: whether the Claude flight engineer can answer
POST /ask      {"q": "..."}: the engineer drives this demo and streams one JSON event per line
POST /mission  {"brief": "..."}: the engineer flies a flight-test campaign, files a checked report,
               then shows the limit on this demo; streams one JSON event per line and records it
               to runs/missions/
POST /replay   {"speed": 20}: replays the latest recorded mission (or docs/missions/sample.jsonl),
               driving this demo through the same live steps; no API or network needed
"""
import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jax
import numpy as np

import falcon  # noqa: F401
from agent import tools
from agent.run import BRIEF, engineer, mission, status
from falcon import track
from falcon.live import LEARNING, Live
from falcon.traj import TRAJECTORIES

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SAMPLE = ROOT / "docs" / "missions" / "sample.jsonl"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
FRAME_S = 0.02
SCENES = {
    "line": {"traj": "inspect", "site": "open", "page": "simple.html"},
    "city": {"traj": "street", "site": "city", "page": "city.html"},
}


class Stage:
    def __init__(self, scene="line", traj=None):
        traj, site = traj or SCENES[scene]["traj"], SCENES[scene]["site"]
        self.scene, self.page = scene, SCENES[scene]["page"]
        self.live, self.lock, self.cond = Live(traj=traj, site=site), threading.Lock(), threading.Condition()
        path = jax.vmap(TRAJECTORIES[traj]())(np.arange(20.0, 32.0, 0.02))  # past the ramp, one or more laps
        self.path = json.dumps(np.round(np.asarray(path), 4).tolist()).encode()
        self.frame, self.seq, self.paused = None, 0, False
        self.hist = deque(maxlen=int(60 / FRAME_S))  # (t, {model: tracking error m, NaN while crashed})
        self.asking = threading.Lock()

    def run(self):
        nxt = time.perf_counter()
        while True:
            if not self.paused:
                with self.lock:
                    frame = self.live.tick()
                ref = np.asarray(frame["ref"])
                self.hist.append((frame["t"], {m: np.nan if r["crashed"] else float(np.linalg.norm(np.asarray(r["pos"]) - ref))
                                               for m, r in frame["models"].items()}))
                with self.cond:
                    self.frame, self.seq = frame, self.seq + 1
                    self.cond.notify_all()
            nxt += FRAME_S
            time.sleep(max(0.0, nxt - time.perf_counter()))
            nxt = max(nxt, time.perf_counter() - FRAME_S)  # never try to catch up a long stall

    def control(self, c: dict):
        with self.lock:
            if "learning" in c and c["learning"] not in LEARNING:
                raise ValueError("learning must be gentle or aggressive")
            mean = c.get("mean")
            if mean is not None and (len(mean) != 2 or (mean[0] ** 2 + mean[1] ** 2) ** 0.5 > 10):
                raise ValueError("mean wind must be [x, y] with speed <= 10 m/s")
            if "gust" in c and not 0 <= float(c["gust"]) <= 4:
                raise ValueError("gust must be in [0, 4] m/s")
            self.live.set_wind(mean=mean, gust=c.get("gust"), learning=c.get("learning"))
            if b := c.get("burst"):
                self.live.burst(min(float(b.get("speed", 6.0)), 9.0), tuple(b.get("dir", (1.0, 0.0))))
            if c.get("reset"):
                self.live.reset()
            if "paused" in c:
                self.paused = bool(c["paused"])

    def observe(self, seconds: float) -> dict:
        """Conditions now, and each model's tracking error over the last `seconds`."""
        with self.lock:
            mean, learning = self.live.mean[:2].tolist(), self.live.learning
        rows = list(self.hist)
        rows = [r for r in rows if r[0] >= rows[-1][0] - seconds] if rows else []
        out = {"window_s": seconds, "mean_wind_xy": mean, "wind_mps": round(float(np.hypot(*mean)), 2), "learning": learning, "models": {}}
        for m in self.live.models:
            e = np.array([r[1][m] for r in rows])
            ok = e[np.isfinite(e)]
            out["models"][m] = {"rms_m": round(float(np.sqrt(np.mean(ok**2))), 4) if ok.size else None,
                                "max_m": round(float(ok.max()), 4) if ok.size else None, "crashed_frames": int((~np.isfinite(e)).sum())}
        return out


STAGE: Stage = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/stream":
            return self.stream()
        if self.path == "/path":
            return self.send(200, STAGE.path)
        if self.path == "/agent":
            return self.send(200, json.dumps(status()).encode())
        if self.path == "/scene":
            return self.send(200, json.dumps({"scene": STAGE.scene}).encode())
        rel = {"/": STAGE.page, "/stage": "live.html"}.get(self.path, self.path.lstrip("/"))
        f = (WEB / rel).resolve()
        if WEB not in f.parents or not f.is_file():
            return self.send(404, b'{"error": "not found"}')
        self.send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/ask":
                return self.ask(str(body.get("q", "")).strip()[:2000], engineer)
            if self.path == "/mission":
                return self.ask(str(body.get("brief", "")).strip()[:2000] or BRIEF, mission)
            if self.path == "/replay":
                return self.replay(min(max(float(body.get("speed", 20)), 1.0), 200.0))
            if self.path != "/control":
                return self.send(404, b'{"error": "not found"}')
            STAGE.control(body)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return self.send(400, json.dumps({"error": str(e)}).encode())
        self.send(200, b'{"ok": true}')

    def ask(self, q: str, agent):
        s = status()
        if not s["ready"]:
            return self.send(503, json.dumps(s).encode())
        if not q:
            return self.send(400, b'{"error": "empty question"}')
        if not STAGE.asking.acquire(blocking=False):
            return self.send(409, b'{"error": "the engineer is already flying a question"}')
        rec = None
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            t0 = time.monotonic()
            if agent is mission:
                (track.RUNS / "missions").mkdir(parents=True, exist_ok=True)
                rec = open(track.RUNS / "missions" / f"mission-{time.strftime('%Y%m%d-%H%M%S')}.jsonl", "w")
                rec.write(json.dumps({"t": 0.0, "type": "brief", "text": q}) + "\n")
            for e in agent(q, STAGE):
                if rec:
                    rec.write(json.dumps({"t": round(time.monotonic() - t0, 2), **e}) + "\n")
                    rec.flush()
                self.wfile.write(json.dumps(e).encode() + b"\n")
                self.wfile.flush()
            self.wfile.write(b'{"type": "done"}\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if rec:
                rec.close()
            STAGE.asking.release()

    def replay(self, speed: float):
        """Play back a recorded mission, compressing idle time by `speed` but running its live steps on
        this demo in real time, so the audience still sees the gust arrive."""
        found = sorted((track.RUNS / "missions").glob("mission-*.jsonl"))
        path = found[-1] if found else SAMPLE
        if not path.is_file():
            return self.send(404, b'{"error": "no recorded mission yet; run one with Clear for flight"}')
        if not STAGE.asking.acquire(blocking=False):
            return self.send(409, b'{"error": "the engineer is already flying"}')
        live = tools.Live(STAGE)
        acts = {"live_set": lambda i: live.live_set(**i), "live_gust": lambda i: live.live_gust(),
                "live_reset": lambda i: live.live_reset()}
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            prev = 0.0
            for line in path.read_text().splitlines():
                e = json.loads(line)
                t = e.pop("t", prev)
                time.sleep(min(max(t - prev, 0.0) / speed, 1.5))
                prev = t
                if e["type"] == "tool" and e["name"] in acts:
                    acts[e["name"]](e["input"])
                if e["type"] == "tool" and e["name"] == "live_watch":
                    time.sleep(min(float(e["input"].get("seconds", 10)), 6.0))
                self.wfile.write(json.dumps(e).encode() + b"\n")
                self.wfile.flush()
            self.wfile.write(b'{"type": "done"}\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            STAGE.asking.release()

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = -1
        try:
            while True:
                with STAGE.cond:
                    STAGE.cond.wait_for(lambda: STAGE.seq != seen, timeout=1.0)
                    frame, seen = STAGE.frame, STAGE.seq
                if frame is not None:
                    self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--scene", default="line", choices=list(SCENES))
    ap.add_argument("--traj", choices=["inspect", "street", "lap", "figure8"], help="override the scene's route")
    a = ap.parse_args()
    global STAGE
    STAGE = Stage(a.scene, a.traj)
    print("compiling the simulator...", flush=True)
    for learning in LEARNING:  # compile both learning rates before the audience is watching
        STAGE.live.set_wind(learning=learning)
        STAGE.live.tick()
    STAGE.live.set_wind(learning="gentle")
    STAGE.live.reset()
    threading.Thread(target=STAGE.run, daemon=True).start()
    print(f"live demo on http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
