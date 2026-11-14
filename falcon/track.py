"""Run logging. Always appends to runs/results.jsonl; also logs to MLflow when
MLFLOW_TRACKING_URI is set (use `databricks` to send runs to a Databricks workspace)."""
import hashlib
import json
import os
import time
from pathlib import Path

RUNS = Path(os.getenv("FALCON_RUNS", "runs"))


def log_run(name: str, params: dict, metrics: dict) -> str:
    """Append a run and return its id, the handle reports cite."""
    t = time.time()
    rid = "r" + hashlib.sha1(f"{name}{t}{metrics}".encode()).hexdigest()[:6]
    record = {"id": rid, "name": name, "time": t, "params": params, "metrics": metrics}
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    if os.getenv("MLFLOW_TRACKING_URI"):
        import mlflow

        mlflow.set_experiment(os.getenv("FALCON_EXPERIMENT", "/Shared/falcon"))
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("falcon_id", rid)  # the id reports cite, so a citation opens in Databricks
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    return rid


def load_runs(limit: int | None = None) -> list[dict]:
    path = RUNS / "results.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return rows[-limit:] if limit else rows


def find_run(rid: str) -> dict | None:
    return next((r for r in reversed(load_runs()) if r.get("id") == rid), None)
