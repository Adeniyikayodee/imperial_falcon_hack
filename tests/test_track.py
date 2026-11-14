import falcon  # noqa: F401
from falcon import track
from falcon.experiment import run_experiment


def test_experiment_logs_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(track, "RUNS", tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    r = run_experiment("none", "hover", (0.0, 0.0, 0.0), seeds=1, duration=6.0)
    rows = track.load_runs()
    assert len(rows) == 1 and rows[0]["metrics"] == r["metrics"]
    assert r["metrics"]["crash_rate"] == 0.0 and r["metrics"]["rmse_m"] < 3e-3  # holds within millimetres on 5 mm estimator noise
