import json

import pytest

from agent import run, tools
from falcon import track


def test_fly_rejects_out_of_envelope_requests():
    with pytest.raises(ValueError):
        tools.fly("ph", "figure8", 9.0, 9.0, 0.0, 0.05, 1)
    with pytest.raises(ValueError):
        tools.fly("ph", "figure8", 1.0, 0.0, 0.0, 5.0, 1)
    with pytest.raises(ValueError):
        tools.fly("rnn", "figure8", 1.0, 0.0, 0.0, 0.05, 1)


def test_fly_and_recent_runs_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(track, "RUNS", tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    out = json.loads(tools.fly("ph", "hover", 2.0, 0.0, 0.0, 0.05, 1))
    assert out["metrics"]["crash_rate"] == 0.0
    assert json.loads(tools.recent_runs(5))[0]["metrics"] == out["metrics"]


def test_vehicle_facts_bound_is_below_controller_gain():
    f = json.loads(tools.vehicle_facts())
    assert f["ph_damping_bound_n_per_mps"] < f["controller_velocity_gain_n_per_mps"]


class FakeStage:
    def __init__(self):
        self.sent = []

    def control(self, c):
        self.sent.append(c)

    def observe(self, seconds):
        return {"window_s": seconds, "mean_wind_xy": [0.0, 4.0], "models": {}}


def test_live_tools_send_page_equivalent_controls_within_limits():
    st = FakeStage()
    live = tools.Live(st)
    live.live_set(4.0, 90.0, "aggressive")
    mean = st.sent[-1]["mean"]
    assert abs(mean[0]) < 1e-12 and abs(mean[1] - 4.0) < 1e-12 and st.sent[-1]["learning"] == "aggressive"
    live.live_gust()
    assert st.sent[-1]["burst"] == {"speed": 2.0, "dir": [0.0, 1.0]}
    with pytest.raises(ValueError):
        live.live_set(9.0, 90.0, "gentle")
    with pytest.raises(ValueError):
        live.live_set(3.0, 90.0, "reckless")


def test_status_waits_for_credentials_without_a_network_call(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    s = run.status()
    assert not s["ready"] and "ant auth login" in s["how"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    assert run.status()["ready"]


def test_tool_schemas_come_from_the_documented_functions():
    pytest.importorskip("anthropic")
    names = [t.name for t in run._tools(FakeStage())]
    assert names == ["fly", "recent_runs", "vehicle_facts", "live_set", "live_gust", "live_watch", "live_reset"]
    assert [t.name for t in run._tools(None)] == names[:3]


def test_cli_exits_cleanly_when_offline(tmp_path, monkeypatch, capsys):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert run.main(["hello"]) == 2 and "offline" in capsys.readouterr().err
