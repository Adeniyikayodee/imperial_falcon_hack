import json

import pytest

from agent import tools
from agent.verify import verify
from falcon import track

FACTS = json.loads(tools.vehicle_facts())


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(track, "RUNS", tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    ids = [track.log_run(f"run{i}", {}, {"rmse_m": 0.0204 + i / 100, "crash_rate": 0.0, "aero_force_xy_n": 0.0661})
           for i in range(3)]
    return ids


def test_runs_get_citable_ids(ledger):
    assert len(set(ledger)) == 3 and track.find_run(ledger[1])["metrics"]["rmse_m"] == pytest.approx(0.0304)


def test_report_with_every_number_cited_and_matching_is_accepted(ledger):
    a, b, c = ledger
    report = (f"GO WITH LIMITS. Learner holds 2.0 cm [{a}.rmse_m], then 3.0 cm [{b}.rmse_m] and 4.04 cm [{c}.rmse_m] "
              f"with 0% [{a}.crash_rate] crashes (95% CI). Drag 66 mN [{a}.aero_force_xy_n] at 4 m/s; "
              f"tilt limit 0.6 rad [facts.max_tilt_rad]; edge 7.6 m/s [pred].")
    res = verify(report, FACTS)
    assert res["accepted"], res["problems"]
    assert len(res["claims"]) == 7 and "Verified against the run log: 7 claims" in res["footer"]


def test_misquoted_uncited_and_invented_numbers_are_rejected(ledger):
    a, b, c = ledger
    res = verify(f"2.1 cm [{a}.rmse_m], 3.0 cm [{b}.rmse_m], 4.0 cm [{c}.rmse_m], max 9 cm, 1 cm [rdeadbe.rmse_m]", FACTS)
    assert not res["accepted"]
    p = " ".join(res["problems"])
    assert "logged rmse_m is 2.04" in p and "'9 cm' has no citation" in p and "no run rdeadbe" in p


def test_report_must_rest_on_measured_runs(ledger):
    res = verify("Edge at 7.6 m/s [pred], tilt 0.6 rad [facts.max_tilt_rad].", FACTS)
    assert not res["accepted"] and any("at least 3" in p for p in res["problems"])


def test_campaign_enforces_build_up_and_stops_at_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(track, "RUNS", tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fake = {2.0: 0.0, 4.0: 1.0}
    monkeypatch.setattr(tools, "fly", lambda m, t, wx, wy, g, lr, s, f=0.0: json.dumps(
        {"id": "r000000", "params": {}, "metrics": {"crash_rate": fake.get(round(abs(wx) + abs(wy), 1), 0.0)}}))
    c = tools.Campaign(tmp_path / "reports")
    with pytest.raises(ValueError, match="build-up"):
        c.fly("ph", "inspect", 0.0, 3.0, 1.0, 0.05, 4)
    c.fly("ph", "inspect", 0.0, 2.0, 1.0, 0.05, 4)
    c.fly("ph", "inspect", 0.0, 4.0, 1.0, 0.05, 4)  # crashes: the edge
    with pytest.raises(ValueError, match="crashed at 4"):
        c.fly("ph", "inspect", 0.0, 4.0, 1.0, 0.05, 4)
    c.fly("ph", "inspect", 0.0, 3.0, 1.0, 0.05, 4)  # refining below the edge is allowed
    c.fly("none", "inspect", 0.0, 2.0, 1.0, 0.05, 4)  # each configuration builds up on its own
    with pytest.raises(ValueError, match="build-up"):
        c.fly("ph", "inspect", 0.0, 3.0, 1.0, 0.5, 4)  # nor does a new learning rate inherit it


def test_accepted_report_is_saved_with_its_verification(ledger, tmp_path):
    a, b, c = ledger
    camp = tools.Campaign(tmp_path / "reports")
    bad = json.loads(camp.submit_report("Holds 9 cm."))
    assert not bad["accepted"] and camp.report is None
    ok = json.loads(camp.submit_report(f"{2.0} cm [{a}.rmse_m], 3.0 cm [{b}.rmse_m], 4.0 cm [{c}.rmse_m]"))
    assert ok["accepted"] and "Verified against the run log" in open(ok["saved"]).read()
    assert [s["accepted"] for s in camp.submissions] == [False, True]


def test_mission_tools_include_code_execution():
    pytest.importorskip("anthropic")
    from agent import run

    ts = run._tools(None, tools.Campaign())
    assert [getattr(t, "name", None) for t in ts[:-1]] == ["fly", "vehicle_facts", "site_forecast", "submit_report"]
    assert ts[-1] == run.CODE


def test_batch_flights_fly_the_live_wind_field(tmp_path, monkeypatch):
    monkeypatch.setattr(track, "RUNS", tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from falcon.experiment import run_experiment

    calm = run_experiment("none", "station", (0.0, 4.0, 0.0), 0.0, 0.05, 1, duration=14.0)["metrics"]
    front = run_experiment("none", "station", (0.0, 4.0, 0.0), 0.0, 0.05, 1, duration=14.0, front=2.0)["metrics"]
    low = run_experiment("none", "hover", (0.0, 4.0, 0.0), 0.0, 0.05, 1, duration=14.0)["metrics"]
    assert front["max_err_m"] > 2 * calm["max_err_m"]  # the gust front reaches the vehicle
    assert calm["aero_force_xy_n"] > 2 * low["aero_force_xy_n"]  # the log profile: 20 m sees far more wind than 1 m
