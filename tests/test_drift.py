from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from energy_forecast.app import app
from energy_forecast.drift import MAE_RATIO, check_and_alert, drift_payload

client = TestClient(app)


def _board(*, hour_n=20, hour_mae=400.0, hour_cov=0.8, hold_mae=200.0):
    return {
        "lab": {
            "packs": [
                {"id": "hour", "holdout_mae": hold_mae, "holdout_coverage": 0.8},
                {"id": "day", "holdout_mae": 850.0, "holdout_coverage": 0.81},
                {"id": "week", "holdout_mae": 1100.0, "holdout_coverage": 0.77},
            ]
        },
        "next30": {"metrics": {"n": hour_n, "mae": hour_mae, "coverage_80": hour_cov}},
        "day": {"current": {"metrics": {"n": 2, "mae": 500.0, "coverage_80": 0.8}}},
        "week": {"current": {"metrics": {"n": 2, "mae": 900.0, "coverage_80": 0.8}}},
    }


def test_drift_mae_ratio_sends_once(tmp_path, monkeypatch):
    (tmp_path / "frozen").mkdir()
    monkeypatch.setattr("energy_forecast.paths.DATA_LIVE", tmp_path)
    monkeypatch.setattr("energy_forecast.live.board_payload", lambda refresh=False: _board())
    monkeypatch.setenv("DRIFT_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("DRIFT_SMTP_HOST", "smtp.example.com")
    sent: list[dict] = []
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    first = check_and_alert(send=sent.append, now=now)
    assert first["alerts"]
    assert any(row["kind"] == "mae" for row in first["alerts"])
    assert first["mailed"] == ["next30:mae"]
    assert sent[0]["preset"] == "next30"
    assert sent[0]["ratio"] >= MAE_RATIO
    second = check_and_alert(send=sent.append, now=now + timedelta(hours=1))
    assert second["mailed"] == []
    assert "next30:mae" in second["skipped"]
    later = check_and_alert(send=sent.append, now=now + timedelta(hours=13))
    assert later["mailed"] == ["next30:mae"]
    assert len(sent) == 2


def test_drift_does_not_mail_without_smtp(tmp_path, monkeypatch):
    (tmp_path / "frozen").mkdir()
    monkeypatch.setattr("energy_forecast.paths.DATA_LIVE", tmp_path)
    monkeypatch.setattr("energy_forecast.live.board_payload", lambda refresh=False: _board())
    monkeypatch.delenv("DRIFT_EMAIL_TO", raising=False)
    monkeypatch.delenv("DRIFT_SMTP_HOST", raising=False)
    sent: list[dict] = []
    out = check_and_alert(send=sent.append, now=datetime(2026, 10, 9, tzinfo=timezone.utc))
    assert out["alerts"]
    assert out["mailed"] == []
    assert sent == []
    assert out["email_ready"] is False


def test_drift_coverage_floor(tmp_path, monkeypatch):
    (tmp_path / "frozen").mkdir()
    monkeypatch.setattr("energy_forecast.paths.DATA_LIVE", tmp_path)
    monkeypatch.setattr(
        "energy_forecast.live.board_payload",
        lambda refresh=False: _board(hour_mae=210.0, hour_cov=0.4),
    )
    monkeypatch.setenv("DRIFT_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("DRIFT_SMTP_HOST", "smtp.example.com")
    sent: list[dict] = []
    out = check_and_alert(send=sent.append, now=datetime(2026, 10, 9, tzinfo=timezone.utc))
    kinds = {row["kind"] for row in out["alerts"]}
    assert "coverage" in kinds
    assert "mae" not in kinds


def test_drift_get_is_local_only(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("drift must not fetch weather")

    monkeypatch.setattr("energy_forecast.data.weather.fetch_forecast_weather", boom)
    monkeypatch.setattr("energy_forecast.data.weather.fetch_historical_weather", boom)
    monkeypatch.setattr("energy_forecast.data.elexon.fetch_indo", boom)
    response = client.get("/drift")
    assert response.status_code == 200
    body = response.json()
    assert "note" in body or "packs" in body
    assert drift_payload()["email_ready"] in {True, False}


def test_dashboard_has_pipeline_tab_not_reload():
    response = client.get("/")
    html = response.text
    assert "Data Pipeline and Maintenance" in html
    assert 'id="pipeline-view"' in html
    assert 'id="reload"' not in html
    assert 'id="capture"' not in html
    assert "Issue due presets" not in html
    assert "Reload board" not in html
    assert "/drift" in html
