from __future__ import annotations

from fastapi.testclient import TestClient

from energy_forecast.app import app

client = TestClient(app)


def test_health_is_fast_and_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_dashboard_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "GB National Demand" in response.text


def test_board_json_shape(monkeypatch):
    import pandas as pd

    monkeypatch.setattr("energy_forecast.live.load_board", lambda: pd.DataFrame())
    monkeypatch.setattr("energy_forecast.live.fetch_indo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    from energy_forecast.live import _BOARD_CACHE

    _BOARD_CACHE["payload"] = None
    response = client.get("/board")
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"as_of", "next30", "day", "week"}
