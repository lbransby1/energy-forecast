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
    assert "Grid Demand UK" in response.text
    assert "three&nbsp;views" in response.text
    assert "Typical size of the miss" in response.text
    assert "This chart is the week ahead" in response.text
    assert "Yesterday’s forecast vs actual" in response.text or "yesterday" in response.text.lower()
    assert "Yesterday forecast CSV" in response.text
    assert "How the guesses did" in response.text
    assert response.text.index('id="next30"') < response.text.index('id="n30-audit"')
    assert "Segoe UI" in response.text
    assert "Georgia" not in response.text
    assert "cdn.jsdelivr.net" not in response.text


def test_chart_js_is_local():
    response = client.get("/chart.js")
    assert response.status_code == 200
    assert len(response.content) > 50_000


def test_board_json_shape(monkeypatch):
    import pandas as pd

    monkeypatch.setattr("energy_forecast.live.load_board", lambda: pd.DataFrame())
    monkeypatch.setattr("energy_forecast.live.load_context_actuals", lambda: pd.DataFrame())
    monkeypatch.setattr("energy_forecast.live.fetch_indo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    from energy_forecast.live import _BOARD_CACHE

    _BOARD_CACHE["payload"] = None
    response = client.get("/board")
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"as_of", "next30", "day", "week", "eval"}
    assert "scored" in body["eval"]
