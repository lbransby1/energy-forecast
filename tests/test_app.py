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
    assert 'href="https://github.com/lbransby1/energy-forecast"' in response.text
    assert "Three&nbsp;views" in response.text
    assert "Typical size of the miss" in response.text
    assert "What the scores mean" in response.text
    assert 'class="shap-row"' in response.text
    assert "This chart is the last two days of last week" in response.text
    assert "Yesterday’s forecast vs actual" in response.text or "yesterday" in response.text.lower()
    assert "Yesterday forecast CSV" in response.text
    assert "How the guesses did" not in response.text
    assert "How this freeze did" in response.text
    assert "How the models were trained" in response.text
    assert "Model Architecture" in response.text
    assert 'id="week-eval-cards"' in response.text
    assert 'id="lab"' in response.text
    assert response.text.index('id="models-view"') < response.text.index('id="lab-packs"')
    assert "This freeze" in response.text
    assert "Holdout (last train)" in response.text
    assert 'id="week-holdout"' in response.text
    assert 'id="day-holdout"' in response.text
    assert 'id="n30-holdout"' in response.text
    assert "holdout_rmse" in response.text
    assert "holdout_wrmsse" in response.text
    assert "holdout_bias" in response.text
    assert "holdout_width" in response.text
    assert 'data-shift="week"' in response.text
    assert 'data-shift="day"' in response.text
    assert 'data-shift="next30"' in response.text
    assert response.text.index('<canvas id="week">') < response.text.index('data-shift="week"')
    assert ">Today</button>" in response.text
    assert 'data-dir="0"' in response.text
    assert "RMSE" in response.text
    assert "After the fact" not in response.text
    assert "Rebuild review" not in response.text
    assert "The project is complete" in response.text
    assert "Freeze cov" not in response.text
    assert "<th>Tuned</th>" not in response.text
    assert "WANDB_PROJECT_URL" not in response.text
    assert "Optuna notebook" not in response.text
    assert "lab-links" not in response.text
    assert "MLflow" not in response.text
    assert 'class="side"' not in response.text
    assert "details.fold" in response.text
    assert ">Scores</summary>" in response.text
    assert "INDO lag check" in response.text
    assert response.text.index('id="next30"') < response.text.index('id="n30-audit"')
    assert response.text.index(">Next 30 minutes</h2>") < response.text.index("/download/next30?which=current")
    assert response.text.index("/inputs/next30") < response.text.index('id="n30-note"')
    assert "Segoe UI" in response.text
    assert "Georgia" not in response.text
    assert "cdn.jsdelivr.net" not in response.text
    assert 'id="reload"' not in response.text
    assert 'id="capture"' not in response.text
    assert "Data Pipeline and Maintenance" in response.text
    assert "How the data moves" not in response.text
    assert "DRIFT_SMTP_HOST" not in response.text
    assert ".pipeline { width: 100%; max-width: none; }" in response.text
    assert 'id="pipeline-view"' in response.text
    assert "setInterval(load" not in response.text
    assert "waitForIndo" not in response.text
    assert '["03", "06", "09", "12", "15", "18", "21"]' in response.text
    assert 'return `${p.hour}:30`' not in response.text


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
    assert set(body) >= {"as_of", "next30", "day", "week", "eval", "lab"}
    assert "scored" in body["eval"]
    assert "archives" in body["day"]
    assert "archives" in body["week"]
    assert {row["id"] for row in body["lab"]["packs"]} == {"hour", "day", "week"}
    hour = next(row for row in body["lab"]["packs"] if row["id"] == "hour")
    assert "holdout_rmse" in hour
    assert "holdout_wrmsse" in hour
    assert "holdout_bias" in hour
    assert "holdout_width" in hour
