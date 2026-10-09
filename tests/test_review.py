from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from energy_forecast.app import app

client = TestClient(app)


def test_notebook_review_helpers_import():
    from energy_forecast.review import (
        _attach_frozen_and_weather,
        _ensure_calendar,
        _load_archive_weather,
        _pack_rows,
        _ready_predict_frame,
        _recompute_weather_features,
        week_review_span,
    )

    assert callable(_pack_rows)
    assert callable(week_review_span)
    assert callable(_attach_frozen_and_weather)
    assert callable(_ensure_calendar)
    assert callable(_load_archive_weather)
    assert callable(_ready_predict_frame)
    assert callable(_recompute_weather_features)


def test_dashboard_has_models_tab_not_review():
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "Model Architecture" in html
    assert "What the scores mean" in html
    assert 'class="shap-row"' in html
    assert "How the models were trained" in html
    assert "How the guesses did" not in html
    assert "How this freeze did" in html
    assert "INDO lag check" in html
    assert ">Scores</summary>" in html
    assert "Holdout (last train)" in html
    assert 'id="models-view"' in html
    assert 'id="pipeline-view"' in html
    assert 'id="live-view"' in html
    assert 'id="reload"' not in html
    assert "After the fact" not in html
    assert "Rebuild review" not in html
    assert 'id="review-view"' not in html
    assert "/shap" in html
    assert "/drift" in html
    assert "the project is complete" in html.lower() or "This site is <b>finished</b>" in html


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


def test_shap_get_is_local_only(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("shap must not fetch weather")

    monkeypatch.setattr("energy_forecast.data.weather.fetch_forecast_weather", boom)
    monkeypatch.setattr("energy_forecast.data.weather.fetch_historical_weather", boom)
    monkeypatch.setattr("energy_forecast.data.elexon.fetch_indo", boom)
    response = client.get("/shap")
    assert response.status_code == 200
    body = response.json()
    assert "packs" in body
    ids = {row["id"] for row in body["packs"]}
    assert ids == {"hour", "day", "week"}


def test_extend_cached_archive_appends_without_wiping(tmp_path, monkeypatch):
    from energy_forecast.data import weather as weather_mod

    idx = pd.date_range("2026-09-24", periods=4, freq="h", tz="Europe/London")
    cached = pd.DataFrame(
        {
            "region_id": "gb",
            "timestamp": idx,
            "settlement_date": pd.to_datetime(idx.date),
            "settlement_period": 1,
            "temperature_2m": [10.0, 11.0, 12.0, 13.0],
        }
    )
    path = tmp_path / "weather_gb_archive.parquet"
    cached.to_parquet(path, index=False)
    extra = pd.DataFrame(
        {
            "region_id": "gb",
            "timestamp": pd.date_range("2026-09-26", periods=2, freq="h", tz="Europe/London"),
            "settlement_date": pd.to_datetime(["2026-09-26", "2026-09-26"]),
            "settlement_period": 1,
            "temperature_2m": [14.0, 15.0],
        }
    )
    monkeypatch.setattr(weather_mod, "_fetch_archive_window", lambda *a, **k: extra)
    out = weather_mod.extend_cached_archive(["gb"], end_date="2026-09-26", processed_dir=tmp_path)
    assert out["settlement_date"].min() == pd.Timestamp("2026-09-24")
    assert out["settlement_date"].max() == pd.Timestamp("2026-09-26")
    assert len(out) == 6
    saved = pd.read_parquet(path)
    assert len(saved) == 6
