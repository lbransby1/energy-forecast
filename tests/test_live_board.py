from __future__ import annotations

import pandas as pd
import pytest

from energy_forecast.live import (
    NEXT30,
    WeekReissueLocked,
    _BOARD_CACHE,
    board_payload,
    capture_preset,
    week_password_ok,
)


def test_week_password_rejects_wrong_secret(monkeypatch):
    monkeypatch.setenv("WEEK_REISSUE_PASSWORD", "correct-horse")
    assert week_password_ok("correct-horse") is True
    assert week_password_ok("nope") is False
    assert week_password_ok("") is False
    assert week_password_ok(None) is False


def test_reissue_week_without_password_raises(monkeypatch):
    monkeypatch.setenv("WEEK_REISSUE_PASSWORD", "secret")
    board = pd.DataFrame(
        {
            "preset": ["week"],
            "window_start": [pd.Timestamp("2026-09-27T23:00:00Z")],
            "timestamp": [pd.Timestamp("2026-09-28T00:00:00Z")],
        }
    )
    monkeypatch.setattr("energy_forecast.live.load_board", lambda: board)
    monkeypatch.setattr("energy_forecast.live.week_window", lambda when=None: pd.Timestamp("2026-09-27T23:00:00Z"))
    with pytest.raises(WeekReissueLocked):
        capture_preset("week", force=True, week_password="wrong")


def test_board_payload_does_not_call_insights(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("fetch_indo must not run on GET /board")

    monkeypatch.setattr("energy_forecast.live.fetch_indo", boom)
    _BOARD_CACHE["payload"] = None
    monkeypatch.setattr("energy_forecast.live.load_board", lambda: pd.DataFrame())
    monkeypatch.setattr("energy_forecast.live.load_context_actuals", lambda: pd.DataFrame())
    payload = board_payload(refresh=False)
    assert "next30" in payload and "day" in payload and "week" in payload
    assert payload["eval"]["scored"] == 0


def test_board_uses_context_indo_without_prior_freeze(monkeypatch):
    context = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-09-27T12:00:00Z", "2026-09-21T12:00:00Z", "2026-09-28T10:00:00Z"]
            ),
            "actual_mw": [25000.0, 24000.0, 26000.0],
        }
    )
    monkeypatch.setattr("energy_forecast.live.load_board", lambda: pd.DataFrame())
    monkeypatch.setattr("energy_forecast.live.load_context_actuals", lambda: context)
    monkeypatch.setattr(
        "energy_forecast.live.london_now",
        lambda: pd.Timestamp("2026-09-28T12:00:00", tz="Europe/London"),
    )
    _BOARD_CACHE["payload"] = None
    payload = board_payload(refresh=False)
    day_mw = [row["actual_mw"] for row in payload["day"]["previous_actuals"]]
    week_mw = [row["actual_mw"] for row in payload["week"]["previous_actuals"]]
    hist_mw = [row["actual_mw"] for row in payload["next30"]["history"]]
    assert 25000.0 in day_mw
    assert 24000.0 in week_mw
    assert 26000.0 in hist_mw


def test_evaluation_tags_outside_band_and_lags():
    from energy_forecast.live import evaluation_payload

    board = pd.DataFrame(
        {
            "preset": ["day", "next30"],
            "timestamp": pd.to_datetime(["2026-09-28T10:00:00Z", "2026-09-28T10:30:00Z"]),
            "p10": [24000.0, 24000.0],
            "p50": [25000.0, 25000.0],
            "p90": [26000.0, 26000.0],
            "actual_mw": [28000.0, 25100.0],
            "demand_lag_1": [20000.0, 25100.0],
            "last_indo_mw": [25000.0, 25100.0],
        }
    )
    out = evaluation_payload(board, pd.DataFrame())
    assert out["scored"] == 2
    assert out["outside_band"] == 1
    assert out["lags"] == 1
