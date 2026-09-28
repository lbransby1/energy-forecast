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
    payload = board_payload(refresh=False)
    assert "next30" in payload and "day" in payload and "week" in payload
    assert payload["next30"]["metrics"]["n"] == 0
