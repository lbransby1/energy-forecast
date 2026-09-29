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


def test_week_previous_is_last_calendar_week(monkeypatch):
    this_monday = pd.Timestamp("2026-09-27T23:00:00Z")
    last_monday = pd.Timestamp("2026-09-20T23:00:00Z")
    overlap = pd.date_range("2026-09-27T21:00:00Z", periods=12, freq="30min", tz="UTC")
    current = pd.date_range(this_monday, periods=6, freq="30min", tz="UTC")
    last_week = pd.date_range("2026-09-21T12:00:00Z", periods=4, freq="30min", tz="UTC")
    board = pd.DataFrame(
        {
            "preset": ["week"] * (len(overlap) + len(current)),
            "window_start": [last_monday] * len(overlap) + [this_monday] * len(current),
            "timestamp": list(overlap) + list(current),
            "issued_at": [pd.Timestamp("2026-09-27T21:00:00Z")] * len(overlap)
            + [this_monday] * len(current),
            "p10": [9000.0] * (len(overlap) + len(current)),
            "p50": [10000.0] * len(overlap) + [20000.0] * len(current),
            "p90": [11000.0] * (len(overlap) + len(current)),
        }
    )
    context = pd.DataFrame({"timestamp": last_week, "actual_mw": [24000.0] * len(last_week)})
    monkeypatch.setattr("energy_forecast.live.load_board", lambda: board)
    monkeypatch.setattr("energy_forecast.live.load_context_actuals", lambda: context)
    monkeypatch.setattr(
        "energy_forecast.live.london_now",
        lambda: pd.Timestamp("2026-09-29T10:00:00", tz="Europe/London"),
    )
    _BOARD_CACHE["payload"] = None
    payload = board_payload(refresh=False)
    prev_ts = [pd.Timestamp(row["timestamp"]) for row in payload["week"]["previous"]["rows"]]
    cur_ts = [pd.Timestamp(row["timestamp"]) for row in payload["week"]["current"]["rows"]]
    assert prev_ts
    assert max(prev_ts) < this_monday
    assert min(cur_ts) >= this_monday
    assert 20000.0 in [row["p50"] for row in payload["week"]["current"]["rows"]]
    assert 24000.0 in [row["actual_mw"] for row in payload["week"]["previous_actuals"]]
    assert len(payload["week"]["previous_actuals"]) == len(last_week)
    labels = prev_ts + cur_ts
    assert labels == sorted(labels)


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
    assert 24000.0 in week_mw
    assert 26000.0 in hist_mw
    assert 25000.0 in day_mw


def test_day_late_freeze_stops_at_midnight(monkeypatch):
    from energy_forecast.live import day_window, previous_london_midnight

    now = pd.Timestamp("2026-09-29T12:00:00", tz="Europe/London")
    yesterday = previous_london_midnight(now)
    today = day_window(now)
    late = pd.date_range("2026-09-28T08:00:00Z", periods=48, freq="30min", tz="UTC")
    current = pd.date_range(today, periods=6, freq="30min", tz="UTC")
    board = pd.DataFrame(
        {
            "preset": ["day"] * (len(late) + len(current)),
            "window_start": [yesterday] * len(late) + [today] * len(current),
            "timestamp": list(late) + list(current),
            "issued_at": [pd.Timestamp("2026-09-28T08:10:00Z")] * len(late) + [today] * len(current),
            "p10": [9000.0] * (len(late) + len(current)),
            "p50": [10000.0] * len(late) + [20000.0] * len(current),
            "p90": [11000.0] * (len(late) + len(current)),
            "actual_mw": [None] * (len(late) + len(current)),
        }
    )
    monkeypatch.setattr("energy_forecast.live.load_board", lambda: board)
    monkeypatch.setattr("energy_forecast.live.load_context_actuals", lambda: pd.DataFrame())
    monkeypatch.setattr("energy_forecast.live.london_now", lambda: now)
    _BOARD_CACHE["payload"] = None
    payload = board_payload(refresh=False)
    prev_ts = [pd.Timestamp(row["timestamp"]) for row in payload["day"]["previous"]["rows"]]
    cur_ts = [pd.Timestamp(row["timestamp"]) for row in payload["day"]["current"]["rows"]]
    assert prev_ts
    assert min(prev_ts) == pd.Timestamp("2026-09-28T08:00:00Z")
    assert max(prev_ts) < today
    assert min(cur_ts) >= today
    assert 20000.0 in [row["p50"] for row in payload["day"]["current"]["rows"]]
    assert pd.Timestamp(payload["day"]["boundary"]) == today


def test_complete_midnight_day_needs_every_half_hour():
    from energy_forecast.live import complete_midnight_day, london_day_range, previous_london_midnight

    now = pd.Timestamp("2026-09-28T12:00:00", tz="Europe/London")
    yesterday = previous_london_midnight(now)
    start, _end, n = london_day_range(yesterday)
    assert n == 48
    idx = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    full = pd.DataFrame({"timestamp": idx, "actual_mw": 20000.0, "p50": 20000.0})
    assert complete_midnight_day(yesterday, full, column="actual_mw", now=now)
    assert not complete_midnight_day(yesterday, full.iloc[:10], column="actual_mw", now=now)


def test_next30_history_skips_incomplete_yesterday():
    from energy_forecast.live import _next30_history

    now = pd.Timestamp("2026-09-29T08:00:00", tz="Europe/London")
    next30 = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-09-28T08:00:00Z", "2026-09-29T07:00:00Z"]),
            "issued_at": pd.to_datetime(["2026-09-28T08:10:00Z", "2026-09-29T07:10:00Z"]),
            "p50": [20000.0, 21000.0],
        }
    )
    live = next30.iloc[-1:]
    out = _next30_history(next30, live, now.tz_convert("UTC"), pd.DataFrame())
    stamps = set(pd.to_datetime(out["timestamp"], utc=True))
    assert pd.Timestamp("2026-09-28T08:00:00Z") not in stamps
    assert pd.Timestamp("2026-09-29T07:00:00Z") in stamps


def test_next30_history_includes_complete_yesterday():
    from energy_forecast.live import _next30_history, london_day_range, previous_london_midnight

    now = pd.Timestamp("2026-09-29T08:00:00", tz="Europe/London")
    yesterday = previous_london_midnight(now)
    start, _end, n = london_day_range(yesterday)
    idx = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    context = pd.DataFrame({"timestamp": idx, "actual_mw": 20000.0})
    next30 = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([idx[16], "2026-09-29T07:00:00Z"]),
            "issued_at": pd.to_datetime(["2026-09-28T08:10:00Z", "2026-09-29T07:10:00Z"]),
            "p50": [20000.0, 21000.0],
            "actual_mw": [20000.0, None],
        }
    )
    live = next30.iloc[-1:]
    out = _next30_history(next30, live, now.tz_convert("UTC"), context)
    stamps = set(pd.to_datetime(out["timestamp"], utc=True))
    assert idx[16] in stamps
    assert pd.Timestamp("2026-09-29T07:00:00Z") in stamps
    assert len(out) >= n


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


def test_width_averages_the_full_issued_fan():
    import numpy as np

    from energy_forecast.live import preset_metrics

    frame = pd.DataFrame(
        {
            "p10": [100.0, 100.0],
            "p50": [200.0, 200.0],
            "p90": [400.0, 300.0],
            "actual_mw": [200.0, np.nan],
        }
    )
    metrics = preset_metrics(frame, compute_wrmsse=False)
    assert metrics["n"] == 1
    assert metrics["interval_width"] == 250.0
