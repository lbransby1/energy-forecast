"""Capture live forecasts and score them against Insights INDO as it arrives."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import time
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd

from energy_forecast.data.elexon import fetch_indo
from energy_forecast.evaluate import _naive_scale_mse, _period_band, _wrmsse
from energy_forecast.features import HOUR_FEATURE_COLUMNS
from energy_forecast.forecast import (
    LIVE_INDO_MAX_AGE_MINUTES,
    attach_indo_actuals,
    run_forecast,
    run_pack,
    validation_metrics,
)
from energy_forecast.modes import MEDIUM, SHORT
from energy_forecast.paths import DATA_LIVE, ensure_data_dirs
from energy_forecast.settlement import LONDON_TZ

ARCHIVE_NAME = "archive.parquet"
BOARD_NAME = "board.parquet"
CONTEXT_NAME = "context_actuals.parquet"
NEXT30 = "next30"
DAY = "day"
WEEK = "week"
PRESETS = (NEXT30, DAY, WEEK)
WEEK_PASSWORD_FILE = "week_reissue_password"
_BOARD_CACHE: dict = {"at": 0.0, "payload": None}
_BOARD_CACHE_S = 2.0
_TICK_LOCK = threading.RLock()


class WeekReissueLocked(PermissionError):
    """Raised when someone tries to overwrite this week's freeze without the password."""

KEEP = (
    "preset",
    "window_start",
    "issued_at",
    "mode",
    "timestamp",
    "settlement_date",
    "settlement_period",
    "lead_hours",
    "horizon",
    "lead_band",
    "p10",
    "p50",
    "p90",
    "interval_width",
    "actual_mw",
    "temperature_2m",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_48",
    "demand_lag_336",
    "last_indo_timestamp",
    "last_indo_mw",
)


def archive_path():
    ensure_data_dirs()
    return DATA_LIVE / ARCHIVE_NAME


def board_path():
    ensure_data_dirs()
    return DATA_LIVE / BOARD_NAME


def context_path():
    ensure_data_dirs()
    return DATA_LIVE / CONTEXT_NAME


def load_context_actuals() -> pd.DataFrame:
    """Published INDO for chart context (yesterday / last week / last 6h). Local file only."""
    path = context_path()
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "actual_mw", "settlement_period"])
    out = pd.read_parquet(path)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "actual_mw" not in out.columns and "demand_mw" in out.columns:
        out = out.rename(columns={"demand_mw": "actual_mw"})
    return out


def save_context_actuals(frame: pd.DataFrame) -> None:
    keep = [column for column in ("timestamp", "actual_mw", "settlement_period", "settlement_date") if column in frame.columns]
    frame.loc[:, keep].sort_values("timestamp").reset_index(drop=True).to_parquet(context_path(), index=False)


def refresh_context_actuals() -> dict:
    """Fetch last week of INDO so charts have history without a previous freeze on disk."""
    now = london_now()
    start = week_window(now) - pd.Timedelta(days=7)
    start_date = start.tz_convert(LONDON_TZ).strftime("%Y-%m-%d")
    try:
        indo = fetch_indo(start_date, date.today().isoformat())
    except Exception as exc:  # noqa: BLE001 — keep the previous context file
        print(f"context INDO fetch failed: {exc}")
        return {"rows": int(len(load_context_actuals())), "error": str(exc)}
    if indo.empty:
        return {"rows": 0}
    out = indo.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "demand_mw" in out.columns:
        out = out.rename(columns={"demand_mw": "actual_mw"})
    save_context_actuals(out)
    _BOARD_CACHE["payload"] = None
    return {"rows": int(len(out))}


def _slice_actuals(context: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if context.empty or "timestamp" not in context.columns:
        return context.iloc[0:0].copy() if hasattr(context, "iloc") else pd.DataFrame()
    ts = pd.to_datetime(context["timestamp"], utc=True)
    mask = (ts >= _as_utc(start)) & (ts < _as_utc(end))
    return context.loc[mask].copy()


def _overlay_actuals(rows: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Stamp published INDO onto forecast rows for display (GET stays local)."""
    if rows.empty or context.empty or "timestamp" not in context.columns:
        return rows
    out = rows.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    lookup = (
        context.dropna(subset=["actual_mw"])
        .drop_duplicates("timestamp")
        .set_index("timestamp")["actual_mw"]
    )
    mapped = out["timestamp"].map(lookup)
    if "actual_mw" not in out.columns:
        out["actual_mw"] = mapped
    else:
        out["actual_mw"] = pd.to_numeric(out["actual_mw"], errors="coerce").combine_first(mapped)
    return out


def _pad_actual_slots(
    rows: pd.DataFrame,
    context: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Keep the black INDO line continuous even when a freeze starts late (e.g. 9am)."""
    trail = _slice_actuals(context, start, end) if context is not None else pd.DataFrame()
    work = rows.copy() if rows is not None and not rows.empty else pd.DataFrame()
    if work.empty and trail.empty:
        return work
    if not work.empty:
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        work = _overlay_actuals(work, context) if context is not None else work
    if trail.empty:
        return work.sort_values("timestamp") if not work.empty else work
    trail = trail.copy()
    trail["timestamp"] = pd.to_datetime(trail["timestamp"], utc=True)
    have = set(work["timestamp"]) if not work.empty and "timestamp" in work.columns else set()
    extra = trail.loc[~trail["timestamp"].isin(have)].copy()
    if extra.empty:
        return work.sort_values("timestamp") if not work.empty else work
    for column in ("p10", "p50", "p90"):
        extra[column] = np.nan
    combined = extra if work.empty else pd.concat([work, extra], ignore_index=True)
    return combined.sort_values("timestamp")


def _align_next30_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Old hour packs targeted the following SP (issued_at + 30min). Plot that guess on the current SP."""
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return frame
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "issued_at" not in out.columns:
        return out
    issued = pd.to_datetime(out["issued_at"], utc=True)
    shifted = (out["timestamp"] - issued) == pd.Timedelta(minutes=30)
    out.loc[shifted, "timestamp"] = issued.loc[shifted]
    return out


def frozen_dir():
    ensure_data_dirs()
    path = DATA_LIVE / "frozen"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_parquet(path, columns=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns or KEEP))
    out = pd.read_parquet(path)
    for column in ("issued_at", "timestamp", "window_start", "last_indo_timestamp"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True)
    if "settlement_date" in out.columns:
        out["settlement_date"] = pd.to_datetime(out["settlement_date"]).dt.tz_localize(None).dt.normalize()
    return out


def load_archive() -> pd.DataFrame:
    return _read_parquet(archive_path())


def save_archive(frame: pd.DataFrame) -> None:
    keep = [column for column in KEEP if column in frame.columns]
    sort = [column for column in ("issued_at", "mode", "timestamp") if column in frame.columns]
    out = frame.loc[:, keep]
    if sort:
        out = out.sort_values(sort)
    out.reset_index(drop=True).to_parquet(archive_path(), index=False)


def load_board() -> pd.DataFrame:
    return _read_parquet(board_path())


def save_board(frame: pd.DataFrame) -> None:
    keep = [column for column in KEEP if column in frame.columns]
    out = frame.loc[:, keep].sort_values(["preset", "window_start", "timestamp"]).reset_index(drop=True)
    out.to_parquet(board_path(), index=False)


def london_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=LONDON_TZ)


def day_window(when: pd.Timestamp | None = None) -> pd.Timestamp:
    stamp = (when or london_now()).tz_convert(LONDON_TZ)
    return stamp.normalize().tz_convert("UTC")


def week_window(when: pd.Timestamp | None = None) -> pd.Timestamp:
    stamp = (when or london_now()).tz_convert(LONDON_TZ)
    monday = stamp.normalize() - pd.Timedelta(days=int(stamp.dayofweek))
    return monday.tz_convert("UTC")


def previous_week_window(when: pd.Timestamp | None = None) -> pd.Timestamp:
    """Last Monday 00:00 Europe/London, as UTC."""
    return week_window(when) - pd.Timedelta(days=7)


def week_previous_chart_start(when: pd.Timestamp | None = None) -> pd.Timestamp:
    """Saturday 00:00 Europe/London of the week just ended (last two days before this Monday)."""
    monday = week_window(when).tz_convert(LONDON_TZ)
    return (monday - pd.Timedelta(days=2)).normalize().tz_convert("UTC")


def _as_utc(stamp: pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(stamp)
    return out.tz_convert("UTC") if out.tzinfo else out.tz_localize("UTC")


def previous_london_midnight(when: pd.Timestamp | None = None) -> pd.Timestamp:
    """Yesterday 00:00 Europe/London, as UTC."""
    today_local = day_window(when).tz_convert(LONDON_TZ)
    return (today_local - pd.Timedelta(days=1)).normalize().tz_convert("UTC")


def london_day_range(start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """UTC [start, end) and expected half-hours for a London calendar day starting at midnight."""
    start = _as_utc(start)
    midnight = start.tz_convert(LONDON_TZ).normalize()
    if start.tz_convert(LONDON_TZ) != midnight:
        return start, start, 0
    end_local = midnight + pd.Timedelta(days=1)
    expected = len(pd.date_range(midnight, end_local, freq="30min", inclusive="left"))
    return midnight.tz_convert("UTC"), end_local.tz_convert("UTC"), int(expected)


def complete_midnight_day(
    start: pd.Timestamp,
    frame: pd.DataFrame,
    *,
    column: str,
    now: pd.Timestamp | None = None,
) -> bool:
    """True when `start` is London midnight, that day has ended, and every SP has `column`."""
    start_utc, end_utc, expected = london_day_range(start)
    if expected <= 0:
        return False
    now = now or london_now()
    if _as_utc(now) < end_utc:
        return False
    if frame is None or frame.empty or "timestamp" not in frame.columns or column not in frame.columns:
        return False
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    slice_ = work.loc[(work["timestamp"] >= start_utc) & (work["timestamp"] < end_utc)].dropna(subset=[column])
    return int(slice_["timestamp"].drop_duplicates().shape[0]) == expected


def _frozen_path(preset: str, window_start: pd.Timestamp):
    key = _as_utc(window_start).strftime("%Y%m%dT%H%M")
    return frozen_dir() / f"{preset}_{key}.parquet"


def save_frozen(preset: str, window_start: pd.Timestamp, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    out = frame.copy()
    out["preset"] = preset
    out["window_start"] = _as_utc(window_start)
    out.to_parquet(_frozen_path(preset, window_start), index=False)
    if preset == NEXT30:
        _prune_next30_frozen()


def _prune_next30_frozen(*, keep_hours: int = 48) -> None:
    """Keep a rolling window of 30-minute input snapshots."""
    cutoff = london_now().tz_convert("UTC") - pd.Timedelta(hours=keep_hours)
    for path in frozen_dir().glob("next30_*.parquet"):
        try:
            stamp = pd.to_datetime(path.stem.replace("next30_", ""), format="%Y%m%dT%H%M", utc=True)
        except ValueError:
            continue
        if stamp < cutoff:
            path.unlink(missing_ok=True)


def load_frozen(preset: str, window_start: pd.Timestamp) -> pd.DataFrame:
    path = _frozen_path(preset, window_start)
    return _read_parquet(path, columns=None) if path.exists() else pd.DataFrame()


def capture(*, mode: str = "both") -> dict:
    """Legacy hourly short+medium capture (kept for the old archive)."""
    issued_at = london_now().floor("1h").tz_convert("UTC")
    raw = run_forecast(mode="both" if mode == "both" else mode, freq="30min")
    frames = raw.values() if isinstance(raw, dict) else [raw]
    pieces: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        piece = frame.copy()
        piece["issued_at"] = issued_at
        if "mode" not in piece.columns:
            piece["mode"] = SHORT
        counts[str(piece["mode"].iloc[0])] = int(len(piece))
        pieces.append(piece)
    if not pieces:
        raise RuntimeError("run_forecast returned no rows")
    new = pd.concat(pieces, ignore_index=True)
    archive = load_archive()
    if not archive.empty:
        archive = archive.loc[
            ~((archive["issued_at"] == issued_at) & (archive["mode"].isin(new["mode"].unique())))
        ]
    save_archive(pd.concat([archive, new], ignore_index=True))
    return {
        "issued_at": issued_at.isoformat(),
        "rows": {name: counts.get(name, 0) for name in (SHORT, MEDIUM)},
        "archive_rows": int(len(load_archive())),
    }


def week_reissue_secret() -> str:
    """Password that overwrites an existing week freeze.

    ``WEEK_REISSUE_PASSWORD`` or ``LIVE_TOKEN`` wins; otherwise the gitignored
    file under ``data/live/``.
    """
    env = (os.environ.get("WEEK_REISSUE_PASSWORD") or os.environ.get("LIVE_TOKEN") or "").strip()
    if env:
        return env
    ensure_data_dirs()
    path = DATA_LIVE / WEEK_PASSWORD_FILE
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def week_password_ok(given: str | None) -> bool:
    expected = week_reissue_secret()
    if not expected or not given:
        return False
    left = given.encode("utf-8")
    right = expected.encode("utf-8")
    if len(left) != len(right):
        return False
    return hmac.compare_digest(left, right)


def _run_pack_retry(
    role: str,
    hours: float,
    *,
    retry: bool,
    origin: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hour pack: retry once after 20s if Insights INDO is still catching up."""
    try:
        return run_pack(role=role, horizon_hours=hours, origin=origin)
    except RuntimeError:
        if not retry:
            raise
        time.sleep(20)
        return run_pack(role=role, horizon_hours=hours, origin=origin)


def capture_preset(preset: str, *, force: bool = False, week_password: str | None = None) -> dict:
    """Issue one board preset. Day/week skip if that window already exists."""
    with _TICK_LOCK:
        result = _capture_preset(preset, force=force, week_password=week_password)
        result["context"] = refresh_context_actuals()
        result["actuals"] = refresh_actuals()
        return result


def _capture_preset(preset: str, *, force: bool = False, week_password: str | None = None) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    now = london_now()
    issued_at = now.floor("30min").tz_convert("UTC")
    if preset == NEXT30:
        window_start = issued_at
        role, hours = "hour", 1.0
    elif preset == DAY:
        window_start = day_window(now)
        role, hours = "day", 24.0
    else:
        window_start = week_window(now)
        role, hours = "week", 168.0

    board = load_board()
    have_window = (
        not board.empty
        and ((board["preset"] == preset) & (board["window_start"] == window_start)).any()
    )
    missing_frozen = not _frozen_path(preset, window_start).exists()
    if not force and preset != NEXT30 and have_window and not missing_frozen:
        return {"preset": preset, "skipped": True, "window_start": window_start.isoformat()}
    if preset == WEEK and have_window:
        if not week_password_ok(week_password):
            raise WeekReissueLocked("week freeze is locked; password required to re-issue")

    origin = None if preset == NEXT30 else window_start
    try:
        forecast, frozen = _run_pack_retry(role, hours, retry=(preset == NEXT30), origin=origin)
    except RuntimeError as exc:
        if preset != NEXT30:
            raise
        return {
            "preset": preset,
            "skipped": True,
            "window_start": window_start.isoformat(),
            "reason": str(exc),
        }
    piece = forecast.copy()
    piece["preset"] = preset
    piece["window_start"] = window_start
    piece["issued_at"] = issued_at
    if "mode" not in piece.columns:
        piece["mode"] = preset
    if not board.empty:
        if preset == NEXT30:
            board = board.loc[~((board["preset"] == NEXT30) & (board["issued_at"] == issued_at))]
        else:
            board = board.loc[~((board["preset"] == preset) & (board["window_start"] == window_start))]
    combined = piece if board.empty else pd.concat([board, piece], ignore_index=True)
    save_board(combined)
    save_frozen(preset, window_start, frozen)
    _BOARD_CACHE["payload"] = None
    return {
        "preset": preset,
        "issued_at": issued_at.isoformat(),
        "window_start": window_start.isoformat(),
        "rows": int(len(piece)),
        "board_rows": int(len(load_board())),
        "inputs": input_health(preset, window_start, issued_at=issued_at),
    }


def import_legacy_next30() -> int:
    """Copy the old hourly-archive shorts onto the next-30 board (once per timestamp)."""
    legacy = scored_frame()
    if legacy.empty or "mode" not in legacy.columns:
        return 0
    shorts = legacy.loc[legacy["mode"] == "short"].copy()
    if shorts.empty:
        return 0
    board = load_board()
    have = set()
    if not board.empty and "preset" in board.columns:
        have = set(pd.to_datetime(board.loc[board["preset"] == NEXT30, "timestamp"], utc=True))
    shorts["timestamp"] = pd.to_datetime(shorts["timestamp"], utc=True)
    add = shorts.loc[~shorts["timestamp"].isin(have)].copy()
    if add.empty:
        return 0
    add["preset"] = NEXT30
    add["issued_at"] = pd.to_datetime(add["issued_at"], utc=True)
    add["window_start"] = add["issued_at"]
    if "mode" not in add.columns:
        add["mode"] = NEXT30
    combined = add if board.empty else pd.concat([board, add], ignore_index=True)
    save_board(combined)
    return int(len(add))


def tick_presets(*, force: bool = False, week_password: str | None = None) -> dict:
    """Issue the 30-minute call, and the day/week freeze if that window is empty."""
    with _TICK_LOCK:
        imported = import_legacy_next30()
        results = {"imported_next30": imported, "next30": _capture_preset(NEXT30, force=True)}
        results["day"] = _capture_preset(DAY, force=force)
        results["week"] = _capture_preset(WEEK, force=force, week_password=week_password)
        results["context"] = refresh_context_actuals()
        results["actuals"] = refresh_actuals()
        results["wrmsse"] = refresh_wrmsse_cache()
        return results


def refresh_actuals() -> dict:
    """Join INDO onto elapsed board and legacy archive rows."""
    updated = {"board": 0, "archive": 0}
    board = load_board()
    if not board.empty:
        before = int(board["actual_mw"].notna().sum()) if "actual_mw" in board.columns else 0
        joined = attach_indo_actuals(board)
        save_board(joined)
        updated["board"] = int(joined["actual_mw"].notna().sum()) - before
    archive = load_archive()
    if not archive.empty:
        before = int(archive["actual_mw"].notna().sum()) if "actual_mw" in archive.columns else 0
        joined = attach_indo_actuals(archive)
        if "actual_mw" in archive.columns:
            joined["actual_mw"] = joined["actual_mw"].combine_first(archive["actual_mw"])
        save_archive(joined)
        updated["archive"] = int(joined["actual_mw"].notna().sum()) - before
    _BOARD_CACHE["payload"] = None
    return updated


def _wrmsse_cache_path():
    ensure_data_dirs()
    return DATA_LIVE / "wrmsse_cache.json"


def load_wrmsse_cache() -> dict:
    path = _wrmsse_cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def refresh_wrmsse_cache() -> dict:
    """Recompute WRMSSE from local board + INDO. Called from the capture loop, not GET /board."""
    cache: dict[str, float | None] = {}
    try:
        indo = fetch_indo(
            (pd.Timestamp.now(tz=LONDON_TZ) - pd.Timedelta(days=56)).strftime("%Y-%m-%d"),
            date.today().isoformat(),
        )
        scale = _naive_scale_mse(indo, "30min", 336)
    except Exception:  # noqa: BLE001
        return load_wrmsse_cache()
    if not np.isfinite(scale) or scale <= 0:
        return load_wrmsse_cache()
    board = load_board()
    for preset in PRESETS:
        rows = _preset_frame(preset, board)
        metrics = preset_metrics(rows, compute_wrmsse=False)
        scored = rows.dropna(subset=["actual_mw", "p50"]) if not rows.empty else rows
        wrmsse = None
        if scored is not None and not scored.empty:
            work = scored.rename(columns={"actual_mw": "demand_mw"}).copy()
            if "settlement_period" in work.columns:
                work["period_band"] = work["settlement_period"].map(_period_band)
            try:
                value = _wrmsse(work, scale)
                wrmsse = float(value) if np.isfinite(value) else None
            except Exception:  # noqa: BLE001
                wrmsse = None
        cache[preset] = wrmsse
        cache[f"{preset}_n"] = metrics["n"]
    _wrmsse_cache_path().write_text(json.dumps(cache), encoding="utf-8")
    return cache


def _preset_frame(preset: str, board: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = board if board is not None else load_board()
    if frame.empty or "preset" not in frame.columns:
        return frame
    return frame.loc[frame["preset"] == preset].copy()


def _windows(preset: str, board: pd.DataFrame | None = None) -> list[pd.Timestamp]:
    frame = _preset_frame(preset, board)
    if frame.empty or "window_start" not in frame.columns:
        return []
    return sorted(pd.to_datetime(frame["window_start"].unique()), reverse=True)


def _window_rows(preset: str, window_start: pd.Timestamp | None, board: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _preset_frame(preset, board)
    if frame.empty or window_start is None:
        return frame.iloc[0:0].copy()
    start = _as_utc(window_start)
    return frame.loc[frame["window_start"] == start].sort_values("timestamp")


def _in_time_range(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame() if frame is None else frame.iloc[0:0].copy()
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    start_u, end_u = _as_utc(start), _as_utc(end)
    return work.loc[(work["timestamp"] >= start_u) & (work["timestamp"] < end_u)].copy()


def _span_rows(preset: str, start: pd.Timestamp, end: pd.Timestamp, board: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast rows whose target timestamps fall in [start, end), preferring that Monday's freeze."""
    frame = _in_time_range(_preset_frame(preset, board), start, end)
    if frame.empty:
        return frame
    start_u = _as_utc(start)
    if "window_start" in frame.columns:
        native = frame.loc[pd.to_datetime(frame["window_start"], utc=True) == start_u]
        if not native.empty and "p50" in native.columns and native["p50"].notna().any():
            return native.sort_values("timestamp")
    if "issued_at" in frame.columns:
        frame = frame.sort_values(["timestamp", "issued_at"]).drop_duplicates("timestamp", keep="last")
    return frame.sort_values("timestamp")


def scored_preset(preset: str, window_start: pd.Timestamp | None = None) -> pd.DataFrame:
    frame = _window_rows(preset, window_start) if window_start is not None else _preset_frame(preset)
    if frame.empty or "actual_mw" not in frame.columns:
        return frame
    return frame.dropna(subset=["actual_mw", "p50"]).copy()


def preset_metrics(frame: pd.DataFrame, *, compute_wrmsse: bool = True) -> dict:
    empty = {
        "n": 0,
        "mae": None,
        "rmse": None,
        "bias": None,
        "coverage_80": None,
        "interval_width": None,
        "wrmsse": None,
    }
    if frame is None or frame.empty:
        return empty
    width = None
    if {"p10", "p90"}.issubset(frame.columns):
        band = (
            pd.to_numeric(frame["p90"], errors="coerce") - pd.to_numeric(frame["p10"], errors="coerce")
        ).dropna()
        if len(band):
            width = float(band.mean())
    if "actual_mw" not in frame.columns or "p50" not in frame.columns:
        return {**empty, "interval_width": width}
    scored = frame.dropna(subset=["actual_mw", "p50"])
    if scored is None or scored.empty:
        return {**empty, "interval_width": width}
    error = scored["actual_mw"] - scored["p50"]
    covered = (
        ((scored["actual_mw"] >= scored["p10"]) & (scored["actual_mw"] <= scored["p90"])).mean()
        if {"p10", "p90"}.issubset(scored.columns)
        else np.nan
    )
    out = {
        "n": int(len(scored)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error.to_numpy() ** 2))),
        "bias": float(error.mean()),
        "coverage_80": float(covered) if pd.notna(covered) else None,
        "interval_width": width,
        "wrmsse": None,
    }
    if not compute_wrmsse:
        return out
    work = scored.rename(columns={"actual_mw": "demand_mw"}).copy()
    if "settlement_period" in work.columns:
        work["period_band"] = work["settlement_period"].map(_period_band)
    try:
        indo = fetch_indo(
            (pd.Timestamp.now(tz=LONDON_TZ) - pd.Timedelta(days=56)).strftime("%Y-%m-%d"),
            date.today().isoformat(),
        )
        scale = _naive_scale_mse(indo, "30min", 336)
        if np.isfinite(scale) and scale > 0:
            out["wrmsse"] = float(_wrmsse(work, scale))
    except Exception:  # noqa: BLE001 — live board must still render
        out["wrmsse"] = None
    return out


def _issue_stamp(frame: pd.DataFrame) -> str | None:
    if frame.empty or "issued_at" not in frame.columns:
        return None
    return pd.Timestamp(frame["issued_at"].iloc[0]).isoformat()


def _actuals_before(forecast: pd.DataFrame, *, hours: int) -> list[dict]:
    """INDO covering the same length as the freeze, ending at its first timestamp."""
    if forecast is None or forecast.empty or "timestamp" not in forecast.columns:
        return []
    end = _as_utc(forecast["timestamp"].min())
    return _indo_records(end - pd.Timedelta(hours=hours), end)


def _indo_records(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Half-hourly INDO between start (inclusive) and end (exclusive), UTC."""
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return []
    begin = start.tz_convert(LONDON_TZ).strftime("%Y-%m-%d")
    last = (end.tz_convert(LONDON_TZ) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d")
    try:
        indo = fetch_indo(begin, last)
    except Exception:  # noqa: BLE001
        return []
    if indo.empty or "timestamp" not in indo.columns:
        return []
    indo = indo.copy()
    indo["timestamp"] = pd.to_datetime(indo["timestamp"], utc=True)
    indo = indo.loc[(indo["timestamp"] >= start) & (indo["timestamp"] < end)]
    if indo.empty:
        return []
    out = pd.DataFrame(
        {
            "timestamp": indo["timestamp"],
            "actual_mw": indo["demand_mw"],
            "settlement_period": indo["settlement_period"] if "settlement_period" in indo.columns else None,
        }
    )
    return _frame_records(out)


MAX_INDO_AGE_MINUTES = LIVE_INDO_MAX_AGE_MINUTES
LAG_MATCH_MW = 25.0


def _iso_stamp(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat()


def input_health(
    preset: str,
    window_start: pd.Timestamp | None,
    *,
    issued_at: pd.Timestamp | None = None,
    rows: pd.DataFrame | None = None,
) -> dict:
    """Check frozen (or board) features: last INDO, short lags, missing hour columns."""
    frozen = load_frozen(preset, window_start) if window_start is not None else pd.DataFrame()
    source = frozen if not frozen.empty else (rows if rows is not None else pd.DataFrame())
    empty = {
        "ok": False,
        "issues": ["no frozen inputs yet — the next issue will snapshot features"],
        "has_frozen": False,
        "last_indo_timestamp": None,
        "last_indo_mw": None,
        "demand_lag_1": None,
        "demand_lag_2": None,
        "demand_lag_48": None,
        "demand_lag_336": None,
        "indo_age_minutes": None,
        "temperature_2m": None,
        "missing_features": [],
        "n_feature_rows": 0,
    }
    if source is None or source.empty:
        return empty
    row = source.iloc[0]
    last_indo_ts = row["last_indo_timestamp"] if "last_indo_timestamp" in source.columns else pd.NaT
    last_indo_mw = row["last_indo_mw"] if "last_indo_mw" in source.columns else np.nan
    lag1 = row["demand_lag_1"] if "demand_lag_1" in source.columns else np.nan
    lag2 = row["demand_lag_2"] if "demand_lag_2" in source.columns else np.nan
    lag48 = row["demand_lag_48"] if "demand_lag_48" in source.columns else np.nan
    lag336 = row["demand_lag_336"] if "demand_lag_336" in source.columns else np.nan
    temp = row["temperature_2m"] if "temperature_2m" in source.columns else np.nan
    if issued_at is None and rows is not None and not rows.empty and "issued_at" in rows.columns:
        issued_at = rows["issued_at"].iloc[0]
    age = None
    if issued_at is not None and pd.notna(last_indo_ts):
        age = (_as_utc(issued_at) - _as_utc(last_indo_ts)).total_seconds() / 60.0
    issues: list[str] = []
    missing: list[str] = []
    if frozen.empty:
        issues.append("frozen snapshot not stored for this window")
    if pd.isna(lag1) or pd.isna(lag2):
        issues.append("demand_lag_1/2 missing — hour model will not see recent INDO")
    elif pd.notna(last_indo_mw) and abs(float(lag1) - float(last_indo_mw)) > LAG_MATCH_MW:
        issues.append(
            f"lag_1 is {float(lag1):.0f} MW but last INDO is {float(last_indo_mw):.0f} MW"
        )
    if age is not None and age > MAX_INDO_AGE_MINUTES:
        issues.append(f"last INDO is {age:.0f} min old at issue — 30-minute call will lag the ramp")
    if pd.isna(last_indo_mw):
        issues.append("no last INDO on the live frame")
    if pd.isna(lag48) or pd.isna(lag336):
        issues.append("day/week lags missing")
    if preset == NEXT30 and not frozen.empty:
        for column in HOUR_FEATURE_COLUMNS:
            if column not in frozen.columns:
                missing.append(column)
            elif frozen[column].isna().any():
                missing.append(column)
        if missing:
            issues.append("missing/NaN hour features: " + ", ".join(missing[:8]))
    return {
        "ok": not issues,
        "issues": issues,
        "has_frozen": not frozen.empty,
        "last_indo_timestamp": _iso_stamp(last_indo_ts),
        "last_indo_mw": None if pd.isna(last_indo_mw) else float(last_indo_mw),
        "demand_lag_1": None if pd.isna(lag1) else float(lag1),
        "demand_lag_2": None if pd.isna(lag2) else float(lag2),
        "demand_lag_48": None if pd.isna(lag48) else float(lag48),
        "demand_lag_336": None if pd.isna(lag336) else float(lag336),
        "indo_age_minutes": None if age is None else float(age),
        "temperature_2m": None if pd.isna(temp) else float(temp),
        "missing_features": missing,
        "n_feature_rows": int(len(frozen)),
    }


def next30_input_audit(frame: pd.DataFrame, *, limit: int = 12, use_network: bool = True) -> dict:
    """For recent 30-minute calls, compare P50 to the INDO that should have been on lag_1."""
    empty = {"n": 0, "stale_share": None, "rows": []}
    if frame is None or frame.empty or "p50" not in frame.columns:
        return empty
    indo = pd.DataFrame()
    if use_network:
        try:
            start = (pd.Timestamp.now(tz=LONDON_TZ) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            indo = fetch_indo(start, date.today().isoformat())
        except Exception:  # noqa: BLE001
            indo = pd.DataFrame()
        if not indo.empty:
            indo = indo.copy()
            indo["timestamp"] = pd.to_datetime(indo["timestamp"], utc=True)
            indo = indo.sort_values("timestamp")
    work = frame.dropna(subset=["p50"]).copy()
    work["issued_at"] = pd.to_datetime(work["issued_at"], utc=True) if "issued_at" in work.columns else pd.NaT
    work = work.sort_values("issued_at").tail(limit)
    rows = []
    stale = 0
    for _, row in work.iterrows():
        issued = _as_utc(row["issued_at"]) if pd.notna(row.get("issued_at")) else _as_utc(row["timestamp"])
        last_mw = None
        last_ts = None
        if not indo.empty:
            available = indo.loc[indo["timestamp"] + pd.Timedelta(minutes=30) <= issued]
            last = available.iloc[-1] if not available.empty else None
            if last is not None:
                last_mw = float(last["demand_mw"])
                last_ts = last["timestamp"]
        elif "last_indo_mw" in work.columns and pd.notna(row.get("last_indo_mw")):
            last_mw = float(row["last_indo_mw"])
            last_ts = row["last_indo_timestamp"] if "last_indo_timestamp" in work.columns else None
        stored = row["demand_lag_1"] if "demand_lag_1" in work.columns else np.nan
        p50 = float(row["p50"])
        actual = row["actual_mw"] if "actual_mw" in work.columns else np.nan
        used = float(stored) if pd.notna(stored) else None
        lag_gap = abs(used - last_mw) if used is not None and last_mw is not None else None
        p50_gap = abs(p50 - last_mw) if last_mw is not None else None
        is_stale = (lag_gap is not None and lag_gap > 400) or (
            lag_gap is None and p50_gap is not None and p50_gap > 1500
        )
        if is_stale:
            stale += 1
        rows.append(
            {
                "issued_at": issued.isoformat(),
                "timestamp": _iso_stamp(row["timestamp"]),
                "p50": p50,
                "actual_mw": None if pd.isna(actual) else float(actual),
                "demand_lag_1": used,
                "available_indo_mw": last_mw,
                "available_indo_timestamp": _iso_stamp(last_ts) if last_ts is not None else None,
                "stale_lags": bool(is_stale),
            }
        )
    return {
        "n": int(len(work)),
        "stale_share": (stale / len(work)) if len(work) else None,
        "rows": rows,
    }


def _window_payload(
    preset: str,
    window_start: pd.Timestamp | None,
    board: pd.DataFrame | None = None,
    *,
    wrmsse: float | None = None,
    context: pd.DataFrame | None = None,
    rows: pd.DataFrame | None = None,
) -> dict:
    rows = _window_rows(preset, window_start, board) if rows is None else rows
    if not rows.empty:
        rows = rows.sort_values("timestamp")
    if context is not None:
        rows = _overlay_actuals(rows, context)
    needed = [column for column in ("actual_mw", "p50") if column in rows.columns]
    scored = rows.dropna(subset=needed) if not rows.empty and needed else rows
    metrics = preset_metrics(rows, compute_wrmsse=False)
    if wrmsse is not None:
        metrics["wrmsse"] = wrmsse
    return {
        "window_start": pd.Timestamp(window_start).isoformat() if window_start is not None else None,
        "issued_at": _issue_stamp(rows),
        "n": int(len(rows)),
        "scored": int(len(scored)) if scored is not None else 0,
        "metrics": metrics,
        "rows": _frame_records(rows),
        "has_frozen": window_start is not None and _frozen_path(preset, window_start).exists(),
        "inputs": input_health(preset, window_start, rows=rows),
    }


def _previous_actuals(
    preset: str,
    windows: list,
    board: pd.DataFrame,
    context: pd.DataFrame | None = None,
) -> list[dict]:
    """Yesterday / last week INDO, clipped to that calendar window (partial days are fine)."""
    now = london_now()
    ctx = context if context is not None else load_context_actuals()
    if preset == DAY:
        start, end, _ = london_day_range(previous_london_midnight(now))
        chunk = _slice_actuals(ctx, start, end)
        if chunk.empty:
            rows = _overlay_actuals(_span_rows(DAY, start, end, board), ctx)
            if not rows.empty and "actual_mw" in rows.columns:
                chunk = rows.dropna(subset=["actual_mw"])
        if chunk.empty:
            return []
        keep = [column for column in ("timestamp", "actual_mw", "settlement_period") if column in chunk.columns]
        return _frame_records(chunk.loc[:, keep].dropna(subset=["actual_mw"]))
    end = week_window(now)
    start = week_previous_chart_start(now)
    chunk = _slice_actuals(ctx, start, end)
    if chunk.empty:
        rows = _overlay_actuals(_span_rows(WEEK, start, end, board), ctx)
        if not rows.empty and "actual_mw" in rows.columns:
            chunk = rows.dropna(subset=["actual_mw"])
    if chunk.empty:
        return []
    keep = [column for column in ("timestamp", "actual_mw", "settlement_period") if column in chunk.columns]
    return _frame_records(chunk.loc[:, keep].dropna(subset=["actual_mw"]))


def _next30_history(
    next30: pd.DataFrame,
    live30: pd.DataFrame,
    now_utc: pd.Timestamp,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """Today from midnight, plus yesterday only if that London day is complete (all half-hours)."""
    today = day_window(now_utc)
    yesterday = previous_london_midnight(now_utc)
    check = context if context is not None and not context.empty else next30
    cutoff = yesterday if complete_midnight_day(yesterday, check, column="actual_mw", now=now_utc) else today
    history = next30.copy()
    if not history.empty:
        if "issued_at" in history.columns:
            history = history.sort_values("issued_at").drop_duplicates("timestamp", keep="last")
        history = history.loc[pd.to_datetime(history["timestamp"], utc=True) >= cutoff]
        if not live30.empty:
            cap = pd.to_datetime(live30["timestamp"].iloc[0], utc=True)
            history = history.loc[pd.to_datetime(history["timestamp"], utc=True) <= cap]
    history = _overlay_actuals(history, context)
    cap = pd.to_datetime(live30["timestamp"].iloc[0], utc=True) if not live30.empty else now_utc
    trail = _slice_actuals(context, cutoff, cap + pd.Timedelta(seconds=1))
    if trail.empty:
        return history.sort_values("timestamp") if not history.empty else history
    have = (
        set(pd.to_datetime(history["timestamp"], utc=True))
        if not history.empty and "timestamp" in history.columns
        else set()
    )
    extra = trail.loc[~pd.to_datetime(trail["timestamp"], utc=True).isin(have)].copy()
    if extra.empty:
        return history.sort_values("timestamp") if not history.empty else history
    for column in ("p10", "p50", "p90"):
        extra[column] = np.nan
    combined = extra if history.empty else pd.concat([history, extra], ignore_index=True)
    return combined.sort_values("timestamp")


def evaluation_payload(
    board: pd.DataFrame,
    context: pd.DataFrame,
    audit: dict | None = None,
) -> dict:
    """Tag scored rows for the validation panel. Local parquet only — no weather API."""
    misses: list[dict] = []
    scored_n = 0
    outside = 0
    lags = 0
    large = 0
    for preset in PRESETS:
        frame = _overlay_actuals(_preset_frame(preset, board), context)
        if frame.empty or "p50" not in frame.columns or "actual_mw" not in frame.columns:
            continue
        work = frame.dropna(subset=["actual_mw", "p50"]).copy()
        if work.empty:
            continue
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        if "issued_at" in work.columns:
            work = work.sort_values("issued_at").drop_duplicates("timestamp", keep="last")
        work = work.reset_index(drop=True)
        scored_n += int(len(work))
        error = pd.to_numeric(work["actual_mw"], errors="coerce") - pd.to_numeric(work["p50"], errors="coerce")
        work["error_mw"] = error
        work["abs_error_mw"] = error.abs()
        band = pd.Series(False, index=work.index)
        if {"p10", "p90"}.issubset(work.columns):
            band = (work["actual_mw"] < work["p10"]) | (work["actual_mw"] > work["p90"])
        lag_issue = pd.Series(False, index=work.index)
        if {"demand_lag_1", "last_indo_mw"}.issubset(work.columns):
            lag_issue = (
                pd.to_numeric(work["demand_lag_1"], errors="coerce")
                - pd.to_numeric(work["last_indo_mw"], errors="coerce")
            ).abs() > 400
        for _, row in work.iterrows():
            tags: list[str] = []
            if bool(lag_issue.loc[row.name]):
                tags.append("lags")
                lags += 1
            if bool(band.loc[row.name]):
                tags.append("outside band")
                outside += 1
            elif float(row["abs_error_mw"]) >= 1500:
                tags.append("large miss")
                large += 1
            misses.append(
                {
                    "preset": preset,
                    "timestamp": _iso_stamp(row["timestamp"]),
                    "p50": float(row["p50"]),
                    "actual_mw": float(row["actual_mw"]),
                    "error_mw": float(row["error_mw"]),
                    "tag": tags[0] if tags else "typical",
                }
            )
    misses.sort(key=lambda row: abs(row["error_mw"]), reverse=True)
    audit = audit or {}
    return {
        "scored": scored_n,
        "outside_band": outside,
        "lags": lags,
        "large_miss": large,
        "stale_share": audit.get("stale_share"),
        "weather": {
            "ready": False,
            "note": "Weather-vs-demand needs issued temperature compared with later actual weather. That fills in over a few days; this panel does not call the weather API.",
        },
        "worst": misses[:8],
    }


def board_payload(*, refresh: bool = False) -> dict:
    """Dashboard JSON. Default path is local parquet only so the page stays fast."""
    if refresh:
        import_legacy_next30()
        refresh_actuals()
        refresh_wrmsse_cache()
        _BOARD_CACHE["payload"] = None
    elif _BOARD_CACHE["payload"] is not None:
        age = time.monotonic() - float(_BOARD_CACHE["at"])
        if age < _BOARD_CACHE_S:
            return _BOARD_CACHE["payload"]
    board = load_board()
    context = load_context_actuals()
    wrmsse = load_wrmsse_cache()
    now = london_now()
    now_utc = now.tz_convert("UTC")
    next30 = _preset_frame(NEXT30, board).copy()
    if not next30.empty:
        next30["timestamp"] = pd.to_datetime(next30["timestamp"], utc=True)
        next30 = _align_next30_target(next30)
        next30 = next30.sort_values("timestamp")
    live30 = next30.iloc[0:0]
    if not next30.empty:
        waiting = next30
        if "actual_mw" in next30.columns:
            waiting = next30.loc[next30["actual_mw"].isna()].sort_values("timestamp")
        live30 = waiting.iloc[:1] if not waiting.empty else next30.iloc[-1:]
    day_windows = _windows(DAY, board)
    week_windows = _windows(WEEK, board)
    history30 = _next30_history(next30, live30, now_utc, context)
    live_window = live30["window_start"].iloc[0] if not live30.empty else None
    n30_metrics = preset_metrics(_overlay_actuals(next30, context), compute_wrmsse=False)
    n30_metrics["wrmsse"] = wrmsse.get(NEXT30)
    audit = next30_input_audit(next30, use_network=False)
    yesterday = previous_london_midnight(now)
    today = day_window(now)
    y_start, y_end, _ = london_day_range(yesterday)
    t_start, t_end, _ = london_day_range(today)
    this_monday = week_window(now)
    last_monday = previous_week_window(now)
    week_left = week_previous_chart_start(now)
    next_monday = this_monday + pd.Timedelta(days=7)
    payload = {
        "as_of": now.isoformat(),
        "next30": {
            **_window_payload(NEXT30, live_window, board, wrmsse=wrmsse.get(NEXT30), context=context),
            "current": _frame_records(live30)[:1],
            "history": _frame_records(history30),
            "metrics": n30_metrics,
            "audit": audit,
        },
        "day": {
            "boundary": t_start.isoformat(),
            "current": _window_payload(
                DAY,
                today,
                board,
                wrmsse=wrmsse.get(DAY),
                context=context,
                rows=_pad_actual_slots(_span_rows(DAY, t_start, t_end, board), context, t_start, t_end),
            ),
            "previous": _window_payload(
                DAY,
                yesterday,
                board,
                context=context,
                rows=_pad_actual_slots(_span_rows(DAY, y_start, y_end, board), context, y_start, y_end),
            ),
            "previous_actuals": _previous_actuals(DAY, day_windows, board, context),
        },
        "week": {
            "current": _window_payload(
                WEEK,
                this_monday,
                board,
                wrmsse=wrmsse.get(WEEK),
                context=context,
                rows=_pad_actual_slots(
                    _span_rows(WEEK, this_monday, next_monday, board), context, this_monday, next_monday
                ),
            ),
            "previous": _window_payload(
                WEEK,
                last_monday,
                board,
                context=context,
                rows=_pad_actual_slots(
                    _span_rows(WEEK, week_left, this_monday, board), context, week_left, this_monday
                ),
            ),
            "previous_actuals": _previous_actuals(WEEK, week_windows, board, context),
        },
        "eval": evaluation_payload(board, context, audit),
    }
    _BOARD_CACHE["at"] = time.monotonic()
    _BOARD_CACHE["payload"] = payload
    return payload


def forecast_csv(preset: str, which: str = "current") -> str:
    if preset == WEEK:
        start = week_window() if which != "previous" else previous_week_window()
        rows = _span_rows(WEEK, start, start + pd.Timedelta(days=7))
        return _to_csv(rows)
    if preset == DAY:
        start = day_window() if which != "previous" else previous_london_midnight()
        start_u, end_u, _ = london_day_range(start)
        rows = _span_rows(DAY, start_u, end_u)
        return _to_csv(rows)
    windows = _windows(preset)
    if not windows:
        return ""
    window = windows[0] if which != "previous" else (windows[1] if len(windows) > 1 else windows[0])
    rows = _window_rows(preset, window)
    return _to_csv(rows)


def frozen_csv(preset: str, which: str = "current") -> str:
    if preset == WEEK:
        start = week_window() if which != "previous" else previous_week_window()
        return _to_csv(load_frozen(WEEK, start))
    windows = _windows(preset)
    if not windows:
        return ""
    window = windows[0] if which != "previous" else (windows[1] if len(windows) > 1 else windows[0])
    return _to_csv(load_frozen(preset, window))


def _iso_utc_series(series: pd.Series) -> pd.Series:
    stamp = pd.to_datetime(series, utc=True)
    formatted = stamp.dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    return formatted.where(stamp.notna(), None)


def _to_csv(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    work = frame.copy()
    for column in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[column]):
            work[column] = _iso_utc_series(work[column])
    buf = StringIO()
    work.to_csv(buf, index=False)
    return buf.getvalue()


def scored_frame(*, latest_issue: bool = True) -> pd.DataFrame:
    """Elapsed rows with INDO on the legacy hourly archive."""
    archive = load_archive()
    if archive.empty or "actual_mw" not in archive.columns:
        return archive
    out = archive.dropna(subset=["actual_mw", "p50"]).copy()
    if "settlement_period" in out.columns and out["settlement_period"].notna().any():
        out = out.loc[out["settlement_period"].notna()]
    if latest_issue and not out.empty and {"issued_at", "mode", "timestamp"}.issubset(out.columns):
        out = out.sort_values("issued_at").drop_duplicates(subset=["mode", "timestamp"], keep="last")
    return out


def live_metrics() -> pd.DataFrame:
    scored = scored_frame()
    if scored.empty:
        return pd.DataFrame(columns=["split", "metric", "value"])
    return validation_metrics(scored)


def latest_forecast(mode: str = SHORT) -> pd.DataFrame:
    archive = load_archive()
    if archive.empty:
        return archive
    if mode in archive["mode"].to_numpy():
        archive = archive.loc[archive["mode"] == mode]
    latest = archive["issued_at"].max()
    return archive.loc[archive["issued_at"] == latest].sort_values("timestamp")


def snapshot() -> dict:
    archive = load_archive()
    scored = scored_frame()
    metrics = live_metrics()
    issued = None
    if not archive.empty:
        issued = pd.Timestamp(archive["issued_at"].max()).isoformat()
    return {
        "issued_at": issued,
        "archive_rows": int(len(archive)),
        "scored_rows": int(len(scored)),
        "metrics": metrics.to_dict(orient="records"),
        "as_of": london_now().isoformat(),
        "today": date.today().isoformat(),
    }


def _frame_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if frame.empty:
        return []
    work = frame.copy()
    if limit is not None:
        work = work.tail(limit)
    for column in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[column]):
            work[column] = _iso_utc_series(work[column])
    return json.loads(work.to_json(orient="records"))


def seconds_until_half_hour() -> float:
    """Sleep until :10 or :40 London so the previous settlement period's INDO is out."""
    now = london_now()
    minute = int(now.minute)
    second = float(now.second) + now.microsecond / 1e6
    if minute < 10:
        wait = (10 - minute) * 60 - second
    elif minute < 40:
        wait = (40 - minute) * 60 - second
    else:
        wait = (70 - minute) * 60 - second
    return max(5.0, wait + 5.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture and score live GB demand forecasts.")
    parser.add_argument("command", choices=("capture", "tick", "refresh", "score", "status", "board"))
    args = parser.parse_args(argv)
    if args.command == "capture":
        print(json.dumps(capture(), indent=2))
        return
    if args.command == "tick":
        print(json.dumps(tick_presets(), indent=2, default=str))
        return
    if args.command == "refresh":
        print(json.dumps(refresh_actuals(), indent=2))
        return
    if args.command == "score":
        refresh_actuals()
        print(live_metrics().to_string(index=False))
        return
    if args.command == "board":
        print(json.dumps(board_payload(), indent=2, default=str)[:4000])
        return
    print(json.dumps(snapshot(), indent=2))


if __name__ == "__main__":
    main()
