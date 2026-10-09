"""Notebook helpers to re-score frozen live rows. Not used by the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.data.weather import HOURLY_VARS, load_cached_weather
from energy_forecast.features import (
    DAY_YESTERDAY_FEATURES,
    DAY_YESTERDAY_LAG_FEATURES,
    HDD_BASE_C,
    WEEK_ENGINEERED_FEATURES,
    WEEK_V3_LAG_FEATURES,
    _add_weather_trajectory,
    _add_week_engineered,
)
from energy_forecast.live import (
    DAY,
    NEXT30,
    WEEK,
    _overlay_actuals,
    _overlay_next30_actuals,
    _pad_actual_slots,
    _preset_frame,
    _span_rows,
    frozen_dir,
    previous_week_window,
    week_window,
)
from energy_forecast.regions import SERIES_ID
from energy_forecast.settlement import LONDON_TZ, normalize_settlement_date

DAY_REVIEW_DAYS = 3
_PREFER_FROZEN = {
    "temperature_2m",
    *HOURLY_VARS,
    "hdd",
    "cdd",
    "hdd_sum_24h",
    "hdd_sum_72h",
    "cdd_sum_24h",
    "cdd_sum_72h",
    "temp_mean_fwd_12h",
    "temp_mean_fwd_24h",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_48",
    "demand_lag_336",
    *WEEK_ENGINEERED_FEATURES,
    *WEEK_V3_LAG_FEATURES,
    *DAY_YESTERDAY_FEATURES,
    *DAY_YESTERDAY_LAG_FEATURES,
    "last_indo_mw",
    "lead_hours",
    "is_weekend",
    "is_holiday",
    "day_of_week",
}


def week_review_span(
    board: pd.DataFrame,
    context: pd.DataFrame,
    when: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Monday–Monday window for the week pack: the freeze with more finished hours."""
    this_monday = week_window(when)
    last_monday = previous_week_window(when)
    this_end = this_monday + pd.Timedelta(days=7)
    n_this = _count_week_scores(board, context, this_monday, this_end)
    n_last = _count_week_scores(board, context, last_monday, this_monday)
    if n_last > n_this:
        return last_monday, this_monday
    return this_monday, this_end


def clip_day_review(frame: pd.DataFrame, *, days: int | None = DAY_REVIEW_DAYS) -> pd.DataFrame:
    """Keep the last ``days`` London calendar days of the day pack.

    ``days=None`` keeps every scored day on the board.
    """
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return frame
    if days is None:
        return frame
    work = frame.copy()
    local = pd.to_datetime(work["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    last_day = local.max().normalize()
    start = last_day - pd.Timedelta(days=int(days) - 1)
    return work.loc[local >= start].copy()


def _count_week_scores(
    board: pd.DataFrame | None,
    context: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    rows = _span_rows(WEEK, start, end, board)
    if rows.empty:
        return 0
    if context is not None:
        rows = _overlay_actuals(rows, context)
    if not {"actual_mw", "p50"}.issubset(rows.columns):
        return 0
    return int(rows.dropna(subset=["actual_mw", "p50"]).shape[0])


def _pack_rows(
    preset: str,
    board: pd.DataFrame,
    context: pd.DataFrame,
    *,
    day_days: int | None = DAY_REVIEW_DAYS,
) -> pd.DataFrame:
    if preset == WEEK:
        start, end = week_review_span(board, context)
        frame = _pad_actual_slots(_span_rows(WEEK, start, end, board), context, start, end)
    else:
        frame = _preset_frame(preset, board)
        if frame.empty:
            return frame
        if preset == NEXT30:
            frame = _overlay_next30_actuals(frame, context)
        else:
            frame = _overlay_actuals(frame, context)
        if preset == DAY:
            frame = clip_day_review(frame, days=day_days)
    work = frame.copy()
    if work.empty:
        return work
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    if "issued_at" in work.columns:
        work["issued_at"] = pd.to_datetime(work["issued_at"], utc=True)
        work = work.sort_values("issued_at").drop_duplicates("timestamp", keep="last")
    if {"actual_mw", "p50"}.issubset(work.columns):
        work["error_mw"] = pd.to_numeric(work["actual_mw"], errors="coerce") - pd.to_numeric(
            work["p50"], errors="coerce"
        )
        work["abs_error_mw"] = work["error_mw"].abs()
    if "window_start" in work.columns:
        work["window_start"] = pd.to_datetime(work["window_start"], utc=True)
    return work.sort_values("timestamp")


def _all_frozen(preset: str) -> pd.DataFrame:
    paths = sorted(frozen_dir().glob(f"{preset}_*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _attach_frozen_and_weather(
    preset: str,
    scored: pd.DataFrame,
    archive: pd.DataFrame | None,
) -> pd.DataFrame:
    frozen = _all_frozen(preset)
    work = scored.copy()
    if not frozen.empty and "timestamp" in frozen.columns:
        frozen = frozen.copy()
        frozen["timestamp"] = pd.to_datetime(frozen["timestamp"], utc=True)
        extra = [
            column
            for column in frozen.columns
            if column not in work.columns or column in _PREFER_FROZEN
        ]
        keys = ["timestamp"]
        if "window_start" in frozen.columns and "window_start" in work.columns:
            frozen["window_start"] = pd.to_datetime(frozen["window_start"], utc=True)
            keys = ["window_start", "timestamp"]
        take = list(dict.fromkeys([*keys, *[c for c in extra if c not in keys]]))
        right = frozen.loc[:, [c for c in take if c in frozen.columns]].drop_duplicates(keys, keep="last")
        work = work.merge(right, on=keys, how="left", suffixes=("", "_frozen"))
        for column in _PREFER_FROZEN:
            frozen_col = f"{column}_frozen"
            if frozen_col in work.columns:
                work[column] = pd.to_numeric(work[frozen_col], errors="coerce").combine_first(
                    pd.to_numeric(work[column], errors="coerce") if column in work.columns else None
                )
    if archive is not None and not archive.empty and "timestamp" in archive.columns:
        era = archive.copy()
        era["timestamp"] = pd.to_datetime(era["timestamp"], utc=True)
        if era["timestamp"].dt.tz is None:
            era["timestamp"] = era["timestamp"].dt.tz_localize("UTC")
        else:
            era["timestamp"] = era["timestamp"].dt.tz_convert("UTC")
        tcol = "temperature_2m"
        if tcol in era.columns and "timestamp" in work.columns:
            right = (
                era.dropna(subset=[tcol])
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .loc[:, ["timestamp", tcol]]
                .rename(columns={tcol: "temperature_actual"})
            )
            left = work.sort_values("timestamp")
            work = pd.merge_asof(
                left,
                right,
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("90min"),
            )
    return work


def _load_archive_weather() -> pd.DataFrame | None:
    try:
        weather = load_cached_weather([SERIES_ID], source="archive")
    except Exception:  # noqa: BLE001
        return None
    if weather.empty or "timestamp" not in weather.columns:
        return None
    out = weather.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("UTC")
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")
    return out


def _ensure_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "settlement_date" in out.columns:
        out["settlement_date"] = normalize_settlement_date(out["settlement_date"])
    timestamp = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    out["hour"] = timestamp.dt.hour
    out["minute"] = timestamp.dt.minute
    out["day_of_week"] = timestamp.dt.dayofweek
    out["month"] = timestamp.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    if "settlement_period" in out.columns:
        period = pd.to_numeric(out["settlement_period"], errors="coerce").fillna(1)
        out["period_sin"] = np.sin(2 * np.pi * period / 48)
        out["period_cos"] = np.cos(2 * np.pi * period / 48)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    if "lead_hours" not in out.columns:
        out["lead_hours"] = 0.0
    return out


def _recompute_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "temperature_2m" not in frame.columns:
        return frame
    work = frame.copy()
    if "region_id" not in work.columns:
        work["region_id"] = SERIES_ID
    work["hdd"] = np.maximum(HDD_BASE_C - pd.to_numeric(work["temperature_2m"], errors="coerce"), 0)
    work["cdd"] = np.maximum(pd.to_numeric(work["temperature_2m"], errors="coerce") - HDD_BASE_C, 0)
    stamp = pd.to_datetime(work["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    work["month"] = stamp.dt.month
    if "settlement_date" in work.columns:
        work["settlement_date"] = normalize_settlement_date(work["settlement_date"])
    if "window_start" in work.columns:
        parts = [
            _add_week_engineered(_add_weather_trajectory(group.copy()))
            for _, group in work.groupby("window_start", sort=False)
        ]
        return pd.concat(parts, ignore_index=True) if parts else work
    return _add_week_engineered(_add_weather_trajectory(work))


def _ready_predict_frame(frame: pd.DataFrame, models) -> pd.DataFrame:
    out = frame.copy()
    if "region_id" not in out.columns:
        out["region_id"] = SERIES_ID
    for column in models.features:
        if column not in out.columns:
            out[column] = np.nan
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out
