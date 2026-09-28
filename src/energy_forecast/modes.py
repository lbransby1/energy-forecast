"""Short (hourly, 24h) and medium (7d) forecast views.

Live plots use ``display_freq``. Holdout and Optuna stay on ``freq``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.settlement import LONDON_TZ

SHORT = "short"
MEDIUM = "medium"

_SEASON = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

MODE_SPEC = {
    SHORT: {
        "freq": "1h",
        "display_freq": "1h",
        "horizon_hours": 24,
        "season": 168,
        "label": "hourly, next 24h",
    },
    MEDIUM: {
        "freq": "6h",
        "display_freq": "1h",
        "horizon_hours": 168,
        "season": 28,
        "label": "hourly, next 7 days",
    },
}
_PERIODS_PER_BIN = {"30min": 1, "1h": 2, "6h": 12}

_MEAN_COLS = (
    "p10",
    "p50",
    "p90",
    "demand_mw",
    "actual_mw",
    "temperature_2m",
    "temp_mean_fwd_12h",
    "temp_mean_fwd_24h",
    "temp_delta_24h",
    "temp_anom_seasonal",
    "temp_anom_7d",
    "climatology_mw",
    "mean_pred",
)
_FIRST_COLS = ("settlement_date", "settlement_period", "region_id", "is_weekend", "is_holiday")


def resample_to_mode(frame: pd.DataFrame, mode: str, freq: str | None = None) -> pd.DataFrame:
    """Average half-hourly rows onto a grid. Default ``freq`` is the eval grid."""
    if mode not in MODE_SPEC:
        raise ValueError(f"mode must be {SHORT!r} or {MEDIUM!r}")
    if frame.empty:
        return frame.copy()

    spec = MODE_SPEC[mode]
    use_freq = freq or spec["freq"]
    work = frame.copy()
    stamp = pd.to_datetime(work["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    work["_bin"] = stamp.dt.floor(use_freq, ambiguous="infer", nonexistent="shift_forward")

    group_cols = ["_bin"]
    if "region_id" in work.columns:
        group_cols = ["region_id", "_bin"]
    if "lead_hours" in work.columns and work.duplicated(["timestamp"]).any():
        group_cols = [*group_cols, "lead_hours"]

    agg: dict[str, str] = {}
    if "lead_hours" in work.columns and "lead_hours" not in group_cols:
        agg["lead_hours"] = "min"
    for column in _MEAN_COLS:
        if column in work.columns:
            agg[column] = "mean"
    for column in _FIRST_COLS:
        if column in work.columns and column not in group_cols:
            agg[column] = "first"

    expected = _PERIODS_PER_BIN.get(use_freq, 2)
    sizes = work.groupby(group_cols).size()
    valid_idx = sizes[sizes >= max(1, expected // 3)].index
    out = work.groupby(group_cols, as_index=False).agg(agg)
    out = out.set_index(group_cols).loc[valid_idx].reset_index()
    out = out.rename(columns={"_bin": "timestamp"})
    out["mode"] = mode
    if {"p10", "p90"}.issubset(out.columns):
        out["interval_width"] = out["p90"] - out["p10"]
    out = _refresh_errors(out)
    stamp = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    out["hour"] = stamp.dt.hour
    out["month"] = stamp.dt.month
    out["year"] = stamp.dt.year
    out["season"] = out["month"].map(_SEASON)
    out["weekday"] = np.where(stamp.dt.dayofweek >= 5, "weekend", "weekday")
    out["period_band"] = out["hour"].map(_hour_band)
    if "is_holiday" in out.columns:
        out["holiday"] = np.where(out["is_holiday"] == 1, "holiday", "normal")
    if "lead_hours" in out.columns:
        out["lead_band"] = out["lead_hours"].map(_lead_band)
    return out.sort_values("timestamp").reset_index(drop=True)


def _refresh_errors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    actual = None
    if "demand_mw" in out.columns and out["demand_mw"].notna().any():
        actual = out["demand_mw"]
    elif "actual_mw" in out.columns:
        actual = out["actual_mw"]
    if actual is None or "p50" not in out.columns:
        return out
    out["error"] = actual - out["p50"]
    out["abs_error"] = out["error"].abs()
    out["ape"] = out["abs_error"] / actual.clip(lower=1)
    if {"p10", "p90"}.issubset(out.columns):
        out["covered"] = (actual >= out["p10"]) & (actual <= out["p90"])
    if "climatology_mw" in out.columns:
        out["clim_abs_error"] = (actual - out["climatology_mw"]).abs()
    return out


def _lead_band(hours: float) -> str:
    if hours <= 24:
        return "0-24h"
    if hours <= 72:
        return "24-72h"
    return "72-168h"


def _hour_band(hour: int) -> str:
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 16:
        return "afternoon"
    if hour < 20:
        return "evening_peak"
    return "late_evening"
