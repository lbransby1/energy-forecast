"""Settlement period helpers (Europe/London, including DST 46/50-period days)."""

from __future__ import annotations

import pandas as pd

LONDON_TZ = "Europe/London"


def normalize_settlement_date(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values).dt.tz_localize(None).dt.normalize()
    text = values.astype(str).str.strip().str.strip('"')
    date_only = text.str.replace(r"\s+.*$", "", regex=True)
    parsed = pd.to_datetime(date_only, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(date_only, format="%d-%b-%Y", errors="coerce"))
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(date_only, format="%d-%b-%y", errors="coerce"))
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(date_only, errors="coerce"))
    return parsed.dt.tz_localize(None).dt.normalize()


def settlement_index(date, n_periods: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(pd.Timestamp(date).date(), tz=LONDON_TZ)
    return pd.date_range(start=start, periods=int(n_periods), freq="30min")


def attach_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a Europe/London timestamp at the start of each settlement period."""
    if frame.empty:
        out = frame.copy()
        out["timestamp"] = pd.Series(dtype="datetime64[ns, Europe/London]")
        return out

    frame = frame.copy()
    frame["settlement_date"] = normalize_settlement_date(frame["settlement_date"])
    parts: list[pd.Series] = []
    for date, group in frame.groupby(frame["settlement_date"], sort=False):
        n_periods = int(group["settlement_period"].max())
        index = settlement_index(date, n_periods)
        mapped = group["settlement_period"].map(
            lambda period, index=index: index[int(period) - 1]
            if 1 <= int(period) <= len(index)
            else pd.NaT
        )
        parts.append(mapped)
    frame["timestamp"] = pd.concat(parts)
    return frame


def add_settlement_keys(frame: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    """Derive settlement_date and settlement_period from a tz-aware timestamp."""
    out = frame.copy()
    local = pd.to_datetime(out[time_col], utc=True).dt.tz_convert(LONDON_TZ)
    out["_local_date"] = local.dt.tz_localize(None).dt.normalize()
    out["settlement_period"] = out.groupby("_local_date", sort=False).cumcount() + 1
    out["settlement_date"] = out["_local_date"]
    return out.drop(columns=["_local_date"])
