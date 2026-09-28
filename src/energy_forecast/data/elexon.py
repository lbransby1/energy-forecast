"""Elexon Insights Initial National Demand Outturn (INDO)."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from energy_forecast.paths import DATA_PROCESSED, ensure_data_dirs
from energy_forecast.regions import SERIES_ID
from energy_forecast.settlement import attach_timestamp, normalize_settlement_date

logger = logging.getLogger(__name__)

INSIGHTS_URL = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
MAX_SPAN_DAYS = 7
TIMEOUT_S = 60
_CACHE_LOCK = threading.Lock()


class InsightsAPIError(RuntimeError):
    pass


def fetch_indo(
    start_date: str,
    end_date: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch half-hourly GB INDO (MW) from Insights. No API key required."""
    ensure_data_dirs()
    cache_path = DATA_PROCESSED / "indo.parquet"
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    with _CACHE_LOCK:
        return _fetch_indo_locked(cache_path, start, end, use_cache=use_cache)


def _fetch_indo_locked(
    cache_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    use_cache: bool,
) -> pd.DataFrame:
    cached = _read_cache(cache_path)
    missing = [] if not use_cache else _missing_dates(cached, start, end)
    if not use_cache:
        missing = [(start.date(), end.date())]
    # Today's (and yesterday's) outturn grows through the day. A morning
    # snapshot in the cache would otherwise look complete and never update.
    live_from = date.today() - timedelta(days=1)
    if use_cache and end.date() >= live_from:
        live_end = end.date()
        live_start = max(start.date(), live_from)
        already = any(a <= live_start and live_end <= b for a, b in missing)
        if not already:
            missing.append((live_start, live_end))
    if missing:
        fetched = [_fetch_window(window_start, window_end) for window_start, window_end in missing]
        cached = pd.concat([cached, *fetched], ignore_index=True)
        cached = _normalise(cached)
        _write_cache(cache_path, cached)
        logger.info("Wrote INDO cache %s", cache_path)

    return cached[
        (cached["settlement_date"] >= start) & (cached["settlement_date"] <= end)
    ].sort_values("timestamp").reset_index(drop=True)


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    """Replace the cache atomically so readers never see a half-written parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _read_cache(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            return _normalise(pd.read_parquet(path))
        except Exception as exc:  # noqa: BLE001 — a torn parquet must not block live
            last_exc = exc
            time.sleep(0.05 * (attempt + 1))
    logger.warning("INDO cache unreadable (%s); refetching", last_exc)
    return pd.DataFrame()


def _missing_dates(
    cached: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[date, date]]:
    needed = pd.date_range(start, end, freq="D")
    have: set[date] = set()
    if not cached.empty:
        have = set(pd.to_datetime(cached["settlement_date"]).dt.date)
    gaps = [ts.date() for ts in needed if ts.date() not in have]
    return _chunk_dates(gaps)


def _chunk_dates(days: list[date]) -> list[tuple[date, date]]:
    if not days:
        return []
    chunks: list[tuple[date, date]] = []
    chunk_start = days[0]
    prev = days[0]
    for day in days[1:]:
        too_long = (day - chunk_start).days >= MAX_SPAN_DAYS
        if day != prev + timedelta(days=1) or too_long:
            chunks.append((chunk_start, prev))
            chunk_start = day
        prev = day
    chunks.append((chunk_start, prev))
    return chunks


def _fetch_window(start: date, end: date) -> pd.DataFrame:
    params = {
        "settlementDateFrom": start.isoformat(),
        "settlementDateTo": end.isoformat(),
        "format": "json",
    }
    logger.info("Fetching Insights INDO %s to %s", start, end)
    try:
        response = requests.get(INSIGHTS_URL, params=params, timeout=TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise InsightsAPIError(f"Insights INDO request failed: {exc}") from exc

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "settlement_date": frame["settlementDate"],
            "settlement_period": frame["settlementPeriod"],
            "demand_mw": pd.to_numeric(frame["initialDemandOutturn"], errors="coerce"),
            "region_id": SERIES_ID,
            "source": "indo",
        }
    )
    return _normalise(out)


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["settlement_date"] = normalize_settlement_date(out["settlement_date"])
    out["settlement_period"] = pd.to_numeric(out["settlement_period"], errors="coerce").astype("Int64")
    out["demand_mw"] = pd.to_numeric(out["demand_mw"], errors="coerce")
    out["region_id"] = SERIES_ID
    out = out.dropna(subset=["settlement_date", "settlement_period", "demand_mw"])
    out["settlement_period"] = out["settlement_period"].astype(int)
    if "source" not in out.columns:
        out["source"] = "indo"
    out = out.drop_duplicates(subset=["settlement_date", "settlement_period"], keep="last")
    if "timestamp" not in out.columns or out["timestamp"].isna().any():
        out = attach_timestamp(out.drop(columns=["timestamp"], errors="ignore"))
    return out.sort_values("timestamp").reset_index(drop=True)
