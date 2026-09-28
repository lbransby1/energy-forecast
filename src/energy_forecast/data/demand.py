"""Load GB National Demand from NESO files and live Insights INDO."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from energy_forecast.data.elexon import fetch_indo
from energy_forecast.paths import DATA_RAW
from energy_forecast.regions import SERIES_ID
from energy_forecast.settlement import attach_timestamp, normalize_settlement_date

logger = logging.getLogger(__name__)


def load_demand(*, include_live: bool = True, raw_dir=None) -> pd.DataFrame:
    """Return half-hourly GB National Demand.

    NESO ``demanddata_*.csv`` ``ND`` is the historic series. When
    ``include_live`` is true, Insights INDO is appended from the day after
    the last NESO date through today so the series can be validated live.
    """
    raw_dir = raw_dir or DATA_RAW
    neso = _load_neso_files(raw_dir)
    frames = [frame for frame in (neso,) if frame is not None and not frame.empty]
    if include_live:
        live_start = _live_start(neso)
        indo = fetch_indo(live_start.isoformat(), date.today().isoformat())
        if not indo.empty:
            frames.append(indo.assign(source="indo"))
    if not frames:
        raise FileNotFoundError(
            f"No NESO demanddata_*.csv in {raw_dir} and Insights INDO returned no rows."
        )

    demand = pd.concat(frames, ignore_index=True)
    demand = _clean_demand(demand)
    return demand.sort_values("timestamp").reset_index(drop=True)


def _live_start(neso: pd.DataFrame) -> date:
    if neso.empty:
        return date.today().replace(day=1)
    last = pd.Timestamp(neso["settlement_date"].max()).date()
    return last + timedelta(days=1)


def _load_neso_files(raw_dir) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("demanddata_*.csv"))
    if not paths:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for path in paths:
        logger.info("Loading NESO demand file %s", path.name)
        raw = pd.read_csv(path)
        raw.columns = [column.strip() for column in raw.columns]
        required = {"SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        parts.append(
            pd.DataFrame(
                {
                    "settlement_date": raw["SETTLEMENT_DATE"],
                    "settlement_period": raw["SETTLEMENT_PERIOD"],
                    "demand_mw": pd.to_numeric(raw["ND"], errors="coerce"),
                    "region_id": SERIES_ID,
                    "source": "neso",
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _clean_demand(demand: pd.DataFrame) -> pd.DataFrame:
    demand = demand.copy()
    demand["settlement_date"] = normalize_settlement_date(demand["settlement_date"])
    demand["settlement_period"] = pd.to_numeric(demand["settlement_period"], errors="coerce")
    demand["demand_mw"] = pd.to_numeric(demand["demand_mw"], errors="coerce")
    demand["region_id"] = SERIES_ID
    demand = demand.dropna(subset=["settlement_date", "settlement_period", "demand_mw"])
    demand["settlement_period"] = demand["settlement_period"].astype(int)
    demand = demand.drop_duplicates(
        subset=["settlement_date", "settlement_period"],
        keep="last",
    )
    if "source" not in demand.columns:
        demand["source"] = "neso"
    return attach_timestamp(demand)
