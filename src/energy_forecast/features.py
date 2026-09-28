"""Calendar, weather, and lag features for demand models."""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from energy_forecast.data.weather import HOURLY_VARS
from energy_forecast.regions import GB
from energy_forecast.settlement import LONDON_TZ, normalize_settlement_date

HDD_BASE_C = 15.5
COLD_C = 5.0
HOT_C = 20.0
SHORT_LAGS = (1, 2)
LAGS = (48, 336)
ALL_LAGS = (*SHORT_LAGS, *LAGS)
PERIODS_6H = 12
PERIODS_12H = 24
PERIODS_24H = 48
PERIODS_72H = 144
PERIODS_7D = 336

TRAJECTORY_FEATURES = [
    "temp_mean_24h",
    "temp_mean_72h",
    "temp_delta_6h",
    "temp_delta_24h",
    "temp_mean_fwd_12h",
    "temp_mean_fwd_24h",
    "temp_anom_7d",
    "temp_anom_seasonal",
    "hdd_sum_24h",
    "hdd_sum_72h",
    "cdd_sum_24h",
    "cdd_sum_72h",
    "cold_frac_72h",
    "hot_frac_72h",
]

FEATURE_COLUMNS = [
    "settlement_period",
    "hour",
    "minute",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "period_sin",
    "period_cos",
    "month_sin",
    "month_cos",
    "lead_hours",
    *HOURLY_VARS,
    "hdd",
    "cdd",
    *TRAJECTORY_FEATURES,
    "demand_lag_48",
    "demand_lag_336",
]

# Next-hour model: recent INDO plus the week features, but no lead_hours.
HOUR_FEATURE_COLUMNS = [
    *[column for column in FEATURE_COLUMNS if column != "lead_hours"],
    "demand_lag_1",
    "demand_lag_2",
]


def merge_demand_weather(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    left = demand.copy()
    right = weather.copy()
    left["settlement_date"] = normalize_settlement_date(left["settlement_date"])
    right["settlement_date"] = normalize_settlement_date(right["settlement_date"])
    keys = ["region_id", "settlement_date", "settlement_period"]
    weather_cols = [
        column
        for column in right.columns
        if column in {"timestamp", "lead_hours", *HOURLY_VARS} or column in keys
    ]
    right = right.loc[:, weather_cols]
    merged = left.merge(right, on=keys, how="inner", suffixes=("", "_wx"))
    if "timestamp_wx" in merged.columns:
        merged = merged.drop(columns=["timestamp_wx"])
    return merged


def build_features(
    frame: pd.DataFrame,
    *,
    climatology: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add calendar, weather-derived, trajectory, and demand-lag features."""
    out = frame.copy()
    out["settlement_date"] = normalize_settlement_date(out["settlement_date"])
    out["settlement_period"] = pd.to_numeric(out["settlement_period"], errors="coerce")
    timestamp = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    out["hour"] = timestamp.dt.hour
    out["minute"] = timestamp.dt.minute
    out["day_of_week"] = timestamp.dt.dayofweek
    out["month"] = timestamp.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday"] = _holiday_flags(out)
    out["period_sin"] = np.sin(2 * np.pi * out["settlement_period"] / 48)
    out["period_cos"] = np.cos(2 * np.pi * out["settlement_period"] / 48)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    if "lead_hours" not in out.columns:
        out["lead_hours"] = 0.0
    out["lead_hours"] = pd.to_numeric(out["lead_hours"], errors="coerce").fillna(0.0)
    out["hdd"] = np.maximum(HDD_BASE_C - out["temperature_2m"], 0)
    out["cdd"] = np.maximum(out["temperature_2m"] - HDD_BASE_C, 0)
    out = _add_weather_trajectory(out)
    out = _add_lags(out, climatology=climatology)
    return out


def demand_climatology(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean demand by region, month, weekday, and settlement period."""
    work = frame.copy()
    timestamp = pd.to_datetime(work["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    work["month"] = timestamp.dt.month
    work["day_of_week"] = timestamp.dt.dayofweek
    profile = (
        work.dropna(subset=["demand_mw"])
        .groupby(["region_id", "month", "day_of_week", "settlement_period"], as_index=False)["demand_mw"]
        .mean()
        .rename(columns={"demand_mw": "climatology_mw"})
    )
    return profile


def attach_demand_lags(demand: pd.DataFrame) -> pd.DataFrame:
    """Shift demand on the unduplicated series so January rows keep a 7-day lag."""
    out = demand.sort_values(["region_id", "timestamp"]).copy()
    grouped = out.groupby("region_id", group_keys=False)
    for lag in ALL_LAGS:
        out[f"demand_lag_{lag}"] = grouped["demand_mw"].shift(lag)
    return out


def _trajectory_group_columns(frame: pd.DataFrame) -> list[str]:
    if "lead_hours" in frame.columns and frame.duplicated(["region_id", "timestamp"]).any():
        return ["region_id", "lead_hours"]
    return ["region_id"]


def _add_weather_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    """Temperature path features: persistence, fronts, and anomalies.

    Trailing windows use only past weather (building thermal inertia).
    Forward windows use the weather forecast path already on the frame.
    Seasonal anomaly is causal: prior years/days for the same month and period.
    """
    group_cols = _trajectory_group_columns(frame)
    out = frame.sort_values([*group_cols, "timestamp"]).copy()
    temp = out.groupby(group_cols, group_keys=False)["temperature_2m"]
    hdd = out.groupby(group_cols, group_keys=False)["hdd"]
    cdd = out.groupby(group_cols, group_keys=False)["cdd"]

    mean_24h = temp.transform(lambda s: s.rolling(PERIODS_24H, min_periods=PERIODS_12H).mean())
    mean_72h = temp.transform(lambda s: s.rolling(PERIODS_72H, min_periods=PERIODS_24H).mean())
    mean_7d = temp.transform(lambda s: s.rolling(PERIODS_7D, min_periods=PERIODS_72H).mean())
    out["temp_mean_24h"] = mean_24h.fillna(out["temperature_2m"])
    out["temp_mean_72h"] = mean_72h.fillna(out["temperature_2m"])
    out["temp_delta_6h"] = temp.transform(lambda s: s - s.shift(PERIODS_6H)).fillna(0)
    out["temp_delta_24h"] = temp.transform(lambda s: s - s.shift(PERIODS_24H)).fillna(0)
    out["temp_mean_fwd_12h"] = temp.transform(
        lambda s: s.shift(-PERIODS_12H).rolling(PERIODS_12H, min_periods=PERIODS_6H).mean()
    ).fillna(out["temperature_2m"])
    out["temp_mean_fwd_24h"] = temp.transform(
        lambda s: s.shift(-PERIODS_24H).rolling(PERIODS_24H, min_periods=PERIODS_12H).mean()
    ).fillna(out["temperature_2m"])
    out["temp_anom_7d"] = (out["temperature_2m"] - mean_7d).fillna(0)

    seasonal_cols = [*group_cols, "month", "settlement_period"]
    seasonal = out.groupby(seasonal_cols, group_keys=False)["temperature_2m"].transform(
        lambda s: s.shift(1).expanding(min_periods=4).mean()
    )
    out["temp_anom_seasonal"] = (out["temperature_2m"] - seasonal).fillna(0)

    out["hdd_sum_24h"] = hdd.transform(lambda s: s.rolling(PERIODS_24H, min_periods=PERIODS_12H).sum()).fillna(
        out["hdd"]
    )
    out["hdd_sum_72h"] = hdd.transform(lambda s: s.rolling(PERIODS_72H, min_periods=PERIODS_24H).sum()).fillna(
        out["hdd"]
    )
    out["cdd_sum_24h"] = cdd.transform(lambda s: s.rolling(PERIODS_24H, min_periods=PERIODS_12H).sum()).fillna(
        out["cdd"]
    )
    out["cdd_sum_72h"] = cdd.transform(lambda s: s.rolling(PERIODS_72H, min_periods=PERIODS_24H).sum()).fillna(
        out["cdd"]
    )
    out["_cold"] = (out["temperature_2m"] < COLD_C).astype(float)
    out["_hot"] = (out["temperature_2m"] > HOT_C).astype(float)
    cold = out.groupby(group_cols, group_keys=False)["_cold"]
    hot = out.groupby(group_cols, group_keys=False)["_hot"]
    out["cold_frac_72h"] = cold.transform(
        lambda s: s.rolling(PERIODS_72H, min_periods=PERIODS_24H).mean()
    ).fillna(0)
    out["hot_frac_72h"] = hot.transform(
        lambda s: s.rolling(PERIODS_72H, min_periods=PERIODS_24H).mean()
    ).fillna(0)
    out = out.drop(columns=["_cold", "_hot"])
    return out


def _add_lags(frame: pd.DataFrame, climatology: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.sort_values(["region_id", "timestamp"]).copy()
    if "demand_mw" not in out.columns:
        out["demand_mw"] = np.nan
    missing = [lag for lag in ALL_LAGS if f"demand_lag_{lag}" not in out.columns]
    if missing:
        grouped = out.groupby(_trajectory_group_columns(out), group_keys=False)
        for lag in missing:
            out[f"demand_lag_{lag}"] = grouped["demand_mw"].shift(lag)

    if climatology is not None and not climatology.empty:
        out = out.merge(
            climatology,
            on=["region_id", "month", "day_of_week", "settlement_period"],
            how="left",
        )
        for lag in LAGS:
            column = f"demand_lag_{lag}"
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(out["climatology_mw"])
        out = out.drop(columns=["climatology_mw"])
    return out


def _holiday_flags(frame: pd.DataFrame) -> pd.Series:
    flags = pd.Series(0, index=frame.index, dtype=int)
    years = sorted(frame["settlement_date"].dt.year.dropna().unique().astype(int).tolist())
    if not years:
        return flags
    calendar = _uk_holidays(tuple(years), GB.holiday_subdiv)
    dates = frame["settlement_date"].dt.date
    flags.loc[frame.index] = dates.map(lambda value: int(value in calendar)).to_numpy()
    return flags


@lru_cache(maxsize=32)
def _uk_holidays(years: tuple[int, ...], subdiv: str) -> holidays.HolidayBase:
    return holidays.country_holidays("GB", subdiv=subdiv, years=list(years))
