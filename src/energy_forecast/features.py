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
DAY_LAGS = (96,)
LONG_LAGS = (672, 1008)
ALL_LAGS = (*SHORT_LAGS, *LAGS, *DAY_LAGS, *LONG_LAGS)
PERIODS_3H = 6
PERIODS_6H = 12
PERIODS_12H = 24
PERIODS_24H = 48
PERIODS_48H = 96
PERIODS_72H = 144
PERIODS_7D = 336
PERIODS_14D = 672
PERIODS_21D = 1008

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

# Week pack that never sees yesterday: last week + weather/calendar only.
# Drops the Monday-freeze climatology slot; live does not need freeze-time or recursive lag_48.
WEEK_LONG_LAG_FEATURES = [column for column in FEATURE_COLUMNS if column != "demand_lag_48"]

# v3: same lag set as v2, plus week-over-week weather. Known at a Monday freeze.
# temp_delta_7d is the exact half-hourly clock match (this T minus last week's same period).
# temp_mean_24h_delta_7d is the smoothed complement. Apparent T is Open-Meteo feels-like (wind chill / heat).
WEEK_ENGINEERED_FEATURES = [
    "temp_delta_7d",
    "temp_mean_24h_delta_7d",
    "apparent_temp_delta_7d",
    "hdd_delta_7d",
    "cdd_delta_7d",
    "shortwave_delta_7d",
    "wind_delta_7d",
    "cloud_delta_7d",
    "shortwave_mean_24h",
    "wind_mean_24h",
]
# hour and period_sin duplicate settlement_period / period_cos on a 48-period clock.
WEEK_V3_DROP = frozenset({"hour", "period_sin"})
# 14-day lag, (14d − 7d) change, 3-week same-clock mean, and weekly second difference.
WEEK_V3_LAG_FEATURES = [
    "demand_lag_672",
    "demand_delta_14d_7d",
    "demand_mean_3w",
    "demand_accel_3w",
]
WEEK_V3_FEATURES = [
    column
    for column in [*WEEK_LONG_LAG_FEATURES, *WEEK_ENGINEERED_FEATURES, *WEEK_V3_LAG_FEATURES]
    if column not in WEEK_V3_DROP
]

# Day pack: midnight freeze already knows yesterday, so keep lag_48.
# Same week-over-week weather and 14d/3w demand extras as week v3.
DAY_V1_FEATURES = [*FEATURE_COLUMNS, *WEEK_ENGINEERED_FEATURES, *WEEK_V3_LAG_FEATURES]

# v2: clock-match vs yesterday (known at midnight), not a 3-week demand mean.
# temp_delta_1d is this T minus yesterday's same period; demand_delta_1d is
# yesterday minus the day before (lag_48 − lag_96). Drops 14d/3w extras that
# competed with true yesterday on v1.
DAY_YESTERDAY_FEATURES = [
    "temp_delta_1d",
    "temp_mean_24h_delta_1d",
    "apparent_temp_delta_1d",
    "hdd_delta_1d",
    "cdd_delta_1d",
    "shortwave_delta_1d",
    "wind_delta_1d",
    "cloud_delta_1d",
]
DAY_YESTERDAY_LAG_FEATURES = [
    "demand_lag_96",
    "demand_delta_1d",
]
# Short-horizon EMAs on the weather path. 3h reacts to fronts; 12h is building
# thermal inertia. Residuals are this half-hour minus the 3h EMA (high-pass).
# 24h T mean already exists as temp_mean_24h, so no 24h EMA.
WEATHER_EMA_SPAN_PERIODS = {
    "3h": PERIODS_3H,
    "12h": PERIODS_12H,
}
WEATHER_EMA_SOURCES = (
    ("temperature_2m", "temp"),
    ("apparent_temperature", "apparent_temp"),
    ("shortwave_radiation", "shortwave"),
    ("wind_speed_10m", "wind"),
    ("hdd", "hdd"),
    ("cdd", "cdd"),
)
WEATHER_EMA_FEATURES = [
    f"{prefix}_ema_{span}"
    for _, prefix in WEATHER_EMA_SOURCES
    for span in WEATHER_EMA_SPAN_PERIODS
] + [
    "temp_ema_resid_3h",
    "shortwave_ema_resid_3h",
    "wind_ema_resid_3h",
]
DAY_V2_FEATURES = [
    *FEATURE_COLUMNS,
    *WEEK_ENGINEERED_FEATURES,
    *DAY_YESTERDAY_FEATURES,
    *DAY_YESTERDAY_LAG_FEATURES,
    *WEATHER_EMA_FEATURES,
]

# v4: production columns plus 24h clock-match deltas (known at midnight).
# Not the v2 EMAs or week-over-week extras.
DAY_V4_FEATURES = [
    *FEATURE_COLUMNS,
    *DAY_YESTERDAY_FEATURES,
    *DAY_YESTERDAY_LAG_FEATURES,
]

# v3: keep the columns v2 SHAP actually used. Drop EMAs, 3w lags, collinear
# calendar duplicates (hour/minute/month), HDD/CDD stacks, and 1d/7d deltas
# that were not wind or shortwave. Midnight freeze still has yesterday.
DAY_V3_FEATURES = [
    "settlement_period",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "period_sin",
    "period_cos",
    "month_sin",
    "month_cos",
    "lead_hours",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "temp_mean_fwd_12h",
    "temp_mean_fwd_24h",
    "demand_lag_48",
    "demand_lag_96",
    "demand_lag_336",
    "wind_delta_1d",
    "wind_delta_7d",
    "shortwave_delta_1d",
    "shortwave_delta_7d",
]

# Next-hour model: recent INDO plus the week features, but no lead_hours.
HOUR_FEATURE_COLUMNS = [
    *[column for column in FEATURE_COLUMNS if column != "lead_hours"],
    "demand_lag_1",
    "demand_lag_2",
]

# Yesterday (lag_48) is only known at freeze if the target is still inside 24h.
FREEZE_LAG_LEAD_HOURS = 24.0
SETTLEMENT_STEP = pd.Timedelta(minutes=30)
DEMAND_LAG_STEPS = {
    "demand_lag_1": 1,
    "demand_lag_2": 2,
    "demand_lag_48": 48,
    "demand_lag_96": 96,
    "demand_lag_336": 336,
    "demand_lag_672": 672,
    "demand_lag_1008": 1008,
}
_ACTUAL_WEATHER_COLUMNS = ("temperature_actual",)


def apply_freeze_time_lags(
    frame: pd.DataFrame,
    *,
    lead_cutoff_hours: float = FREEZE_LAG_LEAD_HOURS,
) -> pd.DataFrame:
    """Use only demand history that exists at freeze time.

    ``demand_lag_48`` is yesterday. On a Monday week freeze that day has not
    happened yet for most of the horizon, so live currently fills it with
    monthly climatology. For leads at or beyond ``lead_cutoff_hours``, copy
    ``demand_lag_336`` (last week, known at issue) into ``demand_lag_48``.
    Shorter leads keep the real yesterday.
    """
    if frame is None or frame.empty:
        return frame
    if "demand_lag_48" not in frame.columns or "lead_hours" not in frame.columns:
        return frame
    out = frame.copy()
    lead = pd.to_numeric(out["lead_hours"], errors="coerce").fillna(0.0)
    mask = lead >= float(lead_cutoff_hours)
    if not mask.any():
        return out
    if "demand_lag_336" in out.columns:
        lagged = pd.to_numeric(out["demand_lag_336"], errors="coerce")
        out.loc[mask, "demand_lag_48"] = lagged.loc[mask].to_numpy()
    else:
        out.loc[mask, "demand_lag_48"] = np.nan
    return out


def demand_lag_columns(features: list[str] | None) -> dict[str, int]:
    names = set(features or [])
    return {column: steps for column, steps in DEMAND_LAG_STEPS.items() if column in names}


def lag_walk_step(features: list[str] | None) -> int | None:
    steps = list(demand_lag_columns(features).values())
    return min(steps) if steps else None


def utc_floor_30min(ts) -> pd.Timestamp:
    stamp = pd.to_datetime(ts, utc=True)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.floor("30min")


def seed_demand_path(
    scored: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> dict[tuple[str, pd.Timestamp], float]:
    """Observed MW known at issue, plus lag columns that point *before* the window.

    Does not ingest ``actual_mw`` on the scored horizon — that would leak this
    week's outturn into Tuesday's ``lag_48``.
    """
    path: dict[tuple[str, pd.Timestamp], float] = {}

    def ingest(frame: pd.DataFrame | None, column: str) -> None:
        if frame is None or frame.empty or column not in frame.columns or "timestamp" not in frame.columns:
            return
        work = frame.copy()
        if "region_id" not in work.columns:
            work["region_id"] = GB.id
        stamps = pd.to_datetime(work["timestamp"], utc=True)
        values = pd.to_numeric(work[column], errors="coerce")
        for region, stamp, mw in zip(work["region_id"].astype(str), stamps, values):
            if pd.isna(mw):
                continue
            path[(str(region), utc_floor_30min(stamp))] = float(mw)

    ingest(history, "demand_mw")
    ingest(scored, "demand_mw")
    if scored is None or scored.empty or "timestamp" not in scored.columns:
        return path
    work = scored.copy()
    if "region_id" not in work.columns:
        work["region_id"] = "gb"
    start = utc_floor_30min(pd.to_datetime(work["timestamp"], utc=True).min())
    for column, steps in DEMAND_LAG_STEPS.items():
        if column not in work.columns:
            continue
        stamps = pd.to_datetime(work["timestamp"], utc=True)
        values = pd.to_numeric(work[column], errors="coerce")
        for region, stamp, mw in zip(work["region_id"].astype(str), stamps, values):
            if pd.isna(mw):
                continue
            origin = utc_floor_30min(stamp) - steps * SETTLEMENT_STEP
            if origin < start:
                path.setdefault((str(region), origin), float(mw))
    return path


def apply_demand_path_lags(
    frame: pd.DataFrame,
    demand_path: dict[tuple[str, pd.Timestamp], float],
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Replace demand lags from the rolling path. Weather columns are untouched."""
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if "region_id" not in out.columns:
        out["region_id"] = "gb"
    lags = demand_lag_columns(features if features is not None else list(out.columns))
    if not lags:
        return out
    stamps = pd.to_datetime(out["timestamp"], utc=True)
    regions = out["region_id"].astype(str)
    for column, steps in lags.items():
        if column not in out.columns:
            out[column] = np.nan
        current = pd.to_numeric(out[column], errors="coerce")
        rolled = []
        for region, stamp, fallback in zip(regions, stamps, current):
            origin = utc_floor_30min(stamp) - steps * SETTLEMENT_STEP
            rolled.append(demand_path.get((str(region), origin), fallback))
        out[column] = np.asarray(rolled, dtype=float)
    return out


def weather_path_from_forecast(
    scored: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """HDD/CDD and T trajectory from issue-time ``temperature_2m``, never ERA5."""
    if scored is None or scored.empty or "temperature_2m" not in scored.columns:
        return scored
    parts: list[pd.DataFrame] = []
    drop = [column for column in _ACTUAL_WEATHER_COLUMNS]
    if history is not None and not history.empty:
        parts.append(history.drop(columns=[c for c in drop if c in history.columns], errors="ignore"))
    parts.append(scored.drop(columns=[c for c in drop if c in scored.columns], errors="ignore"))
    work = pd.concat(parts, ignore_index=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    if "region_id" not in work.columns:
        work["region_id"] = "gb"
    work = work.drop_duplicates(["region_id", "timestamp"], keep="last")
    temp = pd.to_numeric(work["temperature_2m"], errors="coerce")
    work["hdd"] = np.maximum(HDD_BASE_C - temp, 0)
    work["cdd"] = np.maximum(temp - HDD_BASE_C, 0)
    local = work["timestamp"].dt.tz_convert(LONDON_TZ)
    if "month" not in work.columns or work["month"].isna().any():
        work["month"] = local.dt.month
    if "settlement_period" not in work.columns:
        work["settlement_period"] = 1
    work = _add_weather_trajectory(work)
    work = _add_week_engineered(work)
    keys = ["region_id", "timestamp"]
    take = [
        column
        for column in [
            "temperature_2m",
            "hdd",
            "cdd",
            *HOURLY_VARS,
            *TRAJECTORY_FEATURES,
            *WEEK_ENGINEERED_FEATURES,
            *DAY_YESTERDAY_FEATURES,
            *WEATHER_EMA_FEATURES,
        ]
        if column in work.columns
    ]
    right = work.loc[:, list(dict.fromkeys([*keys, *take]))]
    out = scored.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "region_id" not in out.columns:
        out["region_id"] = "gb"
    out["_row"] = np.arange(len(out))
    merged = out.drop(columns=[c for c in take if c in out.columns]).merge(right, on=keys, how="left")
    return merged.sort_values("_row").drop(columns=["_row"]).reset_index(drop=True)


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
    out = _add_week_engineered(out)
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
    """Shift demand on the unduplicated series so January rows keep 7-, 14-, and 21-day lags."""
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
    """Temperature path features: persistence, fronts, anomalies, and short EMAs.

    Trailing windows use only past weather (building thermal inertia).
    Forward windows use the weather forecast path already on the frame.
    Seasonal anomaly is causal: prior years/days for the same month and period.
    EMAs are trailing on the same path (3h and 12h); they include the current step.
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
    return _add_weather_ema(out, group_cols)


def _add_weather_ema(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Trailing EMAs along the weather path (includes the current step)."""
    out = frame
    for column, prefix in WEATHER_EMA_SOURCES:
        if column not in out.columns:
            continue
        source = pd.to_numeric(out[column], errors="coerce")
        grouped = out.groupby(group_cols, group_keys=False)[column]
        for span, periods in WEATHER_EMA_SPAN_PERIODS.items():
            min_periods = max(3, periods // 2)
            ema = grouped.transform(
                lambda s, p=periods, m=min_periods: pd.to_numeric(s, errors="coerce").ewm(
                    span=p, adjust=False, min_periods=m
                ).mean()
            )
            out[f"{prefix}_ema_{span}"] = ema.fillna(source)
        if prefix in {"temp", "shortwave", "wind"} and f"{prefix}_ema_3h" in out.columns:
            out[f"{prefix}_ema_resid_3h"] = (source - out[f"{prefix}_ema_3h"]).fillna(0)
    return out


def _ago_values(
    frame: pd.DataFrame,
    column: str,
    group_cols: list[str],
    *,
    periods: int,
    days: float,
) -> pd.Series:
    """Same clock time ``days`` ago: grouped shift first, then timestamp lookup."""
    empty = pd.Series(np.nan, index=frame.index, dtype="float64")
    if column not in frame.columns:
        return empty
    shifted = frame.groupby(group_cols, group_keys=False)[column].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(periods)
    )
    work = pd.DataFrame(
        {
            "region_id": frame["region_id"].astype(str) if "region_id" in frame.columns else "gb",
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
            "_val": pd.to_numeric(frame[column], errors="coerce"),
        },
        index=frame.index,
    )
    table = (
        work.dropna(subset=["_val"])
        .drop_duplicates(["region_id", "timestamp"], keep="last")
        .set_index(["region_id", "timestamp"])["_val"]
    )
    keys = pd.MultiIndex.from_arrays(
        [work["region_id"], work["timestamp"] - pd.Timedelta(days=days)],
        names=["region_id", "timestamp"],
    )
    looked = pd.Series(table.reindex(keys).to_numpy(), index=frame.index, dtype="float64")
    return pd.to_numeric(shifted, errors="coerce").combine_first(looked)


def _week_ago_values(frame: pd.DataFrame, column: str, group_cols: list[str]) -> pd.Series:
    """Same clock time last week: grouped shift first, then timestamp − 7d lookup."""
    return _ago_values(frame, column, group_cols, periods=PERIODS_7D, days=7)


def _yesterday_values(frame: pd.DataFrame, column: str, group_cols: list[str]) -> pd.Series:
    """Same clock time yesterday: grouped shift first, then timestamp − 1d lookup."""
    return _ago_values(frame, column, group_cols, periods=PERIODS_24H, days=1)


def _add_week_engineered(frame: pd.DataFrame) -> pd.DataFrame:
    """Week-over-week and yesterday weather deltas, plus 14d/3w and 1d demand extras."""
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if "region_id" not in out.columns:
        out["region_id"] = "gb"
    if "timestamp" not in out.columns:
        return out
    group_cols = _trajectory_group_columns(out)
    out = out.sort_values([*group_cols, "timestamp"])

    def rolling_mean(column: str, periods: int, min_periods: int) -> pd.Series:
        return out.groupby(group_cols, group_keys=False)[column].transform(
            lambda s: pd.to_numeric(s, errors="coerce").rolling(periods, min_periods=min_periods).mean()
        )

    if "shortwave_radiation" in out.columns:
        sw = pd.to_numeric(out["shortwave_radiation"], errors="coerce")
        out["shortwave_mean_24h"] = rolling_mean("shortwave_radiation", PERIODS_24H, PERIODS_12H).fillna(sw)
        out["shortwave_delta_7d"] = sw - _week_ago_values(out, "shortwave_radiation", group_cols)
    if "wind_speed_10m" in out.columns:
        wind = pd.to_numeric(out["wind_speed_10m"], errors="coerce")
        out["wind_mean_24h"] = rolling_mean("wind_speed_10m", PERIODS_24H, PERIODS_12H).fillna(wind)
        out["wind_delta_7d"] = wind - _week_ago_values(out, "wind_speed_10m", group_cols)
    if "cloud_cover" in out.columns:
        out["cloud_delta_7d"] = pd.to_numeric(out["cloud_cover"], errors="coerce") - _week_ago_values(
            out, "cloud_cover", group_cols
        )
    if "temperature_2m" in out.columns:
        out["temp_delta_7d"] = pd.to_numeric(out["temperature_2m"], errors="coerce") - _week_ago_values(
            out, "temperature_2m", group_cols
        )
    if "temp_mean_24h" in out.columns:
        out["temp_mean_24h_delta_7d"] = pd.to_numeric(out["temp_mean_24h"], errors="coerce") - _week_ago_values(
            out, "temp_mean_24h", group_cols
        )
    if "apparent_temperature" in out.columns:
        out["apparent_temp_delta_7d"] = pd.to_numeric(out["apparent_temperature"], errors="coerce") - _week_ago_values(
            out, "apparent_temperature", group_cols
        )
    if "hdd" in out.columns:
        out["hdd_delta_7d"] = pd.to_numeric(out["hdd"], errors="coerce") - _week_ago_values(out, "hdd", group_cols)
    if "cdd" in out.columns:
        out["cdd_delta_7d"] = pd.to_numeric(out["cdd"], errors="coerce") - _week_ago_values(out, "cdd", group_cols)

    if "shortwave_radiation" in out.columns:
        out["shortwave_delta_1d"] = pd.to_numeric(out["shortwave_radiation"], errors="coerce") - _yesterday_values(
            out, "shortwave_radiation", group_cols
        )
    if "wind_speed_10m" in out.columns:
        out["wind_delta_1d"] = pd.to_numeric(out["wind_speed_10m"], errors="coerce") - _yesterday_values(
            out, "wind_speed_10m", group_cols
        )
    if "cloud_cover" in out.columns:
        out["cloud_delta_1d"] = pd.to_numeric(out["cloud_cover"], errors="coerce") - _yesterday_values(
            out, "cloud_cover", group_cols
        )
    if "temperature_2m" in out.columns:
        out["temp_delta_1d"] = pd.to_numeric(out["temperature_2m"], errors="coerce") - _yesterday_values(
            out, "temperature_2m", group_cols
        )
    if "temp_mean_24h" in out.columns:
        out["temp_mean_24h_delta_1d"] = pd.to_numeric(out["temp_mean_24h"], errors="coerce") - _yesterday_values(
            out, "temp_mean_24h", group_cols
        )
    if "apparent_temperature" in out.columns:
        out["apparent_temp_delta_1d"] = pd.to_numeric(out["apparent_temperature"], errors="coerce") - _yesterday_values(
            out, "apparent_temperature", group_cols
        )
    if "hdd" in out.columns:
        out["hdd_delta_1d"] = pd.to_numeric(out["hdd"], errors="coerce") - _yesterday_values(out, "hdd", group_cols)
    if "cdd" in out.columns:
        out["cdd_delta_1d"] = pd.to_numeric(out["cdd"], errors="coerce") - _yesterday_values(out, "cdd", group_cols)

    def lag_from_demand(periods: int) -> pd.Series | None:
        if "demand_mw" not in out.columns:
            return None
        return out.groupby(group_cols, group_keys=False)["demand_mw"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(periods)
        )

    def fill_lag(
        column: str,
        periods: int,
        source_lag: str | None = None,
        *,
        ago_days: int = 7,
    ) -> pd.Series | None:
        filled = pd.to_numeric(out[column], errors="coerce") if column in out.columns else None
        from_demand = lag_from_demand(periods)
        if from_demand is not None:
            filled = from_demand if filled is None else filled.combine_first(from_demand)
        if source_lag is not None and source_lag in out.columns:
            ago = (
                _yesterday_values(out, source_lag, group_cols)
                if ago_days == 1
                else _week_ago_values(out, source_lag, group_cols)
            )
            filled = ago if filled is None else filled.combine_first(ago)
        if filled is not None:
            out[column] = filled
        return filled

    lag7 = pd.to_numeric(out["demand_lag_336"], errors="coerce") if "demand_lag_336" in out.columns else None
    lag14 = fill_lag("demand_lag_672", PERIODS_14D, "demand_lag_336")
    lag21 = fill_lag("demand_lag_1008", PERIODS_21D, "demand_lag_672")
    if lag7 is not None and lag14 is not None:
        out["demand_delta_14d_7d"] = lag14 - lag7
    if lag7 is not None and lag14 is not None and lag21 is not None:
        out["demand_mean_3w"] = (lag7 + lag14 + lag21) / 3.0
        out["demand_accel_3w"] = lag7 - (2.0 * lag14) + lag21
    lag_yday = fill_lag("demand_lag_48", PERIODS_24H)
    lag_2d = fill_lag("demand_lag_96", PERIODS_48H, "demand_lag_48", ago_days=1)
    if lag_yday is not None and lag_2d is not None:
        out["demand_delta_1d"] = lag_yday - lag_2d
    return out


def attach_lookback_weather(
    frame: pd.DataFrame,
    weather: pd.DataFrame | None,
    *,
    days: int = 8,
) -> pd.DataFrame:
    """Prepend prior weather so week-over-week deltas exist on a short scored week."""
    if frame is None or frame.empty or weather is None or weather.empty:
        return frame
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    hist = weather.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
    if "region_id" not in out.columns:
        out["region_id"] = "gb"
    if "region_id" not in hist.columns:
        hist["region_id"] = "gb"
    start = out["timestamp"].min()
    hist = hist.loc[(hist["timestamp"] >= start - pd.Timedelta(days=int(days))) & (hist["timestamp"] < start)]
    if hist.empty:
        return out
    keep = [
        column
        for column in ["region_id", "timestamp", "settlement_date", "settlement_period", *HOURLY_VARS]
        if column in hist.columns
    ]
    hist = hist.loc[:, keep]
    return pd.concat([hist, out], ignore_index=True, sort=False)


def attach_lookback_demand(
    frame: pd.DataFrame,
    demand: pd.DataFrame | None,
    *,
    days: int = 22,
) -> pd.DataFrame:
    """Prepend prior actuals so 3-week demand lags exist on a short scored week."""
    if frame is None or frame.empty or demand is None or demand.empty:
        return frame
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    hist = demand.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
    if "region_id" not in out.columns:
        out["region_id"] = "gb"
    if "region_id" not in hist.columns:
        hist["region_id"] = "gb"
    if "demand_mw" not in hist.columns and "actual_mw" in hist.columns:
        hist["demand_mw"] = pd.to_numeric(hist["actual_mw"], errors="coerce")
    if "demand_mw" not in hist.columns:
        return out
    start = out["timestamp"].min()
    hist = hist.loc[(hist["timestamp"] >= start - pd.Timedelta(days=int(days))) & (hist["timestamp"] < start)]
    if hist.empty:
        return out
    keep = [column for column in ["region_id", "timestamp", "demand_mw"] if column in hist.columns]
    return pd.concat([hist.loc[:, keep], out], ignore_index=True, sort=False)


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
