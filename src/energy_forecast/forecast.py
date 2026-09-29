"""Public forecast API: GB quantiles plus live INDO validation."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from energy_forecast.data.demand import load_demand
from energy_forecast.data.elexon import fetch_indo
from energy_forecast.data.weather import (
    PREVIOUS_RUNS_START,
    combine_archive_and_forecast,
    expand_previous_runs,
    fetch_forecast_weather,
    fetch_historical_weather,
    fetch_previous_runs,
    load_cached_weather,
)
from energy_forecast.evaluate import NEXT_DAY_LEAD_HOURS, NEXT_HOUR_LEAD_HOURS, assign_horizon
from energy_forecast.features import HOUR_FEATURE_COLUMNS, attach_demand_lags, build_features
from energy_forecast.modes import MEDIUM, MODE_SPEC, SHORT, resample_to_mode
from energy_forecast.model import (
    QuantileModels,
    load_day_models,
    load_hour_models,
    load_models,
    predict_quantiles,
    prepare_training_frame,
)
from energy_forecast.paths import DATA_MODELS
from energy_forecast.regions import SERIES_ID
from energy_forecast.settlement import LONDON_TZ, normalize_settlement_date

# Extra forecast days so the last horizon hour still has a 24h forward window.
FORWARD_BUFFER_HOURS = 24
ARCHIVE_LAG_DAYS = 7
# Hour pack must not issue if Insights INDO is older than this.
LIVE_INDO_MAX_AGE_MINUTES = 90

FORECAST_COLUMNS = [
    "timestamp",
    "settlement_date",
    "settlement_period",
    "temperature_2m",
    "lead_hours",
    "horizon",
    "mode",
    "temp_mean_fwd_12h",
    "temp_mean_fwd_24h",
    "temp_delta_24h",
    "temp_anom_seasonal",
    "temp_anom_7d",
    "p10",
    "p50",
    "p90",
    "interval_width",
    "actual_mw",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_48",
    "demand_lag_336",
    "last_indo_timestamp",
    "last_indo_mw",
]


def prepare_history(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    include_live: bool = True,
    weather_source: str = "previous_runs",
) -> pd.DataFrame:
    """Join weather with GB demand (NESO + live INDO) and build features.

    Default weather is Open-Meteo Previous Runs expanded to lead days 0-7.
    """
    demand = attach_demand_lags(load_demand(include_live=include_live))
    if end_date is None:
        archive_end = min(
            pd.Timestamp(demand["settlement_date"].max()),
            pd.Timestamp(date.today()) - pd.Timedelta(days=6),
        )
        end_date = archive_end.strftime("%Y-%m-%d")
    if weather_source == "previous_runs":
        if start_date is None:
            start_date = PREVIOUS_RUNS_START
        weather = expand_previous_runs(fetch_previous_runs([SERIES_ID], start_date, end_date))
        return prepare_training_frame(demand, weather)
    if start_date is None:
        start_date = demand["settlement_date"].min().strftime("%Y-%m-%d")
    try:
        weather = load_cached_weather([SERIES_ID], source="archive")
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if weather["settlement_date"].min() > start or weather["settlement_date"].max() < end:
            weather = fetch_historical_weather([SERIES_ID], start_date, end_date)
    except FileNotFoundError:
        weather = fetch_historical_weather([SERIES_ID], start_date, end_date)
    return prepare_training_frame(demand, weather)


def prepare_live_frame(
    *,
    horizon_hours: int = 168,
    past_days: int = ARCHIVE_LAG_DAYS,
    climatology: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Archive weather + recent/forecast weather + demand, with trajectory features.

    Features are built on the full continuous path so an incoming front is
    visible in ``temp_mean_fwd_*`` and ``temp_delta_*`` at forecast time.
    """
    forecast_days = max(2, int((horizon_hours + FORWARD_BUFFER_HOURS + 23) // 24))
    forecast_weather = fetch_forecast_weather(
        [SERIES_ID],
        forecast_days=forecast_days,
        past_days=past_days,
    )
    demand = load_demand(include_live=True)
    archive_weather = _load_archive_weather(demand)
    weather = combine_archive_and_forecast(archive_weather, forecast_weather)
    merged = _merge_weather_left(weather, demand)
    featured = build_features(merged, climatology=climatology)
    featured = _persist_short_lags(featured)
    now = pd.Timestamp.now(tz=LONDON_TZ).floor("30min")
    lead = (featured["timestamp"] - now).dt.total_seconds() / 3600.0
    featured["lead_hours"] = lead.clip(lower=0.0)
    return featured


def run_forecast(
    *,
    mode: str = "both",
    horizon_hours: int | None = None,
    model_dir: Path | None = None,
    models: QuantileModels | None = None,
    freq: str | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Live forecast: next 24h and next 7 days.

    Default ``freq`` is the display grid (hourly). Pass ``30min`` to keep
    settlement periods. Holdout metrics stay on the 6-hour medium grid.
    """
    raw = _run_half_hourly(
        horizon_hours=horizon_hours or MODE_SPEC[MEDIUM]["horizon_hours"],
        model_dir=model_dir,
        models=models,
    )
    short = resample_to_mode(
        _limit_horizon(raw, MODE_SPEC[SHORT]["horizon_hours"]),
        SHORT,
        freq=freq or MODE_SPEC[SHORT]["display_freq"],
    )
    medium = resample_to_mode(
        _limit_horizon(raw, MODE_SPEC[MEDIUM]["horizon_hours"]),
        MEDIUM,
        freq=freq or MODE_SPEC[MEDIUM]["display_freq"],
    )
    if mode == SHORT:
        return short
    if mode == MEDIUM:
        return medium
    if mode == "both":
        return {SHORT: short, MEDIUM: medium}
    raise ValueError("mode must be 'short', 'medium', or 'both'")


def _run_half_hourly(
    *,
    horizon_hours: int,
    model_dir: Path | None,
    models: QuantileModels | None,
) -> pd.DataFrame:
    week_models = models or load_models(model_dir or DATA_MODELS)
    hour_models = _try_load_hour_models()
    day_models = _try_load_day_models()
    featured = prepare_live_frame(horizon_hours=horizon_hours, climatology=week_models.climatology)
    horizon = pd.Timedelta(hours=horizon_hours)
    now = pd.Timestamp.now(tz=LONDON_TZ).floor("30min")
    future_mask = featured["timestamp"] >= now
    if future_mask.any():
        scored = featured.loc[future_mask].copy()
        scored = scored[scored["timestamp"] < now + horizon]
    else:
        scored = featured.copy()
        scored = scored[scored["timestamp"] < scored["timestamp"].min() + horizon]

    predicted = _predict_by_horizon(scored, week_models, hour_models, day_models)
    predicted["horizon"] = assign_horizon(predicted["lead_hours"])
    if {"p10", "p90"}.issubset(predicted.columns):
        predicted["interval_width"] = predicted["p90"] - predicted["p10"]
    predicted = attach_indo_actuals(predicted)
    keep = [column for column in FORECAST_COLUMNS if column in predicted.columns]
    return predicted.loc[:, keep].sort_values("timestamp").reset_index(drop=True)


def run_live_detail(*, horizon_hours: int = 168, model_dir: Path | None = None) -> pd.DataFrame:
    """Same live path as ``run_forecast``, but keep features and which model scored each row."""
    week_models = load_models(model_dir or DATA_MODELS)
    hour_models = _try_load_hour_models()
    day_models = _try_load_day_models()
    featured = prepare_live_frame(horizon_hours=horizon_hours, climatology=week_models.climatology)
    horizon = pd.Timedelta(hours=horizon_hours)
    now = pd.Timestamp.now(tz=LONDON_TZ).floor("30min")
    future_mask = featured["timestamp"] >= now
    if future_mask.any():
        scored = featured.loc[future_mask].copy()
        scored = scored[scored["timestamp"] < now + horizon]
    else:
        scored = featured.copy()
        scored = scored[scored["timestamp"] < scored["timestamp"].min() + horizon]
    predicted = _predict_by_horizon(scored, week_models, hour_models, day_models)
    predicted["horizon"] = assign_horizon(predicted["lead_hours"])
    if {"p10", "p90"}.issubset(predicted.columns):
        predicted["interval_width"] = predicted["p90"] - predicted["p10"]
    return predicted.sort_values("timestamp").reset_index(drop=True)


def run_pack(
    *,
    role: str,
    horizon_hours: float,
    origin: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one pack on the live frame. Returns (forecast, frozen features).

    Day and week packs score ``origin`` plus ``horizon_hours`` (Monday / midnight),
    not “from now”, so a late re-issue still covers that calendar window.
    """
    if role not in {"hour", "day", "week"}:
        raise ValueError("role must be 'hour', 'day', or 'week'")
    week_models = load_models()
    if role == "hour":
        models = _try_load_hour_models() or week_models
        fetch_hours = 2
    elif role == "day":
        models = _try_load_day_models() or week_models
        fetch_hours = 24
    else:
        models = week_models
        fetch_hours = 168
    featured = prepare_live_frame(horizon_hours=fetch_hours, climatology=week_models.climatology)
    now = pd.Timestamp.now(tz=LONDON_TZ).floor("30min")
    start = pd.Timestamp(origin) if origin is not None else now
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    horizon = pd.Timedelta(hours=float(horizon_hours))
    stamps = pd.to_datetime(featured["timestamp"], utc=True)
    scored = featured.loc[(stamps >= start) & (stamps < start + horizon)].copy()
    if role == "hour":
        target = now.tz_convert("UTC") if now.tzinfo else now.tz_localize("UTC")
        scored_ts = pd.to_datetime(scored["timestamp"], utc=True)
        one = scored.loc[scored_ts == target]
        scored = one if not one.empty else scored.head(1)
    if scored.empty:
        raise RuntimeError(f"run_pack({role!r}) produced no future rows")
    if role == "hour":
        last_indo_ts, last_indo_mw, prev_indo_mw = require_live_indo_lags()
        scored = scored.copy()
        scored["demand_lag_1"] = last_indo_mw
        scored["demand_lag_2"] = prev_indo_mw
    else:
        observed = featured.dropna(subset=["demand_mw"]) if "demand_mw" in featured.columns else featured.iloc[0:0]
        last_indo_ts = observed["timestamp"].max() if not observed.empty else pd.NaT
        last_indo_mw = (
            float(observed.loc[observed["timestamp"].idxmax(), "demand_mw"]) if not observed.empty else float("nan")
        )
    predicted = predict_quantiles(scored, models)
    predicted["horizon"] = assign_horizon(predicted["lead_hours"]) if "lead_hours" in predicted.columns else role
    if {"p10", "p90"}.issubset(predicted.columns):
        predicted["interval_width"] = predicted["p90"] - predicted["p10"]
    predicted["last_indo_timestamp"] = last_indo_ts
    predicted["last_indo_mw"] = last_indo_mw
    predicted = attach_indo_actuals(predicted)
    keep = [column for column in FORECAST_COLUMNS if column in predicted.columns]
    forecast = predicted.loc[:, keep].sort_values("timestamp").reset_index(drop=True)
    frozen_wanted = list(_FROZEN_COLUMNS)
    if role == "hour":
        frozen_wanted = list(dict.fromkeys([*frozen_wanted, *HOUR_FEATURE_COLUMNS]))
    frozen_cols = [column for column in frozen_wanted if column in scored.columns]
    frozen = scored.loc[:, frozen_cols].sort_values("timestamp").reset_index(drop=True)
    frozen["last_indo_timestamp"] = last_indo_ts
    frozen["last_indo_mw"] = last_indo_mw
    return forecast, frozen


def require_live_indo_lags() -> tuple[pd.Timestamp, float, float]:
    """Latest published Insights INDO as lag_1, previous print as lag_2.

    Raises if Insights has nothing recent enough for a 30-minute call.
    """
    end = date.today()
    start = end - timedelta(days=1)
    indo = fetch_indo(start.isoformat(), end.isoformat())
    if indo.empty:
        raise RuntimeError("Insights INDO returned no rows for short lags")
    indo = indo.sort_values("timestamp")
    last = indo.iloc[-1]
    prev = indo.iloc[-2] if len(indo) >= 2 else last
    last_ts = pd.to_datetime(last["timestamp"], utc=True)
    age = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60.0
    if age > LIVE_INDO_MAX_AGE_MINUTES:
        raise RuntimeError(
            f"latest INDO is {age:.0f} min old ({last_ts.isoformat()}); not issuing 30-minute call"
        )
    return last_ts, float(last["demand_mw"]), float(prev["demand_mw"])


_FROZEN_COLUMNS = (
    "timestamp",
    "settlement_date",
    "settlement_period",
    "lead_hours",
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "hdd",
    "cdd",
    "hdd_sum_24h",
    "cdd_sum_24h",
    "temp_mean_fwd_12h",
    "temp_mean_fwd_24h",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_48",
    "demand_lag_336",
    "is_weekend",
    "is_holiday",
    "day_of_week",
)


def _limit_horizon(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    if frame.empty or "lead_hours" not in frame.columns:
        return frame
    lead = pd.to_numeric(frame["lead_hours"], errors="coerce").fillna(0.0)
    return frame.loc[lead < hours].copy()


def attach_indo_actuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Join Insights INDO onto rows whose settlement date is not in the future."""
    out = frame.copy()
    if out.empty:
        out["actual_mw"] = np.nan
        return out
    today = pd.Timestamp(date.today()).normalize()
    past = out[pd.to_datetime(out["settlement_date"]).dt.normalize() <= today]
    if past.empty:
        out["actual_mw"] = np.nan
        return out
    start = pd.to_datetime(past["settlement_date"]).min().strftime("%Y-%m-%d")
    end = pd.to_datetime(past["settlement_date"]).max().strftime("%Y-%m-%d")
    indo = fetch_indo(start, end)
    if indo.empty:
        out["actual_mw"] = np.nan
        return out

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    indo = indo.copy()
    indo["timestamp"] = pd.to_datetime(indo["timestamp"], utc=True)

    exact = indo.loc[:, ["timestamp", "demand_mw"]].rename(columns={"demand_mw": "actual_mw"})
    merged = out.drop(columns=["actual_mw"], errors="ignore").merge(exact, on="timestamp", how="left")
    half_hourly = (
        "settlement_period" in merged.columns
        and merged["settlement_period"].notna().any()
    )
    if half_hourly:
        return merged
    hourly = (
        indo.assign(timestamp=indo["timestamp"].dt.floor("1h"))
        .groupby("timestamp", as_index=False)["demand_mw"]
        .mean()
        .rename(columns={"demand_mw": "actual_mw"})
    )
    missing = merged["actual_mw"].isna()
    if missing.any():
        hour_map = hourly.set_index("timestamp")["actual_mw"]
        merged.loc[missing, "actual_mw"] = merged.loc[missing, "timestamp"].dt.floor("1h").map(hour_map)
    return merged


def validation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Score elapsed forecast rows against Insights INDO, overall and by horizon."""
    blocks = [_validation_block(frame, "all")]
    if "mode" in frame.columns:
        for name in (SHORT, MEDIUM):
            group = frame[frame["mode"] == name]
            if group.dropna(subset=["actual_mw", "p50"]).empty:
                continue
            blocks.append(_validation_block(group, name))
    return pd.concat(blocks, ignore_index=True)


def _validation_block(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    scored = frame.dropna(subset=["actual_mw", "p50"]).copy()
    if scored.empty:
        return pd.DataFrame(columns=["split", "metric", "value"])
    error = scored["actual_mw"] - scored["p50"]
    coverage = (
        float(((scored["actual_mw"] >= scored["p10"]) & (scored["actual_mw"] <= scored["p90"])).mean())
        if {"p10", "p90"}.issubset(scored.columns)
        else np.nan
    )
    return pd.DataFrame(
        [
            {"split": split, "metric": "n", "value": float(len(scored))},
            {"split": split, "metric": "mae", "value": float(np.mean(np.abs(error)))},
            {"split": split, "metric": "bias", "value": float(error.mean())},
            {"split": split, "metric": "coverage_80", "value": coverage},
        ]
    )


def _try_load_hour_models() -> QuantileModels | None:
    try:
        return load_hour_models()
    except FileNotFoundError:
        return None


def _try_load_day_models() -> QuantileModels | None:
    try:
        return load_day_models()
    except FileNotFoundError:
        return None


def _predict_by_horizon(
    frame: pd.DataFrame,
    week_models: QuantileModels,
    hour_models: QuantileModels | None,
    day_models: QuantileModels | None = None,
) -> pd.DataFrame:
    """Route next hour, next day, and next week to their own models."""
    if frame.empty or "lead_hours" not in frame.columns:
        return predict_quantiles(frame, week_models)
    lead = pd.to_numeric(frame["lead_hours"], errors="coerce").fillna(0.0)
    hour_mask = lead <= NEXT_HOUR_LEAD_HOURS if hour_models is not None else lead != lead
    day_mask = (lead > NEXT_HOUR_LEAD_HOURS) & (lead <= NEXT_DAY_LEAD_HOURS) if day_models is not None else lead != lead
    week_mask = ~(hour_mask | day_mask)
    parts: list[pd.DataFrame] = []
    if hour_models is not None and hour_mask.any():
        hour_part = predict_quantiles(frame.loc[hour_mask], hour_models)
        hour_part["model"] = "hour"
        parts.append(hour_part)
    if day_models is not None and day_mask.any():
        day_part = predict_quantiles(frame.loc[day_mask], day_models)
        day_part["model"] = "day"
        parts.append(day_part)
    if week_mask.any():
        week_part = predict_quantiles(frame.loc[week_mask], week_models)
        week_part["model"] = "week"
        parts.append(week_part)
    if not parts:
        return predict_quantiles(frame, week_models)
    return pd.concat(parts).sort_index()


def _load_archive_weather(demand: pd.DataFrame) -> pd.DataFrame:
    try:
        return load_cached_weather([SERIES_ID], source="archive")
    except FileNotFoundError:
        if demand.empty:
            return pd.DataFrame()
        return fetch_historical_weather(
            [SERIES_ID],
            demand["settlement_date"].min().strftime("%Y-%m-%d"),
            demand["settlement_date"].max().strftime("%Y-%m-%d"),
        )


def _persist_short_lags(frame: pd.DataFrame) -> pd.DataFrame:
    """On future rows, keep last published INDO in the 30/60-minute lags."""
    if frame.empty or "demand_mw" not in frame.columns:
        return frame
    out = frame.sort_values(["region_id", "timestamp"]).copy()
    for _, group in out.groupby("region_id", sort=False):
        observed = group.dropna(subset=["demand_mw"])
        if observed.empty:
            continue
        last = float(observed["demand_mw"].iloc[-1])
        prev = float(observed["demand_mw"].iloc[-2]) if len(observed) >= 2 else last
        future_idx = group.index[group["demand_mw"].isna()]
        if future_idx.empty:
            continue
        out.loc[future_idx, "demand_lag_1"] = last
        out.loc[future_idx, "demand_lag_2"] = prev
    return out


def _merge_weather_left(weather: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Keep every weather timestamp, including future hours with no demand yet."""
    left = weather.copy()
    left["settlement_date"] = normalize_settlement_date(left["settlement_date"])
    if demand.empty:
        left["demand_mw"] = np.nan
        return left
    right = demand.copy()
    right["settlement_date"] = normalize_settlement_date(right["settlement_date"])
    keys = ["region_id", "settlement_date", "settlement_period"]
    demand_cols = [column for column in [*keys, "demand_mw"] if column in right.columns]
    return left.merge(right.loc[:, demand_cols], on=keys, how="left")
