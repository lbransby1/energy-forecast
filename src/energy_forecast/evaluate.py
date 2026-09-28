"""Backtest and slice GB demand quantile forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.features import FEATURE_COLUMNS
from energy_forecast.model import QuantileModels, fit_models, predict_quantiles
from energy_forecast.modes import MEDIUM, MODE_SPEC, SHORT, resample_to_mode
from energy_forecast.settlement import LONDON_TZ

NEXT_HOUR_LEAD_HOURS = 1.0
NEXT_DAY_LEAD_HOURS = 24.0

SEASON = {
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


def assign_horizon(lead_hours: pd.Series) -> pd.Series:
    """Label next hour / next day / next week from live lead time."""
    hours = pd.to_numeric(lead_hours, errors="coerce").fillna(0.0)
    labels = pd.Series("next_week", index=lead_hours.index, dtype=object)
    labels = labels.mask(hours <= NEXT_DAY_LEAD_HOURS, "next_day")
    labels = labels.mask(hours <= NEXT_HOUR_LEAD_HOURS, "next_hour")
    return labels


def assign_product(lead_hours: pd.Series, *, source: str = "live") -> pd.Series:
    """Product label for a scored frame.

    Live leads are continuous. Previous Runs only has day 0, 1, … 7, so lead 0
    is next-day weather (issued same-day forecast), not the next-hour model.
    """
    if source == "previous_runs":
        hours = pd.to_numeric(lead_hours, errors="coerce").fillna(0.0)
        return pd.Series(
            np.where(hours <= NEXT_DAY_LEAD_HOURS, "next_day", "next_week"),
            index=lead_hours.index,
            dtype=object,
        )
    return assign_horizon(lead_hours)


def score(frame: pd.DataFrame, models: QuantileModels) -> pd.DataFrame:
    """Attach quantile predictions, errors, and analysis slices."""
    features = list(models.features or FEATURE_COLUMNS)
    work = frame.dropna(subset=["demand_mw", *features]).copy()
    out = predict_quantiles(work, models)
    out["error"] = out["demand_mw"] - out["p50"]
    out["abs_error"] = out["error"].abs()
    out["ape"] = out["abs_error"] / out["demand_mw"].clip(lower=1)
    out["covered"] = (out["demand_mw"] >= out["p10"]) & (out["demand_mw"] <= out["p90"])
    out["interval_width"] = out["p90"] - out["p10"]
    out["climatology_mw"] = _row_climatology(out, models.climatology)
    out["clim_abs_error"] = (out["demand_mw"] - out["climatology_mw"]).abs()
    timestamp = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    out["year"] = timestamp.dt.year
    out["month"] = timestamp.dt.month
    out["season"] = out["month"].map(SEASON)
    out["weekday"] = np.where(out["is_weekend"] == 1, "weekend", "weekday")
    out["holiday"] = np.where(out["is_holiday"] == 1, "holiday", "normal")
    out["period_band"] = out["settlement_period"].map(_period_band)
    if "lead_hours" in out.columns:
        out["lead_band"] = out["lead_hours"].map(_lead_band)
        out["horizon"] = assign_horizon(out["lead_hours"])
    out["temp_band"] = pd.cut(
        out["temperature_2m"],
        bins=[-50, 5, 10, 15, 20, 50],
        labels=["freezing", "cold", "mild", "warm", "hot"],
    ).astype(str)
    return out


def overall_metrics(scored: pd.DataFrame, *, scale_mse: float | None = None) -> pd.DataFrame:
    return _metrics_block(scored, split="all", scale_mse=scale_mse)


def score_production_holdouts(
    week_frame: pd.DataFrame,
    hour_frame: pd.DataFrame,
    week_models: QuantileModels,
    hour_models: QuantileModels,
    day_models: QuantileModels | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Score the saved hour, day, and week models on their training holdouts.

    Next hour uses ERA5 plus short INDO lags. Next day uses Previous Runs
    leads 0–24h. Next week uses leads 24–168h.
    """
    hour_start = pd.Timestamp(hour_models.metadata["holdout_start"])
    week_start = pd.Timestamp(week_models.metadata["holdout_start"])
    hour_scored = score(hour_frame[hour_frame["timestamp"] >= hour_start], hour_models)
    hour_scored = hour_scored.assign(product="next_hour")
    week_scored = score(week_frame[week_frame["timestamp"] >= week_start], week_models)
    week_scored = week_scored.assign(
        product=assign_product(week_scored["lead_hours"], source="previous_runs")
    )
    if day_models is not None:
        day_start = pd.Timestamp(day_models.metadata["holdout_start"])
        day_holdout = week_frame[week_frame["timestamp"] >= day_start].copy()
        lead = pd.to_numeric(day_holdout["lead_hours"], errors="coerce").fillna(0.0)
        day_scored = score(day_holdout[lead <= NEXT_DAY_LEAD_HOURS], day_models)
        day_scored = day_scored.assign(product="next_day")
    else:
        day_scored = week_scored[week_scored["product"] == "next_day"].copy()
    scored = {
        "next_hour": hour_scored,
        "next_day": day_scored,
        "next_week": week_scored[week_scored["product"] == "next_week"].copy(),
    }
    metrics = pd.concat(
        [_metrics_block(scored[name], split=name) for name in ("next_hour", "next_day", "next_week") if not scored[name].empty],
        ignore_index=True,
    )
    return metrics, scored


def score_forecast_modes(
    week_frame: pd.DataFrame,
    week_models: QuantileModels,
    day_models: QuantileModels,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Holdout scores at the two display resolutions.

    Short: day model, leads 0–24h, resampled to hourly.
    Medium: week model, all leads, resampled to 6-hourly.
    """
    day_start = pd.Timestamp(day_models.metadata["holdout_start"])
    week_start = pd.Timestamp(week_models.metadata["holdout_start"])
    day_holdout = week_frame[week_frame["timestamp"] >= day_start].copy()
    lead = pd.to_numeric(day_holdout["lead_hours"], errors="coerce").fillna(0.0)
    short = resample_to_mode(score(day_holdout[lead <= NEXT_DAY_LEAD_HOURS], day_models), SHORT)
    week_holdout = week_frame[week_frame["timestamp"] >= week_start].copy()
    medium = resample_to_mode(score(week_holdout, week_models), MEDIUM)
    scored = {SHORT: short, MEDIUM: medium}
    train_history = week_frame[week_frame["timestamp"] < min(day_start, week_start)]
    metrics = pd.concat(
        [
            _metrics_block(
                scored[name],
                split=name,
                scale_mse=_naive_scale_mse(train_history, MODE_SPEC[name]["freq"], MODE_SPEC[name]["season"]),
            )
            for name in (SHORT, MEDIUM)
            if not scored[name].empty
        ],
        ignore_index=True,
    )
    return metrics, scored


def product_slices(scored: pd.DataFrame) -> pd.DataFrame:
    """Usual slices plus lead/horizon when those columns exist."""
    columns = ["season", "period_band", "weekday", "holiday", "temp_band"]
    if "lead_band" in scored.columns and scored["lead_band"].nunique() > 1:
        columns.append("lead_band")
    tables = [slice_metrics(scored, column) for column in columns if column in scored.columns]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def slice_metrics(scored: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [_metrics_block(group, split=str(key)) for key, group in scored.groupby(column, observed=True)]
    table = pd.concat(rows, ignore_index=True)
    table.insert(0, "slice", column)
    return table


def daily_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    daily = scored.groupby(scored["settlement_date"].dt.normalize()).agg(
        demand_mw=("demand_mw", "mean"),
        p50=("p50", "mean"),
        p10=("p10", "mean"),
        p90=("p90", "mean"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
        mape=("ape", "mean"),
        coverage=("covered", "mean"),
        interval_width=("interval_width", "mean"),
        temperature_2m=("temperature_2m", "mean"),
        is_holiday=("is_holiday", "max"),
        is_weekend=("is_weekend", "max"),
    )
    daily = daily.reset_index().rename(columns={"settlement_date": "date"})
    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    daily["season"] = daily["month"].map(SEASON)
    return daily.sort_values("date")


def best_worst_days(daily: pd.DataFrame, n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = daily.dropna(subset=["mae"]).sort_values("mae")
    cols = ["date", "year", "season", "mae", "bias", "mape", "coverage", "demand_mw", "p50", "temperature_2m"]
    cols = [column for column in cols if column in ranked.columns]
    return ranked.head(n).loc[:, cols], ranked.tail(n).iloc[::-1].loc[:, cols]


def year_holdout(frame: pd.DataFrame, *, n_estimators: int = 400) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Train on all earlier years, then test each later calendar year."""
    work = frame.dropna(subset=["demand_mw", *FEATURE_COLUMNS]).copy()
    years = sorted(pd.to_datetime(work["timestamp"], utc=True).dt.year.unique())
    metrics_rows: list[pd.DataFrame] = []
    scored_by_split: dict[str, pd.DataFrame] = {}
    for test_year in years:
        train_years = [year for year in years if year < test_year]
        if not train_years:
            continue
        train = work[pd.to_datetime(work["timestamp"], utc=True).dt.year.isin(train_years)]
        test = work[pd.to_datetime(work["timestamp"], utc=True).dt.year == test_year]
        if train.empty or test.empty:
            continue
        models = fit_models(
            train,
            n_estimators=n_estimators,
            metadata={"eval_split": f"train_{train_years[0]}_{train_years[-1]}_test_{test_year}"},
        )
        scored = score(test, models)
        label = f"train {train_years[0]}-{train_years[-1]} -> {test_year}"
        scored_by_split[label] = scored
        metrics_rows.append(_metrics_block(scored, split=label))
    return pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame(), scored_by_split


def walk_forward_quarters(frame: pd.DataFrame, *, n_estimators: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window quarterly backtest."""
    work = frame.dropna(subset=["demand_mw", *FEATURE_COLUMNS]).copy()
    work = work.sort_values("timestamp")
    period = pd.to_datetime(work["timestamp"], utc=True).dt.to_period("Q")
    work = work.assign(_quarter=period.astype(str))
    quarters = list(dict.fromkeys(work["_quarter"].tolist()))
    metric_rows: list[pd.DataFrame] = []
    scored_parts: list[pd.DataFrame] = []
    for index in range(1, len(quarters)):
        train_q = quarters[:index]
        test_q = quarters[index]
        train = work[work["_quarter"].isin(train_q)]
        test = work[work["_quarter"] == test_q]
        if len(train) < 48 * 60 or test.empty:
            continue
        models = fit_models(train, n_estimators=n_estimators, metadata={"test_quarter": test_q})
        scored = score(test, models)
        scored = scored.assign(split=f"{train_q[0]}-{train_q[-1]} -> {test_q}")
        scored_parts.append(scored)
        metric_rows.append(_metrics_block(scored, split=scored["split"].iloc[0]))
    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    return metrics, scored


def _metrics_block(
    scored: pd.DataFrame,
    split: str,
    *,
    scale_mse: float | None = None,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    actual = scored["demand_mw"].to_numpy()
    p50 = scored["p50"].to_numpy()
    error = actual - p50
    skill = 1.0 - float(scored["abs_error"].mean() / scored["clim_abs_error"].mean())
    row = {
        "split": split,
        "n": int(len(scored)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error) / np.clip(actual, 1, None))),
        "bias": float(np.mean(error)),
        "pinball_10": _pinball(actual, scored["p10"].to_numpy(), 0.1),
        "pinball_50": _pinball(actual, p50, 0.5),
        "pinball_90": _pinball(actual, scored["p90"].to_numpy(), 0.9),
        "coverage_80": float(scored["covered"].mean()),
        "interval_width": float(scored["interval_width"].mean()),
        "skill_vs_climatology": skill,
    }
    if scale_mse is not None and np.isfinite(scale_mse) and scale_mse > 0:
        row["rmsse"] = float(np.sqrt(np.mean(error**2) / scale_mse))
        row["wrmsse"] = _wrmsse(scored, scale_mse)
    return pd.DataFrame([row])


def _naive_scale_mse(frame: pd.DataFrame, freq: str, season: int) -> float:
    """In-sample MSE of a weekly seasonal naive, at the display frequency."""
    if frame.empty or "demand_mw" not in frame.columns:
        return float("nan")
    work = frame.dropna(subset=["demand_mw", "timestamp"]).drop_duplicates("timestamp")
    if work.empty:
        return float("nan")
    binned = pd.to_datetime(work["timestamp"], utc=True).dt.floor(freq)
    series = work.assign(_bin=binned).groupby("_bin")["demand_mw"].mean().to_numpy()
    if series.size <= season:
        return float("nan")
    resid = series[season:] - series[:-season]
    return float(np.mean(resid**2))


def _wrmsse(scored: pd.DataFrame, scale_mse: float, group: str = "period_band") -> float:
    """Demand-weighted RMSSE across time-of-day bands (M5-style, one GB series)."""
    if group not in scored.columns:
        actual = scored["demand_mw"].to_numpy()
        weights = actual / np.clip(actual.sum(), 1, None)
        return float(np.sqrt(np.sum(weights * (actual - scored["p50"].to_numpy()) ** 2) / scale_mse))
    scores: list[float] = []
    weights: list[float] = []
    for _, group_frame in scored.groupby(group, observed=True):
        actual = group_frame["demand_mw"].to_numpy()
        pred = group_frame["p50"].to_numpy()
        if actual.size == 0 or not np.isfinite(actual).any():
            continue
        scores.append(float(np.sqrt(np.mean((actual - pred) ** 2) / scale_mse)))
        weights.append(float(np.nansum(np.abs(actual))))
    if not scores:
        return float("nan")
    weight = np.asarray(weights, dtype=float)
    weight = weight / np.clip(weight.sum(), 1, None)
    return float(np.dot(weight, np.asarray(scores)))


def _pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def _lead_band(hours: float) -> str:
    if hours <= 24:
        return "0-24h"
    if hours <= 72:
        return "24-72h"
    return "72-168h"


def _period_band(period: int) -> str:
    if period <= 12:
        return "night"
    if period <= 24:
        return "morning"
    if period <= 32:
        return "afternoon"
    if period <= 40:
        return "evening_peak"
    return "late_evening"


def _row_climatology(frame: pd.DataFrame, climatology: pd.DataFrame) -> pd.Series:
    if climatology.empty:
        return pd.Series(np.nan, index=frame.index)
    merged = frame.merge(
        climatology,
        on=["region_id", "month", "day_of_week", "settlement_period"],
        how="left",
    )
    return merged["climatology_mw"].to_numpy()
