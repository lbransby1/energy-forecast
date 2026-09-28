"""Fixed diagnostic days: holiday, day-after, clock change, weather, control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_forecast.evaluate import overall_metrics, score
from energy_forecast.explain import OUTPUTS, explain_scored, mean_abs_shap
from energy_forecast.model import QuantileModels
from energy_forecast.modes import SHORT
from energy_forecast.settlement import LONDON_TZ

_HOURLY_MEAN = (
    "p10",
    "p50",
    "p90",
    "demand_mw",
    "mean_pred",
    "temperature_2m",
    "interval_width",
    "abs_error",
    "error",
    "climatology_mw",
    "demand_lag_48",
    "demand_lag_336",
)
_HOURLY_FIRST = ("settlement_date", "region_id", "is_weekend", "is_holiday")
LAG_FEATURES = ("demand_lag_1", "demand_lag_2", "demand_lag_48", "demand_lag_336")


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    why: str
    date: str | None = None
    in_holdout: bool = True


CASES = (
    Case(
        "control",
        "Median weekday",
        "Control: the holdout weekday whose MAE is closest to the median. ~700 MW is typical, not a success.",
        None,
    ),
    Case(
        "holiday",
        "Summer bank holiday",
        "Monday 31 Aug 2026. Demand drops. lag_336 is a working Monday, so the weekly lag pulls the level up.",
        "2026-08-31",
    ),
    Case(
        "after_holiday",
        "Tuesday after the bank holiday",
        "1 Sep 2026. A working Tuesday whose lag_48 is the holiday. This is the case the holiday slice hides.",
        "2026-09-01",
    ),
    Case(
        "clock_change",
        "Clocks forward",
        "29 Mar 2026, last Sunday of March. 46 settlement periods. Calendar features can mis-align for a day.",
        "2026-03-29",
        in_holdout=False,
    ),
    Case(
        "weather",
        "Hottest holdout weekday",
        "The warmest working day in the day-model holdout. The one place weather features should move the mean.",
        None,
    ),
)


def score_same_day(frame: pd.DataFrame, models: QuantileModels) -> pd.DataFrame:
    """Day-model scores on Previous Runs lead 0 (same-day issued weather).

    Returns half-hourly rows so SHAP still has the training features.
    Use ``to_hourly`` for the plots and the MAE table.
    """
    work = frame.copy()
    lead = pd.to_numeric(work["lead_hours"], errors="coerce")
    same_day = work.loc[lead == 0].copy()
    if same_day.empty:
        same_day = work.sort_values("lead_hours").drop_duplicates("timestamp")
    return score(same_day, models)


def to_hourly(scored: pd.DataFrame) -> pd.DataFrame:
    """Average half-hourly scores onto UTC hours (avoids London DST floor bugs)."""
    if scored.empty:
        return scored.copy()
    work = scored.copy()
    work["_bin"] = pd.to_datetime(work["timestamp"], utc=True).dt.floor("1h")
    agg: dict[str, str] = {}
    if "lead_hours" in work.columns:
        agg["lead_hours"] = "min"
    for column in _HOURLY_MEAN:
        if column in work.columns:
            agg[column] = "mean"
    for column in _HOURLY_FIRST:
        if column in work.columns:
            agg[column] = "first"
    out = work.groupby("_bin", as_index=False).agg(agg).rename(columns={"_bin": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    out["mode"] = SHORT
    if {"p10", "p90"}.issubset(out.columns):
        out["interval_width"] = out["p90"] - out["p10"]
    if "demand_mw" in out.columns and "p50" in out.columns:
        out["error"] = out["demand_mw"] - out["p50"]
        out["abs_error"] = out["error"].abs()
        out["ape"] = out["abs_error"] / out["demand_mw"].clip(lower=1)
        if {"p10", "p90"}.issubset(out.columns):
            out["covered"] = (out["demand_mw"] >= out["p10"]) & (out["demand_mw"] <= out["p90"])
        if "climatology_mw" in out.columns:
            out["clim_abs_error"] = (out["demand_mw"] - out["climatology_mw"]).abs()
    return out.sort_values("timestamp").reset_index(drop=True)


def resolve_cases(hourly: pd.DataFrame, holdout_start: str | pd.Timestamp) -> list[Case]:
    """Fill control and weather dates from the holdout; leave the others fixed."""
    cutoff = pd.Timestamp(holdout_start)
    reserved = {case.date for case in CASES if case.date}
    weather_date = _hottest_holdout_weekday(hourly, cutoff, exclude=reserved)
    reserved.add(weather_date)
    control_date = _median_mae_weekday(hourly, cutoff, exclude=reserved)
    resolved: list[Case] = []
    for case in CASES:
        if case.case_id == "weather":
            resolved.append(Case(case.case_id, case.title, case.why, weather_date, in_holdout=True))
        elif case.case_id == "control":
            resolved.append(Case(case.case_id, case.title, case.why, control_date, in_holdout=True))
        else:
            resolved.append(case)
    return resolved


def slice_case(hourly: pd.DataFrame, date: str) -> pd.DataFrame:
    if hourly.empty or not date:
        return hourly.iloc[0:0].copy()
    if "settlement_date" in hourly.columns:
        day = pd.to_datetime(hourly["settlement_date"]).dt.normalize()
        target = pd.Timestamp(date).normalize()
        return hourly.loc[day == target].copy()
    stamp = pd.to_datetime(hourly["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    target = pd.Timestamp(date).tz_localize(LONDON_TZ).normalize()
    return hourly.loc[stamp.dt.normalize() == target].copy()


def weekday_baseline(hourly: pd.DataFrame, holdout_start: str | pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(holdout_start)
    stamp = pd.to_datetime(hourly["timestamp"], utc=True)
    if stamp.dt.tz is not None:
        cutoff = cutoff.tz_convert(stamp.dt.tz) if cutoff.tzinfo else cutoff.tz_localize(stamp.dt.tz)
    work = hourly.loc[stamp >= cutoff].copy()
    if "is_holiday" in work.columns:
        work = work.loc[work["is_holiday"] != 1]
    if "is_weekend" in work.columns:
        work = work.loc[work["is_weekend"] != 1]
    return work


def case_row(case: Case, day: pd.DataFrame, baseline_mae: float) -> dict:
    metrics = overall_metrics(day) if not day.empty else pd.DataFrame()
    mae = float(metrics["mae"].iloc[0]) if not metrics.empty else float("nan")
    return {
        "case": case.case_id,
        "date": case.date,
        "title": case.title,
        "in_holdout": case.in_holdout,
        "n": 0 if day.empty else int(len(day)),
        "mae": mae,
        "vs_weekday": mae - baseline_mae,
        "bias": float(metrics["bias"].iloc[0]) if not metrics.empty else float("nan"),
        "coverage_80": float(metrics["coverage_80"].iloc[0]) if not metrics.empty else float("nan"),
        "demand_mw": float(day["demand_mw"].mean()) if not day.empty else float("nan"),
        "lag_48": float(day["demand_lag_48"].mean()) if not day.empty and "demand_lag_48" in day.columns else float("nan"),
        "lag_gap": _lag_gap(day),
        "temperature_2m": float(day["temperature_2m"].mean()) if not day.empty and "temperature_2m" in day.columns else float("nan"),
    }


def case_note(case: Case, day: pd.DataFrame, baseline_mae: float, shap_tables: dict[str, pd.DataFrame]) -> str:
    if day.empty:
        return "No rows for this date."
    actual = day["demand_mw"] if "demand_mw" in day.columns else day.get("actual_mw")
    pred = day["p50"]
    mae = float((actual - pred).abs().mean())
    bias = float((actual - pred).mean())
    delta = mae - baseline_mae
    leftover = shap_tables.get("mean")
    top = "calendar/weather"
    if leftover is not None and not leftover.empty:
        top = str(leftover["feature"].iloc[0])
    side = "low" if bias > 0 else "high"
    sample = "Holdout" if case.in_holdout else "In-sample"
    gap = _lag_gap(day)
    gap_bit = f" Demand vs yesterday {gap:+,.0f} MW." if pd.notna(gap) else ""
    return (
        f"{sample}: MAE {mae:,.0f} MW ({delta:+,.0f} vs weekday median). "
        f"Model ran {side} (bias {bias:+,.0f}). After lags, mean SHAP led by {top}.{gap_bit}"
    )


def case_shap(
    day: pd.DataFrame,
    models: QuantileModels,
    n: int = 5,
    *,
    drop_lags: bool = True,
) -> dict[str, pd.DataFrame]:
    """Signed leftover effects: feature value on this day and MW added to that output.

    ``shap_mw`` is the mean SHAP (can be negative). ``mw_per_unit`` is a local
    slope on this day only — not a global coefficient.
    """
    empty = {
        output: pd.DataFrame(columns=["feature", "value_mean", "shap_mw", "mw_per_unit", "mean_abs_shap"])
        for output in OUTPUTS
    }
    if day.empty:
        return empty
    shap_by_output = explain_scored(day, models)
    tables = {}
    for output, values in shap_by_output.items():
        work = values.drop(columns=[c for c in LAG_FEATURES if c in values.columns], errors="ignore") if drop_lags else values
        tables[output] = _signed_effects(day, work, n=n)
    return tables


def shap_path(day: pd.DataFrame, models: QuantileModels, feature: str, output: str = "mean") -> pd.DataFrame:
    """Half-hourly feature value and the MW that feature added to ``output``."""
    shap_by_output = explain_scored(day, models)
    contrib = shap_by_output[output]
    if feature not in contrib.columns or feature not in day.columns:
        return pd.DataFrame(columns=["timestamp", "value", "shap_mw"])
    return pd.DataFrame(
        {
            "timestamp": day["timestamp"].to_numpy(),
            "value": pd.to_numeric(day[feature], errors="coerce").to_numpy(),
            "shap_mw": contrib[feature].to_numpy(),
        },
        index=day.index,
    )


def _signed_effects(day: pd.DataFrame, shap_values: pd.DataFrame, n: int) -> pd.DataFrame:
    cols = [c for c in shap_values.columns if c != "shap_base"]
    if not cols:
        return pd.DataFrame(columns=["feature", "value_mean", "shap_mw", "mw_per_unit", "mean_abs_shap"])
    ranking = shap_values.loc[:, cols].abs().mean().sort_values(ascending=False).head(n)
    rows: list[dict] = []
    for feature in ranking.index:
        shap = pd.to_numeric(shap_values[feature], errors="coerce")
        value = pd.to_numeric(day[feature], errors="coerce") if feature in day.columns else pd.Series(dtype=float)
        slope = float("nan")
        if len(value) == len(shap) and float(value.var()) > 1e-12:
            slope = float(np.cov(value.to_numpy(), shap.to_numpy(), ddof=0)[0, 1] / value.var())
        rows.append(
            {
                "feature": feature,
                "value_mean": float(value.mean()) if len(value) else float("nan"),
                "shap_mw": float(shap.mean()),
                "mw_per_unit": slope,
                "mean_abs_shap": float(shap.abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def weekday_median_mae(hourly: pd.DataFrame, holdout_start: str | pd.Timestamp) -> float:
    daily = _daily_holdout(hourly, pd.Timestamp(holdout_start))
    if daily.empty:
        return float("nan")
    return float(daily["mae"].median())


def _daily_holdout(
    hourly: pd.DataFrame,
    holdout_start: pd.Timestamp,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    work = weekday_baseline(hourly, holdout_start)
    if work.empty:
        return pd.DataFrame()
    if "settlement_date" in work.columns:
        key = pd.to_datetime(work["settlement_date"]).dt.normalize()
    else:
        key = pd.to_datetime(work["timestamp"], utc=True).dt.tz_convert(LONDON_TZ).dt.normalize()
    daily = work.groupby(key).agg(
        mae=("abs_error", "mean"),
        temperature_2m=("temperature_2m", "mean") if "temperature_2m" in work.columns else ("abs_error", "size"),
    )
    if exclude:
        blocked = {pd.Timestamp(value).normalize() for value in exclude if value}
        daily = daily.loc[~daily.index.isin(blocked)]
    return daily


def _hottest_holdout_weekday(
    hourly: pd.DataFrame,
    holdout_start: pd.Timestamp,
    exclude: set[str] | None = None,
) -> str | None:
    daily = _daily_holdout(hourly, holdout_start, exclude)
    if daily.empty or "temperature_2m" not in daily.columns:
        return None
    return pd.Timestamp(daily["temperature_2m"].idxmax()).strftime("%Y-%m-%d")


def _median_mae_weekday(
    hourly: pd.DataFrame,
    holdout_start: pd.Timestamp,
    exclude: set[str] | None = None,
) -> str | None:
    daily = _daily_holdout(hourly, holdout_start, exclude)
    if daily.empty:
        return None
    nearest = (daily["mae"] - daily["mae"].median()).abs().idxmin()
    return pd.Timestamp(nearest).strftime("%Y-%m-%d")


def _lag_gap(day: pd.DataFrame) -> float:
    if day.empty or "demand_lag_48" not in day.columns or "demand_mw" not in day.columns:
        return float("nan")
    return float((day["demand_mw"] - day["demand_lag_48"]).mean())
