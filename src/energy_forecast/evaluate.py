"""Backtest and slice GB demand quantile forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.features import FEATURE_COLUMNS, lag_walk_step
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
    return _decorate_scored(predict_quantiles(work, models), models)


def score_recursive(
    frame: pd.DataFrame,
    models: QuantileModels,
    *,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Same slices as ``score``, but demand lags roll from history then P50.

    ``demand_mw`` on ``frame`` is the label only. It is not written into
    ``lag_48`` / ``lag_1`` for later rows. Pass the training fold as ``history``.
    """
    from energy_forecast.forecast import predict_quantiles_recursive

    features = [column for column in (models.features or FEATURE_COLUMNS) if column in frame.columns]
    work = frame.dropna(subset=features).copy()
    if work.empty:
        return work
    keys = [column for column in ("region_id", "timestamp", "lead_hours") if column in work.columns]
    labels = work.loc[:, [column for column in [*keys, "demand_mw"] if column in work.columns]]
    predict_frame = work.copy()
    predict_frame["demand_mw"] = np.nan
    hat = predict_quantiles_recursive(predict_frame, models, history=history)
    hat = hat.drop(columns=["demand_mw"], errors="ignore")
    if keys:
        hat = hat.merge(labels.drop_duplicates(keys), on=keys, how="left")
    else:
        hat["demand_mw"] = labels["demand_mw"].to_numpy()[: len(hat)]
    return _decorate_scored(hat, models)


def _decorate_scored(out: pd.DataFrame, models: QuantileModels) -> pd.DataFrame:
    out = out.copy()
    if "demand_mw" not in out.columns or "p50" not in out.columns:
        return out
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
    if "temperature_2m" in out.columns:
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


def iter_time_series_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    embargo_days: float = 0.0,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds on unique timestamps (sklearn TimeSeriesSplit).

    Rows that share a timestamp (Previous Runs leads 0–7) stay in the same
    fold, so a Tuesday 12:00 lead-168 cannot sit in train while lead-0 is in
    test. Train always ends before test starts. This is not shuffled k-fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return []
    work = frame.sort_values("timestamp").copy()
    stamps = pd.to_datetime(work["timestamp"], utc=True)
    unique = pd.Index(stamps.unique()).sort_values()
    n_times = len(unique)
    test_size = n_times // (n_splits + 1)
    if test_size < 1:
        raise ValueError(f"Not enough unique timestamps ({n_times}) for {n_splits} folds")
    embargo = pd.Timedelta(days=float(embargo_days))
    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for fold in range(n_splits):
        test_start = n_times - (n_splits - fold) * test_size
        test_end = test_start + test_size
        train_times = unique[:test_start]
        test_times = unique[test_start:test_end]
        if embargo_days:
            cutoff = test_times.min() - embargo
            train_times = train_times[train_times <= cutoff]
        train = work.loc[stamps.isin(set(train_times))]
        test = work.loc[stamps.isin(set(test_times))]
        folds.append((fold, train, test))
    return folds


def time_series_kfold(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_estimators: int = 400,
    features: list[str] | None = None,
    mean_params: dict | None = None,
    residual_params: dict | None = None,
    eval_mode: str | None = MEDIUM,
    embargo_days: float = 0.0,
    min_train_rows: int = 48 * 60,
    refit_full: bool = False,
    scoring: tuple[str, ...] = ("direct",),
    max_lead_hours: float | None = None,
    pack: str | None = None,
    return_slices: bool = False,
    verbose: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Fit once per expanding fold; score direct and/or recursive P50 lags.

    ``scoring`` is ``('direct',)``, ``('recursive',)``, or both. Recursive uses
    the training fold as demand history and does not leak test ``demand_mw``
    into later lags. ``eval_mode`` still resamples for the headline MAE; a
    ``grid`` column of ``30min`` is always kept next to that resample.
    """
    feature_cols = list(features or FEATURE_COLUMNS)
    methods = tuple(scoring) or ("direct",)
    work = frame.dropna(subset=["demand_mw", *feature_cols]).copy()
    if max_lead_hours is not None and "lead_hours" in work.columns:
        lead = pd.to_numeric(work["lead_hours"], errors="coerce").fillna(0.0)
        work = work.loc[lead <= float(max_lead_hours)].copy()
    rows: list[pd.DataFrame] = []
    slice_rows: list[pd.DataFrame] = []
    for fold, train, test in iter_time_series_folds(
        work, n_splits=n_splits, embargo_days=embargo_days
    ):
        if len(train) < min_train_rows or test.empty:
            continue
        if verbose:
            print(
                f"{pack or 'fold'} {fold + 1}/{n_splits} train {len(train)} test {len(test)} fit…",
                flush=True,
            )
        models = fit_models(
            train,
            n_estimators=n_estimators,
            features=feature_cols,
            mean_params=mean_params,
            residual_params=residual_params,
            refit_full=refit_full,
            metadata={"fold": fold, "n_splits": n_splits, "pack": pack},
        )
        train_ts = pd.to_datetime(train["timestamp"], utc=True)
        test_ts = pd.to_datetime(test["timestamp"], utc=True)
        if eval_mode:
            spec = MODE_SPEC[eval_mode]
            scale_mode = _naive_scale_mse(train, spec["freq"], spec["season"])
        else:
            scale_mode = None
        scale_hh = _naive_scale_mse(train, "30min", 336)
        lag_step = lag_walk_step(feature_cols)
        for method in methods:
            if method == "recursive" and lag_step == 1:
                if verbose:
                    print(
                        f"{pack or 'fold'} {fold + 1}/{n_splits} skip lag_1 recursive "
                        f"({len(test)} rows; live hour pack is one shot) — direct score",
                        flush=True,
                    )
                scored = score(test, models)
            elif method == "recursive":
                if verbose:
                    print(f"{pack or 'fold'} {fold + 1}/{n_splits} recursive score…", flush=True)
                scored = score_recursive(test, models, history=train)
            elif method == "direct":
                if verbose:
                    print(f"{pack or 'fold'} {fold + 1}/{n_splits} direct score…", flush=True)
                scored = score(test, models)
            else:
                raise ValueError("scoring methods must be 'direct' or 'recursive'")
            if scored.empty:
                continue
            scored = scored.assign(fold=fold, scoring=method, pack=pack or "")
            grids = [("30min", scored, scale_hh)]
            if eval_mode:
                grids.append((eval_mode, resample_to_mode(scored, eval_mode), scale_mode))
            for grid_name, grid_frame, scale in grids:
                block = _metrics_block(
                    grid_frame, split=f"fold {fold + 1}/{n_splits}", scale_mse=scale
                )
                if block.empty:
                    continue
                block.insert(1, "pack", pack or "")
                block.insert(2, "scoring", method)
                block.insert(3, "grid", grid_name)
                block.insert(4, "fold", fold)
                block.insert(5, "train_end", str(train_ts.max()))
                block.insert(6, "test_start", str(test_ts.min()))
                block.insert(7, "test_end", str(test_ts.max()))
                block.insert(8, "n_train", int(len(train)))
                rows.append(block)
            if return_slices:
                sliced = product_slices(scored)
                if not sliced.empty:
                    sliced.insert(0, "pack", pack or "")
                    sliced.insert(1, "scoring", method)
                    sliced.insert(2, "fold", fold)
                    slice_rows.append(sliced)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not table.empty:
        table = pd.concat([table, _kfold_mean_rows(table)], ignore_index=True)
    if return_slices:
        slices = pd.concat(slice_rows, ignore_index=True) if slice_rows else pd.DataFrame()
        return table, slices
    return table


def _kfold_mean_rows(table: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        column
        for column in ("mae", "rmse", "bias", "wrmsse", "coverage_80", "interval_width", "mape")
        if column in table.columns
    ]
    keys = [column for column in ("pack", "scoring", "grid") if column in table.columns]
    work = table.loc[table["split"].astype(str).str.startswith("fold")].copy()
    if work.empty or not keys:
        mean_row = {column: None for column in table.columns}
        mean_row["split"] = "mean"
        if "n" in table:
            mean_row["n"] = int(table["n"].sum())
        for column in metric_cols:
            mean_row[column] = float(table[column].mean())
        return pd.DataFrame([mean_row])
    rows = []
    for _, group in work.groupby(keys, dropna=False):
        mean_row = {column: None for column in table.columns}
        mean_row["split"] = "mean"
        for key in keys:
            mean_row[key] = group[key].iloc[0]
        mean_row["n"] = int(group["n"].sum()) if "n" in group else None
        mean_row["n_train"] = int(group["n_train"].mean()) if "n_train" in group else None
        for column in metric_cols:
            mean_row[column] = float(group[column].mean())
        rows.append(mean_row)
    return pd.DataFrame(rows)


def kfold_all_packs(
    week_frame: pd.DataFrame,
    hour_frame: pd.DataFrame | None = None,
    *,
    n_splits: int = 5,
    embargo_days: float = 0.0,
    scoring: tuple[str, ...] = ("direct", "recursive"),
    refit_full: bool = False,
    min_train_rows: int = 48 * 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding k-fold for hour, day, and week packs. Direct vs recursive.

    ``week_frame`` is Previous Runs (used for day and week). ``hour_frame`` is
    ERA5. Each fold trains once, then scores both methods.
    """
    from energy_forecast.features import HOUR_FEATURE_COLUMNS
    from energy_forecast.model import (
        DAY_MAX_LEAD_HOURS,
        HOUR_MAX_LEAD_HOURS,
        LGB_WEEK,
        WEEK_N_ESTIMATORS,
    )

    configs = [
        {
            "pack": "hour",
            "frame": hour_frame,
            "features": HOUR_FEATURE_COLUMNS,
            "max_lead_hours": HOUR_MAX_LEAD_HOURS,
            "n_estimators": 400,
            "mean_params": None,
            "residual_params": None,
            "eval_mode": None,
        },
        {
            "pack": "day",
            "frame": week_frame,
            "features": FEATURE_COLUMNS,
            "max_lead_hours": DAY_MAX_LEAD_HOURS,
            "n_estimators": 400,
            "mean_params": None,
            "residual_params": None,
            "eval_mode": SHORT,
        },
        {
            "pack": "week",
            "frame": week_frame,
            "features": FEATURE_COLUMNS,
            "max_lead_hours": None,
            "n_estimators": WEEK_N_ESTIMATORS,
            "mean_params": LGB_WEEK,
            "residual_params": LGB_WEEK,
            "eval_mode": MEDIUM,
        },
    ]
    fold_tables: list[pd.DataFrame] = []
    slice_tables: list[pd.DataFrame] = []
    for config in configs:
        frame = config["frame"]
        if frame is None or frame.empty:
            continue
        table, sliced = time_series_kfold(
            frame,
            n_splits=n_splits,
            n_estimators=config["n_estimators"],
            features=config["features"],
            mean_params=config["mean_params"],
            residual_params=config["residual_params"],
            eval_mode=config["eval_mode"],
            embargo_days=embargo_days,
            min_train_rows=min_train_rows,
            refit_full=refit_full,
            scoring=scoring,
            max_lead_hours=config["max_lead_hours"],
            pack=config["pack"],
            return_slices=True,
        )
        if not table.empty:
            fold_tables.append(table)
        if not sliced.empty:
            slice_tables.append(sliced)
    folds = pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()
    slices = pd.concat(slice_tables, ignore_index=True) if slice_tables else pd.DataFrame()
    return folds, slices


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
    clim = float(scored["clim_abs_error"].mean()) if "clim_abs_error" in scored.columns else float("nan")
    abs_mae = float(np.mean(np.abs(error)))
    skill = 1.0 - abs_mae / clim if np.isfinite(clim) and clim > 0 else float("nan")
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


def board_naive_scale(demand: pd.DataFrame | None = None, *, days: int = 56) -> float:
    """Weekly-seasonal naive MSE on 30-minute demand, same scale as the live board WRMSSE."""
    if demand is None:
        from energy_forecast.data.demand import load_demand

        demand = load_demand(include_live=True)
    work = demand.copy()
    if "demand_mw" not in work.columns and "actual_mw" in work.columns:
        work = work.rename(columns={"actual_mw": "demand_mw"})
    if work.empty or "timestamp" not in work.columns or "demand_mw" not in work.columns:
        return float("nan")
    ts = pd.to_datetime(work["timestamp"], utc=True)
    start = ts.max() - pd.Timedelta(days=int(days))
    return _naive_scale_mse(work.loc[ts >= start], "30min", 336)


def as_board_scored(
    frame: pd.DataFrame,
    *,
    actual_col: str = "actual_mw",
    climatology: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Board rows (actual vs P50) in the shape ``overall_metrics`` expects. Half-hours, not 6h bins."""
    need = {actual_col, "p50"}
    if frame is None or frame.empty or not need.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.dropna(subset=list(need)).copy()
    work["demand_mw"] = pd.to_numeric(work[actual_col], errors="coerce")
    work["p50"] = pd.to_numeric(work["p50"], errors="coerce")
    work = work.dropna(subset=["demand_mw", "p50"])
    if work.empty:
        return work
    for column in ("p10", "p90"):
        if column not in work.columns:
            work[column] = work["p50"]
        else:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(work["p50"])
    work["error"] = work["demand_mw"] - work["p50"]
    work["abs_error"] = work["error"].abs()
    work["ape"] = work["abs_error"] / work["demand_mw"].clip(lower=1)
    work["covered"] = (work["demand_mw"] >= work["p10"]) & (work["demand_mw"] <= work["p90"])
    work["interval_width"] = work["p90"] - work["p10"]
    if climatology is not None and not climatology.empty and {"region_id", "month", "day_of_week", "settlement_period"}.issubset(work.columns):
        work["climatology_mw"] = _row_climatology(work, climatology)
    else:
        work["climatology_mw"] = np.nan
    work["clim_abs_error"] = (work["demand_mw"] - work["climatology_mw"]).abs()
    work["clim_abs_error"] = work["clim_abs_error"].fillna(work["abs_error"])
    if "settlement_period" in work.columns:
        work["period_band"] = pd.to_numeric(work["settlement_period"], errors="coerce").map(_period_band)
    return work
