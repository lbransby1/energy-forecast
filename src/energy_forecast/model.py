"""Two-phase GB demand model: mean first, then residual quantiles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from energy_forecast.features import (
    FEATURE_COLUMNS,
    HOUR_FEATURE_COLUMNS,
    attach_demand_lags,
    build_features,
    demand_climatology,
    merge_demand_weather,
)
from energy_forecast.paths import DATA_MODELS, DATA_MODELS_DAY, DATA_MODELS_HOUR, ensure_data_dirs

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
HOLDOUT_DAYS = 56
TARGET_COVERAGE = 0.80
CALIB_DAYS = 21
MEAN_FEATURE = "mean_pred"
# Previous Runs weather is complete from 2024; keep earlier NESO files for lags.
EXCLUDE_TRAIN_YEARS = (2020, 2021, 2022, 2023)
# Next-hour model uses ERA5, so 2022-2023 are usable.
HOUR_EXCLUDE_TRAIN_YEARS = (2020, 2021)
HOUR_MAX_LEAD_HOURS = 1.0
DAY_MAX_LEAD_HOURS = 24.0
# Extra P10–P90 stretch at 168h so the week fan widens with lead.
LEAD_WIDTH_GROWTH = 0.40
LEAD_WIDTH_HOURS = 168.0

LGB_MEAN = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 47,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1,
}
LGB_RESIDUAL = {
    "learning_rate": 0.05,
    "num_leaves": 47,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1,
}
# Optuna winner for medium (6-hourly, 7d) holdout MAE 862 vs 928.
LGB_WEEK = {
    "learning_rate": 0.022965408930055223,
    "num_leaves": 19,
    "min_child_samples": 61,
    "subsample": 0.781673112232677,
    "colsample_bytree": 0.6385450911486689,
}
WEEK_N_ESTIMATORS = 450


@dataclass
class QuantileModels:
    mean_boosters: dict[str, lgb.Booster]
    residual_boosters: dict[tuple[str, float], lgb.Booster]
    interval_scale: dict[str, float]
    features: list[str]
    residual_features: list[str]
    quantiles: tuple[float, ...]
    climatology: pd.DataFrame
    metadata: dict
    boosters: dict[tuple[str, float], lgb.Booster] = field(default_factory=dict)


def prepare_training_frame(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = merge_demand_weather(demand, weather)
    return build_features(merged)


def split_training_frame(
    frame: pd.DataFrame,
    *,
    features: list[str] | None = None,
    holdout_days: int = HOLDOUT_DAYS,
    exclude_years: tuple[int, ...] = EXCLUDE_TRAIN_YEARS,
    max_lead_hours: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same chronological split ``train_models`` uses. Does not fit."""
    feature_cols = list(features or FEATURE_COLUMNS)
    work = frame.dropna(subset=["demand_mw", *feature_cols]).copy()
    if max_lead_hours is not None and "lead_hours" in work.columns:
        lead = pd.to_numeric(work["lead_hours"], errors="coerce").fillna(0.0)
        work = work[lead <= max_lead_hours].copy()
    if exclude_years:
        years = pd.to_datetime(work["timestamp"], utc=True).dt.year
        work = work[~years.isin(exclude_years)].copy()
    if work.empty:
        raise ValueError("Training frame is empty after dropping rows with missing features")
    cutoff = work["timestamp"].max() - pd.Timedelta(days=holdout_days)
    train = work[work["timestamp"] < cutoff]
    holdout = work[work["timestamp"] >= cutoff]
    if train.empty:
        raise ValueError("Not enough history to create a training split")
    return train, holdout


def fit_models(
    train: pd.DataFrame,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    n_estimators: int = 400,
    metadata: dict | None = None,
    target_coverage: float = TARGET_COVERAGE,
    features: list[str] | None = None,
    mean_params: dict | None = None,
    residual_params: dict | None = None,
    refit_full: bool = True,
) -> QuantileModels:
    """Fit a mean model, then demand quantile models on all features plus the mean."""
    feature_cols = list(features or FEATURE_COLUMNS)
    work = train.dropna(subset=["demand_mw", *feature_cols]).copy()
    if work.empty:
        raise ValueError("Training frame is empty after dropping rows with missing features")
    mean_params = {**LGB_MEAN, **(mean_params or {})}
    residual_params = {**LGB_RESIDUAL, **(residual_params or {})}

    cutoff = work["timestamp"].max() - pd.Timedelta(days=CALIB_DAYS)
    fit = work[work["timestamp"] < cutoff]
    calib = work[work["timestamp"] >= cutoff]
    if fit.empty or len(calib) < 48:
        fit, calib = work, work.iloc[0:0]

    climatology = demand_climatology(fit if not fit.empty else work)
    mean_boosters: dict[str, lgb.Booster] = {}
    residual_boosters: dict[tuple[str, float], lgb.Booster] = {}
    interval_scale: dict[str, float] = {}
    residual_features = [*feature_cols, MEAN_FEATURE]

    for series_id, series_fit in fit.groupby("region_id"):
        mean_booster = _fit_mean(series_fit, feature_cols, mean_params, n_estimators)
        mean_hat = np.asarray(mean_booster.predict(series_fit[feature_cols]))
        x_phase2 = series_fit[feature_cols].copy()
        x_phase2[MEAN_FEATURE] = mean_hat
        mean_boosters[str(series_id)] = mean_booster
        for quantile in quantiles:
            residual_boosters[(str(series_id), float(quantile))] = _fit_phase2_quantile(
                x_phase2, series_fit["demand_mw"].to_numpy(), quantile, n_estimators, residual_params
            )

        series_calib = calib[calib["region_id"] == series_id] if not calib.empty else series_fit.iloc[0:0]
        scale_frame = series_calib if not series_calib.empty else series_fit
        interval_scale[str(series_id)] = _coverage_scale(
            scale_frame,
            mean_booster,
            {q: residual_boosters[(str(series_id), float(q))] for q in quantiles},
            feature_cols,
            residual_features,
            target_coverage,
        )

        if not refit_full:
            continue
        series_all = work[work["region_id"] == series_id]
        mean_booster = _fit_mean(series_all, feature_cols, mean_params, n_estimators)
        mean_hat = np.asarray(mean_booster.predict(series_all[feature_cols]))
        x_phase2 = series_all[feature_cols].copy()
        x_phase2[MEAN_FEATURE] = mean_hat
        mean_boosters[str(series_id)] = mean_booster
        for quantile in quantiles:
            residual_boosters[(str(series_id), float(quantile))] = _fit_phase2_quantile(
                x_phase2, series_all["demand_mw"].to_numpy(), quantile, n_estimators, residual_params
            )

    extra = metadata or {}
    return QuantileModels(
        mean_boosters=mean_boosters,
        residual_boosters=residual_boosters,
        interval_scale=interval_scale,
        features=feature_cols,
        residual_features=residual_features,
        quantiles=tuple(quantiles),
        climatology=climatology,
        metadata={
            "architecture": "mean_then_quantiles",
            "quantiles": list(quantiles),
            "features": feature_cols,
            "residual_features": residual_features,
            "regions": sorted(work["region_id"].unique().tolist()),
            "interval_scale": interval_scale,
            "target_coverage": target_coverage,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_estimators": n_estimators,
            "n_train": int(len(work)),
            **extra,
        },
    )


def train_models(
    frame: pd.DataFrame,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    holdout_days: int = HOLDOUT_DAYS,
    model_dir: Path | None = None,
    n_estimators: int = 400,
    exclude_years: tuple[int, ...] = EXCLUDE_TRAIN_YEARS,
    features: list[str] | None = None,
    max_lead_hours: float | None = None,
    role: str = "week",
) -> tuple[QuantileModels, pd.DataFrame]:
    """Fit the two-phase model. Last ``holdout_days`` are held out for metrics."""
    model_dir = model_dir or DATA_MODELS
    ensure_data_dirs()
    feature_cols = list(features or FEATURE_COLUMNS)
    train, holdout = split_training_frame(
        frame,
        features=feature_cols,
        holdout_days=holdout_days,
        exclude_years=exclude_years,
        max_lead_hours=max_lead_hours,
    )
    cutoff = train["timestamp"].max() if holdout.empty else holdout["timestamp"].min()
    tree_params, trees = _role_tree_settings(role, n_estimators)

    models = fit_models(
        train,
        quantiles=quantiles,
        n_estimators=trees,
        features=feature_cols,
        mean_params=tree_params,
        residual_params=tree_params,
        metadata={
            "role": role,
            "holdout_start": str(cutoff),
            "n_holdout": int(len(holdout)),
            "exclude_years": list(exclude_years),
            "max_lead_hours": max_lead_hours,
            "lgb_params": tree_params,
        },
    )

    metric_rows: list[dict] = []
    if not holdout.empty:
        scored = predict_quantiles(holdout, models)
        for quantile in models.quantiles:
            column = _quantile_column(quantile)
            metric_rows.extend(
                _quantile_metrics(
                    quantile=float(quantile),
                    actual=scored["demand_mw"].to_numpy(),
                    predicted=scored[column].to_numpy(),
                )
            )
        covered = (scored["demand_mw"] >= scored["p10"]) & (scored["demand_mw"] <= scored["p90"])
        metric_rows.append(
            {
                "metric": "coverage_80",
                "quantile": 0.8,
                "value": float(covered.mean()),
            }
        )
        metric_rows.append(
            {
                "metric": "interval_scale",
                "quantile": 0.8,
                "value": float(next(iter(models.interval_scale.values()), 1.0)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    save_models(models, model_dir)
    if not metrics.empty:
        metrics.to_csv(model_dir / "metrics.csv", index=False)
    return models, metrics


def train_hour_models(
    frame: pd.DataFrame,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    holdout_days: int = HOLDOUT_DAYS,
    model_dir: Path | None = None,
    n_estimators: int = 400,
    exclude_years: tuple[int, ...] = HOUR_EXCLUDE_TRAIN_YEARS,
) -> tuple[QuantileModels, pd.DataFrame]:
    """Fit the next-hour model on ERA5 weather and recent demand lags."""
    return train_models(
        frame,
        quantiles=quantiles,
        holdout_days=holdout_days,
        model_dir=model_dir or DATA_MODELS_HOUR,
        n_estimators=n_estimators,
        exclude_years=exclude_years,
        features=HOUR_FEATURE_COLUMNS,
        max_lead_hours=HOUR_MAX_LEAD_HOURS,
        role="next_hour",
    )


def load_hour_models(model_dir: Path | None = None) -> QuantileModels:
    return load_models(model_dir or DATA_MODELS_HOUR)


def train_day_models(
    frame: pd.DataFrame,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    holdout_days: int = HOLDOUT_DAYS,
    model_dir: Path | None = None,
    n_estimators: int = 400,
    exclude_years: tuple[int, ...] = EXCLUDE_TRAIN_YEARS,
) -> tuple[QuantileModels, pd.DataFrame]:
    """Fit the next-day model on issued-forecast weather, leads 0–24h only."""
    return train_models(
        frame,
        quantiles=quantiles,
        holdout_days=holdout_days,
        model_dir=model_dir or DATA_MODELS_DAY,
        n_estimators=n_estimators,
        exclude_years=exclude_years,
        features=FEATURE_COLUMNS,
        max_lead_hours=DAY_MAX_LEAD_HOURS,
        role="next_day",
    )


def load_day_models(model_dir: Path | None = None) -> QuantileModels:
    return load_models(model_dir or DATA_MODELS_DAY)


def save_models(models: QuantileModels, model_dir: Path | None = None) -> None:
    model_dir = model_dir or DATA_MODELS
    ensure_data_dirs()
    model_dir.mkdir(parents=True, exist_ok=True)
    for series_id, booster in models.mean_boosters.items():
        booster.save_model(str(_mean_path(model_dir, series_id)))
    for (series_id, quantile), booster in models.residual_boosters.items():
        booster.save_model(str(_residual_path(model_dir, series_id, quantile)))
    models.climatology.to_parquet(model_dir / "climatology.parquet", index=False)
    (model_dir / "metadata.json").write_text(json.dumps(models.metadata, indent=2), encoding="utf-8")


def load_models(model_dir: Path | None = None) -> QuantileModels:
    model_dir = model_dir or DATA_MODELS
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No saved models in {model_dir}. Train models first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("architecture") != "mean_then_quantiles":
        raise ValueError(
            "Saved models are not the current two-phase architecture. Re-run notebook 03 to retrain."
        )
    quantiles = tuple(float(q) for q in metadata["quantiles"])
    regions = metadata["regions"]
    interval_scale = {key: float(value) for key, value in metadata.get("interval_scale", {}).items()}
    mean_boosters = {
        series_id: lgb.Booster(model_file=str(_mean_path(model_dir, series_id))) for series_id in regions
    }
    residual_boosters = {
        (series_id, quantile): lgb.Booster(model_file=str(_residual_path(model_dir, series_id, quantile)))
        for series_id in regions
        for quantile in quantiles
    }
    return QuantileModels(
        mean_boosters=mean_boosters,
        residual_boosters=residual_boosters,
        interval_scale=interval_scale,
        features=list(metadata.get("features", FEATURE_COLUMNS)),
        residual_features=list(metadata.get("residual_features", [*FEATURE_COLUMNS, MEAN_FEATURE])),
        quantiles=quantiles,
        climatology=pd.read_parquet(model_dir / "climatology.parquet"),
        metadata=metadata,
    )


def predict_quantiles(frame: pd.DataFrame, models: QuantileModels) -> pd.DataFrame:
    out = frame.copy()
    for quantile in models.quantiles:
        out[_quantile_column(quantile)] = np.nan
    out["mean_pred"] = np.nan

    for series_id, group in out.groupby("region_id"):
        mean_booster = models.mean_boosters.get(str(series_id))
        if mean_booster is None:
            continue
        mean_hat = np.asarray(mean_booster.predict(group[models.features]))
        out.loc[group.index, "mean_pred"] = mean_hat
        x_phase2 = group[models.features].copy()
        x_phase2[MEAN_FEATURE] = mean_hat
        levels = {}
        for quantile in models.quantiles:
            booster = models.residual_boosters.get((str(series_id), float(quantile)))
            if booster is None:
                continue
            levels[quantile] = np.asarray(booster.predict(x_phase2[models.residual_features]))
        scale = models.interval_scale.get(str(series_id), 1.0)
        role = models.metadata.get("role") or "week"
        if role == "week" and "lead_hours" in group.columns:
            scale = scale * _lead_width_factor(group["lead_hours"].to_numpy())
        center = levels.get(0.5, mean_hat)
        for quantile in models.quantiles:
            level = levels.get(quantile)
            if level is None:
                continue
            if quantile == 0.5:
                out.loc[group.index, _quantile_column(quantile)] = center
            else:
                out.loc[group.index, _quantile_column(quantile)] = center + scale * (level - center)
    return _enforce_quantile_order(out, models.quantiles)


def _lead_width_factor(lead_hours: np.ndarray) -> np.ndarray:
    """Monotone extra interval stretch: 1.0 now, 1 + LEAD_WIDTH_GROWTH at day 7."""
    hours = np.clip(np.asarray(lead_hours, dtype=float), 0.0, LEAD_WIDTH_HOURS)
    return 1.0 + LEAD_WIDTH_GROWTH * (hours / LEAD_WIDTH_HOURS)


def _role_tree_settings(role: str, n_estimators: int) -> tuple[dict, int]:
    if role == "week":
        return dict(LGB_WEEK), WEEK_N_ESTIMATORS
    return {}, n_estimators


def _fit_mean(
    frame: pd.DataFrame,
    features: list[str],
    params: dict | None = None,
    n_estimators: int = 400,
) -> lgb.Booster:
    dataset = lgb.Dataset(frame[features], label=frame["demand_mw"], free_raw_data=False)
    return lgb.train({**LGB_MEAN, **(params or {})}, dataset, num_boost_round=n_estimators)


def _fit_phase2_quantile(
    features: pd.DataFrame,
    demand_mw: np.ndarray,
    quantile: float,
    n_estimators: int,
    params: dict | None = None,
) -> lgb.Booster:
    dataset = lgb.Dataset(features, label=demand_mw, free_raw_data=False)
    return lgb.train(
        {**LGB_RESIDUAL, **(params or {}), "objective": "quantile", "alpha": quantile},
        dataset,
        num_boost_round=n_estimators,
    )


def _coverage_scale(
    frame: pd.DataFrame,
    mean_booster: lgb.Booster,
    residual_boosters: dict[float, lgb.Booster],
    features: list[str],
    residual_features: list[str],
    target_coverage: float,
) -> float:
    if frame.empty or 0.1 not in residual_boosters or 0.9 not in residual_boosters:
        return 1.0
    mean_hat = np.asarray(mean_booster.predict(frame[features]))
    x_phase2 = frame[features].copy()
    x_phase2[MEAN_FEATURE] = mean_hat
    q10 = np.asarray(residual_boosters[0.1].predict(x_phase2[residual_features]))
    q50 = np.asarray(residual_boosters[0.5].predict(x_phase2[residual_features]))
    q90 = np.asarray(residual_boosters[0.9].predict(x_phase2[residual_features]))
    actual = frame["demand_mw"].to_numpy()
    if float(np.mean((actual >= q10) & (actual <= q90))) >= target_coverage:
        return 1.0

    def coverage(scale: float) -> float:
        lower = q50 - scale * (q50 - q10)
        upper = q50 + scale * (q90 - q50)
        return float(np.mean((actual >= lower) & (actual <= upper)))

    low, high = 1.0, 4.0
    if coverage(high) < target_coverage:
        return high
    for _ in range(24):
        mid = (low + high) / 2
        if coverage(mid) >= target_coverage:
            high = mid
        else:
            low = mid
    return float(high)


def _quantile_metrics(*, quantile: float, actual: np.ndarray, predicted: np.ndarray) -> list[dict]:
    error = actual - predicted
    pinball = float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))
    rows = [{"metric": "pinball", "quantile": quantile, "value": pinball}]
    if quantile == 0.5:
        rows.append({"metric": "mae", "quantile": quantile, "value": float(np.mean(np.abs(error)))})
    return rows


def _enforce_quantile_order(frame: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
    ordered = sorted(quantiles)
    columns = [_quantile_column(q) for q in ordered if _quantile_column(q) in frame.columns]
    if len(columns) < 2:
        return frame
    values = np.sort(frame[columns].to_numpy(dtype=float), axis=1)
    out = frame.copy()
    for index, column in enumerate(columns):
        out[column] = values[:, index]
    return out


def _quantile_column(quantile: float) -> str:
    return f"p{int(round(quantile * 100)):02d}"


def _mean_path(model_dir: Path, series_id: str) -> Path:
    return model_dir / f"{series_id.replace('/', '_')}_mean.txt"


def _residual_path(model_dir: Path, series_id: str, quantile: float) -> Path:
    return model_dir / f"{series_id.replace('/', '_')}_resid_q{int(round(quantile * 100)):02d}.txt"
