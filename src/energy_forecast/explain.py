"""SHAP explanations for the live short (24h) and medium (7d) forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.forecast import run_live_detail
from energy_forecast.model import MEAN_FEATURE, QuantileModels, load_day_models, load_hour_models, load_models
from energy_forecast.modes import MODE_SPEC, SHORT, floor_london
from energy_forecast.features import DAY_YESTERDAY_FEATURES, HOURLY_VARS, TRAJECTORY_FEATURES, WEEK_ENGINEERED_FEATURES, WEATHER_EMA_FEATURES
from energy_forecast.regions import SERIES_ID

_MODEL_LOADERS = {
    "hour": load_hour_models,
    "day": load_day_models,
    "week": load_models,
}
OUTPUTS = ("mean", "p10", "p90")
_QUANTILE = {"p10": 0.1, "p90": 0.9}


def live_explained(mode: str = SHORT) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Score the live path and attach SHAP for the mean, P10, and P90 models.

    ``mean_pred`` is not treated as a feature. The mean booster is the P50
    story; P10/P90 SHAP is the extra pull after that mean is known.
    """
    if mode not in MODE_SPEC:
        raise ValueError("mode must be 'short' or 'medium'")
    spec = MODE_SPEC[mode]
    scored = run_live_detail(horizon_hours=spec["horizon_hours"])
    lead = pd.to_numeric(scored["lead_hours"], errors="coerce").fillna(0.0)
    scored = scored.loc[lead < spec["horizon_hours"]].copy()
    shap_by_output = {output: _explain_output(scored, output) for output in OUTPUTS}
    details = _prediction_details(scored, shap_by_output, mode)
    return scored, shap_by_output, details


def explain_scored(scored: pd.DataFrame, models: QuantileModels) -> dict[str, pd.DataFrame]:
    """SHAP for an already-scored frame using one saved model pack."""
    return {output: _shap_output(scored, models, output) for output in OUTPUTS}


def mean_abs_shap(shap_values: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    cols = [c for c in shap_values.columns if c != "shap_base"]
    ranking = shap_values.loc[:, cols].abs().mean().sort_values(ascending=False).head(n)
    return ranking.rename("mean_abs_shap").rename_axis("feature").reset_index()


_CALENDAR_SHAP = {
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
}
_WEATHER_SHAP = {
    *HOURLY_VARS,
    "hdd",
    "cdd",
    *TRAJECTORY_FEATURES,
    *WEEK_ENGINEERED_FEATURES,
    *DAY_YESTERDAY_FEATURES,
    *WEATHER_EMA_FEATURES,
}


def shap_source_share(shap_values: pd.DataFrame) -> pd.DataFrame:
    """Split mean |SHAP| into demand lags vs weather vs calendar vs lead.

    A week tree that is mostly ``demand_lag_*`` will copy yesterday (or the
    climatology fill) and ignore the weather freeze. That is train/serve skew,
    not a shuffled-label leak.
    """
    cols = [column for column in shap_values.columns if column != "shap_base"]
    if not cols:
        return pd.DataFrame()
    weights = shap_values.loc[:, cols].abs().mean()
    total = float(weights.sum())
    if total <= 0:
        return pd.DataFrame()

    def bucket(name: str) -> str:
        if str(name).startswith("demand_"):
            return "demand_lag"
        if name == "lead_hours":
            return "lead"
        if name in _CALENDAR_SHAP:
            return "calendar"
        if name in _WEATHER_SHAP:
            return "weather"
        return "other"

    grouped = weights.groupby(bucket).sum().rename("mean_abs_shap").rename_axis("source")
    out = grouped.reset_index()
    out["share"] = out["mean_abs_shap"] / total
    order = {"demand_lag": 0, "weather": 1, "calendar": 2, "lead": 3, "other": 4}
    return out.sort_values("source", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)


def _explain_output(scored: pd.DataFrame, output: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for model_name, group in scored.groupby("model", sort=False):
        models = _MODEL_LOADERS[str(model_name)]()
        parts.append(_shap_output(group, models, output))
    return pd.concat(parts).reindex(scored.index)


def _shap_output(frame: pd.DataFrame, models: QuantileModels, output: str) -> pd.DataFrame:
    series_id = str(frame["region_id"].iloc[0]) if "region_id" in frame.columns else "gb"
    if output == "mean":
        booster = models.mean_boosters[series_id]
        features = list(models.features)
        matrix = frame.loc[:, features]
    else:
        booster = models.residual_boosters[(series_id, _QUANTILE[output])]
        features = list(models.residual_features)
        matrix = frame.loc[:, features]
    contrib = np.asarray(booster.predict(matrix, pred_contrib=True))
    out = pd.DataFrame(contrib[:, :-1], columns=features, index=frame.index)
    if MEAN_FEATURE in out.columns:
        out = out.drop(columns=[MEAN_FEATURE])
    out["shap_base"] = contrib[:, -1]
    return out


def _prediction_details(
    scored: pd.DataFrame,
    shap_by_output: dict[str, pd.DataFrame],
    mode: str,
) -> pd.DataFrame:
    spec = MODE_SPEC[mode]
    work = scored.copy()
    work["_bin"] = floor_london(work["timestamp"], spec["display_freq"])
    pred = work.groupby("_bin", as_index=False).agg(
        timestamp=("_bin", "first"),
        model=("model", _main_model),
        lead_hours=("lead_hours", "min"),
        mean_pred=("mean_pred", "mean"),
        p10=("p10", "mean"),
        p50=("p50", "mean"),
        p90=("p90", "mean"),
        interval_width=("interval_width", "mean"),
        temperature_2m=("temperature_2m", "mean"),
        n=("p50", "size"),
    )
    binned = {
        output: shap_by_output[output].drop(columns=["shap_base"], errors="ignore").groupby(work["_bin"]).mean()
        for output in OUTPUTS
    }
    rows: list[dict] = []
    for _, row in pred.iterrows():
        detail = {
            "timestamp": row["timestamp"],
            "model": row["model"],
            "lead_hours": row["lead_hours"],
            "mean_pred": row["mean_pred"],
            "p10": row["p10"],
            "p50": row["p50"],
            "p90": row["p90"],
            "interval_width": row["interval_width"],
            "temperature_2m": row["temperature_2m"],
            "n_periods": int(row["n"]),
        }
        for output in OUTPUTS:
            contrib = binned[output].loc[row["timestamp"]] if row["timestamp"] in binned[output].index else pd.Series(dtype=float)
            ranked = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
            for index, (feature, value) in enumerate(ranked.head(3).items(), start=1):
                detail[f"{output}_top{index}"] = feature
                detail[f"{output}_top{index}_shap"] = float(value)
        rows.append(detail)
    return pd.DataFrame(rows)


def _main_model(names: pd.Series) -> str:
    return names.mode().iloc[0] if not names.mode().empty else str(names.iloc[0])


FEATURE_LABELS = {
    "lead_hours": "Lead time after freeze",
    "hour": "Hour of day",
    "minute": "Minute",
    "day_of_week": "Weekday (Mon=0)",
    "is_weekend": "Weekend",
    "is_holiday": "Bank holiday",
    "month": "Month",
    "settlement_period": "Settlement period",
    "temperature_2m": "Temperature",
    "apparent_temperature": "Apparent temperature",
    "shortwave_radiation": "Sun (shortwave)",
    "cloud_cover": "Cloud cover",
    "relative_humidity_2m": "Humidity",
    "wind_speed_10m": "Wind",
    "precipitation": "Rain",
    "hdd": "Heating degrees",
    "cdd": "Cooling degrees",
    "demand_lag_1": "Latest published INDO",
    "demand_lag_2": "Previous INDO print",
    "demand_lag_48": "Demand same time yesterday",
    "demand_lag_96": "Demand same time 2 days ago",
    "demand_lag_336": "Demand same time last week",
    "demand_lag_672": "Demand same time 14 days ago",
    "demand_delta_14d_7d": "Demand 14 days ago minus last week",
    "demand_mean_3w": "Mean demand same time last 3 weeks",
    "demand_accel_3w": "Demand weekly acceleration (3 weeks)",
    "temp_delta_7d": "T vs last week (same half-hour)",
    "temp_mean_24h_delta_7d": "24h mean T vs last week",
    "apparent_temp_delta_7d": "Apparent T vs last week",
    "temp_mean_fwd_12h": "Forecast T next 12h",
    "temp_mean_fwd_24h": "Forecast T next 24h",
    "hdd_sum_24h": "HDD last 24h",
    "hdd_sum_72h": "HDD last 72h",
}

_DASHBOARD_PACKS = (
    {
        "id": "hour",
        "preset": "next30",
        "label": "Next 30 minutes",
        "loader": load_hour_models,
        "how": (
            "ERA5 weather plus the last two published INDO prints (lag_1, lag_2). "
            "Issued every half-hour. Skipped unless lag_1 and lag_2 are the latest two INDO prints (and not older than 90 minutes). "
            "Mean tree, then P10/P90 quantile trees on that mean."
        ),
    },
    {
        "id": "day",
        "preset": "day",
        "label": "Next 24 hours",
        "loader": load_day_models,
        "how": (
            "Previous Runs weather at a midnight London freeze. Leads 0–24h, so yesterday "
            "(lag_48) is known. Stock LightGBM mean, then residual quantiles. "
            "interval_scale stretches P10/P90 toward 80% coverage."
        ),
    },
    {
        "id": "week",
        "preset": "week",
        "label": "Next week",
        "loader": load_models,
        "how": (
            "Monday 00:00 London freeze. No lag_48 — that day has not happened for most of "
            "the horizon. Week-over-week weather and 14-day / 3-week demand extras. "
            "Direct predict (no recursive lag walk). Optuna LGB_WEEK; the fan widens with lead."
        ),
    },
)


def shap_dashboard_payload(*, sample_rows: int = 96) -> dict:
    """Mean-tree SHAP for the three live packs. Frozen parquet + model files only."""
    packs = [_shap_pack_card(spec, sample_rows) for spec in _DASHBOARD_PACKS]
    return {
        "ready": any(bool(pack.get("shap")) for pack in packs),
        "note": (
            "SHAP is LightGBM pred_contrib on the mean tree, averaged over frozen "
            "issue-time rows. No weather API. Positive push raises the MW guess."
        ),
        "packs": packs,
    }


def _shap_pack_card(spec: dict, sample_rows: int) -> dict:
    empty = {
        "id": spec["id"],
        "label": spec["label"],
        "how": spec["how"],
        "ready": False,
        "n": 0,
        "n_features": 0,
        "note": "No frozen inputs or pack on this box yet.",
        "shap": [],
        "sources": [],
    }
    try:
        models = spec["loader"]()
    except Exception:  # noqa: BLE001
        return empty
    frame = _latest_frozen_sample(spec["preset"], models, sample_rows)
    empty["n_features"] = int(len(models.features))
    if frame.empty:
        empty["note"] = "Pack is loaded. Issue a freeze to compute SHAP on frozen rows."
        return empty
    contrib = _shap_output(frame, models, "mean")
    ranking = mean_abs_shap(contrib, n=12)
    sources = shap_source_share(contrib)
    return {
        **empty,
        "ready": True,
        "n": int(len(frame)),
        "note": f"{len(frame)} frozen rows · {len(models.features)} inputs",
        "shap": [
            {
                "feature": str(row["feature"]),
                "label": FEATURE_LABELS.get(str(row["feature"]), str(row["feature"])),
                "mean_abs_shap": float(row["mean_abs_shap"]),
            }
            for _, row in ranking.iterrows()
        ],
        "sources": [
            {
                "source": str(row["source"]),
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "share": float(row["share"]),
            }
            for _, row in sources.iterrows()
        ],
    }


def _latest_frozen_sample(preset: str, models: QuantileModels, sample_rows: int) -> pd.DataFrame:
    from energy_forecast.live import frozen_dir

    paths = sorted(frozen_dir().glob(f"{preset}_*.parquet"))
    if not paths:
        return pd.DataFrame()
    frame = pd.read_parquet(paths[-1])
    if frame.empty:
        return pd.DataFrame()
    if "region_id" not in frame.columns:
        frame["region_id"] = SERIES_ID
    step = max(1, len(frame) // int(sample_rows))
    sample = frame.iloc[::step].head(int(sample_rows)).copy()
    for column in models.features:
        if column not in sample.columns:
            sample[column] = 0.0
        else:
            sample[column] = pd.to_numeric(sample[column], errors="coerce").fillna(0.0)
    return sample.reset_index(drop=True)
