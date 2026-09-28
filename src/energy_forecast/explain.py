"""SHAP explanations for the live short (24h) and medium (7d) forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.forecast import run_live_detail
from energy_forecast.model import MEAN_FEATURE, QuantileModels, load_day_models, load_hour_models, load_models
from energy_forecast.modes import MODE_SPEC, SHORT
from energy_forecast.settlement import LONDON_TZ

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
    stamp = pd.to_datetime(work["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    work["_bin"] = stamp.dt.floor(spec["display_freq"], ambiguous="infer", nonexistent="shift_forward")
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
