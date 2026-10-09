from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.features import (
    apply_demand_path_lags,
    seed_demand_path,
    weather_path_from_forecast,
)
from energy_forecast.forecast import predict_quantiles_recursive
from energy_forecast.model import MEAN_FEATURE, QuantileModels


class _EchoLag:
    def predict(self, X):
        frame = pd.DataFrame(X)
        return frame["demand_lag_48"].to_numpy(dtype=float)


def _models() -> QuantileModels:
    booster = _EchoLag()
    features = ["demand_lag_48", "demand_lag_336", "lead_hours", "temperature_2m"]
    return QuantileModels(
        mean_boosters={"gb": booster},
        residual_boosters={
            ("gb", 0.1): booster,
            ("gb", 0.5): booster,
            ("gb", 0.9): booster,
        },
        interval_scale={"gb": 1.0},
        features=features,
        residual_features=[*features, MEAN_FEATURE],
        quantiles=(0.1, 0.5, 0.9),
        climatology=pd.DataFrame(),
        metadata={"role": "week"},
    )


def _week_frame() -> pd.DataFrame:
    monday = pd.date_range("2026-09-28", periods=96, freq="30min", tz="UTC")
    sunday = monday - pd.Timedelta(days=1)
    lag_48 = np.where(np.arange(96) < 48, 21000.0, 28000.0)
    return pd.DataFrame(
        {
            "region_id": "gb",
            "timestamp": monday,
            "settlement_period": np.tile(np.arange(1, 49), 2),
            "lead_hours": np.arange(96) * 0.5,
            "temperature_2m": 12.0,
            "temperature_actual": 21.0,
            "demand_lag_48": lag_48,
            "demand_lag_336": 19000.0,
            "demand_mw": np.nan,
            "sunday_stamp": sunday,
        }
    )


def test_seed_path_keeps_sunday_from_monday_lag_not_tuesday_climatology():
    frame = _week_frame()
    path = seed_demand_path(frame.drop(columns=["sunday_stamp"]))
    sunday = pd.Timestamp("2026-09-27 12:00", tz="UTC")
    monday = pd.Timestamp("2026-09-28 12:00", tz="UTC")
    assert path[("gb", sunday)] == 21000.0
    assert ("gb", monday) not in path


def test_tuesday_lag_48_rolls_from_monday_p50():
    frame = _week_frame().drop(columns=["sunday_stamp"])
    hat = predict_quantiles_recursive(frame, _models())
    monday = hat.iloc[:48]
    tuesday = hat.iloc[48:]
    assert float(monday["p50"].iloc[24]) == 21000.0
    assert float(tuesday["demand_lag_48"].iloc[24]) == 21000.0
    assert float(tuesday["p50"].iloc[24]) == 21000.0
    assert float(frame["demand_lag_48"].iloc[72]) == 28000.0


def test_weather_stays_on_forecast_not_era5():
    frame = _week_frame().drop(columns=["sunday_stamp"])
    out = weather_path_from_forecast(frame)
    assert float(out["temperature_2m"].mean()) == 12.0
    hat = predict_quantiles_recursive(frame, _models())
    assert float(hat["temperature_2m"].mean()) == 12.0
    assert "temperature_actual" not in hat.columns or float(hat["temperature_2m"].max()) < 20.0


def test_sparse_daily_leads_roll_lag_48_from_p50():
    monday = pd.Timestamp("2026-09-28 12:00", tz="UTC")
    tuesday = monday + pd.Timedelta(days=1)
    frame = pd.DataFrame(
        {
            "region_id": ["gb", "gb"],
            "timestamp": [monday, tuesday],
            "settlement_period": [25, 25],
            "lead_hours": [0.0, 24.0],
            "temperature_2m": [12.0, 12.0],
            "demand_lag_48": [21000.0, 28000.0],
            "demand_lag_336": [19000.0, 19000.0],
            "demand_mw": [np.nan, np.nan],
        }
    )
    hat = predict_quantiles_recursive(frame, _models()).sort_values("timestamp")
    assert float(hat["p50"].iloc[0]) == 21000.0
    assert float(hat["demand_lag_48"].iloc[1]) == 21000.0
    assert float(hat["p50"].iloc[1]) == 21000.0


def test_score_recursive_does_not_leak_test_demand():
    from energy_forecast.evaluate import score_recursive

    monday = pd.date_range("2026-09-28", periods=48, freq="30min", tz="UTC")
    tuesday = monday + pd.Timedelta(days=1)
    stamps = monday.union(tuesday).sort_values()
    frame = pd.DataFrame(
        {
            "region_id": "gb",
            "timestamp": stamps,
            "settlement_period": np.tile(np.arange(1, 49), 2),
            "lead_hours": np.concatenate([np.zeros(48), np.full(48, 24.0)]),
            "temperature_2m": 12.0,
            "demand_lag_48": np.concatenate([np.full(48, 21000.0), np.full(48, 28000.0)]),
            "demand_lag_336": 19000.0,
            "demand_mw": 99999.0,
            "is_weekend": 0,
            "is_holiday": 0,
        }
    )
    scored = score_recursive(frame, _models(), history=None)
    tue = scored.loc[pd.to_datetime(scored["timestamp"], utc=True) >= tuesday[0]]
    assert float(tue["demand_lag_48"].mean()) == 21000.0
    assert float(tue["p50"].mean()) == 21000.0


def test_apply_demand_path_lags_leaves_weather_alone():
    frame = pd.DataFrame(
        {
            "region_id": ["gb", "gb"],
            "timestamp": pd.to_datetime(["2026-09-28 00:00Z", "2026-09-29 00:00Z"]),
            "temperature_2m": [12.0, 13.0],
            "demand_lag_48": [28000.0, 28000.0],
        }
    )
    monday = pd.Timestamp("2026-09-28 00:00", tz="UTC")
    path = {("gb", monday): 22222.0}
    out = apply_demand_path_lags(frame, path, ["demand_lag_48", "temperature_2m"])
    assert float(out["temperature_2m"].iloc[1]) == 13.0
    assert float(out["demand_lag_48"].iloc[1]) == 22222.0
