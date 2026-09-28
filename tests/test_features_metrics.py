from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.evaluate import _naive_scale_mse, _wrmsse
from energy_forecast.features import build_features


def test_weekly_naive_scale_is_one_on_its_own_residuals():
    idx = pd.date_range("2024-01-01", periods=400, freq="30min", tz="UTC")
    # Strictly weekly repeating series: naive residual is 0.
    demand = np.tile(np.linspace(18000, 30000, 336), 2)[:400]
    frame = pd.DataFrame({"timestamp": idx, "demand_mw": demand})
    scale = _naive_scale_mse(frame, "30min", 336)
    assert scale == 0.0 or scale < 1e-6


def test_wrmsse_is_one_when_forecast_equals_weekly_naive_errors():
    n = 48
    actual = np.full(n, 25000.0)
    # Copy-paste last week on this window: error 0 relative to a scale of 100^2.
    scored = pd.DataFrame(
        {
            "demand_mw": actual,
            "p50": actual,
            "period_band": ["morning"] * n,
        }
    )
    assert _wrmsse(scored, scale_mse=100.0**2) == 0.0


def test_build_features_adds_hdd_and_calendar():
    ts = pd.date_range("2026-09-28", periods=4, freq="30min", tz="Europe/London")
    frame = pd.DataFrame(
        {
            "region_id": ["gb"] * 4,
            "timestamp": ts,
            "settlement_date": ["2026-09-28"] * 4,
            "settlement_period": [1, 2, 3, 4],
            "temperature_2m": [10.0, 10.0, 20.0, 20.0],
            "demand_mw": [20000.0, 20100.0, np.nan, np.nan],
        }
    )
    out = build_features(frame, climatology=None)
    assert "hdd" in out.columns and "demand_lag_48" in out.columns
    assert float(out["hdd"].iloc[0]) == 5.5
    assert float(out["cdd"].iloc[2]) == 4.5
