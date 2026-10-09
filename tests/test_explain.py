from __future__ import annotations

import pandas as pd

from energy_forecast.explain import shap_dashboard_payload, shap_source_share


def test_shap_source_share_splits_lags_from_weather():
    shap_values = pd.DataFrame(
        {
            "demand_lag_48": [4000.0, 4000.0],
            "demand_lag_336": [1000.0, 1000.0],
            "temperature_2m": [100.0, 100.0],
            "hour": [50.0, 50.0],
            "lead_hours": [20.0, 20.0],
            "shap_base": [20000.0, 20000.0],
        }
    )
    out = shap_source_share(shap_values)
    share = out.set_index("source")["share"]
    assert share["demand_lag"] > 0.8
    assert share["weather"] < 0.1
    assert "shap_base" not in set(out["source"])


def test_shap_source_share_buckets_week_engineered_as_weather():
    shap_values = pd.DataFrame(
        {
            "temp_delta_7d": [200.0, 200.0],
            "apparent_temp_delta_7d": [100.0, 100.0],
            "temp_delta_1d": [80.0, 80.0],
            "temp_ema_3h": [40.0, 40.0],
            "temp_ema_resid_3h": [20.0, 20.0],
            "shortwave_mean_24h": [50.0, 50.0],
            "demand_lag_336": [40.0, 40.0],
            "demand_delta_14d_7d": [20.0, 20.0],
            "demand_mean_3w": [8.0, 8.0],
            "demand_accel_3w": [4.0, 4.0],
            "hour": [10.0, 10.0],
        }
    )
    share = shap_source_share(shap_values).set_index("source")["share"]
    assert share["weather"] > 0.7
    assert share["demand_lag"] < 0.2


def test_shap_dashboard_payload_lists_three_packs():
    payload = shap_dashboard_payload(sample_rows=8)
    assert {row["id"] for row in payload["packs"]} == {"hour", "day", "week"}
    assert "pred_contrib" in (payload.get("note") or "")
