from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.evaluate import (
    _naive_scale_mse,
    _wrmsse,
    as_board_scored,
    board_naive_scale,
    overall_metrics,
)
from energy_forecast.features import apply_freeze_time_lags, build_features


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


def test_apply_freeze_time_lags_copies_last_week_past_24h():
    frame = pd.DataFrame(
        {
            "lead_hours": [12.0, 24.0, 108.0],
            "demand_lag_48": [21000.0, 28691.0, 25347.0],
            "demand_lag_336": [20000.0, 24045.0, 22923.0],
        }
    )
    out = apply_freeze_time_lags(frame)
    assert float(out["demand_lag_48"].iloc[0]) == 21000.0
    assert float(out["demand_lag_48"].iloc[1]) == 24045.0
    assert float(out["demand_lag_48"].iloc[2]) == 22923.0
    assert float(frame["demand_lag_48"].iloc[1]) == 28691.0


def test_apply_freeze_time_lags_noop_without_lead():
    frame = pd.DataFrame({"demand_lag_48": [1.0], "demand_lag_336": [2.0]})
    out = apply_freeze_time_lags(frame)
    assert float(out["demand_lag_48"].iloc[0]) == 1.0


def test_board_naive_scale_is_finite_on_long_series():
    idx = pd.date_range("2026-01-01", periods=400, freq="30min", tz="UTC")
    demand = pd.DataFrame({"timestamp": idx, "demand_mw": 20000.0 + np.arange(400)})
    scale = board_naive_scale(demand, days=20)
    assert np.isfinite(scale) and scale > 0


def test_as_board_scored_wrmsse_zero_when_p50_matches():
    idx = pd.date_range("2026-09-28", periods=48, freq="30min", tz="Europe/London")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "actual_mw": 22000.0,
            "p50": 22000.0,
            "p10": 20000.0,
            "p90": 24000.0,
            "settlement_period": np.arange(1, 49),
        }
    )
    scored = as_board_scored(frame)
    metrics = overall_metrics(scored, scale_mse=100.0**2)
    assert metrics["mae"].iloc[0] == 0.0
    assert metrics["wrmsse"].iloc[0] == 0.0


def test_week_long_lag_features_drop_yesterday():
    from energy_forecast.features import FEATURE_COLUMNS, WEEK_LONG_LAG_FEATURES

    assert "demand_lag_48" not in WEEK_LONG_LAG_FEATURES
    assert "demand_lag_1" not in WEEK_LONG_LAG_FEATURES
    assert "demand_lag_336" in WEEK_LONG_LAG_FEATURES
    assert "temperature_2m" in WEEK_LONG_LAG_FEATURES
    assert "shortwave_radiation" in WEEK_LONG_LAG_FEATURES
    assert WEEK_LONG_LAG_FEATURES == [c for c in FEATURE_COLUMNS if c != "demand_lag_48"]


def test_week_v3_features_are_engineered_not_yesterday():
    from energy_forecast.features import (
        FEATURE_COLUMNS,
        WEEK_ENGINEERED_FEATURES,
        WEEK_V3_FEATURES,
        WEEK_V3_LAG_FEATURES,
    )

    assert "demand_lag_48" not in WEEK_V3_FEATURES
    assert "demand_lag_336" in WEEK_V3_FEATURES
    assert "demand_lag_672" in WEEK_V3_FEATURES
    assert "demand_delta_14d_7d" in WEEK_V3_FEATURES
    assert "demand_mean_3w" in WEEK_V3_FEATURES
    assert "demand_accel_3w" in WEEK_V3_FEATURES
    assert "demand_lag_1008" not in WEEK_V3_FEATURES
    assert "demand_lag_1008" not in FEATURE_COLUMNS
    assert "temp_delta_7d" in WEEK_V3_FEATURES
    assert "apparent_temp_delta_7d" in WEEK_V3_FEATURES
    assert "temp_mean_24h_delta_7d" in WEEK_V3_FEATURES
    assert "demand_lag_336_mean_24h" not in WEEK_V3_FEATURES
    assert "demand_lag_336_mean_72h" not in WEEK_V3_FEATURES
    assert "hour" not in WEEK_V3_FEATURES
    assert "period_sin" not in WEEK_V3_FEATURES
    assert "period_cos" in WEEK_V3_FEATURES
    assert "settlement_period" in WEEK_V3_FEATURES
    assert "temp_delta_7d" not in FEATURE_COLUMNS
    assert "demand_lag_672" not in FEATURE_COLUMNS
    assert WEEK_V3_FEATURES == [
        column
        for column in [
            *(c for c in FEATURE_COLUMNS if c != "demand_lag_48"),
            *WEEK_ENGINEERED_FEATURES,
            *WEEK_V3_LAG_FEATURES,
        ]
        if column not in {"hour", "period_sin"}
    ]


def test_day_v1_keeps_yesterday_and_adds_engineered():
    from energy_forecast.features import DAY_V1_FEATURES, FEATURE_COLUMNS, WEEK_ENGINEERED_FEATURES, WEEK_V3_LAG_FEATURES

    assert "demand_lag_48" in DAY_V1_FEATURES
    assert "demand_lag_336" in DAY_V1_FEATURES
    assert "hour" in DAY_V1_FEATURES
    assert "period_sin" in DAY_V1_FEATURES
    assert "temp_delta_7d" in DAY_V1_FEATURES
    assert "demand_mean_3w" in DAY_V1_FEATURES
    assert "demand_lag_1" not in DAY_V1_FEATURES
    assert DAY_V1_FEATURES == [*FEATURE_COLUMNS, *WEEK_ENGINEERED_FEATURES, *WEEK_V3_LAG_FEATURES]


def test_day_v2_adds_yesterday_lags_not_three_week_mean():
    from energy_forecast.features import (
        DAY_V2_FEATURES,
        DAY_YESTERDAY_FEATURES,
        DAY_YESTERDAY_LAG_FEATURES,
        FEATURE_COLUMNS,
        WEEK_ENGINEERED_FEATURES,
        WEATHER_EMA_FEATURES,
    )

    assert "demand_lag_48" in DAY_V2_FEATURES
    assert "demand_lag_96" in DAY_V2_FEATURES
    assert "demand_delta_1d" in DAY_V2_FEATURES
    assert "temp_delta_1d" in DAY_V2_FEATURES
    assert "apparent_temp_delta_1d" in DAY_V2_FEATURES
    assert "demand_mean_3w" not in DAY_V2_FEATURES
    assert "demand_lag_672" not in DAY_V2_FEATURES
    assert "demand_lag_1" not in DAY_V2_FEATURES
    assert "temp_ema_3h" in DAY_V2_FEATURES
    assert "temp_ema_resid_3h" in DAY_V2_FEATURES
    assert "temp_ema_3h" not in FEATURE_COLUMNS
    assert DAY_V2_FEATURES == [
        *FEATURE_COLUMNS,
        *WEEK_ENGINEERED_FEATURES,
        *DAY_YESTERDAY_FEATURES,
        *DAY_YESTERDAY_LAG_FEATURES,
        *WEATHER_EMA_FEATURES,
    ]


def test_day_v4_adds_24h_deltas_not_week_or_ema():
    from energy_forecast.features import (
        DAY_V4_FEATURES,
        DAY_YESTERDAY_FEATURES,
        DAY_YESTERDAY_LAG_FEATURES,
        FEATURE_COLUMNS,
    )

    assert DAY_V4_FEATURES == [
        *FEATURE_COLUMNS,
        *DAY_YESTERDAY_FEATURES,
        *DAY_YESTERDAY_LAG_FEATURES,
    ]
    assert "demand_lag_48" in DAY_V4_FEATURES
    assert "temp_delta_24h" in DAY_V4_FEATURES
    assert "temp_delta_1d" in DAY_V4_FEATURES
    assert "demand_delta_1d" in DAY_V4_FEATURES
    assert "demand_lag_96" in DAY_V4_FEATURES
    assert "temp_delta_7d" not in DAY_V4_FEATURES
    assert "temp_ema_3h" not in DAY_V4_FEATURES
    assert "demand_mean_3w" not in DAY_V4_FEATURES
    assert len(DAY_V4_FEATURES) == len(set(DAY_V4_FEATURES))


def test_day_v3_prunes_to_shap_winners():
    from energy_forecast.features import DAY_V2_FEATURES, DAY_V3_FEATURES, FEATURE_COLUMNS

    assert "demand_lag_48" in DAY_V3_FEATURES
    assert "demand_lag_96" in DAY_V3_FEATURES
    assert "demand_lag_336" in DAY_V3_FEATURES
    assert "wind_delta_1d" in DAY_V3_FEATURES
    assert "shortwave_delta_1d" in DAY_V3_FEATURES
    assert "wind_delta_7d" in DAY_V3_FEATURES
    assert "shortwave_delta_7d" in DAY_V3_FEATURES
    assert "temp_mean_fwd_24h" in DAY_V3_FEATURES
    assert "lead_hours" in DAY_V3_FEATURES
    assert "demand_mean_3w" not in DAY_V3_FEATURES
    assert "demand_delta_1d" not in DAY_V3_FEATURES
    assert "temp_ema_3h" not in DAY_V3_FEATURES
    assert "hour" not in DAY_V3_FEATURES
    assert "hdd" not in DAY_V3_FEATURES
    assert "demand_lag_1" not in DAY_V3_FEATURES
    assert len(DAY_V3_FEATURES) < len(FEATURE_COLUMNS)
    assert len(DAY_V3_FEATURES) < len(DAY_V2_FEATURES)
    assert len(DAY_V3_FEATURES) == len(set(DAY_V3_FEATURES))


def test_week_engineered_temp_delta_matches_last_week():
    from energy_forecast.features import build_features

    n = 400
    ts = pd.date_range("2026-01-05", periods=n, freq="30min", tz="UTC")
    temp = np.linspace(5.0, 18.0, n)
    demand = np.linspace(20000.0, 28000.0, n)
    frame = pd.DataFrame(
        {
            "region_id": ["gb"] * n,
            "timestamp": ts,
            "settlement_date": ts.tz_convert("Europe/London").strftime("%Y-%m-%d"),
            "settlement_period": ((ts.tz_convert("Europe/London").hour * 2 + ts.tz_convert("Europe/London").minute // 30) % 48) + 1,
            "temperature_2m": temp,
            "apparent_temperature": temp - 2.0,
            "wind_speed_10m": 4.0,
            "shortwave_radiation": 100.0,
            "cloud_cover": 50.0,
            "demand_mw": demand,
        }
    )
    out = build_features(frame, climatology=None)
    later = 336
    yday = 48
    assert abs(float(out["temp_delta_7d"].iloc[later]) - (temp[later] - temp[0])) < 1e-6
    assert abs(float(out["temp_delta_1d"].iloc[yday]) - (temp[yday] - temp[0])) < 1e-6
    assert abs(float(out["apparent_temp_delta_7d"].iloc[later]) - ((temp[later] - 2.0) - (temp[0] - 2.0))) < 1e-6
    assert "temp_ema_3h" in out.columns
    assert abs(float(out["temp_ema_resid_3h"].iloc[later]) - (temp[later] - float(out["temp_ema_3h"].iloc[later]))) < 1e-6
    assert "demand_lag_336_mean_24h" not in out.columns
    assert "demand_lag_336_mean_72h" not in out.columns


def test_week_v3_14_day_demand_lag_and_week_delta():
    from energy_forecast.features import build_features

    n = 1100
    ts = pd.date_range("2026-01-05", periods=n, freq="30min", tz="UTC")
    demand = 20000.0 + np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "region_id": ["gb"] * n,
            "timestamp": ts,
            "settlement_date": ts.tz_convert("Europe/London").strftime("%Y-%m-%d"),
            "settlement_period": 1,
            "temperature_2m": 10.0,
            "demand_mw": demand,
        }
    )
    out = build_features(frame, climatology=None)
    later = 672
    assert abs(float(out["demand_lag_672"].iloc[later]) - demand[0]) < 1e-6
    assert abs(float(out["demand_lag_336"].iloc[later]) - demand[336]) < 1e-6
    assert abs(float(out["demand_delta_14d_7d"].iloc[later]) - (demand[0] - demand[336])) < 1e-6
    later21 = 1008
    lag7 = demand[later21 - 336]
    lag14 = demand[later21 - 672]
    lag21 = demand[later21 - 1008]
    assert abs(float(out["demand_mean_3w"].iloc[later21]) - (lag7 + lag14 + lag21) / 3.0) < 1e-6
    yday = 96
    assert abs(float(out["demand_lag_96"].iloc[yday]) - demand[0]) < 1e-6
    assert abs(float(out["demand_lag_48"].iloc[yday]) - demand[48]) < 1e-6
    assert abs(float(out["demand_delta_1d"].iloc[yday]) - (demand[48] - demand[0])) < 1e-6
    assert abs(float(out["demand_accel_3w"].iloc[later21]) - (lag7 - 2.0 * lag14 + lag21)) < 1e-6


def test_weather_ema_lags_a_temperature_step():
    from energy_forecast.features import PERIODS_3H, build_features

    n = 40
    ts = pd.date_range("2026-01-05", periods=n, freq="30min", tz="UTC")
    temp = np.concatenate([np.zeros(20), np.full(20, 10.0)])
    frame = pd.DataFrame(
        {
            "region_id": ["gb"] * n,
            "timestamp": ts,
            "settlement_date": ts.tz_convert("Europe/London").strftime("%Y-%m-%d"),
            "settlement_period": 1,
            "temperature_2m": temp,
            "apparent_temperature": temp,
            "wind_speed_10m": 4.0,
            "shortwave_radiation": 100.0,
            "demand_mw": 20000.0,
        }
    )
    out = build_features(frame, climatology=None)
    alpha = 2.0 / (PERIODS_3H + 1)
    assert abs(float(out["temp_ema_3h"].iloc[20]) - (alpha * 10.0)) < 1e-6
    assert abs(float(out["temp_ema_resid_3h"].iloc[20]) - (10.0 - alpha * 10.0)) < 1e-6
    assert float(out["temp_ema_3h"].iloc[22]) > float(out["temp_ema_12h"].iloc[22])


def test_resample_to_mode_spans_several_london_dst_folds():
    from energy_forecast.modes import MEDIUM, SHORT, resample_to_mode

    stamps = pd.concat(
        [
            pd.Series(pd.date_range(f"{year}-10-24", periods=48 * 10, freq="30min", tz="UTC"))
            for year in range(2020, 2027)
        ],
        ignore_index=True,
    )
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "region_id": "gb",
            "demand_mw": 25000.0,
            "p10": 24000.0,
            "p50": 25000.0,
            "p90": 26000.0,
        }
    )
    short = resample_to_mode(frame, SHORT)
    medium = resample_to_mode(frame, MEDIUM)
    assert not short.empty
    assert not medium.empty
