from __future__ import annotations

import optuna

from energy_forecast.lab import model_lab_payload, tracking_links
from energy_forecast.tune import PACK_LABELS, PRODUCTS


def test_model_lab_payload_lists_three_packs():
    out = model_lab_payload()
    ids = [row["id"] for row in out["packs"]]
    assert ids == ["hour", "day", "week"]
    week = next(row for row in out["packs"] if row["id"] == "week")
    assert week["ready"] is True
    assert week["tuned"] is True
    assert week["holdout_mae"] is not None
    day = next(row for row in out["packs"] if row["id"] == "day")
    assert day["tuned"] is False
    studies = {row["product"]: row for row in out["optuna"]}
    assert set(studies) == {"next_hour", "short", "medium"}
    assert studies["medium"]["ready"] is True
    assert studies["medium"]["n_complete"] > 0
    assert studies["next_hour"]["ready"] is True
    assert studies["short"]["ready"] is True


def test_tune_products_cover_all_three_packs():
    from energy_forecast.features import DAY_V3_FEATURES, DAY_V4_FEATURES, FEATURE_COLUMNS, WEEK_V3_FEATURES
    from energy_forecast.modes import MEDIUM

    assert {"next_hour", "short", "medium"}.issubset(PRODUCTS)
    assert set(PACK_LABELS) == set(PRODUCTS)
    assert PRODUCTS["next_hour"]["weather_source"] == "archive"
    assert PRODUCTS["short"]["max_lead_hours"] == 24.0
    assert PRODUCTS["medium"]["max_lead_hours"] is None
    assert PRODUCTS["week_v3"]["features"] == WEEK_V3_FEATURES
    assert PRODUCTS["week_v3"]["eval_mode"] == MEDIUM
    assert "demand_lag_48" not in PRODUCTS["week_v3"]["features"]
    assert PRODUCTS["day_v1"]["max_lead_hours"] == 24.0
    assert "demand_lag_48" in PRODUCTS["day_v1"]["features"]
    assert "temp_delta_7d" in PRODUCTS["day_v1"]["features"]
    assert PRODUCTS["day_v2"]["max_lead_hours"] == 24.0
    assert "demand_lag_48" in PRODUCTS["day_v2"]["features"]
    assert "demand_lag_96" in PRODUCTS["day_v2"]["features"]
    assert "temp_delta_1d" in PRODUCTS["day_v2"]["features"]
    assert "temp_ema_3h" in PRODUCTS["day_v2"]["features"]
    assert "demand_mean_3w" not in PRODUCTS["day_v2"]["features"]
    assert "temp_ema_3h" not in PRODUCTS["week_v3"]["features"]
    assert PRODUCTS["day_v3"]["max_lead_hours"] == 24.0
    assert PRODUCTS["day_v3"]["features"] == DAY_V3_FEATURES
    assert "demand_lag_48" in PRODUCTS["day_v3"]["features"]
    assert "temp_ema_3h" not in PRODUCTS["day_v3"]["features"]
    assert "demand_mean_3w" not in PRODUCTS["day_v3"]["features"]
    assert PRODUCTS["day_v4"]["max_lead_hours"] == 24.0
    assert PRODUCTS["day_v4"]["features"] == DAY_V4_FEATURES
    assert "demand_lag_48" in PRODUCTS["day_v4"]["features"]
    assert "temp_delta_1d" in PRODUCTS["day_v4"]["features"]
    assert "demand_delta_1d" in PRODUCTS["day_v4"]["features"]
    assert "temp_ema_3h" not in PRODUCTS["day_v4"]["features"]
    assert "temp_delta_7d" not in PRODUCTS["day_v4"]["features"]
    assert PRODUCTS["day_v4"]["quantiles"] == (0.1, 0.9)
    assert PRODUCTS["day_v4"]["mean_objective"] == "mae"
    assert 0.5 not in PRODUCTS["day_v4"]["quantiles"]


def test_day_v2_optuna_keeps_feature_fraction_in_band():
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        from energy_forecast.tune import suggest_lgb_params

        params = suggest_lgb_params(trial, "day_v2")
        assert 0.70 <= params["colsample_bytree"] <= 0.90
        return 1.0

    study.optimize(objective, n_trials=3)


def test_day_quantile_params_shrink_leaves():
    from energy_forecast.model import LGB_DAY, day_quantile_params

    out = day_quantile_params(LGB_DAY)
    assert out["num_leaves"] <= 31
    assert out["num_leaves"] == max(15, min(31, int(LGB_DAY["num_leaves"]) // 2))
    assert out["min_child_samples"] >= 80


def test_interval_scale_uses_pre_refit_trees(monkeypatch):
    import numpy as np
    import pandas as pd

    from energy_forecast import model as model_mod

    calls: list[int] = []

    class Dummy:
        def __init__(self, n):
            self.n = n

        def predict(self, X):
            return np.zeros(len(X))

    def fake_mean(frame, features, params, n_estimators):
        return Dummy(len(frame))

    def fake_quantile(features, demand_mw, quantile, n_estimators, params):
        return Dummy(len(features))

    def fake_scale(frame, mean_booster, residual_boosters, *args, **kwargs):
        calls.append(mean_booster.n)
        return 2.0

    monkeypatch.setattr(model_mod, "_fit_mean", fake_mean)
    monkeypatch.setattr(model_mod, "_fit_phase2_quantile", fake_quantile)
    monkeypatch.setattr(model_mod, "_coverage_scale", fake_scale)
    monkeypatch.setattr(model_mod, "demand_climatology", lambda frame: pd.DataFrame())

    idx = pd.date_range("2026-01-01", periods=48 * 30, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {
            "region_id": "gb",
            "timestamp": idx,
            "settlement_period": np.tile(np.arange(1, 49), 30),
            "demand_mw": 20000.0,
            "x": 1.0,
        }
    )
    models = model_mod.fit_models(frame, features=["x"], n_estimators=1, quantiles=(0.1, 0.9))
    fit_rows = int((frame["timestamp"] < frame["timestamp"].max() - pd.Timedelta(days=21)).sum())
    assert calls == [fit_rows]
    assert models.interval_scale["gb"] == 2.0


def test_predict_quantiles_uses_mean_as_p50_without_q50():
    import numpy as np
    import pandas as pd

    from energy_forecast.model import MEAN_FEATURE, TAIL_QUANTILES, QuantileModels, predict_quantiles

    class Const:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    features = ["x"]
    models = QuantileModels(
        mean_boosters={"gb": Const(100.0)},
        residual_boosters={("gb", 0.1): Const(80.0), ("gb", 0.9): Const(130.0)},
        interval_scale={"gb": 1.0},
        features=features,
        residual_features=[*features, MEAN_FEATURE],
        quantiles=TAIL_QUANTILES,
        climatology=pd.DataFrame(),
        metadata={"role": "next_day"},
    )
    frame = pd.DataFrame({"region_id": ["gb", "gb"], "x": [1.0, 2.0]})
    out = predict_quantiles(frame, models)
    assert list(out["p50"]) == [100.0, 100.0]
    assert list(out["p10"]) == [80.0, 80.0]
    assert list(out["p90"]) == [130.0, 130.0]
    assert 0.5 not in models.quantiles


def test_week_v3_optuna_keeps_feature_fraction_in_band():
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        from energy_forecast.tune import suggest_lgb_params

        params = suggest_lgb_params(trial, "week_v3")
        assert 0.60 <= params["colsample_bytree"] <= 0.70
        return 1.0

    study.optimize(objective, n_trials=3)


def test_day_v4_optuna_keeps_feature_fraction_in_band():
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        from energy_forecast.tune import suggest_lgb_params

        params = suggest_lgb_params(trial, "day_v4")
        assert 0.55 <= params["colsample_bytree"] <= 0.75
        return 1.0

    study.optimize(objective, n_trials=3)


def test_live_week_pack_is_v3():
    from energy_forecast.features import WEEK_V3_FEATURES
    from energy_forecast.forecast import frozen_feature_columns
    from energy_forecast.model import load_models

    models = load_models()
    assert models.metadata.get("role") == "week"
    assert models.metadata.get("long_lags_only") is True
    assert models.metadata.get("engineered_week") is True
    assert models.features == WEEK_V3_FEATURES
    assert "demand_lag_48" not in models.features
    frozen = frozen_feature_columns("week", models)
    assert "demand_lag_672" in frozen
    assert "demand_mean_3w" in frozen
    assert "temp_delta_7d" in frozen


def test_tracking_links_read_env(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT_URL", "https://wandb.ai/example/energy")
    links = tracking_links()
    assert links["wandb"].endswith("/energy")
    assert "mlflow" not in links
    assert "06_optuna" in links["notebook"]
