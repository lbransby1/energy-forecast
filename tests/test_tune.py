from __future__ import annotations

import optuna
from optuna.distributions import FloatDistribution, IntDistribution

from energy_forecast.tune import study_trial_rows


def _complete_trial(*, mae: float, learning_rate: float) -> optuna.trial.FrozenTrial:
    return optuna.trial.create_trial(
        value=mae,
        params={
            "learning_rate": learning_rate,
            "num_leaves": 20,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 200,
        },
        distributions={
            "learning_rate": FloatDistribution(0.02, 0.12, log=True),
            "num_leaves": IntDistribution(15, 95),
            "min_child_samples": IntDistribution(10, 80),
            "subsample": FloatDistribution(0.6, 1.0),
            "colsample_bytree": FloatDistribution(0.6, 1.0),
            "n_estimators": IntDistribution(150, 500, step=50),
        },
        user_attrs={"coverage_80": 0.85, "rmse": mae * 1.2, "wrmsse": 0.4},
    )


def test_study_trial_rows_running_best_and_winner():
    study = optuna.create_study(direction="minimize", study_name="short_mae")
    study.add_trial(_complete_trial(mae=200.0, learning_rate=0.05))
    study.add_trial(_complete_trial(mae=150.0, learning_rate=0.03))
    study.add_trial(_complete_trial(mae=180.0, learning_rate=0.04))
    rows = study_trial_rows(study, "short")
    assert [row["mae"] for row in rows] == [200.0, 150.0, 180.0]
    assert [row["best_mae_so_far"] for row in rows] == [200.0, 150.0, 150.0]
    assert [row["is_best"] for row in rows] == [False, True, False]
    assert rows[0]["product"] == "short"
    assert rows[1]["coverage_80"] == 0.85
    assert rows[1]["wrmsse"] == 0.4
