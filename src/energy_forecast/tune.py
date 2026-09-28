"""Optuna search over LightGBM settings for the hour, day, and week models."""

from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd

from energy_forecast.evaluate import overall_metrics, score
from energy_forecast.features import FEATURE_COLUMNS
from energy_forecast.forecast import prepare_history
from energy_forecast.model import (
    DAY_MAX_LEAD_HOURS,
    EXCLUDE_TRAIN_YEARS,
    HOLDOUT_DAYS,
    LGB_MEAN,
    fit_models,
    split_training_frame,
)
from energy_forecast.modes import MEDIUM, SHORT, resample_to_mode
from energy_forecast.paths import DATA_PROCESSED, ensure_data_dirs

PRODUCTS = {
    SHORT: {
        "weather_source": "previous_runs",
        "features": FEATURE_COLUMNS,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": DAY_MAX_LEAD_HOURS,
        "role": "next_day",
        "eval_mode": SHORT,
    },
    MEDIUM: {
        "weather_source": "previous_runs",
        "features": FEATURE_COLUMNS,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": None,
        "role": "week",
        "eval_mode": MEDIUM,
    },
}


def load_product_frame(product: str) -> pd.DataFrame:
    spec = PRODUCTS[product]
    return prepare_history(weather_source=spec["weather_source"])


def product_split(
    frame: pd.DataFrame,
    product: str,
    *,
    holdout_days: int = HOLDOUT_DAYS,
    train_sample: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = PRODUCTS[product]
    train, holdout = split_training_frame(
        frame,
        features=spec["features"],
        holdout_days=holdout_days,
        exclude_years=spec["exclude_years"],
        max_lead_hours=spec["max_lead_hours"],
    )
    if train_sample is not None and 0 < train_sample < 1:
        train = train.sample(frac=train_sample, random_state=42).sort_values("timestamp")
    return train, holdout


def suggest_lgb_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 95),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
    }


def fit_trial(
    train: pd.DataFrame,
    product: str,
    params: dict,
    *,
    refit_full: bool = False,
):
    spec = PRODUCTS[product]
    tree = {key: params[key] for key in ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree")}
    return fit_models(
        train,
        n_estimators=int(params["n_estimators"]),
        features=spec["features"],
        mean_params=tree,
        residual_params=tree,
        refit_full=refit_full,
        metadata={"role": spec["role"], "optuna_params": params},
    )


def holdout_mae(holdout: pd.DataFrame, models, product: str | None = None) -> tuple[float, pd.DataFrame]:
    scored = score(holdout, models)
    if product and PRODUCTS[product].get("eval_mode"):
        scored = resample_to_mode(scored, PRODUCTS[product]["eval_mode"])
    metrics = overall_metrics(scored)
    return float(metrics["mae"].iloc[0]), metrics


def make_objective(train: pd.DataFrame, holdout: pd.DataFrame, product: str):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgb_params(trial)
        models = fit_trial(train, product, params, refit_full=False)
        mae, metrics = holdout_mae(holdout, models, product)
        trial.set_user_attr("coverage_80", float(metrics["coverage_80"].iloc[0]))
        trial.set_user_attr("interval_width", float(metrics["interval_width"].iloc[0]))
        trial.set_user_attr("skill_vs_climatology", float(metrics["skill_vs_climatology"].iloc[0]))
        return mae

    return objective


def study_path(product: str) -> Path:
    ensure_data_dirs()
    return DATA_PROCESSED / f"optuna_{product}.db"


def create_study(product: str) -> optuna.Study:
    path = study_path(product)
    return optuna.create_study(
        study_name=f"{product}_mae",
        direction="minimize",
        storage=f"sqlite:///{path.as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )


def baseline_params() -> dict:
    return {
        "learning_rate": LGB_MEAN["learning_rate"],
        "num_leaves": LGB_MEAN["num_leaves"],
        "min_child_samples": LGB_MEAN["min_child_samples"],
        "subsample": LGB_MEAN["subsample"],
        "colsample_bytree": LGB_MEAN["colsample_bytree"],
        "n_estimators": 400,
    }
