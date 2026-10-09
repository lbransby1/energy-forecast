"""Optuna search over LightGBM settings for the hour, day, and week models."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import optuna
import pandas as pd

from energy_forecast.evaluate import _naive_scale_mse, overall_metrics, score
from energy_forecast.features import DAY_V1_FEATURES, DAY_V2_FEATURES, DAY_V3_FEATURES, DAY_V4_FEATURES, FEATURE_COLUMNS, HOUR_FEATURE_COLUMNS, WEEK_V3_FEATURES
from energy_forecast.forecast import prepare_history
from energy_forecast.model import (
    DAY_MAX_LEAD_HOURS,
    DAY_N_ESTIMATORS,
    DEFAULT_QUANTILES,
    EXCLUDE_TRAIN_YEARS,
    HOLDOUT_DAYS,
    HOUR_EXCLUDE_TRAIN_YEARS,
    HOUR_MAX_LEAD_HOURS,
    LGB_DAY,
    LGB_MEAN,
    LGB_WEEK,
    TAIL_QUANTILES,
    WEEK_N_ESTIMATORS,
    fit_models,
    split_training_frame,
)
from energy_forecast.modes import MEDIUM, MODE_SPEC, SHORT, resample_to_mode
from energy_forecast.paths import DATA_PROCESSED, ensure_data_dirs

NEXT_HOUR = "next_hour"
WEEK_V3 = "week_v3"
DAY_V1 = "day_v1"
DAY_V2 = "day_v2"
DAY_V3 = "day_v3"
DAY_V4 = "day_v4"

PACK_LABELS = {
    NEXT_HOUR: "Next 30 min",
    SHORT: "Next 24 hours",
    MEDIUM: "Next week",
    WEEK_V3: "Week v3 (engineered)",
    DAY_V1: "Day v1 (engineered)",
    DAY_V2: "Day v2 (yesterday lags)",
    DAY_V3: "Day v3 (pruned)",
    DAY_V4: "Day v4 (24h deltas)",
}

PARAM_KEYS = (
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "n_estimators",
)

METRIC_KEYS = (
    "mae",
    "wrmsse",
    "rmse",
    "bias",
    "coverage_80",
    "interval_width",
    "skill_vs_climatology",
)

PRODUCTS = {
    NEXT_HOUR: {
        "weather_source": "archive",
        "features": HOUR_FEATURE_COLUMNS,
        "exclude_years": HOUR_EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": HOUR_MAX_LEAD_HOURS,
        "role": "next_hour",
        "eval_mode": None,
    },
    SHORT: {
        "weather_source": "previous_runs",
        "features": FEATURE_COLUMNS,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": DAY_MAX_LEAD_HOURS,
        "role": "next_day",
        "eval_mode": SHORT,
    },
    DAY_V1: {
        "weather_source": "previous_runs",
        "features": DAY_V1_FEATURES,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": DAY_MAX_LEAD_HOURS,
        "role": "next_day",
        "eval_mode": SHORT,
    },
    DAY_V2: {
        "weather_source": "previous_runs",
        "features": DAY_V2_FEATURES,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": DAY_MAX_LEAD_HOURS,
        "role": "next_day",
        "eval_mode": SHORT,
        "colsample_bytree_range": (0.70, 0.90),
    },
    DAY_V3: {
        "weather_source": "previous_runs",
        "features": DAY_V3_FEATURES,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": DAY_MAX_LEAD_HOURS,
        "role": "next_day",
        "eval_mode": SHORT,
    },
    DAY_V4: {
        "weather_source": "previous_runs",
        "features": DAY_V4_FEATURES,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": DAY_MAX_LEAD_HOURS,
        "role": "next_day",
        "eval_mode": SHORT,
        "colsample_bytree_range": (0.55, 0.75),
        "quantiles": TAIL_QUANTILES,
        "mean_objective": "mae",
    },
    MEDIUM: {
        "weather_source": "previous_runs",
        "features": FEATURE_COLUMNS,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": None,
        "role": "week",
        "eval_mode": MEDIUM,
    },
    WEEK_V3: {
        "weather_source": "previous_runs",
        "features": WEEK_V3_FEATURES,
        "exclude_years": EXCLUDE_TRAIN_YEARS,
        "max_lead_hours": None,
        "role": "week",
        "eval_mode": MEDIUM,
        "colsample_bytree_range": (0.60, 0.70),
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


def suggest_lgb_params(trial: optuna.Trial, product: str | None = None) -> dict:
    spec = PRODUCTS.get(product or "", {})
    col_lo, col_hi = spec.get("colsample_bytree_range", (0.6, 1.0))
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 95),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", float(col_lo), float(col_hi)),
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
    tree["feature_fraction"] = float(tree["colsample_bytree"])
    mean_tree = dict(tree)
    if spec.get("mean_objective"):
        mean_tree["objective"] = spec["mean_objective"]
        mean_tree["metric"] = spec["mean_objective"]
    return fit_models(
        train,
        n_estimators=int(params["n_estimators"]),
        features=spec["features"],
        quantiles=tuple(spec.get("quantiles", DEFAULT_QUANTILES)),
        mean_params=mean_tree,
        residual_params=tree,
        refit_full=refit_full,
        metadata={"role": spec["role"], "optuna_params": params},
    )


def naive_scale_for_product(train: pd.DataFrame, product: str) -> float:
    """Weekly seasonal-naive MSE on train, at the pack's display frequency."""
    mode = PRODUCTS[product].get("eval_mode")
    if mode:
        spec = MODE_SPEC[mode]
        return _naive_scale_mse(train, spec["freq"], spec["season"])
    return _naive_scale_mse(train, "30min", 336)


def holdout_mae(
    holdout: pd.DataFrame,
    models,
    product: str | None = None,
    *,
    train: pd.DataFrame | None = None,
    scale_mse: float | None = None,
) -> tuple[float, pd.DataFrame]:
    scored = score(holdout, models)
    if product and PRODUCTS[product].get("eval_mode"):
        scored = resample_to_mode(scored, PRODUCTS[product]["eval_mode"])
    if scale_mse is None and train is not None and product:
        scale_mse = naive_scale_for_product(train, product)
    metrics = overall_metrics(scored, scale_mse=scale_mse)
    return float(metrics["mae"].iloc[0]), metrics


def _trial_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    row = metrics.iloc[0]
    out: dict[str, float] = {}
    for column in METRIC_KEYS:
        if column in row.index and pd.notna(row[column]):
            out[column] = float(row[column])
    return out


def _trial_metric_map(trial, mae: float | None = None) -> dict[str, float]:
    value = mae if mae is not None else trial.value
    out: dict[str, float] = {}
    if value is not None:
        out["mae"] = float(value)
    for key in METRIC_KEYS:
        if key == "mae":
            continue
        raw = trial.user_attrs.get(key)
        if raw is not None and pd.notna(raw):
            out[key] = float(raw)
    return out


def _wandb_step_payload(trial, mae: float) -> dict:
    running = None
    try:
        import wandb

        if wandb.run is not None:
            running = wandb.run.summary.get("best_mae_so_far")
    except Exception:  # noqa: BLE001
        running = None
    best = mae if running is None else min(float(running), mae)
    payload = {
        "trial": int(trial.number),
        "mae": mae,
        "best_mae_so_far": best,
        **{key: value for key, value in _trial_metric_map(trial, mae).items() if key != "mae"},
        **{f"param/{key}": trial.params[key] for key in PARAM_KEYS if key in trial.params},
    }
    try:
        import wandb

        if wandb.run is not None:
            wandb.run.summary["best_mae_so_far"] = best
    except Exception:  # noqa: BLE001
        pass
    return payload


def log_trial(product: str, trial: optuna.Trial, mae: float, metrics: pd.DataFrame) -> None:
    """Log this trial onto the current W&B run (Charts x-axis = trial number)."""
    extra = _trial_metrics(metrics)
    extra["mae"] = mae
    for key, value in extra.items():
        trial.set_user_attr(key, value)
    try:
        import wandb

        if wandb.run is None:
            return
        wandb.log(_wandb_step_payload(trial, mae), step=int(trial.number))
    except Exception:  # noqa: BLE001
        pass


def _define_wandb_metrics() -> None:
    import wandb

    wandb.define_metric("trial")
    wandb.define_metric("mae", step_metric="trial", summary="min")
    wandb.define_metric("wrmsse", step_metric="trial", summary="min")
    wandb.define_metric("best_mae_so_far", step_metric="trial", summary="min")
    for key in METRIC_KEYS:
        if key in {"mae", "wrmsse"}:
            continue
        wandb.define_metric(key, step_metric="trial")
    for key in PARAM_KEYS:
        wandb.define_metric(f"param/{key}", step_metric="trial")


def _replay_study(study: optuna.Study) -> None:
    """Fill Charts with trials already in sqlite so a resumed search still shows the full curve."""
    import wandb

    if wandb.run is None:
        return
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        wandb.log(_wandb_step_payload(trial, float(trial.value)), step=int(trial.number))


def _wandb_config(product: str) -> dict:
    spec = PRODUCTS[product]
    return {
        "product": product,
        "label": PACK_LABELS[product],
        "objective": "holdout_mae",
        "weather_source": spec["weather_source"],
        "max_lead_hours": spec["max_lead_hours"],
        "eval_mode": spec["eval_mode"] or "half_hour",
        "role": spec["role"],
    }


@contextmanager
def tracking_run(product: str, study: optuna.Study | None = None, *, wandb_on: bool = True):
    """One W&B run per pack. Skip if wandb is missing or login fails."""
    wandb_mod = None
    if wandb_on:
        try:
            import wandb

            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "energy-forecast"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=f"{PACK_LABELS[product]} ({product})",
                group=product,
                job_type="optuna",
                tags=["optuna", product, PACK_LABELS[product]],
                config=_wandb_config(product),
                reinit=True,
            )
            _define_wandb_metrics()
            if study is not None:
                _replay_study(study)
            wandb_mod = wandb
            print("W&B", wandb.run.url)
            print("W&B group", product, "— filter Groups to compare the three packs")
        except Exception as exc:  # noqa: BLE001
            print("W&B skipped:", exc)
    try:
        yield
    finally:
        if wandb_mod is not None:
            try:
                wandb_mod.finish()
            except Exception:  # noqa: BLE001
                pass


def study_trial_rows(study: optuna.Study, product: str | None = None) -> list[dict]:
    """One row per completed trial, in trial order, with running best MAE."""
    rows: list[dict] = []
    running_best = float("inf")
    try:
        best_number = study.best_trial.number
    except ValueError:
        best_number = None
    for trial in sorted(study.trials, key=lambda item: item.number):
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        mae = float(trial.value)
        running_best = min(running_best, mae)
        row = {
            "product": product or study.study_name.removesuffix("_mae"),
            "trial": int(trial.number),
            "mae": mae,
            "best_mae_so_far": running_best,
            "is_best": trial.number == best_number,
            **{key: trial.params.get(key) for key in PARAM_KEYS},
            **{key: trial.user_attrs.get(key) for key in METRIC_KEYS if key != "mae"},
        }
        rows.append(row)
    return rows


def log_study_end(study: optuna.Study, product: str) -> None:
    """Winner summary plus a W&B table of every completed trial."""
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return
    best = study.best_trial
    summary = {
        "best_mae": float(best.value),
        "n_complete": float(len(complete)),
        "n_trials": float(len(study.trials)),
        "best_trial": float(best.number),
    }
    try:
        import wandb

        if wandb.run is None:
            return
        wandb.summary.update(summary)
        wandb.config.update(
            {"best_params": best.params, "best_trial": best.number},
            allow_val_change=True,
        )
        rows = study_trial_rows(study, product)
        if not rows:
            return
        columns = list(rows[0].keys())
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row.get(column) for column in columns])
        wandb.log({"optuna_trials": table})
    except Exception:  # noqa: BLE001
        pass


def make_objective(train: pd.DataFrame, holdout: pd.DataFrame, product: str):
    scale_mse = naive_scale_for_product(train, product)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgb_params(trial, product)
        models = fit_trial(train, product, params, refit_full=False)
        mae, metrics = holdout_mae(holdout, models, product, scale_mse=scale_mse)
        log_trial(product, trial, mae, metrics)
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


def baseline_params(product: str | None = None) -> dict:
    if product == WEEK_V3:
        return {
            "learning_rate": LGB_WEEK["learning_rate"],
            "num_leaves": LGB_WEEK["num_leaves"],
            "min_child_samples": LGB_WEEK["min_child_samples"],
            "subsample": LGB_WEEK["subsample"],
            "colsample_bytree": float(LGB_WEEK.get("feature_fraction", LGB_WEEK.get("colsample_bytree", 0.65))),
            "n_estimators": WEEK_N_ESTIMATORS,
        }
    if product == DAY_V4:
        return {
            "learning_rate": LGB_DAY["learning_rate"],
            "num_leaves": LGB_DAY["num_leaves"],
            "min_child_samples": LGB_DAY["min_child_samples"],
            "subsample": LGB_DAY["subsample"],
            "colsample_bytree": float(LGB_DAY.get("feature_fraction", LGB_DAY.get("colsample_bytree", 0.62))),
            "n_estimators": DAY_N_ESTIMATORS,
        }
    return {
        "learning_rate": LGB_MEAN["learning_rate"],
        "num_leaves": LGB_MEAN["num_leaves"],
        "min_child_samples": LGB_MEAN["min_child_samples"],
        "subsample": LGB_MEAN["subsample"],
        "colsample_bytree": LGB_MEAN["colsample_bytree"],
        "n_estimators": 400,
    }
