"""Holdout cards and Optuna snapshots for the dashboard lab column. Local files only."""

from __future__ import annotations

import json
import os
from pathlib import Path

from energy_forecast.paths import DATA_MODELS, DATA_MODELS_DAY, DATA_MODELS_HOUR, DATA_PROCESSED

PACKS = (
    {
        "id": "hour",
        "label": "Next 30 minutes",
        "dir": DATA_MODELS_HOUR,
        "search": "hour",
        "note": "ERA5 weather and lag_1 / lag_2. No Optuna winner wired in yet.",
    },
    {
        "id": "day",
        "label": "Next 24 hours",
        "dir": DATA_MODELS_DAY,
        "search": "short",
        "note": "Issued-forecast weather, leads up to 24h. Stock LightGBM until a short Optuna winner is copied in.",
    },
    {
        "id": "week",
        "label": "Next week",
        "dir": DATA_MODELS,
        "search": "medium",
        "note": "All leads. Production trees are the medium Optuna winner.",
    },
)

DEFAULT_NOTEBOOK = "https://github.com/lbransby1/energy-forecast/blob/main/notebooks/06_optuna.ipynb"


def _metric_map(model_dir: Path) -> dict[str, float]:
    path = model_dir / "metrics.csv"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    try:
        import pandas as pd

        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            name = str(row.get("metric", ""))
            if name in {"mae", "coverage_80", "interval_scale"}:
                out[name] = float(row["value"])
    except Exception:  # noqa: BLE001
        return {}
    return out


def _pack_card(spec: dict) -> dict:
    model_dir = Path(spec["dir"])
    meta_path = model_dir / "metadata.json"
    empty = {
        "id": spec["id"],
        "label": spec["label"],
        "search": spec["search"],
        "note": spec["note"],
        "ready": False,
        "tuned": False,
        "trained_at": None,
        "n_estimators": None,
        "n_holdout": None,
        "holdout_mae": None,
        "holdout_coverage": None,
        "role": None,
    }
    if not meta_path.exists():
        return empty
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return empty
    metrics = _metric_map(model_dir)
    params = meta.get("lgb_params") or {}
    return {
        **empty,
        "ready": True,
        "tuned": bool(params),
        "trained_at": meta.get("trained_at"),
        "n_estimators": meta.get("n_estimators"),
        "n_holdout": meta.get("n_holdout"),
        "holdout_mae": metrics.get("mae"),
        "holdout_coverage": metrics.get("coverage_80"),
        "role": meta.get("role"),
        "n_train": meta.get("n_train"),
        "max_lead_hours": meta.get("max_lead_hours"),
    }


def _optuna_snapshot(product: str) -> dict:
    path = DATA_PROCESSED / f"optuna_{product}.db"
    base = {
        "product": product,
        "ready": False,
        "n_trials": 0,
        "n_complete": 0,
        "best_mae": None,
        "best_params": None,
    }
    if not path.exists():
        return base
    try:
        import optuna

        study = optuna.load_study(
            study_name=f"{product}_mae",
            storage=f"sqlite:///{path.as_posix()}",
        )
        complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        best = study.best_trial if complete else None
        params = None
        if best is not None and best.params:
            params = {key: best.params[key] for key in sorted(best.params)}
        return {
            **base,
            "ready": True,
            "n_trials": int(len(study.trials)),
            "n_complete": int(len(complete)),
            "best_mae": float(best.value) if best is not None and best.value is not None else None,
            "best_params": params,
        }
    except Exception:  # noqa: BLE001
        return {**base, "error": "study file present but could not be read"}


def tracking_links() -> dict:
    wandb = (os.environ.get("WANDB_PROJECT_URL") or os.environ.get("WANDB_URL") or "").strip()
    mlflow = (os.environ.get("MLFLOW_UI_URL") or "").strip()
    notebook = (os.environ.get("OPTUNA_NOTEBOOK_URL") or DEFAULT_NOTEBOOK).strip()
    return {
        "wandb": wandb or None,
        "mlflow": mlflow or None,
        "notebook": notebook or None,
    }


def model_lab_payload() -> dict:
    """Cards for the three packs plus local Optuna study summaries."""
    packs = [_pack_card(spec) for spec in PACKS]
    searches = [_optuna_snapshot("short"), _optuna_snapshot("medium")]
    return {
        "packs": packs,
        "optuna": searches,
        "tracking": tracking_links(),
    }
