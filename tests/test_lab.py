from __future__ import annotations

from energy_forecast.lab import model_lab_payload, tracking_links


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
    assert studies["medium"]["n_complete"] == 50
    assert studies["next_hour"]["ready"] is True
    assert studies["next_hour"]["n_complete"] == 20


def test_tracking_links_read_env(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT_URL", "https://wandb.ai/example/energy")
    monkeypatch.setenv("MLFLOW_UI_URL", "https://mlflow.example/energy")
    links = tracking_links()
    assert links["wandb"].endswith("/energy")
    assert links["mlflow"].startswith("https://mlflow")
    assert "06_optuna" in links["notebook"]
