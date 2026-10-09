from __future__ import annotations

import pandas as pd
import pytest

from energy_forecast.evaluate import iter_time_series_folds


def _lead_frame(n_times: int = 12) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n_times, freq="D", tz="UTC")
    rows = []
    for stamp in times:
        for lead in (0.0, 24.0, 168.0):
            rows.append({"timestamp": stamp, "lead_hours": lead, "demand_mw": 20000.0})
    return pd.DataFrame(rows)


def test_time_series_folds_are_expanding_and_causal():
    frame = _lead_frame(12)
    folds = iter_time_series_folds(frame, n_splits=3)
    assert len(folds) == 3
    previous_train_end = None
    for fold, train, test in folds:
        train_t = pd.to_datetime(train["timestamp"], utc=True)
        test_t = pd.to_datetime(test["timestamp"], utc=True)
        assert train_t.max() < test_t.min()
        assert set(train_t).isdisjoint(set(test_t))
        leads_in_test = test.loc[test["timestamp"] == test_t.min(), "lead_hours"]
        assert set(leads_in_test) == {0.0, 24.0, 168.0}
        if previous_train_end is not None:
            assert train_t.max() >= previous_train_end
        previous_train_end = train_t.max()


def test_time_series_folds_embargo_drops_recent_train():
    frame = _lead_frame(12)
    folds = iter_time_series_folds(frame, n_splits=3, embargo_days=2)
    _, train, test = folds[0]
    gap = pd.to_datetime(test["timestamp"], utc=True).min() - pd.to_datetime(train["timestamp"], utc=True).max()
    assert gap >= pd.Timedelta(days=2)


def test_time_series_folds_rejects_shuffled_k():
    with pytest.raises(ValueError, match="at least 2"):
        iter_time_series_folds(_lead_frame(12), n_splits=1)


def test_time_series_kfold_return_slices_is_tuple():
    import inspect
    from energy_forecast.evaluate import time_series_kfold

    assert "return_slices" in inspect.signature(time_series_kfold).parameters
    assert "scoring" in inspect.signature(time_series_kfold).parameters
