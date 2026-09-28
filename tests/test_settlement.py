from __future__ import annotations

import pandas as pd

from energy_forecast.settlement import LONDON_TZ, attach_timestamp, normalize_settlement_date


def test_normalize_iso_and_neso_dates():
    raw = pd.Series(["2026-09-28", "28-Sep-26", "31-Oct-23"])
    parsed = normalize_settlement_date(raw)
    assert parsed.dt.strftime("%Y-%m-%d").tolist() == ["2026-09-28", "2026-09-28", "2023-10-31"]


def test_attach_timestamp_is_london_period_start():
    frame = pd.DataFrame(
        {
            "settlement_date": ["2026-09-28", "2026-09-28"],
            "settlement_period": [1, 25],
            "demand_mw": [20000.0, 28000.0],
        }
    )
    out = attach_timestamp(frame)
    ts = pd.to_datetime(out["timestamp"])
    assert str(ts.dt.tz) in {LONDON_TZ, "Europe/London"}
    assert int(ts.iloc[0].hour) == 0 and int(ts.iloc[0].minute) == 0
    assert int(ts.iloc[1].hour) == 12 and int(ts.iloc[1].minute) == 0
