# Technical abilities — GB National Demand quantile forecast

The product is a **P10 / P50 / P90** forecast of Great Britain National Demand, scored against **Elexon Insights INDO** after each half-hour settlement period.

Stack in one line: **Python 3.11, pandas, LightGBM, Optuna, SHAP, FastAPI, Parquet, Docker**, plus public energy and weather APIs.

---

## What was built 

National Demand is a half-hourly GB series (NESO `ND` historically, INDO live). The system produces three live products from separate model packs:

| Product | Horizon | Model | What “good” looks like on holdout |
|---|---|---|---|
| Next 30 minutes | one settlement period | Hour pack (ERA5 + last 30/60 min INDO) | tight band; lags must be the latest published outturn |
| Next 24 hours | midnight freeze | Day pack (Previous Runs weather) | ~830 MW MAE, ~82% in the 80% interval |
| Next 7 days | Monday 00:00 freeze | Week pack (Previous Runs, Optuna) | ~870 MW MAE, fan widens with lead |

The operational test is not the holdout table. A FastAPI service **issues a forecast, stores it, then joins INDO only after that period has ended**. MAE, coverage, bias, interval width, and WRMSSE on that archive are the numbers that matter.

Domain constraint that shaped the engineering: **no GSP / regional models**. There is no public live actual to score them. Only what can be validated live is in scope.

---

## Software engineering (SWE)

### Package and API design

- Installable library (`src/` layout, `pyproject.toml`, `pip install -e .`) with a public forecast API (`run_forecast`, `run_pack`) and a live ops module (`energy_forecast.live`).
- Clear module boundaries: `data/` (NESO, Elexon, Open-Meteo), `features`, `model`, `evaluate`, `forecast`, `live`, `app`, `explain`, `cases`, `tune`.
- Typed-ish Python 3.11 (`from __future__ import annotations`, dataclasses for model bundles and case studies).
- CLI for capture / score / board without the HTTP server (`python -m energy_forecast.live`).

### Time, calendars, and correctness

- Settlement periods in **Europe/London**, including **DST days with 46 or 50 periods**, not a naive 48 × 30 min grid.
- Timestamp convention: period **start**; INDO for period `T` is only known **after `T+30min`**.
- Joins on `(settlement_date, settlement_period)` rather than flooring hours, so a 12:30 row is not silently filled from 12:00.

### Data pipelines and reliability

- Historical demand from NESO CSVs; live series from Insights **without an API key**, with a growing-day cache (today’s file is incomplete at 09:00 and must be refetched all day).
- Weather from three Open-Meteo surfaces (ERA5 archive, operational forecast, Previous Runs) with **retries on 429/502/503/504**, backoff, and forecast-cache fallback when the API is down.
- Parquet caches; **atomic write** (temp file + `os.replace`) and an in-process lock so concurrent dashboard refreshes and the capture loop cannot tear `indo.parquet`.
- Frozen **feature snapshots** per day/week (and next-30) freeze so a live miss can be audited: lags, weather, last INDO timestamp, missing columns.

### Backend and product surface

- **FastAPI + uvicorn**: health, board JSON, CSV downloads, capture endpoints, HTML dashboard with Chart.js fans.
- Background capture loop aligned to **:10 / :40 London** so the previous settlement period’s INDO is usually published before the hour model runs.
- Auth: optional `LIVE_TOKEN` (Bearer) on mutating routes; **password gate on re-issuing the week freeze** so a passer-by cannot reset a running trial (`hmac.compare_digest`, secret from env or gitignored file).
- **Docker** image (`python:3.11-slim`) and Compose with a live-data volume so scores survive restarts.

### Frontend / UX engineering

- Single-page dashboard: three presets, London-time axis ticks, previous-window INDO in grey then the live fan, next-30 **issued history vs live dots**, input-health line, stale-lag audit table.
- ISO-8601 `Z` timestamps so the browser does not parse `+0000` as local and shift the live point by an hour.

### Production debugging

Lived issues that required SWE:

- Stale INDO cache treated a morning snapshot as “today complete” → hour model followed overnight demand through the Monday ramp.
- Hourly fallback copied 12:00 INDO onto 12:30 when settlement period was present.
- Chart.js `spanGaps` drew a diagonal spike across a midnight join; INDO series had to abut without interpolating a gap.
- Concurrent parquet writes; Open-Meteo 503s; port bind leftovers on Windows.

---

## Machine learning engineering (MLE)

### Problem formulation

- **Probabilistic forecast**, not a point estimate: P10 / P50 / P90 with a coverage target of 80%.
- **P50 is a residual quantile around a mean LightGBM**, not the mean booster itself. The mean is the “story” model; tails are a second stage.
- **Horizon-specific packs**: short lags (`demand_lag_1`, `demand_lag_2`) are valid for the next hour and leak or sit empty for day 7. Separate hour / day / week models, routed live by `lead_hours`.
- Train/serve weather **matched by product**: hour pack on ERA5 (nowcast-like); day/week on Previous Runs so holdout weather is the forecast that would have been available at issue, not perfect ERA5.

### Features

- Calendar: hour, settlement period, weekend, UK bank holidays (`holidays` + GB subdiv), Fourier terms.
- Weather: temperature, humidity, wind, shortwave, cloud, rain; **HDD/CDD vs 15.5 °C**.
- Trajectory: rolling past temperature, **forward 12h/24h means** on the forecast path already on the frame (fronts visible at issue time), seasonal and 7-day anomalies.
- Demand: lags 1, 2, 48, 336 (30 min, 60 min, day, week). **Climatology fills only 48/336** on far-future rows; short lags are last published INDO, never monthly-weekday means.
- Explicit **non-result**: `days_since_holiday` / `yesterday_was_holiday` did not fix the day-after-bank-holiday miss; trees still followed `lag_48`. Those flags were reverted.

### Training protocol and leakage control

- LightGBM mean + pinball-style residual boosters; **interval scale** calibrated on a recent window toward 80% coverage.
- Week fan **widens with lead** (monotone stretch to day 7) so coverage is slightly fat on purpose.
- **Exclude 2020–2023** from week/day training (COVID load + Previous Runs completeness from 2024). Hour pack can use 2022–2023 because it trains on ERA5.
- Holdout: last **56 days**, metrics at the **display resolution** (hourly short; 6-hourly medium for Optuna/WRMSSE so tuning is comparable).
- Walk-forward and year slices in the eval notebook; skill vs **period climatology**.

### Metrics that match the product

- MAE / RMSE / bias on P50.
- **80% coverage** and mean interval width (a 0.9 GW hour-model band will miss if lags are 8 GW stale; a 2.8 GW week band will not).
- **WRMSSE**: RMSE relative to a **weekly seasonal naive** (same settlement period last week), demand-weighted toward evening peak. **1.0 = copy-paste last week** on the scale set; **&lt; 1 beats it**. Live week freeze ~0.39 on Monday morning hours is not “39% of a forecast”; it is beating the naive.
- Pinball at 10/50/90 in the eval module.

### Tuning and model ops

- **Optuna** over LightGBM hyperparameters per product; search does not overwrite `data/models` until applied. Week winner logged (e.g. MAE 862 vs 928 baseline on medium holdout).
- Saved artefacts: boosters, climatology parquet, metadata (features, interval scale, role).
- Three on-disk packs (`data/models`, `.../day`, `.../hour`) loaded independently; live routing concatenates by lead mask.

### Failure analysis as part of the model

Case-study notebook: holiday, **working Tuesday after a holiday** (the real miss), clock-change, hot weekday, median weekday. Quantified: 1 Sep 2026 day-after MAE **1,225 MW** vs **715 MW** median weekday. That is how you talk about lag-based models in energy: persistence is the feature and the bug.

---

## Applied AI / AI engineering

### Serve, freeze, score

- Issue-time **feature freeze** (day/week/next-30 parquet + CSV). Retraining or a later weather file cannot rewrite what the model saw.
- Live **input health**: last INDO age, `lag_1` must match last published MW, NaN check on hour-feature columns. Hour pack **refuses to issue** if Insights INDO is older than 90 minutes; one retry, then skip.
- Hour lags **stamped from Insights** immediately before `predict`, not inferred from a weather-left join that can drop morning rows.

### Explainability

- **SHAP** (`pred_contrib`) on the **mean, P10, and P90** boosters for the live path. P50 is not explained as if it were the mean. `mean_pred` is not treated as a SHAP feature for the tails. (not available on online demo as of 28/9/26, is in notebook)
- Explanations routed by which pack scored that row (hour vs day vs week).

### Evaluation culture

- Holdout, walk-forward, **live archive** (issue → wait → join actual). The dashboard is the eval harness.
- Notebooks in order: EDA → weather → train → forecast → evaluate → Optuna → SHAP → cases. Research and production share the same library, not pasted cells.

### Human-in-the-loop

- Automatic Monday 00:00 week freeze vs **password-protected re-issue** so a trial week is not reset from the UI.
- Day freeze at midnight; next-30 every half hour. Operators can re-issue day or next-30 without touching the week test.

---

## Data engineering

- Multi-source merge: NESO historic + Insights live, left-join weather so **future hours exist without demand**.
- Half-hourly expansion of hourly weather (time interpolation) onto the settlement grid.
- UK city average (London, Birmingham, Manchester, Glasgow) as a cheap national weather proxy.
- Cache invalidation rules for a series that **grows through the day**.
- Parquet as the live board/archive format; CSV for human download.

---


## Skills checklist

| Area | Evidence in this repo |
|---|---|
| Python packaging | `src/` layout, setuptools, 3.11 |
| Dataframes / time series | pandas, PyArrow, DST-safe settlement index |
| Gradient boosting | LightGBM two-phase mean + quantile residuals |
| Probabilistic ML | P10/P50/P90, coverage calibration, lead-dependent width |
| Experimentation | Optuna, holdout vs live, notebooks that call the library |
| Explainability | SHAP on the models that actually scored the row |
| HTTP APIs | FastAPI, background tasks, Bearer token, password on a mutating action |
| Frontend enough to ship | Chart.js fans, London axes, CSV/JSON downloads |
| Reliability | retries, atomic parquet, locks, cache-vs-live INDO |
| Containers | Dockerfile, Compose, mounted live volume |
| Metrics design | WRMSSE vs weekly naive, skill vs climatology, MAE at display freq |
| Scientific honesty | reverted holiday features; documented the miss they did not fix |



