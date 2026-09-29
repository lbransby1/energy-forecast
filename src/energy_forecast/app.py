"""Live forecast service: capture, join INDO, show the score."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response

from energy_forecast.live import (
    NEXT30,
    PRESETS,
    WeekReissueLocked,
    capture_preset,
    forecast_csv,
    frozen_csv,
    latest_forecast,
    live_metrics,
    load_archive,
    load_board,
    refresh_actuals,
    scored_frame,
    seconds_until_half_hour,
    snapshot,
    tick_presets,
    _frame_records,
)
from energy_forecast.modes import SHORT

LIVE_TOKEN = os.environ.get("LIVE_TOKEN", "")


def _check_token(authorization: str | None) -> None:
    if not LIVE_TOKEN:
        return
    expected = f"Bearer {LIVE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid token")


async def _capture_loop() -> None:
    """Forecasts run in a worker thread so GET / and GET /board stay responsive."""
    await asyncio.sleep(2)
    while True:
        try:
            await asyncio.to_thread(tick_presets)
        except Exception as exc:  # noqa: BLE001 — loop must not die
            print(f"live capture failed: {exc}")
        await asyncio.sleep(seconds_until_half_hour())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("EF_SKIP_CAPTURE") == "1":
        yield
        return
    from energy_forecast.live import apply_next30_stats_epoch

    apply_next30_stats_epoch()
    task = asyncio.create_task(_capture_loop())
    yield
    task.cancel()


app = FastAPI(title="Grid Demand UK", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)


_STATIC = Path(__file__).resolve().parent / "static"


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/chart.js")
def chart_js() -> FileResponse:
    return FileResponse(
        _STATIC / "chart.umd.min.js",
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/status")
def status() -> dict:
    return snapshot()


@app.get("/forecast")
def forecast(mode: str = Query(default=SHORT)) -> dict:
    frame = latest_forecast(mode)
    return {"mode": mode, "n": int(len(frame)), "rows": _frame_records(frame)}


@app.get("/score")
def score() -> dict:
    try:
        refresh_actuals()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    metrics = live_metrics()
    scored = scored_frame()
    return {
        "metrics": metrics.to_dict(orient="records"),
        "n": int(len(scored)),
        "rows": _frame_records(scored.sort_values("timestamp"), limit=400),
    }


@app.get("/board")
def board() -> dict:
    try:
        from energy_forecast.live import board_payload

        return board_payload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/download/{preset}")
def download_forecast(preset: str, which: str = Query(default="current")) -> Response:
    if preset not in PRESETS:
        raise HTTPException(status_code=404, detail="unknown preset")
    csv = forecast_csv(preset, which)
    if not csv:
        raise HTTPException(status_code=404, detail="no forecast for that window")
    name = f"{preset}_{which}_forecast.csv"
    return Response(
        csv,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/download/{preset}/frozen")
def download_frozen(preset: str, which: str = Query(default="current")) -> Response:
    if preset not in PRESETS:
        raise HTTPException(status_code=404, detail="unknown preset")
    try:
        csv = frozen_csv(preset, which)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not csv:
        raise HTTPException(status_code=404, detail="no frozen inputs for that window")
    name = f"{preset}_{which}_frozen.csv"
    return Response(
        csv,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/inputs/{preset}")
def inputs(preset: str, which: str = Query(default="current")) -> dict:
    if preset not in PRESETS:
        raise HTTPException(status_code=404, detail="unknown preset")
    from energy_forecast.live import (
        _windows,
        input_health,
        load_frozen,
        next30_input_audit,
        _preset_frame,
        previous_week_window,
        week_window,
    )

    if preset == "week":
        window = week_window() if which != "previous" else previous_week_window()
    else:
        windows = _windows(preset)
        if not windows:
            raise HTTPException(status_code=404, detail="no window for that preset")
        window = windows[0] if which != "previous" else (windows[1] if len(windows) > 1 else windows[0])
    payload = input_health(preset, window)
    payload["preset"] = preset
    payload["window_start"] = window.isoformat() if hasattr(window, "isoformat") else str(window)
    frozen = load_frozen(preset, window)
    payload["columns"] = list(frozen.columns) if not frozen.empty else []
    if preset == NEXT30:
        payload["audit"] = next30_input_audit(_preset_frame(NEXT30))
    return payload


@app.post("/capture")
def run_capture(
    authorization: str | None = Header(default=None),
    force: bool = Query(default=False),
    x_week_password: str | None = Header(default=None),
) -> dict:
    _check_token(authorization)
    try:
        return tick_presets(force=force, week_password=x_week_password)
    except WeekReissueLocked as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/capture/{preset}")
def run_capture_preset(
    preset: str,
    authorization: str | None = Header(default=None),
    x_week_password: str | None = Header(default=None),
) -> dict:
    if preset not in PRESETS:
        raise HTTPException(status_code=404, detail="unknown preset")
    _check_token(authorization)
    try:
        result = capture_preset(preset, force=True, week_password=x_week_password)
        result["actuals"] = refresh_actuals()
        return result
    except WeekReissueLocked as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(
        DASHBOARD_HTML,
        headers={"Cache-Control": "no-store"},
    )


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Grid Demand UK</title>
  <style>
    :root {
      font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
      font-variant-numeric: lining-nums tabular-nums;
      font-feature-settings: "tnum" 1, "lnum" 1;
      color: #1a1a1a;
      background: #f7f4ee;
    }
    body { max-width: 1680px; margin: 24px auto; padding: 0 16px 40px; }
    h1 { font-size: 1.55rem; font-weight: 600; margin-bottom: 0.3rem; }
    h2 { font-size: 1.15rem; margin: 0; }
    .muted { color: #5c5c5c; font-size: 0.92rem; }
    .lede { max-width: 70rem; margin: 0 0 12px; line-height: 1.45; }
    .explain { color: #5c5c5c; font-size: 0.9rem; line-height: 1.45; margin: 6px 0 10px; }
    .glossary { display: grid; grid-template-columns: 7.5rem 1fr; gap: 4px 12px; margin: 8px 0 12px; font-size: 0.88rem; line-height: 1.4; color: #444; }
    .glossary dt { font-weight: 600; color: #1a1a1a; }
    .glossary dd { margin: 0; }
    .key { display: flex; flex-wrap: wrap; gap: 14px 22px; margin: 10px 0 4px; font-size: 0.92rem; }
    .key span { display: inline-flex; align-items: center; gap: 8px; }
    .swatch { width: 22px; height: 4px; border-radius: 1px; }
    .swatch.black { background: #111; }
    .swatch.red { background: #c45c26; height: 5px; }
    .swatch.blue { background: #9bb8d3; }
    .panel { background: #fff; border: 1px solid #ddd4c4; padding: 14px 16px 10px; margin: 0; }
    .board { display: grid; grid-template-columns: 1fr 1fr minmax(280px, 24rem); grid-template-areas: "week week side" "day n30 side"; gap: 14px; align-items: start; margin: 14px 0; }
    .side { grid-area: side; display: flex; flex-direction: column; gap: 14px; }
    .panel.week { grid-area: week; }
    .panel.day { grid-area: day; }
    .panel.n30 { grid-area: n30; }
    .eval .cards, .lab .cards { margin-top: 8px; }
    table.eval-miss, table.audit, table.lab-packs { font-variant-numeric: lining-nums tabular-nums; }
    table.eval-miss, table.lab-packs { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 8px; }
    table.eval-miss th, table.eval-miss td, table.lab-packs th, table.lab-packs td { text-align: left; padding: 4px 5px; border-bottom: 1px solid #eee; }
    @media (max-width: 1100px) {
      .board { grid-template-columns: 1fr; grid-template-areas: "week" "day" "n30" "side"; }
    }
    .head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; align-items: baseline; }
    .cards { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 6px; }
    .card { border: 1px solid #eee; padding: 8px 10px; min-width: 110px; min-height: 4.1rem; cursor: help; display: flex; flex-direction: column; justify-content: space-between; }
    .card b { display: block; font-size: 1.05rem; font-weight: 600; line-height: 1.25; letter-spacing: 0; white-space: nowrap; font-variant-numeric: lining-nums tabular-nums; }
    .links a { margin-right: 12px; color: #1a1a1a; }
    .links .muted { margin-right: 12px; }
    button { font: inherit; padding: 8px 14px; cursor: pointer; }
    .panel button { font-size: 0.92rem; padding: 6px 12px; margin: 8px 0 4px; }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .chart-wrap { position: relative; width: 100%; }
    .chart-wrap.week { height: 340px; }
    .chart-wrap.half { height: 300px; }
    canvas { background: #fff; }
    .warn { color: #8b1e1e; }
    .ok { color: #1a5c38; }
    table.audit { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 8px; }
    table.audit th, table.audit td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #eee; }
  </style>
</head>
<body>
  <h1>Grid Demand UK</h1>
  <p class="lede">How much electricity Great Britain is using, and what this site guessed before the official figure was published. There are three&nbsp;views of the forecast (week, day, half-hour), a check on finished half-hours, and a lab column for holdout scores and search. Times are UK. The black line is only drawn after each half-hour has ended.</p>
  <p class="key" aria-label="Chart colour key">
    <span><i class="swatch black"></i> Black — what actually happened (official grid outturn)</span>
    <span><i class="swatch red"></i> Red — the central forecast (most likely demand)</span>
    <span><i class="swatch blue"></i> Blue — a likely range (low to high). The black line should usually sit between them.</span>
  </p>
  <dl class="glossary" aria-label="Score cards">
    <dt>n</dt>
    <dd>How many half-hours already have an official number, so they can be scored. “Waiting” means none yet.</dd>
    <dt>MAE</dt>
    <dd>Typical size of the miss, in megawatts. Average of how far the red line was from the black line. Lower is better.</dd>
    <dt>WRMSSE</dt>
    <dd>Error compared with copying last week. <b>1.0</b> = as good as paste-last-week. Below 1 beats that. Above 1 is worse.</dd>
    <dt>Coverage</dt>
    <dd>How often the black line landed between the two blue lines. About 80% is the aim.</dd>
    <dt>Bias</dt>
    <dd>On average, is the red line too high (positive) or too low (negative), in megawatts.</dd>
    <dt>Width</dt>
    <dd>Average gap between the two blue lines across the <b>whole issued forecast</b> (including hours that have not finished yet). The week fan is meant to get wider later in the week, so that pulls this number up. MAE and coverage still use only finished half-hours (<b>n</b>).</dd>
  </dl>
  <p>
    <button type="button" id="reload">Reload board</button>
    <button type="button" id="capture">Issue due presets</button>
    <span class="muted" id="note">Loading board…</span>
  </p>

  <div class="board">
  <section class="panel week">
    <div class="head">
      <div>
        <h2>Next week</h2>
        <p class="muted" id="week-note">Written each Monday at midnight UK time.</p>
      </div>
      <div class="links">
        <a href="/download/week?which=current">Forecast CSV</a>
        <a href="/download/week/frozen?which=current">Frozen inputs</a>
        <a href="/download/week?which=previous">Previous forecast</a>
        <a href="/download/week/frozen?which=previous">Previous inputs</a>
      </div>
    </div>
    <p class="explain">This chart is the last two days of last week (Saturday and Sunday), then this week from Monday, in time order. That keeps this week’s half-hours readable. If last week’s Monday freeze is still on disk, the pale lines are that forecast against the black official numbers. From this Monday the bright lines are this week’s forecast. Scores in the boxes only use hours of this week that already have a black line.</p>
    <div class="cards" id="week-cards"></div>
    <button type="button" data-preset="week">Re-issue week (password)</button>
    <div class="chart-wrap week"><canvas id="week"></canvas></div>
  </section>

  <div class="side">
  <section class="panel eval" id="eval">
    <div class="head">
      <div>
        <h2>How the guesses did</h2>
        <p class="muted" id="eval-note">Finished half-hours only.</p>
      </div>
    </div>
    <p class="explain">This column is a check, not a new forecast. It looks at half-hours that already have a black line and tags the awkward ones. <b>Outside band</b> — actual missed the blue range (a spike the guess did not cover). <b>Lags</b> — the demand reading the half-hour model used did not match the official figure at issue time (a labelling / freshness problem). <b>Large miss</b> — red line was more than about 1.5 GW out, but still inside the blues. Weather-was-wrong is not scored here yet; that needs a few days of issued temperature vs later actual weather.</p>
    <div class="cards" id="eval-cards"></div>
    <p class="muted" id="eval-weather"></p>
    <table class="eval-miss" id="eval-miss"></table>
  </section>

  <section class="panel lab" id="lab">
    <div class="head">
      <div>
        <h2>How the models were trained</h2>
        <p class="muted" id="lab-note">Holdout scores from the last train, not today’s board.</p>
      </div>
    </div>
    <p class="explain">This column is the research trail. <b>Holdout MAE</b> is the last eight weeks used when the pack was fit — a different sample from the live n on the charts. <b>Tuned</b> means Optuna settings were written into that pack. The week pack has a search winner; the 24-hour pack still uses stock LightGBM until you copy a short-study winner in. Weights &amp; Biases and MLflow links appear when those URLs are set on the server.</p>
    <p class="links" id="lab-links"></p>
    <table class="lab-packs" id="lab-packs"></table>
    <table class="lab-packs" id="lab-optuna"></table>
  </section>
  </div>

  <section class="panel day">
      <div class="head">
        <div>
          <h2>Next 24 hours</h2>
          <p class="muted" id="day-note">Written at midnight UK time.</p>
        </div>
        <div class="links">
          <a href="/download/day?which=current">Forecast CSV</a>
          <a href="/download/day/frozen?which=current">Frozen inputs</a>
          <a href="/download/day?which=previous">Yesterday forecast CSV</a>
        </div>
      </div>
      <p class="explain">The left side is yesterday’s London calendar day, from the first stored half-hour until midnight. A late freeze (for example issued at 9am) still stops at midnight; it does not run into today. From midnight the bright lines are today’s forecast. Scores in the boxes are for today.</p>
      <div class="cards" id="day-cards"></div>
      <button type="button" data-preset="day">Re-issue day</button>
      <div class="chart-wrap half"><canvas id="day"></canvas></div>
  </section>
  <section class="panel n30">
      <div class="head">
        <div>
          <h2>Next 30 minutes</h2>
          <p class="muted" id="n30-note">A new guess every half-hour.</p>
        </div>
        <div class="links">
          <a href="/download/next30?which=current">Forecast CSV</a>
          <a href="/download/next30/frozen?which=current">Frozen inputs</a>
          <a href="/inputs/next30">Input check JSON</a>
        </div>
      </div>
      <p class="explain">This chart is the last six hours, one guess per half-hour. The guess labelled 1:30 is for 1:00–1:30, and the black official number for that half-hour is drawn at 1:30 too (Insights stores it at 1:00). Pale red is what was written at the time. Dots are the live guess still waiting.</p>
      <div class="cards" id="n30-cards"></div>
      <button type="button" data-preset="next30">Re-issue next 30</button>
      <div class="chart-wrap half"><canvas id="next30"></canvas></div>
      <dl class="glossary" aria-label="Next 30 minute extra cards">
        <dt>Target</dt>
        <dd>The end of the half-hour being guessed. 1:30 means 1:00–1:30, matching the black line.</dd>
        <dt>Live P50</dt>
        <dd>The current red-line guess, in megawatts, for that target half-hour.</dd>
      </dl>
      <p class="explain">Logs for the half-hour model sit under the chart so it lines up with today. The status line is a health check on the latest guess: last official outturn (and how old it is), the two most recent demand readings the model was given, and UK temperature. Green means those look complete; red means the guess was skipped or the inputs look wrong. The table is a short history. <b>lag_1</b> is the latest demand the model used; <b>INDO then</b> is the official figure available at issue time; <b>stale lags</b> means those two disagreed, so the guess may be off.</p>
      <p class="muted" id="n30-inputs"></p>
      <table class="audit" id="n30-audit"></table>
  </section>
  </div>

  <script src="/chart.js"></script>
  <script>
    const fmt = (n, d=0) => n == null || Number.isNaN(n) ? "—" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d});
    const charts = {};
    Chart.defaults.font.family = '"Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';

    function londonParts(ts) {
      const date = new Date(ts);
      if (Number.isNaN(date.getTime())) return null;
      const out = {};
      new Intl.DateTimeFormat("en-GB", {
        timeZone: "Europe/London", weekday: "short", day: "numeric", month: "short",
        hour: "2-digit", minute: "2-digit", hour12: false, hourCycle: "h23"
      }).formatToParts(date).forEach((part) => { out[part.type] = part.value; });
      return out;
    }

    function london(ts, withDay=true) {
      const p = londonParts(ts);
      if (!p) return "";
      return withDay ? `${p.day} ${p.month}, ${p.hour}:${p.minute}` : `${p.hour}:${p.minute}`;
    }

    function stampMs(ts) {
      const t = Date.parse(ts);
      return Number.isNaN(t) ? null : t;
    }

    function indexRows(rows) {
      const map = new Map();
      for (const row of rows || []) {
        const ms = stampMs(row.timestamp);
        if (ms == null) continue;
        map.set(ms, Object.assign({}, map.get(ms) || {}, row));
      }
      return map;
    }

    function halfHourGrid(maps) {
      let min = Infinity;
      let max = -Infinity;
      for (const map of maps) {
        for (const ms of map.keys()) {
          if (ms < min) min = ms;
          if (ms > max) max = ms;
        }
      }
      if (!Number.isFinite(min)) return [];
      const step = 30 * 60 * 1000;
      min = Math.floor(min / step) * step;
      max = Math.ceil(max / step) * step;
      const out = [];
      for (let t = min; t <= max; t += step) out.push(t);
      return out;
    }

    function axisTick(ts, mode) {
      const p = londonParts(ts);
      if (!p) return "";
      if (mode === "week") {
        return (p.hour === "00" && p.minute === "00") ? `${p.weekday} ${p.day}` : "";
      }
      if (mode === "day") {
        if (p.hour === "00" && p.minute === "00") return `${p.day} ${p.month}`;
        if (p.minute === "00" && ["06", "12", "18"].includes(p.hour)) return `${p.hour}:00`;
        return "";
      }
      if (mode === "next30") {
        if (p.hour === "00" && p.minute === "00") return `${p.day} ${p.month}`;
        if (p.minute === "00" && ["03", "06", "09", "12", "15", "18", "21"].includes(p.hour)) return `${p.hour}:00`;
        return "";
      }
      if (p.minute === "00") {
        return p.hour === "00" ? `${p.day} ${p.month}` : `${p.hour}:00`;
      }
      return "";
    }

    async function readJson(res) {
      const text = await res.text();
      try { return JSON.parse(text); }
      catch { throw new Error(text.slice(0, 180) || res.statusText); }
    }

    function cards(el, metrics, extra) {
      const m = metrics || {};
      el.innerHTML = `
        ${extra || ""}
        <div class="card" title="How many half-hours already have an official number, so they can be scored."><span class="muted">n</span><b>${fmt(m.n)}</b></div>
        <div class="card" title="Typical size of the miss, in megawatts. Lower is better."><span class="muted">MAE</span><b>${m.mae != null ? fmt(m.mae) + " MW" : "waiting"}</b></div>
        <div class="card" title="Compared with copying last week. 1.0 = copy last week. Below 1 is better."><span class="muted">WRMSSE</span><b>${m.wrmsse != null ? Number(m.wrmsse).toFixed(2) : "—"}</b></div>
        <div class="card" title="How often the black line landed between the two blue lines. About 80% is the aim."><span class="muted">Coverage</span><b>${m.coverage_80 != null ? (100*m.coverage_80).toFixed(0) + "%" : "waiting"}</b></div>
        <div class="card" title="On average, is the red line too high (positive) or too low (negative)."><span class="muted">Bias</span><b>${m.bias != null ? fmt(m.bias) + " MW" : "—"}</b></div>
        <div class="card" title="Average gap between the blue lines over the whole issued forecast, including hours still to come."><span class="muted">Width</span><b>${m.interval_width != null ? fmt(m.interval_width) + " MW" : "—"}</b></div>`;
    }

    function lineChart(id, labels, datasets, title, mode) {
      if (charts[id]) charts[id].destroy();
      charts[id] = new Chart(document.getElementById(id), {
        type: "line",
        data: { labels, datasets },
        options: {
          maintainAspectRatio: false,
          layout: { padding: { bottom: 6 } },
          plugins: {
            title: { display: true, text: title, font: { size: 14 } },
            legend: {
              position: "bottom",
              labels: { filter: () => true }
            }
          },
          scales: {
            x: {
              ticks: {
                autoSkip: false,
                maxRotation: 0,
                minRotation: 0,
                font: { size: 11 },
                color: "#444",
                callback: function(value) {
                  return axisTick(this.getLabelForValue(value), mode);
                }
              },
              grid: { color: (ctx) => {
                const label = labels[ctx.index];
                const p = label ? londonParts(label) : null;
                return (p && p.hour === "00" && p.minute === "00") ? "#d0c8b8" : "rgba(0,0,0,0.04)";
              } }
            },
            y: {
              title: { display: true, text: "MW", font: { size: 11 } },
              ticks: { font: { size: 11 }, color: "#444", callback: (v) => Number(v).toLocaleString() }
            }
          }
        }
      });
    }

    function fan(rows, prefix, colors) {
      return [
        { label: prefix + "P10", data: rows.map(r => r.p10), borderColor: colors.band, pointRadius: 0, borderWidth: 1 },
        { label: prefix + "P50", data: rows.map(r => r.p50), borderColor: colors.p50, pointRadius: 0, borderWidth: 2 },
        { label: prefix + "P90", data: rows.map(r => r.p90), borderColor: colors.band, pointRadius: 0, borderWidth: 1 },
        { label: prefix + "INDO", data: rows.map(r => r.actual_mw), borderColor: colors.indo, pointRadius: 0, borderWidth: 2, spanGaps: true },
      ];
    }

    function rowsBefore(rows, endMs) {
      if (endMs == null) return rows || [];
      return (rows || []).filter((row) => {
        const ms = stampMs(row.timestamp);
        return ms != null && ms < endMs;
      });
    }

    function rowsFrom(rows, startMs) {
      if (startMs == null) return rows || [];
      return (rows || []).filter((row) => {
        const ms = stampMs(row.timestamp);
        return ms != null && ms >= startMs;
      });
    }

    function drawWindow(id, previous, fallbackActuals, current, title, mode, pastLabel, seamMs) {
      const issued = rowsBefore((previous && previous.rows) || [], seamMs);
      const hasPrevFan = issued.some((row) => row.p50 != null || row.p10 != null);
      const prevMap = indexRows(rowsBefore(fallbackActuals || [], seamMs));
      for (const row of issued) {
        const ms = stampMs(row.timestamp);
        if (ms == null) continue;
        prevMap.set(ms, Object.assign({}, prevMap.get(ms) || {}, row));
      }
      const curMap = indexRows(rowsFrom((current && current.rows) || [], seamMs));
      const grid = halfHourGrid([prevMap, curMap]);
      const labels = grid.map((ms) => new Date(ms).toISOString());
      const prevAt = (ms) => prevMap.get(ms) || {};
      const curAt = (ms) => curMap.get(ms) || {};
      const left = (key) => grid.map((ms) => {
        const v = prevAt(ms)[key];
        return v == null ? null : v;
      });
      const right = (key) => grid.map((ms) => {
        const v = curAt(ms)[key];
        return v == null ? null : v;
      });
      const indo = grid.map((ms) => {
        const cur = curAt(ms).actual_mw;
        if (cur != null) return cur;
        const prev = prevAt(ms).actual_mw;
        return prev == null ? null : prev;
      });
      const past = pastLabel || "Previous";
      const datasets = [
        { label: "Actual", data: indo, borderColor: "#111", pointRadius: 0, borderWidth: 2, spanGaps: false },
      ];
      if (hasPrevFan) {
        datasets.push(
          { label: past + " low", data: left("p10"), borderColor: "#d5e0ea", pointRadius: 0, borderWidth: 1.5, spanGaps: false },
          { label: past + " most likely", data: left("p50"), borderColor: "#e0b49a", pointRadius: 0, borderWidth: 2, spanGaps: false },
          { label: past + " high", data: left("p90"), borderColor: "#d5e0ea", pointRadius: 0, borderWidth: 1.5, spanGaps: false },
        );
      }
      datasets.push(
        { label: "Low", data: right("p10"), borderColor: "#9bb8d3", pointRadius: 0, borderWidth: 1.5, spanGaps: false },
        { label: "Most likely", data: right("p50"), borderColor: "#c45c26", pointRadius: 0, borderWidth: 2, spanGaps: false },
        { label: "High", data: right("p90"), borderColor: "#9bb8d3", pointRadius: 0, borderWidth: 1.5, spanGaps: false },
      );
      lineChart(id, labels, datasets, title, mode);
    }

    async function load() {
      const board = await readJson(await fetch("/board"));
      const week = board.week || {};
      const day = board.day || {};
      const n30 = board.next30 || {};
      const live = (n30.current && n30.current[0]) || {};
      document.getElementById("week-note").textContent =
        (week.current && week.current.issued_at)
          ? ("Written " + london(week.current.issued_at)
            + ((week.previous && week.previous.issued_at)
              ? " · left = Saturday–Sunday vs actual"
              : " · left = Saturday–Sunday actual only"))
          : "No week forecast yet — it is written Monday midnight UK, or click Issue due presets.";
      const yestFan = day.previous && day.previous.issued_at;
      const yestActuals = day.previous_actuals && day.previous_actuals.length;
      document.getElementById("day-note").textContent =
        (day.current && day.current.issued_at)
          ? ("Written " + london(day.current.issued_at)
            + (yestFan
              ? " · left = yesterday until midnight"
              : (yestActuals
                ? " · left = yesterday actual until midnight"
                : " · no yesterday freeze in range")))
          : "No day forecast yet — it is written at midnight UK, or click Issue due presets.";
      const dayTitle = (yestFan || yestActuals)
        ? "Yesterday until midnight, then today’s forecast"
        : "Today’s forecast";
      document.getElementById("n30-note").textContent = live.timestamp
        ? ("Guessing " + london(live.timestamp) + " · written " + london(live.issued_at)
          + " · most likely " + fmt(live.p50) + " MW")
        : "Waiting for the first half-hour guess.";
      cards(document.getElementById("week-cards"), week.current && week.current.metrics);
      cards(document.getElementById("day-cards"), day.current && day.current.metrics,
        (day.previous && day.previous.metrics && day.previous.issued_at)
          ? `<div class="card" title="Yesterday’s freeze: typical miss once the day had finished."><span class="muted">Yesterday MAE</span><b>${day.previous.metrics.mae != null ? fmt(day.previous.metrics.mae) + " MW" : "—"}</b></div>`
          : "");
      cards(document.getElementById("n30-cards"), n30.metrics, live.timestamp
        ? `<div class="card" title="Which half-hour the guess is for."><span class="muted">Target</span><b>${london(live.timestamp)}</b></div>
           <div class="card" title="The current red-line guess for that half-hour, in megawatts."><span class="muted">Live P50</span><b>${fmt(live.p50)} MW</b></div>`
        : "");
      const inp = n30.inputs || {};
      const issues = inp.issues || [];
      const inputEl = document.getElementById("n30-inputs");
      const age = inp.indo_age_minutes != null ? Math.round(inp.indo_age_minutes) + " min old" : "—";
      inputEl.className = inp.ok ? "ok" : "warn";
      inputEl.textContent = inp.ok
        ? ("Inputs ok · last INDO " + (inp.last_indo_mw != null ? fmt(inp.last_indo_mw) + " MW" : "—")
          + " at " + london(inp.last_indo_timestamp)
          + " (" + age + ") · lag_1 " + fmt(inp.demand_lag_1) + " · lag_2 " + fmt(inp.demand_lag_2)
          + " · T " + (inp.temperature_2m != null ? Number(inp.temperature_2m).toFixed(1) + "°C" : "—"))
        : (issues.length ? issues.join(" · ") : "No frozen 30-minute inputs yet — click Issue due presets.");
      const audit = (n30.audit && n30.audit.rows) || [];
      const auditEl = document.getElementById("n30-audit");
      if (!audit.length) {
        auditEl.innerHTML = "";
      } else {
        auditEl.innerHTML = "<thead><tr><th title='When the guess was written'>Written</th><th title='Latest demand the model used'>lag_1</th><th title='Official figure available at that time'>INDO then</th><th title='Red-line guess'>Guess</th><th title='Official figure once the half-hour ended'>Actual</th><th title='Whether the model’s demand reading matched the official figure'>Check</th></tr></thead>"
          + "<tbody>" + audit.slice().reverse().map((r) => {
            const mark = r.stale_lags ? "stale lags" : "ok";
            return `<tr class="${r.stale_lags ? "warn" : ""}"><td>${london(r.issued_at)}</td>
              <td>${fmt(r.demand_lag_1)}</td><td>${fmt(r.available_indo_mw)}</td>
              <td>${fmt(r.p50)}</td><td>${fmt(r.actual_mw)}</td><td>${mark}</td></tr>`;
          }).join("") + "</tbody>";
      }
      drawWindow("week", week.previous, week.previous_actuals, week.current, "Saturday–Sunday, then this week’s forecast", "week", "Sat–Sun", stampMs(week.current && week.current.window_start));
      drawWindow("day", day.previous, day.previous_actuals, day.current, dayTitle, "day", "Yesterday", stampMs(day.boundary || (day.current && day.current.window_start)));
      drawNext30(n30.history || [], live);
      drawEval(board.eval || {});
      drawLab(board.lab || {});
    }

    function drawEval(ev) {
      const scored = ev.scored || 0;
      document.getElementById("eval-note").textContent = scored
        ? (scored + " finished half-hours scored")
        : "Waiting for official outturn on issued guesses.";
      const cards = document.getElementById("eval-cards");
      const stale = ev.stale_share != null ? (100 * ev.stale_share).toFixed(0) + "%" : "—";
      cards.innerHTML = `
        <div class="card" title="Half-hours with both a guess and an official number."><span class="muted">Scored</span><b>${fmt(scored)}</b></div>
        <div class="card" title="Actual missed the blue range."><span class="muted">Outside band</span><b>${fmt(ev.outside_band)}</b></div>
        <div class="card" title="Half-hour model used a demand reading that did not match the official figure."><span class="muted">Lags</span><b>${fmt(ev.lags)}</b></div>
        <div class="card" title="Red line more than about 1.5 GW out, but still inside the blues."><span class="muted">Large miss</span><b>${fmt(ev.large_miss)}</b></div>
        <div class="card" title="Share of recent half-hour issues where lag_1 disagreed with INDO."><span class="muted">Stale share</span><b>${stale}</b></div>`;
      const weather = document.getElementById("eval-weather");
      weather.textContent = (ev.weather && ev.weather.note) || "";
      const worst = ev.worst || [];
      const table = document.getElementById("eval-miss");
      if (!worst.length) {
        table.innerHTML = "";
        return;
      }
      table.innerHTML = "<thead><tr><th>When</th><th>View</th><th>Guess</th><th>Actual</th><th>Miss</th><th>Tag</th></tr></thead><tbody>"
        + worst.map((row) => `<tr>
            <td>${london(row.timestamp)}</td>
            <td>${row.preset || ""}</td>
            <td>${fmt(row.p50)}</td>
            <td>${fmt(row.actual_mw)}</td>
            <td>${fmt(row.error_mw)}</td>
            <td>${row.tag || ""}</td>
          </tr>`).join("")
        + "</tbody>";
    }

    function drawLab(lab) {
      const packs = lab.packs || [];
      const optuna = lab.optuna || [];
      const tracking = lab.tracking || {};
      const tuned = packs.filter((row) => row.tuned).length;
      document.getElementById("lab-note").textContent = packs.length
        ? (tuned + " of " + packs.length + " packs have an Optuna winner in production")
        : "No saved packs on this box.";
      const links = [];
      if (tracking.notebook) links.push(`<a href="${tracking.notebook}">Optuna notebook</a>`);
      if (tracking.wandb) links.push(`<a href="${tracking.wandb}">Weights &amp; Biases</a>`);
      else links.push('<span class="muted" title="Set WANDB_PROJECT_URL on the server to show a live W&amp;B project.">W&amp;B — set WANDB_PROJECT_URL</span>');
      if (tracking.mlflow) links.push(`<a href="${tracking.mlflow}">MLflow</a>`);
      else links.push('<span class="muted" title="Set MLFLOW_UI_URL on the server to show the tracking UI.">MLflow — set MLFLOW_UI_URL</span>');
      document.getElementById("lab-links").innerHTML = links.join(" · ");
      const packTable = document.getElementById("lab-packs");
      packTable.innerHTML = "<thead><tr><th>Pack</th><th>Search</th><th>Holdout MAE</th><th>Coverage</th><th>Trees</th><th>Tuned</th></tr></thead><tbody>"
        + packs.map((row) => `<tr>
            <td title="${row.note || ""}">${row.label || row.id}</td>
            <td>${row.search || "—"}</td>
            <td>${row.holdout_mae != null ? fmt(row.holdout_mae) + " MW" : "—"}</td>
            <td>${row.holdout_coverage != null ? (100 * row.holdout_coverage).toFixed(0) + "%" : "—"}</td>
            <td>${fmt(row.n_estimators)}</td>
            <td>${row.tuned ? "Optuna" : "defaults"}</td>
          </tr>`).join("")
        + "</tbody>";
      const studyTable = document.getElementById("lab-optuna");
      studyTable.innerHTML = "<thead><tr><th>Study</th><th>Trials</th><th>Best MAE</th></tr></thead><tbody>"
        + optuna.map((row) => `<tr>
            <td>${row.product || ""}</td>
            <td>${row.ready ? fmt(row.n_complete) + " complete" : "no local db"}</td>
            <td>${row.best_mae != null ? fmt(row.best_mae) + " MW" : "—"}</td>
          </tr>`).join("")
        + "</tbody>";
    }

    function drawNext30(hist, live) {
      const liveTs = live && live.timestamp;
      const rows = hist.slice();
      if (liveTs && !rows.some((row) => row.timestamp === liveTs)) rows.push(live);
      rows.sort((a, b) => (stampMs(a.timestamp) || 0) - (stampMs(b.timestamp) || 0));
      const labels = rows.map((row) => row.timestamp);
      const liveDot = (key) => rows.map((row) => row.timestamp === liveTs ? row[key] : null);
      lineChart("next30", labels, [
        { label: "Actual", data: rows.map((row) => row.actual_mw), borderColor: "#111", pointRadius: 0, borderWidth: 2, spanGaps: true },
        { label: "Guess then", data: rows.map((row) => row.p50), borderColor: "#e0b49a", pointRadius: 0, borderWidth: 1.5, spanGaps: true },
        { label: "Low then", data: rows.map((row) => row.p10), borderColor: "#d5e0ea", pointRadius: 0, borderWidth: 1, spanGaps: true },
        { label: "High then", data: rows.map((row) => row.p90), borderColor: "#d5e0ea", pointRadius: 0, borderWidth: 1, spanGaps: true },
        { label: "Live low", data: liveDot("p10"), borderColor: "#9bb8d3", pointRadius: 4, borderWidth: 1.5, showLine: false },
        { label: "Live guess", data: liveDot("p50"), borderColor: "#c45c26", pointRadius: 6, borderWidth: 2, showLine: false },
        { label: "Live high", data: liveDot("p90"), borderColor: "#9bb8d3", pointRadius: 4, borderWidth: 1.5, showLine: false },
      ], "Pale lines = guesses already written. Dots = the half-hour still waiting for the official number.", "next30");
    }

    async function runCapture(force) {
      const note = document.getElementById("note");
      note.textContent = force ? "Re-issuing…" : "Running due packs…";
      try {
        const res = await fetch("/capture?force=" + (force ? "true" : "false"), { method: "POST" });
        const body = await readJson(res);
        if (!res.ok) {
          note.textContent = body.detail || "failed";
        } else if (body.next30 && body.next30.skipped && body.next30.reason) {
          note.textContent = "Skipped 30-minute call: " + body.next30.reason;
        } else {
          note.textContent = "Updated";
        }
        await load();
      } catch (err) { note.textContent = String(err); }
    }
    async function reissuePreset(preset) {
      const note = document.getElementById("note");
      const names = { week: "week", day: "day", next30: "next 30 minutes" };
      const headers = {};
      if (preset === "week") {
        const pw = window.prompt("Password to re-issue this week's freeze");
        if (pw == null || pw === "") {
          note.textContent = "Week re-issue cancelled";
          return;
        }
        headers["X-Week-Password"] = pw;
      }
      note.textContent = "Re-issuing " + (names[preset] || preset) + "…";
      try {
        const res = await fetch("/capture/" + preset, { method: "POST", headers });
        const body = await readJson(res);
        if (!res.ok) {
          note.textContent = body.detail || "failed";
        } else if (body.skipped && body.reason) {
          note.textContent = "Skipped: " + body.reason;
        } else {
          note.textContent = (names[preset] || preset) + " re-issued";
        }
        await load();
      } catch (err) { note.textContent = String(err); }
    }
    document.getElementById("capture").onclick = () => runCapture(false);
    document.getElementById("reload").onclick = () => reloadBoard();
    document.querySelectorAll("button[data-preset]").forEach((btn) => {
      btn.onclick = () => reissuePreset(btn.dataset.preset);
    });

    async function reloadBoard() {
      const note = document.getElementById("note");
      note.textContent = "Refreshing…";
      try {
        await load();
        note.textContent = "";
      } catch (err) {
        note.textContent = String(err);
      }
    }

    load().then(() => { document.getElementById("note").textContent = ""; }).catch((err) => {
      document.getElementById("note").textContent = String(err);
    });
  </script>
</body>
</html>
"""
