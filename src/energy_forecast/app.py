"""Live forecast service: capture, join INDO, show the score."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, Response

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
    await asyncio.sleep(5)
    while True:
        try:
            tick_presets()
        except Exception as exc:  # noqa: BLE001 — loop must not die
            print(f"live capture failed: {exc}")
        await asyncio.sleep(seconds_until_half_hour())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("EF_SKIP_CAPTURE") == "1":
        yield
        return
    task = asyncio.create_task(_capture_loop())
    yield
    task.cancel()


app = FastAPI(title="GB demand forecast", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


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
    from energy_forecast.live import _windows, input_health, load_frozen, next30_input_audit, _preset_frame

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
  <title>GB demand live score</title>
  <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    :root { font-family: Georgia, serif; color: #1a1a1a; background: #f7f4ee; }
    body { max-width: 1400px; margin: 24px auto; padding: 0 16px 40px; }
    h1 { font-size: 1.55rem; font-weight: 600; margin-bottom: 0.3rem; }
    h2 { font-size: 1.15rem; margin: 0; }
    .muted { color: #5c5c5c; font-size: 0.92rem; }
    .panel { background: #fff; border: 1px solid #ddd4c4; padding: 14px 16px 10px; margin: 14px 0; }
    .head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; align-items: baseline; }
    .cards { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 6px; }
    .card { border: 1px solid #eee; padding: 8px 10px; min-width: 92px; }
    .card b { display: block; font-size: 1.15rem; }
    .links a { margin-right: 12px; color: #1a1a1a; }
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
  <h1>GB National Demand — three presets</h1>
  <p class="muted">Week freeze at Monday 00:00 London. Day freeze at midnight. Next-30-minute call every half-hour. Times are Europe/London. INDO joins after each settlement period. Re-issue lives on each chart: week/day replace that freeze; next-30 only moves the live dots, not the pale history.</p>
  <p>
    <button type="button" id="capture">Issue due presets</button>
    <span class="muted" id="note"></span>
  </p>

  <section class="panel">
    <div class="head">
      <div>
        <h2>Next week</h2>
        <p class="muted" id="week-note">Sunday–Monday freeze, week model. Previous week in grey.</p>
      </div>
      <div class="links">
        <a href="/download/week?which=current">Forecast CSV</a>
        <a href="/download/week/frozen?which=current">Frozen inputs</a>
        <a href="/download/week?which=previous">Previous forecast</a>
        <a href="/download/week/frozen?which=previous">Previous inputs</a>
      </div>
    </div>
    <div class="cards" id="week-cards"></div>
    <button type="button" data-preset="week">Re-issue week (password)</button>
    <div class="chart-wrap week"><canvas id="week"></canvas></div>
  </section>

  <div class="split">
    <section class="panel">
      <div class="head">
        <div>
          <h2>Next 24 hours</h2>
          <p class="muted" id="day-note">Midnight freeze, day model. Yesterday in grey.</p>
        </div>
        <div class="links">
          <a href="/download/day?which=current">Forecast CSV</a>
          <a href="/download/day/frozen?which=current">Frozen inputs</a>
          <a href="/download/day?which=previous">Yesterday</a>
        </div>
      </div>
      <div class="cards" id="day-cards"></div>
      <button type="button" data-preset="day">Re-issue day</button>
      <div class="chart-wrap half"><canvas id="day"></canvas></div>
    </section>
    <section class="panel">
      <div class="head">
        <div>
          <h2>Next 30 minutes</h2>
          <p class="muted" id="n30-note">Hour model. Issued time on the live call.</p>
        </div>
        <div class="links">
          <a href="/download/next30?which=current">Forecast CSV</a>
          <a href="/download/next30/frozen?which=current">Frozen inputs</a>
          <a href="/inputs/next30">Input check JSON</a>
        </div>
      </div>
      <div class="cards" id="n30-cards"></div>
      <p class="muted" id="n30-inputs"></p>
      <table class="audit" id="n30-audit"></table>
      <button type="button" data-preset="next30">Re-issue next 30</button>
      <div class="chart-wrap half"><canvas id="next30"></canvas></div>
    </section>
  </div>

  <script>
    const fmt = (n, d=0) => n == null || Number.isNaN(n) ? "—" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d});
    const charts = {};

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
        if (p.minute === "00") return p.hour === "00" ? `${p.day} ${p.month}` : `${p.hour}:00`;
        if (p.minute === "30") return `${p.hour}:30`;
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
        <div class="card"><span class="muted">n</span><b>${fmt(m.n)}</b></div>
        <div class="card"><span class="muted">MAE</span><b>${m.mae != null ? fmt(m.mae) + " MW" : "waiting"}</b></div>
        <div class="card"><span class="muted">WRMSSE</span><b>${m.wrmsse != null ? Number(m.wrmsse).toFixed(2) : "—"}</b></div>
        <div class="card"><span class="muted">Coverage</span><b>${m.coverage_80 != null ? (100*m.coverage_80).toFixed(0) + "%" : "waiting"}</b></div>
        <div class="card"><span class="muted">Bias</span><b>${m.bias != null ? fmt(m.bias) + " MW" : "—"}</b></div>
        <div class="card"><span class="muted">Width</span><b>${m.interval_width != null ? fmt(m.interval_width) + " MW" : "—"}</b></div>`;
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

    function drawWindow(id, previousActuals, current, title, mode) {
      const prev = previousActuals || [];
      const cur = (current && current.rows) || [];
      const labels = prev.map(r => r.timestamp).concat(cur.map(r => r.timestamp));
      const right = (value) => prev.map(() => null).concat(cur.map(value));
      const indo = prev.map(r => r.actual_mw).concat(cur.map(r => r.actual_mw));
      lineChart(id, labels, [
        { label: "INDO", data: indo, borderColor: "#111", pointRadius: 0, borderWidth: 2, spanGaps: false },
        { label: "P10", data: right(r => r.p10), borderColor: "#9bb8d3", pointRadius: 0, borderWidth: 1.5, spanGaps: false },
        { label: "P50", data: right(r => r.p50), borderColor: "#c45c26", pointRadius: 0, borderWidth: 2, spanGaps: false },
        { label: "P90", data: right(r => r.p90), borderColor: "#9bb8d3", pointRadius: 0, borderWidth: 1.5, spanGaps: false },
      ], title, mode);
    }

    async function load() {
      const board = await readJson(await fetch("/board"));
      const week = board.week || {};
      const day = board.day || {};
      const n30 = board.next30 || {};
      const live = (n30.current && n30.current[0]) || {};
      document.getElementById("week-note").textContent =
        (week.current && week.current.issued_at)
          ? ("Issued " + london(week.current.issued_at) + " · last week is INDO only, this week is the fan")
          : "No week freeze yet — first Monday 00:00 London, or click Issue due presets.";
      document.getElementById("day-note").textContent =
        (day.current && day.current.issued_at)
          ? ("Issued " + london(day.current.issued_at) + " · yesterday is INDO only, today is the fan")
          : "No day freeze yet — midnight London, or click Issue due presets.";
      document.getElementById("n30-note").textContent = live.timestamp
        ? ("Next period " + london(live.timestamp) + " · generated " + london(live.issued_at)
          + " · P50 " + fmt(live.p50) + " MW")
        : "Waiting for the first 30-minute issue.";
      cards(document.getElementById("week-cards"), week.current && week.current.metrics);
      cards(document.getElementById("day-cards"), day.current && day.current.metrics);
      cards(document.getElementById("n30-cards"), n30.metrics, live.timestamp
        ? `<div class="card"><span class="muted">Generated</span><b>${london(live.issued_at)}</b></div>
           <div class="card"><span class="muted">Target</span><b>${london(live.timestamp)}</b></div>
           <div class="card"><span class="muted">Live P50</span><b>${fmt(live.p50)} MW</b></div>`
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
        auditEl.innerHTML = "<thead><tr><th>Issued</th><th>lag_1</th><th>INDO then</th><th>P50</th><th>actual</th><th></th></tr></thead>"
          + "<tbody>" + audit.slice().reverse().map((r) => {
            const mark = r.stale_lags ? "stale lags" : "ok";
            return `<tr class="${r.stale_lags ? "warn" : ""}"><td>${london(r.issued_at)}</td>
              <td>${fmt(r.demand_lag_1)}</td><td>${fmt(r.available_indo_mw)}</td>
              <td>${fmt(r.p50)}</td><td>${fmt(r.actual_mw)}</td><td>${mark}</td></tr>`;
          }).join("") + "</tbody>";
      }
      drawWindow("week", week.previous_actuals, week.current, "Last week INDO, then this week's P10 / P50 / P90", "week");
      drawWindow("day", day.previous_actuals, day.current, "Yesterday INDO, then today's P10 / P50 / P90", "day");
      drawNext30(n30.history || [], live);
    }

    function drawNext30(hist, live) {
      const liveTs = live && live.timestamp;
      const rows = hist.slice();
      if (liveTs && !rows.some((row) => row.timestamp === liveTs)) rows.push(live);
      const labels = rows.map((row) => row.timestamp);
      const liveDot = (key) => rows.map((row) => row.timestamp === liveTs ? row[key] : null);
      lineChart("next30", labels, [
        { label: "INDO", data: rows.map((row) => row.actual_mw), borderColor: "#111", pointRadius: 0, borderWidth: 2, spanGaps: true },
        { label: "Issued P50", data: rows.map((row) => row.p50), borderColor: "#e0b49a", pointRadius: 0, borderWidth: 1.5, spanGaps: true },
        { label: "Issued P10", data: rows.map((row) => row.p10), borderColor: "#d5e0ea", pointRadius: 0, borderWidth: 1, spanGaps: true },
        { label: "Issued P90", data: rows.map((row) => row.p90), borderColor: "#d5e0ea", pointRadius: 0, borderWidth: 1, spanGaps: true },
        { label: "Live P10", data: liveDot("p10"), borderColor: "#9bb8d3", pointRadius: 4, borderWidth: 1.5, showLine: false },
        { label: "Live P50", data: liveDot("p50"), borderColor: "#c45c26", pointRadius: 6, borderWidth: 2, showLine: false },
        { label: "Live P90", data: liveDot("p90"), borderColor: "#9bb8d3", pointRadius: 4, borderWidth: 1.5, showLine: false },
      ], "Pale line = issued calls. Dots sit on the next period still waiting for INDO.", "next30");
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
    document.querySelectorAll("button[data-preset]").forEach((btn) => {
      btn.onclick = () => reissuePreset(btn.dataset.preset);
    });

    function londonClock() {
      const parts = new Intl.DateTimeFormat("en-GB", {
        timeZone: "Europe/London", hour: "2-digit", minute: "2-digit",
        second: "2-digit", hour12: false, hourCycle: "h23"
      }).formatToParts(new Date());
      const num = (type) => Number(parts.find((part) => part.type === type).value);
      return { minute: num("minute"), second: num("second") };
    }
    function msUntilIndoRefresh() {
      const { minute, second } = londonClock();
      const through = minute * 60 + second;
      const slot = through < 10 * 60 ? 10 * 60 : through < 40 * 60 ? 40 * 60 : 70 * 60;
      return (slot - through) * 1000;
    }
    async function waitForIndo() {
      await new Promise((resolve) => setTimeout(resolve, msUntilIndoRefresh()));
      await load();
      waitForIndo();
    }
    load();
    waitForIndo();
  </script>
</body>
</html>
"""
