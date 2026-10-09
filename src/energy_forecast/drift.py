"""Live vs holdout drift check. Runs on the capture loop; emails if configured."""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from energy_forecast.lab import model_lab_payload
from energy_forecast.paths import ensure_data_dirs

PRESET_PACK = {"next30": "hour", "day": "day", "week": "week"}
PACK_PRESET = {value: key for key, value in PRESET_PACK.items()}
LABELS = {
    "next30": "Next 30 minutes",
    "day": "Next 24 hours",
    "week": "Next week",
}
MIN_N = {"next30": 12, "day": 8, "week": 24}
MAE_RATIO = 1.5
COVERAGE_FLOOR = 0.55
COOLDOWN_HOURS = 12


def drift_path() -> Path:
    from energy_forecast.paths import DATA_LIVE

    ensure_data_dirs()
    return DATA_LIVE / "drift.json"


def load_drift_state() -> dict:
    path = drift_path()
    if not path.exists():
        return {"checked_at": None, "alerts": [], "last_sent": {}, "packs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"checked_at": None, "alerts": [], "last_sent": {}, "packs": []}


def save_drift_state(state: dict) -> None:
    path = drift_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def email_configured() -> bool:
    return bool(os.environ.get("DRIFT_EMAIL_TO", "").strip() and os.environ.get("DRIFT_SMTP_HOST", "").strip())


def check_and_alert(*, send=None, now: datetime | None = None) -> dict:
    """Compare this-freeze MAE/coverage with the last-train holdout. Local files only."""
    from energy_forecast.live import board_payload

    when = now or datetime.now(timezone.utc)
    board = board_payload(refresh=False)
    lab = {row["id"]: row for row in (board.get("lab") or {}).get("packs") or model_lab_payload()["packs"]}
    live = {
        "hour": ((board.get("next30") or {}).get("metrics") or {}),
        "day": (((board.get("day") or {}).get("current") or {}).get("metrics") or {}),
        "week": (((board.get("week") or {}).get("current") or {}).get("metrics") or {}),
    }
    packs = []
    firing: list[dict] = []
    for pack_id, preset in PACK_PRESET.items():
        hold = lab.get(pack_id) or {}
        metrics = live.get(pack_id) or {}
        card = _pack_status(pack_id, preset, hold, metrics)
        packs.append(card)
        firing.extend(card["alerts"])

    state = load_drift_state()
    last_sent = dict(state.get("last_sent") or {})
    send_fn = send if send is not None else _send_email
    mailed: list[str] = []
    skipped: list[str] = []
    for alert in firing:
        key = f"{alert['preset']}:{alert['kind']}"
        if _in_cooldown(last_sent.get(key), when):
            skipped.append(key)
            continue
        if not email_configured():
            skipped.append(key)
            continue
        try:
            send_fn(alert)
            last_sent[key] = when.isoformat()
            mailed.append(key)
        except Exception as exc:  # noqa: BLE001
            alert["send_error"] = str(exc)
            skipped.append(key)

    out = {
        "checked_at": when.isoformat(),
        "email_ready": email_configured(),
        "mae_ratio": MAE_RATIO,
        "coverage_floor": COVERAGE_FLOOR,
        "cooldown_hours": COOLDOWN_HOURS,
        "packs": packs,
        "alerts": firing,
        "mailed": mailed,
        "skipped": skipped,
        "last_sent": last_sent,
        "note": "Live MAE vs last-train holdout, on the capture loop. Alerts wait 12 hours before repeating.",
    }
    save_drift_state(out)
    return out


def drift_payload() -> dict:
    """Last check from disk. No weather APIs."""
    state = load_drift_state()
    if not state.get("packs"):
        state["note"] = "No drift check yet. The capture loop writes this after the next half-hour tick."
        state["email_ready"] = email_configured()
    return state


def _pack_status(pack_id: str, preset: str, hold: dict, metrics: dict) -> dict:
    n = int(metrics.get("n") or 0)
    live_mae = metrics.get("mae")
    live_cov = metrics.get("coverage_80")
    hold_mae = hold.get("holdout_mae")
    hold_cov = hold.get("holdout_coverage")
    min_n = MIN_N[preset]
    alerts: list[dict] = []
    ratio = None
    if live_mae is not None and hold_mae not in (None, 0) and n >= min_n:
        ratio = float(live_mae) / float(hold_mae)
        if ratio >= MAE_RATIO:
            alerts.append(
                _alert(
                    preset,
                    "mae",
                    f"{LABELS[preset]} live MAE is {live_mae:.0f} MW vs holdout {hold_mae:.0f} MW "
                    f"(ratio {ratio:.2f}, n={n}).",
                    live_mae=live_mae,
                    holdout_mae=hold_mae,
                    n=n,
                    ratio=ratio,
                )
            )
    if live_cov is not None and n >= min_n and float(live_cov) < COVERAGE_FLOOR:
        alerts.append(
            _alert(
                preset,
                "coverage",
                f"{LABELS[preset]} live coverage is {100 * float(live_cov):.0f}% "
                f"(aim ~80%, floor {100 * COVERAGE_FLOOR:.0f}%, n={n}).",
                live_coverage=live_cov,
                holdout_coverage=hold_cov,
                n=n,
            )
        )
    return {
        "id": pack_id,
        "preset": preset,
        "label": LABELS[preset],
        "n": n,
        "min_n": min_n,
        "live_mae": live_mae,
        "holdout_mae": hold_mae,
        "mae_ratio": ratio,
        "live_coverage": live_cov,
        "holdout_coverage": hold_cov,
        "alerts": alerts,
    }


def _alert(preset: str, kind: str, message: str, **extra) -> dict:
    return {"preset": preset, "kind": kind, "message": message, **extra}


def _in_cooldown(stamp: str | None, when: datetime) -> bool:
    if not stamp:
        return False
    try:
        last = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (when - last).total_seconds() < COOLDOWN_HOURS * 3600


def _send_email(alert: dict) -> None:
    to_addr = os.environ["DRIFT_EMAIL_TO"].strip()
    host = os.environ["DRIFT_SMTP_HOST"].strip()
    port = int(os.environ.get("DRIFT_SMTP_PORT", "587"))
    user = os.environ.get("DRIFT_SMTP_USER", "").strip()
    password = os.environ.get("DRIFT_SMTP_PASSWORD", "")
    from_addr = os.environ.get("DRIFT_EMAIL_FROM", "").strip() or user or to_addr
    msg = EmailMessage()
    msg["Subject"] = f"Grid Demand UK drift — {LABELS.get(alert['preset'], alert['preset'])}"
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(
        "A live score moved far from the last train holdout.\n\n"
        + alert["message"]
        + "\n\nThis is the capture-loop check, not a page refresh. "
        "It will not mail the same pack/kind again for 12 hours.\n"
    )
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as smtp:
            if user:
                smtp.login(user, password)
            smtp.send_message(msg)
        return
    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls(context=context)
        if user:
            smtp.login(user, password)
        smtp.send_message(msg)
