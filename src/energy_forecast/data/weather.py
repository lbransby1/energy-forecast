"""Open-Meteo historical archive and forecast clients."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from energy_forecast.paths import DATA_PROCESSED, ensure_data_dirs
from energy_forecast.regions import SERIES_ID, Series, WeatherPoint, get_series
from energy_forecast.settlement import LONDON_TZ, add_settlement_keys

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HOURLY_VARS = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
)
LEAD_DAYS = (0, 1, 2, 3, 4, 5, 6, 7)
PREVIOUS_RUNS_START = "2024-01-01"
REQUEST_PAUSE_S = 0.25
PREVIOUS_RUNS_PAUSE_S = 2.0
TIMEOUT_S = 180
PREVIOUS_RUNS_RETRIES = 6
FORECAST_RETRIES = 5
RETRY_STATUSES = {429, 502, 503, 504}


class WeatherAPIError(RuntimeError):
    pass


def fetch_historical_weather(
    region_ids: list[str] | None = None,
    start_date: str = "",
    end_date: str = "",
    *,
    use_cache: bool = True,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch ERA5 archive weather and align it to settlement periods."""
    region_ids = region_ids or [SERIES_ID]
    frames = [
        _load_or_fetch_region(
            region_id,
            source="archive",
            use_cache=use_cache,
            processed_dir=processed_dir,
            start_date=start_date,
            end_date=end_date,
        )
        for region_id in region_ids
    ]
    return pd.concat(frames, ignore_index=True)


def fetch_forecast_weather(
    region_ids: list[str] | None = None,
    *,
    forecast_days: int = 7,
    past_days: int = 0,
    use_cache: bool = False,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch the operational weather forecast and align it to settlement periods."""
    region_ids = region_ids or [SERIES_ID]
    frames = [
        _load_or_fetch_region(
            region_id,
            source="forecast",
            use_cache=use_cache,
            processed_dir=processed_dir,
            forecast_days=forecast_days,
            past_days=past_days,
        )
        for region_id in region_ids
    ]
    return pd.concat(frames, ignore_index=True)


def previous_run_column(variable: str, lead_days: int) -> str:
    if lead_days <= 0:
        return variable
    return f"{variable}_previous_day{int(lead_days)}"


def previous_runs_hourly_vars() -> tuple[str, ...]:
    return tuple(previous_run_column(variable, lead) for variable in HOURLY_VARS for lead in LEAD_DAYS)


def fetch_previous_runs(
    region_ids: list[str] | None = None,
    start_date: str = PREVIOUS_RUNS_START,
    end_date: str = "",
    *,
    use_cache: bool = True,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch Open-Meteo Previous Runs (lead days 0-7) and align to settlement periods."""
    region_ids = region_ids or [SERIES_ID]
    if not end_date:
        end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    frames = [
        _load_or_fetch_previous_runs(
            region_id,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            processed_dir=processed_dir,
        )
        for region_id in region_ids
    ]
    return pd.concat(frames, ignore_index=True)


def expand_previous_runs(weather: pd.DataFrame) -> pd.DataFrame:
    """Turn wide previous-run columns into one row per timestamp and lead."""
    keys = ["region_id", "timestamp", "settlement_date", "settlement_period"]
    parts: list[pd.DataFrame] = []
    for lead in LEAD_DAYS:
        piece = weather.loc[:, [column for column in keys if column in weather.columns]].copy()
        for variable in HOURLY_VARS:
            source = previous_run_column(variable, lead)
            piece[variable] = (
                pd.to_numeric(weather[source], errors="coerce") if source in weather.columns else np.nan
            )
        piece["lead_hours"] = float(lead * 24)
        parts.append(piece)
    return pd.concat(parts, ignore_index=True)


def load_cached_weather(
    region_ids: list[str],
    *,
    source: str = "archive",
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    processed_dir = processed_dir or DATA_PROCESSED
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for region_id in region_ids:
        path = _cache_path(processed_dir, region_id, source)
        if not path.exists():
            missing.append(region_id)
            continue
        frames.append(pd.read_parquet(path))
    if missing:
        raise FileNotFoundError(
            "Weather cache missing for "
            + ", ".join(missing)
            + f" under {processed_dir}. Run fetch_historical_weather first."
        )
    weather = pd.concat(frames, ignore_index=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    weather["settlement_date"] = pd.to_datetime(weather["settlement_date"]).dt.tz_localize(None).dt.normalize()
    return weather


def combine_archive_and_forecast(archive: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """ERA5 through its last timestamp, then the operational forecast after that.

    Open-Meteo archive lags by several days. ``forecast`` should include
    ``past_days`` so the join is a continuous half-hourly path.
    """
    forecast = _normalise_weather_timestamps(forecast)
    if archive is None or archive.empty:
        return _fill_half_hourly(forecast)
    archive = _normalise_weather_timestamps(archive)
    cutoff = archive["timestamp"].max()
    weather = pd.concat(
        [archive[archive["timestamp"] <= cutoff], forecast[forecast["timestamp"] > cutoff]],
        ignore_index=True,
    )
    return _fill_half_hourly(weather)


def _normalise_weather_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    return out


def _fill_half_hourly(weather: pd.DataFrame) -> pd.DataFrame:
    """Reindex each region onto a complete 30-minute grid and interpolate weather."""
    parts: list[pd.DataFrame] = []
    for region_id, group in weather.groupby("region_id", sort=False):
        group = group.sort_values("timestamp").drop_duplicates("timestamp")
        if group.empty:
            continue
        full_index = pd.date_range(group["timestamp"].min(), group["timestamp"].max(), freq="30min")
        filled = group.set_index("timestamp").reindex(full_index)
        filled[list(HOURLY_VARS)] = filled[list(HOURLY_VARS)].interpolate(method="time").ffill().bfill()
        filled = filled.reset_index().rename(columns={"index": "timestamp"})
        filled["region_id"] = region_id
        filled = add_settlement_keys(filled, time_col="timestamp")
        parts.append(filled.loc[:, ["region_id", "timestamp", "settlement_date", "settlement_period", *HOURLY_VARS]])
    if not parts:
        return weather.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _load_or_fetch_previous_runs(
    region_id: str,
    *,
    start_date: str,
    end_date: str,
    use_cache: bool,
    processed_dir: Path | None,
) -> pd.DataFrame:
    processed_dir = processed_dir or DATA_PROCESSED
    ensure_data_dirs()
    path = _cache_path(processed_dir, region_id, "previous_runs")
    cached = pd.DataFrame()
    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
        cached["settlement_date"] = pd.to_datetime(cached["settlement_date"]).dt.tz_localize(None).dt.normalize()

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    fetch_start = start
    if not cached.empty:
        fetch_start = cached["settlement_date"].max() + pd.Timedelta(days=1)
        if fetch_start > end:
            return cached[(cached["settlement_date"] >= start) & (cached["settlement_date"] <= end)].copy()

    region = get_series(region_id)
    year_frames = [
        _average_point_frames(
            [_fetch_previous_runs_point(point, year_start, year_end) for point in region.weather_points],
            previous_runs_hourly_vars(),
        )
        for year_start, year_end in _year_windows(fetch_start.strftime("%Y-%m-%d"), end_date)
    ]
    hourly = pd.concat(year_frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    aligned = align_hourly_to_settlement(hourly, region_id=region_id)
    combined = pd.concat([cached, aligned], ignore_index=True)
    combined = combined.drop_duplicates(["region_id", "timestamp"]).sort_values("timestamp")
    combined.to_parquet(path, index=False)
    logger.info("Wrote weather cache %s", path)
    return combined[(combined["settlement_date"] >= start) & (combined["settlement_date"] <= end)].copy()


def _year_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        year_end = min(pd.Timestamp(year=cursor.year, month=12, day=31), end)
        windows.append((cursor.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
        cursor = year_end + pd.Timedelta(days=1)
    return windows


def _fetch_previous_runs_point(point: WeatherPoint, start_date: str, end_date: str) -> pd.DataFrame:
    params: dict[str, object] = {
        "latitude": point.latitude,
        "longitude": point.longitude,
        "hourly": ",".join(previous_runs_hourly_vars()),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "start_date": start_date,
        "end_date": end_date,
        "models": "best_match",
    }
    logger.info("Fetching Open-Meteo previous runs for %s %s to %s", point.name, start_date, end_date)
    try:
        payload = _get_json(PREVIOUS_RUNS_URL, params, retries=PREVIOUS_RUNS_RETRIES)
    except requests.RequestException as exc:
        raise WeatherAPIError(f"Open-Meteo previous-runs request failed for {point.name}: {exc}") from exc
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise WeatherAPIError(f"Unexpected Open-Meteo previous-runs payload for {point.name}: {payload}")
    hourly = pd.DataFrame(payload["hourly"]).rename(columns={"time": "timestamp"})
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    for column in previous_runs_hourly_vars():
        if column in hourly.columns:
            hourly[column] = pd.to_numeric(hourly[column], errors="coerce")
    time.sleep(PREVIOUS_RUNS_PAUSE_S)
    return hourly


def _get_json(url: str, params: dict[str, object], *, retries: int = 1) -> dict:
    last_error: requests.RequestException | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_S)
            if response.status_code in RETRY_STATUSES and attempt + 1 < retries:
                try:
                    wait = float(response.headers.get("Retry-After", 8 * (2**attempt)))
                except (TypeError, ValueError):
                    wait = 8 * (2**attempt)
                logger.warning(
                    "Open-Meteo %s; sleeping %.0fs then retrying",
                    response.status_code,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 >= retries:
                raise
            time.sleep(15 * (2**attempt))
    raise last_error or requests.RequestException("Open-Meteo request failed")


def _average_point_frames(point_frames: list[pd.DataFrame], value_columns: tuple[str, ...]) -> pd.DataFrame:
    columns = [column for column in value_columns if any(column in frame.columns for frame in point_frames)]
    acc = None
    count = None
    for frame in point_frames:
        values = frame.set_index("timestamp")[columns].apply(pd.to_numeric, errors="coerce")
        present = values.notna().astype(float)
        if acc is None:
            acc = values.astype(float)
            count = present
        else:
            acc = acc.add(values, fill_value=0)
            count = count.add(present, fill_value=0)
    return (acc / count).reset_index()


def _load_or_fetch_region(
    region_id: str,
    *,
    source: str,
    use_cache: bool,
    processed_dir: Path | None,
    start_date: str | None = None,
    end_date: str | None = None,
    forecast_days: int = 7,
    past_days: int = 0,
) -> pd.DataFrame:
    processed_dir = processed_dir or DATA_PROCESSED
    ensure_data_dirs()
    path = _cache_path(processed_dir, region_id, source)
    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
        cached["settlement_date"] = pd.to_datetime(cached["settlement_date"]).dt.tz_localize(None).dt.normalize()
        if source == "archive" and start_date and end_date and _cache_covers(cached, start_date, end_date):
            start = pd.Timestamp(start_date).normalize()
            end = pd.Timestamp(end_date).normalize()
            return cached[
                (cached["settlement_date"] >= start) & (cached["settlement_date"] <= end)
            ].copy()
        if source == "forecast":
            return cached

    region = get_series(region_id)
    try:
        hourly = _fetch_region_hourly(
            region,
            source=source,
            start_date=start_date,
            end_date=end_date,
            forecast_days=forecast_days,
            past_days=past_days,
        )
    except WeatherAPIError:
        if source == "forecast" and path.exists():
            logger.warning("Open-Meteo forecast failed; using cached %s", path)
            cached = pd.read_parquet(path)
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
            cached["settlement_date"] = pd.to_datetime(cached["settlement_date"]).dt.tz_localize(None).dt.normalize()
            return cached
        raise
    aligned = align_hourly_to_settlement(hourly, region_id=region_id)
    aligned.to_parquet(path, index=False)
    logger.info("Wrote weather cache %s", path)
    return aligned


def _cache_covers(cached: pd.DataFrame, start_date: str, end_date: str) -> bool:
    if cached.empty:
        return False
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return cached["settlement_date"].min() <= start and cached["settlement_date"].max() >= end


def _cache_path(processed_dir: Path, region_id: str, source: str) -> Path:
    safe_id = region_id.replace("/", "_")
    return processed_dir / f"weather_{safe_id}_{source}.parquet"


def _fetch_region_hourly(
    region: Series,
    *,
    source: str,
    start_date: str | None,
    end_date: str | None,
    forecast_days: int,
    past_days: int = 0,
) -> pd.DataFrame:
    point_frames = [
        _fetch_point_hourly(
            point,
            source=source,
            start_date=start_date,
            end_date=end_date,
            forecast_days=forecast_days,
            past_days=past_days,
        )
        for point in region.weather_points
    ]
    if len(point_frames) == 1:
        return point_frames[0]

    acc = None
    count = None
    for frame in point_frames:
        values = frame.set_index("timestamp")[list(HOURLY_VARS)]
        present = values.notna().astype(float)
        if acc is None:
            acc = values.astype(float)
            count = present
        else:
            acc = acc.add(values, fill_value=0)
            count = count.add(present, fill_value=0)
    return (acc / count).reset_index()


def _fetch_point_hourly(
    point: WeatherPoint,
    *,
    source: str,
    start_date: str | None,
    end_date: str | None,
    forecast_days: int,
    past_days: int = 0,
) -> pd.DataFrame:
    params: dict[str, object] = {
        "latitude": point.latitude,
        "longitude": point.longitude,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    }
    if source == "archive":
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required for archive weather")
        url = ARCHIVE_URL
        params["start_date"] = start_date
        params["end_date"] = end_date
    elif source == "forecast":
        url = FORECAST_URL
        params["forecast_days"] = forecast_days
        params["models"] = "best_match"
        if past_days:
            params["past_days"] = int(past_days)
    else:
        raise ValueError(f"Unknown weather source {source!r}")

    logger.info("Fetching Open-Meteo %s weather for %s", source, point.name)
    try:
        payload = _get_json(url, params, retries=FORECAST_RETRIES)
    except requests.RequestException as exc:
        raise WeatherAPIError(f"Open-Meteo request failed for {point.name}: {exc}") from exc

    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise WeatherAPIError(f"Unexpected Open-Meteo payload for {point.name}: {payload}")

    hourly = pd.DataFrame(payload["hourly"])
    hourly = hourly.rename(columns={"time": "timestamp"})
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True).dt.tz_convert(LONDON_TZ)
    for column in HOURLY_VARS:
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")
    time.sleep(REQUEST_PAUSE_S)
    return hourly[["timestamp", *HOURLY_VARS]]


def align_hourly_to_settlement(hourly: pd.DataFrame, *, region_id: str) -> pd.DataFrame:
    """Interpolate hourly weather onto a 30-minute settlement grid."""
    frame = hourly.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    end = frame.index.max().floor("h") + pd.Timedelta(minutes=30)
    full_index = pd.date_range(start=frame.index.min().floor("30min"), end=end, freq="30min")
    value_cols = [column for column in frame.columns if column != "timestamp"]
    half_hourly = frame.reindex(full_index)
    half_hourly[value_cols] = half_hourly[value_cols].interpolate(method="time").ffill().bfill()
    half_hourly = half_hourly.reset_index().rename(columns={"index": "timestamp"})
    half_hourly = add_settlement_keys(half_hourly, time_col="timestamp")
    half_hourly["region_id"] = region_id
    columns = ["region_id", "timestamp", "settlement_date", "settlement_period", *value_cols]
    return half_hourly.loc[:, [column for column in columns if column in half_hourly.columns]]
