"""GB weather stations used to drive the national demand model."""

from __future__ import annotations

from dataclasses import dataclass

SERIES_ID = "gb"


@dataclass(frozen=True)
class WeatherPoint:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Series:
    id: str
    name: str
    weather_points: tuple[WeatherPoint, ...]
    holiday_subdiv: str = "England"


GB = Series(
    id=SERIES_ID,
    name="Great Britain",
    weather_points=(
        WeatherPoint("London", 51.5074, -0.1278),
        WeatherPoint("Birmingham", 52.4862, -1.8904),
        WeatherPoint("Manchester", 53.4808, -2.2426),
        WeatherPoint("Glasgow", 55.8642, -4.2518),
    ),
    holiday_subdiv="England",
)


def get_series(series_id: str = SERIES_ID) -> Series:
    if series_id != SERIES_ID:
        raise KeyError(f"Only {SERIES_ID!r} is supported; got {series_id!r}")
    return GB
