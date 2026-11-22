"""Live weather for the response LLM, via Open-Meteo (free, no API key).

Flow: geocode the place name -> pull current + 3-day forecast -> format a
compact briefing block that the hub injects into the LLM prompt so GLaDOS
can quote real numbers instead of admitting she has no search access.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("glados_hub.weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 5.0

# WMO weather interpretation codes -> spoken descriptions
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def describe_wmo(code) -> str:
    """Map a WMO weather code to a short spoken description."""
    try:
        return WMO_CODES.get(int(code), "unknown conditions")
    except (TypeError, ValueError):
        return "unknown conditions"


def _fmt_temp(value) -> str:
    return f"{round(float(value))} degrees Fahrenheit"


def format_weather_block(place: str, current: dict, daily: dict) -> str:
    """Render the Open-Meteo JSON payloads into an LLM-ready briefing."""
    lines = [f"Live weather data for {place} (from Open-Meteo, fetched just now, authoritative):"]

    if current:
        lines.append(
            "Right now: "
            f"{_fmt_temp(current.get('temperature_2m', 0))}, "
            f"feels like {_fmt_temp(current.get('apparent_temperature', 0))}, "
            f"{describe_wmo(current.get('weather_code'))}, "
            f"wind {round(float(current.get('wind_speed_10m', 0)))} mph."
        )

    days = daily.get("time", []) if daily else []
    labels = ["today", "tomorrow", "the day after tomorrow"]
    for i, day in enumerate(days[:3]):
        hi = daily.get("temperature_2m_max", [0] * 3)[i]
        lo = daily.get("temperature_2m_min", [0] * 3)[i]
        codes = daily.get("weather_code", [None] * 3)[i]
        probs = daily.get("precipitation_probability_max", [None] * 3)[i]
        prob = f"{probs} percent" if isinstance(probs, (int, float)) else "unknown"
        lines.append(
            f"{labels[i]} ({day}): {describe_wmo(codes)}, "
            f"high {_fmt_temp(hi)}, low {_fmt_temp(lo)}, {prob} chance of precipitation."
        )
    lines.append("Answer the user's weather question using this data.")
    return "\n".join(lines)


async def get_weather_block(location: str, client: httpx.AsyncClient) -> str:
    """Geocode `location` and return a formatted weather briefing, or '' on failure."""
    location = (location or "").strip()
    if not location:
        return ""
    try:
        geo = await client.get(
            GEOCODE_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=_TIMEOUT,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            logger.warning("weather: could not geocode %r", location)
            return ""
        g = results[0]
        place = ", ".join(x for x in (g.get("name"), g.get("admin1"), g.get("country")) if x)

        fc = await client.get(
            FORECAST_URL,
            params={
                "latitude": g["latitude"],
                "longitude": g["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=_TIMEOUT,
        )
        fc.raise_for_status()
        data = fc.json()
        block = format_weather_block(place, data.get("current"), data.get("daily"))
        logger.info("weather: %s", place)
        return block
    except Exception as e:  # network down, bad JSON, etc. - never block the conversation
        logger.warning("weather lookup failed for %r: %s", location, e)
        return ""
