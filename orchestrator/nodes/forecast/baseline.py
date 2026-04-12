from __future__ import annotations

from datetime import datetime, timedelta


def _period_step(grain: str) -> timedelta:
    if grain == "day":
        return timedelta(days=1)
    if grain == "month":
        return timedelta(days=30)
    return timedelta(weeks=1)


def _season_length(grain: str) -> int:
    if grain == "day":
        return 7
    if grain == "month":
        return 12
    return 4


def _generate_periods(last_period: datetime, horizon: int, grain: str) -> list[datetime]:
    step = _period_step(grain)
    return [last_period + step * offset for offset in range(1, horizon + 1)]


def seasonal_naive_forecast(series: list[dict], *, horizon: int, grain: str) -> list[dict]:
    if not series:
        return []
    values = [item["value"] for item in series]
    periods = _generate_periods(series[-1]["period"], horizon, grain)
    season_length = min(_season_length(grain), len(values))
    return [
        {"period": period, "value": float(values[-season_length + (index % season_length)])}
        for index, period in enumerate(periods)
    ]
