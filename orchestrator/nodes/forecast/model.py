from __future__ import annotations

from orchestrator.nodes.forecast.baseline import seasonal_naive_forecast


def run_forecast_model(series: list[dict], *, horizon: int, grain: str) -> dict:
    """Deterministic seasonal naive baseline used for legacy forecast paths."""
    point_forecast = seasonal_naive_forecast(series, horizon=horizon, grain=grain)
    return {
        "point_forecast": point_forecast,
        "prediction_intervals": [],
        "warnings": [],
    }
