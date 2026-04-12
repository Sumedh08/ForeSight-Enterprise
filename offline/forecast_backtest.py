from __future__ import annotations

import json

from infra.metrics_registry import MetricRegistry
from orchestrator.graph import AppServices, AnalyticsOrchestrator
from orchestrator.nodes.forecast.backtest import rolling_backtest
from orchestrator.nodes.forecast.baseline import seasonal_naive_forecast
from orchestrator.nodes.forecast.model import run_forecast_model
from orchestrator.nodes.forecast.prep import prepare_series


def run_metric_backtests() -> dict:
    services = AppServices.bootstrap()
    orchestrator = AnalyticsOrchestrator(services)
    registry = MetricRegistry()

    results = {}
    for metric_key, metric in registry.metrics.items():
        history = orchestrator._fetch_metric_series(metric_key)
        prepared = prepare_series(history, metric.default_grain)
        results[metric_key] = rolling_backtest(
            prepared["series"],
            horizon=metric.default_horizon,
            grain=metric.default_grain,
            forecast_fn=run_forecast_model,
            baseline_fn=seasonal_naive_forecast,
        )
    return results


if __name__ == "__main__":
    print(json.dumps(run_metric_backtests(), indent=2, default=str))
