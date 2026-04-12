from __future__ import annotations

import pandas as pd

from infra.metrics import warning


def rolling_backtest(
    series: list[dict],
    *,
    horizon: int,
    grain: str,
    forecast_fn,
    baseline_fn,
) -> dict:
    if len(series) < max(8, horizon * 2):
        return {
            "mae": None,
            "mape": None,
            "coverage_80": None,
            "coverage_95": None,
            "beats_baseline": False,
            "warnings": [warning("data_quality", "Not enough history for a rolling backtest.")],
        }

    results = []
    n_splits = min(3, max(1, len(series) // horizon - 1))
    for split in range(n_splits, 0, -1):
        cutoff = len(series) - split * horizon
        if cutoff < 8:
            continue
        train = series[:cutoff]
        actuals = series[cutoff : cutoff + horizon]
        forecast = forecast_fn(train, horizon=horizon, grain=grain)
        baseline = baseline_fn(train, horizon=horizon, grain=grain)
        for idx, actual in enumerate(actuals):
            if idx >= len(forecast["point_forecast"]) or idx >= len(baseline):
                break
            interval = forecast["prediction_intervals"][idx]
            results.append(
                {
                    "actual": actual["value"],
                    "forecast": forecast["point_forecast"][idx]["value"],
                    "baseline": baseline[idx]["value"],
                    "low_80": interval["low_80"],
                    "high_80": interval["high_80"],
                    "low_95": interval["low_95"],
                    "high_95": interval["high_95"],
                }
            )

    if not results:
        return {
            "mae": None,
            "mape": None,
            "coverage_80": None,
            "coverage_95": None,
            "beats_baseline": False,
            "warnings": [warning("data_quality", "Backtest splits could not be computed from the available history.")],
        }

    df = pd.DataFrame(results)
    actual_abs = df["actual"].abs().replace(0, 1)
    mae_model = float((df["actual"] - df["forecast"]).abs().mean())
    mae_baseline = float((df["actual"] - df["baseline"]).abs().mean())
    coverage_80 = float(((df["actual"] >= df["low_80"]) & (df["actual"] <= df["high_80"])).mean())
    coverage_95 = float(((df["actual"] >= df["low_95"]) & (df["actual"] <= df["high_95"])).mean())

    warnings = []
    if mae_model >= mae_baseline:
        warnings.append(warning("model_risk", "The forecast model did not beat the seasonal baseline on backtest."))
    if coverage_80 < 0.7:
        warnings.append(warning("model_risk", f"80% intervals only covered {coverage_80:.0%} of backtest actuals."))

    return {
        "mae": mae_model,
        "mape": float(((df["actual"] - df["forecast"]).abs() / actual_abs).mean()),
        "coverage_80": coverage_80,
        "coverage_95": coverage_95,
        "beats_baseline": mae_model < mae_baseline,
        "warnings": warnings,
    }
