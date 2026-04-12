from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    period: datetime
    value: float
    low_80: float
    high_80: float
    low_95: float
    high_95: float


@dataclass(frozen=True, slots=True)
class ForecastEvaluation:
    mae: float
    mape: float
    baseline_mae: float
    baseline_mape: float
    beats_baseline: bool


@dataclass(frozen=True, slots=True)
class ForecastBenchmarkResult:
    synthetic: ForecastEvaluation
    real_world: ForecastEvaluation


@dataclass(frozen=True, slots=True)
class SyntheticForecastDataset:
    series: list[dict[str, float | datetime]]
    horizon: int
    grain: str


class SeasonalNaiveForecaster:
    def predict(self, series: list[dict], *, horizon: int, grain: str) -> list[ForecastPoint]:
        prepared = _prepare_series(series, grain)
        season_length = _season_length(grain, len(prepared))
        values = prepared["value"].to_numpy(dtype=float)
        history = values[-season_length:] if len(values) >= season_length else values
        sigma = float(np.std(values - np.mean(values))) if len(values) > 1 else max(abs(values[-1]) * 0.05, 1.0)
        future_index = _future_index(prepared.index[-1], horizon, grain)
        points = []
        for offset, period in enumerate(future_index):
            base = float(history[offset % len(history)])
            points.append(_make_point(period.to_pydatetime(), base, sigma))
        return points


class SeasonalTrendForecaster:
    def predict(self, series: list[dict], *, horizon: int, grain: str) -> list[ForecastPoint]:
        prepared = _prepare_series(series, grain)
        values = prepared["value"].to_numpy(dtype=float)
        length = len(values)
        time_index = np.arange(length, dtype=float)
        design = np.column_stack([np.ones(length), time_index])
        intercept, slope = np.linalg.lstsq(design, values, rcond=None)[0]
        trend = intercept + slope * time_index

        season_length = _season_length(grain, length)
        detrended = values - trend
        seasonal = np.zeros(season_length, dtype=float)
        counts = np.zeros(season_length, dtype=float)
        for idx, value in enumerate(detrended):
            bucket = idx % season_length
            seasonal[bucket] += value
            counts[bucket] += 1
        counts[counts == 0] = 1
        seasonal /= counts

        fitted = trend + np.array([seasonal[idx % season_length] for idx in range(length)], dtype=float)
        residuals = values - fitted
        spread = float(residuals.std()) if len(residuals) > 1 else max(abs(values[-1]) * 0.05, 1.0)
        q10, q90 = np.quantile(residuals, [0.1, 0.9]) if len(residuals) >= 8 else (-1.28 * spread, 1.28 * spread)
        q05, q95 = np.quantile(residuals, [0.05, 0.95]) if len(residuals) >= 12 else (-1.96 * spread, 1.96 * spread)

        future = _future_index(prepared.index[-1], horizon, grain)
        points = []
        for step, period in enumerate(future, start=1):
            base = intercept + slope * (length + step - 1)
            base += seasonal[(length + step - 1) % season_length]
            points.append(
                ForecastPoint(
                    period=period.to_pydatetime(),
                    value=float(base),
                    low_80=float(base + q10),
                    high_80=float(base + q90),
                    low_95=float(base + q05),
                    high_95=float(base + q95),
                )
            )
        return points


class ForecastingEngine:
    def __init__(self, *, model: SeasonalTrendForecaster | None = None, baseline: SeasonalNaiveForecaster | None = None) -> None:
        self.model = model or SeasonalTrendForecaster()
        self.baseline = baseline or SeasonalNaiveForecaster()

    def evaluate_series(self, series: list[dict], *, horizon: int, grain: str, folds: int = 3) -> ForecastEvaluation:
        prepared = _prepare_series(series, grain)
        values = prepared["value"].to_numpy(dtype=float)
        if len(values) <= horizon * 2:
            raise ValueError("Not enough history for rolling evaluation.")
        predictions: list[float] = []
        baselines: list[float] = []
        actuals: list[float] = []
        max_folds = max(1, min(folds, len(values) // horizon - 1))
        for fold in range(max_folds, 0, -1):
            cutoff = len(values) - fold * horizon
            train = [
                {"period": prepared.index[idx].to_pydatetime(), "value": float(values[idx])}
                for idx in range(cutoff)
            ]
            actual_window = values[cutoff : cutoff + horizon]
            model_points = self.model.predict(train, horizon=horizon, grain=grain)
            baseline_points = self.baseline.predict(train, horizon=horizon, grain=grain)
            for idx, actual in enumerate(actual_window):
                predictions.append(model_points[idx].value)
                baselines.append(baseline_points[idx].value)
                actuals.append(float(actual))
        return _evaluate_arrays(actuals, predictions, baselines)

    def evaluate(
        self,
        *,
        synthetic_dataset: SyntheticForecastDataset,
        real_world_series: list[dict],
    ) -> ForecastBenchmarkResult:
        synthetic = self.evaluate_series(
            synthetic_dataset.series,
            horizon=synthetic_dataset.horizon,
            grain=synthetic_dataset.grain,
        )
        real_world = self.evaluate_series(real_world_series, horizon=synthetic_dataset.horizon, grain=synthetic_dataset.grain)
        return ForecastBenchmarkResult(synthetic=synthetic, real_world=real_world)

    @staticmethod
    def synthetic_dataset(*, periods: int = 96, grain: str = "week", horizon: int = 8, seed: int = 13) -> SyntheticForecastDataset:
        rng = np.random.default_rng(seed)
        freq = {"day": "D", "week": "W-MON", "month": "MS"}[grain]
        index = pd.date_range("2024-01-01", periods=periods, freq=freq)
        season_length = _season_length(grain, periods)
        values = []
        for idx, period in enumerate(index):
            seasonal = 25.0 * np.sin(2 * np.pi * (idx % season_length) / max(season_length, 1))
            trend = 1.2 * idx
            noise = rng.normal(0, 8)
            values.append({"period": period.to_pydatetime(), "value": float(200 + trend + seasonal + noise)})
        return SyntheticForecastDataset(series=values, horizon=horizon, grain=grain)


def _prepare_series(series: Iterable[dict], grain: str) -> pd.DataFrame:
    frame = pd.DataFrame(series)
    if frame.empty:
        raise ValueError("Series is empty.")
    frame["period"] = pd.to_datetime(frame["period"])
    frame = frame.sort_values("period").set_index("period")
    freq = {"day": "D", "week": "W-MON", "month": "MS"}[grain]
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq=freq)
    frame = frame.reindex(full_index)
    frame["value"] = frame["value"].interpolate(method="linear").ffill().bfill()
    return frame


def _season_length(grain: str, size: int) -> int:
    default = {"day": 7, "week": 4, "month": 12}[grain]
    return max(2, min(default, max(2, size // 3)))


def _future_index(last_period: pd.Timestamp, horizon: int, grain: str) -> pd.DatetimeIndex:
    freq = {"day": "D", "week": "W-MON", "month": "MS"}[grain]
    return pd.date_range(last_period, periods=horizon + 1, freq=freq)[1:]


def _make_point(period: datetime, base: float, sigma: float) -> ForecastPoint:
    return ForecastPoint(
        period=period,
        value=float(base),
        low_80=float(base - 1.28 * sigma),
        high_80=float(base + 1.28 * sigma),
        low_95=float(base - 1.96 * sigma),
        high_95=float(base + 1.96 * sigma),
    )


def _evaluate_arrays(actuals: list[float], predictions: list[float], baselines: list[float]) -> ForecastEvaluation:
    actual = np.array(actuals, dtype=float)
    pred = np.array(predictions, dtype=float)
    base = np.array(baselines, dtype=float)
    denom = np.where(actual == 0, 1.0, np.abs(actual))
    mae = float(np.mean(np.abs(actual - pred)))
    baseline_mae = float(np.mean(np.abs(actual - base)))
    mape = float(np.mean(np.abs(actual - pred) / denom))
    baseline_mape = float(np.mean(np.abs(actual - base) / denom))
    return ForecastEvaluation(
        mae=mae,
        mape=mape,
        baseline_mae=baseline_mae,
        baseline_mape=baseline_mape,
        beats_baseline=mae < baseline_mae,
    )
