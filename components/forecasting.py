from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable


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


def _parse_series(series: Iterable[dict], grain: str) -> list[dict[str, float | datetime]]:
    rows: list[dict[str, float | datetime]] = []
    for item in series:
        period = item.get("period")
        value = item.get("value")
        if not isinstance(period, datetime):
            if hasattr(period, "to_pydatetime"):
                period = period.to_pydatetime()
            else:
                period = datetime.fromisoformat(str(period))
        try:
            numeric = float(value)
        except Exception:
            continue
        rows.append({"period": period, "value": numeric})
    rows.sort(key=lambda row: row["period"])

    deduped: dict[datetime, float] = {}
    for row in rows:
        deduped[row["period"]] = float(row["value"])
    normalized = [{"period": period, "value": value} for period, value in sorted(deduped.items())]
    if grain == "day" and normalized:
        normalized = _fill_daily_gaps(normalized)
    return normalized


def _fill_daily_gaps(series: list[dict[str, float | datetime]]) -> list[dict[str, float | datetime]]:
    if not series:
        return []
    filled = [series[0]]
    for item in series[1:]:
        previous = filled[-1]
        previous_period = previous["period"]
        current_period = item["period"]
        assert isinstance(previous_period, datetime)
        assert isinstance(current_period, datetime)
        gap = (current_period - previous_period).days
        if gap > 1:
            step = (float(item["value"]) - float(previous["value"])) / float(gap)
            for offset in range(1, gap):
                filled.append(
                    {
                        "period": previous_period + timedelta(days=offset),
                        "value": float(previous["value"]) + (step * offset),
                    }
                )
        filled.append(item)
    return filled


def _season_length(grain: str, size: int) -> int:
    defaults = {"day": 7, "week": 4, "month": 12}
    default = defaults.get(grain, 4)
    return max(2, min(default, max(2, size // 3)))


def _future_period(last_period: datetime, grain: str, offset: int) -> datetime:
    if grain == "day":
        return last_period + timedelta(days=offset)
    if grain == "week":
        return last_period + timedelta(days=7 * offset)
    current = last_period
    month = current.month - 1 + offset
    year = current.year + month // 12
    month = month % 12 + 1
    day = min(current.day, 28)
    return datetime(year, month, day)


def _make_point(period: datetime, base: float, sigma: float) -> ForecastPoint:
    return ForecastPoint(
        period=period,
        value=float(base),
        low_80=float(base - 1.28 * sigma),
        high_80=float(base + 1.28 * sigma),
        low_95=float(base - 1.96 * sigma),
        high_95=float(base + 1.96 * sigma),
    )


class SeasonalNaiveForecaster:
    def predict(self, series: list[dict], *, horizon: int, grain: str) -> list[ForecastPoint]:
        prepared = _parse_series(series, grain)
        if not prepared:
            raise ValueError("Series is empty.")
        season_length = _season_length(grain, len(prepared))
        history = prepared[-season_length:]
        values = [float(item["value"]) for item in prepared]
        sigma = statistics.pstdev(values) if len(values) > 1 else max(abs(values[-1]) * 0.05, 1.0)
        last_period = prepared[-1]["period"]
        assert isinstance(last_period, datetime)
        points = []
        for offset in range(1, horizon + 1):
            base = float(history[(offset - 1) % len(history)]["value"])
            points.append(_make_point(_future_period(last_period, grain, offset), base, sigma))
        return points


class SeasonalTrendForecaster:
    def predict(self, series: list[dict], *, horizon: int, grain: str) -> list[ForecastPoint]:
        prepared = _parse_series(series, grain)
        if len(prepared) < 2:
            raise ValueError("Series has insufficient history.")

        values = [float(item["value"]) for item in prepared]
        indices = list(range(len(values)))
        mean_x = statistics.mean(indices)
        mean_y = statistics.mean(values)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(indices, values))
        denominator = sum((x - mean_x) ** 2 for x in indices) or 1.0
        slope = numerator / denominator
        intercept = mean_y - (slope * mean_x)

        season_length = _season_length(grain, len(prepared))
        detrended = [value - (intercept + slope * idx) for idx, value in enumerate(values)]
        seasonal_buckets: list[list[float]] = [[] for _ in range(season_length)]
        for idx, value in enumerate(detrended):
            seasonal_buckets[idx % season_length].append(value)
        seasonal = [statistics.mean(bucket) if bucket else 0.0 for bucket in seasonal_buckets]

        fitted = [intercept + slope * idx + seasonal[idx % season_length] for idx in range(len(values))]
        residuals = [actual - estimate for actual, estimate in zip(values, fitted)]
        spread = statistics.pstdev(residuals) if len(residuals) > 1 else max(abs(values[-1]) * 0.05, 1.0)
        last_period = prepared[-1]["period"]
        assert isinstance(last_period, datetime)

        points = []
        for step in range(1, horizon + 1):
            index = len(values) + step - 1
            base = intercept + slope * index + seasonal[index % season_length]
            points.append(_make_point(_future_period(last_period, grain, step), base, spread))
        return points


class ForecastingEngine:
    def __init__(self, *, model: SeasonalTrendForecaster | None = None, baseline: SeasonalNaiveForecaster | None = None) -> None:
        self.model = model or SeasonalTrendForecaster()
        self.baseline = baseline or SeasonalNaiveForecaster()

    def evaluate_series(self, series: list[dict], *, horizon: int, grain: str, folds: int = 3) -> ForecastEvaluation:
        prepared = _parse_series(series, grain)
        values = [float(item["value"]) for item in prepared]
        if len(values) <= horizon * 2:
            raise ValueError("Not enough history for rolling evaluation.")

        predictions: list[float] = []
        baselines: list[float] = []
        actuals: list[float] = []
        max_folds = max(1, min(folds, len(values) // horizon - 1))
        for fold in range(max_folds, 0, -1):
            cutoff = len(values) - fold * horizon
            train = prepared[:cutoff]
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
        rng = random.Random(seed)
        start = datetime(2024, 1, 1)
        series: list[dict[str, float | datetime]] = []
        season_length = _season_length(grain, periods)
        current = start
        for idx in range(periods):
            seasonal = 25.0 * math.sin(2 * math.pi * (idx % season_length) / max(season_length, 1))
            trend = 1.2 * idx
            noise = rng.gauss(0, 8)
            series.append({"period": current, "value": float(200 + trend + seasonal + noise)})
            current = _future_period(current, grain, 1)
        return SyntheticForecastDataset(series=series, horizon=horizon, grain=grain)


def _evaluate_arrays(actuals: list[float], predictions: list[float], baselines: list[float]) -> ForecastEvaluation:
    if not actuals:
        raise ValueError("No actual values were provided.")
    absolute_errors = [abs(actual - pred) for actual, pred in zip(actuals, predictions)]
    baseline_errors = [abs(actual - base) for actual, base in zip(actuals, baselines)]
    denominator = [abs(actual) if actual != 0 else 1.0 for actual in actuals]
    mae = sum(absolute_errors) / len(absolute_errors)
    baseline_mae = sum(baseline_errors) / len(baseline_errors)
    mape = sum(error / denom for error, denom in zip(absolute_errors, denominator)) / len(absolute_errors)
    baseline_mape = sum(error / denom for error, denom in zip(baseline_errors, denominator)) / len(baseline_errors)
    return ForecastEvaluation(
        mae=float(mae),
        mape=float(mape),
        baseline_mae=float(baseline_mae),
        baseline_mape=float(baseline_mape),
        beats_baseline=mae < baseline_mae,
    )
