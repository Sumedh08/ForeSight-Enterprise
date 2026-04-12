from __future__ import annotations

from typing import Any, Literal, TypedDict


class QueryState(TypedDict, total=False):
    request: dict[str, Any]
    route: Literal["sql", "forecast", "unclear"]
    response: dict[str, Any]
    latency_ms: dict[str, float]


class SQLState(TypedDict, total=False):
    question: str
    schema_snapshot: str
    keywords: list[str]
    question_skeleton: str
    retrieved_cells: list[dict[str, Any]]
    retrieved_examples: list[dict[str, Any]]
    sql_candidates: list[str]
    selected_sql: str
    validation_result: dict[str, Any]
    execution_result: dict[str, Any]
    repair_attempts: int
    warnings: list[dict[str, str]]
    latency_ms: dict[str, float]


class ForecastState(TypedDict, total=False):
    question: str
    series_data: list[dict[str, Any]]
    series_id: str
    metric: str
    grain: str
    horizon: int
    prepared_series: list[dict[str, Any]]
    baseline_forecast: list[dict[str, Any]]
    point_forecast: list[dict[str, Any]]
    prediction_intervals: list[dict[str, Any]]
    backtest_metrics: dict[str, Any]
    anomalies: list[dict[str, Any]]
    warnings: list[dict[str, str]]
    latency_ms: dict[str, float]
