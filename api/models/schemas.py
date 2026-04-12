from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WarningItem(BaseModel):
    kind: Literal["data_quality", "model_risk", "safety_block", "degraded_mode", "performance"]
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["auto", "sql", "forecast", "anomaly"] = "auto"
    filters: dict[str, Any] | None = None
    series_id: str | None = None
    metric: str | None = None
    dimensions: dict[str, Any] | None = None
    horizon: int | None = None
    grain: str | None = None
    scenario: dict[str, Any] | None = None


class SQLArtifact(BaseModel):
    generated_sql: str
    selected_tables: list[str]
    validation_status: Literal["valid", "repaired", "failed"]
    row_count: int
    preview_rows: list[dict[str, Any]]


class ForecastArtifact(BaseModel):
    series_id: str
    baseline: list[dict[str, Any]]
    point_forecast: list[dict[str, Any]]
    prediction_intervals: list[dict[str, Any]]
    anomalies: list[dict[str, Any]]
    backtest_metrics: dict[str, Any]


class ScenarioArtifact(BaseModel):
    series_id: str
    baseline_forecast: list[dict[str, Any]]
    scenario_forecast: list[dict[str, Any]]
    baseline_intervals: list[dict[str, Any]]
    scenario_intervals: list[dict[str, Any]]
    scenario_description: str
    comparison_summary: str


class QueryResponse(BaseModel):
    status: Literal["ok", "degraded", "blocked"]
    task_type: Literal["sql", "forecast", "anomaly", "scenario", "unclear"]
    answer: str
    confidence: float
    warnings: list[WarningItem]
    artifacts: SQLArtifact | ForecastArtifact | ScenarioArtifact | None
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, str]
