from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ConnectionType = Literal["postgres", "mysql", "sqlite", "duckdb"]


class WarningItem(BaseModel):
    kind: Literal["data_quality", "model_risk", "safety_block", "degraded_mode", "performance"]
    message: str


class TrainingJobSummary(BaseModel):
    series_id: str
    predictor_name: str
    state: Literal["queued", "training", "ready", "failed"]
    progress_pct: int
    message: str
    poll_after_ms: int = 3000


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["auto", "sql", "forecast", "anomaly", "scenario"] = "auto"
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


class TrainingArtifact(BaseModel):
    series_id: str
    predictor_name: str
    state: Literal["queued", "training", "ready", "failed"]
    progress_pct: int
    message: str
    poll_after_ms: int
    preview_baseline: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioArtifact(BaseModel):
    series_id: str
    baseline_forecast: list[dict[str, Any]]
    scenario_forecast: list[dict[str, Any]]
    baseline_intervals: list[dict[str, Any]]
    scenario_intervals: list[dict[str, Any]]
    scenario_description: str
    comparison_summary: str


class QueryResponse(BaseModel):
    status: Literal["ok", "degraded", "blocked", "training"]
    task_type: Literal["sql", "forecast", "anomaly", "scenario", "unclear"]
    answer: str
    confidence: float
    warnings: list[WarningItem]
    artifacts: SQLArtifact | ForecastArtifact | ScenarioArtifact | TrainingArtifact | None
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, str]
    active_connection: str | None = None
    active_connection_type: ConnectionType | None = None


class ConnectionTestRequest(BaseModel):
    connection_type: ConnectionType
    config: dict[str, Any]


class ConnectionSaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    connection_type: ConnectionType
    config: dict[str, Any]
    activate: bool = True


class ConnectionProfileResponse(BaseModel):
    id: str
    name: str
    connection_type: ConnectionType
    active: bool
    config: dict[str, Any]
    created_at: str
    updated_at: str


class ConnectionListResponse(BaseModel):
    active_profile_id: str | None
    profiles: list[ConnectionProfileResponse]


class ConnectionTestResponse(BaseModel):
    status: Literal["ok", "failed"]
    message: str
    dialect: str | None = None
    table_count: int | None = None
    tables: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    status: Literal["ok", "degraded", "failed"]
    message: str
    table_name: str | None = None
    ingestion_id: str | None = None
    warehouse_profile: str | None = None
    training_jobs: list[TrainingJobSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TrainingStatusResponse(BaseModel):
    status: Literal["ok"]
    ingestion_id: str | None = None
    series_id: str | None = None
    jobs: list[TrainingJobSummary] = Field(default_factory=list)
