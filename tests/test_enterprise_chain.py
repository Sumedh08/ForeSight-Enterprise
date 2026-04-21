from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from api.models.schemas import TrainingArtifact
from components.connectors import ColumnSchema, DatabaseSchema, QueryExecutionResult, TableSchema
from infra.training_store import TrainingStore, build_training_jobs, detect_metric_columns, detect_temporal_column
from infra.vanna_engine import VannaSemanticCache, VannaSemanticError
from orchestrator.enterprise_orchestrator import EnterpriseOrchestrator


ROOT = Path(__file__).resolve().parent.parent


def _series_rows(period_alias: str = "period", value_alias: str = "value") -> list[dict[str, object]]:
    with (ROOT / "MORTGAGE30US.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        frame = list(reader)[:24]
    rows: list[dict[str, object]] = []
    for row in frame:
        rows.append(
            {
                period_alias: row["observation_date"],
                value_alias: row["MORTGAGE30US"],
            }
        )
    return rows


class FakeConnector:
    def __init__(self, rows: list[dict[str, object]], *, fail_dry_run_once: bool = False) -> None:
        self.rows = rows
        self.dry_runs: list[str] = []
        self.executions: list[str] = []
        self.fail_dry_run_once = fail_dry_run_once

    def introspect_schema(self, sample_limit: int = 3) -> DatabaseSchema:
        return DatabaseSchema(
            dialect="postgres",
            tables=[
                TableSchema(
                    name="mortgage30us",
                    columns=[
                        ColumnSchema(name="observation_date", data_type="date", sample_values=["2026-03-12"]),
                        ColumnSchema(name="MORTGAGE30US", data_type="double precision", sample_values=[6.11]),
                    ],
                )
            ],
        )

    def dry_run(self, sql: str) -> None:
        self.dry_runs.append(sql)
        if self.fail_dry_run_once:
            self.fail_dry_run_once = False
            raise RuntimeError("stale cached sql")

    def execute(self, sql: str, preview_limit: int = 50) -> QueryExecutionResult:
        self.executions.append(sql)
        preview = self.rows[:preview_limit]
        return QueryExecutionResult(columns=list(preview[0].keys()), rows=preview, row_count=len(self.rows))


class FakeManager:
    def __init__(self, connector: FakeConnector) -> None:
        self.connector = connector

    def get_active_profile(self) -> dict[str, object]:
        return {"id": "warehouse", "name": "enterprise-warehouse", "type": "postgres", "config": {"dsn": "postgresql://example"}}

    def build_connector(self, profile: dict[str, object], read_only: bool = True) -> FakeConnector:
        return self.connector


class FakeVanna:
    def __init__(self, *, sql: str | None = None, error: Exception | None = None) -> None:
        self.sql = sql or "SELECT observation_date AS period, MORTGAGE30US AS value FROM mortgage30us ORDER BY observation_date"
        self.error = error
        self.calls: list[str] = []

    def train_on_ddl(self, ddl: str) -> None:
        return None

    async def generate_sql(self, question: str) -> str:
        self.calls.append(question)
        if self.error is not None:
            raise self.error
        return self.sql


class FakeMindsDB:
    def __init__(self, *, state: tuple[str, int, str], forecast_rows: list[dict[str, object]] | None = None) -> None:
        self.state = state
        self.forecast_rows = forecast_rows or _series_rows(period_alias="observation_date", value_alias="MORTGAGE30US")

    async def get_predictor_state(self, predictor_name: str) -> tuple[str, int, str]:
        return self.state

    async def run_predictor(self, *, predictor: str, row_cap: int) -> list[dict[str, object]]:
        return self.forecast_rows[:row_cap]


def make_services(tmp_path: Path, *, connector_rows: list[dict[str, object]], mindsdb: FakeMindsDB, training_jobs=None, fail_dry_run_once: bool = False):
    connector = FakeConnector(connector_rows, fail_dry_run_once=fail_dry_run_once)
    training_store = TrainingStore(path=tmp_path / "training_jobs.json")
    if training_jobs:
        training_store.save_jobs(training_jobs)
    return SimpleNamespace(
        connection_manager=FakeManager(connector),
        mindsdb_client=mindsdb,
        training_store=training_store,
        vanna_cache=VannaSemanticCache(path=tmp_path / "vanna_cache.json", train_successes=False),
    )


def test_mortgage_upload_heuristics_identify_date_and_metric():
    with (ROOT / "MORTGAGE30US.csv").open("r", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))
    assert detect_temporal_column(frame) == "observation_date"
    assert detect_metric_columns(frame, date_column="observation_date") == ["MORTGAGE30US"]


def test_vanna_semantic_cache_remember_lookup_and_invalidate(tmp_path: Path):
    cache = VannaSemanticCache(path=tmp_path / "cache.json", train_successes=False)
    cache.remember(
        question="sql::Show the latest 10 mortgage rates",
        sql="SELECT * FROM mortgage30us ORDER BY observation_date DESC LIMIT 10",
        selected_tables=["mortgage30us"],
    )
    entry = cache.lookup("sql::Show the latest 10 mortgage rates")
    assert entry is not None
    assert entry.selected_tables == ["mortgage30us"]
    cache.invalidate(question="sql::Show the latest 10 mortgage rates")
    assert cache.lookup("sql::Show the latest 10 mortgage rates") is None


@pytest.mark.asyncio
async def test_vanna_semantic_miss_blocks_sql_with_define_metric_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import orchestrator.enterprise_orchestrator as enterprise_module

    services = make_services(
        tmp_path,
        connector_rows=_series_rows(),
        mindsdb=FakeMindsDB(state=("training", 72, "still training")),
    )
    monkeypatch.setattr(enterprise_module, "vanna_engine", FakeVanna(error=VannaSemanticError("no relevant data")))
    orchestrator = EnterpriseOrchestrator(services=services)

    response = await orchestrator._run_sql(SimpleNamespace(question="Show refinance applications", mode="sql"))

    assert response.status == "blocked"
    assert "Define a new metric" in response.answer


@pytest.mark.asyncio
async def test_vanna_cache_hit_bypasses_regeneration_and_invalidates_stale_sql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import orchestrator.enterprise_orchestrator as enterprise_module

    fake_vanna = FakeVanna()
    services = make_services(
        tmp_path,
        connector_rows=_series_rows(),
        mindsdb=FakeMindsDB(state=("ready", 100, "ready")),
        fail_dry_run_once=True,
    )
    services.vanna_cache.remember(
        question="sql::Show the latest 10 mortgage rates",
        sql="SELECT observation_date AS period, MORTGAGE30US AS value FROM mortgage30us ORDER BY observation_date",
        selected_tables=["mortgage30us"],
    )
    monkeypatch.setattr(enterprise_module, "vanna_engine", fake_vanna)
    orchestrator = EnterpriseOrchestrator(services=services)

    response = await orchestrator._run_sql(SimpleNamespace(question="Show the latest 10 mortgage rates", mode="sql"))

    assert response.status == "degraded"
    assert fake_vanna.calls == ["Show the latest 10 mortgage rates"]
    assert services.vanna_cache.lookup("sql::Show the latest 10 mortgage rates") is not None


@pytest.mark.asyncio
async def test_forecast_returns_training_status_with_preview_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import orchestrator.enterprise_orchestrator as enterprise_module

    with (ROOT / "MORTGAGE30US.csv").open("r", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))[:24]
    jobs = build_training_jobs(
        frame=frame,
        table_name="mortgage30us",
        ingestion_id="ing-1",
        warehouse_profile={"id": "warehouse", "name": "enterprise-warehouse"},
    )
    services = make_services(
        tmp_path,
        connector_rows=_series_rows(),
        mindsdb=FakeMindsDB(state=("training", 72, "MindsDB is training the predictor.")),
        training_jobs=jobs,
    )
    monkeypatch.setattr(enterprise_module, "vanna_engine", FakeVanna())
    orchestrator = EnterpriseOrchestrator(services=services)

    response = await orchestrator._run_forecast(
        SimpleNamespace(
            question="Forecast the 30-year mortgage rate for the next 12 weeks",
            mode="forecast",
            series_id=None,
            metric=None,
            horizon=12,
            grain="week",
        )
    )

    assert response.status == "training"
    assert isinstance(response.artifacts, TrainingArtifact)
    assert response.artifacts.progress_pct == 72
    assert response.artifacts.preview_baseline


@pytest.mark.asyncio
async def test_forecast_ready_path_returns_mindsdb_points_and_narrative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import orchestrator.enterprise_orchestrator as enterprise_module

    class FakeNIM:
        enabled = True

        async def chat(self, messages, temperature=0.1, max_tokens=220):
            return "Mortgage rates are projected to remain elevated over the forecast horizon."

    monkeypatch.setattr(enterprise_module, "nim_gateway", FakeNIM())
    monkeypatch.setattr(enterprise_module, "vanna_engine", FakeVanna())

    with (ROOT / "MORTGAGE30US.csv").open("r", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))[:24]
    jobs = build_training_jobs(
        frame=frame,
        table_name="mortgage30us",
        ingestion_id="ing-2",
        warehouse_profile={"id": "warehouse", "name": "enterprise-warehouse"},
    )
    forecast_rows = _series_rows(period_alias="observation_date", value_alias="MORTGAGE30US")
    services = make_services(
        tmp_path,
        connector_rows=_series_rows(),
        mindsdb=FakeMindsDB(state=("ready", 100, "ready"), forecast_rows=forecast_rows),
        training_jobs=jobs,
    )
    orchestrator = EnterpriseOrchestrator(services=services)

    response = await orchestrator._run_forecast(
        SimpleNamespace(
            question="Forecast the 30-year mortgage rate for the next 12 weeks",
            mode="forecast",
            series_id=None,
            metric=None,
            horizon=12,
            grain="week",
        )
    )

    assert response.status == "ok"
    assert response.artifacts is not None
    assert response.artifacts.point_forecast
    assert "projected to remain elevated" in response.answer


@pytest.mark.asyncio
async def test_failed_predictor_returns_degraded_naive_forecast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import orchestrator.enterprise_orchestrator as enterprise_module

    with (ROOT / "MORTGAGE30US.csv").open("r", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))[:24]
    jobs = build_training_jobs(
        frame=frame,
        table_name="mortgage30us",
        ingestion_id="ing-3",
        warehouse_profile={"id": "warehouse", "name": "enterprise-warehouse"},
    )
    failed_jobs = []
    for job in jobs:
        failed_jobs.append(job)
    failed_jobs[0].state = "failed"
    failed_jobs[0].progress_pct = 100
    failed_jobs[0].message = "ML engine must be specified when creating a model"

    services = make_services(
        tmp_path,
        connector_rows=_series_rows(),
        mindsdb=FakeMindsDB(state=("failed", 100, "ML engine must be specified when creating a model")),
        training_jobs=failed_jobs,
    )
    monkeypatch.setattr(enterprise_module, "vanna_engine", FakeVanna())
    orchestrator = EnterpriseOrchestrator(services=services)

    response = await orchestrator._run_forecast(
        SimpleNamespace(
            question="Forecast the 30-year mortgage rate for the next 12 weeks",
            mode="forecast",
            series_id=None,
            metric=None,
            horizon=12,
            grain="week",
        )
    )

    assert response.status == "degraded"
    assert response.artifacts is not None
    assert response.artifacts.point_forecast
    assert "seasonal naive baseline" in response.answer
