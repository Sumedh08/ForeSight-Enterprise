from __future__ import annotations

import pytest


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("duckdb")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from api.main import app
from infra.vanna_engine import VannaSemanticError


def test_health_endpoint_contract():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded"}
        assert isinstance(body.get("components"), dict)
        assert "database" in body["components"]


def test_connection_list_contract():
    with TestClient(app) as client:
        response = client.get("/api/connections")
        assert response.status_code == 200
        body = response.json()
        assert "profiles" in body
        assert isinstance(body["profiles"], list)
        assert body["profiles"], "Expected at least one default connection profile."


def test_upload_query_and_forecast_flow_with_mortgage_data(tmp_path, monkeypatch):
    import api.routes.data as data_module
    import orchestrator.enterprise_orchestrator as enterprise_module

    profile = {
        "id": "warehouse-test",
        "name": "enterprise-warehouse",
        "type": "duckdb",
        "config": {"path": str(tmp_path / "warehouse.duckdb")},
    }

    class FakeVanna:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.table_name = "mortgage30us"
            self.metric_column = "MORTGAGE30US"

        def train_on_ddl(self, ddl: str) -> None:
            return None

        async def generate_sql(self, question: str) -> str:
            self.calls.append(question)
            lowered = question.lower()
            if "refinance applications" in lowered:
                raise VannaSemanticError("unknown metric")
            if "latest 10" in lowered:
                return (
                    f'SELECT observation_date, "{self.metric_column}" '
                    f'FROM "{self.table_name}" ORDER BY observation_date DESC LIMIT 10'
                )
            return (
                f'SELECT observation_date AS period, "{self.metric_column}" AS value '
                f'FROM "{self.table_name}" ORDER BY observation_date'
            )

    class FakeMindsDB:
        def __init__(self) -> None:
            self.status_calls: dict[str, int] = {}

        async def create_time_series_predictor(self, **kwargs):
            predictor_name = kwargs["predictor_name"]
            self.status_calls.setdefault(predictor_name, 0)
            return predictor_name

        async def get_predictor_state(self, predictor_name: str):
            calls = self.status_calls.get(predictor_name, 0)
            self.status_calls[predictor_name] = calls + 1
            if calls < 2:
                return "training", 72, "MindsDB is training the predictor."
            return "ready", 100, "Predictor is ready."

        async def run_predictor(self, *, predictor: str, row_cap: int):
            rows = []
            for index in range(1, min(row_cap, 12) + 1):
                rows.append(
                    {
                        "observation_date": f"2026-04-{9 + index:02d}",
                        "MORTGAGE30US": 6.8 + (index * 0.01),
                    }
                )
            return rows

        async def health(self) -> str:
            return "up"

    fake_vanna = FakeVanna()
    fake_mindsdb = FakeMindsDB()

    with TestClient(app) as client:
        services = app.state.services
        services.training_store.path = tmp_path / "training_jobs.json"
        services.vanna_cache.path = tmp_path / "vanna_cache.json"
        monkeypatch.setattr(services.connection_manager, "ensure_enterprise_warehouse_profile", lambda activate=True: profile)
        monkeypatch.setattr(services.connection_manager, "get_active_profile", lambda: profile)
        services.mindsdb_client.create_time_series_predictor = fake_mindsdb.create_time_series_predictor
        services.mindsdb_client.get_predictor_state = fake_mindsdb.get_predictor_state
        services.mindsdb_client.run_predictor = fake_mindsdb.run_predictor
        services.mindsdb_client.health = fake_mindsdb.health
        monkeypatch.setattr(data_module, "vanna_engine", fake_vanna)
        monkeypatch.setattr(enterprise_module, "vanna_engine", fake_vanna)

        with open("MORTGAGE30US.csv", "rb") as handle:
            upload = client.post(
                "/api/data/upload",
                files={"file": ("MORTGAGE30US.csv", handle.read(), "text/csv")},
            )
        assert upload.status_code == 200
        upload_body = upload.json()
        assert upload_body["warehouse_profile"] == "enterprise-warehouse"
        assert upload_body["training_jobs"], "Expected at least one queued training job."
        fake_vanna.table_name = upload_body["table_name"]
        series_id = upload_body["training_jobs"][0]["series_id"]
        ingestion_id = upload_body["ingestion_id"]

        first_sql = client.post(
            "/query",
            json={"question": "Show the latest 10 30-year mortgage rates", "mode": "sql"},
        )
        assert first_sql.status_code == 200
        assert first_sql.json()["status"] in {"ok", "degraded"}

        second_sql = client.post(
            "/query",
            json={"question": "Show the latest 10 30-year mortgage rates", "mode": "sql"},
        )
        assert second_sql.status_code == 200
        assert len([call for call in fake_vanna.calls if "latest 10" in call.lower()]) == 1

        forecast_training = client.post(
            "/query",
            json={"question": "Forecast the 30-year mortgage rate for the next 12 weeks", "mode": "forecast"},
        )
        assert forecast_training.status_code == 200
        forecast_training_body = forecast_training.json()
        assert forecast_training_body["status"] == "training"
        assert forecast_training_body["artifacts"]["preview_baseline"]

        training_status = client.get(f"/api/data/training?ingestion_id={ingestion_id}")
        assert training_status.status_code == 200
        training_body = training_status.json()
        assert training_body["jobs"][0]["state"] == "ready"

        forecast_ready = client.post(
            "/query",
            json={
                "question": "Forecast the 30-year mortgage rate for the next 12 weeks",
                "mode": "forecast",
                "series_id": series_id,
            },
        )
        assert forecast_ready.status_code == 200
        forecast_ready_body = forecast_ready.json()
        assert forecast_ready_body["status"] == "ok"
        assert forecast_ready_body["artifacts"]["point_forecast"]

        semantic_gap = client.post(
            "/query",
            json={"question": "Show refinance applications", "mode": "sql"},
        )
        assert semantic_gap.status_code == 200
        assert semantic_gap.json()["status"] == "blocked"
        assert "Define a new metric" in semantic_gap.json()["answer"]
