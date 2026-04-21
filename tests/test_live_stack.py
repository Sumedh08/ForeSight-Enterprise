from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from api.main import app
from infra.vanna_engine import VannaSemanticError


ROOT = Path(__file__).resolve().parent.parent


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _service_ready() -> bool:
    if not _port_open("127.0.0.1", 5432):
        return False
    if not _port_open("127.0.0.1", 47334):
        return False
    try:
        response = httpx.get("http://127.0.0.1:47334/api/status", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _service_ready(), reason="Live Postgres/MindsDB services are not reachable on localhost.")
def test_live_postgres_and_mindsdb_flow(tmp_path, monkeypatch):
    import api.routes.data as data_module
    import orchestrator.enterprise_orchestrator as enterprise_module

    profile = {
        "id": "live-warehouse",
        "name": "enterprise-warehouse",
        "type": "postgres",
        "config": {
            "dsn": "postgresql://admin:adminpassword@127.0.0.1:5432/natwest_db",
            "mindsdb_dsn": "postgresql://admin:adminpassword@postgres:5432/natwest_db",
        },
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

    fake_vanna = FakeVanna()

    with TestClient(app) as client:
        services = app.state.services
        services.training_store.path = tmp_path / "live_training_jobs.json"
        services.vanna_cache.path = tmp_path / "live_vanna_cache.json"
        services.vanna_cache.train_successes = False
        services.mindsdb_client.base_url = "http://127.0.0.1:47334/api/sql/query"
        services.mindsdb_client.health_url = "http://127.0.0.1:47334/api/status"
        monkeypatch.setattr(services.connection_manager, "ensure_enterprise_warehouse_profile", lambda activate=True: profile)
        monkeypatch.setattr(services.connection_manager, "get_active_profile", lambda: profile)
        monkeypatch.setattr(data_module, "vanna_engine", fake_vanna)
        monkeypatch.setattr(enterprise_module, "vanna_engine", fake_vanna)

        with (ROOT / "MORTGAGE30US.csv").open("rb") as handle:
            upload = client.post(
                "/api/data/upload",
                files={"file": ("MORTGAGE30US.csv", handle.read(), "text/csv")},
            )
        assert upload.status_code == 200
        upload_body = upload.json()
        assert upload_body["status"] == "ok"
        assert upload_body["training_jobs"]
        fake_vanna.table_name = upload_body["table_name"]

        latest = client.post(
            "/query",
            json={"question": "Show the latest 10 30-year mortgage rates", "mode": "sql"},
        )
        assert latest.status_code == 200
        latest_body = latest.json()
        assert latest_body["status"] in {"ok", "degraded"}
        assert latest_body["artifacts"]["row_count"] >= 10

        repeat = client.post(
            "/query",
            json={"question": "Show the latest 10 30-year mortgage rates", "mode": "sql"},
        )
        assert repeat.status_code == 200
        assert float(repeat.json()["latency_ms"].get("cache_hit", 0.0)) >= 1.0

        connector = services.connection_manager.build_connector(profile, read_only=True)
        probe = connector.execute(
            f'SELECT observation_date AS period, "{fake_vanna.metric_column}" AS value '
            f'FROM "{fake_vanna.table_name}" ORDER BY observation_date',
            preview_limit=20,
        )
        assert probe.row_count >= 100
        assert float(probe.rows[0]["value"]) > 0

        forecast = client.post(
            "/query",
            json={"question": "Forecast the 30-year mortgage rate for the next 12 weeks", "mode": "forecast"},
        )
        assert forecast.status_code == 200
        forecast_body = forecast.json()
        assert forecast_body["status"] in {"training", "ok", "degraded"}
        if forecast_body["status"] == "training":
            assert forecast_body["artifacts"]["preview_baseline"]
        else:
            assert forecast_body["artifacts"]["point_forecast"]

        semantic_gap = client.post(
            "/query",
            json={"question": "Show refinance applications", "mode": "sql"},
        )
        assert semantic_gap.status_code == 200
        assert semantic_gap.json()["status"] == "blocked"
        assert "Define a new metric" in semantic_gap.json()["answer"]
