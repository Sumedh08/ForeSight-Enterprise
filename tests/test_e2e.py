from __future__ import annotations

import pytest


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("duckdb")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint_contract():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body.get("components"), dict)
    assert "database" in body["components"]


def test_connection_list_contract():
    client = TestClient(app)
    response = client.get("/api/connections")
    assert response.status_code == 200
    body = response.json()
    assert "profiles" in body
    assert isinstance(body["profiles"], list)
    assert body["profiles"], "Expected at least one default connection profile."
