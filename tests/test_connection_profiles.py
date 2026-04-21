from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("duckdb")

from infra.connection_profiles import ConnectionProfileManager, ConnectionProfileStore, derive_mindsdb_dsn


def test_profile_save_list_activate(tmp_path: Path):
    store_path = tmp_path / "profiles.json"
    manager = ConnectionProfileManager(store=ConnectionProfileStore(store_path))

    profiles = manager.list_profiles(redact=True)
    assert profiles, "Default profile should be created."
    first_id = profiles[0]["id"]

    saved = manager.save_profile(
        name="sqlite-local",
        connection_type="sqlite",
        config={"path": str(tmp_path / "demo.sqlite")},
        activate=True,
    )
    assert saved["id"] != first_id

    listed = manager.list_profiles(redact=True)
    listed_map = {item["id"]: item for item in listed}
    assert listed_map[saved["id"]]["active"] is True
    assert listed_map[saved["id"]]["config"]["path"].endswith("demo.sqlite")


def test_profile_redacts_secret_fields(tmp_path: Path):
    store_path = tmp_path / "profiles.json"
    manager = ConnectionProfileManager(store=ConnectionProfileStore(store_path))
    manager.save_profile(
        name="pg-secret",
        connection_type="postgres",
        config={
            "host": "db",
            "port": 5432,
            "database": "analytics",
            "username": "tester",
            "password": "super-secret",
        },
        activate=False,
    )
    redacted = manager.list_profiles(redact=True)
    found = next(item for item in redacted if item["name"] == "pg-secret")
    assert found["config"]["password"] == "***"

    raw_payload = json.loads(store_path.read_text(encoding="utf-8"))
    raw = next(item for item in raw_payload["profiles"] if item["name"] == "pg-secret")
    assert raw["config"]["password"] == "super-secret"


def test_derive_mindsdb_dsn_rewrites_localhost_for_containers():
    derived = derive_mindsdb_dsn("postgresql://admin:adminpassword@127.0.0.1:5432/natwest_db")
    assert derived == "postgresql://admin:adminpassword@host.docker.internal:5432/natwest_db"
