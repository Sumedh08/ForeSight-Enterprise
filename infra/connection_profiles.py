from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote_plus

from components.connectors import (
    DatabaseConnector,
    DuckDBConnector,
    MySQLConnector,
    PostgresConnector,
    SQLiteConnector,
)
from infra.settings import settings


ConnectionType = Literal["postgres", "mysql", "sqlite", "duckdb"]
SENSITIVE_KEYS = {"password", "token", "secret", "api_key"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_\- ]+", "", name).strip()
    return value or "connection"


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in config.items():
        lowered = key.lower()
        if lowered == "dsn" and value:
            text = str(value)
            if "@" in text and "://" in text:
                prefix, suffix = text.split("@", 1)
                scheme = prefix.split("://", 1)[0]
                redacted[key] = f"{scheme}://***@{suffix}"
            else:
                redacted[key] = "***"
            continue
        if lowered in SENSITIVE_KEYS and value:
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ConnectionProfileStore:
    path: Path

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.connection_profiles_path
        _ensure_parent(self.path)
        if not self.path.exists():
            self._write({"active_profile_id": None, "profiles": []})

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"active_profile_id": None, "profiles": []}

    def _write(self, payload: dict[str, Any]) -> None:
        _ensure_parent(self.path)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def all(self) -> dict[str, Any]:
        return self._read()

    def save(self, payload: dict[str, Any]) -> None:
        self._write(payload)


class ConnectionProfileManager:
    def __init__(self, store: ConnectionProfileStore | None = None) -> None:
        self.store = store or ConnectionProfileStore()
        self.ensure_default_profile()

    def ensure_default_profile(self) -> None:
        payload = self.store.all()
        if payload.get("profiles"):
            return
        default_type = settings.default_connection_type.lower().strip()
        if default_type not in {"duckdb", "sqlite", "postgres", "mysql"}:
            default_type = "duckdb"
        default_config: dict[str, Any]
        if default_type == "duckdb":
            settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            bootstrap_connector = DuckDBConnector(settings.duckdb_path, read_only=False)
            bootstrap_connector.execute("SELECT 1", preview_limit=1)
            default_config = {"path": str(settings.duckdb_path)}
        elif default_type == "sqlite":
            default_config = {"path": str(settings.data_dir / "default.sqlite")}
        else:
            # Keep a safe local fallback for first boot if env is incomplete.
            default_type = "duckdb"
            default_config = {"path": str(settings.duckdb_path)}

        profile = {
            "id": str(uuid.uuid4()),
            "name": "default-local",
            "type": default_type,
            "config": default_config,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        payload["profiles"] = [profile]
        payload["active_profile_id"] = profile["id"]
        self.store.save(payload)

    def list_profiles(self, *, redact: bool = True) -> list[dict[str, Any]]:
        payload = self.store.all()
        active_id = payload.get("active_profile_id")
        profiles = []
        for profile in payload.get("profiles", []):
            item = deepcopy(profile)
            item["active"] = item.get("id") == active_id
            if redact:
                item["config"] = redact_config(item.get("config", {}))
            profiles.append(item)
        return profiles

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        payload = self.store.all()
        for profile in payload.get("profiles", []):
            if profile.get("id") == profile_id:
                return deepcopy(profile)
        raise ValueError(f"Connection profile '{profile_id}' was not found.")

    def get_active_profile(self) -> dict[str, Any]:
        payload = self.store.all()
        active_id = payload.get("active_profile_id")
        for profile in payload.get("profiles", []):
            if profile.get("id") == active_id:
                return deepcopy(profile)
        raise ValueError("No active connection profile is configured.")

    def activate(self, profile_id: str) -> dict[str, Any]:
        payload = self.store.all()
        if not any(item.get("id") == profile_id for item in payload.get("profiles", [])):
            raise ValueError(f"Connection profile '{profile_id}' was not found.")
        payload["active_profile_id"] = profile_id
        self.store.save(payload)
        return self.get_profile(profile_id)

    def save_profile(
        self,
        *,
        name: str,
        connection_type: ConnectionType,
        config: dict[str, Any],
        activate: bool = False,
    ) -> dict[str, Any]:
        normalized = self.normalize_profile(connection_type=connection_type, config=config)
        payload = self.store.all()
        now = utc_now_iso()
        profile = {
            "id": str(uuid.uuid4()),
            "name": sanitize_name(name),
            "type": connection_type,
            "config": normalized,
            "created_at": now,
            "updated_at": now,
        }
        profiles = payload.get("profiles", [])
        profiles.append(profile)
        payload["profiles"] = profiles
        if activate or not payload.get("active_profile_id"):
            payload["active_profile_id"] = profile["id"]
        self.store.save(payload)
        return profile

    def normalize_profile(self, *, connection_type: ConnectionType, config: dict[str, Any]) -> dict[str, Any]:
        candidate = {str(key): value for key, value in config.items()}
        if connection_type in {"duckdb", "sqlite"}:
            path = str(candidate.get("path") or "").strip()
            if not path:
                raise ValueError("The 'path' field is required for file-based databases.")
            return {"path": path}

        dsn = str(candidate.get("dsn") or "").strip()
        if dsn:
            return {"dsn": dsn}

        host = str(candidate.get("host") or "").strip()
        database = str(candidate.get("database") or "").strip()
        username = str(candidate.get("username") or candidate.get("user") or "").strip()
        password = str(candidate.get("password") or "").strip()
        port = int(candidate.get("port") or (5432 if connection_type == "postgres" else 3306))
        if not host or not database or not username:
            raise ValueError("Fields host, database, and username are required when DSN is not provided.")
        return {
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
        }

    def profile_to_dsn(self, profile: dict[str, Any]) -> str:
        connection_type = str(profile.get("type"))
        config = profile.get("config", {})
        if "dsn" in config and config["dsn"]:
            return str(config["dsn"])

        if connection_type == "postgres":
            return (
                "postgresql+psycopg2://"
                f"{quote_plus(str(config.get('username', '')))}:"
                f"{quote_plus(str(config.get('password', '')))}@"
                f"{config.get('host')}:{config.get('port')}/{config.get('database')}"
            )
        if connection_type == "mysql":
            return (
                "mysql+pymysql://"
                f"{quote_plus(str(config.get('username', '')))}:"
                f"{quote_plus(str(config.get('password', '')))}@"
                f"{config.get('host')}:{config.get('port')}/{config.get('database')}"
            )
        raise ValueError(f"Unsupported DSN conversion for connection type '{connection_type}'.")

    def build_connector(self, profile: dict[str, Any], *, read_only: bool = True) -> DatabaseConnector:
        connection_type = str(profile.get("type"))
        config = profile.get("config", {})
        if connection_type == "duckdb":
            return DuckDBConnector(config["path"], read_only=read_only)
        if connection_type == "sqlite":
            return SQLiteConnector(config["path"])
        if connection_type == "postgres":
            return PostgresConnector(self.profile_to_dsn(profile), read_only=read_only)
        if connection_type == "mysql":
            return MySQLConnector(self.profile_to_dsn(profile), read_only=read_only)
        raise ValueError(f"Unsupported connection type '{connection_type}'.")

    def test_profile(self, *, connection_type: ConnectionType, config: dict[str, Any]) -> dict[str, Any]:
        normalized = self.normalize_profile(connection_type=connection_type, config=config)
        candidate = {"type": connection_type, "config": normalized}
        connector = self.build_connector(candidate, read_only=(connection_type != "duckdb"))
        schema = connector.introspect_schema(sample_limit=1)
        return {
            "status": "ok",
            "dialect": schema.dialect,
            "table_count": len(schema.tables),
            "tables": [table.name for table in schema.tables[:20]],
        }
