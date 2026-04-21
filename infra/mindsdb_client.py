from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from infra.connection_profiles import ConnectionProfileManager
from infra.settings import settings


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower()


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _dsn_to_parameters(dsn: str, *, connection_type: str, fallback_config: dict[str, Any]) -> dict[str, Any]:
    parsed = urlparse(dsn)
    username = parsed.username or fallback_config.get("username") or fallback_config.get("user")
    password = parsed.password or fallback_config.get("password")
    database = (parsed.path or "").lstrip("/") or fallback_config.get("database")
    host = parsed.hostname or fallback_config.get("host")
    port = parsed.port or fallback_config.get("port") or (5432 if connection_type == "postgres" else 3306)

    parameters = {
        "host": str(host or ""),
        "port": int(port),
        "user": unquote(str(username or "")),
        "password": unquote(str(password or "")),
        "database": str(database or ""),
    }
    if connection_type == "mysql":
        parameters["type"] = "mysql"
    return parameters


class MindsDBClient:
    def __init__(self, profile_manager: ConnectionProfileManager) -> None:
        self.base_url = settings.mindsdb_api_url.rstrip("/")
        self.health_url = self.base_url.rsplit("/api/sql", 1)[0] + "/api/status"
        self.profile_manager = profile_manager

    async def health(self) -> str:
        try:
            await self.query("SELECT 1 AS ok")
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.get(self.health_url)
            return "up"
        except Exception:
            return "degraded"

    async def query(self, sql: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.base_url, json={"query": sql})
            response.raise_for_status()
            payload = response.json()
        if isinstance(payload, dict) and str(payload.get("type", "")).lower() == "error":
            raise RuntimeError(str(payload.get("error_message") or "MindsDB returned an error payload."))
        return payload

    async def ensure_datasource(self, profile: dict[str, Any]) -> str:
        connection_type = str(profile.get("type"))
        datasource_name = f"profile_{_safe_identifier(str(profile.get('id', 'default')))}"
        if connection_type not in {"postgres", "mysql"}:
            raise RuntimeError("MindsDB datasource setup currently supports Postgres and MySQL profiles.")

        config = profile.get("config", {})
        dsn = self.profile_manager.profile_to_dsn(profile, purpose="mindsdb")
        parameters = _dsn_to_parameters(dsn, connection_type=connection_type, fallback_config=config)
        engine_name = "postgres" if connection_type == "postgres" else "mysql"
        sql = (
            f"CREATE DATABASE IF NOT EXISTS {datasource_name} "
            f"WITH ENGINE = '{engine_name}', "
            f"PARAMETERS = {json.dumps(parameters)};"
        )
        await self.query(sql)
        return datasource_name

    async def create_time_series_predictor(
        self,
        *,
        profile: dict[str, Any],
        table_name: str,
        date_column: str,
        value_column: str,
        predictor_name: str,
        horizon: int = 26,
        window: int = 20,
    ) -> str:
        datasource = await self.ensure_datasource(profile)
        predictor = _safe_identifier(predictor_name)
        table = _quote_identifier(table_name)
        ordered_date = _quote_identifier(date_column)
        metric = _quote_identifier(value_column)
        sql = f"""
        CREATE PREDICTOR mindsdb.{predictor}
        FROM {datasource}
          (
            SELECT *
            FROM {table}
            WHERE {ordered_date} IS NOT NULL AND {metric} IS NOT NULL
          )
        PREDICT {metric}
        ORDER BY {ordered_date}
        WINDOW {int(window)}
        HORIZON {int(horizon)}
        USING ENGINE='lightwood';
        """
        try:
            await self.query(sql)
        except RuntimeError as exc:
            lowered = str(exc).lower()
            if "there is no engine 'lightwood'" not in lowered and 'no engine "lightwood"' not in lowered:
                raise
            fallback_sql = f"""
            CREATE PREDICTOR mindsdb.{predictor}
            FROM {datasource}
              (
                SELECT *
                FROM {table}
                WHERE {ordered_date} IS NOT NULL AND {metric} IS NOT NULL
              )
            PREDICT {metric}
            ORDER BY {ordered_date}
            WINDOW {int(window)}
            HORIZON {int(horizon)};
            """
            await self.query(fallback_sql)
        return predictor

    async def list_predictors(self) -> list[str]:
        payload = await self.query("SHOW PREDICTORS")
        rows = self._payload_to_rows(payload)
        names: list[str] = []
        for row in rows:
            name = str(row.get("NAME", row.get("name", ""))).strip()
            if name:
                names.append(name)
        return names

    async def get_predictor_record(self, predictor_name: str) -> dict[str, Any] | None:
        predictor = _safe_identifier(predictor_name)
        try:
            payload = await self.query("SHOW PREDICTORS")
        except Exception:
            return None
        rows = self._payload_to_rows(payload)
        for row in rows:
            name = str(row.get("NAME", row.get("name", ""))).strip().lower()
            if name == predictor:
                return row
        return None

    async def get_predictor_state(self, predictor_name: str) -> tuple[str, int, str]:
        predictor = _safe_identifier(predictor_name)
        record = await self.get_predictor_record(predictor)
        if record:
            state, progress, message = self._normalize_state(record)
            if state != "queued":
                return state, progress, message

        try:
            await self.run_predictor(predictor=predictor, row_cap=3)
            return "ready", 100, f"Predictor `{predictor}` is ready."
        except Exception as exc:
            message = str(exc).lower()
            if any(token in message for token in ("still training", "in training", "training", "building", "preparing", "initializing")):
                return "training", 72, f"Predictor `{predictor}` is still training."
            if any(token in message for token in ("does not exist", "not found", "unknown table", "can't find")):
                return "queued", 20, f"Predictor `{predictor}` is being registered in MindsDB."
            return "failed", 100, f"Predictor `{predictor}` failed: {exc}"

    async def resolve_predictor(self, *, question: str, table_names: list[str]) -> str | None:
        predictors = await self.list_predictors()
        if not predictors:
            return None

        lowered = question.lower()
        table_hits = [table for table in table_names if table.lower() in lowered]
        if table_hits:
            preferred = table_hits[0].lower()
            for predictor in predictors:
                if preferred in predictor.lower():
                    return predictor

        question_tokens = set(re.findall(r"[a-zA-Z]+|\d+", lowered))
        best: tuple[int, str] | None = None
        for predictor in predictors:
            tokens = set(re.findall(r"[a-zA-Z]+|\d+", predictor.lower()))
            score = len(question_tokens & tokens)
            if best is None or score > best[0]:
                best = (score, predictor)
        return best[1] if best else predictors[0]

    async def run_predictor(self, *, predictor: str, row_cap: int) -> list[dict[str, Any]]:
        payload = await self.query(f"SELECT * FROM mindsdb.{_safe_identifier(predictor)} LIMIT {int(row_cap)}")
        mapped = self._payload_to_rows(payload)
        if mapped:
            return mapped
        raise RuntimeError(f"MindsDB predictor `{_safe_identifier(predictor)}` returned no rows.")

    @staticmethod
    def _normalize_state(record: dict[str, Any]) -> tuple[str, int, str]:
        text_chunks: list[str] = []
        for key, value in record.items():
            if value is None:
                continue
            lowered_key = str(key).lower()
            if lowered_key in {
                "status",
                "training_status",
                "phase",
                "error",
                "current_phase",
                "current_status",
                "update_status",
                "current_training_phase",
                "training_phase_name",
            }:
                text_chunks.append(str(value).lower())
        combined = " ".join(text_chunks)
        if any(token in combined for token in ("error", "failed", "failure")):
            return "failed", 100, "MindsDB reported a training failure."
        if any(token in combined for token in ("complete", "completed", "ready", "trained", "success")):
            return "ready", 100, "MindsDB training completed."
        if any(token in combined for token in ("generating", "preparing", "starting", "queued", "pending")):
            return "training", 35, "MindsDB is preparing the predictor."
        if any(token in combined for token in ("training", "learning", "fitting", "validating", "optimizing")):
            return "training", 72, "MindsDB is training the predictor."
        return "queued", 20, "Predictor registration is still propagating."

    @staticmethod
    def _payload_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        if "data" in payload and isinstance(payload["data"], list):
            data = payload["data"]
            if data and isinstance(data[0], dict):
                return data
            columns = payload.get("column_names") or payload.get("columns") or []
            rows: list[dict[str, Any]] = []
            for row in data:
                if isinstance(row, dict):
                    rows.append(row)
                elif isinstance(row, (list, tuple)):
                    rows.append({str(columns[i]): row[i] for i in range(min(len(columns), len(row)))})
            return rows
        if "result" in payload and isinstance(payload["result"], list):
            return [row for row in payload["result"] if isinstance(row, dict)]
        return []
