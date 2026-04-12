from __future__ import annotations

import re
from typing import Any

import httpx

from infra.connection_profiles import ConnectionProfileManager
from infra.settings import settings


class MindsDBClient:
    def __init__(self, profile_manager: ConnectionProfileManager) -> None:
        self.base_url = settings.mindsdb_api_url.rstrip("/")
        self.profile_manager = profile_manager

    async def health(self) -> str:
        try:
            await self.query("SELECT 1 AS ok")
            return "up"
        except Exception:
            return "degraded"

    async def query(self, sql: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.base_url, json={"query": sql})
            response.raise_for_status()
            return response.json()

    async def ensure_datasource(self, profile: dict[str, Any]) -> str:
        connection_type = str(profile.get("type"))
        config = profile.get("config", {})
        datasource_name = f"profile_{profile['id'].replace('-', '_')}"
        if connection_type not in {"postgres", "mysql"}:
            raise RuntimeError("MindsDB datasource setup currently supports Postgres and MySQL profiles.")

        dsn = self.profile_manager.profile_to_dsn(profile)
        dsn = dsn.replace("\\", "\\\\").replace('"', '\\"')
        engine_name = "postgres" if connection_type == "postgres" else "mysql"
        sql = (
            f"CREATE DATABASE IF NOT EXISTS {datasource_name} "
            f"WITH ENGINE = '{engine_name}', "
            f"PARAMETERS = {{\"connection\": \"{dsn}\"}};"
        )
        await self.query(sql)
        return datasource_name

    async def list_predictors(self) -> list[str]:
        payload = await self.query("SELECT name FROM mindsdb.predictors")
        rows = self._payload_to_rows(payload)
        names: list[str] = []
        for row in rows:
            name = str(row.get("name", "")).strip()
            if name:
                names.append(name)
        return names

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

        question_tokens = set(re.findall(r"[a-zA-Z0-9_]+", lowered))
        best: tuple[int, str] | None = None
        for predictor in predictors:
            tokens = set(re.findall(r"[a-zA-Z0-9_]+", predictor.lower()))
            score = len(question_tokens & tokens)
            if best is None or score > best[0]:
                best = (score, predictor)
        return best[1] if best else predictors[0]

    async def run_predictor(self, *, predictor: str, row_cap: int) -> list[dict[str, Any]]:
        payload = await self.query(f"SELECT * FROM mindsdb.{predictor} LIMIT {row_cap}")
        return self._payload_to_rows(payload)

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
