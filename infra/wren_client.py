"""
Wren AI Client
Generates SQL through Wren AI and exposes lightweight health checks.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from infra.settings import settings


def _extract_sql(text: str) -> str:
    candidate = text.strip()
    if "```sql" in candidate.lower():
        start = candidate.lower().find("```sql")
        candidate = candidate[start + 6 :]
        candidate = candidate.split("```", 1)[0]
    elif "```" in candidate:
        candidate = candidate.split("```", 1)[1].split("```", 1)[0]
    candidate = candidate.strip().strip("`")
    if candidate.upper().startswith(("SELECT", "WITH")):
        return candidate.rstrip(";")

    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.upper().startswith(("SELECT", "WITH")):
            return line.rstrip(";")
    return candidate.rstrip(";")


class WrenClient:
    def __init__(self) -> None:
        self.engine_url = settings.wren_engine_url.rstrip("/")
        self.ai_service_url = settings.wren_ai_service_url.rstrip("/")
        self.timeout_s = 90.0

    async def health(self) -> dict[str, str]:
        ai_status = "degraded"
        engine_status = "degraded"
        async with httpx.AsyncClient(timeout=8.0) as client:
            try:
                ai_response = await client.get(f"{self.ai_service_url}/health")
                if ai_response.status_code < 500:
                    ai_status = "up"
            except Exception:
                ai_status = "degraded"
            try:
                engine_response = await client.get(f"{self.engine_url}/health")
                if engine_response.status_code < 500:
                    engine_status = "up"
            except Exception:
                engine_status = "degraded"
        return {"wren_ai": ai_status, "wren_engine": engine_status}

    async def generate_sql(self, question: str, *, schema_context: str | None = None) -> str:
        system_context = (
            "You are a production text-to-SQL system. "
            "Return only executable SQL (no markdown, no explanation). "
            "Use only tables/columns present in schema context."
        )
        if schema_context:
            system_context += f"\n\nSchema context:\n{schema_context}"

        payload = {
            "model": "wren-ai",
            "messages": [
                {"role": "system", "content": system_context},
                {"role": "user", "content": question},
            ],
            "temperature": 0.0,
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(f"{self.ai_service_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()

        content = ""
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                content = str(message.get("content", "")).strip()
            if not content and isinstance(body.get("sql"), str):
                content = str(body.get("sql", "")).strip()
        if not content:
            raise RuntimeError("Wren did not return SQL content.")

        sql = _extract_sql(content)
        if not sql or not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
            raise RuntimeError("Wren response did not contain a valid read-only SQL query.")
        return sql

    async def execute_semantic_query(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(f"{self.engine_url}/v1/query", json=query)
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data", [])
        return data if isinstance(data, list) else []


wren_client = WrenClient()
