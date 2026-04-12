from __future__ import annotations

from typing import Any

import httpx

from infra.settings import settings


class AirflowClient:
    def __init__(self) -> None:
        self.base_url = settings.airflow_base_url.rstrip("/")
        self.username = settings.airflow_username
        self.password = settings.airflow_password

    async def health(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
            return "up"
        except Exception:
            return "degraded"

    async def trigger_dynamic_discovery(
        self,
        *,
        connection_profile: dict[str, Any],
        tables: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "dag_run_id": None,
            "conf": {
                "connection_profile": connection_profile,
                "tables": tables or [],
            },
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/dags/dynamic_data_discovery/dagRuns",
                auth=(self.username, self.password),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def trigger_dynamic_discovery_sync(
        self,
        *,
        connection_profile: dict[str, Any],
        tables: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "dag_run_id": None,
            "conf": {
                "connection_profile": connection_profile,
                "tables": tables or [],
            },
        }
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                f"{self.base_url}/api/v1/dags/dynamic_data_discovery/dagRuns",
                auth=(self.username, self.password),
                json=payload,
            )
            response.raise_for_status()
            return response.json()


airflow_client = AirflowClient()
