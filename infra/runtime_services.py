import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from infra.airflow_client import AirflowClient
from infra.connection_profiles import ConnectionProfileManager
from infra.mindsdb_client import MindsDBClient
from infra.nim_gateway import nim_gateway
from infra.training_store import TrainingStore
from infra.vanna_engine import VannaSemanticCache


@dataclass(slots=True)
class RuntimeServices:
    connection_manager: ConnectionProfileManager
    airflow_client: AirflowClient
    mindsdb_client: MindsDBClient
    training_store: TrainingStore
    vanna_cache: VannaSemanticCache
    _cache: dict[str, Any] = field(default_factory=dict, init=False)

    async def health_components(self) -> dict[str, str]:
        now = time.time()
        if "data" in self._cache and (now - self._cache.get("ts", 0)) < 25:
            return self._cache["data"]

        # Run all health checks in parallel to save time on slow hardware
        tasks = [
            self._check_db(),
            self.mindsdb_client.health(),
            self.airflow_client.health(),
            nim_gateway.health(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        components: dict[str, str] = {}
        # 1. DB
        components["database"] = results[0] if isinstance(results[0], str) else "degraded"
        # 2. MindsDB
        components["mindsdb"] = results[1] if isinstance(results[1], str) else "degraded"
        # 3. Airflow
        components["airflow"] = results[2] if isinstance(results[2], str) else "degraded"
        # 4. NIM
        components["nim"] = results[3] if isinstance(results[3], str) else "degraded"

        self._cache = {"data": components, "ts": now}
        return components

    async def _check_db(self) -> str:
        try:
            active = self.connection_manager.get_active_profile()
            connector = self.connection_manager.build_connector(active, read_only=True)
            schema = connector.introspect_schema(sample_limit=1)
            return "up" if schema.tables is not None else "degraded"
        except Exception:
            return "degraded"
