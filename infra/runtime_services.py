from __future__ import annotations

from dataclasses import dataclass

from infra.airflow_client import AirflowClient
from infra.connection_profiles import ConnectionProfileManager
from infra.mindsdb_client import MindsDBClient
from infra.nim_gateway import nim_gateway
from infra.wren_client import WrenClient


@dataclass(slots=True)
class RuntimeServices:
    connection_manager: ConnectionProfileManager
    airflow_client: AirflowClient
    wren_client: WrenClient
    mindsdb_client: MindsDBClient

    async def health_components(self) -> dict[str, str]:
        components: dict[str, str] = {}
        # Active DB
        try:
            active = self.connection_manager.get_active_profile()
            connector = self.connection_manager.build_connector(active, read_only=True)
            schema = connector.introspect_schema(sample_limit=1)
            components["database"] = "up" if schema.tables is not None else "degraded"
        except Exception:
            components["database"] = "degraded"

        components.update(await self.wren_client.health())
        components["mindsdb"] = await self.mindsdb_client.health()
        components["airflow"] = await self.airflow_client.health()
        components["nim"] = await nim_gateway.health()
        return components
