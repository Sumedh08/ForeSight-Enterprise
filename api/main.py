from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.logging import configure_logging, logging_middleware
from api.routes.connections import router as connections_router
from api.routes.data import router as data_router
from api.routes.query import router as query_router
from infra.airflow_client import AirflowClient
from infra.connection_profiles import ConnectionProfileManager
from infra.mindsdb_client import MindsDBClient
from infra.runtime_services import RuntimeServices
from infra.settings import settings
from infra.wren_client import WrenClient
from orchestrator.enterprise_orchestrator import EnterpriseOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.ensure_directories()

    connection_manager = ConnectionProfileManager()
    airflow_client = AirflowClient()
    wren_client = WrenClient()
    mindsdb_client = MindsDBClient(connection_manager)
    services = RuntimeServices(
        connection_manager=connection_manager,
        airflow_client=airflow_client,
        wren_client=wren_client,
        mindsdb_client=mindsdb_client,
    )
    app.state.services = services
    app.state.orchestrator = EnterpriseOrchestrator(services=services)
    yield


app = FastAPI(title="NatWest Analytics Assistant", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(logging_middleware)
app.include_router(data_router, prefix="/api/data")
app.include_router(connections_router, prefix="/api/connections")
app.include_router(query_router)
