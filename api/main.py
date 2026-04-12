from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.logging import configure_logging, logging_middleware
from api.routes.data import router as data_router
from api.routes.query import router as query_router
from infra.settings import settings
from offline.schema_refresh import refresh_schema_cache
from orchestrator.enterprise_orchestrator import EnterpriseOrchestrator

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.ensure_directories()
    # Dynamic setup: No longer need disk-based schema refresh as Cube.js handles it
    app.state.services = None # AppServices replaced by Enterprise stack
    app.state.orchestrator = EnterpriseOrchestrator(services=None)
    yield


app = FastAPI(title="NatWest Analytics Assistant", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(logging_middleware)
app.include_router(data_router, prefix="/api/data")
app.include_router(query_router)
