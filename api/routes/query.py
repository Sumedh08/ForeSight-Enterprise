from __future__ import annotations

from fastapi import APIRouter, Request

from api.models.schemas import HealthResponse, QueryRequest, QueryResponse
from infra.nim_gateway import nim_gateway


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    orchestrator = http_request.app.state.orchestrator
    return await orchestrator.run_query(request)


@router.get("/health", response_model=HealthResponse)
async def health(http_request: Request) -> HealthResponse:
    services = http_request.app.state.services
    try:
        from chronos import ChronosPipeline  # noqa: F401
        forecast_status = "up"
    except Exception:
        forecast_status = "degraded"
    components = {
        "duckdb": "up" if services.allowed_tables else "degraded",
        "vector_store": services.retrieval_store.health(),
        "nim": await nim_gateway.health(),
        "forecast_model": forecast_status,
    }
    status = "ok" if all(value == "up" for value in components.values()) else "degraded"
    return HealthResponse(status=status, components=components)
