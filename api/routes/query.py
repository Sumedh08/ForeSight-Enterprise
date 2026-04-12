from __future__ import annotations

from fastapi import APIRouter, Request

from api.models.schemas import HealthResponse, QueryRequest, QueryResponse


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    orchestrator = http_request.app.state.orchestrator
    return await orchestrator.run_query(request)


@router.get("/health", response_model=HealthResponse)
async def health(http_request: Request) -> HealthResponse:
    services = http_request.app.state.services
    components = await services.health_components()
    status = "ok" if all(value == "up" for value in components.values()) else "degraded"
    try:
        active_profile = services.connection_manager.get_active_profile()
    except Exception:
        active_profile = {"name": None, "type": None}
    return HealthResponse(
        status=status,
        components=components,
        active_connection=active_profile.get("name"),
        active_connection_type=active_profile.get("type"),
    )
