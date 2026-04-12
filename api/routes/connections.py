from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.models.schemas import (
    ConnectionListResponse,
    ConnectionProfileResponse,
    ConnectionSaveRequest,
    ConnectionTestRequest,
    ConnectionTestResponse,
)


router = APIRouter()


def _to_response(profile: dict) -> ConnectionProfileResponse:
    connection_type = str(profile.get("type", "duckdb"))
    if connection_type not in {"postgres", "mysql", "sqlite", "duckdb"}:
        connection_type = "duckdb"
    return ConnectionProfileResponse(
        id=str(profile.get("id", "")),
        name=str(profile.get("name", "connection")),
        connection_type=connection_type,
        active=bool(profile.get("active")),
        config=profile.get("config", {}),
        created_at=str(profile.get("created_at", "")),
        updated_at=str(profile.get("updated_at", "")),
    )


@router.get("", response_model=ConnectionListResponse)
async def list_connections(http_request: Request) -> ConnectionListResponse:
    manager = http_request.app.state.services.connection_manager
    payload = manager.store.all()
    profiles = manager.list_profiles(redact=True)
    return ConnectionListResponse(
        active_profile_id=payload.get("active_profile_id"),
        profiles=[_to_response(profile) for profile in profiles],
    )


@router.post("/test", response_model=ConnectionTestResponse)
async def test_connection(request: ConnectionTestRequest, http_request: Request) -> ConnectionTestResponse:
    manager = http_request.app.state.services.connection_manager
    try:
        result = manager.test_profile(
            connection_type=request.connection_type,
            config=request.config,
        )
        return ConnectionTestResponse(
            status="ok",
            message="Connection successful.",
            dialect=result.get("dialect"),
            table_count=result.get("table_count"),
            tables=result.get("tables", []),
        )
    except Exception as exc:
        return ConnectionTestResponse(status="failed", message=str(exc))


@router.post("", response_model=ConnectionProfileResponse)
async def save_connection(request: ConnectionSaveRequest, http_request: Request) -> ConnectionProfileResponse:
    manager = http_request.app.state.services.connection_manager
    try:
        profile = manager.save_profile(
            name=request.name,
            connection_type=request.connection_type,
            config=request.config,
            activate=request.activate,
        )
        saved = manager.get_profile(profile["id"])
        listed = manager.list_profiles(redact=True)
        listed_map = {item["id"]: item for item in listed}
        saved["active"] = manager.store.all().get("active_profile_id") == profile["id"]
        saved["config"] = listed_map.get(profile["id"], {}).get("config", {})
        return _to_response(saved)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{profile_id}/activate", response_model=ConnectionProfileResponse)
async def activate_connection(profile_id: str, http_request: Request) -> ConnectionProfileResponse:
    manager = http_request.app.state.services.connection_manager
    try:
        profile = manager.activate(profile_id)
        profile["active"] = True
        listed = manager.list_profiles(redact=True)
        listed_map = {item["id"]: item for item in listed}
        profile["config"] = listed_map.get(profile_id, {}).get("config", {})
        return _to_response(profile)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
