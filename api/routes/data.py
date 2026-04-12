from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from api.models.schemas import UploadResponse
from infra.data_ingestion import sanitize_table_name, write_dataframe_to_profile
from infra.settings import settings


router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")
    lowered_name = file.filename.lower()
    if not (lowered_name.endswith(".csv") or lowered_name.endswith(".xlsx") or lowered_name.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported.")

    settings.ensure_directories()
    safe_filename = Path(file.filename).name
    temp_path = settings.data_dir / safe_filename
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    profile_manager = request.app.state.services.connection_manager
    airflow_client = request.app.state.services.airflow_client
    warnings: list[str] = []
    try:
        if lowered_name.endswith(".csv"):
            frame = pd.read_csv(temp_path)
        else:
            frame = pd.read_excel(temp_path)

        active_profile = profile_manager.get_active_profile()
        table_name = sanitize_table_name(safe_filename.rsplit(".", 1)[0])
        table_name = write_dataframe_to_profile(
            profile=active_profile,
            frame=frame,
            table_name=table_name,
            profile_manager=profile_manager,
        )

        def trigger_sync() -> None:
            try:
                airflow_client.trigger_dynamic_discovery_sync(
                    connection_profile=active_profile,
                    tables=[table_name],
                )
            except Exception:
                # Upload success should not fail because offline orchestration is unavailable.
                pass

        background_tasks.add_task(trigger_sync)
        warnings.append("Offline model sync was queued in Airflow (if available).")
        return UploadResponse(
            status="ok",
            message=f"Data uploaded successfully to `{table_name}` using the active profile.",
            table_name=table_name,
            warnings=warnings,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()
