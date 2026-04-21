from __future__ import annotations

import asyncio
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile

from api.models.schemas import TrainingStatusResponse, UploadResponse
from infra.data_ingestion import sanitize_table_name, write_dataframe_to_profile
from infra.settings import settings
from infra.training_store import TrainingJobRecord, build_training_jobs
from infra.vanna_engine import vanna_engine
from offline.schema_refresh import refresh_schema_cache


router = APIRouter()


async def _run_ingestion_pipeline(
    *,
    services: Any,
    warehouse_profile: dict[str, Any],
    table_name: str,
    training_jobs: list[TrainingJobRecord],
) -> None:
    try:
        refresh_schema_cache(profile=warehouse_profile, profile_manager=services.connection_manager)
    except Exception:
        pass

    for job in training_jobs:
        services.training_store.update_job(
            job.series_id,
            state="training",
            progress_pct=35,
            message=f"Registering Lightwood predictor `{job.predictor_name}`.",
        )
        try:
            await services.mindsdb_client.create_time_series_predictor(
                profile=warehouse_profile,
                table_name=table_name,
                date_column=job.date_column,
                value_column=job.value_column,
                predictor_name=job.predictor_name,
            )
        except Exception as exc:
            lowered = str(exc).lower()
            if "already exists" not in lowered and "duplicate" not in lowered:
                services.training_store.update_job(
                    job.series_id,
                    state="failed",
                    progress_pct=100,
                    message=f"Predictor `{job.predictor_name}` failed: {exc}",
                    last_error=str(exc),
                )
                continue

        try:
            state, progress_pct, message = await services.mindsdb_client.get_predictor_state(job.predictor_name)
        except Exception as exc:
            state, progress_pct, message = ("training", 55, f"MindsDB accepted `{job.predictor_name}` and is still training ({exc}).")
        services.training_store.update_job(
            job.series_id,
            state=state,
            progress_pct=progress_pct,
            message=message,
        )

    try:
        services.airflow_client.trigger_dynamic_discovery_sync(
            connection_profile=warehouse_profile,
            tables=[table_name],
        )
    except Exception:
        pass


def _run_ingestion_pipeline_sync(
    *,
    services: Any,
    warehouse_profile: dict[str, Any],
    table_name: str,
    training_jobs: list[TrainingJobRecord],
) -> None:
    asyncio.run(
        _run_ingestion_pipeline(
            services=services,
            warehouse_profile=warehouse_profile,
            table_name=table_name,
            training_jobs=training_jobs,
        )
    )


async def _refresh_jobs(request: Request, jobs: list[TrainingJobRecord]) -> list[TrainingJobRecord]:
    services = request.app.state.services
    refreshed: list[TrainingJobRecord] = []
    for job in jobs:
        if job.state in {"queued", "training"}:
            try:
                state, progress_pct, message = await services.mindsdb_client.get_predictor_state(job.predictor_name)
                updated = services.training_store.update_job(
                    job.series_id,
                    state=state,
                    progress_pct=progress_pct,
                    message=message,
                )
                refreshed.append(updated or job)
                continue
            except Exception:
                pass
        refreshed.append(job)
    refreshed.sort(key=lambda item: item.updated_at, reverse=True)
    return refreshed


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

    services = request.app.state.services
    warnings: list[str] = []
    try:
        import pandas as pd

        if lowered_name.endswith(".csv"):
            frame = pd.read_csv(temp_path)
        else:
            frame = pd.read_excel(temp_path)

        warehouse_profile = services.connection_manager.ensure_enterprise_warehouse_profile(activate=True)
        table_name = sanitize_table_name(safe_filename.rsplit(".", 1)[0])
        normalized_columns = {sanitize_table_name(str(column_name)) for column_name in frame.columns}
        if table_name in normalized_columns:
            table_name = f"{table_name}_data"
            warnings.append(
                f"Uploaded table name was changed to `{table_name}` to avoid SQL ambiguity with a column of the same name."
            )
        table_name = write_dataframe_to_profile(
            profile=warehouse_profile,
            frame=frame,
            table_name=table_name,
            profile_manager=services.connection_manager,
        )

        # Autonomous Training: Teach Vanna the new table schema immediately
        try:
            cols = []
            for col_name, dtype in frame.dtypes.items():
                sql_type = "TEXT"
                if "int" in str(dtype).lower(): sql_type = "INTEGER"
                elif "float" in str(dtype).lower(): sql_type = "FLOAT"
                elif "date" in str(dtype).lower() or "time" in str(dtype).lower(): sql_type = "TIMESTAMP"
                cols.append(f'"{col_name}" {sql_type}')
            
            ddl = f"CREATE TABLE \"{table_name}\" (\n  " + ",\n  ".join(cols) + "\n);"
            vanna_engine.train_on_ddl(ddl)
            warnings.append(f"Vanna AI successfully trained on `{table_name}` schema.")
        except Exception as ve:
            print(f"Vanna AI auto-learning failed (non-blocking): {ve}")
            traceback.print_exc()
            warnings.append(f"Vanna AI auto-learning failed: {ve}")

        ingestion_id = str(uuid.uuid4())
        training_jobs = build_training_jobs(
            frame=frame,
            table_name=table_name,
            ingestion_id=ingestion_id,
            warehouse_profile=warehouse_profile,
        )
        if training_jobs:
            services.training_store.save_jobs(training_jobs)
            background_tasks.add_task(
                _run_ingestion_pipeline_sync,
                services=services,
                warehouse_profile=warehouse_profile,
                table_name=table_name,
                training_jobs=training_jobs,
            )
            warnings.append("Lightwood predictor training was queued for the uploaded metrics.")
        else:
            warnings.append("No forecastable date/metric pair was detected, so MindsDB training was not queued.")

        return UploadResponse(
            status="ok",
            message=f"Data uploaded successfully to `{table_name}` in the canonical enterprise warehouse.",
            table_name=table_name,
            ingestion_id=ingestion_id,
            warehouse_profile=str(warehouse_profile.get("name", "enterprise-warehouse")),
            training_jobs=[job.as_summary() for job in training_jobs],
            warnings=warnings,
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"CRITICAL: Upload ingestion failed: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.get("/training", response_model=TrainingStatusResponse)
async def training_status(
    request: Request,
    ingestion_id: str | None = Query(default=None),
    series_id: str | None = Query(default=None),
) -> TrainingStatusResponse:
    services = request.app.state.services
    if series_id:
        selected = [job for job in services.training_store.list_jobs() if job.series_id == series_id]
    elif ingestion_id:
        selected = services.training_store.get_jobs_by_ingestion(ingestion_id)
    else:
        selected = services.training_store.list_jobs()[:12]

    refreshed = await _refresh_jobs(request, selected)
    return TrainingStatusResponse(
        status="ok",
        ingestion_id=ingestion_id,
        series_id=series_id,
        jobs=[job.as_summary() for job in refreshed],
    )
