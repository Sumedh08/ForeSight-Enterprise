from __future__ import annotations

import os
import shutil

from sqlalchemy import create_engine, text

@router.post("/upload")
async def upload_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx") or file.filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
    
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    temp_path = settings.data_dir / file.filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        table_name = file.filename.split(".")[0].replace("-", "_").replace(" ", "_").lower()
        # Use PostgreSQL instead of DuckDB
        pg_url = "postgresql://admin:adminpassword@localhost:5432/natwest_db"
        engine = create_engine(pg_url)
        
        if file.filename.endswith(".csv"):
            df = pd.read_csv(temp_path)
        else:
            df = pd.read_excel(temp_path)
            
        df.to_sql(table_name, engine, if_exists="replace", index=False)
            
        # Notify services of new data for schema-agnostic discovery
        if hasattr(request.app.state, "services"):
            request.app.state.services.refresh_schema()
            
        # NEW: Trigger MindsDB & Cube Discovery via Airflow or Background Task
        from infra.mindsdb_dynamic_setup import setup_mindsdb_datasource, discover_and_train
        background_tasks.add_task(setup_mindsdb_datasource)
        background_tasks.add_task(discover_and_train)

    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path.exists():
            temp_path.unlink()
    
    return {"status": "ok", "message": f"Data successfully loaded into Postgres table `{table_name}`"}
    
    # Clean up temp file safely after connection closure
    try:
        if temp_path.exists():
            temp_path.unlink()
    except Exception as e:
        logger.warning(f"Could not delete temp file {temp_path}: {e}")
    
    return {"status": "ok", "message": f"Data successfully loaded into table `{table_name}`"}


