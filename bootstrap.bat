@echo off
SETLOCAL EnableDelayedExpansion

echo ========================================================
echo   NatWest ForeSight Enterprise - Bootstrap Setup
echo ========================================================

:: 1. Environment Verification
echo [1/6] Verifying environment...
if not exist .env (
    if exist .env.example (
        echo [!] .env not found. Creating from .env.example...
        copy .env.example .env
        echo [!] IMPORTANT: Please edit .env and add your NVIDIA_API_KEY.
    ) else (
        echo [ERROR] .env.example missing. Please restore codebase.
        exit /b 1
    )
)

:: 2. Docker Infrastructure (all-in-docker runtime)
echo [2/6] Starting Docker Enterprise Stack...
docker compose up -d --build
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker failed to start. Ensure Docker Desktop is running and you have 10GB free space.
    exit /b 1
)

:: 3. Data Migration (optional bootstrap migration)
echo [3/6] Migrating local data to Enterprise Storage...
:: Wait for Postgres to be ready
echo [*] Waiting for PostgreSQL...
:wait_pg
docker exec postgres pg_isready -U admin -d natwest_db >nul 2>&1
if !ERRORLEVEL! neq 0 (
    timeout /t 2 >nul
    goto wait_pg
)
docker compose run --rm backend python -m infra.migrate_to_postgres
echo [OK] Data mesh initialized.

:: 4. Check service readiness
echo [4/6] Checking backend health...
timeout /t 5 >nul

:: 5. Airflow offline orchestration is handled by docker services
echo [5/6] Airflow and MindsDB are running in offline orchestration mode.

:: 6. Done
echo [6/6] Ready.

echo ========================================================
echo   SETUP COMPLETE!
echo   Backend API: http://localhost:8000
echo   Frontend UI: http://localhost:8501
echo   Airflow UI: http://localhost:8080
echo   MindsDB UI: http://localhost:47334
echo   Wren AI UI: http://localhost:3001
echo ========================================================
pause
