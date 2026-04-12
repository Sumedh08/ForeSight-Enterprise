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

:: 2. Python Virtual Environment
echo [2/6] Setting up Python virtual environment...
if not exist .venv (
    python -m venv .venv
    echo [OK] Created .venv
)
call .venv\Scripts\activate

:: 3. Dependencies
echo [3/6] Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] pip install failed.
    exit /b 1
)

:: 4. Docker Infrastructure
echo [4/6] Starting Docker Enterprise Stack...
docker compose up -d
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker failed to start. Ensure Docker Desktop is running and you have 10GB free space.
    exit /b 1
)

:: 5. Data Migration (Internal Automation)
echo [5/6] Migrating local data to Enterprise Storage...
:: Wait for Postgres to be ready
echo [*] Waiting for PostgreSQL...
:wait_pg
docker exec postgres pg_isready -U admin -d natwest_db >nul 2>&1
if !ERRORLEVEL! neq 0 (
    timeout /t 2 >nul
    goto wait_pg
)
python -m infra.migrate_to_postgres
echo [OK] Data mesh initialized.

:: 6. Launch Services
echo [6/6] Launching Analytical Interface...
start "" cmd /c "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
start "" cmd /c "streamlit run frontend/app.py"

echo ========================================================
echo   SETUP COMPLETE!
echo   Backend: http://localhost:8000
echo   Frontend: http://localhost:8501 (Streamlit)
echo   MindsDB UI: http://localhost:47334
echo   Wren AI UI: http://localhost:3001
echo ========================================================
pause
