@echo off
setlocal
echo ========================================================
echo   ForeSight Enterprise - High-Efficiency Startup
echo   Optimized for 8GB RAM Hardware
echo ========================================================

echo [STAGE 1/5] Starting Core Infrastructure (Postgres, Redis)...
docker compose up -d postgres redis

echo Waiting for Postgres (5432) to accept connections...
:WAIT_POSTGRES
powershell -Command "Test-NetConnection localhost -Port 5432" | find "TcpTestSucceeded : True" > nul
if errorlevel 1 (
    echo ... Postgres still initializing - swapping RAM ...
    timeout /t 5 /nobreak > nul
    goto WAIT_POSTGRES
)
echo Postgres is ONLINE.

echo Waiting for Redis (6379) to accept connections...
:WAIT_REDIS
powershell -Command "Test-NetConnection localhost -Port 6379" | find "TcpTestSucceeded : True" > nul
if errorlevel 1 (
    echo ... Redis still initializing ...
    timeout /t 5 /nobreak > nul
    goto WAIT_REDIS
)
echo Redis is ONLINE.

echo [STAGE 2/5] Initializing Orchestration (Apache Airflow)...
docker compose up -d airflow-init
timeout /t 10 /nobreak > nul
docker compose up -d airflow-webserver airflow-scheduler
echo Giving Airflow time to finish metadata migration...
timeout /t 15 /nobreak > nul

echo [STAGE 3/5] Launching Semantic Layer (Cube.js)...
docker compose up -d cubejs
timeout /t 10 /nobreak > nul

echo [STAGE 4/5] Activating Intelligence Layer (MindsDB)...
docker compose up -d mindsdb
timeout /t 30 /nobreak > nul

echo [STAGE 5/5] Deploying User Interface (Backend, Frontend)...
REM Only rebuild if code changed: run "docker compose build backend" manually
docker compose up -d backend
docker compose up -d frontend
echo ========================================================
echo   Autonomous Deployment Complete.
echo   - Backend:  http://localhost:8000
echo   - Frontend: http://localhost:8501
echo   - Airflow:  http://localhost:8082
echo ========================================================
pause
