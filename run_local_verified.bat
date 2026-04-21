@echo off
set PYTHONPATH=%CD%
echo Starting ForeSight Autonomous Backend (Local Verified Mode)...
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
