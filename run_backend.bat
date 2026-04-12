@echo off
echo Starting NatWest Analytics Backend...
.venv311\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
