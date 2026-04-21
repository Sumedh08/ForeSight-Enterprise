@echo off
set PYTHONPATH=%CD%
echo Starting ForeSight Analytics Frontend (Local)...
python -m streamlit run frontend/app.py --server.port 8501
pause
