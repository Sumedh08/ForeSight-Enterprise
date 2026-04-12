@echo off
echo Starting NatWest Analytics Frontend...
.venv311\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
pause
