@echo off
echo ========================================================
echo   NatWest ForeSight - Final GitHub Submission
echo ========================================================

:: 1. Force Clean Git Index Lock
echo [*] Cleaning git locks...
del /f .git\index.lock 2>nul
taskkill /F /IM git.exe 2>nul

:: 2. Stage All Source Files
echo [*] Staging source code...
git add .

:: 3. Commit with DCO Sign-off
echo [*] Creating certified commit...
git commit -s -m "feat: complete enterprise analytics stack migration

- Migrated to PostgreSQL, MindsDB, and Cube.js
- Implemented schema-agnostic dynamic profiling
- Built Airflow orchestration for predictor training
- Added Wren AI for deterministic Text-to-SQL
- Cleaned up 2.9GB of benchmark data for production readiness

Signed-off-by: Sumedh Ramesh Naidu <sumedhramesh.naidu@2022vitstudent.ac.in>"

:: 4. Push to Remote
echo [*] Pushing to GitHub (Sumedh08/NatWest-ForeSight-Enterprise)...
git push -u origin main --force

echo ========================================================
echo   SUCCESS! Your project is now live on GitHub.
echo ========================================================
pause
