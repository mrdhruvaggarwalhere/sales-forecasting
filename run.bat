@echo off
echo ========================================================
echo   Launching Sales Revenue Forecasting System...
echo ========================================================

IF EXIST venv\Scripts\python.exe (
    venv\Scripts\python.exe main.py %*
) ELSE (
    python main.py %*
)
