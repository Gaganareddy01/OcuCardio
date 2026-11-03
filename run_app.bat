@echo off
title OcuCardio - AI-Powered Retinal Image Analysis for Cardiovascular Risk Assessment
echo =====================================================================
echo    OCUCARDIO: AI-POWERED RETINAL ANALYSIS & CVD RISK PLATFORM
echo =====================================================================
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo [*] Initializing OcuCardio Backend (PyTorch Multi-Task ResUNet)...
echo [*] Aesthetic: Modern Healthcare AI Dashboard
echo [*] Starting server...

start /b "" "%PYTHON_BIN%" -u backend\server.py 8080 >nul 2>&1

:: Wait briefly for model and server socket initialization
timeout /t 4 /nobreak >nul

set "APP_PORT=8080"
if exist ".active_port" (
    set /p APP_PORT=<.active_port
)

echo [*] Launching OcuCardio in default browser: http://localhost:%APP_PORT%
start http://localhost:%APP_PORT%

echo =====================================================================
echo [*] OcuCardio is running live at: http://localhost:%APP_PORT%
echo [*] Modules: Scan Inspector, Dataset Hub, Training, Benchmarks
echo =====================================================================
pause
