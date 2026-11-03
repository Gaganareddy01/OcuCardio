@echo off
set "PATH=C:\Users\gagan\.gemini\antigravity\scratch\mingit\cmd;C:\Users\gagan\.gemini\antigravity\scratch\mingit\mingw64\bin;%PATH%"
cd /d "C:\Users\gagan\.gemini\antigravity\scratch\retinal_cvd_risk"

echo ========================================================
echo   Pushing OcuCardio Project to GitHub...
echo ========================================================
echo.
echo Target: https://github.com/Gaganareddy01/OcuCardio
echo.
echo (If a browser window pops up, click 'Authorize' to log in to GitHub)
echo.

git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   SUCCESS! OcuCardio is now live on GitHub:
    echo   https://github.com/Gaganareddy01/OcuCardio
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   Push failed. Please ensure:
    echo   1. You created the repository named 'OcuCardio' on https://github.com/new
    echo   2. You are logged into GitHub in your browser
    echo ========================================================
)

echo.
pause

