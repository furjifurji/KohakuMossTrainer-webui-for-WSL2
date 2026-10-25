@echo off
chcp 65001 >nul
title KohakuMoss Trainer Studio (WSL2 WebUI)
cd /d "%~dp0"

echo ============================================================
echo   Starting KohakuMoss Trainer WebUI (WSL2)
echo   URL: http://localhost:7878
echo   Close this window to stop the service.
echo ============================================================
echo.

rem open the browser after 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:7878"

wsl -d Ubuntu -- bash -c ".venv/bin/python webui.py 7878"

pause
