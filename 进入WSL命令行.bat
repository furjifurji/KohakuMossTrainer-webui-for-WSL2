@echo off
chcp 65001 >nul
title KohakuMoss Trainer - WSL命令行终端
cd /d "%~dp0"

echo ============================================================
echo   正在进入 KohakuMossTrainer WSL 终端 (已加载 .venv)...
echo   退出终端请输入 exit 回车即可。
echo ============================================================
echo.

wsl -d Ubuntu -- bash -c "source ~/.bashrc && source .venv/bin/activate && exec bash -i"

pause
