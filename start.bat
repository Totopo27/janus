@echo off
title Janus
cd /d "%~dp0"
echo ========================================================
echo   [*] Iniciando Janus
echo   Directorio: %CD%
echo ========================================================
python run.py
pause
