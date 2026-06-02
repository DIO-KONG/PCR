@echo off
setlocal
cd /d "%~dp0"
".env\python.exe" "scripts\run_standard.py" %*
exit /b %ERRORLEVEL%
