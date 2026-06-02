@echo off
setlocal
cd /d "%~dp0"
".env\python.exe" "scripts\check_env.py" %*
exit /b %ERRORLEVEL%
