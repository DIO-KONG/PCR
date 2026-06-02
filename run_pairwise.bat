@echo off
setlocal
cd /d "%~dp0"
".env\python.exe" "scripts\run_pairwise.py" %*
exit /b %ERRORLEVEL%
