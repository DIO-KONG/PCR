@echo off
setlocal
cd /d "%~dp0"
".env\python.exe" "scripts\run_multi_to_multi.py" %*
exit /b %ERRORLEVEL%
