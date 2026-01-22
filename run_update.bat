@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo Portfolio Analytics - Update Workflow
echo ===================================================

echo Cleaning up old reports for clean run...
if exist "index.html" (
    del "index.html"
    echo Deleted old index.html
)

echo Checking environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv
    pause
    exit /b 1
)

echo Running Analysis Engine...
".venv\Scripts\python.exe" main.py --slippage 5 --commission 10

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Analysis failed with error code %errorlevel%.
    echo Please check the output above.
) else (
    echo.
    echo [SUCCESS] Report generated: index.html
    echo Opening report...
    echo [NOTE] If the report looks old, press CTRL + F5 in the browser to refresh!
    start index.html
)

:done
pause
