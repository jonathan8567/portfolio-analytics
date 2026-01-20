@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo Portfolio Analytics - Update Workflow
echo ===================================================

echo Cleaning up old reports...
if exist "portfolio_report.html" (
    del "portfolio_report.html"
    echo Deleted old portfolio_report.html
)

echo Checking environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv
    pause
    exit /b 1
)

echo Running Analysis Engine...
".venv\Scripts\python.exe" main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Analysis failed with error code %errorlevel%.
    echo Please check the output above.
) else (
    echo.
    echo [SUCCESS] Report generated.
    
    REM Find the newest timestamped file
    for /f "delims=" %%x in ('dir portfolio_report_*.html /b /o-d') do (
        echo Opening newest report: %%x
        start %%x
        goto :done
    )
)

:done
pause
