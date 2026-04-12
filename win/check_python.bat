@echo off
chcp 65001 >nul 2>&1

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found.
    echo Please install Python 3.10 or above.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=1,2" %%a in ('python -c "import sys; print(sys.version_info.major, sys.version_info.minor)"') do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 goto too_old
if %MAJOR% EQU 3 if %MINOR% LSS 10 goto too_old

echo Python %MAJOR%.%MINOR% OK
pause
exit /b 0

:too_old
echo Python %MAJOR%.%MINOR% is too old. Need 3.10 or above.
pause
exit /b 1