@echo off
cd /d "%~dp0"
setlocal EnableDelayedExpansion

REM ================================================
REM   Ramaria Release Build Script
REM ================================================

REM Find Python (venv first)
set "PYTHON="
if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo [1/5] Cleaning old builds...
if exist "dist\Ramaria" rmdir /s /q "dist\Ramaria"
if exist "build\Ramaria" rmdir /s /q "build\Ramaria"
if exist "Ramaria*.zip" del /q "Ramaria*.zip"

echo [2/5] Building (2-5 min)...
%PYTHON% -m PyInstaller ramaria.spec --noconfirm
if errorlevel 1 goto :error

echo [3/5] Copying config templates...
if not exist "dist\Ramaria\config" mkdir "dist\Ramaria\config"
copy /Y "config\persona.toml.example" "dist\Ramaria\config\persona.toml" >nul 2>&1
if exist ".env.example" copy /Y ".env.example" "dist\Ramaria\.env" >nul 2>&1

echo [4/5] Collecting VC++ runtime DLLs...
%PYTHON% scripts\collect_dlls.py >nul 2>&1

REM echo [5/5] Creating zip archive...
REM powershell -Command "Compress-Archive -Path 'dist\Ramaria' -DestinationPath 'dist\Ramaria-win-x64.zip' -Force"
REM echo.
echo [5/5] Skip compression (output: dist\Ramaria\)

echo.
echo ================================================
echo   Done!
echo ================================================
echo.
echo Output: dist\Ramaria\
echo.
pause
exit /b 0

:error
echo.
echo [ERROR] Build failed
pause
exit /b 1
