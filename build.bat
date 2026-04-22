@echo off
cd /d "%~dp0"

REM Find Python (venv first, then system)
set PYTHON=
if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
    echo [Info] Using venv Python
) else (
    set PYTHON=python
    echo [Info] Using system Python
)

echo ================================================
echo   Ramaria Build Script
echo ================================================
echo.
echo [Step 1] Check Python...
%PYTHON% --version
if errorlevel 1 goto error_python

echo.
echo [Step 2] Check PyInstaller...
%PYTHON% -m PyInstaller --version
if errorlevel 1 goto error_pyinstaller

echo.
echo [Step 3] Clean old builds...
if exist "dist\Ramaria" rmdir /s /q "dist\Ramaria"
if exist "build\Ramaria" rmdir /s /q "build\Ramaria"

echo.
echo [Step 4] Building... (This may take 2-5 minutes)
%PYTHON% -m PyInstaller ramaria.spec --noconfirm
if errorlevel 1 goto error_build

echo.
echo [Step 5] Copy config files...
if exist "dist\Ramaria" (
    if not exist "dist\Ramaria\config" mkdir "dist\Ramaria\config"
    xcopy /Y /E "config\*" "dist\Ramaria\config\" >nul 2>&1
    echo [OK] Config files copied
)

echo.
echo [Step 5b] Copy .env file...
if exist "dist\Ramaria" (
    if exist ".env" (
        copy /Y ".env" "dist\Ramaria\.env" >nul 2>&1
        echo [OK] .env file copied
    ) else (
        echo [Warning] .env not found, copying .env.example instead
        copy /Y ".env.example" "dist\Ramaria\.env" >nul 2>&1
    )
)

echo.
echo [Step 6] Generate readme...
if exist "dist\Ramaria" (
    echo Ramaria Distribution Package > "dist\Ramaria\README.txt"
    echo. >> "dist\Ramaria\README.txt"
    echo See README_QuickStart_Distribution.md for usage instructions. >> "dist\Ramaria\README.txt"
)

echo.
echo ================================================
echo   Build Complete!
echo ================================================
echo.
echo Output: dist\Ramaria\
echo.
pause
exit

:error_python
echo.
echo [Error] Python not found. Please install Python 3.10+
pause
exit /b 1

:error_pyinstaller
echo.
echo [Error] PyInstaller not found. Please run:
echo %PYTHON% -m pip install pyinstaller
pause
exit /b 1

:error_build
echo.
echo [Error] Build failed.
pause
exit /b 1
