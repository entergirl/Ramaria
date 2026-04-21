@echo off
chcp 65001 >nul 2>&1
REM ============================================================
REM build.bat — Ramaria Windows 打包脚本
REM
REM 用法：
REM     双击运行 build.bat
REM     或在项目根目录命令行：build.bat
REM
REM 前置条件：
REM     1. 已激活虚拟环境（venv\Scripts\activate）
REM        或直接使用 venv 内的 Python 运行
REM     2. 已安装打包依赖：pip install -e .[bundle]
REM     3. icon.ico 已放置在项目根目录（可选，无则使用系统默认图标）
REM
REM 输出：
REM     dist\Ramaria\        打包产物目录（含 Ramaria.exe 和依赖）
REM     dist\Ramaria.zip     打包压缩包（方便分发）
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ================================================
echo   珊瑚菌 Ramaria 打包脚本
echo ================================================
echo.

REM 检测虚拟环境
if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
    echo [信息] 使用虚拟环境：venv\Scripts\python.exe
) else (
    set PYTHON=python
    echo [警告] 未找到 venv，使用系统 Python
)

REM 检测 PyInstaller
%PYTHON% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未安装 PyInstaller，请先运行：
    echo         pip install -e .[bundle]
    pause
    exit /b 1
)

REM 清理旧的打包产物
echo [步骤 1/3] 清理旧产物…
if exist "dist\Ramaria" (
    rmdir /s /q "dist\Ramaria"
    echo   已删除 dist\Ramaria
)
if exist "build\Ramaria" (
    rmdir /s /q "build\Ramaria"
    echo   已删除 build\Ramaria
)

REM 打包
echo [步骤 2/3] 正在打包（约需 2-5 分钟）…
%PYTHON% -m PyInstaller ramaria.spec --noconfirm
if errorlevel 1 (
    echo.
    echo [错误] 打包失败，请查看上方错误信息。
    pause
    exit /b 1
)

REM 打包成 zip（可选，需要系统自带 PowerShell）
echo [步骤 3/3] 创建压缩包…
if exist "dist\Ramaria" (
    powershell -command "Compress-Archive -Path 'dist\Ramaria' -DestinationPath 'dist\Ramaria.zip' -Force" 2>nul
    if exist "dist\Ramaria.zip" (
        echo   已生成 dist\Ramaria.zip
    )
)

echo.
echo ================================================
echo   打包完成！
echo   产物目录：dist\Ramaria\
echo   启动文件：dist\Ramaria\Ramaria.exe
echo ================================================
echo.

REM 询问是否立即测试
set /p TEST_NOW="是否立即运行测试？(y/N): "
if /i "!TEST_NOW!"=="y" (
    echo 正在启动…
    start "" "dist\Ramaria\Ramaria.exe"
)

pause
