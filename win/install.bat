@echo off
chcp 65001 >nul
echo ================================================
echo 珊瑚菌 Ramaria 安装脚本
echo ================================================
echo.

::: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    echo 下载: https://www.python.org/downloads/
    pause
    exit /b 1
)

::: 创建/更新虚拟环境
echo [步骤 1/3] 环境配置...
if not exist "venv" (
    python -m venv venv
    echo [完成] 虚拟环境已创建
)

call venv\Scripts\activate.bat
pip install --upgrade pip -q
pip install -e . -q
echo [完成] 依赖安装完成

::: 配置文件
echo.
echo [步骤 2/3] 配置文件...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [完成] .env 已创建
    )
) else (
    echo [跳过] .env 已存在
)

::: 初始化数据库
echo.
echo [步骤 3/3] 初始化数据库...
python scripts\setup_db.py >nul 2>&1
if exist "data\assistant.db" (
    echo [完成] 数据库已就绪
) else (
    echo [跳过] 数据库将在首次启动时创建
)

echo.
echo ================================================
echo 安装完成！
echo ================================================
echo.
echo 已就绪: 虚拟环境、依赖、配置、数据库
echo.
echo 启动命令:
echo   start.bat
echo.
pause
