@echo off
chcp 65001 >nul
echo ================================================
echo 珊瑚菌 Ramaria 安装脚本
echo ================================================
echo.

:: 检查 Python 版本
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [检查] Python 版本: %PYTHON_VERSION%

:: 创建虚拟环境
echo.
echo [步骤 1/4] 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo [完成] 虚拟环境已创建
) else (
    echo [跳过] 虚拟环境已存在
)

:: 激活虚拟环境
echo.
echo [步骤 2/4] 激活虚拟环境...
call venv\Scripts\activate.bat
echo [完成] 虚拟环境已激活

:: 安装依赖
echo.
echo [步骤 3/4] 安装依赖...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)
echo [完成] 依赖安装完成

:: 配置环境文件
echo.
echo [步骤 4/4] 配置环境...
if not exist ".env" (
    copy .env.example .env
    echo [完成] 配置文件已创建 (.env)
    echo [提示] 请编辑 .env 文件配置 API Key 和模型路径
) else (
    echo [跳过] .env 文件已存在
)

echo.
echo ================================================
echo 安装完成！
echo ================================================
echo.
echo 接下来请:
echo   1. 编辑 .env 文件，配置 API Key 和模型路径
echo   2. 启动本地模型推理服务 (如 LM Studio)
echo   3. 运行初始化脚本: python scripts\setup_db.py
echo   4. 启动应用: python app\main.py
echo.
echo 启动前请先激活虚拟环境: call venv\Scripts\activate.bat
echo.
pause
