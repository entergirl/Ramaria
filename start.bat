@echo off
chcp 65001 >nul
echo ================================================
echo 珊瑚菌 Ramaria 启动脚本
echo ================================================
echo.

:: 检查虚拟环境是否存在
if not exist "venv" (
    echo [警告] 虚拟环境不存在，正在创建...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    echo.
)

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 检查 .env 文件
if not exist ".env" (
    echo [警告] .env 文件不存在，正在创建...
    copy .env.example .env
    echo [提示] 请先编辑 .env 文件配置 API Key
    echo.
)

:: 检查数据库
if not exist "data" (
    mkdir data
)

:: 启动应用
echo [启动] 珊瑚菌服务...
echo [访问] http://localhost:8000
echo.
python app\main.py

pause
