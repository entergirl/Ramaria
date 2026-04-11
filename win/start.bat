@echo off
chcp 65001 >nul
echo ================================================
echo 珊瑚菌 Ramaria
echo ================================================
echo.

call venv\Scripts\activate.bat

::: 检查首次运行
if not exist ".env" (
    echo [首次运行] 正在初始化配置...
    if exist ".env.example" copy .env.example .env >nul
    echo [提示] 请编辑 .env 文件后重新运行
    echo.
    echo 必填项:
    echo   LOCAL_API_URL    - 本地模型服务地址
    echo   LOCAL_MODEL_NAME - 模型名称
    echo   EMBEDDING_MODEL  - 嵌入模型路径
    echo.
    pause
    exit /b 1
)

::: 确保目录存在
if not exist "data" mkdir data
if not exist "logs" mkdir logs

::: 检查数据库
if not exist "data\assistant.db" (
    echo [提示] 正在初始化数据库...
    python scripts\setup_db.py
)

::: 启动
echo [启动] 珊瑚菌服务...
echo [访问] http://localhost:8000
echo.
python app\main.py

pause
