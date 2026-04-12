@echo off
:: ============================================================
:: start.bat — 珊瑚菌 Ramaria 启动脚本（Windows）
:: 版本：v0.3.6-hotfix
::
:: 执行内容：
::   前置检查 1  虚拟环境是否存在（未存在则引导先运行 install.bat）
::   前置检查 2  .env 配置文件是否存在（未存在则引导配置）
::   前置检查 3  数据库文件是否存在（不存在则自动初始化）
::   启动         python app\main.py
::
:: 注意事项：
::   · 本文件必须保存为 UTF-8 with BOM 编码，否则中文乱码
::   · 运行前请确保本地模型推理服务（LM Studio / Ollama）已启动
::   · 按 Ctrl+C 可停止服务，窗口会提示"终止批处理操作吗"，输入 Y 确认
:: ============================================================

:: 强制使用 UTF-8 代码页，避免中文乱码
chcp 65001 >nul 2>&1

@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================
echo   珊瑚菌 Ramaria
echo ================================================
echo.

:: ============================================================
:: 前置检查 1：虚拟环境是否存在
:: ============================================================
:: 这是 start.bat 直接闪退最常见的原因：
:: venv 不存在时 call activate.bat 会报"找不到文件"，
:: errorlevel 变为非零，后续所有命令连锁失败，
:: 因为没有 pause，窗口瞬间关闭，用户看不到任何提示。
if not exist "venv\Scripts\activate.bat" (
    echo [错误] 未找到虚拟环境。
    echo.
    echo        请先运行 install.bat 完成安装，再使用本脚本启动服务。
    echo.
    pause
    exit /b 1
)

:: ============================================================
:: 前置检查 2：.env 配置文件是否存在
:: ============================================================
if not exist ".env" (
    echo [首次运行] 未找到 .env 配置文件。
    echo.

    if exist ".env.example" (
        :: 自动从模板生成 .env
        copy ".env.example" ".env" >nul
        echo [完成] 已根据 .env.example 生成 .env。
    ) else (
        echo [警告] 也未找到 .env.example 模板。
        echo        请参考 README.md 手动创建 .env 文件。
        echo.
        pause
        exit /b 1
    )

    echo.
    echo [重要] 请先用文本编辑器打开 .env，填写以下必要配置后再重新运行本脚本：
    echo.
    echo          LOCAL_API_URL    — 本地模型服务地址
    echo                            （LM Studio 默认：http://localhost:1234/v1/chat/completions）
    echo          LOCAL_MODEL_NAME — 模型名称（需与推理服务中加载的模型一致）
    echo          EMBEDDING_MODEL  — 嵌入模型本地路径
    echo.
    pause
    exit /b 0
)

:: ============================================================
:: 激活虚拟环境
:: ============================================================
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo [错误] 虚拟环境激活失败。
    echo        请尝试删除 venv 文件夹，重新运行 install.bat。
    echo.
    pause
    exit /b 1
)

:: ============================================================
:: 前置检查 3：确保目录和数据库存在
:: ============================================================
:: 确保 data/ logs/ 目录存在，mkdir 对已存在目录报错，用 2>nul 静默
mkdir data 2>nul
mkdir logs 2>nul

if not exist "data\assistant.db" (
    echo [提示] 未找到数据库文件，正在初始化...
    python scripts\setup_db.py
    if errorlevel 1 (
        echo.
        echo [警告] 数据库初始化脚本报错，将尝试继续启动。
        echo        如果服务启动后出现数据库相关报错，
        echo        请手动运行：python scripts\setup_db.py
        echo.
    ) else (
        echo [完成] 数据库初始化成功。
    )
    echo.
)

:: ============================================================
:: 启动服务
:: ============================================================
echo [启动] 正在启动珊瑚菌服务...
echo.
echo   本地访问：http://localhost:8000
echo   局域网访问：http://^<本机IP^>:8000
echo.
echo   按 Ctrl+C 停止服务
echo ------------------------------------------------
echo.

:: 启动 FastAPI 服务
:: app\main.py 内部使用 uvicorn，会阻塞在此行直到用户 Ctrl+C
python app\main.py

:: ============================================================
:: 服务退出后的处理
:: ============================================================
:: 正常到达这里说明服务已停止（Ctrl+C 或程序自行退出）
:: pause 确保用户能看到退出前的最后输出，不会直接闪退
echo.
echo [停止] 珊瑚菌服务已停止。
echo.
pause
