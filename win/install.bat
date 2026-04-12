@echo off
:: ============================================================
:: install.bat — 珊瑚菌 Ramaria 一键安装脚本（Windows）
:: 版本：v0.3.6-hotfix
::
:: 执行内容：
::   步骤 1  检查 Python 版本（要求 3.10+）
::   步骤 2  创建或复用 venv 虚拟环境
::   步骤 3  安装项目依赖（pip install -e .）
::   步骤 4  生成 .env 配置文件（已存在则跳过）
::   步骤 5  初始化数据库（setup_db.py）
::
:: 注意事项：
::   · 本文件必须保存为 UTF-8 with BOM 编码，否则中文乱码
::   · 首次运行后请编辑 .env，填写本地模型地址和嵌入模型路径
::   · 安装完成后使用 start.bat 启动服务
:: ============================================================

:: 强制使用 UTF-8 代码页，避免中文乱码
:: 65001 = UTF-8；若终端本身不支持 UTF-8 字体，仍可能显示方块，
:: 但脚本逻辑不受影响
chcp 65001 >nul 2>&1

:: 关闭命令回显，保持输出整洁
@echo off

:: 开启变量延迟展开，允许在 if/for 块内使用 !变量名! 语法
setlocal enabledelayedexpansion

echo.
echo ================================================
echo   珊瑚菌 Ramaria 安装脚本
echo ================================================
echo.

:: ============================================================
:: 步骤 1：检查 Python 是否已安装，且版本 >= 3.10
:: ============================================================
echo [步骤 1/5] 检查 Python 版本...

:: 先确认 python 命令本身存在
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [错误] 未检测到 Python，请先安装 Python 3.10 或以上版本。
    echo        下载地址：https://www.python.org/downloads/
    echo        安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:: 获取 Python 主版本号和次版本号，用于版本门控
:: python -c 输出形如 "3 10"，分别赋给 PY_MAJOR 和 PY_MINOR
for /f "tokens=1,2" %%a in ('python -c "import sys; print(sys.version_info.major, sys.version_info.minor)"') do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

:: Python 版本必须 >= 3.10
:: 先判断主版本（必须是 3），再判断次版本（必须 >= 10）
if !PY_MAJOR! LSS 3 (
    echo [错误] Python 版本过低，当前为 !PY_MAJOR!.!PY_MINOR!，需要 3.10+。
    pause
    exit /b 1
)
if !PY_MAJOR! EQU 3 (
    if !PY_MINOR! LSS 10 (
        echo [错误] Python 版本过低，当前为 !PY_MAJOR!.!PY_MINOR!，需要 3.10+。
        pause
        exit /b 1
    )
)

echo [完成] Python !PY_MAJOR!.!PY_MINOR! 检测通过。

:: ============================================================
:: 步骤 2：创建或复用 venv 虚拟环境
:: ============================================================
echo.
echo [步骤 2/5] 配置虚拟环境...

if exist "venv\Scripts\activate.bat" (
    :: 虚拟环境已存在，直接复用，不重新创建
    echo [跳过] 虚拟环境已存在，复用现有环境。
) else (
    echo        正在创建虚拟环境，请稍候...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo [错误] 虚拟环境创建失败。
        echo        可能原因：磁盘空间不足，或 Python 安装不完整。
        echo.
        pause
        exit /b 1
    )
    echo [完成] 虚拟环境已创建。
)

:: 激活虚拟环境
:: call 使激活脚本在当前 cmd 进程中执行，而非新开子进程
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo [错误] 虚拟环境激活失败。
    echo        请尝试删除 venv 文件夹后重新运行本脚本。
    echo.
    pause
    exit /b 1
)

:: ============================================================
:: 步骤 3：安装项目依赖
:: ============================================================
echo.
echo [步骤 3/5] 安装项目依赖（首次安装约需 3-10 分钟）...

:: 先升级 pip，避免老版本 pip 安装部分包时报错
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [警告] pip 升级失败，将继续使用当前版本。
)

:: pip install -e . 以"可编辑模式"安装当前项目
:: -e 表示 editable：修改源码后无需重新安装即可生效
:: --quiet 减少冗余输出，只在出错时打印信息
pip install -e . --quiet
if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败。
    echo        常见原因：
    echo          1. 网络连接不稳定，可尝试切换网络或使用代理
    echo          2. pip 版本过低（已在上一步尝试升级）
    echo          3. 某个依赖包与 Python 版本不兼容
    echo        建议重新运行本脚本，或手动执行：pip install -e .
    echo.
    pause
    exit /b 1
)

echo [完成] 依赖安装成功。

:: ============================================================
:: 步骤 4：生成 .env 配置文件
:: ============================================================
echo.
echo [步骤 4/5] 检查配置文件...

if exist ".env" (
    :: .env 已存在，跳过（避免覆盖用户已填写的配置）
    echo [跳过] .env 已存在，保留现有配置。
) else (
    if exist ".env.example" (
        :: 从模板复制生成初始 .env
        copy ".env.example" ".env" >nul
        if errorlevel 1 (
            echo [警告] .env 文件生成失败，请手动将 .env.example 复制为 .env。
        ) else (
            echo [完成] 已根据 .env.example 生成 .env 配置文件。
            echo.
            echo [重要] 安装完成后，请用文本编辑器打开 .env，填写以下必要配置：
            echo          LOCAL_API_URL    — 本地模型服务地址（如 LM Studio 默认为 http://localhost:1234/v1/chat/completions）
            echo          LOCAL_MODEL_NAME — 模型名称（需与推理服务中加载的模型一致）
            echo          EMBEDDING_MODEL  — 嵌入模型本地路径
        )
    ) else (
        :: .env.example 也不存在（不正常，提示用户）
        echo [警告] 未找到 .env.example 模板文件，请手动创建 .env 并填写配置。
        echo        参考文档：README.md 快速开始章节
    )
)

:: ============================================================
:: 步骤 5：初始化数据库
:: ============================================================
echo.
echo [步骤 5/5] 初始化数据库...

:: 确保 data/ 和 logs/ 目录存在
:: mkdir 在目录已存在时会报错，2>nul 将报错信息丢弃
mkdir data  2>nul
mkdir logs  2>nul

if exist "data\assistant.db" (
    echo [跳过] 数据库已存在，跳过初始化。
) else (
    :: 调用 setup_db.py 进行数据库建表和默认数据写入
    :: setup_db.py 本身是幂等的，重复运行不会破坏已有数据
    python scripts\setup_db.py
    if errorlevel 1 (
        echo.
        echo [警告] 数据库初始化脚本报错。
        echo        这不一定是致命错误；数据库也可能在首次启动服务时自动创建。
        echo        如启动后出现数据库相关报错，请手动运行：python scripts\setup_db.py
        echo.
    ) else (
        echo [完成] 数据库初始化成功。
    )
)

:: ============================================================
:: 安装完成
:: ============================================================
echo.
echo ================================================
echo   安装完成！
echo ================================================
echo.
echo   下一步：
echo     1. 确认 .env 中的配置已填写完整
echo     2. 启动本地模型推理服务（LM Studio 或 Ollama）
echo     3. 双击运行 start.bat 启动珊瑚菌服务
echo.
echo   访问地址：http://localhost:8000
echo.
pause
