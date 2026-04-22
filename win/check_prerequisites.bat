@echo off
chcp 65001 >nul 2>&1
REM ============================================================
REM 检查前置条件.bat — 珊瑚菌运行前置检查
REM ============================================================
REM
REM 用法：
REM     双击运行此脚本，或在 exe 所在目录打开终端运行
REM
REM 检查项目：
REM     1. LM Studio / Ollama 是否安装
REM     2. 嵌入模型目录是否存在
REM     3. 端口 8000 是否被占用
REM     4. .env 配置是否完整
REM
REM 此脚本独立运行，不依赖 Python 环境
REM ============================================================

setlocal enabledelayedexpansion

:start
cls
echo.
echo  ================================================
echo     珊瑚菌 · 前置条件检查
echo  ================================================
echo.
echo  正在检查运行环境，请稍候...
echo.

REM ─────────────────────────────────────────────────────────────────
REM 检查 1：LM Studio 或 Ollama 是否安装
REM ─────────────────────────────────────────────────────────────────
echo [1/4] 检查本地推理服务...

set "lm_found="
set "ollama_found="

REM 检查 LM Studio 默认安装路径
if exist "%LOCALAPPDATA%\LM Studio\lm-studio.exe" (
    set "lm_found=!LOCALAPPDATA!\LM Studio"
) else if exist "%PROGRAMFILES%\LM Studio\lm-studio.exe" (
    set "lm_found=%PROGRAMFILES%\LM Studio"
) else if exist "%PROGRAMFILES^(X86^)%\LM Studio\lm-studio.exe" (
    set "lm_found=%PROGRAMFILES^(X86^)%\LM Studio"
)

REM 检查 Ollama
where ollama >nul 2>&1
if !errorlevel! equ 0 (
    set "ollama_found=已安装（已在 PATH 中）"
)

REM 检查 Ollama 默认安装路径
if exist "%LOCALAPPDATA%\Ollama\ollama.exe" (
    if not defined ollama_found set "ollama_found=!LOCALAPPDATA!\Ollama"
)

REM 显示推理服务检查结果
echo.
if defined lm_found (
    echo   [OK] LM Studio 已安装
    echo         !lm_found!
) else if defined ollama_found (
    echo   [OK] Ollama 已安装
    echo         !ollama_found!
) else (
    echo   [警告] 未检测到 LM Studio 或 Ollama
    echo.
    echo   请先安装本地推理服务：
    echo     - LM Studio: https://lmstudio.ai
    echo     - Ollama:   https://ollama.com
    echo.
    set /p continue_check="是否继续检查其他项目？(y/N): "
    if /i "!continue_check!" neq "y" goto end
)


REM ─────────────────────────────────────────────────────────────────
REM 检查 2：嵌入模型目录
REM ─────────────────────────────────────────────────────────────────
echo.
echo [2/4] 检查嵌入模型配置...

REM 从 .env 读取配置（如果有）
set "embedding_path="
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("\.env") do (
        if "%%a"=="EMBEDDING_MODEL" set "embedding_path=%%b"
    )
)

REM 清理路径中的引号和空格
set "embedding_path=!embedding_path: =!"
set "embedding_path=!embedding_path:"=!"

REM 标准化路径分隔符
set "embedding_path=!embedding_path:/=\!"

if not defined embedding_path (
    echo   [未配置] EMBEDDING_MODEL 未填写
    echo.
    echo   首次运行时需要在配置向导中填写嵌入模型路径
) else (
    if exist "!embedding_path!" (
        echo   [OK] 嵌入模型目录存在
        echo         !embedding_path!
    ) else (
        echo   [警告] 嵌入模型目录不存在
        echo         !embedding_path!
        echo.
        echo   请确保：
        echo     1. 嵌入模型已完整下载
        echo     2. 路径填写正确（应为文件夹路径，非文件路径）
    )
)


REM ─────────────────────────────────────────────────────────────────
REM 检查 3：端口占用
REM ─────────────────────────────────────────────────────────────────
echo.
echo [3/4] 检查端口 8000 是否被占用...

netstat -ano | findstr :8000 > temp_port.txt
set "port_used="
set "port_pid="

for /f "tokens=5" %%a in (temp_port.txt) do (
    set "port_used=是"
    set "port_pid=%%a"
)

REM 清理临时文件
del temp_port.txt >nul 2>&1

if defined port_used (
    echo   [警告] 端口 8000 已被占用
    echo.
    echo   PID: !port_pid!
    echo.
    echo   如需结束占用进程，请运行：
    echo     taskkill /PID !port_pid! /F
    echo.
    echo   或修改 .env 中的 SERVER_PORT 为其他端口
) else (
    echo   [OK] 端口 8000 可用
)


REM ─────────────────────────────────────────────────────────────────
REM 检查 4：.env 配置文件
REM ─────────────────────────────────────────────────────────────────
echo.
echo [4/4] 检查配置文件...

if exist ".env" (
    echo   [OK] .env 文件存在
) else (
    if exist ".env.example" (
        echo   [提示] .env 不存在，将从 .env.example 生成
    ) else (
        echo   [警告] .env 和 .env.example 均不存在
    )
)

REM 检查必要配置项
set "local_api="
set "local_model="
set "embed_model="

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("\.env") do (
        if "%%a"=="LOCAL_API_URL" (
            set "local_api=%%b"
            set "local_api=!local_api: =!"
            set "local_api=!local_api:"=!"
        )
        if "%%a"=="LOCAL_MODEL_NAME" (
            set "local_model=%%b"
            set "local_model=!local_model: =!"
            set "local_model=!local_model:"=!"
        )
        if "%%a"=="EMBEDDING_MODEL" (
            set "embed_model=%%b"
            set "embed_model=!embed_model: =!"
            set "embed_model=!embed_model:"=!"
        )
    )

    echo.
    echo   配置项检查：
    if defined local_api (
        echo     LOCAL_API_URL     [OK] !local_api!
    ) else (
        echo     LOCAL_API_URL     [未填写]
    )

    if defined local_model (
        echo     LOCAL_MODEL_NAME  [OK] !local_model!
    ) else (
        echo     LOCAL_MODEL_NAME  [未填写]
    )

    if defined embed_model (
        echo     EMBEDDING_MODEL   [OK] !embed_model!
    ) else (
        echo     EMBEDDING_MODEL   [未填写]
    )
)


REM ─────────────────────────────────────────────────────────────────
REM 检查结果汇总
REM ─────────────────────────────────────────────────────────────────
echo.
echo  ================================================
echo     检查完成
echo  ================================================
echo.
echo  前置条件说明：
echo    - 首次使用请先安装 LM Studio 或 Ollama
echo    - 首次运行会自动打开配置向导
echo    - 如遇问题请查看 logs 目录下的日志文件
echo.

:end
echo 按任意键退出...
pause >nul
endlocal
