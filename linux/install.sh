#!/usr/bin/env bash
# ==============================================================================
# install.sh — Ramaria 安装脚本 (Linux)
# ==============================================================================
#
# 用法：
#     bash linux/install.sh
#   或在项目根目录：
#     bash install.sh
#
# 执行内容：
#   步骤 1  检查 Python 版本（要求 3.10+）
#   步骤 2  创建或复用 venv 虚拟环境
#   步骤 3  升级 pip
#   步骤 4  安装项目依赖（pip install -e .）
#   步骤 5  生成 .env 配置文件（已存在则跳过）
#   步骤 6  初始化数据库（scripts/setup_db.py）
#
# ==============================================================================

set -e   # 任意命令失败立即退出
set -u   # 引用未定义变量时报错

# ------------------------------------------------------------------------------
# 定位项目根目录
# 无论从哪个目录调用本脚本，ROOT 始终指向项目根目录
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 兼容两种调用方式：
#   · bash linux/install.sh  → SCRIPT_DIR 是 linux/，ROOT 是上一级
#   · bash install.sh        → SCRIPT_DIR 就是根目录
if [ "$(basename "$SCRIPT_DIR")" = "linux" ]; then
    ROOT="$(dirname "$SCRIPT_DIR")"
else
    ROOT="$SCRIPT_DIR"
fi

VENV_DIR="$ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------

banner() {
    echo ""
    echo "=================================================="
    echo "  $1"
    echo "=================================================="
}

step() {
    echo ""
    echo "[步骤 $1/$2] $3"
}

ok()   { echo "  [完成] $1"; }
skip() { echo "  [跳过] $1"; }
warn() { echo "  [警告] $1"; }

# 打印错误并退出（set -e 已开启，此函数主要用于提供友好提示）
error_exit() {
    echo ""
    echo "  [错误] $1"
    echo ""
    exit 1
}

# ------------------------------------------------------------------------------
# 步骤 1：检查 Python 版本
# ------------------------------------------------------------------------------
check_python() {
    step 1 6 "检查 Python 版本..."

    # 优先使用 python3，其次 python
    if command -v python3 &>/dev/null; then
        PY_CMD="python3"
    elif command -v python &>/dev/null; then
        PY_CMD="python"
    else
        error_exit "未检测到 Python。\n\n  请先安装 Python 3.10+：\n    Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip\n    Fedora:        sudo dnf install python3\n    Arch:          sudo pacman -S python"
    fi

    # 读取主版本号和次版本号
    read -r PY_MAJOR PY_MINOR < <($PY_CMD -c "import sys; print(sys.version_info.major, sys.version_info.minor)")

    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
        error_exit "Python 版本过低（当前 ${PY_MAJOR}.${PY_MINOR}），需要 3.10 或以上版本。\n\n  Ubuntu 22.04+ 自带 Python 3.10，较旧版本可使用 deadsnakes PPA：\n    sudo add-apt-repository ppa:deadsnakes/ppa\n    sudo apt install python3.11 python3.11-venv"
    fi

    ok "Python ${PY_MAJOR}.${PY_MINOR} 版本检测通过。"
}

# ------------------------------------------------------------------------------
# 步骤 2：创建或复用 venv 虚拟环境
# ------------------------------------------------------------------------------
create_venv() {
    step 2 6 "配置虚拟环境..."

    if [ -f "$VENV_PYTHON" ]; then
        skip "虚拟环境已存在，复用现有环境。"
        return
    fi

    echo "  正在创建虚拟环境，请稍候..."

    # 部分 Linux 发行版需要单独安装 python3-venv
    if ! $PY_CMD -m venv "$VENV_DIR" 2>/dev/null; then
        error_exit "虚拟环境创建失败。\n\n  可能缺少 venv 模块，请安装：\n    Ubuntu/Debian: sudo apt install python3-venv\n    Fedora:        sudo dnf install python3-virtualenv\n\n  安装后重新运行本脚本。"
    fi

    ok "虚拟环境已创建：$VENV_DIR"
}

# ------------------------------------------------------------------------------
# 步骤 3：升级 pip
# ------------------------------------------------------------------------------
upgrade_pip() {
    step 3 6 "升级 pip..."

    if "$VENV_PYTHON" -m pip install --upgrade pip --quiet; then
        ok "pip 已升级。"
    else
        warn "pip 升级失败，将继续使用当前版本。"
    fi
}

# ------------------------------------------------------------------------------
# 步骤 4：安装项目依赖
# ------------------------------------------------------------------------------
install_deps() {
    step 4 6 "安装项目依赖（首次安装约需 3-10 分钟）..."
    echo "  网络不稳定时可重新运行本脚本，pip 会自动跳过已安装的包。"

    # 切换到项目根目录，确保 pip install -e . 能找到 pyproject.toml
    cd "$ROOT"

    if ! "$VENV_PYTHON" -m pip install -e . --quiet; then
        error_exit "依赖安装失败。\n\n  常见原因：\n    1. 网络不稳定，可重新运行或使用代理\n    2. 缺少系统级编译工具：\n         Ubuntu/Debian: sudo apt install build-essential python3-dev\n         Fedora:        sudo dnf install gcc python3-devel\n    3. 磁盘空间不足\n\n  可手动运行查看完整报错：\n    $VENV_PYTHON -m pip install -e ."
    fi

    ok "依赖安装成功。"
}

# ------------------------------------------------------------------------------
# 步骤 5：生成 .env 配置文件
# ------------------------------------------------------------------------------
setup_env() {
    step 5 6 "检查配置文件..."

    ENV_FILE="$ROOT/.env"
    ENV_EXAMPLE="$ROOT/.env.example"

    if [ -f "$ENV_FILE" ]; then
        skip ".env 已存在，保留现有配置。"
        return
    fi

    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        ok ".env 已根据 .env.example 生成。"
        echo ""
        echo "  ┌──────────────────────────────────────────────────────┐"
        echo "  │  重要：安装完成后请打开 .env 填写以下必要配置项        │"
        echo "  ├──────────────────────────────────────────────────────┤"
        echo "  │  LOCAL_API_URL    本地模型服务地址                    │"
        echo "  │  LOCAL_MODEL_NAME 模型名称（与推理服务保持一致）       │"
        echo "  │  EMBEDDING_MODEL  嵌入模型的本地文件夹完整路径         │"
        echo "  └──────────────────────────────────────────────────────┘"
    else
        warn "未找到 .env.example 模板，请手动创建 .env 并参考 README.md 填写配置。"
    fi
}

# ------------------------------------------------------------------------------
# 步骤 6：初始化数据库
# ------------------------------------------------------------------------------
init_db() {
    step 6 6 "初始化数据库..."

    DB_FILE="$ROOT/data/assistant.db"
    SETUP_SCRIPT="$ROOT/scripts/setup_db.py"

    # 确保 data/ logs/ 目录存在
    mkdir -p "$ROOT/data"
    mkdir -p "$ROOT/logs"

    if [ -f "$DB_FILE" ]; then
        skip "数据库已存在，跳过初始化。"
        return
    fi

    if [ ! -f "$SETUP_SCRIPT" ]; then
        warn "未找到 scripts/setup_db.py，数据库将在首次启动服务时自动创建。"
        return
    fi

    echo "  正在初始化数据库..."
    if "$VENV_PYTHON" "$SETUP_SCRIPT"; then
        ok "数据库初始化成功。"
    else
        warn "数据库初始化脚本报错，服务启动时会尝试自动创建。\n  如启动后出现数据库报错，请手动运行：$VENV_PYTHON scripts/setup_db.py"
    fi
}

# ==============================================================================
# 主流程
# ==============================================================================
main() {
    banner "Ramaria 安装脚本 (Linux)"
    echo "  项目目录：$ROOT"

    check_python
    create_venv
    upgrade_pip
    install_deps
    setup_env
    init_db

    banner "安装完成！"
    echo ""
    echo "  下一步："
    echo "    1. 打开 .env，确认配置已填写完整"
    echo "    2. 启动本地模型推理服务（LM Studio 或 Ollama）"
    echo "    3. 运行  bash linux/start.sh  启动珊瑚菌"
    echo ""
    echo "  访问地址：http://localhost:8000"
    echo ""
}

main
