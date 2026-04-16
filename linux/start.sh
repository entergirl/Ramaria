#!/usr/bin/env bash
# ==============================================================================
# start.sh - Ramaria 启动脚本 (Linux)
# ==============================================================================
#
# 用法：
#     bash linux/start.sh
#   或在项目根目录：
#     bash start.sh
#
# 前置检查 1  虚拟环境是否存在
# 前置检查 2  .env 是否存在且必填项已填写
# 前置检查 3  EMBEDDING_MODEL 路径是否真实存在
# 前置检查 4  数据库是否存在（不存在则自动初始化）
# 启动        使用 venv 内的 Python 运行 app/main.py
#
# ==============================================================================

set -u

# ------------------------------------------------------------------------------
# 定位项目根目录
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(basename "$SCRIPT_DIR")" = "linux" ]; then
    ROOT="$(dirname "$SCRIPT_DIR")"
else
    ROOT="$SCRIPT_DIR"
fi

VENV_PYTHON="$ROOT/venv/bin/python"
REQUIRED_KEYS=("LOCAL_API_URL" "LOCAL_MODEL_NAME" "EMBEDDING_MODEL")

# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------

banner() {
    echo ""
    echo "=================================================="
    echo "  $1"
    echo "=================================================="
    echo ""
}

info() { echo "  [信息] $1"; }
warn() { echo "  [警告] $1"; }

error_exit() {
    echo ""
    echo "  [错误] $1"
    echo ""
    exit 1
}

# 从 .env 文件中读取指定 key 的值，去除注释和引号
get_env_value() {
    local key="$1"
    local file="$2"
    grep -E "^${key}[[:space:]]*=" "$file" 2>/dev/null \
        | head -1 \
        | sed 's/^[^=]*=//' \
        | sed 's/#.*//' \
        | sed "s/^[[:space:]]*//;s/[[:space:]]*$//" \
        | sed "s/^['\"]//;s/['\"]$//"
}

# ------------------------------------------------------------------------------
# 前置检查 1：虚拟环境
# ------------------------------------------------------------------------------
check_venv() {
    if [ ! -f "$VENV_PYTHON" ]; then
        error_exit "未找到虚拟环境。\n\n  请先运行安装脚本：\n    bash linux/install.sh"
    fi
    info "虚拟环境检测通过。"
}

# ------------------------------------------------------------------------------
# 前置检查 2：.env 存在且必填项已填写
# ------------------------------------------------------------------------------
check_env_file() {
    local env_file="$ROOT/.env"
    local env_example="$ROOT/.env.example"

    if [ ! -f "$env_file" ]; then
        if [ -f "$env_example" ]; then
            cp "$env_example" "$env_file"
            echo "  [提示] 未找到 .env，已根据 .env.example 自动生成。"
        else
            error_exit "未找到 .env 配置文件。请参考 README.md 创建 .env 并填写配置。"
        fi
        echo ""
        echo "  请打开 .env 填写以下必要配置后重新运行本脚本："
        echo "    LOCAL_API_URL    - 本地模型服务地址"
        echo "    LOCAL_MODEL_NAME - 模型名称"
        echo "    EMBEDDING_MODEL  - 嵌入模型本地文件夹路径"
        echo ""
        exit 0
    fi

    local missing=()
    for key in "${REQUIRED_KEYS[@]}"; do
        local val
        val="$(get_env_value "$key" "$env_file")"
        if [ -z "$val" ]; then
            missing+=("$key")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo "  [错误] .env 中以下必填项尚未配置："
        for key in "${missing[@]}"; do
            echo "           * $key"
        done
        echo ""
        echo "  请打开 .env 填写后重新运行本脚本。"
        echo ""
        exit 1
    fi

    info ".env 配置检测通过。"
}

# ------------------------------------------------------------------------------
# 前置检查 3：EMBEDDING_MODEL 路径存在性
# ------------------------------------------------------------------------------
check_embedding_model() {
    local env_file="$ROOT/.env"
    local path
    path="$(get_env_value "EMBEDDING_MODEL" "$env_file")"

    if [ -z "$path" ]; then
        return
    fi

    if [ ! -e "$path" ]; then
        echo ""
        warn "EMBEDDING_MODEL 路径不存在："
        echo "           $path"
        echo ""
        echo "  这会导致服务在启动阶段崩溃。"
        echo "  请确认嵌入模型已下载完整，且 .env 中的路径正确。"
        echo ""
        printf "  是否仍要继续启动？(y/N): "
        read -r answer
        if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
            exit 0
        fi
    else
        info "嵌入模型路径检测通过。"
    fi
}

# ------------------------------------------------------------------------------
# 前置检查 4：数据库
# ------------------------------------------------------------------------------
check_and_init_db() {
    mkdir -p "$ROOT/data" "$ROOT/logs"

    if [ -f "$ROOT/data/assistant.db" ]; then
        info "数据库检测通过。"
        return
    fi

    if [ ! -f "$ROOT/scripts/setup_db.py" ]; then
        warn "未找到 scripts/setup_db.py，跳过初始化。"
        return
    fi

    echo "  [提示] 未找到数据库，正在初始化..."
    if "$VENV_PYTHON" "$ROOT/scripts/setup_db.py"; then
        info "数据库初始化成功。"
    else
        warn "数据库初始化失败，将尝试继续启动。"
    fi
}

# ------------------------------------------------------------------------------
# 启动服务
# ------------------------------------------------------------------------------
start_service() {
    if [ ! -f "$ROOT/app/main.py" ]; then
        error_exit "未找到 app/main.py，请确认项目文件完整。"
    fi

    # 将 .env 中的配置导出为环境变量，确保子进程可以继承
    set -a
    source "$ROOT/.env"
    set +a

    echo ""
    echo "  --------------------------------------------------"
    echo "  本地访问：  http://localhost:8000"
    local local_ip
    local_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -n "$local_ip" ]; then
        echo "  局域网访问：http://${local_ip}:8000"
    fi
    echo "  按 Ctrl+C 停止服务"
    echo "  --------------------------------------------------"
    echo ""

    cd "$ROOT"
    "$VENV_PYTHON" app/main.py
    local code=$?

    echo ""
    if [ $code -eq 0 ] || [ $code -eq 130 ]; then
        info "服务已停止。"
    else
        echo "  [错误] 服务异常退出（退出码：$code）。"
        echo ""
        echo "  常见原因："
        echo "    * 本地推理服务未启动（LM Studio / Ollama）"
        echo "    * EMBEDDING_MODEL 路径不存在"
        echo "    * 端口 8000 被占用（检查：lsof -i:8000）"
        exit $code
    fi
}

# ==============================================================================
# 主流程
# ==============================================================================
main() {
    banner "Ramaria"
    echo "  正在进行前置检查..."
    echo ""
    check_venv
    check_env_file
    check_embedding_model
    check_and_init_db
    echo ""
    info "前置检查全部通过，正在启动服务..."
    start_service
}

main
