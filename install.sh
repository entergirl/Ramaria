#!/bin/bash
# =============================================================================
# 珊瑚菌 Ramaria 安装脚本 (Linux/macOS)
# =============================================================================

set -e

echo "================================================"
echo "珊瑚菌 Ramaria 安装脚本"
echo "================================================"
echo ""

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python 3，请先安装 Python 3.10+"
    echo "下载地址: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "[检查] Python 版本: $PYTHON_VERSION"

# 检查 pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "[错误] 未检测到 pip，请先安装 pip"
    exit 1
fi

# 创建虚拟环境
echo ""
echo "[步骤 1/4] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[完成] 虚拟环境已创建"
else
    echo "[跳过] 虚拟环境已存在"
fi

# 激活虚拟环境
echo ""
echo "[步骤 2/4] 激活虚拟环境..."
source venv/bin/activate
echo "[完成] 虚拟环境已激活 (venv)"

# 安装依赖
echo ""
echo "[步骤 3/4] 安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt
echo "[完成] 依赖安装完成"

# 配置环境文件
echo ""
echo "[步骤 4/4] 配置环境..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[完成] 配置文件已创建 (.env)"
    echo "[提示] 请编辑 .env 文件配置 API Key 和模型路径"
else
    echo "[跳过] .env 文件已存在"
fi

echo ""
echo "================================================"
echo "安装完成！"
echo "================================================"
echo ""
echo "接下来请:"
echo "  1. 编辑 .env 文件，配置 API Key 和模型路径"
echo "  2. 启动本地模型推理服务 (如 LM Studio / Ollama)"
echo "  3. 运行初始化脚本: python scripts/setup_db.py"
echo "  4. 启动应用: python app/main.py"
echo ""
echo "启动前请先激活虚拟环境: source venv/bin/activate"
echo ""
