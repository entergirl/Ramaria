#!/bin/bash
# =============================================================================
# 珊瑚菌 Ramaria 安装脚本 (Linux/macOS)
# =============================================================================

set -e

echo "================================================"
echo "珊瑚菌 Ramaria 安装脚本"
echo "================================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python 3，请先安装"
    echo "下载: https://www.python.org/downloads/"
    exit 1
fi

# 创建/更新虚拟环境
echo "[步骤 1/3] 环境配置..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[完成] 虚拟环境已创建"
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
echo "[完成] 依赖安装完成"

# 配置文件
echo ""
echo "[步骤 2/3] 配置文件..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "[完成] .env 已创建"
    fi
else
    echo "[跳过] .env 已存在"
fi

# 初始化数据库
echo ""
echo "[步骤 3/3] 初始化数据库..."
python scripts/setup_db.py >/dev/null 2>&1
if [ -f "data/assistant.db" ]; then
    echo "[完成] 数据库已就绪"
else
    echo "[跳过] 数据库将在首次启动时创建"
fi

echo ""
echo "================================================"
echo "安装完成！"
echo "================================================"
echo ""
echo "已就绪: 虚拟环境、依赖、配置、数据库"
echo ""
echo "启动命令: bash start.sh"
echo ""
