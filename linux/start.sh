#!/bin/bash
# =============================================================================
# 珊瑚菌 Ramaria 启动脚本 (Linux/macOS)
# =============================================================================

source venv/bin/activate

# 检查首次运行
if [ ! -f ".env" ]; then
    echo "[首次运行] 正在初始化配置..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    fi
    echo "[提示] 请编辑 .env 文件后重新运行"
    echo ""
    echo "必填项:"
    echo "  LOCAL_API_URL    - 本地模型服务地址"
    echo "  LOCAL_MODEL_NAME - 模型名称"
    echo "  EMBEDDING_MODEL  - 嵌入模型路径"
    echo ""
    exit 1
fi

# 确保目录存在
mkdir -p data logs

# 检查数据库
if [ ! -f "data/assistant.db" ]; then
    echo "[提示] 正在初始化数据库..."
    python scripts/setup_db.py
fi

# 启动
echo "[启动] 珊瑚菌服务..."
echo "[访问] http://localhost:8000"
echo ""
python app/main.py
