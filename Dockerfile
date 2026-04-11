# =============================================================================
# 珊瑚菌 Ramaria Dockerfile
# =============================================================================
# 构建: docker build -t ramaria .
# 运行: docker run -p 8000:8000 -v ./data:/app/data ramaria

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (ChromaDB 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1

# 使用 pyproject.toml 安装依赖
RUN pip install --upgrade pip && \
    pip install -e .

# 复制项目文件
COPY . .

# 创建必要目录
RUN mkdir -p data logs config

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 启动命令
CMD ["python", "app/main.py"]
