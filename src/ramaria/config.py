"""
src/ramaria/config.py — 运行时参数配置

职责：
    集中管理所有运行时可调整的参数：路径、模型地址、业务阈值等。
    不放 Prompt 文本（→ 各自负责的模块）
    不放静态常量（→ 顶层 constants.py）

调整参数时只改这一个文件。
"""

from __future__ import annotations

import os
from pathlib import Path


# =============================================================================
# 路径
# =============================================================================

# 项目根目录
# 本文件位于 src/ramaria/config.py
# parents[0] = src/ramaria/
# parents[1] = src/
# parents[2] = 项目根目录
ROOT_DIR = Path(__file__).parents[2]

DATA_DIR   = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"
LOG_DIR    = ROOT_DIR / "logs"

# 数据库文件路径（data/ 目录下，整个目录都在 .gitignore 中）
DB_PATH = DATA_DIR / "assistant.db"

# Chroma 向量索引持久化目录
CHROMA_DIR = DATA_DIR / "chroma_db"

# 人设配置文件
PERSONA_PATH = CONFIG_DIR / "persona.toml"


# =============================================================================
# 向量检索
# =============================================================================

# 嵌入模型路径（本地文件），切换模型后需重建全部索引
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "")

# L0 滑动窗口大小：以最相关消息为中心，前后各取几条作为上下文
L0_WINDOW_SIZE = 3

# 各层检索默认返回条数
L0_RETRIEVE_TOP_K = 3
L1_RETRIEVE_TOP_K = 4
L2_RETRIEVE_TOP_K = 2

# 语义相似度过滤阈值（余弦距离，超过此值视为不相关）
SIMILARITY_THRESHOLD = 0.6

# RRF 融合参数
# RRF_K：倒数排名融合的平滑系数，标准取值 60
RRF_K: int = 60

# BM25 通道相对向量通道的权重（1.0 = 等权）
BM25_WEIGHT: float = 1.0

# 图谱通道相对向量通道的权重（冷启动阶段保守设置）
GRAPH_WEIGHT: float = 0.8

# 检索结果注入 prompt 时各层的权重系数
# 系数 < 1.0 → 该层更容易排前面（"加分"）
# 系数 = 1.0 → 不额外加权
RETRIEVAL_WEIGHT_L2: float = 0.8  # L2 粒度粗，优先展示
RETRIEVAL_WEIGHT_L1: float = 1.0  # L1 主力检索层，不额外加权


# =============================================================================
# 记忆衰减（Ebbinghaus 遗忘曲线）
# =============================================================================
#
# 衰减公式：R = e^(-t / S)
#   R：保留率（0~1），越大越容易被检索到
#   t：距记忆生成的天数（基于 created_at）
#   S：稳定性系数，越大衰减越慢
#
# S 值设计：
#   L0 = 10 → 细节信息衰减最快
#   L1 = 30 → 单次摘要衰减适中
#   L2 = 60 → 聚合摘要衰减最慢

MEMORY_DECAY_S_L0: int = 10
MEMORY_DECAY_S_L1: int = 30
MEMORY_DECAY_S_L2: int = 60

# last_accessed_at 保底加成
# 开启后：若记忆在 RECENT_BOOST_DAYS 天内被访问过，R 不低于 FLOOR
# 这是 R 的下限，不替换主衰减公式
MEMORY_DECAY_ENABLE_ACCESS_BOOST: bool  = True
MEMORY_DECAY_RECENT_BOOST_DAYS:   int   = 7
MEMORY_DECAY_RECENT_BOOST_FLOOR:  float = 0.5

# salience 对稳定性的加成系数
# S_adjusted = S × (1 + salience × MULTIPLIER)
# salience=1.0 时稳定性提升 50%
SALIENCE_DECAY_MULTIPLIER: float = 0.5


# =============================================================================
# Session 管理
# =============================================================================

# 空闲超过此时长（分钟）自动触发 L1 摘要生成
L1_IDLE_MINUTES: int = 10

# 空闲检测线程轮询间隔（秒）
IDLE_CHECK_INTERVAL_SECONDS: int = 60

# L2 定时检查线程轮询间隔（秒），每天执行一次
L2_CHECK_INTERVAL_SECONDS: int = 86400


# =============================================================================
# 记忆层触发阈值
# =============================================================================

# 未吸收的 L1 累计达到此条数时触发 L2 合并
L2_TRIGGER_COUNT: int = 5

# 最早一条未吸收 L1 距今超过此天数时触发 L2 合并
L2_TRIGGER_DAYS: int = 7


# =============================================================================
# 本地模型（LM Studio / Ollama，兼容 OpenAI API 格式）
# =============================================================================

LOCAL_API_URL:             str   = os.environ.get("LOCAL_API_URL", "http://localhost:1234/v1/chat/completions")
LOCAL_MODEL_NAME:          str   = os.environ.get("LOCAL_MODEL_NAME", "qwen/qwen3.5-9b")
LOCAL_TEMPERATURE:         float = float(os.environ.get("LOCAL_TEMPERATURE", "0.3"))
LOCAL_MAX_TOKENS_SUMMARY:  int   = int(os.environ.get("LOCAL_MAX_TOKENS_SUMMARY", "512"))
LOCAL_MAX_TOKENS_CHAT:     int   = int(os.environ.get("LOCAL_MAX_TOKENS_CHAT", "1024"))


# =============================================================================
# Claude API
# =============================================================================

# 从环境变量读取，不硬编码
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# 标识符中的日期是模型训练快照日期，非当前日期
# 更新模型版本时在此处修改
CLAUDE_MODEL_NAME:  str   = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS:  int   = 2048
CLAUDE_TEMPERATURE: float = 0.7


# =============================================================================
# Web 服务
# =============================================================================

SERVER_HOST: str  = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int  = int(os.environ.get("SERVER_PORT", "8000"))
DEBUG:       bool = os.environ.get("APP_DEBUG", "true").lower() == "true"


# =============================================================================
# BM25 索引增量更新
# =============================================================================

# 缓冲区积累超过此条数时触发合并重建（每次 L1/L2 写入先进缓冲，不立即重建）
BM25_INCREMENTAL_THRESHOLD: int = 10

# 后台定时重建间隔（秒），兜底保证即使写入量不足阈值也会定期重建
# 300 秒 = 5 分钟
BM25_REBUILD_INTERVAL: int = 300


# =============================================================================
# 杂项
# =============================================================================

DEFAULT_MODEL: str = "local"

# L1 time_period 合法值，summarizer 和 database 共同依赖
# 放在 config 而非 constants 是因为这是业务参数，未来可能扩展
TIME_PERIOD_OPTIONS: list[str] = ["清晨", "上午", "下午", "傍晚", "夜间", "深夜"]