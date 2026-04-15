"""
src/ramaria/logger.py — 统一日志管理模块

职责：
    为整个项目提供统一的 Logger，封装 Python 标准库 logging 的配置细节。
    各模块只需一行 import 即可获得格式一致、行为可控的 logger。

日志输出目标（双通道）：
    1. 控制台（stdout）
       级别：INFO（正常）/ DEBUG（DEBUG=True 时）
    2. 文件（logs/coral.log）
       级别：DEBUG（始终最详细）
       轮转：单文件最大 10MB，保留最近 5 个备份

使用方法：
    from ramaria.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


# =============================================================================
# 路径配置
# =============================================================================

# 本文件在 src/ramaria/，项目根目录是上三级
# src/ramaria/logger.py → src/ramaria/ → src/ → 项目根
_ROOT_DIR = Path(__file__).parent.parent.parent

LOG_DIR  = _ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "coral.log"


# =============================================================================
# 格式配置
# =============================================================================

CONSOLE_FORMAT  = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
CONSOLE_DATEFMT = "%Y-%m-%d %H:%M:%S"

FILE_FORMAT  = "%(asctime)s [%(name)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

LOG_MAX_BYTES    = 10 * 1024 * 1024   # 10MB
LOG_BACKUP_COUNT = 5


# =============================================================================
# 内部状态
# =============================================================================

_initialized_loggers: set[str] = set()


# =============================================================================
# 核心函数
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    获取已配置双通道（控制台 + 文件）的 Logger。

    首次调用时初始化，后续对同一 name 的调用直接返回缓存实例。

    参数：
        name — logger 名称，推荐传入 __name__

    返回：
        logging.Logger — 配置好的 logger 实例
    """
    logger = logging.getLogger(name)

    if name in _initialized_loggers:
        return logger

    # 读取 DEBUG 开关
    try:
        from ramaria.config import DEBUG
        console_level = logging.DEBUG if DEBUG else logging.INFO
    except ImportError:
        console_level = logging.INFO

    logger.setLevel(logging.DEBUG)

    # ── Handler 1：控制台 ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(fmt=CONSOLE_FORMAT, datefmt=CONSOLE_DATEFMT)
    )

    # ── Handler 2：文件轮转 ──
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename     = LOG_FILE,
        maxBytes     = LOG_MAX_BYTES,
        backupCount  = LOG_BACKUP_COUNT,
        encoding     = "utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=FILE_FORMAT, datefmt=FILE_DATEFMT)
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    _initialized_loggers.add(name)
    return logger