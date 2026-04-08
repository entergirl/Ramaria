"""
logger.py — 统一日志管理模块

职责：
    为整个项目提供统一的 Logger，封装 Python 标准库 logging 的配置细节。
    各模块只需一行 import 即可获得格式一致、行为可控的 logger。

日志输出目标（双通道）：
    1. 控制台（stdout）
       级别：INFO（正常）/ DEBUG（DEBUG=True 时）
       格式：时间 + 模块名 + 级别 + 消息
    2. 文件（logs/coral.log）
       级别：DEBUG（始终最详细）
       格式：含文件名 + 行号
       轮转：单文件最大 10MB，保留最近 5 个备份

与 config 的关系：
    读取 ramaria.config.DEBUG 决定控制台日志级别。
    读取失败时默认 INFO。

使用方法：
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("消息")
    logger.debug("调试信息")
    logger.warning("警告")
    logger.error("错误")
    logger.exception("异常（自动附带 traceback）")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


# =============================================================================
# 路径配置
# =============================================================================

# 项目根目录：logger.py 在根目录，所以 parent 就是根
_ROOT_DIR = Path(__file__).parent

# 日志目录和文件
LOG_DIR  = _ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "coral.log"


# =============================================================================
# 格式配置
# =============================================================================

# 控制台格式：简洁，便于开发期快速阅读
# 示例：2026-03-22 14:30:05 [summarizer] INFO: L1 摘要已写入，id = 3
CONSOLE_FORMAT  = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
CONSOLE_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 文件格式：详细，含文件名和行号，便于精确定位
# 示例：2026-03-22 14:30:05,123 [summarizer] INFO summarizer.py:142 - L1 摘要已写入
FILE_FORMAT  = "%(asctime)s [%(name)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 文件轮转参数
LOG_MAX_BYTES    = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5


# =============================================================================
# 内部状态
# =============================================================================

# 已初始化的 logger 集合，防止重复添加 handler 导致日志重复输出
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

    # 已初始化则直接返回，避免重复添加 handler
    if name in _initialized_loggers:
        return logger

    # 从 ramaria.config 读取 DEBUG 开关
    try:
        from ramaria.config import DEBUG
        console_level = logging.DEBUG if DEBUG else logging.INFO
    except ImportError:
        # ramaria 包尚未安装或路径未配置时，降级为 INFO
        console_level = logging.INFO

    # logger 本身的级别设为 DEBUG，让所有消息都能流到 handler
    # 由各 handler 自己决定是否输出
    logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Handler 1：控制台（stdout）
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(fmt=CONSOLE_FORMAT, datefmt=CONSOLE_DATEFMT)
    )

    # ------------------------------------------------------------------
    # Handler 2：文件轮转
    # ------------------------------------------------------------------
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename     = LOG_FILE,
        maxBytes     = LOG_MAX_BYTES,
        backupCount  = LOG_BACKUP_COUNT,
        encoding     = "utf-8",  # 显式指定，避免 Windows 中文乱码
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=FILE_FORMAT, datefmt=FILE_DATEFMT)
    )

    # ------------------------------------------------------------------
    # 绑定 Handler
    # ------------------------------------------------------------------
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 阻止传播到 root logger，避免在 uvicorn 等框架下重复输出
    logger.propagate = False

    _initialized_loggers.add(name)
    return logger