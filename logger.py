"""
logger.py — 统一日志管理模块
=====================================================================

职责：
    为珊瑚菌项目的所有模块提供统一的日志记录器（Logger）。
    封装 Python 标准库 logging 的配置细节，各模块只需一行
    import 即可获得格式一致、行为可控的 logger。

日志输出目标（双通道）：
    1. 控制台（stdout）
       - 级别：INFO（正常运行时）/ DEBUG（DEBUG=True 时）
       - 格式：简洁，只含时间 + 模块名 + 级别 + 消息
       - 用途：开发期实时观察运行状态

    2. 文件（logs/coral.log）
       - 级别：DEBUG（始终记录最详细的日志，便于事后排查）
       - 格式：详细，含文件名 + 行号，完整追溯路径
       - 轮转：单文件最大 10MB，保留最近 5 个备份
       - 用途：生产期问题排查、运行记录归档

日志目录：
    默认在项目根目录下创建 logs/ 文件夹。
    路径基于本文件位置动态计算，无需手动配置。

与 config.py 的关系：
    - 读取 config.DEBUG 决定控制台日志级别：
        DEBUG=True  → 控制台输出 DEBUG 级别（开发模式，信息最详细）
        DEBUG=False → 控制台输出 INFO 级别（生产模式，只看关键信息）
    - 文件日志始终是 DEBUG 级别，不受 DEBUG 开关影响。

使用方法（在其他模块里）：
    from logger import get_logger
    logger = get_logger(__name__)

    # 之后替换原来的 print() 调用：
    # 旧：print(f"[summarizer] L1 摘要已写入，id = {l1_id}")
    # 新：logger.info(f"L1 摘要已写入，id = {l1_id}")

    # 各级别的使用场景：
    logger.debug("细节信息，开发期调试用，生产环境控制台不显示")
    logger.info("正常流程节点，如'session 已关闭'、'L1 写入成功'")
    logger.warning("非致命异常，流程可继续，如'关键词写回失败'")
    logger.error("操作失败，需要关注，如'模型调用失败'")
    logger.critical("严重错误，系统可能无法继续运行")

    # 记录异常详情（自动附带 traceback）：
    try:
        risky_operation()
    except Exception as e:
        logger.exception(f"操作失败：{e}")   # exception() 自动附带完整堆栈
"""

import logging
import logging.handlers
import sys
from pathlib import Path


# =============================================================================
# 常量配置
# =============================================================================

# 日志目录：项目根目录下的 logs/ 文件夹
# Path(__file__).parent 取本文件所在目录（即项目根目录）
LOG_DIR = Path(__file__).parent / "logs"

# 日志文件名
LOG_FILE = LOG_DIR / "coral.log"

# 文件日志轮转参数
LOG_MAX_BYTES   = 10 * 1024 * 1024   # 单文件最大 10MB
LOG_BACKUP_COUNT = 5                  # 最多保留 5 个备份文件

# 控制台日志格式：简洁，便于开发期快速阅读
# 示例输出：2026-03-22 14:30:05 [summarizer] INFO: L1 摘要已写入，id = 3
CONSOLE_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
CONSOLE_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 文件日志格式：详细，包含文件名和行号，便于精确定位问题
# 示例输出：2026-03-22 14:30:05,123 [summarizer] INFO summarizer.py:142 - L1 摘要已写入
FILE_FORMAT = "%(asctime)s [%(name)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# 内部状态：已初始化的 logger 集合，避免重复添加 handler
# =============================================================================

# Python logging 模块本身有层级缓存机制，
# 但我们额外维护一个集合，用来判断某个 logger 是否已经配置过 handler，
# 防止多次调用 get_logger(name) 时重复添加 handler 导致日志重复输出。
_initialized_loggers: set[str] = set()


# =============================================================================
# 核心函数
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    获取一个已配置好双通道（控制台 + 文件）的 Logger。

    首次调用时会：
      1. 创建 logs/ 目录（已存在则跳过）
      2. 配置控制台 Handler
      3. 配置文件轮转 Handler
      4. 将两个 Handler 绑定到对应 logger

    后续对同一 name 的调用直接返回缓存的 logger，不重复添加 handler。

    参数：
        name — logger 的名称，推荐传入 __name__（即模块名）。
               传入 __name__ 后，日志条目里会显示模块名，便于定位来源。
               例如在 summarizer.py 里传入 __name__ 会得到 "summarizer"。

    返回：
        logging.Logger — 配置好的 logger 实例

    使用示例：
        # 在模块顶部（import 区域后）写一次，模块内全部使用这个 logger
        from logger import get_logger
        logger = get_logger(__name__)
    """
    # 如果已经初始化过，直接返回缓存的 logger，不重复配置
    logger = logging.getLogger(name)
    if name in _initialized_loggers:
        return logger

    # ------------------------------------------------------------------
    # 确定控制台日志级别
    # 从 config 读取 DEBUG 开关，读取失败时默认按 INFO 处理
    # ------------------------------------------------------------------
    try:
        from config import DEBUG
        console_level = logging.DEBUG if DEBUG else logging.INFO
    except ImportError:
        # config.py 不存在时（单独运行 logger.py 做测试），默认 INFO
        console_level = logging.INFO

    # ------------------------------------------------------------------
    # 设置 logger 本身的最低级别为 DEBUG，
    # 确保 DEBUG 消息能"流到" handler，由 handler 各自决定是否输出
    # ------------------------------------------------------------------
    logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Handler 1：控制台输出（StreamHandler → stdout）
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(fmt=CONSOLE_FORMAT, datefmt=CONSOLE_DATEFMT)
    )

    # ------------------------------------------------------------------
    # Handler 2：文件轮转输出（RotatingFileHandler）
    # 创建日志目录（不存在时自动创建，已存在时跳过）
    # ------------------------------------------------------------------
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename    = LOG_FILE,
        maxBytes    = LOG_MAX_BYTES,
        backupCount = LOG_BACKUP_COUNT,
        encoding    = "utf-8",          # 显式指定 UTF-8，避免 Windows 下中文乱码
    )
    file_handler.setLevel(logging.DEBUG)   # 文件始终记录最详细的级别
    file_handler.setFormatter(
        logging.Formatter(fmt=FILE_FORMAT, datefmt=FILE_DATEFMT)
    )

    # ------------------------------------------------------------------
    # 将两个 Handler 绑定到 logger
    # ------------------------------------------------------------------
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # 阻止日志向上传播到 root logger，避免在某些框架（如 uvicorn）中
    # 出现重复输出的问题
    # ------------------------------------------------------------------
    logger.propagate = False

    # 标记为已初始化，下次调用直接复用
    _initialized_loggers.add(name)

    return logger


# =============================================================================
# 直接运行此文件时：验证日志配置是否正常
# =============================================================================

if __name__ == "__main__":
    print("=== logger.py 验证测试 ===\n")

    # 获取两个不同名称的 logger，验证互不干扰
    log_a = get_logger("summarizer")
    log_b = get_logger("merger")

    print("--- 测试一：各级别输出 ---")
    log_a.debug("这是 DEBUG 消息（生产模式下控制台不显示）")
    log_a.info("这是 INFO 消息（正常流程节点）")
    log_a.warning("这是 WARNING 消息（非致命异常）")
    log_a.error("这是 ERROR 消息（操作失败）")
    log_a.critical("这是 CRITICAL 消息（严重错误）")

    print("\n--- 测试二：不同模块的 logger 互不干扰 ---")
    log_b.info("来自 merger 模块的日志")
    log_a.info("来自 summarizer 模块的日志")

    print("\n--- 测试三：重复获取同一 logger 不会重复输出 ---")
    log_a2 = get_logger("summarizer")   # 应该复用缓存，不重复添加 handler
    log_a2.info("如果这条消息只出现一次，说明 handler 没有重复添加 ✓")

    print("\n--- 测试四：异常详情记录 ---")
    try:
        _ = 1 / 0
    except ZeroDivisionError as e:
        log_a.exception(f"捕获到异常（exception() 会自动附带 traceback）：{e}")

    print(f"\n验证完成。日志文件已写入：{LOG_FILE}")
    print("检查 logs/coral.log 确认文件日志是否正常。")
