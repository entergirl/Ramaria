"""
importer/qq_importer.py — 历史消息数据库写入层
=====================================================================

职责：
    接收 qq_parser.py 解析出的 session 列表，将消息写入数据库的 L0 层。
    复用 database.py 的接口，不触发 L1 摘要生成。

    写入流程（每个 session）：
        1. 调用 new_session_with_time()        — 创建历史 session（保留原始时间）
        2. 逐条调用 save_message_with_fingerprint() — 写入消息（同时写入指纹字段）
        3. 调用 close_session_with_time()      — 关闭 session（保留原始时间）
        4. 不触发 L1 摘要生成                 — L1 由后续批量流程处理

    为什么不立即触发 L1：
        历史消息可能有几千条，导入时立即触发摘要会连续调用本地模型上百次，
        时间很长且难以中断。设计上分离"导入 L0"和"生成 L1"两个步骤，
        让用户可以先确认 L0 数据正确，再批量触发 L1。

    指纹写入：
        每条消息的 import_fingerprint 字段在写入时同步存入数据库，
        供后续重复导入预检使用。这要求 messages 表已经执行过迁移：
            ALTER TABLE messages ADD COLUMN import_fingerprint TEXT;
        database_patch.py 的 get_all_message_fingerprints() 会在预检时
        自动检查此列是否存在，未迁移时给出友好错误提示。

依赖 database.py 新增的函数（见 database_patch.py）：
    new_session_with_time(started_at_iso)
    save_message_with_fingerprint(session_id, role, content, created_at, fingerprint)
    close_session_with_time(session_id, ended_at_iso)

使用方法：
    from importer.qq_importer import import_sessions_to_db

    stats = import_sessions_to_db(result.parsed_sessions)
    print(f"写入 {stats['sessions_written']} 个 session，共 {stats['messages_written']} 条消息")
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 里
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from importer.qq_parser import ParsedMessage
from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 数据库写入（对外接口）
# =============================================================================

def import_sessions_to_db(
    parsed_sessions: list,   # list[list[ParsedMessage]]
) -> dict:
    """
    将解析后的 session 列表批量写入数据库（仅写入 L0，不触发 L1 摘要）。

    每个 session 的写入是独立的事务，单个 session 失败不会中止整体导入，
    失败信息记录到返回值的 failed_details 里。

    参数：
        parsed_sessions — parse_qq_export() 返回的 parsed_sessions 字段
                          每个元素是 list[ParsedMessage]，已过滤重复

    返回：
        dict，包含：
            sessions_written  — 成功写入的 session 数
            messages_written  — 成功写入的消息总数
            sessions_failed   — 写入失败的 session 数
            failed_details    — 失败 session 的错误信息列表
    """
    # 延迟导入，避免模块加载时就连接数据库
    from database import (
        new_session_with_time,
        save_message_with_fingerprint,
        close_session_with_time,
    )

    stats = {
        "sessions_written": 0,
        "messages_written": 0,
        "sessions_failed":  0,
        "failed_details":   [],
    }

    total = len(parsed_sessions)
    logger.info(f"开始写入 {total} 个历史 session")

    for idx, session_msgs in enumerate(parsed_sessions, start=1):
        if not session_msgs:
            continue

        # 取第一条和最后一条的时间戳作为 session 的开始和结束时间
        started_at = session_msgs[0].timestamp
        ended_at   = session_msgs[-1].timestamp

        try:
            # 1. 创建历史 session（用原始时间戳，不用 datetime.now()）
            session_id = new_session_with_time(started_at)

            # 2. 逐条写入消息，同时写入指纹
            msg_count = 0
            for msg in session_msgs:
                save_message_with_fingerprint(
                    session_id  = session_id,
                    role        = msg.role,
                    content     = msg.content,
                    created_at  = msg.timestamp,
                    fingerprint = msg.fingerprint,
                )
                msg_count += 1

            # 3. 关闭 session
            close_session_with_time(session_id, ended_at)

            stats["sessions_written"] += 1
            stats["messages_written"] += msg_count

            # 每10个 session 打印一次进度
            if idx % 10 == 0 or idx == total:
                logger.info(f"进度：{idx}/{total} 个 session 已写入")

        except Exception as e:
            # 单个 session 失败不中止整体，记录错误后继续
            stats["sessions_failed"] += 1
            error_info = f"session {idx}（{started_at[:10]}）：{e}"
            stats["failed_details"].append(error_info)
            logger.error(f"写入失败 — {error_info}")

    logger.info(
        f"写入完成：成功 {stats['sessions_written']} 个 session，"
        f"共 {stats['messages_written']} 条消息，"
        f"失败 {stats['sessions_failed']} 个"
    )

    return stats
