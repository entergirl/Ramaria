"""
src/ramaria/importer/qq/importer.py — 历史消息数据库写入层

职责：
    接收 parser.py 解析出的 session 列表，将消息写入数据库的 L0 层。
    复用 database.py 的接口，不触发 L1 摘要生成。

    写入流程（每个 session）：
        1. 调用 new_session_with_time()
        2. 逐条调用 save_message_with_fingerprint()
        3. 调用 close_session_with_time()
        4. 不触发 L1 摘要生成（L1 由后续批量流程处理）

"""

from ramaria.importer.qq.parser import ParsedMessage
from logger import get_logger

logger = get_logger(__name__)


def import_sessions_to_db(
    parsed_sessions: list,  # list[list[ParsedMessage]]
) -> dict:
    """
    将解析后的 session 列表批量写入数据库（仅写入 L0，不触发 L1 摘要）。

    每个 session 的写入是独立的事务，单个 session 失败不会中止整体导入。

    参数：
        parsed_sessions — parser.parse_qq_export() 返回的 parsed_sessions 字段

    返回：
        dict — {
            "sessions_written": int,
            "messages_written": int,
            "sessions_failed":  int,
            "failed_details":   list[str],
        }
    """
    from ramaria.storage.database import (
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

        started_at = session_msgs[0].timestamp
        ended_at   = session_msgs[-1].timestamp

        try:
            session_id = new_session_with_time(started_at)

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

            close_session_with_time(session_id, ended_at)

            stats["sessions_written"] += 1
            stats["messages_written"] += msg_count

            if idx % 10 == 0 or idx == total:
                logger.info(f"进度：{idx}/{total} 个 session 已写入")

        except Exception as e:
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