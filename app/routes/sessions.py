"""
app/routes/sessions.py — Session 查询接口

包含接口：
    GET /api/sessions                       — 获取所有 session 列表（含统计）
    GET /api/sessions/{session_id}/messages — 获取指定 session 的消息列表
"""

import sqlite3

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ramaria.config import DB_PATH
from ramaria.storage.database import get_messages, get_session

from logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _get_all_sessions_with_stats() -> list:
    """
    获取所有 session 的摘要列表，含统计信息。

    每条记录包含：
        id                  — session ID
        started_at          — session 开始时间
        ended_at            — session 结束时间（进行中则为 null）
        message_count       — 本 session 的消息总数
        last_message_at     — 最后一条消息的时间
        last_message_preview — 最后一条消息的前 80 字符预览

    按最后活动时间倒序排列（活跃的在最前）。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sql = """
        SELECT
          s.id, s.started_at, s.ended_at,
          (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)
              AS message_count,
          (SELECT MAX(m.created_at) FROM messages m WHERE m.session_id = s.id)
              AS last_message_at,
          (SELECT m.content FROM messages m WHERE m.session_id = s.id
           ORDER BY m.created_at DESC LIMIT 1)
              AS last_message_preview
        FROM sessions s
        ORDER BY COALESCE(
          (SELECT MAX(m.created_at) FROM messages m WHERE m.session_id = s.id),
          s.started_at
        ) DESC
    """

    cursor.execute(sql)
    results = []
    for row in cursor.fetchall():
        preview = row["last_message_preview"]
        if preview and len(preview) > 80:
            preview = preview[:80] + "…"

        results.append({
            "id":                   row["id"],
            "started_at":           row["started_at"],
            "ended_at":             row["ended_at"],
            "message_count":        row["message_count"],
            "last_message_at":      row["last_message_at"],
            "last_message_preview": preview,
        })

    conn.close()
    return results


@router.get("/api/sessions")
async def api_get_sessions():
    """获取所有 session 的摘要列表。"""
    try:
        sessions = _get_all_sessions_with_stats()
        return JSONResponse(sessions)
    except Exception as e:
        logger.error(f"获取 session 列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部错误: {e}")


@router.get("/api/sessions/{session_id}/messages")
async def api_get_session_messages(session_id: int):
    """
    获取指定 session 的所有消息。

    异常：
        404 — session 不存在
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = get_messages(session_id)
    result = [
        {
            "role":       row["role"],
            "content":    row["content"],
            "created_at": row["created_at"],
        }
        for row in messages
    ]

    return JSONResponse(result)