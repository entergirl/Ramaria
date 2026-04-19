"""
app/routes/sessions.py — Session 查询接口

包含接口：
    GET /api/sessions                       — 获取所有 session 列表（含统计）
    GET /api/sessions/{session_id}/messages — 获取指定 session 的消息列表
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ramaria.storage.database import (
    get_all_sessions_with_stats,
    get_messages,
    get_session,
)

from ramaria.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.get("/api/sessions")
async def api_get_sessions():
    """获取所有 session 的摘要列表。"""
    try:
        sessions = get_all_sessions_with_stats()
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
