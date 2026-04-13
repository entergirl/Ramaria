"""
app/routes/memory.py — 记忆可视化接口

包含接口：
    GET    /api/memory/l1       — L1 摘要列表（分页，含实时衰减值 R）
    GET    /api/memory/l2       — L2 聚合摘要列表（分页）
    GET    /api/memory/profile  — L3 用户画像（复用已有逻辑）
    DELETE /api/memory/l1/{id}  — 删除单条 L1（SQLite + Chroma 双向删除）
    DELETE /api/memory/l2/{id}  — 删除单条 L2（SQLite + Chroma 双向删除）

设计说明：
    · 衰减值 R 在请求时实时计算，确保准确性
    · 分页在数据库层用 LIMIT/OFFSET 实现，不拉全量数据
    · 删除操作先删 Chroma 向量索引，再删 SQLite 记录
      顺序很重要：若 SQLite 先删，Chroma 孤儿索引无法溯源清理
    · Chroma 删除失败时记录警告但不阻断 SQLite 删除，
      避免索引抖动导致记忆数据无法删除
"""

import math
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ramaria.config import (
    MEMORY_DECAY_S_L1,
    MEMORY_DECAY_S_L2,
    MEMORY_DECAY_ENABLE_ACCESS_BOOST,
    MEMORY_DECAY_RECENT_BOOST_DAYS,
    MEMORY_DECAY_RECENT_BOOST_FLOOR,
    SALIENCE_DECAY_MULTIPLIER,
)
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# 内部工具：实时衰减值计算
# =============================================================================

def _calc_decay_r(
    created_at_str: str,
    decay_s: float,
    last_accessed_at_str: str | None = None,
    salience: float | None = None,
) -> float:
    """
    实时计算单条记忆的 Ebbinghaus 保留率 R，逻辑与 vector_store._calc_decay_factor 完全一致。

    此处复制而非直接引用，是为了避免 routes 层依赖 storage 层的私有函数（下划线前缀）。
    若 vector_store 的衰减公式未来发生变更，此处需同步修改。

    公式：R = exp(-t / S_adjusted)
        t           = 距 created_at 的天数
        S_adjusted  = decay_s × (1 + salience × SALIENCE_DECAY_MULTIPLIER)

    保底加成（MEMORY_DECAY_ENABLE_ACCESS_BOOST=True 时）：
        若 last_accessed_at 在 RECENT_BOOST_DAYS 天内，R = max(R, RECENT_BOOST_FLOOR)

    返回：
        float，保留 4 位小数，范围 (0, 1]
        解析时间失败时安全返回 1.0
    """
    now = datetime.now(timezone.utc)

    # 解析 created_at，计算距今天数
    try:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        t_days = max((now - created_at).total_seconds() / 86400, 0)
    except (ValueError, TypeError):
        logger.warning(f"无法解析 created_at={created_at_str!r}，返回 R=1.0")
        return 1.0

    # salience 加成稳定性系数（None 时默认 0.5，与 vector_store 行为一致）
    s_val = salience if salience is not None else 0.5
    s_val = max(0.0, min(1.0, s_val))
    decay_s_adjusted = decay_s * (1.0 + s_val * SALIENCE_DECAY_MULTIPLIER)

    # Ebbinghaus 公式
    R = math.exp(-t_days / decay_s_adjusted)

    # last_accessed_at 保底加成
    if MEMORY_DECAY_ENABLE_ACCESS_BOOST and last_accessed_at_str:
        try:
            last_accessed = datetime.fromisoformat(last_accessed_at_str)
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            days_since = (now - last_accessed).total_seconds() / 86400
            if days_since <= MEMORY_DECAY_RECENT_BOOST_DAYS:
                R = max(R, MEMORY_DECAY_RECENT_BOOST_FLOOR)
        except (ValueError, TypeError):
            pass  # 解析失败时静默忽略，不影响主衰减值

    return round(R, 4)


# =============================================================================
# 内部工具：SQLite 直接查询（分页）
# =============================================================================

def _query_l1_page(page: int, limit: int) -> tuple[list[dict], int]:
    """
    从 memory_l1 表分页查询，同时返回总记录数。

    参数：
        page  — 页码，从 1 开始
        limit — 每页条数

    返回：
        (rows_as_dicts, total_count)
    """
    import sqlite3
    from ramaria.config import DB_PATH

    offset = (page - 1) * limit

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 总数（用于前端分页控件）
    cursor.execute("SELECT COUNT(*) AS cnt FROM memory_l1")
    total = cursor.fetchone()["cnt"]

    # 分页数据，按生成时间降序（最新的在前）
    cursor.execute(
        """
        SELECT id, session_id, summary, keywords, time_period, atmosphere,
               valence, salience, absorbed, created_at, last_accessed_at
        FROM memory_l1
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows, total


def _query_l2_page(page: int, limit: int) -> tuple[list[dict], int]:
    """
    从 memory_l2 表分页查询，同时返回总记录数。
    """
    import sqlite3
    from ramaria.config import DB_PATH

    offset = (page - 1) * limit

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS cnt FROM memory_l2")
    total = cursor.fetchone()["cnt"]

    cursor.execute(
        """
        SELECT id, summary, keywords, period_start, period_end,
               created_at, last_accessed_at
        FROM memory_l2
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows, total


# =============================================================================
# 内部工具：删除操作（Chroma + SQLite 双向）
# =============================================================================

def _delete_l1_record(l1_id: int) -> None:
    """
    删除单条 L1 记录，顺序：先 Chroma → 后 SQLite。

    Chroma 删除失败只记警告，不阻断 SQLite 删除。
    SQLite 删除失败会抛出异常，由调用方处理。

    注意：此函数不检查记录是否存在，调用前应先查询确认。
    """
    import sqlite3
    from ramaria.config import DB_PATH

    # 第一步：删除 Chroma 向量索引
    try:
        from ramaria.storage.vector_store import _get_collection, _COLL_L1
        collection = _get_collection(_COLL_L1)
        collection.delete(ids=[str(l1_id)])
        logger.info(f"L1 Chroma 索引已删除，id={l1_id}")
    except Exception as e:
        # Chroma 删除失败不阻断后续 SQLite 删除
        # 常见情况：该条 L1 生成时向量化失败，索引本就不存在
        logger.warning(f"L1 Chroma 索引删除失败（可能本就不存在），id={l1_id} — {e}")

    # 第二步：删除 SQLite 记录
    # 级联关系说明：
    #   · l2_sources 表有外键引用 memory_l1.id，但 SQLite 默认外键约束关闭
    #   · database.py 的 _get_connection() 已开启 foreign_keys=ON
    #   · 若此 L1 已被 L2 吸收，l2_sources 中存在引用，删除会触发外键约束失败
    #   · 这里用独立连接并开启外键约束，保证一致性
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("DELETE FROM memory_l1 WHERE id = ?", (l1_id,))
        conn.commit()
        logger.info(f"L1 SQLite 记录已删除，id={l1_id}")
    except sqlite3.IntegrityError as e:
        conn.rollback()
        # 外键约束失败：该 L1 已被 L2 引用，需要先删除对应 L2 或解除引用
        raise HTTPException(
            status_code=409,
            detail=(
                f"L1 id={l1_id} 已被 L2 摘要引用（l2_sources 表），"
                "请先删除对应的 L2 记录，或在 l2_sources 中解除关联后再删除。"
            ),
        ) from e
    finally:
        conn.close()


def _delete_l2_record(l2_id: int) -> None:
    """
    删除单条 L2 记录，顺序：先 Chroma → 后 SQLite。

    同时删除 l2_sources 中的关联行（告知被吸收的 L1 可以独立存在），
    但不修改 memory_l1.absorbed 字段——已吸收的标记保留，
    避免这些 L1 被重新触发合并。

    Chroma 删除失败只记警告，不阻断 SQLite 删除。
    """
    import sqlite3
    from ramaria.config import DB_PATH

    # 第一步：删除 Chroma 向量索引
    try:
        from ramaria.storage.vector_store import _get_collection, _COLL_L2
        collection = _get_collection(_COLL_L2)
        collection.delete(ids=[str(l2_id)])
        logger.info(f"L2 Chroma 索引已删除，id={l2_id}")
    except Exception as e:
        logger.warning(f"L2 Chroma 索引删除失败，id={l2_id} — {e}")

    # 第二步：删除 SQLite 记录（同一事务内先删 l2_sources，再删 memory_l2）
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        # 先删关联表，解除外键约束
        conn.execute("DELETE FROM l2_sources WHERE l2_id = ?", (l2_id,))
        # 再删主记录
        conn.execute("DELETE FROM memory_l2 WHERE id = ?", (l2_id,))
        conn.commit()
        logger.info(f"L2 SQLite 记录及 l2_sources 关联行已删除，id={l2_id}")
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"L2 SQLite 删除失败：{e}",
        ) from e
    finally:
        conn.close()


# =============================================================================
# GET /api/memory/l1 — L1 列表（分页 + 实时衰减值）
# =============================================================================

@router.get("/api/memory/l1")
async def api_get_memory_l1(
    page:  int = Query(default=1,  ge=1,   description="页码，从 1 开始"),
    limit: int = Query(default=20, ge=1, le=100, description="每页条数，最大 100"),
):
    """
    返回 L1 摘要列表，每条附带实时计算的衰减值 R。

    返回格式：
        {
            "total":    int,        # 总记录数
            "page":     int,        # 当前页码
            "limit":    int,        # 每页条数
            "items": [
                {
                    "id":               int,
                    "session_id":       int,
                    "summary":          str,
                    "keywords":         str | null,
                    "time_period":      str | null,
                    "atmosphere":       str | null,
                    "valence":          float,
                    "salience":         float,
                    "absorbed":         int,      # 0=未吸收，1=已被L2吸收
                    "created_at":       str,      # ISO 8601
                    "last_accessed_at": str | null,
                    "decay_r":          float,    # 实时衰减保留率，范围(0,1]
                },
                ...
            ]
        }
    """
    try:
        rows, total = _query_l1_page(page, limit)
    except Exception as e:
        logger.error(f"L1 分页查询失败 — {e}")
        raise HTTPException(status_code=500, detail=f"数据库查询失败：{e}")

    # 为每条记录计算实时衰减值
    for row in rows:
        row["decay_r"] = _calc_decay_r(
            created_at_str       = row.get("created_at", ""),
            decay_s              = MEMORY_DECAY_S_L1,
            last_accessed_at_str = row.get("last_accessed_at"),
            salience             = row.get("salience"),
        )

    return JSONResponse({
        "total": total,
        "page":  page,
        "limit": limit,
        "items": rows,
    })


# =============================================================================
# GET /api/memory/l2 — L2 列表（分页）
# =============================================================================

@router.get("/api/memory/l2")
async def api_get_memory_l2(
    page:  int = Query(default=1,  ge=1,   description="页码，从 1 开始"),
    limit: int = Query(default=10, ge=1, le=100, description="每页条数，最大 100"),
):
    """
    返回 L2 聚合摘要列表。

    L2 没有 salience 字段，衰减值 R 使用默认 salience=0.5 计算。

    返回格式：
        {
            "total":    int,
            "page":     int,
            "limit":    int,
            "items": [
                {
                    "id":               int,
                    "summary":          str,
                    "keywords":         str | null,
                    "period_start":     str,      # 覆盖时间段起点
                    "period_end":       str,      # 覆盖时间段终点
                    "created_at":       str,
                    "last_accessed_at": str | null,
                    "decay_r":          float,
                },
                ...
            ]
        }
    """
    try:
        rows, total = _query_l2_page(page, limit)
    except Exception as e:
        logger.error(f"L2 分页查询失败 — {e}")
        raise HTTPException(status_code=500, detail=f"数据库查询失败：{e}")

    for row in rows:
        row["decay_r"] = _calc_decay_r(
            created_at_str       = row.get("created_at", ""),
            decay_s              = MEMORY_DECAY_S_L2,
            last_accessed_at_str = row.get("last_accessed_at"),
            salience             = None,   # L2 无 salience，使用默认值 0.5
        )

    return JSONResponse({
        "total": total,
        "page":  page,
        "limit": limit,
        "items": rows,
    })


# =============================================================================
# GET /api/memory/profile — L3 用户画像
# =============================================================================

@router.get("/api/memory/profile")
async def api_get_memory_profile():
    """
    返回当前生效的 L3 用户画像，按六大板块组织。

    直接复用 database.get_current_profile()，不做额外处理。

    返回格式：
        {
            "basic_info":      str | null,
            "personal_status": str | null,
            "interests":       str | null,
            "social":          str | null,
            "history":         str | null,
            "recent_context":  str | null,
        }
    """
    from ramaria.storage.database import get_current_profile
    from constants import PROFILE_FIELDS

    try:
        profile = get_current_profile()
    except Exception as e:
        logger.error(f"L3 画像查询失败 — {e}")
        raise HTTPException(status_code=500, detail=f"数据库查询失败：{e}")

    # 确保六个板块都有 key，未填写的返回 null 而非缺失
    result = {field: profile.get(field) for field in PROFILE_FIELDS}
    return JSONResponse(result)


# =============================================================================
# DELETE /api/memory/l1/{id} — 删除单条 L1
# =============================================================================

@router.delete("/api/memory/l1/{l1_id}")
async def api_delete_memory_l1(l1_id: int):
    """
    删除单条 L1 记忆（SQLite + Chroma 双向删除）。

    注意：若该 L1 已被 L2 引用（l2_sources 表），删除会返回 409 冲突，
    需要先删除对应 L2 才能删除此 L1。

    返回：
        200 {"status": "ok", "deleted_id": l1_id}
        404 记录不存在
        409 被 L2 引用，无法删除
        500 内部错误
    """
    from ramaria.storage.database import get_l1_by_id

    # 先确认记录存在
    row = get_l1_by_id(l1_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"L1 id={l1_id} 不存在")

    # 执行双向删除（内部已处理 409/500 场景）
    _delete_l1_record(l1_id)

    return JSONResponse({"status": "ok", "deleted_id": l1_id})


# =============================================================================
# DELETE /api/memory/l2/{id} — 删除单条 L2
# =============================================================================

@router.delete("/api/memory/l2/{l2_id}")
async def api_delete_memory_l2(l2_id: int):
    """
    删除单条 L2 摘要（SQLite + Chroma 双向删除，同时清理 l2_sources 关联行）。

    注意：删除 L2 不会修改其来源 L1 的 absorbed 标记，
    这些 L1 仍保持 absorbed=1 状态，不会被重新触发合并。
    如需重新合并，需手动将 absorbed 改回 0，或直接触发 check_and_merge()。

    返回：
        200 {"status": "ok", "deleted_id": l2_id}
        404 记录不存在
        500 内部错误
    """
    import sqlite3
    from ramaria.config import DB_PATH

    # 先确认记录存在
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM memory_l2 WHERE id = ?", (l2_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"L2 id={l2_id} 不存在")

    # 执行双向删除
    _delete_l2_record(l2_id)

    return JSONResponse({"status": "ok", "deleted_id": l2_id})