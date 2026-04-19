"""
src/ramaria/storage/database.py — 数据库操作层

封装所有与 assistant.db 的读写操作，其他模块通过调用这里的函数访问数据库。
不包含建表逻辑（→ scripts/init_db.py）。

设计原则：
    · 每个函数只做一件事，职责单一
    · 每个函数自己开连接、自己关连接，不依赖外部传入的连接对象
    · 所有时间统一使用 UTC ISO 8601 格式存储
    · _get_connection 是模块私有函数，不对外暴露
    · 外部模块一律通过公开函数访问数据

"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from ramaria.config import DB_PATH, TIME_PERIOD_OPTIONS
from ramaria.logger import get_logger

logger = get_logger(__name__)

_LAYER_TABLE_MAP = {
    "l1": "memory_l1",
    "l2": "memory_l2",
}


# =============================================================================
# 内部工具函数（模块私有，不对外暴露）
# =============================================================================

def _get_connection():
    """
    获取数据库连接（内部使用）。
    - row_factory 让查询结果支持列名访问，如 row["content"]
    - 开启外键约束（SQLite 默认关闭）

    注意：此函数为模块私有，外部模块不应直接 import 使用。
          模块内部也应优先使用 _db_conn() 上下文管理器。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _db_conn():
    """
    数据库连接上下文管理器（内部使用）。

    确保无论正常退出还是异常退出，连接都会被关闭，
    避免 SQLite 文件锁残留（Windows 环境尤其重要）。

    使用示例：
        def some_db_function():
            with _db_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
                return cursor.fetchall()
    """
    conn = _get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _now() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串（内部使用）。"""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Sessions 表操作
# =============================================================================

def new_session() -> int:
    """
    创建一个新 session，返回新 session 的 id。

    调用时机：
        用户发来消息，且当前没有活跃 session 时，由 session_manager 调用。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (started_at) VALUES (?)",
            (_now(),)
        )
        conn.commit()
        return cursor.lastrowid


def close_session(session_id: int) -> None:
    """
    关闭一个 session，将 ended_at 写入当前时间。

    调用时机：空闲检测超时后，由 session_manager 调用。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (_now(), session_id)
        )
        conn.commit()


def get_session(session_id: int):
    """
    取出一个 session 的完整信息。

    返回：
        sqlite3.Row — 包含 id / started_at / ended_at；不存在时返回 None
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return cursor.fetchone()


def get_active_sessions() -> list:
    """
    取出所有尚未关闭的 session（ended_at 为 NULL）。

    用途：
        后端启动时检查是否有遗留的未关闭 session。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY started_at ASC"
        )
        return cursor.fetchall()


# =============================================================================
# Messages 表操作
# =============================================================================

def save_message(session_id: int, role: str, content: str) -> int:
    """
    保存一条消息到 messages 表（L0 原始记录）。

    参数：
        session_id — 当前 session 的 id
        role       — 发言方，只能是 "user" 或 "assistant"
        content    — 消息正文文本

    返回：
        int — 新插入消息的 id
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role 只能是 'user' 或 'assistant'，收到：{role!r}")

    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now())
        )
        conn.commit()
        return cursor.lastrowid


def get_messages(session_id: int) -> list:
    """
    取出某个 session 的全部消息，按时间升序排列。

    返回：
        list[sqlite3.Row]，可用 row["role"]、row["content"] 访问。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        )
        return cursor.fetchall()


def get_messages_as_dicts(session_id: int) -> list[dict]:
    """
    取出某个 session 的全部消息，以字典列表形式返回。

    返回格式与 OpenAI / LM Studio API 的 messages 参数一致，可直接拼进请求体。

    返回格式示例：
        [
            {"role": "user",      "content": "你好"},
            {"role": "assistant", "content": "你好！"}
        ]
    """
    rows = get_messages(session_id)
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def get_last_message_time(session_id: int) -> str | None:
    """
    取出某个 session 最后一条消息的时间戳字符串。

    返回：
        str  — ISO 8601 时间戳
        None — 该 session 没有任何消息时返回 None

    用途：空闲检测时判断距离上次消息过了多长时间。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT created_at FROM messages "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        )
        row = cursor.fetchone()
        return row["created_at"] if row else None


def update_message_time_for_test(session_id: int, fake_time_str: str) -> None:
    """
    将某 session 所有消息的时间戳改为指定值。

    ⚠️  此函数仅限测试使用，不应在正式业务代码中调用。
        函数名后缀 _for_test 是有意为之的警示标记。
    """
    with _db_conn() as conn:
        conn.execute(
            "UPDATE messages SET created_at = ? WHERE session_id = ?",
            (fake_time_str, session_id)
        )
        conn.commit()


# =============================================================================
# memory_l1 表操作
# =============================================================================

def save_l1_summary(
    session_id: int,
    summary: str,
    keywords: str | None,
    time_period: str | None,
    atmosphere: str | None,
    valence: float = 0.0,
    salience: float = 0.5,
    created_at: str | None = None,
) -> int:
    """
    将一条 L1 摘要写入 memory_l1 表。

    参数：
        session_id  — 对应的 session id
        summary     — 一句话摘要（第三人称，只记结论，不超过50字）
        keywords    — 名词标签字符串，逗号分隔，如 "项目,后端,FastAPI"
        time_period — 时间段标签，六选一：清晨/上午/下午/傍晚/夜间/深夜
        atmosphere  — 对话氛围，四字以内，如 "专注高效"
        valence     — 情绪效价，五档：-1.0/-0.5/0.0/0.5/1.0，默认中性 0.0
        salience    — 情感显著性，五档：0.0/0.25/0.5/0.75/1.0，默认中等 0.5
        created_at  — 可选，摘要的基准时间（ISO 8601 字符串）。
                      正常对话路径不传此参数，使用 _now()（摘要生成时刻）。
                      历史数据导入路径传入 session 的原始结束时间，
                      确保衰减计算和前端显示反映真实的对话发生时间，
                      而非批处理运行时间。

    返回：
        int — 新插入 L1 记录的 id
    """
    if time_period not in TIME_PERIOD_OPTIONS:
        logger.warning(f"time_period 值 {time_period!r} 不在合法列表内，已置为 None")
        time_period = None

    # created_at 未传入时使用当前时间（正常对话路径）
    # 传入时使用调用方指定的时间（历史导入路径，保留原始对话时间）
    record_time = created_at if created_at is not None else _now()

    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory_l1
                (session_id, summary, keywords, time_period, atmosphere,
                 valence, salience, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, summary, keywords, time_period, atmosphere,
             valence, salience, record_time)
        )
        conn.commit()
        return cursor.lastrowid


def get_l1_by_id(l1_id: int):
    """
    按主键查询一条 L1 摘要记录。

    返回：
        sqlite3.Row — 完整的 L1 记录行；id 不存在时返回 None
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory_l1 WHERE id = ?", (l1_id,))
        return cursor.fetchone()


def get_l1_salience(l1_id: int) -> float:
    """
    快速查询单条 L1 记录的 salience 值。

    返回：
        float — salience 值；记录不存在或字段为 NULL 时返回默认值 0.5
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT salience FROM memory_l1 WHERE id = ?", (l1_id,))
        row = cursor.fetchone()

    if row is None or row["salience"] is None:
        return 0.5
    return float(row["salience"])


def get_unabsorbed_l1(limit: int | None = None) -> list:
    """
    取出所有尚未被 L2 吸收的 L1 摘要，按生成时间升序排列。

    参数：
        limit — 最多返回多少条；不传则返回全部
    """
    sql    = "SELECT * FROM memory_l1 WHERE absorbed = 0 ORDER BY created_at ASC"
    params = ()
    if limit:
        sql   += " LIMIT ?"
        params = (limit,)

    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()


def get_all_l1(conn: sqlite3.Connection | None = None) -> list:
    """
    返回 memory_l1 表中的全部记录。

    用途：vector_store.rebuild_all_indexes() 重建 L1 向量索引时调用。

    参数：
        conn — 可选的外部数据库连接。
               传入时复用该连接（调用方负责连接的生命周期），不额外开关连接。
               不传时内部自己开关连接（原有行为，向后兼容）。

    v0.4.0 变更：新增 conn 可选参数，供 BM25 rebuild 路径复用连接，
               减少 BM25 重建时的重复开关连接开销。
    """
    sql = "SELECT id, summary, keywords, session_id, created_at FROM memory_l1"

    if conn is not None:
        # 复用外部连接：直接查询，不关闭连接
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    else:
        # 原有行为：自己管理连接
        with _db_conn() as _conn:
            cursor = _conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()


def get_all_l2(conn: sqlite3.Connection | None = None) -> list:
    """
    返回 memory_l2 表中的全部记录。

    用途：vector_store.rebuild_all_indexes() 重建 L2 向量索引时调用。

    参数：
        conn — 可选的外部数据库连接，含义同 get_all_l1()。
    """
    sql = "SELECT id, summary, keywords, period_start, period_end FROM memory_l2"

    if conn is not None:
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    else:
        with _db_conn() as _conn:
            cursor = _conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()


def get_all_session_ids() -> list[int]:
    """
    返回 messages 表中全部去重后的 session_id 列表。

    用途：vector_store.rebuild_all_indexes() 重建 L0 向量索引时调用。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM messages")
        rows = cursor.fetchall()
    return [row["session_id"] for row in rows]


def mark_l1_absorbed(l1_ids: list[int]) -> None:
    """将一批 L1 记录标记为已被 L2 吸收（absorbed = 1）。"""
    if not l1_ids:
        return
    placeholders = ",".join("?" * len(l1_ids))
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE memory_l1 SET absorbed = 1 WHERE id IN ({placeholders})",
            l1_ids
        )
        conn.commit()


# =============================================================================
# memory_l2 表操作
# =============================================================================

def save_l2_summary(
    summary: str,
    keywords: str | None,
    period_start: str,
    period_end: str,
    l1_ids: list[int],
) -> int:
    """
    将一条 L2 摘要写入 memory_l2 表，同时写入 l2_sources 关联表。
    两步写入包裹在同一个事务里，保证原子性。

    参数：
        summary      — 合并后的摘要文本
        keywords     — 关键词字符串，逗号分隔
        period_start — 覆盖时间段起点（ISO 8601）
        period_end   — 覆盖时间段终点（ISO 8601）
        l1_ids       — 参与合并的 L1 id 列表

    返回：
        int — 新插入 L2 记录的 id
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO memory_l2
                    (summary, keywords, period_start, period_end, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (summary, keywords, period_start, period_end, _now())
            )
            l2_id = cursor.lastrowid

            l2_source_rows = [(l2_id, l1_id) for l1_id in l1_ids]
            cursor.executemany(
                "INSERT OR IGNORE INTO l2_sources (l2_id, l1_id) VALUES (?, ?)",
                l2_source_rows
            )

            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"save_l2_summary 事务失败，已回滚 — {e}")
            raise

    return l2_id


def get_recent_l2(limit: int = 3) -> list:
    """
    取出最近几条 L2 摘要，按生成时间降序排列（最新的在前）。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM memory_l2 ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()


def get_latest_l1():
    """
    取出最近一条 L1 摘要（不限 absorbed 状态）。

    注意：
        此函数仅用于 prompt_builder 注入上下文，
        不应用于 conflict_checker / profile_manager 的 L1 校验。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM memory_l1 ORDER BY created_at DESC LIMIT 1"
        )
        return cursor.fetchone()


def get_recent_l1(limit: int = 3) -> list:
    """取出最近几条 L1 摘要，按生成时间降序排列。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, session_id, summary, keywords, time_period, atmosphere,
                   created_at, last_accessed_at
            FROM memory_l1
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()


def get_l1_by_session(session_id: int):
    """取出某个 session 对应的 L1 摘要（通常只有一条）。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM memory_l1 "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        )
        return cursor.fetchone()


# =============================================================================
# Settings 表操作
# =============================================================================

def get_setting(key: str, default: str | None = None) -> str | None:
    """读取一个配置项的值（字符串格式）。key 不存在时返回 default。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
    return row["value"] if row else default


def get_all_sessions_with_stats() -> list[dict]:
    """获取所有 session 的摘要列表，含消息数和最后一条消息预览。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
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
        )
        rows = cursor.fetchall()

    results = []
    for row in rows:
        preview = row["last_message_preview"]
        if preview and len(preview) > 80:
            preview = preview[:80] + "..."

        results.append({
            "id": row["id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "message_count": row["message_count"],
            "last_message_at": row["last_message_at"],
            "last_message_preview": preview,
        })

    return results


def set_setting(key: str, value: str) -> None:
    """
    写入或更新一个配置项。
    key 已存在时更新 value 和 updated_at；不存在时插入新行。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value      = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, str(value), _now())
        )
        conn.commit()


# =============================================================================
# user_profile 表操作
# =============================================================================

def get_current_profile() -> dict:
    """
    取出当前生效的完整用户画像，返回字典，key 为板块名。

    返回格式示例：
        {"basic_info": "烧酒，19岁，...", "interests": "..."}

    调用时机：构建 system prompt 时，将 L3 记忆注入对话上下文。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT field, content FROM user_profile
            WHERE is_current = 1 AND status = 'approved'
            """
        )
        rows = cursor.fetchall()
    return {row["field"]: row["content"] for row in rows}


def update_profile_field(
    field: str,
    new_content: str,
    source_l1_id: int | None = None,
) -> int:
    """
    更新用户画像的某个板块。

    采用追加写入策略：旧版本标记为历史（is_current=0），插入新版本（is_current=1）。
    历史版本永久保留，支持溯源和回滚。

    参数：
        field        — 板块名，如 "personal_status"
        new_content  — 新的板块内容文本
        source_l1_id — 触发此更新的 L1 id；手动更新时传 None

    返回：
        int — 新插入画像记录的 id
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        # 先把该字段旧的当前版本标记为历史
        cursor.execute(
            "UPDATE user_profile SET is_current = 0 "
            "WHERE field = ? AND is_current = 1",
            (field,)
        )
        # 插入新版本
        cursor.execute(
            """
            INSERT INTO user_profile
                (field, content, source_l1_id, status, is_current, updated_at)
            VALUES (?, ?, ?, 'approved', 1, ?)
            """,
            (field, new_content, source_l1_id, _now())
        )
        conn.commit()
        return cursor.lastrowid


# =============================================================================
# keyword_pool 表操作
# =============================================================================

def upsert_keywords(keywords: list[str]) -> None:
    """
    批量写入关键词到词典表。
    词条已存在时：use_count + 1，更新 last_used_at。
    词条不存在时：插入新行，use_count 从 1 开始。
    """
    if not keywords:
        return

    now = _now()
    with _db_conn() as conn:
        cursor = conn.cursor()
        for kw in keywords:
            kw = kw.strip()
            if not kw:
                continue
            cursor.execute(
                """
                INSERT INTO keyword_pool (keyword, use_count, last_used_at, created_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(keyword) DO UPDATE SET
                    use_count    = use_count + 1,
                    last_used_at = excluded.last_used_at
                """,
                (kw, now, now)
            )
        conn.commit()


def get_all_keywords() -> list[str]:
    """取出词典表中所有关键词，按使用频次降序排列。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT keyword FROM keyword_pool ORDER BY use_count DESC")
        rows = cursor.fetchall()
    return [row["keyword"] for row in rows]


def get_top_keywords(limit: int = 50) -> list[str]:
    """取出使用频次最高的前 N 个关键词。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT keyword FROM keyword_pool ORDER BY use_count DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
    return [row["keyword"] for row in rows]


# =============================================================================
# conflict_queue 表操作
# =============================================================================

def save_conflict(
    source_l1_id: int,
    field: str,
    old_content: str,
    new_content: str,
    conflict_desc: str,
) -> int:
    """
    将一条冲突记录写入 conflict_queue 表，状态初始为 pending。

    返回：int — 新插入冲突记录的 id
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conflict_queue
                (source_l1_id, field, old_content, new_content,
                 conflict_desc, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (source_l1_id, field, old_content, new_content, conflict_desc, _now())
        )
        conn.commit()
        return cursor.lastrowid


def get_pending_conflicts() -> list:
    """
    取出所有状态为 pending 的冲突记录，按写入时间升序排列。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM conflict_queue "
            "WHERE status = 'pending' ORDER BY created_at ASC"
        )
        return cursor.fetchall()


def resolve_conflict(conflict_id: int) -> None:
    """将一条冲突记录标记为 resolved（用户确认接受新内容）。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conflict_queue SET status = 'resolved', resolved_at = ? WHERE id = ?",
            (_now(), conflict_id)
        )
        conn.commit()


def ignore_conflict(conflict_id: int) -> None:
    """将一条冲突记录标记为 ignored（用户选择忽略）。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conflict_queue SET status = 'ignored', resolved_at = ? WHERE id = ?",
            (_now(), conflict_id)
        )
        conn.commit()


# =============================================================================
# 导入功能专用函数（历史消息导入，保留原始时间戳）
# =============================================================================

def new_session_with_time(started_at_iso: str) -> int:
    """
    创建一个历史 session，使用调用方指定的开始时间。
    用于历史消息导入，保留原始时间线。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (started_at) VALUES (?)",
            (started_at_iso,)
        )
        conn.commit()
        return cursor.lastrowid


def save_message_with_fingerprint(
    session_id: int,
    role: str,
    content: str,
    created_at: str,
    fingerprint: str,
) -> int:
    """
    保存一条历史消息，同时写入指纹字段，用于后续重复导入预检。

    参数：
        session_id  — 所属 session 的 id
        role        — "user" 或 "assistant"
        content     — 消息正文
        created_at  — 消息原始时间，UTC ISO 8601 格式
        fingerprint — 16位十六进制指纹字符串
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role 只能是 'user' 或 'assistant'，收到：{role!r}")

    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages
                (session_id, role, content, created_at, import_fingerprint)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, created_at, fingerprint)
        )
        conn.commit()
        return cursor.lastrowid


def close_session_with_time(session_id: int, ended_at_iso: str) -> None:
    """
    关闭一个历史 session，使用调用方指定的结束时间。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (ended_at_iso, session_id)
        )
        conn.commit()


def get_all_message_fingerprints() -> set:
    """
    取出 messages 表中所有消息的指纹集合，用于重复导入预检。

    前置条件：messages 表必须已有 import_fingerprint 列。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(messages)")
        columns = [row["name"] for row in cursor.fetchall()]

        if "import_fingerprint" not in columns:
            raise RuntimeError(
                "messages 表缺少 import_fingerprint 列。\n"
                "请先执行：ALTER TABLE messages ADD COLUMN import_fingerprint TEXT;"
            )

        cursor.execute(
            "SELECT import_fingerprint FROM messages "
            "WHERE import_fingerprint IS NOT NULL"
        )
        rows = cursor.fetchall()

    return {row["import_fingerprint"] for row in rows}


def get_sessions_without_l1() -> list[dict]:
    """
    查询所有"已关闭但尚未生成 L1 摘要"的 session。

    返回：list[dict]，每个元素包含 id 和 started_at
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.started_at
            FROM sessions s
            LEFT JOIN memory_l1 l1 ON s.id = l1.session_id
            WHERE l1.id IS NULL
              AND s.ended_at IS NOT NULL
            ORDER BY s.started_at ASC
            """
        )
        rows = cursor.fetchall()
    return [{"id": row["id"], "started_at": row["started_at"]} for row in rows]


# =============================================================================
# 记忆衰减相关函数
# =============================================================================

def add_last_accessed_at_columns() -> dict:
    """
    数据库迁移：给 memory_l1 和 memory_l2 表各新增 last_accessed_at 列。
    幂等保护：列已存在时自动跳过，可以安全地反复调用。

    返回：
        dict — {"l1": bool, "l2": bool}，True 表示本次新增了该列
    """
    result = {"l1": False, "l2": False}

    with _db_conn() as conn:
        cursor = conn.cursor()

        def _column_exists(table: str, column: str) -> bool:
            cursor.execute(f"PRAGMA table_info({table})")
            return any(row["name"] == column for row in cursor.fetchall())

        if not _column_exists("memory_l1", "last_accessed_at"):
            cursor.execute("ALTER TABLE memory_l1 ADD COLUMN last_accessed_at TEXT")
            result["l1"] = True

        if not _column_exists("memory_l2", "last_accessed_at"):
            cursor.execute("ALTER TABLE memory_l2 ADD COLUMN last_accessed_at TEXT")
            result["l2"] = True

        conn.commit()

    if result["l1"]:
        logger.info("memory_l1 表已新增 last_accessed_at 列")
    if result["l2"]:
        logger.info("memory_l2 表已新增 last_accessed_at 列")

    return result


def batch_update_last_accessed(layer: str, id_list: list[int]) -> int:
    """
    批量更新一批记忆记录的 last_accessed_at 为当前时间。
    由后台回写线程（AccessBoostWorker）调用。

    使用白名单字典强制映射表名，非法 layer 值记录日志并返回 0。

    参数：
        layer   — "l1" 或 "l2"
        id_list — 需要更新的记录 id 列表

    返回：
        int — 实际更新的行数；失败时返回 0
    """
    if not id_list:
        return 0

    try:
        table = _LAYER_TABLE_MAP[layer]
    except KeyError:
        logger.warning(
            f"batch_update_last_accessed: 未知 layer={layer!r}，"
            f"合法值为 {list(_LAYER_TABLE_MAP.keys())}，跳过"
        )
        return 0

    now          = _now()
    placeholders = ",".join("?" * len(id_list))

    try:
        with _db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {table} SET last_accessed_at = ? "
                f"WHERE id IN ({placeholders})",
                [now] + list(id_list)
            )
            conn.commit()
            updated = cursor.rowcount

        logger.debug(
            f"batch_update_last_accessed: layer={layer}，"
            f"更新 {updated} 条"
        )
        return updated

    except Exception as e:
        logger.warning(f"batch_update_last_accessed 写入失败，layer={layer} — {e}")
        return 0


def get_last_accessed_at(layer: str, record_id: int) -> str | None:
    """
    读取单条记忆记录的 last_accessed_at 字段。

    参数：
        layer     — "l1" 或 "l2"
        record_id — 主键 id

    返回：
        str  — ISO 8601 时间戳字符串
        None — 字段为 NULL 或记录不存在
    """
    try:
        table = _LAYER_TABLE_MAP[layer]
    except KeyError:
        logger.warning(f"get_last_accessed_at: 未知 layer={layer!r}，返回 None")
        return None

    try:
        with _db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT last_accessed_at FROM {table} WHERE id = ?",
                (record_id,)
            )
            row = cursor.fetchone()
        return row["last_accessed_at"] if row else None

    except Exception as e:
        logger.warning(
            f"get_last_accessed_at 查询失败，layer={layer} id={record_id} — {e}"
        )
        return None


# =============================================================================
# graph_nodes 表操作
# =============================================================================

def get_node_by_name(entity_name: str):
    """按实体名精确查询图谱节点。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM graph_nodes WHERE entity_name = ?",
            (entity_name,)
        )
        return cursor.fetchone()


def get_or_create_node(
    entity_name: str,
    entity_type: str,
    source_l1_id: int,
) -> int:
    """
    获取或创建一个图谱节点，返回节点 id。
    节点已存在：use_count + 1。
    节点不存在：插入新行。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM graph_nodes WHERE entity_name = ?",
            (entity_name,)
        )
        row = cursor.fetchone()

        if row:
            cursor.execute(
                "UPDATE graph_nodes SET use_count = use_count + 1 WHERE id = ?",
                (row["id"],)
            )
            node_id = row["id"]
        else:
            cursor.execute(
                """
                INSERT INTO graph_nodes
                    (entity_name, entity_type, source_l1_id, created_at, use_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (entity_name, entity_type, source_l1_id, _now())
            )
            node_id = cursor.lastrowid

        conn.commit()
    return node_id


def get_canonical_name(word_id: int) -> str | None:
    """给定 keyword_pool 里的词 id，追溯到规范词的文本。"""
    with _db_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT keyword, canonical_id FROM keyword_pool WHERE rowid = ?",
            (word_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        if row["canonical_id"] is None:
            return row["keyword"]

        cursor.execute(
            "SELECT keyword FROM keyword_pool WHERE rowid = ?",
            (row["canonical_id"],)
        )
        canonical_row = cursor.fetchone()

    return canonical_row["keyword"] if canonical_row else None


def get_all_canonical_keywords() -> list[dict]:
    """
    取出 keyword_pool 里所有规范词（canonical_id 为 NULL 且已确认的词）。
    用于实体归一化时计算向量相似度。
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT rowid, keyword, use_count
            FROM keyword_pool
            WHERE canonical_id IS NULL
              AND alias_status = 'confirmed'
            ORDER BY use_count DESC
            """
        )
        rows = cursor.fetchall()
    return [
        {"id": r["rowid"], "keyword": r["keyword"], "use_count": r["use_count"]}
        for r in rows
    ]


# =============================================================================
# graph_edges 表操作
# =============================================================================

def save_edge(
    source_node_id: int,
    target_node_id: int,
    relation_type: str,
    relation_detail: str,
    source_l1_id: int,
) -> int:
    """保存一条图谱边，返回新边的 id。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO graph_edges
                (source_node_id, target_node_id, relation_type,
                 relation_detail, source_l1_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source_node_id, target_node_id, relation_type,
             relation_detail, source_l1_id, _now())
        )
        conn.commit()
        return cursor.lastrowid


def get_l1_ids_by_node(node_id: int, max_hops: int = 2) -> list[dict]:
    """
    从指定节点出发，做广度优先遍历，返回关联的 L1 id 列表。

    返回：list[dict]，每个元素包含 l1_id 和 hops
    """
    with _db_conn() as conn:
        cursor = conn.cursor()

        visited_nodes = {node_id}
        current_level = {node_id}
        result        = []
        seen_l1_ids   = set()

        for hop in range(1, max_hops + 1):
            if not current_level:
                break

            placeholders = ",".join("?" * len(current_level))
            cursor.execute(
                f"""
                SELECT source_l1_id, source_node_id, target_node_id
                FROM graph_edges
                WHERE source_node_id IN ({placeholders})
                   OR target_node_id IN ({placeholders})
                """,
                list(current_level) + list(current_level)
            )
            edges = cursor.fetchall()

            next_level = set()
            for edge in edges:
                l1_id = edge["source_l1_id"]
                if l1_id not in seen_l1_ids:
                    result.append({"l1_id": l1_id, "hops": hop})
                    seen_l1_ids.add(l1_id)

                for neighbor_id in (edge["source_node_id"], edge["target_node_id"]):
                    if neighbor_id not in visited_nodes:
                        next_level.add(neighbor_id)
                        visited_nodes.add(neighbor_id)

            current_level = next_level

    return result


def get_all_l1_ids_in_graph() -> set[int]:
    """返回图谱中已经处理过的所有 L1 id 集合。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT source_l1_id FROM graph_edges")
        rows = cursor.fetchall()
    return {row["source_l1_id"] for row in rows}


def get_node_by_id(node_id: int):
    """按节点 id 查询图谱节点。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM graph_nodes WHERE id = ?", (node_id,))
        return cursor.fetchone()


# =============================================================================
# 实体别名归一化操作
# =============================================================================

def save_keyword_with_alias(
    keyword: str,
    canonical_id: int | None,
    alias_status: str,
) -> int:
    """
    写入一个带别名状态的关键词到 keyword_pool。

    参数：
        keyword      — 关键词文本
        canonical_id — 规范词的 rowid；为 None 表示本词就是规范词
        alias_status — 'confirmed' / 'pending' / 'canonical' 三态之一

    返回：
        int — 新写入词条的 rowid
    """
    now = _now()
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO keyword_pool
                (keyword, use_count, last_used_at, created_at,
                 canonical_id, alias_status)
            VALUES (?, 1, ?, ?, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET
                use_count    = use_count + 1,
                last_used_at = excluded.last_used_at,
                canonical_id = COALESCE(canonical_id, excluded.canonical_id),
                alias_status = CASE
                    WHEN alias_status = 'confirmed' THEN 'confirmed'
                    ELSE excluded.alias_status
                END
            """,
            (keyword, now, now, canonical_id, alias_status)
        )
        conn.commit()
        cursor.execute(
            "SELECT rowid FROM keyword_pool WHERE keyword = ?",
            (keyword,)
        )
        row = cursor.fetchone()
    return row["rowid"]


def get_pending_aliases() -> list:
    """查询所有待确认的别名记录（alias_status = 'pending'）。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT kp.rowid, kp.keyword, kp.canonical_id, kp.alias_status,
                   canonical.keyword AS canonical_keyword
            FROM keyword_pool kp
            LEFT JOIN keyword_pool canonical ON kp.canonical_id = canonical.rowid
            WHERE kp.alias_status = 'pending'
            ORDER BY kp.created_at ASC
            """
        )
        return cursor.fetchall()


def confirm_alias(new_word_id: int, canonical_id: int) -> bool:
    """
    确认别名关系：将 new_word_id 对应的词正式归入 canonical_id 对应的规范词。

    执行三步操作（同一事务）：
      1. 更新 keyword_pool：alias_status 改为 confirmed
      2. 迁移 graph_edges：所有指向 new_word 节点的边改指向规范词节点
      3. 合并 graph_nodes：use_count 累加，删除 new_word 节点

    返回：bool — True 表示成功
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT keyword FROM keyword_pool WHERE rowid = ?",
                (new_word_id,)
            )
            new_word_row = cursor.fetchone()

            cursor.execute(
                "SELECT keyword FROM keyword_pool WHERE rowid = ?",
                (canonical_id,)
            )
            canonical_row = cursor.fetchone()

            if not new_word_row or not canonical_row:
                logger.warning(
                    f"confirm_alias: 找不到词条 "
                    f"new_word_id={new_word_id} 或 canonical_id={canonical_id}"
                )
                return False

            new_word       = new_word_row["keyword"]
            canonical_word = canonical_row["keyword"]

            cursor.execute(
                "SELECT id, use_count FROM graph_nodes WHERE entity_name = ?",
                (new_word,)
            )
            new_node = cursor.fetchone()

            cursor.execute(
                "SELECT id FROM graph_nodes WHERE entity_name = ?",
                (canonical_word,)
            )
            canonical_node = cursor.fetchone()

            # 步骤 1：更新 keyword_pool
            cursor.execute(
                """
                UPDATE keyword_pool
                SET canonical_id = ?, alias_status = 'confirmed'
                WHERE rowid = ?
                """,
                (canonical_id, new_word_id)
            )

            # 步骤 2 & 3：迁移图谱（只在两个节点都存在时执行）
            if new_node and canonical_node:
                new_node_id       = new_node["id"]
                canonical_node_id = canonical_node["id"]
                merged_count      = new_node["use_count"]

                cursor.execute(
                    "UPDATE graph_edges SET source_node_id = ? "
                    "WHERE source_node_id = ?",
                    (canonical_node_id, new_node_id)
                )
                cursor.execute(
                    "UPDATE graph_edges SET target_node_id = ? "
                    "WHERE target_node_id = ?",
                    (canonical_node_id, new_node_id)
                )
                cursor.execute(
                    "UPDATE graph_nodes SET use_count = use_count + ? WHERE id = ?",
                    (merged_count, canonical_node_id)
                )
                cursor.execute(
                    "DELETE FROM graph_nodes WHERE id = ?",
                    (new_node_id,)
                )

            conn.commit()
            logger.info(
                f"confirm_alias: '{new_word}' → '{canonical_word}' 已确认"
            )
            return True

        except Exception as e:
            conn.rollback()
            logger.error(f"confirm_alias 事务失败，已回滚 — {e}")
            return False


def reject_alias(new_word_id: int) -> bool:
    """拒绝别名关系：将 pending 词独立为新的规范词。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE keyword_pool
                SET canonical_id = NULL, alias_status = 'confirmed'
                WHERE rowid = ?
                """,
                (new_word_id,)
            )
            conn.commit()
            logger.info(f"reject_alias: word_id={new_word_id} 已独立为规范词")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"reject_alias 失败 — {e}")
            return False


# =============================================================================
# conflict_queue 扩展：别名确认类冲突
# =============================================================================

def save_alias_conflict(
    source_l1_id: int,
    old_word: str,
    new_word: str,
    new_word_kp_id: int,
    canonical_kp_id: int,
    similarity: float,
) -> int:
    """
    将一条待确认的别名关系写入 conflict_queue，等待用户在对话中确认。

    返回：int — 新冲突记录的 id
    """
    conflict_desc = (
        f'"{new_word}"和"{old_word}"是指同一个事物吗？'
        f'（相似度 {similarity:.0%}，'
        f'回复"是"确认合并，回复"不是"保持独立）'
    )

    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conflict_queue
                (source_l1_id, field, old_content, new_content,
                 conflict_desc, status, created_at, conflict_type)
            VALUES (?, 'keyword_alias', ?, ?, ?, 'pending', ?, 'alias_confirm')
            """,
            (source_l1_id, old_word, new_word, conflict_desc, _now())
        )
        conn.commit()
        conflict_id = cursor.lastrowid

        # 在 old_content 字段末尾追加 kp_id 信息，供 handle_conflict_reply 读取
        cursor.execute(
            "UPDATE conflict_queue SET old_content = ? WHERE id = ?",
            (
                f"{old_word}\n__kp_ids__{new_word_kp_id}:{canonical_kp_id}",
                conflict_id
            )
        )
        conn.commit()

    logger.info(
        f"save_alias_conflict: '{new_word}' vs '{old_word}' "
        f"（相似度 {similarity:.2f}），conflict_id={conflict_id}"
    )
    return conflict_id


def get_alias_kp_ids_from_conflict(
    conflict_id: int,
) -> tuple[int, int] | None:
    """
    从 conflict_queue 记录中解析出 (new_word_kp_id, canonical_kp_id)。

    返回：
        (new_word_kp_id, canonical_kp_id) — 解析成功
        None — 解析失败
    """
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT old_content FROM conflict_queue WHERE id = ?",
            (conflict_id,)
        )
        row = cursor.fetchone()

    if not row:
        return None

    old_content = row["old_content"]
    try:
        parts        = old_content.split("\n__kp_ids__")
        ids_str      = parts[1]
        new_id, canonical_id = ids_str.split(":")
        return int(new_id), int(canonical_id)
    except Exception as e:
        logger.warning(f"get_alias_kp_ids_from_conflict 解析失败：{e}")
        return None


# =============================================================================
# pending_push 表操作（主动推送功能）
# =============================================================================

def save_pending_push(content: str) -> int:
    """将一条主动推送消息写入 pending_push 表，状态为 pending。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO pending_push (content, created_at, status) "
            "VALUES (?, ?, 'pending')",
            (content, _now())
        )
        conn.commit()
        return cursor.lastrowid


def get_pending_pushes() -> list:
    """取出所有状态为 pending 的主动推送消息，按创建时间升序排列。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM pending_push "
            "WHERE status = 'pending' ORDER BY created_at ASC"
        )
        return cursor.fetchall()


def mark_push_sent(push_id: int) -> None:
    """将一条推送记录标记为已发送（status = 'sent'）。"""
    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE pending_push SET status = 'sent', sent_at = ? WHERE id = ?",
            (_now(), push_id)
        )
        conn.commit()


def get_push_count_today() -> int:
    """查询今天已触发的推送条数（包含 pending 和 sent）。"""
    from datetime import date
    today_str = date.today().isoformat()

    with _db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM pending_push WHERE created_at LIKE ?",
            (f"{today_str}%",)
        )
        row = cursor.fetchone()
    return row["cnt"] if row else 0
