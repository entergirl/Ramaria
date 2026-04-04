"""
database.py — 数据库操作层
封装所有与 assistant.db 的读写操作，其他模块通过调用这里的函数访问数据库。
不包含建表逻辑（那部分在 init_db.py）。

设计原则：
  - 每个函数只做一件事，职责单一
  - 每个函数自己开连接、自己关连接，不依赖外部传入的连接对象
  - 所有时间统一使用 UTC ISO 8601 格式存储，如 "2025-03-14T22:00:00+00:00"
  - _get_connection 是模块私有函数，不对外暴露；外部模块一律通过公开函数访问数据

变更记录：
  v2 — 新增 get_l1_by_id(l1_id)，修复审查报告问题2（并发竞态风险）
  v3 — 新增 get_all_l1() / get_all_l2() / get_all_session_ids()，
       修复审查报告问题A：vector_store.rebuild_all_indexes() 原来直接
       import 私有函数 _get_connection 并手写原始 SQL，破坏数据层封装。
       改为通过这三个公开函数访问，彻底消除私有函数外漏。
  v4 — 新增 update_message_time_for_test()，修复代码优化清单 P1-2：
       session_manager.py 的 __main__ 测试块原来直接调用私有函数
       _get_connection() 来篡改消息时间戳，造成封装原则不一致。
       现在将该操作封装为仅限测试使用的公开函数，测试块改为调用此函数。
     — get_unabsorbed_l1() 合并重复 SQL，修复代码优化清单 P3-1：
       原来 limit 有无两段 SQL 字面几乎完全相同，只是末尾 LIMIT 子句不同，
       违反 DRY 原则。改为条件拼接 SQL，合并为一段。

使用方法（在其他模块里）：
    from database import save_message, get_messages, save_l1_summary, ...
"""

import sqlite3
from datetime import datetime, timezone
from config import DB_PATH
from logger import get_logger
logger = get_logger(__name__)

# =============================================================================
# 内部工具函数（模块私有，不对外暴露）
# =============================================================================

def _get_connection():
    """
    获取数据库连接（内部使用）。
    - row_factory 让查询结果支持列名访问，如 row["content"]
    - 开启外键约束（SQLite 默认关闭）

    注意：此函数为模块私有，外部模块不应直接 import 使用。
          外部模块需要查询数据时，请使用下方的公开函数。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now():
    """
    返回当前 UTC 时间的 ISO 8601 字符串（内部使用）。
    统一在这里生成时间戳，方便以后切换时区策略。
    """
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Sessions 表操作
# =============================================================================

def new_session():
    """
    创建一个新 session，返回新 session 的 id（int）。

    调用时机：
        用户发来消息，且当前没有活跃 session 时，由 session_manager 调用。

    返回：
        int — 新建 session 的 id
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (started_at) VALUES (?)",
        (_now(),)
    )
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id


def close_session(session_id):
    """
    关闭一个 session，将 ended_at 写入当前时间。

    调用时机：
        空闲检测超时后，由 session_manager 调用。

    参数：
        session_id — 要关闭的 session id
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (_now(), session_id)
    )
    conn.commit()
    conn.close()


def get_session(session_id):
    """
    取出一个 session 的完整信息。

    参数：
        session_id — session id

    返回：
        sqlite3.Row — 包含 id / started_at / ended_at 字段；不存在时返回 None
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_active_sessions():
    """
    取出所有尚未关闭的 session（ended_at 为 NULL）。

    返回：
        列表，每个元素是 sqlite3.Row；没有活跃 session 时返回空列表。

    用途：
        后端启动时检查是否有遗留的未关闭 session（例如上次异常退出遗留的）。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY started_at ASC"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# =============================================================================
# Messages 表操作
# =============================================================================

def save_message(session_id, role, content):
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

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, _now())
    )
    conn.commit()
    message_id = cursor.lastrowid
    conn.close()
    return message_id


def get_messages(session_id):
    """
    取出某个 session 的全部消息，按时间升序排列。

    参数：
        session_id — session id

    返回：
        列表，每个元素是 sqlite3.Row，可用 row["role"]、row["content"] 访问。
        没有消息时返回空列表。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_messages_as_dicts(session_id):
    """
    取出某个 session 的全部消息，以字典列表形式返回。

    与 get_messages 的区别：
        本函数返回普通 dict，格式与 OpenAI / LM Studio API 的 messages 参数一致，
        可以直接拼进请求体，不需要再做转换。

    返回格式示例：
        [
            {"role": "user",      "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"}
        ]
    """
    rows = get_messages(session_id)
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def get_last_message_time(session_id):
    """
    取出某个 session 最后一条消息的时间戳字符串。

    参数：
        session_id — session id

    返回：
        str  — ISO 8601 时间戳，如 "2025-03-14T22:00:00+00:00"
        None — 该 session 没有任何消息时返回 None

    用途：
        空闲检测时，用来判断距离上次消息过了多长时间。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT created_at FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row["created_at"] if row else None


def update_message_time_for_test(session_id, fake_time_str):
    """
    将某 session 所有消息的时间戳改为指定值。

    【v4 新增】修复代码优化清单 P1-2：
        session_manager.py 的 __main__ 测试块原来直接调用私有函数
        _get_connection() 来篡改消息时间戳，用于模拟空闲超时场景：

            旧写法（违反封装原则）：
                conn = database._get_connection()          # ← 直接访问私有函数
                conn.execute("UPDATE messages SET ...", ...)
                conn.commit()
                conn.close()

            新写法（通过公开函数访问）：
                from database import update_message_time_for_test
                update_message_time_for_test(sid, fake_time)

    ⚠️  此函数仅限测试使用，不应在正式业务代码中调用。
        函数名后缀 _for_test 是有意为之的警示标记。

    参数：
        session_id    — 要修改的 session id
        fake_time_str — 目标时间戳字符串，ISO 8601 格式，
                        如 "2026-03-25T10:00:00+00:00"
    """
    conn = _get_connection()
    conn.execute(
        "UPDATE messages SET created_at = ? WHERE session_id = ?",
        (fake_time_str, session_id)
    )
    conn.commit()
    conn.close()


# =============================================================================
# memory_l1 表操作
# =============================================================================

def save_l1_summary(session_id, summary, keywords, time_period, atmosphere,
                    valence=0.0, salience=0.5):
    """
    将一条 L1 摘要写入 memory_l1 表。

    参数：
        session_id  — 对应的 session id
        summary     — 一句话摘要（第三人称，只记结论，不超过50字）
        keywords    — 名词标签字符串，逗号分隔，如 "项目,后端,FastAPI"
        time_period — 时间段标签，六选一：清晨/上午/下午/傍晚/夜间/深夜
        atmosphere  — 对话氛围，四字以内，如 "专注高效"
        valence     — 情绪效价，五档浮点：-1.0/-0.5/0.0/0.5/1.0，默认中性 0.0
        salience    — 情感显著性，五档浮点：0.0/0.25/0.5/0.75/1.0，默认中等 0.5

    返回：
        int — 新插入 L1 记录的 id
    """
    from config import TIME_PERIOD_OPTIONS
    if time_period not in TIME_PERIOD_OPTIONS:
        logger.warning(f"time_period 值 {time_period!r} 不在合法列表内，已置为 None")
        time_period = None

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memory_l1
            (session_id, summary, keywords, time_period, atmosphere,
             valence, salience, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, summary, keywords, time_period, atmosphere,
          valence, salience, _now()))
    conn.commit()
    l1_id = cursor.lastrowid
    conn.close()
    return l1_id


def get_l1_by_id(l1_id):
    """
    按主键查询一条 L1 摘要记录。

    [v2 新增] 修复审查报告问题2：conflict_checker 和 profile_manager 原来用
    get_latest_l1() 读取最新一条再比对 id，存在并发竞态风险。
    改为直接按主键查询，彻底消除竞态风险。

    参数：
        l1_id — memory_l1 表的主键 id（int）

    返回：
        sqlite3.Row — 完整的 L1 记录行；id 不存在时返回 None
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM memory_l1 WHERE id = ?", (l1_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_l1_salience(l1_id: int) -> float:
    """
    快速查询单条 L1 记录的 salience 值。

    用于 vector_store._calc_decay_factor() 计算衰减时读取显著性加成。
    比 get_l1_by_id() 轻量，只查一个字段。

    参数：
        l1_id — memory_l1 表的主键 id

    返回：
        float — salience 值，范围 [0.0, 1.0]
                记录不存在或字段为 NULL 时返回默认值 0.5（不影响衰减）
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT salience FROM memory_l1 WHERE id = ?",
        (l1_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None or row["salience"] is None:
        return 0.5   # 默认中等显著性，不影响衰减计算
    return float(row["salience"])


def get_unabsorbed_l1(limit=None):
    """
    取出所有尚未被 L2 吸收的 L1 摘要，按生成时间升序排列。

    参数：
        limit — 最多返回多少条；不传则返回全部

    返回：
        列表，每个元素是 sqlite3.Row；没有未吸收记录时返回空列表。

    用途：
        L2 触发判断 + L2 合并时读取待处理的 L1 列表。

    【v4 修复】代码优化清单 P3-1：合并重复 SQL，消除 DRY 违反。
 
    """
    conn = _get_connection()
    cursor = conn.cursor()

    sql    = "SELECT * FROM memory_l1 WHERE absorbed = 0 ORDER BY created_at ASC"
    params = ()
    if limit:
        sql   += " LIMIT ?"
        params = (limit,)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_l1():
    """
    返回 memory_l1 表中的全部记录（id, summary, keywords, session_id）。

    [v3 新增] 修复审查报告问题A：vector_store.rebuild_all_indexes() 原来直接
    import 私有函数 _get_connection 并手写原始 SQL 查询所有 L1，破坏了
    database.py 作为数据层统一出口的封装原则。

    用途：
        vector_store.rebuild_all_indexes() 重建 L1 向量索引时调用。

    返回：
        列表，每个元素是 sqlite3.Row，包含 id / summary / keywords / session_id 字段。
        表为空时返回空列表。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, summary, keywords, session_id FROM memory_l1"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_l2():
    """
    返回 memory_l2 表中的全部记录（id, summary, keywords, period_start, period_end）。

    [v3 新增] 修复审查报告问题A：与 get_all_l1() 同理。

    用途：
        vector_store.rebuild_all_indexes() 重建 L2 向量索引时调用。

    返回：
        列表，每个元素是 sqlite3.Row，包含 id / summary / keywords /
        period_start / period_end 字段。表为空时返回空列表。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, summary, keywords, period_start, period_end FROM memory_l2"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_session_ids():
    """
    返回 messages 表中全部去重后的 session_id 列表。

    [v3 新增] 修复审查报告问题A：与 get_all_l1() 同理。

    用途：
        vector_store.rebuild_all_indexes() 按 session 逐个重建 L0 向量索引时调用。

    返回：
        list[int] — 去重后的 session_id 整数列表；messages 表为空时返回空列表。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id FROM messages")
    rows = cursor.fetchall()
    conn.close()
    return [row["session_id"] for row in rows]


def mark_l1_absorbed(l1_ids):
    """
    将一批 L1 记录标记为已被 L2 吸收（absorbed = 1）。

    参数：
        l1_ids — L1 id 的列表，如 [1, 2, 3]

    用途：
        L2 合并完成后，调用此函数更新这批 L1 的状态。
    """
    if not l1_ids:
        return
    conn = _get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(l1_ids))
    cursor.execute(
        f"UPDATE memory_l1 SET absorbed = 1 WHERE id IN ({placeholders})",
        l1_ids
    )
    conn.commit()
    conn.close()


# =============================================================================
# memory_l2 表操作
# =============================================================================

def save_l2_summary(summary, keywords, period_start, period_end, l1_ids):
    """
    将一条 L2 摘要写入 memory_l2 表，同时写入 l2_sources 关联表。
    两步写入包裹在同一个事务里，保证原子性——要么都成功，要么都回滚。

    参数：
        summary      — 合并后的摘要文本（两到三句话）
        keywords     — 关键词字符串，英文逗号分隔
        period_start — 覆盖时间段起点（ISO 8601），取最早 L1 的 created_at
        period_end   — 覆盖时间段终点（ISO 8601），取最晚 L1 的 created_at
        l1_ids       — 参与合并的 L1 id 列表，如 [1, 2, 3, 4, 5]

    返回：
        int — 新插入 L2 记录的 id
    """
    conn = _get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO memory_l2 (summary, keywords, period_start, period_end, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (summary, keywords, period_start, period_end, _now()))
        l2_id = cursor.lastrowid

        l2_source_rows = [(l2_id, l1_id) for l1_id in l1_ids]
        cursor.executemany("""
            INSERT OR IGNORE INTO l2_sources (l2_id, l1_id) VALUES (?, ?)
        """, l2_source_rows)

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"save_l2_summary 事务失败，已回滚 — {e}")
        raise
    finally:
        conn.close()

    return l2_id


def get_recent_l2(limit=3):
    """
    取出最近几条 L2 摘要，按生成时间降序排列（最新的在前）。

    参数：
        limit — 最多返回多少条，默认 3（对应 config.L2_INJECT_COUNT）

    返回：
        列表，每个元素是 sqlite3.Row；没有 L2 时返回空列表。

    用途：
        构建 system prompt 时注入近期 L2，代表用户近期状态。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM memory_l2
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_latest_l1():
    """
    取出最近一条 L1 摘要（不限 absorbed 状态）。

    返回：
        sqlite3.Row 或 None

    用途：
        构建 system prompt 时注入当日最新 L1，代表最新上下文。

    注意：
        此函数仅用于 prompt_builder 注入上下文，不应用于
        conflict_checker / profile_manager 的 L1 校验（请改用 get_l1_by_id）。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM memory_l1
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return row


def get_l1_by_session(session_id):
    """
    取出某个 session 对应的 L1 摘要（通常只有一条）。

    参数：
        session_id — session id

    返回：
        sqlite3.Row 或 None
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM memory_l1 WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row


# =============================================================================
# Settings 表操作
# =============================================================================

def get_setting(key, default=None):
    """
    读取一个配置项的值（字符串格式）。

    参数：
        key     — 配置项名称
        default — key 不存在时的返回值，默认为 None
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key, value):
    """
    写入或更新一个配置项。
    key 已存在时更新 value 和 updated_at；不存在时插入新行。

    参数：
        key   — 配置项名称
        value — 配置项值（传入字符串；数字请先 str() 转换）
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (key, str(value), _now()))
    conn.commit()
    conn.close()


# =============================================================================
# user_profile 表操作
# =============================================================================

def get_current_profile():
    """
    取出当前生效的完整用户画像，返回字典，key 为板块名。

    返回格式示例：
        {
            "basic_info":      "烧酒，19岁，...",
            "interests":       "编程、...",
            "personal_status": "...",
        }

    调用时机：
        构建 system prompt 时，将 L3 记忆注入对话上下文。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT field, content FROM user_profile
        WHERE is_current = 1 AND status = 'approved'
    """)
    rows = cursor.fetchall()
    conn.close()
    return {row["field"]: row["content"] for row in rows}


# =============================================================================
# keyword_pool 表操作
# =============================================================================

def upsert_keywords(keywords):
    """
    批量写入关键词到词典表。
    词条已存在时：use_count + 1，更新 last_used_at。
    词条不存在时：插入新行，use_count 从 1 开始。

    参数：
        keywords — 关键词字符串列表，如 ["数据库", "后端", "FastAPI"]
    """
    if not keywords:
        return

    now = _now()
    conn = _get_connection()
    cursor = conn.cursor()

    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        cursor.execute("""
            INSERT INTO keyword_pool (keyword, use_count, last_used_at, created_at)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET
                use_count    = use_count + 1,
                last_used_at = excluded.last_used_at
        """, (kw, now, now))

    conn.commit()
    conn.close()


def get_all_keywords():
    """
    取出词典表中所有关键词，按使用频次降序排列。

    返回：
        字符串列表；词典为空时返回空列表。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT keyword FROM keyword_pool ORDER BY use_count DESC")
    rows = cursor.fetchall()
    conn.close()
    return [row["keyword"] for row in rows]


def get_top_keywords(limit=50):
    """
    取出使用频次最高的前 N 个关键词。

    参数：
        limit — 最多返回多少个，默认 50

    返回：
        字符串列表，按 use_count 降序排列。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT keyword FROM keyword_pool
        ORDER BY use_count DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [row["keyword"] for row in rows]


# =============================================================================
# conflict_queue 表操作
# =============================================================================

def save_conflict(source_l1_id, field, old_content, new_content, conflict_desc):
    """
    将一条冲突记录写入 conflict_queue 表，状态初始为 pending。

    参数：
        source_l1_id  — 触发此冲突的 L1 记录 id
        field         — 涉及的画像板块名，如 "personal_status"
        old_content   — 现有画像内容（写入时的快照）
        new_content   — L1 里检测到的新信息（与旧内容矛盾的部分）
        conflict_desc — 模型生成的冲突描述，供展示给用户时使用

    返回：
        int — 新插入冲突记录的 id
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conflict_queue
            (source_l1_id, field, old_content, new_content, conflict_desc, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'pending', ?)
    """, (source_l1_id, field, old_content, new_content, conflict_desc, _now()))
    conn.commit()
    conflict_id = cursor.lastrowid
    conn.close()
    return conflict_id


def get_pending_conflicts():
    """
    取出所有状态为 pending 的冲突记录，按写入时间升序排列（最早的优先处理）。

    返回：
        列表，每个元素是 sqlite3.Row；没有待确认冲突时返回空列表。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM conflict_queue
        WHERE status = 'pending'
        ORDER BY created_at ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def resolve_conflict(conflict_id):
    """
    将一条冲突记录标记为 resolved（用户确认接受新内容）。

    参数：
        conflict_id — conflict_queue 表的记录 id
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE conflict_queue
        SET status = 'resolved', resolved_at = ?
        WHERE id = ?
    """, (_now(), conflict_id))
    conn.commit()
    conn.close()


def ignore_conflict(conflict_id):
    """
    将一条冲突记录标记为 ignored（用户选择忽略，新旧内容均保留）。

    参数：
        conflict_id — conflict_queue 表的记录 id
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE conflict_queue
        SET status = 'ignored', resolved_at = ?
        WHERE id = ?
    """, (_now(), conflict_id))
    conn.commit()
    conn.close()


def update_profile_field(field, new_content, source_l1_id=None):
    """
    更新用户画像的某个板块，将旧版本标记为历史，插入新版本。

    设计说明：
        user_profile 表采用追加写入策略，不覆盖旧行，而是将旧行的
        is_current 置为 0，插入新行并将 is_current 置为 1。
        这样保留了完整的历史版本，支持将来回溯或撤销。

    参数：
        field        — 板块名，如 "personal_status"
        new_content  — 新的板块内容文本
        source_l1_id — 触发此更新的 L1 id；手动更新时传 None

    返回：
        int — 新插入画像记录的 id
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # 先把该字段旧的「当前版本」标记为历史
    cursor.execute("""
        UPDATE user_profile SET is_current = 0
        WHERE field = ? AND is_current = 1
    """, (field,))

    # 再插入新版本，is_current = 1 表示当前生效
    cursor.execute("""
        INSERT INTO user_profile
            (field, content, source_l1_id, status, is_current, updated_at)
        VALUES (?, ?, ?, 'approved', 1, ?)
    """, (field, new_content, source_l1_id, _now()))

    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

"""
database_patch.py — database.py 新增函数补丁
=====================================================================

将以下三个函数追加到 database.py 末尾（在 __main__ 块之前）。

这三个函数专门为历史消息导入设计，接受自定义时间戳参数，
不使用 datetime.now()，确保历史 session 的时间线准确。

与现有函数的区别：
    new_session()                 → started_at = datetime.now()
    new_session_with_time()       → started_at = 调用方传入的 ISO 字符串

    save_message()                → created_at = datetime.now()
    save_message_with_fingerprint → created_at = 调用方传入的 ISO 字符串 + 指纹字段

    close_session()               → ended_at = datetime.now()
    close_session_with_time()     → ended_at = 调用方传入的 ISO 字符串

使用场景：
    仅限 importer/qq_importer.py 调用。
    正常业务流程（实时对话）继续使用原有函数。
"""

def new_session_with_time(started_at_iso: str) -> int:
    """
    创建一个历史 session，使用调用方指定的开始时间。

    与 new_session() 的唯一区别：
        started_at 由调用方传入，而不是 datetime.now()。
        用于历史消息导入，保留原始时间线。

    参数：
        started_at_iso — session 开始时间，UTC ISO 8601 格式
                         如 "2024-06-18T08:55:51+00:00"

    返回：
        int — 新建 session 的 id
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (started_at) VALUES (?)",
        (started_at_iso,)
    )
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id


def save_message_with_fingerprint(
    session_id: int,
    role: str,
    content: str,
    created_at: str,
    fingerprint: str,
) -> int:
    """
    保存一条历史消息，同时写入指纹字段，用于后续重复导入预检。

    与 save_message() 的区别：
        · created_at 由调用方传入（保留原始时间戳，不用 datetime.now()）
        · 额外写入 import_fingerprint 字段（重复预检用）

    前置条件：
        messages 表必须已执行以下迁移，否则写入时会报列不存在：
            ALTER TABLE messages ADD COLUMN import_fingerprint TEXT;

    参数：
        session_id  — 所属 session 的 id
        role        — "user" 或 "assistant"
        content     — 消息正文（对方消息含 [发送人名] 前缀）
        created_at  — 消息原始时间，UTC ISO 8601 格式
        fingerprint — 由 qq_parser._make_fingerprint() 生成的16位十六进制字符串

    返回：
        int — 新插入消息的 id
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role 只能是 'user' 或 'assistant'，收到：{role!r}")

    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO messages (session_id, role, content, created_at, import_fingerprint)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, role, content, created_at, fingerprint)
    )
    conn.commit()
    message_id = cursor.lastrowid
    conn.close()
    return message_id


def close_session_with_time(session_id: int, ended_at_iso: str) -> None:
    """
    关闭一个历史 session，使用调用方指定的结束时间。

    与 close_session() 的唯一区别：
        ended_at 由调用方传入，而不是 datetime.now()。
        用于历史消息导入，保留原始时间线。

    参数：
        session_id   — 要关闭的 session id
        ended_at_iso — session 结束时间，UTC ISO 8601 格式
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (ended_at_iso, session_id)
    )
    conn.commit()
    conn.close()


# =============================================================================
# 重复导入预检专用函数
# =============================================================================

def get_all_message_fingerprints() -> set:
    """
    取出 messages 表中所有消息的指纹集合，用于重复导入预检。

    指纹由 qq_parser._make_fingerprint() 计算，存储在 messages 表的
    import_fingerprint 列（需要先执行下方的迁移 SQL 添加此列）。

    ── 迁移 SQL（首次使用前手动执行一次）──
        ALTER TABLE messages ADD COLUMN import_fingerprint TEXT;

    未设置指纹的消息（实时对话产生的）import_fingerprint 为 NULL，
    查询时自动过滤掉 NULL 值，只返回有指纹的条目。

    返回：
        set — 所有已存在指纹的字符串集合；messages 表为空或无指纹时返回空集合

    调用方：
        importer/qq_parser.py 的 _check_duplicates_against_db()
    """
    conn   = _get_connection()
    cursor = conn.cursor()

    # 检查 import_fingerprint 列是否存在
    # 如果用户还没执行迁移 SQL，这里会给出友好提示而不是崩溃
    cursor.execute("PRAGMA table_info(messages)")
    columns = [row["name"] for row in cursor.fetchall()]
    conn.close()

    if "import_fingerprint" not in columns:
        raise RuntimeError(
            "messages 表缺少 import_fingerprint 列。\n"
            "请先在 SQLite 中执行以下迁移 SQL：\n"
            "    ALTER TABLE messages ADD COLUMN import_fingerprint TEXT;\n"
            "执行后重新运行导入即可。"
        )

    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT import_fingerprint FROM messages WHERE import_fingerprint IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

    return {row["import_fingerprint"] for row in rows}

# ===========================================================================
# 查询所有"已关闭但尚未生成 L1 摘要"的历史 session。
# ===========================================================================

def get_sessions_without_l1() -> list:
    """
    SQL 逻辑：
        sessions LEFT JOIN memory_l1 ON session_id
        WHERE memory_l1.id IS NULL          -- 没有对应 L1
          AND sessions.ended_at IS NOT NULL  -- 已关闭的 session
        ORDER BY started_at ASC             -- 按时间升序，从最早的开始处理

    返回：
        list[dict]，每个元素包含：
            id         — session id（int）
            started_at — session 开始时间（ISO 8601 字符串）

    用途：
        importer/l1_batch.py 的 _get_pending_sessions() 调用此函数，
        获取批处理的待处理列表。
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.id, s.started_at
        FROM sessions s
        LEFT JOIN memory_l1 l1 ON s.id = l1.session_id
        WHERE l1.id IS NULL
          AND s.ended_at IS NOT NULL
        ORDER BY s.started_at ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    # sqlite3.Row 转 dict，方便后续序列化和传参
    return [{"id": row["id"], "started_at": row["started_at"]} for row in rows]

# ===========================================================================
# 记忆衰减功能的数据库补丁
# ===========================================================================

def add_last_accessed_at_columns():
    """
    数据库迁移：给 memory_l1 和 memory_l2 表各新增 last_accessed_at 列。

    last_accessed_at 用于记忆衰减的"保底加成"机制：
        当一条记忆在近期（MEMORY_DECAY_RECENT_BOOST_DAYS 天内）被检索命中过，
        即使它创建时间很早，也不会让其保留率 R 跌得太低。

    列定义：
        last_accessed_at TEXT  -- UTC ISO 8601，可为 NULL
        NULL 表示该记忆自创建后从未被检索命中过（或在此功能上线前就已存在）

    幂等保护：
        SQLite 的 ALTER TABLE ADD COLUMN 在列已存在时会抛出 OperationalError，
        函数内部捕获这个错误并静默跳过，可以安全地反复调用。

    调用时机：
        main.py 的 lifespan startup 阶段调用一次即可，后续自动跳过。

    返回：
        dict — {"l1": bool, "l2": bool}，True 表示本次新增了该列，False 表示已存在
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    result = {"l1": False, "l2": False}

    # memory_l1 表新增 last_accessed_at
    try:
        cursor.execute(
            "ALTER TABLE memory_l1 ADD COLUMN last_accessed_at TEXT"
        )
        result["l1"] = True
    except Exception as e:
        # 列已存在时 SQLite 抛出 OperationalError: duplicate column name
        # 其他异常也静默处理，不阻断启动流程
        if "duplicate column name" not in str(e).lower():
            logger.warning(f"memory_l1 添加 last_accessed_at 列时出现非预期异常 — {e}")

    # memory_l2 表新增 last_accessed_at
    try:
        cursor.execute(
            "ALTER TABLE memory_l2 ADD COLUMN last_accessed_at TEXT"
        )
        result["l2"] = True
    except Exception as e:
        if "duplicate column name" not in str(e).lower():
            logger.warning(f"memory_l2 添加 last_accessed_at 列时出现非预期异常 — {e}")

    conn.commit()
    conn.close()

    if result["l1"]:
        logger.info("memory_l1 表已新增 last_accessed_at 列")
    if result["l2"]:
        logger.info("memory_l2 表已新增 last_accessed_at 列")

    return result


def batch_update_last_accessed(layer: str, id_list: list) -> int:
    """
    批量更新一批记忆记录的 last_accessed_at 为当前时间。

    由后台回写线程（AccessBoostWorker）消费队列时调用，
    不在检索主路径上执行，不阻塞对话响应。

    参数：
        layer   — 要更新的表层级，只能是 "l1" 或 "l2"
        id_list — 需要更新的记录 id 列表，如 [3, 7, 12]
                  列表为空时直接返回 0，不执行任何 SQL

    返回：
        int — 实际更新的行数；失败时返回 0

    异常处理：
        更新失败时只记录 warning 日志，不向上抛出，
        保证后台线程不因单次写入失败而崩溃。
    """
    if not id_list:
        return 0

    # 只允许操作这两张表，防止参数注入
    table_map = {
        "l1": "memory_l1",
        "l2": "memory_l2",
    }
    if layer not in table_map:
        logger.warning(f"batch_update_last_accessed: 未知 layer={layer!r}，跳过")
        return 0

    table        = table_map[layer]
    now          = _now()
    placeholders = ",".join("?" * len(id_list))

    try:
        conn   = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {table} SET last_accessed_at = ? WHERE id IN ({placeholders})",
            [now] + list(id_list)
        )
        conn.commit()
        updated = cursor.rowcount
        conn.close()
        logger.debug(
            f"batch_update_last_accessed: layer={layer}，"
            f"更新 {updated} 条（ids={id_list[:5]}{'...' if len(id_list) > 5 else ''}）"
        )
        return updated

    except Exception as e:
        logger.warning(f"batch_update_last_accessed 写入失败，layer={layer} — {e}")
        return 0


def get_last_accessed_at(layer: str, record_id: int):
    """
    读取单条记忆记录的 last_accessed_at 字段。

    供 vector_store._retrieve() 在计算保底加成时使用：
        如果 last_accessed_at 不为 NULL 且在 RECENT_BOOST_DAYS 天内，
        则对该条记忆的保留率 R 设置下限。

    参数：
        layer     — "l1" 或 "l2"
        record_id — memory_l1 或 memory_l2 表的主键 id

    返回：
        str  — ISO 8601 时间戳字符串，如 "2026-03-28T14:30:00+00:00"
        None — 字段为 NULL（从未被访问）或记录不存在时返回 None

    注意：
        此函数在每次检索命中时都会被调用，需保持轻量。
        SQLite 的单行 SELECT 性能足够，通常在 1ms 以内。
    """
    table_map = {"l1": "memory_l1", "l2": "memory_l2"}
    if layer not in table_map:
        return None

    table = table_map[layer]

    try:
        conn   = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT last_accessed_at FROM {table} WHERE id = ?",
            (record_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return row["last_accessed_at"] if row else None

    except Exception as e:
        logger.warning(f"get_last_accessed_at 查询失败，layer={layer} id={record_id} — {e}")
        return None

# =============================================================================
# graph_nodes 表操作
# =============================================================================

def get_node_by_name(entity_name: str):
    """
    按实体名精确查询图谱节点。

    注意：查询前调用方应已完成归一化，传入的是规范词。
    本函数不做归一化处理，只做精确匹配。

    参数：
        entity_name — 归一化后的实体名

    返回：
        sqlite3.Row — 节点记录；不存在时返回 None
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM graph_nodes WHERE entity_name = ?",
        (entity_name,)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_or_create_node(entity_name: str, entity_type: str, source_l1_id: int) -> int:
    """
    获取或创建一个图谱节点，返回节点 id。

    如果节点已存在：use_count + 1，返回现有 id。
    如果节点不存在：插入新行，返回新 id。

    参数：
        entity_name  — 归一化后的实体名（规范词）
        entity_type  — 实体类型，五选一：person/project/module/concept/time
        source_l1_id — 触发此节点创建的 L1 摘要 id

    返回：
        int — 节点 id
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # 先查是否已存在
    cursor.execute(
        "SELECT id FROM graph_nodes WHERE entity_name = ?",
        (entity_name,)
    )
    row = cursor.fetchone()

    if row:
        # 已存在，use_count + 1
        cursor.execute(
            "UPDATE graph_nodes SET use_count = use_count + 1 WHERE id = ?",
            (row["id"],)
        )
        node_id = row["id"]
    else:
        # 不存在，插入新节点
        cursor.execute(
            """
            INSERT INTO graph_nodes (entity_name, entity_type, source_l1_id, created_at, use_count)
            VALUES (?, ?, ?, ?, 1)
            """,
            (entity_name, entity_type, source_l1_id, _now())
        )
        node_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return node_id


def get_canonical_name(word_id: int) -> str | None:
    """
    给定 keyword_pool 里的词 id，追溯到规范词的文本。

    如果 canonical_id 为 NULL，说明本词就是规范词，直接返回自身的 keyword。
    如果 canonical_id 非 NULL，跳转到规范词，返回规范词的 keyword。

    参数：
        word_id — keyword_pool 表的主键 id

    返回：
        str  — 规范词文本
        None — word_id 不存在时返回 None
    """
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT keyword, canonical_id FROM keyword_pool WHERE rowid = ?",
        (word_id,)
    )
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return None

    if row["canonical_id"] is None:
        # 本词就是规范词
        conn.close()
        return row["keyword"]

    # 追溯到规范词（只追一层，设计上不允许多级别名）
    cursor.execute(
        "SELECT keyword FROM keyword_pool WHERE rowid = ?",
        (row["canonical_id"],)
    )
    canonical_row = cursor.fetchone()
    conn.close()
    return canonical_row["keyword"] if canonical_row else None


def get_all_canonical_keywords() -> list[dict]:
    """
    取出 keyword_pool 里所有规范词（canonical_id 为 NULL 且已确认的词）。

    用于实体归一化时计算向量相似度：
    只和规范词比较，不和别名比较，避免别名之间互相匹配。

    返回：
        list[dict] — 每个元素包含 rowid / keyword / use_count
    """
    conn = _get_connection()
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
    conn.close()
    return [{"id": r["rowid"], "keyword": r["keyword"], "use_count": r["use_count"]}
            for r in rows]


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
    """
    保存一条图谱边，返回新边的 id。

    注意：不做重复检查，同一对节点之间可以存在多条边
    （来自不同 L1，代表不同时间发生的同类关系）。
    如果需要查重，由调用方在写入前自行检查。

    参数：
        source_node_id  — 主语节点 id（graph_nodes.id）
        target_node_id  — 宾语节点 id（graph_nodes.id）
        relation_type   — 关系类型，七类大类之一
        relation_detail — 模型提取的原始细节描述
        source_l1_id    — 来源 L1 摘要 id

    返回：
        int — 新边的 id
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO graph_edges
            (source_node_id, target_node_id, relation_type, relation_detail,
             source_l1_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_node_id, target_node_id, relation_type, relation_detail,
         source_l1_id, _now())
    )
    conn.commit()
    edge_id = cursor.lastrowid
    conn.close()
    return edge_id


def get_l1_ids_by_node(node_id: int, max_hops: int = 2) -> list[dict]:
    """
    从指定节点出发，做广度优先遍历，返回关联的 L1 id 列表。

    用于图谱检索通道：给定查询实体的节点 id，
    找出所有在逻辑上与该实体相关的历史 L1 摘要。

    参数：
        node_id  — 起始节点 id（graph_nodes.id）
        max_hops — 最大跳数，默认 2（1跳=直接相连，2跳=通过中间节点间接相连）

    返回：
        list[dict] — 每个元素包含：
            l1_id — L1 摘要 id
            hops  — 距起始节点的跳数（用于 RRF 排名，跳数越少排名越靠前）
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # 广度优先遍历：用集合记录已访问节点，避免环路
    visited_nodes = {node_id}
    current_level = {node_id}
    result = []         # [(l1_id, hops), ...]
    seen_l1_ids = set() # 同一个 L1 可能通过多条路径被找到，只记一次

    for hop in range(1, max_hops + 1):
        if not current_level:
            break

        # 查询当前层所有节点的出边和入边（无向图遍历）
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
            # 收集 L1 id
            l1_id = edge["source_l1_id"]
            if l1_id not in seen_l1_ids:
                result.append({"l1_id": l1_id, "hops": hop})
                seen_l1_ids.add(l1_id)

            # 收集下一层节点（未访问过的）
            for neighbor_id in (edge["source_node_id"], edge["target_node_id"]):
                if neighbor_id not in visited_nodes:
                    next_level.add(neighbor_id)
                    visited_nodes.add(neighbor_id)

        current_level = next_level

    conn.close()
    return result


def get_all_l1_ids_in_graph() -> set[int]:
    """
    返回图谱中已经处理过的所有 L1 id 集合。

    用于批处理时判断哪些 L1 还没有提取图谱三元组。
    graph_builder.py 的 _get_pending_l1() 会调用此函数。

    返回：
        set[int] — 已在图谱中出现的 L1 id 集合
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT source_l1_id FROM graph_edges")
    rows = cursor.fetchall()
    conn.close()
    return {row["source_l1_id"] for row in rows}


def get_node_by_id(node_id: int):
    """
    按节点 id 查询图谱节点。

    参数：
        node_id — graph_nodes.id

    返回：
        sqlite3.Row — 节点记录；不存在时返回 None
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM graph_nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    conn.close()
    return row


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

    与原有 upsert_keywords() 的区别：
        upsert_keywords 用于普通关键词写入，不涉及别名字段。
        本函数专用于图谱实体归一化流程，需要同时写入 canonical_id 和 alias_status。

    参数：
        keyword      — 关键词文本
        canonical_id — 规范词的 rowid；为 None 表示本词就是规范词
        alias_status — 'confirmed' / 'pending' / 'canonical' 三态之一

    返回：
        int — 新写入词条的 rowid
    """
    now = _now()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO keyword_pool (keyword, use_count, last_used_at, created_at,
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
    # 获取这条记录的 rowid
    cursor.execute("SELECT rowid FROM keyword_pool WHERE keyword = ?", (keyword,))
    row = cursor.fetchone()
    conn.close()
    return row["rowid"]


def get_pending_aliases() -> list:
    """
    查询所有待确认的别名记录（alias_status = 'pending'）。

    返回：
        list[sqlite3.Row] — 每行包含 rowid / keyword / canonical_id / alias_status
    """
    conn = _get_connection()
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
    rows = cursor.fetchall()
    conn.close()
    return rows


def confirm_alias(new_word_id: int, canonical_id: int) -> bool:
    """
    确认别名关系：将 new_word_id 对应的词正式归入 canonical_id 对应的规范词。

    执行三步操作（包裹在同一事务中）：
      1. 更新 keyword_pool：alias_status 改为 confirmed，canonical_id 固定
      2. 迁移 graph_edges：所有指向 new_word 节点的边改指向规范词节点
      3. 合并 graph_nodes：将 new_word 节点的 use_count 累加到规范词节点，
         然后删除 new_word 节点

    参数：
        new_word_id  — 待归入的词在 keyword_pool 的 rowid
        canonical_id — 规范词在 keyword_pool 的 rowid

    返回：
        bool — True 表示成功，False 表示任何步骤失败（已自动回滚）
    """
    conn = _get_connection()
    cursor = conn.cursor()

    try:
        # 查出 new_word 和规范词各自对应的 graph_nodes.id
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
                f"confirm_alias: 找不到词条 new_word_id={new_word_id} "
                f"或 canonical_id={canonical_id}"
            )
            conn.close()
            return False

        new_word      = new_word_row["keyword"]
        canonical_word = canonical_row["keyword"]

        # 查出对应的 graph_nodes.id
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

        # ── 步骤 1：更新 keyword_pool ──
        cursor.execute(
            """
            UPDATE keyword_pool
            SET canonical_id = ?, alias_status = 'confirmed'
            WHERE rowid = ?
            """,
            (canonical_id, new_word_id)
        )

        # ── 步骤 2 & 3：迁移图谱（只在两个节点都存在时执行）──
        if new_node and canonical_node:
            new_node_id       = new_node["id"]
            canonical_node_id = canonical_node["id"]
            merged_count      = new_node["use_count"]

            # 迁移边：source 指向 new_node 的改为指向 canonical_node
            cursor.execute(
                """
                UPDATE graph_edges
                SET source_node_id = ?
                WHERE source_node_id = ?
                """,
                (canonical_node_id, new_node_id)
            )

            # 迁移边：target 指向 new_node 的改为指向 canonical_node
            cursor.execute(
                """
                UPDATE graph_edges
                SET target_node_id = ?
                WHERE target_node_id = ?
                """,
                (canonical_node_id, new_node_id)
            )

            # 合并 use_count
            cursor.execute(
                """
                UPDATE graph_nodes
                SET use_count = use_count + ?
                WHERE id = ?
                """,
                (merged_count, canonical_node_id)
            )

            # 删除已被吸收的节点
            cursor.execute(
                "DELETE FROM graph_nodes WHERE id = ?",
                (new_node_id,)
            )

        conn.commit()
        logger.info(
            f"confirm_alias: '{new_word}' → '{canonical_word}' 别名关系已确认"
        )
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"confirm_alias 事务失败，已回滚 — {e}")
        return False
    finally:
        conn.close()


def reject_alias(new_word_id: int) -> bool:
    """
    拒绝别名关系：将 pending 词独立为新的规范词。

    参数：
        new_word_id — 待处理词在 keyword_pool 的 rowid

    返回：
        bool — True 表示成功
    """
    conn = _get_connection()
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
    finally:
        conn.close()


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

    与普通画像冲突的区别：
        conflict_type = 'alias_confirm'
        field         = 'keyword_alias'（复用 field 字段做类型标记）
        old_content   = 候选规范词（old_word）
        new_content   = 待确认词（new_word）
        conflict_desc = 供对话展示的询问文本

    参数：
        source_l1_id    — 触发此别名检测的 L1 摘要 id
        old_word        — 候选规范词（keyword_pool 中已有的高频词）
        new_word        — 新提取的词（相似度在 0.85-0.95 之间）
        new_word_kp_id  — new_word 在 keyword_pool 的 rowid
        canonical_kp_id — old_word（规范词）在 keyword_pool 的 rowid
        similarity      — 两词的余弦相似度（写入供调试参考）

    返回：
        int — 新冲突记录的 id
    """
    conflict_desc = (
        f'"{new_word}"和"{old_word}"是指同一个事物吗？'
        f'（相似度 {similarity:.0%}，回复"是"确认合并，回复"不是"保持独立）'
    )

    conn = _get_connection()
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

    # 同时在 conflict_queue 记录里用 resolved_at 字段暂存两个 kp_id
    # 格式：new_word_kp_id:canonical_kp_id（用冒号分隔，解析时 split(':')）
    # 注意：resolved_at 正常情况存时间戳，这里临时借用存储 id，
    # 一旦用户确认后会被写入真实时间戳覆盖。
    # 为了避免歧义，改为在 old_content 字段末尾追加 id 信息（用 \n 分隔）。
    #
    # 更清洁的做法：直接在 conflict_desc 里解析，或者另建一张 alias_pending 表。
    # 当前选择用 old_content 追加，减少新增表，后续如果觉得别扭可以重构。
    cursor.execute(
        """
        UPDATE conflict_queue
        SET old_content = ?
        WHERE id = ?
        """,
        (f"{old_word}\n__kp_ids__{new_word_kp_id}:{canonical_kp_id}", conflict_id)
    )
    conn.commit()
    conn.close()

    logger.info(
        f"save_alias_conflict: '{new_word}' vs '{old_word}' "
        f"（相似度 {similarity:.2f}），conflict_id={conflict_id}"
    )
    return conflict_id


def get_alias_kp_ids_from_conflict(conflict_id: int) -> tuple[int, int] | None:
    """
    从 conflict_queue 记录中解析出 (new_word_kp_id, canonical_kp_id)。

    供 conflict_checker.handle_conflict_reply() 的 alias_confirm 分支调用。

    参数：
        conflict_id — conflict_queue 表的记录 id

    返回：
        (new_word_kp_id, canonical_kp_id) — 解析成功时返回二元组
        None — 解析失败时返回 None
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT old_content FROM conflict_queue WHERE id = ?",
        (conflict_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    old_content = row["old_content"]
    try:
        # 格式："{old_word}\n__kp_ids__{new_word_kp_id}:{canonical_kp_id}"
        parts      = old_content.split("\n__kp_ids__")
        ids_str    = parts[1]
        new_id, canonical_id = ids_str.split(":")
        return int(new_id), int(canonical_id)
    except Exception as e:
        logger.warning(f"get_alias_kp_ids_from_conflict 解析失败：{e}")
        return None

# =============================================================================
# 直接运行此文件时：执行连通性验证
# =============================================================================

if __name__ == "__main__":
    print("=== database.py 连通性验证 ===\n")

    # 1. 新建 session
    sid = new_session()
    print(f"[1] new_session()                    → session id = {sid}")

    # 2. 存两条消息
    mid1 = save_message(sid, "user", "database.py 验证消息")
    mid2 = save_message(sid, "assistant", "收到，数据库读写正常。")
    print(f"[2] save_message()                   → message id = {mid1}, {mid2}")

    # 3. 读回消息
    msgs = get_messages_as_dicts(sid)
    print(f"[3] get_messages_as_dicts()          → {len(msgs)} 条消息")

    # 4. 写一条 L1 摘要
    l1_id = save_l1_summary(
        session_id=sid, summary="烧酒完成了 database.py 的连通性验证。",
        keywords="数据库,验证,后端", time_period="夜间", atmosphere="专注高效"
    )
    print(f"[4] save_l1_summary()                → L1 id = {l1_id}")

    # 5. 按主键查询 L1
    row = get_l1_by_id(l1_id)
    print(f"[5] get_l1_by_id({l1_id})               → {row['summary'] if row else None}")
    print(f"    get_l1_by_id(99999)              → {get_l1_by_id(99999)}（应为 None）")

    # 6. 验证 get_unabsorbed_l1() 的 P3-1 修复：有无 limit 都走同一段 SQL
    print(f"\n--- P3-1 修复验证：get_unabsorbed_l1() ---")
    rows_all   = get_unabsorbed_l1()
    rows_limit = get_unabsorbed_l1(limit=1)
    print(f"[6] get_unabsorbed_l1()              → {len(rows_all)} 条（全量）")
    print(f"    get_unabsorbed_l1(limit=1)       → {len(rows_limit)} 条（应为 1）")

    # 7. 验证三个公开函数（v3 问题A修复）
    print(f"\n--- 问题A修复验证 ---")
    print(f"[7] get_all_l1()                     → {len(get_all_l1())} 条")
    print(f"    get_all_l2()                     → {len(get_all_l2())} 条")
    print(f"    get_all_session_ids()            → {get_all_session_ids()}")

    # 8. 验证 update_message_time_for_test()（P1-2 修复核心）
    print(f"\n--- P1-2 修复验证：update_message_time_for_test() ---")
    from datetime import timedelta
    fake_time = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    update_message_time_for_test(sid, fake_time)
    new_time = get_last_message_time(sid)
    print(f"[8] update_message_time_for_test()   → 改后最后消息时间 = {new_time[:19]}")
    assert fake_time[:19] == new_time[:19], "时间戳修改未生效"
    print(f"    时间戳修改验证通过 ✓")

    # 9. 关闭 session 并读配置
    close_session(sid)
    print(f"\n[9] close_session()                  → ok")
    print(f"[10] get_setting()                   → l1_idle_minutes = {get_setting('l1_idle_minutes', '未找到')}")

    print("\n验证通过。")
