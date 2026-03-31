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
