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

使用方法（在其他模块里）：
    from database import save_message, get_messages, save_l1_summary, ...
"""

import sqlite3
from datetime import datetime, timezone
from config import DB_PATH


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


# =============================================================================
# memory_l1 表操作
# =============================================================================

def save_l1_summary(session_id, summary, keywords, time_period, atmosphere):
    """
    将一条 L1 摘要写入 memory_l1 表。

    参数：
        session_id  — 对应的 session id
        summary     — 一句话摘要（第三人称，只记结论，不超过50字）
        keywords    — 名词标签字符串，英文逗号分隔，如 "项目,后端,FastAPI"
        time_period — 时间段标签，六选一：清晨/上午/下午/傍晚/夜间/深夜
        atmosphere  — 对话氛围，四字以内，如 "专注高效"

    返回：
        int — 新插入 L1 记录的 id
    """
    from config import TIME_PERIOD_OPTIONS
    if time_period not in TIME_PERIOD_OPTIONS:
        print(f"[database] 警告：time_period 值 {time_period!r} 不在合法列表内，已置为 None")
        time_period = None

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memory_l1 (session_id, summary, keywords, time_period, atmosphere, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, summary, keywords, time_period, atmosphere, _now()))
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


def get_unabsorbed_l1(limit=None):
    """
    取出所有尚未被 L2 吸收的 L1 摘要，按生成时间升序排列。

    参数：
        limit — 最多返回多少条；不传则返回全部

    返回：
        列表，每个元素是 sqlite3.Row；没有未吸收记录时返回空列表。

    用途：
        L2 触发判断 + L2 合并时读取待处理的 L1 列表。
    """
    conn = _get_connection()
    cursor = conn.cursor()
    if limit:
        cursor.execute("""
            SELECT * FROM memory_l1
            WHERE absorbed = 0
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,))
    else:
        cursor.execute("""
            SELECT * FROM memory_l1
            WHERE absorbed = 0
            ORDER BY created_at ASC
        """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_l1():
    """
    返回 memory_l1 表中的全部记录（id, summary, keywords, session_id）。

    [v3 新增] 修复审查报告问题A：vector_store.rebuild_all_indexes() 原来直接
    import 私有函数 _get_connection 并手写原始 SQL 查询所有 L1，破坏了
    database.py 作为数据层统一出口的封装原则。

    改为通过本函数访问，外部模块无需感知表结构细节：
      旧写法（vector_store.py）：
        from database import _get_connection          # 私有函数泄漏
        conn = _get_connection()
        conn.cursor().execute("SELECT id, summary, keywords, session_id FROM memory_l1")
      新写法：
        from database import get_all_l1
        rows = get_all_l1()

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

    [v3 新增] 修复审查报告问题A：与 get_all_l1() 同理，消除 vector_store.py
    对私有函数 _get_connection 的依赖。

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

    [v3 新增] 修复审查报告问题A：与 get_all_l1() 同理，消除 vector_store.py
    对私有函数 _get_connection 的依赖。

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
        print(f"[database] 错误：save_l2_summary 事务失败，已回滚 — {e}")
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

    cursor.execute("""
        UPDATE user_profile SET is_current = 0
        WHERE field = ? AND is_current = 1
    """, (field,))

    cursor.execute("""
        INSERT INTO user_profile
            (field, content, source_l1_id, status, is_current, updated_at)
        VALUES (?, ?, ?, 'approved', 1, ?)
    """, (field, new_content, source_l1_id, _now()))

    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


# =============================================================================
# 直接运行此文件时：执行连通性验证
# =============================================================================

if __name__ == "__main__":
    print("=== database.py 连通性验证 ===\n")

    # 1. 新建 session
    sid = new_session()
    print(f"[1] new_session()              → session id = {sid}")

    # 2. 存两条消息
    mid1 = save_message(sid, "user", "database.py 验证消息")
    mid2 = save_message(sid, "assistant", "收到，数据库读写正常。")
    print(f"[2] save_message()             → message id = {mid1}, {mid2}")

    # 3. 读回消息
    msgs = get_messages_as_dicts(sid)
    print(f"[3] get_messages_as_dicts()    → {len(msgs)} 条消息")

    # 4. 写一条 L1 摘要
    l1_id = save_l1_summary(
        session_id=sid, summary="烧酒完成了 database.py 的连通性验证。",
        keywords="数据库,验证,后端", time_period="夜间", atmosphere="专注高效"
    )
    print(f"[4] save_l1_summary()          → L1 id = {l1_id}")

    # 5. 按主键查询 L1
    row = get_l1_by_id(l1_id)
    print(f"[5] get_l1_by_id({l1_id})         → {row['summary'] if row else None}")
    print(f"    get_l1_by_id(99999)        → {get_l1_by_id(99999)}（应为 None）")

    # 6. 验证三个新增公开函数（问题A修复核心）
    print(f"\n--- 问题A修复验证 ---")
    print(f"[6] get_all_l1()               → {len(get_all_l1())} 条")
    print(f"    get_all_l2()               → {len(get_all_l2())} 条")
    print(f"    get_all_session_ids()      → {get_all_session_ids()}")

    # 7. 关闭 session 并读配置
    close_session(sid)
    print(f"\n[7] close_session()            → ok")
    print(f"[8] get_setting()              → l1_idle_minutes = {get_setting('l1_idle_minutes', '未找到')}")

    print("\n验证通过。")
