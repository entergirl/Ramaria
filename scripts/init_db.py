"""
init_db.py — 数据库初始化脚本
运行此脚本会在同目录下创建 assistant.db 文件，并建好所有表。
可以反复运行，已存在的表不会被覆盖（IF NOT EXISTS 保护）。

变更记录：
    v2 — 给 memory_l1 表新增 time_period（时间段）和 atmosphere（对话氛围）字段，
         与 L1 摘要 Prompt 的输出结构对齐。
         如需对已有数据库执行同等变更，请运行 migrate_add_l1_fields.py。

使用方法：
    python init_db.py
"""

import sqlite3
import os
from datetime import datetime, timezone
from ramaria.config import DB_PATH

# =============================================================================
# 数据库连接
# =============================================================================

def get_connection():
    """
    获取数据库连接。
    后续所有操作都通过这个函数拿连接，方便以后统一修改路径。
    开启外键约束（SQLite 默认不开启），保证 FOREIGN KEY 生效。
    """
    conn = sqlite3.connect(DB_PATH)
    # 让查询结果可以用列名访问，比如 row["content"] 而不是 row[0]
    conn.row_factory = sqlite3.Row
    # SQLite 需要每次连接时手动开启外键支持
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# =============================================================================
# 表结构初始化
# =============================================================================

def init_db():
    """
    创建所有表。每张表都用 IF NOT EXISTS，重复运行不会报错、不会覆盖数据。
    """
    conn = get_connection()
    cursor = conn.cursor()

    # -------------------------------------------------------------------------
    # 表1：sessions — 对话 session 管理
    #
    # 每次用户发起对话，创建一个新 session。
    # 空闲超过阈值后，由后端将 ended_at 写入，标志 session 结束。
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- 唯一标识，自动递增
            started_at  TEXT NOT NULL,                      -- session 开始时间（ISO 8601 格式）
            ended_at    TEXT                                -- session 结束时间；对话进行中时为 NULL
        )
    """)

    # -------------------------------------------------------------------------
    # 表2：messages — L0 原始消息流水
    #
    # 每一条用户或助手的消息存一行，永久保留，不删除。
    # 是所有上层摘要（L1/L2）的原始数据来源。
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,                   -- 属于哪个 session（外键）
            role        TEXT    NOT NULL,                   -- 发言方："user" 或 "assistant"
            content     TEXT    NOT NULL,                   -- 消息正文
            created_at  TEXT    NOT NULL,                   -- 发送时间（ISO 8601 格式）

            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # -------------------------------------------------------------------------
    # 表3：memory_l1 — 单次 session 摘要（L1 记忆层）
    #
    # 每个 session 结束后（空闲超过阈值），由后端自动触发生成，存一行。
    # 字段说明：
    #   summary     — 第三人称客观描述，只记结论，不记过程
    #   keywords    — 3~5 个名词标签，逗号分隔，无语义重复
    #   time_period — 对话发生的大致时间段，六选一（见下方合法值）
    #   atmosphere  — 对话整体氛围，四字以内
    #   absorbed    — 是否已被 L2 合并吸收：0=未吸收，1=已吸收
    #
    # time_period 合法值（六选一）：
    #   "清晨"（06:00前）、"上午"（06-12）、"下午"（12-18）、
    #   "傍晚"（18-21）、"夜间"（21-24）、"深夜"（00-06）
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_l1 (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,                   -- 对应哪个 session（外键）
            summary     TEXT    NOT NULL,                   -- 一句话摘要（第三人称，只记结论）
            keywords    TEXT,                               -- 名词标签，逗号分隔，如 "项目,后端,FastAPI"
            time_period TEXT,                               -- 时间段标签，六选一，见上方说明
            atmosphere  TEXT,                               -- 对话氛围，四字以内，如 "专注高效"
            created_at  TEXT    NOT NULL,                   -- 摘要生成时间（ISO 8601 格式）
            absorbed    INTEGER NOT NULL DEFAULT 0,         -- L2 吸收标志：0=未吸收，1=已吸收

            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # -------------------------------------------------------------------------
    # 表4：memory_l2 — 时间段聚合摘要（L2 记忆层）
    #
    # 由多条 L1 合并生成，触发条件（满足其一即触发）：
    #   - 未吸收的 L1 累计达到 N 条（默认5条，见 settings 表）
    #   - 最早一条未吸收 L1 距今超过 M 天（默认7天）
    # period_start / period_end 记录这条 L2 覆盖的实际时间范围。
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_l2 (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            summary         TEXT    NOT NULL,               -- 聚合摘要正文
            keywords        TEXT,                           -- 关键词，逗号分隔
            period_start    TEXT    NOT NULL,               -- 覆盖时间段的起点（ISO 8601）
            period_end      TEXT    NOT NULL,               -- 覆盖时间段的终点（ISO 8601）
            created_at      TEXT    NOT NULL                -- 本条 L2 的生成时间（ISO 8601）
        )
    """)

    # -------------------------------------------------------------------------
    # 表5：l2_sources — L2 溯源关联表
    #
    # 记录每条 L2 是由哪些 L1 合并而来。
    # 这是一张多对多关系表：一条 L2 对应多条 L1。
    # 用途：调试时追溯 L2 的原始来源；将来支持重新生成或回滚。
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS l2_sources (
            l2_id   INTEGER NOT NULL,                       -- 关联 memory_l2.id
            l1_id   INTEGER NOT NULL,                       -- 关联 memory_l1.id

            PRIMARY KEY (l2_id, l1_id),                    -- 组合主键，防止重复关联
            FOREIGN KEY (l2_id) REFERENCES memory_l2(id),
            FOREIGN KEY (l1_id) REFERENCES memory_l1(id)
        )
    """)

    # -------------------------------------------------------------------------
    # 表6：user_profile — 长期用户画像（L3 记忆层）
    #
    # 按六大板块分行存储，每次更新插入新行（不覆盖旧行），保留历史版本。
    # is_current = 1 表示当前生效版本，注入 system prompt 时只读这些行。
    # is_current = 0 表示历史版本，保留用于追踪变化。
    # status 三态：pending（待确认）→ approved（已生效）或 rejected（已拒绝）
    #
    # field 合法值（六选一）：
    #   "basic_info"      基础信息（姓名、年龄、所在地等）
    #   "personal_status" 个人状况（当前状态、近期情绪等）
    #   "interests"       兴趣爱好与关注领域
    #   "social"          社交情况（重要人际关系）
    #   "history"         历史事件（重要经历与里程碑）
    #   "recent_context"  近期背景（作为 L3 的动态补充，可选）
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            field           TEXT    NOT NULL,               -- 板块名，见上方合法值
            content         TEXT    NOT NULL,               -- 该板块的内容文本
            source_l1_id    INTEGER,                        -- 来源 L1 的 id（手动录入时为 NULL）
            status          TEXT    NOT NULL DEFAULT 'pending',
                                                            -- "pending"=待确认 / "approved"=已生效 / "rejected"=已拒绝
            is_current      INTEGER NOT NULL DEFAULT 0,     -- 当前生效版本标志：1=是，0=否
            updated_at      TEXT    NOT NULL,               -- 本行写入时间（ISO 8601）

            FOREIGN KEY (source_l1_id) REFERENCES memory_l1(id)
        )
    """)

    # -------------------------------------------------------------------------
    # 表7：settings — 全局运行配置
    #
    # key-value 结构，避免将配置硬编码在业务逻辑里。
    # value 统一存为字符串，调用方在读取后自行转换类型。
    #
    # 内置配置项说明：
    #   profile_mode     → "manual"（需用户确认）或 "auto"（全自动写入）
    #   l1_idle_minutes  → 触发 L1 摘要的空闲分钟数，默认 "10"
    #   l2_trigger_count → 触发 L2 合并所需的未吸收 L1 条数，默认 "5"
    #   l2_trigger_days  → 触发 L2 合并的最大等待天数，默认 "7"
    #   default_model    → 默认调用的模型："local" 或 "claude"
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key         TEXT PRIMARY KEY,                   -- 配置项名称（唯一）
            value       TEXT NOT NULL,                      -- 配置项值（字符串格式）
            updated_at  TEXT NOT NULL                       -- 最近修改时间（ISO 8601）
        )
    """)

    # 写入默认配置
    # INSERT OR IGNORE：key 已存在则跳过，不会覆盖用户修改过的值
    now = datetime.now(timezone.utc).isoformat()
    default_settings = [
        ("profile_mode",     "manual", now),   # 默认半自动：profile 更新需用户确认
        ("l1_idle_minutes",  "10",     now),   # 空闲 10 分钟触发 L1 摘要生成
        ("l2_trigger_count", "5",      now),   # 积累 5 条未吸收 L1 时触发 L2 合并
        ("l2_trigger_days",  "7",      now),   # 或最早一条 L1 已超过 7 天时触发
        ("default_model",    "local",  now),   # 默认使用本地模型处理日常任务
    ]
    cursor.executemany("""
        INSERT OR IGNORE INTO settings (key, value, updated_at)
        VALUES (?, ?, ?)
    """, default_settings)

    # -------------------------------------------------------------------------
    # 表8：keyword_pool — 关键词词典
    #
    # 存储历史上出现过的所有关键词，供 L1 摘要生成时复用，避免同义词发散。
    # keyword 本身作为主键，天然去重。
    # 每次 L1 生成时，将词典现有词条作为候选列表喂给模型；
    # 模型选用已有词时更新 use_count 和 last_used_at；
    # 确实无法匹配时新增词条，use_count 从 1 开始。
    #
    # 字段说明：
    #   keyword     — 关键词文本，唯一主键
    #   use_count   — 累计被使用次数，首次写入时为 1
    #   last_used_at — 最近一次被使用的时间（ISO 8601）
    #   created_at  — 首次写入时间（ISO 8601）
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS keyword_pool (
            keyword      TEXT PRIMARY KEY,    -- 关键词本身，唯一，天然去重
            use_count    INTEGER NOT NULL DEFAULT 1,  -- 累计使用次数
            last_used_at TEXT    NOT NULL,    -- 最近使用时间（ISO 8601）
            created_at   TEXT    NOT NULL     -- 首次写入时间（ISO 8601）
        )
    """)

    # -------------------------------------------------------------------------
    # 表9：conflict_queue — 冲突检测待确认队列
    #
    # L1 摘要生成后，冲突检测模块将发现的矛盾写入此表，等待用户确认。
    # 不直接修改任何记忆层，只记录"发现了什么冲突、旧值是什么、新值是什么"。
    # 用户通过对话界面确认后，由 conflict_checker 根据 status 决定后续操作。
    #
    # status 三态流转：
    #   pending   — 刚写入，尚未向用户询问，或用户还未回复
    #   resolved  — 用户确认接受新内容，对应记忆层已更新
    #   ignored   — 用户选择忽略，新旧两个版本均保留，本条记录仅作存档
    #
    # field 对应 user_profile 表的六大板块（见 user_profile 表注释）
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conflict_queue (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source_l1_id INTEGER NOT NULL,    -- 触发此冲突的 L1 记录 id（外键）
            field        TEXT    NOT NULL,    -- 涉及的画像板块，如 "personal_status"
            old_content  TEXT    NOT NULL,    -- 现有画像内容（冲突发生时的快照）
            new_content  TEXT    NOT NULL,    -- L1 里检测到的新信息（与旧内容矛盾）
            conflict_desc TEXT   NOT NULL,    -- 模型生成的冲突描述，用于向用户展示
            status       TEXT    NOT NULL DEFAULT 'pending',
                                              -- "pending" / "resolved" / "ignored"
            created_at   TEXT    NOT NULL,    -- 写入时间（ISO 8601）
            resolved_at  TEXT,               -- 用户确认时间；pending 时为 NULL

            FOREIGN KEY (source_l1_id) REFERENCES memory_l1(id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"数据库初始化完成：{DB_PATH}")
    print("已创建表：sessions, messages, memory_l1, memory_l2, l2_sources, user_profile, settings, keyword_pool, conflict_queue, pending_push")


# -------------------------------------------------------------------------
    # 表10：pending_push — 主动推送消息暂存表
    #
    # 用于暂存用户离线时触发的主动推送消息。
    # 用户上线（WebSocket 连接建立）时，由 main.py 检查此表并推送。
    #
    # status 两态：
    #   pending — 尚未推送（用户离线或推送中）
    #   sent    — 已成功推送给用户
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_push (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'pending',
            sent_at     TEXT
        )
    """)

    # 写入推送相关默认配置项（INSERT OR IGNORE，不覆盖已有值）
    push_settings = [
        ("debounce_seconds",  "3",  now),
        ("push_enabled",      "1",  now),
        ("push_window_start", "8",  now),
        ("push_window_end",   "24", now),
        ("push_daily_limit",  "4",  now),
    ]
    cursor.executemany("""
        INSERT OR IGNORE INTO settings (key, value, updated_at)
        VALUES (?, ?, ?)
    """, push_settings)

# ------------------------------------------------------------------
    # 建立高频查询字段的索引
    #
    # 放在建表语句之后、commit 之前。
    # CREATE INDEX IF NOT EXISTS 保证幂等：重复运行 init_db.py 不会报错。
    #
    # 索引作用说明：
    #   idx_messages_session_id    — get_messages() 每次对话都走，最高频
    #   idx_messages_created_at    — get_last_message_time() 排序，空闲检测每分钟触发
    #   idx_memory_l1_session_id   — get_l1_by_session() 查询
    #   idx_memory_l1_absorbed     — get_unabsorbed_l1() 每次 L1 写入后触发
    #   idx_memory_l1_created_at   — L2 触发检查排序 / get_latest_l1()
    #   idx_conflict_queue_status  — get_pending_conflicts() 每次对话前检查
    #   idx_pending_push_status    — get_pending_pushes() WebSocket 建立时查询
    # ------------------------------------------------------------------
    index_sqls = [
        "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_session_id ON memory_l1(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_absorbed ON memory_l1(absorbed)",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_created_at ON memory_l1(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_conflict_queue_status ON conflict_queue(status)",
        "CREATE INDEX IF NOT EXISTS idx_pending_push_status ON pending_push(status)",
    ]
    for sql in index_sqls:
        cursor.execute(sql)

# =============================================================================
# 辅助函数 — 后续模块直接 import 使用
# =============================================================================

def new_session():
    """
    创建一个新 session，返回新 session 的 id（int）。
    调用时机：用户发来第一条消息，且当前没有活跃 session 时。
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("INSERT INTO sessions (started_at) VALUES (?)", (now,))
    conn.commit()
    session_id = cursor.lastrowid   # 取刚插入行的自增 id
    conn.close()
    return session_id


def close_session(session_id):
    """
    关闭一个 session，将 ended_at 写入当前时间。
    调用时机：空闲检测超时后，由后端触发。
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        UPDATE sessions SET ended_at = ? WHERE id = ?
    """, (now, session_id))
    conn.commit()
    conn.close()


def save_message(session_id, role, content):
    """
    保存一条消息到 messages 表（L0 原始记录）。

    参数：
        session_id  — 当前 session 的 id
        role        — 发言方，"user" 或 "assistant"
        content     — 消息正文文本
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO messages (session_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, now))
    conn.commit()
    conn.close()


def get_messages(session_id):
    """
    取出某个 session 的全部消息，按时间升序排列。

    返回：
        列表，每个元素是 sqlite3.Row 对象，可用 row["role"]、row["content"] 访问。
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM messages
        WHERE session_id = ?
        ORDER BY created_at ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def save_l1_summary(session_id, summary, keywords, time_period, atmosphere):
    """
    将一条 L1 摘要写入 memory_l1 表。

    参数：
        session_id  — 对应的 session id
        summary     — 一句话摘要（第三人称，只记结论）
        keywords    — 名词标签字符串，逗号分隔，如 "项目,后端,FastAPI"
        time_period — 时间段标签，六选一：清晨/上午/下午/傍晚/夜间/深夜
        atmosphere  — 对话氛围，四字以内，如 "专注高效"

    返回：
        新插入行的 id（int）
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO memory_l1 (session_id, summary, keywords, time_period, atmosphere, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, summary, keywords, time_period, atmosphere, now))
    conn.commit()
    l1_id = cursor.lastrowid
    conn.close()
    return l1_id


def get_unabsorbed_l1(limit=None):
    """
    取出所有尚未被 L2 吸收的 L1 摘要，按生成时间升序排列。

    参数：
        limit — 最多返回多少条，不传则返回全部

    返回：
        列表，每个元素是 sqlite3.Row 对象。
    """
    conn = get_connection()
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


def get_current_profile():
    """
    取出当前生效的完整用户画像，返回字典，key 为板块名。

    示例返回：
        {"basic_info": "烧酒，19岁，...", "interests": "...", ...}

    调用时机：构建 system prompt 时注入 L3 记忆。
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT field, content FROM user_profile
        WHERE is_current = 1 AND status = 'approved'
    """)
    rows = cursor.fetchall()
    conn.close()
    return {row["field"]: row["content"] for row in rows}


def get_setting(key):
    """
    读取一个配置项的值（字符串格式）。

    示例：
        get_setting("l1_idle_minutes")  →  "10"
        int(get_setting("l1_idle_minutes"))  →  10

    返回：
        配置值字符串；key 不存在时返回 None。
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else None


def set_setting(key, value):
    """
    写入或更新一个配置项。

    参数：
        key   — 配置项名称
        value — 配置项值（传入字符串）
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (key, value, now))
    conn.commit()
    conn.close()


# =============================================================================
# 直接运行此脚本时：初始化数据库并执行简单验证
# =============================================================================

if __name__ == "__main__":
    init_db()

    print("\n--- 验证测试 ---")

    # 1. 新建 session
    sid = new_session()
    print(f"新建 session，id = {sid}")

    # 2. 存两条消息
    save_message(sid, "user", "你好！我是测试消息。")
    save_message(sid, "assistant", "你好！连接正常。")

    # 3. 读回消息
    messages = get_messages(sid)
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content']}")

    # 4. 写一条 L1 摘要（验证新字段）
    l1_id = save_l1_summary(
        session_id  = sid,
        summary     = "对话完成了初始化。",
        keywords    = "初始化",
        time_period = "夜间",
        atmosphere  = "专注高效"
    )
    print(f"\n写入 L1 摘要，id = {l1_id}")

    # 5. 读回未吸收的 L1，验证新字段是否正常存取
    l1_rows = get_unabsorbed_l1()
    for row in l1_rows:
        print(f"  L1 [{row['id']}] {row['summary']}")
        print(f"       keywords={row['keywords']}  time_period={row['time_period']}  atmosphere={row['atmosphere']}")

    # 6. 关闭 session
    close_session(sid)
    print(f"\nsession {sid} 已关闭")

    # 7. 读配置
    print(f"\n默认配置 l1_idle_minutes = {get_setting('l1_idle_minutes')} 分钟")
    print("验证通过，可以开始开发了。")
