"""
scripts/setup_db.py — 数据库一键初始化与迁移脚本
=====================================================================

职责：
    · 新环境：建立完整、正确的数据库结构（含所有表、索引、默认配置）
    · 已有环境：检测缺失的列/表/索引并幂等补齐，不破坏现有数据

设计原则：
    1. 单一入口：此文件是数据库结构的唯一权威来源，废弃原有的
       init_db.py 和所有 migrate_*.py，不再维护那些文件。
    2. 幂等安全：所有操作使用 IF NOT EXISTS / ALTER TABLE 前检查，
       可以反复运行，不会报错，不会覆盖用户数据。
    3. 完整正确：修复了原 init_db.py 中的以下已知问题：
       - pending_push 表建表 SQL 悬空在函数体外（不会被执行）
       - memory_l1 缺少 valence/salience/last_accessed_at 三列
       - messages 表缺少 import_fingerprint 列
       - graph_nodes/graph_edges 表完全不在原建表脚本中
       - conflict_queue 缺少 conflict_type 列
       - keyword_pool 缺少 canonical_id/alias_status 两列
       - 所有索引不在原建表脚本中
    4. 外键约束：SQLite 默认不开启外键检查，每次连接需手动执行
       PRAGMA foreign_keys = ON，本脚本在所有连接上均已开启。

使用方法：
    # 新环境（首次运行）或已有环境（升级）均使用同一条命令
    python scripts/setup_db.py

    # 可选：强制重建（会删除已有数据库，谨慎使用）
    python scripts/setup_db.py --force-rebuild

依赖：
    · Python 标准库 sqlite3，无需额外安装
    · 需要 src/ 在 Python 路径中（脚本会自动添加）

版本历史：
    v1.0  初始版本，整合全部历史迁移脚本
"""

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# 路径配置
# =============================================================================

# 脚本自身在 scripts/ 下，项目根目录是它的上一级
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR    = _SCRIPTS_DIR.parent
_SRC_DIR     = _ROOT_DIR / "src"

# 将 src/ 加入模块搜索路径，以便 import ramaria.config
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# 延迟导入，等路径设置完成后再 import
from ramaria.config import DB_PATH  # noqa: E402


# =============================================================================
# 数据库连接工具
# =============================================================================

def _get_conn() -> sqlite3.Connection:
    """
    获取数据库连接。
    · row_factory = sqlite3.Row 支持按列名访问，如 row["content"]
    · foreign_keys = ON 强制执行外键约束（SQLite 默认关闭）
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串，用于写入时间戳字段。"""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# 表结构定义
# =============================================================================
# 所有建表 SQL 集中在此处，便于查阅和修改。
# 修改表结构时：
#   · 新增列 → 同时在 _MIGRATIONS 列表里加对应的 ALTER TABLE
#   · 新增表 → 在 _create_all_tables() 里加建表语句
#   · 新增索引 → 在 _create_all_indexes() 里加
# =============================================================================

# -----------------------------------------------------------------------------
# 表1：sessions
# 管理对话 session 的生命周期。每次用户开启新对话创建一条记录。
# started_at：session 开始时间（注：有意保留 started_at 而非 created_at，
#             与 database.py 中大量引用保持一致，避免引入修改风险）
# ended_at：session 结束时间，对话进行中时为 NULL
# -----------------------------------------------------------------------------
_SQL_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT
)
"""

# -----------------------------------------------------------------------------
# 表2：messages
# L0 原始消息层，永久保留所有消息，是上层摘要的唯一原始来源。
#
# role：发言方，只能是 "user" 或 "assistant"（由 database.py 在写入前校验）
# import_fingerprint：历史数据导入时写入的 SHA-256 前16位指纹，
#                     用于重复导入预检；正常对话产生的消息此列为 NULL
# -----------------------------------------------------------------------------
_SQL_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         INTEGER NOT NULL,
    role               TEXT    NOT NULL,
    content            TEXT    NOT NULL,
    created_at         TEXT    NOT NULL,
    import_fingerprint TEXT,

    FOREIGN KEY (session_id) REFERENCES sessions(id)
)
"""

# -----------------------------------------------------------------------------
# 表3：memory_l1
# 单次 session 摘要（L1 记忆层）。
#
# 情感元数据：
#   valence  — 情绪效价，五档：-1.0 / -0.5 / 0.0 / 0.5 / 1.0
#              -1.0=非常消极，0.0=中性，1.0=非常积极
#   salience — 情感显著性，五档：0.0 / 0.25 / 0.5 / 0.75 / 1.0
#              影响 Ebbinghaus 衰减速率：salience 越高衰减越慢
#
# 访问时间：
#   last_accessed_at — 最近一次被检索命中的时间，用于衰减保底加成
#                      （由 AccessBoostWorker 后台线程异步写入）
#
# absorbed：是否已被 L2 合并吸收，0=未吸收，1=已吸收
#
# time_period 合法值（六选一）：
#   清晨 / 上午 / 下午 / 傍晚 / 夜间 / 深夜
# -----------------------------------------------------------------------------
_SQL_CREATE_MEMORY_L1 = """
CREATE TABLE IF NOT EXISTS memory_l1 (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL,
    summary          TEXT    NOT NULL,
    keywords         TEXT,
    time_period      TEXT,
    atmosphere       TEXT,
    valence          REAL    NOT NULL DEFAULT 0.0,
    salience         REAL    NOT NULL DEFAULT 0.5,
    absorbed         INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT    NOT NULL,
    last_accessed_at TEXT,

    FOREIGN KEY (session_id) REFERENCES sessions(id)
)
"""

# -----------------------------------------------------------------------------
# 表4：memory_l2
# 时间段聚合摘要（L2 记忆层）。
# 由多条 L1 合并生成，触发条件（满足其一）：
#   · 未吸收 L1 累计达到 N 条（见 settings.l2_trigger_count）
#   · 最早一条未吸收 L1 距今超过 M 天（见 settings.l2_trigger_days）
#
# last_accessed_at：同 memory_l1，用于衰减保底加成
# -----------------------------------------------------------------------------
_SQL_CREATE_MEMORY_L2 = """
CREATE TABLE IF NOT EXISTS memory_l2 (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    summary          TEXT    NOT NULL,
    keywords         TEXT,
    period_start     TEXT    NOT NULL,
    period_end       TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    last_accessed_at TEXT
)
"""

# -----------------------------------------------------------------------------
# 表5：l2_sources
# L2 溯源关联表（多对多）。
# 记录每条 L2 是由哪些 L1 合并而来，支持追溯和回滚。
# -----------------------------------------------------------------------------
_SQL_CREATE_L2_SOURCES = """
CREATE TABLE IF NOT EXISTS l2_sources (
    l2_id   INTEGER NOT NULL,
    l1_id   INTEGER NOT NULL,

    PRIMARY KEY (l2_id, l1_id),
    FOREIGN KEY (l2_id) REFERENCES memory_l2(id),
    FOREIGN KEY (l1_id) REFERENCES memory_l1(id)
)
"""

# -----------------------------------------------------------------------------
# 表6：user_profile
# 长期用户画像（L3 记忆层）。
#
# 设计：追加写入，不覆盖历史版本。
#   · is_current=1：当前生效版本，注入 system prompt 时只读这些行
#   · is_current=0：历史版本，保留用于追踪状态变化和回滚
#
# status 三态说明：
#   · 'approved'：当前已生效（profile_manager.py 写入时直接设为此状态）
#   · 'pending'：待用户确认（预留，当前无代码路径产生此状态，
#               为后续"需要用户二次确认才写入画像"功能预留）
#   · 'rejected'：用户拒绝写入（预留，当前无代码路径产生此状态）
#
# field 合法值（六选一，见 constants.py PROFILE_FIELDS）：
#   basic_info / personal_status / interests / social / history / recent_context
# -----------------------------------------------------------------------------
_SQL_CREATE_USER_PROFILE = """
CREATE TABLE IF NOT EXISTS user_profile (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    field        TEXT    NOT NULL,
    content      TEXT    NOT NULL,
    source_l1_id INTEGER,
    status       TEXT    NOT NULL DEFAULT 'approved',
    is_current   INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT    NOT NULL,

    FOREIGN KEY (source_l1_id) REFERENCES memory_l1(id)
)
"""

# -----------------------------------------------------------------------------
# 表7：keyword_pool
# 关键词词典，防止摘要关键词随时间发散为大量同义词。
#
# 机制：L1 生成时将已有词条作为候选列表喂给模型，引导复用。
#
# canonical_id：指向本表自身的 rowid（隐式主键）。
#   · NULL → 本词是规范词（canonical word）
#   · 非NULL → 本词是某规范词的别名，值为规范词的 rowid
#   注：此处故意引用 rowid 而非显式 id 列，与 database.py 中
#       get_all_canonical_keywords() 的查询逻辑保持一致。
#
# alias_status 三态：
#   · 'confirmed' → 别名关系已确认（或本词就是规范词）
#   · 'pending'   → 待用户确认是否与规范词合并（相似度 0.85~0.95）
#   · 'canonical' → 显式标记为规范词（预留，当前使用 canonical_id=NULL 区分）
# -----------------------------------------------------------------------------
_SQL_CREATE_KEYWORD_POOL = """
CREATE TABLE IF NOT EXISTS keyword_pool (
    keyword      TEXT    PRIMARY KEY,
    use_count    INTEGER NOT NULL DEFAULT 1,
    last_used_at TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    canonical_id INTEGER,
    alias_status TEXT    NOT NULL DEFAULT 'confirmed'
)
"""

# -----------------------------------------------------------------------------
# 表8：graph_nodes
# 知识图谱实体节点。
#
# entity_type 合法值（五选一）：
#   person / project / module / concept / time
#
# use_count：被图谱边引用的次数，类比 keyword_pool.use_count，
#            每次有新的 graph_edge 指向本节点时 +1
# -----------------------------------------------------------------------------
_SQL_CREATE_GRAPH_NODES = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_name  TEXT    NOT NULL UNIQUE,
    entity_type  TEXT    NOT NULL,
    source_l1_id INTEGER,
    created_at   TEXT    NOT NULL,
    use_count    INTEGER NOT NULL DEFAULT 1,

    FOREIGN KEY (source_l1_id) REFERENCES memory_l1(id)
)
"""

# -----------------------------------------------------------------------------
# 表9：graph_edges
# 知识图谱关系边，存储三元组（主语 → 关系 → 宾语）。
#
# relation_type 合法值（七类）：
#   TASK_STATUS   任务状态（开始/进行中/完成/放弃）
#   OBSTACLE      遇到障碍（报错/卡住/待解决）
#   USES_DEPENDS  使用或依赖（工具/库/方案）
#   BELONGS_TO    归属（某模块属于某项目）
#   EMOTION_STATE 情感状态（情绪/疲惫/开心）
#   SOCIAL_EVENT  社交事件（与某人的互动）
#   TIME_ANCHOR   时间归属（某事发生于某时间段）
#
# source_l1_id：三元组的来源 L1，通过此字段可追溯至原始对话
# -----------------------------------------------------------------------------
_SQL_CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS graph_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id  INTEGER NOT NULL,
    target_node_id  INTEGER NOT NULL,
    relation_type   TEXT    NOT NULL,
    relation_detail TEXT,
    source_l1_id    INTEGER NOT NULL,
    created_at      TEXT    NOT NULL,

    FOREIGN KEY (source_node_id) REFERENCES graph_nodes(id),
    FOREIGN KEY (target_node_id) REFERENCES graph_nodes(id),
    FOREIGN KEY (source_l1_id)   REFERENCES memory_l1(id)
)
"""

# -----------------------------------------------------------------------------
# 表10：conflict_queue
# 冲突检测待确认队列。
# 发现冲突时写入此表，在下一次对话中以关心口吻询问用户，
# 不直接修改任何记忆层。
#
# conflict_type 两类：
#   'profile'       — 用户画像字段的前后矛盾（最常见）
#   'alias_confirm' — 知识图谱实体别名待确认
#
# status 三态流转：pending → resolved（用户接受新内容）
#                          → ignored（用户保持现状）
# -----------------------------------------------------------------------------
_SQL_CREATE_CONFLICT_QUEUE = """
CREATE TABLE IF NOT EXISTS conflict_queue (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_l1_id  INTEGER NOT NULL,
    field         TEXT    NOT NULL,
    old_content   TEXT    NOT NULL,
    new_content   TEXT    NOT NULL,
    conflict_desc TEXT    NOT NULL,
    conflict_type TEXT    NOT NULL DEFAULT 'profile',
    status        TEXT    NOT NULL DEFAULT 'pending',
    created_at    TEXT    NOT NULL,
    resolved_at   TEXT,

    FOREIGN KEY (source_l1_id) REFERENCES memory_l1(id)
)
"""

# -----------------------------------------------------------------------------
# 表11：pending_push
# 主动推送消息暂存表。
# 用于缓冲用户离线时触发的主动消息，上线后按 created_at 顺序推送。
#
# status 两态：
#   'pending' → 等待推送（用户离线或尚未发送）
#   'sent'    → 已成功推送至客户端
# -----------------------------------------------------------------------------
_SQL_CREATE_PENDING_PUSH = """
CREATE TABLE IF NOT EXISTS pending_push (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'pending',
    sent_at    TEXT
)
"""

# -----------------------------------------------------------------------------
# 表12：settings
# 全局运行配置，key-value 结构。
# value 统一存为 TEXT，调用方自行转换类型（见 database.py get_setting()）。
# 所有合法的 key 及默认值见下方 _DEFAULT_SETTINGS。
# -----------------------------------------------------------------------------
_SQL_CREATE_SETTINGS = """
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


# =============================================================================
# 默认配置项
# =============================================================================
# 所有配置项的默认值集中在此处管理。
# 写入时使用 INSERT OR IGNORE，不会覆盖用户已修改的值。
# 新增配置项时在此列表末尾追加即可。
# =============================================================================

_DEFAULT_SETTINGS = [
    # key                  value   说明
    ("profile_mode",       "manual",   # L3 画像更新模式：manual=半自动（需询问用户）
                                       # auto=全自动写入（预留，当前未使用）
    ),
    ("l1_idle_minutes",    "10",       # 对话空闲超过此分钟数后触发 L1 摘要生成
    ),
    ("l2_trigger_count",   "5",        # 未吸收 L1 累计达到此条数时触发 L2 合并
    ),
    ("l2_trigger_days",    "7",        # 最早未吸收 L1 距今超过此天数时触发 L2 合并
    ),
    ("default_model",      "local",    # 默认路由：local=本地模型，cloud=云端模型
    ),
    ("debounce_seconds",   "3",        # WebSocket 消息防抖等待时长（秒）
    ),
    ("push_enabled",       "1",        # 主动推送总开关：1=开启，0=关闭
    ),
    ("push_window_start",  "8",        # 推送时间窗口开始（24小时制整点，含）
    ),
    ("push_window_end",    "24",       # 推送时间窗口结束（24小时制，24表示午夜，含）
    ),
    ("push_daily_limit",   "4",        # 每日主动推送条数上限
    ),
]


# =============================================================================
# 索引定义
# =============================================================================
# 格式：(索引名, 建索引 SQL, 说明)
# 说明字段仅用于控制台输出，不写入数据库。
# =============================================================================

_INDEXES = [
    # ── messages 表 ──────────────────────────────────────────────────────────
    (
        "idx_messages_session_id",
        "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)",
        "get_messages() 每次对话调用，最高频查询",
    ),
    (
        "idx_messages_created_at",
        "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
        "get_last_message_time() 排序，空闲检测每分钟触发",
    ),
    # ── memory_l1 表 ─────────────────────────────────────────────────────────
    (
        "idx_memory_l1_session_id",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_session_id ON memory_l1(session_id)",
        "get_l1_by_session() 查询",
    ),
    (
        "idx_memory_l1_absorbed",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_absorbed ON memory_l1(absorbed)",
        "get_unabsorbed_l1() 每次 L1 写入后触发",
    ),
    (
        "idx_memory_l1_created_at",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_created_at ON memory_l1(created_at)",
        "L2 触发检查的排序字段，get_latest_l1()",
    ),
    # ── user_profile 表 ───────────────────────────────────────────────────────
    (
        "idx_user_profile_field_current",
        "CREATE INDEX IF NOT EXISTS idx_user_profile_field_current "
        "ON user_profile(field, is_current)",
        "get_current_profile() 每次对话前调用，复合索引覆盖两个过滤条件",
    ),
    # ── keyword_pool 表 ───────────────────────────────────────────────────────
    (
        "idx_keyword_pool_alias_status",
        "CREATE INDEX IF NOT EXISTS idx_keyword_pool_alias_status "
        "ON keyword_pool(alias_status)",
        "get_all_canonical_keywords() 按 alias_status 过滤",
    ),
    # ── graph_edges 表 ────────────────────────────────────────────────────────
    (
        "idx_graph_edges_source",
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_source "
        "ON graph_edges(source_node_id)",
        "图谱遍历：从节点出发找所有出边",
    ),
    (
        "idx_graph_edges_target",
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_target "
        "ON graph_edges(target_node_id)",
        "图谱遍历：找所有指向某节点的入边",
    ),
    (
        "idx_graph_edges_l1",
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_l1 "
        "ON graph_edges(source_l1_id)",
        "按 L1 溯源：查找某条摘要产生了哪些图谱边",
    ),
    (
        "idx_graph_edges_relation_type",
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_relation_type "
        "ON graph_edges(relation_type)",
        "按关系类型检索图谱边（如只查 TASK_STATUS 类）",
    ),
    # ── conflict_queue 表 ─────────────────────────────────────────────────────
    (
        "idx_conflict_queue_status",
        "CREATE INDEX IF NOT EXISTS idx_conflict_queue_status "
        "ON conflict_queue(status)",
        "get_pending_conflicts() 每次对话前检查",
    ),
    # ── pending_push 表 ───────────────────────────────────────────────────────
    (
        "idx_pending_push_status",
        "CREATE INDEX IF NOT EXISTS idx_pending_push_status "
        "ON pending_push(status)",
        "get_pending_pushes() WebSocket 连接建立时查询",
    ),
]


# =============================================================================
# 所有表的建表语句，按依赖顺序排列（被引用的表在前）
# =============================================================================

_ALL_TABLES = [
    ("sessions",       _SQL_CREATE_SESSIONS),
    ("messages",       _SQL_CREATE_MESSAGES),
    ("memory_l1",      _SQL_CREATE_MEMORY_L1),
    ("memory_l2",      _SQL_CREATE_MEMORY_L2),
    ("l2_sources",     _SQL_CREATE_L2_SOURCES),
    ("user_profile",   _SQL_CREATE_USER_PROFILE),
    ("keyword_pool",   _SQL_CREATE_KEYWORD_POOL),
    ("graph_nodes",    _SQL_CREATE_GRAPH_NODES),
    ("graph_edges",    _SQL_CREATE_GRAPH_EDGES),
    ("conflict_queue", _SQL_CREATE_CONFLICT_QUEUE),
    ("pending_push",   _SQL_CREATE_PENDING_PUSH),
    ("settings",       _SQL_CREATE_SETTINGS),
]


# =============================================================================
# 迁移定义
# =============================================================================
# 用于已有数据库的升级，检测并补齐历史遗漏的列。
# 格式：(表名, 列名, ALTER TABLE SQL, 备注)
#
# 扩展方法：
#   · 新增列时，在此列表末尾追加一条记录
#   · setup_db.py 会在运行时自动检查该列是否已存在，存在则跳过
# =============================================================================

_COLUMN_MIGRATIONS = [
    # messages 表：补充历史遗漏的导入指纹列
    (
        "messages",
        "import_fingerprint",
        "ALTER TABLE messages ADD COLUMN import_fingerprint TEXT",
        "历史导入消息的去重指纹，正常对话消息此列为 NULL",
    ),
    # memory_l1 表：补充情感元数据列（原通过 emotion_migration.py 加入）
    (
        "memory_l1",
        "valence",
        "ALTER TABLE memory_l1 ADD COLUMN valence REAL NOT NULL DEFAULT 0.0",
        "情绪效价，五档浮点，0.0=中性",
    ),
    (
        "memory_l1",
        "salience",
        "ALTER TABLE memory_l1 ADD COLUMN salience REAL NOT NULL DEFAULT 0.5",
        "情感显著性，五档浮点，影响衰减速率",
    ),
    (
        "memory_l1",
        "last_accessed_at",
        "ALTER TABLE memory_l1 ADD COLUMN last_accessed_at TEXT",
        "最近检索命中时间，由 AccessBoostWorker 异步写入",
    ),
    # memory_l2 表：补充访问时间列
    (
        "memory_l2",
        "last_accessed_at",
        "ALTER TABLE memory_l2 ADD COLUMN last_accessed_at TEXT",
        "最近检索命中时间，由 AccessBoostWorker 异步写入",
    ),
    # keyword_pool 表：补充别名归一化列（原通过 migrate_add_graph_tables.py 加入）
    (
        "keyword_pool",
        "canonical_id",
        "ALTER TABLE keyword_pool ADD COLUMN canonical_id INTEGER",
        "规范词的 rowid，NULL 表示本词就是规范词",
    ),
    (
        "keyword_pool",
        "alias_status",
        "ALTER TABLE keyword_pool ADD COLUMN alias_status TEXT NOT NULL DEFAULT 'confirmed'",
        "别名状态：confirmed / pending / canonical",
    ),
    # conflict_queue 表：补充冲突类型列（原通过 migrate_add_graph_tables.py 加入）
    (
        "conflict_queue",
        "conflict_type",
        "ALTER TABLE conflict_queue ADD COLUMN conflict_type TEXT NOT NULL DEFAULT 'profile'",
        "冲突来源类型：profile=画像矛盾，alias_confirm=实体别名待确认",
    ),
    # messages 表：补充消息来源列
    (
        "messages",
        "source",
        "ALTER TABLE messages ADD COLUMN source TEXT NOT NULL DEFAULT 'local'",
        "消息来源：local=本地模型，online=云端API，import=历史导入",
    ),
]


# =============================================================================
# 核心函数：全新建库
# =============================================================================

def _create_fresh_db(conn: sqlite3.Connection) -> None:
    """
    在全新数据库上建立完整的表结构、索引和默认配置。
    此函数假设数据库文件刚刚创建，不做任何存在性检查。
    """
    cursor = conn.cursor()

    # ── 建表 ──────────────────────────────────────────────────────────────────
    print("\n[1/3] 建立表结构")
    for table_name, sql in _ALL_TABLES:
        cursor.execute(sql)
        print(f"  ✓  {table_name}")

    # ── 建索引 ────────────────────────────────────────────────────────────────
    print("\n[2/3] 建立索引")
    for index_name, sql, desc in _INDEXES:
        cursor.execute(sql)
        print(f"  ✓  {index_name}")
        print(f"       {desc}")

    # ── 写入默认配置 ──────────────────────────────────────────────────────────
    print("\n[3/3] 写入默认配置")
    now = _now()
    for key, value in _DEFAULT_SETTINGS:
        cursor.execute(
            "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        print(f"  ✓  {key} = {value}")

    conn.commit()


# =============================================================================
# 核心函数：已有数据库迁移
# =============================================================================

def _migrate_existing_db(conn: sqlite3.Connection) -> None:
    """
    对已有数据库执行幂等迁移：
      1. 检测并补齐缺失的列（ALTER TABLE）
      2. 检测并新建缺失的表（已有表不动）
      3. 检测并新建缺失的索引（已有索引不动）
      4. 补充缺失的默认配置项（INSERT OR IGNORE）

    所有操作均先检查再执行，重复运行安全。
    """
    cursor = conn.cursor()

    # ── 迁移1：补齐缺失的列 ───────────────────────────────────────────────────
    print("\n[1/4] 检查并补齐缺失列")

    # 缓存已检查过的表的列名，避免重复 PRAGMA 查询
    _table_columns: dict[str, set[str]] = {}

    def _get_columns(table: str) -> set[str]:
        if table not in _table_columns:
            cursor.execute(f"PRAGMA table_info({table})")
            _table_columns[table] = {row["name"] for row in cursor.fetchall()}
        return _table_columns[table]

    for table, col, sql, desc in _COLUMN_MIGRATIONS:
        # 先检查表是否存在（历史数据库可能没有 graph_nodes 等新表）
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not cursor.fetchone():
            # 表本身不存在，跳过列迁移（后面会在迁移2里建表）
            print(f"  –  {table}.{col}（表不存在，跳过，将在建表步骤创建）")
            continue

        if col in _get_columns(table):
            print(f"  ✓  {table}.{col}（已存在，跳过）")
        else:
            cursor.execute(sql)
            # 清除该表的列缓存，下次重新查询
            _table_columns.pop(table, None)
            print(f"  +  {table}.{col}（新增）— {desc}")

    # ── 迁移2：新建缺失的表 ───────────────────────────────────────────────────
    print("\n[2/4] 检查并补齐缺失表")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row["name"] for row in cursor.fetchall()}

    for table_name, sql in _ALL_TABLES:
        if table_name in existing_tables:
            print(f"  ✓  {table_name}（已存在，跳过）")
        else:
            cursor.execute(sql)
            print(f"  +  {table_name}（新建）")

    # ── 迁移3：新建缺失的索引 ─────────────────────────────────────────────────
    print("\n[3/4] 检查并补齐缺失索引")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    existing_indexes = {row["name"] for row in cursor.fetchall()}

    for index_name, sql, desc in _INDEXES:
        if index_name in existing_indexes:
            print(f"  ✓  {index_name}（已存在，跳过）")
        else:
            cursor.execute(sql)
            print(f"  +  {index_name}（新建）— {desc}")

    # ── 迁移4：补充缺失的默认配置项 ──────────────────────────────────────────
    print("\n[4/4] 检查并补充缺失配置项")
    now = _now()
    for key, value in _DEFAULT_SETTINGS:
        cursor.execute(
            "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        if cursor.rowcount > 0:
            print(f"  +  {key} = {value}（新增）")
        else:
            print(f"  ✓  {key}（已存在，保留现有值）")

    conn.commit()


# =============================================================================
# 工具函数：验证数据库完整性
# =============================================================================

def _verify_db(conn: sqlite3.Connection) -> bool:
    """
    建库或迁移完成后，验证数据库结构是否符合预期。
    检查内容：
      · 所有表是否存在
      · 所有索引是否存在
      · settings 表是否包含所有默认配置项
    返回 True 表示验证通过，False 表示存在问题。
    """
    cursor = conn.cursor()
    all_ok = True

    # 检查表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row["name"] for row in cursor.fetchall()}
    expected_tables = {name for name, _ in _ALL_TABLES}
    missing_tables  = expected_tables - existing_tables
    if missing_tables:
        print(f"\n  [ERROR] 缺少表：{missing_tables}")
        all_ok = False

    # 检查索引
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    existing_indexes = {row["name"] for row in cursor.fetchall()}
    expected_indexes = {name for name, _, _ in _INDEXES}
    missing_indexes  = expected_indexes - existing_indexes
    if missing_indexes:
        print(f"\n  [ERROR] 缺少索引：{missing_indexes}")
        all_ok = False

    # 检查配置项
    cursor.execute("SELECT key FROM settings")
    existing_keys = {row["key"] for row in cursor.fetchall()}
    expected_keys = {key for key, _ in _DEFAULT_SETTINGS}
    missing_keys  = expected_keys - existing_keys
    if missing_keys:
        print(f"\n  [ERROR] 缺少配置项：{missing_keys}")
        all_ok = False

    return all_ok


# =============================================================================
# 工具函数：强制重建（危险操作，需 --force-rebuild 参数）
# =============================================================================

def _force_rebuild() -> None:
    """
    删除已有数据库文件后全新建库。
    调用前会要求用户二次确认，防止误操作。
    """
    if not DB_PATH.exists():
        print("数据库文件不存在，直接建库。")
        return

    print(f"\n[警告] 即将删除数据库文件：{DB_PATH}")
    print("       此操作不可逆，所有历史数据将永久丢失！")
    confirm = input("\n请输入 'YES' 确认删除，或按回车取消：").strip()
    if confirm != "YES":
        print("已取消，数据库未改动。")
        sys.exit(0)

    # 删除前先备份到同目录下，文件名加时间戳
    backup_path = DB_PATH.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    )
    shutil.copy2(DB_PATH, backup_path)
    print(f"已备份至：{backup_path}")

    DB_PATH.unlink()
    print("旧数据库已删除。")


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    """
    主入口函数。
    自动判断数据库状态：
      · 文件不存在 → 全新建库
      · 文件已存在 → 迁移已有数据库
      · --force-rebuild 参数 → 删除后全新建库
    """
    # ── 命令行参数解析 ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="珊瑚菌数据库初始化与迁移脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python scripts/setup_db.py                 # 新建或升级数据库
  python scripts/setup_db.py --force-rebuild # 删除后重建（会提示确认）
        """,
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="删除已有数据库后重新建库（危险操作，会要求二次确认）",
    )
    args = parser.parse_args()

    # ── 确认数据库路径 ────────────────────────────────────────────────────────
    print(f"数据库路径：{DB_PATH}")

    # ── 强制重建 ──────────────────────────────────────────────────────────────
    if args.force_rebuild:
        _force_rebuild()

    # ── 确保数据目录存在 ──────────────────────────────────────────────────────
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── 判断新建还是迁移 ──────────────────────────────────────────────────────
    is_new_db = not DB_PATH.exists()

    conn = _get_conn()
    try:
        if is_new_db:
            print("\n=== 全新数据库，开始建库 ===")
            _create_fresh_db(conn)
        else:
            print("\n=== 已有数据库，开始迁移 ===")
            _migrate_existing_db(conn)

        # ── 验证 ──────────────────────────────────────────────────────────────
        print("\n=== 验证数据库完整性 ===")
        ok = _verify_db(conn)
        if ok:
            print("\n✅  数据库就绪，可以运行 python app/main.py 启动服务。")
        else:
            print("\n❌  验证发现问题，请检查上方错误信息后重试。")
            sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()