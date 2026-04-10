"""
src/ramaria/adapters/mcp/tools/read_tools.py — MCP 只读工具集

包含五个无副作用的查询工具：
    search_memory       语义检索 L1/L2 向量索引
    get_profile         读取 L3 用户画像
    get_recent_context  获取最近 N 条 L1 时间序摘要
    get_index_stats     查看三层向量索引的条目统计
    get_pending_sessions 查询有多少 session 尚未生成 L1

每个工具函数的职责：
    1. 参数解析与边界校验
    2. 调用 database.py / vector_store.py 中的现有函数
    3. 将结果格式化为可读字符串，返回给 MCP 客户端

"""

import sqlite3
from ramaria.config import DB_PATH, L1_RETRIEVE_TOP_K, L2_RETRIEVE_TOP_K
from constants import PROFILE_FIELDS, PROFILE_FIELD_LIST


# =============================================================================
# search_memory
# =============================================================================

def search_memory(arguments: dict) -> str:
    """
    在 L1/L2 向量索引中执行语义检索，返回最相关的历史摘要。

    参数（来自 MCP 调用方）：
        query  — 检索文本，必填
        top_k  — 每层最多返回条数，可选，默认使用 config 里的设置

    返回：
        格式化的文本，包含 L2 和 L1 的命中结果及相关度分数。
        无命中时返回说明文字。
    """
    from ramaria.storage.vector_store import retrieve_combined

    query = str(arguments.get("query", "")).strip()
    if not query:
        return "错误：query 参数不能为空。"

    top_k_l1 = min(int(arguments.get("top_k", L1_RETRIEVE_TOP_K)), 10)
    top_k_l2 = min(int(arguments.get("top_k", L2_RETRIEVE_TOP_K)), 5)

    try:
        results = retrieve_combined(query)
    except Exception as e:
        return f"检索失败：{e}"

    l2_hits = results.get("l2", [])[:top_k_l2]
    l1_hits = results.get("l1", [])[:top_k_l1]

    if not l2_hits and not l1_hits:
        return (
            f"未找到与「{query}」相关的历史记忆。\n"
            "可能原因：向量索引尚未建立，或相关内容的语义距离超过阈值。"
        )

    lines = [f"# 语义检索结果\n查询：「{query}」\n"]

    if l2_hits:
        lines.append("## L2 时间段摘要（粒度粗，话题定位）")
        for i, hit in enumerate(l2_hits, 1):
            doc  = hit.get("document", "").strip()
            dist = hit.get("distance", 0)
            meta = hit.get("metadata", {})
            period = ""
            if meta.get("period_start") and meta.get("period_end"):
                period = f"（{meta['period_start'][:10]} ~ {meta['period_end'][:10]}）"
            lines.append(f"  [{i}] {period} 相关度={dist:.3f}")
            lines.append(f"      {doc}")
        lines.append("")

    if l1_hits:
        lines.append("## L1 单次对话摘要（粒度细，细节参考）")
        for i, hit in enumerate(l1_hits, 1):
            doc  = hit.get("document", "").strip()
            dist = hit.get("distance", 0)
            meta = hit.get("metadata", {})
            kw      = meta.get("keywords", "")
            created = meta.get("created_at", "")[:10] if meta.get("created_at") else ""
            kw_str   = f" 关键词：{kw}" if kw else ""
            date_str = f" [{created}]" if created else ""
            lines.append(f"  [{i}]{date_str} 相关度={dist:.3f}{kw_str}")
            lines.append(f"      {doc}")

    return "\n".join(lines)


# =============================================================================
# get_profile
# =============================================================================

def get_profile(arguments: dict) -> str:
    """
    读取当前生效的 L3 用户长期画像。

    参数（来自 MCP 调用方）：
        field — 可选，只返回指定板块
                不传则返回全部六个板块

    合法的 field 值：
        basic_info / personal_status / interests / social / history / recent_context
    """
    from ramaria.storage.database import get_current_profile

    field_labels = PROFILE_FIELDS

    try:
        profile = get_current_profile()
    except Exception as e:
        return f"读取画像失败：{e}"

    if not profile:
        return (
            "当前用户画像为空（冷启动阶段）。\n"
            "珊瑚菌会在每次对话结束后自动从摘要中提取画像信息，"
            "随着对话积累画像会逐渐丰富。"
        )

    field_filter = str(arguments.get("field", "")).strip()
    if field_filter:
        if field_filter not in field_labels:
            valid = "、".join(field_labels.keys())
            return f"错误：field 值 {field_filter!r} 不合法。\n合法值：{valid}"
        profile = {field_filter: profile.get(field_filter, "")}

    lines = ["# L3 用户长期画像\n"]
    for field_key, label in field_labels.items():
        content = profile.get(field_key, "").strip()
        if content:
            lines.append(f"## {label}（{field_key}）")
            lines.append(content)
            lines.append("")

    if len(lines) == 1:
        return f"板块「{field_filter}」暂无内容。"

    return "\n".join(lines).strip()


# =============================================================================
# 内部辅助：直接查表（database.py 无对应公开函数）
# =============================================================================

def _get_recent_l1_rows(limit: int) -> list:
    """
    取出最近 N 条 L1 摘要，按 created_at 降序排列。
    database.py 没有"最近 N 条"的公开函数，故在此处独立实现。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, session_id, summary, keywords, time_period, atmosphere, created_at
        FROM memory_l1
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def _get_db_counts() -> dict:
    """
    查询 SQLite 中 memory_l1、memory_l2 和 sessions 的记录数。
    供 get_index_stats 使用。
    """
    result = {"l1": "查询失败", "l2": "查询失败", "sessions": "查询失败"}
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM memory_l1")
        result["l1"] = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) AS cnt FROM memory_l2")
        result["l2"] = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(DISTINCT session_id) AS cnt FROM messages")
        result["sessions"] = cursor.fetchone()["cnt"]
        conn.close()
    except Exception:
        pass
    return result


# =============================================================================
# get_recent_context
# =============================================================================

def get_recent_context(arguments: dict) -> str:
    """
    获取最近几条 L1 摘要（时间序，不做语义检索）。

    与 search_memory 的区别：
        search_memory 是语义检索（找最相关的）
        get_recent_context 是时间序（找最近的）

    参数（来自 MCP 调用方）：
        limit — 返回条数，可选，默认5，最多20
    """
    limit = min(int(arguments.get("limit", 5)), 20)

    try:
        rows = _get_recent_l1_rows(limit)
    except Exception as e:
        return f"查询失败：{e}"

    if not rows:
        return "暂无 L1 摘要记录（尚未有 session 完成总结）。"

    lines = [f"# 最近 {len(rows)} 条对话摘要（L1，时间降序）\n"]
    for row in rows:
        date_str = row["created_at"][:16] if row["created_at"] else "未知时间"
        tp   = row["time_period"] or ""
        atm  = row["atmosphere"]  or ""
        kw   = row["keywords"]    or ""
        meta = "、".join(filter(None, [tp, atm]))
        kw_str = f"\n  关键词：{kw}" if kw else ""
        lines.append(f"## [{date_str}] {meta}")
        lines.append(f"  {row['summary']}{kw_str}")
        lines.append("")

    return "\n".join(lines).strip()


# =============================================================================
# get_index_stats
# =============================================================================

def get_index_stats(arguments: dict) -> str:
    """
    查看三层向量索引（L0/L1/L2）的当前条目数。
    无参数。用途：运维检查，确认索引是否正常写入。
    """
    from ramaria.storage.vector_store import get_index_stats as _get_stats

    try:
        stats = _get_stats()
    except Exception as e:
        return f"查询索引统计失败：{e}"

    db_counts  = _get_db_counts()
    db_l1      = db_counts["l1"]
    db_l2      = db_counts["l2"]
    db_sessions = db_counts["sessions"]

    lines = [
        "# 向量索引统计\n",
        "## Chroma 向量索引（实际可检索条目）",
        f"  L0（原始消息切片）：{stats.get('l0', 0)} 条",
        f"  L1（单次对话摘要）：{stats.get('l1', 0)} 条",
        f"  L2（时间段聚合摘要）：{stats.get('l2', 0)} 条",
        "",
        "## SQLite 原始数据（参考）",
        f"  memory_l1 表：{db_l1} 条",
        f"  memory_l2 表：{db_l2} 条",
        f"  涉及 session 数：{db_sessions} 个",
        "",
        "提示：向量索引条目数 < SQLite 条目数时，说明有数据尚未建立索引。",
        "可运行 vector_store.rebuild_all_indexes() 重建全部索引。",
    ]
    return "\n".join(lines)


# =============================================================================
# get_pending_sessions
# =============================================================================

def get_pending_sessions(arguments: dict) -> str:
    """
    查询有多少历史 session 尚未生成 L1 摘要。
    无参数。用途：导入聊天记录后确认待处理量。
    """
    from ramaria.storage.database import get_sessions_without_l1

    try:
        pending = get_sessions_without_l1()
    except Exception as e:
        return f"查询失败：{e}"

    count = len(pending)

    if count == 0:
        return "所有 session 均已生成 L1 摘要，无待处理项。"

    earliest = pending[0]["started_at"][:16]  if pending else ""
    latest   = pending[-1]["started_at"][:16] if pending else ""

    lines = [
        f"# 待生成 L1 摘要的 session\n",
        f"共 {count} 个 session 尚未生成摘要。",
        f"时间跨度：{earliest} ~ {latest}",
        "",
        "处理方式：",
        "  · 使用 trigger_l1 工具逐个触发（传入 session_id）",
        "  · 或在 Web 界面（/import 页面）批量生成",
        "  · 或运行命令行：python -m ramaria.importer.qq.cli --generate-l1",
        "",
        f"前10个待处理 session ID：",
    ]

    for s in pending[:10]:
        lines.append(f"  session {s['id']}（{s['started_at'][:16]}）")

    if count > 10:
        lines.append(f"  ... 还有 {count - 10} 个")

    return "\n".join(lines)