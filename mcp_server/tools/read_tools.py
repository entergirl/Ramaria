"""
mcp_server/tools/read_tools.py — 只读工具集
=====================================================================

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

返回格式统一为纯文本（TextContent），方便 Claude Desktop 直接阅读。
需要结构化数据时（如 search_memory），以 JSON 格式内嵌在文本中。

与其他模块的关系：
    - 调用 vector_store.retrieve_combined()
    - 调用 database.get_current_profile()
    - 调用 database.get_latest_l1() / get_recent_l2()
    - 调用 vector_store.get_index_stats()
    - 调用 database.get_sessions_without_l1()
"""

import json
import sys
from pathlib import Path

# ── 将项目根目录加入 sys.path，使 MCP Server 能直接 import 项目模块 ──
# MCP Server 作为独立进程运行，需要显式设置路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from constants import PROFILE_FIELDS, VALID_FIELD_KEYS, PROFILE_FIELD_LIST

# =============================================================================
# search_memory
# =============================================================================

def search_memory(arguments: dict) -> str:
    """
    在 L1/L2 向量索引中执行语义检索，返回最相关的历史摘要。

    参数（来自 MCP 调用方）：
        query  — 检索文本，必填，如 "最近有没有写代码熬夜"
        top_k  — 每层最多返回条数，可选，默认使用 config 里的设置

    工作原理：
        直接调用 vector_store.retrieve_combined()，同时查 L1 和 L2，
        结果已按语义相关度排序（距离越小越相关）。
        距离超过 SIMILARITY_THRESHOLD 的结果会被自动过滤。

    返回：
        格式化的文本，包含 L2 和 L1 的命中结果及其相关度分数。
        无命中时返回说明文字。
    """
    from vector_store import retrieve_combined
    from config import L1_RETRIEVE_TOP_K, L2_RETRIEVE_TOP_K

    # 参数解析
    query = str(arguments.get("query", "")).strip()
    if not query:
        return "错误：query 参数不能为空。"

    # top_k 允许调用方覆盖，但不能超过合理上限（避免结果过多）
    top_k_l1 = min(int(arguments.get("top_k", L1_RETRIEVE_TOP_K)), 10)
    top_k_l2 = min(int(arguments.get("top_k", L2_RETRIEVE_TOP_K)), 5)

    try:
        # retrieve_combined 返回 {"l2": [...], "l1": [...]}
        # 每条结果是 {"document": str, "distance": float, "metadata": dict, ...}
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

    # L2 结果（时间段聚合摘要，粒度粗，代表较长时间的规律）
    if l2_hits:
        lines.append("## L2 时间段摘要（粒度粗，话题定位）")
        for i, hit in enumerate(l2_hits, 1):
            doc  = hit.get("document", "").strip()
            dist = hit.get("distance", 0)
            adj  = hit.get("adjusted_distance", dist)
            meta = hit.get("metadata", {})
            # period 信息来自 metadata，格式如 "2026-01-01 ~ 2026-01-31"
            period = ""
            if meta.get("period_start") and meta.get("period_end"):
                period = f"（{meta['period_start'][:10]} ~ {meta['period_end'][:10]}）"
            lines.append(f"  [{i}] {period} 相关度={dist:.3f}")
            lines.append(f"      {doc}")
        lines.append("")

    # L1 结果（单次对话摘要，粒度细，主力参考层）
    if l1_hits:
        lines.append("## L1 单次对话摘要（粒度细，细节参考）")
        for i, hit in enumerate(l1_hits, 1):
            doc  = hit.get("document", "").strip()
            dist = hit.get("distance", 0)
            meta = hit.get("metadata", {})
            kw   = meta.get("keywords", "")
            created = meta.get("created_at", "")[:10] if meta.get("created_at") else ""
            kw_str = f" 关键词：{kw}" if kw else ""
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
        field — 可选，只返回指定板块，如 "basic_info"
                不传则返回全部六个板块

    合法的 field 值（与 user_profile 表保持一致）：
        basic_info      基础信息（姓名、年龄等稳定属性）
        personal_status 近期状态（情绪、健康、压力等动态信息）
        interests       兴趣爱好
        social          社交情况
        history         重要经历
        recent_context  近期背景（项目进展等）

    返回：
        画像文本，按板块分段展示。
        画像为空时给出冷启动说明。
    """
    from database import get_current_profile

    # 板块名到中文标签的映射
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

    # 过滤指定板块
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
        # 有 profile 但过滤后为空（指定了没有内容的板块）
        return f"板块「{field_filter}」暂无内容。"

    return "\n".join(lines).strip()


# =============================================================================
# 内部辅助：封装需要直接查表的 database 操作
# （database.py 没有对应的公开函数，在此处内联实现，避免调用私有函数）
# =============================================================================

def _get_recent_l1_rows(limit: int) -> list:
    """
    取出最近 N 条 L1 摘要，按 created_at 降序排列。

    database.py 提供了 get_latest_l1()（只取一条）和 get_all_l1()（全量），
    但没有"最近 N 条"的公开函数，故在此处独立实现。
    使用与 database._get_connection 相同的连接逻辑，但直接 import DB_PATH。

    返回：
        list[sqlite3.Row]，按 created_at DESC 排序，最多 limit 条。
    """
    import sqlite3
    from config import DB_PATH

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
    供 get_index_stats 使用，与 Chroma 索引数量做对比。

    返回：
        dict — {"l1": int, "l2": int, "sessions": int}
        查询失败时对应值为字符串 "查询失败"。
    """
    import sqlite3
    from config import DB_PATH

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
        search_memory 是语义检索（找最相关的），适合"找某个话题的历史"
        get_recent_context 是时间序（找最近的），适合"了解最近发生了什么"

    参数（来自 MCP 调用方）：
        limit — 返回条数，可选，默认5，最多20

    返回：
        最近 N 条 L1 摘要，按时间降序（最新的在前）。
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

    无参数。

    用途：
        运维检查，确认索引是否正常写入。
        切换嵌入模型后执行 rebuild_all_indexes() 前后可以对比。

    返回：
        三层索引的条目数及简要说明。
    """
    from vector_store import get_index_stats as _get_stats

    try:
        stats = _get_stats()
    except Exception as e:
        return f"查询索引统计失败：{e}"

    # 同时查一下 SQLite 侧的数量，方便对比（索引和 DB 数量不一致可能说明有脏数据）
    db_counts = _get_db_counts()
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

    无参数。

    用途：
        导入 QQ 聊天记录后，确认待处理量。
        也可用于监控日常摘要生成是否正常。

    返回：
        待处理 session 数量，以及最早和最晚的 session 时间。
    """
    from database import get_sessions_without_l1

    try:
        pending = get_sessions_without_l1()
    except Exception as e:
        return f"查询失败：{e}"

    count = len(pending)

    if count == 0:
        return "所有 session 均已生成 L1 摘要，无待处理项。"

    # 取时间范围
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
        "  · 或运行命令行：python -m importer.qq_import_cli --generate-l1",
        "",
        f"前10个待处理 session ID：",
    ]

    for s in pending[:10]:
        lines.append(f"  session {s['id']}（{s['started_at'][:16]}）")

    if count > 10:
        lines.append(f"  ... 还有 {count - 10} 个")

    return "\n".join(lines)
