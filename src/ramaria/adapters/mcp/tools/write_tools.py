"""
src/ramaria/adapters/mcp/tools/write_tools.py — MCP 写入工具集

包含三个有副作用的写入工具：
    save_message    向 L0 写入一条消息
    trigger_l1      触发指定 session 的 L1 摘要生成
    update_profile  直接更新 L3 用户画像某个板块（默认禁用）

设计原则：
    1. 最小副作用：save_message 只写 L0，trigger_l1 只触发摘要
    2. 外部 session 隔离：MCP 写入不复用 FastAPI 活跃 session
    3. 权限检查：每个写入函数先调用 permissions.is_allowed()
    4. 幂等性提示：trigger_l1 检查 session 是否已有摘要

"""

import hashlib
from datetime import datetime, timezone

from ramaria.adapters.mcp.permissions import is_allowed
from constants import PROFILE_FIELDS, VALID_FIELD_KEYS


# =============================================================================
# 内部工具函数
# =============================================================================

def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _make_fingerprint(timestamp_ms: int, role: str, content: str) -> str:
    """
    计算消息唯一指纹（与 qq/parser.py 的 _make_fingerprint 逻辑一致）。
    取 SHA-256 的前16位十六进制字符（8字节）。
    """
    raw = f"{timestamp_ms}|{role}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _get_or_create_mcp_session() -> int:
    """
    创建一个 MCP 专用的外部写入 session。

    每次调用 save_message 时都创建新的独立 session，
    避免不相关消息混入同一 session 破坏摘要语义完整性。
    started_at 和 ended_at 同时写入当前时间，不参与空闲检测。
    """
    from ramaria.storage.database import new_session_with_time, close_session_with_time

    now        = _now_iso()
    session_id = new_session_with_time(now)
    close_session_with_time(session_id, now)
    return session_id


# =============================================================================
# save_message
# =============================================================================

def save_message(arguments: dict) -> str:
    """
    向 L0 原始消息层写入一条消息，不触发任何自动后台流程。

    适合：将外部工具的对话记录注入珊瑚菌的记忆库，或手动补录重要信息。
    写入后请调用 trigger_l1(session_id) 将其提炼进记忆，
    否则这条消息只存在于 L0，不会被检索到。

    参数（来自 MCP 调用方）：
        role         — 消息角色："user" 或 "assistant"，必填
        content      — 消息内容，必填
        source_hint  — 可选，来源说明，会附在消息末尾

    返回：
        写入结果，包含 session_id（用于后续 trigger_l1）。
    """
    if not is_allowed("save_message"):
        return "错误：save_message 工具当前已禁用。请在 permissions.py 中开启。"

    role = str(arguments.get("role", "")).strip().lower()
    if role not in ("user", "assistant"):
        return "错误：role 必须是 'user' 或 'assistant'。"

    content = str(arguments.get("content", "")).strip()
    if not content:
        return "错误：content 不能为空。"

    source_hint = str(arguments.get("source_hint", "")).strip()
    if source_hint:
        content = f"{content}\n\n[来源：{source_hint}]"

    try:
        from ramaria.storage.database import save_message_with_fingerprint

        session_id = _get_or_create_mcp_session()

        ts_ms       = int(datetime.now(timezone.utc).timestamp() * 1000)
        fingerprint = _make_fingerprint(ts_ms, role, content)
        now         = _now_iso()

        msg_id = save_message_with_fingerprint(
            session_id  = session_id,
            role        = role,
            content     = content,
            created_at  = now,
            fingerprint = fingerprint,
        )

    except Exception as e:
        return f"写入失败：{e}"

    return (
        f"消息已写入 L0。\n"
        f"  session_id : {session_id}\n"
        f"  message_id : {msg_id}\n"
        f"  role       : {role}\n"
        f"  时间       : {now[:16]}\n"
        f"\n"
        f"提示：此消息尚未生成摘要。\n"
        f"需要提炼进记忆时，请调用 trigger_l1(session_id={session_id})。"
    )


# =============================================================================
# trigger_l1
# =============================================================================

def trigger_l1(arguments: dict) -> str:
    """
    触发指定 session 的 L1 摘要生成。

    此工具会调用 summarizer.generate_l1_summary()，
    和实时对话结束后的自动触发路径完全一致，包括：
      · 调用本地模型生成摘要（需要 LM Studio 运行中）
      · 写入 memory_l1 表
      · 写入 L1 向量索引
      · 触发画像提取（profile_manager）
      · 触发冲突检测（conflict_checker）
      · 触发 L2 条数检查（merger）

    注意：生成耗时通常 10~30 秒，请耐心等待。

    参数（来自 MCP 调用方）：
        session_id — 必填，要提炼的 session id（整数）
    """
    if not is_allowed("trigger_l1"):
        return "错误：trigger_l1 工具当前已禁用。请在 permissions.py 中开启。"

    session_id_raw = arguments.get("session_id")
    if session_id_raw is None:
        return "错误：session_id 参数必填。可通过 get_pending_sessions 查询待处理列表。"

    try:
        session_id = int(session_id_raw)
    except (ValueError, TypeError):
        return f"错误：session_id 必须是整数，收到：{session_id_raw!r}"

    try:
        from ramaria.storage.database import (
            get_messages,
            get_session,
            get_l1_by_session,
        )

        session = get_session(session_id)
        if session is None:
            return f"错误：session {session_id} 不存在。"

        messages = get_messages(session_id)
        if not messages:
            return f"session {session_id} 没有消息，跳过摘要生成。"

        existing_l1 = get_l1_by_session(session_id)
        warning = ""
        if existing_l1:
            warning = (
                f"\n警告：session {session_id} 已有 L1 摘要"
                f"（id={existing_l1['id']}）。\n"
                f"本次调用将额外生成一条新摘要。\n"
            )

    except Exception as e:
        return f"查询 session 信息失败：{e}"

    try:
        from ramaria.memory.summarizer import generate_l1_summary
        l1_id = generate_l1_summary(session_id)
    except Exception as e:
        return f"摘要生成时发生异常：{e}"

    if l1_id is None:
        return (
            f"摘要生成失败（session_id={session_id}）。\n"
            "可能原因：本地模型未启动（需要 LM Studio 运行中），"
            "或消息内容不足以生成有效摘要。"
        )

    try:
        from ramaria.storage.database import get_l1_by_id
        l1_row = get_l1_by_id(l1_id)
    except Exception:
        l1_row = None

    result_lines = [
        f"L1 摘要生成成功。{warning}",
        f"  l1_id      : {l1_id}",
        f"  session_id : {session_id}",
        f"  消息条数   : {len(messages)}",
    ]

    if l1_row:
        result_lines += [
            f"",
            f"摘要内容：",
            f"  {l1_row['summary']}",
            f"  关键词  ：{l1_row['keywords']  or '无'}",
            f"  时间段  ：{l1_row['time_period'] or '未知'}",
            f"  氛围    ：{l1_row['atmosphere']  or '未知'}",
        ]

    result_lines += [
        f"",
        f"后续流程已自动触发：",
        f"  · L1 向量索引写入",
        f"  · 画像候选提取（profile_manager）",
        f"  · 冲突检测（conflict_checker）",
        f"  · L2 条数触发检查（merger）",
    ]

    return "\n".join(result_lines)


# =============================================================================
# update_profile
# =============================================================================

def update_profile(arguments: dict) -> str:
    """
    直接更新 L3 用户画像的某个板块。

    ⚠️  高风险操作，默认禁用。
    需要在 src/ramaria/adapters/mcp/permissions.py 中将 update_profile 设为 True。

    参数（来自 MCP 调用方）：
        field   — 要更新的画像板块名，必填
                  合法值：basic_info / personal_status / interests /
                          social / history / recent_context
        content — 新的板块内容，必填，将完全替换当前内容
    """
    if not is_allowed("update_profile"):
        return (
            "错误：update_profile 工具当前已禁用（默认关闭的高风险操作）。\n"
            "如需开启，请在 src/ramaria/adapters/mcp/permissions.py 中"
            "将 update_profile 设为 True，然后重启 MCP Server。\n"
            "\n"
            "注意：此操作会直接覆写 L3 用户画像，请谨慎使用。"
        )

    valid_fields = VALID_FIELD_KEYS
    field_labels = PROFILE_FIELDS

    field = str(arguments.get("field", "")).strip()
    if not field:
        return "错误：field 参数必填。"
    if field not in valid_fields:
        valid_list = "、".join(valid_fields)
        return f"错误：field 值 {field!r} 不合法。\n合法值：{valid_list}"

    content = str(arguments.get("content", "")).strip()
    if not content:
        return "错误：content 不能为空。"

    try:
        from ramaria.storage.database import get_current_profile
        old_profile = get_current_profile()
        old_content = old_profile.get(field, "（空）")
    except Exception:
        old_content = "（读取失败）"

    try:
        from ramaria.storage.database import update_profile_field
        new_id = update_profile_field(
            field        = field,
            new_content  = content,
            source_l1_id = None,
        )
    except Exception as e:
        return f"更新失败：{e}"

    label = field_labels.get(field, field)
    return (
        f"L3 画像「{label}」已更新。\n"
        f"  新记录 id  : {new_id}\n"
        f"  field      : {field}\n"
        f"\n"
        f"旧内容（已标记为历史版本）：\n"
        f"  {old_content[:100]}{'...' if len(old_content) > 100 else ''}\n"
        f"\n"
        f"新内容（当前生效）：\n"
        f"  {content[:100]}{'...' if len(content) > 100 else ''}"
    )