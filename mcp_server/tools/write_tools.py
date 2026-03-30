"""
mcp_server/tools/write_tools.py — 写入工具集
=====================================================================

包含三个有副作用的写入工具：
    save_message    向 L0 写入一条消息（不触发摘要，不影响 session 状态）
    trigger_l1      触发指定 session 的 L1 摘要生成
    update_profile  直接更新 L3 用户画像某个板块（默认禁用）

设计原则：
    1. 最小副作用：
       · save_message 只写 L0，不触发任何后台流程
       · trigger_l1   只触发摘要，不关闭/创建 session
       · update_profile 只修改画像，不级联其他操作

    2. 外部 session 隔离：
       · MCP 写入的消息不复用 FastAPI 的活跃 session，
         避免与正在进行的实时对话混淆
       · 外部 session 的 started_at 记录为写入时间，ended_at 也同时写入
         （因为 MCP 写入的是"历史"或"外部注入"消息，不需要空闲检测）

    3. 权限检查：
       · 每个写入函数在执行前都调用 permissions.is_allowed() 检查
       · update_profile 默认禁用，需要在 permissions.py 中手动开启

    4. 幂等性提示：
       · trigger_l1 会检查 session 是否已有摘要，避免重复生成

与其他模块的关系：
    - 调用 database.new_session_with_time() / save_message_with_fingerprint() / close_session_with_time()
    - 调用 summarizer.generate_l1_summary()
    - 调用 database.update_profile_field()
    - 调用 permissions.is_allowed() 检查权限
"""

import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# ── 将项目根目录加入 sys.path ──
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp_server.permissions import is_allowed


# =============================================================================
# 内部工具函数
# =============================================================================

def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _make_fingerprint(timestamp_ms: int, role: str, content: str) -> str:
    """
    计算消息唯一指纹（与 qq_parser._make_fingerprint 逻辑一致）。
    用于重复导入预检，防止同一条消息被写入多次。
    取 SHA-256 的前16位十六进制字符（8字节），碰撞概率极低。
    """
    raw = f"{timestamp_ms}|{role}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _get_or_create_mcp_session() -> int:
    """
    获取或创建 MCP 专用的外部写入 session。

    策略：
        每次调用 save_message 时都创建一个新的独立 session。
        session 的 started_at 和 ended_at 都设为当前时间，
        表示这是一个"即时完成"的外部注入，不参与空闲检测流程。

    为什么不复用已有的外部 session？
        MCP 调用可能来自不同工具、不同时间点，
        将不相关的消息强行塞进同一个 session 会破坏摘要的语义完整性。
        每条消息独立成 session，trigger_l1 时可以精确控制哪个 session 提炼。

    返回：
        int — 新创建的 session id
    """
    from database import new_session_with_time, close_session_with_time

    now = _now_iso()
    session_id = new_session_with_time(now)
    # 立即关闭：外部注入不需要空闲检测
    close_session_with_time(session_id, now)
    return session_id


# =============================================================================
# save_message
# =============================================================================

def save_message(arguments: dict) -> str:
    """
    向 L0 原始消息层写入一条消息。

    设计说明：
        此工具专门用于外部工具（如 Claude Desktop）向珊瑚菌注入信息，
        例如："今天在 Claude Desktop 里讨论了 XXX，记录一下"。

        写入后不会触发任何自动流程（不触发 L1 摘要、不更新 session 状态）。
        需要将这条消息提炼进记忆时，调用 trigger_l1 并传入返回的 session_id。

    参数（来自 MCP 调用方）：
        role         — 消息角色，"user" 或 "assistant"，必填
        content      — 消息内容，必填
        source_hint  — 可选，来源说明，如 "来自Claude Desktop的记录"
                       会被拼接进 content 的末尾（用换行分隔）

    返回：
        写入结果，包含 session_id（用于后续 trigger_l1）。
    """
    if not is_allowed("save_message"):
        return "错误：save_message 工具当前已禁用。请在 permissions.py 中开启。"

    # 参数解析
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
        from database import save_message_with_fingerprint

        # 创建专属的外部 session
        session_id = _get_or_create_mcp_session()

        # 计算指纹，防止重复写入
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        fingerprint = _make_fingerprint(ts_ms, role, content)
        now = _now_iso()

        # 写入消息
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
      · 调用本地 Qwen 模型生成摘要
      · 写入 memory_l1 表
      · 写入 L1 向量索引
      · 触发画像提取（profile_manager）
      · 触发冲突检测（conflict_checker）
      · 触发 L2 条数检查（merger）

    注意事项：
      · 本工具会调用本地 LLM，需要 LM Studio 正在运行
      · 如果 session 已有 L1 摘要，会再生成一条（幂等性由调用方保证）
      · 生成耗时较长（通常10~30秒），请耐心等待

    参数（来自 MCP 调用方）：
        session_id — 必填，要提炼的 session id（整数）
                     通常来自 save_message 的返回值，或 get_pending_sessions 的列表

    返回：
        摘要生成结果（成功时包含 L1 摘要内容）。
    """
    if not is_allowed("trigger_l1"):
        return "错误：trigger_l1 工具当前已禁用。请在 permissions.py 中开启。"

    # 参数解析
    session_id_raw = arguments.get("session_id")
    if session_id_raw is None:
        return "错误：session_id 参数必填。可通过 get_pending_sessions 查询待处理列表。"

    try:
        session_id = int(session_id_raw)
    except (ValueError, TypeError):
        return f"错误：session_id 必须是整数，收到：{session_id_raw!r}"

    # 检查 session 是否存在且有消息
    try:
        from database import get_messages, get_session, get_l1_by_session

        session = get_session(session_id)
        if session is None:
            return f"错误：session {session_id} 不存在。"

        messages = get_messages(session_id)
        if not messages:
            return f"session {session_id} 没有消息，跳过摘要生成。"

        # 提示已有摘要（但不阻止重复生成，让调用方决定）
        existing_l1 = get_l1_by_session(session_id)
        warning = ""
        if existing_l1:
            warning = (
                f"\n警告：session {session_id} 已有 L1 摘要（id={existing_l1['id']}）。\n"
                f"本次调用将额外生成一条新摘要。\n"
            )

    except Exception as e:
        return f"查询 session 信息失败：{e}"

    # 触发摘要生成（调用路径和实时对话完全一致）
    try:
        from summarizer import generate_l1_summary
        l1_id = generate_l1_summary(session_id)
    except Exception as e:
        return f"摘要生成时发生异常：{e}"

    if l1_id is None:
        return (
            f"摘要生成失败（session_id={session_id}）。\n"
            "可能原因：本地模型未启动（需要 LM Studio 运行中），"
            "或消息内容不足以生成有效摘要。"
        )

    # 读取刚生成的摘要内容，返回给调用方
    try:
        from database import get_l1_by_id
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
    需要在 mcp_server/permissions.py 中将 update_profile 设为 True 才能使用。

    与 profile_manager 的区别：
        profile_manager 是自动从 L1 摘要中提取信息（被动更新）
        update_profile  是外部工具直接覆写指定板块（主动覆盖）

    操作语义：
        采用和 database.update_profile_field() 相同的追加写入策略，
        旧版本标记为历史（is_current=0），插入新版本（is_current=1）。
        不会丢失历史数据，可以追溯。

    参数（来自 MCP 调用方）：
        field   — 要更新的画像板块名，必填
                  合法值：basic_info / personal_status / interests /
                          social / history / recent_context
        content — 新的板块内容，必填

    返回：
        更新结果。
    """
    if not is_allowed("update_profile"):
        return (
            "错误：update_profile 工具当前已禁用（默认关闭的高风险操作）。\n"
            "如需开启，请在 mcp_server/permissions.py 中将 update_profile 设为 True，"
            "然后重启 MCP Server。\n"
            "\n"
            "注意：此操作会直接覆写 L3 用户画像，请谨慎使用。"
        )

    # 合法 field 集合（与 user_profile 表和 conflict_checker 保持一致）
    valid_fields = {
        "basic_info", "personal_status", "interests",
        "social", "history", "recent_context",
    }
    field_labels = {
        "basic_info":      "基础信息",
        "personal_status": "近期状态",
        "interests":       "兴趣爱好",
        "social":          "社交情况",
        "history":         "重要经历",
        "recent_context":  "近期背景",
    }

    # 参数解析
    field = str(arguments.get("field", "")).strip()
    if not field:
        return "错误：field 参数必填。"
    if field not in valid_fields:
        valid_list = "、".join(valid_fields)
        return f"错误：field 值 {field!r} 不合法。\n合法值：{valid_list}"

    content = str(arguments.get("content", "")).strip()
    if not content:
        return "错误：content 不能为空。"

    # 读取旧版本，用于回显
    try:
        from database import get_current_profile
        old_profile = get_current_profile()
        old_content = old_profile.get(field, "（空）")
    except Exception:
        old_content = "（读取失败）"

    # 执行更新（追加写入，旧版本保留为历史）
    try:
        from database import update_profile_field
        new_id = update_profile_field(
            field        = field,
            new_content  = content,
            source_l1_id = None,   # MCP 直接写入，无对应 L1
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
