"""
src/ramaria/importer/qq/parser.py — QQ 聊天记录解析核心

适配格式：shuakami/qq-chat-exporter v5.x 导出的 JSON 文件

职责：
    1. 读取并校验 QQ Chat Exporter 导出的 JSON 文件
    2. 解析每条消息，识别消息类型，提取有效文本
    3. 按时间间隔切割 session
    4. 生成详细的诊断报告（dry run），列出成功/降级/跳过三类
    5. 导入前批量预检，识别已经写入数据库的重复消息
    6. 输出标准化消息列表，供导入层写入数据库

"""

import hashlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ========= 终端 UTF-8 修复（Windows 兼容） =========
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# =============================================================================
# 已知消息 type 常量
# =============================================================================

TYPE_TEXT    = "type_1"
TYPE_REPLY   = "type_3"
TYPE_AUDIO   = "type_6"
TYPE_CARD    = "type_7"
TYPE_VIDEO   = "type_9"
TYPE_FORWARD = "type_11"

ELEM_IMAGE = "image"
ELEM_REPLY = "reply"


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class ParsedMessage:
    """解析后的单条消息，是写入数据库的基本单元。"""
    role:        str   # "user" 或 "assistant"
    content:     str   # 消息正文（已处理占位符和前缀）
    timestamp:   str   # UTC ISO 8601
    original_ts: int   # 原始 Unix 毫秒时间戳，用于 session 切割和重复检测
    fingerprint: str   # 唯一指纹（sha256 前16位），用于重复预检


@dataclass
class SkippedItem:
    """被跳过或降级处理的单条消息记录，用于诊断报告。"""
    reason:          str
    time:            str
    original_type:   str
    content_preview: str


@dataclass
class ParseReport:
    """完整的诊断报告，包含三个桶 + 重复预检结果。"""
    file_path:  str = ""
    self_uid:   str = ""
    self_name:  str = ""
    chat_name:  str = ""
    chat_type:  str = ""

    time_start: str = ""
    time_end:   str = ""

    total_raw:     int = 0
    dedup_removed: int = 0

    success_text:          int = 0
    success_image:         int = 0
    success_reply:         int = 0
    success_other_sender:  int = 0

    degraded_reply_fallback: int = 0
    degraded_forward:        int = 0
    degraded_card:           int = 0
    degraded_audio:          int = 0
    degraded_video:          int = 0

    skipped_recalled: int = 0
    skipped_empty:    int = 0
    skipped_unknown:  int = 0

    unknown_types_seen: list = field(default_factory=list)

    duplicate_check_enabled: bool = False
    duplicates_found:         int = 0

    skipped_items:  list = field(default_factory=list)
    degraded_items: list = field(default_factory=list)

    session_count: int = 0
    gap_minutes:   int = 10

    def print_summary(self):
        """在终端打印完整诊断报告，命令行版 dry run 时调用。"""
        total_success = self.success_text + self.success_image + self.success_reply
        total_degraded = (
            self.degraded_reply_fallback + self.degraded_forward
            + self.degraded_card + self.degraded_audio + self.degraded_video
        )
        total_skipped = self.skipped_recalled + self.skipped_empty + self.skipped_unknown

        print()
        print("=" * 60)
        print("  QQ 聊天记录导入 · 诊断报告")
        print("=" * 60)
        print(f"\n【文件信息】")
        print(f"  文件路径    : {self.file_path}")
        print(f"  导出用户    : {self.self_name}（{self.self_uid}）")
        print(f"  对话对象    : {self.chat_name}（{self.chat_type}）")
        print(f"  时间跨度    : {self.time_start} ~ {self.time_end}")
        print(f"\n【总览】")
        print(f"  原始消息数  : {self.total_raw} 条")
        print(f"  文件内去重  : {self.dedup_removed} 条")
        print(f"  切割 session: {self.session_count} 个（间隔阈值 {self.gap_minutes} 分钟）")
        if self.duplicate_check_enabled:
            print(f"\n【重复导入预检】")
            print(f"  发现重复消息: {self.duplicates_found} 条（将跳过）")
        print(f"\n✅  成功解析（共 {total_success} 条）")
        print(f"  纯文本: {self.success_text}  含图片: {self.success_image}  回复: {self.success_reply}")
        print(f"\n⚠️   降级处理（共 {total_degraded} 条）")
        print(f"\n❌  完全跳过（共 {total_skipped} 条）")
        print()
        print("=" * 60)

    def to_dict(self) -> dict:
        """序列化为字典，供 FastAPI 接口返回给前端。"""
        total_success = self.success_text + self.success_image + self.success_reply
        total_degraded = (
            self.degraded_reply_fallback + self.degraded_forward
            + self.degraded_card + self.degraded_audio + self.degraded_video
        )
        total_skipped = self.skipped_recalled + self.skipped_empty + self.skipped_unknown

        def item_to_dict(item: SkippedItem) -> dict:
            return {
                "reason":          item.reason,
                "time":            item.time,
                "original_type":   item.original_type,
                "content_preview": item.content_preview,
            }

        return {
            "file_info": {
                "file_path":  self.file_path,
                "self_uid":   self.self_uid,
                "self_name":  self.self_name,
                "chat_name":  self.chat_name,
                "chat_type":  self.chat_type,
                "time_start": self.time_start,
                "time_end":   self.time_end,
            },
            "overview": {
                "total_raw":     self.total_raw,
                "dedup_removed": self.dedup_removed,
                "session_count": self.session_count,
                "gap_minutes":   self.gap_minutes,
            },
            "duplicate_check": {
                "enabled":          self.duplicate_check_enabled,
                "duplicates_found": self.duplicates_found,
            },
            "success": {
                "total":        total_success,
                "text":         self.success_text,
                "image":        self.success_image,
                "reply":        self.success_reply,
                "other_sender": self.success_other_sender,
            },
            "degraded": {
                "total":          total_degraded,
                "reply_fallback": self.degraded_reply_fallback,
                "forward":        self.degraded_forward,
                "card":           self.degraded_card,
                "audio":          self.degraded_audio,
                "video":          self.degraded_video,
                "items":          [item_to_dict(i) for i in self.degraded_items[:50]],
            },
            "skipped": {
                "total":         total_skipped,
                "recalled":      self.skipped_recalled,
                "empty":         self.skipped_empty,
                "unknown":       self.skipped_unknown,
                "unknown_types": sorted(set(self.unknown_types_seen)),
                "items":         [item_to_dict(i) for i in self.skipped_items[:50]],
            },
        }


@dataclass
class ParseResult:
    """parse_qq_export() 的完整返回值。"""
    parsed_sessions: list   # list[list[ParsedMessage]]
    report:          ParseReport


# =============================================================================
# 工具函数
# =============================================================================

def _ts_ms_to_iso(ts_ms: int) -> str:
    """Unix 毫秒时间戳 → UTC ISO 8601 字符串。"""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.isoformat()


def _content_preview(text: str, max_len: int = 50) -> str:
    """截取前 max_len 个字符作为预览。"""
    if not text:
        return "（空）"
    preview = text.replace("\n", " ").strip()
    return preview[:max_len] + "…" if len(preview) > max_len else preview


def _has_image_element(elements: list) -> bool:
    """判断 elements 列表中是否包含图片类型的元素。"""
    return any(e.get("type") == ELEM_IMAGE for e in elements)


def _get_reply_element(elements: list) -> Optional[dict]:
    """从 elements 列表中提取 type=reply 的元素数据。"""
    for elem in elements:
        if elem.get("type") == ELEM_REPLY:
            return elem.get("data", {})
    return None


def _clean_image_placeholders(text: str) -> str:
    """将导出工具生成的图片占位符统一替换为 [图片]。"""
    return re.sub(r'\[图片:\s*[^\]]+\]', '[图片]', text).strip()


def _extract_reply_body(content_text: str) -> str:
    """
    从 type_3 消息的 content.text 中提取回复正文（去掉引用头部）。
    作为降级处理，当 elements 里找不到 reply 元素时调用。
    """
    if "\n" in content_text:
        body = content_text.split("\n", 1)[1].strip()
        if body:
            return body

    cleaned = re.sub(r'^\[回复[^\]]*\]\s*', '', content_text).strip()
    if cleaned and cleaned != content_text:
        return cleaned

    return content_text


def _make_fingerprint(original_ts: int, role: str, content: str) -> str:
    """计算消息唯一指纹，用于重复导入预检。"""
    raw = f"{original_ts}|{role}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_json_file(file_path: str) -> dict:
    """尝试多种编码方式读取 JSON 文件，兼容 Windows GBK 环境。"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]
    last_error = None

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            last_error = e
            continue

    raise ValueError(
        f"无法解析文件 {file_path}，已尝试编码：{encodings}，"
        f"最后错误：{last_error}"
    )


# =============================================================================
# 单条消息解析
# =============================================================================

def _parse_single_message(
    raw_msg: dict,
    self_uid: str,
    report: ParseReport,
) -> Optional[ParsedMessage]:
    """
    解析单条原始消息，返回 ParsedMessage 或 None（跳过时）。
    同时原地更新 report 的统计计数和详细列表。
    """
    timestamp   = raw_msg.get("timestamp", 0)
    time_str    = raw_msg.get("time", "未知时间")
    msg_type    = raw_msg.get("type", "")
    recalled    = raw_msg.get("recalled", False)
    content     = raw_msg.get("content", {})
    sender      = raw_msg.get("sender", {})
    elements    = content.get("elements", [])
    raw_text    = content.get("text", "").strip()
    sender_uid  = sender.get("uid", "")
    sender_name = sender.get("name", "")

    # 规则1：撤回消息直接跳过
    if recalled:
        report.skipped_recalled += 1
        report.skipped_items.append(SkippedItem(
            reason="recalled（撤回消息）",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))
        return None

    # 规则2：content.text 完全为空，跳过
    if not raw_text:
        report.skipped_empty += 1
        report.skipped_items.append(SkippedItem(
            reason="content.text 为空",
            time=time_str,
            original_type=msg_type,
            content_preview="（空）",
        ))
        return None

    # 规则3：根据 type 分流
    if msg_type == TYPE_TEXT:
        if _has_image_element(elements):
            final_text = _clean_image_placeholders(raw_text)
            if not final_text:
                final_text = "[图片]"
            report.success_image += 1
        else:
            final_text = raw_text
            report.success_text += 1

    elif msg_type == TYPE_REPLY:
        reply_elem = _get_reply_element(elements)
        if reply_elem:
            quoted_sender  = reply_elem.get("senderName", "").strip()
            quoted_content = reply_elem.get("content", "").strip()
            reply_body     = _extract_reply_body(raw_text)
            if len(quoted_content) > 30:
                quoted_content = quoted_content[:30] + "…"
            final_text = f"「回复 {quoted_sender}: {quoted_content}」{reply_body}"
            report.success_reply += 1
        else:
            final_text = _extract_reply_body(raw_text)
            report.degraded_reply_fallback += 1
            report.degraded_items.append(SkippedItem(
                reason="回复消息（无reply元素，已提取正文）",
                time=time_str,
                original_type=msg_type,
                content_preview=_content_preview(raw_text),
            ))

    elif msg_type == TYPE_AUDIO:
        final_text = "[语音]"
        report.degraded_audio += 1
        report.degraded_items.append(SkippedItem(
            reason="语音消息 → [语音]",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))

    elif msg_type == TYPE_VIDEO:
        final_text = "[视频]"
        report.degraded_video += 1
        report.degraded_items.append(SkippedItem(
            reason="视频消息 → [视频]",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))

    elif msg_type == TYPE_FORWARD:
        final_text = "[转发消息]"
        report.degraded_forward += 1
        report.degraded_items.append(SkippedItem(
            reason="合并转发 → [转发消息]",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))

    elif msg_type == TYPE_CARD:
        final_text = "[卡片消息]"
        report.degraded_card += 1
        report.degraded_items.append(SkippedItem(
            reason="卡片消息 → [卡片消息]",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))

    else:
        report.skipped_unknown += 1
        report.unknown_types_seen.append(msg_type)
        report.skipped_items.append(SkippedItem(
            reason=f"未知 type：{msg_type}",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))
        return None

    # 规则4：role 映射
    if sender_uid == self_uid:
        role          = "user"
        content_final = final_text
    else:
        role   = "assistant"
        prefix = f"[{sender_name}] " if sender_name else "[对方] "
        content_final = prefix + final_text
        if msg_type in (TYPE_TEXT, TYPE_REPLY) and (
            msg_type != TYPE_REPLY or _get_reply_element(elements) is not None
        ):
            report.success_other_sender += 1

    fingerprint = _make_fingerprint(timestamp, role, content_final)

    return ParsedMessage(
        role        = role,
        content     = content_final,
        timestamp   = _ts_ms_to_iso(timestamp),
        original_ts = timestamp,
        fingerprint = fingerprint,
    )


# =============================================================================
# session 切割
# =============================================================================

def _split_into_sessions(messages: list, gap_minutes: int) -> list:
    """按时间间隔将消息列表切割为若干 session。"""
    if not messages:
        return []

    sessions        = []
    current_session = [messages[0]]
    gap_ms          = gap_minutes * 60 * 1000

    for msg in messages[1:]:
        if msg.original_ts - current_session[-1].original_ts > gap_ms:
            sessions.append(current_session)
            current_session = [msg]
        else:
            current_session.append(msg)

    if current_session:
        sessions.append(current_session)

    return sessions


# =============================================================================
# 重复导入预检
# =============================================================================

def _check_duplicates_against_db(
    parsed_messages: list,
    report: ParseReport,
) -> list:
    """
    将解析结果与数据库中的现有消息比对，过滤掉已经导入过的消息。
    预检失败时降级为不过滤，不阻断导入流程。
    """
    report.duplicate_check_enabled = True

    try:
        from ramaria.storage.database import get_all_message_fingerprints
        existing_fps = get_all_message_fingerprints()
    except ImportError:
        print("（提示）无法导入 database 模块，跳过重复导入预检")
        report.duplicate_check_enabled = False
        return parsed_messages
    except Exception as e:
        print(f"（警告）重复预检查询失败，跳过预检 — {e}")
        report.duplicate_check_enabled = False
        return parsed_messages

    filtered   = []
    duplicates = 0

    for msg in parsed_messages:
        if msg.fingerprint in existing_fps:
            duplicates += 1
        else:
            filtered.append(msg)

    report.duplicates_found = duplicates
    return filtered


# =============================================================================
# 主解析函数（对外接口）
# =============================================================================

def parse_qq_export(
    file_path: str,
    gap_minutes: int = 10,
    check_duplicates: bool = True,
) -> ParseResult:
    """
    解析 QQ Chat Exporter v5 导出的 JSON 文件。
    这是本模块唯一的对外接口。

    参数：
        file_path        — JSON 文件的绝对或相对路径
        gap_minutes      — session 切割的时间间隔阈值（分钟），默认10分钟
        check_duplicates — 是否执行重复导入预检（默认True）

    返回：
        ParseResult：parsed_sessions + report

    异常：
        FileNotFoundError — 文件不存在时抛出
        ValueError        — 文件格式不匹配时抛出
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    raw_data = _load_json_file(str(path))

    if "chatInfo" not in raw_data or "messages" not in raw_data:
        raise ValueError(
            "文件格式不匹配：缺少 chatInfo 或 messages 字段。\n"
            "请确认文件是由 shuakami/qq-chat-exporter 导出的 JSON 格式。"
        )

    chat_info    = raw_data.get("chatInfo", {})
    raw_messages = raw_data.get("messages", [])

    self_uid  = chat_info.get("selfUid", "")
    self_name = chat_info.get("selfName", "")
    chat_name = chat_info.get("name", "")
    chat_type = chat_info.get("type", "unknown")

    report = ParseReport(
        file_path   = str(path.resolve()),
        self_uid    = self_uid,
        self_name   = self_name,
        chat_name   = chat_name,
        chat_type   = chat_type,
        total_raw   = len(raw_messages),
        gap_minutes = gap_minutes,
    )

    # 文件内去重（以 id + timestamp 为联合键）
    seen_keys    = set()
    deduped_msgs = []
    for msg in raw_messages:
        key = (msg.get("id", ""), msg.get("timestamp", 0))
        if key in seen_keys:
            report.dedup_removed += 1
            continue
        seen_keys.add(key)
        deduped_msgs.append(msg)

    # 按时间戳排序
    deduped_msgs.sort(key=lambda m: m.get("timestamp", 0))

    # 逐条解析
    parsed_messages = []
    for raw_msg in deduped_msgs:
        result = _parse_single_message(raw_msg, self_uid, report)
        if result is not None:
            parsed_messages.append(result)

    # 重复导入预检
    if check_duplicates and parsed_messages:
        parsed_messages = _check_duplicates_against_db(parsed_messages, report)

    # 切割 session
    parsed_sessions      = _split_into_sessions(parsed_messages, gap_minutes)
    report.session_count = len(parsed_sessions)

    # 更新时间范围
    if parsed_messages:
        report.time_start = parsed_messages[0].timestamp[:10]
        report.time_end   = parsed_messages[-1].timestamp[:10]

    return ParseResult(
        parsed_sessions = parsed_sessions,
        report          = report,
    )