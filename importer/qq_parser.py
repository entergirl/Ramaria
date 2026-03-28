"""
importer/qq_parser.py — QQ 聊天记录解析核心
=====================================================================

适配格式：shuakami/qq-chat-exporter v5.x 导出的 JSON 文件

职责：
    1. 读取并校验 QQ Chat Exporter 导出的 JSON 文件
    2. 解析每条消息，识别消息类型，提取有效文本
    3. 按时间间隔切割 session
    4. 生成详细的诊断报告（dry run），列出成功/降级/跳过三类
    5. 导入前批量预检，识别已经写入数据库的重复消息
    6. 输出标准化消息列表，供导入层写入数据库

消息类型处理策略（已适配）：
    type_1  普通消息（文字/图片/混合）
        · 纯文本       → 直接提取 content.text，归入 ✅ 成功
        · 含图片元素   → [图片: xxx] 替换为 [图片]，归入 ✅ 成功
    type_3  回复消息
        · 从 elements 中找 type=reply 元素，提取被引用者和原文
        · 格式化为 「回复 发送人: 被引用内容」新消息正文，归入 ✅ 成功
        · 找不到 reply 元素时，截取 content.text 的正文部分，归入 ⚠️ 降级
    type_6  语音消息  → 替换为 [语音]，归入 ⚠️ 降级
    type_9  视频消息  → 替换为 [视频]，归入 ⚠️ 降级
    type_11 合并转发  → 替换为 [转发消息]，归入 ⚠️ 降级
    type_7  卡片消息  → 替换为 [卡片消息]，归入 ⚠️ 降级
    recalled=true 撤回 → 整条跳过，归入 ❌ 跳过
    重复消息（id+ts）  → 整条跳过，归入 ❌ 跳过
    content.text 为空  → 整条跳过，归入 ❌ 跳过
    未知 type         → 整条跳过，归入 ❌ 跳过（记录 type 值供后续适配）

重复导入预检机制：
    调用 parse_qq_export() 时传入 check_duplicates=True（默认开启），
    解析完成后自动查询数据库中的现有消息，计算指纹（timestamp_ms + role + content）
    并与解析结果对比，将已存在的消息过滤掉，在报告中单独列出。
    干跑模式下也会做预检，让用户知道实际会写入多少条。

session 切割规则：
    相邻两条消息时间间隔超过 gap_minutes（默认10分钟）时，切断为新 session。
    与项目主流程的 L1_IDLE_MINUTES 保持一致，可通过参数覆盖。

role 映射规则：
    sender.uid == chatInfo.selfUid  → role = "user"
    其他 uid                        → role = "assistant"，content 加 [发送人名] 前缀

使用方法：
    from importer.qq_parser import parse_qq_export

    # 标准用法（含重复预检）
    result = parse_qq_export("path/to/export.json", gap_minutes=10)
    result.report.print_summary()

    # 跳过重复预检（数据库为空时可加速）
    result = parse_qq_export("path/to/export.json", check_duplicates=False)
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

TYPE_TEXT    = "type_1"    # 普通消息（文字/图片/混合）
TYPE_REPLY   = "type_3"    # 回复某条特定消息
TYPE_AUDIO   = "type_6"    # 语音消息
TYPE_CARD    = "type_7"    # 卡片/分享消息
TYPE_VIDEO   = "type_9"    # 视频消息
TYPE_FORWARD = "type_11"   # 合并转发

# elements 中的元素类型
ELEM_IMAGE = "image"
ELEM_REPLY = "reply"


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class ParsedMessage:
    """
    解析后的单条消息，是写入数据库的基本单元。
    timestamp 保留原始历史时间，不使用 datetime.now()。
    """
    role: str           # "user" 或 "assistant"
    content: str        # 消息正文（已处理占位符和前缀）
    timestamp: str      # UTC ISO 8601，如 "2024-06-18T08:55:51+00:00"
    original_ts: int    # 原始 Unix 毫秒时间戳，用于 session 切割和重复检测
    fingerprint: str    # 唯一指纹（sha256 前16位），用于重复预检


@dataclass
class SkippedItem:
    """
    被跳过或降级处理的单条消息记录，用于诊断报告。
    """
    reason: str           # 跳过/降级原因
    time: str             # 消息时间字符串，便于人工定位
    original_type: str    # 原始 type 字段值
    content_preview: str  # content.text 的前50字，便于人工核查


@dataclass
class ParseReport:
    """
    完整的诊断报告，包含三个桶 + 重复预检结果。
    """
    # 文件元信息
    file_path: str = ""
    self_uid: str = ""
    self_name: str = ""
    chat_name: str = ""
    chat_type: str = ""   # "private" 或 "group"

    # 时间范围
    time_start: str = ""
    time_end: str = ""

    # 统计数字
    total_raw: int = 0       # JSON 里 messages 数组的原始条目数
    dedup_removed: int = 0   # 文件内去重删除的条数（id+timestamp 相同）

    # ✅ 成功解析（含具体子类型计数）
    success_text: int = 0           # 纯文本消息
    success_image: int = 0          # 含图片（已替换为[图片]）
    success_reply: int = 0          # 回复消息（已格式化为「回复 x: y」z）
    success_other_sender: int = 0   # 对方消息（已加[发送人名]前缀）
    # 注：success_other_sender 与 text/image/reply 是正交维度，不重复计算总数

    # ⚠️ 降级处理（保留占位符）
    degraded_reply_fallback: int = 0  # 回复消息但找不到 reply 元素，降级提取正文
    degraded_forward: int = 0         # 合并转发 → [转发消息]
    degraded_card: int = 0            # 卡片消息 → [卡片消息]
    degraded_audio: int = 0           # 语音消息 → [语音]
    degraded_video: int = 0           # 视频消息 → [视频]

    # ❌ 完全跳过
    skipped_recalled: int = 0   # 撤回消息
    skipped_empty: int = 0      # content.text 为空
    skipped_unknown: int = 0    # 未知 type

    # 未知 type 的具体值列表，供后续适配参考
    unknown_types_seen: list = field(default_factory=list)

    # 重复预检结果（与数据库中现有消息比对）
    duplicate_check_enabled: bool = False   # 是否执行了重复预检
    duplicates_found: int = 0               # 发现的重复消息数（将被跳过）

    # 跳过/降级的详细条目，供逐条查看
    skipped_items: list = field(default_factory=list)    # ❌ 跳过的详细列表
    degraded_items: list = field(default_factory=list)   # ⚠️ 降级的详细列表

    # 切割结果（重复预检后的实际待写入量）
    session_count: int = 0
    gap_minutes: int = 10

    def print_summary(self):
        """
        在终端打印完整诊断报告。
        命令行版 dry run 时调用此方法。
        """
        total_success = self.success_text + self.success_image + self.success_reply
        total_degraded = (
            self.degraded_reply_fallback
            + self.degraded_forward
            + self.degraded_card
            + self.degraded_audio
            + self.degraded_video
        )
        total_skipped = (
            self.skipped_recalled + self.skipped_empty + self.skipped_unknown
        )

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
        print(f"  文件内去重  : {self.dedup_removed} 条（id+timestamp 相同）")
        print(f"  切割 session: {self.session_count} 个（间隔阈值 {self.gap_minutes} 分钟）")

        if self.duplicate_check_enabled:
            print(f"\n【重复导入预检】（与数据库现有消息比对）")
            print(f"  发现重复消息: {self.duplicates_found} 条（将跳过，不重复写入）")
        else:
            print(f"\n【重复导入预检】未执行（check_duplicates=False）")

        print(f"\n✅  成功解析（共 {total_success} 条）")
        print(f"  纯文本消息          : {self.success_text} 条")
        print(f"  含图片（已替换）     : {self.success_image} 条")
        print(f"  回复消息（已格式化） : {self.success_reply} 条")
        print(f"  其中对方消息        : {self.success_other_sender} 条（含前缀，与上方正交统计）")

        print(f"\n⚠️   降级处理（共 {total_degraded} 条，保留占位符）")
        print(f"  回复消息（无reply元素，提取正文）: {self.degraded_reply_fallback} 条")
        print(f"  合并转发 → [转发消息]            : {self.degraded_forward} 条")
        print(f"  卡片消息 → [卡片消息]            : {self.degraded_card} 条")
        print(f"  语音消息 → [语音]                : {self.degraded_audio} 条")
        print(f"  视频消息 → [视频]                : {self.degraded_video} 条")
        if self.degraded_items:
            print(f"  详细列表（前10条）：")
            for item in self.degraded_items[:10]:
                print(f"    [{item.time}] {item.reason}")
                print(f"      预览: {item.content_preview}")

        print(f"\n❌  完全跳过（共 {total_skipped} 条）")
        print(f"  撤回消息    : {self.skipped_recalled} 条")
        print(f"  内容为空    : {self.skipped_empty} 条")
        print(f"  未知 type   : {self.skipped_unknown} 条")
        if self.unknown_types_seen:
            print(f"  → 发现的未知 type 值（供后续适配参考）：")
            for t in sorted(set(self.unknown_types_seen)):
                cnt = self.unknown_types_seen.count(t)
                print(f"    {t}（{cnt} 条）")
        if self.skipped_items:
            print(f"  详细列表（前10条）：")
            for item in self.skipped_items[:10]:
                print(f"    [{item.time}] {item.reason}")
                print(f"      预览: {item.content_preview}")

        print()
        print("=" * 60)

    def to_dict(self) -> dict:
        """
        序列化为字典，供 FastAPI 接口返回给前端。
        skipped_items / degraded_items 只返回前50条，避免响应体过大。
        """
        total_success = self.success_text + self.success_image + self.success_reply
        total_degraded = (
            self.degraded_reply_fallback
            + self.degraded_forward
            + self.degraded_card
            + self.degraded_audio
            + self.degraded_video
        )
        total_skipped = (
            self.skipped_recalled + self.skipped_empty + self.skipped_unknown
        )

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
    """
    parse_qq_export() 的完整返回值。
    parsed_sessions 已经过重复预检过滤，可直接传给导入层写入。
    """
    parsed_sessions: list   # list[list[ParsedMessage]]，切割后的 session 列表
    report: ParseReport     # 详细诊断报告


# =============================================================================
# 工具函数
# =============================================================================

def _ts_ms_to_iso(ts_ms: int) -> str:
    """
    将 Unix 毫秒时间戳转换为 UTC ISO 8601 字符串。
    例：1718700951000 → "2024-06-18T08:55:51+00:00"
    """
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.isoformat()


def _content_preview(text: str, max_len: int = 50) -> str:
    """截取前 max_len 个字符作为预览，去除换行。"""
    if not text:
        return "（空）"
    preview = text.replace("\n", " ").strip()
    return preview[:max_len] + "…" if len(preview) > max_len else preview


def _has_image_element(elements: list) -> bool:
    """判断 elements 列表中是否包含图片类型的元素。"""
    return any(e.get("type") == ELEM_IMAGE for e in elements)


def _get_reply_element(elements: list) -> Optional[dict]:
    """
    从 elements 列表中提取 type=reply 的元素数据。

    type_3 消息的 elements 中，reply 元素结构为：
        {
            "type": "reply",
            "data": {
                "messageId": "...",
                "referencedMessageId": "...",
                "senderName": "被引用者的uid或名字",
                "content": "被引用的原始消息内容"
            }
        }

    返回 data 字典，找不到时返回 None。
    """
    for elem in elements:
        if elem.get("type") == ELEM_REPLY:
            return elem.get("data", {})
    return None


def _clean_image_placeholders(text: str) -> str:
    """
    将导出工具生成的图片占位符统一替换为 [图片]。
    原格式：[图片: 69AECF05E777DCC429944D670C993CFE.jpg]
    目标格式：[图片]
    """
    return re.sub(r'\[图片:\s*[^\]]+\]', '[图片]', text).strip()


def _extract_reply_body(content_text: str) -> str:
    """
    从 type_3 消息的 content.text 中提取回复正文（去掉引用头部）。

    content.text 格式为：
        "[回复 uid: 被引用内容]\n实际回复正文"

    此函数作为降级处理，当 elements 里找不到 reply 元素时调用。
    优先按第一个换行符切分，其次用正则去掉 [回复 xxx:] 前缀。
    两种方式都失败时，返回原始 content_text（宁可有噪声，不要丢内容）。
    """
    # 方式一：按第一个换行符切分
    if "\n" in content_text:
        body = content_text.split("\n", 1)[1].strip()
        if body:
            return body

    # 方式二：正则去掉 [回复 xxx: yyy] 前缀
    cleaned = re.sub(r'^\[回复[^\]]*\]\s*', '', content_text).strip()
    if cleaned and cleaned != content_text:
        return cleaned

    # 都失败，返回原始文本
    return content_text


def _make_fingerprint(original_ts: int, role: str, content: str) -> str:
    """
    计算消息的唯一指纹，用于重复导入预检。

    以 (timestamp_ms, role, content) 三元组计算 SHA-256，
    取前16位十六进制字符（8字节），足够在实际规模下避免碰撞。

    不用 message_id 是因为导出工具存在 id 重复的 bug（样本中已出现）。
    """
    raw = f"{original_ts}|{role}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_json_file(file_path: str) -> dict:
    """
    尝试多种编码方式读取 JSON 文件，兼容 Windows GBK 环境。
    所有编码均失败时抛出 ValueError。
    """
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
        f"无法解析文件 {file_path}，已尝试编码：{encodings}，最后错误：{last_error}"
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
    解析单条原始消息，返回 ParsedMessage 或 None（跳过时返回 None）。
    同时原地更新 report 的统计计数和详细列表。

    参数：
        raw_msg  — messages 数组中的单个原始消息字典
        self_uid — chatInfo.selfUid，用于判断 role
        report   — 诊断报告对象，原地更新
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

    # ------------------------------------------------------------------
    # 规则1：撤回消息直接跳过
    # ------------------------------------------------------------------
    if recalled:
        report.skipped_recalled += 1
        report.skipped_items.append(SkippedItem(
            reason="recalled（撤回消息）",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))
        return None

    # ------------------------------------------------------------------
    # 规则2：content.text 完全为空，跳过
    # ------------------------------------------------------------------
    if not raw_text:
        report.skipped_empty += 1
        report.skipped_items.append(SkippedItem(
            reason="content.text 为空",
            time=time_str,
            original_type=msg_type,
            content_preview="（空）",
        ))
        return None

    # ------------------------------------------------------------------
    # 规则3：根据 type 分流，生成 final_text
    # ------------------------------------------------------------------

    if msg_type == TYPE_TEXT:
        # ── type_1：普通消息（文字/图片/混合） ──
        if _has_image_element(elements):
            final_text = _clean_image_placeholders(raw_text)
            if not final_text:
                final_text = "[图片]"   # 纯图片消息，替换后为空时用占位符
            report.success_image += 1
        else:
            final_text = raw_text
            report.success_text += 1

    elif msg_type == TYPE_REPLY:
        # ── type_3：回复某条特定消息 ──
        reply_elem = _get_reply_element(elements)

        if reply_elem:
            # 成功找到 reply 元素，格式化为「回复 发送人: 被引用内容」正文
            quoted_sender  = reply_elem.get("senderName", "").strip()
            quoted_content = reply_elem.get("content", "").strip()
            reply_body     = _extract_reply_body(raw_text)

            # 被引用内容过长时截断，避免污染 L1 摘要
            if len(quoted_content) > 30:
                quoted_content = quoted_content[:30] + "…"

            # 用中文书名号「」包裹引用部分，与 [前缀] 视觉区分
            final_text = f"「回复 {quoted_sender}: {quoted_content}」{reply_body}"
            report.success_reply += 1

        else:
            # 找不到 reply 元素（导出格式变化），降级只保留正文
            final_text = _extract_reply_body(raw_text)
            report.degraded_reply_fallback += 1
            report.degraded_items.append(SkippedItem(
                reason="回复消息（无reply元素，已提取正文）",
                time=time_str,
                original_type=msg_type,
                content_preview=_content_preview(raw_text),
            ))

    elif msg_type == TYPE_AUDIO:
        # ── type_6：语音消息 ──
        final_text = "[语音]"
        report.degraded_audio += 1
        report.degraded_items.append(SkippedItem(
            reason="语音消息 → [语音]",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))

    elif msg_type == TYPE_VIDEO:
        # ── type_9：视频消息 ──
        final_text = "[视频]"
        report.degraded_video += 1
        report.degraded_items.append(SkippedItem(
            reason="视频消息 → [视频]",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))

    elif msg_type == TYPE_FORWARD:
        # ── type_11：合并转发 ──
        final_text = "[转发消息]"
        report.degraded_forward += 1
        report.degraded_items.append(SkippedItem(
            reason="合并转发 → [转发消息]",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))

    elif msg_type == TYPE_CARD:
        # ── type_7：卡片/分享消息 ──
        final_text = "[卡片消息]"
        report.degraded_card += 1
        report.degraded_items.append(SkippedItem(
            reason="卡片消息 → [卡片消息]",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))

    else:
        # ── 未知 type：完全跳过，记录 type 值供后续适配 ──
        report.skipped_unknown += 1
        report.unknown_types_seen.append(msg_type)
        report.skipped_items.append(SkippedItem(
            reason=f"未知 type：{msg_type}",
            time=time_str,
            original_type=msg_type,
            content_preview=_content_preview(raw_text),
        ))
        return None

    # ------------------------------------------------------------------
    # 规则4：role 映射
    # selfUid 的消息 → user
    # 其他 uid 的消息 → assistant，content 加 [发送人名] 前缀
    # ------------------------------------------------------------------
    if sender_uid == self_uid:
        role          = "user"
        content_final = final_text
    else:
        role   = "assistant"
        # 方括号前缀，避免与摘要格式"助手：xxx"产生冲突
        prefix = f"[{sender_name}] " if sender_name else "[对方] "
        content_final = prefix + final_text
        # success_other_sender 是"对方消息"维度，与 text/image/reply 正交
        # 只对 type_1 和 type_3 的成功解析（非降级）条目计入
        if msg_type in (TYPE_TEXT, TYPE_REPLY) and (
            msg_type != TYPE_REPLY or _get_reply_element(elements) is not None
        ):
            report.success_other_sender += 1

    # ------------------------------------------------------------------
    # 生成指纹（用于后续重复预检）
    # ------------------------------------------------------------------
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

def _split_into_sessions(
    messages: list,     # list[ParsedMessage]，已按时间排序
    gap_minutes: int,
) -> list:              # list[list[ParsedMessage]]
    """
    按时间间隔将消息列表切割为若干 session。

    相邻两条消息的 original_ts 差值超过 gap_minutes 分钟时，切断为新 session。
    用毫秒值直接比较，避免时区转换引入误差。
    """
    if not messages:
        return []

    sessions        = []
    current_session = [messages[0]]
    gap_ms          = gap_minutes * 60 * 1000  # 分钟转毫秒

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
    parsed_messages: list,   # list[ParsedMessage]
    report: ParseReport,
) -> list:                   # list[ParsedMessage]，过滤掉重复的
    """
    将解析结果与数据库中的现有消息比对，过滤掉已经导入过的消息。

    从数据库取出所有现有消息的指纹集合（由 database.get_all_message_fingerprints()
    提供），与本次解析结果的指纹做集合差运算，只保留新消息。

    预检失败（数据库不可用、查询出错）时，降级为不过滤，
    不能因为预检失败而阻断整个导入流程。
    """
    report.duplicate_check_enabled = True

    try:
        # 延迟导入，避免模块加载时就连接数据库
        import sys
        from pathlib import Path
        _root = Path(__file__).parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from database import get_all_message_fingerprints
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

    这是本模块唯一的对外接口，命令行版和 FastAPI 版都调用此函数。

    参数：
        file_path        — JSON 文件的绝对或相对路径
        gap_minutes      — session 切割的时间间隔阈值（分钟），默认10分钟
        check_duplicates — 是否执行重复导入预检（默认True）；
                           数据库为空时可设 False 加快速度

    返回：
        ParseResult：
            parsed_sessions — 切割后的 session 列表（已过滤重复）
            report          — 详细诊断报告

    异常：
        FileNotFoundError — 文件不存在时抛出
        ValueError        — 文件格式不匹配时抛出
    """
    # ------------------------------------------------------------------
    # 第一步：读取并校验文件
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 第二步：初始化诊断报告
    # ------------------------------------------------------------------
    report = ParseReport(
        file_path   = str(path.resolve()),
        self_uid    = self_uid,
        self_name   = self_name,
        chat_name   = chat_name,
        chat_type   = chat_type,
        total_raw   = len(raw_messages),
        gap_minutes = gap_minutes,
    )

    # ------------------------------------------------------------------
    # 第三步：文件内去重（以 id + timestamp 为联合键）
    # 针对 qq-chat-exporter 已知的 id 重复 bug
    # ------------------------------------------------------------------
    seen_keys    = set()
    deduped_msgs = []
    for msg in raw_messages:
        key = (msg.get("id", ""), msg.get("timestamp", 0))
        if key in seen_keys:
            report.dedup_removed += 1
            continue
        seen_keys.add(key)
        deduped_msgs.append(msg)

    # ------------------------------------------------------------------
    # 第四步：按时间戳排序（防止导出文件乱序）
    # ------------------------------------------------------------------
    deduped_msgs.sort(key=lambda m: m.get("timestamp", 0))

    # ------------------------------------------------------------------
    # 第五步：逐条解析
    # ------------------------------------------------------------------
    parsed_messages = []
    for raw_msg in deduped_msgs:
        result = _parse_single_message(raw_msg, self_uid, report)
        if result is not None:
            parsed_messages.append(result)

    # ------------------------------------------------------------------
    # 第六步：重复导入预检（与数据库现有消息指纹比对）
    # ------------------------------------------------------------------
    if check_duplicates and parsed_messages:
        parsed_messages = _check_duplicates_against_db(parsed_messages, report)

    # ------------------------------------------------------------------
    # 第七步：按时间间隔切割 session
    # ------------------------------------------------------------------
    parsed_sessions      = _split_into_sessions(parsed_messages, gap_minutes)
    report.session_count = len(parsed_sessions)

    # ------------------------------------------------------------------
    # 第八步：更新报告时间范围（基于实际解析结果）
    # ------------------------------------------------------------------
    if parsed_messages:
        report.time_start = parsed_messages[0].timestamp[:10]
        report.time_end   = parsed_messages[-1].timestamp[:10]

    return ParseResult(
        parsed_sessions = parsed_sessions,
        report          = report,
    )
