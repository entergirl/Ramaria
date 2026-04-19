"""
src/ramaria/memory/conflict_checker.py — 记忆冲突检测模块

负责在 L1 摘要生成后，检测新内容是否与现有用户画像存在矛盾，
将发现的冲突写入 conflict_queue 表，并在对话中以关心口吻询问用户。

核心流程：
    1. 读取最新 L1 摘要 + 当前生效的 L3 用户画像
    2. 调用本地模型做语义比对，返回冲突列表（JSON 数组）
    3. 将每条冲突写入 conflict_queue 表，状态为 pending
    4. 对外暴露两个接口供 main.py 调用：
           get_conflict_question()  — 查询是否有待确认冲突
           handle_conflict_reply()  — 处理用户的"更新"/"忽略"回复

"""

import json
import re

from ramaria.core.llm_client import call_local_summary, strip_thinking
from ramaria.storage.database import (
    get_current_profile,
    get_l1_by_id,
    get_pending_conflicts,
    ignore_conflict,
    resolve_conflict,
    save_conflict,
    update_profile_field,
)

from ramaria.constants import PROFILE_FIELDS, PROFILE_FIELD_LIST, VALID_FIELD_KEYS
from ramaria.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Prompt 模板
# =============================================================================
#
# 从 config.py 迁移到此处，由本模块统一维护。

CONFLICT_CHECK_PROMPT = """你是一个记忆一致性检测助手。请比对下面的"最新对话摘要"和"现有用户画像"，判断是否存在相互矛盾的信息。

【重要提示】
- 只报告真实矛盾，不报告补充信息（新增信息不是矛盾）
- 不报告语境不同导致的表述差异
- 宁可漏报，不要误报，误报会打扰用户

【输出格式要求】
严格按照以下 JSON 格式输出，不要输出任何其他内容（不要加 markdown 代码块，不要加说明文字）：

有冲突时：
[
  {{
    "field": "涉及的画像板块名，只能是以下六个之一：basic_info / personal_status / interests / social / history / recent_context",
    "old_content": "现有画像中与新信息矛盾的具体内容（原文摘录）",
    "new_content": "最新摘要中与旧内容矛盾的具体内容（原文摘录）",
    "conflict_desc": "用一句关心的口吻描述这个矛盾，例如：之前记得你说过……但今天好像……是有什么变化吗？"
  }}
]

无冲突时，直接输出：
[]

【现有用户画像】
{profile_text}

【最新对话摘要】
{l1_summary}
"""

CONFLICT_ASK_TEMPLATE = (
    '{conflict_desc}\n\n'
    '你可以回复"更新"让我记住新的情况，或者回复"忽略"保持现状。'
)


# =============================================================================
# 用户画像格式化
# =============================================================================

# 板块名到中文标签的映射
_FIELD_LABELS: dict[str, str] = PROFILE_FIELDS


def _format_profile_for_check(profile_dict: dict) -> str:
    """
    将用户画像字典格式化为适合冲突检测 Prompt 的文本。

    明确标注英文字段名，方便模型在输出冲突时准确填写 field 字段。

    示例输出：
        [basic_info 基础信息] 烧酒，19岁，学生
        [personal_status 近期状态] 最近在做毕业设计，压力较大

    参数：
        profile_dict — get_current_profile() 返回的字典

    返回：
        str — 格式化后的文本；profile_dict 为空时返回空字符串
    """
    if not profile_dict:
        return ""

    lines = []
    for field_key, label in _FIELD_LABELS.items():
        content = profile_dict.get(field_key, "").strip()
        if content:
            lines.append(f"[{field_key} {label}] {content}")

    return "\n".join(lines)


# =============================================================================
# 解析模型输出
# =============================================================================

def _parse_conflict_json(raw_text: str) -> list | None:
    """
    从模型返回的原始文本中解析冲突列表。

    模型应输出 JSON 数组：有冲突时包含若干冲突对象，无冲突时输出空数组 []。

    解析策略（三步递进）：
        第一步：剥离思考链后直接解析
        第二步：正则提取第一个完整 JSON 数组

    返回：
        list — 冲突对象列表；无冲突时返回 []；解析失败时返回 None
    """
    # 第一步：剥离思考链
    stripped = strip_thinking(raw_text)
    if stripped != raw_text.strip():
        logger.debug("检测到思考链输出，已自动剥离")

    # 第二步：直接尝试解析
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            logger.warning("模型返回了对象而非数组，已自动包装")
            return [result]
    except json.JSONDecodeError:
        pass

    # 第三步：正则提取第一个完整 JSON 数组
    match = re.search(r'\[.*?\]', stripped, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法解析模型输出，原始输出：{raw_text[:200]}")
    return None


def _validate_conflict_item(item: dict) -> bool:
    """
    校验单条冲突对象的字段完整性。

    参数：
        item — _parse_conflict_json() 返回列表中的单个字典

    返回：
        bool — True 表示校验通过，False 表示应丢弃
    """
    for key in ("field", "old_content", "new_content", "conflict_desc"):
        if not str(item.get(key, "")).strip():
            logger.warning(f"冲突条目缺少字段 {key!r}，已丢弃")
            return False

    if item["field"] not in VALID_FIELD_KEYS:
        logger.warning(f"field 值 {item['field']!r} 不合法，已丢弃")
        return False

    return True


# =============================================================================
# 对外接口
# =============================================================================

def check_conflicts(l1_id: int) -> int | None:
    """
    对指定的 L1 摘要执行冲突检测，将发现的冲突写入 conflict_queue 表。
    由 summarizer.generate_l1_summary() 在 L1 写入后调用。

    参数：
        l1_id — 刚生成的 L1 记录 id

    返回：
        int  — 本次检测写入的冲突条数（0 表示无冲突或检测跳过）
        None — 模型调用失败时返回 None
    """
    logger.info(f"开始对 L1 {l1_id} 执行冲突检测")

    # 第一步：按主键读取 L1（直接查，不用最新记录，避免并发竞态）
    l1_row = get_l1_by_id(l1_id)
    if l1_row is None:
        logger.warning(f"找不到 L1 id={l1_id}，跳过冲突检测")
        return 0

    l1_summary = l1_row["summary"]
    logger.debug(f"读取 L1 摘要：{l1_summary[:80]}")

    # 第二步：读取当前生效的用户画像
    profile_dict = get_current_profile()
    if not profile_dict:
        logger.debug("用户画像为空，跳过冲突检测（冷启动阶段）")
        return 0

    # 第三步：格式化内容，填入 Prompt
    profile_text = _format_profile_for_check(profile_dict)
    prompt = CONFLICT_CHECK_PROMPT.format(
        profile_text = profile_text,
        l1_summary   = l1_summary,
    )

    # 第四步：调用本地模型
    raw_output = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="conflict_checker",
    )
    if raw_output is None:
        logger.error("模型调用失败，冲突检测中止")
        return None

    logger.debug(f"模型原始输出：{raw_output[:200]}")

    # 第五步：解析冲突列表
    conflicts = _parse_conflict_json(raw_output)
    if conflicts is None:
        logger.error("冲突列表解析失败，本次检测结果丢弃")
        return 0

    if not conflicts:
        logger.debug("未检测到冲突")
        return 0

    logger.info(f"检测到 {len(conflicts)} 条潜在冲突，开始写入")

    # 第六步：校验并写入每条冲突
    written = 0
    for item in conflicts:
        if not _validate_conflict_item(item):
            continue

        conflict_id = save_conflict(
            source_l1_id  = l1_id,
            field         = item["field"],
            old_content   = item["old_content"].strip(),
            new_content   = item["new_content"].strip(),
            conflict_desc = item["conflict_desc"].strip(),
        )
        logger.info(f"冲突已写入 id={conflict_id}，field={item['field']}")
        written += 1

    logger.info(f"本次共写入 {written} 条冲突")
    return written


def get_conflict_question() -> tuple[int, str] | None:
    """
    查询是否有待确认的冲突，有则返回第一条冲突的 id 和询问文本。
    由 main.py 的 /chat 路由在构建助手回复之前调用。

    每次只取最早的一条 pending 冲突，避免一次性抛出多个问题。

    返回：
        (int, str) — (conflict_id, 询问文本)
        None       — 没有待确认冲突时返回 None
    """
    pending = get_pending_conflicts()
    if not pending:
        return None

    conflict  = pending[0]
    question  = CONFLICT_ASK_TEMPLATE.format(
        conflict_desc=conflict["conflict_desc"]
    )
    return conflict["id"], question


def handle_conflict_reply(conflict_id: int, action: str) -> str:
    """
    处理用户对冲突询问的回复。

    参数：
        conflict_id — conflict_queue 表的记录 id
        action      — "resolve"（接受新内容）或 "ignore"（保留现状）

    返回：
        str — 操作结果的简短描述，供 main.py 拼入助手回复文本
    """
    pending = get_pending_conflicts()
    target  = next((c for c in pending if c["id"] == conflict_id), None)

    if target is None:
        logger.warning(f"找不到 conflict_id={conflict_id}，可能已处理")
        return "这条记录好像已经处理过了。"

    if target.get("conflict_type") == "alias_confirm":
        from ramaria.storage.database import (
            confirm_alias,
            get_alias_kp_ids_from_conflict,
            reject_alias,
        )

        kp_ids = get_alias_kp_ids_from_conflict(conflict_id)
        if kp_ids is None:
            logger.warning(
                f"alias_confirm conflict_id={conflict_id} 解析 kp_ids 失败"
            )
            return "这条记录解析出问题了，我先跳过，不影响其他功能。"

        new_word_kp_id, canonical_kp_id = kp_ids

        old_content = target["old_content"]
        old_word    = old_content.split("\n__kp_ids__")[0]
        new_word    = target["new_content"]

        if action == "resolve":
            ok = confirm_alias(new_word_kp_id, canonical_kp_id)
            resolve_conflict(conflict_id)
            if ok:
                logger.info(
                    f"alias_confirm {conflict_id} 已确认合并："
                    f"'{new_word}' → '{old_word}'"
                )
                return (
                    f'好的，我已经把"{new_word}"和"{old_word}"识别为同一个东西了，'
                    f'以后会统一用"{old_word}"来记。'
                )
            else:
                return "合并的时候出了点问题，我记录下来，不影响继续对话。"

        elif action == "ignore":
            reject_alias(new_word_kp_id)
            ignore_conflict(conflict_id)
            logger.info(
                f"alias_confirm {conflict_id} 已拒绝合并："
                f"'{new_word}' 独立为规范词"
            )
            return (
                f'好的，"{new_word}"和"{old_word}"我会分开记，不混在一起。'
            )

    field_label = _FIELD_LABELS.get(target["field"], target["field"])

    if action == "resolve":
        update_profile_field(
            field        = target["field"],
            new_content  = target["new_content"],
            source_l1_id = target["source_l1_id"],
        )
        resolve_conflict(conflict_id)
        logger.info(
            f"冲突 {conflict_id} 已 resolve，画像 {target['field']} 已更新"
        )
        return f'好的，我已经把"{field_label}"更新成新的情况了。'

    if action == "ignore":
        ignore_conflict(conflict_id)
        logger.info(f"冲突 {conflict_id} 已 ignore，画像保持不变")
        return (
            f'好的，我会继续记住之前的"{field_label}"，'
            f'这次的变化先不更新。'
        )

    logger.warning(f"未知 action {action!r}，不做任何操作")
    return '没有理解你的选择，可以回复"更新"或"忽略"。'
