"""
src/ramaria/memory/profile_manager.py — L3 画像半自动维护模块

设计理念：
    不打扰用户，静默补充画像；发现矛盾时才开口询问。
    像真实朋友一样——默默记住你说的话，只在前后说的不一样时才自然提起。

核心流程：
    触发时机：summarizer.generate_l1_summary() 写入 L1 后，
              在 conflict_checker 之前调用 extract_and_update()

    Step 1  读取刚生成的 L1 摘要内容（直接按主键查）
    Step 2  读取当前生效的 L3 用户画像
    Step 3  调用本地模型，判断 L1 里哪些是"新信息"
    Step 4  解析模型返回的 JSON，校验字段合法性
    Step 5  逐字段追加写入 user_profile 表（带时间戳前缀）

时间戳说明：
    每条写入内容都带上本地日期前缀，例如：
        "[2026-03-20] 正在开发珊瑚菌记忆系统，进入阶段二收尾"
    让模型能区分"两个月前的状态"和"最近的状态"。

"""

import json
import re
from datetime import datetime, timezone

from ramaria.core.llm_client import call_local_summary, strip_thinking
from ramaria.storage.database import (
    get_current_profile,
    get_l1_by_id,
    update_profile_field,
)
from constants import PROFILE_FIELDS, VALID_FIELD_KEYS, PROFILE_FIELD_LIST
from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Prompt 模板
# =============================================================================

# 标准版：画像已有内容时使用
EXTRACT_PROMPT = """你是一个用户画像维护助手。请根据最新对话摘要，提取其中值得长期记住的新信息。

【重要规则】
- 只提取"新信息"：L3 画像里没有记录过的内容
- 不提取"冲突信息"：与现有画像矛盾的内容（那部分由其他模块处理）
- 不提取已经完整记录过的细节（避免重复）
- 如果没有任何值得写入的新信息，返回空对象 {{}}
- 宁可少提取，不要滥提取；闲聊、日常问候、一次性事件不需要记录

【field 合法值（六选一）】
- basic_info      基础信息（姓名、年龄、所在地、职业等稳定属性）
- personal_status 近期状态（当前情绪、健康、压力等动态信息）
- interests       兴趣爱好（长期关注的领域、喜好）
- social          社交情况（重要人际关系）
- history         历史事件（重要经历与里程碑）
- recent_context  近期背景（项目进展、阶段性动态）

【输出格式要求】
严格按照以下 JSON 格式输出，不要输出任何其他内容：

{{
  "field名": "这个字段需要新增的内容，简洁客观，用第三人称描述烧酒"
}}

没有任何新信息时，直接输出：
{{}}

【当前 L3 用户画像】
{profile_text}

【最新 L1 摘要】
{l1_summary}
"""

# 冷启动版：画像为空时使用
EXTRACT_PROMPT_COLD_START = """你是一个用户画像维护助手。请根据最新对话摘要，提取其中值得长期记住的信息，初始化用户画像。

【重要规则】
- 只提取有长期价值的信息（稳定属性、兴趣、重要经历等）
- 不提取一次性事件、日常闲聊、临时状态
- 如果没有值得提取的信息，返回空对象 {{}}
- 宁可少提取，不要滥提取

【field 合法值（六选一）】
- basic_info      基础信息（姓名、年龄、所在地、职业等稳定属性）
- personal_status 近期状态（当前情绪、健康、压力等动态信息）
- interests       兴趣爱好（长期关注的领域、喜好）
- social          社交情况（重要人际关系）
- history         历史事件（重要经历与里程碑）
- recent_context  近期背景（项目进展、阶段性动态）

【输出格式要求】
严格按照以下 JSON 格式输出，不要输出任何其他内容：

{{
  "field名": "提取到的内容，简洁客观，用第三人称描述烧酒"
}}

没有值得提取的信息时，直接输出：
{{}}

【最新 L1 摘要】
{l1_summary}
"""


# =============================================================================
# 工具函数
# =============================================================================

def _get_today_str() -> str:
    """
    返回当前本地日期字符串，格式 "2026-03-20"。
    用于给写入内容加时间戳前缀，避免模型混淆历史记忆的时间。
    """
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")


def _add_timestamp(content: str) -> str:
    """
    给画像内容加上今日日期前缀。

    参数：
        content — 原始内容文本，如 "正在开发珊瑚菌记忆系统"

    返回：
        str — 如 "[2026-03-20] 正在开发珊瑚菌记忆系统"
    """
    return f"[{_get_today_str()}] {content}"


def _format_profile_for_prompt(profile_dict: dict) -> str:
    """
    将 L3 用户画像字典格式化为适合填入 Prompt 的文本。
    同时标注英文字段名，方便模型在输出时准确填写 field。

    返回：
        str — 格式化后的文本；画像为空时返回 "（暂无画像记录）"
    """
    if not profile_dict:
        return "（暂无画像记录）"

    lines = []
    for field_key, label in PROFILE_FIELD_LIST:
        content = profile_dict.get(field_key, "").strip()
        if content:
            lines.append(f"[{field_key} {label}] {content}")

    return "\n".join(lines) if lines else "（暂无画像记录）"


# =============================================================================
# 解析模型输出
# =============================================================================

def _parse_extract_json(raw_text: str) -> dict | None:
    """
    从模型返回的原始文本中解析出画像提取结果。

    返回：
        dict — 解析成功时返回字典（可能为空字典 {}）
        None — 三步都失败时返回 None
    """
    # 第一步：直接解析
    try:
        result = json.loads(raw_text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 第二步：剥离思考链后再解析
    stripped = strip_thinking(raw_text)
    if stripped != raw_text.strip():
        logger.debug("检测到思考链输出，已自动剥离")
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 第三步：正则提取第一个完整 JSON 对象
    match = re.search(r'\{.*?\}', stripped, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法解析模型输出，原始输出：{raw_text[:200]}")
    return None


def _validate_extract_result(result_dict: dict) -> dict:
    """
    校验提取结果，过滤掉字段名不合法或内容为空的条目。

    返回：
        dict — 过滤后的合法条目字典；可能为空字典
    """
    validated = {}

    for field, content in result_dict.items():
        if field not in VALID_FIELD_KEYS:
            logger.warning(f"字段名 {field!r} 不合法，已跳过")
            continue

        content = str(content).strip()
        if not content:
            logger.warning(f"字段 {field} 内容为空，已跳过")
            continue

        validated[field] = content

    return validated


# =============================================================================
# 画像写入逻辑
# =============================================================================

def _append_to_profile_field(
    field: str,
    new_content: str,
    source_l1_id: int,
    profile_dict: dict,
) -> int:
    """
    将新内容追加写入指定画像字段。

    追加规则：
        · 字段已有值 → 在末尾用"；"分隔追加新内容
        · 字段无值   → 直接写入

    新内容会自动加上今日时间戳前缀。

    参数：
        field        — 画像字段名，如 "recent_context"
        new_content  — 需要写入的新内容（不含时间戳）
        source_l1_id — 触发此更新的 L1 记录 id
        profile_dict — 当前生效的完整画像字典

    返回：
        int — update_profile_field() 返回的新记录 id
    """
    timestamped_content = _add_timestamp(new_content)
    existing = profile_dict.get(field, "").strip()

    if existing:
        merged_content = f"{existing}；{timestamped_content}"
        logger.info(f"字段 {field} 追加：{timestamped_content[:50]}")
    else:
        merged_content = timestamped_content
        logger.info(f"字段 {field} 新建：{timestamped_content[:50]}")

    return update_profile_field(
        field        = field,
        new_content  = merged_content,
        source_l1_id = source_l1_id,
    )


# =============================================================================
# 对外接口
# =============================================================================

def extract_and_update(l1_id: int) -> int | None:
    """
    从指定 L1 摘要中提取新信息，静默写入 L3 用户画像。
    这是本模块唯一的对外接口，由 summarizer.generate_l1_summary() 调用。

    调用时机：
        L1 写入 SQLite 后，conflict_checker.check_conflicts() 之前。
        顺序很重要——冲突检测需要拿到最新的 L3 才能正确比对。

    参数：
        l1_id — 刚生成的 L1 记录 id

    返回：
        int  — 成功写入的字段数量（0 表示无新信息或跳过）
        None — 模型调用失败时返回 None
    """
    logger.info(f"开始为 L1 {l1_id} 提取画像候选条目")

    # Step 1：按主键读取 L1（直接查，避免并发竞态）
    l1_row = get_l1_by_id(l1_id)
    if l1_row is None:
        logger.warning(f"找不到 L1 id={l1_id}，跳过画像提取")
        return 0

    l1_summary    = l1_row["summary"]
    logger.debug(f"读取 L1 摘要：{l1_summary[:80]}")

    # Step 2：读取当前 L3 画像
    profile_dict  = get_current_profile()
    is_cold_start = not profile_dict

    if is_cold_start:
        logger.debug("画像为空，使用冷启动 Prompt")
    else:
        logger.debug(f"当前画像已有 {len(profile_dict)} 个字段，使用标准 Prompt")

    # Step 3：构建 Prompt，调用本地模型
    if is_cold_start:
        prompt = EXTRACT_PROMPT_COLD_START.format(l1_summary=l1_summary)
    else:
        profile_text = _format_profile_for_prompt(profile_dict)
        prompt = EXTRACT_PROMPT.format(
            profile_text = profile_text,
            l1_summary   = l1_summary,
        )

    raw_output = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="profile_manager",
    )
    if raw_output is None:
        logger.error("模型调用失败，画像提取中止")
        return None

    logger.debug(f"模型原始输出：{raw_output[:200]}")

    # Step 4：解析并校验模型输出
    result = _parse_extract_json(raw_output)
    if result is None:
        logger.error("解析失败，本次画像提取跳过")
        return 0

    if not result:
        logger.debug("L1 中无新画像信息，画像保持不变")
        return 0

    validated = _validate_extract_result(result)
    if not validated:
        logger.debug("校验后无合法条目，画像保持不变")
        return 0

    logger.info(f"提取到 {len(validated)} 条新信息，开始写入")

    # Step 5：逐字段写入
    written = 0
    for field, content in validated.items():
        try:
            _append_to_profile_field(
                field        = field,
                new_content  = content,
                source_l1_id = l1_id,
                profile_dict = profile_dict,
            )
            written += 1
        except Exception as e:
            logger.warning(f"字段 {field} 写入失败 — {e}")

    logger.info(f"画像更新完成，共写入 {written} 个字段")
    return written