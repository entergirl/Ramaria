"""
src/ramaria/memory/merger.py — L2 时间段摘要合并模块

负责将多条 L1 摘要压缩合并为一条 L2 时间段摘要，写入 memory_l2 表。

触发路径（两条，互相独立）：
    路径A — 即时触发：每次 L1 写入后，由 summarizer.py 调用 check_and_merge()
             条件：未吸收 L1 条数 ≥ L2_TRIGGER_COUNT（默认5条）
    路径B — 定时触发：session_manager 后台线程每天轮询，调用 check_and_merge()
             条件：最早一条未吸收 L1 距今 ≥ L2_TRIGGER_DAYS（默认7天）

"""

import json
import re
from datetime import datetime, timezone

from ramaria.config import (
    L2_TRIGGER_COUNT,
    L2_TRIGGER_DAYS,
)
from ramaria.core.llm_client import call_local_summary, strip_thinking
from ramaria.storage.database import (
    get_setting,
    get_unabsorbed_l1,
    mark_l1_absorbed,
    save_l2_summary,
)

from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Prompt 模板
# =============================================================================
#
# 从 config.py 迁移到此处，由本模块统一维护。

L2_MERGE_PROMPT = """你是一个记忆压缩助手。下面是一段时间内的多条对话摘要（L1），请将它们压缩合并为一条时间段摘要（L2）。

【输出格式要求】
严格按照以下 JSON 格式输出，不要输出任何其他内容（不要加 markdown 代码块，不要加说明文字）：

{{
  "summary": "第三人称客观描述这段时间内的主要活动与规律，两到三句话，不超过120字",
  "keywords": "5到8个名词标签，用英文逗号分隔，覆盖这段时间的核心主题，避免语义重复"
}}

【字段说明】
- summary：提炼规律性信息和重要事件，用"烧酒"指代用户；保留里程碑事件，忽略琐碎细节；客观陈述
- keywords：取各条 L1 关键词的并集，去除重复和近义词，按重要性排序

【待合并的 L1 摘要列表】
{l1_summaries}
"""


# =============================================================================
# 触发条件检查
# =============================================================================

def _should_trigger(l1_rows) -> tuple[bool, str]:
    """
    判断是否满足 L2 合并触发条件。两个条件只要满足其一即触发。

    参数：
        l1_rows — get_unabsorbed_l1() 返回的未吸收 L1 列表（按 created_at ASC）

    返回：
        (bool, str) — (是否触发, 触发原因描述)
    """
    if not l1_rows:
        return False, "无未吸收 L1"

    # 条件一：条数触发
    trigger_count = int(get_setting("l2_trigger_count") or L2_TRIGGER_COUNT)
    if len(l1_rows) >= trigger_count:
        return True, (
            f"未吸收 L1 已达 {len(l1_rows)} 条（阈值 {trigger_count} 条）"
        )

    # 条件二：时间触发
    # l1_rows 已按 created_at ASC 排序，第一条就是最早的
    trigger_days      = int(get_setting("l2_trigger_days") or L2_TRIGGER_DAYS)
    earliest_time_str = l1_rows[0]["created_at"]

    try:
        earliest_time = datetime.fromisoformat(earliest_time_str)
    except ValueError:
        logger.warning(f"L1 时间戳格式异常：{earliest_time_str}，跳过时间触发检查")
        return False, "时间格式异常，跳过"

    # naive / aware 统一处理：SQLite 存储的时间可能不含时区信息
    if earliest_time.tzinfo is None:
        earliest_time = earliest_time.replace(tzinfo=timezone.utc)

    now          = datetime.now(timezone.utc)
    days_elapsed = (now - earliest_time).total_seconds() / 86400

    if days_elapsed >= trigger_days:
        return True, (
            f"最早 L1 距今 {days_elapsed:.1f} 天（阈值 {trigger_days} 天）"
        )

    return False, (
        f"未达触发条件（当前 {len(l1_rows)} 条，"
        f"最早 {days_elapsed:.1f} 天前）"
    )


# =============================================================================
# L1 内容格式化
# =============================================================================

def _format_l1_list(l1_rows) -> str:
    """
    将 L1 列表格式化为适合填入 Prompt 的纯文本块。

    每条 L1 格式化为：
        [序号] 时间段 | 氛围：xxx
        摘要：xxx
        关键词：xxx

    参数：
        l1_rows — sqlite3.Row 列表

    返回：
        str — 格式化后的多行文本
    """
    lines = []
    for i, row in enumerate(l1_rows, start=1):
        time_period = row["time_period"] or "未知时段"
        atmosphere  = row["atmosphere"]  or "未记录"
        keywords    = row["keywords"]    or "无"

        lines.append(f"[{i}] {time_period} | 氛围：{atmosphere}")
        lines.append(f"    摘要：{row['summary']}")
        lines.append(f"    关键词：{keywords}")
        lines.append("")
    return "\n".join(lines).strip()


# =============================================================================
# 解析与校验模型输出
# =============================================================================

def _parse_l2_json(raw_text: str) -> dict | None:
    """
    从模型返回的原始文本中解析出 L2 摘要的结构化字段。

    解析策略：
        前置步骤：剥离思考链
        第一步：直接尝试解析剥离后的文本
        第二步：用正则提取第一个 JSON 对象

    返回：
        dict — 包含 summary / keywords 两个键
        None — 无论如何都无法解析时返回 None
    """
    # 前置步骤：剥离思考链
    stripped = strip_thinking(raw_text)
    if stripped != raw_text.strip():
        logger.debug("检测到思考链输出，已自动剥离")

    # 第一步：直接尝试解析
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 第二步：正则提取第一个 JSON 对象
    match = re.search(r'\{.*?\}', stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法从模型输出中解析 JSON，原始输出：{raw_text[:200]}")
    return None


def _validate_l2(parsed: dict) -> dict:
    """
    校验 L2 解析结果，对不合规的值进行降级处理。

    参数：
        parsed — _parse_l2_json() 返回的字典

    返回：
        dict — 校验/修复后的字典，保证 summary 存在且有值
    """
    result = {}

    result["summary"] = str(parsed.get("summary", "")).strip()
    if not result["summary"]:
        result["summary"] = "（本次时间段摘要生成失败，请参考原始对话记录）"

    keywords = str(parsed.get("keywords", "")).strip()
    result["keywords"] = keywords if keywords else None

    return result


# =============================================================================
# 时间范围提取
# =============================================================================

def _get_time_range(l1_rows) -> tuple[str, str]:
    """
    从 L1 列表中提取时间范围。

    参数：
        l1_rows — sqlite3.Row 列表，已按 created_at ASC 排序

    返回：
        (period_start, period_end) — ISO 8601 格式
    """
    return l1_rows[0]["created_at"], l1_rows[-1]["created_at"]


# =============================================================================
# 对外接口
# =============================================================================

def check_and_merge() -> int | None:
    """
    检查是否满足 L2 合并条件，满足时执行合并，不满足时静默返回。
    这是本模块唯一的对外接口。

    完整流程：
        1. 读取所有未吸收 L1
        2. 判断是否触发
        3. 格式化 L1 内容，填入 Prompt
        4. 调用本地模型生成 L2 摘要
        5. 解析 JSON，校验字段
        6. 写入 memory_l2 表 + l2_sources 关联表
        7. 批量标记 L1 为已吸收
        8. 写入 L2 向量索引

    返回：
        int  — 成功时返回新写入的 L2 记录 id
        None — 未触发或任何步骤失败时返回 None
    """
    logger.debug("开始检查 L2 合并触发条件")

    # 第一步：读取所有未吸收 L1
    l1_rows = get_unabsorbed_l1()
    should_trigger, reason = _should_trigger(l1_rows)
    logger.debug(f"触发判断：{reason}")

    if not should_trigger:
        return None

    logger.info(f"触发 L2 合并，共 {len(l1_rows)} 条 L1 待合并")

    # 第二步：格式化 L1 内容，填入 Prompt
    l1_text = _format_l1_list(l1_rows)
    prompt  = L2_MERGE_PROMPT.format(l1_summaries=l1_text)

    # 第三步：调用本地模型
    raw_output = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="merger",
    )
    if raw_output is None:
        logger.error("模型调用失败，L2 合并中止")
        return None

    logger.debug(f"模型原始输出：{raw_output[:200]}")

    # 第四步：解析 JSON（已包含思考链剥离）
    parsed = _parse_l2_json(raw_output)
    if parsed is None:
        logger.error(f"JSON 解析失败，原始输出：{raw_output[:500]}")
        parsed = {
            "summary":  "（本次时间段摘要生成失败，请参考原始对话记录）",
            "keywords": None,
        }

    # 第五步：校验字段
    validated = _validate_l2(parsed)

    # 第六步：写入 memory_l2 表（内部同时写 l2_sources）
    period_start, period_end = _get_time_range(l1_rows)
    l1_ids = [row["id"] for row in l1_rows]

    l2_id = save_l2_summary(
        summary      = validated["summary"],
        keywords     = validated["keywords"],
        period_start = period_start,
        period_end   = period_end,
        l1_ids       = l1_ids,
    )

    # 第七步：批量标记 L1 为已吸收
    mark_l1_absorbed(l1_ids)

    logger.info(f"L2 摘要已写入，id = {l2_id}")
    logger.debug(
        f"summary={validated['summary']} | "
        f"keywords={validated['keywords']} | "
        f"覆盖时间={period_start[:10]}~{period_end[:10]} | "
        f"吸收L1={l1_ids}"
    )

    # 第八步：写入 L2 向量索引
    try:
        from ramaria.storage.vector_store import index_l2
        index_l2(
            l2_id        = l2_id,
            summary      = validated["summary"],
            keywords     = validated["keywords"],
            period_start = period_start,
            period_end   = period_end,
        )
    except Exception as e:
        logger.warning(f"L2 向量索引写入失败 — {e}")

    return l2_id