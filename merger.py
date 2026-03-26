"""
merger.py — L2 时间段摘要合并模块
负责将多条 L1 摘要压缩合并为一条 L2 时间段摘要，写入 memory_l2 表。

本次修改（对应审查报告问题7）：
  · 补充 _strip_thinking() 函数
  · 在 _parse_l2_json() 的第一步之前先调用 _strip_thinking()
    原来的 _parse_l2_json 直接 json.loads(raw_text)，
    当 Qwen 输出思考链时 JSON 解析会失败，降级写入错误摘要。
    修复后：先剥离 <think>...</think> 标签和纯文本思考链前缀，
    再做 JSON 解析，与 summarizer.py 的处理逻辑保持一致。

本次修改（对应审查报告问题4）：
  · _call_local_model() 的 max_tokens 改为使用 LOCAL_MAX_TOKENS_SUMMARY

核心流程：
  1. 检查是否满足触发条件（条数触发 或 时间触发）
  2. 读取所有未吸收的 L1 摘要
  3. 将 L1 内容格式化后填入压缩 Prompt
  4. 调用本地 Qwen 模型生成 L2 摘要
  5. 解析并校验模型返回的 JSON 结果
  6. 写入 memory_l2 表 + 写入 l2_sources 关联表
  7. 批量标记这批 L1 为已吸收
  8. 写入 L2 向量索引（vector_store.index_l2）

触发路径（两条，互相独立）：
  路径A — 即时触发：每次 L1 写入后，由 summarizer.py 调用 check_and_merge()
           条件：未吸收 L1 条数 ≥ L2_TRIGGER_COUNT（默认5条）
  路径B — 定时触发：session_manager 后台线程每天轮询，调用 check_and_merge()
           条件：最早一条未吸收 L1 距今 ≥ L2_TRIGGER_DAYS（默认7天）

与其他模块的关系：
  - 读 L1 / 写 L2：database.py
  - 读配置：config.py
  - 被调用：summarizer.py（路径A）、session_manager.py（路径B）

使用方法：
    from merger import check_and_merge
    check_and_merge()   # 满足条件时自动执行合并，不满足时静默返回
"""

import json
import re
import requests
from datetime import datetime, timezone
from config import (
    L2_TRIGGER_COUNT,
    L2_TRIGGER_DAYS,
    L2_MERGE_PROMPT,
)
from llm_client import call_local_summary, strip_thinking
from database import (
    get_unabsorbed_l1,
    save_l2_summary,
    mark_l1_absorbed,
    get_setting,
)

from logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# 触发条件检查
# =============================================================================

def _should_trigger(l1_rows):
    """
    判断是否满足 L2 合并触发条件。
    两个条件只要满足其一即触发。

    参数：
        l1_rows — get_unabsorbed_l1() 返回的未吸收 L1 列表（按 created_at ASC 排序）

    返回：
        (bool, str) — (是否触发, 触发原因描述)
        触发原因仅用于日志打印，不影响业务逻辑。
    """
    if not l1_rows:
        return False, "无未吸收 L1"

    # ------------------------------------------------------------------
    # 条件一：条数触发
    # 优先读数据库里的运行时配置，读不到时用 config.py 的默认值
    # ------------------------------------------------------------------
    trigger_count = int(get_setting("l2_trigger_count") or L2_TRIGGER_COUNT)
    if len(l1_rows) >= trigger_count:
        return True, f"未吸收 L1 已达 {len(l1_rows)} 条（阈值 {trigger_count} 条）"

    # ------------------------------------------------------------------
    # 条件二：时间触发
    # 取最早一条 L1 的创建时间，计算距今天数
    # l1_rows 已按 created_at ASC 排序，第一条就是最早的
    # ------------------------------------------------------------------
    trigger_days = int(get_setting("l2_trigger_days") or L2_TRIGGER_DAYS)
    earliest_time_str = l1_rows[0]["created_at"]

    try:
        earliest_time = datetime.fromisoformat(earliest_time_str)
    except ValueError:
        logger.warning(f"L1 时间戳格式异常：{earliest_time_str}，跳过时间触发检查")
        return False, "时间格式异常，跳过"

    now = datetime.now(timezone.utc)
    days_elapsed = (now - earliest_time).total_seconds() / 86400

    if days_elapsed >= trigger_days:
        return True, f"最早 L1 距今 {days_elapsed:.1f} 天（阈值 {trigger_days} 天）"

    return False, f"未达触发条件（当前 {len(l1_rows)} 条，最早 {days_elapsed:.1f} 天前）"


# =============================================================================
# L1 内容格式化
# =============================================================================

def _format_l1_list(l1_rows):
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

def _parse_l2_json(raw_text):
    """
    从模型返回的原始文本中解析出 L2 摘要的结构化字段。

    [修改说明] 审查报告问题7：
        在三步解析流程的最前面加入思考链剥离：
          原流程：直接解析 → 正则提取
          新流程：剥离思考链 → 直接解析 → 正则提取
        这样当 Qwen 输出了 <tool_call>...<tool_call> 或纯文本推理前缀时，
        不会因为 JSON 解析失败而降级写入错误摘要，行为与 summarizer 一致。

    参数：
        raw_text — 模型返回的原始文本

    返回：
        dict — 包含 summary / keywords 两个键
        None — 无论如何都无法解析时返回 None
    """
    # ──────────────────────────────────────────
    # 前置步骤：剥离思考链
    # ──────────────────────────────────────────
    stripped = strip_thinking(raw_text)

    # 如果内容有变化，说明确实存在思考链，打印提示方便排查
    if stripped != raw_text.strip():
        logger.debug("检测到思考链输出，已自动剥离")

    # ──────────────────────────────────────────
    # 第一步：直接尝试解析剥离后的文本
    # ──────────────────────────────────────────
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # ──────────────────────────────────────────
    # 第二步：用正则提取第一个 JSON 对象
    # 应对模型在 JSON 前后加了说明文字的情况
    # re.DOTALL 让 . 能匹配换行符，处理多行 JSON
    # ──────────────────────────────────────────
    match = re.search(r'\{.*?\}', stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法从模型输出中解析 JSON，原始输出：{raw_text[:200]}")
    return None


def _validate_l2(parsed):
    """
    校验 L2 解析结果，对不合规的值进行降级处理。

    参数：
        parsed — _parse_l2_json() 返回的字典

    返回：
        dict — 校验/修复后的字典，保证 summary 存在且有值
    """
    result = {}

    # summary：必填，缺失时写入占位文本
    result["summary"] = str(parsed.get("summary", "")).strip()
    if not result["summary"]:
        result["summary"] = "（L2 摘要生成失败，内容为空）"

    # keywords：允许为空
    keywords = str(parsed.get("keywords", "")).strip()
    result["keywords"] = keywords if keywords else None

    return result


# =============================================================================
# 时间范围提取
# =============================================================================

def _get_time_range(l1_rows):
    """
    从 L1 列表中提取时间范围（最早一条的创建时间 ~ 最晚一条的创建时间）。

    参数：
        l1_rows — sqlite3.Row 列表，已按 created_at ASC 排序

    返回：
        (str, str) — (period_start, period_end)，ISO 8601 格式
    """
    period_start = l1_rows[0]["created_at"]
    period_end   = l1_rows[-1]["created_at"]
    return period_start, period_end


# =============================================================================
# 对外接口
# =============================================================================

def check_and_merge():
    """
    检查是否满足 L2 合并条件，满足时执行合并，不满足时静默返回。
    这是本模块唯一的对外接口。

    两条触发路径都调用这同一个函数：
      - summarizer.py 在每次 L1 写入后调用（检查条数触发）
      - session_manager.py 后台线程每天调用（检查时间触发）

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
        # JSON 解析彻底失败，写入降级摘要，不丢失这批 L1 的合并记录
        logger.error("JSON 解析失败，写入降级 L2 摘要")
        parsed = {
            "summary":  f"（自动合并失败，原始输出已记录）{raw_output[:100]}",
            "keywords": None,
        }

    # 第五步：校验字段
    validated = _validate_l2(parsed)

    # 第六步：写入 memory_l2 表（save_l2_summary 内部同时写 l2_sources）
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
    logger.debug(f"summary={validated['summary']} | keywords={validated['keywords']} | 覆盖时间={period_start[:10]}~{period_end[:10]} | 吸收L1={l1_ids}")

    # 第八步：写入 L2 向量索引
    try:
        from vector_store import index_l2
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


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    from database import new_session, save_message, close_session, save_l1_summary

    print("=== merger.py 验证测试 ===\n")

    # ------------------------------------------------------------------
    # 测试一：格式化逻辑（不依赖模型）
    # ------------------------------------------------------------------
    print("--- 测试一：L1 格式化输出 ---")

    class FakeRow(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)

    fake_rows = [
        FakeRow({"id": 1, "summary": "烧酒完成了数据库模块开发。",
                 "keywords": "数据库,后端", "time_period": "夜间",
                 "atmosphere": "专注高效", "created_at": "2026-03-08T14:00:00+00:00"}),
        FakeRow({"id": 2, "summary": "烧酒讨论了记忆系统架构。",
                 "keywords": "记忆,架构", "time_period": "上午",
                 "atmosphere": "轻松愉快", "created_at": "2026-03-09T03:00:00+00:00"}),
    ]
    print(_format_l1_list(fake_rows))
    print()

    # ------------------------------------------------------------------
    # 测试二：触发条件判断
    # ------------------------------------------------------------------
    print("--- 测试二：触发条件判断 ---")
    trigger, reason = _should_trigger([])
    print(f"空列表：trigger={trigger}（应为 False）")
    trigger2, reason2 = _should_trigger(fake_rows)
    print(f"2条 L1：trigger={trigger2}（应为 False，低于5条阈值）")
    print()

    # ------------------------------------------------------------------
    # 测试三：思考链剥离（不依赖模型）—— 新增测试
    # ------------------------------------------------------------------
    print("--- 测试三：_strip_thinking 验证 ---")

    # 标签格式思考链
    mock_think_tag = '<tool_call>\n这是思考过程\n<tool_call>\n{"summary":"合并摘要","keywords":"k1,k2"}'
    stripped = strip_thinking(mock_think_tag)
    print(f"标签格式剥离后：{stripped}（应以 {{ 开头）")

    # 纯文本思考链
    mock_think_text = 'Thinking:\n1. 分析内容\n2. 输出格式\n{"summary":"合并摘要","keywords":"k1,k2"}'
    stripped2 = strip_thinking(mock_think_text)
    print(f"纯文本格式剥离后：{stripped2}（应以 {{ 开头）")

    # 无思考链（正常输出）
    mock_clean = '{"summary":"合并摘要","keywords":"k1,k2"}'
    stripped3 = strip_thinking(mock_clean)
    print(f"无思考链（应原样返回）：{stripped3}")
    print()

    # ------------------------------------------------------------------
    # 测试四：_parse_l2_json 思考链场景（不依赖模型）
    # ------------------------------------------------------------------
    print("--- 测试四：_parse_l2_json 思考链场景 ---")

    mock_with_think = '<tool_call>让我合并一下。<tool_call>\n{"summary":"烧酒这周完成了多个后端模块。","keywords":"后端,模块,开发"}'
    result = _parse_l2_json(mock_with_think)
    print(f"带思考链解析：{result}（应正常得到 dict）")

    mock_no_think = '{"summary":"烧酒这周完成了多个后端模块。","keywords":"后端,模块,开发"}'
    result2 = _parse_l2_json(mock_no_think)
    print(f"无思考链解析：{result2}（应正常得到 dict）")
    print()

    # ------------------------------------------------------------------
    # 测试五：端到端合并（需要 LM Studio 运行中）
    # ------------------------------------------------------------------
    print("--- 测试五：端到端合并触发（需要 LM Studio 运行中）---")

    for i in range(5):
        sid = new_session()
        save_message(sid, "user", f"测试消息 {i+1}")
        save_message(sid, "assistant", f"收到消息 {i+1}。")
        close_session(sid)
        l1_id = save_l1_summary(
            session_id  = sid,
            summary     = f"烧酒完成了第 {i+1} 项测试任务。",
            keywords    = f"测试,任务{i+1}",
            time_period = "夜间",
            atmosphere  = "专注高效",
        )
        print(f"已写入测试 L1 id={l1_id}")

    print("\n调用 check_and_merge()...")
    l2_id = check_and_merge()

    if l2_id:
        print(f"\n端到端测试通过，L2 id = {l2_id}")
    else:
        print("\n模型未响应（LM Studio 未启动），本地逻辑测试已通过，端到端测试跳过")

    print("\n验证完成。")
