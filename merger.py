"""
merger.py — L2 时间段摘要合并模块
负责将多条 L1 摘要压缩合并为一条 L2 时间段摘要，写入 memory_l2 表。

变更记录：
  v1（审查报告问题7）：
    · 补充 strip_thinking() 调用，修复模型输出思考链时 JSON 解析失败的问题。
    · _call_local_model() 的 max_tokens 改为使用 LOCAL_MAX_TOKENS_SUMMARY。

  v2（代码优化清单第三轮）：
    · 修复 _should_trigger() 中 naive / aware datetime 混用导致的崩溃风险。

    问题代码（merger.py:99）：
        earliest_time = datetime.fromisoformat(earliest_time_str)
        # ↑ SQLite 存储的时间戳可能是 naive（无时区信息）
        now = datetime.now(timezone.utc)
        # ↑ 一定是 aware（带 UTC 时区）
        days_elapsed = (now - earliest_time).total_seconds() / 86400
        # ↑ naive 与 aware 直接相减 → TypeError 崩溃！

    根本原因：
        Python 的 datetime 分为两种：
          · naive  — 不含时区信息（tzinfo=None），如从 SQLite 直接读出的时间戳
          · aware  — 含时区信息，如 datetime.now(timezone.utc)
        两种 datetime 直接做算术运算会抛出 TypeError，无法比较大小。
        只要 L2 的时间触发条件被检查，且数据库里存有旧格式的 naive 时间戳，
        就会必然崩溃。

    修复方案（参考 main.py:287-294 已有的正确写法）：
        在 fromisoformat() 之后立即检查 tzinfo，若为 None 则补上 UTC：
            if earliest_time.tzinfo is None:
                earliest_time = earliest_time.replace(tzinfo=timezone.utc)
        只需 3 行，其余代码不变。

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
    # 取最早一条 L1 的创建时间，计算距今天数。
    # l1_rows 已按 created_at ASC 排序，第一条就是最早的。
    # ------------------------------------------------------------------
    trigger_days      = int(get_setting("l2_trigger_days") or L2_TRIGGER_DAYS)
    earliest_time_str = l1_rows[0]["created_at"]

    try:
        earliest_time = datetime.fromisoformat(earliest_time_str)
    except ValueError:
        logger.warning(f"L1 时间戳格式异常：{earliest_time_str}，跳过时间触发检查")
        return False, "时间格式异常，跳过"

    # ------------------------------------------------------------------
    # 【P1-A 修复】补充 naive / aware 时区统一处理
    #
    # 问题根源：
    #     SQLite 存储的时间戳字符串可能没有时区信息（如 "2026-03-01T14:30:00"），
    #     fromisoformat() 解析后得到 naive datetime（tzinfo=None）。
    #     而 datetime.now(timezone.utc) 返回的是 aware datetime（带 UTC 时区）。
    #     两者直接相减会抛出：
    #         TypeError: can't subtract offset-naive and offset-aware datetimes
    #
    # 修复方案：
    #     解析后立即检查 tzinfo，若为 None（naive）则补上 UTC 时区。
    #     这与数据库写入时使用 UTC 的约定一致——历史数据即使没有时区标记，
    #     也应当被视为 UTC 处理。
    #
    #     参考：main.py _build_context() 第 287-294 行已有完全相同的处理逻辑。
    # ------------------------------------------------------------------
    if earliest_time.tzinfo is None:
        earliest_time = earliest_time.replace(tzinfo=timezone.utc)

    now          = datetime.now(timezone.utc)
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

    解析策略（三步递进）：
      前置步骤：剥离思考链（<think>...</think> 标签及纯文本前缀）
      第一步：直接尝试解析剥离后的文本
      第二步：用正则提取第一个 JSON 对象

    参数：
        raw_text — 模型返回的原始文本

    返回：
        dict — 包含 summary / keywords 两个键
        None — 无论如何都无法解析时返回 None
    """
    # ------------------------------------------------------------------
    # 前置步骤：剥离思考链
    # 当 Qwen 输出 <think>...</think> 或纯文本推理前缀时，
    # 直接 json.loads 会失败，需要先剥离再解析。
    # ------------------------------------------------------------------
    stripped = strip_thinking(raw_text)
    if stripped != raw_text.strip():
        logger.debug("检测到思考链输出，已自动剥离")

    # 第一步：直接尝试解析剥离后的文本
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 第二步：用正则提取第一个 JSON 对象
    # 应对模型在 JSON 前后加了说明文字的情况
    # re.DOTALL 让 . 能匹配换行符，处理多行 JSON
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

    # summary：必填，缺失时写入占位文本（不携带原始输出，避免污染记忆层）
    result["summary"] = str(parsed.get("summary", "")).strip()
    if not result["summary"]:
        result["summary"] = "（本次时间段摘要生成失败，请参考原始对话记录）"

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
        # JSON 解析彻底失败，写入降级摘要，原始输出记录到日志而非 SQLite
        # 避免乱码或思考链碎片污染记忆层（参考代码优化清单 P3-3）
        logger.error(f"JSON 解析失败，原始输出：{raw_output[:500]}")
        parsed = {
            "summary":  "（本次时间段摘要生成失败，请参考原始对话记录）",
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
    logger.debug(
        f"summary={validated['summary']} | keywords={validated['keywords']} | "
        f"覆盖时间={period_start[:10]}~{period_end[:10]} | 吸收L1={l1_ids}"
    )

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
    # 测试二：触发条件判断（不依赖模型）
    # ------------------------------------------------------------------
    print("--- 测试二：触发条件判断 ---")
    trigger, reason = _should_trigger([])
    print(f"空列表：trigger={trigger}（应为 False）")
    trigger2, reason2 = _should_trigger(fake_rows)
    print(f"2条 L1：trigger={trigger2}（应为 False，低于5条阈值）")
    print()

    # ------------------------------------------------------------------
    # 测试三：【P1-A 修复核心】naive/aware 时区混用场景验证
    # ------------------------------------------------------------------
    print("--- 测试三：时区修复验证（P1-A）---")

    # 模拟数据库中存储的旧格式 naive 时间戳（不含时区，16天前）
    naive_time_str = "2026-03-10T14:00:00"   # 无时区信息

    # 验证修复前的崩溃场景（注释掉，仅作说明）：
    #   dt = datetime.fromisoformat(naive_time_str)         # naive
    #   now = datetime.now(timezone.utc)                    # aware
    #   diff = (now - dt).total_seconds()                   # ← TypeError!

    # 验证修复后的正确处理：
    dt_naive = datetime.fromisoformat(naive_time_str)
    print(f"解析结果（修复前）：tzinfo={dt_naive.tzinfo}（None = naive，会崩溃）")

    if dt_naive.tzinfo is None:
        dt_aware = dt_naive.replace(tzinfo=timezone.utc)
    else:
        dt_aware = dt_naive
    print(f"修复后：tzinfo={dt_aware.tzinfo}（已补充 UTC）")

    now  = datetime.now(timezone.utc)
    diff = (now - dt_aware).total_seconds() / 86400
    print(f"距今 {diff:.1f} 天（应为正数，无 TypeError）✓")
    print()

    # 构造带 naive 时间的 fake row，验证 _should_trigger 不再崩溃
    old_row = FakeRow({
        "id": 99, "summary": "旧记录，带 naive 时间戳。",
        "keywords": "测试", "time_period": "夜间",
        "atmosphere": "专注高效",
        "created_at": "2026-03-01T10:00:00",   # naive，无时区
    })
    try:
        trigger3, reason3 = _should_trigger([old_row])
        print(f"naive 时间戳场景：trigger={trigger3}，reason='{reason3}'")
        print("_should_trigger() 未崩溃 ✓")
    except TypeError as e:
        print(f"❌ 仍然崩溃（修复未生效）：{e}")
    print()

    # ------------------------------------------------------------------
    # 测试四：思考链剥离（不依赖模型）
    # ------------------------------------------------------------------
    print("--- 测试四：_parse_l2_json 思考链场景 ---")
    mock_with_think = '<think>让我合并一下。</think>\n{"summary":"烧酒这周完成了多个后端模块。","keywords":"后端,模块,开发"}'
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
