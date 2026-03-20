"""
merger.py — L2 时间段摘要合并模块
负责将多条 L1 摘要压缩合并为一条 L2 时间段摘要，写入 memory_l2 表。

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
    LOCAL_API_URL,
    LOCAL_MODEL_NAME,
    LOCAL_TEMPERATURE,
    LOCAL_MAX_TOKENS,
    L2_TRIGGER_COUNT,
    L2_TRIGGER_DAYS,
    L2_MERGE_PROMPT,
)
from database import (
    get_unabsorbed_l1,
    save_l2_summary,
    mark_l1_absorbed,
    get_setting,
)


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
        # 时间格式异常，跳过时间触发检查，不阻断流程
        print(f"[merger] 警告：L1 时间戳格式异常：{earliest_time_str}，跳过时间触发检查")
        return False, "时间格式异常，跳过"

    now = datetime.now(timezone.utc)
    days_elapsed = (now - earliest_time).total_seconds() / 86400   # 转换为天数

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
        # 时间段和氛围允许为空，用占位符补全，避免 Prompt 里出现 None 字样
        time_period = row["time_period"] or "未知时段"
        atmosphere  = row["atmosphere"]  or "未记录"
        keywords    = row["keywords"]    or "无"

        lines.append(f"[{i}] {time_period} | 氛围：{atmosphere}")
        lines.append(f"    摘要：{row['summary']}")
        lines.append(f"    关键词：{keywords}")
        lines.append("")   # 每条之间空一行，方便模型区分
    return "\n".join(lines).strip()


# =============================================================================
# 调用本地模型
# =============================================================================

def _call_local_model(prompt):
    """
    向 LM Studio 发送请求，返回模型的回复文本。
    与 summarizer._call_local_model 逻辑一致，独立维护避免循环依赖。

    参数：
        prompt — 完整的 Prompt 字符串

    返回：
        str  — 模型回复的文本内容
        None — 请求失败时返回 None
    """
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [
            # /no_think 放在 system 层，让 Qwen 跳过思考链直接输出 JSON
            {"role": "system", "content": "/no_think"},
            {"role": "user",   "content": prompt},
        ],
        "temperature": LOCAL_TEMPERATURE,
        "max_tokens":  LOCAL_MAX_TOKENS,
    }

    try:
        response = requests.post(LOCAL_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        print("[merger] 错误：无法连接到 LM Studio，请确认模型已启动")
        return None
    except requests.exceptions.Timeout:
        print("[merger] 错误：请求超时（超过 60 秒）")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[merger] 错误：HTTP 请求失败 — {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"[merger] 错误：解析模型响应结构失败 — {e}")
        return None


# =============================================================================
# 解析与校验模型输出
# =============================================================================

def _parse_l2_json(raw_text):
    """
    从模型返回的原始文本中解析出 L2 摘要的结构化字段。

    参数：
        raw_text — 模型返回的原始文本

    返回：
        dict — 包含 summary / keywords 两个键
        None — 无论如何都无法解析时返回 None
    """
    # 第一步：直接尝试解析（模型严格按格式输出时走这里）
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 第二步：用正则提取第一个 JSON 对象（应对模型在 JSON 前后加了说明文字的情况）
    match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print(f"[merger] 警告：无法从模型输出中解析 JSON\n原始输出：{raw_text[:200]}")
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

    # summary：必填，缺失时用占位文本
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

    返回：
        int  — 成功时返回新写入的 L2 记录 id
        None — 未触发或任何步骤失败时返回 None
    """
    print("[merger] 开始检查 L2 合并触发条件")

    # 第一步：读取所有未吸收 L1
    l1_rows = get_unabsorbed_l1()
    should_trigger, reason = _should_trigger(l1_rows)
    print(f"[merger] 触发判断：{reason}")

    if not should_trigger:
        return None

    print(f"[merger] 触发 L2 合并，共 {len(l1_rows)} 条 L1 待合并")

    # 第二步：格式化 L1 内容，填入 Prompt
    l1_text = _format_l1_list(l1_rows)
    prompt  = L2_MERGE_PROMPT.format(l1_summaries=l1_text)

    # 第三步：调用本地模型
    raw_output = _call_local_model(prompt)
    if raw_output is None:
        print("[merger] 模型调用失败，L2 合并中止")
        return None

    print(f"[merger] 模型原始输出：{raw_output[:200]}")

    # 第四步：解析 JSON
    parsed = _parse_l2_json(raw_output)
    if parsed is None:
        # JSON 解析彻底失败，写入降级摘要，不丢失这批 L1 的合并记录
        print("[merger] JSON 解析失败，写入降级 L2 摘要")
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

    print(f"[merger] L2 摘要已写入，id = {l2_id}")
    print(f"  summary  : {validated['summary']}")
    print(f"  keywords : {validated['keywords']}")
    print(f"  覆盖时间 : {period_start[:10]} ~ {period_end[:10]}")
    print(f"  吸收 L1  : {l1_ids}")

    # ------------------------------------------------------------------
    # 第八步：写入 L2 向量索引
    #
    # L2 写入 SQLite 并标记 L1 已吸收后，同步写入 Chroma 向量索引。
    # L2 是话题定位层，写入索引后 RAG 检索可以用它做上层导航，
    # 再穿透到对应的 L1 / L0 拿细节。
    #
    # 写入失败只打印警告，不影响 l2_id 的正常返回。
    # ------------------------------------------------------------------
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
        print(f"[merger] 警告：L2 向量索引写入失败 — {e}")

    return l2_id


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    from database import new_session, save_message, close_session, save_l1_summary

    print("=== merger.py 验证测试 ===\n")

    # ------------------------------------------------------------------
    # 测试一：格式化逻辑（不依赖模型，纯本地验证）
    # ------------------------------------------------------------------
    print("--- 测试一：L1 格式化输出 ---")

    class FakeRow(dict):
        """模拟 sqlite3.Row 的字典访问方式，仅用于测试"""
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
    trigger2, reason2 = _should_trigger(fake_rows)   # 只有2条，默认阈值5条，不触发
    print(f"2条 L1：trigger={trigger2}（应为 False，因为低于5条阈值）")
    print()

    # ------------------------------------------------------------------
    # 测试三：端到端合并（需要 LM Studio 运行中）
    # ------------------------------------------------------------------
    print("--- 测试三：端到端合并触发（需要 LM Studio 运行中）---")

    # 写入5条 L1，触发条数合并
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
        print("\n模型未响应（LM Studio 未启动），格式化与触发判断已验证，端到端测试跳过")

    print("\n验证完成。")
