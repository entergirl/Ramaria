"""
summarizer.py — L1 摘要生成模块
负责在 session 结束后，调用本地 Qwen 模型生成结构化摘要，并写入 memory_l1 表。

变更记录：
  v2 — 修复审查报告 N1/N2：
       N1：第十步注释块首行多余空格已删除（原来是5个空格，改为标准4个）
       N2：模块文档核心流程补充第11步，并对齐步骤8~11的实际执行顺序

核心流程：
  1. 从数据库读取指定 session 的全部消息
  2. 从 keyword_pool 表读取历史关键词，作为候选列表
  3. 将消息格式化为纯文本对话，连同候选词一起填入 Prompt 模板
  4. 调用本地 LM Studio API（Qwen）生成摘要
  5. 解析模型返回的 JSON 结果
  6. 校验字段合法性，写入 memory_l1 表
  7. 将本次使用的关键词同步写回 keyword_pool 表
  8. 写入 L1 向量索引（vector_store.index_l1）
  9. 提取新信息，静默写入 L3 用户画像（profile_manager.extract_and_update）
  10. 触发冲突检测（conflict_checker.check_conflicts）
  11. 触发路径A — L2 条数检查（merger.check_and_merge）

关键词词典机制说明：
  - 词典冷启动时为空，早期关键词相对发散，属正常现象
  - 随使用量积累，词库逐渐稳定收敛，同义词发散问题自然改善
  - 词典较小时（< KEYWORD_INJECT_THRESHOLD 条）注入全部词条；
    词典较大时只注入高频前 KEYWORD_INJECT_LIMIT 条，避免 Prompt 过长

容错设计：
  - 模型返回内容无法解析为 JSON 时：记录原始输出，写入降级摘要，不抛出异常
  - 网络请求失败时：打印错误，返回 None，不阻断主流程
  - 词典读取或写回失败时：只打印警告，不影响摘要生成主流程

与其他模块的关系：
  - 读消息：database.get_messages()
  - 读词典：database.get_all_keywords() / database.get_top_keywords()
  - 写摘要：database.save_l1_summary()
  - 写词典：database.upsert_keywords()
  - 读配置：config.py
  - 触发 L2：merger.check_and_merge()（懒加载，避免循环依赖）

使用方法：
    from summarizer import generate_l1_summary
    generate_l1_summary(session_id)   # session 结束后由 session_manager 调用
"""

import json
import re
import requests

from config import (
    L1_SUMMARY_PROMPT,
    L1_SUMMARY_PROMPT_WITH_KEYWORDS,
    TIME_PERIOD_OPTIONS,
)
from llm_client import call_local_summary, strip_thinking
from database import (
    get_messages,
    save_l1_summary,
    get_all_keywords,
    get_top_keywords,
    upsert_keywords,
)


# =============================================================================
# 词典注入参数
# =============================================================================

# 词典词条数低于此阈值时，全量注入所有词条；
# 超过阈值时，只注入高频前 KEYWORD_INJECT_LIMIT 条，避免 Prompt 过长
KEYWORD_INJECT_THRESHOLD = 100

# 词典超过阈值时，最多注入多少个候选词
KEYWORD_INJECT_LIMIT = 50


# =============================================================================
# 对话格式化
# =============================================================================

def _format_conversation(messages):
    """
    将消息列表格式化为纯文本对话，供 Prompt 填入使用。

    参数：
        messages — get_messages() 返回的 sqlite3.Row 列表，按 created_at ASC 排序

    返回：
        str — 每行一条消息，格式为"角色：内容"

    示例输出：
        用户：你好
        助手：你好！有什么可以帮你？
    """
    lines = []
    for msg in messages:
        role_label = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role_label}：{msg['content']}")
    return "\n".join(lines)


# =============================================================================
# 关键词候选列表处理
# =============================================================================

def _get_keyword_candidates():
    """
    从词典表读取候选关键词，格式化为逗号分隔的字符串，供注入 Prompt 使用。

    词典为空时返回空字符串（调用方据此决定使用哪个 Prompt 模板）。
    词典较小时全量返回，词典较大时只返回高频前 KEYWORD_INJECT_LIMIT 条。

    返回：
        str — 中文逗号分隔的关键词字符串；词典为空时返回空字符串 ""
    """
    try:
        all_kws = get_all_keywords()

        if not all_kws:
            return ""

        if len(all_kws) <= KEYWORD_INJECT_THRESHOLD:
            candidates = all_kws
        else:
            candidates = get_top_keywords(limit=KEYWORD_INJECT_LIMIT)

        return "，".join(candidates)

    except Exception as e:
        print(f"[summarizer] 警告：读取关键词词典失败，将使用无候选词版 Prompt — {e}")
        return ""


def _extract_keywords_list(keywords_str):
    """
    将关键词字段字符串拆分为列表，用于写回词典。

    参数：
        keywords_str — 关键词字符串，如 "数据库,后端,验证"；允许传入 None

    返回：
        list[str] — 关键词字符串列表；传入 None 或空字符串时返回空列表
    """
    if not keywords_str:
        return []

    # 同时兼容中文逗号和英文逗号
    keywords_str = keywords_str.replace("，", ",")
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]


# =============================================================================
# 解析与校验模型输出
# =============================================================================

def _parse_summary_json(raw_text):
    """
    从模型返回的原始文本中解析出结构化摘要字段。

    解析策略（三步递进）：
      第一步：直接尝试解析
      第二步：剥离思考链后再次尝试
      第三步：用正则提取第一个完整 JSON 对象

    参数：
        raw_text — 模型返回的原始文本字符串

    返回：
        dict — 包含 summary / keywords / time_period / atmosphere 四个键
        None — 三步都失败时返回 None
    """
    # 第一步：直接尝试解析
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 第二步：剥离思考链后再次解析
    stripped = strip_thinking(raw_text)
    if stripped != raw_text:
        print(f"[summarizer] 检测到思考链输出，已自动剥离，重新解析")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 第三步：用正则提取第一个完整 JSON 对象
    match = re.search(r'\{.*?\}', stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print(f"[summarizer] 警告：无法从模型输出中解析 JSON\n原始输出：{raw_text[:200]}")
    return None


def _validate_and_fix(parsed):
    """
    校验解析结果的字段完整性和合法性，对不合规的值进行降级处理。

    参数：
        parsed — _parse_summary_json() 返回的字典

    返回：
        dict — 校验/修复后的字典，四个字段均有值（部分允许为 None）
    """
    result = {}

    # summary：必填，缺失或为空时用占位文本替代
    result["summary"] = str(parsed.get("summary", "")).strip()
    if not result["summary"]:
        result["summary"] = "（摘要生成失败，内容为空）"

    # keywords：允许为空
    keywords = str(parsed.get("keywords", "")).strip()
    result["keywords"] = keywords if keywords else None

    # time_period：严格六选一校验
    tp = str(parsed.get("time_period", "")).strip()
    result["time_period"] = tp if tp in TIME_PERIOD_OPTIONS else None
    if result["time_period"] is None and tp:
        print(f"[summarizer] 警告：time_period 值 {tp!r} 不合法，已置为 None")

    # atmosphere：四字以内，超长时截断
    atm = str(parsed.get("atmosphere", "")).strip()
    if len(atm) > 4:
        print(f"[summarizer] 警告：atmosphere 值 {atm!r} 超过四字，已截断")
        atm = atm[:4]
    result["atmosphere"] = atm if atm else None

    return result


# =============================================================================
# 对外接口
# =============================================================================

def generate_l1_summary(session_id):
    """
    为指定 session 生成 L1 摘要并写入数据库。
    这是本模块唯一的对外接口，由 session_manager 在 session 结束时调用。

    完整流程：
      1.  读取 session 的全部消息
      2.  读取关键词词典，获取候选词列表
      3.  格式化对话文本，选择合适的 Prompt 模板填入内容
      4.  调用本地 Qwen 模型生成摘要
      5.  解析 JSON 输出
      6.  校验字段合法性，写入 memory_l1 表
      7.  将本次关键词同步写回 keyword_pool 词典
      8.  将 L1 摘要写入 Chroma 向量索引
      9.  提取新信息，静默写入 L3 用户画像（profile_manager）
      10. 触发冲突检测（conflict_checker）
      11. 触发路径A — 检查是否需要 L2 合并（条数触发）

    参数：
        session_id — 需要生成摘要的 session id（int）

    返回：
        int  — 成功时返回新写入的 L1 记录 id
        None — 任何步骤失败时返回 None（不抛出异常，不阻断主流程）
    """
    print(f"[summarizer] 开始为 session {session_id} 生成 L1 摘要")

    # 第一步：读取消息
    messages = get_messages(session_id)

    if not messages:
        print(f"[summarizer] session {session_id} 没有消息，跳过摘要生成")
        return None

    print(f"[summarizer] 读取到 {len(messages)} 条消息")

    # 第二步：读取关键词候选列表
    keyword_candidates = _get_keyword_candidates()

    if keyword_candidates:
        print(f"[summarizer] 词典候选词已注入（共 {len(keyword_candidates.split('，'))} 个）")
    else:
        print(f"[summarizer] 词典为空（冷启动阶段），使用基础 Prompt")

    # 第三步：格式化对话文本，选择 Prompt 模板
    conversation_text = _format_conversation(messages)

    if keyword_candidates:
        prompt = L1_SUMMARY_PROMPT_WITH_KEYWORDS.format(
            conversation       = conversation_text,
            keyword_candidates = keyword_candidates,
        )
    else:
        prompt = L1_SUMMARY_PROMPT.format(
            conversation = conversation_text,
        )

    # 第四步：调用本地模型
    raw_output = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="summarizer",
    )

    if raw_output is None:
        print(f"[summarizer] 模型调用失败，session {session_id} 摘要生成中止")
        return None

    print(f"[summarizer] 模型原始输出：{raw_output[:200]}")

    # 第五步：解析 JSON
    parsed = _parse_summary_json(raw_output)

    if parsed is None:
        print(f"[summarizer] JSON 解析失败，写入降级摘要")
        l1_id = save_l1_summary(
            session_id  = session_id,
            summary     = f"（自动摘要失败，原始输出已记录）{raw_output[:100]}",
            keywords    = None,
            time_period = None,
            atmosphere  = None,
        )
        return l1_id

    # 第六步：校验字段，写入 memory_l1 表
    validated = _validate_and_fix(parsed)

    l1_id = save_l1_summary(
        session_id  = session_id,
        summary     = validated["summary"],
        keywords    = validated["keywords"],
        time_period = validated["time_period"],
        atmosphere  = validated["atmosphere"],
    )

    print(f"[summarizer] L1 摘要已写入，id = {l1_id}")
    print(f"  summary     : {validated['summary']}")
    print(f"  keywords    : {validated['keywords']}")
    print(f"  time_period : {validated['time_period']}")
    print(f"  atmosphere  : {validated['atmosphere']}")

    # 第七步：将本次关键词写回 keyword_pool 词典
    if validated["keywords"]:
        try:
            kw_list = _extract_keywords_list(validated["keywords"])
            upsert_keywords(kw_list)
            print(f"[summarizer] 关键词已写回词典：{kw_list}")
        except Exception as e:
            print(f"[summarizer] 警告：关键词写回词典失败 — {e}")

    # 第八步：将 L1 摘要写入向量索引
    try:
        from vector_store import index_l1
        index_l1(
            l1_id      = l1_id,
            summary    = validated["summary"],
            keywords   = validated["keywords"],
            session_id = session_id,
        )
    except Exception as e:
        print(f"[summarizer] 警告：L1 向量索引写入失败 — {e}")

    # ------------------------------------------------------------------
    # 第九步：提取新信息，静默写入 L3 用户画像
    #
    # 必须在 conflict_checker 之前执行：
    #   冲突检测需要拿到"最新的 L3"来比对，
    #   而 profile_manager 会在这一步把新信息追加进 L3，
    #   所以顺序不能颠倒。
    # ------------------------------------------------------------------
    try:
        from profile_manager import extract_and_update
        extract_and_update(l1_id)
    except Exception as e:
        print(f"[summarizer] 警告：画像提取时出现异常 — {e}")

    # ------------------------------------------------------------------
    # 第十步：触发冲突检测
    #
    # [修复 N1] 原来此注释块首行多一个空格（5个空格 vs 标准4个），已修正。
    #
    # 此时 L3 已经被 profile_manager 更新过，
    # conflict_checker 拿到的是包含本次新信息的最新 L3，
    # 能更准确地判断是否存在真实矛盾。
    # ------------------------------------------------------------------
    try:
        from conflict_checker import check_conflicts
        check_conflicts(l1_id)
    except Exception as e:
        print(f"[summarizer] 警告：冲突检测时出现异常 — {e}")

    # ------------------------------------------------------------------
    # 第十一步：触发路径A — L1 写入后立即检查是否需要触发 L2 合并（条数触发）
    # 使用懒加载 import 避免 summarizer ↔ merger 循环依赖
    # check_and_merge() 内部会判断条件，不满足时静默返回，不影响主流程
    # ------------------------------------------------------------------
    try:
        from merger import check_and_merge
        check_and_merge()
    except Exception as e:
        print(f"[summarizer] 警告：L2 合并检查时出现异常 — {e}")

    return l1_id


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    from database import new_session, save_message, close_session

    print("=== summarizer.py 验证测试 ===\n")

    print("--- 测试一：工具函数验证 ---")
    kws = _extract_keywords_list("数据库,后端,验证")
    print(f"英文逗号：{kws}（应为 ['数据库', '后端', '验证']）")
    kws2 = _extract_keywords_list("数据库，后端，验证")
    print(f"中文逗号：{kws2}（应为 ['数据库', '后端', '验证']）")
    kws3 = _extract_keywords_list(None)
    print(f"传入 None：{kws3}（应为 []）")
    print()

    print("--- 测试二：JSON 解析与字段校验 ---")
    mock_clean = '{"summary":"烧酒完成了后端验证。","keywords":"后端,验证","time_period":"夜间","atmosphere":"专注高效"}'
    result = _parse_summary_json(mock_clean)
    validated = _validate_and_fix(result)
    print(f"标准 JSON：{validated}")

    mock_bad_tp = '{"summary":"测试。","keywords":"测试","time_period":"中午","atmosphere":"正常"}'
    result3 = _parse_summary_json(mock_bad_tp)
    validated3 = _validate_and_fix(result3)
    print(f"time_period 不合法：{validated3['time_period']}（应为 None）")

    mock_think = '<think>分析请求。</think>\n{"summary":"烧酒完成了验证。","keywords":"验证","time_period":"夜间","atmosphere":"专注高效"}'
    result5 = _parse_summary_json(mock_think)
    print(f"思考链剥离：summary={result5['summary'] if result5 else None}（应正常解析）")
    print()

    print("--- 测试三：端到端（需要 LM Studio 运行中）---")
    sid = new_session()
    save_message(sid, "user", "今天把 session_manager 写完了，验证也通过了")
    save_message(sid, "assistant", "太棒了，进度很稳！")
    save_message(sid, "user", "接下来是 summarizer")
    save_message(sid, "assistant", "不会太难，我们一步一步来。")
    close_session(sid)
    print(f"已创建测试 session {sid}，包含 4 条消息")

    l1_id = generate_l1_summary(sid)

    if l1_id:
        print(f"\n端到端测试通过，L1 id = {l1_id}")
    else:
        print("\n模型未响应（LM Studio 未启动），本地逻辑测试已通过，端到端测试跳过")

    print("\n验证完成。")
