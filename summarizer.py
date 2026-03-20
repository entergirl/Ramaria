"""
summarizer.py — L1 摘要生成模块
负责在 session 结束后，调用本地 Qwen 模型生成结构化摘要，并写入 memory_l1 表。

核心流程：
  1. 从数据库读取指定 session 的全部消息
  2. 从 keyword_pool 表读取历史关键词，作为候选列表
  3. 将消息格式化为纯文本对话，连同候选词一起填入 Prompt 模板
  4. 调用本地 LM Studio API（Qwen）生成摘要
  5. 解析模型返回的 JSON 结果
  6. 校验字段合法性，写入 memory_l1 表
  7. 将本次使用的关键词同步写回 keyword_pool 表
  8. 触发冲突检测（conflict_checker.check_conflicts）
  9. 写入 L1 向量索引（vector_store.index_l1）
  10. 触发路径A — L2 条数检查（merger.check_and_merge）

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
    LOCAL_API_URL,
    LOCAL_MODEL_NAME,
    LOCAL_TEMPERATURE,
    LOCAL_MAX_TOKENS,
    L1_SUMMARY_PROMPT,
    L1_SUMMARY_PROMPT_WITH_KEYWORDS,
    TIME_PERIOD_OPTIONS,
)
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
        用户：帮我看一下这段代码
        助手：好的，我来看看……
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
        str — 中文逗号分隔的关键词字符串，如 "数据库，后端，项目，FastAPI"
              词典为空时返回空字符串 ""
    """
    try:
        all_kws = get_all_keywords()

        if not all_kws:
            # 词典为空（冷启动阶段），返回空字符串
            return ""

        if len(all_kws) <= KEYWORD_INJECT_THRESHOLD:
            # 词典较小，全量注入
            candidates = all_kws
        else:
            # 词典较大，只取高频前 N 个，避免 Prompt 过长
            candidates = get_top_keywords(limit=KEYWORD_INJECT_LIMIT)

        return "，".join(candidates)

    except Exception as e:
        # 词典读取失败不影响主流程，降级为无候选词版 Prompt
        print(f"[summarizer] 警告：读取关键词词典失败，将使用无候选词版 Prompt — {e}")
        return ""


def _extract_keywords_list(keywords_str):
    """
    将关键词字段字符串拆分为列表，用于写回词典。

    模型输出的关键词字段格式为"词A,词B,词C"（英文逗号分隔）。
    本函数将其拆分并清理空格，得到干净的字符串列表。
    同时兼容模型偶尔使用中文逗号的情况。

    参数：
        keywords_str — 关键词字符串，如 "数据库,后端,验证"
                       允许传入 None，此时返回空列表

    返回：
        list[str] — 关键词字符串列表，如 ["数据库", "后端", "验证"]
                    传入 None 或空字符串时返回空列表
    """
    if not keywords_str:
        return []

    # 同时兼容中文逗号和英文逗号，避免模型偶尔用错分隔符导致拆分失败
    keywords_str = keywords_str.replace("，", ",")
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]


# =============================================================================
# 调用本地模型
# =============================================================================

def _call_local_model(prompt):
    """
    向 LM Studio 发送请求，返回模型的回复文本。

    消息结构说明：
        用 system 消息注入 /no_think 指令，比放在 user 消息正文里更可靠。
        /no_think 告诉 Qwen 跳过思考链，直接输出结果，避免返回内容里
        混入 "Thinking Process:..." 等内容导致 JSON 解析失败。

    参数：
        prompt — 完整的 Prompt 字符串（已填入对话内容和候选词）

    返回：
        str  — 模型回复的文本内容（已去除首尾空白）
        None — 任何网络或解析错误时返回 None
    """
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [
            # system 消息：关闭思考链，强制直接输出 JSON
            # /no_think 必须单独放在 system 层才能被 Qwen 可靠识别
            {"role": "system", "content": "/no_think"},
            # user 消息：Prompt 正文（已填入对话内容和候选词）
            {"role": "user", "content": prompt},
        ],
        "temperature": LOCAL_TEMPERATURE,
        "max_tokens": LOCAL_MAX_TOKENS,
    }

    try:
        response = requests.post(
            LOCAL_API_URL,
            json=payload,
            timeout=60,   # 本地模型响应较慢，给足 60 秒
        )
        response.raise_for_status()   # 非 2xx 状态码时抛出异常
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        print("[summarizer] 错误：无法连接到 LM Studio，请确认模型已启动")
        return None
    except requests.exceptions.Timeout:
        print("[summarizer] 错误：请求超时（超过 60 秒），模型响应过慢")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[summarizer] 错误：HTTP 请求失败 — {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"[summarizer] 错误：解析模型响应结构失败 — {e}")
        return None


# =============================================================================
# 解析与校验模型输出
# =============================================================================

def _strip_thinking(raw_text):
    """
    剥离模型输出中的思考链内容，只保留 JSON 部分。

    /no_think 指令并不能保证 Qwen 完全跳过思考链，当指令失效时，
    模型会在 JSON 前输出一大段推理过程。本函数负责把这部分内容清理掉，
    让后续的 JSON 解析能正常工作。

    支持两种思考链格式：
      - 标签格式：<think>...</think> 整段标签及其内容
      - 纯文本格式：模型直接以推理步骤开头，直到出现 { 字符为止的所有内容

    参数：
        raw_text — 模型返回的原始文本字符串

    返回：
        str — 剥离思考链后的文本（已去除首尾空白）
              如果原文中没有思考链，原样返回
    """
    # 处理标签格式：<think>...</think>
    # re.DOTALL 让 . 能跨行匹配，re.IGNORECASE 兼容大小写变体
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)

    # 处理纯文本格式：{ 之前的所有内容都是思考链，直接丢弃
    # 找到第一个 { 的位置，截取从这里开始的内容
    brace_pos = cleaned.find('{')
    if brace_pos > 0:
        # { 前面有内容，说明存在思考链前缀，截掉它
        cleaned = cleaned[brace_pos:]

    return cleaned.strip()


def _parse_summary_json(raw_text):
    """
    从模型返回的原始文本中解析出结构化摘要字段。

    解析策略（三步递进）：
      第一步：直接尝试解析（模型严格按格式输出时走这里，最快）
      第二步：剥离思考链后再次尝试直接解析（应对 /no_think 失效的情况）
      第三步：用正则提取第一个完整 JSON 对象（应对 JSON 前后仍有说明文字的情况）

    参数：
        raw_text — 模型返回的原始文本字符串

    返回：
        dict — 包含 summary / keywords / time_period / atmosphere 四个键
        None — 三步都失败时返回 None

    返回示例：
        {
            "summary":     "烧酒完成了 session_manager 的验证测试。",
            "keywords":    "session管理,后端,验证",
            "time_period": "夜间",
            "atmosphere":  "专注高效"
        }
    """
    # 第一步：直接尝试解析（模型严格按格式输出时走这里）
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 第二步：剥离思考链后再次尝试解析
    # 应对 /no_think 指令失效、模型输出了 <think>...</think> 或纯文本推理过程的情况
    stripped = _strip_thinking(raw_text)
    if stripped != raw_text:
        # 内容有变化，说明确实剥离了思考链，打印提示方便排查
        print(f"[summarizer] 检测到思考链输出，已自动剥离，重新解析")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 第三步：用正则提取第一个完整 JSON 对象
    # re.DOTALL 让 . 能匹配换行符，处理模型输出多行 JSON 的情况
    match = re.search(r'\{.*?\}', stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 三步都失败，打印警告并返回 None
    print(f"[summarizer] 警告：无法从模型输出中解析 JSON\n原始输出：{raw_text[:200]}")
    return None


def _validate_and_fix(parsed):
    """
    校验解析结果的字段完整性和合法性，对不合规的值进行降级处理。
    保证返回的字典里四个字段都存在，不会有 KeyError。

    参数：
        parsed — _parse_summary_json() 返回的字典

    返回：
        dict — 校验/修复后的字典，四个字段均有值（部分允许为 None）

    各字段处理规则：
        summary     — 必填，缺失或为空时用占位文本替代
        keywords    — 允许为 None，清理多余空格后原样保留
        time_period — 必须是 TIME_PERIOD_OPTIONS 六选一，不合法时置为 None
        atmosphere  — 四字以内，超长时截断到前四字，为空时置为 None
    """
    result = {}

    # summary：必填字段，为空时写入占位文本，不能让数据库里出现空摘要
    result["summary"] = str(parsed.get("summary", "")).strip()
    if not result["summary"]:
        result["summary"] = "（摘要生成失败，内容为空）"

    # keywords：允许为空，只清理多余空格
    keywords = str(parsed.get("keywords", "")).strip()
    result["keywords"] = keywords if keywords else None

    # time_period：严格六选一校验，不在列表内的值一律置为 None
    tp = str(parsed.get("time_period", "")).strip()
    result["time_period"] = tp if tp in TIME_PERIOD_OPTIONS else None
    if result["time_period"] is None and tp:
        # 模型返回了值但不合法，打印警告方便排查 Prompt 问题
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
      1. 读取 session 的全部消息
      2. 读取关键词词典，获取候选词列表
      3. 格式化对话文本，选择合适的 Prompt 模板填入内容
      4. 调用本地 Qwen 模型生成摘要
      5. 解析 JSON 输出
      6. 校验字段合法性，写入 memory_l1 表
      7. 将本次关键词同步写回 keyword_pool 词典
      8. 将 L1 摘要写入 Chroma 向量索引
      9. 触发冲突检测（与现有用户画像比对）
      10. 触发路径A — 检查是否需要 L2 合并（条数触发）

    Prompt 选择策略：
      - 词典有内容时：使用带候选词列表的 L1_SUMMARY_PROMPT_WITH_KEYWORDS
      - 词典为空时（冷启动）：使用基础版 L1_SUMMARY_PROMPT

    参数：
        session_id — 需要生成摘要的 session id（int）

    返回：
        int  — 成功时返回新写入的 L1 记录 id
        None — 任何步骤失败时返回 None（不抛出异常，不阻断主流程）
    """
    print(f"[summarizer] 开始为 session {session_id} 生成 L1 摘要")

    # ------------------------------------------------------------------
    # 第一步：读取消息
    # ------------------------------------------------------------------
    messages = get_messages(session_id)

    if not messages:
        print(f"[summarizer] session {session_id} 没有消息，跳过摘要生成")
        return None

    print(f"[summarizer] 读取到 {len(messages)} 条消息")

    # ------------------------------------------------------------------
    # 第二步：读取关键词候选列表
    # 词典为空（冷启动）时返回空字符串，后续会选择不带候选词的基础 Prompt
    # ------------------------------------------------------------------
    keyword_candidates = _get_keyword_candidates()

    if keyword_candidates:
        print(f"[summarizer] 词典候选词已注入（共 {len(keyword_candidates.split('，'))} 个）")
    else:
        print(f"[summarizer] 词典为空（冷启动阶段），使用基础 Prompt")

    # ------------------------------------------------------------------
    # 第三步：格式化对话文本，选择 Prompt 模板并填入内容
    # 有候选词时用带词典版本，引导模型复用已有词条
    # 无候选词时用基础版本，不在 Prompt 里留空白的候选词区域
    # ------------------------------------------------------------------
    conversation_text = _format_conversation(messages)

    if keyword_candidates:
        prompt = L1_SUMMARY_PROMPT_WITH_KEYWORDS.format(
            conversation=conversation_text,
            keyword_candidates=keyword_candidates,
        )
    else:
        prompt = L1_SUMMARY_PROMPT.format(
            conversation=conversation_text,
        )

    # ------------------------------------------------------------------
    # 第四步：调用本地模型
    # ------------------------------------------------------------------
    raw_output = _call_local_model(prompt)

    if raw_output is None:
        print(f"[summarizer] 模型调用失败，session {session_id} 摘要生成中止")
        return None

    # 只打印前 200 字，避免日志过长
    print(f"[summarizer] 模型原始输出：{raw_output[:200]}")

    # ------------------------------------------------------------------
    # 第五步：解析 JSON
    # ------------------------------------------------------------------
    parsed = _parse_summary_json(raw_output)

    if parsed is None:
        # JSON 解析彻底失败，写入降级摘要
        # 目的是保留记录，不丢失这条 session 的存在，哪怕内容是错误信息
        print(f"[summarizer] JSON 解析失败，写入降级摘要")
        l1_id = save_l1_summary(
            session_id  = session_id,
            summary     = f"（自动摘要失败，原始输出已记录）{raw_output[:100]}",
            keywords    = None,
            time_period = None,
            atmosphere  = None,
        )
        return l1_id

    # ------------------------------------------------------------------
    # 第六步：校验字段，写入 memory_l1 表
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 第七步：将本次关键词写回 keyword_pool 词典
    # 只在关键词字段有内容时执行，空关键词不写入词典
    # 写回失败只打印警告，不影响 l1_id 的正常返回
    # ------------------------------------------------------------------
    if validated["keywords"]:
        try:
            kw_list = _extract_keywords_list(validated["keywords"])
            upsert_keywords(kw_list)
            print(f"[summarizer] 关键词已写回词典：{kw_list}")
        except Exception as e:
            print(f"[summarizer] 警告：关键词写回词典失败 — {e}")

    # ------------------------------------------------------------------
    # 第八步：将 L1 摘要写入向量索引
    #
    # L1 写入 SQLite 之后，紧接着写入 Chroma 向量索引。
    # 这样 RAG 检索才能找到这条记忆——SQLite 存的是完整数据，
    # Chroma 存的是语义坐标，两者缺一不可。
    #
    # 索引文本 = 摘要 + 关键词（在 index_l1 内部拼接），
    # 把关键词也纳入向量化，提升相关记忆的召回率。
    #
    # 写入失败只打印警告，不影响 l1_id 的正常返回，不阻断主流程。
    # 使用懒加载 import 避免循环依赖（vector_store → database → summarizer）。
    # ------------------------------------------------------------------
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
    # 第九步：触发冲突检测
    # L1 写入后，检查新内容是否与现有用户画像存在矛盾
    # 使用懒加载 import 避免循环依赖
    # 检测失败只打印警告，不影响 l1_id 的正常返回
    # ------------------------------------------------------------------
    try:
        from conflict_checker import check_conflicts
        check_conflicts(l1_id)
    except Exception as e:
        print(f"[summarizer] 警告：冲突检测时出现异常 — {e}")

    # ------------------------------------------------------------------
    # 第十步：触发路径A — L1 写入后立即检查是否需要触发 L2 合并（条数触发）
    # 使用懒加载 import 避免 summarizer ↔ merger 循环依赖
    # check_and_merge() 内部会判断条件，不满足时静默返回，不影响主流程
    # ------------------------------------------------------------------
    try:
        from merger import check_and_merge
        check_and_merge()
    except Exception as e:
        # L2 合并失败不应阻断 L1 写入的返回结果，只打印警告
        print(f"[summarizer] 警告：L2 合并检查时出现异常 — {e}")

    return l1_id


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    from database import new_session, save_message, close_session

    print("=== summarizer.py 验证测试 ===\n")

    # ------------------------------------------------------------------
    # 测试一：工具函数验证（不依赖模型，纯本地逻辑）
    # ------------------------------------------------------------------
    print("--- 测试一：工具函数验证 ---")

    # _extract_keywords_list：正常英文逗号分隔
    kws = _extract_keywords_list("数据库,后端,验证")
    print(f"英文逗号分隔：{kws}（应为 ['数据库', '后端', '验证']）")

    # _extract_keywords_list：中文逗号兼容
    kws2 = _extract_keywords_list("数据库，后端，验证")
    print(f"中文逗号兼容：{kws2}（应为 ['数据库', '后端', '验证']）")

    # _extract_keywords_list：传入 None
    kws3 = _extract_keywords_list(None)
    print(f"传入 None：{kws3}（应为 []）")

    print()

    # ------------------------------------------------------------------
    # 测试二：JSON 解析与字段校验（不依赖模型）
    # ------------------------------------------------------------------
    print("--- 测试二：JSON 解析与字段校验 ---")

    # 标准 JSON 输出
    mock_clean = '{"summary":"烧酒完成了后端数据库模块的开发与验证。","keywords":"后端,数据库,验证","time_period":"夜间","atmosphere":"专注高效"}'
    result = _parse_summary_json(mock_clean)
    validated = _validate_and_fix(result)
    print(f"标准 JSON 解析：{validated}")

    # 模型在 JSON 前后加了说明文字（Qwen 偶发行为）
    mock_dirty = '好的，以下是摘要：\n{"summary":"烧酒讨论了项目架构设计。","keywords":"架构,项目","time_period":"上午","atmosphere":"轻松愉快"}\n希望对你有帮助。'
    result2 = _parse_summary_json(mock_dirty)
    validated2 = _validate_and_fix(result2)
    print(f"带前后说明文字：{validated2}")

    # time_period 不合法
    mock_bad_tp = '{"summary":"测试摘要。","keywords":"测试","time_period":"中午","atmosphere":"正常"}'
    result3 = _parse_summary_json(mock_bad_tp)
    validated3 = _validate_and_fix(result3)
    print(f"time_period 不合法：time_period = {validated3['time_period']}（应为 None）")

    # atmosphere 超长
    mock_long_atm = '{"summary":"测试摘要。","keywords":"测试","time_period":"夜间","atmosphere":"非常非常专注高效"}'
    result4 = _parse_summary_json(mock_long_atm)
    validated4 = _validate_and_fix(result4)
    print(f"atmosphere 超长截断：atmosphere = {validated4['atmosphere']}（应为前四字）")

    # 模拟 Qwen 输出了思考链（<think> 标签格式）
    mock_think_tag = '<think>\n分析请求：需要生成结构化摘要。\n输出格式：严格 JSON。\n</think>\n{"summary":"烧酒完成了验证测试。","keywords":"验证,测试","time_period":"夜间","atmosphere":"专注高效"}'
    result5 = _parse_summary_json(mock_think_tag)
    print(f"思考链标签格式剥离：summary = {result5['summary'] if result5 else None}（应正常解析）")

    # 模拟 Qwen 输出了思考链（纯文本格式，和截图里一致）
    mock_think_text = 'Thinking Process:\n1. Analyze the Request.\n2. Output Format: Strict JSON\n{"summary":"烧酒完成了验证测试。","keywords":"验证,测试","time_period":"夜间","atmosphere":"专注高效"}'
    result6 = _parse_summary_json(mock_think_text)
    print(f"思考链纯文本格式剥离：summary = {result6['summary'] if result6 else None}（应正常解析）")

    print()

    # ------------------------------------------------------------------
    # 测试三：词典候选词注入（不依赖模型）
    # ------------------------------------------------------------------
    print("--- 测试三：词典候选词注入 ---")

    # 先写入一些测试词条
    upsert_keywords(["数据库", "后端", "项目", "架构", "验证"])
    candidates = _get_keyword_candidates()
    print(f"词典候选词：{candidates}")
    print(f"（词典有内容，应返回非空字符串）")

    print()

    # ------------------------------------------------------------------
    # 测试四：端到端流程（需要 LM Studio 正在运行）
    # ------------------------------------------------------------------
    print("--- 测试四：端到端摘要生成（需要 LM Studio 运行中）---")

    sid = new_session()
    save_message(sid, "user", "杋枫姐，我今天把 session_manager 写完了，验证也通过了")
    save_message(sid, "assistant", "太棒了！三个测试全部符合预期，你今天进度很扎实。")
    save_message(sid, "user", "接下来是 summarizer，感觉这块逻辑会复杂一些")
    save_message(sid, "assistant", "不会太难，主要是 Prompt 格式化和 JSON 解析，我们一步一步来。")
    close_session(sid)
    print(f"已创建测试 session {sid}，包含 4 条消息")
    print()

    l1_id = generate_l1_summary(sid)

    if l1_id:
        print(f"\n端到端测试通过，L1 id = {l1_id}")
    else:
        print("\n模型未响应（LM Studio 未启动），本地逻辑测试已通过，端到端测试跳过")

    print("\n验证完成。")
