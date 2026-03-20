"""
profile_manager.py — L3 画像半自动维护模块
=====================================================================

负责从 L1 摘要中自动提取新信息，静默更新 L3 用户画像。
是阶段二收尾的最后一个核心模块。

─────────────────────────────────────────────────────────────
设计理念
─────────────────────────────────────────────────────────────

  不打扰用户，静默补充画像；发现矛盾时才开口询问。
  像真实朋友一样——默默记住你说的话，只在"你前后说的不一样"
  时才自然地提起，而不是每次都追问"我可以记下来吗"。

─────────────────────────────────────────────────────────────
核心流程
─────────────────────────────────────────────────────────────

  触发时机：summarizer.generate_l1_summary() 写入 L1 后，
            在 conflict_checker 之前调用 extract_and_update()

  Step 1  读取刚生成的 L1 摘要内容
  Step 2  读取当前生效的 L3 用户画像（作为上下文传给模型）
  Step 3  调用本地 Qwen，让模型判断 L1 里哪些是"新信息"
          （模型只返回 L3 没有记录的内容，冲突内容不返回）
  Step 4  解析模型返回的 JSON，校验字段合法性
  Step 5  逐字段写入 user_profile 表：
            · 字段已有值 → 追加到末尾（静默补充，不覆盖）
            · 字段无值   → 直接写入（新建）
          写入内容带时间戳前缀，格式：[2026-03-20] 内容

─────────────────────────────────────────────────────────────
与其他模块的关系
─────────────────────────────────────────────────────────────

  · 被调用：summarizer.py（L1 写入后，conflict_checker 之前）
  · 读 L1：database.get_latest_l1()
  · 读画像：database.get_current_profile()
  · 写画像：database.update_profile_field()（追加模式）
  · 读配置：config.py

  调用顺序（summarizer 第九、十步）：
    第九步：profile_manager.extract_and_update(l1_id)  ← 本模块
    第十步：conflict_checker.check_conflicts(l1_id)    ← 需要拿最新 L3 比对

─────────────────────────────────────────────────────────────
时间戳说明
─────────────────────────────────────────────────────────────

  每条写入内容都带上本地日期前缀，例如：
    "[2026-03-20] 正在开发珊瑚菌记忆系统，进入阶段二收尾"

  注入 system prompt 时保留这个格式，让模型能区分
  "两个月前的状态"和"最近的状态"，避免时间混淆。

使用方法：
    from profile_manager import extract_and_update
    extract_and_update(l1_id)   # 由 summarizer 在 L1 写入后调用
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
)
from database import (
    get_current_profile,
    get_latest_l1,
    update_profile_field,
)


# =============================================================================
# Prompt 模板
# =============================================================================

# 标准版：画像已有内容时使用
# 传入当前 L3 + L1 摘要，模型只返回"新增信息"，冲突信息不在这里处理
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
严格按照以下 JSON 格式输出，不要输出任何其他内容（不要加 markdown 代码块，不要加说明文字）：

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
# 不需要新旧对比，直接从 L1 里提取有长期价值的信息
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
# 合法字段集合（与 user_profile 表和 conflict_checker 保持一致）
# =============================================================================

VALID_FIELDS = {
    "basic_info",
    "personal_status",
    "interests",
    "social",
    "history",
    "recent_context",
}


# =============================================================================
# 工具函数
# =============================================================================

def _get_today_str():
    """
    返回当前本地日期字符串，格式 "2026-03-20"。
    用于给写入内容加时间戳前缀，避免模型混淆历史记忆的时间。

    返回：
        str — 如 "2026-03-20"
    """
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")


def _add_timestamp(content):
    """
    给画像内容加上今日日期前缀。

    参数：
        content — 原始内容文本，如 "正在开发珊瑚菌记忆系统"

    返回：
        str — 带前缀的内容，如 "[2026-03-20] 正在开发珊瑚菌记忆系统"
    """
    return f"[{_get_today_str()}] {content}"


def _format_profile_for_prompt(profile_dict):
    """
    将 L3 用户画像字典格式化为适合填入 Prompt 的文本。
    同时标注英文字段名，方便模型在输出时准确填写 field。

    参数：
        profile_dict — get_current_profile() 返回的字典

    返回：
        str — 格式化后的文本；画像为空时返回 "（暂无画像记录）"
    """
    if not profile_dict:
        return "（暂无画像记录）"

    # 按固定顺序输出，每行标注英文字段名 + 中文标签
    field_labels = [
        ("basic_info",      "基础信息"),
        ("personal_status", "近期状态"),
        ("interests",       "兴趣爱好"),
        ("social",          "社交情况"),
        ("history",         "历史事件"),
        ("recent_context",  "近期背景"),
    ]

    lines = []
    for field_key, label in field_labels:
        content = profile_dict.get(field_key, "").strip()
        if content:
            lines.append(f"[{field_key} {label}] {content}")

    return "\n".join(lines) if lines else "（暂无画像记录）"


# =============================================================================
# 调用本地模型
# =============================================================================

def _call_local_model(prompt):
    """
    向 LM Studio 发送请求，返回模型的回复文本。
    与其他模块的同名函数逻辑一致，独立维护避免循环依赖。

    参数：
        prompt — 完整的 Prompt 字符串

    返回：
        str  — 模型回复文本（已去除首尾空白）
        None — 任何错误时返回 None
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
        print("[profile_manager] 错误：无法连接到 LM Studio，请确认模型已启动")
        return None
    except requests.exceptions.Timeout:
        print("[profile_manager] 错误：请求超时（超过 60 秒）")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[profile_manager] 错误：HTTP 请求失败 — {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"[profile_manager] 错误：解析模型响应结构失败 — {e}")
        return None


# =============================================================================
# 解析模型输出
# =============================================================================

def _strip_thinking(raw_text):
    """
    剥离模型输出中的思考链，只保留 JSON 部分。
    与其他模块的同名函数逻辑完全一致，独立维护避免循环依赖。

    参数：
        raw_text — 模型返回的原始文本

    返回：
        str — 剥离后的文本（已去除首尾空白）
    """
    # 处理 <think>...</think> 标签格式
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text,
                     flags=re.DOTALL | re.IGNORECASE)

    # 处理纯文本思考链：找到第一个 { 之前的内容，截掉
    brace_pos = cleaned.find('{')
    if brace_pos > 0:
        cleaned = cleaned[brace_pos:]

    return cleaned.strip()


def _parse_extract_json(raw_text):
    """
    从模型返回的原始文本中解析出画像提取结果。

    模型应输出一个 JSON 对象：
      · 有新信息时：{"field名": "内容", ...}
      · 无新信息时：{}

    解析策略（三步递进）：
      第一步：直接解析（模型严格按格式输出时）
      第二步：剥离思考链后再解析（/no_think 失效时）
      第三步：正则提取第一个 JSON 对象（JSON 前后有多余文字时）

    参数：
        raw_text — 模型返回的原始文本

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
    stripped = _strip_thinking(raw_text)
    if stripped != raw_text:
        print("[profile_manager] 检测到思考链输出，已自动剥离")
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

    print(f"[profile_manager] 警告：无法解析模型输出\n原始输出：{raw_text[:200]}")
    return None


def _validate_extract_result(result_dict):
    """
    校验提取结果，过滤掉字段名不合法或内容为空的条目。

    参数：
        result_dict — _parse_extract_json() 返回的字典

    返回：
        dict — 过滤后的合法条目字典；可能为空字典
    """
    validated = {}

    for field, content in result_dict.items():
        # 校验字段名是否在合法集合内
        if field not in VALID_FIELDS:
            print(f"[profile_manager] 警告：字段名 {field!r} 不合法，已跳过")
            continue

        # 校验内容非空
        content = str(content).strip()
        if not content:
            print(f"[profile_manager] 警告：字段 {field} 内容为空，已跳过")
            continue

        validated[field] = content

    return validated


# =============================================================================
# 画像写入逻辑
# =============================================================================

def _append_to_profile_field(field, new_content, source_l1_id, profile_dict):
    """
    将新内容追加写入指定画像字段。

    追加规则：
      · 字段已有值 → 在末尾用"；"分隔追加新内容
      · 字段无值   → 直接写入

    新内容会自动加上今日时间戳前缀，写入前先读旧值再合并，
    最终调用 update_profile_field() 写入（它内部会把旧版本标记为历史）。

    参数：
        field        — 画像字段名，如 "recent_context"
        new_content  — 需要写入的新内容（不含时间戳）
        source_l1_id — 触发此更新的 L1 记录 id
        profile_dict — 当前生效的完整画像字典（用于判断字段是否已有值）

    返回：
        int — update_profile_field() 返回的新记录 id
    """
    # 给新内容加时间戳
    timestamped_content = _add_timestamp(new_content)

    existing = profile_dict.get(field, "").strip()

    if existing:
        # 字段已有内容，追加在末尾
        merged_content = f"{existing}；{timestamped_content}"
        print(f"[profile_manager] 字段 {field} 追加：{timestamped_content[:50]}")
    else:
        # 字段为空，直接写入
        merged_content = timestamped_content
        print(f"[profile_manager] 字段 {field} 新建：{timestamped_content[:50]}")

    return update_profile_field(
        field        = field,
        new_content  = merged_content,
        source_l1_id = source_l1_id,
    )


# =============================================================================
# 对外接口
# =============================================================================

def extract_and_update(l1_id):
    """
    从指定 L1 摘要中提取新信息，静默写入 L3 用户画像。
    这是本模块唯一的对外接口，由 summarizer.generate_l1_summary() 调用。

    调用时机：
        L1 写入 SQLite 后，conflict_checker.check_conflicts() 之前。
        顺序很重要——冲突检测需要拿到最新的 L3 才能正确比对。

    完整流程：
      Step 1  读取指定 L1 的摘要内容（校验 id 是否匹配）
      Step 2  读取当前 L3 画像（决定使用标准版还是冷启动 Prompt）
      Step 3  构建 Prompt，调用本地 Qwen 提取新信息
      Step 4  解析并校验模型输出
      Step 5  逐字段追加写入，带时间戳前缀

    参数：
        l1_id — 刚生成的 L1 记录 id（int）

    返回：
        int  — 成功写入的字段数量（0 表示无新信息或跳过）
        None — 模型调用失败时返回 None
    """
    print(f"[profile_manager] 开始为 L1 {l1_id} 提取画像候选条目")

    # ------------------------------------------------------------------
    # Step 1：读取 L1 摘要内容
    # 通过 get_latest_l1() 读取，适用于 summarizer 刚写完立即调用的场景
    # ------------------------------------------------------------------
    l1_row = get_latest_l1()

    if l1_row is None or l1_row["id"] != l1_id:
        # 传入的 l1_id 和数据库里最新的 L1 不匹配，说明调用时序有问题
        print(f"[profile_manager] 警告：找不到 L1 id={l1_id}，跳过画像提取")
        return 0

    l1_summary = l1_row["summary"]
    print(f"[profile_manager] 读取 L1 摘要：{l1_summary[:80]}")

    # ------------------------------------------------------------------
    # Step 2：读取当前 L3 画像
    # 画像为空 → 冷启动模式，直接提取有价值信息
    # 画像有内容 → 标准模式，只提取新增部分，不提取已记录或有冲突的内容
    # ------------------------------------------------------------------
    profile_dict = get_current_profile()
    is_cold_start = not profile_dict

    if is_cold_start:
        print("[profile_manager] 画像为空，使用冷启动 Prompt")
    else:
        print(f"[profile_manager] 当前画像已有 {len(profile_dict)} 个字段，使用标准 Prompt")

    # ------------------------------------------------------------------
    # Step 3：构建 Prompt，调用本地模型
    # ------------------------------------------------------------------
    if is_cold_start:
        prompt = EXTRACT_PROMPT_COLD_START.format(
            l1_summary = l1_summary,
        )
    else:
        profile_text = _format_profile_for_prompt(profile_dict)
        prompt = EXTRACT_PROMPT.format(
            profile_text = profile_text,
            l1_summary   = l1_summary,
        )

    raw_output = _call_local_model(prompt)

    if raw_output is None:
        print("[profile_manager] 模型调用失败，画像提取中止")
        return None

    print(f"[profile_manager] 模型原始输出：{raw_output[:200]}")

    # ------------------------------------------------------------------
    # Step 4：解析并校验模型输出
    # ------------------------------------------------------------------
    result = _parse_extract_json(raw_output)

    if result is None:
        # 解析彻底失败，静默跳过，不写入任何内容
        # 宁可漏记，不要写入错误内容
        print("[profile_manager] 解析失败，本次画像提取跳过")
        return 0

    if not result:
        # 空字典，模型判断 L1 里没有值得写入的新信息，属正常情况
        print("[profile_manager] L1 中无新画像信息，画像保持不变")
        return 0

    validated = _validate_extract_result(result)

    if not validated:
        print("[profile_manager] 校验后无合法条目，画像保持不变")
        return 0

    print(f"[profile_manager] 提取到 {len(validated)} 条新信息，开始写入")

    # ------------------------------------------------------------------
    # Step 5：逐字段写入
    # 追加模式：字段已有值则在末尾追加，无值则新建
    # 写入内容带今日时间戳前缀，方便 system prompt 里的时间定位
    # ------------------------------------------------------------------
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
            # 单个字段写入失败不影响其他字段，打印警告继续
            print(f"[profile_manager] 警告：字段 {field} 写入失败 — {e}")

    print(f"[profile_manager] 画像更新完成，共写入 {written} 个字段")
    return written


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    from database import (
        new_session, save_message, close_session,
        save_l1_summary, update_profile_field,
    )

    print("=== profile_manager.py 验证测试 ===\n")

    # ------------------------------------------------------------------
    # 测试一：工具函数验证（不依赖模型）
    # ------------------------------------------------------------------
    print("--- 测试一：工具函数验证 ---")

    # 时间戳格式
    ts = _add_timestamp("正在开发珊瑚菌记忆系统")
    print(f"时间戳格式：{ts}")
    assert ts.startswith("[20"), "时间戳格式不对"

    # 画像格式化（有内容）
    mock_profile = {
        "basic_info":     "烧酒，19岁，本科在读",
        "recent_context": "[2026-03-01] 开发珊瑚菌阶段一",
    }
    formatted = _format_profile_for_prompt(mock_profile)
    print(f"画像格式化（有内容）：\n{formatted}")

    # 画像格式化（空画像）
    empty_formatted = _format_profile_for_prompt({})
    print(f"画像格式化（空）：{empty_formatted}")
    print()

    # ------------------------------------------------------------------
    # 测试二：JSON 解析与字段校验（不依赖模型）
    # ------------------------------------------------------------------
    print("--- 测试二：JSON 解析与字段校验 ---")

    # 标准输出（有新信息）
    mock_valid = '{"recent_context": "正在完成珊瑚菌阶段二收尾工作", "interests": "AI记忆系统开发"}'
    r1 = _parse_extract_json(mock_valid)
    v1 = _validate_extract_result(r1)
    print(f"标准解析结果：{v1}")

    # 空对象（无新信息）
    r2 = _parse_extract_json('{}')
    print(f"空对象解析：{r2}（应为 {{}}）")

    # 非法字段名过滤
    mock_bad = '{"mood": "很好", "recent_context": "在写代码"}'
    r3 = _parse_extract_json(mock_bad)
    v3 = _validate_extract_result(r3)
    print(f"非法字段过滤：{v3}（mood 应被过滤，只剩 recent_context）")

    # 带思考链的输出
    mock_think = '<think>分析新信息。</think>\n{"recent_context": "完成了向量检索模块"}'
    r4 = _parse_extract_json(mock_think)
    print(f"思考链剥离：{r4}（应正常解析）")
    print()

    # ------------------------------------------------------------------
    # 测试三：追加写入逻辑（需要数据库）
    # ------------------------------------------------------------------
    print("--- 测试三：追加写入逻辑 ---")

    # 先写入一条旧记录
    update_profile_field("recent_context", "[2026-03-01] 完成了阶段一开发")
    current = get_current_profile()
    print(f"写入前：{current.get('recent_context', '（空）')}")

    # 创建测试 L1
    sid = new_session()
    save_message(sid, "user", "今天把阶段二的向量检索写完了")
    save_message(sid, "assistant", "太棒了，进度很扎实！")
    close_session(sid)
    l1_id = save_l1_summary(
        session_id  = sid,
        summary     = "烧酒完成了阶段二向量检索模块的开发。",
        keywords    = "向量检索,阶段二",
        time_period = "夜间",
        atmosphere  = "专注高效",
    )

    # 手动触发追加（绕过模型调用，直接测试写入逻辑）
    _append_to_profile_field(
        field        = "recent_context",
        new_content  = "完成了阶段二向量检索模块",
        source_l1_id = l1_id,
        profile_dict = current,
    )

    updated = get_current_profile()
    print(f"追加后：{updated.get('recent_context', '（空）')}")
    assert "[2026-03-01]" in updated["recent_context"], "旧内容不应被覆盖"
    print("验证通过：旧内容保留，新内容附加在末尾")
    print()

    # ------------------------------------------------------------------
    # 测试四：端到端提取（需要 LM Studio 运行中）
    # ------------------------------------------------------------------
    print("--- 测试四：端到端画像提取（需要 LM Studio 运行中）---")

    sid2 = new_session()
    save_message(sid2, "user", "今天把 profile_manager 写完了，阶段二全部收尾了")
    save_message(sid2, "assistant", "恭喜！这是一个重要的里程碑。")
    close_session(sid2)
    l1_id2 = save_l1_summary(
        session_id  = sid2,
        summary     = "烧酒完成了珊瑚菌阶段二全部开发工作，系统进入可用状态。",
        keywords    = "珊瑚菌,阶段二,里程碑",
        time_period = "夜间",
        atmosphere  = "专注高效",
    )
    print(f"已创建测试 L1 id={l1_id2}")

    count = extract_and_update(l1_id2)

    if count is None:
        print("\n模型未响应（LM Studio 未启动），本地逻辑测试已通过，端到端测试跳过")
    elif count == 0:
        print("\n模型判断 L1 中无新画像信息（属正常情况，取决于测试数据）")
    else:
        print(f"\n端到端测试通过，成功提取并写入 {count} 个字段")
        final_profile = get_current_profile()
        for field, content in final_profile.items():
            print(f"  [{field}] {content[:80]}")

    print("\n验证完成。")
