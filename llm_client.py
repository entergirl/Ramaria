"""
llm_client.py — 本地模型调用公共模块
=====================================================================

变更记录：
  v2 — 新增公开函数 strip_thinking()，修复审查报告问题3。
       原来 summarizer.py / merger.py / conflict_checker.py /
       profile_manager.py 四个文件各自维护了一份几乎相同的
       _strip_thinking() 私有函数，逻辑有细微差异且难以统一维护。
       现将其合并到本模块作为公开函数对外暴露，四个文件删掉各自的
       私有版本，统一 from llm_client import strip_thinking 使用。

职责：
    1. 封装对本地模型服务（Ollama / LM Studio）的 HTTP 请求，
       统一管理调用参数、错误处理和日志输出。
    2. 提供 strip_thinking() 工具函数，剥离 Qwen 等模型输出中
       的思考链内容，供各摘要/检测模块解析 JSON 前调用。

对外暴露三个函数，各模块按需 import：
    strip_thinking()     — 剥离模型思考链，返回纯 JSON 文本
    call_local_summary() — 摘要类任务（L1/L2 生成、冲突检测、画像提取）
    call_local_chat()    — 对话类任务（日常聊天，需要更长回复空间）

token 上限统一在 config.py 里管理：
    LOCAL_MAX_TOKENS_SUMMARY = 512
    LOCAL_MAX_TOKENS_CHAT    = 1024

使用方法：
    # 剥离思考链（在 JSON 解析前调用）
    from llm_client import strip_thinking
    cleaned = strip_thinking(raw_output)
    result  = json.loads(cleaned)

    # 摘要类任务
    from llm_client import call_local_summary
    reply = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="summarizer",
    )

    # 对话类任务
    from llm_client import call_local_chat
    reply = call_local_chat(
        messages=history,
        caller="main",
    )

    # 返回 None 说明调用失败，调用方自行处理降级逻辑
    if reply is None:
        ...
"""

import re
import requests

from config import (
    LOCAL_API_URL,
    LOCAL_MODEL_NAME,
    LOCAL_TEMPERATURE,
    LOCAL_MAX_TOKENS_SUMMARY,
    LOCAL_MAX_TOKENS_CHAT,
)


# =============================================================================
# 工具函数：思考链剥离
# =============================================================================

def strip_thinking(raw_text: str) -> str:
    """
    剥离模型输出中的思考链内容，只保留 JSON 部分。

    [v2 新增] 修复审查报告问题3：
        原来四个模块（summarizer / merger / conflict_checker / profile_manager）
        各自维护了一份私有的 _strip_thinking() 函数，存在两处差异：

        差异一：截断起点不同
          · summarizer / merger / profile_manager：从第一个 { 开始截
            （因为它们的模型输出是 JSON 对象）
          · conflict_checker：从第一个 [ 开始截
            （因为它的模型输出是 JSON 数组）

        合并方案：同时查找 { 和 [ 的位置，取两者中更靠前的那个作为截断点。
        这样对象和数组两种格式都能正确处理，不需要调用方关心输出是哪种格式。

        差异二：函数可见性
          · 原来四份都是私有函数（_strip_thinking），无法跨模块复用
          · 现在作为公开函数（strip_thinking）在 llm_client.py 统一维护

    支持两种思考链格式：

      格式一：标签格式（Qwen /think 模式开启时）
        输入：<think>\\n这是思考过程\\n推理步骤\\n</think>\\n{"summary": "..."}
        输出：{"summary": "..."}

      格式二：纯文本前缀格式（模型直接以推理步骤开头）
        输入：Let me analyze this...\\n\\n{"summary": "..."}
        输出：{"summary": "..."}

      格式三：无思考链（正常输出，直接返回去除首尾空白的原文）
        输入：{"summary": "..."}
        输出：{"summary": "..."}

    参数：
        raw_text — 模型返回的原始文本字符串

    返回：
        str — 剥离思考链后的文本（已去除首尾空白）。
              如果原文中没有思考链，返回去除首尾空白后的原文。
              注意：本函数只做文本处理，不做 JSON 解析，调用方负责后续解析。

    使用示例：
        raw = call_local_summary(messages=[...], caller="summarizer")
        if raw:
            cleaned = strip_thinking(raw)
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                # 继续用正则提取等兜底逻辑
                ...
    """
    # ------------------------------------------------------------------
    # 第一步：处理标签格式的思考链
    # <think>...</think> 标签及其内部全部内容替换为空字符串
    # re.DOTALL  — 让 . 能跨行匹配（思考链通常是多行文本）
    # re.IGNORECASE — 兼容大小写变体，如 <Think> / <THINK>
    # ------------------------------------------------------------------
    cleaned = re.sub(
        r'<think>.*?</think>',
        '',
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # 第二步：处理纯文本前缀格式的思考链
    # 同时查找 { 和 [ 的位置，取更靠前的那个作为 JSON 起点。
    #
    # 为什么要同时找两个：
    #   · { 对应 JSON 对象输出（summarizer / merger / profile_manager）
    #   · [ 对应 JSON 数组输出（conflict_checker）
    #
    # 为什么用 -1 而不是 None：
    #   str.find() 找不到时返回 -1，用 -1 可以直接参与大小比较，
    #   通过 > 0 过滤掉"未找到"和"在开头位置"两种无效情况。
    # ------------------------------------------------------------------
    brace_pos   = cleaned.find('{')   # JSON 对象起点
    bracket_pos = cleaned.find('[')   # JSON 数组起点

    # 找出两者中更靠前（index 更小）且有效（> 0）的位置
    # 只有 pos > 0 时才代表"JSON 前面确实有需要截掉的前缀"
    # pos == 0 说明 JSON 在开头，不需要截；pos == -1 说明根本没有
    candidates = [pos for pos in (brace_pos, bracket_pos) if pos > 0]

    if candidates:
        # 取最靠前的位置，从这里开始才是真正的 JSON 内容
        cut_pos = min(candidates)
        cleaned = cleaned[cut_pos:]

    return cleaned.strip()


# =============================================================================
# 对外接口：模型调用
# =============================================================================

def call_local_summary(messages: list[dict], caller: str = "llm_client") -> str | None:
    """
    摘要类任务的本地模型调用入口。

    适用场景：
        L1 摘要生成（summarizer）、L2 合并（merger）、
        冲突检测（conflict_checker）、画像提取（profile_manager）。
        这类任务只需要短 JSON 输出，使用 LOCAL_MAX_TOKENS_SUMMARY（512）。

    参数：
        messages — OpenAI 格式的消息列表，例如：
                   [{"role": "user", "content": "你的 prompt"}]
                   摘要任务通常只有一条 user 消息。

        caller   — 调用方标识，用于日志输出，方便定位问题。
                   例如传入 "summarizer"，报错时日志显示
                   "[caller=summarizer] 无法连接到本地模型"。

    返回：
        str  — 模型回复的文本内容（已去除首尾空白）
        None — 任何网络或解析错误时返回 None
    """
    return _call(messages, LOCAL_MAX_TOKENS_SUMMARY, caller)


def call_local_chat(messages: list[dict], caller: str = "llm_client") -> str | None:
    """
    对话类任务的本地模型调用入口。

    适用场景：
        日常聊天（main.py），注入完整对话历史和记忆上下文后，
        需要更长的回复空间，使用 LOCAL_MAX_TOKENS_CHAT（1024）。
        如果回复经常被截断，可在 config.py 里调大 LOCAL_MAX_TOKENS_CHAT。

    参数：
        messages — OpenAI 格式的消息列表，通常包含 system prompt + 完整历史：
                   [
                       {"role": "system",    "content": "你是助手..."},
                       {"role": "user",      "content": "你好"},
                       {"role": "assistant", "content": "你好！"},
                       {"role": "user",      "content": "今天天气怎样"},
                   ]

        caller   — 调用方标识，用于日志输出。

    返回：
        str  — 模型回复的文本内容（已去除首尾空白）
        None — 任何网络或解析错误时返回 None
    """
    return _call(messages, LOCAL_MAX_TOKENS_CHAT, caller)


# =============================================================================
# 内部实现：HTTP 请求
# =============================================================================

def _call(messages: list[dict], max_tokens: int, caller: str) -> str | None:
    """
    实际发起 HTTP 请求的内部函数，被 call_local_summary 和 call_local_chat 共用。

    参数：
        messages   — OpenAI 格式消息列表
        max_tokens — 最大生成 token 数
        caller     — 调用方标识，用于日志

    返回：
        str 或 None
    """
    payload = {
        "model":       LOCAL_MODEL_NAME,
        "messages":    messages,
        "temperature": LOCAL_TEMPERATURE,
        "max_tokens":  max_tokens,
        "stream":      False,   # 关闭流式，避免 Ollama 下首 token 丢失导致回复截断
    }

    try:
        response = requests.post(LOCAL_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        print(f"[{caller}] 错误：无法连接到本地模型（{LOCAL_API_URL}），请确认服务已启动")
        return None

    except requests.exceptions.Timeout:
        print(f"[{caller}] 错误：请求超时（超过 120 秒），模型响应过慢")
        return None

    except requests.exceptions.HTTPError as e:
        print(f"[{caller}] 错误：HTTP 请求失败 — {e}")
        return None

    except (KeyError, IndexError) as e:
        print(f"[{caller}] 错误：解析模型响应结构失败 — {e}")
        return None


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    print("=== llm_client.py 验证测试 ===\n")

    # ------------------------------------------------------------------
    # 测试一：strip_thinking — 标签格式思考链
    # ------------------------------------------------------------------
    print("--- 测试一：标签格式思考链剥离 ---")

    mock_think_tag = '<think>\n这是思考过程\n第二行推理\n</think>\n{"summary": "摘要内容", "keywords": "k1,k2"}'
    result = strip_thinking(mock_think_tag)
    print(f"输入：{mock_think_tag[:50]}...")
    print(f"输出：{result}")
    assert result.startswith('{"summary"'), f"标签格式剥离失败：{result}"
    print("✓ 通过\n")

    # ------------------------------------------------------------------
    # 测试二：strip_thinking — 纯文本前缀格式（JSON 对象）
    # ------------------------------------------------------------------
    print("--- 测试二：纯文本前缀格式剥离（JSON 对象）---")

    mock_think_text = 'Let me analyze this conversation.\n\n{"summary": "摘要内容", "keywords": "k1,k2"}'
    result2 = strip_thinking(mock_think_text)
    print(f"输入：{mock_think_text[:50]}...")
    print(f"输出：{result2}")
    assert result2.startswith('{"summary"'), f"纯文本前缀剥离失败：{result2}"
    print("✓ 通过\n")

    # ------------------------------------------------------------------
    # 测试三：strip_thinking — JSON 数组格式（conflict_checker 场景）
    # ------------------------------------------------------------------
    print("--- 测试三：JSON 数组格式剥离（conflict_checker 场景）---")

    mock_think_array = '<think>分析冲突。</think>\n[{"field": "personal_status", "conflict_desc": "有矛盾"}]'
    result3 = strip_thinking(mock_think_array)
    print(f"输入：{mock_think_array[:60]}...")
    print(f"输出：{result3}")
    assert result3.startswith('[{"field"'), f"数组格式剥离失败：{result3}"
    print("✓ 通过\n")

    # ------------------------------------------------------------------
    # 测试四：strip_thinking — 无思考链（正常输出，原样返回）
    # ------------------------------------------------------------------
    print("--- 测试四：无思考链，原样返回 ---")

    mock_clean = '{"summary": "摘要内容", "keywords": "k1,k2"}'
    result4 = strip_thinking(mock_clean)
    print(f"输入：{mock_clean}")
    print(f"输出：{result4}")
    assert result4 == mock_clean, f"无思考链时输出不应改变：{result4}"
    print("✓ 通过\n")

    # ------------------------------------------------------------------
    # 测试五：strip_thinking — 数组和对象同时存在时，取更靠前的
    # ------------------------------------------------------------------
    print("--- 测试五：{ 和 [ 同时存在，取更靠前的 ---")

    # [ 在 { 前面 → 应该从 [ 开始截
    mock_bracket_first = 'some prefix [{"field": "f1"}] and {"key": "val"}'
    result5 = strip_thinking(mock_bracket_first)
    print(f"输入：{mock_bracket_first}")
    print(f"输出：{result5}")
    assert result5.startswith('[{"field"'), f"应从 [ 开始截：{result5}"
    print("✓ 通过\n")

    # ------------------------------------------------------------------
    # 测试六：call_local_summary / call_local_chat（需要 LM Studio 运行中）
    # ------------------------------------------------------------------
    print("--- 测试六：call_local_summary（需要 LM Studio 运行中）---")
    reply = call_local_summary(
        messages=[{"role": "user", "content": "你好，请用一句话介绍自己。"}],
        caller="test",
    )
    if reply:
        print(f"回复：{reply}")
        print("call_local_summary 验证通过 ✓")
    else:
        print("调用失败（LM Studio 未启动，属正常，跳过）")

    print("\n--- 测试七：call_local_chat（需要 LM Studio 运行中）---")
    reply2 = call_local_chat(
        messages=[
            {"role": "system",    "content": "你是一个简洁的助手，每次只回复一句话。"},
            {"role": "user",      "content": "今天天气真好。"},
            {"role": "assistant", "content": "是啊，适合出去走走。"},
            {"role": "user",      "content": "你喜欢什么季节？"},
        ],
        caller="test",
    )
    if reply2:
        print(f"回复：{reply2}")
        print("call_local_chat 验证通过 ✓")
    else:
        print("调用失败（LM Studio 未启动，属正常，跳过）")

    print("\n验证完成。")
