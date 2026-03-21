"""
llm_client.py — 本地模型调用公共模块
=====================================================================

职责：
    封装对本地模型服务（Ollama / LM Studio）的 HTTP 请求，
    统一管理调用参数、错误处理和日志输出。

背景（审查报告问题D）：
    原来 summarizer.py / merger.py / conflict_checker.py /
    profile_manager.py 四个文件各自维护了一份几乎相同的
    _call_local_model() 函数，代码重复、难以统一维护。
    新增本模块后，所有调用本地模型的地方统一 import，
    参数、错误处理、日志格式全部在这里集中管理。

同时修复：
    原各模块未设置 "stream": False，在 Ollama 下会导致流式输出
    的第一个 token 被丢弃，造成回复开头截断的问题。
    本模块统一加上 stream=False，彻底解决这个问题。

对外暴露两个函数，调用方按任务类型选用，不需要关心 token 数字：
    call_local_summary() — 摘要类任务（L1/L2 生成、冲突检测、画像提取）
    call_local_chat()    — 对话类任务（日常聊天，需要更长回复空间）

token 上限统一在 config.py 里管理：
    LOCAL_MAX_TOKENS_SUMMARY = 512
    LOCAL_MAX_TOKENS_CHAT    = 1024

使用方法：
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

import requests

from config import (
    LOCAL_API_URL,
    LOCAL_MODEL_NAME,
    LOCAL_TEMPERATURE,
    LOCAL_MAX_TOKENS_SUMMARY,
    LOCAL_MAX_TOKENS_CHAT,
)


# =============================================================================
# 对外接口
# =============================================================================

def call_local_summary(messages, caller="llm_client"):
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
                   "[summarizer] 错误：无法连接到本地模型"。

    返回：
        str  — 模型回复的文本内容（已去除首尾空白）
        None — 任何网络或解析错误时返回 None
    """
    return _call(messages, LOCAL_MAX_TOKENS_SUMMARY, caller)


def call_local_chat(messages, caller="llm_client"):
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
# 内部实现
# =============================================================================

def _call(messages, max_tokens, caller):
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
# 直接运行此文件时：连通性验证
# =============================================================================

if __name__ == "__main__":
    print("=== llm_client.py 连通性验证 ===\n")

    # 测试一：摘要类调用
    print("--- 测试一：call_local_summary ---")
    reply = call_local_summary(
        messages=[{"role": "user", "content": "你好，请用一句话介绍自己。"}],
        caller="test",
    )
    if reply:
        print(f"回复：{reply}")
        print("call_local_summary 验证通过 ✓")
    else:
        print("调用失败，请确认本地模型服务已启动")

    # 测试二：对话类调用（带 system prompt 和历史）
    print("\n--- 测试二：call_local_chat ---")
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
        print("调用失败")

    print("\n验证完成。")
