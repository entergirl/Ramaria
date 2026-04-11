"""
src/ramaria/core/llm_client.py — 本地模型调用公共模块

职责：
    1. 封装对本地模型服务（Ollama / LM Studio）的 HTTP 请求
    2. 提供 strip_thinking() 工具函数，剥离本地模型输出中的思考链

对外暴露三个函数：
    strip_thinking()     — 剥离模型思考链，返回纯 JSON 文本
    call_local_summary() — 摘要类任务（L1/L2 生成、冲突检测、画像提取）
    call_local_chat()    — 对话类任务（日常聊天，需要更长回复空间）

    
"""

import re

import requests

from ramaria.config import (
    LOCAL_API_URL,
    LOCAL_MAX_TOKENS_CHAT,
    LOCAL_MAX_TOKENS_SUMMARY,
    LOCAL_MODEL_NAME,
    LOCAL_TEMPERATURE,
)

from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 工具函数：思考链剥离
# =============================================================================

def strip_thinking(raw_text: str) -> str:
    """
    剥离模型输出中的思考链内容，只保留 JSON 部分。

    支持三种格式：

        格式一：标签格式（思考链模式开启时）
            输入：<think>\\n这是思考过程\\n</think>\\n{"summary": "..."}
            输出：{"summary": "..."}

        格式二：纯文本前缀格式（模型直接以推理步骤开头）
            输入：Let me analyze this...\\n\\n{"summary": "..."}
            输出：{"summary": "..."}

        格式三：无思考链（正常输出，直接返回去除首尾空白的原文）

    同时支持 JSON 对象（{）和 JSON 数组（[）两种起始格式，
    取两者中更靠前的位置作为截断点。

    参数：
        raw_text — 模型返回的原始文本字符串

    返回：
        str — 剥离思考链后的文本（已去除首尾空白）
    """
    # 第一步：处理标签格式的思考链
    cleaned = re.sub(
        r'<think>.*?</think>',
        '',
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 第二步：处理纯文本前缀格式
    # 同时查找 { 和 [ 的位置，取更靠前的那个作为 JSON 起点
    brace_pos   = cleaned.find('{')
    bracket_pos = cleaned.find('[')

    candidates = [pos for pos in (brace_pos, bracket_pos) if pos > 0]

    if candidates:
        cut_pos = min(candidates)
        cleaned = cleaned[cut_pos:]

    return cleaned.strip()


# =============================================================================
# 对外接口：模型调用
# =============================================================================

def call_local_summary(
    messages: list[dict],
    caller: str = "llm_client",
) -> str | None:
    """
    摘要类任务的本地模型调用入口。

    适用场景：
        L1 摘要生成（summarizer）、L2 合并（merger）、
        冲突检测（conflict_checker）、画像提取（profile_manager）。
        使用 LOCAL_MAX_TOKENS_SUMMARY（512）。

    参数：
        messages — OpenAI 格式的消息列表
        caller   — 调用方标识，用于日志输出

    返回：
        str  — 模型回复的文本内容（已去除首尾空白）
        None — 任何网络或解析错误时返回 None
    """
    return _call(messages, LOCAL_MAX_TOKENS_SUMMARY, caller)


def call_local_chat(
    messages: list[dict],
    caller: str = "llm_client",
) -> str | None:
    """
    对话类任务的本地模型调用入口。

    适用场景：
        日常聊天（main.py），注入完整对话历史和记忆上下文后，
        需要更长的回复空间。使用 LOCAL_MAX_TOKENS_CHAT（1024）。

    参数：
        messages — OpenAI 格式的消息列表（通常包含 system + 完整历史）
        caller   — 调用方标识，用于日志输出

    返回：
        str  — 模型回复的文本内容（已去除首尾空白）
        None — 任何网络或解析错误时返回 None
    """
    return _call(messages, LOCAL_MAX_TOKENS_CHAT, caller)


# =============================================================================
# 内部实现：HTTP 请求
# =============================================================================

def _call(
    messages: list[dict],
    max_tokens: int,
    caller: str,
) -> str | None:
    """
    实际发起 HTTP 请求的内部函数，被 call_local_summary 和 call_local_chat 共用。

    参数：
        messages   — OpenAI 格式消息列表
        max_tokens — 最大生成 token 数
        caller     — 调用方标识，用于日志

    返回：str 或 None
    """
    payload = {
        "model":       LOCAL_MODEL_NAME,
        "messages":    messages,
        "temperature": LOCAL_TEMPERATURE,
        "max_tokens":  max_tokens,
        # 关闭流式，避免 Ollama 下首 token 丢失导致回复截断
        "stream":      False,
    }

    try:
        response = requests.post(LOCAL_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        logger.error(
            f"[caller={caller}] 无法连接到本地模型（{LOCAL_API_URL}），"
            f"请确认服务已启动"
        )
        return None

    except requests.exceptions.Timeout:
        logger.error(
            f"[caller={caller}] 请求超时（超过 120 秒），模型响应过慢"
        )
        return None

    except requests.exceptions.HTTPError as e:
        logger.error(f"[caller={caller}] HTTP 请求失败 — {e}")
        return None

    except (KeyError, IndexError) as e:
        logger.error(f"[caller={caller}] 解析模型响应结构失败 — {e}")
        return None