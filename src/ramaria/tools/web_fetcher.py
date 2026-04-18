"""
src/ramaria/tools/web_fetcher.py — 联网查询底层封装

职责：
    提供统一的 HTTP 请求工具函数，供上层各联网工具模块调用。
    封装超时、错误处理、重试逻辑，上层模块无需关心网络细节。

设计原则（可扩展性）：
    · 所有联网工具（天气、未来可能的新闻摘要等）都通过本模块发起请求
    · 新增联网工具时只需 import fetch_json / fetch_text，不重复实现超时/错误处理
    · 本模块不包含任何业务逻辑，只负责「把数据取回来」

对外接口：
    fetch_json(url, params, timeout) -> dict | None
        发起 GET 请求，解析并返回 JSON 响应体。
        失败时返回 None，不抛出异常。

    fetch_text(url, params, timeout) -> str | None
        发起 GET 请求，返回纯文本响应体。
        失败时返回 None，不抛出异常。
"""

from __future__ import annotations

from typing import Any

import requests

from ramaria.logger import get_logger

logger = get_logger(__name__)

# 默认超时时间（秒）
# 联网工具调用在对话链路上，超时过长会影响响应速度
DEFAULT_TIMEOUT = 8


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict | list | None:
    """
    发起 GET 请求，解析 JSON 响应。

    参数：
        url     — 请求地址
        params  — 查询参数字典，会自动编码到 URL
        timeout — 超时时间（秒），默认 8 秒

    返回：
        dict | list — 解析成功的 JSON 对象
        None        — 网络错误、超时、非 JSON 响应时返回 None
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"fetch_json 超时（>{timeout}s）：{url}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"fetch_json 连接失败（无网络？）：{url}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.warning(f"fetch_json HTTP 错误：{e}  url={url}")
        return None
    except ValueError:
        logger.warning(f"fetch_json 响应不是合法 JSON：{url}")
        return None
    except Exception as e:
        logger.warning(f"fetch_json 未知错误：{e}  url={url}")
        return None


def fetch_text(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str | None:
    """
    发起 GET 请求，返回纯文本响应体。

    参数：
        url     — 请求地址
        params  — 查询参数字典
        timeout — 超时时间（秒），默认 8 秒

    返回：
        str  — 响应文本（已去除首尾空白）
        None — 任何错误时返回 None
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.text.strip()
    except requests.exceptions.Timeout:
        logger.warning(f"fetch_text 超时（>{timeout}s）：{url}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"fetch_text 连接失败（无网络？）：{url}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.warning(f"fetch_text HTTP 错误：{e}  url={url}")
        return None
    except Exception as e:
        logger.warning(f"fetch_text 未知错误：{e}  url={url}")
        return None
