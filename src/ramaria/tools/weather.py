"""
src/ramaria/tools/weather.py — 天气查询模块

职责：
    1. 通过 ip-api.com 自动定位当前城市（无需 API Key）
    2. 通过 wttr.in 查询天气（无需 API Key）
    3. 格式化为可注入 System Prompt 的简洁文本

数据来源：
    · 定位：http://ip-api.com/json（免费，无 Key，每分钟限 45 次）
    · 天气：https://wttr.in/{城市}?format=j1（免费，无 Key）

优先级：
    1. config.py 的 WEATHER_CITY（用户在 .env 中手动指定）
    2. ip-api.com 自动定位
    3. 两者均失败时返回空字符串，静默降级

缓存策略：
    · 城市定位结果：进程生命周期内缓存，不重复请求
    · 天气数据：10 分钟内缓存，同一 session 高频触发时复用上次结果
      避免每次对话都请求 wttr.in

对外接口：
    get_weather() -> str
        返回格式化后的天气文本；任何步骤失败时返回空字符串。

    可扩展性说明：
        未来新增「新闻摘要」「汇率」等即时状态查询时，
        参考本模块结构新建对应文件，通过 web_fetcher 发请求即可。
"""

from __future__ import annotations

import time
from typing import Optional

from ramaria.logger import get_logger

logger = get_logger(__name__)

# ── 缓存 ─────────────────────────────────────────────────────────────────────

# 城市定位缓存（进程生命周期内不重新定位）
_cached_city: Optional[str] = None
_city_resolved: bool = False   # 是否已尝试过定位（无论成功失败）

# 天气数据缓存
_cached_weather_text: Optional[str] = None
_weather_cache_time: float = 0.0

# 天气缓存有效期（秒）
_WEATHER_CACHE_TTL = 600   # 10 分钟

# ── API 端点 ──────────────────────────────────────────────────────────────────

_IP_API_URL  = "http://ip-api.com/json"
_WTTR_URL    = "https://wttr.in/{city}"

# wttr.in 天气状况代码 → 中文描述映射
# 只映射常见代码，未覆盖的用英文原文兜底
_WEATHER_CODE_ZH: dict[int, str] = {
    113: "晴",
    116: "局部多云",
    119: "多云",
    122: "阴",
    143: "有雾",
    176: "局部小雨",
    179: "局部小雪",
    182: "冻雨",
    185: "冻雨",
    200: "雷阵雨",
    227: "吹雪",
    230: "暴风雪",
    248: "雾",
    260: "冰雾",
    263: "零星小雨",
    266: "小雨",
    281: "冻毛毛雨",
    284: "冻毛毛雨",
    293: "局部小雨",
    296: "小雨",
    299: "中雨",
    302: "中雨",
    305: "大雨",
    308: "大雨",
    311: "冻雨",
    314: "冻雨",
    317: "小雨夹雪",
    320: "雨夹雪",
    323: "局部小雪",
    326: "小雪",
    329: "中雪",
    332: "中雪",
    335: "大雪",
    338: "大雪",
    350: "冰雹",
    353: "小阵雨",
    356: "中到大阵雨",
    359: "暴雨",
    362: "小雨夹雪",
    365: "中到大雨夹雪",
    368: "小阵雪",
    371: "中到大阵雪",
    374: "小冰雹",
    377: "冰雹",
    386: "雷阵雨",
    389: "强雷阵雨",
    392: "雷阵雪",
    395: "强雷阵雪",
}

# 风向代码 → 中文
_WIND_DIR_ZH: dict[str, str] = {
    "N": "北风", "NNE": "北偏东风", "NE": "东北风", "ENE": "东偏北风",
    "E": "东风", "ESE": "东偏南风", "SE": "东南风", "SSE": "南偏东风",
    "S": "南风", "SSW": "南偏西风", "SW": "西南风", "WSW": "西偏南风",
    "W": "西风", "WNW": "西偏北风", "NW": "西北风", "NNW": "北偏西风",
}


# =============================================================================
# 城市定位
# =============================================================================

def _get_city() -> Optional[str]:
    """
    获取当前城市名（英文），供 wttr.in 查询使用。

    优先级：
        1. config.py 中的 WEATHER_CITY（用户手动指定）
        2. ip-api.com 自动定位
        3. 两者均失败 → 返回 None

    结果在进程生命周期内缓存，不重复请求。
    """
    global _cached_city, _city_resolved

    # 已尝试过定位，直接返回缓存结果（即使是 None）
    if _city_resolved:
        return _cached_city

    _city_resolved = True

    # ── 优先级1：用户手动配置 ─────────────────────────────────────────────
    try:
        from ramaria.config import WEATHER_CITY
        if WEATHER_CITY and WEATHER_CITY.strip():
            _cached_city = WEATHER_CITY.strip()
            logger.info(f"天气城市使用配置值：{_cached_city}")
            return _cached_city
    except (ImportError, AttributeError):
        pass

    # ── 优先级2：IP 自动定位 ──────────────────────────────────────────────
    try:
        from ramaria.tools.web_fetcher import fetch_json
        data = fetch_json(_IP_API_URL, timeout=5)

        if data and data.get("status") == "success":
            city = data.get("city", "")
            if city:
                _cached_city = city
                logger.info(f"IP 定位城市：{city}（{data.get('country', '')}）")
                return _cached_city
            else:
                logger.warning("ip-api 返回成功但城市字段为空")
        else:
            logger.warning(f"ip-api 定位失败：{data}")

    except Exception as e:
        logger.warning(f"IP 定位时发生异常 — {e}")

    logger.warning("城市定位失败，天气功能将不可用")
    return None


# =============================================================================
# 天气数据获取与格式化
# =============================================================================

def _weather_cache_valid() -> bool:
    """判断天气缓存是否仍在有效期内。"""
    return (
        _cached_weather_text is not None
        and time.time() - _weather_cache_time < _WEATHER_CACHE_TTL
    )


def _fetch_weather_raw(city: str) -> dict | None:
    """
    向 wttr.in 请求指定城市的天气 JSON 数据。

    wttr.in 的 format=j1 返回结构化 JSON，包含：
        current_condition — 当前天气
        weather           — 未来3天预报
        nearest_area      — 最近城市信息

    参数：
        city — 城市名（英文），如 "Shanghai"

    返回：
        dict — wttr.in 返回的原始 JSON
        None — 请求失败
    """
    from ramaria.tools.web_fetcher import fetch_json

    url  = _WTTR_URL.format(city=city)
    data = fetch_json(url, params={"format": "j1", "lang": "zh"}, timeout=8)

    if data is None:
        logger.warning(f"wttr.in 请求失败，城市：{city}")
    return data


def _parse_weather(data: dict, city: str) -> str:
    """
    解析 wttr.in 的 JSON 响应，格式化为简洁的中文天气文本。

    格式示例：
        上海 · 当前天气
        🌤 多云，14°C（体感 12°C）
        湿度 68%，东南风 3 级（18 km/h）
        今日：8°C ~ 17°C

    参数：
        data — wttr.in 返回的原始 JSON
        city — 城市名，用于显示

    返回：
        str — 格式化后的天气文本；解析失败时返回基础信息
    """
    try:
        current = data["current_condition"][0]

        # 温度
        temp_c      = current.get("temp_C", "?")
        feels_like  = current.get("FeelsLikeC", "?")

        # 天气状况
        weather_code = int(current.get("weatherCode", 0))
        weather_desc = _WEATHER_CODE_ZH.get(
            weather_code,
            current.get("weatherDesc", [{}])[0].get("value", "未知")
        )

        # 湿度
        humidity = current.get("humidity", "?")

        # 风向和风速
        wind_dir_raw = current.get("winddir16Point", "")
        wind_dir     = _WIND_DIR_ZH.get(wind_dir_raw, wind_dir_raw)
        wind_kmh     = current.get("windspeedKmph", "?")

        # 风力等级（粗略换算：0-5→1级，6-11→2级，12-19→3级，20-28→4级，≥29→5+级）
        try:
            kmh = int(wind_kmh)
            if kmh < 6:       wind_level = "1"
            elif kmh < 12:    wind_level = "2"
            elif kmh < 20:    wind_level = "3"
            elif kmh < 29:    wind_level = "4"
            elif kmh < 39:    wind_level = "5"
            else:             wind_level = "6+"
        except (ValueError, TypeError):
            wind_level = "?"

        # 今日最高/最低温度
        today_high = "?"
        today_low  = "?"
        if data.get("weather"):
            today = data["weather"][0]
            today_high = today.get("maxtempC", "?")
            today_low  = today.get("mintempC", "?")

        # 拼装最终文本
        lines = [
            f"{city} · 当前天气",
            f"{weather_desc}，{temp_c}°C（体感 {feels_like}°C）",
            f"湿度 {humidity}%，{wind_dir} {wind_level} 级（{wind_kmh} km/h）",
            f"今日：{today_low}°C ~ {today_high}°C",
        ]
        return "\n".join(lines)

    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"天气数据解析失败 — {e}")
        # 降级：返回尽量简单的信息
        try:
            current = data["current_condition"][0]
            return f"{city} · {current.get('temp_C', '?')}°C"
        except Exception:
            return f"{city} · 天气数据解析失败"


# =============================================================================
# 对外接口
# =============================================================================

def get_weather() -> str:
    """
    获取当前城市的天气，返回格式化文本。

    使用 10 分钟缓存，避免高频对话反复请求 wttr.in。

    返回：
        str — 格式化天气文本（4行左右）
        ""  — 城市定位失败或网络不可用时返回空字符串
    """
    global _cached_weather_text, _weather_cache_time

    # 缓存有效，直接返回
    if _weather_cache_valid():
        logger.debug("天气数据使用缓存")
        return _cached_weather_text

    # 获取城市
    city = _get_city()
    if not city:
        return ""

    # 请求天气数据
    raw = _fetch_weather_raw(city)
    if raw is None:
        return ""

    # 解析并缓存
    text = _parse_weather(raw, city)
    if text:
        _cached_weather_text = text
        _weather_cache_time  = time.time()
        logger.info(f"天气数据已更新：{city}")

    return text or ""
