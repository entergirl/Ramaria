"""
src/ramaria/tools/tool_registry.py — 工具意图检测与分发中心

职责：
    1. 通过语义相似度（嵌入模型）判断用户消息是否需要调用感知工具
    2. 分发到对应工具函数，收集结果
    3. 返回结构化的 tool_results 字典，供 _build_context 注入 Block B

对外接口：
    resolve_tool_results(user_message: str) -> dict
        分析用户消息，按需调用工具，返回结果字典。
        结果字典结构：
            {
                "hardware": str | None,   # 硬件状态文本，未触发时为 None
                "fs_scan":  str | None,   # 文件扫描文本，未触发时为 None
            }

意图检测机制（方案Z）：
    预定义若干「意图示例句」，用嵌入模型计算用户消息与示例句的余弦相似度，
    超过阈值时判定为该工具的触发意图。

    · 嵌入模型直接加载 SentenceTransformer 模型
    · 避免重复加载模型，首次调用时缓存嵌入向量
    · 嵌入模型不可用时降级为关键词匹配兜底

防抖设计：
    同一 session（进程生命周期内）硬件感知最多每 60 秒触发一次，
    避免高频对话反复采集拖慢响应。
    使用内存变量（重启后重置），不写入数据库。
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ramaria.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 意图检测配置
# =============================================================================

# 语义相似度触发阈值（余弦相似度，越高越严格）
# 0.60 在实践中对中文意图句表现较好，可根据实际效果在 config.py 中调整
INTENT_THRESHOLD = 0.60

# 硬件感知防抖间隔（秒）
HARDWARE_DEBOUNCE_SECONDS = 60

# ── 硬件感知意图示例句 ──────────────────────────────────────────────────────
# 覆盖多种表达方式：直接询问、间接暗示、专业术语
_HARDWARE_INTENT_EXAMPLES: list[str] = [
    "现在电脑CPU占用多少",
    "查一下硬件状态",
    "内存够不够用",
    "电脑跑得动吗",
    "现在机器负载怎么样",
    "CPU温度高不高",
    "内存还剩多少",
    "电池还有多少电",
    "电脑卡了",
    "程序跑得很慢",
    "系统快崩了",
    "内存不够了",
    "CPU使用率是多少",
    "帮我看看电脑状态",
    "现在有哪些进程在跑",
    "GPU占用怎么样",
    "跑模型的时候CPU怎么样",
]

# ── 文件扫描意图示例句 ──────────────────────────────────────────────────────
# 文件扫描需要用户提供路径，所以示例句都带有路径的意象
_FS_SCAN_INTENT_EXAMPLES: list[str] = [
    "帮我看看这个目录里有什么文件",
    "扫描一下这个文件夹",
    "列出目录结构",
    "看看这个路径下有什么",
    "这个文件夹里有多少文件",
    "帮我看看项目目录",
    "查看一下这个目录",
    "列出文件树",
]

# ── 天气查询意图示例句 ──────────────────────────────────────────────────────
# 覆盖多种表达方式：温度询问、天气状况、降水预测、穿着建议
# 避免歧义：移除了与情绪相关的表达，确保都是明确的天气查询
_WEATHER_INTENT_EXAMPLES: list[str] = [
    "今天天气怎么样",
    "现在外面多少度",
    "今天气温是多少",
    "明天会下雨吗",
    "今天会下雪吗",
    "今晚会不会下雨",
    "现在外面是晴天还是阴天",
    "今天风大吗",
    "今天有雾霾吗",
    "空气质量怎么样",
    "湿度是多少",
    "会打雷吗",
    "今天适合跑步吗",
    "出门要不要带伞",
    "现在需要穿外套吗",
    "今天太阳大吗",
]


# =============================================================================
# 嵌入向量缓存
# =============================================================================

# 缓存结构：{"hardware": np.ndarray, "fs_scan": np.ndarray}
# 每类工具的示例句向量取平均，作为该工具的「意图向量」
_intent_vectors: dict[str, Optional[np.ndarray]] = {
    "hardware": None,
    "fs_scan":  None,
    "weather":  None,
}

# 是否已完成初始化（避免重复计算）
_vectors_initialized = False


def _get_embedding_model():
    """直接加载 SentenceTransformer 模型，不走 chromadb 封装。"""
    try:
        from sentence_transformers import SentenceTransformer
        from ramaria.config import EMBEDDING_MODEL
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        logger.warning(f"嵌入模型加载失败，将使用关键词兜底 — {e}")
        return None


def _encode_text(text: str, model) -> Optional[np.ndarray]:
    """用 SentenceTransformer 直接编码，归一化后返回。"""
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        logger.warning(f"文本编码失败：{text[:30]}… — {e}")
        return None


def _build_intent_vectors() -> None:
    """
    预计算各工具意图的平均向量，缓存到 _intent_vectors。
    首次调用 resolve_tool_results 时触发，之后复用缓存。
    """
    global _vectors_initialized

    model = _get_embedding_model()
    if model is None:
        _vectors_initialized = True   # 标记为已初始化（降级模式），避免重复尝试
        return

    def _mean_vector(examples: list[str]) -> Optional[np.ndarray]:
        """计算多个示例句向量的平均，作为该意图的代表向量。"""
        vecs = []
        for ex in examples:
            v = _encode_text(ex, model)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None
        mean = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    _intent_vectors["hardware"] = _mean_vector(_HARDWARE_INTENT_EXAMPLES)
    _intent_vectors["fs_scan"]  = _mean_vector(_FS_SCAN_INTENT_EXAMPLES)
    _intent_vectors["weather"] = _mean_vector(_WEATHER_INTENT_EXAMPLES)

    logger.debug("工具意图向量初始化完成")
    _vectors_initialized = True


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个已归一化向量的余弦相似度（直接点积）。"""
    return float(np.dot(v1, v2))


# =============================================================================
# 关键词兜底（嵌入模型不可用时）
# =============================================================================

_HARDWARE_KEYWORDS: set[str] = {
    "cpu", "gpu", "内存", "内存占用", "cpu占用", "负载", "卡", "慢", "崩",
    "电池", "进程", "硬件", "温度", "显存", "ram", "memory",
}

_FS_SCAN_KEYWORDS: set[str] = {
    "目录", "文件夹", "文件树", "扫描", "列出", "路径", "目录结构",
}


_WEATHER_KEYWORDS: set[str] = {
    "天气", "气温", "温度", "下雨", "雨伞", "下雪", "雪", "晴天",
    "阴天", "多云", "雾", "雾霾", "空气质量", "预报", "太阳",
    "雷暴", "风", "湿度", "体感温度", "冷", "热", "降温", "升温",
    "暴雨", "户外", "出门", "穿外套", "穿什么",
}


def _keyword_match(message: str, keywords: set[str]) -> bool:
    """关键词兜底匹配，任意一个词命中即返回 True。"""
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in keywords)


# =============================================================================
# 防抖状态（内存变量，重启后重置）
# =============================================================================

# 上次硬件感知触发的时间戳（Unix 时间戳，0 表示从未触发）
_last_hardware_trigger_time: float = 0.0


def _hardware_debounce_ok() -> bool:
    """
    检查硬件感知是否已过防抖冷却期。

    返回：
        True  — 可以触发
        False — 冷却中，跳过本次
    """
    global _last_hardware_trigger_time
    now = time.time()
    if now - _last_hardware_trigger_time >= HARDWARE_DEBOUNCE_SECONDS:
        return True
    remaining = int(HARDWARE_DEBOUNCE_SECONDS - (now - _last_hardware_trigger_time))
    logger.debug(f"硬件感知防抖冷却中，剩余 {remaining} 秒")
    return False


def _update_hardware_trigger_time() -> None:
    """更新硬件感知的最后触发时间戳。"""
    global _last_hardware_trigger_time
    _last_hardware_trigger_time = time.time()


# =============================================================================
# 意图判断（核心逻辑）
# =============================================================================

def _should_trigger_hardware(message: str) -> bool:
    """
    判断用户消息是否有硬件感知意图。

    优先使用语义相似度（嵌入模型），不可用时降级为关键词匹配。
    防抖检查放在此函数之外（resolve_tool_results 调用前检查）。
    """
    intent_vec = _intent_vectors.get("hardware")

    if intent_vec is not None:
        model = _get_embedding_model()
        if model is not None:
            msg_vec = _encode_text(message, model)
            if msg_vec is not None:
                sim = _cosine_similarity(msg_vec, intent_vec)
                logger.debug(f"硬件意图相似度：{sim:.4f}（阈值 {INTENT_THRESHOLD}）")
                return sim >= INTENT_THRESHOLD

    # 降级：关键词匹配
    return _keyword_match(message, _HARDWARE_KEYWORDS)


def _should_trigger_fs_scan(message: str) -> bool:
    """
    判断用户消息是否有文件扫描意图。

    文件扫描需要用户提供路径，所以语义匹配只是「第一关」，
    路径提取失败时不会实际触发扫描。
    """
    intent_vec = _intent_vectors.get("fs_scan")

    if intent_vec is not None:
        model = _get_embedding_model()
        if model is not None:
            msg_vec = _encode_text(message, model)
            if msg_vec is not None:
                sim = _cosine_similarity(msg_vec, intent_vec)
                logger.debug(f"文件扫描意图相似度：{sim:.4f}（阈值 {INTENT_THRESHOLD}）")
                return sim >= INTENT_THRESHOLD

    # 降级：关键词匹配
    return _keyword_match(message, _FS_SCAN_KEYWORDS)


def _should_trigger_weather(message: str) -> bool:
    """
    判断用户消息是否有天气查询意图。

    与硬件、文件扫描不同，天气查询没有防抖限制，
    因为天气数据本身有 10 分钟缓存，不会频繁请求网络。
    """
    intent_vec = _intent_vectors.get("weather")

    if intent_vec is not None:
        ef = _get_embedding_model()
        if ef is not None:
            msg_vec = _encode_text(message, ef)
            if msg_vec is not None:
                sim = _cosine_similarity(msg_vec, intent_vec)
                logger.debug(f"天气意图相似度：{sim:.4f}（阈值 {INTENT_THRESHOLD}）")
                return sim >= INTENT_THRESHOLD

    # 降级：关键词匹配
    return _keyword_match(message, _WEATHER_KEYWORDS)


# =============================================================================
# 对外接口
# =============================================================================

def resolve_tool_results(user_message: str) -> dict:
    """
    分析用户消息，按需调用感知工具，返回结果字典。

    调用时机：chat.py 的 _build_context() 在构建 Block B 之前调用此函数。

    参数：
        user_message — 用户发送的消息文本（防抖合并后的版本）

    返回：
        dict，结构如下：
        {
            "hardware": str | None,   # 硬件状态文本，未触发时为 None
            "fs_scan":  str | None,   # 文件扫描文本，未触发时为 None
            "weather":  str | None,   # 天气查询文本，未触发时为 None
        }

    设计：
        · 任何工具调用失败只记日志，不抛出异常
        · 两个工具相互独立，一个失败不影响另一个
    """
    results: dict = {
        "hardware": None,
        "fs_scan":  None,
        "weather":  None,
    }

    if not user_message or not user_message.strip():
        return results

    # 首次调用时初始化意图向量（懒加载，避免服务启动时阻塞）
    if not _vectors_initialized:
        _build_intent_vectors()

    # ── 硬件感知 ──────────────────────────────────────────────────────────
    try:
        if _should_trigger_hardware(user_message) and _hardware_debounce_ok():
            from ramaria.tools.hardware_monitor import get_hardware_stats
            stats = get_hardware_stats()
            if stats:
                results["hardware"] = stats
                _update_hardware_trigger_time()
                logger.info("硬件感知工具已触发")
    except Exception as e:
        logger.warning(f"硬件感知工具调用失败 — {e}")

    # ── 文件系统扫描 ──────────────────────────────────────────────────────
    try:
        if _should_trigger_fs_scan(user_message):
            from ramaria.tools.fs_scanner import (
                extract_path_from_message,
                scan_directory,
            )
            path = extract_path_from_message(user_message)
            if path:
                scan_result = scan_directory(path)
                results["fs_scan"] = scan_result
                logger.info(f"文件扫描工具已触发，路径：{path}")
            else:
                # 检测到扫描意图但没有路径，返回提示信息
                results["fs_scan"] = (
                    "[提示] 用户似乎想扫描某个目录，但我没能提取到路径。"
                    "请回复：你可以把路径用引号括起来告诉我，比如扫描 \"F:\\你的路径\""
                )
                logger.debug("文件扫描意图命中但未找到路径，已添加提示信息")
    except Exception as e:
        logger.warning(f"文件扫描工具调用失败 — {e}")

    # ── 天气查询 ──────────────────────────────────────────────────────────
    try:
        if _should_trigger_weather(user_message):
            from ramaria.tools.weather import get_weather
            weather_text = get_weather()
            if weather_text:
                results["weather"] = weather_text
                logger.info("天气查询工具已触发")
    except Exception as e:
        logger.warning(f"天气查询工具调用失败 — {e}")
        
    return results
