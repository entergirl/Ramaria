"""
共享的记忆衰减计算逻辑。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ramaria.config import (
    MEMORY_DECAY_ENABLE_ACCESS_BOOST,
    MEMORY_DECAY_RECENT_BOOST_DAYS,
    MEMORY_DECAY_RECENT_BOOST_FLOOR,
    SALIENCE_DECAY_MULTIPLIER,
)
from ramaria.logger import get_logger

logger = get_logger(__name__)


def calc_decay_r(
    created_at_str: str,
    decay_s: float,
    last_accessed_at_str: str | None = None,
    salience: float | None = None,
) -> float:
    """根据 Ebbinghaus 遗忘曲线计算记忆保留率 R。"""
    now = datetime.now(timezone.utc)

    try:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        t_days = max((now - created_at).total_seconds() / 86400, 0)
    except (ValueError, TypeError):
        logger.warning(f"无法解析 created_at={created_at_str!r}，衰减跳过（R=1.0）")
        return 1.0

    s_val = salience if salience is not None else 0.5
    s_val = max(0.0, min(1.0, s_val))
    decay_s_adjusted = decay_s * (1.0 + s_val * SALIENCE_DECAY_MULTIPLIER)
    decay_r = math.exp(-t_days / decay_s_adjusted)

    if MEMORY_DECAY_ENABLE_ACCESS_BOOST and last_accessed_at_str:
        try:
            last_accessed = datetime.fromisoformat(last_accessed_at_str)
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            days_since = (now - last_accessed).total_seconds() / 86400
            if days_since <= MEMORY_DECAY_RECENT_BOOST_DAYS:
                decay_r = max(decay_r, MEMORY_DECAY_RECENT_BOOST_FLOOR)
        except (ValueError, TypeError):
            pass

    return round(decay_r, 4)
