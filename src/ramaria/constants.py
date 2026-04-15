"""
src/ramaria/constants.py — 项目级静态常量

职责：
    只放跨模块共享的、纯静态的定义。
    不放运行时参数（→ src/ramaria/config.py）
    不放 Prompt 文本（→ 各自负责的模块）
    不放任何业务逻辑
"""

from __future__ import annotations


# =============================================================================
# 用户画像字段定义
# =============================================================================

PROFILE_FIELDS: dict[str, str] = {
    "basic_info":      "基础信息",
    "personal_status": "近期状态",
    "interests":       "兴趣爱好",
    "social":          "社交情况",
    "history":         "历史事件",
    "recent_context":  "近期背景",
}

VALID_FIELD_KEYS: set[str] = set(PROFILE_FIELDS.keys())

PROFILE_FIELD_LIST: list[tuple[str, str]] = list(PROFILE_FIELDS.items())