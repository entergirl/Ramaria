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


# =============================================================================
# 对话路由关键词定义
# =============================================================================
#
# 用户画像冲突处理关键词：
#   用户回复中包含这些词时，判断是确认更新还是忽略冲突
#   使用包含匹配（substring），兼容"好啊"/"嗯嗯"等变体
#

# 用户确认"更新画像"的关键词
RESOLVE_KEYWORDS: set[str] = {
    "更新", "接受", "对", "是的", "没错", "update",
    "是", "确认", "合并", "好", "好的", "行", "可以", "嗯",
}

# 用户选择"忽略冲突"的关键词
IGNORE_KEYWORDS: set[str] = {
    "忽略", "不用", "不", "算了", "保持", "ignore",
    "不是", "分开", "不要", "不行", "no",
}

# 超过此长度的消息 → 视为正常对话，不拦截
CONFLICT_REPLY_MAX_LEN: int = 20