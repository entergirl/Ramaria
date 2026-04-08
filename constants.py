"""
constants.py — 项目级静态常量

职责：
    只放跨模块共享的、纯静态的定义。
    不放运行时参数（→ src/ramaria/config.py）
    不放 Prompt 文本（→ 各自负责的模块）
    不放任何业务逻辑

被以下模块引用：
    src/ramaria/memory/conflict_checker.py
    src/ramaria/memory/profile_manager.py
    src/ramaria/adapters/mcp/tools/read_tools.py
    src/ramaria/adapters/mcp/tools/write_tools.py
    src/ramaria/core/prompt_builder.py
"""

from __future__ import annotations


# =============================================================================
# 用户画像字段定义
# =============================================================================
#
# 这是唯一权威数据源。
# 新增字段时只改这里，所有引用方自动同步。
# 同时需要在 scripts/init_db.py 的 user_profile 表注释中补充说明。
#
# 字段含义：
#   basic_info      基础信息（姓名、年龄、所在地、职业等稳定属性）
#   personal_status 近期状态（当前情绪、健康、压力等动态信息）
#   interests       兴趣爱好（长期关注的领域、喜好）
#   social          社交情况（重要人际关系）
#   history         历史事件（重要经历与里程碑）
#   recent_context  近期背景（项目进展、阶段性动态）

PROFILE_FIELDS: dict[str, str] = {
    "basic_info":      "基础信息",
    "personal_status": "近期状态",
    "interests":       "兴趣爱好",
    "social":          "社交情况",
    "history":         "历史事件",
    "recent_context":  "近期背景",
}

# 字段名集合，用于 O(1) 合法性校验
# 用法：if field not in VALID_FIELD_KEYS: raise ValueError(...)
VALID_FIELD_KEYS: set[str] = set(PROFILE_FIELDS.keys())

# 保持固定顺序的 (key, label) 列表，用于格式化输出
# dict 在 Python 3.7+ 虽然保持插入顺序，但显式列表语义更清晰
# 用法：for key, label in PROFILE_FIELD_LIST: ...
PROFILE_FIELD_LIST: list[tuple[str, str]] = list(PROFILE_FIELDS.items())