"""
constants.py — 项目级常量定义
=====================================================================

职责：
    集中管理跨模块共享的常量，避免同一个定义散落在多处文件。
    其他模块从这里 import，不再各自维护重复副本。

当前包含：
    PROFILE_FIELDS     — 用户画像六大板块的字段名→中文标签映射
    VALID_FIELD_KEYS   — 仅含字段名的集合，用于合法性校验
    PROFILE_FIELD_LIST — 保持固定顺序的 (key, label) 列表，用于格式化输出

变更记录：
    v1 — 从 conflict_checker / profile_manager / mcp_server / prompt_builder
         四处重复定义中提取，修复审查报告 P1-3。

使用方法：
    from constants import PROFILE_FIELDS, VALID_FIELD_KEYS, PROFILE_FIELD_LIST

    # 合法性校验
    if field not in VALID_FIELD_KEYS:
        raise ValueError(f"非法字段：{field}")

    # 格式化输出（按固定顺序遍历）
    for key, label in PROFILE_FIELD_LIST:
        content = profile.get(key, "")
        ...

    # 通过字段名取中文标签
    label = PROFILE_FIELDS["basic_info"]   # → "基础信息"
"""

# =============================================================================
# 用户画像字段定义
# =============================================================================

# 字段名 → 中文标签的权威映射
# 这是唯一的数据源，其他所有模块从这里 import，不再各自维护副本。
#
# 字段说明（与 user_profile 表的 field 列合法值完全对应）：
#   basic_info      基础信息（姓名、年龄、所在地、职业等稳定属性）
#   personal_status 近期状态（当前情绪、健康、压力等动态信息）
#   interests       兴趣爱好（长期关注的领域、喜好）
#   social          社交情况（重要人际关系）
#   history         历史事件（重要经历与里程碑）
#   recent_context  近期背景（项目进展、阶段性动态）
#
# ⚠️  新增字段时，只改这里，其他文件自动同步。
#     同时需要在 init_db.py 的 user_profile 表注释里补充说明。
PROFILE_FIELDS: dict[str, str] = {
    "basic_info":      "基础信息",
    "personal_status": "近期状态",
    "interests":       "兴趣爱好",
    "social":          "社交情况",
    "history":         "历史事件",
    "recent_context":  "近期背景",
}

# 仅含字段名的集合，用于 O(1) 合法性校验
# 与 PROFILE_FIELDS.keys() 内容完全一致，单独定义是为了语义清晰
# 用法：if field not in VALID_FIELD_KEYS: ...
VALID_FIELD_KEYS: set[str] = set(PROFILE_FIELDS.keys())

# 保持固定显示顺序的 (key, label) 列表
# dict 在 Python 3.7+ 虽然有插入顺序，但显式列表更清晰、更安全
# 用法：for key, label in PROFILE_FIELD_LIST: ...
PROFILE_FIELD_LIST: list[tuple[str, str]] = list(PROFILE_FIELDS.items())