"""
src/ramaria/adapters/mcp/permissions.py — MCP 工具权限边界配置

设计思路：
    本文件是 MCP Server 的权限控制中枢。
    每个 Tool 都在下方的 TOOL_PERMISSIONS 字典中声明是否默认开启。

    分层权限策略：
      · 只读工具（read）         — 默认全部开启，无副作用，安全
      · 消息写入（write_message） — 默认开启，写 L0 但不触发摘要，可控
      · 摘要触发（write_trigger） — 默认开启，手动 trigger 才提炼，可控
      · 画像写入（write_profile） — 默认关闭，直接修改长期画像，高风险

修改方法：
    需要开启某个工具时，将对应的值改为 True 并重启 MCP Server。
    也可以在运行时通过环境变量覆盖（详见 _load_env_overrides 函数）。

环境变量覆盖（可选，部署时灵活控制）：
    RAMARIA_MCP_ENABLE_UPDATE_PROFILE=1   开启画像写入
    RAMARIA_MCP_DISABLE_TRIGGER_L1=1      禁用 L1 触发
    格式：RAMARIA_MCP_ENABLE_{TOOL_NAME_UPPER} / RAMARIA_MCP_DISABLE_{TOOL_NAME_UPPER}

"""

import os


# =============================================================================
# 工具权限默认配置
# key   — Tool 名称，与 server.py 注册的名称保持一致
# value — True 表示默认开启，False 表示默认关闭
# =============================================================================

TOOL_PERMISSIONS: dict[str, bool] = {
    # ── 只读工具（无副作用，全部默认开启）──
    "search_memory":        True,
    "get_profile":          True,
    "get_recent_context":   True,
    "get_index_stats":      True,
    "get_pending_sessions": True,

    # ── 消息写入（写 L0，不触发摘要，默认开启）──
    "save_message":         True,

    # ── 摘要触发（触发 L1 生成，默认开启）──
    "trigger_l1":           True,

    # ── 画像写入（直接修改 L3，高风险，默认关闭）──
    "update_profile":       False,
}


def _load_env_overrides(permissions: dict[str, bool]) -> dict[str, bool]:
    """
    从环境变量读取权限覆盖，返回合并后的权限字典。

    环境变量格式：
        RAMARIA_MCP_ENABLE_{TOOL}=1   强制开启某工具（不区分大小写）
        RAMARIA_MCP_DISABLE_{TOOL}=1  强制禁用某工具（不区分大小写）

    示例：
        export RAMARIA_MCP_ENABLE_UPDATE_PROFILE=1  # 开启画像写入
        export RAMARIA_MCP_DISABLE_TRIGGER_L1=1     # 禁用 L1 触发

    优先级：环境变量 > 代码默认值。
    ENABLE 和 DISABLE 同时设置时，DISABLE 优先（更安全）。
    """
    result = dict(permissions)

    for tool_name in result:
        env_key = tool_name.upper()

        enable_val = os.environ.get(f"RAMARIA_MCP_ENABLE_{env_key}", "").strip()
        if enable_val in ("1", "true", "yes"):
            result[tool_name] = True

        disable_val = os.environ.get(f"RAMARIA_MCP_DISABLE_{env_key}", "").strip()
        if disable_val in ("1", "true", "yes"):
            result[tool_name] = False

    return result


# 运行时生效的权限表（已合并环境变量覆盖）
EFFECTIVE_PERMISSIONS: dict[str, bool] = _load_env_overrides(TOOL_PERMISSIONS)


def is_allowed(tool_name: str) -> bool:
    """
    检查指定工具是否有执行权限。

    参数：
        tool_name — Tool 名称，与 TOOL_PERMISSIONS 的 key 一致

    返回：
        True  — 该工具已启用，可以执行
        False — 该工具已禁用，应返回权限错误
    """
    return EFFECTIVE_PERMISSIONS.get(tool_name, False)


def get_permission_summary() -> dict:
    """
    返回当前所有工具的权限状态摘要，用于调试和日志记录。

    返回示例：
        {
            "search_memory":  {"enabled": True,  "category": "read"},
            "update_profile": {"enabled": False, "category": "write_profile"},
        }
    """
    categories = {
        "search_memory":        "read",
        "get_profile":          "read",
        "get_recent_context":   "read",
        "get_index_stats":      "read",
        "get_pending_sessions": "read",
        "save_message":         "write_message",
        "trigger_l1":           "write_trigger",
        "update_profile":       "write_profile",
    }

    return {
        name: {
            "enabled":  enabled,
            "category": categories.get(name, "unknown"),
        }
        for name, enabled in EFFECTIVE_PERMISSIONS.items()
    }