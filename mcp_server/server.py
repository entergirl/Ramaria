"""
mcp_server/server.py — 珊瑚菌 MCP Server 主入口
=====================================================================

传输方式：stdio（本地进程通信）
协议版本：MCP 1.x（Anthropic 官方 SDK mcp>=1.0.0）
Python   ：3.10+

启动方式：
    # 直接运行（调试用）
    python mcp_server/server.py

    # Claude Desktop 配置方式（见本文件末尾的配置示例）
    在 claude_desktop_config.json 中配置 command + args 即可。

架构说明：
    · Server 对象统一注册两个处理器：list_tools 和 call_tool
    · list_tools 返回全部已启用工具的 Tool 定义（含参数 schema）
    · call_tool  根据工具名分发到 read_tools 或 write_tools 的对应函数

    · 每个工具函数签名统一为 fn(arguments: dict) -> str
      返回值是纯文本字符串，server.py 统一包装为 TextContent 返回给客户端

工具清单（7个）：
    只读组（默认开启）：
        search_memory       语义检索 L1/L2 向量索引
        get_profile         读取 L3 用户画像
        get_recent_context  获取最近 N 条 L1 摘要（时间序）
        get_index_stats     查看三层索引的条目统计
        get_pending_sessions 查询待生成 L1 的 session 数
    写入组：
        save_message        写入 L0 消息（默认开启）
        trigger_l1          触发 L1 摘要生成（默认开启）
        update_profile      更新 L3 画像板块（默认关闭）

依赖安装：
    pip install mcp
"""

import asyncio
import json
import sys
from pathlib import Path

# ── 将项目根目录加入 sys.path，使 MCP Server 作为独立进程时能找到项目模块 ──
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from mcp_server.permissions import EFFECTIVE_PERMISSIONS, get_permission_summary
from mcp_server.tools import read_tools, write_tools


# =============================================================================
# 工具注册表
# =============================================================================

# 每个条目描述一个 Tool 的完整信息：
#   handler    — 实际执行函数，签名为 fn(arguments: dict) -> str
#   definition — MCP Tool 对象，包含 name/description/inputSchema
#
# inputSchema 遵循 JSON Schema Draft 7，只需描述 properties 和 required。
# 参数类型说明：
#   type: "string"  — 字符串
#   type: "integer" — 整数
#   type: "number"  — 数字（含浮点）

TOOL_REGISTRY: list[dict] = [

    # ── 只读工具 ──────────────────────────────────────────────────────────

    {
        "handler": read_tools.search_memory,
        "definition": types.Tool(
            name        = "search_memory",
            description = (
                "在珊瑚菌的记忆库中进行语义检索，返回与查询最相关的历史摘要。"
                "同时搜索 L1（单次对话摘要）和 L2（时间段聚合摘要）两层，"
                "结果按语义相关度排序，已过期或不相关的内容会被自动过滤。"
                "适用场景：查找过去讨论过的话题、了解某段时间的活动规律。"
            ),
            inputSchema = {
                "type":       "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "检索文本，如「最近有没有熬夜写代码」、「情绪低落的记录」",
                    },
                    "top_k": {
                        "type":        "integer",
                        "description": "每层最多返回条数，默认使用系统配置（L1=4，L2=2），最大10",
                        "minimum":     1,
                        "maximum":     10,
                    },
                },
                "required": ["query"],
            },
        ),
    },

    {
        "handler": read_tools.get_profile,
        "definition": types.Tool(
            name        = "get_profile",
            description = (
                "读取当前生效的 L3 用户长期画像。"
                "画像按六个板块组织：基础信息、近期状态、兴趣爱好、社交情况、重要经历、近期背景。"
                "可以读取全部板块，也可以通过 field 参数只读取某一个板块。"
            ),
            inputSchema = {
                "type":       "object",
                "properties": {
                    "field": {
                        "type":        "string",
                        "description": (
                            "可选。只读取指定板块。"
                            "合法值：basic_info / personal_status / interests / "
                            "social / history / recent_context。"
                            "不传则返回全部板块。"
                        ),
                        "enum": [
                            "basic_info", "personal_status", "interests",
                            "social", "history", "recent_context",
                        ],
                    },
                },
                "required": [],
            },
        ),
    },

    {
        "handler": read_tools.get_recent_context,
        "definition": types.Tool(
            name        = "get_recent_context",
            description = (
                "获取最近 N 条 L1 对话摘要（按时间降序，最新的在前）。"
                "与 search_memory 不同，此工具是时间序查询而非语义检索，"
                "适合了解「最近发生了什么」，而非「某个话题的历史」。"
            ),
            inputSchema = {
                "type":       "object",
                "properties": {
                    "limit": {
                        "type":        "integer",
                        "description": "返回条数，默认5，最大20",
                        "minimum":     1,
                        "maximum":     20,
                        "default":     5,
                    },
                },
                "required": [],
            },
        ),
    },

    {
        "handler": read_tools.get_index_stats,
        "definition": types.Tool(
            name        = "get_index_stats",
            description = (
                "查看三层向量索引（L0/L1/L2）当前的条目数，"
                "以及 SQLite 数据库中的原始数据量。"
                "用于运维检查：确认索引是否正常写入，"
                "切换嵌入模型或重建索引后验证数量是否对齐。"
            ),
            inputSchema = {
                "type":       "object",
                "properties": {},
                "required":   [],
            },
        ),
    },

    {
        "handler": read_tools.get_pending_sessions,
        "definition": types.Tool(
            name        = "get_pending_sessions",
            description = (
                "查询有多少历史 session 尚未生成 L1 摘要。"
                "返回待处理数量、时间范围，以及前10个待处理 session 的 ID。"
                "可配合 trigger_l1 工具批量处理历史记录。"
            ),
            inputSchema = {
                "type":       "object",
                "properties": {},
                "required":   [],
            },
        ),
    },

    # ── 写入工具 ──────────────────────────────────────────────────────────

    {
        "handler": write_tools.save_message,
        "definition": types.Tool(
            name        = "save_message",
            description = (
                "向 L0 原始消息层写入一条消息，不触发任何自动后台流程。"
                "适合：将外部工具的对话记录注入珊瑚菌的记忆库，"
                "或手动补录某段重要的信息。"
                "写入后请调用 trigger_l1(session_id) 将其提炼进记忆，"
                "否则这条消息只存在于 L0，不会被检索到。"
            ),
            inputSchema = {
                "type":       "object",
                "properties": {
                    "role": {
                        "type":        "string",
                        "description": "消息角色：user（用户/烧酒）或 assistant（助手）",
                        "enum":        ["user", "assistant"],
                    },
                    "content": {
                        "type":        "string",
                        "description": "消息内容，不能为空",
                    },
                    "source_hint": {
                        "type":        "string",
                        "description": "可选，来源说明，如「来自Claude Desktop的记录」，会附在消息末尾",
                    },
                },
                "required": ["role", "content"],
            },
        ),
    },

    {
        "handler": write_tools.trigger_l1,
        "definition": types.Tool(
            name        = "trigger_l1",
            description = (
                "触发指定 session 的 L1 摘要生成。"
                "此工具会调用本地 Qwen 模型（需要 LM Studio 运行中），"
                "生成结构化摘要并写入记忆层，同时触发：画像提取、冲突检测、L2 条数检查。"
                "通常在 save_message 之后调用，将写入的消息提炼进长期记忆。"
                "注意：生成耗时10~30秒，请耐心等待。"
            ),
            inputSchema = {
                "type":       "object",
                "properties": {
                    "session_id": {
                        "type":        "integer",
                        "description": (
                            "要提炼的 session ID（整数）。"
                            "通常来自 save_message 的返回值，"
                            "或 get_pending_sessions 返回的列表。"
                        ),
                    },
                },
                "required": ["session_id"],
            },
        ),
    },

    {
        "handler": write_tools.update_profile,
        "definition": types.Tool(
            name        = "update_profile",
            description = (
                "⚠️ 高风险操作，默认禁用。"
                "直接更新 L3 用户画像的某个板块，覆盖当前内容。"
                "旧版本会保留为历史记录（不会丢失），但当前生效内容会被替换。"
                "需要在 mcp_server/permissions.py 中将 update_profile 设为 True 才能使用。"
            ),
            inputSchema = {
                "type":       "object",
                "properties": {
                    "field": {
                        "type":        "string",
                        "description": "要更新的板块名",
                        "enum": [
                            "basic_info", "personal_status", "interests",
                            "social", "history", "recent_context",
                        ],
                    },
                    "content": {
                        "type":        "string",
                        "description": "新的板块内容，将完全替换当前内容",
                    },
                },
                "required": ["field", "content"],
            },
        ),
    },
]

# 构建工具名 → 处理函数的查找表，供 call_tool 分发
_HANDLER_MAP: dict[str, callable] = {
    entry["definition"].name: entry["handler"]
    for entry in TOOL_REGISTRY
}


# =============================================================================
# MCP Server 初始化
# =============================================================================

server = Server(
    name    = "ramaria-memory",
    version = "1.0.0",
)


# =============================================================================
# 处理器注册：list_tools
# =============================================================================

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """
    返回当前已启用的工具列表。
    未启用（permissions.py 中为 False）的工具不会出现在列表中，
    这样 MCP 客户端的 UI 里就不会展示被禁用的工具。
    """
    enabled_tools = []
    for entry in TOOL_REGISTRY:
        tool_name = entry["definition"].name
        if EFFECTIVE_PERMISSIONS.get(tool_name, False):
            enabled_tools.append(entry["definition"])
    return enabled_tools


# =============================================================================
# 处理器注册：call_tool
# =============================================================================

@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict,
) -> list[types.TextContent]:
    """
    接收 MCP 客户端的工具调用请求，分发到对应的处理函数。

    执行流程：
      1. 检查工具名是否在注册表中
      2. 检查工具是否有执行权限（二次确认，write_tools 内部也会检查）
      3. 调用对应的处理函数（同步函数，在线程池中执行以避免阻塞事件循环）
      4. 将字符串结果包装为 TextContent 列表返回

    错误处理：
      · 工具不存在   → 返回错误文本（不抛异常，避免 MCP 连接断开）
      · 权限不足     → 返回权限错误文本
      · 处理函数异常 → 捕获并返回错误信息

    参数（由 MCP SDK 框架传入）：
        name      — 工具名称字符串
        arguments — 工具参数字典，已由 SDK 根据 inputSchema 校验格式
    """
    # 工具存在性检查
    if name not in _HANDLER_MAP:
        error_msg = (
            f"错误：工具 {name!r} 不存在。\n"
            f"可用工具：{', '.join(_HANDLER_MAP.keys())}"
        )
        return [types.TextContent(type="text", text=error_msg)]

    # 权限检查（已禁用的工具即使被强行调用也会被拦截）
    if not EFFECTIVE_PERMISSIONS.get(name, False):
        error_msg = (
            f"错误：工具 {name!r} 当前已禁用。\n"
            f"请在 mcp_server/permissions.py 中将 {name} 设为 True，然后重启 MCP Server。"
        )
        return [types.TextContent(type="text", text=error_msg)]

    # 调用处理函数
    # 工具函数内部调用了 database.py（同步 SQLite）和 vector_store.py（同步 Chroma），
    # 使用 asyncio.to_thread 在线程池中执行，避免阻塞 asyncio 事件循环
    handler = _HANDLER_MAP[name]
    try:
        result_text = await asyncio.to_thread(handler, arguments)
    except Exception as e:
        result_text = (
            f"工具 {name!r} 执行时发生未预期的错误：\n"
            f"  {type(e).__name__}: {e}\n"
            f"\n"
            f"请检查：\n"
            f"  1. 项目依赖是否完整（database.py / vector_store.py 是否可 import）\n"
            f"  2. assistant.db 是否存在（运行过 init_db.py）\n"
            f"  3. 如果是写入操作，LM Studio 是否正在运行"
        )

    return [types.TextContent(type="text", text=result_text)]


# =============================================================================
# 启动入口
# =============================================================================

async def main():
    """
    MCP Server 主协程。
    使用 stdio_server 作为传输层，接收来自 MCP 客户端（如 Claude Desktop）的连接。

    生命周期：
      1. stdio_server 建立 stdin/stdout 通信管道
      2. server.run() 进入消息循环，持续处理 JSON-RPC 请求
      3. 客户端断开连接时协程退出

    启动时打印权限摘要到 stderr（不影响 stdio 通信），方便调试。
    """
    # 将权限摘要输出到 stderr（stdout 用于 MCP 协议通信，不能污染）
    summary = get_permission_summary()
    enabled  = [name for name, info in summary.items() if info["enabled"]]
    disabled = [name for name, info in summary.items() if not info["enabled"]]

    print("珊瑚菌 MCP Server 启动中…", file=sys.stderr)
    print(f"  已启用工具（{len(enabled)} 个）：{', '.join(enabled)}", file=sys.stderr)
    if disabled:
        print(f"  已禁用工具（{len(disabled)} 个）：{', '.join(disabled)}", file=sys.stderr)
    print("等待 MCP 客户端连接（stdio）…", file=sys.stderr)

    # 进入 stdio 通信循环
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Claude Desktop 配置示例
# =============================================================================
#
# 将以下配置添加到 Claude Desktop 的 claude_desktop_config.json：
#
# macOS 路径：~/Library/Application Support/Claude/claude_desktop_config.json
# Windows 路径：%APPDATA%\Claude\claude_desktop_config.json
#
# {
#   "mcpServers": {
#     "ramaria": {
#       "command": "python",
#       "args": [
#         "F:/9700/demo/mcp_server/server.py"
#       ],
#       "env": {
#         "PYTHONIOENCODING": "utf-8",
#
#         // 可选：通过环境变量覆盖权限（不修改代码）
#         // "RAMARIA_MCP_ENABLE_UPDATE_PROFILE": "1"
#       }
#     }
#   }
# }
#
# Windows 注意事项：
#   · 路径使用正斜杠 / 或双反斜杠 \\
#   · command 可以改为 python 的完整路径，如 "C:/Python310/python.exe"
#   · 如果使用虚拟环境：command 改为 venv 里的 python.exe 路径
#
# 配置完成后重启 Claude Desktop，在对话框旁边会出现工具图标。
# =============================================================================
