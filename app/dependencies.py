"""
app/dependencies.py — 全局单例与共享依赖

职责：
    · 持有 session_manager 和 router 两个全局单例
    · 持有 WebSocket 连接池 _ws_connections
    · 暴露 ws_broadcast() 和 is_user_online() 供 PushScheduler 注入
    · 所有路由模块从此文件导入单例，确保全局唯一

设计原则：
    此文件不 import 任何路由模块，只被路由模块单向依赖，
    避免循环导入。
"""

from ramaria.core.session_manager import SessionManager
from ramaria.core.router import Router

# =============================================================================
# 全局单例
# =============================================================================

# session_manager：管理对话 session 的生命周期
# 在 main.py 的 lifespan startup 阶段调用 .start()
session_manager = SessionManager()

# router：任务路由，判断消息走本地模型还是 Claude API
router = Router()

# =============================================================================
# WebSocket 连接池
# =============================================================================
#
# key   — WebSocket 实例
# value — 该连接当前绑定的 session_id（连接建立后赋值，初始为 None）
# 使用字典而非集合：便于按连接快速查找对应的 session_id
_ws_connections: dict = {}


# =============================================================================
# WebSocket 广播与在线状态
# =============================================================================

async def ws_broadcast(data: dict) -> None:
    """
    向所有当前在线的 WebSocket 客户端广播消息。

    PushScheduler 触发主动推送时调用此函数。
    如果连接池为空（用户离线），调用方应改为写入 pending_push 表。

    参数：
        data — 要广播的数据字典，必须包含 "type" 字段
    """
    from app.routes.chat import _ws_send

    for ws in list(_ws_connections.keys()):
        await _ws_send(ws, data)


def is_user_online() -> bool:
    """
    判断当前是否有用户在线（连接池非空）。
    PushScheduler 在决定直接推送还是暂存时调用此函数。

    返回：bool — True 表示至少有一个 WebSocket 连接存在
    """
    return len(_ws_connections) > 0