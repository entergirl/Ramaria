"""
app/dependencies.py — 全局单例与共享依赖

职责：
    · 持有 session_manager 和 router 两个全局单例
    · 持有 WebSocket 连接池 _ws_connections（线程安全）
    · 暴露 ws_broadcast() 和 is_user_online() 供 PushScheduler 注入
    · 生成并管理 admin API key
    · 所有路由模块从此文件导入单例，确保全局唯一

设计原则：
    此文件不 import 任何路由模块，只被路由模块单向依赖，
    避免循环导入。
"""

import asyncio
import secrets
import threading

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

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
# WebSocket 连接池（线程安全）
# =============================================================================
#
# key   — WebSocket 实例
# value — 该连接当前绑定的 session_id（连接建立后赋值，初始为 None）
# 使用字典而非集合：便于按连接快速查找对应的 session_id
#
# _ws_lock 保护所有对 _ws_connections 的读写：
#   - 异步协程（WebSocket endpoint）写入
#   - 后台线程（PushScheduler）读取 keys
#   避免 dict changed size during iteration
_ws_connections: dict = {}
_ws_lock = threading.Lock()

# 主事件循环引用，供 PushScheduler 等后台线程使用
# run_coroutine_threadsafe() 将协程调度到主循环，避免跨事件循环操作 WebSocket
_main_event_loop: "asyncio.AbstractEventLoop | None" = None


# =============================================================================
# Admin API Key 认证
# =============================================================================

_admin_api_key: str = secrets.token_urlsafe(32)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_admin_api_key() -> str:
    """返回当前会话的 admin API key（供前端注入请求头）。"""
    return _admin_api_key


def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """保存主事件循环引用，供后台线程使用 run_coroutine_threadsafe。"""
    global _main_event_loop
    _main_event_loop = loop


def get_main_event_loop() -> "asyncio.AbstractEventLoop | None":
    """获取主事件循环引用。"""
    return _main_event_loop


async def verify_admin_key(key: str = Security(API_KEY_HEADER)):
    """验证 API Key，失败返回 401。用于保护 admin/memory 写端点。"""
    if not key or key != _admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


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

    with _ws_lock:
        targets = list(_ws_connections.keys())

    for ws in targets:
        await _ws_send(ws, data)


def is_user_online() -> bool:
    """
    判断当前是否有用户在线（连接池非空）。
    PushScheduler 在决定直接推送还是暂存时调用此函数。

    返回：bool — True 表示至少有一个 WebSocket 连接存在
    """
    with _ws_lock:
        return len(_ws_connections) > 0