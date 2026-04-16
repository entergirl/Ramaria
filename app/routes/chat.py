"""
app/routes/chat.py — 核心对话路由

包含接口：
    POST /chat     — 核心对话（HTTP 兼容接口，保留）
    POST /save     — 手动保存当前 session
    WS   /ws       — WebSocket 实时双向通信（主对话通道）

辅助函数（模块内部使用）：
    _build_context()        — 组装 RAG 上下文字典
    _format_rag_results()   — 格式化 RAG 检索结果为可注入文本
    _call_local()           — 调用本地模型
    _handle_local()         — 本地路由公共逻辑（构建 prompt + 调用 + 保存）
    _detect_conflict_action() — 检测用户消息是否是冲突回复
    _ws_send()              — 向单个 WebSocket 客户端发送数据
"""

import asyncio
import re as _re
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ramaria.config import RETRIEVAL_WEIGHT_L1, RETRIEVAL_WEIGHT_L2
from ramaria.core.llm_client import call_local_chat
from ramaria.core.prompt_builder import build_system_prompt
from ramaria.memory.conflict_checker import get_conflict_question, handle_conflict_reply
from ramaria.storage.database import (
    get_current_profile,
    get_latest_l1,
    get_messages,
    get_messages_as_dicts,
    get_pending_pushes,
    get_recent_l2,
    get_setting,
    mark_push_sent,
    save_message,
)

from ramaria.constants import PROFILE_FIELD_LIST
from ramaria.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# 数据模型
# =============================================================================

class ChatRequest(BaseModel):
    """POST /chat 请求体。"""
    content: str


class ChatResponse(BaseModel):
    """
    /chat 接口的响应体。

    mode 字段说明：
        "local"   — 由本地模型生成
        "online"  — 由云端 API 生成
        "confirm" — 等待用户确认是否调用云端 API
    """
    reply:      str
    session_id: int
    mode:       str


class ToggleRequest(BaseModel):
    """POST /router/toggle 请求体。"""
    online: bool


# =============================================================================
# 冲突回复检测
# =============================================================================

# 用户确认"更新画像"的关键词（包含匹配）
_RESOLVE_KEYWORDS = {"更新", "接受", "对", "是的", "没错", "update", "是", "确认", "合并"}

# 用户选择"忽略冲突"的关键词（包含匹配）
_IGNORE_KEYWORDS  = {"忽略", "不用", "不", "算了", "保持", "ignore", "不是", "分开"}

# 超过此长度且未命中关键词 → 视为正常对话
_CONFLICT_REPLY_MAX_LEN = 20


def _detect_conflict_action(text: str) -> str | None:
    """
    检测用户消息是否是对冲突询问的回复，返回对应 action。

    返回：
        "resolve" — 用户确认接受新内容
        "ignore"  — 用户选择忽略冲突
        None      — 不是冲突回复，交给正常对话流程
    """
    t = text.lower().strip()

    if any(kw in t for kw in _RESOLVE_KEYWORDS):
        return "resolve"
    if any(kw in t for kw in _IGNORE_KEYWORDS):
        return "ignore"

    # 超长消息视为正常对话，不拦截
    if len(text) > _CONFLICT_REPLY_MAX_LEN:
        return None

    return None


# =============================================================================
# RAG 结果格式化
# =============================================================================

def _sort_key_l2(hit, weight_l2: float):
    """L2 排序键：处理 adjusted_distance 可能为 None 的情况。"""
    dist = hit.get("adjusted_distance") or hit.get("distance") or 1.0
    return dist * weight_l2


def _sort_key_l1(hit, weight_l1: float):
    """L1 排序键：处理 adjusted_distance 可能为 None 的情况。"""
    dist = hit.get("adjusted_distance") or hit.get("distance") or 1.0
    return dist * weight_l1


def _format_rag_results(
    rag_result: dict,
    weight_l2: float = RETRIEVAL_WEIGHT_L2,
    weight_l1: float = RETRIEVAL_WEIGHT_L1,
) -> str | None:
    """
    将 retrieve_combined() 的返回结果格式化为可注入 prompt 的纯文本。

    块内按 final_score = adjusted_distance × layer_weight 升序排列，
    分数越小越相关，越靠前展示。

    L2 块在前（宏观背景），L1 块在后（具体细节）。

    参数：
        rag_result — retrieve_combined() 的返回值：{"l2": [...], "l1": [...]}
        weight_l2  — L2 层级权重系数（< 1.0 表示 L2 更容易排前面）
        weight_l1  — L1 层级权重系数

    返回：
        str  — 格式化后的多行文本
        None — L2 和 L1 均无命中时返回 None
    """
    l2_hits = rag_result.get("l2", [])
    l1_hits = rag_result.get("l1", [])

    if not l2_hits and not l1_hits:
        return None

    lines = []

    # ── L2 块 ──
    if l2_hits:
        lines.append("[语义相关 · L2 时间段摘要]")

        l2_sorted = sorted(
            l2_hits,
            key=lambda hit: _sort_key_l2(hit, weight_l2),
        )

        for hit in l2_sorted:
            adj_dist    = hit.get("adjusted_distance") or hit.get("distance") or 1.0
            final_score = adj_dist * weight_l2
            doc         = hit.get("document", "").strip()
            meta        = hit.get("metadata", {})

            period_start = meta.get("period_start", "")[:7]
            period_end   = meta.get("period_end",   "")[:7]
            if period_start and period_end and period_start != period_end:
                date_prefix = f"{period_start} ~ {period_end}"
            elif period_start:
                date_prefix = period_start
            else:
                date_prefix = ""

            # 去掉 doc 内置的 [...] 前缀，避免重复显示日期
            doc_clean = _re.sub(r'^\[.*?\]\s*', '', doc)

            if date_prefix:
                line = f"{date_prefix} {doc_clean}（加权分数：{final_score:.2f}）"
            else:
                line = f"{doc_clean}（加权分数：{final_score:.2f}）"

            lines.append(line)

        lines.append("")  # L2/L1 之间空一行

    # ── L1 块 ──
    if l1_hits:
        lines.append("[语义相关 · L1 单次摘要]")

        l1_sorted = sorted(
            l1_hits,
            key=lambda hit: _sort_key_l1(hit, weight_l1),
        )

        for hit in l1_sorted:
            adj_dist    = hit.get("adjusted_distance") or hit.get("distance") or 1.0
            final_score = adj_dist * weight_l1
            doc         = hit.get("document", "").strip()
            meta        = hit.get("metadata", {})

            created_at = meta.get("created_at", "")[:10]
            doc_clean  = _re.sub(r'^\[.*?\]\s*', '', doc)

            if created_at:
                line = f"{created_at} {doc_clean}（加权分数：{final_score:.2f}）"
            else:
                line = f"{doc_clean}（加权分数：{final_score:.2f}）"

            lines.append(line)

    return "\n".join(lines).strip()


# =============================================================================
# 上下文组装
# =============================================================================

def _build_context(session_id: int, user_message: str | None = None) -> dict:
    """
    组装传给 build_system_prompt() 的 context 字典。

    参数：
        session_id   — 当前活跃 session 的 ID
        user_message — 用户本次消息文本。
                       有值时触发 RAG 语义检索；
                       None 时跳过 RAG，纯走时间序（降级安全）

    返回：
        context 字典，结构与 prompt_builder.PromptBuilder.build() 一致：
        {
            "last_session_time": datetime | None,
            "l3_profile":        str | None,
            "retrieved_l1l2":    str | None,
            "raw_fragments":     None,          # 预留，暂不使用
            "session_id":        int,
            "session_index":     None,          # 不显示不准确的编号
        }
    """
    # ── L3 用户长期画像 ──
    profile_dict = get_current_profile()
    if profile_dict:
        lines = []
        for key, label in PROFILE_FIELD_LIST:
            val = profile_dict.get(key, "").strip()
            if val:
                lines.append(f"{label}：{val}")
        l3_profile = "\n".join(lines) if lines else None
    else:
        l3_profile = None

    # ── 时间序内容：近期 L2（最多3条）+ 最新 L1（1条）──
    # 无论是否有 RAG 结果，时间序都会追加在后面，
    # 确保模型感知"最近发生了什么"
    time_seq_parts = []

    l2_rows = get_recent_l2(limit=3)
    for row in l2_rows:
        date_str = row["created_at"][:10]
        kw       = f"（{row['keywords']}）" if row["keywords"] else ""
        time_seq_parts.append(f"[时间段摘要 {date_str}] {row['summary']}{kw}")

    l1_row = get_latest_l1()
    if l1_row:
        tp   = l1_row["time_period"] or ""
        atm  = l1_row["atmosphere"]  or ""
        meta = f"（{tp}，{atm}）" if tp or atm else ""
        time_seq_parts.append(f"[最近一次对话] {l1_row['summary']}{meta}")

    # ── 上次 session 结束时间（跨日检测用）──
    last_session_time = None
    if l1_row and l1_row["created_at"]:
        try:
            dt = datetime.fromisoformat(l1_row["created_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            last_session_time = dt
        except ValueError:
            pass

    # ── RAG 语义检索 + 与时间序内容融合 ──
    #
    # 融合规则：
    #   有 RAG + 有时间序 → RAG 在前，"── 近期动态 ──" 分隔，时间序在后
    #   有 RAG + 无时间序 → 纯 RAG 文本
    #   无 RAG + 有时间序 → 纯时间序（降级路径）
    #   两者都无          → None
    retrieved_l1l2 = None

    if user_message:
        rag_text = None
        try:
            from ramaria.storage.vector_store import retrieve_combined
            rag_result = retrieve_combined(user_message)
            rag_text   = _format_rag_results(
                rag_result,
                weight_l2 = RETRIEVAL_WEIGHT_L2,
                weight_l1 = RETRIEVAL_WEIGHT_L1,
            )

            if rag_text:
                logger.debug(
                    f"RAG 命中：L2={len(rag_result.get('l2', []))} 条，"
                    f"L1={len(rag_result.get('l1', []))} 条"
                )
            else:
                logger.debug("RAG 无相关结果，降级到纯时间序")

        except Exception as e:
            logger.warning(f"RAG 检索失败，降级到纯时间序 — {e}")
            rag_text = None

        if rag_text and time_seq_parts:
            retrieved_l1l2 = (
                rag_text
                + "\n\n── 近期动态 ──\n"
                + "\n".join(time_seq_parts)
            )
        elif rag_text:
            retrieved_l1l2 = rag_text
        elif time_seq_parts:
            retrieved_l1l2 = "\n".join(time_seq_parts)

    else:
        # user_message=None（/save 等无消息场景），跳过 RAG
        retrieved_l1l2 = "\n".join(time_seq_parts) if time_seq_parts else None

    return {
        "last_session_time": last_session_time,
        "l3_profile":        l3_profile,
        "retrieved_l1l2":    retrieved_l1l2,
        "raw_fragments":     None,
        "session_id":        session_id,
        "session_index":     None,
    }


# =============================================================================
# 本地模型调用封装
# =============================================================================

def _call_local(messages: list[dict]) -> str:
    """
    调用本地 Qwen 模型，返回回复文本。
    失败时返回固定提示文本，不向上抛出异常。
    """
    result = call_local_chat(messages, caller="chat")
    if result is None:
        return "（错误：本地模型调用失败，请确认 LM Studio 服务已启动）"
    return result


def _handle_local(session_id: int, content: str) -> ChatResponse:
    """
    统一处理走本地模型的分支。

    流程：
        1. 取出当前 session 的完整对话历史
        2. 调用 _build_context()，触发 RAG 检索并构建 System Prompt
        3. 拼装消息列表，调用本地模型
        4. 保存回复，返回 ChatResponse

    参数：
        session_id — 当前活跃 session 的 id
        content    — 用户消息文本
    """
    history = get_messages_as_dicts(session_id)
    context = _build_context(session_id, user_message=content)
    system  = build_system_prompt(context)

    msgs = [
        {"role": "system", "content": system},
        *history[:-1],
        {"role": "user",   "content": content},
    ]

    reply = _call_local(msgs)
    save_message(session_id, "assistant", reply)
    return ChatResponse(reply=reply, session_id=session_id, mode="local")


# =============================================================================
# WebSocket 工具函数
# =============================================================================

async def _ws_send(ws: WebSocket, data: dict) -> None:
    """
    向单个 WebSocket 客户端发送 JSON 数据。
    封装异常处理，发送失败时静默记录日志，不向上抛出。

    消息类型（data["type"]）说明：
        "reply" — 模型生成的对话回复
        "push"  — 主动推送消息
        "error" — 错误通知
    """
    try:
        await ws.send_json(data)
    except Exception as e:
        logger.warning(f"WebSocket 发送失败 — {e}")


# =============================================================================
# POST /chat — 核心对话接口（HTTP 版）
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    核心对话接口（HTTP 版，保留用于兼容）。
    主要对话通道已迁移到 WebSocket /ws。

    处理流程：
        1. 基础校验
        2. Session 管理
        3. 保存用户消息
        4. 冲突回复检测
        5. 冲突询问推送
        6. 路由判断 → 对应分支
    """
    from fastapi import HTTPException
    from app.dependencies import session_manager
    from app.dependencies import router as app_router

    if not req.content.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    session_id = session_manager.on_message()
    save_message(session_id, "user", req.content)

    # 冲突回复检测
    conflict_action = _detect_conflict_action(req.content.strip())
    if conflict_action is not None:
        cr = get_conflict_question()
        if cr is not None:
            conflict_id = cr["conflict_id"]
            reply = handle_conflict_reply(conflict_id, conflict_action)
            save_message(session_id, "assistant", reply)
            return ChatResponse(reply=reply, session_id=session_id, mode="local")

    # 冲突询问推送
    cr = get_conflict_question()
    if cr is not None:
        question = cr["conflict_question"]
        save_message(session_id, "assistant", question)
        return ChatResponse(reply=question, session_id=session_id, mode="local")

    # 路由判断
    result = app_router.route(req.content.strip())
    action = result["action"]

    if action == "ask_confirm":
        txt = result["text"]
        save_message(session_id, "assistant", txt)
        return ChatResponse(reply=txt, session_id=session_id, mode="confirm")

    if action in ("online", "confirm_yes"):
        reply = app_router.call_claude(result["message"])
        save_message(session_id, "assistant", reply)
        return ChatResponse(reply=reply, session_id=session_id, mode="online")

    if action == "confirm_no":
        return _handle_local(session_id, result["message"])

    return _handle_local(session_id, req.content.strip())


# =============================================================================
# POST /save — 手动保存 session
# =============================================================================

@router.post("/save")
async def save():
    """
    手动结束当前 session 并触发 L1 摘要生成。
    _build_context 调用时 user_message=None，跳过 RAG，属正常降级。
    """
    from app.dependencies import session_manager
    session_manager.force_close_current_session()
    return JSONResponse({"status": "ok"})


# =============================================================================
# WebSocket /ws — 实时双向通信
# =============================================================================

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket 主路由，处理客户端的实时双向通信。

    连接生命周期：
        1. 握手建立 → 注册连接池 → 推送积压的 pending_push 消息
        2. 消息循环 → 接收用户消息 → 存入缓冲区 → 防抖计时器到期 → 合并触发
        3. 连接断开 → 取消计时器 → 从连接池移除

    防抖逻辑：
        每个连接维护独立的 _buf 和 _timer。
        每来一条消息：追加到 _buf，取消旧计时器，创建新的 debounce_seconds 秒计时器。
        计时器到期时：合并 _buf 里的所有消息 → 走路由 → 回复。
    """
    from app.dependencies import session_manager, router as app_router, _ws_connections

    await ws.accept()
    logger.info("WebSocket 连接建立")

    _ws_connections[ws] = None

    # 每个连接独立的防抖状态
    # 使用列表包装，便于在嵌套函数里修改
    _buf:   list[str]                 = []      # 消息缓冲区
    _timer: list[asyncio.Task | None] = [None]  # 防抖计时任务

    async def _flush():
        """
        防抖计时器到期时触发：合并缓冲区消息，走路由 → 生成 → 回复。
        """
        if not _buf:
            return

        combined = "\n".join(_buf)
        _buf.clear()
        _timer[0] = None

        session_id = _ws_connections.get(ws)
        if session_id is None:
            logger.warning("_flush 触发时 session_id 为 None，跳过")
            return

        logger.debug(f"防抖触发，合并消息：{combined[:80]}")

        # ── 冲突回复检测 ──
        conflict_action = _detect_conflict_action(combined)
        if conflict_action is not None:
            cr = get_conflict_question()
            if cr is not None:
                conflict_id = cr["conflict_id"]
                reply = handle_conflict_reply(conflict_id, conflict_action)
                save_message(session_id, "assistant", reply)
                await _ws_send(ws, {
                    "type":       "reply",
                    "reply":      reply,
                    "session_id": session_id,
                    "mode":       "local",
                })
                return

        # ── 冲突询问推送 ──
        cr = get_conflict_question()
        if cr is not None:
            question = cr["conflict_question"]
            save_message(session_id, "assistant", question)
            await _ws_send(ws, {
                "type":       "reply",
                "reply":      question,
                "session_id": session_id,
                "mode":       "local",
            })
            return

        # ── 路由判断 ──
        result = app_router.route(combined)
        action = result["action"]

        if action == "ask_confirm":
            txt = result["text"]
            save_message(session_id, "assistant", txt)
            await _ws_send(ws, {
                "type":       "reply",
                "reply":      txt,
                "session_id": session_id,
                "mode":       "confirm",
            })

        elif action in ("online", "confirm_yes"):
            reply = app_router.call_claude(result["message"])
            save_message(session_id, "assistant", reply)
            await _ws_send(ws, {
                "type":       "reply",
                "reply":      reply,
                "session_id": session_id,
                "mode":       "online",
            })

        elif action == "confirm_no":
            response = _handle_local(session_id, result["message"])
            await _ws_send(ws, {
                "type":       "reply",
                "reply":      response.reply,
                "session_id": session_id,
                "mode":       response.mode,
            })

        else:
            # 默认本地
            response = _handle_local(session_id, combined)
            await _ws_send(ws, {
                "type":       "reply",
                "reply":      response.reply,
                "session_id": session_id,
                "mode":       response.mode,
            })

    try:
        # ── 用户上线：推送离线积压消息 ──
        pending_pushes = get_pending_pushes()
        if pending_pushes:
            logger.info(f"用户上线，推送 {len(pending_pushes)} 条积压消息")
            for push in pending_pushes:
                await _ws_send(ws, {
                    "type":       "push",
                    "content":    push["content"],
                    "created_at": push["created_at"],
                })
                mark_push_sent(push["id"])

        # ── 消息循环 ──
        while True:
            data = await ws.receive_json()

            msg_type = data.get("type")
            content  = data.get("content", "").strip()

            if msg_type != "chat" or not content:
                continue

            # Session 管理
            session_id = session_manager.on_message()
            _ws_connections[ws] = session_id

            # 每条消息单独存入 L0（永久保留原始消息）
            save_message(session_id, "user", content)

            # 存入缓冲区
            _buf.append(content)

            # 读取防抖时长（从数据库读，支持用户实时修改）
            try:
                debounce_sec = float(get_setting("debounce_seconds") or "3")
            except ValueError:
                debounce_sec = 3.0

            # 取消旧计时器
            if _timer[0] is not None and not _timer[0].done():
                _timer[0].cancel()

            # 创建新计时器
            async def _delayed_flush(delay: float):
                await asyncio.sleep(delay)
                await _flush()

            _timer[0] = asyncio.create_task(_delayed_flush(debounce_sec))

    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
    except Exception as e:
        logger.error(f"WebSocket 处理异常 — {e}")
    finally:
        # 连接断开时取消未触发的计时器
        if _timer[0] is not None and not _timer[0].done():
            _timer[0].cancel()
            logger.debug("已取消未触发的防抖计时器")
        _ws_connections.pop(ws, None)
        logger.debug(f"当前在线连接数：{len(_ws_connections)}")