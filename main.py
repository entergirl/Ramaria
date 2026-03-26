"""
main.py — FastAPI 应用入口
版本：0.3.3
=====================================================================

本次修改（前后端分离）：
  · 将内联 HTML 字符串提取到独立文件 static/index.html。
  · GET / 路由不再返回硬编码的 HTML 字符串，改为读取并返回
    static/index.html 的文件内容。
  · 新增 STATIC_DIR 常量指向 static/ 目录（与本文件同级）。
  · 新增 HTML_FILE 常量指向 static/index.html。
  · 其余接口和业务逻辑完全不变。

目录结构变化：
  修改前：
    demo/
    └── main.py   （HTML 内联在 Python 字符串里）

  修改后：
    demo/
    ├── main.py
    └── static/
        └── index.html

接口列表（无变化）：
  GET  /               — 对话页面（现改为从文件读取）
  POST /chat           — 核心对话接口
  POST /save           — 手动保存 session
  GET  /router/status  — 查询路由状态
  POST /router/toggle  — 切换线上/本地模式
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import (
    DEBUG,
    SERVER_HOST,
    SERVER_PORT,
)
from llm_client import call_local_chat
from database import (
    get_messages_as_dicts,
    save_message,
    get_current_profile,
    get_latest_l1,
    get_recent_l2,
    get_active_sessions,
)
from prompt_builder import build_system_prompt
from session_manager import SessionManager
from conflict_checker import get_conflict_question, handle_conflict_reply
from router import Router

from logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# 静态文件路径
# =============================================================================

# 项目根目录（main.py 所在目录）
BASE_DIR = Path(__file__).parent

# 静态文件目录：demo/static/
# 如需调整目录名，只改这一行。
STATIC_DIR = BASE_DIR / "static"

# 前端入口文件：demo/static/index.html
HTML_FILE = STATIC_DIR / "index.html"


# =============================================================================
# 全局单例
# =============================================================================

session_manager = SessionManager()
router          = Router()


# =============================================================================
# 应用生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动中…")
    session_manager.start()
    logger.info("就绪，访问 http://localhost:8000")
    yield
    logger.info("关闭中…")
    session_manager.stop()
    logger.info("已停止")


app = FastAPI(
    title       = "珊瑚菌 · 个人 AI 陪伴助手",
    description = "本地运行，支持分层记忆与任务路由",
    version     = "0.3.3",
    lifespan    = lifespan,
)


# =============================================================================
# 数据模型
# =============================================================================

class ChatRequest(BaseModel):
    content: str

class ChatResponse(BaseModel):
    reply:      str
    session_id: int
    mode:       str   # "local" | "online" | "confirm"

class ToggleRequest(BaseModel):
    online: bool   # True = 切换到线上，False = 切回本地


# =============================================================================
# 冲突回复关键词与检测函数
# =============================================================================

# 用户确认"更新画像"的关键词集合
_RESOLVE_KEYWORDS = {"更新", "接受", "对", "是的", "没错", "update"}

# 用户选择"忽略冲突"的关键词集合
_IGNORE_KEYWORDS  = {"忽略", "不用", "不", "算了", "保持", "ignore"}

# 字数兜底阈值：消息超过此长度且未命中关键词，视为正常对话而非冲突回复
_CONFLICT_REPLY_MAX_LEN = 20


def _detect_conflict_action(text):
    """
    检测用户消息是否是对冲突询问的回复，返回对应 action。

    检测策略（两步，优先级从高到低）：
      1. 关键词匹配（优先）：
           · 消息中包含确认词（"更新"/"接受"/"是的" 等）→ 返回 "resolve"
           · 消息中包含忽略词（"忽略"/"不用"/"算了" 等）→ 返回 "ignore"
      2. 字数兜底（次要）：
           · 未命中任何关键词，且消息超过 _CONFLICT_REPLY_MAX_LEN 字 → 返回 None

    参数：
        text — 用户消息文本（已去除首尾空白）

    返回：
        "resolve" — 用户确认接受新内容
        "ignore"  — 用户选择忽略冲突
        None      — 不是冲突回复，交给正常对话逻辑处理
    """
    t = text.lower().strip()

    # 第一步：关键词匹配（优先，不受字数限制）
    if any(kw in t for kw in _RESOLVE_KEYWORDS):
        return "resolve"
    if any(kw in t for kw in _IGNORE_KEYWORDS):
        return "ignore"

    # 第二步：字数兜底，超长消息直接视为正常对话
    if len(text) > _CONFLICT_REPLY_MAX_LEN:
        return None

    return None


# =============================================================================
# context 组装辅助函数
# =============================================================================

def _build_context(session_id: int, user_message: str | None = None) -> dict:
    """
    组装传给 build_system_prompt() 的 context 字典。

    从数据库读取所有动态信息，统一封装后返回。
    所有 datetime 对象统一转换为带时区版本，避免与数据库 UTC 时间戳相减时
    触发 "can't subtract offset-naive and offset-aware datetimes" 错误。

    参数：
        session_id:   当前活跃 session 的 ID。
        user_message: 用户本次消息（预留，供将来接入 RAG 检索）。

    返回：
        context 字典，结构与 prompt_builder.PromptBuilder.build() 一致。
    """
    from datetime import datetime, timezone

    # ------------------------------------------------------------------
    # L3 用户长期画像
    # ------------------------------------------------------------------
    profile_dict = get_current_profile()
    if profile_dict:
        field_labels = [
            ("basic_info",      "基础信息"),
            ("personal_status", "近期状态"),
            ("interests",       "兴趣爱好"),
            ("social",          "社交情况"),
            ("history",         "重要经历"),
            ("recent_context",  "近期背景"),
        ]
        lines = []
        for key, label in field_labels:
            val = profile_dict.get(key, "").strip()
            if val:
                lines.append(f"{label}：{val}")
        l3_profile = "\n".join(lines) if lines else None
    else:
        l3_profile = None

    # ------------------------------------------------------------------
    # 近期 L2 + 最新 L1 拼成 retrieved_l1l2
    # ------------------------------------------------------------------
    memory_parts = []

    l2_rows = get_recent_l2(limit=3)
    for row in l2_rows:
        date_str = row["created_at"][:10]
        kw = f"（{row['keywords']}）" if row["keywords"] else ""
        memory_parts.append(f"[时间段摘要 {date_str}] {row['summary']}{kw}")

    l1_row = get_latest_l1()
    if l1_row:
        tp  = l1_row["time_period"] or ""
        atm = l1_row["atmosphere"]  or ""
        meta = f"（{tp}，{atm}）" if tp or atm else ""
        memory_parts.append(f"[最近一次对话] {l1_row['summary']}{meta}")

    retrieved_l1l2 = "\n".join(memory_parts) if memory_parts else None

    # ------------------------------------------------------------------
    # 上次 session 结束时间（用于时间感知和跨日检测）
    # ------------------------------------------------------------------
    last_session_time = None
    if l1_row and l1_row["created_at"]:
        try:
            dt = datetime.fromisoformat(l1_row["created_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            last_session_time = dt
        except ValueError:
            pass

    return {
        "last_session_time": last_session_time,
        "l3_profile":        l3_profile,
        "retrieved_l1l2":    retrieved_l1l2,
        "raw_fragments":     None,
        "session_id":        session_id,
        "session_index":     session_id,
    }


# =============================================================================
# 调用本地模型
# =============================================================================

def _call_local(messages: list[dict]) -> str:
    result = call_local_chat(messages, caller="main")
    if result is None:
        return "（错误：本地模型调用失败，请确认服务已启动）"
    return result


# =============================================================================
# GET / — 对话页面
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    返回前端对话页面。

    从 static/index.html 读取文件内容后返回，不再内联 HTML 字符串。

    文件不存在时返回 503，避免用户看到 Python 异常堆栈。
    这种情况通常是部署时忘记创建 static/ 目录或拷贝 index.html。
    """
    if not HTML_FILE.exists():
        # 给出明确的错误提示，方便排查
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"前端文件未找到：{HTML_FILE}\n"
                "请确认 static/index.html 存在于项目根目录下。"
            ),
        )

    # 读取并返回 HTML 文件内容
    # encoding 显式指定 utf-8，避免 Windows 默认编码（GBK）导致中文乱码
    html_content = HTML_FILE.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


# =============================================================================
# GET /router/status
# =============================================================================

@app.get("/router/status")
async def get_router_status():
    """返回当前路由状态，供前端 UI 初始化。"""
    return JSONResponse(router.get_status())


# =============================================================================
# POST /router/toggle — 切换线上/本地
# =============================================================================

@app.post("/router/toggle")
async def toggle_router(req: ToggleRequest):
    """
    Toggle 拨动时调用。

    online=True  → force_online()，标记下一条消息走 Claude
    online=False → 重置路由状态，回到本地模式
    """
    if req.online:
        tip = router.force_online()
        return JSONResponse({"ok": True, "mode": "pending", "message": tip})
    else:
        router._reset()
        return JSONResponse({"ok": True, "mode": "local", "message": None})


# =============================================================================
# POST /chat — 核心对话接口
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    核心对话接口，处理流程：
      1. session 管理
      2. 保存用户消息
      3. 冲突回复检测
      4. 冲突询问检查
      5. 路由判断（ask_confirm / online / confirm_yes / confirm_no / local）
      6. 调用对应模型
      7. 保存并返回
    """
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    # 1. session
    session_id = session_manager.on_message()

    # 2. 保存用户消息
    save_message(session_id, "user", req.content)

    # 3. 冲突回复检测
    ca = _detect_conflict_action(req.content.strip())
    if ca is not None:
        cr = get_conflict_question()
        if cr is not None:
            cid, _ = cr
            reply  = handle_conflict_reply(cid, ca)
            save_message(session_id, "assistant", reply)
            return ChatResponse(reply=reply, session_id=session_id, mode="local")

    # 4. 冲突询问
    cr = get_conflict_question()
    if cr is not None:
        _, question = cr
        save_message(session_id, "assistant", question)
        return ChatResponse(reply=question, session_id=session_id, mode="local")

    # 5. 路由判断
    result = router.route(req.content.strip())
    action = result["action"]

    # A. 发确认询问
    if action == "ask_confirm":
        txt = result["text"]
        save_message(session_id, "assistant", txt)
        return ChatResponse(reply=txt, session_id=session_id, mode="confirm")

    # B. 走 Claude
    if action in ("online", "confirm_yes"):
        reply = router.call_claude(result["message"])
        save_message(session_id, "assistant", reply)
        return ChatResponse(reply=reply, session_id=session_id, mode="online")

    # C. 用户拒绝，缓存消息走本地
    if action == "confirm_no":
        history = get_messages_as_dicts(session_id)
        context = _build_context(session_id, user_message=req.content.strip())
        system  = build_system_prompt(context)
        msgs = [
            {"role": "system", "content": system},
            *history[:-1],
            {"role": "user", "content": req.content.strip()},
        ]
        reply = _call_local(msgs)
        save_message(session_id, "assistant", reply)
        return ChatResponse(reply=reply, session_id=session_id, mode="local")

    # D. 默认本地
    history = get_messages_as_dicts(session_id)
    context = _build_context(session_id, user_message=req.content.strip())
    system  = build_system_prompt(context)
    msgs = [
        {"role": "system", "content": system},
        *history[:-1],
        {"role": "user", "content": req.content.strip()},
    ]
    reply = _call_local(msgs)
    save_message(session_id, "assistant", reply)
    return ChatResponse(reply=reply, session_id=session_id, mode="local")


# =============================================================================
# POST /save
# =============================================================================

@app.post("/save")
async def save():
    """手动结束当前 session 并触发 L1 摘要。"""
    session_manager.force_close_current_session()
    return JSONResponse({"status": "ok"})


# =============================================================================
# 启动
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host   = SERVER_HOST,
        port   = SERVER_PORT,
        reload = DEBUG,
    )
