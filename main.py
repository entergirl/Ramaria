"""
main.py — FastAPI 应用入口
版本：0.3.5
=====================================================================

变更记录：

  v0.3.4 — P1-3 提取 _handle_local()，P1-4 router.disable_online()，
            P2-4 修复 session_index 误用。

  v0.3.5 — 两项改动（对应代码优化清单第三轮 P3-A、P2-A）：

    【P3-A】删除 _detect_conflict_action() 中的死代码
      问题：原函数末尾有两个连续的 return None，第一个之后第二个
            永远触达不到，属于无意义死代码。
      修复：删除多余的 return None，逻辑完全不变。
            Python 函数末尾无 return 时自动返回 None，行为一致。

    【P2-A】_build_context() 接入 RAG 语义检索
      问题：retrieve_combined() 接口已就绪，ChromaDB 索引正常写入，
            但 _build_context() 从未调用语义检索，retrieved_l1l2
            始终是纯时间序（最近 L2 + 最新 L1），向量索引功能闲置。
      修复：在 _build_context() 中调用 retrieve_combined(user_message)，
            将语义检索结果与时间序内容融合后注入 retrieved_l1l2。

      融合策略：
        · RAG 语义结果放在前面（语义相关的历史，最有参考价值）
        · 时间序内容追加在后面（最近发生的事，代表当前状态）
        · 两部分用"── 近期动态 ──"分隔，让模型能区分内容来源
        · user_message 为 None 时（/save 等场景）跳过 RAG，
          纯走时间序，行为与修改前完全一致，降级路径安全

目录结构（无变化）：
    demo/
    ├── main.py
    └── static/
        └── index.html

接口列表（无变化）：
  GET  /               — 对话页面（从 static/index.html 读取）
  POST /chat           — 核心对话接口
  POST /save           — 手动保存 session
  GET  /router/status  — 查询路由状态
  POST /router/toggle  — 切换线上/本地模式
"""

import os
# 禁止 HuggingFace 在启动时联网检查模型更新，避免离线环境启动卡顿
os.environ["HF_HUB_OFFLINE"] = "1"

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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

BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
HTML_FILE  = STATIC_DIR / "index.html"


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
    """
    FastAPI 生命周期管理器。
    startup：启动 SessionManager（含两个后台线程）。
    shutdown：优雅停止 SessionManager。
    """
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
    version     = "0.3.5",
    lifespan    = lifespan,
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# 数据模型
# =============================================================================

class ChatRequest(BaseModel):
    """用户发送消息的请求体。"""
    content: str


class ChatResponse(BaseModel):
    """
    /chat 接口的响应体。

    字段说明：
      reply      — 助手的回复文本
      session_id — 当前活跃 session 的数据库 id
      mode       — 本次回复的处理路径：
                     "local"   — 由本地 Qwen 生成
                     "online"  — 由 Claude API 生成
                     "confirm" — 等待用户确认是否调用 Claude
    """
    reply:      str
    session_id: int
    mode:       str


class ToggleRequest(BaseModel):
    """
    /router/toggle 接口的请求体。
    online=True 表示切换到线上模式，False 表示切回本地模式。
    """
    online: bool


# =============================================================================
# 冲突回复关键词与检测函数
# =============================================================================

# 用户确认"更新画像"的关键词集合（包含匹配，不区分大小写）
_RESOLVE_KEYWORDS = {"更新", "接受", "对", "是的", "没错", "update"}

# 用户选择"忽略冲突"的关键词集合（包含匹配，不区分大小写）
_IGNORE_KEYWORDS  = {"忽略", "不用", "不", "算了", "保持", "ignore"}

# 字数兜底阈值：消息超过此长度且未命中关键词，视为正常对话而非冲突回复
_CONFLICT_REPLY_MAX_LEN = 20


def _detect_conflict_action(text: str) -> str | None:
    """
    检测用户消息是否是对冲突询问的回复，返回对应 action。

    检测策略（两步，优先级从高到低）：
      1. 关键词包含匹配（优先）：
           · 消息中包含确认词 → 返回 "resolve"
           · 消息中包含忽略词 → 返回 "ignore"
      2. 字数兜底（次要）：
           · 未命中任何关键词，且消息超过 _CONFLICT_REPLY_MAX_LEN 字
             → 直接 return None，视为正常对话，不拦截

    参数：
        text — 用户消息文本（已去除首尾空白）

    返回：
        "resolve" — 用户确认接受新内容
        "ignore"  — 用户选择忽略冲突
        None      — 不是冲突回复，交给正常对话流程处理

    【P3-A 修复】删除了函数末尾重复的 return None：
        原来末尾两行：
            if len(text) > _CONFLICT_REPLY_MAX_LEN:
                return None
            return None      ← 死代码，无论如何这里都会返回 None
        修复后：只保留 if 块内的 return None，函数末尾不再显式 return。
        Python 函数执行到末尾无 return 时自动返回 None，行为完全一致。
    """
    t = text.lower().strip()

    # 关键词包含匹配：兼容"好啊"/"不用了"等自然变体
    if any(kw in t for kw in _RESOLVE_KEYWORDS):
        return "resolve"
    if any(kw in t for kw in _IGNORE_KEYWORDS):
        return "ignore"

    # 超长消息直接视为正常对话
    # 【P3-A 修复】此处之后不再有多余的 return None
    if len(text) > _CONFLICT_REPLY_MAX_LEN:
        return None


# =============================================================================
# RAG 结果格式化辅助函数
# =============================================================================

def _format_rag_results(rag_result: dict) -> str | None:
    """
    将 retrieve_combined() 的返回结果格式化为可注入 prompt 的纯文本。

    参数：
        rag_result — retrieve_combined() 的返回值：
                     {"l2": list[dict], "l1": list[dict]}
                     每个 dict 包含 "document"（文本）和 "distance"（余弦距离）

    返回：
        str  — 格式化后的多行文本，各条结果按检索顺序（相关度从高到低）排列
        None — L2 和 L1 均无命中时返回 None

    格式示例：
        [语义相关 · L2 时间段摘要]
        烧酒这周完成了多个后端模块...（相关度：0.32）

        [语义相关 · L1 单次摘要]
        烧酒今晚完成了 summarizer 模块...（相关度：0.28）
        烧酒今天讨论了记忆系统架构...（相关度：0.41）

    注意：
        "相关度"这里直接显示余弦距离值（越小越相关），
        不做反转处理，模型通过标签和上下文能理解其含义。
        后续如需改为相似度（1-distance）展示，只改这里即可。
    """
    lines = []

    l2_hits = rag_result.get("l2", [])
    l1_hits = rag_result.get("l1", [])

    # L2 语义结果（时间段聚合摘要，粒度粗，话题定位用）
    if l2_hits:
        lines.append("[语义相关 · L2 时间段摘要]")
        for hit in l2_hits:
            dist = hit.get("distance", 0)
            doc  = hit.get("document", "").strip()
            if doc:
                lines.append(f"{doc}（相关度：{dist:.2f}）")
        lines.append("")   # 空行分隔 L2 和 L1

    # L1 语义结果（单次对话摘要，粒度细，主力注入层）
    if l1_hits:
        lines.append("[语义相关 · L1 单次摘要]")
        for hit in l1_hits:
            dist = hit.get("distance", 0)
            doc  = hit.get("document", "").strip()
            if doc:
                lines.append(f"{doc}（相关度：{dist:.2f}）")

    if not lines:
        return None

    return "\n".join(lines).strip()


# =============================================================================
# context 组装辅助函数
# =============================================================================

def _build_context(session_id: int, user_message: str | None = None) -> dict:
    """
    组装传给 build_system_prompt() 的 context 字典。

    从数据库读取所有动态信息，统一封装后返回。
    所有 datetime 对象统一转换为带时区版本，避免 naive/aware 相减报错。

    参数：
        session_id:   当前活跃 session 的 ID。
        user_message: 用户本次消息文本。
                      · 有值时：触发 RAG 语义检索，语义结果优先注入
                      · 为 None 时：跳过 RAG，纯走时间序（降级安全）

    返回：
        context 字典，结构与 prompt_builder.PromptBuilder.build() 一致。

    【P2-A】retrieved_l1l2 融合结构：

        有 RAG 命中时：
            [语义相关 · L2 时间段摘要]
            ...

            [语义相关 · L1 单次摘要]
            ...

            ── 近期动态 ──
            [时间段摘要 2026-03-20] ...
            [最近一次对话] ...

        无 RAG 命中或 user_message=None 时：
            [时间段摘要 2026-03-20] ...
            [最近一次对话] ...
            （与修改前完全一致）
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
    # 时间序内容：近期 L2（最多3条）+ 最新 L1（1条）
    # 无论是否有 RAG 结果，时间序都会追加，确保模型感知"最近发生了什么"
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 上次 session 结束时间（跨日检测用）
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

    # ------------------------------------------------------------------
    # 【P2-A】RAG 语义检索 + 与时间序内容融合
    #
    # 触发条件：user_message 有值（/save 等无消息场景跳过）
    #
    # 融合规则：
    #   · 有 RAG 命中 + 有时间序 → RAG 在前，"── 近期动态 ──"分隔，时间序在后
    #   · 有 RAG 命中 + 无时间序 → 纯 RAG 文本
    #   · 无 RAG 命中 + 有时间序 → 纯时间序（与改前行为一致）
    #   · 两者都无     → None
    #
    # 异常处理：RAG 失败时 warning 降级，主流程不受影响
    # ------------------------------------------------------------------
    retrieved_l1l2 = None

    if user_message:
        rag_text = None
        try:
            from vector_store import retrieve_combined
            rag_result = retrieve_combined(user_message)
            rag_text   = _format_rag_results(rag_result)

            if rag_text:
                logger.debug(
                    f"RAG 命中：L2={len(rag_result.get('l2', []))} 条，"
                    f"L1={len(rag_result.get('l1', []))} 条"
                )
            else:
                logger.debug("RAG 无相关结果（距离超过阈值），降级到纯时间序")

        except Exception as e:
            logger.warning(f"RAG 检索失败，降级到纯时间序 — {e}")
            rag_text = None

        # 四种情况的融合
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
        # 两者都没有时保持 None

    else:
        # user_message=None，跳过 RAG，纯时间序
        retrieved_l1l2 = "\n".join(time_seq_parts) if time_seq_parts else None

    return {
        "last_session_time": last_session_time,
        "l3_profile":        l3_profile,
        "retrieved_l1l2":    retrieved_l1l2,
        "raw_fragments":     None,   # 预留：将来接入 L0 穿透召回后填充
        "session_id":        session_id,
        "session_index":     None,   # P2-4：不显示不准确的编号
    }


# =============================================================================
# 调用本地模型
# =============================================================================

def _call_local(messages: list[dict]) -> str:
    """
    调用本地 Qwen 模型，返回回复文本。
    失败时返回固定提示文本，不向上抛出异常。
    """
    result = call_local_chat(messages, caller="main")
    if result is None:
        return "（错误：本地模型调用失败，请确认 LM Studio 服务已启动）"
    return result


# =============================================================================
# 本地处理公共函数
# =============================================================================

def _handle_local(session_id: int, content: str) -> "ChatResponse":
    """
    统一处理走本地模型的分支（分支 C 用户拒绝 / 分支 D 默认本地）。

    流程：
      1. 取出当前 session 的完整对话历史
      2. 调用 _build_context()，触发 RAG 语义检索并构建 System Prompt
      3. 拼装消息列表（system + history[:-1] + 当前消息）
      4. 调用本地模型，保存回复，返回 ChatResponse

    参数：
        session_id — 当前活跃 session 的 id
        content    — 用户消息文本（已去除首尾空白）

    返回：
        ChatResponse，mode 固定为 "local"
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
# GET / — 对话页面
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端对话页面，从 static/index.html 读取。"""
    if not HTML_FILE.exists():
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"前端文件未找到：{HTML_FILE}\n"
                "请确认 static/index.html 存在于项目根目录下。"
            ),
        )
    return HTMLResponse(content=HTML_FILE.read_text(encoding="utf-8"))


# =============================================================================
# GET /router/status
# =============================================================================

@app.get("/router/status")
async def get_router_status():
    """返回当前路由状态，供前端 UI 初始化时同步显示。"""
    return JSONResponse(router.get_status())


# =============================================================================
# POST /router/toggle — 切换线上/本地
# =============================================================================

@app.post("/router/toggle")
async def toggle_router(req: ToggleRequest):
    """Toggle 拨动时调用，切换线上/本地模式。"""
    if req.online:
        tip = router.force_online()
        return JSONResponse({"ok": True, "mode": "pending", "message": tip})
    else:
        router.disable_online()
        return JSONResponse({"ok": True, "mode": "local", "message": None})


# =============================================================================
# POST /chat — 核心对话接口
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    核心对话接口。

    处理流程：
      1. 基础校验
      2. Session 管理
      3. 保存用户消息
      4. 冲突回复检测（优先处理"更新/忽略"回复）
      5. 冲突询问推送（有待确认冲突时插入询问）
      6. 路由判断 → 对应分支处理
    """
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    session_id = session_manager.on_message()
    save_message(session_id, "user", req.content)

    # 冲突回复检测
    conflict_action = _detect_conflict_action(req.content.strip())
    if conflict_action is not None:
        cr = get_conflict_question()
        if cr is not None:
            conflict_id, _ = cr
            reply = handle_conflict_reply(conflict_id, conflict_action)
            save_message(session_id, "assistant", reply)
            return ChatResponse(reply=reply, session_id=session_id, mode="local")

    # 冲突询问推送
    cr = get_conflict_question()
    if cr is not None:
        _, question = cr
        save_message(session_id, "assistant", question)
        return ChatResponse(reply=question, session_id=session_id, mode="local")

    # 路由判断
    result = router.route(req.content.strip())
    action = result["action"]

    # 分支 A：发确认询问
    if action == "ask_confirm":
        txt = result["text"]
        save_message(session_id, "assistant", txt)
        return ChatResponse(reply=txt, session_id=session_id, mode="confirm")

    # 分支 B：走 Claude API
    if action in ("online", "confirm_yes"):
        reply = router.call_claude(result["message"])
        save_message(session_id, "assistant", reply)
        return ChatResponse(reply=reply, session_id=session_id, mode="online")

    # 分支 C：用户拒绝，走本地（含 RAG）
    if action == "confirm_no":
        return _handle_local(session_id, result["message"])

    # 分支 D：默认本地（含 RAG）
    return _handle_local(session_id, req.content.strip())


# =============================================================================
# POST /save
# =============================================================================

@app.post("/save")
async def save():
    """
    手动结束当前 session 并触发 L1 摘要生成。
    内部调用 _build_context 时 user_message=None，跳过 RAG，属正常降级。
    """
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
