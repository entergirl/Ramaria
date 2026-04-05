"""
main.py — FastAPI 应用入口
版本：0.3.6
=====================================================================

变更记录：

  v0.3.6 — 检索质量优化：L1/L2 分层加权排序（对应优化清单 RAG-W1）

    【RAG-W1】_format_rag_results() + _build_context() 引入层级权重

      问题：
        原来 retrieve_combined() 返回的 L1/L2 结果在注入 prompt 时，
        是"L2 全部拼前、L1 全部拼后"的硬分块，块内按 adjusted_distance 排序。
        但两层的 adjusted_distance 量纲相同，混排才能真正体现相关度优先。

      优化方案（分层加权，保留分块结构）：
        · 引入 RETRIEVAL_WEIGHT_L2 / RETRIEVAL_WEIGHT_L1（config.py）
        · final_score = adjusted_distance × layer_weight
        · 输出格式保持分块（L2 块在前、L1 块在后），但块内按 final_score 升序排列
        · 调用方在 _build_context() 里从 config 读取权重，传给 _format_rag_results()

      改动范围：
        · _format_rag_results()：新增 weight_l2 / weight_l1 参数，块内排序改为 final_score
        · _build_context()：从 config 读取权重常量，传给 _format_rag_results()
        · config.py：新增 RETRIEVAL_WEIGHT_L2 = 0.8 / RETRIEVAL_WEIGHT_L1 = 1.0
        · 其余所有函数、路由、逻辑完全不变


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

import os, tempfile, asyncio
# 禁止 HuggingFace 在启动时联网检查模型更新，避免离线环境启动卡顿
os.environ["HF_HUB_OFFLINE"] = "1"

from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import (
    DEBUG,
    SERVER_HOST,
    SERVER_PORT,
    RETRIEVAL_WEIGHT_L2,   # [v0.3.6 新增] L2 层级权重系数
    RETRIEVAL_WEIGHT_L1,   # [v0.3.6 新增] L1 层级权重系数
)
from llm_client import call_local_chat
from database import (
    get_messages,
    get_messages_as_dicts,
    save_message,
    get_current_profile,
    get_latest_l1,
    get_recent_l2,
    get_active_sessions,
    get_session,
    get_pending_pushes,
    mark_push_sent,
    get_setting,          # 防抖：运行时读取 debounce_seconds 配置
)
from prompt_builder import build_system_prompt
from session_manager import SessionManager
from conflict_checker import get_conflict_question, handle_conflict_reply
from router import Router
from importer.l1_batch import start_batch, stop_batch, get_status, get_pending_count
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

# WebSocket 连接池
# key: WebSocket 实例，value: 对应的 session_id（连接建立时赋值）
# 使用字典便于按连接快速查找和删除
_ws_connections: dict = {}


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

    # 数据库迁移：确保 last_accessed_at 列存在（幂等，可重复运行）
    from database import add_last_accessed_at_columns
    add_last_accessed_at_columns()

    # 启动后台访问回写线程（MEMORY_DECAY_ENABLE_ACCESS_BOOST=False 时自动跳过）
    from vector_store import _start_access_worker
    _start_access_worker()

    # 启动时预热 BM25 索引（从 SQLite 全量读取，通常 < 1 秒）
    # 必须在 session_manager.start() 之前完成，
    # 确保第一条消息到来时 BM25 已就绪
    from vector_store import _bm25_index
    _bm25_index.rebuild("l1")
    _bm25_index.rebuild("l2")
    logger.info("BM25 索引预热完成")

    session_manager.start()

    # 启动主动推送调度器
    # 在 session_manager.start() 之后启动，确保 session 状态已就绪
    # 注入三个运行时函数：广播、在线判断、获取当前 session_id
    from push_scheduler import PushScheduler
    _push_scheduler = PushScheduler(
        ws_broadcast_fn = ws_broadcast,
        is_online_fn    = is_user_online,
        session_id_fn   = session_manager.get_current_session_id,
    )
    _push_scheduler.start()
    # 将调度器引用存入 session_manager，方便 stop() 时统一停止
    session_manager._push_scheduler = _push_scheduler

    logger.info("就绪，访问 http://localhost:8000")
    yield

    logger.info("关闭中…")
    from vector_store import _stop_access_worker
    _stop_access_worker()
    session_manager.stop()   # 内部会调用 _push_scheduler.stop()
    logger.info("已停止")


app = FastAPI(
    title       = "珊瑚菌 · 个人 AI 陪伴助手",
    description = "本地运行，支持分层记忆与任务路由",
    version     = "0.3.6",
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
_RESOLVE_KEYWORDS = {"更新", "接受", "对", "是的", "没错", "update", "是", "确认", "合并"}

# 用户选择"忽略冲突"的关键词集合（包含匹配，不区分大小写）
_IGNORE_KEYWORDS  = {"忽略", "不用", "不", "算了", "保持", "ignore", "不是", "分开"}

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
    """
    t = text.lower().strip()

    # 关键词包含匹配：兼容"好啊"/"不用了"等自然变体
    if any(kw in t for kw in _RESOLVE_KEYWORDS):
        return "resolve"
    if any(kw in t for kw in _IGNORE_KEYWORDS):
        return "ignore"

    # 超长消息直接视为正常对话
    if len(text) > _CONFLICT_REPLY_MAX_LEN:
        return None


# =============================================================================
# RAG 结果格式化辅助函数
# =============================================================================

def _format_rag_results(
    rag_result: dict,
    weight_l2: float = RETRIEVAL_WEIGHT_L2,
    weight_l1: float = RETRIEVAL_WEIGHT_L1,
) -> str | None:
    """
    将 retrieve_combined() 的返回结果格式化为可注入 prompt 的纯文本。

    [v0.3.6 改动]
      · 新增 weight_l2 / weight_l1 参数，对每条结果计算 final_score
      · final_score = adjusted_distance × layer_weight
      · 输出格式保持分块（L2 块在前、L1 块在后），块内按 final_score 升序排列
      · 显示"加权分数"而非原来的"相关度"，数值即 final_score

    设计说明：
      保留分块结构是为了给模型提供层级暗示：
        · 先看 L2 块 → 感知长期规律和话题背景
        · 再看 L1 块 → 获取具体事件细节
      块内排序按 final_score 是为了在同一块里把最相关的放前面，
      避免模型因"第一条不相关"而忽视后面更好的结果。

    参数：
        rag_result — retrieve_combined() 的返回值：
                     {"l2": list[dict], "l1": list[dict]}
                     每个 dict 包含：
                       "document"          — 索引文本（含日期前缀）
                       "distance"          — 原始语义距离（Chroma 返回）
                       "adjusted_distance" — 衰减调整后距离（vector_store 计算）
                       "metadata"          — 元数据字典
        weight_l2  — L2 层级权重系数，默认读取 config.RETRIEVAL_WEIGHT_L2
        weight_l1  — L1 层级权重系数，默认读取 config.RETRIEVAL_WEIGHT_L1

    返回：
        str  — 格式化后的多行文本，L2 块在前、L1 块在后，块内按 final_score 升序
        None — L2 和 L1 均无命中时返回 None

    格式示例：
        [语义相关 · L2 时间段摘要]
        2026-03 ~ 2026-03 烧酒这周在改 bug（加权分数：0.24）
        2026-02 ~ 2026-02 主要做前端优化（加权分数：0.35）

        [语义相关 · L1 单次摘要]
        2026-03-28 周五完成了 merger 模块（加权分数：0.28）
        2026-03-26 周三讨论了 RAG 方案（加权分数：0.31）
    """
    l2_hits = rag_result.get("l2", [])
    l1_hits = rag_result.get("l1", [])

    # 两层都没有命中，直接返回 None，调用方降级到纯时间序
    if not l2_hits and not l1_hits:
        return None

    lines = []

    # ── L2 块：时间段聚合摘要 ──
    if l2_hits:
        lines.append("[语义相关 · L2 时间段摘要]")

        # 计算每条 L2 结果的 final_score，按升序排列（越小越相关）
        l2_sorted = sorted(
            l2_hits,
            key=lambda hit: hit.get("adjusted_distance", hit.get("distance", 1.0)) * weight_l2,
        )

        for hit in l2_sorted:
            adj_dist   = hit.get("adjusted_distance", hit.get("distance", 1.0))
            final_score = adj_dist * weight_l2
            doc         = hit.get("document", "").strip()
            meta        = hit.get("metadata", {})

            # 从 metadata 取时间范围，拼出可读的日期前缀
            # period_start / period_end 格式如 "2026-03-01T00:00:00+00:00"
            period_start = meta.get("period_start", "")[:7]   # 取 YYYY-MM
            period_end   = meta.get("period_end",   "")[:7]
            if period_start and period_end and period_start != period_end:
                date_prefix = f"{period_start} ~ {period_end}"
            elif period_start:
                date_prefix = period_start
            else:
                date_prefix = ""

            # 组装一行：日期前缀 + 摘要文本 + 加权分数
            # doc 本身已含 [YYYY-MM-DD ~ YYYY-MM-DD] 前缀（index_l2 写入时拼的）
            # 这里不重复打印 doc 的内置前缀，而是用 metadata 里的 period 信息重新格式化
            # 原因：doc 里的前缀格式是 [date ~ date]，这里统一成无方括号的更简洁格式
            #
            # 注意：doc 里可能含有完整的摘要文本（含内置日期前缀），
            # 为了不重复显示日期，这里先把 doc 里的 [...] 前缀去掉再拼
            import re as _re
            doc_clean = _re.sub(r'^\[.*?\]\s*', '', doc)   # 去掉 doc 开头的 [...] 前缀

            if date_prefix:
                line = f"{date_prefix} {doc_clean}（加权分数：{final_score:.2f}）"
            else:
                line = f"{doc_clean}（加权分数：{final_score:.2f}）"

            lines.append(line)

        lines.append("")   # L2 块和 L1 块之间空一行

    # ── L1 块：单次对话摘要 ──
    if l1_hits:
        lines.append("[语义相关 · L1 单次摘要]")

        # 计算每条 L1 结果的 final_score，按升序排列
        l1_sorted = sorted(
            l1_hits,
            key=lambda hit: hit.get("adjusted_distance", hit.get("distance", 1.0)) * weight_l1,
        )

        for hit in l1_sorted:
            adj_dist    = hit.get("adjusted_distance", hit.get("distance", 1.0))
            final_score = adj_dist * weight_l1
            doc         = hit.get("document", "").strip()
            meta        = hit.get("metadata", {})

            # 从 metadata 取创建日期，格式化为 YYYY-MM-DD
            created_at = meta.get("created_at", "")[:10]   # 取 YYYY-MM-DD 部分

            # 同样去掉 doc 内置的 [...] 前缀，用 metadata 里的日期重新格式化
            import re as _re
            doc_clean = _re.sub(r'^\[.*?\]\s*', '', doc)

            if created_at:
                line = f"{created_at} {doc_clean}（加权分数：{final_score:.2f}）"
            else:
                line = f"{doc_clean}（加权分数：{final_score:.2f}）"

            lines.append(line)

    # 去掉末尾可能多出的空行，返回拼接结果
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

    retrieved_l1l2 融合结构（v0.3.6 起块内按加权分数排序）：

        有 RAG 命中时：
            [语义相关 · L2 时间段摘要]
            2026-03 ~ 2026-03 这周在改 bug（加权分数：0.24）
            ...

            [语义相关 · L1 单次摘要]
            2026-03-28 完成了 merger 模块（加权分数：0.28）
            ...

            ── 近期动态 ──
            [时间段摘要 2026-03-20] ...
            [最近一次对话] ...

        无 RAG 命中或 user_message=None 时（降级）：
            [时间段摘要 2026-03-20] ...
            [最近一次对话] ...
            （与 v0.3.5 行为完全一致）
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
    # 无论是否有 RAG 结果，时间序都会追加在后面，
    # 确保模型感知"最近发生了什么"，补充 RAG 语义结果可能遗漏的近期动态
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
    # RAG 语义检索 + 与时间序内容融合
    #
    # [v0.3.6 改动]
    #   · _format_rag_results() 现在接收 weight_l2 / weight_l1 参数
    #   · 块内按 final_score = adjusted_distance × layer_weight 升序排列
    #   · 权重从 config 读取，方便调整而不动逻辑代码
    #
    # 触发条件：user_message 有值（/save 等无消息场景跳过）
    #
    # 融合规则：
    #   · 有 RAG 命中 + 有时间序 → RAG 在前，"── 近期动态 ──"分隔，时间序在后
    #   · 有 RAG 命中 + 无时间序 → 纯 RAG 文本
    #   · 无 RAG 命中 + 有时间序 → 纯时间序（与 v0.3.5 行为一致）
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

            # [v0.3.6] 传入权重系数，块内按加权分数排序
            rag_text = _format_rag_results(
                rag_result,
                weight_l2 = RETRIEVAL_WEIGHT_L2,
                weight_l1 = RETRIEVAL_WEIGHT_L1,
            )

            if rag_text:
                logger.debug(
                    f"RAG 命中（加权排序）：L2={len(rag_result.get('l2', []))} 条，"
                    f"L1={len(rag_result.get('l1', []))} 条，"
                    f"weight_l2={RETRIEVAL_WEIGHT_L2}，weight_l1={RETRIEVAL_WEIGHT_L1}"
                )
            else:
                logger.debug("RAG 无相关结果（距离超过阈值），降级到纯时间序")

        except Exception as e:
            logger.warning(f"RAG 检索失败，降级到纯时间序 — {e}")
            rag_text = None

        # 四种情况的融合（逻辑与 v0.3.5 一致，格式内容已由 _format_rag_results 改变）
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
        # user_message=None，跳过 RAG，纯时间序（降级路径）
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

def _handle_local(session_id: int, content: str) -> ChatResponse:
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
# GET /api/sessions — session 列表接口
# =============================================================================

def get_all_sessions_with_stats():
    """
    获取所有 session 的摘要列表，含统计信息。

    返回所有 session 的基本信息加统计：
      - id: session ID
      - started_at: session 开始时间
      - ended_at: session 结束时间（进行中则为 null）
      - message_count: 本 session 的消息总数
      - last_message_at: 本 session 最后一条消息的时间（可能为 null）
      - last_message_preview: 最后一条消息的前 80 个字符预览（可能为 null）

    按最后活动时间倒序排列（活跃优先）。
    """
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sql = """
        SELECT
          s.id, s.started_at, s.ended_at,
          (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count,
          (SELECT MAX(m.created_at) FROM messages m WHERE m.session_id = s.id) AS last_message_at,
          (SELECT m.content FROM messages m WHERE m.session_id = s.id
           ORDER BY m.created_at DESC LIMIT 1) AS last_message_preview
        FROM sessions s
        ORDER BY COALESCE(
          (SELECT MAX(m.created_at) FROM messages m WHERE m.session_id = s.id),
          s.started_at
        ) DESC
    """

    cursor.execute(sql)
    results = []
    for row in cursor.fetchall():
        preview = row["last_message_preview"]
        if preview and len(preview) > 80:
            preview = preview[:80] + "…"

        results.append({
            "id":                   row["id"],
            "started_at":           row["started_at"],
            "ended_at":             row["ended_at"],
            "message_count":        row["message_count"],
            "last_message_at":      row["last_message_at"],
            "last_message_preview": preview,
        })

    conn.close()
    return results


@app.get("/api/sessions")
async def api_get_sessions():
    """获取所有 session 的摘要列表。"""
    try:
        sessions = get_all_sessions_with_stats()
        return JSONResponse(sessions)
    except Exception as e:
        logger.error(f"获取 session 列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部错误: {e}")


# =============================================================================
# GET /api/sessions/{session_id}/messages — 获取指定 session 的消息
# =============================================================================

@app.get("/api/sessions/{session_id}/messages")
async def api_get_session_messages(session_id: int):
    """
    获取指定 session 的所有消息。

    异常：
      - 404: session 不存在
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = get_messages(session_id)
    result = [
        {
            "role":       row["role"],
            "content":    row["content"],
            "created_at": row["created_at"],
        }
        for row in messages
    ]

    return JSONResponse(result)

# =============================================================================
# POST /import/qq/preview — 解析预览（不写入数据库）
# =============================================================================

@app.post("/import/qq/preview")
async def import_qq_preview(
    file: UploadFile = File(...),
    gap: int = 10,
):
    """
    上传 QQ Chat Exporter JSON 文件，返回详细诊断报告。
    此接口不写入数据库，仅用于预览解析结果。

    参数：
        file — 上传的 JSON 文件（multipart/form-data）
        gap  — session 切割时间间隔（分钟），默认10分钟

    返回：
        report — 诊断报告字典（见 ParseReport.to_dict()）
    """
    from importer.qq_parser import parse_qq_export

    # 将上传文件写入临时文件，解析完成后删除
    # 使用 suffix 保留 .json 扩展名，方便调试
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="wb"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = parse_qq_export(tmp_path, gap_minutes=gap)
        return JSONResponse(result.report.to_dict())

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"import/qq/preview 处理失败 — {e}")
        raise HTTPException(status_code=500, detail=f"解析失败：{e}")
    finally:
        # 无论成功还是失败，都删除临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# POST /import/qq/confirm — 真正写入数据库
# =============================================================================

@app.post("/import/qq/confirm")
async def import_qq_confirm(
    file: UploadFile = File(...),
    gap: int = 10,
):
    """
    上传 QQ Chat Exporter JSON 文件，解析后写入数据库（仅写入 L0）。
    不触发 L1 摘要生成，L1 由后续批量流程处理。

    参数：
        file — 上传的 JSON 文件（multipart/form-data）
        gap  — session 切割时间间隔（分钟），默认10分钟

    返回：
        stats — 写入统计：
            sessions_written  写入成功的 session 数
            messages_written  写入成功的消息总数
            sessions_failed   写入失败的 session 数
            failed_details    失败详情列表
    """
    from importer.qq_parser import parse_qq_export
    from importer.qq_importer import import_sessions_to_db

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="wb"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 第一阶段：解析
        result = parse_qq_export(tmp_path, gap_minutes=gap)

        # 第二阶段：写入数据库
        stats = import_sessions_to_db(result.parsed_sessions)

        return JSONResponse({
            "status":  "ok",
            "stats":   stats,
            "report_overview": {
                "total_raw":     result.report.total_raw,
                "session_count": result.report.session_count,
                "time_start":    result.report.time_start,
                "time_end":      result.report.time_end,
            },
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"import/qq/confirm 处理失败 — {e}")
        raise HTTPException(status_code=500, detail=f"导入失败：{e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# =============================================================================
# GET /import — 导入页面
# =============================================================================

@app.get("/import", response_class=HTMLResponse)
async def import_page():
    import_html = BASE_DIR / "static" / "import.html"
    if not import_html.exists():
        raise HTTPException(status_code=503, detail="import.html 未找到")
    return HTMLResponse(content=import_html.read_text(encoding="utf-8"))


# =============================================================================
# POST /import/qq/preview — QQ 聊天记录解析预览（不写入数据库）
# =============================================================================

@app.post("/import/qq/preview")
async def import_qq_preview(
    file: UploadFile = File(...),
    gap: int = 10,
):
    """
    上传 QQ Chat Exporter JSON 文件，返回详细诊断报告。
    不写入数据库，仅用于预览解析结果。
    """
    from importer.qq_parser import parse_qq_export

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = parse_qq_export(tmp_path, gap_minutes=gap)
        return JSONResponse(result.report.to_dict())

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"import/qq/preview 处理失败 — {e}")
        raise HTTPException(status_code=500, detail=f"解析失败：{e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# POST /import/qq/confirm — QQ 聊天记录导入（写入 L0）
# =============================================================================

@app.post("/import/qq/confirm")
async def import_qq_confirm(
    file: UploadFile = File(...),
    gap: int = 10,
):
    """
    上传 QQ Chat Exporter JSON 文件，解析后写入数据库 L0 层。
    不触发 L1 摘要生成，L1 由 /import/qq/start_l1 单独触发。
    """
    from importer.qq_parser import parse_qq_export
    from importer.qq_importer import import_sessions_to_db

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = parse_qq_export(tmp_path, gap_minutes=gap)
        stats  = import_sessions_to_db(result.parsed_sessions)

        return JSONResponse({
            "status": "ok",
            "stats":  stats,
            "report_overview": {
                "total_raw":     result.report.total_raw,
                "session_count": result.report.session_count,
                "time_start":    result.report.time_start,
                "time_end":      result.report.time_end,
            },
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"import/qq/confirm 处理失败 — {e}")
        raise HTTPException(status_code=500, detail=f"导入失败：{e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# GET /import/qq/pending_l1 — 查询待处理 session 数量
# =============================================================================

@app.get("/import/qq/pending_l1")
async def get_pending_l1():
    """查询当前有多少历史 session 尚未生成 L1 摘要。"""
    from importer.l1_batch import get_pending_count
    count = get_pending_count()
    return JSONResponse({"count": count})


# =============================================================================
# POST /import/qq/start_l1 — 启动 L1 批量生成
# =============================================================================

@app.post("/import/qq/start_l1")
async def start_l1_batch():
    """启动后台批处理线程，非阻塞，立即返回当前状态。"""
    from importer.l1_batch import start_batch
    status = start_batch(session_manager=session_manager)
    return JSONResponse(status)


# =============================================================================
# GET /import/qq/l1_progress — 查询批处理进度（前端轮询）
# =============================================================================

@app.get("/import/qq/l1_progress")
async def get_l1_progress():
    """返回当前批处理状态，供前端进度面板轮询（建议间隔 2 秒）。"""
    from importer.l1_batch import get_status
    return JSONResponse(get_status())


# =============================================================================
# POST /import/qq/stop_l1 — 停止批处理
# =============================================================================

@app.post("/import/qq/stop_l1")
async def stop_l1_batch():
    """请求停止批处理，等当前 session 处理完后退出。"""
    from importer.l1_batch import stop_batch
    status = stop_batch()
    return JSONResponse(status)


# =============================================================================
# WebSocket 连接池工具函数
# =============================================================================

async def _ws_send(ws: WebSocket, data: dict):
    """
    向单个 WebSocket 客户端发送 JSON 数据。
    封装异常处理，发送失败时静默记录日志，不向上抛出。

    所有服务端 → 客户端的消息都通过此函数发送，统一格式为：
        {
            "type":  消息类型字符串（见下方类型说明）,
            ...      其他字段根据 type 不同而不同
        }

    消息类型说明：
        "reply"    — 模型生成的对话回复
        "push"     — 主动推送消息（push_scheduler 触发）
        "status"   — 状态通知（如"消息已收到，正在思考"）
        "error"    — 错误通知

    参数：
        ws   — 目标 WebSocket 连接实例
        data — 要发送的数据字典，必须包含 "type" 字段
    """
    try:
        await ws.send_json(data)
    except Exception as e:
        logger.warning(f"WebSocket 发送失败 — {e}")


async def ws_broadcast(data: dict):
    """
    向所有当前在线的 WebSocket 客户端广播消息。

    push_scheduler 触发主动推送时调用此函数。
    如果连接池为空（用户离线），调用方应改为写入 pending_push 表。

    参数：
        data — 要广播的数据字典
    """
    for ws in list(_ws_connections.keys()):
        await _ws_send(ws, data)


def is_user_online() -> bool:
    """
    判断当前是否有用户在线（即连接池非空）。
    push_scheduler 在决定直接推送还是暂存时调用此函数。

    返回：
        bool — True 表示至少有一个 WebSocket 连接存在
    """
    return len(_ws_connections) > 0


# =============================================================================
# WebSocket 路由
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket 主路由，处理客户端的实时双向通信。
    v2：新增服务端防抖聚合，用户连发多条消息时等待停顿后统一回复。

    连接生命周期：
        1. 握手建立 → 注册连接池 → 推送积压的 pending_push 消息
        2. 消息循环 → 接收用户消息 → 存入缓冲区 → 防抖计时器到期 → 合并触发
        3. 连接断开 → 取消计时器 → 从连接池移除

    防抖逻辑：
        每个连接维护独立的 _buf（消息缓冲区）和 _timer（asyncio 计时任务）。
        每来一条消息：
          · 追加到 _buf
          · 若 _timer 已存在则取消
          · 重新创建一个 debounce_seconds 秒后触发的 _timer
        _timer 到期时：
          · 把 _buf 里所有消息按换行合并为一条
          · 清空 _buf
          · 走正常的路由 → 模型生成 → 回复流程
    """
    await ws.accept()
    logger.info("WebSocket 连接建立")

    _ws_connections[ws] = None

    # ------------------------------------------------------------------
    # 每个连接独立的防抖状态
    # 使用列表包装，便于在嵌套函数（_flush）里修改
    # ------------------------------------------------------------------
    _buf:   list[str]                      = []     # 消息缓冲区
    _timer: list[asyncio.Task | None]      = [None] # 防抖计时任务（用列表包装以便内层修改）

    async def _flush():
        """
        防抖计时器到期时触发：合并缓冲区消息，走路由→生成→回复流程。
        此函数运行在同一个事件循环里，不存在并发安全问题。
        """
        if not _buf:
            return

        # 合并缓冲区里的所有消息，按换行分隔
        # 例如用户连发："在吗" + "想问你个问题" → "在吗\n想问你个问题"
        combined = "\n".join(_buf)
        _buf.clear()
        _timer[0] = None

        # 取当前连接绑定的 session_id
        session_id = _ws_connections.get(ws)
        if session_id is None:
            logger.warning("_flush 触发时 session_id 为 None，跳过")
            return

        logger.debug(f"防抖触发，合并消息：{combined[:80]}")

        # ------------------------------------------------------------------
        # 以下逻辑与原路由完全一致
        # ------------------------------------------------------------------

        # 冲突回复检测
        conflict_action = _detect_conflict_action(combined)
        if conflict_action is not None:
            cr = get_conflict_question()
            if cr is not None:
                conflict_id, _ = cr
                reply = handle_conflict_reply(conflict_id, conflict_action)
                save_message(session_id, "assistant", reply)
                await _ws_send(ws, {
                    "type":       "reply",
                    "reply":      reply,
                    "session_id": session_id,
                    "mode":       "local",
                })
                AppState_loading_off()
                return

        # 冲突询问推送
        cr = get_conflict_question()
        if cr is not None:
            _, question = cr
            save_message(session_id, "assistant", question)
            await _ws_send(ws, {
                "type":       "reply",
                "reply":      question,
                "session_id": session_id,
                "mode":       "local",
            })
            AppState_loading_off()
            return

        # 路由判断
        result = router.route(combined)
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
            reply = router.call_claude(result["message"])
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

    def AppState_loading_off():
        """
        占位函数：loading 状态由前端 onMessage 回调在收到 reply 后自动关闭。
        服务端不需要额外通知，此函数仅作注释说明用，实际为空操作。
        """
        pass

    try:
        # ------------------------------------------------------------------
        # 用户上线：推送离线积压消息
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 消息循环
        # ------------------------------------------------------------------
        while True:
            data = await ws.receive_json()

            msg_type = data.get("type")
            content  = data.get("content", "").strip()

            if msg_type != "chat" or not content:
                continue

            # --------------------------------------------------------------
            # Session 管理：每条消息到来时确保有活跃 session
            # 注意：save_message 在 _flush 里不再调用，
            # 这里只做 session 初始化和单条消息的持久化
            # --------------------------------------------------------------
            session_id = session_manager.on_message()
            _ws_connections[ws] = session_id

            # 每条消息单独存入数据库（L0 永久保留原始消息）
            save_message(session_id, "user", content)

            # 存入缓冲区
            _buf.append(content)

            # 取防抖时长（从数据库读，支持用户实时修改配置）
            try:
                debounce_sec = float(get_setting("debounce_seconds") or "3")
            except ValueError:
                debounce_sec = 3.0

            # 取消旧计时器（如果存在）
            if _timer[0] is not None and not _timer[0].done():
                _timer[0].cancel()

            # 创建新计时器：debounce_sec 秒后触发 _flush
            async def _delayed_flush(delay: float):
                await asyncio.sleep(delay)
                await _flush()

            _timer[0] = asyncio.create_task(_delayed_flush(debounce_sec))

    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
    except Exception as e:
        logger.error(f"WebSocket 处理异常 — {e}")
    finally:
        # 连接断开时取消未触发的计时器，避免悬空任务
        if _timer[0] is not None and not _timer[0].done():
            _timer[0].cancel()
            logger.debug("已取消未触发的防抖计时器")
        _ws_connections.pop(ws, None)
        logger.debug(f"当前在线连接数：{len(_ws_connections)}")

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
