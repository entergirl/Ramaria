"""
main.py — FastAPI 应用入口
版本：0.3.2
=====================================================================

本次修改（对应审查报告问题6）：
  · _detect_conflict_action() 改为关键词优先匹配。
    旧写法：len(text) > 10 时直接返回 None，跳过关键词检测。
    问题：用户回复"好帮我更新一下"（11字）会被直接忽略，
          造成冲突询问没有响应，体验很差。
    新写法：先做关键词匹配，命中则返回对应 action；
           未命中时才用字数（20字）兜底过滤，避免把正常对话误判为冲突回复。

界面与后端逻辑沿用 0.3.1，无其他改动。

接口列表：
  GET  /               — 对话页面
  POST /chat           — 核心对话接口
  POST /save           — 手动保存 session
  GET  /router/status  — 查询路由状态
  POST /router/toggle  — 切换线上/本地模式
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
from contextlib import asynccontextmanager
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
    print("[main] 应用启动中…")
    session_manager.start()
    print("[main] 就绪，访问 http://localhost:8000")
    yield
    print("[main] 关闭中…")
    session_manager.stop()
    print("[main] 已停止")


app = FastAPI(
    title       = "珊瑚菌 · 个人 AI 陪伴助手",
    description = "本地运行，支持分层记忆与任务路由",
    version     = "0.3.2",
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
# [修改说明] 审查报告问题6：
#   旧写法：if len(text) > 10: return None
#     问题：用户回复"好帮我更新一下"（11字）会被直接截断，
#           冲突询问得不到响应，用户体验很差。
#   新写法：关键词匹配优先，命中即返回；
#           未命中时再用字数（20字）兜底过滤。
#     效果：兜住了"好帮我更新一下"、"忽略就行"、"更新一下吧"等自然语言表达，
#           同时对"我今天很开心，想和你聊聊天"这类长消息仍能正确过滤。
_CONFLICT_REPLY_MAX_LEN = 20


def _detect_conflict_action(text):
    """
    检测用户消息是否是对冲突询问的回复，返回对应 action。

    检测策略（两步，优先级从高到低）：
      1. 关键词匹配（优先）：
           · 消息中包含确认词（"更新"/"接受"/"是的" 等）→ 返回 "resolve"
           · 消息中包含忽略词（"忽略"/"不用"/"算了" 等）→ 返回 "ignore"
           · 关键词匹配不依赖消息长度，长消息也能正确识别

      2. 字数兜底（次要）：
           · 未命中任何关键词，且消息超过 _CONFLICT_REPLY_MAX_LEN 字 → 返回 None
           · 超长消息几乎不可能是冲突回复，直接放行给正常对话逻辑

    参数：
        text — 用户消息文本（已去除首尾空白）

    返回：
        "resolve" — 用户确认接受新内容
        "ignore"  — 用户选择忽略冲突
        None      — 不是冲突回复，交给正常对话逻辑处理
    """
    t = text.lower().strip()

    # 第一步：关键词匹配（优先，不受字数限制）
    # 用 any() + 子串匹配，兜住"好帮我更新一下"、"更新一下吧"等自然语言
    if any(kw in t for kw in _RESOLVE_KEYWORDS):
        return "resolve"
    if any(kw in t for kw in _IGNORE_KEYWORDS):
        return "ignore"

    # 第二步：字数兜底
    # 未命中关键词且超过阈值，判定为正常对话，不做冲突处理
    if len(text) > _CONFLICT_REPLY_MAX_LEN:
        return None

    # 短消息但未命中关键词（如"嗯""好的""明白"等），也不当作冲突回复
    # 交给正常对话逻辑，避免"好的"这类词在没有冲突待确认时被误触
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

    Args:
        session_id:   当前活跃 session 的 ID。
        user_message: 用户本次消息（预留，供将来接入 RAG 检索）。

    Returns:
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
    #
    # 关键：fromisoformat 解析带时区字符串（如 "2026-03-20T23:00:00+00:00"）
    # 结果是 offset-aware datetime；若字符串不带时区则手动补 UTC，
    # 确保与 prompt_builder 里 datetime.now(timezone.utc) 可以直接相减。
    # ------------------------------------------------------------------
    last_session_time = None
    if l1_row and l1_row["created_at"]:
        try:
            dt = datetime.fromisoformat(l1_row["created_at"])
            # 若解析结果没有时区信息，补 UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            last_session_time = dt
        except ValueError:
            pass  # 时间戳格式异常时降级为 None，不影响主流程

    return {
        "last_session_time": last_session_time,
        "l3_profile":        l3_profile,
        "retrieved_l1l2":    retrieved_l1l2,
        "raw_fragments":     None,       # L0 穿透暂未接入，预留
        "session_id":        session_id,
        "session_index":     session_id, # 用自增主键近似表示第几个 session
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
# 对话页面 HTML
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    返回完整对话页面。CSS/JS 全部内联，无外部依赖，离线可用。
    """
    html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>珊瑚菌 · 个人 AI 陪伴助手</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@300;400;500&display=swap" rel="stylesheet">

  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:           #f7f4ef;
      --surface:      #ffffff;
      --surface-alt:  #fdf8f3;
      --border:       #e8e0d4;
      --border-light: #ede7dd;
      --text:         #2d2720;
      --text-mid:     #7a6e63;
      --text-dim:     #b0a89e;
      --coral:        #c97c5a;
      --coral-light:  #e8c4b0;
      --coral-dim:    #d4956f;
      --dot-local:    #7cad7c;
      --dot-online:   #c97c5a;
      --dot-pending:  #7fafd4;
      --font-title:   'Noto Serif SC', Georgia, serif;
      --font-body:    'Noto Serif SC', Georgia, serif;
      --font-mono:    'SFMono-Regular', 'Consolas', monospace;
      --radius:       16px;
      --radius-sm:    8px;
    }

    html, body { height: 100%; }

    body {
      background: var(--bg);
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
      color: var(--text);
      font-family: var(--font-body);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 0 16px;
    }

    header { width: 100%; max-width: 740px; padding: 16px 0 12px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--bg); z-index: 10; }

    .header-title { display: flex; flex-direction: column; gap: 1px; flex-shrink: 0; }
    .title-main { font-family: var(--font-title); font-size: .98rem; font-weight: 500; color: var(--coral); letter-spacing: .06em; line-height: 1; }
    .title-sub { font-size: .62rem; color: var(--text-dim); font-family: var(--font-mono); letter-spacing: .05em; }

    #session-label { font-size: .66rem; color: var(--text-dim); font-family: var(--font-mono); background: var(--border-light); padding: 2px 8px; border-radius: 20px; flex-shrink: 0; }

    #controls { margin-left: auto; display: flex; align-items: center; gap: 9px; }

    #mode-badge { display: flex; align-items: center; gap: 5px; font-size: .66rem; font-family: var(--font-mono); color: var(--text-mid); flex-shrink: 0; }
    #mode-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--dot-local); transition: background .4s; flex-shrink: 0; }
    #mode-dot.online { background: var(--dot-online); }
    #mode-dot.pending { background: var(--dot-pending); animation: blink 1.4s ease-in-out infinite; }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: .2; } }

    .sep { width: 1px; height: 14px; background: var(--border); flex-shrink: 0; }

    .toggle-wrap { display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
    .toggle-label-text { font-size: .66rem; font-family: var(--font-mono); color: var(--text-mid); white-space: nowrap; }
    .toggle-input { display: none; }
    .toggle-track { position: relative; width: 34px; height: 18px; border-radius: 9px; background: var(--border); transition: background .25s; flex-shrink: 0; }
    .toggle-track::after { content: ''; position: absolute; top: 2px; left: 2px; width: 14px; height: 14px; border-radius: 50%; background: var(--text-dim); transition: transform .25s, background .25s; }
    .toggle-input:checked + .toggle-track { background: var(--coral-light); }
    .toggle-input:checked + .toggle-track::after { transform: translateX(16px); background: var(--coral); }

    #save-btn { background: none; border: 1px solid var(--border); color: var(--text-mid); font-size: .66rem; font-family: var(--font-mono); padding: 3px 10px; border-radius: 20px; cursor: pointer; white-space: nowrap; flex-shrink: 0; transition: border-color .2s, color .2s, background .2s; }
    #save-btn:hover { border-color: var(--coral-light); color: var(--coral); background: rgba(201,124,90,.06); }

    #messages { width: 100%; max-width: 740px; flex: 1; overflow-y: auto; padding: 26px 0 14px; display: flex; flex-direction: column; gap: 12px; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }

    .msg { display: flex; flex-direction: column; gap: 3px; animation: rise .2s ease both; }
    @keyframes rise { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

    .msg-label { font-size: .63rem; font-family: var(--font-mono); color: var(--text-dim); letter-spacing: .05em; display: flex; align-items: center; gap: 5px; }
    .msg.user .msg-label { justify-content: flex-end; color: var(--coral-light); }

    .tag-online { font-size: .58rem; color: var(--coral); border: 1px solid var(--coral-light); border-radius: 3px; padding: 0 3px; line-height: 1.5; }

    .msg-bubble { padding: 11px 16px; border-radius: var(--radius); line-height: 1.78; font-size: .92rem; white-space: pre-wrap; word-break: break-word; max-width: 84%; box-shadow: 0 1px 4px rgba(0,0,0,.05); }

    .msg.user { align-items: flex-end; }
    .msg.user .msg-bubble { background: var(--surface-alt); border: 1px solid var(--border-light); }
    .msg.assistant .msg-bubble { background: var(--surface); border: 1px solid var(--border); }
    .msg.assistant.confirm .msg-bubble { border: 1.5px dashed var(--dot-pending); }
    .msg.assistant.is-online .msg-bubble { border-left: 3px solid var(--coral-light); }

    .think-block { font-size: .73rem; color: var(--text-dim); font-family: var(--font-mono); border-left: 2px solid var(--border); padding: 4px 10px; margin-bottom: 8px; cursor: pointer; user-select: none; transition: color .2s; }
    .think-block:hover { color: var(--text-mid); }
    .think-content { display: none; margin-top: 5px; }
    .think-block.open .think-content { display: block; }

    .sys-notice { text-align: center; color: var(--text-dim); font-size: .68rem; font-family: var(--font-mono); padding: 5px 0; letter-spacing: .04em; }

    .typing { display: flex; gap: 4px; align-items: center; padding: 2px 0; }
    .typing span { width: 5px; height: 5px; border-radius: 50%; background: var(--coral-light); animation: bounce .9s infinite; }
    .typing span:nth-child(2) { animation-delay: .15s; }
    .typing span:nth-child(3) { animation-delay: .30s; }
    @keyframes bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-5px); } }

    #input-area { width: 100%; max-width: 740px; padding: 10px 0 22px; display: flex; gap: 10px; align-items: flex-end; }

    #user-input { flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); font-family: var(--font-body); font-size: .92rem; padding: 12px 16px; resize: none; line-height: 1.6; min-height: 48px; max-height: 160px; outline: none; box-shadow: 0 1px 3px rgba(0,0,0,.04); transition: border-color .2s, box-shadow .2s; }
    #user-input:focus { border-color: var(--coral-light); box-shadow: 0 0 0 3px rgba(201,124,90,.1); }
    #user-input::placeholder { color: var(--text-dim); }

    #send-btn { width: 48px; height: 48px; border-radius: var(--radius-sm); border: none; background: var(--coral); color: #fff; font-size: 1.1rem; cursor: pointer; flex-shrink: 0; box-shadow: 0 2px 8px rgba(201,124,90,.35); transition: background .2s, transform .1s, box-shadow .2s; }
    #send-btn:hover:not(:disabled) { background: var(--coral-dim); box-shadow: 0 3px 12px rgba(201,124,90,.45); }
    #send-btn:active:not(:disabled) { transform: scale(.95); }
    #send-btn:disabled { background: var(--border); box-shadow: none; cursor: not-allowed; }
  </style>
</head>
<body>

<header>
  <div class="header-title">
    <span class="title-main">珊瑚菌 · 个人 AI 陪伴助手</span>
    <span class="title-sub">Ramaria · Personal Companion</span>
  </div>
  <span id="session-label">初始化中</span>
  <div id="controls">
    <div id="mode-badge">
      <div id="mode-dot"></div>
      <span id="mode-text">本地</span>
    </div>
    <div class="sep"></div>
    <label class="toggle-wrap" title="切换本地 / 线上模式">
      <span class="toggle-label-text">线上模式</span>
      <input type="checkbox" id="mode-toggle" class="toggle-input" onchange="onToggleChange(this.checked)">
      <div class="toggle-track"></div>
    </label>
    <div class="sep"></div>
    <button id="save-btn" onclick="saveSession()">保存对话</button>
  </div>
</header>

<div id="messages"></div>

<div id="input-area">
  <textarea id="user-input" placeholder="输入消息，Enter 发送，Shift+Enter 换行" rows="1"></textarea>
  <button id="send-btn" onclick="sendMessage()">↑</button>
</div>

<script>
let isLoading = false;
let isOnline  = false;

const textarea = document.getElementById('user-input');
textarea.addEventListener('input', () => {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
});
textarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function onToggleChange(checked) {
  isOnline = checked;
  updateModeUI(checked ? 'pending' : 'local');
  try {
    const res  = await fetch('/router/toggle', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({online: checked}) });
    const data = await res.json();
    if (data.message) _appendBubble(data.message, 'assistant', false, false);
    updateModeUI(data.mode || (checked ? 'pending' : 'local'));
  } catch (_) {
    isOnline = !checked;
    document.getElementById('mode-toggle').checked = !checked;
    updateModeUI('local');
  }
}

function updateModeUI(mode) {
  const dot = document.getElementById('mode-dot');
  const text = document.getElementById('mode-text');
  dot.className = '';
  switch (mode) {
    case 'online':  dot.classList.add('online');  text.textContent = '线上'; break;
    case 'pending': dot.classList.add('pending'); text.textContent = '等待确认'; break;
    default: text.textContent = '本地';
  }
}

function renderAssistant(text, mode) {
  hideTyping();
  const parts = text.split('||').map(s => s.trim()).filter(Boolean);
  parts.forEach((part, i) => {
    const last = i === parts.length - 1;
    _appendBubble(part, 'assistant', last && mode === 'online', last && mode === 'confirm');
  });
}

function _appendBubble(text, role, isOnlineMsg, isConfirm) {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role + (isConfirm ? ' confirm' : '') + (isOnlineMsg ? ' is-online' : '');
  const label = document.createElement('div');
  label.className = 'msg-label';
  if (role === 'user') {
    label.textContent = '你';
  } else {
    label.textContent = '助手';
    if (isOnlineMsg) { const tag = document.createElement('span'); tag.className = 'tag-online'; tag.textContent = 'API'; label.appendChild(tag); }
  }
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  if (role === 'assistant') {
    const m = text.match(/^<think>([\s\S]*?)<\/think>\s*/);
    if (m) {
      const td = document.createElement('div');
      td.className = 'think-block';
      td.innerHTML = '▸ 思考过程（点击展开）<div class="think-content">' + _esc(m[1].trim()) + '</div>';
      td.onclick = () => td.classList.toggle('open');
      bubble.appendChild(td);
      text = text.slice(m[0].length);
    }
  }
  bubble.appendChild(document.createTextNode(text));
  div.appendChild(label);
  div.appendChild(bubble);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function showTyping() {
  const c = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.id = 'typing-indicator';
  el.innerHTML = '<div class="msg-label">助手</div><div class="msg-bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  c.appendChild(el);
  c.scrollTop = c.scrollHeight;
}
function hideTyping() { const el = document.getElementById('typing-indicator'); if (el) el.remove(); }

async function sendMessage() {
  if (isLoading) return;
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';
  _appendBubble(text, 'user', false, false);
  isLoading = true;
  document.getElementById('send-btn').disabled = true;
  showTyping();
  try {
    const res = await fetch('/chat', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({content: text}) });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || 'HTTP ' + res.status); }
    const data = await res.json();
    renderAssistant(data.reply, data.mode);
    if (data.mode === 'confirm') {
      updateModeUI('pending');
    } else if (data.mode === 'online') {
      updateModeUI('online');
      setTimeout(() => { updateModeUI('local'); isOnline = false; document.getElementById('mode-toggle').checked = false; }, 2000);
    } else {
      if (!isOnline) updateModeUI('local');
    }
    document.getElementById('session-label').textContent = 'session #' + data.session_id;
  } catch (err) {
    hideTyping();
    _appendBubble('（网络错误：' + err.message + '）', 'assistant', false, false);
    updateModeUI('local');
  } finally {
    isLoading = false;
    document.getElementById('send-btn').disabled = false;
    input.focus();
  }
}

async function saveSession() {
  if (isLoading) return;
  try {
    await fetch('/save', { method: 'POST' });
    document.getElementById('session-label').textContent = '已保存';
    const c = document.getElementById('messages');
    const el = document.createElement('div');
    el.className = 'sys-notice';
    el.textContent = '── 对话已保存，新消息将开启新 session ──';
    c.appendChild(el);
    c.scrollTop = c.scrollHeight;
  } catch (e) { alert('保存失败：' + e.message); }
}

function _esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function syncStatus() {
  try {
    const data = await fetch('/router/status').then(r => r.json());
    isOnline = data.mode === 'pending';
    document.getElementById('mode-toggle').checked = isOnline;
    updateModeUI(data.mode || 'local');
  } catch (_) {}
}
syncStatus();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


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
      3. 冲突回复检测（改为关键词优先匹配，见 _detect_conflict_action）
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
        target  = result["message"]
        history = get_messages_as_dicts(session_id)
        context = _build_context(session_id, user_message=req.content.strip())
        system  = build_system_prompt(context)
        # 去掉最后一条（即刚存入的 user 消息），单独作为当前轮传入
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
