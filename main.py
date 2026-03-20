"""
main.py — FastAPI 应用入口
版本：0.3.0（浅色主题 + 路由控制面板重构）
=====================================================================

界面更新：
  · 浅色系配色，米白底色，珊瑚色主调
  · 标题改为"珊瑚菌 · 个人 AI 陪伴助手"
  · 顶栏：session 标签、模式指示灯、线上/本地 toggle、保存按钮
  · 助手消息支持 || 拆分为多条气泡

后端逻辑（与 0.2.0 一致）：
  · router.py 管理本地/线上模式切换
  · 本地走 LM Studio Qwen，线上走 Claude API
  · Claude 不带记忆上下文，用完即回本地

接口列表：
  GET  /               — 对话页面
  POST /chat           — 核心对话接口
  POST /save           — 手动保存 session
  GET  /router/status  — 查询路由状态
  POST /router/toggle  — 切换线上/本地模式
"""

from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import (
    DEBUG,
    LOCAL_API_URL,
    LOCAL_MAX_TOKENS,
    LOCAL_MODEL_NAME,
    LOCAL_TEMPERATURE,
    SERVER_HOST,
    SERVER_PORT,
)
from database import get_messages_as_dicts, save_message
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
    version     = "0.3.0",
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
# 冲突回复关键词（与原版一致）
# =============================================================================

_RESOLVE_KEYWORDS = {"更新", "接受", "对", "是的", "没错", "update"}
_IGNORE_KEYWORDS  = {"忽略", "不用", "不", "算了", "保持", "ignore"}


def _detect_conflict_action(text):
    """检测用户消息是否是对冲突询问的回复。"""
    if len(text) > 10:
        return None
    t = text.lower().strip()
    if t in _RESOLVE_KEYWORDS:
        return "resolve"
    if t in _IGNORE_KEYWORDS:
        return "ignore"
    return None


# =============================================================================
# 调用本地模型
# =============================================================================

def _call_local(messages: list[dict]) -> str:
    """
    向 LM Studio 发送完整对话历史，获取助手回复。

    参数：
        messages — OpenAI 格式消息列表，首位是 system prompt

    返回：
        str — 模型回复文本；失败时返回括号包裹的错误说明
    """
    payload = {
        "model":       LOCAL_MODEL_NAME,
        "messages":    messages,
        "temperature": LOCAL_TEMPERATURE,
        "max_tokens":  LOCAL_MAX_TOKENS,
    }
    try:
        resp = requests.post(LOCAL_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "（错误：无法连接到本地模型，请确认 LM Studio 已启动）"
    except requests.exceptions.Timeout:
        return "（错误：模型响应超时）"
    except requests.exceptions.HTTPError as e:
        return f"（错误：{e}）"
    except (KeyError, IndexError):
        return "（错误：解析模型响应失败）"


# =============================================================================
# 对话页面 HTML
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    返回完整对话页面。CSS/JS 全部内联，无外部依赖，离线可用。

    设计风格：
        浅色系，米白底色，珊瑚橙主调，衬线字体。
        温暖、清爽，适合长时间使用。
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
    /* ── 全局重置 ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    /* ── 设计令牌 ── */
    :root {
      --bg:           #f7f4ef;   /* 页面底色：暖米白 */
      --surface:      #ffffff;   /* 气泡白 */
      --surface-alt:  #fdf8f3;   /* 用户气泡淡杏 */
      --border:       #e8e0d4;
      --border-light: #ede7dd;
      --text:         #2d2720;   /* 深棕主文字 */
      --text-mid:     #7a6e63;
      --text-dim:     #b0a89e;
      --coral:        #c97c5a;   /* 珊瑚橙主色 */
      --coral-light:  #e8c4b0;
      --coral-dim:    #d4956f;
      --dot-local:    #7cad7c;   /* 绿：本地 */
      --dot-online:   #c97c5a;   /* 珊瑚：线上 */
      --dot-pending:  #7fafd4;   /* 蓝：等待确认 */
      --font-title:   'Noto Serif SC', Georgia, serif;
      --font-body:    'Noto Serif SC', Georgia, serif;
      --font-mono:    'SFMono-Regular', 'Consolas', monospace;
      --radius:       16px;
      --radius-sm:    8px;
    }

    html, body { height: 100%; }

    body {
      background: var(--bg);
      /* 细腻纸纹 */
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
      color: var(--text);
      font-family: var(--font-body);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 0 16px;
    }

    /* ══ 顶栏 ══ */
    header {
      width: 100%;
      max-width: 740px;
      padding: 16px 0 12px;
      display: flex;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      background: var(--bg);
      z-index: 10;
    }

    /* 标题块 */
    .header-title {
      display: flex;
      flex-direction: column;
      gap: 1px;
      flex-shrink: 0;
    }
    .title-main {
      font-family: var(--font-title);
      font-size: .98rem;
      font-weight: 500;
      color: var(--coral);
      letter-spacing: .06em;
      line-height: 1;
    }
    .title-sub {
      font-size: .62rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      letter-spacing: .05em;
    }

    /* session 标签 */
    #session-label {
      font-size: .66rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      background: var(--border-light);
      padding: 2px 8px;
      border-radius: 20px;
      flex-shrink: 0;
    }

    /* 右侧控制区 */
    #controls {
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 9px;
    }

    /* 模式指示器 */
    #mode-badge {
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: .66rem;
      font-family: var(--font-mono);
      color: var(--text-mid);
      flex-shrink: 0;
    }
    #mode-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--dot-local);
      transition: background .4s;
      flex-shrink: 0;
    }
    #mode-dot.online  { background: var(--dot-online); }
    #mode-dot.pending {
      background: var(--dot-pending);
      animation: blink 1.4s ease-in-out infinite;
    }
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50%       { opacity: .2; }
    }

    /* 分隔线 */
    .sep {
      width: 1px;
      height: 14px;
      background: var(--border);
      flex-shrink: 0;
    }

    /* ── Toggle 开关 ──
       label 包裹隐藏的 checkbox + 可见轨道。
       :checked 状态 = 线上模式。
    ── */
    .toggle-wrap {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    .toggle-label-text {
      font-size: .66rem;
      font-family: var(--font-mono);
      color: var(--text-mid);
      white-space: nowrap;
    }
    /* 隐藏原生 checkbox，保留可访问性 */
    .toggle-input { display: none; }

    /* Toggle 轨道（用 ::after 作滑块） */
    .toggle-track {
      position: relative;
      width: 34px;
      height: 18px;
      border-radius: 9px;
      background: var(--border);
      transition: background .25s;
      flex-shrink: 0;
    }
    .toggle-track::after {
      content: '';
      position: absolute;
      top: 2px;
      left: 2px;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--text-dim);
      transition: transform .25s, background .25s;
    }
    /* 开启（线上）状态 */
    .toggle-input:checked + .toggle-track {
      background: var(--coral-light);
    }
    .toggle-input:checked + .toggle-track::after {
      transform: translateX(16px);
      background: var(--coral);
    }

    /* 保存按钮 */
    #save-btn {
      background: none;
      border: 1px solid var(--border);
      color: var(--text-mid);
      font-size: .66rem;
      font-family: var(--font-mono);
      padding: 3px 10px;
      border-radius: 20px;
      cursor: pointer;
      white-space: nowrap;
      flex-shrink: 0;
      transition: border-color .2s, color .2s, background .2s;
    }
    #save-btn:hover {
      border-color: var(--coral-light);
      color: var(--coral);
      background: rgba(201,124,90,.06);
    }

    /* ══ 消息区 ══ */
    #messages {
      width: 100%;
      max-width: 740px;
      flex: 1;
      overflow-y: auto;
      padding: 26px 0 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      scrollbar-width: thin;
      scrollbar-color: var(--border) transparent;
    }

    /* 单条消息容器 */
    .msg {
      display: flex;
      flex-direction: column;
      gap: 3px;
      animation: rise .2s ease both;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(5px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* 发言方标签 */
    .msg-label {
      font-size: .63rem;
      font-family: var(--font-mono);
      color: var(--text-dim);
      letter-spacing: .05em;
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .msg.user .msg-label {
      justify-content: flex-end;
      color: var(--coral-light);
    }

    /* 线上回复小标签 */
    .tag-online {
      font-size: .58rem;
      color: var(--coral);
      border: 1px solid var(--coral-light);
      border-radius: 3px;
      padding: 0 3px;
      line-height: 1.5;
    }

    /* 气泡 */
    .msg-bubble {
      padding: 11px 16px;
      border-radius: var(--radius);
      line-height: 1.78;
      font-size: .92rem;
      white-space: pre-wrap;
      word-break: break-word;
      max-width: 84%;
      box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }

    /* 用户气泡：右对齐，淡杏 */
    .msg.user { align-items: flex-end; }
    .msg.user .msg-bubble {
      background: var(--surface-alt);
      border: 1px solid var(--border-light);
    }

    /* 助手气泡：左对齐，纯白 */
    .msg.assistant .msg-bubble {
      background: var(--surface);
      border: 1px solid var(--border);
    }

    /* 确认询问：蓝色虚线边框，提示用户需要操作 */
    .msg.assistant.confirm .msg-bubble {
      border: 1.5px dashed var(--dot-pending);
    }

    /* 线上模式回复：珊瑚色左边装饰线 */
    .msg.assistant.is-online .msg-bubble {
      border-left: 3px solid var(--coral-light);
    }

    /* 思考链折叠块 */
    .think-block {
      font-size: .73rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      border-left: 2px solid var(--border);
      padding: 4px 10px;
      margin-bottom: 8px;
      cursor: pointer;
      user-select: none;
      transition: color .2s;
    }
    .think-block:hover { color: var(--text-mid); }
    .think-content { display: none; margin-top: 5px; }
    .think-block.open .think-content { display: block; }

    /* 系统提示条 */
    .sys-notice {
      text-align: center;
      color: var(--text-dim);
      font-size: .68rem;
      font-family: var(--font-mono);
      padding: 5px 0;
      letter-spacing: .04em;
    }

    /* 输入中动画 */
    .typing { display: flex; gap: 4px; align-items: center; padding: 2px 0; }
    .typing span {
      width: 5px; height: 5px;
      border-radius: 50%;
      background: var(--coral-light);
      animation: bounce .9s infinite;
    }
    .typing span:nth-child(2) { animation-delay: .15s; }
    .typing span:nth-child(3) { animation-delay: .30s; }
    @keyframes bounce {
      0%,80%,100% { transform: translateY(0); }
      40%          { transform: translateY(-5px); }
    }

    /* ══ 输入区 ══ */
    #input-area {
      width: 100%;
      max-width: 740px;
      padding: 10px 0 22px;
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }

    #user-input {
      flex: 1;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      font-family: var(--font-body);
      font-size: .92rem;
      padding: 12px 16px;
      resize: none;
      line-height: 1.6;
      min-height: 48px;
      max-height: 160px;
      outline: none;
      box-shadow: 0 1px 3px rgba(0,0,0,.04);
      transition: border-color .2s, box-shadow .2s;
    }
    #user-input:focus {
      border-color: var(--coral-light);
      box-shadow: 0 0 0 3px rgba(201,124,90,.1);
    }
    #user-input::placeholder { color: var(--text-dim); }

    #send-btn {
      width: 48px; height: 48px;
      border-radius: var(--radius-sm);
      border: none;
      background: var(--coral);
      color: #fff;
      font-size: 1.1rem;
      cursor: pointer;
      flex-shrink: 0;
      box-shadow: 0 2px 8px rgba(201,124,90,.35);
      transition: background .2s, transform .1s, box-shadow .2s;
    }
    #send-btn:hover:not(:disabled) {
      background: var(--coral-dim);
      box-shadow: 0 3px 12px rgba(201,124,90,.45);
    }
    #send-btn:active:not(:disabled) { transform: scale(.95); }
    #send-btn:disabled {
      background: var(--border);
      box-shadow: none;
      cursor: not-allowed;
    }
  </style>
</head>
<body>

<!-- ══ 顶栏 ══ -->
<header>
  <div class="header-title">
    <span class="title-main">珊瑚菌 · 个人 AI 陪伴助手</span>
    <span class="title-sub">Ramaria · Personal Companion</span>
  </div>

  <span id="session-label">初始化中</span>

  <div id="controls">
    <!-- 模式指示灯 -->
    <div id="mode-badge">
      <div id="mode-dot"></div>
      <span id="mode-text">本地</span>
    </div>

    <div class="sep"></div>

    <!-- 线上/本地 Toggle
         checked = 线上模式，onchange 通知后端 -->
    <label class="toggle-wrap" title="切换本地 / 线上模式">
      <span class="toggle-label-text">线上模式</span>
      <input type="checkbox" id="mode-toggle" class="toggle-input"
             onchange="onToggleChange(this.checked)">
      <div class="toggle-track"></div>
    </label>

    <div class="sep"></div>

    <button id="save-btn" onclick="saveSession()">保存对话</button>
  </div>
</header>

<!-- ══ 消息列表 ══ -->
<div id="messages"></div>

<!-- ══ 输入区 ══ -->
<div id="input-area">
  <textarea id="user-input"
    placeholder="输入消息，Enter 发送，Shift+Enter 换行"
    rows="1"></textarea>
  <button id="send-btn" onclick="sendMessage()">↑</button>
</div>

<script>
// ──────────────────────────────────────────────
// 全局状态
// ──────────────────────────────────────────────
let isLoading = false;   // 防止重复提交
let isOnline  = false;   // Toggle 当前状态


// ──────────────────────────────────────────────
// 输入框自动调高 + 快捷键
// ──────────────────────────────────────────────
const textarea = document.getElementById('user-input');

textarea.addEventListener('input', () => {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
});

textarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});


// ──────────────────────────────────────────────
// Toggle 切换：通知后端，更新指示灯
// ──────────────────────────────────────────────
async function onToggleChange(checked) {
  isOnline = checked;
  updateModeUI(checked ? 'pending' : 'local');

  try {
    const res  = await fetch('/router/toggle', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ online: checked }),
    });
    const data = await res.json();

    // 后端返回提示文字时，显示在对话区
    if (data.message) {
      _appendBubble(data.message, 'assistant', false, false);
    }
    updateModeUI(data.mode || (checked ? 'pending' : 'local'));

  } catch (_) {
    // 网络失败：回滚 Toggle
    isOnline = !checked;
    document.getElementById('mode-toggle').checked = !checked;
    updateModeUI('local');
  }
}


// ──────────────────────────────────────────────
// 更新模式指示灯
// mode: 'local' | 'online' | 'pending'
// ──────────────────────────────────────────────
function updateModeUI(mode) {
  const dot  = document.getElementById('mode-dot');
  const text = document.getElementById('mode-text');
  dot.className = '';
  switch (mode) {
    case 'online':
      dot.classList.add('online');
      text.textContent = '线上';
      break;
    case 'pending':
      dot.classList.add('pending');
      text.textContent = '等待确认';
      break;
    default:
      text.textContent = '本地';
  }
}


// ──────────────────────────────────────────────
// 渲染助手回复
// 支持 || 拆分为多条气泡（模拟真人连发短消息）
// ──────────────────────────────────────────────
function renderAssistant(text, mode) {
  hideTyping();
  const parts = text.split('||').map(s => s.trim()).filter(Boolean);
  parts.forEach((part, i) => {
    const last      = i === parts.length - 1;
    const isOnlineMsg = last && mode === 'online';
    const isConfirm   = last && mode === 'confirm';
    _appendBubble(part, 'assistant', isOnlineMsg, isConfirm);
  });
}


// ──────────────────────────────────────────────
// 内部：生成并追加单个气泡
// ──────────────────────────────────────────────
function _appendBubble(text, role, isOnlineMsg, isConfirm) {
  const container = document.getElementById('messages');

  const div = document.createElement('div');
  div.className = 'msg ' + role
    + (isConfirm   ? ' confirm'   : '')
    + (isOnlineMsg ? ' is-online' : '');

  // 标签
  const label = document.createElement('div');
  label.className = 'msg-label';
  if (role === 'user') {
    label.textContent = '你';
  } else {
    label.textContent = '助手';
    if (isOnlineMsg) {
      const tag = document.createElement('span');
      tag.className = 'tag-online';
      tag.textContent = 'API';
      label.appendChild(tag);
    }
  }

  // 气泡
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  // 思考链折叠（仅助手消息）
  if (role === 'assistant') {
    const m = text.match(/^<think>([\s\S]*?)<\/think>\s*/);
    if (m) {
      const td = document.createElement('div');
      td.className = 'think-block';
      td.innerHTML = '▸ 思考过程（点击展开）<div class="think-content">'
        + _esc(m[1].trim()) + '</div>';
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


// ──────────────────────────────────────────────
// Typing 占位符
// ──────────────────────────────────────────────
function showTyping() {
  const c  = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.id = 'typing-indicator';
  el.innerHTML =
    '<div class="msg-label">助手</div>' +
    '<div class="msg-bubble"><div class="typing">' +
    '<span></span><span></span><span></span></div></div>';
  c.appendChild(el);
  c.scrollTop = c.scrollHeight;
}
function hideTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}


// ──────────────────────────────────────────────
// 发送消息
// ──────────────────────────────────────────────
async function sendMessage() {
  if (isLoading) return;
  const input = document.getElementById('user-input');
  const text  = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  _appendBubble(text, 'user', false, false);

  isLoading = true;
  document.getElementById('send-btn').disabled = true;
  showTyping();

  try {
    const res = await fetch('/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ content: text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'HTTP ' + res.status);
    }
    const data = await res.json();

    // 渲染回复（支持 || 拆分）
    renderAssistant(data.reply, data.mode);

    // 同步指示灯
    if (data.mode === 'confirm') {
      updateModeUI('pending');
    } else if (data.mode === 'online') {
      updateModeUI('online');
      // 线上回复完成后 2s 恢复本地，并把 Toggle 拨回
      setTimeout(() => {
        updateModeUI('local');
        isOnline = false;
        document.getElementById('mode-toggle').checked = false;
      }, 2000);
    } else {
      if (!isOnline) updateModeUI('local');
    }

    document.getElementById('session-label').textContent =
      'session #' + data.session_id;

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


// ──────────────────────────────────────────────
// 保存 session
// ──────────────────────────────────────────────
async function saveSession() {
  if (isLoading) return;
  try {
    await fetch('/save', { method: 'POST' });
    document.getElementById('session-label').textContent = '已保存';
    const c  = document.getElementById('messages');
    const el = document.createElement('div');
    el.className   = 'sys-notice';
    el.textContent = '── 对话已保存，新消息将开启新 session ──';
    c.appendChild(el);
    c.scrollTop = c.scrollHeight;
  } catch (e) {
    alert('保存失败：' + e.message);
  }
}


// ──────────────────────────────────────────────
// HTML 转义（防 XSS）
// ──────────────────────────────────────────────
function _esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}


// ──────────────────────────────────────────────
// 初始化：同步后端路由状态
// ──────────────────────────────────────────────
async function syncStatus() {
  try {
    const data = await fetch('/router/status').then(r => r.json());
    isOnline   = data.mode === 'pending';
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
    """返回当前路由状态，供前端初始化。"""
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
        target  = result["message"]
        history = get_messages_as_dicts(session_id)
        system  = build_system_prompt(query_text=target)
        msgs    = [{"role": "system", "content": system}, *history]
        reply   = _call_local(msgs)
        save_message(session_id, "assistant", reply)
        return ChatResponse(reply=reply, session_id=session_id, mode="local")

    # D. 默认本地
    history = get_messages_as_dicts(session_id)
    system  = build_system_prompt(query_text=req.content.strip())
    msgs    = [{"role": "system", "content": system}, *history]
    reply   = _call_local(msgs)
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
