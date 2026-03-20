"""
main.py — FastAPI 应用入口
启动 Web 服务，提供对话接口和浏览器对话页面。

接口列表：
  GET  /        — 返回浏览器对话页面（HTML）
  POST /chat    — 接收用户消息，返回助手回复（JSON）
  POST /save    — 手动立即保存并结束当前 session

启动方式：
  首次安装依赖：pip install fastapi uvicorn
  启动服务：    python main.py
  浏览器访问：  http://localhost:8000

/chat 路由的完整处理流程：
  1. 获取或新建当前活跃 session
  2. 保存用户消息
  3. 判断用户消息是否是对冲突询问的回复（"更新" / "忽略"）
     → 是：处理冲突回复，返回确认消息，不走正常对话流程
     → 否：继续往下走
  4. 检查是否有待确认的冲突
     → 有：直接返回冲突询问文本，不调用模型生成回复
     → 无：继续往下走
  5. 读取对话历史，构建带记忆的消息列表
  6. 调用本地模型生成回复
  7. 保存助手回复，返回结果

冲突处理的设计说明：
  每次对话最多处理一条待确认冲突（取最早的一条），避免一次性抛出多个问题。
  用户回复"更新"或"忽略"后，下次对话才会处理下一条（如果有的话）。
  冲突询问和正常对话共用同一个消息气泡，不需要特殊的前端适配。

依赖模块：
  config.py           — 配置项
  database.py         — 数据库读写
  session_manager.py  — session 生命周期管理
  prompt_builder.py   — system prompt 构建
  conflict_checker.py — 冲突检测与处理
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


# =============================================================================
# 全局 SessionManager 实例
# 整个应用共享一个，负责维护 session 状态和后台空闲检测
# =============================================================================

session_manager = SessionManager()


# =============================================================================
# 应用生命周期：启动 / 关闭
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器。
    yield 之前的代码在应用启动时执行（相当于 startup）。
    yield 之后的代码在应用关闭时执行（相当于 shutdown）。
    """
    # --- 启动阶段 ---
    print("[main] 应用启动中…")
    session_manager.start()   # 启动后台空闲检测线程和 L2 定时检查线程
    print("[main] 应用就绪，访问 http://localhost:8000 开始对话")

    yield   # 应用正常运行期间在这里等待

    # --- 关闭阶段 ---
    print("[main] 应用关闭中…")
    session_manager.stop()    # 优雅停止所有后台线程
    print("[main] 已停止")


# =============================================================================
# FastAPI 应用实例
# =============================================================================

app = FastAPI(
    title="个人陪伴助手",
    description="本地运行的个人 AI 陪伴助手，支持分层记忆。",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# 请求 / 响应数据模型
# =============================================================================

class ChatRequest(BaseModel):
    """
    /chat 接口的请求体。
    content: 用户发送的消息文本
    """
    content: str


class ChatResponse(BaseModel):
    """
    /chat 接口的响应体。
    reply:      助手的回复文本
    session_id: 当前 session 的 id（调试用）
    """
    reply: str
    session_id: int


# =============================================================================
# 冲突回复识别
# =============================================================================

# 用户回复"更新"时触发接受冲突的关键词列表（全部转为小写后匹配）
_RESOLVE_KEYWORDS = {"更新", "接受", "对", "是的", "没错", "update"}

# 用户回复"忽略"时触发忽略冲突的关键词列表
_IGNORE_KEYWORDS  = {"忽略", "不用", "不", "算了", "保持", "ignore"}


def _detect_conflict_action(text):
    """
    检测用户消息是否是对冲突询问的明确回复。

    只做简单的关键词匹配，不调用模型，避免引入额外延迟。
    消息较短（≤ 10 字）且命中关键词时才认定为冲突回复，
    消息较长时视为正常对话内容，避免把包含"更新"字样的正常句子误判为冲突回复。

    参数：
        text — 用户消息文本（已去除首尾空白）

    返回：
        "resolve" — 用户选择接受新内容
        "ignore"  — 用户选择忽略，保持现状
        None      — 不是冲突回复，走正常对话流程
    """
    # 消息过长时不做冲突回复判断，防止误判
    if len(text) > 10:
        return None

    text_lower = text.lower().strip()

    if text_lower in _RESOLVE_KEYWORDS:
        return "resolve"
    if text_lower in _IGNORE_KEYWORDS:
        return "ignore"

    return None


# =============================================================================
# 调用本地模型生成对话回复
# =============================================================================

def _call_local_model_for_chat(messages: list[dict]) -> str:
    """
    向 LM Studio 发送对话历史，获取助手回复。

    与 summarizer._call_local_model 的区别：
      - 这里传入完整的多轮对话历史，摘要模块只传单条 Prompt
      - 这里不需要 /no_think，正常对话允许模型自由思考
      - timeout 设为 120 秒，对话生成比摘要慢，需要更长等待时间

    参数：
        messages — 符合 OpenAI 格式的消息列表，首位是 system prompt，
                   其后是完整对话历史，如：
                   [
                       {"role": "system",    "content": "你是..."},
                       {"role": "user",      "content": "你好"},
                       {"role": "assistant", "content": "你好！"},
                       {"role": "user",      "content": "今天天气怎么样？"},
                   ]

    返回：
        str — 助手的回复文本（已去除首尾空白）
        str — 请求失败时返回括号包裹的错误提示字符串，不抛异常，让对话继续
    """
    payload = {
        "model":       LOCAL_MODEL_NAME,
        "messages":    messages,
        "temperature": LOCAL_TEMPERATURE,
        "max_tokens":  LOCAL_MAX_TOKENS,
    }

    try:
        response = requests.post(
            LOCAL_API_URL,
            json=payload,
            timeout=120,   # 对话生成比摘要慢，给足 120 秒
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return "（错误：无法连接到本地模型，请确认 LM Studio 已启动）"
    except requests.exceptions.Timeout:
        return "（错误：模型响应超时，请稍后重试）"
    except requests.exceptions.HTTPError as e:
        return f"（错误：HTTP 请求失败 — {e}）"
    except (KeyError, IndexError):
        return "（错误：解析模型响应失败）"


# =============================================================================
# 路由：GET / — 对话页面
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    返回浏览器对话页面。
    单文件 HTML，内嵌 CSS 和 JS，不依赖外部资源，离线可用。
    """
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>陪伴助手</title>
  <style>
    /* ── 基础重置 ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    /* ── 全局变量 ── */
    :root {
      --bg:        #0f0f11;
      --surface:   #1a1a1f;
      --border:    #2a2a32;
      --accent:    #a78bfa;
      --accent-dim:#6d5fc4;
      --text:      #e8e8f0;
      --text-dim:  #888899;
      --user-bg:   #1e1b2e;
      --ai-bg:     #161618;
      --radius:    14px;
      --font:      'Georgia', serif;
      --font-mono: 'Courier New', monospace;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      height: 100dvh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 0 16px;
    }

    /* ── 顶栏 ── */
    header {
      width: 100%;
      max-width: 720px;
      padding: 20px 0 12px;
      display: flex;
      align-items: baseline;
      gap: 12px;
      border-bottom: 1px solid var(--border);
    }
    header h1 {
      font-size: 1.1rem;
      font-weight: normal;
      letter-spacing: .06em;
      color: var(--accent);
    }
    header span {
      font-size: .75rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }
    #save-btn {
      margin-left: auto;
      background: none;
      border: 1px solid var(--border);
      color: var(--text-dim);
      font-size: .75rem;
      padding: 4px 12px;
      border-radius: 20px;
      cursor: pointer;
      transition: border-color .2s, color .2s;
      font-family: var(--font-mono);
    }
    #save-btn:hover { border-color: var(--accent); color: var(--accent); }

    /* ── 消息区 ── */
    #messages {
      width: 100%;
      max-width: 720px;
      flex: 1;
      overflow-y: auto;
      padding: 24px 0;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scrollbar-width: thin;
      scrollbar-color: var(--border) transparent;
    }

    .msg {
      display: flex;
      flex-direction: column;
      gap: 4px;
      animation: fadeUp .25s ease both;
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .msg-label {
      font-size: .7rem;
      font-family: var(--font-mono);
      color: var(--text-dim);
      letter-spacing: .08em;
    }
    .msg.user .msg-label { text-align: right; color: var(--accent-dim); }

    .msg-bubble {
      padding: 12px 16px;
      border-radius: var(--radius);
      line-height: 1.7;
      font-size: .95rem;
      white-space: pre-wrap;
      word-break: break-word;
      max-width: 88%;
    }
    .msg.user  { align-items: flex-end; }
    .msg.user  .msg-bubble {
      background: var(--user-bg);
      border: 1px solid #2d2547;
    }
    .msg.assistant .msg-bubble {
      background: var(--ai-bg);
      border: 1px solid var(--border);
    }

    /* 思考链折叠块 */
    .think-block {
      font-size: .8rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      border-left: 2px solid var(--border);
      padding: 6px 10px;
      margin-bottom: 8px;
      cursor: pointer;
      user-select: none;
    }
    .think-content { display: none; margin-top: 6px; }
    .think-block.open .think-content { display: block; }

    /* 加载动画 */
    .typing span {
      display: inline-block;
      width: 5px; height: 5px;
      border-radius: 50%;
      background: var(--text-dim);
      margin: 0 2px;
      animation: bounce .9s infinite;
    }
    .typing span:nth-child(2) { animation-delay: .15s; }
    .typing span:nth-child(3) { animation-delay: .30s; }
    @keyframes bounce {
      0%,80%,100% { transform: translateY(0); }
      40%          { transform: translateY(-6px); }
    }

    /* ── 输入区 ── */
    #input-area {
      width: 100%;
      max-width: 720px;
      padding: 12px 0 24px;
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
      font-family: var(--font);
      font-size: .95rem;
      padding: 12px 16px;
      resize: none;
      line-height: 1.6;
      min-height: 48px;
      max-height: 160px;
      outline: none;
      transition: border-color .2s;
    }
    #user-input:focus { border-color: var(--accent-dim); }
    #user-input::placeholder { color: var(--text-dim); }

    #send-btn {
      background: var(--accent-dim);
      border: none;
      color: #fff;
      width: 48px; height: 48px;
      border-radius: 12px;
      cursor: pointer;
      font-size: 1.1rem;
      transition: background .2s, transform .1s;
      flex-shrink: 0;
    }
    #send-btn:hover   { background: var(--accent); }
    #send-btn:active  { transform: scale(.95); }
    #send-btn:disabled { background: var(--border); cursor: not-allowed; }
  </style>
</head>
<body>

<header>
  <h1>陪伴助手</h1>
  <span id="session-label">初始化中…</span>
  <button id="save-btn" onclick="saveSession()">保存并结束</button>
</header>

<div id="messages"></div>

<div id="input-area">
  <textarea
    id="user-input"
    placeholder="输入消息，Enter 发送，Shift+Enter 换行"
    rows="1"
  ></textarea>
  <button id="send-btn" onclick="sendMessage()">↑</button>
</div>

<script>
  // ── 状态 ──────────────────────────────────────────────────
  let isLoading = false;

  // ── 自动调整输入框高度 ────────────────────────────────────
  const textarea = document.getElementById('user-input');
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  });

  // Enter 发送，Shift+Enter 换行
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // ── 渲染消息 ──────────────────────────────────────────────
  function renderMessage(role, text) {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg ' + role;

    const label = document.createElement('div');
    label.className = 'msg-label';
    label.textContent = role === 'user' ? '你' : '助手';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    // 如果是助手消息，检测并折叠思考链 <think>...</think>
    if (role === 'assistant') {
      const thinkMatch = text.match(/^<think>([\\s\\S]*?)<\\/think>\\s*/);
      if (thinkMatch) {
        const thinkDiv = document.createElement('div');
        thinkDiv.className = 'think-block';
        thinkDiv.innerHTML =
          '▸ 思考过程（点击展开）' +
          '<div class="think-content">' + escapeHtml(thinkMatch[1].trim()) + '</div>';
        thinkDiv.onclick = () => thinkDiv.classList.toggle('open');
        bubble.appendChild(thinkDiv);
        text = text.slice(thinkMatch[0].length);
      }
      bubble.appendChild(document.createTextNode(text));
    } else {
      bubble.textContent = text;
    }

    div.appendChild(label);
    div.appendChild(bubble);
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
  }

  function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // ── 加载动画 ──────────────────────────────────────────────
  function showTyping() {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.id = 'typing-indicator';
    div.innerHTML =
      '<div class="msg-label">助手</div>' +
      '<div class="msg-bubble"><div class="typing">' +
      '<span></span><span></span><span></span></div></div>';
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }
  function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  // ── 发送消息 ──────────────────────────────────────────────
  async function sendMessage() {
    if (isLoading) return;
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text) return;

    // 清空输入框
    input.value = '';
    input.style.height = 'auto';

    // 渲染用户消息
    renderMessage('user', text);

    // 锁定发送，防止重复提交
    isLoading = true;
    document.getElementById('send-btn').disabled = true;
    showTyping();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '请求失败');
      }

      const data = await res.json();
      hideTyping();
      renderMessage('assistant', data.reply);

      // 更新顶栏 session 标签
      document.getElementById('session-label').textContent =
        'session #' + data.session_id;

    } catch (err) {
      hideTyping();
      renderMessage('assistant', '（网络错误：' + err.message + '）');
    } finally {
      isLoading = false;
      document.getElementById('send-btn').disabled = false;
      input.focus();
    }
  }

  // ── 保存并结束当前 session ────────────────────────────────
  async function saveSession() {
    if (isLoading) return;
    try {
      await fetch('/save', { method: 'POST' });
      document.getElementById('session-label').textContent = '已保存';
      // 在消息区显示一条系统提示
      const container = document.getElementById('messages');
      const div = document.createElement('div');
      div.style.cssText =
        'text-align:center;color:var(--text-dim);font-size:.75rem;' +
        'font-family:var(--font-mono);padding:8px 0;';
      div.textContent = '── 对话已保存，新消息将开启新 session ──';
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
    } catch (e) {
      alert('保存失败：' + e.message);
    }
  }
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# =============================================================================
# 路由：POST /chat — 接收消息，返回回复
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    核心对话接口。

    完整处理流程（见文件头注释）：
      1. 获取或新建当前活跃 session
      2. 保存用户消息
      3. 检测是否是冲突回复 → 是则处理后直接返回，不走模型
      4. 检查是否有待确认冲突 → 有则返回冲突询问，不走模型
      5. 构建带记忆的消息列表
      6. 调用本地模型生成回复
      7. 保存助手回复，返回结果

    请求体：{ "content": "用户消息文本" }
    响应体：{ "reply": "助手回复文本", "session_id": 当前session id }
    """
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    # ------------------------------------------------------------------
    # 第一步：获取或新建当前活跃 session
    # ------------------------------------------------------------------
    session_id = session_manager.on_message()

    # ------------------------------------------------------------------
    # 第二步：保存用户消息到数据库
    # 必须在判断冲突之前保存，确保 L0 消息流水完整
    # ------------------------------------------------------------------
    save_message(session_id, "user", req.content)

    # ------------------------------------------------------------------
    # 第三步：检测用户消息是否是对冲突询问的明确回复
    #
    # 流程：
    #   a. 用简单关键词匹配判断是否是"更新"/"忽略"类回复
    #   b. 是的话查询最早一条 pending 冲突
    #   c. 调用 handle_conflict_reply() 处理，返回确认消息
    #   d. 把确认消息也保存到数据库，让对话历史保持完整
    #   e. 直接返回，不调用模型，节省一次推理时间
    # ------------------------------------------------------------------
    action = _detect_conflict_action(req.content.strip())

    if action is not None:
        # 用户回复了"更新"或"忽略"，检查是否真的有待处理的冲突
        conflict_result = get_conflict_question()

        if conflict_result is not None:
            conflict_id, _ = conflict_result
            # 处理冲突：更新画像或标记忽略
            reply = handle_conflict_reply(conflict_id, action)
            # 保存助手的确认回复，保持对话历史完整
            save_message(session_id, "assistant", reply)
            print(f"[main] 冲突 {conflict_id} 已处理，action={action}")
            return ChatResponse(reply=reply, session_id=session_id)

        # 用了"更新"/"忽略"这类词，但没有待处理的冲突
        # 让消息正常进入对话流程，不做特殊处理
        print(f"[main] 检测到冲突关键词但无待处理冲突，按正常对话处理")

    # ------------------------------------------------------------------
    # 第四步：检查是否有尚未询问用户的待确认冲突
    #
    # 流程：
    #   a. 查询 conflict_queue 里最早一条 pending 记录
    #   b. 有的话，把冲突询问文本作为助手回复直接返回
    #   c. 同样保存到数据库，让对话历史保持完整
    #   d. 直接返回，本次不调用模型生成正常回复
    #
    # 设计说明：
    #   每次对话只插入一条冲突询问，避免一次性抛出多个问题让用户困惑。
    #   用户处理完这条后，下次发消息会重新走第三步，处理下一条冲突（如有）。
    # ------------------------------------------------------------------
    conflict_result = get_conflict_question()

    if conflict_result is not None:
        conflict_id, question = conflict_result
        # 把冲突询问作为助手的这轮回复
        save_message(session_id, "assistant", question)
        print(f"[main] 有待确认冲突 id={conflict_id}，已向用户展示询问")
        return ChatResponse(reply=question, session_id=session_id)

    # ------------------------------------------------------------------
    # 第五步：读取对话历史，构建带记忆注入的消息列表
    #
    # system prompt 由 prompt_builder 每次动态构建，包含四层记忆：
    #   - L3 用户画像（稳定层）
    #   - 近期 L2 摘要（近期层）
    #   - 最新 L1 摘要（当日层）
    #   - 语义检索结果（动态层）← 阶段二新增，基于当前消息实时检索
    #
    # 将用户当前消息传入 build_system_prompt(query_text=...)，
    # 动态层会同时用这条消息 + 最新 L1 内容两路检索相关历史记忆，
    # 合并去重后注入 prompt，让模型能"主动想起"相关历史。
    # ------------------------------------------------------------------
    history = get_messages_as_dicts(session_id)
    system_prompt = build_system_prompt(query_text=req.content.strip())
    messages_with_memory = [
        {"role": "system", "content": system_prompt},
        *history,
    ]

    # ------------------------------------------------------------------
    # 第六步：调用本地模型生成回复
    # ------------------------------------------------------------------
    reply = _call_local_model_for_chat(messages_with_memory)

    # ------------------------------------------------------------------
    # 第七步：保存助手回复，返回结果
    # ------------------------------------------------------------------
    save_message(session_id, "assistant", reply)

    return ChatResponse(reply=reply, session_id=session_id)


# =============================================================================
# 路由：POST /save — 手动保存并结束当前 session
# =============================================================================

@app.post("/save")
async def save():
    """
    手动立即结束当前 session 并触发 L1 摘要生成。
    对应页面上的"保存并结束"按钮。

    正常情况下 session 由后台空闲检测线程自动关闭，
    此接口供用户主动触发（例如对话告一段落时立即保存）。
    """
    session_manager.force_close_current_session()
    return JSONResponse({"status": "ok", "message": "session 已保存"})


# =============================================================================
# 启动入口
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=DEBUG,    # DEBUG=True 时代码改动后自动重启，开发阶段很方便
    )
