/**
 * ui.js — UI 渲染模块
 * =============================================================================
 * 负责所有 DOM 操作：
 *   · 消息气泡的创建与插入
 *   · 打字动画的显示/隐藏
 *   · 模式状态徽章的更新
 *   · 系统提示条的插入
 *   · Toast 通知的显示
 *   · 输入框高度的自动调整
 *
 * 依赖：
 *   · AppState（state.js）— 读取当前状态
 *
 * 对外暴露：
 *   UI.appendBubble(text, role, isOnlineMsg, isConfirm)
 *   UI.renderAssistantReply(text, mode)
 *   UI.showTyping()
 *   UI.hideTyping()
 *   UI.updateModeUI(mode)
 *   UI.appendSysNotice(text)
 *   UI.showToast(message, type, duration)
 *   UI.autoResizeTextarea(textarea)
 *   UI.scrollToBottom()
 * =============================================================================
 */

const UI = (() => {

  // Toast 自动消失的计时器 id，用于在新 toast 出现时清除上一个
  let _toastTimer = null;


  /**
   * HTML 转义。
   * 防止消息内容中的特殊字符被解析为 HTML 标签（XSS 防护）。
   * 注意：普通消息内容通过 createTextNode 插入，不需要调用此函数；
   *       此函数仅用于思考链内容（因为需要 innerHTML 才能支持折叠结构）。
   *
   * @param {string} s - 原始字符串
   * @returns {string} - 转义后的字符串
   */
  function _esc(s) {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }


  /**
   * 滚动消息列表到底部。
   * 每次插入新消息后调用，确保最新消息可见。
   */
  function scrollToBottom() {
    const container = document.getElementById('messages');
    if (container) container.scrollTop = container.scrollHeight;
  }


  /**
   * 创建并插入一条消息气泡。
   * 所有消息（用户/助手/系统）都通过此函数创建，确保结构统一。
   *
   * @param {string}  text        - 气泡文本内容
   * @param {'user'|'assistant'} role - 发言方
   * @param {boolean} isOnlineMsg - 是否是线上（Claude API）回复
   * @param {boolean} isConfirm   - 是否是等待确认消息（蓝色虚线边框）
   */
  function appendBubble(text, role, isOnlineMsg, isConfirm) {
    const container = document.getElementById('messages');
    if (!container) return;

    // ── 外层容器 ──
    const div = document.createElement('div');
    div.className = [
      'msg',
      role,
      isConfirm   ? 'confirm'   : '',
      isOnlineMsg ? 'is-online' : '',
    ].filter(Boolean).join(' ');

    // ── 发言方标签 ──
    const label = document.createElement('div');
    label.className = 'msg-label';

    if (role === 'user') {
      label.textContent = '你';
    } else {
      label.textContent = '助手';
      if (isOnlineMsg) {
        // 线上模式额外显示 API 来源标签
        const tag = document.createElement('span');
        tag.className   = 'tag-online';
        tag.textContent = 'API';
        label.appendChild(tag);
      }
    }

    // ── 气泡主体 ──
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    // 助手消息：尝试提取并折叠思考链 <think>...</think>
    if (role === 'assistant') {
      const thinkMatch = text.match(/^<think>([\s\S]*?)<\/think>\s*/);
      if (thinkMatch) {
        // 构建可折叠的思考过程块
        const thinkDiv = document.createElement('div');
        thinkDiv.className = 'think-block';
        // think-content 内容使用转义后的 HTML，防止 XSS
        thinkDiv.innerHTML =
          '▸ 思考过程（点击展开）'
          + `<div class="think-content">${_esc(thinkMatch[1].trim())}</div>`;
        thinkDiv.addEventListener('click', () => thinkDiv.classList.toggle('open'));
        bubble.appendChild(thinkDiv);
        // 去掉 <think>...</think> 部分，只展示实际回复文本
        text = text.slice(thinkMatch[0].length);
      }
    }

    // 消息正文用 createTextNode 插入，保证特殊字符不被解析为 HTML
    bubble.appendChild(document.createTextNode(text));

    div.appendChild(label);
    div.appendChild(bubble);
    container.appendChild(div);

    scrollToBottom();
  }


  /**
   * 渲染助手回复。
   * 支持用 || 分隔的多段消息：每段独立渲染为一个气泡，模拟多条消息发送。
   *
   * @param {string} text  - 助手回复全文（可能包含 || 分隔符）
   * @param {string} mode  - 回复模式：'local' | 'online' | 'confirm'
   */
  function renderAssistantReply(text, mode) {
    hideTyping();

    // 按 || 拆分，过滤空段落
    const parts = text.split('||').map(s => s.trim()).filter(Boolean);

    parts.forEach((part, i) => {
      const isLast = i === parts.length - 1;
      appendBubble(
        part,
        'assistant',
        isLast && mode === 'online',    // 只在最后一段标注 API 来源
        isLast && mode === 'confirm',   // 只在最后一段显示确认样式
      );
    });
  }


  /**
   * 显示打字动画（三点跳动的等待指示器）。
   * 在发送消息后、收到回复前调用。
   */
  function showTyping() {
    const container = document.getElementById('messages');
    if (!container) return;

    const el = document.createElement('div');
    el.className = 'msg assistant';
    el.id = 'typing-indicator';
    el.innerHTML =
      '<div class="msg-label">助手</div>'
      + '<div class="msg-bubble">'
      + '<div class="typing" aria-label="助手正在输入" role="status">'
      + '<span></span><span></span><span></span>'
      + '</div></div>';
    container.appendChild(el);
    scrollToBottom();
  }


  /**
   * 隐藏打字动画。
   * 在收到后端回复时调用（渲染实际内容前）。
   */
  function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }


  /**
   * 更新顶部模式状态徽章。
   *
   * @param {'local'|'online'|'pending'} mode
   */
  function updateModeUI(mode) {
    const dot  = document.getElementById('mode-dot');
    const text = document.getElementById('mode-text');
    if (!dot || !text) return;

    // 先清空所有状态类，再按 mode 添加对应类
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
        // 'local' 或未知值，保持默认（绿点，无额外类）
        text.textContent = '本地';
    }
  }


  /**
   * 在消息列表插入一条系统提示条。
   * 用于"对话已保存"等非对话内容的提示。
   *
   * @param {string} text - 提示文本
   */
  function appendSysNotice(text) {
    const container = document.getElementById('messages');
    if (!container) return;

    const notice = document.createElement('div');
    notice.className   = 'sys-notice';
    // 使用 textContent 而非 innerHTML，防止 XSS
    notice.textContent = text;
    container.appendChild(notice);
    scrollToBottom();
  }


  /**
   * 显示 Toast 通知。
   * 3秒（默认）后自动消失。如果上一个 Toast 还未消失，立即替换。
   *
   * @param {string} message              - 通知文本
   * @param {'default'|'error'|'success'} [type='default'] - 通知类型
   * @param {number} [duration=3000]      - 自动消失时长（毫秒）
   */
  function showToast(message, type = 'default', duration = 3000) {
    let toast = document.getElementById('toast');
    // 如果 DOM 中还没有 toast 元素，动态创建一个
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'toast';
      toast.setAttribute('role', 'alert');       // 无障碍：通知角色
      toast.setAttribute('aria-live', 'polite'); // 屏幕阅读器会读取内容变化
      document.body.appendChild(toast);
    }

    // 清除上一个自动消失计时器
    if (_toastTimer) clearTimeout(_toastTimer);

    // 重置类名，设置新类型
    toast.className = type !== 'default' ? type : '';
    toast.textContent = message;

    // 触发显示（通过 CSS transition）
    // 先移除 show 再立即重加，确保动画能重新触发
    toast.classList.remove('show');
    void toast.offsetWidth; // 强制回流，让浏览器感知到 class 移除
    toast.classList.add('show');

    // 定时隐藏
    _toastTimer = setTimeout(() => {
      toast.classList.remove('show');
    }, duration);
  }


  /**
   * 自动调整 textarea 高度，跟随内容增长。
   * 上限由 CSS max-height 控制（160px）。
   *
   * @param {HTMLTextAreaElement} textarea
   */
  function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }


  // ── 暴露公开接口 ──
  return {
    appendBubble,
    renderAssistantReply,
    showTyping,
    hideTyping,
    updateModeUI,
    appendSysNotice,
    showToast,
    autoResizeTextarea,
    scrollToBottom,
  };

})();
