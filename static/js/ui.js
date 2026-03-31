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
  
  // 内部状态：记录已加载的历史 session id
  let _loadedHistorySessionId = null;
  
  // T7：上一条消息的时间（用于时间戳判断）
  let _lastMessageTime = null;


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
   * 创建气泡 DOM 元素（供 appendBubble 和 loadHistorySession 共用）
   * @private
   */
  function _createBubbleElement(text, role, isOnlineMsg, isConfirm) {
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
    
    return div;
  }

  /**
   * 创建并插入一条消息气泡。
   * 所有消息（用户/助手/系统）都通过此函数创建，确保结构统一。
   *
   * @param {string} text - 气泡文本内容
   * @param {'user'|'assistant'} role - 发言方
   * @param {boolean} isOnlineMsg - 是否是线上（Claude API）回复
   * @param {boolean} isConfirm - 是否是等待确认消息（蓝色虚线边框）
   * @param {string} created_at - 可选，消息时间戳（ISO 8601）
   */
  function appendBubble(text, role, isOnlineMsg, isConfirm, created_at) {
    const container = document.getElementById('messages');
    if (!container) return;

    // T7: 检查是否需要插入时间戳
    if (created_at && _lastMessageTime && _shouldInsertTimeStamp(_lastMessageTime, created_at)) {
      const timeStampEl = document.createElement('div');
      timeStampEl.className = 'msg-timestamp';
      timeStampEl.textContent = _formatTime(created_at);
      container.appendChild(timeStampEl);
    }

    const bubbleEl = _createBubbleElement(text, role, isOnlineMsg, isConfirm);
    container.appendChild(bubbleEl);

    // 更新上一条消息时间
    if (created_at) {
      _lastMessageTime = created_at;
    }

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
        new Date().toISOString(),       // T7: 使用当前时间作为时间戳
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
      + '</div>'
      + '<div class="typing-text">正在回复...</div>'
      + '</div>';
    container.appendChild(el);
    scrollToBottom();
    
    // T8: 超过10秒时切换提示文字
    el._typingTimeout = setTimeout(() => {
      const textEl = el.querySelector('.typing-text');
      if (textEl) {
        textEl.textContent = '可能需要一点时间…';
        textEl.classList.add('typing-long');
      }
    }, 10000); // 10秒
  }


  /**
   * 隐藏打字动画。
   * 在收到后端回复时调用（渲染实际内容前）。
   */
  function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) {
      // 清除超时定时器
      if (el._typingTimeout) {
        clearTimeout(el._typingTimeout);
      }
      el.remove();
    }
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
   * 增强的错误处理 - 显示更友好的错误气泡
   * @param {Error|string} error - 错误对象或错误消息
   * @param {string} context - 发生错误的上下文（例如 'sendMessage', 'toggleOnline', 'saveSession'）
   * @param {boolean} showToast - 是否同时显示toast通知
   */
  function showEnhancedError(error, context = 'unknown', showToast = true) {
    let errorMessage = error;
    let errorDetails = '';
    
    // 提取错误信息
    if (error instanceof Error) {
      errorMessage = error.message;
      // 从常见网络错误中提取友好消息
      if (errorMessage.includes('NetworkError') || errorMessage.includes('network')) {
        errorMessage = '网络连接失败，请检查网络设置';
        errorDetails = '无法连接到服务器，请确保网络连接正常。';
      } else if (errorMessage.includes('Failed to fetch')) {
        errorMessage = '无法连接到服务器';
        errorDetails = '后台服务可能暂时不可用，请稍后重试。';
      } else if (errorMessage.includes('timeout') || errorMessage.includes('超时')) {
        errorMessage = '请求超时';
        errorDetails = '服务器响应时间过长，请稍后重试。';
      }
    }
    
    // 根据上下文提供更具体的指导
    if (context === 'sendMessage') {
      errorDetails = '消息发送失败，消息已存入本地缓存，重连后将自动重试。';
    } else if (context === 'toggleOnline') {
      errorDetails = '模式切换失败，已自动切换到本地模式，可以正常使用，稍后重试联网功能。';
    } else if (context === 'saveSession') {
      errorDetails = '对话保存失败，对话内容仍在当前会话中，刷新前不会丢失。';
    }
    
    // 显示错误气泡（比普通toast更明显）
    const errorBubble = document.createElement('div');
    errorBubble.className = 'msg assistant error';
    errorBubble.innerHTML = `
      <div class="msg-label">助手</div>
      <div class="msg-bubble">
        <div class="error-message">⚠️ <strong>${errorMessage}</strong></div>
        <div class="error-details">${errorDetails}</div>
        ${context !== 'unknown' ? `<div class="error-context">（错误场景：${context}）</div>` : ''}
      </div>
    `;
    
    const container = document.getElementById('messages');
    if (container) {
      container.appendChild(errorBubble);
      scrollToBottom();
    }
    
    // 可选：显示toast通知
    if (showToast) {
      showToast(errorMessage, 'error', 5000);
    }
    
    console.error(`[${context}] ${errorMessage}`, error);
  }

  /**
   * 屏幕阅读器通知 - 用于向屏幕阅读器用户提供即时通知
   * @param {string} message - 通知消息
   * @param {'polite'|'assertive'} [politeness='polite'] - 通知紧急程度
   * @param {number} [delay=0] - 延迟时间（毫秒）
   */
  function srNotify(message, politeness = 'polite', delay = 0) {
    const execute = () => {
      // 寻找或创建屏幕阅读器通知容器
      let srContainer = document.getElementById('sr-announce');
      if (!srContainer) {
        srContainer = document.createElement('div');
        srContainer.id = 'sr-announce';
        srContainer.className = 'sr-only';
        srContainer.setAttribute('aria-live', 'assertive');
        srContainer.setAttribute('aria-atomic', 'true');
        srContainer.setAttribute('role', 'status');
        document.body.appendChild(srContainer);
      }
      
      // 设置aria-live属性
      srContainer.setAttribute('aria-live', politeness);
      
      // 清除现有内容并设置新消息（强制屏幕阅读器读取）
      srContainer.textContent = '';
      setTimeout(() => {
        srContainer.textContent = message;
      }, 50);
      
      // 5秒后清除消息（如果不再需要）
      setTimeout(() => {
        srContainer.textContent = '';
      }, 5000);
    };
    
    if (delay > 0) {
      setTimeout(execute, delay);
    } else {
      execute();
    }
  }

  /**
   * 显示加载状态指示器
   * @param {string} message - 加载状态消息
   * @param {string} id - 加载器ID（可选）
   * @returns {HTMLElement} - 加载状态元素
   */
  function showLoadingIndicator(message = '正在加载...', id = null) {
    const container = document.getElementById('messages');
    if (!container) return null;
    
    const loaderId = id || `loader-${Date.now()}`;
    const loaderEl = document.createElement('div');
    loaderEl.id = loaderId;
    loaderEl.className = 'loading-indicator';
    loaderEl.innerHTML = `
      <div class="loading-spinner"></div>
      <div class="loading-text">${message}</div>
    `;
    loaderEl.setAttribute('role', 'alert');
    loaderEl.setAttribute('aria-live', 'polite');
    
    container.appendChild(loaderEl);
    scrollToBottom();
    
    return loaderEl;
  }

  /**
   * 隐藏加载状态指示器
   * @param {string} id - 加载器ID
   */
  function hideLoadingIndicator(id) {
    const loader = document.getElementById(id);
    if (loader) {
      loader.classList.add('hide');
      setTimeout(() => loader.remove(), 300);
    }
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


  // =====================================================================
  // T4: 侧边栏功能
  // =====================================================================

  /**
   * 显示侧边栏
   */
  function showSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (sidebar && overlay) {
      sidebar.classList.add('open');
      overlay.classList.add('visible');
      // 防止背景内容滚动
      document.body.style.overflow = 'hidden';
    }
  }

  /**
   * 隐藏侧边栏
   */
  function hideSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (sidebar && overlay) {
      sidebar.classList.remove('open');
      overlay.classList.remove('visible');
      // 恢复背景滚动
      document.body.style.overflow = '';
    }
  }

  /**
   * 渲染 session 列表
   * @param {Array} sessions - session 数据数组
   */
  function renderSessionList(sessions) {
    const sessionListEl = document.getElementById('session-list');
    if (!sessionListEl) return;

    // 清空现有内容
    sessionListEl.innerHTML = '';

    if (!sessions || sessions.length === 0) {
      // 显示空状态
      const emptyEl = document.createElement('div');
      emptyEl.className = 'session-item empty-state';
      emptyEl.innerHTML = `
        <div class="session-title">暂无历史对话</div>
        <div class="session-preview">开始与珊瑚菌对话后，对话记录会显示在这里</div>
      `;
      sessionListEl.appendChild(emptyEl);
      return;
    }

    // 渲染每个 session
    sessions.forEach(session => {
      const sessionEl = document.createElement('div');
      sessionEl.className = 'session-item';
      sessionEl.dataset.sessionId = session.id;
      
      // 无障碍访问属性
      sessionEl.setAttribute('role', 'button');
      sessionEl.setAttribute('tabindex', '0');
      sessionEl.setAttribute('aria-label', `查看Session ${session.id}的对话记录`);
      if (session.last_message_preview) {
        sessionEl.setAttribute('aria-describedby', `session-${session.id}-preview`);
      }
      
      // 添加键盘和点击事件
      sessionEl.addEventListener('click', () => {
        closeAndLoadHistory(session.id);
      });
      
      sessionEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          closeAndLoadHistory(session.id);
        }
      });
      
      // 格式化时间
      const timeStr = session.last_message_at ? 
        _formatTimeForSidebar(session.last_message_at) : 
        _formatTimeForSidebar(session.started_at);
      
      // 截断预览文本
      const preview = session.last_message_preview || '';
      
      // 添加无障碍ID以便describedby引用
      const previewId = `session-${session.id}-preview`;
      
      sessionEl.innerHTML = `
        <div class="session-title">Session #${session.id}</div>
        <div id="${previewId}" class="session-preview" title="${preview}">${preview}</div>
        <div class="session-time">${timeStr}</div>
      `;
      
      sessionListEl.appendChild(sessionEl);
    });
  }

  /**
   * 高亮当前活跃的 session
   * @param {number|null} sessionId
   */
  function highlightActiveSession(sessionId) {
    const items = document.querySelectorAll('.session-item');
    items.forEach(item => {
      if (item.dataset.sessionId && parseInt(item.dataset.sessionId) === sessionId) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });
  }

  /**
   * 格式化时间戳（T7使用，这里先实现）
   * @private
   */
  function _formatTime(isoStr) {
    if (!isoStr) return '';
    
    try {
      const date = new Date(isoStr);
      const now = new Date();
      const hhmm = date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      const isToday = date.toDateString() === now.toDateString();
      
      if (isToday) return hhmm;
      
      const yesterday = new Date(now);
      yesterday.setDate(now.getDate() - 1);
      const isYesterday = date.toDateString() === yesterday.toDateString();
      
      if (isYesterday) return '昨天 ' + hhmm;
      
      const month = date.getMonth() + 1;
      const day = date.getDate();
      return `${month}/${day} ${hhmm}`;
      
    } catch (e) {
      return '';
    }
  }

  /**
   * 检查两条消息时间间隔是否需要插入时间戳
   * @private
   */
  function _shouldInsertTimeStamp(prevTime, currTime) {
    if (!prevTime || !currTime) return false;
    
    try {
      const prevDate = new Date(prevTime);
      const currDate = new Date(currTime);
      const diffMs = currDate - prevDate;
      return diffMs > 5 * 60 * 1000; // 5分钟
    } catch (e) {
      return false;
    }
  }

  /**
   * 加载历史对话
   * @param {number} sessionId
   */
  function loadHistorySession(sessionId) {
    // 如果已经有历史块，先移除
    const existingBlock = document.getElementById('history-block');
    if (existingBlock) {
      existingBlock.remove();
    }
    
    // 如果是同一个 session，则直接返回
    if (_loadedHistorySessionId === sessionId) {
      hideSidebar();
      return;
    }
    
    // 保存当前加载的 session id
    _loadedHistorySessionId = sessionId;
    
    // 获取消息
    API.getSessionMessages(sessionId)
      .then(messages => {
        if (messages.length === 0) {
          hideSidebar();
          return;
        }
        
        const container = document.getElementById('messages');
        if (!container) return;
        
        // 创建历史块容器
        const historyBlock = document.createElement('div');
        historyBlock.id = 'history-block';
        historyBlock.className = 'history-block';
        
        // 添加上方虚线分隔符
        const topDivider = document.createElement('div');
        topDivider.className = 'history-divider top';
        topDivider.innerHTML = `<span>Session #${sessionId}</span>`;
        historyBlock.appendChild(topDivider);
        
        // 添加消息
        let prevMessageTime = null;
        messages.forEach(msg => {
          // 检查是否需要插入时间戳
          if (msg.created_at && prevMessageTime && _shouldInsertTimeStamp(prevMessageTime, msg.created_at)) {
            const timeStampEl = document.createElement('div');
            timeStampEl.className = 'msg-timestamp';
            timeStampEl.textContent = _formatTime(msg.created_at);
            historyBlock.appendChild(timeStampEl);
          }
          
          // 添加消息气泡
          const bubbleEl = _createBubbleElement(msg.content, msg.role, false, false);
          historyBlock.appendChild(bubbleEl);
          
          prevMessageTime = msg.created_at;
        });
        
        // 添加下方虚线分隔符
        const bottomDivider = document.createElement('div');
        bottomDivider.className = 'history-divider bottom';
        bottomDivider.innerHTML = '<span>以上是历史聊天</span>';
        historyBlock.appendChild(bottomDivider);
        
        // 插入到消息列表顶部
        if (container.firstChild) {
          container.insertBefore(historyBlock, container.firstChild);
        } else {
          container.appendChild(historyBlock);
        }
        
        // 滚动到历史块底部（即当前对话的开始）
        const historyHeight = historyBlock.offsetHeight;
        container.scrollTop = historyHeight;
        
        hideSidebar();
      })
      .catch(err => {
        console.error('加载历史消息失败:', err);
        hideSidebar();
      });
  }

  /**
   * 关闭侧边栏并加载历史对话
   * @param {number} sessionId
   */
  function closeAndLoadHistory(sessionId) {
    hideSidebar();
    loadHistorySession(sessionId);
  }

  /**
   * 侧边栏内部使用的格式化时间函数
   * @param {string} isoStr - ISO 8601 时间字符串
   * @returns {string} - 格式化后的时间
   */
  function _formatTimeForSidebar(isoStr) {
    if (!isoStr) return '';
    
    try {
      const date = new Date(isoStr);
      const now = new Date();
      const diffMs = now - date;
      
      // 1小时内：显示"X分钟前"
      if (diffMs < 60 * 60 * 1000) {
        const mins = Math.floor(diffMs / (60 * 1000));
        return mins === 0 ? '刚才' : `${mins}分钟前`;
      }
      
      // 今天：显示时间
      if (date.toDateString() === now.toDateString()) {
        return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      }
      
      // 昨天
      const yesterday = new Date(now);
      yesterday.setDate(now.getDate() - 1);
      if (date.toDateString() === yesterday.toDateString()) {
        return '昨天 ' + date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      }
      
      // 一周内：显示星期
      const weekMs = 7 * 24 * 60 * 60 * 1000;
      if (diffMs < weekMs) {
        const days = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
        return days[date.getDay()];
      }
      
      // 更久：显示日期
      const month = date.getMonth() + 1;
      const day = date.getDate();
      return `${month}/${day}`;
      
    } catch (e) {
      return '';
    }
  }

  /**
   * 渲染记忆强度滑块组件
   */
  function renderMemorySlider() {
    const container = document.getElementById('messages');
    if (!container) return;
    
    // 如果已经存在滑块，先移除
    const existingSlider = document.getElementById('memory-slider-container');
    if (existingSlider) {
      existingSlider.remove();
    }
    
    // 创建滑块容器
    const sliderContainer = document.createElement('div');
    sliderContainer.id = 'memory-slider-container';
    sliderContainer.className = 'memory-slider-container';
    
    // 滑块参数
    const minValue = 0;
    const maxValue = 10;
    const defaultValue = 5;
    
    // 预定义模式
    const presets = [
      { value: 0, label: '轻量聊天', hint: '只使用最近对话，快速响应' },
      { value: 3, label: '日常陪伴', hint: '平衡速度与记忆深度，适中陪伴' },
      { value: 5, label: '深度思考', hint: '深度检索记忆，提供详细反馈' },
      { value: 8, label: '研究模式', hint: '全面检索所有层级的记忆' },
      { value: 10, label: '全记忆', hint: '调取所有可用记忆，最详细的分析' }
    ];
    
    // 滑块HTML结构
    sliderContainer.innerHTML = `
      <div class="memory-slider-label">记忆深度</div>
      <div class="memory-slider-track">
        <div class="memory-slider-fill" id="memory-slider-fill"></div>
        <input 
          type="range" 
          class="memory-slider" 
          id="memory-slider"
          min="${minValue}" 
          max="${maxValue}" 
          value="${defaultValue}"
          aria-label="记忆深度控制器"
          aria-describedby="memory-slider-hint"
        >
        <div class="memory-slider-ticks" id="memory-slider-ticks"></div>
      </div>
      <div class="memory-slider-value" id="memory-slider-value">${defaultValue}</div>
    `;
    
    // 添加预设模式按钮
    const presetGroup = document.createElement('div');
    presetGroup.className = 'memory-preset-group';
    
    presets.forEach(preset => {
      const btn = document.createElement('button');
      btn.className = `memory-preset-btn ${preset.value === defaultValue ? 'active' : ''}`;
      btn.type = 'button';
      btn.textContent = preset.label;
      btn.dataset.value = preset.value;
      presetGroup.appendChild(btn);
    });
    
    // 添加提示文本
    const hintEl = document.createElement('div');
    hintEl.id = 'memory-slider-hint';
    hintEl.className = 'memory-slider-hint';
    hintEl.textContent = presets.find(p => p.value === defaultValue)?.hint || '';
    
    // 插入到容器中
    container.appendChild(sliderContainer);
    sliderContainer.appendChild(presetGroup);
    sliderContainer.appendChild(hintEl);
    
    // 初始化滑块
    _initializeMemorySlider();
    
    // 将滑块定位到消息列表顶部（在当前会话之后）
    setTimeout(() => {
      container.scrollTop = 0;
      const sliderEl = document.getElementById('memory-slider-container');
      if (sliderEl) {
        sliderEl.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
    
    return sliderContainer;
  }
  
  /**
   * 初始化记忆滑块的事件监听
   * @private
   */
  function _initializeMemorySlider() {
    const slider = document.getElementById('memory-slider');
    const fill = document.getElementById('memory-slider-fill');
    const valueEl = document.getElementById('memory-slider-value');
    const hintEl = document.getElementById('memory-slider-hint');
    const ticks = document.getElementById('memory-slider-ticks');
    
    if (!slider || !fill || !valueEl || !hintEl) return;
    
    const minValue = parseInt(slider.min);
    const maxValue = parseInt(slider.max);
    
    // 预定义模式和提示
    const presets = [
      { value: 0, label: '轻量聊天', hint: '只使用最近对话，快速响应' },
      { value: 3, label: '日常陪伴', hint: '平衡速度与记忆深度，适中陪伴' },
      { value: 5, label: '深度思考', hint: '深度检索记忆，提供详细反馈' },
      { value: 8, label: '研究模式', hint: '全面检索所有层级的记忆' },
      { value: 10, label: '全记忆', hint: '调取所有可用记忆，最详细的分析' }
    ];
    
    // 创建刻度标记
    if (ticks) {
      for (let i = minValue; i <= maxValue; i++) {
        const tick = document.createElement('div');
        tick.className = 'memory-slider-tick';
        if ([0, 3, 5, 8, 10].includes(i)) {
          tick.classList.add('active');
        }
        ticks.appendChild(tick);
      }
    }
    
    // 更新滑块UI
    function updateSliderUI(value) {
      const percentage = ((value - minValue) / (maxValue - minValue)) * 100;
      fill.style.width = `${percentage}%`;
      valueEl.textContent = value;
      
      // 找到最接近的预设并更新提示
      const closestPreset = presets.reduce((prev, curr) => {
        return Math.abs(curr.value - value) < Math.abs(prev.value - value) ? curr : prev;
      });
      
      if (closestPreset) {
        hintEl.textContent = closestPreset.hint;
      }
      
      // 更新预设按钮状态
      document.querySelectorAll('.memory-preset-btn').forEach(btn => {
        const btnValue = parseInt(btn.dataset.value);
        if (Math.abs(btnValue - value) <= 1) { // 容差1
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });
    }
    
    // 初始更新
    const initialValue = parseInt(slider.value);
    updateSliderUI(initialValue);
    
    // 滑块输入事件
    slider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      updateSliderUI(value);
    });
    
    // 滑块改变事件（松开鼠标或完成拖动）
    slider.addEventListener('change', (e) => {
      const value = parseInt(e.target.value);
      updateSliderUI(value);
      // 发送记忆深度变更通知（未来可连接到API）
      _onMemoryDepthChange(value);
    });
    
    // 预设按钮点击事件
    document.querySelectorAll('.memory-preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const value = parseInt(btn.dataset.value);
        slider.value = value;
        updateSliderUI(value);
        _onMemoryDepthChange(value);
      });
    });
  }
  
  /**
   * 记忆深度变更处理函数
   * @private
   */
  function _onMemoryDepthChange(value) {
    // 这里可以添加后端API调用，在后续阶段实现
    const depthLabels = {
      0: '浅层记忆（仅最近对话）',
      3: '中等深度（近期摘要）',
      5: '深度记忆（完整摘要）',
      8: '深层检索（全部层级）',
      10: '全记忆（最详细分析）'
    };
    
    const label = depthLabels[value] || '自定义深度';
    
    // 显示临时提示
    showToast(`记忆深度设为：${label}`, 'default', 1500);
    
    console.log(`记忆深度设置为: ${value} (${label})`);
    // TODO: 后续连接后端API
    // API.setMemoryDepth(value);
  }
  
  /**
   * 检查是否显示记忆滑块
   */
  function checkAndShowMemorySlider() {
    // 只在特定条件下显示（例如当前不是历史会话视图）
    if (!_loadedHistorySessionId) {
      // 延迟显示，让页面先加载完成
      setTimeout(() => {
        renderMemorySlider();
      }, 500);
    }
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
    // 增强的错误处理和加载状态
    showEnhancedError,
    showLoadingIndicator,
    hideLoadingIndicator,
    srNotify,
    // T4 侧边栏函数
    showSidebar,
    hideSidebar,
    renderSessionList,
    highlightActiveSession,
    loadHistorySession,
    closeAndLoadHistory,
    // 记忆滑块函数
    renderMemorySlider,
    checkAndShowMemorySlider,
  };

})();
