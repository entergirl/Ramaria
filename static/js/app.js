/**
 * app.js — 应用入口，事件绑定与初始化
 * =============================================================================
 * 职责：
 *   · 页面加载完成后执行初始化（syncStatus、restoreTheme）
 *   · 绑定所有用户交互事件（发送、保存、主题切换、Toggle）
 *   · 协调 AppState / API / UI 三个模块完成业务流程
 *
 * 加载顺序（index.html 中 script 标签的顺序）：
 *   state.js → api.js → ui.js → app.js
 *   app.js 是最后加载的，可以安全引用前三个模块。
 *
 * 主要流程：
 *   sendMessage()       — 发送消息的完整生命周期
 *   onToggleChange()    — Toggle 拨动事件处理
 *   saveSession()       — 手动保存对话
 *   toggleTheme()       — 切换主题
 *   syncStatus()        — 页面加载时同步后端路由状态
 * =============================================================================
 */


/* =============================================================================
   发送消息
   ============================================================================= */

/**
 * 发送用户消息，并处理后端回复的完整流程。
 *
 * 流程：
 *   1. 校验输入（非空、非加载中）
 *   2. 清空输入框，显示用户气泡
 *   3. 进入加载状态，显示打字动画
 *   4. 调用 API.chat()
 *   5. 根据 mode 渲染回复，更新状态徽章
 *   6. 无论成功失败，退出加载状态
 */
async function sendMessage() {
  // 正在等待回复时，忽略重复发送
  if (AppState.isLoading) return;

  const textarea = document.getElementById('user-input');
  const content  = textarea.value.trim();
  if (!content) return;

  // 清空输入框，重置高度
  textarea.value = '';
  textarea.style.height = 'auto';

  // 立即渲染用户消息气泡（不等后端，提升响应感）
  UI.appendBubble(content, 'user', false, false);

  // 进入加载状态
  AppState.setLoading(true);
  UI.showTyping();

  try {
    const data = await API.chat(content);

    // 渲染助手回复（支持 || 多段分割）
    UI.renderAssistantReply(data.reply, data.mode);

    // 更新 session id 标签
    AppState.setSessionId(data.session_id);

    // 根据回复模式更新状态徽章
    if (data.mode === 'confirm') {
      // 等待用户确认是否调用 Claude
      UI.updateModeUI('pending');

    } else if (data.mode === 'online') {
      // 线上模式回复完成：徽章短暂显示"线上"，2秒后自动回到"本地"
      UI.updateModeUI('online');
      AppState.setOnline(true);
      setTimeout(() => {
        UI.updateModeUI('local');
        AppState.setOnline(false);
      }, 2000);

    } else {
      // 本地模式：只在 toggle 未手动开启时重置徽章
      if (!AppState.isOnline) {
        UI.updateModeUI('local');
      }
    }

  } catch (err) {
    // 网络错误或 HTTP 错误：隐藏打字动画，显示错误气泡 + Toast
    UI.hideTyping();
    UI.appendBubble(`（发生错误：${err.message}）`, 'assistant', false, false);
    UI.showToast(err.message, 'error');
    UI.updateModeUI('local');

  } finally {
    // 无论成功还是失败，都退出加载状态，恢复输入
    AppState.setLoading(false);
    textarea.focus();
  }
}


/* =============================================================================
   Toggle 拨动：切换线上 / 本地模式
   ============================================================================= */

/**
 * Toggle checkbox 改变时调用。
 * 通知后端切换路由模式，并根据后端返回的状态更新 UI。
 *
 * @param {boolean} checked - Toggle 当前是否选中（true = 切换到线上）
 */
async function onToggleChange(checked) {
  // 先在 UI 上显示等待状态，给用户即时反馈
  AppState.setOnline(checked);
  UI.updateModeUI(checked ? 'pending' : 'local');

  try {
    const data = await API.toggleRouter(checked);

    // 后端可能返回提示文本（如"下一条消息走线上 API"）
    if (data.message) {
      UI.appendBubble(data.message, 'assistant', false, false);
    }

    // 用后端返回的实际 mode 更新徽章（而非依赖前端推测）
    UI.updateModeUI(data.mode || (checked ? 'pending' : 'local'));

  } catch (err) {
    // 请求失败：回滚 toggle 状态，不让界面停留在错误状态
    AppState.setOnline(!checked);
    UI.updateModeUI('local');
    UI.showToast(`切换失败：${err.message}`, 'error');
  }
}


/* =============================================================================
   手动保存对话
   ============================================================================= */

/**
 * 点击"保存对话"按钮时调用。
 * 关闭当前 session，触发后端 L1 摘要生成。
 */
async function saveSession() {
  if (AppState.isLoading) return;

  try {
    await API.save();

    // 更新 session 标签
    const label = document.getElementById('session-label');
    if (label) label.textContent = '已保存';

    // 在消息列表中插入系统分隔提示
    UI.appendSysNotice('── 对话已保存，新消息将开启新 session ──');

    UI.showToast('对话已保存', 'success');

  } catch (err) {
    UI.showToast(`保存失败：${err.message}`, 'error');
  }
}


/* =============================================================================
   主题切换
   ============================================================================= */

/**
 * 点击主题切换按钮时调用。
 * 在亮色 / 暗色间切换，结果持久化到 localStorage。
 */
function toggleTheme() {
  AppState.toggleTheme();
}


/* =============================================================================
   页面初始化：同步后端路由状态
   ============================================================================= */

/**
 * 页面加载后立即向后端查询路由状态，
 * 确保 Toggle 和徽章与后端实际状态一致。
 * 避免刷新页面后 UI 与后端状态不一致的问题。
 */
async function syncStatus() {
  try {
    const data = await API.getRouterStatus();
    const isOnline = data.mode === 'pending';
    AppState.setOnline(isOnline);
    UI.updateModeUI(data.mode || 'local');
  } catch (_) {
    // 同步失败时保持默认本地状态，不影响使用
    UI.updateModeUI('local');
  }
}


/* =============================================================================
   事件绑定与初始化（DOMContentLoaded）
   ============================================================================= */

document.addEventListener('DOMContentLoaded', () => {

  const textarea = document.getElementById('user-input');
  const sendBtn  = document.getElementById('send-btn');

  // ── 输入框：自动调整高度 ──
  textarea.addEventListener('input', () => {
    UI.autoResizeTextarea(textarea);
  });

  // ── 输入框：Enter 发送，Shift+Enter 换行 ──
  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // ── 发送按钮点击 ──
  sendBtn.addEventListener('click', sendMessage);

  // ── 发送按钮触摸反馈（移动端） ──
  // CSS :active 在部分移动端浏览器延迟较大，通过 touchstart/touchend 补充即时反馈
  sendBtn.addEventListener('touchstart', () => {
    if (!sendBtn.disabled) sendBtn.style.transform = 'scale(0.95)';
  }, { passive: true });
  sendBtn.addEventListener('touchend', () => {
    sendBtn.style.transform = '';
  }, { passive: true });

  // ── 移动端键盘弹出时，确保最新消息仍然可见 ──
  // 当虚拟键盘弹出时 window.innerHeight 会缩小，此时滚动到底部
  const initialHeight = window.innerHeight;
  window.addEventListener('resize', () => {
    if (window.innerHeight < initialHeight) {
      UI.scrollToBottom();
    }
  });

  // ── 主题切换按钮 ──
  const themeBtn = document.getElementById('theme-btn');
  if (themeBtn) {
    themeBtn.addEventListener('click', toggleTheme);
  }

  // ── 保存按钮 ──
  const saveBtn = document.getElementById('save-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', saveSession);
  }

  // ── Toggle：onchange 由 HTML 内联绑定（onchange="onToggleChange(this.checked)"）──
  // 保留 HTML 内联写法，方便直接在 HTML 中看清绑定关系

  // ── 初始化：恢复主题 + 同步后端状态 ──
  AppState.restoreTheme();
  syncStatus();

  // ── 初始化完成后，聚焦到输入框（桌面端体验） ──
  // 移动端不自动聚焦，避免键盘弹出遮挡内容
  if (window.innerWidth > 768) {
    textarea.focus();
  }

});
