/**
 * state.js — 全局状态管理模块
 * =============================================================================
 * 将原来散落在全局的变量（isLoading、isOnline）集中管理。
 * 所有状态的读写都通过 AppState 的方法进行，
 * 修改状态时同步更新 UI，避免状态与界面不一致。
 *
 * 暴露给其他模块的接口：
 *   AppState.isLoading          — 只读，是否正在等待后端回复
 *   AppState.isOnline           — 只读，是否处于线上模式
 *   AppState.currentSessionId   — 只读，当前 session id（number | null）
 *   AppState.currentTheme       — 只读，当前主题 'light' | 'dark'
 *
 *   AppState.setLoading(bool)   — 设置加载状态，同步更新发送按钮
 *   AppState.setOnline(bool)    — 设置线上模式，同步更新 toggle
 *   AppState.setSessionId(id)   — 更新 session id，同步更新顶部标签
 *   AppState.setTheme(theme)    — 切换主题，持久化到 localStorage
 *   AppState.toggleTheme()      — 在 light / dark 间切换
 * =============================================================================
 */

const AppState = (() => {
  // ── 内部私有状态（外部不可直接修改） ──
  let _isLoading        = false;
  let _isOnline         = false;
  let _currentSessionId = null;
  let _currentTheme     = 'light';

  // ── 公开只读属性与方法 ──
  return {

    /* 只读 getter */
    get isLoading()        { return _isLoading; },
    get isOnline()         { return _isOnline; },
    get currentSessionId() { return _currentSessionId; },
    get currentTheme()     { return _currentTheme; },

    /**
     * 设置加载状态。
     * 同步禁用/启用发送按钮，防止重复发送。
     * @param {boolean} loading
     */
    setLoading(loading) {
      _isLoading = loading;
      const btn = document.getElementById('send-btn');
      if (btn) btn.disabled = loading;
    },

    /**
     * 设置线上模式状态。
     * 同步更新 header 中的 toggle checkbox 位置。
     * @param {boolean} online
     */
    setOnline(online) {
      _isOnline = online;
      const toggle = document.getElementById('mode-toggle');
      if (toggle) toggle.checked = online;
    },

    /**
     * 更新当前 session id。
     * （T3：session-label 元素已移除，该函数现在仅更新内部状态）
     * @param {number|null} id
     */
    setSessionId(id) {
      _currentSessionId = id;
      // T3: session-label 元素已移到侧边栏，此处仅更新内部状态
      // 显示逻辑将在侧边栏中实现（T4）
    },

    /**
     * 切换并持久化主题。
     * @param {'light'|'dark'} theme
     */
    setTheme(theme) {
      _currentTheme = theme;
      // 写入 HTML 根元素的 data-theme 属性，触发 CSS 变量切换
      document.documentElement.setAttribute('data-theme', theme === 'dark' ? 'dark' : '');
      // 更新主题按钮的 emoji 提示
      const btn = document.getElementById('theme-btn');
      if (btn) {
        btn.textContent = theme === 'dark' ? '☀️' : '🌙';
        btn.setAttribute('title', theme === 'dark' ? '切换到亮色模式' : '切换到暗色模式');
        btn.setAttribute('aria-label', theme === 'dark' ? '切换到亮色模式' : '切换到暗色模式');
      }
      // 持久化到 localStorage，下次打开时恢复
      try {
        localStorage.setItem('ramaria-theme', theme);
      } catch (_) {
        // localStorage 不可用时静默忽略（无痕模式等场景）
      }
    },

    /** 在 light / dark 间切换 */
    toggleTheme() {
      this.setTheme(_currentTheme === 'dark' ? 'light' : 'dark');
    },

    /**
     * 从 localStorage 恢复上次的主题设置。
     * 在 app.js 初始化时调用一次。
     */
    restoreTheme() {
      try {
        const saved = localStorage.getItem('ramaria-theme');
        if (saved === 'dark' || saved === 'light') {
          this.setTheme(saved);
        }
      } catch (_) {
        // 读取失败时保持默认亮色
      }
    },

  };
})();
