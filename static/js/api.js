/**
 * api.js — 后端 API 调用封装
 * =============================================================================
 * 将所有 fetch 请求集中在此模块，统一处理：
 *   · 请求头设置
 *   · HTTP 错误响应（4xx / 5xx）的解析与抛出
 *   · 网络层错误（断线、超时）的统一包装
 *
 * 对外暴露的函数：
 *   API.chat(content)       — POST /chat，发送消息
 *   API.save()              — POST /save，手动保存 session
 *   API.getRouterStatus()   — GET  /router/status，查询路由状态
 *   API.toggleRouter(bool)  — POST /router/toggle，切换路由模式
 *
 * 错误处理约定：
 *   所有函数在失败时抛出 Error，调用方用 try/catch 处理，
 *   通过 UI.showToast() 向用户展示错误信息。
 * =============================================================================
 */

const API = (() => {

  /**
   * 内部通用 fetch 封装。
   * 统一处理 JSON 序列化、HTTP 错误状态码解析。
   *
   * @param {string} url        - 请求路径（相对路径）
   * @param {object} [options]  - fetch 选项，可覆盖默认值
   * @returns {Promise<any>}    - 解析后的 JSON 响应体
   * @throws {Error}            - HTTP 错误或网络错误
   */
  async function _request(url, options = {}) {
    const defaultOptions = {
      headers: { 'Content-Type': 'application/json' },
    };
    // 合并选项，调用方传入的选项优先级更高
    const mergedOptions = { ...defaultOptions, ...options };
    if (options.headers) {
      mergedOptions.headers = { ...defaultOptions.headers, ...options.headers };
    }

    let response;
    try {
      response = await fetch(url, mergedOptions);
    } catch (networkErr) {
      // fetch 本身抛出的错误（断网、DNS 失败、超时等）
      throw new Error(`网络连接失败，请检查网络后重试（${networkErr.message}）`);
    }

    if (!response.ok) {
      // HTTP 层面的错误（4xx / 5xx）
      let detail = `HTTP ${response.status}`;
      try {
        const errBody = await response.json();
        // FastAPI 的错误格式是 { "detail": "..." }
        if (errBody.detail) detail = errBody.detail;
      } catch (_) {
        // 响应体不是合法 JSON，用默认的 HTTP 状态码描述
      }
      throw new Error(detail);
    }

    return response.json();
  }


  // ── 公开 API 方法 ──
  return {

    /**
     * 发送一条用户消息，获取助手回复。
     *
     * @param {string} content - 用户消息文本
     * @returns {Promise<{reply: string, session_id: number, mode: string}>}
     * @throws {Error}
     */
    async chat(content) {
      return _request('/chat', {
        method: 'POST',
        body: JSON.stringify({ content }),
      });
    },

    /**
     * 手动触发 session 保存，关闭当前 session 并触发 L1 摘要生成。
     *
     * @returns {Promise<{status: string}>}
     * @throws {Error}
     */
    async save() {
      return _request('/save', { method: 'POST' });
    },

    /**
     * 查询当前路由状态（本地/线上/等待确认）。
     * 页面加载时调用一次，同步前端 UI 与后端状态。
     *
     * @returns {Promise<{api_enabled: boolean, waiting_confirm: boolean, mode: string}>}
     * @throws {Error}
     */
    async getRouterStatus() {
      return _request('/router/status');
    },

    /**
     * 切换路由模式。
     *
     * @param {boolean} online - true=切换到线上模式，false=切回本地模式
     * @returns {Promise<{ok: boolean, mode: string, message: string|null}>}
     * @throws {Error}
     */
    async toggleRouter(online) {
      return _request('/router/toggle', {
        method: 'POST',
        body: JSON.stringify({ online }),
      });
    },

    // =====================================================================
    // T1/T2: 新增 API 方法
    // =====================================================================

    /**
     * 获取所有 session 的摘要列表（T1）。
     *
     * @returns {Promise<Array<{
     *   id: number,
     *   started_at: string,
     *   ended_at: string|null,
     *   message_count: number,
     *   last_message_at: string|null,
     *   last_message_preview: string|null
     * }>>}
     * @throws {Error}
     */
    async getSessions() {
      return _request('/api/sessions');
    },

    /**
     * 获取指定 session 的消息列表（T2）。
     *
     * @param {number} sessionId - session id
     * @returns {Promise<Array<{
     *   role: 'user'|'assistant',
     *   content: string,
     *   created_at: string
     * }>>}
     * @throws {Error}
     */
    async getSessionMessages(sessionId) {
      return _request(`/api/sessions/${sessionId}/messages`);
    },

  };
})();
