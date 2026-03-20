"""
router.py — 任务路由层
=====================================================================

负责判断当前消息是否需要切换到 Claude API 处理，
并管理线上/本地模式的全局状态。

核心设计原则：
  · 本地 Qwen 是主体，负责日常对话和记忆维护
  · Claude 是一次性专家工具，处理完当前消息立即回到本地模式
  · Claude 不注入任何记忆上下文，只拿到用户的当前消息
  · 自动检测到可能需要 Claude 时，先向用户发确认询问，不直接调用

─────────────────────────────────────────────────────────────
状态机
─────────────────────────────────────────────────────────────

  模式状态（mode）：
    "local"   — 默认状态，消息走本地 Qwen
    "online"  — 当前消息走 Claude API（处理完后自动回到 local）

  API 开关（api_enabled）：
    True  — 允许自动检测和手动切换到 Claude（默认）
    False — 完全禁用 Claude，所有消息强制走本地

  确认等待状态（waiting_confirm）：
    True  — 上一条消息触发了自动检测，正在等待用户确认是否调用 Claude
    False — 正常对话流程

─────────────────────────────────────────────────────────────
自动触发条件
─────────────────────────────────────────────────────────────

  满足以下任一条件时，系统会向用户发确认询问：
    1. 消息中包含代码块（``` 包裹的内容）
    2. 消息中包含调试/分析类关键词（见 TRIGGER_KEYWORDS）

  用户回复"好"/"是"等确认词 → 调用 Claude 处理上一条消息
  用户回复"不用"/"算了"等拒绝词 → 本地 Qwen 处理上一条消息

─────────────────────────────────────────────────────────────
与其他模块的关系
─────────────────────────────────────────────────────────────

  · 被调用：main.py 的 /chat 路由
  · 调用：Claude API（通过 anthropic SDK 或 requests）
  · 读配置：config.py

使用方法：
    from router import Router
    router = Router()

    # 每条消息到来时
    result = router.process(user_message, session_id)
    # result["action"] 决定后续行为：
    #   "ask_confirm"  → 返回确认询问文本给用户，暂不处理消息
    #   "local"        → 走本地 Qwen 正常处理
    #   "online"       → 调用 Claude API，返回结果
    #   "confirm_yes"  → 用户确认调用，取出缓存消息走 Claude
    #   "confirm_no"   → 用户拒绝调用，取出缓存消息走本地
"""

import re
import requests

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL_NAME,
    CLAUDE_MAX_TOKENS,
    CLAUDE_TEMPERATURE,
)


# =============================================================================
# 路由配置常量
# =============================================================================

# 自动触发关键词列表（检测到这些词时发确认询问）
# 全部小写，匹配时会先把用户消息转小写再比对
TRIGGER_KEYWORDS = [
    # 调试类
    "报错", "错误", "bug", "debug", "报了个错", "跑不起来", "不对劲",
    "traceback", "exception", "error",
    # 分析类
    "帮我分析", "帮我看看", "分析一下", "为什么", "怎么回事",
    "优化", "重构", "怎么优化", "性能",
    # 架构类
    "怎么设计", "架构", "方案", "怎么实现",
]

# 用户确认调用 Claude 的关键词
CONFIRM_YES_KEYWORDS = {"好", "好的", "行", "是", "是的", "要", "可以", "嗯", "ok", "yes"}

# 用户拒绝调用 Claude 的关键词
CONFIRM_NO_KEYWORDS = {"不用", "不", "算了", "不要", "不行", "no", "本地", "本地处理"}

# 自动检测触发时，发给用户的确认询问文本
# 刻意写得自然口语，融入对话而非系统提示
CONFIRM_ASK_TEXT = "要不我用线上 API 为你解决？"

# Claude API 端点
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"


# =============================================================================
# Router 类
# =============================================================================

class Router:
    """
    任务路由管理器。

    内部状态：
        _api_enabled      — 线上 API 总开关（用户从 UI 控制）
        _waiting_confirm  — 是否正在等待用户确认是否调用 Claude
        _pending_message  — 触发确认时缓存的原始用户消息
                            确认后取出来交给 Claude 或本地模型处理

    典型对话流程：
        用户："帮我分析一下这段代码为什么报错"
        → 检测到"报错"关键词
        → 返回 action="ask_confirm"，文本="要不我用线上 API 为你解决？"
        → _waiting_confirm = True，_pending_message = 用户消息

        用户："好"
        → 检测到确认词
        → 取出 _pending_message，调用 Claude
        → 返回 action="confirm_yes"，reply=Claude 的回复
        → _waiting_confirm = False（自动回到本地模式）

        用户："不用"
        → 检测到拒绝词
        → 取出 _pending_message，交给本地 Qwen 处理
        → 返回 action="confirm_no"
        → _waiting_confirm = False
    """

    def __init__(self):
        # 线上 API 总开关，True = 允许使用 Claude
        self._api_enabled = True

        # 等待确认状态
        self._waiting_confirm = False

        # 缓存触发确认时的原始用户消息
        # 确认/拒绝后取出来交给对应模型处理
        self._pending_message = None

    # -------------------------------------------------------------------------
    # 公开接口：状态查询
    # -------------------------------------------------------------------------

    @property
    def api_enabled(self):
        """线上 API 是否开启"""
        return self._api_enabled

    @property
    def waiting_confirm(self):
        """是否正在等待用户确认"""
        return self._waiting_confirm

    @property
    def pending_message(self):
        """缓存的待处理消息（确认/拒绝后由 main.py 取用）"""
        return self._pending_message

    def get_status(self):
        """
        返回当前路由状态字典，供前端 UI 轮询显示。

        返回格式：
            {
                "api_enabled":     bool,   # 线上 API 开关状态
                "waiting_confirm": bool,   # 是否等待确认
                "mode":            str,    # "local" 或 "online"（等待确认时显示 "pending"）
            }
        """
        if self._waiting_confirm:
            mode = "pending"
        elif not self._api_enabled:
            mode = "local"
        else:
            mode = "local"  # 默认本地，online 状态只在实际调用 Claude 时短暂存在

        return {
            "api_enabled":     self._api_enabled,
            "waiting_confirm": self._waiting_confirm,
            "mode":            mode,
        }

    # -------------------------------------------------------------------------
    # 公开接口：状态控制
    # -------------------------------------------------------------------------

    def set_api_enabled(self, enabled: bool):
        """
        设置线上 API 开关（由前端 UI 的 toggle 调用）。

        关闭时同时清除等待确认状态，避免残留。

        参数：
            enabled — True = 开启，False = 关闭
        """
        self._api_enabled = enabled
        if not enabled:
            # 关闭 API 时清除所有等待状态
            self._waiting_confirm = False
            self._pending_message = None
        print(f"[router] 线上 API {'开启' if enabled else '关闭'}")

    def force_online(self):
        """
        手动强制切换到线上模式（由前端"切换"按钮或 /online 指令触发）。
        不发确认询问，直接标记下一条消息走 Claude。

        返回：
            str — 提示文本，供 main.py 展示给用户
        """
        if not self._api_enabled:
            return "线上 API 当前已关闭，请先在面板中开启。"
        if not ANTHROPIC_API_KEY:
            return "未检测到 ANTHROPIC_API_KEY，请先配置环境变量。"

        # 标记等待确认但跳过询问（pending_message 留空，下一条消息直接用 Claude）
        # 这里用一个特殊标记值区分"手动触发"和"自动触发后等待确认"
        self._waiting_confirm  = True
        self._pending_message  = "__FORCE_ONLINE__"
        print("[router] 手动切换到线上模式，下一条消息将使用 Claude API")
        return "好，下一条消息我会用线上 API 处理。"

    # -------------------------------------------------------------------------
    # 公开接口：消息路由判断
    # -------------------------------------------------------------------------

    def route(self, message: str):
        """
        判断当前消息的处理路径。
        这是路由层的核心判断函数，由 main.py 在每条消息到来时调用。

        判断优先级（从高到低）：
          1. 正在等待确认 → 检测用户是否回复了确认/拒绝词
          2. API 已关闭  → 强制本地
          3. 手动触发标记（__FORCE_ONLINE__）→ 直接走 Claude
          4. 自动检测     → 检测代码块和关键词，触发时发确认询问
          5. 默认         → 本地 Qwen

        参数：
            message — 用户发送的消息文本（已去除首尾空白）

        返回：
            dict，包含以下字段：
              action  — 字符串，指示 main.py 下一步操作：
                          "ask_confirm"  等待确认，返回询问文本
                          "local"        本地 Qwen 处理
                          "online"       Claude API 处理
                          "confirm_yes"  用户确认，处理缓存消息
                          "confirm_no"   用户拒绝，本地处理缓存消息
              text    — 询问/提示文本（仅 ask_confirm 时有值）
              message — 实际需要处理的消息（confirm_yes/no 时是缓存消息）
        """
        # ------------------------------------------------------------------
        # 优先级 1：正在等待用户确认
        # ------------------------------------------------------------------
        if self._waiting_confirm:
            return self._handle_confirm_reply(message)

        # ------------------------------------------------------------------
        # 优先级 2：API 开关已关闭，强制本地
        # ------------------------------------------------------------------
        if not self._api_enabled:
            return {"action": "local", "message": message}

        # ------------------------------------------------------------------
        # 优先级 3：API Key 未配置，无法使用线上模式
        # ------------------------------------------------------------------
        if not ANTHROPIC_API_KEY:
            return {"action": "local", "message": message}

        # ------------------------------------------------------------------
        # 优先级 4：自动检测——代码块或触发关键词
        # ------------------------------------------------------------------
        if self._should_trigger(message):
            # 缓存原始消息，等用户确认后取出处理
            self._pending_message = message
            self._waiting_confirm = True
            print(f"[router] 自动检测触发，等待用户确认")
            return {
                "action": "ask_confirm",
                "text":   CONFIRM_ASK_TEXT,
                "message": message,
            }

        # ------------------------------------------------------------------
        # 优先级 5：默认走本地
        # ------------------------------------------------------------------
        return {"action": "local", "message": message}

    # -------------------------------------------------------------------------
    # 公开接口：调用 Claude API
    # -------------------------------------------------------------------------

    def call_claude(self, message: str):
        """
        调用 Claude API 处理单条消息，返回回复文本。

        Claude 作为无状态工具：
          · 不注入任何记忆上下文（L3/L2/L1/RAG 全部不带）
          · 只带一个简短的角色说明 system prompt
          · 处理完后路由自动回到本地模式

        参数：
            message — 需要 Claude 处理的用户消息

        返回：
            str — Claude 的回复文本
            str — 调用失败时返回括号包裹的错误信息
        """
        print(f"[router] 调用 Claude API，消息长度={len(message)}")

        # Claude 的 system prompt：简洁的角色定位，不带任何用户记忆
        system_prompt = (
            "你是一个专业、精准的 AI 助手。"
            "请直接回答用户的问题，给出清晰、实用的解答。"
            "如果是代码问题，请给出可运行的代码和必要的解释。"
        )

        headers = {
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }

        payload = {
            "model":      CLAUDE_MODEL_NAME,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "system":     system_prompt,
            "messages": [
                {"role": "user", "content": message}
            ],
        }

        try:
            response = requests.post(
                CLAUDE_API_URL,
                headers = headers,
                json    = payload,
                timeout = 60,
            )
            response.raise_for_status()
            data  = response.json()
            reply = data["content"][0]["text"].strip()
            print(f"[router] Claude 回复成功，长度={len(reply)}")
            return reply

        except requests.exceptions.ConnectionError:
            return "（线上 API 连接失败，请检查网络）"
        except requests.exceptions.Timeout:
            return "（线上 API 响应超时，请稍后重试）"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "（API Key 无效，请检查 ANTHROPIC_API_KEY 配置）"
            return f"（线上 API 请求失败：{e}）"
        except (KeyError, IndexError) as e:
            return f"（解析 Claude 响应失败：{e}）"
        finally:
            # 无论成功还是失败，用完立即回到本地模式
            self._reset()

    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------

    def _should_trigger(self, message: str):
        """
        检测消息是否满足自动触发条件。

        两个条件满足其一即触发：
          1. 包含代码块（``` 包裹）
          2. 包含触发关键词（不区分大小写）

        参数：
            message — 用户消息文本

        返回：
            bool — True 表示应该触发确认询问
        """
        # 条件一：包含代码块
        if "```" in message:
            print("[router] 检测到代码块，触发确认")
            return True

        # 条件二：包含触发关键词（转小写匹配）
        message_lower = message.lower()
        for kw in TRIGGER_KEYWORDS:
            if kw in message_lower:
                print(f"[router] 检测到关键词 [{kw}]，触发确认")
                return True

        return False

    def _handle_confirm_reply(self, message: str):
        """
        处理用户对确认询问的回复。

        逻辑：
          · 检测到确认词 → 取出缓存消息，标记走 Claude
          · 检测到拒绝词 → 取出缓存消息，标记走本地
          · 都不是（用户说了其他话）→ 视为拒绝，本地处理，清除等待状态

        参数：
            message — 用户的回复消息

        返回：
            dict — route() 标准返回格式
        """
        message_lower = message.lower().strip()

        # 取出缓存消息（无论确认还是拒绝都需要）
        cached = self._pending_message

        # 特殊情况：手动强制触发（force_online）时缓存的是占位符
        # 此时用户下一条真实消息才是要处理的内容
        if cached == "__FORCE_ONLINE__":
            self._reset()
            return {"action": "online", "message": message}

        # 正常确认/拒绝流程
        if message_lower in CONFIRM_YES_KEYWORDS:
            print("[router] 用户确认，走 Claude API")
            self._reset()
            return {"action": "confirm_yes", "message": cached}

        if message_lower in CONFIRM_NO_KEYWORDS:
            print("[router] 用户拒绝，走本地 Qwen")
            self._reset()
            return {"action": "confirm_no", "message": cached}

        # 用户说了其他话（既不是确认也不是拒绝）
        # 视为取消，清除等待状态，把这条新消息按正常流程处理
        print(f"[router] 未识别的确认回复，清除等待状态，按正常消息处理")
        self._reset()
        return {"action": "local", "message": message}

    def _reset(self):
        """
        重置等待确认状态，回到本地模式。
        每次 Claude 调用完成（无论成功失败）或用户拒绝后调用。
        """
        self._waiting_confirm = False
        self._pending_message = None


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    print("=== router.py 验证测试 ===\n")

    router = Router()

    # ------------------------------------------------------------------
    # 测试一：正常消息，走本地
    # ------------------------------------------------------------------
    print("--- 测试一：普通消息 → 本地 ---")
    result = router.route("你好，今天天气怎么样")
    print(f"action={result['action']}（应为 local）\n")

    # ------------------------------------------------------------------
    # 测试二：包含关键词，触发确认
    # ------------------------------------------------------------------
    print("--- 测试二：触发关键词 → 询问确认 ---")
    result = router.route("帮我分析一下这段代码为什么报错")
    print(f"action={result['action']}（应为 ask_confirm）")
    print(f"text={result.get('text')}（应为确认询问文本）\n")

    # ------------------------------------------------------------------
    # 测试三：用户确认
    # ------------------------------------------------------------------
    print("--- 测试三：用户确认 → confirm_yes ---")
    result2 = router.route("好")
    print(f"action={result2['action']}（应为 confirm_yes）")
    print(f"message={result2.get('message')}（应为触发时的原始消息）\n")

    # ------------------------------------------------------------------
    # 测试四：触发后用户拒绝
    # ------------------------------------------------------------------
    print("--- 测试四：触发关键词 → 用户拒绝 → confirm_no ---")
    router.route("帮我优化这段代码")   # 触发
    result3 = router.route("不用")
    print(f"action={result3['action']}（应为 confirm_no）\n")

    # ------------------------------------------------------------------
    # 测试五：代码块触发
    # ------------------------------------------------------------------
    print("--- 测试五：代码块 → 询问确认 ---")
    result4 = router.route("这段代码有问题：\n```python\nprint('hello')\n```")
    print(f"action={result4['action']}（应为 ask_confirm）\n")
    router.route("不用")   # 清除状态

    # ------------------------------------------------------------------
    # 测试六：API 开关关闭
    # ------------------------------------------------------------------
    print("--- 测试六：API 关闭 → 强制本地 ---")
    router.set_api_enabled(False)
    result5 = router.route("帮我分析报错")
    print(f"action={result5['action']}（应为 local，API 已关闭）\n")
    router.set_api_enabled(True)

    # ------------------------------------------------------------------
    # 测试七：force_online
    # ------------------------------------------------------------------
    print("--- 测试七：force_online → 下一条直接走 Claude ---")
    tip = router.force_online()
    print(f"提示：{tip}")
    result6 = router.route("帮我写一个快速排序")
    print(f"action={result6['action']}（应为 online）\n")

    print("验证完成（Claude API 实际调用未测试，需要有效的 API Key）")
