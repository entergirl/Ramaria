"""
router.py — 任务路由层
=====================================================================

变更记录：
  v2 — force_online() 新增时间戳记录，_handle_confirm_reply() 入口处检测
       是否超时（默认30分钟），超时自动 reset 回本地模式。
       修复审查报告问题C：原来 force_online 后若用户关闭浏览器，
       _waiting_confirm 状态会一直挂着，下次重开时第一条消息会意外直达 Claude。

       新增内部属性：
         _force_online_time — 记录 force_online() 的触发时刻（datetime 对象）
         FORCE_ONLINE_TIMEOUT_MINUTES — 超时阈值（默认30分钟，可在此文件顶部调整）

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

─────────────────────────────────────────────────────────────
与其他模块的关系
─────────────────────────────────────────────────────────────

  · 被调用：main.py 的 /chat 路由
  · 调用：Claude API（通过 requests）
  · 读配置：config.py

使用方法：
    from router import Router
    router = Router()
    result = router.route(user_message)
"""

import re
import requests
from datetime import datetime, timezone, timedelta

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL_NAME,
    CLAUDE_MAX_TOKENS,
    CLAUDE_TEMPERATURE,
)
from logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# 路由配置常量
# =============================================================================

# 自动触发关键词列表（检测到这些词时发确认询问）
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
CONFIRM_ASK_TEXT = "要不我用线上 API 为你解决？"

# ─────────────────────────────────────────────────────────────
# [新增] force_online 超时阈值（分钟）
#
# 问题背景（审查报告问题C）：
#   force_online() 触发后，_waiting_confirm=True 和
#   _pending_message="__FORCE_ONLINE__" 状态会一直保持，
#   直到用户发下一条消息。如果用户切换线上模式后关闭浏览器，
#   下次重开时状态仍挂着，第一条消息会意外直达 Claude。
#
# 修复方案：
#   force_online() 记录触发时刻（_force_online_time），
#   _handle_confirm_reply() 每次处理消息前检查是否超时，
#   超时则自动 reset 回本地模式，再正常处理当前消息。
#
# 调整此阈值：如果希望线上模式等待窗口更长或更短，修改这个数字即可。
# ─────────────────────────────────────────────────────────────
FORCE_ONLINE_TIMEOUT_MINUTES = 30

# Claude API 端点
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"


# =============================================================================
# Router 类
# =============================================================================

class Router:
    """
    任务路由管理器。

    内部状态：
        _api_enabled       — 线上 API 总开关
        _waiting_confirm   — 是否正在等待用户确认是否调用 Claude
        _pending_message   — 触发确认时缓存的原始用户消息
        _force_online_time — [新增] force_online() 的触发时刻（datetime 或 None）
                             用于超时检测：超过 FORCE_ONLINE_TIMEOUT_MINUTES 分钟
                             无新消息时自动 reset 回本地模式
    """

    def __init__(self):
        self._api_enabled      = True
        self._waiting_confirm  = False
        self._pending_message  = None
        self._force_online_time = None   # [新增] 记录 force_online 触发时刻

    # -------------------------------------------------------------------------
    # 公开接口：状态查询
    # -------------------------------------------------------------------------

    @property
    def api_enabled(self):
        return self._api_enabled

    @property
    def waiting_confirm(self):
        return self._waiting_confirm

    @property
    def pending_message(self):
        return self._pending_message

    def get_status(self):
        """
        返回当前路由状态字典，供前端 UI 轮询显示。

        返回格式：
            {
                "api_enabled":     bool,
                "waiting_confirm": bool,
                "mode":            str,   # "local" / "pending"
            }
        """
        if self._waiting_confirm:
            mode = "pending"
        else:
            mode = "local"

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
        """
        self._api_enabled = enabled
        if not enabled:
            self._reset()
        logger.info(f"线上 API {'开启' if enabled else '关闭'}")

    def force_online(self):
        """
        手动强制切换到线上模式（由前端"切换"按钮触发）。
        标记下一条消息走 Claude，并记录触发时刻用于超时检测。

        [v2 修改] 新增 _force_online_time = datetime.now(utc)：
            _handle_confirm_reply() 在处理消息前会检查触发时刻是否超过
            FORCE_ONLINE_TIMEOUT_MINUTES，超时则自动 reset，避免状态残留。

        返回：
            str — 提示文本，供 main.py 展示给用户
        """
        if not self._api_enabled:
            return "线上 API 当前已关闭，请先在面板中开启。"
        if not ANTHROPIC_API_KEY:
            return "未检测到 ANTHROPIC_API_KEY，请先配置环境变量。"

        self._waiting_confirm   = True
        self._pending_message   = "__FORCE_ONLINE__"
        self._force_online_time = datetime.now(timezone.utc)   # [新增] 记录触发时刻

        logger.info(f"手动切换到线上模式，超时阈值 {FORCE_ONLINE_TIMEOUT_MINUTES} 分钟")
        return "好，下一条消息我会用线上 API 处理。"

    # -------------------------------------------------------------------------
    # 公开接口：消息路由判断
    # -------------------------------------------------------------------------

    def route(self, message: str):
        """
        判断当前消息的处理路径。

        判断优先级（从高到低）：
          1. 正在等待确认 → 检测用户是否回复了确认/拒绝词（含超时检查）
          2. API 已关闭  → 强制本地
          3. API Key 未配置 → 强制本地
          4. 自动检测     → 检测代码块和关键词，触发时发确认询问
          5. 默认         → 本地 Qwen

        参数：
            message — 用户发送的消息文本（已去除首尾空白）

        返回：
            dict，包含以下字段：
              action  — "ask_confirm" / "local" / "online" / "confirm_yes" / "confirm_no"
              text    — 询问文本（仅 ask_confirm 时有值）
              message — 实际需要处理的消息
        """
        if self._waiting_confirm:
            return self._handle_confirm_reply(message)

        if not self._api_enabled:
            return {"action": "local", "message": message}

        if not ANTHROPIC_API_KEY:
            return {"action": "local", "message": message}

        if self._should_trigger(message):
            self._pending_message  = message
            self._waiting_confirm  = True
            self._force_online_time = datetime.now(timezone.utc)   # 自动触发也记录时刻
            logger.info("自动检测触发，等待用户确认")
            return {
                "action":  "ask_confirm",
                "text":    CONFIRM_ASK_TEXT,
                "message": message,
            }

        return {"action": "local", "message": message}

    # -------------------------------------------------------------------------
    # 公开接口：调用 Claude API
    # -------------------------------------------------------------------------

    def call_claude(self, message: str):
        """
        调用 Claude API 处理单条消息，返回回复文本。

        Claude 作为无状态工具：
          · 不注入任何记忆上下文
          · 只带一个简短的角色说明 system prompt
          · 处理完后路由自动回到本地模式

        参数：
            message — 需要 Claude 处理的用户消息

        返回：
            str — Claude 的回复文本
            str — 调用失败时返回括号包裹的错误信息
        """
        logger.info(f"调用 Claude API，消息长度={len(message)}")

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
            "messages":   [{"role": "user", "content": message}],
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
            logger.info(f"Claude 回复成功，长度={len(reply)}")
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
        """
        if "```" in message:
            logger.debug("检测到代码块，触发确认")
            return True

        message_lower = message.lower()
        for kw in TRIGGER_KEYWORDS:
            if kw in message_lower:
                logger.debug(f"检测到关键词 [{kw}]，触发确认")
                return True

        return False

    def _handle_confirm_reply(self, message: str):
        """
        处理用户对确认询问的回复。

        [v2 修改] 在处理任何回复之前，先检查 force_online 触发时刻是否超时：
          · 未超时：正常执行确认/拒绝逻辑
          · 已超时：自动 reset，把当前消息当作普通消息走本地模式，
                    并打印提示避免用户困惑

        超时阈值：FORCE_ONLINE_TIMEOUT_MINUTES（默认30分钟）
        触发时刻：_force_online_time（force_online() 或自动触发时写入）

        参数：
            message — 用户的回复消息

        返回：
            dict — route() 标准返回格式
        """
        # ------------------------------------------------------------------
        # [新增] 超时检查：force_online 触发后超过阈值未发消息，自动 reset
        # ------------------------------------------------------------------
        if self._force_online_time is not None:
            now     = datetime.now(timezone.utc)
            elapsed = (now - self._force_online_time).total_seconds() / 60

            if elapsed >= FORCE_ONLINE_TIMEOUT_MINUTES:
                logger.warning(
                    f"force_online 已超时（{elapsed:.1f} 分钟 >= "
                    f"{FORCE_ONLINE_TIMEOUT_MINUTES} 分钟），自动 reset 回本地模式"
                )
                self._reset()
                # 超时后把当前消息按普通本地流程处理，不触发 Claude
                return {"action": "local", "message": message}

        # ------------------------------------------------------------------
        # 正常确认/拒绝流程（未超时）
        # ------------------------------------------------------------------
        message_lower = message.lower().strip()
        cached        = self._pending_message

        # 特殊情况：手动 force_online，下一条真实消息就是要处理的内容
        if cached == "__FORCE_ONLINE__":
            self._reset()
            return {"action": "online", "message": message}

        # 用户确认
        if message_lower in CONFIRM_YES_KEYWORDS:
            logger.info("用户确认，走 Claude API")
            self._reset()
            return {"action": "confirm_yes", "message": cached}

        # 用户拒绝
        if message_lower in CONFIRM_NO_KEYWORDS:
            logger.info("用户拒绝，走本地 Qwen")
            self._reset()
            return {"action": "confirm_no", "message": cached}

        # 用户说了其他话（既不是确认也不是拒绝），视为取消
        logger.debug("未识别的确认回复，清除等待状态，按正常消息处理")
        self._reset()
        return {"action": "local", "message": message}

    def _reset(self):
        """
        重置等待确认状态，回到本地模式。
        每次 Claude 调用完成（无论成功失败）或用户拒绝后调用。

        [v2 修改] 同时清除 _force_online_time，避免时间戳残留。
        """
        self._waiting_confirm   = False
        self._pending_message   = None
        self._force_online_time = None   # [新增] 清除触发时刻


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    print("=== router.py 验证测试 ===\n")

    router = Router()

    print("--- 测试一：普通消息 → 本地 ---")
    result = router.route("你好，今天天气怎么样")
    print(f"action={result['action']}（应为 local）\n")

    print("--- 测试二：触发关键词 → 询问确认 ---")
    result = router.route("帮我分析一下这段代码为什么报错")
    print(f"action={result['action']}（应为 ask_confirm）")
    print(f"text={result.get('text')}\n")

    print("--- 测试三：用户确认 → confirm_yes ---")
    result2 = router.route("好")
    print(f"action={result2['action']}（应为 confirm_yes）\n")

    print("--- 测试四：触发后用户拒绝 → confirm_no ---")
    router.route("帮我优化这段代码")
    result3 = router.route("不用")
    print(f"action={result3['action']}（应为 confirm_no）\n")

    print("--- 测试五：代码块触发 ---")
    result4 = router.route("这段代码有问题：\n```python\nprint('hello')\n```")
    print(f"action={result4['action']}（应为 ask_confirm）")
    router.route("不用")
    print()

    print("--- 测试六：API 关闭 → 强制本地 ---")
    router.set_api_enabled(False)
    result5 = router.route("帮我分析报错")
    print(f"action={result5['action']}（应为 local，API 已关闭）")
    router.set_api_enabled(True)
    print()

    print("--- 测试七：force_online → 下一条走 Claude ---")
    tip = router.force_online()
    print(f"提示：{tip}")
    result6 = router.route("帮我写一个快速排序")
    print(f"action={result6['action']}（应为 online）")
    print()

    print("--- 测试八：超时自动 reset（问题C修复核心）---")
    # 模拟 force_online 触发后超过30分钟
    router.force_online()
    # 手动把触发时刻倒推35分钟，模拟超时
    from datetime import timedelta
    router._force_online_time = datetime.now(timezone.utc) - timedelta(minutes=35)
    result7 = router.route("这条消息应该走本地，因为超时了")
    print(f"action={result7['action']}（应为 local，force_online 已超时自动 reset）")
    print(f"_waiting_confirm={router._waiting_confirm}（应为 False）")
    print(f"_force_online_time={router._force_online_time}（应为 None）")
    print()

    print("验证完成（Claude API 实际调用未测试，需要有效的 API Key）")
