"""
src/ramaria/memory/push_scheduler.py — 主动推送调度器

职责：
    · 在用户设定的时间窗口内，随机选取一个时刻主动向用户发送消息
    · 调用本地模型生成自然的主动消息内容（基于近期记忆上下文）
    · 用户在线时直接通过 WebSocket 推送；离线时写入 pending_push 表暂存
    · 每日推送条数受 push_daily_limit 配置限制

设计原则：
    · 通过依赖注入接收三个回调函数，不直接依赖 FastAPI 路由层
    · 推送消息的生成基于真实记忆上下文，不凭空捏造
    · 时间窗口和上限从 settings 表动态读取，支持用户实时修改后生效

调用方：
    app/main.py 的 lifespan startup 阶段注入回调并调用 start()
"""

import asyncio
import random
import threading
import time
from datetime import datetime, timezone, date, timedelta
from typing import Callable, Awaitable, Optional

from ramaria.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 推送消息生成 Prompt 模板
# =============================================================================
#
# 占位符说明：
#   {assistant_name} — 助手名字，从 persona.toml [identity] 读取
#   {user_name}      — 用户名字，从 persona.toml [identity] 读取
#   {current_time}   — 当前时间，如 "2026-04-11 14:30"
#   {weekday}        — 星期，如 "周五"
#   {time_period}    — 时间段，如 "下午"，从当前小时推断
#   {recent_context} — 近期 L1 摘要，最多3条，提供真实记忆上下文

_PUSH_PROMPT_TEMPLATE = """\
你是{assistant_name}，正在主动给{user_name}发一条消息。

【当前时间】
{current_time}（{weekday}，{time_period}）

【关于{user_name}的近期记忆】
{recent_context}

【发消息的原则】
- 像朋友日常发消息，随意自然，不要像助手汇报
- 可以从记忆里找一个细节延伸，也可以完全从当前时间出发
- 不能提到记忆里没有的事实，不能捏造{user_name}说过的话
- 不能和当前时间矛盾，{time_period}就该有{time_period}的语气
- 长度：1到3句话，如果有多句用||分隔
- 不要以问候语开头，直接说事

【输出格式】
只输出消息内容本身，不加任何解释或标注。\
"""

# 时间段推断表：小时 → 时间段文字
# 与 config.py 的 TIME_PERIOD_OPTIONS 对应
_HOUR_TO_PERIOD = {
    range(0, 6):   "深夜",
    range(6, 9):   "清晨",
    range(9, 12):  "上午",
    range(12, 18): "下午",
    range(18, 21): "傍晚",
    range(21, 24): "夜间",
}

# 中文星期映射（weekday() 返回 0=周一 ... 6=周日）
_WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _get_time_period(hour: int) -> str:
    """根据小时数返回对应的中文时间段名称。"""
    for hour_range, period in _HOUR_TO_PERIOD.items():
        if hour in hour_range:
            return period
    return "夜间"  # 安全降级


# =============================================================================
# PushScheduler 类
# =============================================================================

class PushScheduler:
    """
    主动推送调度器。

    生命周期：
        main.py lifespan startup → start()
        main.py lifespan shutdown → stop()（由 session_manager.stop() 间接调用）

    线程模型：
        调度器运行在一个独立的守护线程中。
        每次循环：
            1. 读取 settings 表获取当前配置
            2. 判断当前时刻是否在推送窗口内
            3. 若在窗口内且今日未达上限，计算随机触发时刻
            4. 等待到触发时刻，调用模型生成消息
            5. 在线则直接推送，离线则写入 pending_push 表
            6. 睡眠至下一个检查周期（固定60秒）

    注入的回调函数签名：
        ws_broadcast_fn(data: dict) -> Awaitable[None]
            向所有在线 WebSocket 客户端广播消息

        is_online_fn() -> bool
            判断当前是否有用户在线

        session_id_fn() -> Optional[int]
            获取当前活跃 session 的 id（用于读取近期记忆上下文）
    """

    # 调度器主循环的检查间隔（秒）
    # 每隔此时长唤醒一次，检查是否需要触发推送
    _CHECK_INTERVAL = 60

    def __init__(
        self,
        ws_broadcast_fn: Callable[[dict], Awaitable[None]],
        is_online_fn:    Callable[[], bool],
        session_id_fn:   Callable[[], Optional[int]],
    ):
        self._ws_broadcast_fn = ws_broadcast_fn
        self._is_online_fn    = is_online_fn
        self._session_id_fn   = session_id_fn

        self._thread:  Optional[threading.Thread] = None
        self._running: bool                        = False
        self._lock:    threading.Lock              = threading.Lock()

        # 记录今日已推送次数和对应日期，用于跨日重置
        # （数据库 get_push_count_today 是权威来源，这里只是快速校验缓存）
        self._today_date:  Optional[date] = None
        self._today_count: int            = 0

        # 记录今日已计划的推送触发时刻（datetime），避免在同一天重复计划
        self._scheduled_time: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """
        启动推送调度器后台线程。
        幂等：已在运行时再次调用无效果。
        """
        with self._lock:
            if self._running:
                logger.warning("PushScheduler 已在运行，忽略重复启动")
                return

            self._running = True

        self._thread = threading.Thread(
            target  = self._run_loop,
            name    = "PushScheduler",
            daemon  = True,   # 守护线程，主进程退出时自动结束
        )
        self._thread.start()
        logger.info("PushScheduler 已启动")

    def stop(self) -> None:
        """
        温和停止推送调度器。
        设置停止标志后等待当前循环自然结束（最多等待一个检查周期）。
        """
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread and self._thread.is_alive():
            # 最多等待一个检查周期 + 2秒缓冲
            self._thread.join(timeout=self._CHECK_INTERVAL + 2)
            if self._thread.is_alive():
                logger.warning("PushScheduler 线程未在预期时间内退出")
            else:
                logger.info("PushScheduler 已停止")

    # -------------------------------------------------------------------------
    # 内部：主循环
    # -------------------------------------------------------------------------

    def _run_loop(self) -> None:
        """
        调度器主循环。
        每隔 _CHECK_INTERVAL 秒执行一次调度检查。
        """
        logger.info(
            f"PushScheduler 循环启动，检查间隔 {self._CHECK_INTERVAL} 秒"
        )

        while self._running:
            try:
                self._tick()
            except Exception as e:
                # 单次 tick 失败不中止整个调度器
                logger.exception(f"PushScheduler tick 异常，已跳过本次 — {e}")

            # 小步睡眠：每秒检查一次 _running，使 stop() 能快速响应
            for _ in range(self._CHECK_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)

        logger.info("PushScheduler 循环已退出")

    def _tick(self) -> None:
        """
        单次调度逻辑，由主循环周期性调用。

        执行流程：
            1. 读取推送配置（开关、时间窗口、每日上限）
            2. 检查推送开关
            3. 检查当前时刻是否在推送窗口内
            4. 检查今日是否已达上限
            5. 若尚未为今天计划推送时刻，随机选取一个窗口内的时刻
            6. 若当前时刻已过计划触发时刻，执行推送
        """
        # ── 读取配置 ──────────────────────────────────────────────────────────
        from ramaria.storage.database import get_setting, get_push_count_today

        push_enabled    = int(get_setting("push_enabled")      or "1")
        window_start    = int(get_setting("push_window_start") or "8")
        window_end      = int(get_setting("push_window_end")   or "24")
        daily_limit     = int(get_setting("push_daily_limit")  or "4")

        if not push_enabled:
            return

        # ── 当前时间信息 ──────────────────────────────────────────────────────
        now       = datetime.now()          # 使用本地时间判断窗口（用户配置的是本地时间）
        today     = now.date()
        cur_hour  = now.hour

        # ── 跨日重置计划时刻 ──────────────────────────────────────────────────
        if self._today_date != today:
            self._today_date   = today
            self._today_count  = 0
            self._scheduled_time = None
            logger.debug(f"PushScheduler 日期更新为 {today}，重置计划")

        # ── 检查今日已推送次数（从数据库取权威值）────────────────────────────
        self._today_count = get_push_count_today()
        if self._today_count >= daily_limit:
            logger.debug(
                f"今日已推送 {self._today_count} 条，达到上限 {daily_limit}，跳过"
            )
            return

        # ── 检查当前是否在推送窗口内 ──────────────────────────────────────────
        # window_end == 24 表示到午夜，需要特殊处理
        in_window = (
            window_start <= cur_hour < window_end
            if window_end < 24
            else window_start <= cur_hour < 24
        )
        if not in_window:
            logger.debug(
                f"当前 {cur_hour}:xx 不在推送窗口 "
                f"[{window_start}:00, {window_end}:00)，跳过"
            )
            return

        # ── 为今天计划一个随机触发时刻 ────────────────────────────────────────
        if self._scheduled_time is None or self._scheduled_time.date() != today:
            self._scheduled_time = self._pick_random_time(
                today, window_start, window_end
            )
            logger.info(
                f"今日推送计划时刻：{self._scheduled_time.strftime('%H:%M:%S')}"
            )

        # ── 判断是否已到触发时刻 ──────────────────────────────────────────────
        if now < self._scheduled_time:
            logger.debug(
                f"距计划触发还有 "
                f"{int((self._scheduled_time - now).total_seconds())} 秒"
            )
            return

        # ── 执行推送 ──────────────────────────────────────────────────────────
        logger.info("推送触发时刻已到，开始生成推送消息")
        self._do_push()

        # 推送完成后重置计划时刻，下次循环不会重复触发
        self._scheduled_time = None

    def _pick_random_time(
        self,
        target_date: date,
        window_start: int,
        window_end: int,
    ) -> datetime:
        """
        在指定日期的时间窗口内随机选取一个触发时刻。

        参数：
            target_date  — 目标日期
            window_start — 窗口开始小时（含）
            window_end   — 窗口结束小时（不含，24表示午夜）

        返回：
            datetime — 随机选取的触发时刻（精确到秒）
        """
        end_hour = min(window_end, 23)   # 防止 hour=24 导致 datetime 异常

        # 窗口内的总秒数
        window_seconds = (end_hour - window_start) * 3600
        if window_seconds <= 0:
            # 窗口异常时降级为窗口开始整点
            return datetime(
                target_date.year, target_date.month, target_date.day,
                window_start, 0, 0
            )

        # 随机偏移秒数
        offset_seconds = random.randint(0, window_seconds - 1)
        base = datetime(
            target_date.year, target_date.month, target_date.day,
            window_start, 0, 0
        )
        return base.replace(second=0) + timedelta(seconds=offset_seconds)
        

    # -------------------------------------------------------------------------
    # 内部：执行推送
    # -------------------------------------------------------------------------

    def _do_push(self) -> None:
        """
        执行一次主动推送：
            1. 生成推送消息文本
            2. 判断用户是否在线
            3. 在线则广播；离线则写入 pending_push 表
        """
        # 生成消息内容
        content = self._generate_push_content()
        if not content:
            logger.warning("推送消息生成失败，本次跳过")
            return

        # 构造推送数据包
        push_data = {
            "type":       "push",
            "content":    content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._is_online_fn():
            # 用户在线：通过 WebSocket 直接广播
            # ws_broadcast_fn 是异步函数，在同步线程中需要创建新事件循环执行
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._ws_broadcast_fn(push_data))
                loop.close()
                logger.info(f"推送消息已实时发送：{content[:40]}…")
            except Exception as e:
                logger.error(f"WebSocket 推送失败，降级写入 pending_push — {e}")
                self._save_to_pending(content)
        else:
            # 用户离线：写入 pending_push 表，等用户上线后补发
            self._save_to_pending(content)
            logger.info(f"用户离线，推送消息已暂存：{content[:40]}…")

    def _save_to_pending(self, content: str) -> None:
        """将推送消息写入 pending_push 表。"""
        try:
            from ramaria.storage.database import save_pending_push
            save_pending_push(content)
        except Exception as e:
            logger.error(f"写入 pending_push 失败 — {e}")

    # -------------------------------------------------------------------------
    # 内部：生成推送消息内容
    # -------------------------------------------------------------------------

    def _generate_push_content(self) -> Optional[str]:
        """
        调用本地模型生成主动推送消息。

        步骤：
            1. 从 persona.toml 读取助手名和用户名
            2. 从数据库读取近期 L1 摘要作为记忆上下文
            3. 拼装 prompt，调用本地模型
            4. 返回生成的消息文本

        返回：
            str  — 生成的消息文本
            None — 任何步骤失败时返回 None
        """
        # ── 读取身份信息 ──────────────────────────────────────────────────────
        try:
            from ramaria.core.prompt_builder import get_builder
            identity       = get_builder().get_identity()
            assistant_name = identity["assistant_name"]
            user_name      = identity["user_name"]
        except Exception as e:
            logger.warning(f"读取 persona identity 失败，使用默认值 — {e}")
            assistant_name = "助手"
            user_name      = "用户"

        # ── 读取近期记忆上下文 ────────────────────────────────────────────────
        recent_context = self._build_recent_context()

        # ── 拼装时间信息 ──────────────────────────────────────────────────────
        now          = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M")
        weekday      = _WEEKDAY_ZH[now.weekday()]
        time_period  = _get_time_period(now.hour)

        # ── 拼装 Prompt ───────────────────────────────────────────────────────
        prompt = _PUSH_PROMPT_TEMPLATE.format(
            assistant_name = assistant_name,
            user_name      = user_name,
            current_time   = current_time,
            weekday        = weekday,
            time_period    = time_period,
            recent_context = recent_context,
        )

        # ── 调用本地模型 ──────────────────────────────────────────────────────
        try:
            from ramaria.core.llm_client import call_local_summary
            result = call_local_summary(
                messages = [{"role": "user", "content": prompt}],
                caller   = "push_scheduler",
            )
            if not result or not result.strip():
                logger.warning("模型返回了空内容")
                return None
            return result.strip()
        except Exception as e:
            logger.error(f"调用本地模型生成推送消息失败 — {e}")
            return None

    def _build_recent_context(self) -> str:
        """
        从数据库读取最近3条 L1 摘要，格式化为 prompt 注入文本。

        如果数据库查询失败或无记录，返回占位文本，
        让模型退化为基于时间的纯随机消息（仍然安全）。

        返回：
            str — 格式化后的近期记忆文本
        """
        try:
            from ramaria.storage.database import get_recent_l2
            # 复用 get_recent_l2 获取最近摘要；
            # 这里取 L1 层级更细，但 database.py 没有直接暴露
            # "最近N条L1按时间降序" 的函数，复用 read_tools 里的私有函数
            from ramaria.adapters.mcp.tools.read_tools import _get_recent_l1_rows

            rows = _get_recent_l1_rows(limit=3)
            if not rows:
                return "（暂无近期对话记录）"

            lines = []
            for row in rows:
                date_str = row["created_at"][:10] if row["created_at"] else "未知日期"
                tp       = row["time_period"] or ""
                atm      = row["atmosphere"]  or ""
                meta     = f"（{tp}，{atm}）" if (tp or atm) else ""
                lines.append(f"[{date_str}]{meta} {row['summary']}")

            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"读取近期记忆失败，使用占位文本 — {e}")
            return "（近期记忆读取失败，请根据时间自由发挥）"