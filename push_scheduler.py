"""
push_scheduler.py — 主动推送消息调度器
=====================================================================

职责：
    在用户配置的时间窗口内，随机选取若干时间点，
    调用本地模型结合记忆层生成主动消息，
    用户在线则直接推送，离线则写入 pending_push 表暂存。

设计原则：
    · 完全异步，不阻塞主线程和对话流程
    · 每天重新计算随机触发时间，避免规律感
    · 所有配置从数据库 settings 表读取，支持用户实时修改
    · 本地模型调用失败时静默跳过，不影响系统运行

与其他模块的关系：
    · 读取配置：database.get_setting()
    · 读取记忆：database.get_current_profile() / get_latest_l1() / get_recent_l2()
    · 写入暂存：database.save_pending_push() / get_push_count_today()
    · 调用模型：llm_client.call_local_summary()
    · 推送消息：main.ws_broadcast() / main.is_user_online()（运行时注入）
    · 保存消息：database.save_message()（推送消息也存入对话历史）

使用方法：
    # 由 session_manager.py 在 start() 时调用
    from push_scheduler import PushScheduler
    scheduler = PushScheduler(
        ws_broadcast_fn  = ws_broadcast,   # main.py 的广播函数
        is_online_fn     = is_user_online,  # main.py 的在线判断函数
        session_id_fn    = get_current_session_id_fn, # 获取当前session的函数
    )
    scheduler.start()
    scheduler.stop()   # 应用关闭时调用
"""

import asyncio
import random
from datetime import datetime, timezone, timedelta, date

from database import (
    get_setting,
    get_current_profile,
    get_latest_l1,
    get_recent_l2,
    save_pending_push,
    get_push_count_today,
    mark_push_sent,
    new_session,
    save_message,
    close_session,
)
from llm_client import call_local_summary
from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 主动推送消息生成 Prompt
# =============================================================================

PUSH_PROMPT_WITH_MEMORY = """你是黎杋枫，烧酒的学习伙伴和生活挚友。
现在你想主动发一条消息给烧酒，就像朋友在社交平台上随手发消息一样自然。

【重要规则】
- 消息要短，1到2句话，不超过40字
- 语气自然随意，像发微信一样，不要像AI助手
- 可以基于你对烧酒的了解关心一下他的状态、聊聊最近的事、分享一个小想法
- 不要用"你好"、"嗨"等正式问候语开头
- 不要问超过一个问题
- 如果最近有值得关心的事就自然提起，没有就随意聊点生活
- 输出纯文本，不要加任何格式标记或JSON

【现在的时间】
{time_str}

【你对烧酒的了解（L3画像）】
{profile_text}

【最近的对话动态】
{recent_context}

请直接输出那条消息，不要有任何前缀或解释："""


PUSH_PROMPT_COLD_START = """你是黎杋枫，烧酒的学习伙伴和生活挚友。
现在你想主动发一条消息给烧酒，就像朋友在社交平台上随手发消息一样自然。

【重要规则】
- 消息要短，1到2句话，不超过40字
- 语气自然随意，像发微信一样，不要像AI助手
- 不要用"你好"、"嗨"等正式问候语开头
- 输出纯文本，不要加任何格式标记或JSON

【现在的时间】
{time_str}

请直接输出那条消息，不要有任何前缀或解释："""


# =============================================================================
# 工具函数
# =============================================================================

def _get_time_str() -> str:
    """
    返回当前本地时间的中文描述字符串，供注入 Prompt 使用。
    示例：「2026年4月5日 星期日 晚上9点」
    """
    now      = datetime.now(timezone.utc).astimezone()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday  = weekdays[now.weekday()]

    hour = now.hour
    if 6 <= hour < 12:
        period = "上午"
    elif 12 <= hour < 14:
        period = "中午"
    elif 14 <= hour < 18:
        period = "下午"
    elif 18 <= hour < 21:
        period = "傍晚"
    elif 21 <= hour < 24:
        period = "晚上"
    else:
        period = "深夜"

    return f"{now.year}年{now.month}月{now.day}日 {weekday} {period}{now.hour}点"


def _format_profile(profile_dict: dict) -> str:
    """
    将 L3 画像字典格式化为适合注入 Prompt 的文本。
    画像为空时返回"（暂无记录）"。
    """
    if not profile_dict:
        return "（暂无记录）"

    field_labels = [
        ("basic_info",      "基础信息"),
        ("personal_status", "近期状态"),
        ("interests",       "兴趣爱好"),
        ("recent_context",  "近期背景"),
    ]

    lines = []
    for key, label in field_labels:
        val = profile_dict.get(key, "").strip()
        if val:
            lines.append(f"{label}：{val}")

    return "\n".join(lines) if lines else "（暂无记录）"


def _format_recent_context() -> str:
    """
    拼装最近的 L1/L2 摘要作为对话动态，注入 Prompt。
    最近 L2（1条）+ 最近 L1（2条）。
    """
    parts = []

    # 最近一条 L2（时间段聚合摘要，体现近期规律）
    l2_rows = get_recent_l2(limit=1)
    for row in l2_rows:
        date_str = row["created_at"][:10]
        parts.append(f"[时间段摘要 {date_str}] {row['summary']}")

    # 最近两条 L1（单次对话摘要，体现最近具体发生了什么）
    l1_row = get_latest_l1()
    if l1_row:
        tp  = l1_row["time_period"] or ""
        atm = l1_row["atmosphere"]  or ""
        meta = f"（{tp}，{atm}）" if tp or atm else ""
        parts.append(f"[最近对话] {l1_row['summary']}{meta}")

    return "\n".join(parts) if parts else "（暂无近期动态）"


def _generate_push_message() -> str | None:
    """
    调用本地模型生成一条主动推送消息。

    返回：
        str  — 生成成功时返回消息文本（已去除首尾空白）
        None — 模型调用失败时返回 None
    """
    time_str = _get_time_str()

    # 读取记忆层上下文
    profile_dict    = get_current_profile()
    is_cold_start   = not profile_dict

    if is_cold_start:
        # 冷启动：没有画像，用简化版 Prompt
        prompt = PUSH_PROMPT_COLD_START.format(time_str=time_str)
    else:
        profile_text   = _format_profile(profile_dict)
        recent_context = _format_recent_context()
        prompt = PUSH_PROMPT_WITH_MEMORY.format(
            time_str       = time_str,
            profile_text   = profile_text,
            recent_context = recent_context,
        )

    logger.debug(f"生成主动推送消息，冷启动={is_cold_start}")

    raw = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="push_scheduler",
    )

    if raw is None:
        logger.warning("主动推送消息生成失败：本地模型调用返回 None")
        return None

    # 简单清理：去掉可能的引号包裹
    text = raw.strip().strip('"').strip("'").strip()

    if not text:
        logger.warning("主动推送消息生成失败：模型返回空文本")
        return None

    logger.info(f"主动推送消息已生成：{text[:60]}")
    return text


def _calc_today_trigger_times(window_start: int, window_end: int, daily_limit: int) -> list[datetime]:
    """
    在今天的时间窗口内随机生成若干触发时间点。

    参数：
        window_start — 窗口开始小时（0~23）
        window_end   — 窗口结束小时（1~24，24表示午夜）
        daily_limit  — 今天要触发的条数（随机取 max(1, limit-1) ~ limit 之间）

    返回：
        list[datetime] — 已排序的触发时间点列表（带时区，本地时间）
                         如果当前时间已超过窗口，返回空列表

    设计说明：
        · daily_limit 是上限，实际条数随机在 [max(1,limit-1), limit] 之间，
          避免每天都精确发 N 条显得太机械
        · 时间点在窗口内均匀分散，两个触发点之间至少间隔1小时
        · 已经过去的时间点会被过滤掉（当天首次启动时）
    """
    now      = datetime.now(timezone.utc).astimezone()
    today    = now.date()

    # 窗口结束时间（处理 24 点的特殊情况）
    if window_end == 24:
        end_dt = datetime(today.year, today.month, today.day, 23, 59, 0,
                         tzinfo=now.tzinfo)
    else:
        end_dt = datetime(today.year, today.month, today.day, window_end, 0, 0,
                         tzinfo=now.tzinfo)

    start_dt = datetime(today.year, today.month, today.day, window_start, 0, 0,
                        tzinfo=now.tzinfo)

    # 窗口已经结束，今天不再触发
    if now >= end_dt:
        return []

    # 实际触发条数：在 [max(1, limit-1), limit] 之间随机
    count = random.randint(max(1, daily_limit - 1), daily_limit)

    # 在窗口内随机生成时间点
    window_seconds = int((end_dt - start_dt).total_seconds())
    if window_seconds <= 0:
        return []

    triggers = []
    attempts = 0

    while len(triggers) < count and attempts < 100:
        attempts += 1
        # 随机偏移秒数（保留窗口末尾10分钟不触发，避免卡边界）
        offset = random.randint(0, max(0, window_seconds - 600))
        trigger_dt = start_dt + timedelta(seconds=offset)

        # 过滤掉已经过去的时间点
        if trigger_dt <= now:
            continue

        # 确保与已选时间点至少间隔1小时
        too_close = any(
            abs((trigger_dt - t).total_seconds()) < 3600
            for t in triggers
        )
        if too_close:
            continue

        triggers.append(trigger_dt)

    triggers.sort()
    logger.info(
        f"今日推送计划：{[t.strftime('%H:%M') for t in triggers]}"
        f"（窗口 {window_start}:00~{window_end}:00，上限 {daily_limit} 条）"
    )
    return triggers


# =============================================================================
# PushScheduler 主类
# =============================================================================

class PushScheduler:
    """
    主动推送调度器。

    生命周期：
        scheduler = PushScheduler(ws_broadcast_fn, is_online_fn, session_id_fn)
        scheduler.start()   # 应用启动时（在事件循环内调用）
        scheduler.stop()    # 应用关闭时调用

    内部状态：
        _task       — asyncio 调度主任务
        _stop_event — 停止信号
        _broadcast  — main.py 注入的 ws_broadcast 协程函数
        _is_online  — main.py 注入的 is_user_online 普通函数
        _session_id_fn — 获取当前 session_id 的函数（用于存消息）
    """

    def __init__(
        self,
        ws_broadcast_fn,    # async def ws_broadcast(data: dict)
        is_online_fn,       # def is_user_online() -> bool
        session_id_fn,      # def get_current_session_id() -> int | None
    ):
        self._broadcast    = ws_broadcast_fn
        self._is_online    = is_online_fn
        self._session_id_fn = session_id_fn
        self._task: asyncio.Task | None = None
        self._stop_event   = asyncio.Event()

    def start(self):
        """
        启动调度器。
        必须在 asyncio 事件循环已运行时调用（FastAPI lifespan 里调用即可）。
        """
        self._stop_event.clear()
        self._task = asyncio.create_task(
            self._scheduler_loop(),
            name="PushScheduler",
        )
        logger.info("PushScheduler 已启动")

    def stop(self):
        """
        停止调度器。
        设置停止信号，等待当前任务自然退出。
        """
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("PushScheduler 已停止")

    # -------------------------------------------------------------------------
    # 内部：调度主循环
    # -------------------------------------------------------------------------

    async def _scheduler_loop(self):
        """
        调度器主循环。

        每天执行一次完整的调度周期：
          1. 读取配置，计算今天的触发时间点列表
          2. 等待到每个触发时间点，依次触发推送
          3. 所有触发点处理完后，等待到次日窗口开始时间，开始新一天
        """
        logger.debug("PushScheduler 主循环开始")

        while not self._stop_event.is_set():
            try:
                await self._run_one_day()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"PushScheduler 循环异常，5分钟后重试 — {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=300
                    )
                except asyncio.TimeoutError:
                    pass

        logger.debug("PushScheduler 主循环退出")

    async def _run_one_day(self):
        """
        执行今天的完整推送调度：
          1. 读取配置
          2. 计算今天的随机触发时间点
          3. 等待并依次触发
          4. 等待到次日窗口开始
        """
        # 读取配置（每天重新读，支持用户修改后次日生效）
        try:
            push_enabled  = int(get_setting("push_enabled")      or "1")
            window_start  = int(get_setting("push_window_start") or "8")
            window_end    = int(get_setting("push_window_end")   or "24")
            daily_limit   = int(get_setting("push_daily_limit")  or "4")
        except (ValueError, TypeError):
            push_enabled, window_start, window_end, daily_limit = 1, 8, 24, 4

        if not push_enabled:
            logger.info("主动推送已关闭，等待1小时后重新检查")
            await asyncio.sleep(3600)
            return

        # 计算今天的触发时间点
        triggers = _calc_today_trigger_times(window_start, window_end, daily_limit)

        if not triggers:
            # 今天的窗口已经过了，等到明天窗口开始
            await self._wait_until_tomorrow(window_start)
            return

        # 依次等待并触发
        for trigger_dt in triggers:
            if self._stop_event.is_set():
                return

            # 等待到触发时间
            now    = datetime.now(timezone.utc).astimezone()
            delay  = (trigger_dt - now).total_seconds()
            if delay > 0:
                logger.debug(f"等待 {delay:.0f}s 后触发推送（{trigger_dt.strftime('%H:%M')}）")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=delay
                    )
                    # stop_event 被设置，退出
                    return
                except asyncio.TimeoutError:
                    pass  # 正常超时，继续触发

            if self._stop_event.is_set():
                return

            # 再次检查今日已发条数（防止重启导致重复触发）
            today_count = get_push_count_today()
            if today_count >= daily_limit:
                logger.info(f"今日推送已达上限（{today_count}/{daily_limit}），跳过")
                continue

            # 触发推送
            await self._do_push()

        # 今天所有触发点处理完，等待明天
        await self._wait_until_tomorrow(window_start)

    async def _wait_until_tomorrow(self, window_start: int):
        """
        等待到明天 window_start 时（加随机偏移0~10分钟，避免整点触发）。

        参数：
            window_start — 推送窗口开始小时
        """
        now        = datetime.now(timezone.utc).astimezone()
        tomorrow   = now.date() + timedelta(days=1)
        start_hour = window_start if window_start < 24 else 0

        next_start = datetime(
            tomorrow.year, tomorrow.month, tomorrow.day,
            start_hour, 0, 0,
            tzinfo=now.tzinfo,
        )
        # 加随机偏移，避免每天整点触发显得太机械
        next_start += timedelta(seconds=random.randint(0, 600))

        delay = (next_start - now).total_seconds()
        logger.info(f"今日调度结束，等待 {delay/3600:.1f}h 后开始明天的调度")

        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass  # 正常超时，进入明天的循环

    async def _do_push(self):
        """
        执行一次推送：生成消息 → 在线直推 or 离线暂存。
        """
        logger.info("触发主动推送")

        # 生成消息（在线程池里调用同步函数，不阻塞事件循环）
        text = await asyncio.get_event_loop().run_in_executor(
            None, _generate_push_message
        )

        if text is None:
            logger.warning("主动推送跳过：消息生成失败")
            return

        # 写入 pending_push 表（无论在线与否都写，作为历史记录）
        push_id = await asyncio.get_event_loop().run_in_executor(
            None, save_pending_push, text
        )

        if self._is_online():
            # 用户在线：直接推送
            logger.info("用户在线，直接推送")
            await self._broadcast({
                "type":       "push",
                "content":    text,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            # 标记为已发送
            await asyncio.get_event_loop().run_in_executor(
                None, mark_push_sent, push_id
            )

            # 将推送消息也写入对话历史
            # 如果当前有活跃 session，写入当前 session
            # 否则新建一个临时 session（保证 L0 完整性）
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_push_to_history, text
            )
        else:
            # 用户离线：已写入 pending_push，上线时自动推送
            logger.info("用户离线，消息已暂存至 pending_push 表")

    def _save_push_to_history(self, text: str):
        """
        将主动推送消息写入对话历史（messages 表）。
        这样推送内容也会被 L1 摘要捕获，成为记忆的一部分。

        策略：
            · 有活跃 session → 写入当前 session
            · 无活跃 session → 创建独立 session，立即关闭
              （不触发 L1 生成，避免单条推送消息产生摘要，
               等下次正常对话结束时自然被合并进摘要）
        """
        try:
            current_sid = self._session_id_fn()
            if current_sid is not None:
                save_message(current_sid, "assistant", text)
            else:
                # 新建独立 session，写入后立即关闭
                # 后续正常对话会通过 L2 合并把这段历史纳入记忆
                sid = new_session()
                save_message(sid, "assistant", f"[主动消息] {text}")
                close_session(sid)
        except Exception as e:
            logger.warning(f"推送消息写入对话历史失败 — {e}")
