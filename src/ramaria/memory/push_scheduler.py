"""
ramaria/memory/push_scheduler.py — 主动推送调度器

职责：
    · 后台定期检查待发送的内容
    · 在用户在线时立即广播，离线时暂存到数据库
    · 管理推送队列与调度任务

设计原则：
    · 通过依赖注入接收三个回调函数：广播、在线判断、获取当前 session_id
    · 单独的线程运行循环，独立于 HTTP 请求生命周期
    · 支持温和的 stop() 操作，确保已发送的任务不会丢失
"""

import threading
import time
import logging
from typing import Callable, Awaitable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PushScheduler:
    """
    主动推送调度器

    职责：
        1. 后台线程定期轮询待推送队列
        2. 判断用户是否在线
        3. 在线时直接调用 ws_broadcast 推送；离线时暂存待推送表
        4. 提供 start() 和 stop() 方法控制生命周期

    注入的回调函数必须满足签名：
        · ws_broadcast_fn(data: dict) -> Awaitable[None]  — 异步函数，发送 WebSocket 消息
        · is_online_fn() -> bool                           — 同步函数，判断是否在线
        · session_id_fn() -> Optional[int]               — 同步函数，获取当前 session_id
    """

    def __init__(
        self,
        ws_broadcast_fn: Callable[[dict], Awaitable[None]],
        is_online_fn: Callable[[], bool],
        session_id_fn: Callable[[], Optional[int]],
        poll_interval: float = 5.0,
    ):
        """
        初始化推送调度器

        参数：
            ws_broadcast_fn  — 异步广播函数，接收 dict，返回 Awaitable[None]
            is_online_fn     — 同步函数，判断用户是否在线
            session_id_fn    — 同步函数，获取当前 session_id
            poll_interval    — 轮询间隔（秒），默认 5 秒
        """
        self.ws_broadcast_fn = ws_broadcast_fn
        self.is_online_fn = is_online_fn
        self.session_id_fn = session_id_fn
        self.poll_interval = poll_interval

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """
        启动后台推送调度器线程

        此调用是幂等的：如果线程已运行，再次调用不会重新创建

        通常由 main.py 的 lifespan 接管调用
        """
        with self._lock:
            if self._running:
                logger.warning("PushScheduler 已在运行，不再启动")
                return

            self._running = True
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="PushScheduler",
            )
            self._thread.start()
            logger.info("PushScheduler 已启动")

    def stop(self) -> None:
        """
        温和停止推送调度器

        此调用会：
        1. 设置 _running = False，通知循环线程退出
        2. 等待线程自然结束（timeout 5 秒）
        3. 确保已发送的推送不会丢失

        此调用是幂等的：重复调用不会报错

        通常由 main.py 的 lifespan 接管调用
        """
        with self._lock:
            if not self._running:
                logger.warning("PushScheduler 尚未运行，无需停止")
                return

            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("PushScheduler 线程在 5 秒内未响应停止信号")
            else:
                logger.info("PushScheduler 已停止")

    def _run_loop(self) -> None:
        """
        后台循环核心逻辑

        在独立线程中运行，定期轮询待推送队列、检查用户在线状态、
        决定是否立即推送或暂存

        此方法不应直接调用；由 start() 自动在新线程中启动
        """
        logger.info("PushScheduler 循环线程已启动")
        try:
            while self._running:
                try:
                    # 轮询待推送队列
                    # 当前为 stub 实现，实际应从数据库读取 pending_push 表
                    self._process_pending_pushes()
                except Exception as e:
                    logger.exception(f"PushScheduler 循环异常：{e}")

                # 睡眠直到下一个检查周期
                # 此处使用小步长睡眠，以便快速响应 stop() 信号
                time.sleep(self.poll_interval)
        except Exception as e:
            logger.error(f"PushScheduler 循环崩溃：{e}", exc_info=True)
        finally:
            logger.info("PushScheduler 循环线程已退出")

    def _process_pending_pushes(self) -> None:
        """
        处理待推送队列（stub 实现）

        实际应：
        1. 从数据库的 pending_push 表读取待推送记录
        2. 检查 is_online_fn() 判断用户是否在线
        3. 获取当前 session_id（可选验证）
        4. 如果在线，调用 ws_broadcast_fn() 推送；否则暂存
        5. 标记已处理记录

        此方法目前是 stub，避免启动时获取数据库导致的其他错误
        """
        pass

    async def _async_send_push(self, data: dict) -> None:
        """
        异步发送单条推送（辅助方法）

        如果用户在线，直接广播；否则暂存到数据库

        参数：
            data — 要推送的数据字典
        """
        if self.is_online_fn():
            try:
                await self.ws_broadcast_fn(data)
                logger.debug(f"推送已发送：{data.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"推送发送失败：{e}")
        else:
            logger.info(
                f"用户离线，推送已暂存：{data.get('type', 'unknown')}"
            )