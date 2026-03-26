"""
session_manager.py — Session 生命周期管理
负责维护当前活跃 session 的状态，并在后台持续检测空闲超时和 L2 定时合并。

核心职责：
  1. 维护当前活跃 session_id（内存中）
  2. 每条消息到来时，确保有活跃 session，并记录最后活跃时间
  3. 后台线程A（空闲检测）：每隔 IDLE_CHECK_INTERVAL_SECONDS 秒轮询一次：
       距上次消息超过 L1_IDLE_MINUTES 分钟
       → 关闭当前 session
       → 写入 L0 向量索引（滑动窗口切片，vector_store.index_l0_session）
       → 触发 L1 摘要生成（summarizer）
       → L1 写入后自动触发 L1 向量索引写入和 L2 条数检查（路径A，在 summarizer 内部完成）
  4. 后台线程B（L2 定时检查）：每隔 L2_CHECK_INTERVAL_SECONDS 秒轮询一次：
       → 调用 merger.check_and_merge()
       → merger 内部判断时间触发条件（路径B）

变更记录：
  v2 — 修复代码优化清单 P1-2：
       __main__ 测试块中移除对 database._get_connection() 私有函数的直接调用。
       原来为了模拟空闲超时，测试块直接用私有函数连接数据库并手写 UPDATE SQL，
       造成封装原则不一致，且若 _get_connection 内部实现变更会波及此处。
       修复方案：database.py 新增公开函数 update_message_time_for_test()，
       测试块改为通过该函数修改时间戳，不再直接接触私有函数。

与其他模块的关系：
  - 读写数据库：database.py
  - 读取配置：config.py
  - 触发 L1 摘要：summarizer.generate_l1_summary()
  - 触发 L2 合并：merger.check_and_merge()（懒加载，避免循环依赖）

使用方法：
    from session_manager import SessionManager
    sm = SessionManager()
    sm.start()                    # 启动两个后台线程，在 FastAPI startup 时调用
    session_id = sm.on_message()  # 每条消息到来时调用
    sm.stop()                     # 应用关闭时调用
"""

import threading
from datetime import datetime, timezone, timedelta

from config import (
    L1_IDLE_MINUTES,
    IDLE_CHECK_INTERVAL_SECONDS,
    L2_CHECK_INTERVAL_SECONDS,
)
from summarizer import generate_l1_summary
from database import (
    new_session,
    close_session,
    get_active_sessions,
    get_last_message_time,
)

from logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# L1 摘要触发函数
# =============================================================================

def trigger_l1_summary(session_id):
    """
    触发指定 session 的 L1 摘要生成。
    L1 写入后，summarizer 内部会自动触发 L2 条数检查（路径A）。

    参数：
        session_id — 需要生成摘要的 session id
    """
    logger.info(f"触发 session {session_id} 的 L1 摘要生成")
    generate_l1_summary(session_id)


# =============================================================================
# SessionManager 类
# =============================================================================

class SessionManager:
    """
    Session 生命周期管理器。

    内部状态：
        _current_session_id   — 当前活跃 session 的 id；None 表示没有活跃 session
        _lock                 — 线程锁，保护 _current_session_id 的并发读写
        _stop_event           — 通知所有后台线程停止的事件标志（共用一个）
        _idle_checker_thread  — 线程A：空闲检测
        _l2_checker_thread    — 线程B：L2 定时检查

    典型使用流程：
        sm = SessionManager()
        sm.start()                    # 启动两个后台线程
        sid = sm.on_message()         # 每条用户消息到来时调用，返回 session_id
        sm.stop()                     # 应用关闭时调用，优雅停止所有后台线程
    """

    def __init__(self):
        self._current_session_id  = None
        self._lock                = threading.Lock()
        self._stop_event          = threading.Event()
        self._idle_checker_thread = None
        self._l2_checker_thread   = None

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def start(self):
        """
        启动 SessionManager：
          1. 检查并处理遗留的未关闭 session
          2. 启动线程A（空闲检测）
          3. 启动线程B（L2 定时检查）

        调用时机：FastAPI 应用启动时（lifespan 的 startup 阶段）。
        """
        self._recover_active_sessions()
        self._start_idle_checker()
        self._start_l2_checker()
        logger.info("已启动，线程A（空闲检测）+ 线程B（L2 定时检查）均运行中")

    def stop(self):
        """
        优雅停止 SessionManager：
          1. 设置 stop_event，通知所有后台线程退出
          2. 等待两个线程结束

        调用时机：FastAPI 应用关闭时（lifespan 的 shutdown 阶段）。
        """
        self._stop_event.set()

        if self._idle_checker_thread and self._idle_checker_thread.is_alive():
            self._idle_checker_thread.join(timeout=IDLE_CHECK_INTERVAL_SECONDS + 1)

        if self._l2_checker_thread and self._l2_checker_thread.is_alive():
            self._l2_checker_thread.join(timeout=L2_CHECK_INTERVAL_SECONDS + 1)

        logger.info("已停止")

    def on_message(self):
        """
        每条用户消息到来时必须调用此函数。

        作用：
          - 如果当前没有活跃 session，自动新建一个
          - 返回当前活跃 session 的 id

        返回：
            int — 当前活跃 session 的 id

        注意：
            本函数只管理 session 状态，不负责将消息写入数据库。
            写消息由 main.py 的路由函数负责，调用 database.save_message()。
        """
        with self._lock:
            if self._current_session_id is None:
                self._current_session_id = new_session()
                logger.info(f"新建 session，id = {self._current_session_id}")
            return self._current_session_id

    def get_current_session_id(self):
        """
        读取当前活跃 session 的 id，不触发任何副作用。

        返回：
            int 或 None
        """
        with self._lock:
            return self._current_session_id

    def force_close_current_session(self):
        """
        手动立即关闭当前 session 并触发 L1 摘要生成。
        调用时机：用户主动点击"保存/结束对话"时，由 main.py 路由函数调用。
        """
        with self._lock:
            sid = self._current_session_id
            if sid is None:
                logger.debug("没有活跃 session，无需关闭")
                return
            self._close_and_summarize(sid)
            self._current_session_id = None

    # -------------------------------------------------------------------------
    # 内部方法 — 启动流程
    # -------------------------------------------------------------------------

    def _recover_active_sessions(self):
        """
        应用启动时，检查数据库中是否有未关闭的遗留 session。

        处理策略：
          - 一个遗留 session：恢复为当前活跃 session（正常热重启）
          - 多个遗留 session：异常情况，全部关闭并触发摘要
          - 无遗留 session：正常启动，不做任何操作
        """
        active = get_active_sessions()

        if len(active) == 0:
            logger.debug("启动检查：无遗留 session")

        elif len(active) == 1:
            sid = active[0]["id"]
            self._current_session_id = sid
            logger.info(f"启动检查：恢复遗留 session，id = {sid}")

        else:
            logger.warning(f"启动检查：发现 {len(active)} 个未关闭 session，全部关闭")
            for s in active:
                sid = s["id"]
                close_session(sid)
                trigger_l1_summary(sid)
                logger.info(f"已关闭遗留 session {sid}")

    def _start_idle_checker(self):
        """
        创建并启动线程A（空闲检测线程）。
        """
        self._stop_event.clear()
        self._idle_checker_thread = threading.Thread(
            target=self._idle_check_loop,
            name="IdleChecker",
            daemon=True,
        )
        self._idle_checker_thread.start()

    def _start_l2_checker(self):
        """
        创建并启动线程B（L2 定时检查线程）。
        """
        self._l2_checker_thread = threading.Thread(
            target=self._l2_check_loop,
            name="L2Checker",
            daemon=True,
        )
        self._l2_checker_thread.start()

    # -------------------------------------------------------------------------
    # 内部方法 — 线程A：空闲检测
    # -------------------------------------------------------------------------

    def _idle_check_loop(self):
        """
        线程A主循环：每隔 IDLE_CHECK_INTERVAL_SECONDS 秒执行一次空闲检测。
        """
        logger.debug(f"线程A（空闲检测）启动，轮询间隔 {IDLE_CHECK_INTERVAL_SECONDS} 秒")

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=IDLE_CHECK_INTERVAL_SECONDS)
            if self._stop_event.is_set():
                break
            self._check_idle_timeout()

        logger.debug("线程A（空闲检测）已退出")

    def _check_idle_timeout(self):
        """
        执行一次空闲超时检测。

        逻辑：
          1. 读取当前 session id（加锁快速读取后立即释放）
          2. 没有活跃 session 则跳过
          3. 查询最后消息时间，计算空闲时长
          4. 超过阈值则关闭 session 并触发 L1 摘要
        """
        with self._lock:
            sid = self._current_session_id

        if sid is None:
            return

        last_time_str = get_last_message_time(sid)
        if last_time_str is None:
            return

        last_time = datetime.fromisoformat(last_time_str)
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        idle_minutes = (now - last_time).total_seconds() / 60

        logger.debug(f"检测 session {sid}，已空闲 {idle_minutes:.1f} 分钟")

        if idle_minutes >= L1_IDLE_MINUTES:
            logger.info(f"session {sid} 空闲超时（{idle_minutes:.1f} 分钟），触发关闭")
            with self._lock:
                # 二次确认：避免拿到锁之前 session 已被其他路径关闭
                if self._current_session_id == sid:
                    self._close_and_summarize(sid)
                    self._current_session_id = None

    # -------------------------------------------------------------------------
    # 内部方法 — 线程B：L2 定时检查
    # -------------------------------------------------------------------------

    def _l2_check_loop(self):
        """
        线程B主循环：每隔 L2_CHECK_INTERVAL_SECONDS 秒执行一次 L2 触发检查。
        这是 L2 合并的路径B（时间触发），负责检查"最早 L1 距今 ≥ 7天"的条件。
        """
        logger.debug(f"线程B（L2 定时检查）启动，轮询间隔 {L2_CHECK_INTERVAL_SECONDS} 秒")

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=L2_CHECK_INTERVAL_SECONDS)
            if self._stop_event.is_set():
                break
            self._trigger_l2_check()

        logger.debug("线程B（L2 定时检查）已退出")

    def _trigger_l2_check(self):
        """
        执行一次 L2 合并触发检查（路径B：时间触发）。
        调用 merger.check_and_merge()，不满足条件时静默返回。

        使用懒加载 import 避免循环依赖。
        """
        logger.debug("线程B 触发 L2 定时检查")
        try:
            from merger import check_and_merge
            check_and_merge()
        except Exception as e:
            logger.warning(f"L2 定时检查出现异常 — {e}")

    # -------------------------------------------------------------------------
    # 内部方法 — 关闭 session 并触发摘要
    # -------------------------------------------------------------------------

    def _close_and_summarize(self, session_id):
        """
        关闭 session，触发 L0 向量索引写入，再触发 L1 摘要生成。
        唯一负责"关闭 + 索引 + 摘要"三步的地方，确保三个动作总是成对执行。

        执行顺序说明：
          1. close_session()      — 先关闭 session，打上 ended_at 时间戳
          2. index_l0_session()   — 对原始消息做滑动窗口切片，写入 L0 向量索引
                                    必须在 L1 摘要生成之前完成，因为 L1 写入后
                                    会触发 L2 合并，届时检索链路需要 L0 已就位
          3. trigger_l1_summary() — 调用 Qwen 生成 L1 摘要并写入 SQLite，
                                    summarizer 内部会继续触发 L1 向量索引写入
                                    和 L2 条数检查（路径A）

        注意：
            调用此函数前必须已持有 self._lock，或在单线程环境中调用。
        """
        close_session(session_id)
        logger.info(f"session {session_id} 已关闭")

        # L0 向量索引：对原始消息做滑动窗口切片写入 Chroma
        # 放在 L1 摘要生成之前，确保 L0 层先就位
        # 失败只打印警告，不阻断后续的摘要生成
        try:
            from vector_store import index_l0_session
            index_l0_session(session_id)
        except Exception as e:
            logger.warning(f"L0 向量索引写入失败，session_id={session_id} — {e}")

        # L1 摘要生成（summarizer 内部会继续触发 L1 索引写入和 L2 条数检查）
        trigger_l1_summary(session_id)


# =============================================================================
# 直接运行此文件时：模拟完整 session 生命周期
# =============================================================================

if __name__ == "__main__":
    from database import (
        save_message,
        update_message_time_for_test,   # 【P1-2 修复】用公开函数替换私有函数调用
    )

    print("=== session_manager.py 验证测试 ===\n")

    sm = SessionManager()
    sm.start()
    print()

    # ------------------------------------------------------------------
    # 测试一：空闲超时自动关闭
    #
    # 【P1-2 修复说明】
    # 原来直接调用私有函数 database._get_connection()：
    #
    #   import database
    #   conn = database._get_connection()          # ← 直接访问私有函数，破坏封装
    #   conn.execute(
    #       "UPDATE messages SET created_at = ? WHERE session_id = ?",
    #       (fake_time, sid)
    #   )
    #   conn.commit()
    #   conn.close()
    #
    # 修复后，通过 database.py 新增的公开函数访问：
    #
    #   from database import update_message_time_for_test
    #   update_message_time_for_test(sid, fake_time)
    #
    # 两者效果完全一致，但后者不依赖私有实现，封装性更好。
    # ------------------------------------------------------------------
    print("--- 测试一：空闲超时自动关闭 ---")

    sid = sm.on_message()
    save_message(sid, "user", "你好，这是验证消息")
    save_message(sid, "assistant", "收到！")
    print(f"消息已写入 session {sid}")

    # 模拟空闲超时：将该 session 的所有消息时间改为 15 分钟前
    fake_time = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()

    # 【P1-2 修复】改为调用公开函数，不再直接访问 _get_connection
    update_message_time_for_test(sid, fake_time)
    print(f"已将 session {sid} 的消息时间改为 15 分钟前，模拟超时")

    sm._check_idle_timeout()
    print(f"检测后当前活跃 session：{sm.get_current_session_id()}（应为 None）")
    print()

    # ------------------------------------------------------------------
    # 测试二：新消息到来时自动新建 session
    # ------------------------------------------------------------------
    print("--- 测试二：新消息自动新建 session ---")
    sid2 = sm.on_message()
    save_message(sid2, "user", "新的一条消息")
    print(f"新消息写入 session {sid2}（应 > {sid}）")
    print()

    # ------------------------------------------------------------------
    # 测试三：手动强制关闭 session
    # ------------------------------------------------------------------
    print("--- 测试三：手动强制关闭 ---")
    sm.force_close_current_session()
    print(f"手动关闭后当前活跃 session：{sm.get_current_session_id()}（应为 None）")
    print()

    # ------------------------------------------------------------------
    # 测试四：L2 定时检查（手动触发）
    # ------------------------------------------------------------------
    print("--- 测试四：L2 定时检查 ---")
    sm._trigger_l2_check()
    print("L2 定时检查调用完毕（无未吸收 L1 时应静默返回）")
    print()

    sm.stop()
    print("\n验证完成。")
