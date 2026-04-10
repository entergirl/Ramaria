"""
src/ramaria/importer/batch.py — L1 摘要批量生成模块

职责：
    对"已导入 L0 但尚未生成 L1 摘要"的历史 session 批量调用
    summarizer.generate_l1_summary()，并维护全局批处理状态，
    供前端进度面板和命令行版实时查询。

设计原则：
    1. 与实时对话不冲突：检测到活跃 session 时暂停等待
    2. 可中止、可续跑：stop_batch() 后在当前 session 完成后退出
    3. 单例运行：全局只允许一个批处理任务同时运行
    4. 线程安全：BatchState 的所有字段通过 threading.Lock 保护

对外暴露的接口：
    start_batch(session_manager)
    stop_batch()
    get_status()
    get_pending_count()
    run_batch_cli(session_manager, progress_callback)

"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 批处理状态数据结构
# =============================================================================

@dataclass
class BatchState:
    """
    全局批处理状态，由后台线程写入，主线程/前端读取。
    所有字段通过外部 Lock 保护，不要在 Lock 之外修改。

    状态机：
        idle → running → done
                      → stopped（用户主动停止）
                      → error（异常中止）
                      → waiting（检测到活跃对话，暂停等待）
    """
    status: str = "idle"

    total:     int = 0
    done:      int = 0
    succeeded: int = 0
    failed:    int = 0
    skipped:   int = 0

    current_session_id:   Optional[int] = None
    current_session_date: str = ""

    started_at:  Optional[str] = None
    finished_at: Optional[str] = None

    failed_session_ids: list = field(default_factory=list)
    stop_requested:     bool = False

    def eta_seconds(self) -> Optional[int]:
        """估算剩余时间（秒）。数据不足时返回 None。"""
        if self.done == 0 or not self.started_at:
            return None
        try:
            start = datetime.fromisoformat(self.started_at)
            now   = datetime.now(timezone.utc)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            elapsed         = (now - start).total_seconds()
            avg_per_session = elapsed / self.done
            remaining       = self.total - self.done
            return int(avg_per_session * remaining)
        except Exception:
            return None

    def to_dict(self) -> dict:
        """序列化为字典，供 FastAPI 接口和命令行版使用。"""
        return {
            "status":               self.status,
            "total":                self.total,
            "done":                 self.done,
            "succeeded":            self.succeeded,
            "failed":               self.failed,
            "skipped":              self.skipped,
            "current_session_id":   self.current_session_id,
            "current_session_date": self.current_session_date,
            "started_at":           self.started_at,
            "finished_at":          self.finished_at,
            "eta_seconds":          self.eta_seconds(),
            "failed_session_ids":   self.failed_session_ids[:20],
        }


# =============================================================================
# 全局单例
# =============================================================================

_state      = BatchState()
_state_lock = threading.Lock()
_batch_thread: Optional[threading.Thread] = None


# =============================================================================
# 内部工具函数
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_pending_sessions() -> list:
    """
    查询所有"已关闭但尚未生成 L1"的 session，按时间升序排列。
    返回：list[dict]，每个元素包含 id 和 started_at
    """
    from ramaria.storage.database import get_sessions_without_l1
    return get_sessions_without_l1()


def _is_active_session(session_manager) -> bool:
    """检查当前是否有活跃的实时对话 session。"""
    if session_manager is None:
        return False
    try:
        return session_manager.get_current_session_id() is not None
    except Exception:
        return False


def _process_one_session(session_id: int) -> str:
    """
    对单个 session 调用 generate_l1_summary()。

    返回：
        "success" — L1 生成成功
        "failed"  — 模型调用失败或异常
        "skipped" — session 没有消息
    """
    try:
        from ramaria.memory.summarizer import generate_l1_summary
        result = generate_l1_summary(session_id)

        if result is None:
            from ramaria.storage.database import get_messages
            msgs = get_messages(session_id)
            if not msgs:
                return "skipped"
            else:
                return "failed"
        else:
            return "success"

    except Exception as e:
        logger.error(f"处理 session {session_id} 时异常 — {e}")
        return "failed"


# =============================================================================
# 后台批处理线程
# =============================================================================

def _batch_worker(pending_sessions: list, session_manager):
    """
    后台批处理线程的主函数。
    逐个处理 pending_sessions，每处理一个前检查实时对话冲突和停止标志。
    """
    global _state

    logger.info(f"批处理线程启动，共 {len(pending_sessions)} 个 session 待处理")

    for session_info in pending_sessions:
        session_id   = session_info["id"]
        session_date = session_info.get("started_at", "")[:10]

        with _state_lock:
            if _state.stop_requested:
                _state.status      = "stopped"
                _state.finished_at = _now_iso()
                logger.info("批处理已停止（用户请求）")
                return

        wait_logged = False
        while _is_active_session(session_manager):
            with _state_lock:
                if _state.stop_requested:
                    _state.status      = "stopped"
                    _state.finished_at = _now_iso()
                    return
                _state.status = "waiting"

            if not wait_logged:
                logger.info("检测到活跃对话，批处理暂停等待…")
                wait_logged = True

            time.sleep(5)

        with _state_lock:
            _state.status               = "running"
            _state.current_session_id   = session_id
            _state.current_session_date = session_date

        logger.info(f"处理 session {session_id}（{session_date}）")

        outcome = _process_one_session(session_id)

        with _state_lock:
            _state.done += 1
            if outcome == "success":
                _state.succeeded += 1
            elif outcome == "failed":
                _state.failed += 1
                _state.failed_session_ids.append(session_id)
            else:
                _state.skipped += 1

        logger.info(
            f"session {session_id} → {outcome}  "
            f"（{_state.done}/{_state.total}）"
        )

    with _state_lock:
        _state.status               = "done"
        _state.finished_at          = _now_iso()
        _state.current_session_id   = None
        _state.current_session_date = ""

    logger.info(
        f"批处理完成：成功 {_state.succeeded}，"
        f"失败 {_state.failed}，跳过 {_state.skipped}"
    )


# =============================================================================
# 对外接口：后台线程版（供 FastAPI 使用）
# =============================================================================

def start_batch(session_manager=None) -> dict:
    """
    启动后台批处理线程，非阻塞返回。
    如果已有批处理在运行，直接返回当前状态，不重复启动。

    参数：
        session_manager — SessionManager 实例，用于实时对话冲突检测。
                          传 None 时跳过冲突检测。
    """
    global _state, _batch_thread

    with _state_lock:
        if _state.status in ("running", "waiting"):
            return _state.to_dict()

        try:
            pending = _get_pending_sessions()
        except Exception as e:
            logger.error(f"查询待处理 session 失败 — {e}")
            _state.status = "error"
            return _state.to_dict()

        if not pending:
            _state.status = "done"
            _state.total  = 0
            return _state.to_dict()

        _state = BatchState(
            status     = "running",
            total      = len(pending),
            started_at = _now_iso(),
        )

    _batch_thread = threading.Thread(
        target  = _batch_worker,
        args    = (pending, session_manager),
        name    = "L1BatchWorker",
        daemon  = True,
    )
    _batch_thread.start()
    logger.info(f"批处理线程已启动，待处理 session 数：{len(pending)}")

    with _state_lock:
        return _state.to_dict()


def stop_batch() -> dict:
    """
    请求停止批处理。
    不立即中止，而是设置停止标志，后台线程在当前 session 处理完后退出。
    """
    global _state

    with _state_lock:
        if _state.status not in ("running", "waiting"):
            return _state.to_dict()
        _state.stop_requested = True

    logger.info("已发送停止请求，等待当前 session 处理完毕")

    with _state_lock:
        return _state.to_dict()


def get_status() -> dict:
    """查询当前批处理状态，供前端进度面板轮询。"""
    with _state_lock:
        return _state.to_dict()


def get_pending_count() -> int:
    """
    查询当前待处理 session 数量，不启动批处理。
    供前端页面初始化时显示"N 个 session 待处理"。

    返回：int；查询失败时返回 -1
    """
    try:
        pending = _get_pending_sessions()
        return len(pending)
    except Exception as e:
        logger.error(f"查询待处理数量失败 — {e}")
        return -1


# =============================================================================
# 对外接口：同步版（供命令行使用）
# =============================================================================

def run_batch_cli(
    session_manager=None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    同步执行批处理，阻塞直到全部完成。命令行版使用此函数。

    参数：
        session_manager   — SessionManager 实例；命令行独立运行时传 None
        progress_callback — 每完成一个 session 时调用的回调，
                            接受参数 (done, total, outcome, session_date)
    """
    try:
        pending = _get_pending_sessions()
    except Exception as e:
        logger.error(f"查询待处理 session 失败 — {e}")
        return {"status": "error", "error": str(e)}

    if not pending:
        print("没有待处理的 session（所有历史 session 已生成摘要）")
        return {"status": "done", "total": 0}

    total     = len(pending)
    succeeded = 0
    failed    = 0
    skipped   = 0
    failed_ids = []

    print(f"\n开始批量生成 L1 摘要，共 {total} 个 session\n")

    for idx, session_info in enumerate(pending, start=1):
        session_id   = session_info["id"]
        session_date = session_info.get("started_at", "")[:10]

        wait_shown = False
        while _is_active_session(session_manager):
            if not wait_shown:
                print(f"\n⏸  检测到活跃对话，暂停等待…（按 Ctrl+C 退出）")
                wait_shown = True
            time.sleep(5)

        outcome = _process_one_session(session_id)

        if outcome == "success":
            succeeded += 1
            mark = "✓"
        elif outcome == "failed":
            failed += 1
            failed_ids.append(session_id)
            mark = "✗"
        else:
            skipped += 1
            mark = "–"

        print(f"[{idx:>4}/{total}] {mark}  session {session_id}（{session_date}）")

        if progress_callback:
            progress_callback(idx, total, outcome, session_date)

    print()
    print("=" * 50)
    print(f"  批处理完成")
    print(f"  成功生成 L1 : {succeeded} 个")
    print(f"  失败        : {failed} 个")
    print(f"  跳过（无消息）: {skipped} 个")
    if failed_ids:
        print(f"  失败的 session_id: {failed_ids}")
    print("=" * 50)

    return {
        "status":    "done",
        "total":     total,
        "succeeded": succeeded,
        "failed":    failed,
        "skipped":   skipped,
        "failed_session_ids": failed_ids,
    }