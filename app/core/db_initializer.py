"""
app/core/db_initializer.py — 数据库初始化模块

职责：
    · 检查数据库文件是否存在
    · 不存在时自动调用 scripts/setup_db.py 进行初始化（全新建库）
    · 存在时调用 scripts/setup_db.py 执行幂等迁移（补齐新增列/表/索引）
    · 支持命令行启动（经过 start.py）和 PyInstaller exe 直接启动两种场景

设计决策：
    · 开发模式：通过 subprocess 调用 setup_db.py，与 admin.py 的
      admin_init_db 路径统一。好处：
      1. setup_db.py 的 print 输出全部被 capture_output=True 捕获，
         不再污染启动日志。
      2. setup_db.py 内部异常不会直接崩溃主进程，而是以返回码上报。
      3. 只在关键节点输出一行 logger.info，启动日志简洁。
    · 打包模式（PyInstaller exe）：直接在进程内调用 setup_db.main()，
      因为打包后 sys.executable 指向 exe 自身而非 Python 解释器，
      无法通过 subprocess 执行 .py 文件。

调用时机：
    app/main.py 的 lifespan 函数中最优先执行

        from app.core.db_initializer import ensure_db_ready
        ensure_db_ready()
"""

import io
import os
import subprocess
import sys
from pathlib import Path

from ramaria.config import DB_PATH
from ramaria.logger import get_logger

logger = get_logger(__name__)

_IS_FROZEN = getattr(sys, "frozen", False)


def ensure_db_ready() -> None:
    """
    确保数据库已准备好（文件存在、所有表和列均存在）。

    运行模式：
        · 打包模式（frozen）：直接在进程内调用 setup_db.main()，
          避免 subprocess 无法执行 .py 文件的问题。
        · 开发模式：通过 subprocess 调用 scripts/setup_db.py，
          stdout/stderr 被完全捕获，保持启动日志简洁。

    异常：
        RuntimeError — 初始化或迁移失败时抛出，含诊断信息
    """
    db_file_exists = DB_PATH.exists()

    if not db_file_exists:
        logger.info("数据库文件不存在，正在初始化…")
    else:
        logger.info("检查数据库结构，补齐缺失迁移…")

    # 确保 data/ 目录存在（setup_db.py 需要写数据库文件）
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _IS_FROZEN:
        _run_in_process(db_file_exists)
    else:
        _run_subprocess(db_file_exists)

    # 成功：只输出一行简洁日志
    if not db_file_exists:
        logger.info("数据库初始化完成")
    else:
        logger.info("数据库迁移检查完成")


def _run_in_process(db_file_exists: bool) -> None:
    """
    打包模式下在当前进程内直接调用 setup_db.main()。

    优势：
        · 无需 Python 解释器（打包后没有独立解释器）
        · 无 subprocess 开销
        · 无编码问题（stdout 被重定向到 StringIO）

    异常：
        RuntimeError — setup_db.main() 抛出异常时转换为此异常
    """
    try:
        # 导入 setup_db 模块
        setup_db = _import_setup_db()

        # 重定向 stdout/stderr 到 StringIO，防止 print 输出污染日志
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            sys.stdout = captured_stdout  # type: ignore[assignment]
            sys.stderr = captured_stderr  # type: ignore[assignment]
            setup_db.main()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        # 获取捕获的输出用于诊断
        stdout_text = captured_stdout.getvalue()[-500:].strip() if captured_stdout else ""
        stderr_text = captured_stderr.getvalue()[-500:].strip() if captured_stderr else ""

        action = "初始化" if not db_file_exists else "迁移"
        logger.error(
            f"数据库{action}失败\n"
            f"数据库路径：{DB_PATH}\n"
            f"异常：{e}\n"
            + (f"stderr：\n{stderr_text}\n" if stderr_text else "")
            + (f"stdout 末尾：\n{stdout_text}" if stdout_text else "")
        )
        raise RuntimeError(
            f"数据库{action}失败：{e}。详情请查看 logs/coral.log。"
        ) from e


def _import_setup_db():
    """
    动态导入 setup_db 模块。

    打包模式下 setup_db.py 在 _MEIPASS/scripts/ 目录中，
    需要将其添加到 sys.path 才能导入。
    """
    root = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    scripts_dir = root / "scripts"

    # 将 scripts/ 加入路径，使得 "import setup_db" 可以工作
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # 同时确保 src/ 在路径中（setup_db 依赖 ramaria.config）
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import setup_db  # noqa: F401
    return setup_db


def _run_subprocess(db_file_exists: bool) -> None:
    """
    开发模式下通过 subprocess 调用 scripts/setup_db.py。

    设置 PYTHONIOENCODING=utf-8 防止 Windows GBK 编码错误。
    stdout/stderr 被完全捕获，不输出到控制台。
    """
    setup_script = _find_setup_db_script()

    # 构造子进程环境变量，确保 UTF-8 输出
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            # 工作目录设为项目根目录，与 setup_db.py 内部路径推断一致
            cwd=str(setup_script.parent.parent),
            # 捕获全部输出，不打印到控制台
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"找不到 Python 解释器或 setup_db.py：{e}"
        ) from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("数据库初始化超时（>60s），请检查磁盘状态")
    except Exception as e:
        raise RuntimeError(f"启动 setup_db.py 子进程失败：{e}") from e

    if result.returncode != 0:
        # 失败时才输出详情，帮助排查
        _log_subprocess_failure(result, db_file_exists)
        raise RuntimeError(
            f"数据库{'初始化' if not db_file_exists else '迁移'}失败，"
            f"退出码：{result.returncode}。"
            f"详情请查看 logs/coral.log。"
        )


def _find_setup_db_script() -> Path:
    """
    定位 scripts/setup_db.py 的绝对路径。

    仅在开发模式（非打包）下使用。

    异常：
        FileNotFoundError — 找不到 setup_db.py 时抛出
    """
    # 开发模式：app/core/db_initializer.py → app/core → app → 项目根
    root = Path(__file__).resolve().parents[2]

    script = root / "scripts" / "setup_db.py"

    if not script.exists():
        raise FileNotFoundError(
            f"setup_db.py 未找到：{script}\n"
            "可能原因：项目文件不完整，缺少 scripts/setup_db.py"
        )

    return script


def _log_subprocess_failure(result: subprocess.CompletedProcess, is_new_db: bool) -> None:
    """
    数据库操作失败时，将 setup_db.py 的输出摘要写入日志，帮助排查。

    只输出 stderr 的最后 500 字符（通常是关键错误行），
    避免大量 print 输出淹没日志文件。
    """
    action = "初始化" if is_new_db else "迁移"
    stderr_tail = (result.stderr or "")[-500:].strip()
    stdout_tail = (result.stdout or "")[-500:].strip()

    logger.error(
        f"数据库{action}失败（退出码：{result.returncode}）\n"
        f"数据库路径：{DB_PATH}\n"
        + (f"stderr：\n{stderr_tail}\n" if stderr_tail else "")
        + (f"stdout 末尾：\n{stdout_tail}" if stdout_tail else "")
    )