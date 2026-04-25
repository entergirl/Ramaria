"""
app/core/db_initializer.py — 数据库初始化模块

职责：
    · 检查数据库文件是否存在
    · 不存在时自动调用 scripts/setup_db.py 进行初始化（全新建库）
    · 存在时调用 scripts/setup_db.py 执行幂等迁移（补齐新增列/表/索引）
    · 支持命令行启动（经过 start.py）和 PyInstaller exe 直接启动两种场景

设计决策：
    v0.6.0 改为通过 subprocess 调用 setup_db.py，与 admin.py 的
    admin_init_db 路径统一。好处：
    1. setup_db.py 的 print 输出全部被 capture_output=True 捕获，
       不再污染启动日志。
    2. setup_db.py 内部异常不会直接崩溃主进程，而是以返回码上报。
    3. 只在关键节点输出一行 logger.info，启动日志简洁。

调用时机：
    app/main.py 的 lifespan 函数中最优先执行

        from app.core.db_initializer import ensure_db_ready
        ensure_db_ready()
"""

import subprocess
import sys
from pathlib import Path

from ramaria.config import DB_PATH
from ramaria.logger import get_logger

logger = get_logger(__name__)


def ensure_db_ready() -> None:
    """
    确保数据库已准备好（文件存在、所有表和列均存在）。

    内部通过 subprocess 调用 scripts/setup_db.py：
        · 数据库文件不存在 → setup_db 执行全新建库
        · 数据库文件存在   → setup_db 执行幂等迁移，补齐缺失结构

    subprocess 的 stdout/stderr 被完全捕获，不输出到控制台，
    仅在失败时通过 logger.error 输出摘要，保持启动日志简洁。

    异常：
        RuntimeError — 初始化或迁移失败时抛出，含诊断信息
    """
    db_file_exists = DB_PATH.exists()

    if not db_file_exists:
        logger.info("数据库文件不存在，正在初始化…")
    else:
        logger.info("检查数据库结构，补齐缺失迁移…")

    setup_script = _find_setup_db_script()

    # 确保 data/ 目录存在（setup_db.py 需要写数据库文件）
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            # 工作目录设为项目根目录，与 setup_db.py 内部路径推断一致
            cwd=str(setup_script.parent.parent),
            # 捕获全部输出，不打印到控制台（这是修复问题1的关键）
            capture_output=True,
            text=True,
            timeout=60,          # 60秒超时，防止卡死
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
        _log_failure(result, db_file_exists)
        raise RuntimeError(
            f"数据库{'初始化' if not db_file_exists else '迁移'}失败，"
            f"退出码：{result.returncode}。"
            f"详情请查看 logs/coral.log。"
        )

    # 成功：只输出一行简洁日志
    if not db_file_exists:
        logger.info("✅ 数据库初始化完成")
    else:
        logger.info("✅ 数据库迁移检查完成")


def _find_setup_db_script() -> Path:
    """
    定位 scripts/setup_db.py 的绝对路径。

    支持两种运行环境：
        · 开发模式：本文件在 app/core/，项目根目录是上两级
        · PyInstaller 打包模式：sys._MEIPASS 是解压后的只读资源目录

    异常：
        FileNotFoundError — 找不到 setup_db.py 时抛出
    """
    if getattr(sys, "frozen", False):
        # 打包模式：从 _MEIPASS（资源目录）查找
        root = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # 开发模式：app/core/db_initializer.py → app/core → app → 项目根
        root = Path(__file__).resolve().parents[2]

    script = root / "scripts" / "setup_db.py"

    if not script.exists():
        raise FileNotFoundError(
            f"setup_db.py 未找到：{script}\n"
            "可能原因：\n"
            "  1. 打包时 ramaria.spec 的 datas 未包含 scripts/ 目录\n"
            "  2. 项目文件不完整，缺少 scripts/setup_db.py\n"
            "解决：确认 ramaria.spec 包含 (str(ROOT / 'scripts'), 'scripts') 后重新打包"
        )

    return script


def _log_failure(result: subprocess.CompletedProcess, is_new_db: bool) -> None:
    """
    数据库操作失败时，将 setup_db.py 的输出摘要写入日志，帮助排查。

    只输出 stderr 的最后 500 字符（通常是关键错误行），
    避免大量 print 输出淹没日志文件。
    """
    action = "初始化" if is_new_db else "迁移"
    stderr_tail = (result.stderr or "")[-500:].strip()
    stdout_tail = (result.stdout or "")[-500:].strip()

    logger.error(
        f"❌ 数据库{action}失败（退出码：{result.returncode}）\n"
        f"数据库路径：{DB_PATH}\n"
        + (f"stderr：\n{stderr_tail}\n" if stderr_tail else "")
        + (f"stdout 末尾：\n{stdout_tail}" if stdout_tail else "")
    )