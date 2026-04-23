"""
app/core/db_initializer.py — 数据库初始化模块

职责：
    · 检查数据库文件和表是否存在
    · 不存在时自动调用 scripts/setup_db.py 进行初始化
    · 数据库存在时执行幂等迁移（补齐新增列/表/索引）
    · 支持命令行启动（经过 start.py）和 PyInstaller exe 直接启动两种场景
    · 所有操作幂等，多次调用安全

调用时机：
    app/main.py 的 lifespan 函数启动时最优先执行

        from app.core.db_initializer import ensure_db_ready
        ensure_db_ready()
"""

import importlib.util
import sys
from pathlib import Path

from ramaria.config import DB_PATH
from ramaria.logger import get_logger

logger = get_logger(__name__)


def ensure_db_ready() -> None:
    """
    确保数据库已准备好（文件存在、所有表和列均存在）。

    流程：
        · 数据库文件不存在 → 全新建库（_create_fresh_db）
        · 数据库文件存在   → 幂等迁移（_migrate_existing_db），补齐缺失结构

    异常：
        RuntimeError — 初始化或迁移失败时抛出，含诊断信息
    """
    db_file_exists = DB_PATH.exists()

    if not db_file_exists:
        logger.info("数据库文件不存在，创建新数据库…")
        try:
            _call_setup_db(is_new_db=True)
            logger.info("✅ 新数据库初始化完成")
        except FileNotFoundError as e:
            _log_error_scripts_not_found(str(e))
            raise RuntimeError(f"数据库创建失败：setup_db.py 不存在 — {e}") from e
        except Exception as e:
            _log_error_generic(str(e))
            raise RuntimeError(f"数据库创建失败：{e}") from e
    else:
        logger.info("数据库已存在，执行迁移检查（补齐新增列/表/索引）…")
        try:
            _call_setup_db(is_new_db=False)
            logger.info("✅ 数据库迁移检查完成")
        except FileNotFoundError as e:
            _log_error_scripts_not_found(str(e))
            raise RuntimeError(f"数据库迁移失败：setup_db.py 不存在 — {e}") from e
        except Exception as e:
            _log_error_generic(str(e))
            raise RuntimeError(f"数据库迁移失败：{e}") from e


def _call_setup_db(is_new_db: bool) -> None:
    """
    动态加载并调用 setup_db 模块的初始化函数。

    参数：
        is_new_db — True 表示全新建库；False 表示执行幂等迁移

    异常：
        FileNotFoundError — setup_db.py 不存在
        RuntimeError      — 模块加载或执行失败
        AttributeError    — setup_db 缺少必需函数
    """
    # 定位项目根目录
    # PyInstaller 打包后 sys._MEIPASS 是解压后的只读资源目录
    # 开发环境下通过文件路径计算
    if getattr(sys, "frozen", False):
        project_root = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # app/core/db_initializer.py → app/core → app → 项目根
        project_root = Path(__file__).resolve().parents[2]

    setup_db_path = project_root / "scripts" / "setup_db.py"

    if not setup_db_path.exists():
        raise FileNotFoundError(
            f"setup_db.py 不存在：{setup_db_path}"
        )

    # 确保 data/ 目录存在（setup_db 需要写数据库文件）
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 动态加载 setup_db 模块，避免循环导入和路径依赖
    module_key = "setup_db_dynamic"
    spec = importlib.util.spec_from_file_location(module_key, setup_db_path)

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法为 setup_db.py 创建模块规范：{setup_db_path}"
        )

    setup_db_module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = setup_db_module

    try:
        spec.loader.exec_module(setup_db_module)
    except Exception as e:
        raise RuntimeError(f"执行 setup_db 模块时异常：{e}") from e

    # 提取所需函数
    _create_fresh_db    = getattr(setup_db_module, "_create_fresh_db",    None)
    _migrate_existing_db = getattr(setup_db_module, "_migrate_existing_db", None)
    _get_conn            = getattr(setup_db_module, "_get_conn",            None)

    if not all([_create_fresh_db, _migrate_existing_db, _get_conn]):
        raise AttributeError(
            "setup_db.py 中缺少必需函数：_create_fresh_db / _migrate_existing_db / _get_conn"
        )

    conn = _get_conn()
    try:
        if is_new_db:
            logger.info("全新数据库，执行初始化建库…")
            _create_fresh_db(conn)
        else:
            logger.info("已有数据库，执行幂等迁移…")
            _migrate_existing_db(conn)
    finally:
        conn.close()


# =============================================================================
# 错误诊断日志
# =============================================================================

def _log_error_scripts_not_found(detail: str) -> None:
    logger.error(
        "❌ setup_db.py 不存在\n"
        f"详情：{detail}\n"
        "\n"
        "可能的原因：\n"
        "  1. PyInstaller 打包时没有包含 scripts/ 目录\n"
        "  2. 项目文件不完整（缺少 scripts/setup_db.py）\n"
        "\n"
        "解决方案：\n"
        "  1. 检查 ramaria.spec 的 datas 是否包含 scripts 目录\n"
        "  2. 重新打包：pyinstaller ramaria.spec --noconfirm\n"
        "  3. 命令行启动请使用 python win/start.py"
    )


def _log_error_generic(detail: str) -> None:
    logger.error(
        "❌ 数据库初始化/迁移失败\n"
        f"错误详情：{detail}\n"
        "\n"
        "常见原因及解决方案：\n"
        f"  1. 权限问题 - data/ 目录不可写\n"
        f"       → 检查目录权限，或以管理员身份运行\n"
        f"\n"
        f"  2. 磁盘空间不足\n"
        f"       → 检查磁盘可用空间\n"
        f"\n"
        f"  3. 数据库文件被锁定\n"
        f"       → 关闭所有 Ramaria 进程后重试\n"
        f"       → 删除后重建：del data\\assistant.db\n"
        f"\n"
        f"数据库路径：{DB_PATH}"
    )
