"""
app/core/db_initializer.py — 数据库初始化模块

职责：
    · 检查数据库文件和表是否存在
    · 不存在时自动调用 scripts/setup_db.py 进行初始化
    · 支持命令行启动（经过 start.py）和 PyInstaller exe 直接启动两种场景
    · 所有操作幂等，多次调用安全

调用时机：
    app/main.py 的 lifespan 函数启动时
    从 app.core.db_initializer import ensure_db_ready
    ensure_db_ready()  # 应该是 lifespan 中最优先执行的操作

设计原理：
    - win/install.py 和 win/start.py 已集中在 scripts/ 处理初始化
    - 此模块为 exe 直接启动的应急方案
    - 集中在 app/core/ 符合现有架构（与 env_checker.py 平行）
    - 职责清晰：检查 → 初始化 → 报错诊断
"""

import importlib.util
import sys
from pathlib import Path

from ramaria.config import DB_PATH
from ramaria.logger import get_logger

logger = get_logger(__name__)


def ensure_db_ready() -> None:
    """
    确保数据库已准备好（文件存在、必需表存在）。
    
    如果检测到数据库或表不存在，自动调用 setup_db 进行初始化。
    所有操作幂等：如果数据库和表都已存在，直接返回。
    
    异常处理：
    - 初始化失败时抛出 RuntimeError，并给出详细诊断信息
    - 错误信息包括常见原因和解决方案
    
    调用时机（app/main.py 的 lifespan）：
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info("应用启动中…")
            
            # 第一步：确保数据库准备好
            from app.core.db_initializer import ensure_db_ready
            ensure_db_ready()
            
            # 后续步骤...
    """
    # ─────────────────────────────────────────────────────────────────────────
    # 第一阶段：检查数据库和表是否存在
    # ─────────────────────────────────────────────────────────────────────────
    
    db_file_exists = DB_PATH.exists()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 阶段1：数据库不存在时，创建新库
    # ─────────────────────────────────────────────────────────────────────────
    
    if not db_file_exists:
        logger.info("数据库文件不存在，创建新数据库...")
        try:
            _call_setup_db(is_new_db=True)
            logger.info("✅ 新数据库初始化完成")
            return
        except Exception as e:
            _log_error_generic(str(e))
            raise RuntimeError(f"数据库创建失败：{e}") from e
    
    # ─────────────────────────────────────────────────────────────────────────
    # 阶段2：数据库存在，执行幂等迁移（检查并补齐缺失的列/表/索引）
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info("数据库已存在，执行迁移检查（补齐新增列/表/索引）...")
    
    try:
        _call_setup_db(is_new_db=False)
        logger.info("✅ 数据库迁移检查完成")
    
    except FileNotFoundError as e:
        _log_error_scripts_not_found(str(e))
        raise RuntimeError("数据库迁移失败：setup_db.py 不存在") from e
    
    except Exception as e:
        _log_error_generic(str(e))
        raise RuntimeError(f"数据库迁移失败：{e}") from e
        logger.info("✅ 数据库初始化完成")
    
    except FileNotFoundError as e:
        _log_error_scripts_not_found(str(e))
        raise RuntimeError("数据库初始化失败：setup_db.py 不存在") from e
    
    except Exception as e:
        _log_error_generic(str(e))
        raise RuntimeError(f"数据库初始化失败：{e}") from e


def _check_table_exists() -> bool:
    """
    检查 memory_l1 表是否存在。
    
    返回：
        True  - 表存在，数据库结构完整
        False - 表不存在（数据库为空或表被意外删除）
    """
    try:
        import sqlite3
        
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_l1'"
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()
    
    except Exception as e:
        logger.error(f"检查表时异常：{e}")
        return False


def _call_setup_db(is_new_db: bool) -> None:
    """
    动态导入并调用 setup_db 模块的初始化函数。
    
    参数：
        is_new_db - True 表示数据库文件不存在（全新建库）
                   False 表示数据库存在但表缺失（迁移）
    
    异常处理：
        FileNotFoundError   - setup_db.py 或相关模块不存在
        RuntimeError        - 无法加载或执行 setup_db
        AttributeError      - setup_db 中缺少必需函数
    """
    # 获取项目根目录
    # PyInstaller 打包环境中，sys._MEIPASS 是解压后的根目录
    # 命令行环境中，通过相对路径计算根目录
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后
        project_root = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # 开发环境：app/core/db_initializer.py → app/core → app → . (项目根)
        project_root = Path(__file__).resolve().parents[2]
    
    setup_db_path = project_root / "scripts" / "setup_db.py"
    
    if not setup_db_path.exists():
        raise FileNotFoundError(
            f"setup_db.py 不存在：{setup_db_path}"
        )
    
    # 确保 data/ 目录存在（setup_db 需要写数据库文件）
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用 importlib 动态加载 setup_db 模块
    # 这样避免了硬依赖，setup_db 可能因路径问题找不到
    spec = importlib.util.spec_from_file_location("setup_db_dynamic", setup_db_path)
    
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法为 setup_db.py 创建模块规范：{setup_db_path}"
        )
    
    setup_db_module = importlib.util.module_from_spec(spec)
    sys.modules["setup_db_dynamic"] = setup_db_module
    
    try:
        spec.loader.exec_module(setup_db_module)
    except Exception as e:
        raise RuntimeError(f"执行 setup_db 模块时异常：{e}") from e
    
    # 获取初始化函数
    _create_fresh_db = getattr(setup_db_module, "_create_fresh_db", None)
    _migrate_existing_db = getattr(setup_db_module, "_migrate_existing_db", None)
    _get_conn = getattr(setup_db_module, "_get_conn", None)
    
    if not all([_create_fresh_db, _migrate_existing_db, _get_conn]):
        raise AttributeError(
            "setup_db.py 中缺少必需函数：_create_fresh_db / _migrate_existing_db / _get_conn"
        )
    
    # 调用相应的初始化函数
    conn = _get_conn()
    try:
        if is_new_db:
            logger.info("全新数据库，执行初始化建库...")
            _create_fresh_db(conn)
        else:
            logger.info("已有数据库，执行迁移...")
            _migrate_existing_db(conn)
    finally:
        conn.close()


def _log_error_scripts_not_found(detail: str) -> None:
    """
    输出 scripts/setup_db.py 不存在的错误诊断信息。
    """
    logger.error(
        "❌ setup_db.py 不存在\n"
        f"详情：{detail}\n"
        "\n"
        "可能的原因：\n"
        "  1. PyInstaller 打包时没有包含 scripts/ 目录\n"
        "  2. 项目文件不完整（缺少 scripts/setup_db.py）\n"
        "  3. 运行路径异常\n"
        "\n"
        "解决方案：\n"
        "  1. 检查项目结构，确保 scripts/setup_db.py 存在\n"
        "  2. 检查 ramaria.spec 的 datas 列表是否包含 scripts 目录：\n"
        "       datas = [(str(ROOT / 'scripts'), 'scripts'), ...]\n"
        "  3. 重新打包：pyinstaller ramaria.spec --noconfirm\n"
        "  4. 如果使用命令行启动，请使用 python win/start.py"
    )


def _log_error_generic(detail: str) -> None:
    """
    输出通用的数据库初始化错误诊断信息。
    """
    logger.error(
        "❌ 数据库初始化失败\n"
        f"错误详情：{detail}\n"
        "\n"
        "常见原因及解决方案：\n"
        f"  1. 权限问题 - data/ 目录不可写\n"
        f"       → 检查目录权限：chmod 755 data/\n"
        f"       → 删除后重试：rm -rf data/\n"
        f"\n"
        f"  2. 磁盘空间不足\n"
        f"       → 检查磁盘使用：df -h /\n"
        f"\n"
        f"  3. setup_db 依赖缺失\n"
        f"       → 确保 Python 3.10+ 已安装\n"
        f"       → 手动运行：python scripts/setup_db.py\n"
        f"\n"
        f"  4. 数据库文件被锁定\n"
        f"       → 关闭所有 ramaria 进程\n"
        f"       → 删除数据库：rm data/assistant.db\n"
        f"       → 重新运行\n"
        f"\n"
        f"数据库路径：{DB_PATH}\n"
        f"如果问题持续，请提交 issue"
    )
