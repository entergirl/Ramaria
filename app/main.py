"""
app/main.py — FastAPI 应用入口

职责：
    · 创建 FastAPI 实例，配置 lifespan
    · 挂载所有路由 Blueprint
    · 挂载静态文件
    · 启动 uvicorn（直接运行时）

lifespan 负责：
    · 数据库迁移（幂等）
    · BM25 索引预热
    · 知识图谱加载
    · SessionManager 启动
    · PushScheduler 启动

所有业务逻辑都在 routes/ 里，此文件只做组装。
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 禁止 HuggingFace 在启动时联网检查模型更新，避免离线环境启动卡顿
os.environ["HF_HUB_OFFLINE"] = "1"

# 直接执行 app/main.py 时，将项目根目录和 src/ 加入 Python 模块搜索路径
_ROOT_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR  = _ROOT_DIR / "src"

# 将项目根目录加入 sys.path，以便能通过 `from app.xxx` 导入 app 目录下的模块
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ramaria.config import DEBUG, SERVER_HOST, SERVER_PORT
from ramaria.logger import get_logger

logger = get_logger(__name__)

# 静态文件目录（相对于项目根目录）
_ROOT_DIR   = Path(__file__).parent.parent
_STATIC_DIR = _ROOT_DIR / "static"
_HTML_FILE  = _STATIC_DIR / "index.html"


# =============================================================================
# 应用生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器。

    startup 阶段（yield 之前）：
        1. 数据库迁移（幂等，确保 last_accessed_at 列存在）
        2. 启动后台访问回写线程（AccessBoostWorker）
        3. BM25 索引预热（从 SQLite 全量读取，通常 < 1秒）
        4. 图谱加载到 NetworkX 内存图
        5. SessionManager.start()（含空闲检测和 L2 定时检查两个后台线程）
        6. PushScheduler.start()（主动推送调度器）

    shutdown 阶段（yield 之后）：
        1. 停止 AccessBoostWorker
        2. SessionManager.stop()（内部会停止 PushScheduler）
    """
    import time
    
    logger.info("=" * 60)
    logger.info("应用启动中…")
    logger.info("=" * 60)
    
    startup_start = time.time()

    # 步骤 -1：验证关键资源
    logger.info("[0/6] 验证应用资源…")
    if not _HTML_FILE.exists():
        logger.error(f"❌ 致命错误：前端文件缺失")
        logger.error(f"   路径：{_HTML_FILE}")
        logger.error(f"   请确认 static/ 目录完整，或重新拉取代码")
        raise FileNotFoundError(f"前端文件缺失：{_HTML_FILE}")
    else:
        logger.info("      ✓ 前端资源就绪")

    # 步骤 0：确保数据库已初始化（PyInstaller exe 首次运行时自动初始化）
    logger.info("[1/6] 检查数据库…")
    from app.core.db_initializer import ensure_db_ready
    ensure_db_ready()

    # 步骤1：数据库迁移
    logger.info("[2/6] 数据库迁移…")
    from ramaria.storage.database import add_last_accessed_at_columns
    add_last_accessed_at_columns()
    logger.info("      ✓ 数据库迁移完成")

    # 步骤2：启动访问回写线程
    logger.info("[3/6] 启动访问回写线程…")
    from ramaria.storage.vector_store import _start_access_worker
    _start_access_worker()
    logger.info("      ✓ 访问回写线程就绪")

    # 步骤3：BM25 索引预热
    logger.info("[4/6] 预热 BM25 索引…")
    from ramaria.storage.vector_store import _bm25_index, _start_bm25_timer
    t0 = time.time()
    _bm25_index.rebuild("l1")
    _bm25_index.rebuild("l2")
    logger.info(f"      ✓ BM25 索引预热完成 ({time.time() - t0:.2f}s)")

    # 步骤3.5：启动 BM25 后台定时重建线程（预热之后启动，避免启动时重复重建）
    _start_bm25_timer()

    # 步骤4：图谱加载
    logger.info("[5/6] 加载知识图谱…")
    try:
        t0 = time.time()
        from ramaria.memory.graph_builder import load_graph_to_memory
        load_graph_to_memory()
        logger.info(f"      ✓ 知识图谱加载完成 ({time.time() - t0:.2f}s)")
    except Exception as e:
        logger.warning(f"      ⚠ 知识图谱加载出错（可继续运行）: {e}")

    # 步骤5：SessionManager 启动
    logger.info("[6/6] 启动 SessionManager…")
    from app.dependencies import session_manager
    session_manager.start()
    logger.info("      ✓ SessionManager 启动完成")

    # 步骤6：PushScheduler 启动
    logger.info("      启动 PushScheduler…")
    from ramaria.memory.push_scheduler import PushScheduler
    from app.dependencies import ws_broadcast, is_user_online

    _push_scheduler = PushScheduler(
        ws_broadcast_fn = ws_broadcast,
        is_online_fn    = is_user_online,
        session_id_fn   = session_manager.get_current_session_id,
    )
    _push_scheduler.start()

    # 将 scheduler 引用存入 session_manager，
    # 方便 stop() 时统一停止
    session_manager._push_scheduler = _push_scheduler
    logger.info("      ✓ PushScheduler 启动完成")

    total_time = time.time() - startup_start
    logger.info("=" * 60)
    logger.info(f"✓ 就绪 (总耗时: {total_time:.2f}s)")
    logger.info(f"  访问 http://localhost:{SERVER_PORT}")
    logger.info("=" * 60)
    yield

    # ── shutdown ──
    logger.info("=" * 60)
    logger.info("关闭中…")
    logger.info("=" * 60)
    from ramaria.storage.vector_store import _stop_access_worker, _stop_bm25_timer
    _stop_access_worker()
    _stop_bm25_timer() 
    session_manager.stop()
    logger.info("✓ 已停止")
    logger.info("=" * 60)


# =============================================================================
# FastAPI 实例
# =============================================================================

app = FastAPI(
    title       = "珊瑚菌 · 个人 AI 陪伴助手",
    description = "本地运行，支持分层记忆与任务路由",
    lifespan    = lifespan,
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# =============================================================================
# 首页路由（直接定义在 main.py，不值得单独建路由文件）
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端对话页面（static/index.html）。"""
    from fastapi import HTTPException
    if not _HTML_FILE.exists():
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"前端文件未找到：{_HTML_FILE}\n"
                "请确认 static/index.html 存在于项目根目录下。"
            ),
        )
    return HTMLResponse(content=_HTML_FILE.read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """返回系统总览页面（static/dashboard.html）。"""
    from fastapi import HTTPException
    dashboard_file = _STATIC_DIR / "dashboard.html"
    if not dashboard_file.exists():
        raise HTTPException(
            status_code = 503,
            detail      = f"页面未找到：{dashboard_file}",
        )
    return HTMLResponse(content=dashboard_file.read_text(encoding="utf-8"))


# =============================================================================
# 注册路由模块
# =============================================================================

from app.routes.admin       import router as admin_router
from app.routes.chat        import router as chat_router
from app.routes.router_ctrl import router as router_ctrl_router
from app.routes.import_ctrl import router as import_ctrl_router
from app.routes.sessions    import router as sessions_router
from app.routes.settings    import router as settings_router
from app.routes.graph_ctrl  import router as graph_ctrl_router
from app.routes.memory      import router as memory_router

app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(router_ctrl_router)
app.include_router(import_ctrl_router)
app.include_router(sessions_router)
app.include_router(settings_router)
app.include_router(graph_ctrl_router)
app.include_router(memory_router)


# =============================================================================
# 启动
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host   = SERVER_HOST,
        port   = SERVER_PORT,
        reload = DEBUG,
    )