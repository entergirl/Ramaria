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

# 直接执行 app/main.py 时，将 src/ 加入 Python 模块搜索路径
_ROOT_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ramaria.config import DEBUG, SERVER_HOST, SERVER_PORT
from logger import get_logger

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
    logger.info("应用启动中…")

    # 步骤1：数据库迁移
    from ramaria.storage.database import add_last_accessed_at_columns
    add_last_accessed_at_columns()

    # 步骤2：启动访问回写线程
    from ramaria.storage.vector_store import _start_access_worker
    _start_access_worker()

    # 步骤3：BM25 索引预热
    # 必须在 SessionManager.start() 之前完成，
    # 确保第一条消息到来时 BM25 已就绪
    from ramaria.storage.vector_store import _bm25_index
    _bm25_index.rebuild("l1")
    _bm25_index.rebuild("l2")
    logger.info("BM25 索引预热完成")

    # 步骤4：图谱加载
    from ramaria.memory.graph_builder import load_graph_to_memory
    load_graph_to_memory()

    # 步骤5：SessionManager 启动
    from app.dependencies import session_manager
    session_manager.start()

    # 步骤6：PushScheduler 启动
    # 注入三个运行时函数：广播、在线判断、获取当前 session_id
    # 这三个函数都定义在 app/dependencies.py 或 app/routes/chat.py
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

    logger.info(f"就绪，访问 http://localhost:{SERVER_PORT}")
    yield

    # ── shutdown ──
    logger.info("关闭中…")
    from ramaria.storage.vector_store import _stop_access_worker
    _stop_access_worker()
    session_manager.stop()   # 内部调用 _push_scheduler.stop()
    logger.info("已停止")


# =============================================================================
# FastAPI 实例
# =============================================================================

app = FastAPI(
    title       = "珊瑚菌 · 个人 AI 陪伴助手",
    description = "本地运行，支持分层记忆与任务路由",
    version     = "0.3.6",
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


# =============================================================================
# 注册路由模块
# =============================================================================

from app.routes.chat        import router as chat_router
from app.routes.router_ctrl import router as router_ctrl_router
from app.routes.import_ctrl import router as import_ctrl_router
from app.routes.sessions    import router as sessions_router
from app.routes.settings    import router as settings_router
from app.routes.graph_ctrl  import router as graph_ctrl_router
from app.routes.memory      import router as memory_router

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