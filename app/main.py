"""
app/main.py — FastAPI 应用入口

职责：
    · 创建 FastAPI 实例，配置 lifespan
    · 挂载所有路由 Blueprint
    · 挂载静态文件
    · 启动 uvicorn（直接运行时）
    · 提供 init_app() 函数供懒加载初始化

lifespan 仅做：
    · 验证静态资源存在
    · 日志打印启动信息

init_app() 负责完整初始化：
    · 数据库迁移（幂等）
    · BM25 索引预热
    · 知识图谱加载
    · SessionManager 启动
    · PushScheduler 启动

InitProgress 提供初始化进度追踪，前端通过轮询展示进度。

所有业务逻辑都在 routes/ 里，此文件只做组装。
"""

import asyncio
import os
import sys
import time
import threading
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

# 禁止 HuggingFace 在启动时联网检查模型更新，避免离线环境启动卡顿
os.environ["HF_HUB_OFFLINE"] = "1"

# 抑制 requests 库的 urllib3/charset_normalizer 版本警告
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")

# ── 直接执行 app/main.py 时，将项目根目录和 src/ 加入 Python 模块搜索路径 ──
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

from ramaria.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# 延迟加载 config 常量（避免 .env 缺失时模块级 import 即崩溃）
# =============================================================================
# 原代码在模块顶层 from ramaria.config import DEBUG, SERVER_HOST, SERVER_PORT
# 若 .env 不存在，ramaria.config 的模块级 os.environ.get() 会触发异常链，
# 导致 uvicorn 无法启动 → 配置向导页面永远无法显示。
# 改为通过 _get_config() 延迟获取，仅在需要时才 import ramaria.config。
_config_cache: dict[str, object] = {}

def _get_config(key: str, default: object = None) -> object:
    """延迟获取 ramaria.config 中的值，首次调用时缓存，避免 .env 缺失崩溃。"""
    global _config_cache
    if not _config_cache:
        try:
            from ramaria.config import (
                DEBUG, SERVER_HOST, SERVER_PORT,
            )
            _config_cache = {
                "DEBUG":        DEBUG,
                "SERVER_HOST":  SERVER_HOST,
                "SERVER_PORT":  SERVER_PORT,
            }
        except Exception as e:
            logger.warning(f"无法加载 ramaria.config，使用默认值: {e}")
            _config_cache = {
                "DEBUG":        False,
                "SERVER_HOST":  "127.0.0.1",
                "SERVER_PORT":  8000,
            }
    return _config_cache.get(key, default)


# =============================================================================
# 静态文件路径常量
# =============================================================================

_STATIC_DIR = _ROOT_DIR / "static"
_HTML_FILE  = _STATIC_DIR / "index.html"


# =============================================================================
# 初始化进度追踪
# =============================================================================

@dataclass
class InitProgress:
    """
    懒加载初始化的每一步进度。

    属性：
        step       — 当前步骤序号（1-based），0 表示未开始，-1 表示完成
        total      — 总步骤数
        step_name  — 人类可读的步骤名，如 "初始化数据库…"
        elapsed_ms — 当前步骤已用毫秒数
        done       — 所有步骤是否已完成
        started    — init_app() 是否已被调用
        error      — 错误信息（空字符串表示无错误）
    """
    step:       int    = 0
    total:      int    = 6
    step_name:  str    = ""
    elapsed_ms: int    = 0
    done:       bool   = False
    started:    bool   = False
    error:      str    = ""

    def to_dict(self) -> dict:
        return {
            "step":       self.step,
            "total":      self.total,
            "step_name":  self.step_name,
            "elapsed_ms": self.elapsed_ms,
            "done":       self.done,
            "started":    self.started,
            "error":      self.error,
        }


# 进程级单例（admin.py 通过路由读取，init_app() 写入）
_init_progress = InitProgress()
_init_lock     = threading.Lock()


def get_init_progress() -> InitProgress:
    """返回当前初始化进度的快照（线程安全）。"""
    with _init_lock:
        # 返回副本避免并发修改问题
        return InitProgress(
            step       = _init_progress.step,
            total      = _init_progress.total,
            step_name  = _init_progress.step_name,
            elapsed_ms = _init_progress.elapsed_ms,
            done       = _init_progress.done,
            started    = _init_progress.started,
            error      = _init_progress.error,
        )


# =============================================================================
# 应用生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器。

    startup 阶段（yield 之前）：
        仅验证静态资源存在 + 打印启动信息。
        不再执行任何重量级初始化——这些已移入 init_app()。

    shutdown 阶段（yield 之后）：
        停止 SessionManager、AccessBoostWorker、BM25 定时器等。
    """
    logger.info("=" * 60)
    logger.info("应用启动中…（轻量模式，核心模块待配置后初始化）")
    logger.info("=" * 60)

    # 保存主事件循环引用，供后台线程使用 run_coroutine_threadsafe
    from app.dependencies import set_main_event_loop
    loop = asyncio.get_running_loop()
    set_main_event_loop(loop)
    logger.debug("主事件循环引用已保存")

    # 验证静态资源存在（无前端则无法显示配置向导）
    if not _HTML_FILE.exists():
        logger.error(f"❌ 致命错误：前端文件缺失 — {_HTML_FILE}")
        raise FileNotFoundError(f"前端文件缺失：{_HTML_FILE}")

    logger.info(f"      ✓ 前端资源就绪")
    logger.info(f"      ✓ 服务就绪 — 可在浏览器打开 http://localhost:{_get_config('SERVER_PORT', 8000)}")
    logger.info("=" * 60)

    yield

    # ── shutdown：优雅停止所有后台组件 ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("关闭中…")
    logger.info("=" * 60)

    # 停止 AccessBoostWorker 和 BM25 定时器（仅在已启动时）
    try:
        from ramaria.storage.vector_store import _stop_access_worker, _stop_bm25_timer
        _stop_access_worker()
        _stop_bm25_timer()
    except Exception as e:
        logger.debug(f"停止向量存储组件时忽略异常: {e}")

    # 停止 SessionManager（内部会停止 PushScheduler）
    try:
        from app.dependencies import session_manager
        session_manager.stop()
    except Exception as e:
        logger.debug(f"停止 SessionManager 时忽略异常: {e}")

    logger.info("✓ 已停止")
    logger.info("=" * 60)


# =============================================================================
# init_app() — 完整初始化（由 POST /api/admin/init 触发）
# =============================================================================

def init_app() -> dict:
    """
    执行重量级初始化（原 lifespan startup 内容）。

    6 个步骤，每步完成后更新 _init_progress，供前端轮询展示进度。

    返回：
        {"ok": True, "elapsed_s": float} 成功
        {"ok": False, "error": str}       失败

    幂等性：
        如果 _init_progress.done == True，直接返回（不重复初始化）。
    """
    global _init_progress

    with _init_lock:
        if _init_progress.done:
            return {"ok": True, "elapsed_s": 0, "message": "已完成，跳过"}
        _init_progress.started = True
        _init_progress.step    = 0
        _init_progress.done    = False
        _init_progress.error   = ""

    total_start = time.time()
    step_start  = total_start
    steps_total = 6

    # 辅助函数：完成一步并更新进度
    def _complete_step(step_num: int, name: str) -> None:
        """标记当前步骤完成，记录下一步的名称和耗时。"""
        nonlocal step_start
        now = time.time()
        elapsed = int((now - step_start) * 1000)
        with _init_lock:
            _init_progress.step       = step_num
            _init_progress.step_name  = name
            _init_progress.elapsed_ms = elapsed
        step_start = now
        logger.info(f"      [{_init_progress.step}/{_init_progress.total}] {name} ({elapsed}ms)")

    try:
        # ── [1/6] 初始化数据库 ──────────────────────────────────────────────
        _complete_step(1, "初始化数据库…")

        from app.core.db_initializer import ensure_db_ready
        ensure_db_ready()

        # ── [2/6] 数据库迁移 ────────────────────────────────────────────────
        _complete_step(2, "数据库迁移…")

        from ramaria.storage.database import add_last_accessed_at_columns
        add_last_accessed_at_columns()

        # ── [3/6] 预热 BM25 索引 ────────────────────────────────────────────
        _complete_step(3, "预热 BM25 索引…")

        from ramaria.storage.vector_store import (
            _bm25_index,
            _start_bm25_timer,
            _start_access_worker,
        )
        _start_access_worker()
        _bm25_index.rebuild("l1")
        _bm25_index.rebuild("l2")
        _start_bm25_timer()

        # ── [4/6] 加载知识图谱 ──────────────────────────────────────────────
        _complete_step(4, "加载知识图谱…")

        try:
            from ramaria.memory.graph_builder import load_graph_to_memory
            load_graph_to_memory()
        except Exception as e:
            logger.warning(f"知识图谱加载出错（可继续运行）: {e}")

        # ── [5/6] 启动 SessionManager ───────────────────────────────────────
        _complete_step(5, "启动 SessionManager…")

        from app.dependencies import session_manager
        session_manager.start()

        # ── [6/6] 启动 PushScheduler ────────────────────────────────────────
        _complete_step(6, "启动 PushScheduler…")

        from ramaria.memory.push_scheduler import PushScheduler
        from app.dependencies import ws_broadcast, is_user_online

        _push_scheduler = PushScheduler(
            ws_broadcast_fn = ws_broadcast,
            is_online_fn    = is_user_online,
            session_id_fn   = session_manager.get_current_session_id,
        )
        _push_scheduler.start()
        session_manager._push_scheduler = _push_scheduler

        # ── 完成 ────────────────────────────────────────────────────────────
        total_elapsed = time.time() - total_start
        with _init_lock:
            _init_progress.done       = True
            _init_progress.step_name  = f"就绪 (总耗时 {total_elapsed:.2f}s)"
            _init_progress.elapsed_ms = int(total_elapsed * 1000)

        logger.info("=" * 60)
        logger.info(f"✓ 就绪 (总耗时: {total_elapsed:.2f}s)")
        logger.info(f"  访问 http://localhost:{_get_config('SERVER_PORT', 8000)}")
        logger.info("=" * 60)

        return {"ok": True, "elapsed_s": round(total_elapsed, 2)}

    except Exception as e:
        logger.error(f"初始化失败: {e}")
        logger.exception(e)
        with _init_lock:
            _init_progress.error = str(e)
        return {"ok": False, "error": str(e)}


# =============================================================================
# FastAPI 实例
# =============================================================================

app = FastAPI(
    title       = "珊瑚菌 · 个人 AI 陪伴助手",
    description = "本地运行，支持分层记忆与任务路由",
    lifespan    = lifespan,
)


# =============================================================================
# 安全响应头中间件
# =============================================================================

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """为所有响应添加安全相关的 HTTP 头。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        # 本地桌面应用：放宽脚本/样式限制以支持 WebView 和 inline 代码
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' ws: wss: http://localhost:*; "
            "img-src 'self' data:"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# 挂载静态文件目录（配置向导、加载页、聊天界面等所有前端资源）
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# =============================================================================
# 首页路由（直接定义在 main.py，不值得单独建路由文件）
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端对话页面（static/index.html），注入 API key 元标签。"""
    from fastapi import HTTPException
    if not _HTML_FILE.exists():
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"前端文件未找到：{_HTML_FILE}\n"
                "请确认 static/index.html 存在于项目根目录下。"
            ),
        )
    from app.dependencies import get_admin_api_key
    html = _HTML_FILE.read_text(encoding="utf-8")
    # 在 </head> 前注入 API key 元标签
    api_key_meta = f'<meta name="api-key" content="{get_admin_api_key()}">'
    html = html.replace("</head>", f"  {api_key_meta}\n</head>", 1)
    return HTMLResponse(content=html)


@app.get("/loading", response_class=HTMLResponse)
async def loading():
    """返回加载页面（static/loading.html），注入 API key 元标签。"""
    from fastapi import HTTPException
    from app.dependencies import get_admin_api_key

    loading_file = _STATIC_DIR / "loading.html"
    if not loading_file.exists():
        raise HTTPException(
            status_code = 503,
            detail      = f"加载页面未找到：{loading_file}",
        )
    html = loading_file.read_text(encoding="utf-8")
    api_key_meta = f'<meta name="api-key" content="{get_admin_api_key()}">'
    html = html.replace("</head>", f"  {api_key_meta}\n</head>", 1)
    return HTMLResponse(content=html)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """返回系统总览页面（static/dashboard.html），注入 API key 元标签。"""
    from fastapi import HTTPException
    from app.dependencies import get_admin_api_key

    dashboard_file = _STATIC_DIR / "dashboard.html"
    if not dashboard_file.exists():
        raise HTTPException(
            status_code = 503,
            detail      = f"页面未找到：{dashboard_file}",
        )
    html = dashboard_file.read_text(encoding="utf-8")
    api_key_meta = f'<meta name="api-key" content="{get_admin_api_key()}">'
    html = html.replace("</head>", f"  {api_key_meta}\n</head>", 1)
    return HTMLResponse(content=html)


@app.get("/memory", response_class=HTMLResponse)
async def memory():
    """返回记忆查看器页面（static/memory.html），注入 API key 元标签。"""
    from fastapi import HTTPException
    from app.dependencies import get_admin_api_key

    memory_file = _STATIC_DIR / "memory.html"
    if not memory_file.exists():
        raise HTTPException(
            status_code = 503,
            detail      = f"页面未找到：{memory_file}",
        )
    html = memory_file.read_text(encoding="utf-8")
    api_key_meta = f'<meta name="api-key" content="{get_admin_api_key()}">'
    html = html.replace("</head>", f"  {api_key_meta}\n</head>", 1)
    return HTMLResponse(content=html)


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
# 直接启动入口（开发模式：python app/main.py）
# =============================================================================

if __name__ == "__main__":
    _host = _get_config("SERVER_HOST", "127.0.0.1")
    _port = _get_config("SERVER_PORT", 8000)
    _debug = _get_config("DEBUG", False)

    uvicorn.run(
        "app.main:app",
        host   = _host,
        port   = int(_port),
        reload = bool(_debug),
    )