"""
app/bundle.py — Ramaria 桌面版主入口

双击 Ramaria.exe 时执行此文件（PyInstaller 打包入口）。
开发调试时也可直接 python app/bundle.py 运行。

启动流程：
    1. 修复打包后的模块路径（PyInstaller _MEIPASS）
    2. 加载 .env 中的环境变量到当前进程
    3. 在后台线程启动 FastAPI（uvicorn）
    4. 等待服务就绪（轮询 /api/admin/status）
    5. 判断是否需要显示配置向导：
       · 首次启动 / .env 缺失 → launcher.html → setup.html
       · 配置完整             → 直接打开 index.html
    6. 创建系统托盘图标
    7. 打开 pywebview 窗口（阻塞，直到用户关闭）

线程模型：
    主线程    — pywebview（必须在主线程运行，macOS/Windows GUI 要求）
    后台线程  — uvicorn（FastAPI 服务）
    后台线程  — pystray（系统托盘，守护线程）
"""

from __future__ import annotations

import os
import platform
import sys
import threading
import time
from pathlib import Path

# =============================================================================
# 步骤 0：路径修复（PyInstaller 打包时资源在 _MEIPASS 目录）
# =============================================================================

def _get_root() -> Path:
    """
    返回项目根目录：
      · 开发模式：本文件在 app/，根目录是上一级
      · 打包模式：_MEIPASS 就是解压后的根目录
    """
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后，sys._MEIPASS 是解压目录
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


ROOT = _get_root()

# 将 src/ 加入模块搜索路径（开发模式需要）
_src = ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =============================================================================
# 步骤 1：加载 .env 环境变量
# =============================================================================

def _load_dotenv() -> None:
    """
    简单解析 .env 文件，将 KEY=VALUE 写入 os.environ。
    不依赖 python-dotenv，使用纯标准库。
    已有同名环境变量时不覆盖（允许系统环境变量优先）。
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.split("#")[0].strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()

# =============================================================================
# 步骤 2：常量
# =============================================================================

_IS_WINDOWS   = platform.system() == "Windows"
_SERVER_PORT  = int(os.environ.get("SERVER_PORT", "8000"))
_BASE_URL     = f"http://localhost:{_SERVER_PORT}"
_ICON_PATH    = ROOT / "icon.ico"      # 托盘 / 窗口图标，可选
_TITLE        = "珊瑚菌 · Ramaria"

# 等待服务就绪的最长时间（秒）和轮询间隔
_MAX_WAIT_SECONDS  = 120   # 启动时 ML 模型加载较慢
_POLL_INTERVAL     = 0.5

# =============================================================================
# 步骤 2.5：修复 config 路径（仅打包模式）
# =============================================================================

def _patch_config_paths() -> None:
    """
    修复打包模式下 config.py 和 logger.py 中的路径变量。

    打包后这些模块的 ROOT_DIR 计算错误（指向打包内部路径），
    此函数在导入其他模块之前调用，用正确路径覆盖。
    """
    if not getattr(sys, "frozen", False):
        return  # 仅打包模式需要

    import ramaria.config as config
    import ramaria.logger as logger_module

    # 修复 config 路径
    config.ROOT_DIR = ROOT
    config.DATA_DIR = ROOT / "data"
    config.CONFIG_DIR = ROOT / "config"
    config.LOG_DIR = ROOT / "logs"
    config.DB_PATH = config.DATA_DIR / "assistant.db"
    config.CHROMA_DIR = config.DATA_DIR / "chroma_db"
    config.PERSONA_PATH = config.CONFIG_DIR / "persona.toml"

    # 修复 logger 路径
    logger_module._ROOT_DIR = ROOT
    logger_module.LOG_DIR = ROOT / "logs"
    logger_module.LOG_FILE = logger_module.LOG_DIR / "coral.log"

    # 确保目录存在
    for d in [config.DATA_DIR, config.LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 步骤 3：后台启动 FastAPI 服务
# =============================================================================

_uvicorn_thread: threading.Thread | None = None
_service_ready  = threading.Event()     # 服务就绪信号


def _start_uvicorn() -> None:
    """
    在后台线程中启动 uvicorn（运行 FastAPI app）。
    服务启动后设置 _service_ready 事件，通知主线程打开窗口。
    """
    import uvicorn

    # uvicorn 的 on_startup 钩子在 lifespan 之后触发，
    # 这里用轮询代替回调（更可靠，跨平台行为一致）
    config = uvicorn.Config(
        app       = "app.main:app",
        host      = "127.0.0.1",
        port      = _SERVER_PORT,
        log_level = "warning",     # 减少控制台噪声
    )
    server = uvicorn.Server(config)
    server.run()


def _wait_for_service() -> bool:
    """
    轮询 /api/admin/status 直到服务响应，或超时返回 False。

    返回：True = 服务就绪，False = 超时
    """
    import urllib.error
    import urllib.request

    url      = f"{_BASE_URL}/api/admin/status"
    deadline = time.time() + _MAX_WAIT_SECONDS

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(_POLL_INTERVAL)

    return False


def start_backend() -> bool:
    """
    启动后台 uvicorn 线程，等待服务就绪。

    返回：True = 就绪，False = 启动超时
    """
    global _uvicorn_thread

    _uvicorn_thread = threading.Thread(
        target = _start_uvicorn,
        name   = "UvicornServer",
        daemon = True,   # 主线程退出时自动结束
    )
    _uvicorn_thread.start()

    return _wait_for_service()


# =============================================================================
# 步骤 4：决定打开哪个页面
# =============================================================================

def _get_start_url() -> str:
    """
    根据当前配置状态决定首页 URL：
      · 配置不完整 → 配置向导（/api/admin/launcher/page）
      · 配置完整   → 聊天主界面（/）
    """
    from app.core.env_checker import can_start_directly

    ok, _ = can_start_directly()
    if ok:
        return _BASE_URL + "/"
    return _BASE_URL + "/api/admin/launcher/page"


# =============================================================================
# 步骤 5：webview 窗口管理
# =============================================================================

_webview_window = None   # pywebview window 对象，全局保存用于托盘回调


def _open_or_focus_window(url: str | None = None) -> None:
    """
    显示/恢复 webview 窗口。
    若窗口已存在则恢复到前台；若已销毁则重新创建（不常用场景）。

    参数：
        url — 打开新 URL；None 表示仅恢复焦点
    """
    global _webview_window
    import webview

    if _webview_window is not None:
        try:
            _webview_window.show()
            if url:
                _webview_window.load_url(url)
            return
        except Exception:
            _webview_window = None

    # 窗口不存在时创建新窗口
    target_url = url or (_BASE_URL + "/")
    _webview_window = webview.create_window(
        title       = _TITLE,
        url         = target_url,
        width       = 1024,
        height      = 720,
        min_size    = (800, 600),
        resizable   = True,
    )


def _open_setup_in_window() -> None:
    """从托盘菜单打开配置向导页面。"""
    _open_or_focus_window(_BASE_URL + "/api/admin/launcher/page")


# =============================================================================
# 步骤 6：重启服务
# =============================================================================

def _restart_service() -> None:
    """
    重启 uvicorn 后端服务。
    当前实现：重新启动整个进程（最简单可靠的方式）。
    """
    import subprocess
    subprocess.Popen([sys.executable] + sys.argv)
    _quit_app()


# =============================================================================
# 步骤 7：退出
# =============================================================================

def _quit_app() -> None:
    """彻底退出应用：销毁 webview 窗口，退出 Python 进程。"""
    global _webview_window
    import webview

    try:
        # destroy() 会触发 webview 的主循环退出
        webview.windows[0].destroy()
    except Exception:
        pass

    # 确保进程退出（uvicorn/tray 是守护线程，主进程结束后自动结束）
    os._exit(0)


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """
    Ramaria 桌面版启动入口。
    """
    import webview
    from ramaria.logger import get_logger

    logger = get_logger("bundle")
    logger.info(f"Ramaria 桌面版启动，根目录：{ROOT}")

    # ── 修复打包模式下的 config 路径 ──────────────────────────────────────
    # 必须在导入 app.main 之前执行，否则 app.main 导入时就用了错误路径
    _patch_config_paths()
    logger.info("路径修复完成")

    # ── 启动后台服务 ───────────────────────────────────────────────────────
    logger.info("正在启动后台服务…")
    ready = start_backend()
    if not ready:
        # 服务启动超时，打开一个错误页面告知用户
        logger.error("后台服务启动超时")
        # 仍然打开窗口，显示错误提示
        start_url = _BASE_URL + "/api/admin/launcher/page"
    else:
        logger.info("后台服务就绪")
        start_url = _get_start_url()

    # ── 创建系统托盘 ───────────────────────────────────────────────────────
    from app.system.tray import RamariaTrayx

    tray = RamariaTrayx(
        open_window_fn = lambda: _open_or_focus_window(),
        restart_fn     = _restart_service,
        quit_fn        = _quit_app,
        open_setup_fn  = _open_setup_in_window,
        exe_path       = Path(sys.executable) if getattr(sys, "frozen", False) else None,
        icon_path      = _ICON_PATH if _ICON_PATH.exists() else None,
    )
    tray.start()

    # ── 创建 webview 窗口 ──────────────────────────────────────────────────
    global _webview_window

    # 窗口关闭时只隐藏到托盘，不退出进程
    def _on_window_closed():
        pass  # 托盘图标仍在，用户可以通过托盘重新打开

    _webview_window = webview.create_window(
        title       = _TITLE,
        url         = start_url,
        width       = 1024,
        height      = 720,
        min_size    = (800, 600),
        resizable   = True,
        on_top      = False,
    )

    # 加载图标（可选）
    icon_arg = {}
    if _ICON_PATH.exists():
        icon_arg["icon"] = str(_ICON_PATH)

    logger.info(f"打开窗口：{start_url}")

    # webview.start() 阻塞主线程，直到所有窗口关闭
    # gui=None 表示使用系统默认（Windows 用 EdgeChromium/MSHTML，macOS 用 WebKit）
    webview.start(
        http_server = False,   # 不使用内置 HTTP 服务器，我们自己运行 uvicorn
        **icon_arg,
    )

    # webview 退出后整个程序退出
    logger.info("窗口已关闭，程序退出")
    os._exit(0)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    main()
