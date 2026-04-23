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
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


ROOT = _get_root()

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

    加载顺序（打包模式）：
    1. exe 同级目录的 .env（用户自定义，优先）
    2. 资源目录中的 .env（打包时的模板，作为兜底）
    """
    env_file = None
    exe_dir  = None

    if getattr(sys, "frozen", False):
        exe_dir  = Path(sys.executable).parent
        user_env = exe_dir / ".env"
        if user_env.exists():
            env_file = user_env
            print(f"[bundle] 加载用户 .env: {env_file}")
        else:
            bundled_env = ROOT / ".env"
            if bundled_env.exists():
                env_file = bundled_env
                print(f"[bundle] 加载 bundled .env: {env_file}")
    else:
        env_file = ROOT / ".env"
        print(f"[bundle] 加载开发 .env: {env_file}")

    if not env_file or not env_file.exists():
        print(f"[bundle] 警告: 未找到 .env 文件")
        return

    # 打包模式下首次运行：把模板 .env 复制到 exe 同级目录供用户编辑
    if getattr(sys, "frozen", False) and exe_dir:
        user_env = exe_dir / ".env"
        if not user_env.exists():
            import shutil
            try:
                shutil.copy(env_file, user_env)
                print(f"[bundle] 已复制 .env 到: {user_env}")
            except Exception as e:
                print(f"[bundle] 复制 .env 失败: {e}")

    loaded_keys = []
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            # 去除行尾注释（# 前面有空格才截断，避免误伤 URL 中的 # 字符）
            if " #" in v:
                v = v.split(" #")[0].strip()
            v = v.strip().strip("'\"")
            if k:
                if k not in os.environ:
                    os.environ[k] = v
                    loaded_keys.append(k)
                else:
                    loaded_keys.append(f"{k}(skipped)")
    print(f"[bundle] 加载的环境变量: {loaded_keys}")


_load_dotenv()

# =============================================================================
# 步骤 2：常量
# =============================================================================

_IS_WINDOWS   = platform.system() == "Windows"
_SERVER_PORT  = int(os.environ.get("SERVER_PORT", "8000"))
_BASE_URL     = f"http://localhost:{_SERVER_PORT}"
_ICON_PATH    = ROOT / "icon.ico"
_TITLE        = "珊瑚菌 · Ramaria"

_MAX_WAIT_SECONDS  = 120
_POLL_INTERVAL     = 0.5

# =============================================================================
# 步骤 2.5：修复 config 路径（仅打包模式）
# =============================================================================

def _patch_config_paths() -> None:
    """
    修复打包模式下 config.py 和 logger.py 中的路径变量。

    打包后这些模块的 ROOT_DIR 计算错误（指向只读的 _internal/），
    此函数在导入其他模块之前调用，用正确路径（exe 同级可写目录）覆盖。

    同时处理配置文件的首次初始化：
      · .env      — 从打包资源目录复制到 exe 同级（_load_dotenv 已处理）
      · config/   — 整个目录复制（仅当目录不存在时）
      · persona.toml — 只在不存在时从 example 复制，不覆盖用户已有配置
    """
    if not getattr(sys, "frozen", False):
        return  # 仅打包模式需要

    import ramaria.config as config
    import ramaria.logger as logger_module

    # exe 同级目录（可写），用于存放用户数据
    exe_dir = Path(sys.executable).parent

    # 修复 config 路径（用户数据放在 exe 同级目录，不放只读的 _internal/）
    config.ROOT_DIR    = ROOT
    config.DATA_DIR    = exe_dir / "data"
    config.CONFIG_DIR  = exe_dir / "config"
    config.LOG_DIR     = exe_dir / "logs"
    config.DB_PATH     = config.DATA_DIR    / "assistant.db"
    config.CHROMA_DIR  = config.DATA_DIR    / "chroma_db"
    config.PERSONA_PATH = config.CONFIG_DIR / "persona.toml"

    # 修复 logger 路径
    logger_module._ROOT_DIR = exe_dir
    logger_module.LOG_DIR   = exe_dir / "logs"
    logger_module.LOG_FILE  = logger_module.LOG_DIR / "coral.log"

    # 确保必要目录存在
    for d in [config.DATA_DIR, config.CONFIG_DIR, config.LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 首次运行：从打包资源复制 config/ 目录 ────────────────────────────
    bundled_config = ROOT / "config"
    if bundled_config.exists() and not config.CONFIG_DIR.exists():
        import shutil
        try:
            shutil.copytree(bundled_config, config.CONFIG_DIR)
            print(f"[bundle] config/ 目录已复制到: {config.CONFIG_DIR}")
        except Exception as e:
            print(f"[bundle] 复制 config/ 目录失败: {e}")

    # ── persona.toml：只在不存在时从 example 复制，不覆盖用户已有配置 ────
    persona_path   = config.CONFIG_DIR / "persona.toml"
    persona_example = config.CONFIG_DIR / "persona.toml.example"

    # 也检查打包资源中的 example（config/ 目录可能刚刚复制过来）
    bundled_example = bundled_config / "persona.toml.example"

    if not persona_path.exists():
        # 优先用 exe 同级 config/ 中的 example
        src_example = None
        if persona_example.exists():
            src_example = persona_example
        elif bundled_example.exists():
            src_example = bundled_example

        if src_example:
            import shutil
            try:
                shutil.copy(src_example, persona_path)
                print(f"[bundle] persona.toml 已从 example 复制: {persona_path}")
            except Exception as e:
                print(f"[bundle] 复制 persona.toml.example 失败: {e}")
        else:
            print(
                "[bundle] 警告: persona.toml 和 persona.toml.example 均不存在，"
                "服务启动时可能因人格配置缺失而报错。"
                "请手动创建 config/persona.toml 或参考 README 配置。"
            )
    else:
        print(f"[bundle] persona.toml 已存在，跳过自动创建: {persona_path}")


# =============================================================================
# 步骤 3：后台启动 FastAPI 服务
# =============================================================================

_uvicorn_thread: threading.Thread | None = None


def _start_uvicorn() -> None:
    """在后台线程中启动 uvicorn（运行 FastAPI app）。"""
    import uvicorn

    config = uvicorn.Config(
        app       = "app.main:app",
        host      = "127.0.0.1",
        port      = _SERVER_PORT,
        log_level = "warning",
    )
    server = uvicorn.Server(config)
    server.run()


def _wait_for_service() -> bool:
    """轮询 /api/admin/status 直到服务响应，或超时返回 False。"""
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
    """启动后台 uvicorn 线程，等待服务就绪。返回 True = 就绪，False = 超时。"""
    global _uvicorn_thread

    _uvicorn_thread = threading.Thread(
        target = _start_uvicorn,
        name   = "UvicornServer",
        daemon = True,
    )
    _uvicorn_thread.start()

    return _wait_for_service()


# =============================================================================
# 步骤 4：决定打开哪个页面
# =============================================================================

def _get_start_url() -> str:
    """根据当前配置状态决定首页 URL。"""
    from app.core.env_checker import can_start_directly

    ok, _ = can_start_directly()
    if ok:
        return _BASE_URL + "/"
    return _BASE_URL + "/api/admin/launcher/page"


# =============================================================================
# 步骤 5：webview 窗口管理
# =============================================================================

_webview_window = None


def _open_or_focus_window(url: str | None = None) -> None:
    """显示/恢复 webview 窗口。窗口已存在则恢复焦点；不存在则创建。"""
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
    """重启应用：启动新进程后退出当前进程。"""
    import subprocess
    subprocess.Popen([sys.executable] + sys.argv)
    _quit_app()


# =============================================================================
# 步骤 7：退出
# =============================================================================

def _quit_app() -> None:
    """彻底退出应用。"""
    global _webview_window
    import webview

    try:
        webview.windows[0].destroy()
    except Exception:
        pass

    os._exit(0)


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """Ramaria 桌面版启动入口。"""
    import webview
    from ramaria.logger import get_logger

    logger = get_logger("bundle")
    logger.info(f"Ramaria 桌面版启动，根目录：{ROOT}")

    # 修复打包模式下的 config 路径（必须在导入 app.main 之前执行）
    _patch_config_paths()
    logger.info("路径修复完成")

    # 启动后台服务
    logger.info("正在启动后台服务…")
    ready = start_backend()
    if not ready:
        logger.error("后台服务启动超时")
        start_url = _BASE_URL + "/api/admin/launcher/page"
    else:
        logger.info("后台服务就绪")
        start_url = _get_start_url()

    # 创建系统托盘
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

    # 创建 webview 窗口
    global _webview_window

    _webview_window = webview.create_window(
        title       = _TITLE,
        url         = start_url,
        width       = 1024,
        height      = 720,
        min_size    = (800, 600),
        resizable   = True,
        on_top      = False,
    )

    icon_arg = {}
    if _ICON_PATH.exists():
        icon_arg["icon"] = str(_ICON_PATH)

    logger.info(f"打开窗口：{start_url}")

    webview.start(
        http_server = False,
        **icon_arg,
    )

    logger.info("窗口已关闭，程序退出")
    os._exit(0)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    main()
