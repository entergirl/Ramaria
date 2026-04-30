"""
app/bundle.py — Ramaria 桌面版主入口

双击 Ramaria.exe 时执行此文件（PyInstaller 打包入口）。
开发调试时也可直接 python app/bundle.py 运行。

启动流程（优化版 — 先显示加载页）：
    1. 修复打包后的模块路径（PyInstaller _MEIPASS）
    2. 加载 .env 中的环境变量到当前进程
    3. 创建 pywebview 窗口，显示 loading.html（加载动画）
    4. 后台线程启动 FastAPI（uvicorn）
    5. loading.html 轮询 /api/admin/status，实时更新进度
    6. 服务就绪后，自动跳转到 index.html 或 launcher.html
    7. 系统托盘在后台运行（守护线程）

这样设计的好处：
    · 用户立即看到加载动画，而不是等待服务启动的黑屏
    · 加载进度透明化，提升用户体验
    · 复用前端的 LoadingScreen 动画逻辑

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
    解析 .env 文件，将 KEY=VALUE 写入 os.environ。
    不依赖 python-dotenv，使用纯标准库。
    已有同名环境变量时不覆盖（允许系统环境变量优先）。

    打包模式路径策略：
        只读取 exe 同级目录的 .env（用户通过配置向导创建的）。
        不再从 _MEIPASS 资源目录读取或复制 .env，因为：
          1. 打包时不再包含 .env（避免开发者配置泄露到用户环境）
          2. 首次运行时用户没有 .env → can_start_directly() 返回 False
          3. 配置向导通过 write_env() 在 exe 同级目录创建 .env
          4. 下次启动时自动加载用户配置

    开发模式路径策略：
        直接从项目根目录的 .env 加载。
    """
    env_file = None

    if getattr(sys, "frozen", False):
        # 打包模式：只读取 exe 同级目录的用户 .env
        exe_dir  = Path(sys.executable).parent
        user_env = exe_dir / ".env"
        if user_env.exists():
            env_file = user_env
            print(f"[bundle] 加载用户 .env: {env_file}")
        else:
            # 首次运行：用户还没有 .env，不创建空文件，
            # 让 can_start_directly() 自然判定需要引导
            print(f"[bundle] 首次运行：未找到用户 .env，将进入配置向导")
            return
    else:
        env_file = ROOT / ".env"
        print(f"[bundle] 加载开发 .env: {env_file}")

    if not env_file or not env_file.exists():
        print(f"[bundle] 警告: 未找到 .env 文件")
        return

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
      · .env      — 不再自动创建，首次运行时由配置向导写入（见 _load_dotenv）
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
_uvicorn_server: "uvicorn.Server | None" = None  # 保存 server 引用，用于 shutdown
_uvicorn_started: threading.Event = threading.Event()  # 标记 uvicorn 已启动（非就绪）


def _start_uvicorn() -> None:
    """在后台线程中启动 uvicorn（运行 FastAPI app）。"""
    import uvicorn

    # 标记启动中（让主线程知道线程已创建，可以开始轮询）
    _uvicorn_started.set()

    config = uvicorn.Config(
        app       = "app.main:app",
        host      = "127.0.0.1",
        port      = _SERVER_PORT,
        log_level = "warning",
    )
    global _uvicorn_server
    _uvicorn_server = uvicorn.Server(config)
    _uvicorn_server.run()


def _wait_for_service() -> bool:
    """轮询 /api/admin/status 直到服务响应，或超时返回 False。"""
    import urllib.error
    import urllib.request

    url      = f"{_BASE_URL}/api/admin/status"
    deadline = time.time() + _MAX_WAIT_SECONDS
    start_time = time.time()
    poll_count = 0

    print(f"\n  [启动] 等待后台服务就绪…")
    print(f"  [启动] 确保本地模型推理服务已启动（LM Studio / Ollama）\n")

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start_time
                    print(f"  [启动] ✓ 后台服务就绪 ({elapsed:.1f}s)")
                    return True
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        poll_count += 1
        
        # 每 10s 打印一次进度
        if poll_count % 20 == 0:
            remaining = _MAX_WAIT_SECONDS - elapsed
            print(
                f"  [启动] 仍在等待… ({elapsed:.0f}s/{_MAX_WAIT_SECONDS}s, "
                f"剩余 {remaining:.0f}s)"
            )
        
        time.sleep(_POLL_INTERVAL)

    elapsed = time.time() - start_time
    print(f"\n  [错误] ❌ 启动超时（{elapsed:.0f}s）")
    print(
        f"\n  可能的原因：\n"
        f"    1. 本地模型推理服务未启动\n"
        f"       → 请先启动 LM Studio 或 Ollama\n\n"
        f"    2. .env 中的配置不正确\n"
        f"       → 检查 LOCAL_API_URL 是否指向推理服务\n"
        f"       → 检查 LOCAL_MODEL_NAME 是否正确\n"
        f"       → 检查 EMBEDDING_MODEL 路径是否存在\n\n"
        f"    3. 嵌入模型加载缓慢\n"
        f"       → 等待更长时间（2-3 分钟）再试\n"
        f"       → 查看日志：logs/coral.log\n\n"
        f"  调试步骤：\n"
        f"    1. 查看日志文件了解详细错误：logs/coral.log\n"
        f"    2. 尝试手动访问：{_BASE_URL}\n"
        f"    3. 验证推理服务：\n"
        f"       - LM Studio: http://localhost:1234/v1/models\n"
        f"       - Ollama: http://localhost:11434/api/tags\n"
    )
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
# 步骤 5：webview 窗口管理 & JS 桥接
# =============================================================================

# JS 桥接对象（暴露给 loading.html 的 JavaScript）
class WebviewBridge:
    """暴露给前端 JavaScript 的 API"""

    def navigate_to(self, url: str) -> None:
        """
        加载页面就绪后，调用此方法跳转到目标页面。
        这是推荐的方式，因为它通过 pywebview 内部机制处理页面切换。
        """
        global _webview_window
        if _webview_window:
            # 构造完整 URL
            target_url = url if url.startswith('http') else _BASE_URL + url
            _webview_window.load_url(target_url)
            print(f"[Bridge] 导航到: {target_url}")

    def get_start_url(self) -> str:
        """获取应该显示的起始页面（index 或 launcher）"""
        return _get_start_url()


# 全局桥接实例
_webview_bridge = WebviewBridge()

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
    """彻底退出应用，先优雅关闭服务再退出进程。"""
    global _webview_window, _uvicorn_server
    import webview
    from ramaria.logger import get_logger

    logger = get_logger("bundle")

    # 步骤1：优雅关闭 uvicorn 服务（触发 lifespan shutdown 释放资源）
    if _uvicorn_server is not None:
        try:
            logger.info("正在关闭后台服务…")
            # stop() 会触发 lifespan 的 shutdown 阶段，
            # 正确关闭数据库连接、停止后台线程、释放端口
            _uvicorn_server.stop()
            logger.info("后台服务已关闭")
        except Exception as e:
            logger.warning(f"关闭服务时出错（可忽略）：{e}")

    # 步骤2：销毁 webview 窗口
    try:
        if webview.windows:
            webview.windows[0].destroy()
    except Exception:
        pass

    # 步骤3：安全退出进程
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

    # ── 步骤 1：立即创建 webview 窗口，显示加载页面 ──────────────────────────
    # 先打开窗口让用户看到加载动画，提升启动体验
    global _webview_window

    loading_url = _BASE_URL + "/loading.html"

    try:
        logger.info(f"创建 WebView 窗口（显示加载页面）…")
        _webview_window = webview.create_window(
            title           = _TITLE,
            url             = loading_url,
            width           = 1024,
            height          = 720,
            min_size        = (800, 600),
            resizable       = True,
            on_top          = False,
            background_color = "#faf5f3",
        )
        logger.info(f"WebView 窗口创建成功")

        icon_arg = {}
        if _ICON_PATH.exists():
            icon_arg["icon"] = str(_ICON_PATH)

        # ── 步骤 2：后台启动 uvicorn（在窗口显示后立即开始）────────────────────
        logger.info("启动后台服务线程…")
        _uvicorn_thread = threading.Thread(
            target  = _start_uvicorn,
            name    = "UvicornServer",
            daemon  = True,
        )
        _uvicorn_thread.start()

        # 等待 uvicorn 线程真正启动（避免 webview 启动后立即退出）
        _uvicorn_started.wait(timeout=5)
        logger.info("后台服务线程已启动")

        # ── 步骤 3：创建系统托盘图标 ──────────────────────────────────────────
        # 托盘需要独立线程运行，不阻塞主线程
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
        logger.info("系统托盘已启动")

        # ── 步骤 3：启动 WebView（主线程阻塞）───────────────────────────────────
        # 加载页面内部会轮询服务状态，就绪后自动跳转
        logger.info("启动 WebView 主循环…")

        webview.start(
            func       = None,          # 不需要单独的 init 函数
            js_api     = _webview_bridge,  # 暴露桥接对象的方法给 JS
            http_server = False,
            **icon_arg,
        )

    except ImportError as e:
        logger.error(f"❌ WebView2 库缺失或加载失败: {e}")
        _handle_webview_fallback(logger, "启动中")
        return

    except Exception as e:
        logger.error(f"❌ WebView 启动异常: {e}")
        logger.exception(e)
        _handle_webview_fallback(logger, "启动中")
        return

    finally:
        logger.info("窗口已关闭，程序退出")
        _quit_app()


def _handle_webview_fallback(logger, page_hint: str = "") -> None:
    """
    WebView 失败时的降级处理：尝试用浏览器打开。
    同时在后台启动服务。
    """
    import webbrowser

    # 先启动后台服务
    logger.info("降级方案：启动后台服务…")
    global _uvicorn_thread
    _uvicorn_thread = threading.Thread(
        target  = _start_uvicorn,
        name    = "UvicornServer",
        daemon  = True,
    )
    _uvicorn_thread.start()

    # 等待服务就绪
    ready = _wait_for_service()
    if not ready:
        print(f"\n[错误] 后台服务启动失败，请检查日志")
        return

    # 用浏览器打开
    start_url = _get_start_url()
    print(
        f"\n[警告] WebView 不可用，已用浏览器打开应用\n"
        f"访问地址：{start_url}\n"
        f"按 Ctrl+C 退出。\n"
    )

    try:
        webbrowser.open(start_url)
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    except Exception as e2:
        logger.error(f"浏览器打开失败: {e2}")


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    main()
