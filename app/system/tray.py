"""
app/system/tray.py — 系统托盘管理模块

职责：
    创建 Windows 系统托盘图标，提供右键菜单控制珊瑚菌的运行状态。
    托盘在独立线程中运行，不阻塞 FastAPI 主服务。

右键菜单项：
    ├─ 打开界面      → 调用 open_window_fn() 显示/恢复 webview 窗口
    ├─ ─────────── 分隔线
    ├─ 服务状态      → 显示服务是否运行（灰色，不可点击）
    ├─ 重启服务      → 调用 restart_fn()
    ├─ ─────────── 分隔线
    ├─ 设置          → 打开配置向导
    ├─ 开机自启  ✓   → 切换开机自启状态
    ├─ ─────────── 分隔线
    └─ 退出          → 调用 quit_fn()

依赖：
    pystray >= 0.19
    Pillow  >= 9.0

使用方式：
    tray = RamariaTrayx(
        open_window_fn = lambda: ...,
        restart_fn     = lambda: ...,
        quit_fn        = lambda: ...,
        exe_path       = Path("Ramaria.exe"),
    )
    tray.start()   # 在后台线程启动，立即返回
    ...
    tray.stop()    # 退出时调用
"""

from __future__ import annotations

import io
import platform
import threading
from pathlib import Path
from typing import Callable, Optional

_IS_WINDOWS = platform.system() == "Windows"


# =============================================================================
# 托盘图标生成（纯 Pillow，不依赖外部图标文件）
# =============================================================================

def _make_icon_image(size: int = 64) -> "Image.Image":
    """
    用 Pillow 绘制一个简单的珊瑚橙色圆形托盘图标。
    在没有 icon.ico 文件时作为默认图标使用。

    参数：
        size — 图标边长（像素），默认 64
    """
    from PIL import Image, ImageDraw

    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 珊瑚橙色填充圆形
    coral = (201, 124, 90, 255)
    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], fill=coral)

    # 中间画一个简单的"R"形白色菌状图案（两个白色圆点）
    dot_size = size // 8
    cx, cy   = size // 2, size // 2
    white    = (255, 255, 255, 230)
    draw.ellipse([cx - dot_size * 2, cy - dot_size, cx - dot_size, cy + dot_size], fill=white)
    draw.ellipse([cx + dot_size, cy - dot_size, cx + dot_size * 2, cy + dot_size], fill=white)

    return img


def _load_icon_image(icon_path: Path | None) -> "Image.Image":
    """
    加载图标文件；文件不存在或加载失败时使用 _make_icon_image() 生成默认图标。

    参数：
        icon_path — icon.ico 的路径；传 None 时直接使用默认图标
    """
    if icon_path and icon_path.exists():
        try:
            from PIL import Image
            return Image.open(icon_path).convert("RGBA")
        except Exception:
            pass
    return _make_icon_image()


# =============================================================================
# RamariaTrayx 托盘管理类
# =============================================================================

class RamariaTrayx:
    """
    珊瑚菌系统托盘管理类。

    在独立的守护线程中运行 pystray.Icon，与 FastAPI 服务互不阻塞。

    参数：
        open_window_fn  — 点击"打开界面"时调用的函数（无参数）
        restart_fn      — 点击"重启服务"时调用的函数（无参数）
        quit_fn         — 点击"退出"时调用的函数（无参数）
        open_setup_fn   — 点击"设置"时调用的函数（无参数），可选
        exe_path        — Ramaria.exe 路径，用于开机自启写入注册表
        icon_path       — icon.ico 路径，可选；不存在时使用默认生成图标
    """

    def __init__(
        self,
        open_window_fn:  Callable[[], None],
        restart_fn:      Callable[[], None],
        quit_fn:         Callable[[], None],
        open_setup_fn:   Optional[Callable[[], None]] = None,
        exe_path:        Optional[Path]               = None,
        icon_path:       Optional[Path]               = None,
    ):
        self._open_window_fn = open_window_fn
        self._restart_fn     = restart_fn
        self._quit_fn        = quit_fn
        self._open_setup_fn  = open_setup_fn or open_window_fn
        self._exe_path       = exe_path
        self._icon_path      = icon_path
        self._icon_obj       = None   # pystray.Icon 实例
        self._thread         = None   # 托盘运行线程

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """在后台守护线程中启动托盘图标，立即返回。"""
        if not _IS_WINDOWS:
            return  # 非 Windows 平台静默跳过

        self._thread = threading.Thread(
            target  = self._run,
            name    = "RamariaTrayx",
            daemon  = True,  # 主进程退出时自动结束
        )
        self._thread.start()

    def stop(self) -> None:
        """停止并移除托盘图标。"""
        if self._icon_obj:
            try:
                self._icon_obj.stop()
            except Exception:
                pass
            self._icon_obj = None

    def update_status(self, running: bool) -> None:
        """
        更新托盘菜单中的服务状态文字。
        pystray 菜单是静态生成的，通过重建菜单实现"刷新"。

        参数：
            running — True = 服务运行中，False = 服务已停止
        """
        self._service_running = running
        if self._icon_obj:
            try:
                self._icon_obj.menu = self._build_menu()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 内部实现
    # -------------------------------------------------------------------------

    _service_running: bool = True  # 服务运行状态，用于菜单文字显示

    def _run(self) -> None:
        """托盘线程主函数。"""
        try:
            import pystray

            icon_img = _load_icon_image(self._icon_path)
            menu     = self._build_menu()

            self._icon_obj = pystray.Icon(
                name  = "Ramaria",
                icon  = icon_img,
                title = "珊瑚菌 · Ramaria",
                menu  = menu,
            )
            # 左键单击直接打开窗口
            self._icon_obj.default_action = self._on_open

            self._icon_obj.run()
        except Exception as e:
            from ramaria.logger import get_logger
            get_logger(__name__).error(f"系统托盘运行出错 — {e}")

    def _build_menu(self) -> "pystray.Menu":
        """构建右键菜单，每次 update_status() 调用时重建。"""
        import pystray

        from app.system.autostart import is_enabled as autostart_is_enabled

        autostart_on  = autostart_is_enabled()
        status_text   = "● 服务运行中" if self._service_running else "○ 服务已停止"

        items = [
            pystray.MenuItem("打开界面",          self._on_open,        default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(status_text,         None,                 enabled=False),
            pystray.MenuItem("重启服务",           self._on_restart),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("设置",              self._on_setup),
            pystray.MenuItem(
                "开机自启",
                self._on_toggle_autostart,
                checked=lambda item: autostart_is_enabled(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出",              self._on_quit),
        ]
        return pystray.Menu(*items)

    # ── 菜单事件回调 ──────────────────────────────────────────────────────────

    def _on_open(self, icon=None, item=None) -> None:
        """打开/恢复界面窗口。"""
        try:
            self._open_window_fn()
        except Exception as e:
            from ramaria.logger import get_logger
            get_logger(__name__).warning(f"打开窗口失败 — {e}")

    def _on_restart(self, icon=None, item=None) -> None:
        """重启后端服务。"""
        try:
            self._restart_fn()
        except Exception as e:
            from ramaria.logger import get_logger
            get_logger(__name__).warning(f"重启服务失败 — {e}")

    def _on_setup(self, icon=None, item=None) -> None:
        """打开配置向导。"""
        try:
            self._open_setup_fn()
        except Exception as e:
            from ramaria.logger import get_logger
            get_logger(__name__).warning(f"打开设置失败 — {e}")

    def _on_toggle_autostart(self, icon=None, item=None) -> None:
        """切换开机自启。"""
        from app.system.autostart import toggle as autostart_toggle
        if self._exe_path:
            autostart_toggle(self._exe_path)
        # pystray 的 checked 回调会自动重新读取 is_enabled()，无需手动刷新

    def _on_quit(self, icon=None, item=None) -> None:
        """退出应用。先停止托盘图标，再调用退出回调。"""
        self.stop()
        try:
            self._quit_fn()
        except Exception:
            pass
