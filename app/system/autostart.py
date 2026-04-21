"""
app/system/autostart.py — 开机自启管理（Windows）

职责：
    通过 Windows 注册表的 HKCU\Software\Microsoft\Windows\CurrentVersion\Run
    键管理开机自启，只操作当前用户，不需要管理员权限。

对外接口：
    is_enabled()   → bool      检测当前是否已开机自启
    enable(exe)    → None      写入注册表（exe 为 Ramaria.exe 的完整路径）
    disable()      → None      删除注册表键
    toggle(exe)    → bool      切换状态，返回切换后的新状态

注意：非 Windows 平台调用时静默返回 False，不抛出异常。
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

# 注册表路径和应用名称
_REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
_APP_NAME = "Ramaria"

_IS_WINDOWS = platform.system() == "Windows"


def is_enabled() -> bool:
    """检测 Ramaria 是否已加入开机自启。非 Windows 始终返回 False。"""
    if not _IS_WINDOWS:
        return False

    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_PATH, 0, winreg.KEY_READ)
        winreg.QueryValueEx(key, _APP_NAME)
        winreg.CloseKey(key)
        return True
    except FileNotFoundError:
        # 键不存在，正常情况（未设置自启）
        return False
    except Exception:
        return False


def enable(exe_path: str | Path) -> bool:
    """
    将 exe_path 写入注册表，开启开机自启。

    参数：
        exe_path — Ramaria.exe 的完整绝对路径

    返回：
        True = 成功，False = 失败（非 Windows 或写入出错）
    """
    if not _IS_WINDOWS:
        return False

    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _REG_PATH,
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, _APP_NAME, 0, winreg.REG_SZ, str(exe_path))
        winreg.CloseKey(key)
        return True
    except Exception as e:
        from ramaria.logger import get_logger
        get_logger(__name__).warning(f"开机自启写入失败 — {e}")
        return False


def disable() -> bool:
    """
    从注册表删除开机自启键。

    返回：
        True = 成功（包括本来就没有的情况），False = 出现意外错误
    """
    if not _IS_WINDOWS:
        return False

    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _REG_PATH,
            0, winreg.KEY_SET_VALUE
        )
        winreg.DeleteValue(key, _APP_NAME)
        winreg.CloseKey(key)
        return True
    except FileNotFoundError:
        # 本来就没有，视为成功
        return True
    except Exception as e:
        from ramaria.logger import get_logger
        get_logger(__name__).warning(f"开机自启删除失败 — {e}")
        return False


def toggle(exe_path: str | Path) -> bool:
    """
    切换开机自启状态。

    返回：
        True = 切换后处于开启状态，False = 切换后处于关闭状态
    """
    if is_enabled():
        disable()
        return False
    else:
        enable(exe_path)
        return True
