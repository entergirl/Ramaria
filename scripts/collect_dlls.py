"""
打包后自动复制必要的 DLL 到输出目录

使用方法：
1. 先运行 pyinstaller ramaria.spec --noconfirm
2. 然后运行 python collect_dlls.py

这会从以下位置复制 VC++ 运行时 DLL：
- Python 安装目录的 DLLs 文件夹
- VC++ Redistributable 安装目录
"""
import os
import sys
import shutil
from pathlib import Path


def find_python_dir():
    """查找 Python 安装目录"""
    if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix:
        # 虚拟环境
        return Path(sys.prefix)
    return Path(sys.prefix)


def find_vc_redist_dlls():
    """查找 VC++ Redistributable 的运行时 DLL"""
    dlls = []

    # 常见的 VC++ Redistributable 安装位置
    search_paths = [
        # Python DLLs 目录
        find_python_dir() / "DLLs",

        # VC++ 2015-2022 Redistributable
        Path("C:/Windows/System32"),  # 有时 DLL 会被复制到这里

        # Visual Studio 安装目录
        *list(Path("C:/Program Files/Microsoft Visual Studio").glob("**/VC/Redist/MSVC/*/x64/")),
        *list(Path("C:/Program Files (x86)/Microsoft Visual Studio").glob("**/VC/Redist/MSVC/*/x64/")),
    ]

    # 需要收集的 DLL 列表
    needed_dlls = [
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "python310.dll",
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue
        for dll_name in needed_dlls:
            dll_path = search_path / dll_name
            if dll_path.exists() and dll_path not in [d[0] for d in dlls]:
                dlls.append((dll_path, dll_name))
                print(f"Found: {dll_path}")

    return dlls


def copy_dlls_to_dist():
    """复制 DLL 到 dist 目录"""
    # 查找 dist/Ramaria 目录（相对于项目根目录）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dist_path = project_root / "dist" / "Ramaria"

    if not dist_path.exists():
        print(f"Error: {dist_path} not found. Run pyinstaller first.")
        return False

    print(f"\nTarget directory: {dist_path}")
    print("-" * 50)

    dlls = find_vc_redist_dlls()

    if not dlls:
        print("Warning: No VC++ runtime DLLs found!")
        print("Users may need to install Visual C++ Redistributable manually.")
        return False

    copied = 0
    for src_path, dll_name in dlls:
        dst_path = dist_path / dll_name
        try:
            shutil.copy2(src_path, dst_path)
            print(f"[OK] Copied {dll_name} to {dist_path}")
            copied += 1
        except Exception as e:
            print(f"[FAIL] {dll_name}: {e}")

    print("-" * 50)
    print(f"Copied {copied} DLLs")
    return copied > 0


if __name__ == "__main__":
    print("=" * 50)
    print("Ramaria DLL Collector")
    print("=" * 50)
    copy_dlls_to_dist()
    print("\nDone! You can now run the executable on other machines.")
