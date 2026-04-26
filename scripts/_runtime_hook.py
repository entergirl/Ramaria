"""
Runtime Hook: 确保 python310.dll 和 VC++ 运行时库能被正确加载

使用方法：
1. 将此文件放在项目根目录
2. 在 spec 文件的 runtime_hooks 中引用此文件
3. DLL 文件会自动从打包的 _internal 目录加载
"""
import os
import sys
import ctypes

def fix_dll_loading():
    """修复 DLL 加载路径问题"""
    if sys.platform != 'win32':
        return

    # 获取 _internal 目录（PyInstaller 打包后的资源目录）
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS  # PyInstaller 打包后的临时目录
        app_path = os.path.dirname(sys.executable)
        internal_path = os.path.join(app_path, '_internal')
    else:
        return  # 非打包模式下不需要

    # 将 _internal 目录添加到 DLL 搜索路径
    if os.path.exists(internal_path):
        os.add_dll_directory(internal_path)
        os.environ['PATH'] = internal_path + os.pathsep + os.environ.get('PATH', '')

    # 尝试预加载关键 DLL（如果存在）
    critical_dlls = [
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'msvcp140.dll',
        'msvcp140_1.dll',
        'msvcp140_2.dll',
    ]

    for dll_name in critical_dlls:
        dll_path = os.path.join(internal_path, dll_name)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(dll_path)
                print(f"[OK] Loaded {dll_name}")
            except Exception as e:
                print(f"[WARN] Failed to load {dll_name}: {e}")

fix_dll_loading()
