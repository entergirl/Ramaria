# -*- mode: python ; coding: utf-8 -*-
"""
ramaria.spec — PyInstaller 打包配置

打包说明：
    在项目根目录运行：
        pyinstaller ramaria.spec

    输出在 dist/Ramaria/ 目录（文件夹模式，非单文件 exe）。
    文件夹模式比单文件 exe 启动快约 3-5 倍，推荐使用。

排除项：
    · 向量嵌入模型（用户在 .env 中填绝对路径，不打包进 exe）
    · 开发调试工具（pytest、ipython 等）
    · 不必要的大型库（避免打包体积过大）

注意事项：
    · 首次打包前需 pip install pyinstaller
    · 打包环境需与运行环境位数相同（64-bit → 64-bit exe）
    · icon.ico 需提前准备好（16/32/48/64 px 多尺寸 ico 文件）
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

# 项目根目录（spec 文件所在位置）
ROOT = Path(sys.argv[0]).parent.resolve()

# =============================================================================
# 自动收集第三方库依赖
# =============================================================================
# 自动收集 chromadb 的所有必要资源（数据、二进制文件、隐式模块）
chromadb_datas, chromadb_binaries, chromadb_hiddenimports = collect_all('chromadb')

# =============================================================================
# 数据文件：打包时需要随 exe 一起携带的非 Python 文件
# =============================================================================
datas = [
    # 静态前端文件（HTML/CSS/JS）
    (str(ROOT / "static"),          "static"),
    # 配置模板（用户未创建 persona.toml 时的提示）
    (str(ROOT / "config"),          "config"),
    # 数据库初始化脚本（打包模式下由 db_initializer 进程内调用）
    (str(ROOT / "scripts"),         "scripts"),
    # .env.example 模板（首次启动时复制生成空 .env，引导用户填写）
    # 注意：不打包 .env 本身，避免开发者配置泄露到用户环境
    (str(ROOT / ".env.example"),    "."),
] + chromadb_datas

# =============================================================================
# 隐式导入：PyInstaller 分析不到、但运行时需要的模块
# =============================================================================
hiddenimports = [
    # FastAPI 相关
    "fastapi",
    "fastapi.routing",
    "fastapi.staticfiles",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    # ramaria 包
    "ramaria",
    "ramaria.config",
    "ramaria.logger",
    "ramaria.storage.database",
    "ramaria.storage.vector_store",
    "ramaria.memory.summarizer",
    "ramaria.memory.merger",
    "ramaria.memory.profile_manager",
    "ramaria.memory.conflict_checker",
    "ramaria.memory.push_scheduler",
    "ramaria.memory.graph_builder",
    "ramaria.memory.decay",
    "ramaria.core.session_manager",
    "ramaria.core.router",
    "ramaria.core.prompt_builder",
    "ramaria.core.llm_client",
    "ramaria.tools.tool_registry",
    "ramaria.tools.hardware_monitor",
    "ramaria.tools.fs_scanner",
    "ramaria.tools.weather",
    "ramaria.tools.web_fetcher",
    "ramaria.importer.batch",
    "ramaria.importer.qq.parser",
    "ramaria.importer.qq.importer",
    # app 包
    "app.main",
    "app.core.env_checker",
    "app.routes.admin",
    "app.routes.chat",
    "app.routes.sessions",
    "app.routes.settings",
    "app.routes.memory",
    "app.routes.import_ctrl",
    "app.routes.graph_ctrl",
    "app.routes.router_ctrl",
    "app.system.tray",
    "app.system.autostart",
    "app.dependencies",
    # 系统集成
    "pystray",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "webview",
    # 数据库和向量
    "chromadb",
    "chromadb.utils.embedding_functions",
    "chromadb.api.models.Collection",
    "chromadb.api.models.CollectionCommon",
    "sentence_transformers",
    "rank_bm25",
    "jieba",
    "networkx",
    "onnxruntime",
    # 其他
    "tomllib" if sys.version_info >= (3, 11) else "tomli",
    "psutil",
    "requests",
    "winreg",        # Windows 注册表（仅 Windows 可用）
    # 消除警告的模块
    "email_validator",   # FastAPI EmailStr 类型需要
    "pycparser",         # CFFI 依赖
] + chromadb_hiddenimports

# =============================================================================
# 排除模块：明确不需要的包，减小体积
# =============================================================================
excludes = [
    # 测试工具
    "pytest",
    "pytest_asyncio",
    "_pytest",
    # 交互式环境
    "IPython",
    "ipykernel",
    "jupyter",
    "notebook",
    # 文档生成
    "sphinx",
    "docutils",
    # 开发工具
    "black",
    "isort",
    "mypy",
    "pylint",
    # 不需要的科学计算
    "matplotlib",
    "pandas",
    # "sklearn",  # 需要用于向量检索/BM25，不能排除
    "scipy",
    # Telegram 适配（打包版不需要）
    "telegram",
    "python_telegram_bot",
]

# =============================================================================
# PyInstaller 打包配置
# =============================================================================

# 收集 Python DLL 和运行时库（解决 python310.dll 加载失败）
import sysconfig
import os

python_dir = sysconfig.get_path("scripts").replace("\\Scripts", "")

extra_binaries = []
if sys.platform == "win32":
    # Python DLL
    python_dll = sysconfig.get_config_var("LDLIBRARY") or "python310.dll"
    python_dll_path = str(Path(python_dir) / python_dll)
    if Path(python_dll_path).exists():
        extra_binaries.append((python_dll_path, "."))

    # 收集 DLLs 目录下的所有 DLL（C 运行时库等）
    dlls_dir = Path(python_dir) / "DLLs"
    if dlls_dir.exists():
        for dll in dlls_dir.glob("vcruntime*.dll"):
            extra_binaries.append((str(dll), "."))
        for dll in dlls_dir.glob("msvcp*.dll"):
            extra_binaries.append((str(dll), "."))
        for dll in dlls_dir.glob("python*.dll"):
            if str(dll) not in [p for p, _ in extra_binaries]:
                extra_binaries.append((str(dll), "."))
        # WebView2Loader.dll (pywebview 需要)
        webview2_dll = dlls_dir / "WebView2Loader.dll"
        if webview2_dll.exists():
            extra_binaries.append((str(webview2_dll), "."))


a = Analysis(
    scripts    = [str(ROOT / "app" / "bundle.py")],
    pathex     = [str(ROOT), str(ROOT / "src")],
    binaries   = chromadb_binaries + extra_binaries,
    datas      = datas,
    hiddenimports = hiddenimports,
    hookspath  = [],
    hooksconfig= {},
    runtime_hooks = [str(ROOT / "scripts" / "_runtime_hook.py")],
    excludes   = excludes,
    noarchive  = False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries = True,         # 使用文件夹模式
    name             = "Ramaria",
    debug            = False,
    bootloader_ignore_signals = False,
    strip            = False,
    upx              = False,        # UPX 压缩已禁用（会导致 python310.dll 损坏）
    console          = True,         # 显示控制台窗口用于调试
    disable_windowed_traceback = False,
    target_arch      = None,
    codesign_identity = None,
    entitlements_file = None,
    icon             = str(ROOT / "icon.ico") if (ROOT / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip    = False,
    upx      = False,
    upx_exclude = [],
    name     = "Ramaria",
)
