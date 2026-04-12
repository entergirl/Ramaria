"""
install.py — 珊瑚菌 Ramaria 一键安装脚本
=========================================
用法：
    python install.py

执行内容：
    步骤 1  检查 Python 版本（要求 3.10+）
    步骤 2  创建或复用 venv 虚拟环境
    步骤 3  升级 pip
    步骤 4  安装项目依赖（pip install -e .）
    步骤 5  生成 .env 配置文件（已存在则跳过）
    步骤 6  初始化数据库（scripts/setup_db.py）

本文件使用纯 Python 标准库，无任何第三方依赖，
可在安装依赖之前安全运行。
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── 项目根目录（本文件所在目录）──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── 虚拟环境目录 ──────────────────────────────────────────────────────────────
VENV_DIR = ROOT / "venv"

# ── 判断当前操作系统 ──────────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"

# venv 内 Python / pip 可执行文件的路径因平台而异
if IS_WINDOWS:
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP    = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP    = VENV_DIR / "bin" / "pip"


# =============================================================================
# 工具函数
# =============================================================================

def banner(text: str) -> None:
    """打印带分隔线的阶段标题。"""
    print(f"\n{'=' * 52}")
    print(f"  {text}")
    print(f"{'=' * 52}")


def step(n: int, total: int, text: str) -> None:
    """打印步骤进度行。"""
    print(f"\n[步骤 {n}/{total}] {text}")


def ok(text: str) -> None:
    print(f"  [完成] {text}")


def skip(text: str) -> None:
    print(f"  [跳过] {text}")


def warn(text: str) -> None:
    print(f"  [警告] {text}")


def error_exit(text: str) -> None:
    """打印错误并退出，暂停等待用户确认（双击运行时窗口不会消失）。"""
    print(f"\n  [错误] {text}")
    print("\n按回车键退出...")
    input()
    sys.exit(1)


def run(cmd: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """
    运行子进程命令。
    cmd     — 命令列表，如 [str(VENV_PYTHON), "-m", "pip", "install", "-e", "."]
    check   — True 时命令失败会抛出 CalledProcessError
    capture — True 时捕获 stdout/stderr，不打印到终端
    """
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"]   = True
    return subprocess.run(cmd, check=check, **kwargs)


# =============================================================================
# 步骤实现
# =============================================================================

def check_python_version() -> None:
    """步骤 1：确认当前 Python 版本 >= 3.10。"""
    major = sys.version_info.major
    minor = sys.version_info.minor
    print(f"  当前 Python 版本：{major}.{minor}.{sys.version_info.micro}")

    if major < 3 or (major == 3 and minor < 10):
        error_exit(
            f"Python 版本过低（当前 {major}.{minor}），需要 3.10 或以上版本。\n"
            "  下载地址：https://www.python.org/downloads/\n"
            "  安装时请勾选 'Add Python to PATH'"
        )
    ok(f"Python {major}.{minor} 版本检测通过。")


def create_venv() -> None:
    """步骤 2：创建或复用 venv 虚拟环境。"""
    if VENV_PYTHON.exists():
        skip("虚拟环境已存在，复用现有环境。")
        return

    print("  正在创建虚拟环境，请稍候...")
    try:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError:
        error_exit(
            "虚拟环境创建失败。\n"
            "  可能原因：磁盘空间不足，或 Python 安装不完整（缺少 venv 模块）。"
        )
    ok("虚拟环境已创建。")


def upgrade_pip() -> None:
    """步骤 3：升级 venv 内的 pip，避免老版本 pip 安装部分包时报错。"""
    print("  正在升级 pip...")
    try:
        run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        ok("pip 已升级。")
    except subprocess.CalledProcessError:
        # pip 升级失败不阻断安装，只打警告
        warn("pip 升级失败，将继续使用当前版本。")


def install_deps() -> None:
    """步骤 4：以可编辑模式安装项目依赖。"""
    print("  正在安装依赖，首次安装约需 3-10 分钟，请耐心等待...")
    print("  （安装过程中网络不稳定可能导致失败，可重新运行本脚本重试）")

    # 切换到项目根目录，确保 pip install -e . 能找到 pyproject.toml
    try:
        run([str(VENV_PYTHON), "-m", "pip", "install", "-e", ".", "--quiet"],
            check=True)
    except subprocess.CalledProcessError:
        error_exit(
            "依赖安装失败，常见原因：\n"
            "  1. 网络不稳定，可尝试切换网络或使用代理\n"
            "  2. 缺少 C++ 编译工具（Windows 用户需安装 Microsoft C++ Build Tools）\n"
            "     下载地址：https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            "  3. 磁盘空间不足\n\n"
            "  也可手动运行：\n"
            f"  {VENV_PYTHON} -m pip install -e .\n"
            "  以查看完整错误信息。"
        )
    ok("依赖安装成功。")


def setup_env() -> None:
    """步骤 5：从 .env.example 生成 .env，已存在则跳过。"""
    env_file     = ROOT / ".env"
    env_example  = ROOT / ".env.example"

    if env_file.exists():
        skip(".env 已存在，保留现有配置。")
        return

    if env_example.exists():
        shutil.copy(env_example, env_file)
        ok(".env 已根据 .env.example 生成。")
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  重要：安装完成后请打开 .env，填写以下必要配置项      │")
        print("  ├─────────────────────────────────────────────────────┤")
        print("  │  LOCAL_API_URL    本地模型服务地址                   │")
        print("  │  LOCAL_MODEL_NAME 模型名称（与推理服务保持一致）      │")
        print("  │  EMBEDDING_MODEL  嵌入模型的本地文件夹路径           │")
        print("  └─────────────────────────────────────────────────────┘")
    else:
        warn(
            "未找到 .env.example 模板文件。\n"
            "  请参考 README.md 手动创建 .env 并填写配置。"
        )


def init_db() -> None:
    """步骤 6：初始化数据库（幂等，重复运行不会破坏已有数据）。"""
    db_file     = ROOT / "data" / "assistant.db"
    setup_script = ROOT / "scripts" / "setup_db.py"

    # 确保 data/ logs/ 目录存在
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    if db_file.exists():
        skip("数据库已存在，跳过初始化。")
        return

    if not setup_script.exists():
        warn("未找到 scripts/setup_db.py，跳过数据库初始化。\n"
             "  数据库将在首次启动服务时自动创建。")
        return

    print("  正在初始化数据库...")
    try:
        # 使用 venv 内的 Python 运行，确保依赖可用
        run([str(VENV_PYTHON), str(setup_script)])
        ok("数据库初始化成功。")
    except subprocess.CalledProcessError:
        # 数据库初始化失败不阻断安装，服务首次启动时会自动重试
        warn(
            "数据库初始化脚本报错。\n"
            "  这不一定是致命错误，服务启动时会尝试自动创建数据库。\n"
            "  如启动后出现数据库相关报错，请手动运行：\n"
            f"  {VENV_PYTHON} scripts/setup_db.py"
        )


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    # 切换工作目录到项目根目录，确保相对路径全部有效
    os.chdir(ROOT)

    banner("珊瑚菌 Ramaria 安装脚本")
    print(f"  项目目录：{ROOT}")
    print(f"  操作系统：{platform.system()} {platform.release()}")

    TOTAL = 6
    step(1, TOTAL, "检查 Python 版本"); check_python_version()
    step(2, TOTAL, "配置虚拟环境");     create_venv()
    step(3, TOTAL, "升级 pip");        upgrade_pip()
    step(4, TOTAL, "安装项目依赖");     install_deps()
    step(5, TOTAL, "生成配置文件");     setup_env()
    step(6, TOTAL, "初始化数据库");     init_db()

    banner("安装完成！")
    print()
    print("  下一步：")
    print("    1. 打开 .env，确认配置已填写完整")
    print("    2. 启动本地模型推理服务（LM Studio 或 Ollama）")
    print("    3. 运行  python start.py  启动珊瑚菌")
    print()
    print("  访问地址：http://localhost:8000")
    print()
    input("按回车键退出...")


if __name__ == "__main__":
    main()
