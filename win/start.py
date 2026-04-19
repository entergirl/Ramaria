"""
start.py — 珊瑚菌 Ramaria 启动脚本
=====================================
用法：
    python start.py

执行内容：
    前置检查 1  虚拟环境是否存在（未安装则引导先运行 install.py）
    前置检查 2  .env 是否存在且填写了必要配置项
    前置检查 3  数据库是否存在（不存在则自动初始化）
    启动        在 venv 内运行 python app/main.py

注意：
    · 运行前请确保本地模型推理服务（LM Studio / Ollama）已启动
    · 按 Ctrl+C 可停止服务
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

# ── 项目根目录 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── venv 内 Python 可执行文件路径 ─────────────────────────────────────────────
IS_WINDOWS  = platform.system() == "Windows"
VENV_PYTHON = (
    ROOT / "venv" / "Scripts" / "python.exe"
    if IS_WINDOWS
    else ROOT / "venv" / "bin" / "python"
)

# ── .env 中必须有值的配置项 ───────────────────────────────────────────────────
# 启动前做简单检查，在进程拉起前就提示用户，而不是等服务深处才报错
REQUIRED_ENV_KEYS = [
    "LOCAL_API_URL",
    "LOCAL_MODEL_NAME",
    "EMBEDDING_MODEL",
]


# =============================================================================
# 工具函数
# =============================================================================

def banner(text: str) -> None:
    print(f"\n{'=' * 52}")
    print(f"  {text}")
    print(f"{'=' * 52}\n")


def info(text: str) -> None:
    print(f"  [信息] {text}")


def warn(text: str) -> None:
    print(f"  [警告] {text}")


def error_exit(text: str) -> None:
    """打印错误后暂停，防止双击运行时窗口瞬间关闭。"""
    print(f"\n  [错误] {text}")
    print("\n按回车键退出...")
    input()
    sys.exit(1)


def load_env_file(env_path: Path) -> dict:
    """
    简单解析 .env 文件，返回 {key: value} 字典。
    只处理 KEY=VALUE 格式，忽略注释行（# 开头）和空行。
    不依赖 python-dotenv，使用纯标准库实现。
    """
    result = {}
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            # 去除行尾注释（# 后面的部分）和首尾引号
            value = value.split("#")[0].strip().strip("'\"")
            result[key.strip()] = value
    return result


# =============================================================================
# 前置检查
# =============================================================================

def check_venv() -> None:
    """前置检查 1：虚拟环境是否存在。"""
    if not VENV_PYTHON.exists():
        error_exit(
            "未找到虚拟环境。\n\n"
            "  请先运行安装脚本完成安装：\n"
            "      python install.py\n\n"
            "  安装完成后再运行本脚本。"
        )
    info(f"虚拟环境检测通过：{VENV_PYTHON}")


def check_env_file() -> None:
    """
    前置检查 2：.env 是否存在，且必填项已填写。
    检测到 .env 不存在时自动从模板生成，然后提示用户填写后重新启动。
    检测到必填项为空时打印具体缺失的 key，不直接启动。
    """
    import shutil

    env_file    = ROOT / ".env"
    env_example = ROOT / ".env.example"

    # .env 不存在：尝试从模板生成
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print()
            print("  [提示] 未找到 .env，已根据 .env.example 自动生成。")
        else:
            error_exit(
                "未找到 .env 配置文件，也找不到 .env.example 模板。\n"
                "  请参考 README.md 手动创建 .env 并填写配置。"
            )

        # 生成后提示用户填写，不直接启动（避免带空配置启动后在深处报错）
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  请打开 .env 文件，填写以下必要配置后重新运行本脚本   │")
        print("  ├─────────────────────────────────────────────────────┤")
        print("  │  LOCAL_API_URL    本地模型服务地址                   │")
        print("  │                   LM Studio 默认：                   │")
        print("  │                   http://localhost:1234/v1/...       │")
        print("  │  LOCAL_MODEL_NAME 模型名称（与推理服务中保持一致）    │")
        print("  │  EMBEDDING_MODEL  嵌入模型的本地文件夹完整路径        │")
        print("  └─────────────────────────────────────────────────────┘")
        print()
        input("按回车键退出...")
        sys.exit(0)

    # .env 存在：检查必填项是否有值
    env_vars = load_env_file(env_file)

    # 将 .env 中的配置加载到当前进程的环境变量
    # 这样后续启动的子进程可以继承这些配置
    for key, value in env_vars.items():
        os.environ[key] = value

    missing  = [k for k in REQUIRED_ENV_KEYS if not env_vars.get(k, "").strip()]

    if missing:
        print()
        print("  [错误] .env 中以下必填项尚未配置：")
        for key in missing:
            print(f"           · {key}")
        print()
        print("  请打开 .env 填写上述配置后重新运行本脚本。")
        print()
        input("按回车键退出...")
        sys.exit(1)

    info(".env 配置检测通过。")

    # 特别检查 EMBEDDING_MODEL 路径是否真实存在（最常见的启动失败原因）
    embedding_path = env_vars.get("EMBEDDING_MODEL", "").strip()
    if embedding_path and not Path(embedding_path).exists():
        print()
        warn(
            f"EMBEDDING_MODEL 路径不存在：\n"
            f"           {embedding_path}\n\n"
            "  这会导致服务在启动阶段崩溃。\n"
            "  请确认嵌入模型已下载完整，且 .env 中的路径正确。"
        )
        print()
        answer = input("  是否仍要继续启动？(y/N): ").strip().lower()
        if answer != "y":
            sys.exit(0)


def check_and_init_db() -> None:
    """前置检查 3：数据库不存在时自动初始化。"""
    db_file      = ROOT / "data" / "assistant.db"
    setup_script = ROOT / "scripts" / "setup_db.py"

    # 确保 data/ logs/ 目录存在
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    if db_file.exists():
        info("数据库检测通过。")
        return

    if not setup_script.exists():
        warn("未找到 scripts/setup_db.py，跳过数据库初始化。")
        return

    print("  [提示] 未找到数据库，正在初始化...")
    try:
        subprocess.run([str(VENV_PYTHON), str(setup_script)], check=True)
        info("数据库初始化成功。")
    except subprocess.CalledProcessError:
        warn(
            "数据库初始化脚本报错，将尝试继续启动。\n"
            "  如启动后出现数据库相关报错，请手动运行：\n"
            f"  python scripts/setup_db.py"
        )


# =============================================================================
# 启动服务
# =============================================================================

def run_health_check() -> None:
    """
    以子进程方式运行 scripts/health_check.py。

    使用子进程而非直接 import，好处是：
        · health_check 失败时不污染 start.py 的进程环境
        · 用户可以单独运行 health_check.py 进行排障，行为与 start.py 调用完全一致
        · health_check 内部的 sys.exit() 不会直接终止 start.py

    退出码约定：
        0 — 全部通过，继续启动
        非 0 — 有失败项，终止启动（错误信息已在子进程中打印）
    """
    health_check_script = ROOT / "scripts" / "health_check.py"

    if not health_check_script.exists():
        # health_check.py 不存在时跳过检查，允许继续启动
        # 避免脚本缺失导致所有用户无法启动
        warn(
            "未找到 scripts/health_check.py，跳过启动前检查。\n"
            "  建议确认项目文件完整，或重新拉取最新代码。"
        )
        return

    print("\n  正在执行启动前健康检查...\n")

    result = subprocess.run(
        [str(VENV_PYTHON), str(health_check_script)],
        cwd=str(ROOT),   # 工作目录设为项目根目录，与 health_check 内的路径推断一致
    )

    if result.returncode != 0:
        # 错误信息已由 health_check.py 打印，这里只补充操作提示
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  健康检查未通过，启动已终止。                        │")
        print("  │  请根据上方提示修复问题后重新运行 start.py。          │")
        print("  │                                                     │")
        print("  │  如需跳过检查直接启动（调试用）：                    │")
        print("  │    python app/main.py                               │")
        print("  └─────────────────────────────────────────────────────┘")
        print()
        input("按回车键退出...")
        sys.exit(1)

    print("  健康检查通过，正在启动服务...\n")

def start_service() -> None:
    """通过 venv 内的 Python 启动 app/main.py。"""
    main_script = ROOT / "app" / "main.py"

    if not main_script.exists():
        error_exit(f"未找到 app/main.py，请确认项目文件完整。")

    print()
    print("  ------------------------------------------------")
    print("  本地访问：  http://localhost:8000")
    print("  局域网访问：http://<本机IP>:8000")
    print("  按 Ctrl+C 停止服务")
    print("  ------------------------------------------------")
    print()

    try:
        # 切换工作目录到项目根目录，与原来直接运行 python app/main.py 的行为一致
        subprocess.run(
            [str(VENV_PYTHON), str(main_script)],
            cwd=str(ROOT),
            check=True,
        )
    except KeyboardInterrupt:
        # Ctrl+C 是正常退出方式，不算错误
        print()
        info("服务已停止（Ctrl+C）。")
    except subprocess.CalledProcessError as e:
        print()
        print(f"  [错误] 服务异常退出，退出码：{e.returncode}")
        print()
        print("  常见原因：")
        print("    · 本地模型推理服务未启动（LM Studio / Ollama）")
        print("    · EMBEDDING_MODEL 路径不存在或模型文件不完整")
        print("    · 端口 8000 已被其他程序占用")
        print()
        print("  查看上方的完整错误信息可以定位具体原因。")
        print()
        input("按回车键退出...")
        sys.exit(1)


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    os.chdir(ROOT)

    banner("珊瑚菌 Ramaria")

    print("  正在进行前置检查...\n")
    check_venv()
    check_env_file()
    check_and_init_db()
    run_health_check()
    print()
    info("前置检查全部通过，正在启动服务...")
    start_service()


if __name__ == "__main__":
    main()
