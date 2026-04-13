"""
scripts/health_check.py — 启动前健康检查脚本

检查项（共五项，任一失败立即退出并给出修复指引）：
    1. venv 存在且依赖完整（pip check）
    2. .env 必填项非空
    3. EMBEDDING_MODEL 路径存在
    4. 本地推理服务可达（GET /v1/models，超时 3 秒）
    5. assistant.db 存在且表结构完整（PRAGMA integrity_check）

退出码约定：
    0 — 全部通过，start.py 可继续拉起主服务
    1 — 有检查项失败，start.py 应终止启动

使用方式：
    # 单独手动排障
    python scripts/health_check.py

    # 被 start.py 以子进程方式调用
    result = subprocess.run([venv_python, "scripts/health_check.py"])
    if result.returncode != 0:
        sys.exit(1)

设计原则：
    · 本脚本使用纯标准库 + 项目已有依赖，不引入新包
    · 每项检查独立封装为函数，便于单独调试
    · 失败时打印具体修复步骤，而非只抛异常
    · 脚本本身的异常（如 import 失败）不应静默吞掉，直接抛出让调用方感知
"""

from __future__ import annotations

import os
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path


# =============================================================================
# 路径配置
# =============================================================================

# 本脚本在 scripts/ 目录下，项目根目录是上一级
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR    = _SCRIPTS_DIR.parent
_SRC_DIR     = _ROOT_DIR / "src"

# 将 src/ 加入路径，以便 import ramaria.config
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# venv 内 Python 可执行文件路径（与 start.py 保持一致）
_IS_WINDOWS  = platform.system() == "Windows"
_VENV_PYTHON = (
    _ROOT_DIR / "venv" / "Scripts" / "python.exe"
    if _IS_WINDOWS
    else _ROOT_DIR / "venv" / "bin" / "python"
)

# .env 中必须有值的配置项（与 start.py 保持一致）
_REQUIRED_ENV_KEYS = ["LOCAL_API_URL", "LOCAL_MODEL_NAME", "EMBEDDING_MODEL"]


# =============================================================================
# 终端输出工具
# =============================================================================

# ANSI 颜色码，Windows 旧版终端不支持时自动降级为无色
_USE_COLOR = (
    sys.stdout.isatty()
    and not (
        _IS_WINDOWS
        and int(platform.version().split(".")[2]) < 10586
        if _IS_WINDOWS and "." in platform.version()
        else False
    )
)

def _green(s: str)  -> str: return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _red(s: str)    -> str: return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _USE_COLOR else s
def _bold(s: str)   -> str: return f"\033[1m{s}\033[0m"  if _USE_COLOR else s


def _pass(item: str, detail: str = "") -> None:
    """打印通过项。"""
    line = f"  {_green('✓')}  {item}"
    if detail:
        line += f"\n       {detail}"
    print(line)


def _fail(item: str, detail: str = "", fix: str = "") -> None:
    """打印失败项（含修复指引）。"""
    print(f"  {_red('✗')}  {_bold(item)}")
    if detail:
        print(f"       原因：{detail}")
    if fix:
        # 修复指引可能是多行，逐行对齐缩进
        for line in fix.strip().splitlines():
            print(f"       {line}")


def _warn(item: str, detail: str = "") -> None:
    """打印警告项（不影响退出码）。"""
    line = f"  {_yellow('!')}  {item}"
    if detail:
        line += f"\n       {detail}"
    print(line)


def _section(title: str) -> None:
    print(f"\n{_bold(title)}")
    print("  " + "─" * 48)


# =============================================================================
# 工具：解析 .env 文件（纯标准库，不依赖 python-dotenv）
# =============================================================================

def _load_env(env_path: Path) -> dict[str, str]:
    """
    简单解析 .env 文件，返回 {key: value} 字典。
    只处理 KEY=VALUE 格式，忽略注释行和空行，去除行尾注释和引号。
    """
    result: dict[str, str] = {}
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
# 检查项 1：venv 存在且依赖完整
# =============================================================================

def check_venv() -> bool:
    """
    检查虚拟环境是否存在，且 pip check 无依赖冲突。

    pip check 会验证已安装包的依赖关系是否完整，
    检测到缺失或版本冲突时输出具体包名。

    返回：True=通过，False=失败
    """
    _section("检查 1 / 5：虚拟环境")

    # 1a. venv 目录存在性
    if not _VENV_PYTHON.exists():
        _fail(
            "虚拟环境不存在",
            detail=f"未找到 {_VENV_PYTHON}",
            fix=(
                "修复步骤：\n"
                "       请先运行安装脚本完成安装：\n"
                "         Windows : python win/install.py\n"
                "         Linux   : bash linux/install.sh\n"
                "         macOS   : bash mac/install.sh"
            ),
        )
        return False

    _pass("虚拟环境目录存在", detail=str(_VENV_PYTHON.parent.parent))

    # 1b. pip check（依赖完整性）
    result = subprocess.run(
        [str(_VENV_PYTHON), "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        _fail(
            "依赖存在冲突或缺失",
            detail=result.stdout.strip() or result.stderr.strip(),
            fix=(
                "修复步骤：\n"
                "       重新安装依赖：\n"
                f"         {_VENV_PYTHON} -m pip install -e ."
            ),
        )
        return False

    _pass("依赖完整性检查通过（pip check）")
    return True


# =============================================================================
# 检查项 2：.env 必填项非空
# =============================================================================

def check_env_file() -> bool:
    """
    检查 .env 文件存在，且三个必填项（LOCAL_API_URL、LOCAL_MODEL_NAME、
    EMBEDDING_MODEL）均已填写非空值。

    返回：True=通过，False=失败
    """
    _section("检查 2 / 5：.env 配置文件")

    env_path = _ROOT_DIR / ".env"

    if not env_path.exists():
        _fail(
            ".env 文件不存在",
            fix=(
                "修复步骤：\n"
                "       将 .env.example 复制为 .env 并填写配置：\n"
                "         cp .env.example .env\n"
                "       然后用文本编辑器打开 .env，填写必要配置项。"
            ),
        )
        return False

    _pass(".env 文件存在")

    try:
        env_vars = _load_env(env_path)
    except Exception as e:
        _fail(".env 解析失败", detail=str(e))
        return False

    # 逐一检查必填项
    missing = [k for k in _REQUIRED_ENV_KEYS if not env_vars.get(k, "").strip()]

    if missing:
        _fail(
            f"以下必填项未配置：{', '.join(missing)}",
            fix=(
                "修复步骤：\n"
                "       用文本编辑器打开 .env，填写上述配置项。\n"
                "       参考说明：\n"
                "         LOCAL_API_URL    — 本地推理服务地址\n"
                "                           LM Studio 默认：http://localhost:1234/v1/chat/completions\n"
                "         LOCAL_MODEL_NAME — 模型名称（与推理服务中保持一致）\n"
                "         EMBEDDING_MODEL  — 嵌入模型的本地文件夹完整路径"
            ),
        )
        return False

    _pass("所有必填项已配置", detail=f"已检查：{', '.join(_REQUIRED_ENV_KEYS)}")
    return True


# =============================================================================
# 检查项 3：EMBEDDING_MODEL 路径存在
# =============================================================================

def check_embedding_model() -> bool:
    """
    检查 .env 中 EMBEDDING_MODEL 指向的路径真实存在，
    且目录内包含 config.json（确认是完整的模型目录，而非空文件夹）。

    前置条件：.env 存在且 EMBEDDING_MODEL 已填写（check_env_file 已通过）。

    返回：True=通过，False=失败
    """
    _section("检查 3 / 5：嵌入模型路径")

    env_path = _ROOT_DIR / ".env"
    env_vars = _load_env(env_path)
    model_path_str = env_vars.get("EMBEDDING_MODEL", "").strip()

    # 理论上 check_env_file 已确保非空，此处做防御性处理
    if not model_path_str:
        _fail("EMBEDDING_MODEL 未填写（应已在检查 2 中报错）")
        return False

    model_path = Path(model_path_str)

    if not model_path.exists():
        _fail(
            "嵌入模型路径不存在",
            detail=str(model_path),
            fix=(
                "修复步骤：\n"
                "       确认嵌入模型已下载完整，并在 .env 中填写正确路径。\n"
                "       快速下载方式（在 venv 环境中执行）：\n"
                f"         {_VENV_PYTHON} -c \"\n"
                "           from huggingface_hub import snapshot_download\n"
                "           snapshot_download('Qwen/Qwen3-Embedding-0.6B',\n"
                "                             local_dir='./models/Qwen3-Embedding-0.6B')\n"
                "         \"\n"
                "       国内用户请先设置镜像：\n"
                "         set HF_ENDPOINT=https://hf-mirror.com  (Windows)\n"
                "         export HF_ENDPOINT=https://hf-mirror.com  (Linux/macOS)"
            ),
        )
        return False

    _pass("模型路径存在", detail=str(model_path))

    # 进一步检查目录内是否有 config.json（模型完整性的基础标志）
    config_json = model_path / "config.json"
    if not config_json.exists():
        _warn(
            "模型目录下未找到 config.json，模型文件可能不完整",
            detail=f"路径：{model_path}\n       如果模型下载中断，请重新下载。",
        )
        # 不视为致命错误，降级为警告，允许继续启动（可能是非标准模型目录结构）

    else:
        _pass("模型目录包含 config.json（文件完整性基础验证通过）")

    return True


# =============================================================================
# 检查项 4：本地推理服务可达
# =============================================================================

def check_local_model_service() -> bool:
    """
    向本地推理服务的 /v1/models 端点发送 GET 请求，验证服务可达。

    超时设为 3 秒（足够宽裕，避免误报）。
    只验证网络连通性，不验证具体模型是否加载。

    返回：True=通过，False=失败
    """
    import re
    import urllib.request
    import urllib.error

    _section("检查 4 / 5：本地推理服务")

    env_path = _ROOT_DIR / ".env"
    env_vars = _load_env(env_path)
    api_url  = env_vars.get("LOCAL_API_URL", "").strip()

    if not api_url:
        _fail("LOCAL_API_URL 未填写（应已在检查 2 中报错）")
        return False

    # 从 chat completions 地址推断 /v1/models 地址
    # 例：http://localhost:1234/v1/chat/completions → http://localhost:1234/v1/models
    models_url = re.sub(r"/v1/.*$", "/v1/models", api_url)
    if "/v1/" not in models_url:
        # 地址格式不标准，直接在末尾追加
        models_url = api_url.rstrip("/") + "/v1/models"

    print(f"       探测地址：{models_url}")

    try:
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            status = resp.status
            if status == 200:
                _pass(f"推理服务可达（HTTP {status}）", detail=models_url)
                return True
            else:
                # 非 200 但有响应，说明服务在运行但端点异常（较罕见）
                _warn(
                    f"推理服务响应了但状态码异常（HTTP {status}）",
                    detail="服务可能在运行，但建议确认模型已正常加载。",
                )
                return True  # 有响应就视为可达，降级为警告

    except urllib.error.URLError as e:
        reason = str(e.reason) if hasattr(e, "reason") else str(e)
        _fail(
            "无法连接到本地推理服务",
            detail=reason,
            fix=(
                "修复步骤：\n"
                "       请先启动推理服务：\n"
                "         LM Studio：打开 → Local Server → 选择模型 → Start Server\n"
                "         Ollama    ：ollama serve\n"
                "       确认服务启动后，重新运行本检查脚本。\n"
                f"       探测地址：{models_url}"
            ),
        )
        return False

    except Exception as e:
        _fail("推理服务检查时发生未预期错误", detail=str(e))
        return False


# =============================================================================
# 检查项 5：数据库存在且表结构完整
# =============================================================================

def check_database() -> bool:
    """
    检查 assistant.db 存在，并执行 PRAGMA integrity_check 验证文件完整性。
    同时检查 12 张核心表是否全部存在。

    不存在时提示运行 setup_db.py，不在此处自动初始化，
    保持健康检查与数据库初始化职责分离。

    返回：True=通过，False=失败
    """
    _section("检查 5 / 5：数据库")

    # 尝试从 config 读取路径，失败时使用默认路径
    try:
        from ramaria.config import DB_PATH
        db_path = Path(DB_PATH)
    except ImportError:
        db_path = _ROOT_DIR / "data" / "assistant.db"

    if not db_path.exists():
        _fail(
            "数据库文件不存在",
            detail=str(db_path),
            fix=(
                "修复步骤：\n"
                "       运行数据库初始化脚本：\n"
                f"         {_VENV_PYTHON} scripts/setup_db.py"
            ),
        )
        return False

    _pass("数据库文件存在", detail=str(db_path))

    # PRAGMA integrity_check：验证 SQLite 文件本身无损坏
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("PRAGMA integrity_check")
        results = [row[0] for row in cursor.fetchall()]

        if results == ["ok"]:
            _pass("数据库完整性检查通过（PRAGMA integrity_check = ok）")
        else:
            _fail(
                "数据库完整性检查失败",
                detail="\n       ".join(results),
                fix=(
                    "修复步骤：\n"
                    "       数据库文件可能已损坏，建议备份后重新初始化：\n"
                    f"         cp {db_path} {db_path}.backup\n"
                    f"         {_VENV_PYTHON} scripts/setup_db.py --force-rebuild"
                ),
            )
            conn.close()
            return False

    except sqlite3.DatabaseError as e:
        _fail(
            "数据库文件无法打开（可能已损坏）",
            detail=str(e),
            fix=(
                "修复步骤：\n"
                "       备份并重新初始化数据库：\n"
                f"         {_VENV_PYTHON} scripts/setup_db.py --force-rebuild"
            ),
        )
        return False

    # 检查 12 张核心表是否存在
    expected_tables = {
        "sessions", "messages", "memory_l1", "memory_l2", "l2_sources",
        "user_profile", "keyword_pool", "graph_nodes", "graph_edges",
        "conflict_queue", "pending_push", "settings",
    }

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    missing_tables = expected_tables - existing_tables

    if missing_tables:
        _fail(
            f"数据库缺少 {len(missing_tables)} 张表",
            detail=f"缺失：{', '.join(sorted(missing_tables))}",
            fix=(
                "修复步骤：\n"
                "       运行数据库迁移脚本补齐缺失的表：\n"
                f"         {_VENV_PYTHON} scripts/setup_db.py"
            ),
        )
        return False

    _pass(f"全部 {len(expected_tables)} 张核心表存在")
    return True


# =============================================================================
# 主入口
# =============================================================================

def main() -> int:
    """
    执行全部五项检查，返回退出码。

    退出码：
        0 — 全部通过
        1 — 有失败项

    设计：检查项之间有依赖关系，失败时立即退出，不继续执行后续检查：
        · 检查 1 失败 → venv 不存在，后续 import ramaria 均会失败，无需继续
        · 检查 2 失败 → .env 缺失，检查 3/4 无法读取配置，无需继续
        · 检查 3/4/5 失败 → 独立，互不影响，仍继续执行后续检查
    """
    print()
    print(_bold("=" * 52))
    print(_bold("  珊瑚菌 Ramaria — 启动前健康检查"))
    print(_bold("=" * 52))

    # 检查 1：venv（失败则立即退出）
    if not check_venv():
        _print_summary(passed=0, failed=1, total=5, early_exit=True)
        return 1

    # 检查 2：.env（失败则立即退出，因为后续需要读取配置）
    if not check_env_file():
        _print_summary(passed=1, failed=1, total=5, early_exit=True)
        return 1

    # 检查 3/4/5：独立执行，收集所有结果后再汇报
    passed = 2  # 检查 1/2 已通过
    failed = 0

    for check_fn in (check_embedding_model, check_local_model_service, check_database):
        if check_fn():
            passed += 1
        else:
            failed += 1

    _print_summary(passed=passed, failed=failed, total=5)
    return 0 if failed == 0 else 1


def _print_summary(passed: int, failed: int, total: int, early_exit: bool = False) -> None:
    """打印汇总行。"""
    print()
    print(_bold("=" * 52))
    if early_exit:
        print(f"  检查中止（关键项失败，跳过剩余检查）")
    elif failed == 0:
        print(f"  {_green('全部检查通过')} ✓  ({passed}/{total})")
    else:
        print(f"  {_red(f'有 {failed} 项检查失败')}  ({passed}/{total} 通过)")
    print(_bold("=" * 52))
    print()


if __name__ == "__main__":
    sys.exit(main())