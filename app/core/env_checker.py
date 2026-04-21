"""
app/core/env_checker.py — 环境状态检测模块

职责：
    集中管理应用启动前和运行时的环境检测逻辑，
    供 admin.py 路由和 bundle.py 启动器两处复用。

检测项：
    1. Python 版本（>= 3.10）
    2. 虚拟环境是否已创建（venv 目录）
    3. 关键依赖是否已安装（fastapi / chromadb 等）
    4. .env 文件是否存在，必填项是否已填写
    5. 嵌入模型路径是否存在（目录 + config.json 存在性）
    6. 本地推理服务是否可达（GET /v1/models，超时 3s）
    7. 数据库是否存在且完整（sqlite3 integrity_check）
    8. 端口 8000（或配置的端口）是否被占用

对外接口：
    run_all_checks()      → dict[str, CheckResult]  完整检测，供前端轮询
    can_start_directly()  → (bool, list[str])        快速判断能否直接启动
    get_env_value(key)    → str | None               读取 .env 中的值
    get_all_env_values()  → dict                     读取全部 .env 值
    write_env(data)       → None                     写入/更新 .env
"""

from __future__ import annotations

import platform
import re
import socket
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── 路径常量 ────────────────────────────────────────────────────────────────
# 本文件在 app/core/，项目根目录是上两级
_ROOT         = Path(__file__).resolve().parents[2]
_ENV_FILE     = _ROOT / ".env"
_VENV_DIR     = _ROOT / "venv"
_DB_PATH      = _ROOT / "data" / "assistant.db"

# venv 内 Python 可执行文件（Windows / Unix 路径不同）
_IS_WINDOWS   = platform.system() == "Windows"
_VENV_PYTHON  = (
    _VENV_DIR / "Scripts" / "python.exe"
    if _IS_WINDOWS
    else _VENV_DIR / "bin" / "python"
)

# .env 中必须有值的三个配置项
_REQUIRED_ENV_KEYS = ["LOCAL_API_URL", "LOCAL_MODEL_NAME", "EMBEDDING_MODEL"]

# 将 /v1/chat/completions 等路径替换为 /v1/models 用于推理服务检测
_MODELS_PATH_RE = re.compile(r"/v1/.*$")


# =============================================================================
# 结果数据结构
# =============================================================================

@dataclass
class CheckResult:
    """单项检测的结果。"""
    ok:      bool        # True = 通过
    message: str         # 人类可读的结果描述
    detail:  str  = ""   # 扩展说明或修复建议
    data:    Any  = None # 可选的附加数据（如端口号、版本字符串）

    def to_dict(self) -> dict:
        return {
            "ok":      self.ok,
            "message": self.message,
            "detail":  self.detail,
            "data":    self.data,
        }


# =============================================================================
# .env 读写工具（公开接口）
# =============================================================================

def get_env_value(key: str) -> str | None:
    """
    从 .env 文件读取指定 key 的值。
    去除行尾注释、首尾引号和空白。文件不存在或 key 不存在时返回 None。
    """
    if not _ENV_FILE.exists():
        return None

    with open(_ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                v = v.split("#")[0].strip().strip("'\"")
                return v or None
    return None


def get_all_env_values() -> dict[str, str]:
    """
    读取 .env 文件中所有 KEY=VALUE 行，返回字典。
    注释行和空行忽略，值已去除首尾引号和行尾注释。
    """
    result: dict[str, str] = {}
    if not _ENV_FILE.exists():
        return result

    with open(_ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.split("#")[0].strip().strip("'\"")
            result[k.strip()] = v
    return result


def write_env(data: dict[str, str]) -> None:
    """
    将 data 中的键值对写入 .env 文件。

    策略：
      · .env 已存在 → 逐行扫描替换已有 key，找不到则在末尾追加
      · .env 不存在 → 先从 .env.example 复制（若存在），再写入
      · 空字符串值写为 KEY= 格式，不加引号

    参数：
        data — 要写入/更新的 {key: value} 字典
    """
    # 确保 data/ 目录存在
    (_ROOT / "data").mkdir(exist_ok=True)

    # .env 不存在时先从模板复制
    if not _ENV_FILE.exists():
        example = _ROOT / ".env.example"
        if example.exists():
            import shutil
            shutil.copy(example, _ENV_FILE)
        else:
            _ENV_FILE.touch()

    lines = _ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_keys: set[str] = set()

    # 替换已存在的 key
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in data:
            lines[i] = f"{k}={data[k]}\n"
            updated_keys.add(k)

    # 追加不存在的 key
    new_keys = [k for k in data if k not in updated_keys]
    if new_keys:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        for k in new_keys:
            lines.append(f"{k}={data[k]}\n")

    _ENV_FILE.write_text("".join(lines), encoding="utf-8")


# =============================================================================
# 各项检测函数（每个函数独立，返回 CheckResult）
# =============================================================================

def check_python_version() -> CheckResult:
    """检测 Python 版本 >= 3.10。"""
    major, minor  = sys.version_info.major, sys.version_info.minor
    version_str   = f"{major}.{minor}.{sys.version_info.micro}"

    if major < 3 or (major == 3 and minor < 10):
        return CheckResult(
            ok=False,
            message=f"Python 版本过低（当前 {version_str}）",
            detail="需要 Python 3.10 或以上版本。请访问 https://www.python.org/downloads/",
            data={"version": version_str},
        )
    return CheckResult(ok=True, message=f"Python {version_str}", data={"version": version_str})


def check_venv() -> CheckResult:
    """检测 venv 虚拟环境是否已创建。"""
    if _VENV_PYTHON.exists():
        return CheckResult(ok=True, message="虚拟环境已就绪", data={"path": str(_VENV_DIR)})
    return CheckResult(
        ok=False,
        message="未找到虚拟环境",
        detail="请先运行安装脚本：python win/install.py 或 bash linux/install.sh",
        data={"path": str(_VENV_DIR)},
    )


def check_dependencies() -> CheckResult:
    """
    检测关键依赖是否已安装。
    通过 importlib.import_module 逐个探测，不调用 pip，速度快。
    """
    import importlib

    required = [
        ("fastapi",               "FastAPI"),
        ("uvicorn",               "Uvicorn"),
        ("chromadb",              "ChromaDB"),
        ("sentence_transformers", "SentenceTransformers"),
        ("rank_bm25",             "rank-bm25"),
        ("jieba",                 "jieba"),
        ("networkx",              "NetworkX"),
        ("webview",               "pywebview"),
        ("pystray",               "pystray"),
        ("PIL",                   "Pillow"),
    ]

    missing = []
    for module, display_name in required:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(display_name)

    if missing:
        return CheckResult(
            ok=False,
            message=f"缺少 {len(missing)} 个依赖包",
            detail=f"缺少：{', '.join(missing)}\n请运行：pip install -e .[bundle]",
            data={"missing": missing},
        )
    return CheckResult(ok=True, message="所有依赖已安装", data={"missing": []})


def check_env_file() -> CheckResult:
    """检测 .env 文件是否存在，且三个必填项均已填写。"""
    if not _ENV_FILE.exists():
        return CheckResult(
            ok=False,
            message=".env 配置文件不存在",
            detail="请通过配置向导填写必要参数，向导将自动创建 .env 文件。",
        )

    env_vals = get_all_env_values()
    missing  = [k for k in _REQUIRED_ENV_KEYS if not env_vals.get(k, "").strip()]

    if missing:
        return CheckResult(
            ok=False,
            message=f"必填项未配置：{', '.join(missing)}",
            detail="请打开配置向导补充填写这些字段。",
            data={"missing": missing, "filled": [k for k in _REQUIRED_ENV_KEYS if k not in missing]},
        )
    return CheckResult(
        ok=True,
        message=".env 配置完整",
        data={"missing": [], "filled": _REQUIRED_ENV_KEYS},
    )


def check_embedding_model() -> CheckResult:
    """
    检测 EMBEDDING_MODEL 路径是否存在，
    且目录内包含 config.json（基础完整性验证）。
    """
    path_str = get_env_value("EMBEDDING_MODEL")
    if not path_str:
        return CheckResult(
            ok=False,
            message="嵌入模型路径未填写",
            detail="请在配置向导中填写嵌入模型的本地文件夹绝对路径。",
        )

    model_path = Path(path_str)
    if not model_path.exists():
        return CheckResult(
            ok=False,
            message="嵌入模型路径不存在",
            detail=f"路径 {model_path} 不存在。请确认模型已下载并填写正确的文件夹路径。",
            data={"path": str(model_path)},
        )

    if not (model_path / "config.json").exists():
        return CheckResult(
            ok=False,
            message="嵌入模型文件不完整",
            detail=f"目录 {model_path} 中未找到 config.json，模型可能下载不完整。",
            data={"path": str(model_path)},
        )

    return CheckResult(ok=True, message="嵌入模型路径有效", data={"path": str(model_path)})


def check_inference_service() -> CheckResult:
    """
    检测本地推理服务是否可达（GET /v1/models，超时 3s）。
    从 LOCAL_API_URL 推断 /v1/models 端点地址。
    """
    import urllib.error
    import urllib.request

    api_url = get_env_value("LOCAL_API_URL")
    if not api_url:
        return CheckResult(
            ok=False,
            message="LOCAL_API_URL 未配置",
            detail="请先在配置向导中填写推理服务地址。",
        )

    models_url = _MODELS_PATH_RE.sub("/v1/models", api_url)
    if "/v1/" not in models_url:
        models_url = api_url.rstrip("/") + "/v1/models"

    try:
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return CheckResult(ok=True, message="推理服务运行中", data={"url": models_url})
            return CheckResult(
                ok=False,
                message=f"推理服务响应异常（HTTP {resp.status}）",
                detail="服务可能正在启动，请稍后重试。",
                data={"url": models_url},
            )
    except urllib.error.URLError:
        return CheckResult(
            ok=False,
            message="推理服务未启动",
            detail=(
                "LM Studio：打开 → Local Server → 选择模型 → Start Server\n"
                "Ollama：运行 ollama serve"
            ),
            data={"url": models_url},
        )
    except Exception as e:
        return CheckResult(ok=False, message="推理服务检测失败", detail=str(e))


def check_database() -> CheckResult:
    """检测 assistant.db 是否存在且通过 integrity_check。"""
    if not _DB_PATH.exists():
        return CheckResult(
            ok=False,
            message="数据库不存在",
            detail="首次启动时将自动初始化数据库，无需手动创建。",
            data={"path": str(_DB_PATH), "needs_init": True},
        )

    try:
        conn = sqlite3.connect(str(_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        results = [row[0] for row in cursor.fetchall()]
        conn.close()

        if results == ["ok"]:
            return CheckResult(ok=True, message="数据库完整", data={"path": str(_DB_PATH), "needs_init": False})
        return CheckResult(
            ok=False,
            message="数据库文件损坏",
            detail="\n".join(results),
            data={"path": str(_DB_PATH), "needs_init": False},
        )
    except sqlite3.DatabaseError as e:
        return CheckResult(ok=False, message="数据库无法打开", detail=str(e))


def check_port(port: int = 8000) -> CheckResult:
    """检测指定端口是否被占用。被占用时返回 ok=False。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return CheckResult(
                    ok=False,
                    message=f"端口 {port} 已被占用",
                    detail=f"请关闭占用端口 {port} 的程序，或在 .env 中修改 SERVER_PORT。",
                    data={"port": port},
                )
        return CheckResult(ok=True, message=f"端口 {port} 可用", data={"port": port})
    except Exception as e:
        # 检测失败不阻断启动，视为可用
        return CheckResult(ok=True, message="端口检测跳过", detail=str(e), data={"port": port})


# =============================================================================
# 聚合接口（公开）
# =============================================================================

def run_all_checks() -> dict[str, dict]:
    """
    执行全部检测项，返回可直接 JSON 序列化的字典。
    供 admin.py 的 GET /api/admin/status 接口调用。
    """
    try:
        port = int(get_env_value("SERVER_PORT") or "8000")
    except (ValueError, TypeError):
        port = 8000

    checks: dict[str, CheckResult] = {
        "python":            check_python_version(),
        "venv":              check_venv(),
        "dependencies":      check_dependencies(),
        "env_file":          check_env_file(),
        "embedding_model":   check_embedding_model(),
        "inference_service": check_inference_service(),
        "database":          check_database(),
        "port":              check_port(port),
    }

    return {name: result.to_dict() for name, result in checks.items()}


def can_start_directly() -> tuple[bool, list[str]]:
    """
    快速判断是否可以跳过向导直接启动服务。
    只检测阻塞启动的必要项，不检测推理服务（允许先启动再连接）。

    返回：
        (True,  [])          — 可直接启动
        (False, [失败项名])  — 有阻塞项
    """
    try:
        port = int(get_env_value("SERVER_PORT") or "8000")
    except (ValueError, TypeError):
        port = 8000

    blocking: dict[str, CheckResult] = {
        "env_file":        check_env_file(),
        "embedding_model": check_embedding_model(),
        "port":            check_port(port),
    }

    failed = [name for name, result in blocking.items() if not result.ok]
    return (len(failed) == 0, failed)
