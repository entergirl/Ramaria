"""
app/core/env_checker.py — 环境状态检测模块

职责：
    集中管理应用启动前和运行时的环境检测逻辑，
    供 admin.py 路由和 bundle.py 启动器两处复用。

检测项：
    1. Python 版本（>= 3.10）
    2. 虚拟环境是否已创建（venv 目录）—— 打包模式下跳过
    3. 关键依赖是否已安装（fastapi / chromadb 等）—— 打包模式下跳过
    4. .env 文件是否存在，必填项是否已填写
    5. 嵌入模型路径是否存在（目录 + config.json 存在性）
    6. 本地推理服务是否可达（GET /v1/models，超时 3s）
    7. 数据库是否存在且完整（sqlite3 integrity_check + 关键列校验）
    8. 端口 8000（或配置的端口）是否被占用

打包模式（PyInstaller exe）路径说明：
    · 打包后 ramaria.config 中的路径由 bundle.py._patch_config_paths() 修复，
      指向 exe 同级可写目录（data/、config/、logs/ 等）。
    · 本模块的路径常量（_ROOT / _ENV_FILE / _DB_PATH 等）从 ramaria.config
      动态获取，确保打包模式和开发模式路径一致。
    · 不再从 __file__ 推算路径（打包后 __file__ 指向 _MEIPASS 只读目录）。

对外接口：
    run_all_checks()      → dict[str, CheckResult]  完整检测，供前端轮询
    can_start_directly()  → (bool, list[str])        快速判断能否直接启动
    get_env_value(key)    → str | None               读取 .env 中的值
    get_all_env_values()  → dict                     读取全部 .env 值
    write_env(data)       → None                     写入/更新 .env
"""

from __future__ import annotations

import os
import platform
import re
import socket
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# 路径常量（支持打包模式和开发模式）
# =============================================================================
# 关键设计：打包后 __file__ 指向 _MEIPASS 只读目录，不能用来推算用户数据路径。
# 必须从 ramaria.config 获取路径（由 bundle.py._patch_config_paths() 修复）。
#
# 但 ramaria.config 在首次 import 时会执行 os.environ.get()，
# 所以必须在 bundle.py._load_dotenv() 之后才能 import。
# 此处使用延迟计算：_get_root() / _get_env_file() / _get_db_path()
# 在首次调用时才 import ramaria.config 并缓存结果。

_cached_root: Path | None = None
_cached_env_file: Path | None = None
_cached_db_path: Path | None = None

_IS_FROZEN = getattr(sys, "frozen", False)


def _get_root() -> Path:
    """
    获取项目根目录（延迟计算，首次调用时缓存）。

    · 打包模式：从 ramaria.config.ROOT_DIR 获取（已由 bundle.py 修复为 exe 同级目录）
    · 开发模式：从 __file__ 推算（app/core/ → app/ → 项目根）
    """
    global _cached_root
    if _cached_root is not None:
        return _cached_root

    if _IS_FROZEN:
        from ramaria.config import ROOT_DIR
        _cached_root = ROOT_DIR
    else:
        _cached_root = Path(__file__).resolve().parents[2]

    return _cached_root


def _get_env_file() -> Path:
    """
    获取 .env 文件路径（延迟计算，首次调用时缓存）。

    · 打包模式：exe 同级目录的 .env（用户实际配置）
    · 开发模式：项目根目录的 .env
    """
    global _cached_env_file
    if _cached_env_file is not None:
        return _cached_env_file

    if _IS_FROZEN:
        # 打包模式：.env 在 exe 同级目录（由 bundle.py 复制/用户编辑）
        _cached_env_file = Path(sys.executable).parent / ".env"
    else:
        _cached_env_file = _get_root() / ".env"

    return _cached_env_file


def _get_db_path() -> Path:
    """
    获取数据库文件路径（延迟计算，首次调用时缓存）。

    · 打包模式：从 ramaria.config.DB_PATH 获取（已由 bundle.py 修复）
    · 开发模式：项目根目录 / data / assistant.db
    """
    global _cached_db_path
    if _cached_db_path is not None:
        return _cached_db_path

    if _IS_FROZEN:
        from ramaria.config import DB_PATH
        _cached_db_path = DB_PATH
    else:
        _cached_db_path = _get_root() / "data" / "assistant.db"

    return _cached_db_path


# venv 相关路径（仅开发模式使用，打包模式下 check_venv() 直接跳过）
_VENV_DIR = _get_root() / "venv" if not _IS_FROZEN else Path("")

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
# 数据库列级校验清单（v0.6.0 新增）
# =============================================================================
# 格式：(表名, 列名, 说明)
# 这些列在旧版数据库（v0.3.x 之前）可能不存在，迁移脚本负责补齐。
# 此处只做"是否存在"的检查，不做修复——修复由 ensure_db_ready() 负责。
_REQUIRED_COLUMNS: list[tuple[str, str, str]] = [
    # messages 表新增列（v0.3.x 历史导入功能）
    ("messages", "source",             "消息来源字段，区分本地/线上/导入"),
    ("messages", "import_fingerprint", "历史导入去重指纹"),
    # memory_l1 表情感元数据列（v0.3.x 情感功能）
    ("memory_l1", "valence",           "情绪效价"),
    ("memory_l1", "salience",          "情感显著性"),
    ("memory_l1", "last_accessed_at",  "最近访问时间，用于衰减保底"),
    # memory_l2 表访问时间列
    ("memory_l2", "last_accessed_at",  "最近访问时间，用于衰减保底"),
    # keyword_pool 别名归一化列（v0.3.x 图谱功能）
    ("keyword_pool", "canonical_id",   "规范词 rowid"),
    ("keyword_pool", "alias_status",   "别名状态"),
    # conflict_queue 冲突类型列
    ("conflict_queue", "conflict_type", "冲突来源类型"),
]


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
    env_file = _get_env_file()
    if not env_file.exists():
        return None

    with open(env_file, encoding="utf-8") as f:
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
    env_file = _get_env_file()
    if not env_file.exists():
        return result

    with open(env_file, encoding="utf-8") as f:
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
    root = _get_root()
    env_file = _get_env_file()

    (root / "data").mkdir(exist_ok=True)

    if not env_file.exists():
        example = root / ".env.example"
        if example.exists():
            import shutil
            shutil.copy(example, env_file)
        else:
            env_file.touch()

    lines = env_file.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_keys: set[str] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in data:
            lines[i] = f"{k}={data[k]}\n"
            updated_keys.add(k)

    new_keys = [k for k in data if k not in updated_keys]
    if new_keys:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        for k in new_keys:
            lines.append(f"{k}={data[k]}\n")

    env_file.write_text("".join(lines), encoding="utf-8")

    # 同步更新 os.environ，使当前进程中的 os.environ.get() 能读到最新值
    for k, v in data.items():
        os.environ[k] = v


# =============================================================================
# 各项检测函数
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
    """
    检测 venv 虚拟环境是否已创建。

    打包模式下自动跳过（exe 自带 Python 运行时，无需虚拟环境）。
    """
    if _IS_FROZEN:
        return CheckResult(ok=True, message="打包模式，无需虚拟环境")

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

    打包模式下自动跳过（exe 已包含所有依赖）。
    """
    if _IS_FROZEN:
        return CheckResult(ok=True, message="打包模式，所有依赖已内置")

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
    env_file = _get_env_file()
    if not env_file.exists():
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
    且目录内包含 config.json（确认是完整的模型目录）。
    """
    path_str = get_env_value("EMBEDDING_MODEL")
    if not path_str:
        return CheckResult(
            ok=False,
            message="嵌入模型路径未填写",
            detail="请在配置向导中填写嵌入模型的本地文件夹绝对路径（路径分隔符请使用 / ）。",
        )

    model_path = Path(path_str)
    # 统一显示路径使用正斜杠，保持跨平台一致性
    display_path = path_str.replace("\\", "/")
    if not model_path.exists():
        return CheckResult(
            ok=False,
            message="嵌入模型路径不存在",
            detail=f"路径 {display_path} 不存在。请确认模型已下载并填写正确的文件夹路径。",
            data={"path": display_path},
        )

    if not (model_path / "config.json").exists():
        return CheckResult(
            ok=False,
            message="嵌入模型文件不完整",
            detail=f"目录 {display_path} 中未找到 config.json，模型可能下载不完整。",
            data={"path": display_path},
        )

    return CheckResult(ok=True, message="嵌入模型路径有效", data={"path": display_path})


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
    """
    检测 assistant.db 是否存在、通过 integrity_check，
    以及 v0.3.x 之后新增的关键列是否都已迁移。

    v0.6.0 新增列级校验：
        检查 _REQUIRED_COLUMNS 中列出的所有列是否存在。
        若有缺失，说明数据库是旧版本且迁移未完成，
        返回警告（ok=False）并告知用户运行 setup_db.py 修复。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        return CheckResult(
            ok=False,
            message="数据库不存在",
            detail="首次启动时将自动初始化数据库，无需手动创建。",
            data={"path": str(db_path), "needs_init": True},
        )

    # ── integrity_check ──────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("PRAGMA integrity_check")
        results = [row[0] for row in cursor.fetchall()]

        if results != ["ok"]:
            conn.close()
            return CheckResult(
                ok=False,
                message="数据库文件损坏",
                detail="\n".join(results),
                data={"path": str(db_path), "needs_init": False},
            )

    except sqlite3.DatabaseError as e:
        return CheckResult(ok=False, message="数据库无法打开", detail=str(e))

    # ── 12 张核心表校验 ───────────────────────────────────────────────────────
    expected_tables = {
        "sessions", "messages", "memory_l1", "memory_l2", "l2_sources",
        "user_profile", "keyword_pool", "graph_nodes", "graph_edges",
        "conflict_queue", "pending_push", "settings",
    }

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    missing_tables  = expected_tables - existing_tables

    if missing_tables:
        conn.close()
        return CheckResult(
            ok=False,
            message=f"数据库缺少 {len(missing_tables)} 张表",
            detail=(
                f"缺失：{', '.join(sorted(missing_tables))}\n"
                f"修复：python scripts/setup_db.py"
            ),
            data={"path": str(db_path), "needs_init": False},
        )

    # ── v0.6.0 新增：关键列校验 ──────────────────────────────────────────────
    # 构建各表现有列的缓存，避免对同一张表重复 PRAGMA 查询
    table_columns: dict[str, set[str]] = {}

    def _get_columns(table: str) -> set[str]:
        """获取指定表的列名集合，结果缓存避免重复查询。"""
        if table not in table_columns:
            cursor.execute(f"PRAGMA table_info({table})")
            table_columns[table] = {row["name"] for row in cursor.fetchall()}
        return table_columns[table]

    missing_columns: list[str] = []
    for table, col, desc in _REQUIRED_COLUMNS:
        # 只检查已存在的表（表不存在的情况已在上面 missing_tables 里处理了）
        if table in existing_tables and col not in _get_columns(table):
            missing_columns.append(f"{table}.{col}（{desc}）")

    conn.close()

    if missing_columns:
        return CheckResult(
            ok=False,
            message=f"数据库缺少 {len(missing_columns)} 个字段（需要迁移）",
            detail=(
                f"缺失字段：\n  " + "\n  ".join(missing_columns) + "\n\n"
                f"这是因为数据库是旧版本创建的，新功能所需的列尚未添加。\n"
                f"修复方法：运行 python scripts/setup_db.py\n"
                f"（幂等操作，不会删除已有数据）"
            ),
            data={
                "path": str(db_path),
                "needs_init": False,
                "missing_columns": missing_columns,
            },
        )

    return CheckResult(
        ok=True,
        message="数据库完整",
        data={"path": str(db_path), "needs_init": False},
    )


def check_port(port: int = 8000) -> CheckResult:
    """
    检测指定端口是否可以使用。

    v0.6.0 修复逻辑：
        原实现直接判断"能连上就报占用"，导致 FastAPI 服务本身启动后
        调用此接口时（/api/admin/status），永远报告端口被占用。

        修复策略：
            1. 尝试连接端口，连不上 → 端口空闲，返回 ok=True（最常见路径）
            2. 连得上（端口被占用）→ 进一步用 psutil 检查占用进程：
               a. psutil 不可用 → 降级，尝试绑定端口
               b. 占用进程是本进程（os.getpid()）或父进程 → 端口被自己占，
                  返回 ok=True（FastAPI 已启动状态下调用此函数的正常情况）
               c. 占用进程是其他进程 → 返回 ok=False，报告进程名和 PID
            3. 端口绑定测试（psutil 不可用时的降级方案）：
               尝试 SO_REUSEADDR 绑定，成功说明端口可用（同进程可重用）

    参数：
        port — 要检测的端口号，默认 8000

    返回：
        CheckResult，ok=True 表示端口可用（或被本进程占用）
    """
    # 第一步：尝试连接，连不上直接返回"端口可用"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                # 连接失败 = 端口没有监听者 = 可用
                return CheckResult(ok=True, message=f"端口 {port} 可用", data={"port": port})
    except Exception:
        # socket 操作本身出错，视为可用（不阻断启动）
        return CheckResult(ok=True, message="端口检测跳过", data={"port": port})

    # 第二步：端口有监听者，判断是不是自己（uvicorn/本进程）
    own_pid    = os.getpid()
    own_ppid   = os.getppid() if hasattr(os, "getppid") else -1

    try:
        import psutil

        # 找出占用该端口的所有连接，提取对端 PID
        occupying_pids: set[int] = set()
        for conn in psutil.net_connections(kind="tcp"):
            # LISTEN 状态的连接，laddr.port 是监听端口
            if (
                conn.status == "LISTEN"
                and conn.laddr
                and conn.laddr.port == port
                and conn.pid is not None
            ):
                occupying_pids.add(conn.pid)

        if not occupying_pids:
            # 找不到具体 PID（权限问题等），降级视为可用
            return CheckResult(ok=True, message=f"端口 {port} 状态未知，视为可用", data={"port": port})

        # 判断是否是本进程或父进程在监听
        self_pids = {own_pid, own_ppid}
        foreign_pids = occupying_pids - self_pids

        if not foreign_pids:
            # 全部是自己的进程（uvicorn 运行中的正常状态）
            return CheckResult(
                ok=True,
                message=f"端口 {port} 由本服务占用（正常）",
                data={"port": port, "self_occupied": True},
            )

        # 有其他进程占用，报告进程信息
        proc_names: list[str] = []
        for pid in foreign_pids:
            try:
                proc = psutil.Process(pid)
                proc_names.append(f"{proc.name()}（PID {pid}）")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc_names.append(f"PID {pid}（无法获取进程名）")

        return CheckResult(
            ok=False,
            message=f"端口 {port} 被其他程序占用",
            detail=(
                f"占用进程：{', '.join(proc_names)}\n"
                f"请关闭占用程序，或在 .env 中修改 SERVER_PORT。\n"
                f"Windows 可用：netstat -ano | findstr :{port}\n"
                f"Linux/macOS：lsof -i:{port}"
            ),
            data={"port": port, "occupying_pids": list(foreign_pids)},
        )

    except ImportError:
        # psutil 不可用，降级：尝试用 SO_REUSEADDR 绑定端口
        # SO_REUSEADDR 允许同进程重用端口，如果是 uvicorn 占用则绑定仍可成功
        pass

    # psutil 不可用时的降级方案：尝试绑定
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
        # 绑定成功：端口对本进程可用（可能是被本进程的 uvicorn 占用）
        return CheckResult(
            ok=True,
            message=f"端口 {port} 可用（或由本服务占用）",
            data={"port": port},
        )
    except OSError:
        # 绑定失败：端口确实被其他程序占用，但没有进程信息
        return CheckResult(
            ok=False,
            message=f"端口 {port} 已被占用",
            detail=(
                f"无法确定占用进程（psutil 未安装）。\n"
                f"请安装 psutil 获取详情：pip install psutil\n"
                f"或手动检查：netstat -ano | findstr :{port}"
            ),
            data={"port": port},
        )


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