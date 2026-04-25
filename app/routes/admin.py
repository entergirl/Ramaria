"""
app/routes/admin.py — 管理与配置 API 路由

提供以下接口，供配置向导前端和系统托盘调用：

    GET  /api/admin/status          — 执行全部检测，返回检测结果
    GET  /api/admin/config          — 读取当前 .env 所有值（前端回显用）
    POST /api/admin/config          — 保存配置（写入 .env）
    POST /api/admin/init_db         — 初始化/迁移数据库
    GET  /api/admin/setup/page      — 返回配置向导 HTML 页面
    GET  /api/admin/launcher/page   — 返回启动过渡页 HTML 页面
    GET  /api/persona/get           — 读取当前 persona.toml
    POST /api/persona/save          — 保存 persona.toml
    GET  /api/persona/example       — 读取 persona.toml.example（配置向导用）

设计原则：
    · 所有配置写操作通过 env_checker.write_env() 统一处理
    · 初始化 DB 在打包模式下进程内调用，开发模式下子进程调用
    · 返回统一的 {ok, message, data} JSON 结构，方便前端判断

路径说明（打包模式兼容）：
    · 开发模式：路径从 __file__ 推算（app/routes/ → 项目根）
    · 打包模式：路径从 ramaria.config 获取（由 bundle.py._patch_config_paths() 修复）
      静态文件（HTML/CSS/JS）从 _MEIPASS 加载（只读资源）
      用户数据（.env、config/、data/）从 exe 同级目录加载（可写）
"""

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.core.env_checker import (
    can_start_directly,
    get_all_env_values,
    run_all_checks,
    write_env,
)

from ramaria.logger import get_logger

logger  = get_logger(__name__)
router  = APIRouter()

# =============================================================================
# 路径常量（支持打包模式和开发模式）
# =============================================================================
_IS_FROZEN = getattr(sys, "frozen", False)

if _IS_FROZEN:
    # 打包模式：
    #   · 静态文件从 _MEIPASS 加载（只读，打包时嵌入）
    #   · 用户数据路径从 ramaria.config 获取（已由 bundle.py 修复）
    _MEIPASS      = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    _STATIC_DIR   = _MEIPASS / "static"

    from ramaria.config import CONFIG_DIR
    _PERSONA_PATH    = CONFIG_DIR / "persona.toml"
    _PERSONA_EXAMPLE = CONFIG_DIR / "persona.toml.example"
else:
    # 开发模式：从 __file__ 推算
    _ROOT           = Path(__file__).resolve().parents[2]
    _STATIC_DIR     = _ROOT / "static"
    _PERSONA_PATH   = _ROOT / "config" / "persona.toml"
    _PERSONA_EXAMPLE = _ROOT / "config" / "persona.toml.example"


# =============================================================================
# 请求/响应模型
# =============================================================================

class ConfigPayload(BaseModel):
    """POST /api/admin/config 的请求体。"""
    values: dict[str, str]


# =============================================================================
# 通用响应构建
# =============================================================================

def _reload_config_values(values: dict[str, str]) -> None:
    """
    配置保存后刷新 ramaria.config 模块级常量。

    ramaria.config 中的变量（LOCAL_MODEL_NAME 等）是模块级常量，
    在 import 时从 os.environ.get() 读取一次，后续不会自动更新。
    此函数在 .env 写入后手动同步这些变量，使 llm_client 等模块
    无需重启即可使用新值。
    """
    import ramaria.config as config

    _CONFIG_KEY_MAP = {
        "LOCAL_API_URL":            "LOCAL_API_URL",
        "LOCAL_MODEL_NAME":         "LOCAL_MODEL_NAME",
        "LOCAL_TEMPERATURE":        "LOCAL_TEMPERATURE",
        "LOCAL_MAX_TOKENS_SUMMARY": "LOCAL_MAX_TOKENS_SUMMARY",
        "LOCAL_MAX_TOKENS_CHAT":    "LOCAL_MAX_TOKENS_CHAT",
        "SERVER_HOST":              "SERVER_HOST",
        "SERVER_PORT":              "SERVER_PORT",
        "EMBEDDING_MODEL":          "EMBEDDING_MODEL",
        "ANTHROPIC_API_KEY":        "ANTHROPIC_API_KEY",
        "WEATHER_CITY":             "WEATHER_CITY",
    }

    _TYPE_CONVERTERS = {
        "LOCAL_TEMPERATURE":        float,
        "LOCAL_MAX_TOKENS_SUMMARY": int,
        "LOCAL_MAX_TOKENS_CHAT":    int,
        "SERVER_PORT":              int,
    }

    for env_key, config_attr in _CONFIG_KEY_MAP.items():
        if env_key in values:
            new_value = values[env_key]
            converter = _TYPE_CONVERTERS.get(config_attr)
            if converter:
                try:
                    new_value = converter(new_value)
                except (ValueError, TypeError):
                    continue
            setattr(config, config_attr, new_value)


def _ok(message: str = "ok", data: dict | None = None) -> JSONResponse:
    return JSONResponse({"ok": True, "message": message, "data": data or {}})


def _fail(message: str, detail: str = "", data: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "message": message, "detail": detail, "data": data or {}},
        status_code=400,
    )


# =============================================================================
# 环境检测接口
# =============================================================================

@router.get("/api/admin/status")
async def admin_status():
    """
    执行全部环境检测，返回每一项的 ok / message / detail。
    同时返回顶层 can_start 字段，供前端决定是否显示「立即启动」按钮。
    """
    checks    = run_all_checks()
    can_start, failed_items = can_start_directly()

    return JSONResponse({
        "ok":         can_start,
        "can_start":  can_start,
        "failed":     failed_items,
        "checks":     checks,
    })


# =============================================================================
# 配置读写接口
# =============================================================================

@router.get("/api/admin/config")
async def admin_get_config():
    """读取当前 .env 的全部值，供配置向导页面回显已有配置。"""
    values = get_all_env_values()
    return JSONResponse({"ok": True, "values": values})


@router.post("/api/admin/config")
async def admin_save_config(payload: ConfigPayload):
    """
    接收前端提交的配置表单，写入 .env 文件。
    只写入 payload.values 中的字段，不影响其他已有配置项。
    """
    if not payload.values:
        return _fail("请求体为空，未写入任何配置")

    try:
        write_env(payload.values)
        logger.info(f"配置已更新：{list(payload.values.keys())}")
    except Exception as e:
        logger.error(f"写入 .env 失败 — {e}")
        return _fail("写入配置失败", detail=str(e))

    # 刷新 ramaria.config 模块级常量，使 llm_client 等模块立即使用新值
    _reload_config_values(payload.values)

    from app.core.env_checker import check_embedding_model, check_env_file
    return JSONResponse({
        "ok":      True,
        "message": "配置已保存",
        "checks": {
            "env_file":        check_env_file().to_dict(),
            "embedding_model": check_embedding_model().to_dict(),
        },
    })


# =============================================================================
# 数据库初始化接口
# =============================================================================

@router.post("/api/admin/init_db")
async def admin_init_db():
    """
    初始化/迁移数据库。

    打包模式：直接在进程内调用 setup_db.main()。
    开发模式：以子进程调用 scripts/setup_db.py。

    同时在数据库初始化前确保 persona.toml 存在：
      · 如果 persona.toml 已存在 → 不覆盖
      · 如果不存在但 persona.toml.example 存在 → 从 example 复制
      · 两者均不存在 → 跳过（不影响数据库初始化）
    """
    # 确保 persona.toml 存在（不覆盖已有配置）
    _ensure_persona_exists()

    if getattr(sys, "frozen", False):
        # 打包模式：直接在进程内调用
        return _init_db_in_process()
    else:
        # 开发模式：子进程调用
        return _init_db_subprocess()


def _init_db_in_process() -> JSONResponse:
    """打包模式下直接在进程内调用 setup_db.main()。"""
    import io

    try:
        from app.core.db_initializer import _import_setup_db
        setup_db = _import_setup_db()

        # 重定向 stdout/stderr，防止 print 污染日志
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        try:
            sys.stdout = captured_stdout  # type: ignore[assignment]
            sys.stderr = captured_stderr  # type: ignore[assignment]
            setup_db.main()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        logger.info("数据库初始化完成")
        return _ok("数据库初始化完成")

    except Exception as e:
        logger.error(f"数据库初始化失败：{e}")
        return _fail("数据库初始化失败", detail=str(e))


def _init_db_subprocess() -> JSONResponse:
    """开发模式下以子进程调用 scripts/setup_db.py。"""
    import os

    # 开发模式下的项目根目录和脚本路径
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    setup_script = scripts_dir / "setup_db.py"
    root_dir = scripts_dir.parent

    if not setup_script.exists():
        return _fail("未找到 scripts/setup_db.py", detail="请确认项目文件完整。")

    # 设置 PYTHONIOENCODING=utf-8 防止 Windows GBK 编码错误
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode == 0:
            logger.info("数据库初始化完成")
            return _ok("数据库初始化完成", data={"stdout": result.stdout[-500:]})
        else:
            logger.error(f"数据库初始化失败：{result.stderr}")
            return _fail(
                "数据库初始化失败",
                detail=result.stderr[-500:],
                data={"returncode": result.returncode},
            )
    except subprocess.TimeoutExpired:
        return _fail("数据库初始化超时（>30s）")
    except Exception as e:
        return _fail("数据库初始化异常", detail=str(e))


def _ensure_persona_exists() -> None:
    """
    确保 persona.toml 存在：
      · 已存在 → 不做任何操作（不覆盖用户自定义）
      · 不存在 + example 存在 → 从 example 复制
      · 两者均不存在 → 静默跳过
    """
    if _PERSONA_PATH.exists():
        logger.debug("persona.toml 已存在，跳过自动创建")
        return

    _PERSONA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _PERSONA_EXAMPLE.exists():
        import shutil
        try:
            shutil.copy(_PERSONA_EXAMPLE, _PERSONA_PATH)
            logger.info(f"persona.toml 已从 example 复制：{_PERSONA_PATH}")
        except Exception as e:
            logger.warning(f"复制 persona.toml.example 失败 — {e}")
    else:
        logger.warning(
            "persona.toml 和 persona.toml.example 均不存在，跳过自动创建。"
            "服务启动时可能因人格配置缺失而报错。"
        )


# =============================================================================
# 页面接口
# =============================================================================

@router.get("/api/admin/setup/page", response_class=HTMLResponse)
async def admin_setup_page():
    """返回配置向导 HTML 页面（static/setup.html）。"""
    setup_html = _STATIC_DIR / "setup.html"
    if not setup_html.exists():
        return HTMLResponse("<h1>setup.html 未找到</h1>", status_code=404)
    return HTMLResponse(content=setup_html.read_text(encoding="utf-8"))


@router.get("/api/admin/launcher/page", response_class=HTMLResponse)
async def admin_launcher_page():
    """返回启动过渡页 HTML（static/launcher.html）。"""
    launcher_html = _STATIC_DIR / "launcher.html"
    if not launcher_html.exists():
        return HTMLResponse("<h1>launcher.html 未找到</h1>", status_code=404)
    return HTMLResponse(content=launcher_html.read_text(encoding="utf-8"))


# =============================================================================
# persona.toml 读写接口
# =============================================================================

@router.get("/api/persona/get")
async def get_persona():
    """
    读取当前 persona.toml 内容。
    文件不存在时尝试返回 example 内容；均不存在时返回 404。
    """
    # 优先读取用户自定义版本
    if _PERSONA_PATH.exists():
        return Response(
            content=_PERSONA_PATH.read_text(encoding="utf-8"),
            media_type="text/plain",
        )

    # 降级到 example
    if _PERSONA_EXAMPLE.exists():
        return Response(
            content=_PERSONA_EXAMPLE.read_text(encoding="utf-8"),
            media_type="text/plain",
        )

    return Response(status_code=404, content="persona.toml 不存在")


@router.post("/api/persona/save")
async def save_persona(request: Request):
    """
    保存 persona.toml 内容。
    自动创建 config/ 目录（如不存在）。
    """
    content = await request.body()
    text    = content.decode("utf-8").strip()

    if not text:
        return JSONResponse({"ok": False, "message": "内容不能为空"}, status_code=400)

    try:
        _PERSONA_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PERSONA_PATH.write_text(text, encoding="utf-8")
        logger.info(f"persona.toml 已保存：{_PERSONA_PATH}")
        return JSONResponse({"ok": True, "message": "保存成功"})
    except Exception as e:
        logger.error(f"保存 persona.toml 失败 — {e}")
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


@router.get("/api/persona/example")
async def get_persona_example():
    """
    读取 persona.toml.example 内容，供配置向导「恢复默认」使用。
    文件不存在时返回 404（前端回退到 JS 内置模板）。
    """
    if _PERSONA_EXAMPLE.exists():
        return Response(
            content=_PERSONA_EXAMPLE.read_text(encoding="utf-8"),
            media_type="text/plain",
        )
    return Response(status_code=404, content="persona.toml.example 不存在")
