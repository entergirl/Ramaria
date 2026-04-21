"""
app/routes/admin.py — 管理与配置 API 路由

提供以下接口，供配置向导前端和系统托盘调用：

    GET  /api/admin/status          — 执行全部检测，返回检测结果
    GET  /api/admin/config          — 读取当前 .env 所有值（前端回显用）
    POST /api/admin/config          — 保存配置（写入 .env）
    POST /api/admin/init_db         — 初始化/迁移数据库
    GET  /api/admin/setup/page      — 返回配置向导 HTML 页面
    GET  /api/admin/launcher/page   — 返回启动过渡页 HTML 页面

设计原则：
    · 所有配置写操作通过 env_checker.write_env() 统一处理
    · 初始化 DB 调用已有的 scripts/setup_db.py（以子进程运行，隔离异常）
    · 返回统一的 {ok, message, data} JSON 结构，方便前端判断
"""

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter
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

# 项目根目录（本文件在 app/routes/，向上两级）
_ROOT        = Path(__file__).resolve().parents[2]
_STATIC_DIR  = _ROOT / "static"
_SCRIPTS_DIR = _ROOT / "scripts"


# =============================================================================
# 请求/响应模型
# =============================================================================

class ConfigPayload(BaseModel):
    """
    POST /api/admin/config 的请求体。
    使用宽松的 dict 结构，允许前端只传需要更新的字段。
    """
    values: dict[str, str]


# =============================================================================
# 通用响应构建
# =============================================================================

def _ok(message: str = "ok", data: dict | None = None) -> JSONResponse:
    return JSONResponse({"ok": True, "message": message, "data": data or {}})


def _fail(message: str, detail: str = "", data: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "message": message, "detail": detail, "data": data or {}},
        status_code=400,
    )


# =============================================================================
# 接口实现
# =============================================================================

@router.get("/api/admin/status")
async def admin_status():
    """
    执行全部环境检测，返回每一项的 ok / message / detail。
    同时返回顶层 can_start 字段，供前端决定是否显示"立即启动"按钮。

    前端每隔 2s 轮询此接口以实时更新状态显示。
    """
    checks    = run_all_checks()
    can_start, failed_items = can_start_directly()

    return JSONResponse({
        "ok":         can_start,
        "can_start":  can_start,
        "failed":     failed_items,
        "checks":     checks,
    })


@router.get("/api/admin/config")
async def admin_get_config():
    """
    读取当前 .env 的全部值，供配置向导页面回显已有配置。

    注意：敏感字段（API Key）原样返回，前端自行处理是否遮罩显示。
    """
    values = get_all_env_values()
    return JSONResponse({"ok": True, "values": values})


@router.post("/api/admin/config")
async def admin_save_config(payload: ConfigPayload):
    """
    接收前端提交的配置表单，写入 .env 文件。

    只写入 payload.values 中的字段，不影响其他已有配置项。
    写入完成后重新检测 env_file 和 embedding_model 项，返回最新状态。
    """
    if not payload.values:
        return _fail("请求体为空，未写入任何配置")

    try:
        write_env(payload.values)
        logger.info(f"配置已更新：{list(payload.values.keys())}")
    except Exception as e:
        logger.error(f"写入 .env 失败 — {e}")
        return _fail("写入配置失败", detail=str(e))

    # 写入后立刻重新检测，把最新结果返回给前端
    from app.core.env_checker import check_embedding_model, check_env_file
    return JSONResponse({
        "ok":      True,
        "message": "配置已保存",
        "checks": {
            "env_file":        check_env_file().to_dict(),
            "embedding_model": check_embedding_model().to_dict(),
        },
    })


@router.post("/api/admin/init_db")
async def admin_init_db():
    """
    以子进程调用 scripts/setup_db.py 初始化/迁移数据库。
    子进程隔离，避免 setup_db 的 sys.exit() 影响主进程。

    首次使用或数据库缺失时由前端在向导最后一步调用。
    """
    setup_script = _SCRIPTS_DIR / "setup_db.py"
    if not setup_script.exists():
        return _fail("未找到 scripts/setup_db.py", detail="请确认项目文件完整。")

    try:
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
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
