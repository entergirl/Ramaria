"""
app/routes/import_ctrl.py — 聊天记录导入接口

包含接口：
    GET  /import                   — 导入页面
    POST /import/qq/preview        — 解析预览（不写入数据库）
    POST /import/qq/confirm        — 确认导入 L0
    GET  /import/qq/pending_l1     — 查询待处理 session 数量
    POST /import/qq/start_l1       — 启动 L1 批量生成
    GET  /import/qq/l1_progress    — 查询批处理进度（前端轮询）
    POST /import/qq/stop_l1        — 停止批处理
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ramaria.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# 静态文件目录（相对于项目根目录）
_STATIC_DIR = Path(__file__).parents[3] / "static"


# =============================================================================
# GET /import — 导入页面
# =============================================================================

@router.get("/import", response_class=HTMLResponse)
async def import_page():
    """返回聊天记录导入页面（static/import.html）。"""
    import_html = _STATIC_DIR / "import.html"
    if not import_html.exists():
        raise HTTPException(status_code=503, detail="import.html 未找到")
    return HTMLResponse(content=import_html.read_text(encoding="utf-8"))


# =============================================================================
# POST /import/qq/preview — 解析预览
# =============================================================================

@router.post("/import/qq/preview")
async def import_qq_preview(
    file: UploadFile = File(...),
    gap: int = 10,
):
    """
    上传 QQ Chat Exporter JSON 文件，返回详细诊断报告。
    此接口不写入数据库，仅用于预览解析结果。

    参数：
        file — 上传的 JSON 文件（multipart/form-data）
        gap  — session 切割时间间隔（分钟），默认10
    """
    from ramaria.importer.qq.parser import parse_qq_export

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="wb"
        ) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = parse_qq_export(tmp_path, gap_minutes=gap)
        return JSONResponse(result.report.to_dict())

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"import/qq/preview 处理失败 — {e}")
        raise HTTPException(status_code=500, detail=f"解析失败：{e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# POST /import/qq/confirm — 确认导入 L0
# =============================================================================

@router.post("/import/qq/confirm")
async def import_qq_confirm(
    file: UploadFile = File(...),
    gap: int = 10,
):
    """
    上传 QQ Chat Exporter JSON 文件，解析后写入数据库 L0 层。
    不触发 L1 摘要生成，L1 由 /import/qq/start_l1 单独触发。
    """
    from ramaria.importer.qq.parser import parse_qq_export
    from ramaria.importer.qq.importer import import_sessions_to_db

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="wb"
        ) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = parse_qq_export(tmp_path, gap_minutes=gap)
        stats  = import_sessions_to_db(result.parsed_sessions)

        return JSONResponse({
            "status": "ok",
            "stats":  stats,
            "report_overview": {
                "total_raw":     result.report.total_raw,
                "session_count": result.report.session_count,
                "time_start":    result.report.time_start,
                "time_end":      result.report.time_end,
            },
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"import/qq/confirm 处理失败 — {e}")
        raise HTTPException(status_code=500, detail=f"导入失败：{e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# GET /import/qq/pending_l1 — 查询待处理 session 数量
# =============================================================================

@router.get("/import/qq/pending_l1")
async def get_pending_l1():
    """查询当前有多少历史 session 尚未生成 L1 摘要。"""
    from ramaria.importer.batch import get_pending_count
    count = get_pending_count()
    return JSONResponse({"count": count})


# =============================================================================
# POST /import/qq/start_l1 — 启动 L1 批量生成
# =============================================================================

@router.post("/import/qq/start_l1")
async def start_l1_batch():
    """启动后台批处理线程，非阻塞，立即返回当前状态。"""
    from ramaria.importer.batch import start_batch
    from app.dependencies import session_manager
    status = start_batch(session_manager=session_manager)
    return JSONResponse(status)


# =============================================================================
# GET /import/qq/l1_progress — 查询批处理进度
# =============================================================================

@router.get("/import/qq/l1_progress")
async def get_l1_progress():
    """返回当前批处理状态，供前端进度面板轮询（建议间隔2秒）。"""
    from ramaria.importer.batch import get_status
    return JSONResponse(get_status())


# =============================================================================
# POST /import/qq/stop_l1 — 停止批处理
# =============================================================================

@router.post("/import/qq/stop_l1")
async def stop_l1_batch():
    """请求停止批处理，等当前 session 处理完后退出。"""
    from ramaria.importer.batch import stop_batch
    status = stop_batch()
    return JSONResponse(status)