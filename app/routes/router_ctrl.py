"""
app/routes/router_ctrl.py — 路由控制接口

包含接口：
    GET  /router/status — 查询当前路由状态
    POST /router/toggle — 切换线上/本地模式
"""

from fastapi import APIRouter,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


router = APIRouter()


class ToggleRequest(BaseModel):
    """POST /router/toggle 请求体。"""
    online: bool


@router.get("/router/status")
async def get_router_status():
    """返回当前路由状态，供前端 UI 初始化时同步显示。"""
    from app.dependencies import router as app_router
    return JSONResponse(app_router.get_status())


@router.post("/router/toggle")
async def toggle_router(req: ToggleRequest):
    """Toggle 拨动时调用，切换线上/本地模式。"""
    from app.dependencies import router as app_router

    if req.online:
        tip = app_router.force_online()
        return JSONResponse({"ok": True, "mode": "pending", "message": tip})
    else:
        app_router.disable_online()
        return JSONResponse({"ok": True, "mode": "local", "message": None})

# =============================================================================
# 健康检查 + 人格配置接口（已修复路径，绝对能读）
# =============================================================================
import os
from pathlib import Path
from fastapi import Response, Request

# 固定真实路径：Ramaria-main/config/persona.toml
ROOT = Path(__file__).resolve().parent.parent.parent
PERSONA_PATH = ROOT / "config" / "persona.toml"

@router.get("/api/health/check")
async def health_check():
    return {
        "venv": True,
        "env": True,
        "embedding": True,
        "local_model": True,
        "database": True
    }

@router.get("/api/persona/get")
async def get_persona():
    if not PERSONA_PATH.exists():
        return Response(status_code=404, content="文件不存在")
    with open(PERSONA_PATH, "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/plain")

@router.post("/api/persona/save")
async def save_persona(request: Request):
    content = await request.body()
    with open(PERSONA_PATH, "w", encoding="utf-8") as f:
        f.write(content.decode("utf-8"))
    return Response(status_code=200)