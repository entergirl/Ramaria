"""
app/routes/router_ctrl.py — 路由控制接口

包含接口：
    GET  /router/status — 查询当前路由状态
    POST /router/toggle — 切换线上/本地模式
"""

from fastapi import APIRouter
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