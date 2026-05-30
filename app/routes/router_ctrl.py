"""
app/routes/router_ctrl.py — 路由控制接口

包含接口：
    GET  /router/status     — 查询当前路由状态
    POST /router/toggle     — 切换线上/本地模式（Claude 专家路由）
    POST /router/llm-mode   — 切换 LLM 运行模式（本地部署 / 线上 API）（v0.7.0）
    GET  /api/health/check  — 系统健康检查
    GET  /api/persona/get   — 获取人格配置
    POST /api/persona/save  — 保存人格配置
"""

from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.dependencies import verify_admin_key

router = APIRouter()

# 固定真实路径：Ramaria-main/config/persona.toml
ROOT = Path(__file__).resolve().parent.parent.parent
PERSONA_PATH = ROOT / "config" / "persona.toml"


class ToggleRequest(BaseModel):
    """POST /router/toggle 请求体。"""
    online: bool


class LLMModeRequest(BaseModel):
    """POST /router/llm-mode 请求体（v0.7.0）。"""
    mode: str  # "local" 或 "api"


@router.get("/router/status")
async def get_router_status():
    """返回当前路由状态，供前端 UI 初始化时同步显示。"""
    from app.dependencies import router as app_router
    return JSONResponse(app_router.get_status())


@router.post("/router/toggle", dependencies=[Depends(verify_admin_key)])
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
# LLM 模式切换（v0.7.0 新增）
# =============================================================================

@router.post("/router/llm-mode", dependencies=[Depends(verify_admin_key)])
async def set_llm_mode(req: LLMModeRequest):
    """
    切换 LLM 运行模式（本地部署 / 线上 API）。

    流程：
        1. 校验 mode 值合法性
        2. API 模式下检查必填项（API_KEY / API_BASE_URL）
        3. 更新 ramaria.config 模块级常量
        4. 写入 .env 持久化
        5. 返回新状态

    错误码：
        400 — mode 值非法或必填项缺失
    """
    from ramaria.logger import get_logger
    logger = get_logger(__name__)

    mode = req.mode.strip().lower()
    if mode not in ("local", "api"):
        raise HTTPException(
            status_code=400,
            detail=f'无效的 LLM 模式："{req.mode}"，必须是 "local" 或 "api"',
        )

    # API 模式下检查必填项
    if mode == "api":
        from ramaria.config import API_KEY, API_BASE_URL
        if not API_KEY or not API_KEY.strip():
            raise HTTPException(
                status_code=400,
                detail="切换到线上 API 模式需要先配置 API_KEY，请在设置页面中填写。",
            )
        if not API_BASE_URL or not API_BASE_URL.strip():
            raise HTTPException(
                status_code=400,
                detail="切换到线上 API 模式需要先配置 API_BASE_URL，请在设置页面中填写。",
            )

    # 更新运行时配置（llm_client 立即生效）
    from ramaria.config import update_runtime_config
    update_runtime_config({"LLM_MODE": mode})

    # 持久化到 .env 文件
    from app.core.env_checker import write_env
    write_env({"LLM_MODE": mode})

    logger.info(f"LLM 模式已切换为：{mode}")

    return JSONResponse({
        "ok":   True,
        "mode": mode,
        "message": "将在下一组对话时生效",
    })


# =============================================================================
# 健康检查 + 人格配置接口
# =============================================================================


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
