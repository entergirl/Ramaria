"""
app/routes/settings.py — 用户配置读写接口

包含接口：
    GET  /api/settings — 读取所有用户配置项
    POST /api/settings — 保存用户配置项
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ramaria.storage.database import get_setting, set_setting


router = APIRouter()


class SettingsRequest(BaseModel):
    """
    设置面板提交的配置项。
    所有字段可选，只更新传入的字段，其余保持不变。
    """
    debounce_seconds:  float | None = None
    push_enabled:      int   | None = None
    push_window_start: int   | None = None
    push_window_end:   int   | None = None
    push_daily_limit:  int   | None = None


@router.get("/api/settings")
async def api_get_settings():
    """
    读取前端设置面板需要的所有配置项。
    返回字典，key 为配置项名，value 为已转换好类型的值。
    """
    return JSONResponse({
        "debounce_seconds":  float(get_setting("debounce_seconds")  or "3"),
        "push_enabled":      int(get_setting("push_enabled")         or "1"),
        "push_window_start": int(get_setting("push_window_start")    or "8"),
        "push_window_end":   int(get_setting("push_window_end")      or "24"),
        "push_daily_limit":  int(get_setting("push_daily_limit")     or "4"),
    })


@router.post("/api/settings")
async def api_post_settings(req: SettingsRequest):
    """
    保存用户配置到 settings 表。
    只更新请求体中明确传入的字段。

    参数合法性校验：
        debounce_seconds  — 1~10
        push_window_start — 0~23
        push_window_end   — 1~24，且大于 push_window_start
        push_daily_limit  — 1~10
    """
    errors = []
    updates: dict[str, str] = {}

    if req.debounce_seconds is not None:
        if not (1 <= req.debounce_seconds <= 10):
            errors.append("debounce_seconds 必须在 1~10 之间")
        else:
            updates["debounce_seconds"] = str(req.debounce_seconds)

    if req.push_enabled is not None:
        if req.push_enabled not in (0, 1):
            errors.append("push_enabled 必须是 0 或 1")
        else:
            updates["push_enabled"] = str(req.push_enabled)

    if req.push_window_start is not None:
        if not (0 <= req.push_window_start <= 23):
            errors.append("push_window_start 必须在 0~23 之间")
        else:
            updates["push_window_start"] = str(req.push_window_start)

    if req.push_window_end is not None:
        if not (1 <= req.push_window_end <= 24):
            errors.append("push_window_end 必须在 1~24 之间")
        else:
            updates["push_window_end"] = str(req.push_window_end)

    if req.push_daily_limit is not None:
        if not (1 <= req.push_daily_limit <= 10):
            errors.append("push_daily_limit 必须在 1~10 之间")
        else:
            updates["push_daily_limit"] = str(req.push_daily_limit)

    # 窗口起止时间交叉校验
    if req.push_window_start is not None and req.push_window_end is not None:
        if req.push_window_end <= req.push_window_start:
            errors.append("push_window_end 必须大于 push_window_start")

    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    for key, value in updates.items():
        set_setting(key, value)

    return JSONResponse({"status": "ok"})
