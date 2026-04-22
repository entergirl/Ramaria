"""
app/routes/settings.py — 用户配置读写接口

包含接口：
    GET  /api/settings — 读取所有用户配置项
    POST /api/settings — 保存用户配置项
    GET  /api/model-status — 获取向量模型加载状态
    POST /api/model-path — 修改向量模型路径并重新加载
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from pathlib import Path

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


# =============================================================================
# 向量模型管理接口
# =============================================================================

@router.get("/api/model-status")
async def api_get_model_status():
    """
    获取向量模型的加载状态。
    
    返回：
        {
            "current_path": str or null,  # 当前模型路径
            "is_loaded": bool,            # 是否成功加载
            "error": str or null,         # 如果加载失败，返回错误信息
        }
    """
    from ramaria.storage.vector_store import get_model_status
    status = get_model_status()
    return JSONResponse(status)


class ModelPathRequest(BaseModel):
    """
    修改模型路径的请求体
    """
    model_path: str


@router.post("/api/model-path")
async def api_set_model_path(req: ModelPathRequest):
    """
    修改向量模型路径，验证有效性后热重载。
    
    流程：
        1. 验证输入的路径有效性
        2. 调用 reload_model() 热重载模型
        3. 更新 .env 文件中的 EMBEDDING_MODEL 变量
        4. 返回操作结果
    
    错误响应：
        - 422: 请求体格式错误
        - 400: 路径验证失败或模型重载失败
    """
    if not req.model_path or not str(req.model_path).strip():
        raise HTTPException(
            status_code=400,
            detail="模型路径不能为空"
        )
    
    try:
        # 调用向量存储的热重载函数
        from ramaria.storage.vector_store import reload_model
        success, msg = reload_model(req.model_path)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=msg
            )
        
        # 热重载成功，更新 .env 文件
        _update_env_file("EMBEDDING_MODEL", req.model_path)
        
        return JSONResponse({
            "status": "ok",
            "message": msg,
            "new_path": req.model_path,
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"模型路径修改失败：{str(e)}"
        )


def _update_env_file(key: str, value: str) -> None:
    """
    更新 .env 文件中的配置项。
    
    逻辑：
        1. 从项目根目录找到 .env 文件
        2. 查找对应的 key 并更新 value
        3. 如果 key 不存在，则在文件末尾追加
    
    参数：
        key - 配置项名称
        value - 配置项值
    """
    from pathlib import Path
    
    # 获取项目根目录下的 .env 文件
    # 当前文件是 app/routes/settings.py，往上两级是项目根目录
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / ".env"
    
    if not env_path.exists():
        # .env 文件不存在，创建新文件
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
        return
    
    # 读取 .env 文件
    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    
    # 查找并更新对应的 key
    found = False
    for i, line in enumerate(lines):
        # 跳过注释行和空行
        if line.strip().startswith("#") or not line.strip():
            continue
        
        if "=" in line:
            config_key, _ = line.split("=", 1)
            if config_key.strip() == key:
                lines[i] = f"{key}={value}\n"
                found = True
                break
    
    # 如果未找到，在文件末尾追加
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={value}\n")
    
    # 写回 .env 文件
    env_path.write_text("".join(lines), encoding="utf-8")
