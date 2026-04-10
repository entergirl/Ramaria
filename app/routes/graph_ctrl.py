"""
app/routes/graph_ctrl.py — 图谱构建接口

包含接口：
    GET  /graph/pending — 查询待构建图谱的 L1 数量
    POST /graph/start   — 启动图谱构建后台线程
    POST /graph/stop    — 请求停止图谱构建
    GET  /graph/status  — 查询图谱构建进度（前端轮询）
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/graph/pending")
async def graph_pending():
    """查询当前待构建图谱的 L1 数量。"""
    from ramaria.memory.graph_builder import get_graph_pending_count
    count = get_graph_pending_count()
    return JSONResponse({"count": count})


@router.post("/graph/start")
async def graph_start():
    """启动图谱构建后台线程，非阻塞，立即返回当前状态。"""
    from ramaria.memory.graph_builder import start_graph_build, reload_graph

    status = start_graph_build()

    # 无待处理 L1 时确保内存图是最新的
    if status.get("status") == "done" and status.get("total", 0) == 0:
        reload_graph()

    return JSONResponse(status)


@router.post("/graph/stop")
async def graph_stop():
    """请求停止图谱构建，等当前 L1 处理完后退出。"""
    from ramaria.memory.graph_builder import stop_graph_build
    status = stop_graph_build()
    return JSONResponse(status)


@router.get("/graph/status")
async def graph_status():
    """查询图谱构建进度，供前端轮询（建议间隔3秒）。"""
    from ramaria.memory.graph_builder import get_graph_status, reload_graph

    status = get_graph_status()

    # 批处理刚完成时，重新加载内存图
    if status.get("status") == "done" and status.get("total", 0) > 0:
        reload_graph()

    return JSONResponse(status)