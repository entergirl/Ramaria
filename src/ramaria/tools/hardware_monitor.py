"""
src/ramaria/tools/hardware_monitor.py — 硬件状态感知模块

职责：
    通过 psutil 采集当前机器的硬件状态快照，
    格式化为可注入 System Prompt 的结构化文本。

对外接口：
    get_hardware_stats() -> str
        返回格式化后的硬件状态文本；采集失败时返回空字符串。

    is_high_load() -> bool
        判断当前是否处于高负载状态（CPU > 80% 且持续感知到压力）。
        供 L1 摘要生成时写入 atmosphere 字段使用。

设计约束：
    · 不做任何写操作，只读取系统状态
    · psutil 不可用时静默降级，不抛出异常，不影响主对话流程
    · 进程列表只取 CPU 占用前 5 名，避免 Prompt 过长
"""

from __future__ import annotations

from ramaria.logger import get_logger

logger = get_logger(__name__)

# CPU 高负载判断阈值（百分比）
_CPU_HIGH_THRESHOLD = 80.0

# 内存高占用判断阈值（百分比）
_MEM_HIGH_THRESHOLD = 85.0

# 进程列表最多展示条数
_TOP_PROCESS_COUNT = 5


def get_hardware_stats() -> str:
    """
    采集当前硬件状态并格式化为文本。

    返回示例：
        CPU 占用 45.2%，内存使用 9.8 GB / 16.0 GB（61.3%）
        电池：82%，正在充电
        活跃进程（CPU 占用前5）：LM Studio 38.1%、VSCode 12.3%、Chrome 8.9%

    返回：
        str — 格式化后的硬件状态文本
        ""  — psutil 未安装或采集失败时返回空字符串
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil 未安装，硬件感知不可用。请执行：pip install psutil")
        return ""

    lines: list[str] = []

    # ── CPU 与内存 ──────────────────────────────────────────────────────────
    try:
        # interval=0.5 表示采样 0.5 秒取平均，比 interval=None 的瞬时值更准确
        cpu_pct = psutil.cpu_percent(interval=0.5)

        mem     = psutil.virtual_memory()
        mem_gb  = mem.total / (1024 ** 3)
        used_gb = mem.used  / (1024 ** 3)

        lines.append(
            f"CPU 占用 {cpu_pct:.1f}%，"
            f"内存使用 {used_gb:.1f} GB / {mem_gb:.1f} GB（{mem.percent:.1f}%）"
        )
    except Exception as e:
        logger.warning(f"CPU/内存采集失败 — {e}")

    # ── 电池状态（笔记本场景，无电池时跳过）────────────────────────────────
    try:
        battery = psutil.sensors_battery()
        if battery is not None:
            charging = "正在充电" if battery.power_plugged else "使用电池"
            lines.append(f"电池：{battery.percent:.0f}%，{charging}")
    except Exception:
        # 台式机或不支持电池感知时静默跳过
        pass

    # ── 活跃进程（CPU 占用前 N 名）────────────────────────────────────────
    try:
        procs: list[tuple[float, str]] = []
        for proc in psutil.process_iter(["name", "cpu_percent"]):
            try:
                cpu = proc.info["cpu_percent"] or 0.0
                name = proc.info["name"] or "unknown"
                if cpu > 0.1:
                    procs.append((cpu, name))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 按 CPU 占用降序，取前 N 名
        procs.sort(reverse=True)
        top = procs[:_TOP_PROCESS_COUNT]

        if top:
            proc_str = "、".join(f"{name} {cpu:.1f}%" for cpu, name in top)
            lines.append(f"活跃进程（CPU 占用前{_TOP_PROCESS_COUNT}）：{proc_str}")

    except Exception as e:
        logger.warning(f"进程列表采集失败 — {e}")

    if not lines:
        return ""

    return "\n".join(lines)


def is_high_load() -> bool:
    """
    判断当前是否处于高负载状态。
    高负载定义：CPU > 80% 或内存占用 > 85%。

    供 L1 摘要生成时额外标注 atmosphere 字段（如「深夜高负载编码」）使用。

    返回：
        True  — 高负载
        False — 正常负载或 psutil 不可用
    """
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.5)
        mem_pct = psutil.virtual_memory().percent
        return cpu_pct > _CPU_HIGH_THRESHOLD or mem_pct > _MEM_HIGH_THRESHOLD
    except Exception:
        return False
