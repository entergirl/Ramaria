"""
src/ramaria/core/prompt_builder.py — System Prompt 构建模块

职责：
    将静态人设（config/persona.toml）与动态上下文（时间、记忆、session）
    拼装成每轮对话的完整 system prompt。

Block 结构（按拼接顺序）：
    [A] 角色核心人设     ← 从 persona.toml 读取，静态
    [B] 时间与状态上下文 ← 每次对话动态生成
    [C] 记忆上下文       ← RAG 检索结果注入（由 main._build_context() 组装）
    [D] 当前 session 信息← session 元信息
    [E] 交互规则与格式   ← 从 persona.toml 读取，静态

context 字典结构（所有字段均为可选）：
    {
        "last_session_time": datetime | None,  # 上次对话的结束时间
        "l3_profile":        str | None,       # L3 用户长期画像文本
        "retrieved_l1l2":    str | None,       # RAG 检索到的 L1/L2 摘要
        "raw_fragments":     str | None,       # 穿透至 L0 的原始对话片段
        "session_id":        int | None,       # 当前 session 数据库 ID
        "session_index":     int | None,       # 当前是第几个 session（从1计）
    }

"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Python 3.11+ 内置 tomllib；低版本需要：pip install tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Python < 3.11 需要安装 tomli：pip install tomli"
        )

from ramaria.config import PERSONA_PATH
from ramaria.constants import PROFILE_FIELD_LIST
from ramaria.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 常量
# =============================================================================

# persona.toml 默认路径：使用 ramaria.config 里定义的统一路径
# 原来是 Path(__file__).parent / "persona.toml"，
# 迁移后 __file__ 指向 src/ramaria/core/，不再能直接找到 config/ 下的文件
DEFAULT_PERSONA_PATH: Path = PERSONA_PATH

# Block 之间的分隔符
BLOCK_SEPARATOR = "\n\n---\n\n"

# 中文星期映射
WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# =============================================================================
# PromptBuilder 主类
# =============================================================================

class PromptBuilder:
    """
    System Prompt 构建器。

    启动时加载一次 persona.toml，之后每次调用 build() 动态拼装完整 prompt。
    """

    def __init__(
    self,
        persona_path: str | Path = DEFAULT_PERSONA_PATH,
    ):
        """
        初始化构建器，加载静态人设文件。
        """
        # _load_persona 现在返回包含 blocks 和 identity 的完整字典
        _data          = self._load_persona(persona_path)
        self._blocks   = _data["blocks"]
        self._identity = _data["identity"]


    # -------------------------------------------------------------------------
    # 公开方法
    # -------------------------------------------------------------------------

    def build(self, context: dict | None = None) -> str:
        """
        构建完整的 system prompt。

        参数：
            context — 包含动态信息的字典（结构见文件顶部注释）。
                      传 None 时所有动态块降级为最小化内容。

        返回：
            拼装好的完整 system prompt 字符串。
        """
        if context is None:
            context = {}

        blocks = [
            self._blocks.get("A_persona", ""),
            self._build_current_state_block(
                context.get("last_session_time"),
                context.get("tool_results"),
            ),
            self._build_memory_block(
                context.get("l3_profile"),
                context.get("retrieved_l1l2"),
                context.get("raw_fragments"),
            ),
            self._build_session_block(
                context.get("session_id"),
                context.get("session_index"),
            ),
            self._blocks.get("E_rules", ""),
        ]

        return BLOCK_SEPARATOR.join(
            b.strip() for b in blocks if b and b.strip()
        )

    # -------------------------------------------------------------------------
    # 私有方法：文件加载
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_persona(path: str | Path) -> dict:
        """
        加载并解析 persona.toml，返回完整数据字典。

        返回：
            dict，包含：
                "blocks"   → {"A_persona": "...", "E_rules": "...", ...}
                "identity" → {"assistant_name": "...", "user_name": "..."}

        异常：
            FileNotFoundError — 文件不存在时抛出
            ValueError        — 文件中没有 [blocks] 节时抛出
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"persona.toml 未找到：{path}\n"
                "请确认 config/persona.toml 存在，"
                "或通过 PromptBuilder(persona_path=...) 指定路径。"
            )
        with open(path, "rb") as f:
            data = tomllib.load(f)

        blocks = data.get("blocks", {})
        if not blocks:
            raise ValueError(
                "persona.toml 中未找到 [blocks] 节，请检查文件格式。"
            )

        # 读取 [identity] 块，提供默认值保证向后兼容
        # 旧版 persona.toml 没有此块时不会报错
        identity = data.get("identity", {})
        identity.setdefault("assistant_name", "助手")
        identity.setdefault("user_name", "用户")

        return {"blocks": blocks, "identity": identity}

    # -------------------------------------------------------------------------
    # 私有方法：Block B — 时间与状态上下文
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_current_state_block(
        last_session_time,
        tool_results=None,
    ) -> str:
        """
        生成 Block B：当前状态上下文（v0.5.0 改造版）。

        包含三个子节，按需组装：
            1. 时间背景    — 必然出现，与原 _build_time_block 逻辑完全一致
            2. 硬件状态    — 仅当 tool_results["hardware"] 有值时出现
            3. 文件扫描    — 仅当 tool_results["fs_scan"] 有值时出现

        参数：
            last_session_time — 上次 session 结束的 datetime，无则传 None
            tool_results      — resolve_tool_results() 的返回字典，无则传 None
                            结构：{"hardware": str|None, "fs_scan": str|None}
        """
        from datetime import timezone as _tz
        from datetime import datetime

        # 复用原有的星期和跨日逻辑常量
        WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

        now     = datetime.now(_tz.utc).astimezone()
        weekday = WEEKDAY_ZH[now.weekday()]

        # ── 子节1：时间背景（原 _build_time_block 逻辑，完全保留）─────────────
        current_time_str = (
            f"现在是 {now.strftime('%Y年%m月%d日 %H:%M')}，{weekday}。"
        )

        if last_session_time is None:
            gap_str           = "这是与烧酒的第一次对话，没有历史时间参照。"
            cross_day_warning = ""
        else:
            delta         = now - last_session_time
            total_seconds = int(delta.total_seconds())
            days          = delta.days
            hours         = total_seconds // 3600
            minutes       = (total_seconds % 3600) // 60

            if days >= 1:
                gap_str = f"距离上次对话已过去约 {days} 天。"
            elif hours > 0:
                gap_str = f"距离上次对话已过去约 {hours} 小时 {minutes} 分钟。"
            elif minutes > 0:
                gap_str = f"距离上次对话已过去约 {minutes} 分钟。"
            else:
                gap_str = "与上次对话间隔极短（不足1分钟）。"

            if last_session_time.date() < now.date():
                last_date_str     = last_session_time.strftime("%m月%d日")
                cross_day_warning = (
                    "[注意] 上次对话发生在 " + last_date_str + "，"
                    "本次对话已跨越自然日。烧酒可能经历了连续工作或熬夜，"
                    "请在问候中自然体现这一时间跨度（如询问休息情况），"
                    "而不是用普通的你好开场。"
                )
            else:
                cross_day_warning = ""

        time_lines = ["### 时间背景", current_time_str, gap_str]
        if cross_day_warning:
            time_lines.append(cross_day_warning)

        # ── 子节2：硬件状态（按需，未触发时跳过）────────────────────────────
        hardware_lines = []
        if tool_results and tool_results.get("hardware"):
            hardware_lines = [
                "### 硬件状态",
                tool_results["hardware"],
            ]

        # ── 子节3：文件扫描（按需，手动触发时附加）──────────────────────────
        fs_lines = []
        if tool_results and tool_results.get("fs_scan"):
            fs_lines = [
                "### 文件目录",
                tool_results["fs_scan"],
            ]

        # ── 拼装 Block B ──────────────────────────────────────────────────────
        # 将非空子节用空行分隔组合
        sections = []
        sections.append("\n".join(time_lines))
        if hardware_lines:
            sections.append("\n".join(hardware_lines))
        if fs_lines:
            sections.append("\n".join(fs_lines))

        header = "## 当前状态"
        return header + "\n\n" + "\n\n".join(sections)


    # -------------------------------------------------------------------------
    # 私有方法：Block D — 当前 Session 信息
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_session_block(
        session_id: int | None,
        session_index: int | None,
    ) -> str:
        """
        生成 Block D：当前 session 的元信息。

        参数：
            session_id    — 数据库中的 session ID。
            session_index — 这是第几个 session（从1计）。
                            传 None 时不显示编号。
        """
        if session_id is None and session_index is None:
            return ""

        lines = ["## 当前 Session 信息"]
        if session_index is not None:
            lines.append(f"- 这是与烧酒的第 {session_index} 个对话 session。")
        if session_id is not None:
            lines.append(f"- Session ID：{session_id}")

        return "\n".join(lines)


# =============================================================================
# 便捷函数（供其他模块快速调用）
# =============================================================================

# 模块级单例，避免重复读取文件
_default_builder: PromptBuilder | None = None


def get_builder(
    persona_path: str | Path = DEFAULT_PERSONA_PATH,
) -> PromptBuilder:
    """
    获取模块级单例 PromptBuilder。

    首次调用时初始化，后续复用同一实例。
    若需切换 persona 文件，请先调用 reset_builder()。

    参数：
        persona_path — persona.toml 路径，仅首次调用时生效。

    返回：
        PromptBuilder 实例。
    """
    global _default_builder
    if _default_builder is None:
        _default_builder = PromptBuilder(persona_path)
    return _default_builder


def reset_builder() -> None:
    """
    重置模块级单例，下次调用 get_builder() 时重新从文件初始化。

    使用场景：
        · 单元测试中切换 persona 文件
        · 需要重新加载 persona.toml 的场景（如热更新）

    正常业务运行中不需要调用此函数。
    """
    global _default_builder
    _default_builder = None
    logger.debug("PromptBuilder 单例已重置，下次调用将重新从文件初始化")


def build_system_prompt(context: dict | None = None) -> str:
    """
    便捷函数：直接构建 system prompt，无需手动管理 builder 实例。

    等价于 get_builder().build(context)。

    参数：
        context — 动态上下文字典（结构见文件顶部注释）。

    返回：
        完整的 system prompt 字符串。
    """
    return get_builder().build(context)
    
def get_identity(self) -> dict:
    """
    返回 persona.toml 中的 [identity] 配置。

    返回格式：
        {
            "assistant_name": str,  # 助手名字
            "user_name":      str,  # 用户名字
        }

    调用方：push_scheduler.py 的推送消息生成函数
    """
    return dict(self._identity)