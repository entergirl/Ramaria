# =============================================================================
# prompt_builder.py —— 珊瑚菌虚拟伙伴系统 · System Prompt 构建模块
# =============================================================================
#
# 职责：
#   将静态人设（persona.toml）与动态上下文（时间、记忆、session）
#   拼装成每轮对话的完整 system prompt。
#
# Block 结构（按拼接顺序）：
#   [A] 角色核心人设     ← 从 persona.toml 读取，静态
#   [B] 时间与状态上下文 ← 每次对话动态生成
#   [C] 记忆上下文       ← RAG 检索结果注入（由 main._build_context() 组装）
#   [D] 当前 session 信息← session 元信息
#   [E] 交互规则与格式   ← 从 persona.toml 读取，静态
#
# 变更记录：
#   v2 — 新增 reset_builder()，修复代码优化清单第三轮 P3-C：
#        原来模块级单例 _default_builder 没有重置机制，
#        在单元测试中切换 persona 文件时无法重置，总是复用第一次初始化的实例。
#        新增 reset_builder() 将单例置为 None，下次调用 get_builder() 时重新初始化。
#
# 使用方式：
#   builder = PromptBuilder()           # 启动时初始化一次
#   prompt  = builder.build(context)    # 每次对话时调用
#
# context 字典结构（所有字段均为可选）：
#   {
#       "last_session_time": datetime | None,  # 上次对话的结束时间
#       "l3_profile":        str | None,       # L3 用户长期画像文本
#       "retrieved_l1l2":    str | None,       # RAG 检索到的 L1/L2 摘要
#       "raw_fragments":     str | None,       # 穿透至 L0 的原始对话片段
#       "session_id":        int | None,       # 当前 session 数据库 ID
#       "session_index":     int | None,       # 当前是第几个 session（从1计）
#   }
# =============================================================================

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

from logger import get_logger
logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# 常量
# -----------------------------------------------------------------------------

# persona.toml 默认路径：与本文件同目录
DEFAULT_PERSONA_PATH = Path(__file__).parent / "persona.toml"

# Block 之间的分隔符（--- 让模型也能感知到结构边界）
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

    def __init__(self, persona_path: str | Path = DEFAULT_PERSONA_PATH):
        """
        初始化构建器，加载静态人设文件。

        Args:
            persona_path: persona.toml 的路径，默认与本文件同目录。
        """
        self._blocks = self._load_persona(persona_path)

    # -------------------------------------------------------------------------
    # 公开方法
    # -------------------------------------------------------------------------

    def build(self, context: dict | None = None) -> str:
        """
        构建完整的 system prompt。

        Args:
            context: 包含动态信息的字典（结构见文件顶部注释）。
                     传 None 时所有动态块降级为最小化内容。

        Returns:
            拼装好的完整 system prompt 字符串。
        """
        if context is None:
            context = {}

        # 按顺序收集各 Block，空 Block 自动跳过
        blocks = [
            self._blocks.get("A_persona", ""),           # [A] 静态人设
            self._build_time_block(                       # [B] 动态时间
                context.get("last_session_time")
            ),
            self._build_memory_block(                     # [C] 动态记忆
                context.get("l3_profile"),
                context.get("retrieved_l1l2"),
                context.get("raw_fragments"),
            ),
            self._build_session_block(                    # [D] 动态 session
                context.get("session_id"),
                context.get("session_index"),
            ),
            self._blocks.get("E_rules", ""),              # [E] 静态规则
        ]

        # 过滤空块后拼接
        return BLOCK_SEPARATOR.join(b.strip() for b in blocks if b and b.strip())

    # -------------------------------------------------------------------------
    # 私有方法：文件加载
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_persona(path: str | Path) -> dict:
        """
        加载并解析 persona.toml，返回 blocks 字典。

        Args:
            path: toml 文件路径。

        Returns:
            {"A_persona": "...", "E_rules": "...", ...}
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"persona.toml 未找到：{path}\n"
                "请确认文件存在，或通过 PromptBuilder(persona_path=...) 指定路径。"
            )
        with open(path, "rb") as f:
            data = tomllib.load(f)

        blocks = data.get("blocks", {})
        if not blocks:
            raise ValueError("persona.toml 中未找到 [blocks] 节，请检查文件格式。")

        return blocks

    # -------------------------------------------------------------------------
    # 私有方法：Block B —— 时间与状态上下文
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_time_block(last_session_time: datetime | None) -> str:
        """
        生成 Block B：当前时间 + 距上次对话时长 + 跨日标记。

        Args:
            last_session_time: 上次 session 结束的 datetime，无则传 None。

        Returns:
            格式化的时间上下文字符串。
        """
        from datetime import timezone as _tz
        now     = datetime.now(_tz.utc).astimezone()   # 带本地时区的当前时间
        weekday = WEEKDAY_ZH[now.weekday()]
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

            # 跨日检测：上次对话日期 != 今天
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

        lines = ["## 时间背景", current_time_str, gap_str]
        if cross_day_warning:
            lines.append(cross_day_warning)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 私有方法：Block C —— 记忆上下文
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_memory_block(
        l3_profile: str | None,
        retrieved_l1l2: str | None,
        raw_fragments: str | None,
    ) -> str:
        """
        生成 Block C：三层记忆内容。

        retrieved_l1l2 由 main._build_context() 组装，在 P2-A 修复后包含：
          · RAG 语义检索结果（与本次消息最相关的历史，已格式化）
          · 时间序内容（最近 L2 + 最新 L1，追加在 RAG 结果后）
        本函数直接将其注入，不做二次处理。

        任意层级为 None 时自动跳过，全部为 None 时返回空字符串（Block 被跳过）。

        Args:
            l3_profile:     L3 用户长期画像的文本内容。
            retrieved_l1l2: RAG 检索到的 L1/L2 摘要文本（已含时间序内容）。
            raw_fragments:  从 L0 穿透召回的原始对话片段（预留，当前为 None）。

        Returns:
            格式化的记忆上下文字符串，或空字符串。
        """
        parts = []

        if l3_profile:
            parts.append(
                "### 用户长期画像（L3）\n"
                "以下是关于烧酒的长期稳定特征，优先级最高：\n"
                f"{l3_profile}"
            )

        if retrieved_l1l2:
            parts.append(
                "### 相关历史摘要（L1/L2）\n"
                "以下是与当前对话语义相关的历史摘要及近期动态：\n"
                f"{retrieved_l1l2}"
            )

        if raw_fragments:
            parts.append(
                "### 原始对话片段（L0）\n"
                "以下是从历史消息中召回的具体片段，可辅助理解细节：\n"
                f"{raw_fragments}"
            )

        if not parts:
            return ""

        header = (
            "## 记忆上下文\n"
            "以下内容来自历史对话记忆，请自然地融入当前对话，"
            "不要生硬地报出【我记得你说过】，而是像老朋友一样自然提及。"
        )
        return header + "\n\n" + "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # 私有方法：Block D —— 当前 Session 信息
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_session_block(
        session_id: int | None,
        session_index: int | None,
    ) -> str:
        """
        生成 Block D：当前 session 的元信息。

        Args:
            session_id:    数据库中的 session ID。
            session_index: 这是第几个 session（从1计）。
                           P2-4 修复后传 None，此块不显示编号。

        Returns:
            格式化的 session 信息字符串，或空字符串。
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


def get_builder(persona_path: str | Path = DEFAULT_PERSONA_PATH) -> PromptBuilder:
    """
    获取模块级单例 PromptBuilder。

    首次调用时初始化，后续直接复用同一实例。
    适合在 main.py 或 session_manager.py 中统一调用。

    Args:
        persona_path: persona.toml 路径，仅首次调用时生效。
                      若需要切换 persona 文件，请先调用 reset_builder()。

    Returns:
        PromptBuilder 实例。
    """
    global _default_builder
    if _default_builder is None:
        _default_builder = PromptBuilder(persona_path)
    return _default_builder


def reset_builder():
    """
    重置模块级单例，下次调用 get_builder() 时重新从文件初始化。

    【P3-C 新增】修复代码优化清单第三轮 P3-C：
        原来 _default_builder 一旦初始化就无法重置，
        在单元测试中切换 persona 文件时会复用旧实例，导致测试相互污染。

        典型使用场景：
            # 测试场景一：使用默认 persona
            prompt_a = build_system_prompt(context_a)

            # 切换到另一个 persona 文件进行测试
            reset_builder()
            prompt_b = get_builder(persona_path="test_persona.toml").build(context_b)

            # 恢复默认
            reset_builder()
            prompt_c = build_system_prompt(context_c)   # 重新从默认路径加载

        注意：正常业务运行中不需要调用此函数。
              此函数主要面向测试代码和需要动态切换 persona 的场景。
    """
    global _default_builder
    _default_builder = None
    logger.debug("PromptBuilder 单例已重置，下次调用将重新从文件初始化")


def build_system_prompt(context: dict | None = None) -> str:
    """
    便捷函数：直接构建 system prompt，无需手动管理 builder 实例。

    等价于 get_builder().build(context)。

    Args:
        context: 动态上下文字典（结构见文件顶部注释）。

    Returns:
        完整的 system prompt 字符串。
    """
    return get_builder().build(context)


# =============================================================================
# 调试入口：python prompt_builder.py 直接运行可预览效果
# =============================================================================

if __name__ == "__main__":
    from datetime import timedelta, timezone

    print("=" * 60)
    print("【调试模式】预览生成的 System Prompt")
    print("=" * 60)

    # 模拟跨日场景（2.5小时前的 session，带时区）
    fake_last_session = datetime.now(timezone.utc) - timedelta(hours=2, minutes=30)

    test_context = {
        "last_session_time": fake_last_session,
        "l3_profile": (
            "基础信息：在读本科生，ID 烧酒\n"
            "近期状态：正在开发珊瑚菌虚拟伙伴系统，压力较大但状态积极\n"
            "兴趣爱好：编程、AI 系统设计"
        ),
        # 模拟 P2-A 接通 RAG 后的融合格式
        "retrieved_l1l2": (
            "[语义相关 · L1 单次摘要]\n"
            "烧酒完成了 vector_store 的多线程初始化修复。（相关度：0.31）\n"
            "\n── 近期动态 ──\n"
            "[时间段摘要 2026-03-25] 烧酒这周持续推进珊瑚菌代码审查修复工作。\n"
            "[最近一次对话] 烧酒完成了第二批代码修复，包括 merger 时区修复。（夜间，专注高效）"
        ),
        "raw_fragments": None,
        "session_id":    5,
        "session_index": None,   # P2-4：不显示编号
    }

    prompt = build_system_prompt(test_context)
    print(prompt)
    print("\n" + "=" * 60)
    print(f"总字符数：{len(prompt)}")

    # ------------------------------------------------------------------
    # 验证 reset_builder()（P3-C 修复核心）
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("【P3-C 验证】reset_builder() 测试")
    print("=" * 60)

    b1 = get_builder()
    print(f"第一次 get_builder() → id={id(b1)}")

    b2 = get_builder()
    print(f"第二次 get_builder() → id={id(b2)}（应与第一次相同，复用单例）")
    assert b1 is b2, "单例复用失败"

    reset_builder()
    print("reset_builder() 已调用")

    b3 = get_builder()
    print(f"reset 后 get_builder() → id={id(b3)}（应与之前不同，重新初始化）")
    assert b3 is not b1, "reset 后应创建新实例"

    print("reset_builder() 验证通过 ✓")
