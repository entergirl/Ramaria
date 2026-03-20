"""
prompt_builder.py — System Prompt 构建模块
负责在每次对话时，从数据库和向量索引读取各层记忆，组装完整的 system prompt。

─────────────────────────────────────────────────────────────
注入顺序（从稳定到动态，共四层）
─────────────────────────────────────────────────────────────

  1. L3 用户画像（稳定层）
       每次全量注入，内容变化频率最低，是模型理解烧酒的基础底色。

  2. 近期 L2 摘要（近期层）
       注入最新 N 条时间段聚合摘要（默认3条），代表近期状态和规律。

  3. 最新 L1 摘要（当日层）
       注入最近一条单次对话摘要，代表上一次对话的最新上下文。

  4. 语义检索结果（动态层）★ 阶段二新增
       基于两个查询来源实时检索，合并去重后注入：
         · 查询源 A：用户本次发送的消息内容
         · 查询源 B：最新一条 L1 摘要的文本（上次对话的话题线索）
       两个查询分别在 L1 / L2 向量索引中检索，结果合并、按距离排序、去重，
       过滤掉距离超过阈值的噪声后注入 prompt。
       这一层让模型能"主动想起"与当前话题相关的历史记忆。

─────────────────────────────────────────────────────────────
空层处理
─────────────────────────────────────────────────────────────

  任意一层没有数据时，对应段落整体不拼入 prompt，不留空白占位符。
  冷启动阶段（首次使用，没有任何历史记忆）可以正常运行。

─────────────────────────────────────────────────────────────
与其他模块的关系
─────────────────────────────────────────────────────────────

  · 读取结构化记忆：database.py
  · 读取向量检索结果：vector_store.py（retrieve_combined）
  · 读取模板配置：config.py
  · 被调用：main.py 的 /chat 路由函数

使用方法：
    from prompt_builder import build_system_prompt

    # 不传 query_text 时，动态层仅用最新 L1 内容做检索
    system_prompt = build_system_prompt()

    # 传入用户当前消息时，动态层同时用消息内容 + 最新 L1 两条路径检索
    system_prompt = build_system_prompt(query_text="用户这条消息的内容")
"""

from datetime import datetime, timezone

from config import (
    SYSTEM_PROMPT_TEMPLATE,
    PROFILE_SECTION_HEADER,
    L2_SECTION_HEADER,
    L1_SECTION_HEADER,
    RAG_SECTION_HEADER,
    L2_INJECT_COUNT,
    TIME_FORMAT,
    WEEKDAY_CN,
)
from database import (
    get_current_profile,
    get_recent_l2,
    get_latest_l1,
)


# =============================================================================
# 各记忆层内容格式化
# =============================================================================

def _format_profile(profile_dict):
    """
    将 L3 用户画像字典格式化为适合注入 system prompt 的文本块。

    字段顺序按重要性排列，每个板块独占一行，板块名称用中文显示。
    没有填写的板块自动跳过，不留空行。

    参数：
        profile_dict — get_current_profile() 返回的字典，key 为板块名
                       如 {"basic_info": "...", "interests": "..."}

    返回：
        str — 格式化后的文本；profile_dict 为空时返回空字符串
    """
    if not profile_dict:
        return ""

    # 板块名到中文标签的映射，同时定义注入顺序
    field_labels = [
        ("basic_info",      "基础信息"),
        ("personal_status", "近期状态"),
        ("interests",       "兴趣爱好"),
        ("social",          "社交情况"),
        ("history",         "重要经历"),
        ("recent_context",  "近期背景"),
    ]

    lines = []
    for field_key, label in field_labels:
        content = profile_dict.get(field_key, "").strip()
        if content:
            lines.append(f"{label}：{content}")

    return "\n".join(lines)


def _format_l2_list(l2_rows):
    """
    将近期 L2 摘要列表格式化为适合注入 system prompt 的文本块。

    L2 是时间段摘要，按时间倒序注入（最新的在前），方便模型优先关注近期状态。

    参数：
        l2_rows — get_recent_l2() 返回的列表，按 created_at DESC 排序

    返回：
        str — 格式化后的文本；列表为空时返回空字符串
    """
    if not l2_rows:
        return ""

    lines = []
    for row in l2_rows:
        # 取日期部分（前10位），去掉时间和时区，保持简洁
        date_str = row["created_at"][:10] if row["created_at"] else "未知日期"
        keywords = f"（{row['keywords']}）" if row["keywords"] else ""
        lines.append(f"· [{date_str}] {row['summary']}{keywords}")

    return "\n".join(lines)


def _format_l1(l1_row):
    """
    将最新 L1 摘要格式化为适合注入 system prompt 的文本块。

    参数：
        l1_row — get_latest_l1() 返回的 sqlite3.Row 或 None

    返回：
        str — 格式化后的文本；l1_row 为 None 时返回空字符串
    """
    if l1_row is None:
        return ""

    time_period = l1_row["time_period"] or ""
    atmosphere  = l1_row["atmosphere"]  or ""
    meta        = f"（{time_period}，{atmosphere}）" if time_period or atmosphere else ""

    return f"{l1_row['summary']}{meta}"


def _format_rag_results(combined_results):
    """
    将语义检索的合并结果格式化为适合注入 system prompt 的文本块。

    格式设计说明：
      · L2 结果标注"时间段记忆"，表明这是一段时间的聚合，宏观背景
      · L1 结果标注"对话记忆"，表明这是某次具体对话的摘要，细节精度高
      · 每条记忆前加"·"对齐，让模型容易区分各条之间的边界
      · 不展示距离分数（那是内部调试信息，不需要让模型看到）

    参数：
        combined_results — _retrieve_and_merge() 返回的去重合并列表
                           每个元素是 dict，包含 source / document / distance 字段

    返回：
        str — 格式化后的文本；列表为空时返回空字符串
    """
    if not combined_results:
        return ""

    lines = []
    for item in combined_results:
        # source 字段标注来源层级，让模型知道这条记忆的粒度
        source_label = "时间段记忆" if item["source"] == "l2" else "对话记忆"
        lines.append(f"· [{source_label}] {item['document']}")

    return "\n".join(lines)


# =============================================================================
# 语义检索与合并（动态层核心逻辑）
# =============================================================================

def _retrieve_and_merge(query_a, query_b=None):
    """
    执行两路语义检索并合并去重，返回最终注入的记忆列表。

    两路查询设计：
      · query_a（主查询）：用户本次发送的消息内容
                           直接反映当前对话意图，精准度最高
      · query_b（辅查询）：最新 L1 摘要的文本内容
                           代表上次对话的话题线索，扩大相关记忆的召回范围
                           例如用户继续昨天未完的话题，这条查询能把昨天的
                           相关历史也一并捞出来

    合并去重规则：
      · 以 document 文本作为去重 key，避免同一条记忆被两路查询重复返回
      · 相同 document 出现两次时，保留距离更小（相似度更高）的那条
      · 最终按距离升序排列，最相关的放在最前面

    参数：
        query_a — 主查询文本（通常是用户当前消息），不能为空
        query_b — 辅查询文本（通常是最新 L1 摘要），可以为 None

    返回：
        list[dict] — 合并去重后的结果列表，每个元素包含：
            {
                "source":   "l1" 或 "l2"，标注来自哪层索引
                "document": str，索引时存入的文本
                "distance": float，语义距离（越小越相似）
            }
        无相关结果时返回空列表。
    """
    try:
        from vector_store import retrieve_combined
    except ImportError:
        # vector_store 未安装（冷启动或环境问题），静默跳过，不影响对话
        print("[prompt_builder] 警告：vector_store 模块未找到，跳过语义检索")
        return []

    # ------------------------------------------------------------------
    # 第一路：主查询（用户当前消息）
    # ------------------------------------------------------------------
    raw_a = retrieve_combined(query_a)   # 返回 {"l2": [...], "l1": [...]}

    # ------------------------------------------------------------------
    # 第二路：辅查询（最新 L1 摘要内容）
    # query_b 为 None 时（没有历史 L1 或调用方未传入）跳过这一路
    # ------------------------------------------------------------------
    raw_b = {"l2": [], "l1": []}
    if query_b:
        raw_b = retrieve_combined(query_b)

    # ------------------------------------------------------------------
    # 合并两路结果，打上 source 标签，统一放入一个列表
    # ------------------------------------------------------------------
    all_results = []

    for item in raw_a.get("l2", []):
        all_results.append({"source": "l2", **item})
    for item in raw_a.get("l1", []):
        all_results.append({"source": "l1", **item})
    for item in raw_b.get("l2", []):
        all_results.append({"source": "l2", **item})
    for item in raw_b.get("l1", []):
        all_results.append({"source": "l1", **item})

    if not all_results:
        return []

    # ------------------------------------------------------------------
    # 去重：以 document 文本为 key
    # 同一条记忆被两路查询都召回时，保留距离更小的那条
    # ------------------------------------------------------------------
    seen = {}   # key: document文本, value: 当前保留的最优结果dict
    for item in all_results:
        doc = item["document"]
        if doc not in seen or item["distance"] < seen[doc]["distance"]:
            seen[doc] = item

    # 按距离升序排列，最相关的放在最前面
    merged = sorted(seen.values(), key=lambda x: x["distance"])

    return merged


# =============================================================================
# 当前时间格式化
# =============================================================================

def _get_current_time():
    """
    获取当前本地时间，格式化为适合注入 system prompt 的中文字符串。

    为什么需要这个函数：
        模型没有感知当前时间的能力。如果不主动注入，模型只能依赖历史记忆里
        出现过的时间信息，导致时间感知停留在过去。
        每次请求都重新生成，确保模型始终知道"现在"是什么时间。

    返回：
        str — 如 "2026年03月19日 星期四 15:30"
    """
    now = datetime.now(timezone.utc).astimezone()
    weekday_str = WEEKDAY_CN[now.weekday()]
    return now.strftime(TIME_FORMAT).replace("{weekday}", weekday_str)


# =============================================================================
# 对外接口
# =============================================================================

def build_system_prompt(query_text=None):
    """
    读取各层记忆，组装完整的 system prompt。
    这是本模块唯一的对外接口，由 main.py 的 /chat 路由函数调用。

    四层注入顺序（从稳定到动态）：
      L3 Profile → 近期 L2 → 最新 L1 → 语义检索结果

    参数：
        query_text — 用户当前发送的消息文本（可选）。
                     传入时，动态层会同时用这条消息 + 最新 L1 两路检索；
                     不传时，动态层仅用最新 L1 内容做检索；
                     两者都没有数据时，动态层为空，不影响其余三层正常注入。

    返回：
        str — 完整的 system prompt 文本
    """
    # ------------------------------------------------------------------
    # 获取当前时间（每次请求都重新生成，确保实时性）
    # ------------------------------------------------------------------
    current_time = _get_current_time()

    # ------------------------------------------------------------------
    # 第一层：读取 L3 用户画像（稳定层）
    # ------------------------------------------------------------------
    profile_dict = get_current_profile()
    profile_text = _format_profile(profile_dict)

    # ------------------------------------------------------------------
    # 第二层：读取近期 L2 摘要（近期层）
    # ------------------------------------------------------------------
    l2_rows = get_recent_l2(limit=L2_INJECT_COUNT)
    l2_text = _format_l2_list(l2_rows)

    # ------------------------------------------------------------------
    # 第三层：读取最新 L1 摘要（当日层）
    # 同时提取 L1 的纯摘要文本，供动态层辅查询使用
    # ------------------------------------------------------------------
    l1_row  = get_latest_l1()
    l1_text = _format_l1(l1_row)

    # 只取 summary 字段，不带时间段和氛围括号，语义更干净
    l1_query = l1_row["summary"] if l1_row else None

    # ------------------------------------------------------------------
    # 第四层：语义检索（动态层）
    #
    # 查询来源组合：
    #   · 主查询 = query_text（用户当前消息）
    #   · 辅查询 = l1_query（最新 L1 摘要内容）
    #
    # 降级逻辑：
    #   主查询有值时 → 双路检索（主查询 + 辅查询）
    #   主查询为 None → 辅查询单路检索（用 L1 内容做检索）
    #   两者都为 None → 跳过，冷启动阶段无数据
    # ------------------------------------------------------------------
    rag_text = ""

    # 确定实际主查询：优先用用户消息，没有时降级用 L1 内容
    effective_query_a = query_text or l1_query

    if effective_query_a:
        # query_text 存在时，l1_query 作辅查询；否则辅查询为 None（避免重复查询）
        effective_query_b = l1_query if (query_text and l1_query) else None

        rag_results = _retrieve_and_merge(effective_query_a, effective_query_b)
        rag_text    = _format_rag_results(rag_results)

        if rag_text:
            print(f"[prompt_builder] 语义检索注入 {len(rag_results)} 条相关记忆")
        else:
            print(f"[prompt_builder] 语义检索无相关结果，动态层跳过")
    else:
        print(f"[prompt_builder] 冷启动阶段，跳过语义检索")

    # ------------------------------------------------------------------
    # 按模板拼装各段落
    # 有内容时拼入段落标题 + 内容 + 两个换行；没有内容时段落为空字符串
    # ------------------------------------------------------------------
    profile_section = PROFILE_SECTION_HEADER.format(content=profile_text) if profile_text else ""
    l2_section      = L2_SECTION_HEADER.format(content=l2_text)           if l2_text      else ""
    l1_section      = L1_SECTION_HEADER.format(content=l1_text)           if l1_text      else ""
    rag_section     = RAG_SECTION_HEADER.format(content=rag_text)          if rag_text     else ""

    # ------------------------------------------------------------------
    # 填入主模板，生成最终 system prompt
    # ------------------------------------------------------------------
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        current_time    = current_time,
        profile_section = profile_section,
        l2_section      = l2_section,
        l1_section      = l1_section,
        rag_section     = rag_section,    # 新增，空时为 ""
    )

    return system_prompt


# =============================================================================
# 直接运行此文件时：打印当前 system prompt
# =============================================================================

if __name__ == "__main__":
    print("=== prompt_builder.py 验证测试 ===\n")

    # ------------------------------------------------------------------
    # 测试一：不传 query_text（模拟应用启动时的状态）
    # ------------------------------------------------------------------
    print("--- 测试一：无 query_text ---\n")
    prompt_a = build_system_prompt()
    print(prompt_a)

    # ------------------------------------------------------------------
    # 测试二：传入 query_text（模拟正常对话时的状态）
    # ------------------------------------------------------------------
    print("\n--- 测试二：传入 query_text ---\n")
    prompt_b = build_system_prompt(query_text="最近睡眠很差，有点焦虑")
    print(prompt_b)

    print("\n--- 验证完成 ---")
    print("（若记忆为空属正常，首次运行时各层均无数据）")
    print("（若【相关记忆】段落不出现，说明向量索引为空或无相关结果，属正常）")
