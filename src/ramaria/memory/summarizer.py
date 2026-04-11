"""
src/ramaria/memory/summarizer.py — L1 摘要生成模块

负责在 session 结束后，调用本地模型生成结构化摘要，
并写入 memory_l1 表。

核心流程：
    1.  从数据库读取指定 session 的全部消息
    2.  从 keyword_pool 表读取历史关键词，作为候选列表
    3.  将消息格式化为纯文本对话，连同候选词一起填入 Prompt 模板
    4.  调用本地 LM Studio API 生成摘要
    5.  解析模型返回的 JSON 结果
    6.  校验字段合法性，写入 memory_l1 表
    7.  将本次使用的关键词同步写回 keyword_pool 表
    8.  写入 L1 向量索引
    9.  提取新信息，静默写入 L3 用户画像（profile_manager）
    10. 触发冲突检测（conflict_checker）
    11. 触发路径A — L2 条数检查（merger）

关键词词典机制：
    词典冷启动时为空，随使用量积累逐渐收敛。
    词典较小时（< KEYWORD_INJECT_THRESHOLD 条）注入全部词条；
    词典较大时只注入高频前 KEYWORD_INJECT_LIMIT 条，避免 Prompt 过长。

"""

import json
import re

from ramaria.config import TIME_PERIOD_OPTIONS
from ramaria.core.llm_client import call_local_summary, strip_thinking
from ramaria.storage.database import (
    get_all_keywords,
    get_messages,
    get_top_keywords,
    save_l1_summary,
    upsert_keywords,
)

from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Prompt 模板
# =============================================================================
#
# 从 config.py 迁移到此处，由本模块统一维护。
# 有四个版本，summarizer 根据"词典是否为空"和"是否需要情感字段"自动选择。
#
# valence 档位（五档，模型只能从这五个值中选一个）：
#   -1.0  非常消极（崩溃、绝望、强烈愤怒）
#   -0.5  偏消极  （疲惫、担心、轻度低落）
#    0.0  中性    （平静日常、技术问答）
#    0.5  偏积极  （放松、满意、轻度开心）
#    1.0  非常积极（兴奋、强烈成就感、里程碑）
#
# salience 档位（五档）：
#   0.0   平淡（纯闲聊或技术问答）
#   0.25  轻微
#   0.5   中等
#   0.75  较高（情绪明显，或话题对用户有重要意义）
#   1.0   极高（强烈情绪，或人生重要节点/里程碑）

_L1_PROMPT_BASE = """你是一个对话摘要助手。请根据下面的对话内容，生成一条结构化摘要。

【输出格式要求】
严格按照以下 JSON 格式输出，不要输出任何其他内容（不要加 markdown 代码块，不要加说明文字）：

{{
  "summary": "第三人称客观描述本次对话的核心结论，一句话，不超过50字",
  "keywords": "3到5个名词标签，用英文逗号分隔，标签之间不能语义重复",
  "time_period": "从以下六个选项中选一个：清晨、上午、下午、傍晚、夜间、深夜",
  "atmosphere": "四字以内描述对话整体氛围，例如：专注高效、轻松愉快、情绪低落",
  "valence": 0.0,
  "salience": 0.5
}}

【valence 情绪效价说明】
只能从以下五个值中选一个，不要输出其他数值：
-1.0  非常消极（崩溃、绝望、强烈愤怒）
-0.5  偏消极（疲惫、担心、轻度低落）
 0.0  中性（平静日常、技术问答、无明显情绪）
 0.5  偏积极（放松、满意、轻度开心）
 1.0  非常积极（兴奋、强烈成就感、里程碑）

【salience 情感显著性说明】
只能从以下五个值中选一个，不要输出其他数值：
0.0   平淡（纯闲聊或技术问答，无情感投入）
0.25  轻微（有轻微情绪但不重要）
0.5   中等（正常对话，有情感内容）
0.75  较高（情绪明显，或话题对用户有重要意义）
1.0   极高（强烈情绪，或人生重要节点/里程碑）

【其他字段说明】
- summary：只记结论，不记过程；用"烧酒"指代用户；客观陈述，不加主观评价
- keywords：只取名词，不取动词或形容词；避免同义词重复
- time_period：根据对话发生的时间判断；若无法判断，根据内容氛围推断
- atmosphere：优先反映整体基调；四字以内，不超过四字
"""

# 不含关键词候选列表的版本（词典为空时使用）
L1_SUMMARY_PROMPT_WITH_EMOTIONS = (
    _L1_PROMPT_BASE
    + """
【对话内容】
{conversation}
"""
)

# 含关键词候选列表的版本（词典有内容时使用）
L1_SUMMARY_PROMPT_WITH_EMOTIONS_AND_KEYWORDS = (
    _L1_PROMPT_BASE
    + """
【关键词候选列表】
以下是历史上出现过的关键词，请优先从中选取语义匹配的词，确实无法匹配时再新造词：
{keyword_candidates}

【对话内容】
{conversation}
"""
)


# =============================================================================
# 词典注入参数
# =============================================================================

# 词典词条数低于此阈值时，全量注入所有词条
KEYWORD_INJECT_THRESHOLD = 100

# 词典超过阈值时，最多注入多少个候选词
KEYWORD_INJECT_LIMIT = 50


# =============================================================================
# 对话格式化
# =============================================================================

def _format_conversation(messages) -> str:
    """
    将消息列表格式化为纯文本对话，供 Prompt 填入使用。

    参数：
        messages — get_messages() 返回的 sqlite3.Row 列表

    返回：
        str — 每行一条消息，格式为"角色：内容"
    """
    lines = []
    for msg in messages:
        role_label = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role_label}：{msg['content']}")
    return "\n".join(lines)


# =============================================================================
# 关键词候选列表处理
# =============================================================================

def _get_keyword_candidates() -> str:
    """
    从词典表读取候选关键词，格式化为逗号分隔的字符串。

    词典为空时返回空字符串（调用方据此选择不含候选词的 Prompt 模板）。

    返回：
        str — 中文逗号分隔的关键词字符串；词典为空时返回 ""
    """
    try:
        all_kws = get_all_keywords()

        if not all_kws:
            return ""

        if len(all_kws) <= KEYWORD_INJECT_THRESHOLD:
            candidates = all_kws
        else:
            candidates = get_top_keywords(limit=KEYWORD_INJECT_LIMIT)

        return "，".join(candidates)

    except Exception as e:
        logger.warning(f"读取关键词词典失败，将使用无候选词版 Prompt — {e}")
        return ""


def _extract_keywords_list(keywords_str: str | None) -> list[str]:
    """
    将关键词字段字符串拆分为列表，用于写回词典。

    参数：
        keywords_str — 关键词字符串，如 "数据库,后端,验证"；允许传入 None

    返回：
        list[str] — 关键词列表；传入 None 或空字符串时返回空列表
    """
    if not keywords_str:
        return []

    # 同时兼容中文逗号和英文逗号
    keywords_str = keywords_str.replace("，", ",")
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]


# =============================================================================
# 解析与校验模型输出
# =============================================================================

def _parse_summary_json(raw_text: str) -> dict | None:
    """
    从模型返回的原始文本中解析出结构化摘要字段。

    解析策略（三步递进）：
        第一步：直接尝试解析
        第二步：剥离思考链后再次尝试
        第三步：用正则提取第一个完整 JSON 对象

    返回：
        dict — 包含 summary / keywords / time_period / atmosphere 等键
        None — 三步都失败时返回 None
    """
    # 第一步：直接尝试解析
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 第二步：剥离思考链后再次解析
    stripped = strip_thinking(raw_text)
    if stripped != raw_text:
        logger.debug("检测到思考链输出，已自动剥离，重新解析")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 第三步：用正则提取第一个完整 JSON 对象
    match = re.search(r'\{.*?\}', stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法从模型输出中解析 JSON，原始输出：{raw_text[:200]}")
    return None


# valence 合法档位集合
_VALID_VALENCE  = {-1.0, -0.5, 0.0, 0.5, 1.0}
# salience 合法档位集合
_VALID_SALIENCE = {0.0, 0.25, 0.5, 0.75, 1.0}


def _validate_and_fix(parsed: dict) -> dict:
    """
    校验解析结果的字段完整性和合法性，对不合规的值进行降级处理。

    参数：
        parsed — _parse_summary_json() 返回的字典

    返回：
        dict — 校验/修复后的字典，六个字段均有值（部分允许为 None）
    """
    result = {}

    # summary：必填，缺失或为空时用占位文本替代
    result["summary"] = str(parsed.get("summary", "")).strip()
    if not result["summary"]:
        result["summary"] = "（摘要生成失败，内容为空）"

    # keywords：允许为空
    keywords = str(parsed.get("keywords", "")).strip()
    result["keywords"] = keywords if keywords else None

    # time_period：严格六选一校验
    tp = str(parsed.get("time_period", "")).strip()
    result["time_period"] = tp if tp in TIME_PERIOD_OPTIONS else None
    if result["time_period"] is None and tp:
        logger.warning(f"time_period 值 {tp!r} 不合法，已置为 None")

    # atmosphere：四字以内，超长时截断
    atm = str(parsed.get("atmosphere", "")).strip()
    if len(atm) > 4:
        logger.warning(f"atmosphere 值 {atm!r} 超过四字，已截断")
        atm = atm[:4]
    result["atmosphere"] = atm if atm else None

    # valence：五档浮点校验，自动对齐到最近合法档位
    try:
        raw_valence = float(parsed.get("valence", 0.0))
        closest     = min(_VALID_VALENCE, key=lambda v: abs(v - raw_valence))
        if closest != raw_valence:
            logger.debug(
                f"valence {raw_valence} 不在合法档位，自动对齐到 {closest}"
            )
        result["valence"] = closest
    except (ValueError, TypeError):
        logger.warning("valence 解析失败，使用默认值 0.0")
        result["valence"] = 0.0

    # salience：五档浮点校验，自动对齐到最近合法档位
    try:
        raw_salience = float(parsed.get("salience", 0.5))
        closest      = min(_VALID_SALIENCE, key=lambda v: abs(v - raw_salience))
        if closest != raw_salience:
            logger.debug(
                f"salience {raw_salience} 不在合法档位，自动对齐到 {closest}"
            )
        result["salience"] = closest
    except (ValueError, TypeError):
        logger.warning("salience 解析失败，使用默认值 0.5")
        result["salience"] = 0.5

    return result


# =============================================================================
# 对外接口
# =============================================================================

def generate_l1_summary(session_id: int) -> int | None:
    """
    为指定 session 生成 L1 摘要并写入数据库。
    这是本模块唯一的对外接口，由 session_manager 在 session 结束时调用。

    完整流程：
        1.  读取 session 的全部消息
        2.  读取关键词词典，获取候选词列表
        3.  格式化对话文本，选择合适的 Prompt 模板填入内容
        4.  调用本地模型生成摘要
        5.  解析 JSON 输出
        6.  校验字段合法性，写入 memory_l1 表
        7.  将本次关键词同步写回 keyword_pool 词典
        8.  将 L1 摘要写入 Chroma 向量索引
        9.  提取新信息，静默写入 L3 用户画像（profile_manager）
        10. 触发冲突检测（conflict_checker）
        11. 触发 L2 条数检查（merger，路径A）

    参数：
        session_id — 需要生成摘要的 session id

    返回：
        int  — 成功时返回新写入的 L1 记录 id
        None — 任何步骤失败时返回 None（不抛出异常，不阻断主流程）
    """
    logger.info(f"开始为 session {session_id} 生成 L1 摘要")

    # 第一步：读取消息
    messages = get_messages(session_id)
    if not messages:
        logger.debug(f"session {session_id} 没有消息，跳过摘要生成")
        return None

    logger.debug(f"读取到 {len(messages)} 条消息")

    # 第二步：读取关键词候选列表
    keyword_candidates = _get_keyword_candidates()
    if keyword_candidates:
        logger.debug(
            f"词典候选词已注入"
            f"（共 {len(keyword_candidates.split('，'))} 个）"
        )
    else:
        logger.debug("词典为空（冷启动阶段），使用基础 Prompt")

    # 第三步：格式化对话文本，选择 Prompt 模板
    conversation_text = _format_conversation(messages)

    if keyword_candidates:
        prompt = L1_SUMMARY_PROMPT_WITH_EMOTIONS_AND_KEYWORDS.format(
            conversation       = conversation_text,
            keyword_candidates = keyword_candidates,
        )
    else:
        prompt = L1_SUMMARY_PROMPT_WITH_EMOTIONS.format(
            conversation = conversation_text,
        )

    # 第四步：调用本地模型
    raw_output = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="summarizer",
    )
    if raw_output is None:
        logger.error(f"模型调用失败，session {session_id} 摘要生成中止")
        return None

    logger.debug(f"模型原始输出：{raw_output[:200]}")

    # 第五步：解析 JSON
    parsed = _parse_summary_json(raw_output)
    if parsed is None:
        logger.error("JSON 解析失败，写入降级摘要")
        l1_id = save_l1_summary(
            session_id  = session_id,
            summary     = f"（自动摘要失败，原始输出已记录）{raw_output[:100]}",
            keywords    = None,
            time_period = None,
            atmosphere  = None,
            valence     = 0.0,
            salience    = 0.5,
        )
        return l1_id

    # 第六步：校验字段，写入 memory_l1 表
    validated = _validate_and_fix(parsed)

    l1_id = save_l1_summary(
        session_id  = session_id,
        summary     = validated["summary"],
        keywords    = validated["keywords"],
        time_period = validated["time_period"],
        atmosphere  = validated["atmosphere"],
        valence     = validated["valence"],
        salience    = validated["salience"],
    )

    logger.info(f"L1 摘要已写入，id = {l1_id}")
    logger.debug(
        f"summary={validated['summary']} | "
        f"keywords={validated['keywords']} | "
        f"time_period={validated['time_period']} | "
        f"atmosphere={validated['atmosphere']}"
    )

    # 第七步：将本次关键词写回 keyword_pool 词典
    if validated["keywords"]:
        try:
            kw_list = _extract_keywords_list(validated["keywords"])
            upsert_keywords(kw_list)
            logger.debug(f"关键词已写回词典：{kw_list}")
        except Exception as e:
            logger.warning(f"关键词写回词典失败 — {e}")

    # 第八步：将 L1 摘要写入向量索引
    try:
        from ramaria.storage.vector_store import index_l1
        from ramaria.storage.database import get_l1_by_id as _get_l1
        l1_row = _get_l1(l1_id)
        index_l1(
            l1_id      = l1_id,
            summary    = validated["summary"],
            keywords   = validated["keywords"],
            session_id = session_id,
            created_at = l1_row["created_at"] if l1_row else None,
            salience   = validated["salience"],
        )
    except Exception as e:
        logger.warning(f"L1 向量索引写入失败 — {e}")

    # 第九步：提取新信息，静默写入 L3 用户画像
    # 必须在 conflict_checker 之前执行，冲突检测需要拿最新的 L3 比对
    try:
        from ramaria.memory.profile_manager import extract_and_update
        extract_and_update(l1_id)
    except Exception as e:
        logger.warning(f"画像提取时出现异常 — {e}")

    # 第十步：触发冲突检测
    try:
        from ramaria.memory.conflict_checker import check_conflicts
        check_conflicts(l1_id)
    except Exception as e:
        logger.warning(f"冲突检测时出现异常 — {e}")

    # 第十一步：触发路径A — L2 条数检查
    # 使用懒加载 import 避免 summarizer ↔ merger 循环依赖
    try:
        from ramaria.memory.merger import check_and_merge
        check_and_merge()
    except Exception as e:
        logger.warning(f"L2 合并检查时出现异常 — {e}")

    return l1_id