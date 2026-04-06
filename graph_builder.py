"""
graph_builder.py — 知识图谱批处理核心模块
=====================================================================

职责：
    从已生成的 L1 摘要中批量提取三元组，写入 graph_nodes 和 graph_edges 表，
    同时维护实体与 keyword_pool 的归一化联动。

设计原则：
    · 不挂在实时对话链路上，作为独立后台批处理运行
    · 只处理 salience >= 0.25 的 L1（过滤纯闲聊）
    · 实体归一化分三档：高置信度自动合并、中置信度转交用户确认、低置信度独立新词
    · 三元组提取失败不中止整体流程，记录错误后跳过当条 L1

与其他模块的关系：
    · 读 L1 数据：database.get_l1_by_id / get_all_l1_ids_in_graph
    · 写图谱数据：database.get_or_create_node / save_edge
    · 归一化联动：database.get_all_canonical_keywords / save_keyword_with_alias
    · 别名确认：database.save_alias_conflict（写入 conflict_queue）
    · 向量计算：复用 vector_store 的嵌入模型实例
    · 调用模型：llm_client.call_local_summary

FastAPI 路由（在 main.py 中注册）：
    POST /graph/start    启动后台批处理线程
    POST /graph/stop     停止
    GET  /graph/status   查询进度

使用方法：
    # 后台线程模式（FastAPI 路由调用）
    from graph_builder import start_graph_build, stop_graph_build, get_graph_status

    # 命令行同步模式（调试用）
    python graph_builder.py
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from database import (
    get_l1_by_id,
    get_all_l1_ids_in_graph,
    get_or_create_node,
    save_edge,
    get_all_canonical_keywords,
    save_keyword_with_alias,
    save_alias_conflict,
)
from llm_client import call_local_summary, strip_thinking
from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 常量配置
# =============================================================================

# 实体归一化的向量相似度阈值
# 高于 HIGH_THRESHOLD：直接自动合并为别名
# 介于两者之间：写入 pending，转交用户确认
# 低于 LOW_THRESHOLD：独立为新规范词
ALIAS_HIGH_THRESHOLD = 0.95
ALIAS_LOW_THRESHOLD  = 0.85

# 每条 L1 最多提取的三元组数量（防止模型输出膨胀）
MAX_TRIPLES_PER_L1 = 5

# salience 过滤阈值：低于此值的 L1 跳过图谱提取（纯闲聊过滤）
MIN_SALIENCE = 0.25

# 合法的关系类型（七大类，与设计文档对齐）
VALID_RELATION_TYPES = {
    "TASK_STATUS",    # 任务状态类
    "OBSTACLE",       # 遇到障碍类
    "USES_DEPENDS",   # 使用/依赖类
    "BELONGS_TO",     # 归属类
    "EMOTION_STATE",  # 情感状态类
    "SOCIAL_EVENT",   # 社交事件类
    "TIME_ANCHOR",    # 时间归属类
}

# 合法的实体类型（五类）
VALID_ENTITY_TYPES = {"person", "project", "module", "concept", "time"}

# 后台线程消费间隔（秒）
BATCH_SLEEP_INTERVAL = 2


# =============================================================================
# 三元组提取 Prompt
# =============================================================================

EXTRACT_TRIPLES_PROMPT = """你是一个知识图谱提取助手。请从下面的 L1 摘要中提取知识三元组。

【关系类型（必须从以下七类中选一个填入 relation_type）】
TASK_STATUS     任务状态类（开始/进行中/完成/放弃某任务）
OBSTACLE        遇到障碍类（报错/卡住/待解决的问题）
USES_DEPENDS    使用或依赖类（使用了某工具/库/方案）
BELONGS_TO      归属类（某模块/功能属于某项目）
EMOTION_STATE   情感状态类（情绪、疲惫、开心、压力等）
SOCIAL_EVENT    社交事件类（与某人发生的互动）
TIME_ANCHOR     时间归属类（某事发生于某时间段）

【实体类型（必须从以下五类中选一个填入 subject_type / object_type）】
person   人物（如烧酒、撒老师）
project  项目/产品（如珊瑚菌、Ramaria）
module   代码模块/文件（如 vector_store.py、summarizer.py）
concept  抽象概念/技术（如 BM25、RRF、遗忘曲线）
time     时间节点（如 2026-03、夜间）

【提取优先级（重要！）】
1. 【最高优先】与已有状态发生变更或冲突的信息
   例："不用 Chroma 了改用 Qdrant" → 优先于普通的使用关系
2. 【次高优先】里程碑/完成类事件（relation_type = TASK_STATUS）
3. 【一般优先】其他关系

【硬性限制】
· 最多输出 {max_triples} 个三元组，超过时按优先级截断
· 主语和宾语必须是具体命名实体，不要用"用户"、"他"、"某系统"等模糊词
· 用"烧酒"指代用户，不要用"我"或真实姓名
· 优先使用【已知实体列表】中的名称，避免自造别名

【输出格式】
严格按照 JSON 数组输出，不要输出任何其他内容（不要加 markdown 代码块）：
[
  {{
    "subject":         "主语实体名",
    "subject_type":    "五类之一",
    "relation_type":   "七类之一",
    "relation_detail": "一句话描述具体关系内容",
    "object":          "宾语实体名",
    "object_type":     "五类之一"
  }}
]

无有效三元组时输出：[]

【已知实体列表（优先使用这里的名称）】
{entity_candidates}

【L1 摘要内容】
{l1_summary}
"""


# =============================================================================
# 批处理状态数据结构
# =============================================================================

@dataclass
class GraphBuildState:
    """
    全局批处理状态，结构参考 l1_batch.py 的 BatchState。
    所有字段通过外部 Lock 保护，不在 Lock 外直接修改。
    """
    # 运行状态
    # idle     — 未启动或已完成重置
    # running  — 正在处理
    # done     — 全部处理完毕
    # stopped  — 用户主动停止
    # error    — 异常中止
    status: str = "idle"

    # 进度数据
    total:     int = 0   # 本次批处理的 L1 总数
    done:      int = 0   # 已完成（成功+跳过+失败）
    succeeded: int = 0   # 成功提取三元组的 L1 数
    skipped:   int = 0   # 跳过（salience 太低或无有效三元组）
    failed:    int = 0   # 提取失败的 L1 数

    # 当前正在处理的 L1 信息
    current_l1_id: Optional[int] = None

    # 时间信息
    started_at:  Optional[str] = None
    finished_at: Optional[str] = None

    # 失败的 L1 id 列表，供排查
    failed_l1_ids: list = field(default_factory=list)

    # 停止标志（主线程写，后台线程读）
    stop_requested: bool = False

    def to_dict(self) -> dict:
        """序列化为字典，供 FastAPI 路由返回。"""
        return {
            "status":        self.status,
            "total":         self.total,
            "done":          self.done,
            "succeeded":     self.succeeded,
            "skipped":       self.skipped,
            "failed":        self.failed,
            "current_l1_id": self.current_l1_id,
            "started_at":    self.started_at,
            "finished_at":   self.finished_at,
            "failed_l1_ids": self.failed_l1_ids[:20],  # 最多返回前 20 个
        }


# =============================================================================
# 全局单例
# =============================================================================

_state       = GraphBuildState()
_state_lock  = threading.Lock()
_build_thread: Optional[threading.Thread] = None

# 嵌入模型缓存（懒加载，避免模块导入时就加载模型）
_embedding_model = None
_embedding_lock  = threading.Lock()


# =============================================================================
# 嵌入模型管理
# =============================================================================

def _get_embedding_model():
    """
    获取（或初始化）嵌入模型实例。
    使用双重检查锁保证线程安全，懒加载避免启动时拖慢服务。

    返回：
        SentenceTransformer 实例；加载失败时返回 None
    """
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    from config import EMBEDDING_MODEL
                    logger.info(f"正在加载嵌入模型：{EMBEDDING_MODEL}")
                    _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                    logger.info("嵌入模型加载完成")
                except Exception as e:
                    logger.error(f"嵌入模型加载失败 — {e}")
                    return None
    return _embedding_model


def _encode(text: str) -> Optional[np.ndarray]:
    """
    将文本编码为向量。

    参数：
        text — 待编码的文本

    返回：
        np.ndarray — 归一化后的向量；失败时返回 None
    """
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec
    except Exception as e:
        logger.warning(f"向量编码失败：{text[:30]}… — {e}")
        return None


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个归一化向量的余弦相似度。
    因为 _encode 已做归一化，直接点积即为余弦相似度。

    参数：
        v1, v2 — 已归一化的向量

    返回：
        float — 相似度，范围 [-1, 1]
    """
    return float(np.dot(v1, v2))


# =============================================================================
# 实体归一化
# =============================================================================

def _normalize_entity(raw_name: str, source_l1_id: int) -> str:
    """
    对模型提取的原始实体名做归一化，返回最终写入图谱的规范名。

    归一化流程：
      1. 计算 raw_name 的向量
      2. 与 keyword_pool 里所有规范词（canonical_id=NULL）计算余弦相似度
      3. 取相似度最高的规范词：
         · > 0.95：直接归入该规范词，返回规范词名
         · 0.85~0.95：写入 pending 状态，写入 conflict_queue 等待确认，
                       本次暂时返回 raw_name（等用户确认后再迁移）
         · < 0.85 或 keyword_pool 为空：new_name 独立成新规范词
      4. 无论哪种情况，都确保 raw_name 已写入 keyword_pool

    参数：
        raw_name     — 模型提取的原始实体名
        source_l1_id — 触发此归一化的 L1 摘要 id

    返回：
        str — 归一化后应写入 graph_nodes.entity_name 的名称
              （可能是规范词，也可能是 raw_name 本身）
    """
    # 计算原始名称的向量
    vec_new = _encode(raw_name)

    # 如果嵌入模型不可用，直接以 raw_name 写入，跳过归一化
    if vec_new is None:
        logger.warning(f"嵌入模型不可用，{raw_name!r} 跳过归一化，直接写入")
        save_keyword_with_alias(raw_name, canonical_id=None, alias_status="confirmed")
        return raw_name

    # 取出 keyword_pool 里所有规范词
    canonical_keywords = get_all_canonical_keywords()

    if not canonical_keywords:
        # keyword_pool 为空，直接写入新规范词
        save_keyword_with_alias(raw_name, canonical_id=None, alias_status="confirmed")
        return raw_name

    # 计算与所有规范词的相似度，找出最高的一个
    best_sim  = -1.0
    best_word = None
    best_id   = None

    for kw in canonical_keywords:
        # 跳过和自身完全一样的词（已存在的情况）
        if kw["keyword"] == raw_name:
            return raw_name   # 已是规范词，直接返回

        vec_existing = _encode(kw["keyword"])
        if vec_existing is None:
            continue

        sim = _cosine_similarity(vec_new, vec_existing)
        if sim > best_sim:
            best_sim  = sim
            best_word = kw["keyword"]
            best_id   = kw["id"]

    # 根据相似度阈值分三档处理
    if best_sim >= ALIAS_HIGH_THRESHOLD:
        # 档位一：高置信度，直接自动归入规范词
        logger.info(
            f"实体归一化（自动）：'{raw_name}' → '{best_word}'（相似度 {best_sim:.3f}）"
        )
        save_keyword_with_alias(raw_name, canonical_id=best_id, alias_status="confirmed")
        return best_word   # 使用规范词名写入图谱

    elif best_sim >= ALIAS_LOW_THRESHOLD:
        # 档位二：中置信度，写入 pending，推送给用户确认
        logger.info(
            f"实体归一化（待确认）：'{raw_name}' vs '{best_word}'（相似度 {best_sim:.3f}）"
        )
        # 先把 raw_name 写入 keyword_pool，状态为 pending
        new_kp_id = save_keyword_with_alias(
            raw_name,
            canonical_id=best_id,
            alias_status="pending",
        )
        # 写入 conflict_queue，等待用户在对话中确认
        save_alias_conflict(
            source_l1_id    = source_l1_id,
            old_word        = best_word,
            new_word        = raw_name,
            new_word_kp_id  = new_kp_id,
            canonical_kp_id = best_id,
            similarity      = best_sim,
        )
        # 本次暂时用 raw_name 写入图谱，等用户确认后 confirm_alias() 会迁移
        return raw_name

    else:
        # 档位三：低置信度，独立为新规范词
        logger.debug(
            f"实体归一化（新词）：'{raw_name}'（与最近规范词相似度 {best_sim:.3f}，低于阈值）"
        )
        save_keyword_with_alias(raw_name, canonical_id=None, alias_status="confirmed")
        return raw_name


# =============================================================================
# 三元组提取
# =============================================================================

def _build_entity_candidates() -> str:
    """
    从 keyword_pool 取出所有规范词，格式化为供 Prompt 注入的候选列表。

    返回：
        str — 用换行分隔的候选词列表；keyword_pool 为空时返回"（暂无已知实体）"
    """
    keywords = get_all_canonical_keywords()
    if not keywords:
        return "（暂无已知实体，请根据摘要内容自由提取）"
    # 最多注入前 80 个高频词，避免 Prompt 过长
    top_keywords = keywords[:80]
    return "\n".join(f"· {kw['keyword']}" for kw in top_keywords)


def _parse_triples(raw_text: str) -> Optional[list[dict]]:
    """
    从模型返回的原始文本中解析三元组列表。

    解析策略（三步递进）：
      第一步：剥离思考链后直接解析
      第二步：正则提取第一个 JSON 数组
      第三步：返回 None（解析彻底失败）

    参数：
        raw_text — 模型返回的原始文本

    返回：
        list[dict] — 三元组列表；无有效三元组时返回 []；解析失败时返回 None
    """
    # 第一步：剥离思考链
    stripped = strip_thinking(raw_text)

    # 第二步：直接尝试解析
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 第三步：正则提取第一个完整 JSON 数组
    # re.DOTALL 让 . 能跨行匹配（模型输出的 JSON 通常是多行格式）
    match = re.search(r'\[.*?\]', stripped, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning(f"三元组解析失败，原始输出：{raw_text[:200]}")
    return None


def _validate_triple(triple: dict) -> bool:
    """
    校验单条三元组的字段完整性和合法性。

    参数：
        triple — 从 _parse_triples() 返回列表中取出的单条字典

    返回：
        bool — True 表示校验通过，False 表示应丢弃
    """
    # 检查必填字段是否存在且非空
    required_fields = ("subject", "subject_type", "relation_type",
                       "relation_detail", "object", "object_type")
    for f in required_fields:
        if not str(triple.get(f, "")).strip():
            logger.debug(f"三元组缺少字段 {f!r}，已丢弃：{triple}")
            return False

    # 校验关系类型
    if triple["relation_type"] not in VALID_RELATION_TYPES:
        logger.debug(
            f"relation_type {triple['relation_type']!r} 不合法，已丢弃"
        )
        return False

    # 校验实体类型
    if triple["subject_type"] not in VALID_ENTITY_TYPES:
        logger.debug(
            f"subject_type {triple['subject_type']!r} 不合法，已丢弃"
        )
        return False

    if triple["object_type"] not in VALID_ENTITY_TYPES:
        logger.debug(
            f"object_type {triple['object_type']!r} 不合法，已丢弃"
        )
        return False

    # 主语和宾语不能相同（自环无意义）
    if triple["subject"].strip() == triple["object"].strip():
        logger.debug(f"主语和宾语相同，已丢弃：{triple['subject']}")
        return False

    return True


def _extract_triples_for_l1(l1_id: int) -> Optional[int]:
    """
    对单条 L1 摘要执行三元组提取，写入图谱。

    完整流程：
      1. 读取 L1 记录（同时检查 salience 过滤条件）
      2. 构建 Prompt，调用本地模型
      3. 解析并校验三元组列表
      4. 逐条执行实体归一化
      5. 写入 graph_nodes 和 graph_edges

    参数：
        l1_id — 需要处理的 L1 摘要 id

    返回：
        int  — 成功写入的三元组数量（0 表示跳过或无有效三元组）
        None — 模型调用失败时返回 None
    """
    # ── 第一步：读取 L1，检查 salience ──
    l1_row = get_l1_by_id(l1_id)
    if l1_row is None:
        logger.warning(f"找不到 L1 id={l1_id}，跳过")
        return 0

    salience = l1_row["salience"] if l1_row["salience"] is not None else 0.5
    if salience < MIN_SALIENCE:
        logger.debug(
            f"L1 {l1_id} salience={salience:.2f} < {MIN_SALIENCE}，跳过图谱提取"
        )
        return 0

    l1_summary = l1_row["summary"]

    # ── 第二步：构建 Prompt，调用本地模型 ──
    entity_candidates = _build_entity_candidates()
    prompt = EXTRACT_TRIPLES_PROMPT.format(
        max_triples       = MAX_TRIPLES_PER_L1,
        entity_candidates = entity_candidates,
        l1_summary        = l1_summary,
    )

    raw_output = call_local_summary(
        messages = [{"role": "user", "content": prompt}],
        caller   = "graph_builder",
    )

    if raw_output is None:
        logger.error(f"L1 {l1_id} 模型调用失败")
        return None

    # ── 第三步：解析并校验三元组 ──
    triples = _parse_triples(raw_output)
    if triples is None:
        logger.error(f"L1 {l1_id} 三元组解析失败，原始输出：{raw_output[:200]}")
        return None

    if not triples:
        logger.debug(f"L1 {l1_id} 无有效三元组")
        return 0

    # 校验并过滤
    valid_triples = [t for t in triples if _validate_triple(t)]
    # 截断到上限
    valid_triples = valid_triples[:MAX_TRIPLES_PER_L1]

    if not valid_triples:
        logger.debug(f"L1 {l1_id} 校验后无合法三元组")
        return 0

    # ── 第四步：实体归一化 + 写入图谱 ──
    written = 0
    for triple in valid_triples:
        try:
            # 归一化主语
            subject_name = _normalize_entity(
                triple["subject"].strip(), source_l1_id=l1_id
            )
            # 归一化宾语
            object_name = _normalize_entity(
                triple["object"].strip(), source_l1_id=l1_id
            )

            # 获取或创建主语节点
            source_node_id = get_or_create_node(
                entity_name  = subject_name,
                entity_type  = triple["subject_type"],
                source_l1_id = l1_id,
            )

            # 获取或创建宾语节点
            target_node_id = get_or_create_node(
                entity_name  = object_name,
                entity_type  = triple["object_type"],
                source_l1_id = l1_id,
            )

            # 写入边
            save_edge(
                source_node_id  = source_node_id,
                target_node_id  = target_node_id,
                relation_type   = triple["relation_type"],
                relation_detail = triple["relation_detail"].strip(),
                source_l1_id    = l1_id,
            )

            written += 1
            logger.debug(
                f"三元组写入：{subject_name} "
                f"—[{triple['relation_type']}]→ {object_name}"
            )

        except Exception as e:
            logger.warning(f"L1 {l1_id} 单条三元组写入失败 — {e}")
            continue

    logger.info(f"L1 {l1_id} 图谱提取完成，写入 {written} 条三元组")
    return written


# =============================================================================
# 查询待处理 L1
# =============================================================================

def _get_pending_l1_ids() -> list[int]:
    """
    查询所有"已生成 L1 但尚未提取图谱三元组"的 L1 id 列表。

    逻辑：
      · 取出所有 memory_l1 的 id
      · 排除掉已在 graph_edges 里出现过 source_l1_id 的 id
      · 同时排除 salience < MIN_SALIENCE 的 L1（直接在这里过滤，
        避免把它们放进批处理队列再跳过，浪费一次模型调用）

    返回：
        list[int] — 待处理的 L1 id 列表，按 id 升序排列
    """
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 取出所有满足 salience 条件且尚未进图谱的 L1
    cursor.execute(
        """
        SELECT id FROM memory_l1
        WHERE (salience IS NULL OR salience >= ?)
          AND id NOT IN (
              SELECT DISTINCT source_l1_id FROM graph_edges
          )
        ORDER BY id ASC
        """,
        (MIN_SALIENCE,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [row["id"] for row in rows]


# =============================================================================
# 后台批处理线程
# =============================================================================

def _build_worker(pending_ids: list[int]):
    """
    后台批处理线程的主函数。
    逐个处理 pending_ids 中的 L1，每处理一个前检查停止标志。

    参数：
        pending_ids — 待处理的 L1 id 列表
    """
    global _state

    logger.info(f"图谱构建线程启动，共 {len(pending_ids)} 条 L1 待处理")

    for l1_id in pending_ids:

        # ── 检查停止标志 ──
        with _state_lock:
            if _state.stop_requested:
                _state.status      = "stopped"
                _state.finished_at = _now_iso()
                logger.info("图谱构建已停止（用户请求）")
                return

        # ── 更新当前处理状态 ──
        with _state_lock:
            _state.current_l1_id = l1_id

        # ── 执行三元组提取 ──
        result = _extract_triples_for_l1(l1_id)

        # ── 更新进度 ──
        with _state_lock:
            _state.done += 1
            if result is None:
                # 模型调用失败
                _state.failed += 1
                _state.failed_l1_ids.append(l1_id)
            elif result == 0:
                # 跳过（salience 太低或无有效三元组）
                _state.skipped += 1
            else:
                # 成功提取
                _state.succeeded += 1

        logger.info(
            f"L1 {l1_id} 处理完成（{_state.done}/{_state.total}）"
            f" 结果：{'失败' if result is None else f'{result}条三元组'}"
        )

        # 两条 L1 之间稍微休眠，避免对模型服务造成持续压力
        time.sleep(BATCH_SLEEP_INTERVAL)

    # ── 全部处理完毕 ──
    with _state_lock:
        _state.status        = "done"
        _state.finished_at   = _now_iso()
        _state.current_l1_id = None

    logger.info(
        f"图谱构建完成：成功 {_state.succeeded}，"
        f"跳过 {_state.skipped}，失败 {_state.failed}"
    )


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# 对外接口：后台线程版（供 FastAPI 路由调用）
# =============================================================================

def start_graph_build() -> dict:
    """
    启动图谱构建后台线程，非阻塞返回。

    如果已有批处理在运行，直接返回当前状态，不重复启动。
    调用方通过轮询 get_graph_status() 获取进度。

    返回：
        dict — 当前批处理状态（与 get_graph_status() 格式相同）
    """
    global _state, _build_thread

    with _state_lock:
        # 已在运行时直接返回当前状态
        if _state.status == "running":
            return _state.to_dict()

        # 查询待处理 L1
        try:
            pending_ids = _get_pending_l1_ids()
        except Exception as e:
            logger.error(f"查询待处理 L1 失败 — {e}")
            _state.status = "error"
            return _state.to_dict()

        if not pending_ids:
            _state.status = "done"
            _state.total  = 0
            return _state.to_dict()

        # 初始化状态
        _state = GraphBuildState(
            status     = "running",
            total      = len(pending_ids),
            started_at = _now_iso(),
        )

    # 启动后台线程（在 Lock 外启动，避免持锁时阻塞）
    _build_thread = threading.Thread(
        target  = _build_worker,
        args    = (pending_ids,),
        name    = "GraphBuildWorker",
        daemon  = True,
    )
    _build_thread.start()
    logger.info(f"图谱构建线程已启动，待处理 L1 数：{len(pending_ids)}")

    with _state_lock:
        return _state.to_dict()


def stop_graph_build() -> dict:
    """
    请求停止图谱构建。

    不立即中止，而是设置停止标志，后台线程在当前 L1 处理完后退出。

    返回：
        dict — 当前批处理状态
    """
    global _state

    with _state_lock:
        if _state.status != "running":
            return _state.to_dict()
        _state.stop_requested = True

    logger.info("已发送停止请求，等待当前 L1 处理完毕")

    with _state_lock:
        return _state.to_dict()


def get_graph_status() -> dict:
    """
    查询当前图谱构建状态，供 FastAPI 路由轮询。

    返回：
        dict — GraphBuildState.to_dict() 的结果
    """
    with _state_lock:
        return _state.to_dict()


def get_graph_pending_count() -> int:
    """
    查询当前待处理 L1 数量，不启动批处理。
    供前端页面初始化时显示"N 条 L1 待构建图谱"。

    返回：
        int — 待处理数量；查询失败时返回 -1
    """
    try:
        return len(_get_pending_l1_ids())
    except Exception as e:
        logger.error(f"查询待处理 L1 数量失败 — {e}")
        return -1


# =============================================================================
# 启动时加载图谱到内存（供 main.py lifespan 调用）
# =============================================================================

def load_graph_to_memory():
    """
    从 SQLite 读取全部节点和边，构建 NetworkX 内存图。
    在 main.py 的 lifespan startup 阶段调用一次。

    图对象存储在模块级变量 _nx_graph 中，供 vector_store.retrieve_graph() 使用。
    节点和边数量为 0 时（图谱还没建立）正常返回，不报错。
    """
    global _nx_graph

    try:
        import networkx as nx
        import sqlite3
        from config import DB_PATH

        conn   = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 读取所有节点
        cursor.execute("SELECT id, entity_name, entity_type FROM graph_nodes")
        nodes = cursor.fetchall()

        # 读取所有边
        cursor.execute(
            "SELECT source_node_id, target_node_id, relation_type, "
            "relation_detail, source_l1_id FROM graph_edges"
        )
        edges = cursor.fetchall()
        conn.close()

        # 构建无向图（无向：图遍历时双向可达）
        G = nx.Graph()

        for node in nodes:
            G.add_node(
                node["id"],
                entity_name = node["entity_name"],
                entity_type = node["entity_type"],
            )

        for edge in edges:
            # 用列表存 l1_id，同一对节点可能有多条边（来自不同 L1）
            if G.has_edge(edge["source_node_id"], edge["target_node_id"]):
                # 边已存在，追加 l1_id
                G[edge["source_node_id"]][edge["target_node_id"]]["l1_ids"].append(
                    edge["source_l1_id"]
                )
            else:
                G.add_edge(
                    edge["source_node_id"],
                    edge["target_node_id"],
                    relation_type   = edge["relation_type"],
                    relation_detail = edge["relation_detail"],
                    l1_ids          = [edge["source_l1_id"]],
                )

        _nx_graph = G
        logger.info(
            f"NetworkX 图谱加载完成：{G.number_of_nodes()} 个节点，"
            f"{G.number_of_edges()} 条边"
        )

    except ImportError:
        logger.warning("networkx 未安装，图谱检索功能不可用。请执行：pip install networkx")
        _nx_graph = None
    except Exception as e:
        logger.error(f"图谱加载失败 — {e}")
        _nx_graph = None


# 模块级图对象，load_graph_to_memory() 写入，retrieve_graph() 读取
_nx_graph = None


def get_nx_graph():
    """
    获取当前内存中的 NetworkX 图对象。
    供 vector_store.retrieve_graph() 调用。

    返回：
        nx.Graph — 图对象；尚未加载时返回 None
    """
    return _nx_graph


def reload_graph():
    """
    重新从 SQLite 加载图谱到内存。
    在批处理完成后或手动触发时调用，确保内存图与数据库同步。
    """
    load_graph_to_memory()
    logger.info("图谱已重新加载")


# =============================================================================
# 命令行同步模式（调试用）
# =============================================================================

def run_build_cli() -> dict:
    """
    同步执行图谱构建，阻塞直到全部完成。命令行调试用。

    返回：
        dict — 最终构建状态
    """
    pending_ids = _get_pending_l1_ids()

    if not pending_ids:
        print("没有待处理的 L1（所有满足条件的 L1 均已提取图谱）")
        return {"status": "done", "total": 0}

    total     = len(pending_ids)
    succeeded = 0
    skipped   = 0
    failed    = 0
    failed_ids = []

    print(f"\n开始图谱构建，共 {total} 条 L1 待处理\n")

    for idx, l1_id in enumerate(pending_ids, start=1):
        result = _extract_triples_for_l1(l1_id)

        if result is None:
            failed += 1
            failed_ids.append(l1_id)
            mark = "✗"
        elif result == 0:
            skipped += 1
            mark = "–"
        else:
            succeeded += 1
            mark = "✓"

        print(f"[{idx:>4}/{total}] {mark}  L1 {l1_id}（{result if result is not None else '失败'}）")

    print()
    print("=" * 50)
    print(f"  图谱构建完成")
    print(f"  成功提取三元组 : {succeeded} 条 L1")
    print(f"  跳过           : {skipped} 条 L1")
    print(f"  失败           : {failed} 条 L1")
    if failed_ids:
        print(f"  失败的 L1 id   : {failed_ids}")
    print("=" * 50)

    # 构建完成后重新加载内存图
    reload_graph()

    return {
        "status":    "done",
        "total":     total,
        "succeeded": succeeded,
        "skipped":   skipped,
        "failed":    failed,
        "failed_ids": failed_ids,
    }


# =============================================================================
# 直接运行此文件时：命令行调试模式
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="珊瑚菌知识图谱构建工具")
    parser.add_argument(
        "--status",
        action="store_true",
        help="只查询待处理数量，不启动构建",
    )
    args = parser.parse_args()

    if args.status:
        count = get_graph_pending_count()
        if count < 0:
            print("查询失败，请检查数据库连接。")
        elif count == 0:
            print("所有满足条件的 L1 均已提取图谱，无待处理项。")
        else:
            print(f"共 {count} 条 L1 待提取图谱三元组。")
            print("运行 python graph_builder.py 开始构建。")
    else:
        run_build_cli()
