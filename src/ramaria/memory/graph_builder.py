"""
src/ramaria/memory/graph_builder.py — 知识图谱批处理核心模块

职责：
    从已生成的 L1 摘要中批量提取三元组，写入 graph_nodes 和 graph_edges 表，
    同时维护实体与 keyword_pool 的归一化联动。

设计原则：
    · 不挂在实时对话链路上，作为独立后台批处理运行
    · 只处理 salience >= MIN_SALIENCE 的 L1（过滤纯闲聊）
    · 实体归一化分三档：高置信度自动合并、中置信度转交用户确认、低置信度独立新词

"""

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ramaria.config import DB_PATH, EMBEDDING_MODEL
from ramaria.core.llm_client import call_local_summary, strip_thinking
from ramaria.storage.database import (
    get_all_canonical_keywords,
    get_all_l1_ids_in_graph,
    get_l1_by_id,
    get_or_create_node,
    save_alias_conflict,
    save_edge,
    save_keyword_with_alias,
)

from ramaria.logger import get_logger

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

# 每条 L1 最多提取的三元组数量
MAX_TRIPLES_PER_L1 = 5

# salience 过滤阈值：低于此值的 L1 跳过图谱提取（纯闲聊过滤）
MIN_SALIENCE = 0.25

# 合法的关系类型（七大类）
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

【提取优先级】
1. 【最高优先】与已有状态发生变更或冲突的信息
2. 【次高优先】里程碑/完成类事件（relation_type = TASK_STATUS）
3. 【一般优先】其他关系

【硬性限制】
· 最多输出 {max_triples} 个三元组，超过时按优先级截断
· 主语和宾语必须是具体命名实体，不要用"用户"、"他"等模糊词
· 用"烧酒"指代用户
· 优先使用【已知实体列表】中的名称，避免自造别名

【输出格式】
严格按照 JSON 数组输出，不要输出任何其他内容：
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
    全局批处理状态，由后台线程写入，主线程/前端读取。
    所有字段通过外部 Lock 保护。
    """
    status: str = "idle"   # idle / running / done / stopped / error

    total:     int = 0
    done:      int = 0
    succeeded: int = 0
    skipped:   int = 0
    failed:    int = 0

    current_l1_id: Optional[int] = None

    started_at:  Optional[str] = None
    finished_at: Optional[str] = None

    failed_l1_ids: list = field(default_factory=list)
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
            "failed_l1_ids": self.failed_l1_ids[:20],
        }


# =============================================================================
# 全局单例
# =============================================================================

_state       = GraphBuildState()
_state_lock  = threading.Lock()
_build_thread: Optional[threading.Thread] = None

# 嵌入模型缓存（懒加载）
_embedding_model = None
_embedding_lock  = threading.Lock()


# =============================================================================
# 嵌入模型管理
# =============================================================================

def _get_embedding_model():
    """
    获取（或初始化）嵌入模型实例。
    双重检查锁保证线程安全，懒加载避免启动时拖慢服务。
    """
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"正在加载嵌入模型：{EMBEDDING_MODEL}")
                    _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                    logger.info("嵌入模型加载完成")
                except Exception as e:
                    logger.error(f"嵌入模型加载失败 — {e}")
                    return None
    return _embedding_model


def _encode(text: str) -> Optional[np.ndarray]:
    """将文本编码为归一化向量。失败时返回 None。"""
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        return model.encode(text, normalize_embeddings=True)
    except Exception as e:
        logger.warning(f"向量编码失败：{text[:30]}… — {e}")
        return None


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个归一化向量的余弦相似度。
    因为 _encode 已做归一化，直接点积即为余弦相似度。
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
        2. 与 keyword_pool 里所有规范词计算余弦相似度
        3. 取相似度最高的规范词：
           · > 0.95：直接归入该规范词
           · 0.85~0.95：写入 pending，等待用户确认
           · < 0.85 或词典为空：独立为新规范词
    """
    vec_new = _encode(raw_name)

    if vec_new is None:
        logger.warning(f"嵌入模型不可用，{raw_name!r} 跳过归一化")
        save_keyword_with_alias(raw_name, canonical_id=None, alias_status="confirmed")
        return raw_name

    canonical_keywords = get_all_canonical_keywords()

    if not canonical_keywords:
        save_keyword_with_alias(raw_name, canonical_id=None, alias_status="confirmed")
        return raw_name

    best_sim  = -1.0
    best_word = None
    best_id   = None

    for kw in canonical_keywords:
        if kw["keyword"] == raw_name:
            return raw_name

        vec_existing = _encode(kw["keyword"])
        if vec_existing is None:
            continue

        sim = _cosine_similarity(vec_new, vec_existing)
        if sim > best_sim:
            best_sim  = sim
            best_word = kw["keyword"]
            best_id   = kw["id"]

    if best_sim >= ALIAS_HIGH_THRESHOLD:
        logger.info(
            f"实体归一化（自动）：'{raw_name}' → '{best_word}'"
            f"（相似度 {best_sim:.3f}）"
        )
        save_keyword_with_alias(raw_name, canonical_id=best_id, alias_status="confirmed")
        return best_word

    elif best_sim >= ALIAS_LOW_THRESHOLD:
        logger.info(
            f"实体归一化（待确认）：'{raw_name}' vs '{best_word}'"
            f"（相似度 {best_sim:.3f}）"
        )
        new_kp_id = save_keyword_with_alias(
            raw_name, canonical_id=best_id, alias_status="pending"
        )
        save_alias_conflict(
            source_l1_id    = source_l1_id,
            old_word        = best_word,
            new_word        = raw_name,
            new_word_kp_id  = new_kp_id,
            canonical_kp_id = best_id,
            similarity      = best_sim,
        )
        return raw_name

    else:
        logger.debug(
            f"实体归一化（新词）：'{raw_name}'"
            f"（与最近规范词相似度 {best_sim:.3f}，低于阈值）"
        )
        save_keyword_with_alias(raw_name, canonical_id=None, alias_status="confirmed")
        return raw_name


# =============================================================================
# 三元组提取
# =============================================================================

def _build_entity_candidates() -> str:
    """
    从 keyword_pool 取出规范词，格式化为供 Prompt 注入的候选列表。
    """
    keywords = get_all_canonical_keywords()
    if not keywords:
        return "（暂无已知实体，请根据摘要内容自由提取）"
    top_keywords = keywords[:80]
    return "\n".join(f"· {kw['keyword']}" for kw in top_keywords)


def _parse_triples(raw_text: str) -> list[dict] | None:
    """
    从模型返回的原始文本中解析三元组列表。

    返回：
        list[dict] — 三元组列表；无有效三元组时返回 []；解析失败时返回 None
    """
    stripped = strip_thinking(raw_text)

    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

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

    返回：bool — True 表示校验通过
    """
    required_fields = (
        "subject", "subject_type", "relation_type",
        "relation_detail", "object", "object_type"
    )
    for f in required_fields:
        if not str(triple.get(f, "")).strip():
            logger.debug(f"三元组缺少字段 {f!r}，已丢弃：{triple}")
            return False

    if triple["relation_type"] not in VALID_RELATION_TYPES:
        logger.debug(f"relation_type {triple['relation_type']!r} 不合法，已丢弃")
        return False

    if triple["subject_type"] not in VALID_ENTITY_TYPES:
        logger.debug(f"subject_type {triple['subject_type']!r} 不合法，已丢弃")
        return False

    if triple["object_type"] not in VALID_ENTITY_TYPES:
        logger.debug(f"object_type {triple['object_type']!r} 不合法，已丢弃")
        return False

    if triple["subject"].strip() == triple["object"].strip():
        logger.debug(f"主语和宾语相同，已丢弃：{triple['subject']}")
        return False

    return True


def _extract_triples_for_l1(l1_id: int) -> int | None:
    """
    对单条 L1 摘要执行三元组提取，写入图谱。

    返回：
        int  — 成功写入的三元组数量（0 表示跳过或无有效三元组）
        None — 模型调用失败时返回 None
    """
    l1_row = get_l1_by_id(l1_id)
    if l1_row is None:
        logger.warning(f"找不到 L1 id={l1_id}，跳过")
        return 0

    salience = l1_row["salience"] if l1_row["salience"] is not None else 0.5
    if salience < MIN_SALIENCE:
        logger.debug(
            f"L1 {l1_id} salience={salience:.2f} < {MIN_SALIENCE}，"
            f"跳过图谱提取"
        )
        return 0

    l1_summary = l1_row["summary"]

    entity_candidates = _build_entity_candidates()
    prompt = EXTRACT_TRIPLES_PROMPT.format(
        max_triples       = MAX_TRIPLES_PER_L1,
        entity_candidates = entity_candidates,
        l1_summary        = l1_summary,
    )

    raw_output = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="graph_builder",
    )
    if raw_output is None:
        logger.error(f"L1 {l1_id} 模型调用失败")
        return None

    triples = _parse_triples(raw_output)
    if triples is None:
        logger.error(f"L1 {l1_id} 三元组解析失败，原始输出：{raw_output[:200]}")
        return None

    if not triples:
        logger.debug(f"L1 {l1_id} 无有效三元组")
        return 0

    valid_triples = [t for t in triples if _validate_triple(t)]
    valid_triples = valid_triples[:MAX_TRIPLES_PER_L1]

    if not valid_triples:
        logger.debug(f"L1 {l1_id} 校验后无合法三元组")
        return 0

    written = 0
    for triple in valid_triples:
        try:
            subject_name = _normalize_entity(triple["subject"].strip(), l1_id)
            object_name  = _normalize_entity(triple["object"].strip(),  l1_id)

            source_node_id = get_or_create_node(
                entity_name  = subject_name,
                entity_type  = triple["subject_type"],
                source_l1_id = l1_id,
            )
            target_node_id = get_or_create_node(
                entity_name  = object_name,
                entity_type  = triple["object_type"],
                source_l1_id = l1_id,
            )
            save_edge(
                source_node_id  = source_node_id,
                target_node_id  = target_node_id,
                relation_type   = triple["relation_type"],
                relation_detail = triple["relation_detail"].strip(),
                source_l1_id    = l1_id,
            )
            for triple in valid_triples:
                try:
                    subject_name = _normalize_entity(triple["subject"].strip(), l1_id)
                    object_name  = _normalize_entity(triple["object"].strip(),  l1_id)

                    source_node_id = get_or_create_node(
                        entity_name  = subject_name,
                        entity_type  = triple["subject_type"],
                        source_l1_id = l1_id,
                    )
                    target_node_id = get_or_create_node(
                        entity_name  = object_name,
                        entity_type  = triple["object_type"],
                        source_l1_id = l1_id,
                    )
                    save_edge(
                        source_node_id  = source_node_id,
                        target_node_id  = target_node_id,
                        relation_type   = triple["relation_type"],
                        relation_detail = triple["relation_detail"].strip(),
                        source_l1_id    = l1_id,
                    )

                    # v0.4.0：增量更新内存图谱，无需全量重载
                    _add_edge_to_graph(
                        source_node_id  = source_node_id,
                        source_name     = subject_name,
                        source_type     = triple["subject_type"],
                        target_node_id  = target_node_id,
                        target_name     = object_name,
                        target_type     = triple["object_type"],
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
    同时排除 salience < MIN_SALIENCE 的 L1。
    """
    conn   = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

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

def _build_worker(pending_ids: list[int]) -> None:
    """
    后台批处理线程的主函数。
    逐个处理 pending_ids 中的 L1。
    """
    global _state
    logger.info(f"图谱构建线程启动，共 {len(pending_ids)} 条 L1 待处理")

    for l1_id in pending_ids:
        with _state_lock:
            if _state.stop_requested:
                _state.status      = "stopped"
                _state.finished_at = _now_iso()
                logger.info("图谱构建已停止（用户请求）")
                return

        with _state_lock:
            _state.current_l1_id = l1_id

        result = _extract_triples_for_l1(l1_id)

        with _state_lock:
            _state.done += 1
            if result is None:
                _state.failed += 1
                _state.failed_l1_ids.append(l1_id)
            elif result == 0:
                _state.skipped += 1
            else:
                _state.succeeded += 1

        logger.info(
            f"L1 {l1_id} 处理完成（{_state.done}/{_state.total}）"
        )
        time.sleep(BATCH_SLEEP_INTERVAL)

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
# 对外接口：后台线程版
# =============================================================================

def start_graph_build() -> dict:
    """
    启动图谱构建后台线程，非阻塞返回。
    已有批处理在运行时直接返回当前状态。
    """
    global _state, _build_thread

    with _state_lock:
        if _state.status == "running":
            return _state.to_dict()

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

        _state = GraphBuildState(
            status     = "running",
            total      = len(pending_ids),
            started_at = _now_iso(),
        )

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
    """请求停止图谱构建，等当前 L1 处理完后退出。"""
    global _state

    with _state_lock:
        if _state.status != "running":
            return _state.to_dict()
        _state.stop_requested = True

    logger.info("已发送停止请求，等待当前 L1 处理完毕")

    with _state_lock:
        return _state.to_dict()


def get_graph_status() -> dict:
    """查询当前图谱构建状态，供 FastAPI 路由轮询。"""
    with _state_lock:
        return _state.to_dict()


def get_graph_pending_count() -> int:
    """查询当前待处理 L1 数量，不启动批处理。"""
    try:
        return len(_get_pending_l1_ids())
    except Exception as e:
        logger.error(f"查询待处理 L1 数量失败 — {e}")
        return -1


# =============================================================================
# 图谱内存加载（供 main.py lifespan 调用）
# =============================================================================

def load_graph_to_memory() -> None:
    """
    从 SQLite 读取全部节点和边，构建 NetworkX 内存图。
    在 main.py 的 lifespan startup 阶段调用一次。
    图对象存储在模块级变量 _nx_graph 中。
    """
    global _nx_graph

    try:
        import networkx as nx

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT id, entity_name, entity_type FROM graph_nodes")
        nodes = cursor.fetchall()

        cursor.execute(
            "SELECT source_node_id, target_node_id, relation_type, "
            "relation_detail, source_l1_id FROM graph_edges"
        )
        edges = cursor.fetchall()
        conn.close()

        G = nx.Graph()

        for node in nodes:
            G.add_node(
                node["id"],
                entity_name = node["entity_name"],
                entity_type = node["entity_type"],
            )

        for edge in edges:
            if G.has_edge(edge["source_node_id"], edge["target_node_id"]):
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

        with _nx_graph_lock:
            _nx_graph = G
        logger.info(
            f"NetworkX 图谱加载完成：{G.number_of_nodes()} 个节点，"
            f"{G.number_of_edges()} 条边"
        )

    except ImportError:
        logger.warning("networkx 未安装，图谱检索功能不可用")
        with _nx_graph_lock:
            _nx_graph = None
    except Exception as e:
        logger.error(f"图谱加载失败 — {e}")
        with _nx_graph_lock:
            _nx_graph = None


# 模块级图对象
_nx_graph = None

# 图谱读写锁：保护 _nx_graph 的并发读写
# 使用 RLock 防止同一线程（如 graph_builder 批处理）内重入时死锁
_nx_graph_lock = threading.RLock()


def get_nx_graph():
    """
    获取当前内存中的 NetworkX 图对象。
    供 vector_store.retrieve_graph() 调用。
    加 _nx_graph_lock 保护，防止读取时图正在被替换。
    """
    with _nx_graph_lock:
        return _nx_graph


def _add_edge_to_graph(
    source_node_id: int,
    source_name:    str,
    source_type:    str,
    target_node_id: int,
    target_name:    str,
    target_type:    str,
    relation_type:  str,
    relation_detail: str,
    source_l1_id:   int,
) -> None:
    """
    增量更新内存图谱：将一条新边（及其节点）追加到 _nx_graph，
    无需全量重载整个图，避免 reload_graph() 的高开销。

    调用时机：_extract_triples_for_l1() 成功写入数据库后立即调用。

    节点处理策略：
        · 节点已存在（entity_name 相同）→ 只更新 use_count（+1）
        · 节点不存在 → 新增节点，use_count 初始为 1

    边处理策略：
        · 两节点间边已存在 → 在现有边的 l1_ids 列表中追加 source_l1_id
        · 两节点间边不存在 → 新增边，l1_ids 初始为 [source_l1_id]

    线程安全：所有操作在 _nx_graph_lock 内完成。

    参数：
        source_node_id  — 起点节点的数据库 id（graph_nodes.id）
        source_name     — 起点实体名
        source_type     — 起点实体类型
        target_node_id  — 终点节点的数据库 id
        target_name     — 终点实体名
        target_type     — 终点实体类型
        relation_type   — 关系类型（七类之一）
        relation_detail — 关系描述
        source_l1_id    — 来源 L1 的 id
    """
    global _nx_graph

    with _nx_graph_lock:
        # 图谱未初始化时静默跳过，不阻断主流程
        # 下次 reload_graph() 会把这条边从数据库全量加载进来
        if _nx_graph is None:
            logger.debug(
                f"_add_edge_to_graph: 图谱未初始化，跳过增量更新 "
                f"({source_name} → {target_name})"
            )
            return

        # ── 处理起点节点 ──
        if _nx_graph.has_node(source_node_id):
            # 节点已存在，更新 use_count
            _nx_graph.nodes[source_node_id]["use_count"] = (
                _nx_graph.nodes[source_node_id].get("use_count", 1) + 1
            )
        else:
            _nx_graph.add_node(
                source_node_id,
                entity_name = source_name,
                entity_type = source_type,
                use_count   = 1,
            )

        # ── 处理终点节点 ──
        if _nx_graph.has_node(target_node_id):
            _nx_graph.nodes[target_node_id]["use_count"] = (
                _nx_graph.nodes[target_node_id].get("use_count", 1) + 1
            )
        else:
            _nx_graph.add_node(
                target_node_id,
                entity_name = target_name,
                entity_type = target_type,
                use_count   = 1,
            )

        # ── 处理边 ──
        if _nx_graph.has_edge(source_node_id, target_node_id):
            # 边已存在，追加 l1_id（避免重复）
            existing_l1_ids = _nx_graph[source_node_id][target_node_id].get(
                "l1_ids", []
            )
            if source_l1_id not in existing_l1_ids:
                existing_l1_ids.append(source_l1_id)
                _nx_graph[source_node_id][target_node_id]["l1_ids"] = existing_l1_ids
        else:
            _nx_graph.add_edge(
                source_node_id,
                target_node_id,
                relation_type   = relation_type,
                relation_detail = relation_detail,
                l1_ids          = [source_l1_id],
            )

    logger.debug(
        f"图谱增量更新：{source_name} —[{relation_type}]→ {target_name}，"
        f"L1 id={source_l1_id}"
    )


def reload_graph() -> None:
    """
    重新从 SQLite 加载图谱到内存。
    批处理完成后调用。reload 会重新全量加载，覆盖增量更新的内容，
    但数据库是权威来源，结果是一致的。
    """
    load_graph_to_memory()   # 内部已加锁
    logger.info("图谱已重新加载")


# =============================================================================
# 命令行同步模式
# =============================================================================

def run_build_cli() -> dict:
    """
    同步执行图谱构建，阻塞直到全部完成。命令行调试用。
    """
    pending_ids = _get_pending_l1_ids()

    if not pending_ids:
        print("没有待处理的 L1（所有满足条件的 L1 均已提取图谱）")
        return {"status": "done", "total": 0}

    total      = len(pending_ids)
    succeeded  = 0
    skipped    = 0
    failed     = 0
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

        print(f"[{idx:>4}/{total}] {mark}  L1 {l1_id}")

    print()
    print("=" * 50)
    print(f"  图谱构建完成")
    print(f"  成功 : {succeeded} 条 L1")
    print(f"  跳过 : {skipped} 条 L1")
    print(f"  失败 : {failed} 条 L1")
    if failed_ids:
        print(f"  失败的 L1 id：{failed_ids}")
    print("=" * 50)

    reload_graph()

    return {
        "status":    "done",
        "total":     total,
        "succeeded": succeeded,
        "skipped":   skipped,
        "failed":    failed,
        "failed_ids": failed_ids,
    }