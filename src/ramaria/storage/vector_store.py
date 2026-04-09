"""
src/ramaria/storage/vector_store.py — 向量索引与语义检索模块

负责维护三层向量索引（L0 / L1 / L2），并对外提供语义检索接口。
是分层 RAG 检索链路的核心模块。

"""

import os
import json
import math
import queue
import threading
from datetime import datetime, timezone
import jieba
import jieba.analyse
from rank_bm25 import BM25Okapi

import chromadb
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

from ramaria.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    L0_WINDOW_SIZE,
    L0_RETRIEVE_TOP_K,
    L1_RETRIEVE_TOP_K,
    L2_RETRIEVE_TOP_K,
    SIMILARITY_THRESHOLD,
    MEMORY_DECAY_S_L0,
    MEMORY_DECAY_S_L1,
    MEMORY_DECAY_S_L2,
    MEMORY_DECAY_ENABLE_ACCESS_BOOST,
    MEMORY_DECAY_RECENT_BOOST_DAYS,
    MEMORY_DECAY_RECENT_BOOST_FLOOR,
    RRF_K,
    BM25_WEIGHT,
)
from ramaria.storage.database import get_messages

from logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# 后台访问回写线程配置
# =============================================================================

ACCESS_WRITE_INTERVAL_SECONDS = 30

_access_queue: queue.Queue = queue.Queue(maxsize=0)
_access_worker_thread: threading.Thread | None = None
_access_worker_stop   = threading.Event()


def _start_access_worker():
    """
    启动后台访问回写线程（AccessBoostWorker）。

    线程行为：
        每隔 ACCESS_WRITE_INTERVAL_SECONDS 秒，
        从 _access_queue 取出所有待写记录，按层级分组后批量调用
        database.batch_update_last_accessed()。

    调用时机：main.py 的 lifespan startup 阶段。
    MEMORY_DECAY_ENABLE_ACCESS_BOOST=False 时直接返回，不启动线程。
    """
    global _access_worker_thread

    if not MEMORY_DECAY_ENABLE_ACCESS_BOOST:
        logger.debug("访问加成开关关闭，AccessBoostWorker 不启动")
        return

    if _access_worker_thread and _access_worker_thread.is_alive():
        logger.debug("AccessBoostWorker 已在运行，跳过重复启动")
        return

    _access_worker_stop.clear()
    _access_worker_thread = threading.Thread(
        target=_access_worker_loop,
        name="AccessBoostWorker",
        daemon=True,
    )
    _access_worker_thread.start()
    logger.info("AccessBoostWorker 已启动")


def _stop_access_worker():
    """通知后台访问回写线程退出。在 lifespan shutdown 阶段调用。"""
    _access_worker_stop.set()
    if _access_worker_thread and _access_worker_thread.is_alive():
        _access_worker_thread.join(timeout=ACCESS_WRITE_INTERVAL_SECONDS + 2)
    logger.debug("AccessBoostWorker 已停止")


def _access_worker_loop():
    """后台回写线程主循环。"""
    logger.debug(
        f"AccessBoostWorker 进入循环，"
        f"写入间隔 {ACCESS_WRITE_INTERVAL_SECONDS} 秒"
    )

    while not _access_worker_stop.is_set():
        _access_worker_stop.wait(timeout=ACCESS_WRITE_INTERVAL_SECONDS)

        pending: dict[str, list[int]] = {}
        while True:
            try:
                layer, record_id = _access_queue.get_nowait()
                pending.setdefault(layer, []).append(record_id)
            except queue.Empty:
                break

        if not pending:
            continue

        from ramaria.storage.database import batch_update_last_accessed
        for layer, id_list in pending.items():
            unique_ids = list(set(id_list))
            batch_update_last_accessed(layer, unique_ids)

    logger.debug("AccessBoostWorker 退出循环")


def _enqueue_access(layer: str, record_id: int):
    """将一条检索命中记录放入访问队列，供后台线程异步写入。"""
    if not MEMORY_DECAY_ENABLE_ACCESS_BOOST:
        return
    try:
        _access_queue.put_nowait((layer, record_id))
    except queue.Full:
        pass


# =============================================================================
# BM25 索引管理
# =============================================================================

class BM25Index:
    """
    BM25 关键词索引，与 Chroma 向量索引并行运行。

    职责：
        · 维护内存中的 BM25 索引（基于 memory_l1 / memory_l2 的摘要文本）
        · 提供线程安全的检索接口
        · 在新记录写入后自动重建索引（rank_bm25 不支持增量更新）

    线程安全：使用 threading.RLock 保护索引读写。
    """

    def __init__(self):
        self._lock = threading.RLock()

        self._l1_ids:   list[int] = []
        self._l1_docs:  list[str] = []
        self._l1_bm25:  BM25Okapi | None = None

        self._l2_ids:   list[int] = []
        self._l2_docs:  list[str] = []
        self._l2_bm25:  BM25Okapi | None = None

        self._jieba_dict_loaded = False

    def _tokenize(self, text: str) -> list[str]:
        """
        对文本进行中文分词，返回词列表。
        过滤空字符串和单字符，保留长度 >= 2 的词。
        """
        if not text:
            return []
        tokens = jieba.cut(text, cut_all=False)
        return [t.strip() for t in tokens if len(t.strip()) >= 2]

    def _build_doc_text(self, summary: str, keywords: str | None) -> str:
        """
        将 summary 和 keywords 拼成一段完整的索引文本。
        关键词用空格分隔追加，让 jieba 把每个关键词当独立词处理，
        同时提升关键词在 BM25 词频统计中的权重。
        """
        if keywords:
            kw_text = keywords.replace(",", " ").replace("，", " ")
            return f"{summary} {kw_text}"
        return summary

    def _load_jieba_custom_dict(self):
        """
        从 keyword_pool 表加载自定义词典到 jieba。
        只在首次重建索引时执行一次，提升项目专有词汇的分词准确性。
        """
        if self._jieba_dict_loaded:
            return

        try:
            from ramaria.storage.database import get_all_keywords
            keywords = get_all_keywords()
            for kw in keywords:
                if kw and len(kw) >= 2:
                    jieba.add_word(kw, freq=10000, tag='n')
            self._jieba_dict_loaded = True
            logger.debug(f"jieba 自定义词典加载完成，共 {len(keywords)} 个词条")
        except Exception as e:
            logger.warning(f"jieba 自定义词典加载失败，使用默认分词 — {e}")
            self._jieba_dict_loaded = True

    def rebuild(self, layer: str):
        """
        从数据库读取全量数据，重建指定层的 BM25 索引。

        调用时机：
            · 服务启动时（main.py lifespan startup）
            · 新 L1/L2 写入后（index_l1/index_l2 调用结束时）

        参数：
            layer — "l1" 或 "l2"
        """
        if layer not in ("l1", "l2"):
            logger.warning(f"BM25 rebuild: 未知 layer={layer!r}，跳过")
            return

        self._load_jieba_custom_dict()

        try:
            if layer == "l1":
                from ramaria.storage.database import get_all_l1
                rows = get_all_l1()
            else:
                from ramaria.storage.database import get_all_l2
                rows = get_all_l2()
        except Exception as e:
            logger.warning(f"BM25 rebuild({layer}): 读取数据库失败 — {e}")
            return

        if not rows:
            logger.debug(f"BM25 rebuild({layer}): 数据为空，跳过")
            return

        ids       = []
        docs      = []
        tokenized = []

        for row in rows:
            doc_text = self._build_doc_text(
                summary  = row["summary"]  or "",
                keywords = row["keywords"] if "keywords" in row.keys() else None,
            )
            tokens = self._tokenize(doc_text)
            if not tokens:
                continue

            ids.append(row["id"])
            docs.append(doc_text)
            tokenized.append(tokens)

        if not tokenized:
            logger.debug(f"BM25 rebuild({layer}): 分词后无有效文档，跳过")
            return

        new_bm25 = BM25Okapi(tokenized)

        with self._lock:
            if layer == "l1":
                self._l1_ids  = ids
                self._l1_docs = docs
                self._l1_bm25 = new_bm25
            else:
                self._l2_ids  = ids
                self._l2_docs = docs
                self._l2_bm25 = new_bm25

        logger.debug(f"BM25 rebuild({layer}): 完成，共 {len(ids)} 条记录")

    def search(self, query: str, layer: str, top_k: int) -> list[dict]:
        """
        在指定层的 BM25 索引中检索，返回最相关的前 top_k 条结果。

        返回：
            list[dict]，每个元素包含：
                id, document, bm25_score, rank
            索引为空时返回 []。
        """
        with self._lock:
            if layer == "l1":
                bm25 = self._l1_bm25
                ids  = self._l1_ids
                docs = self._l1_docs
            else:
                bm25 = self._l2_bm25
                ids  = self._l2_ids
                docs = self._l2_docs

        if bm25 is None or not ids:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked = [(idx, score) for idx, score in ranked if score > 0]
        ranked = ranked[:top_k]

        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            results.append({
                "id":         ids[idx],
                "document":   docs[idx],
                "bm25_score": float(score),
                "rank":       rank,
            })

        return results


# BM25 全局单例
_bm25_index = BM25Index()


# =============================================================================
# 内部：Chroma 客户端与 Collection 管理
# =============================================================================

_client      = None
_client_lock = threading.Lock()


def _get_client():
    """获取（或初始化）全局 Chroma 客户端。双重检查锁保证线程安全。"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                os.makedirs(CHROMA_DIR, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                logger.info(f"Chroma 客户端已初始化，索引目录：{CHROMA_DIR}")
    return _client


def _get_embedding_function():
    """根据 config.EMBEDDING_MODEL 返回对应的嵌入函数。"""
    if EMBEDDING_MODEL == "default":
        return DefaultEmbeddingFunction()
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _get_collection(name: str):
    """获取（或创建）指定名称的 Chroma Collection，幂等操作。"""
    client = _get_client()
    ef     = _get_embedding_function()
    return client.get_or_create_collection(
        name               = name,
        embedding_function = ef,
    )


# Collection 名称常量
_COLL_L0 = "memory_l0"
_COLL_L1 = "memory_l1"
_COLL_L2 = "memory_l2"

# Collection 名称到层级的映射
_COLL_TO_LAYER = {
    _COLL_L0: "l0",
    _COLL_L1: "l1",
    _COLL_L2: "l2",
}


# =============================================================================
# 内部：衰减计算
# =============================================================================

def _calc_decay_factor(
    created_at_str: str,
    decay_s: float,
    last_accessed_at_str: str | None = None,
    salience: float | None = None,
) -> float:
    """
    根据 Ebbinghaus 遗忘曲线计算记忆保留率 R，并集成 salience 稳定性加成。

    公式：R = e^(-t / S_adjusted)
        t           = 距记忆生成（created_at）的天数
        S_adjusted  = decay_s × (1 + salience × SALIENCE_DECAY_MULTIPLIER)

    salience 加成：
        salience=0.0 → S 不变
        salience=0.5 → S × 1.25（MULTIPLIER=0.5时）
        salience=1.0 → S × 1.5

    last_accessed_at 保底加成（MEMORY_DECAY_ENABLE_ACCESS_BOOST=True 时）：
        若在 RECENT_BOOST_DAYS 天内访问过，R = max(R, RECENT_BOOST_FLOOR)

    返回：
        float — 保留率 R，范围 (0, 1]；解析时间失败时返回 1.0（安全降级）
    """
    from ramaria.config import SALIENCE_DECAY_MULTIPLIER

    now = datetime.now(timezone.utc)

    try:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        t_days = max((now - created_at).total_seconds() / 86400, 0)
    except (ValueError, TypeError):
        logger.warning(f"无法解析 created_at={created_at_str!r}，衰减跳过（R=1.0）")
        return 1.0

    # salience 调整稳定性系数
    s_val = salience if salience is not None else 0.5
    s_val = max(0.0, min(1.0, s_val))
    decay_s_adjusted = decay_s * (1.0 + s_val * SALIENCE_DECAY_MULTIPLIER)

    R = math.exp(-t_days / decay_s_adjusted)

    # 保底加成
    if (
        MEMORY_DECAY_ENABLE_ACCESS_BOOST
        and last_accessed_at_str is not None
    ):
        try:
            last_accessed = datetime.fromisoformat(last_accessed_at_str)
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            days_since_access = (now - last_accessed).total_seconds() / 86400
            if days_since_access <= MEMORY_DECAY_RECENT_BOOST_DAYS:
                R = max(R, MEMORY_DECAY_RECENT_BOOST_FLOOR)
        except (ValueError, TypeError):
            pass

    return R


def _adjust_distance(semantic_distance: float, R: float) -> float:
    """
    用保留率 R 调整语义检索距离。

    公式：adjusted_distance = semantic_distance / max(R, 0.1)
    max(R, 0.1) 防止极老记忆导致距离无限爆炸（最多放大 10 倍）。
    """
    return semantic_distance / max(R, 0.1)


# =============================================================================
# 内部：L0 滑动窗口切片生成
# =============================================================================

def _make_l0_chunks(messages, session_id: int, window_size: int = L0_WINDOW_SIZE):
    """
    将一个 session 的消息列表切分为滑动窗口片段，用于 L0 索引。

    切片规则：步长为 1，每次向前移动一条消息。
    """
    chunks   = []
    msg_list = list(messages)

    if not msg_list:
        return chunks

    for start in range(len(msg_list) - window_size + 1):
        end    = start + window_size - 1
        window = msg_list[start : end + 1]

        lines = []
        for msg in window:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}：{msg['content']}")
        document = "\n".join(lines)

        message_ids = [msg["id"] for msg in window]

        chunks.append({
            "id":       f"l0_{session_id}_{start}",
            "document": document,
            "metadata": {
                "session_id":  session_id,
                "start_idx":   start,
                "end_idx":     end,
                "message_ids": json.dumps(message_ids),
                "created_at":  msg_list[start]["created_at"] if msg_list else "",
            },
        })

    return chunks


# =============================================================================
# 对外接口：写入索引
# =============================================================================

def index_l1(
    l1_id: int,
    summary: str,
    keywords: str | None = None,
    session_id: int | None = None,
    created_at: str | None = None,
    salience: float | None = None,
):
    """
    将一条 L1 摘要写入向量索引。

    参数：
        l1_id      — memory_l1 表的主键 id
        summary    — L1 摘要文本
        keywords   — 关键词字符串，逗号分隔；可为 None
        session_id — 对应的 session id；可为 None
        created_at — 摘要生成时间（ISO 8601）；为 None 时不加日期前缀
        salience   — 情感显著性；写入 metadata 供衰减计算使用
    """
    try:
        collection = _get_collection(_COLL_L1)

        if created_at:
            date_prefix = created_at[:10]
            base_text   = f"[{date_prefix}] {summary}"
        else:
            base_text = summary

        document = f"{base_text} 关键词：{keywords}" if keywords else base_text

        metadata = {"l1_id": l1_id}
        if session_id is not None:
            metadata["session_id"] = session_id
        if keywords:
            metadata["keywords"] = keywords
        if created_at:
            metadata["created_at"] = created_at
        if salience is not None:
            metadata["salience"] = float(salience)

        collection.upsert(
            ids       = [str(l1_id)],
            documents = [document],
            metadatas = [metadata],
        )
        logger.info(f"L1 索引写入成功，l1_id={l1_id}")

        _bm25_index.rebuild("l1")

    except Exception as e:
        logger.warning(f"L1 索引写入失败，l1_id={l1_id} — {e}")


def index_l2(
    l2_id: int,
    summary: str,
    keywords: str | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
):
    """
    将一条 L2 聚合摘要写入向量索引。

    索引文本格式：[YYYY-MM-DD ~ YYYY-MM-DD] <摘要文本> 关键词：<关键词>
    """
    try:
        collection = _get_collection(_COLL_L2)

        if period_start and period_end:
            date_prefix = f"[{period_start[:10]} ~ {period_end[:10]}]"
            base_text   = f"{date_prefix} {summary}"
        elif period_start:
            base_text = f"[{period_start[:10]}] {summary}"
        else:
            base_text = summary

        document = f"{base_text} 关键词：{keywords}" if keywords else base_text

        metadata = {"l2_id": l2_id}
        if keywords:
            metadata["keywords"] = keywords
        if period_start:
            metadata["period_start"] = period_start
            metadata["created_at"]   = period_start
        if period_end:
            metadata["period_end"] = period_end

        collection.upsert(
            ids       = [str(l2_id)],
            documents = [document],
            metadatas = [metadata],
        )
        logger.info(f"L2 索引写入成功，l2_id={l2_id}")

        _bm25_index.rebuild("l2")

    except Exception as e:
        logger.warning(f"L2 索引写入失败，l2_id={l2_id} — {e}")


def index_l0_session(session_id: int) -> int | None:
    """
    对一个 session 的所有原始消息做滑动窗口切片，批量写入 L0 向量索引。

    返回：
        int  — 成功写入的切片数量
        None — 跳过或失败时返回 None
    """
    try:
        messages = get_messages(session_id)

        if not messages:
            logger.debug(f"session {session_id} 无消息，跳过 L0 索引")
            return None

        if len(messages) < L0_WINDOW_SIZE:
            # 消息数不足一个窗口，整体作为一条索引
            logger.debug(
                f"session {session_id} 消息数({len(messages)}) < "
                f"窗口({L0_WINDOW_SIZE})，使用整体索引"
            )
            lines = []
            for msg in messages:
                role_label = "用户" if msg["role"] == "user" else "助手"
                lines.append(f"{role_label}：{msg['content']}")
            document    = "\n".join(lines)
            message_ids = [msg["id"] for msg in messages]
            created_at  = messages[0]["created_at"] if messages else ""

            collection = _get_collection(_COLL_L0)
            collection.upsert(
                ids       = [f"l0_{session_id}_0"],
                documents = [document],
                metadatas = [{
                    "session_id":  session_id,
                    "start_idx":   0,
                    "end_idx":     len(messages) - 1,
                    "message_ids": json.dumps(message_ids),
                    "created_at":  created_at,
                }],
            )
            return 1

        chunks = _make_l0_chunks(messages, session_id)

        if not chunks:
            return None

        collection = _get_collection(_COLL_L0)
        collection.upsert(
            ids       = [c["id"]       for c in chunks],
            documents = [c["document"] for c in chunks],
            metadatas = [c["metadata"] for c in chunks],
        )

        logger.info(
            f"L0 索引写入成功，session_id={session_id}，"
            f"共 {len(chunks)} 个切片"
        )
        return len(chunks)

    except Exception as e:
        logger.warning(f"L0 索引写入失败，session_id={session_id} — {e}")
        return None


# =============================================================================
# 对外接口：语义检索
# =============================================================================

def retrieve_l2(query_text: str, top_k: int = L2_RETRIEVE_TOP_K) -> list:
    """在 L2 索引中做语义检索，返回最相关的聚合摘要列表。"""
    return _retrieve(
        _COLL_L2, query_text, top_k,
        id_field="l2_id", decay_s=MEMORY_DECAY_S_L2
    )


def retrieve_l1(query_text: str, top_k: int = L1_RETRIEVE_TOP_K) -> list:
    """在 L1 索引中做语义检索，返回最相关的单次对话摘要列表。"""
    return _retrieve(
        _COLL_L1, query_text, top_k,
        id_field="l1_id", decay_s=MEMORY_DECAY_S_L1
    )


def retrieve_l0(query_text: str, top_k: int = L0_RETRIEVE_TOP_K) -> list:
    """在 L0 索引中做语义检索，返回最相关的原始消息切片列表。"""
    return _retrieve(
        _COLL_L0, query_text, top_k,
        id_field=None, decay_s=MEMORY_DECAY_S_L0
    )


def retrieve_combined(query_text: str) -> dict:
    """
    同时在 L1 和 L2 索引中检索，融合向量检索和 BM25 关键词检索的结果。

    融合策略：RRF（倒数排名融合）
        rrf_score = BM25_WEIGHT / (RRF_K + bm25_rank)
                  + 1.0         / (RRF_K + vector_rank)

    返回：
        dict — {"l2": list[dict], "l1": list[dict]}
               每条结果新增 bm25_score 和 rrf_score 字段
    """
    l1_candidate_k = L1_RETRIEVE_TOP_K * 2
    l2_candidate_k = L2_RETRIEVE_TOP_K * 2

    vector_l1 = _retrieve(
        _COLL_L1, query_text, l1_candidate_k,
        id_field="l1_id", decay_s=MEMORY_DECAY_S_L1,
    )
    bm25_l1 = _bm25_index.search(query_text, layer="l1", top_k=l1_candidate_k)

    vector_l2 = _retrieve(
        _COLL_L2, query_text, l2_candidate_k,
        id_field="l2_id", decay_s=MEMORY_DECAY_S_L2,
    )
    bm25_l2 = _bm25_index.search(query_text, layer="l2", top_k=l2_candidate_k)

    fused_l1 = _rrf_fuse(vector_l1, bm25_l1, "l1_id", L1_RETRIEVE_TOP_K)
    fused_l2 = _rrf_fuse(vector_l2, bm25_l2, "l2_id", L2_RETRIEVE_TOP_K)

    logger.debug(
        f"retrieve_combined: "
        f"L1 向量={len(vector_l1)} BM25={len(bm25_l1)} 融合后={len(fused_l1)} | "
        f"L2 向量={len(vector_l2)} BM25={len(bm25_l2)} 融合后={len(fused_l2)}"
    )

    return {"l2": fused_l2, "l1": fused_l1}


def _rrf_fuse(
    vector_results: list[dict],
    bm25_results:   list[dict],
    id_field:       str,
    top_k:          int,
    graph_results:  list[dict] | None = None,
) -> list[dict]:
    """
    对向量、BM25、图谱三路检索结果执行 RRF 融合。

    RRF 公式（三路版）：
        rrf_score = BM25_WEIGHT   / (RRF_K + rank_bm25)
                  + 1.0           / (RRF_K + rank_vector)
                  + GRAPH_WEIGHT  / (RRF_K + rank_graph)

    graph_results 为 None 时退化为两路融合。
    """
    from ramaria.config import GRAPH_WEIGHT

    penalty_rank = top_k * 2 + 1

    bm25_rank_map: dict[int, tuple[int, float]] = {}
    for item in bm25_results:
        bm25_rank_map[item["id"]] = (item["rank"], item["bm25_score"])

    vector_rank_map: dict[int, tuple[int, dict]] = {}
    for rank, item in enumerate(vector_results, start=1):
        record_id = item.get(id_field)
        if record_id is not None:
            vector_rank_map[record_id] = (rank, item)

    graph_rank_map: dict[int, int] = {}
    if graph_results:
        for item in graph_results:
            record_id = item.get(id_field)
            if record_id is not None:
                graph_rank_map[record_id] = item["graph_rank"]

    all_ids: set[int] = (
        set(vector_rank_map.keys())
        | set(bm25_rank_map.keys())
        | set(graph_rank_map.keys())
    )

    scored: list[tuple[float, int]] = []

    for doc_id in all_ids:
        v_rank          = vector_rank_map[doc_id][0] if doc_id in vector_rank_map else penalty_rank
        b_rank, b_score = bm25_rank_map.get(doc_id, (penalty_rank, 0.0))
        g_rank          = graph_rank_map.get(doc_id, penalty_rank)

        rrf_score = (
            BM25_WEIGHT   / (RRF_K + b_rank)
            + 1.0         / (RRF_K + v_rank)
            + GRAPH_WEIGHT / (RRF_K + g_rank)
        )
        scored.append((rrf_score, doc_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:top_k]

    fused = []
    for rrf_score, doc_id in scored:
        if doc_id in vector_rank_map:
            _, result_dict = vector_rank_map[doc_id]
            entry = dict(result_dict)
        else:
            bm25_doc = next(
                (r["document"] for r in bm25_results if r["id"] == doc_id),
                ""
            )
            graph_doc = next(
                (r["document"] for r in (graph_results or [])
                 if r.get(id_field) == doc_id),
                ""
            )
            entry = {
                id_field:            doc_id,
                "document":          bm25_doc or graph_doc,
                "distance":          None,
                "adjusted_distance": None,
                "decay_r":           None,
                "metadata":          {},
            }

        b_rank, b_score = bm25_rank_map.get(doc_id, (penalty_rank, 0.0))
        entry["bm25_score"]  = b_score
        entry["rrf_score"]   = rrf_score
        entry["graph_rank"]  = graph_rank_map.get(doc_id, None)

        fused.append(entry)

    return fused


# =============================================================================
# 内部：通用检索逻辑（含衰减）
# =============================================================================

def _retrieve(
    collection_name: str,
    query_text: str,
    top_k: int,
    id_field: str | None,
    decay_s: float,
) -> list:
    """
    通用检索函数，被 retrieve_l0/l1/l2 调用。

    在语义检索基础上叠加衰减调整：
        1. Chroma 返回原始语义距离
        2. 从 metadata["created_at"] 取时间，计算保留率 R
        3. 若开启访问加成，读取 last_accessed_at 做保底
        4. adjusted_distance = distance / max(R, 0.1)
        5. 按 adjusted_distance 过滤（阈值 SIMILARITY_THRESHOLD）并排序
    """
    try:
        collection = _get_collection(collection_name)

        if collection.count() == 0:
            return []

        candidate_k = min(top_k * 3, collection.count())

        raw = collection.query(
            query_texts = [query_text],
            n_results   = candidate_k,
        )

        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]

        results    = []
        layer      = _COLL_TO_LAYER.get(collection_name)
        access_ids = []

        for doc_id, doc, dist, meta in zip(ids, documents, distances, metadatas):

            created_at_str       = meta.get("created_at", "")
            last_accessed_at_str = None

            if MEMORY_DECAY_ENABLE_ACCESS_BOOST and id_field and id_field in meta:
                record_id = meta[id_field]
                try:
                    from ramaria.storage.database import get_last_accessed_at
                    last_accessed_at_str = get_last_accessed_at(layer, record_id)
                except Exception:
                    pass

            meta_salience = meta.get("salience", None)

            R                 = _calc_decay_factor(
                created_at_str,
                decay_s,
                last_accessed_at_str,
                salience=meta_salience,
            )
            adjusted_distance = _adjust_distance(dist, R)

            if adjusted_distance > SIMILARITY_THRESHOLD:
                continue

            result = {
                "document":          doc,
                "distance":          dist,
                "adjusted_distance": adjusted_distance,
                "decay_r":           round(R, 4),
                "metadata":          meta,
            }

            if id_field and id_field in meta:
                record_id = meta[id_field]
                result[id_field] = record_id
                if layer in ("l1", "l2"):
                    access_ids.append((layer, record_id))

            results.append(result)

        results.sort(key=lambda x: x["adjusted_distance"])
        results = results[:top_k]

        for layer_key, rec_id in access_ids[:top_k]:
            _enqueue_access(layer_key, rec_id)

        return results

    except Exception as e:
        err_str = str(e).lower()
        if "does not exist" in err_str or "invalidcollection" in err_str:
            return []
        logger.error(
            f"检索失败（非预期异常），collection={collection_name} — {e}"
        )
        return []


# =============================================================================
# 对外接口：索引管理工具
# =============================================================================

def get_index_stats() -> dict:
    """
    返回三个 Collection 当前的索引条数。

    返回：dict — {"l0": int, "l1": int, "l2": int}
    """
    stats  = {"l0": 0, "l1": 0, "l2": 0}
    client = _get_client()

    for key, name in [("l0", _COLL_L0), ("l1", _COLL_L1), ("l2", _COLL_L2)]:
        try:
            coll       = client.get_collection(name)
            stats[key] = coll.count()
        except Exception:
            pass

    return stats


def rebuild_all_indexes() -> dict:
    """
    重建全部三层向量索引（切换嵌入模型时使用）。

    ⚠️  此操作不可逆，旧索引会被彻底删除，重建期间检索功能暂时不可用。

    返回：
        dict — {"l0_chunks": int, "l1_count": int, "l2_count": int}
    """
    from ramaria.storage.database import (
        get_all_l1, get_all_l2, get_all_session_ids,
    )

    logger.info("开始重建全部向量索引…")
    client = _get_client()

    for name in [_COLL_L0, _COLL_L1, _COLL_L2]:
        try:
            client.delete_collection(name)
            logger.info(f"已删除旧 Collection：{name}")
        except Exception:
            pass

    counts = {"l0_chunks": 0, "l1_count": 0, "l2_count": 0}

    l1_rows = get_all_l1()
    for row in l1_rows:
        index_l1(
            l1_id      = row["id"],
            summary    = row["summary"],
            keywords   = row["keywords"],
            session_id = row["session_id"],
            created_at = row["created_at"] if "created_at" in row.keys() else None,
        )
        counts["l1_count"] += 1

    l2_rows = get_all_l2()
    for row in l2_rows:
        index_l2(
            l2_id        = row["id"],
            summary      = row["summary"],
            keywords     = row["keywords"],
            period_start = row["period_start"],
            period_end   = row["period_end"],
        )
        counts["l2_count"] += 1

    session_ids = get_all_session_ids()
    for sid in session_ids:
        n = index_l0_session(sid)
        if n:
            counts["l0_chunks"] += n

    logger.info(
        f"索引重建完成：L1={counts['l1_count']} 条，"
        f"L2={counts['l2_count']} 条，L0={counts['l0_chunks']} 个切片"
    )
    return counts


# =============================================================================
# 图谱检索通道
# =============================================================================

def retrieve_graph(
    query_text: str,
    top_k: int = L1_RETRIEVE_TOP_K,
) -> list[dict]:
    """
    基于知识图谱的语义检索通道（第三路，配合 retrieve_combined 使用）。

    流程：
        1. 从查询文本提取候选实体名（jieba 分词 + keyword_pool 匹配）
        2. 在 NetworkX 图里找对应节点
        3. BFS 广度优先遍历（最多两跳）
        4. 按跳数排名，返回关联的 L1 id 列表

    返回：
        list[dict]，每条包含 l1_id, document, graph_rank, hops 等字段
        图谱为空或无命中时返回 []
    """
    from ramaria.adapters.mcp import get_nx_graph  # 懒加载，避免循环依赖

    # 注意：graph_builder 迁移后 get_nx_graph 的 import 路径会更新
    # 此处暂时使用占位，第五批处理 graph_builder 时一并修正
    try:
        import sys
        # 兼容迁移前后两种路径
        try:
            from ramaria.memory.graph_builder import get_nx_graph
        except ImportError:
            from graph_builder import get_nx_graph
    except ImportError:
        logger.warning("graph_builder 未找到，图谱检索跳过")
        return []

    G = get_nx_graph()
    if G is None or G.number_of_nodes() == 0:
        logger.debug("图谱为空或未加载，跳过图谱检索")
        return []

    from ramaria.storage.database import get_all_canonical_keywords

    canonical_keywords = {kw["keyword"] for kw in get_all_canonical_keywords()}
    if not canonical_keywords:
        logger.debug("keyword_pool 为空，图谱检索无法提取查询实体")
        return []

    tokens = list(jieba.cut(query_text, cut_all=False))
    query_entities = [
        t.strip() for t in tokens
        if t.strip() in canonical_keywords and len(t.strip()) >= 2
    ]

    if not query_entities:
        logger.debug(f"查询文本中未找到已知实体，跳过图谱检索：{query_text[:50]}")
        return []

    logger.debug(f"图谱检索实体：{query_entities}")

    entity_to_node = {
        data["entity_name"]: node_id
        for node_id, data in G.nodes(data=True)
    }

    start_node_ids = [
        entity_to_node[e] for e in query_entities
        if e in entity_to_node
    ]

    if not start_node_ids:
        logger.debug("查询实体在图谱中无对应节点")
        return []

    all_l1_results: dict[int, int] = {}

    for start_node_id in start_node_ids:
        import networkx as nx
        try:
            path_lengths = nx.single_source_shortest_path_length(
                G, start_node_id, cutoff=2
            )
        except Exception:
            continue

        for neighbor_node_id, hops in path_lengths.items():
            if hops == 0:
                continue

            if G.has_edge(start_node_id, neighbor_node_id):
                edge_data = G[start_node_id][neighbor_node_id]
            else:
                edge_data = {}

            l1_ids_on_edge = edge_data.get("l1_ids", [])
            for l1_id in l1_ids_on_edge:
                if l1_id not in all_l1_results:
                    all_l1_results[l1_id] = hops
                else:
                    all_l1_results[l1_id] = min(all_l1_results[l1_id], hops)

    if not all_l1_results:
        logger.debug("图谱遍历无命中结果")
        return []

    sorted_l1s = sorted(all_l1_results.items(), key=lambda x: x[1])
    sorted_l1s = sorted_l1s[:top_k]

    results = []
    for rank, (l1_id, hops) in enumerate(sorted_l1s, start=1):
        results.append({
            "l1_id":             l1_id,
            "document":          "",
            "graph_rank":        rank,
            "hops":              hops,
            "distance":          None,
            "adjusted_distance": None,
            "decay_r":           None,
            "metadata":          {},
        })

    logger.debug(f"图谱检索命中 {len(results)} 条 L1")
    return results