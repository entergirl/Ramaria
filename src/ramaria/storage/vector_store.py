"""
src/ramaria/storage/vector_store.py — 向量索引与语义检索模块

负责维护三层向量索引（L0 / L1 / L2），并对外提供语义检索接口。
是分层 RAG 检索链路的核心模块。

v0.4.0 变更：
    · BM25Index 支持增量更新：新记录先进 _pending 缓冲区，
      累积到 BM25_INCREMENTAL_THRESHOLD 条或定时器触发时合并重建，
      重建期间用旧索引继续服务，完成后原子替换（threading.Lock 保护）
    · 新增 _start_bm25_timer() / _stop_bm25_timer() 供 main.py lifespan 调用
    · get_all_l1() / get_all_l2() 支持传入外部连接复用，减少 BM25 重建时的连接开销
"""

import json
import math
import os
import queue
import threading
from datetime import datetime, timezone

import chromadb
import jieba
import jieba.analyse
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)
from rank_bm25 import BM25Okapi

from ramaria.config import (
    BM25_INCREMENTAL_THRESHOLD,
    BM25_REBUILD_INTERVAL,
    BM25_WEIGHT,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    L0_RETRIEVE_TOP_K,
    L0_WINDOW_SIZE,
    L1_RETRIEVE_TOP_K,
    L2_RETRIEVE_TOP_K,
    MEMORY_DECAY_ENABLE_ACCESS_BOOST,
    MEMORY_DECAY_RECENT_BOOST_DAYS,
    MEMORY_DECAY_RECENT_BOOST_FLOOR,
    MEMORY_DECAY_S_L0,
    MEMORY_DECAY_S_L1,
    MEMORY_DECAY_S_L2,
    RRF_K,
    SALIENCE_DECAY_MULTIPLIER,
    SIMILARITY_THRESHOLD,
)
from ramaria.storage.database import get_messages

from ramaria.logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# 后台访问回写线程配置（与原版完全一致）
# =============================================================================

ACCESS_WRITE_INTERVAL_SECONDS = 30

_access_queue: queue.Queue = queue.Queue(maxsize=0)
_access_worker_thread: threading.Thread | None = None
_access_worker_stop   = threading.Event()


def _start_access_worker():
    """
    启动后台访问回写线程（AccessBoostWorker）。
    调用时机：main.py 的 lifespan startup 阶段。
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


def _db_conn_for_rebuild():
    """
    为 BM25 重建提供数据库连接上下文管理器。
    独立封装避免在 vector_store 内直接 import database 的私有函数 _db_conn。

    使用方式：
        with _db_conn_for_rebuild() as conn:
            rows = get_all_l1(conn=conn)
    """
    import sqlite3
    from contextlib import contextmanager
    from ramaria.config import DB_PATH

    @contextmanager
    def _ctx():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    return _ctx()


# =============================================================================
# BM25 索引管理（v0.4.0 增量更新版）
# =============================================================================

class BM25Index:
    """
    BM25 关键词索引，v0.4.0 起支持增量更新。

    增量更新流程：
        1. index_l1() / index_l2() 写入新记录后，调用 add_pending()
           将新文档追加到 _pending_l1 / _pending_l2 缓冲区
        2. 缓冲区条数达到 BM25_INCREMENTAL_THRESHOLD，或后台定时器触发，
           调用 rebuild() 合并重建索引
        3. 重建在锁外完成计算，只在最终替换时持锁（原子替换），
           重建期间旧索引继续正常服务检索请求

    线程安全：
        · _rw_lock（threading.RLock）保护所有对索引变量的读写
        · 重建计算（tokenize + BM25Okapi）在锁外执行，只有赋值时加锁
        · _pending 缓冲区的追加和清空都在 _rw_lock 内完成
    """

    def __init__(self):
        # 读写锁：RLock 允许同一线程重入，防止 rebuild 内部调用 search 时死锁
        self._rw_lock = threading.RLock()

        # 当前生效的 L1 索引
        self._l1_ids:  list[int] = []
        self._l1_docs: list[str] = []
        self._l1_bm25: BM25Okapi | None = None

        # 当前生效的 L2 索引
        self._l2_ids:  list[int] = []
        self._l2_docs: list[str] = []
        self._l2_bm25: BM25Okapi | None = None

        # 增量缓冲区：存放尚未合并进主索引的新记录
        # 格式：[(id, doc_text), ...]
        self._pending_l1: list[tuple[int, str]] = []
        self._pending_l2: list[tuple[int, str]] = []

        # jieba 自定义词典是否已加载（只加载一次）
        self._jieba_dict_loaded = False

    # -------------------------------------------------------------------------
    # 分词与文档构建（私有）
    # -------------------------------------------------------------------------

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
        将 summary 和 keywords 拼成完整索引文本。
        关键词用空格分隔追加，提升关键词在 BM25 词频统计中的权重。
        """
        if keywords:
            kw_text = keywords.replace(",", " ").replace("，", " ")
            return f"{summary} {kw_text}"
        return summary

    def _load_jieba_custom_dict(self):
        """
        从 keyword_pool 表加载自定义词典到 jieba。
        只在首次重建索引时执行一次，之后标记为已加载跳过。
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

    # -------------------------------------------------------------------------
    # 增量写入接口（供 index_l1 / index_l2 调用）
    # -------------------------------------------------------------------------

    def add_pending(self, layer: str, record_id: int, summary: str,
                    keywords: str | None) -> None:
        """
        将一条新记录追加到增量缓冲区。
        缓冲区达到阈值时自动触发合并重建。

        调用时机：index_l1() / index_l2() 写入 Chroma 后立即调用。

        参数：
            layer     — "l1" 或 "l2"
            record_id — memory_l1 / memory_l2 的主键 id
            summary   — 摘要文本
            keywords  — 关键词字符串，可为 None
        """
        if layer not in ("l1", "l2"):
            logger.warning(f"add_pending: 未知 layer={layer!r}，跳过")
            return

        doc_text = self._build_doc_text(summary, keywords)

        with self._rw_lock:
            if layer == "l1":
                self._pending_l1.append((record_id, doc_text))
                pending_count = len(self._pending_l1)
            else:
                self._pending_l2.append((record_id, doc_text))
                pending_count = len(self._pending_l2)

        logger.debug(
            f"add_pending: layer={layer} id={record_id}，"
            f"缓冲区当前 {pending_count} 条"
        )

        # 缓冲区达到阈值，触发合并重建
        if pending_count >= BM25_INCREMENTAL_THRESHOLD:
            logger.info(
                f"BM25 缓冲区达到阈值（{pending_count} >= "
                f"{BM25_INCREMENTAL_THRESHOLD}），触发增量合并重建"
            )
            self.rebuild(layer)

    # -------------------------------------------------------------------------
    # 全量重建（启动预热 / 定时器 / 阈值触发 均调用此函数）
    # -------------------------------------------------------------------------

    def rebuild(self, layer: str) -> None:
        """
        重建指定层的 BM25 索引。

        v0.4.0 重建策略：
            1. 从数据库读取全量数据（在锁外完成，不阻塞检索）
            2. 合并缓冲区中的待写条目（在锁内取出缓冲，锁外计算）
            3. 对合并后的完整数据集重新 tokenize + 构建 BM25Okapi
            4. 只在最终赋值时加锁（原子替换），旧索引在步骤 1-3 期间继续服务

        参数：
            layer — "l1" 或 "l2"
        """
        if layer not in ("l1", "l2"):
            logger.warning(f"BM25 rebuild: 未知 layer={layer!r}，跳过")
            return

        self._load_jieba_custom_dict()

        # ── 步骤 1：从数据库读取全量数据（锁外，不阻塞检索）──
        # 使用单个连接完成查询后立即关闭，减少连接开销
        try:
            with _db_conn_for_rebuild() as rebuild_conn:
                if layer == "l1":
                    from ramaria.storage.database import get_all_l1
                    rows = get_all_l1(conn=rebuild_conn)
                else:
                    from ramaria.storage.database import get_all_l2
                    rows = get_all_l2(conn=rebuild_conn)
        except Exception as e:
            logger.warning(f"BM25 rebuild({layer}): 读取数据库失败 — {e}")
            return

        # ── 步骤 2：取出当前缓冲区（加锁取出后立即清空，缩短持锁时间）──
        with self._rw_lock:
            if layer == "l1":
                pending_snapshot = list(self._pending_l1)
                self._pending_l1.clear()
            else:
                pending_snapshot = list(self._pending_l2)
                self._pending_l2.clear()

        # ── 步骤 3：在锁外合并全量数据和缓冲区数据，完整重新分词 ──
        # 先用数据库全量数据建立基础
        ids:       list[int] = []
        docs:      list[str] = []
        tokenized: list[list[str]] = []

        # 数据库全量数据中已有的 id 集合，用于去重（缓冲区可能包含已入库的条目）
        db_ids: set[int] = set()

        for row in rows:
            doc_text = self._build_doc_text(
                summary  = row["summary"] or "",
                keywords = row["keywords"] if "keywords" in row.keys() else None,
            )
            tokens = self._tokenize(doc_text)
            if not tokens:
                continue
            ids.append(row["id"])
            docs.append(doc_text)
            tokenized.append(tokens)
            db_ids.add(row["id"])

        # 追加缓冲区中尚未入库（或刚入库）的新条目，跳过已在全量数据中的 id
        for record_id, doc_text in pending_snapshot:
            if record_id in db_ids:
                # 已在数据库全量数据中，无需重复追加
                continue
            tokens = self._tokenize(doc_text)
            if not tokens:
                continue
            ids.append(record_id)
            docs.append(doc_text)
            tokenized.append(tokens)

        if not tokenized:
            logger.debug(f"BM25 rebuild({layer}): 合并后无有效文档，跳过")
            return

        new_bm25 = BM25Okapi(tokenized)

        # ── 步骤 4：原子替换（加锁，只做赋值，极短持锁时间）──
        with self._rw_lock:
            if layer == "l1":
                self._l1_ids  = ids
                self._l1_docs = docs
                self._l1_bm25 = new_bm25
            else:
                self._l2_ids  = ids
                self._l2_docs = docs
                self._l2_bm25 = new_bm25

        logger.debug(
            f"BM25 rebuild({layer}): 完成，"
            f"全量 {len(db_ids)} 条 + 缓冲 {len(pending_snapshot)} 条 "
            f"= 最终 {len(ids)} 条"
        )

    # -------------------------------------------------------------------------
    # 检索接口（与原版接口完全一致，加锁保护读操作）
    # -------------------------------------------------------------------------

    def search(self, query: str, layer: str, top_k: int) -> list[dict]:
        """
        在指定层的 BM25 索引中检索，返回最相关的前 top_k 条结果。

        加 _rw_lock 读锁保护，防止检索时索引正在被替换。

        返回：
            list[dict]，每个元素包含：id, document, bm25_score, rank
            索引为空时返回 []。
        """
        with self._rw_lock:
            if layer == "l1":
                bm25 = self._l1_bm25
                ids  = list(self._l1_ids)
                docs = list(self._l1_docs)
            else:
                bm25 = self._l2_bm25
                ids  = list(self._l2_ids)
                docs = list(self._l2_docs)

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
# BM25 后台定时重建线程
# =============================================================================

_bm25_timer_thread: threading.Thread | None = None
_bm25_timer_stop   = threading.Event()


def _start_bm25_timer() -> None:
    """
    启动 BM25 后台定时重建线程。

    每隔 BM25_REBUILD_INTERVAL 秒检查一次两个层的缓冲区：
        · 缓冲区非空 → 触发合并重建（兜底，保证低写入量场景下索引也能更新）
        · 缓冲区为空 → 跳过（无新数据，不做无效重建）

    调用时机：main.py lifespan startup，在 BM25 预热之后调用。
    """
    global _bm25_timer_thread

    if _bm25_timer_thread and _bm25_timer_thread.is_alive():
        logger.debug("BM25 定时重建线程已在运行，跳过重复启动")
        return

    _bm25_timer_stop.clear()
    _bm25_timer_thread = threading.Thread(
        target=_bm25_timer_loop,
        name="BM25TimerRebuilder",
        daemon=True,
    )
    _bm25_timer_thread.start()
    logger.info(
        f"BM25 定时重建线程已启动，间隔 {BM25_REBUILD_INTERVAL} 秒"
    )


def _stop_bm25_timer() -> None:
    """
    通知 BM25 定时重建线程退出。
    在 lifespan shutdown 阶段调用。
    """
    _bm25_timer_stop.set()
    if _bm25_timer_thread and _bm25_timer_thread.is_alive():
        _bm25_timer_thread.join(timeout=BM25_REBUILD_INTERVAL + 2)
    logger.debug("BM25 定时重建线程已停止")


def _bm25_timer_loop() -> None:
    """BM25 定时重建线程主循环。"""
    logger.debug(
        f"BM25TimerRebuilder 进入循环，间隔 {BM25_REBUILD_INTERVAL} 秒"
    )

    while not _bm25_timer_stop.is_set():
        _bm25_timer_stop.wait(timeout=BM25_REBUILD_INTERVAL)
        if _bm25_timer_stop.is_set():
            break

        # 检查两个层的缓冲区，非空才重建（避免无意义重建）
        for layer in ("l1", "l2"):
            with _bm25_index._rw_lock:
                pending = (
                    _bm25_index._pending_l1
                    if layer == "l1"
                    else _bm25_index._pending_l2
                )
                has_pending = len(pending) > 0

            if has_pending:
                logger.info(
                    f"BM25 定时重建触发：layer={layer}，"
                    f"缓冲区有待合并数据"
                )
                _bm25_index.rebuild(layer)
            else:
                logger.debug(
                    f"BM25 定时重建跳过：layer={layer}，缓冲区为空"
                )

    logger.debug("BM25TimerRebuilder 退出循环")


# =============================================================================
# 内部：Chroma 客户端与 Collection 管理（与原版完全一致）
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

_COLL_TO_LAYER = {
    _COLL_L0: "l0",
    _COLL_L1: "l1",
    _COLL_L2: "l2",
}


# =============================================================================
# 内部：衰减计算（与原版完全一致）
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
    """
    now = datetime.now(timezone.utc)

    try:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        t_days = max((now - created_at).total_seconds() / 86400, 0)
    except (ValueError, TypeError):
        logger.warning(f"无法解析 created_at={created_at_str!r}，衰减跳过（R=1.0）")
        return 1.0

    s_val = salience if salience is not None else 0.5
    s_val = max(0.0, min(1.0, s_val))
    decay_s_adjusted = decay_s * (1.0 + s_val * SALIENCE_DECAY_MULTIPLIER)

    R = math.exp(-t_days / decay_s_adjusted)

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
    """用保留率 R 调整语义检索距离。adjusted = distance / max(R, 0.1)"""
    return semantic_distance / max(R, 0.1)


# =============================================================================
# 内部：L0 滑动窗口切片生成（与原版完全一致）
# =============================================================================

def _make_l0_chunks(messages, session_id: int, window_size: int = L0_WINDOW_SIZE):
    """将一个 session 的消息列表切分为滑动窗口片段，用于 L0 索引。"""
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
# 对外接口：写入索引（v0.4.0：写入后调用 add_pending 进增量缓冲区）
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
    将一条 L1 摘要写入向量索引，同时追加到 BM25 增量缓冲区。

    v0.4.0 变更：写入 Chroma 后调用 _bm25_index.add_pending()，
    不再立即触发全量重建，由增量机制决定重建时机。
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
        logger.info(f"L1 向量索引写入成功，l1_id={l1_id}")

        # v0.4.0：改为增量追加，不再全量重建
        _bm25_index.add_pending("l1", l1_id, summary, keywords)

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
    将一条 L2 聚合摘要写入向量索引，同时追加到 BM25 增量缓冲区。

    v0.4.0 变更：同 index_l1，改为增量追加。
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
        logger.info(f"L2 向量索引写入成功，l2_id={l2_id}")

        # v0.4.0：改为增量追加
        _bm25_index.add_pending("l2", l2_id, summary, keywords)

    except Exception as e:
        logger.warning(f"L2 索引写入失败，l2_id={l2_id} — {e}")


def index_l0_session(session_id: int) -> int | None:
    """
    对一个 session 的所有原始消息做滑动窗口切片，批量写入 L0 向量索引。
    L0 不参与 BM25 索引，此函数与原版完全一致。
    """
    try:
        messages = get_messages(session_id)

        if not messages:
            logger.debug(f"session {session_id} 无消息，跳过 L0 索引")
            return None

        if len(messages) < L0_WINDOW_SIZE:
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
# 对外接口：语义检索（与原版完全一致）
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
    与原版完全一致。
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
    与原版完全一致。
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
            BM25_WEIGHT    / (RRF_K + b_rank)
            + 1.0          / (RRF_K + v_rank)
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
                (r["document"] for r in bm25_results if r["id"] == doc_id), ""
            )
            graph_doc = next(
                (r["document"] for r in (graph_results or [])
                 if r.get(id_field) == doc_id), ""
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
        entry["bm25_score"] = b_score
        entry["rrf_score"]  = rrf_score
        entry["graph_rank"] = graph_rank_map.get(doc_id, None)

        fused.append(entry)

    return fused


# =============================================================================
# 内部：通用检索逻辑（与原版完全一致）
# =============================================================================

def _retrieve(
    collection_name: str,
    query_text: str,
    top_k: int,
    id_field: str | None,
    decay_s: float,
) -> list:
    """
    通用检索函数，含衰减调整。与原版完全一致。
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
# 对外接口：索引管理工具（与原版完全一致）
# =============================================================================

def get_index_stats() -> dict:
    """返回三个 Collection 当前的索引条数。"""
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
    同时触发 BM25 全量重建。
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

    # 重建后强制全量刷新 BM25（绕过增量缓冲区）
    _bm25_index.rebuild("l1")
    _bm25_index.rebuild("l2")

    logger.info(
        f"索引重建完成：L1={counts['l1_count']} 条，"
        f"L2={counts['l2_count']} 条，L0={counts['l0_chunks']} 个切片"
    )
    return counts


# =============================================================================
# 图谱检索通道（与原版完全一致）
# =============================================================================

def retrieve_graph(
    query_text: str,
    top_k: int = L1_RETRIEVE_TOP_K,
) -> list[dict]:
    """
    基于知识图谱的语义检索通道（第三路）。
    与原版完全一致。
    """
    try:
        from ramaria.memory.graph_builder import get_nx_graph
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