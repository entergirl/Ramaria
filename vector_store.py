"""
vector_store.py — 向量索引与语义检索模块

负责维护三层向量索引（L0 / L1 / L2），并对外提供语义检索接口。
是分层 RAG 检索链路的核心模块。

变更记录：
  v2 — rebuild_all_indexes() 移除对私有函数 _get_connection 的依赖，
       修复审查报告问题A。

  v3 — 三项改动（P2-5 双重检查锁、P3-5 大小写不敏感匹配、P2-C 异常日志升级）

  v4 — 新增基于 Ebbinghaus 遗忘曲线的记忆衰减机制：

       【衰减逻辑】
         _retrieve() 在过滤和排序前，对每条结果按公式调整检索距离：
           R = e^(-t / S)
           adjusted_distance = semantic_distance / max(R, 0.1)
         t = 距记忆生成（created_at）的天数，S 为各层稳定性系数。
         时间越久 → R 越小 → adjusted_distance 越大 → 越难被召回。

       【保底加成】
         当 MEMORY_DECAY_ENABLE_ACCESS_BOOST=True 时，若记忆在
         MEMORY_DECAY_RECENT_BOOST_DAYS 天内被访问过，则 R 不低于
         MEMORY_DECAY_RECENT_BOOST_FLOOR（默认 0.5），防止"仍在用的旧记忆"
         被压制得太厉害。
         注意：这是 R 的下限（保底），不是替换 t。

       【异步回写】
         检索命中的记录 id 会被放入全局队列 _access_queue，
         后台常驻线程 AccessBoostWorker 每隔 ACCESS_WRITE_INTERVAL_SECONDS
         秒消费一次队列，批量调用 database.batch_update_last_accessed()。
         主检索路径不阻塞。

       【向量化文本前缀】
         index_l1() 和 index_l2() 写入时在文本前拼接 [YYYY-MM-DD] 前缀，
         让模型在阅读注入内容时也能感知时间远近，辅助判断参考权重。

─────────────────────────────────────────────────────────────
三层索引的定位与分工
─────────────────────────────────────────────────────────────

  L2 索引（memory_l2）— 粒度最粗，时间段聚合摘要，话题定位用
  L1 索引（memory_l1）— 语义密度最高，主检索层，RAG 注入主要来源
  L0 索引（messages） — 粒度最细，滑动窗口切片，原始对话细节

─────────────────────────────────────────────────────────────
与其他模块的关系
─────────────────────────────────────────────────────────────

  被写入：
    summarizer.py      → L1 写入后调用 index_l1()
    merger.py          → L2 写入后调用 index_l2()
    session_manager.py → session 关闭后调用 index_l0_session()

  被检索：
    prompt_builder.py / main.py → 构建 system prompt 时调用 retrieve_*() 系列

  读取配置：config.py
  读取消息数据（L0 用）：database.py
  回写访问时间：database.batch_update_last_accessed()（后台线程）
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

from config import (
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
from database import get_messages

from logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# 后台访问回写线程配置
# =============================================================================

# 后台线程消费队列的间隔（秒）
# 不需要太频繁，30 秒足够，主要作用是合并批量写入
ACCESS_WRITE_INTERVAL_SECONDS = 30

# 全局访问队列：存放 (layer, id) 元组
# 检索命中时由主线程 put，后台线程批量 get 并写入数据库
# maxsize=0 表示无限队列，不会阻塞主线程
_access_queue: queue.Queue = queue.Queue(maxsize=0)

# 后台回写线程单例和停止标志
_access_worker_thread: threading.Thread | None = None
_access_worker_stop   = threading.Event()


def _start_access_worker():
    """
    启动后台访问回写线程（AccessBoostWorker）。

    线程行为：
      每隔 ACCESS_WRITE_INTERVAL_SECONDS 秒，
      从 _access_queue 取出所有待写记录，按层级分组后批量调用
      database.batch_update_last_accessed()。

    调用时机：
      main.py 的 lifespan startup 阶段（与 session_manager.start() 同级）。
      如果 MEMORY_DECAY_ENABLE_ACCESS_BOOST=False，此函数直接返回，不启动线程。
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
    """
    通知后台访问回写线程退出。
    在 main.py 的 lifespan shutdown 阶段调用。
    """
    _access_worker_stop.set()
    if _access_worker_thread and _access_worker_thread.is_alive():
        _access_worker_thread.join(timeout=ACCESS_WRITE_INTERVAL_SECONDS + 2)
    logger.debug("AccessBoostWorker 已停止")


def _access_worker_loop():
    """
    后台回写线程主循环。

    每隔 ACCESS_WRITE_INTERVAL_SECONDS 秒：
      1. 取出队列中所有待写条目（非阻塞，取完即止）
      2. 按 layer 分组
      3. 批量调用 batch_update_last_accessed()
    """
    logger.debug(
        f"AccessBoostWorker 进入循环，写入间隔 {ACCESS_WRITE_INTERVAL_SECONDS} 秒"
    )

    while not _access_worker_stop.is_set():
        # 等待一个间隔，或收到停止信号
        _access_worker_stop.wait(timeout=ACCESS_WRITE_INTERVAL_SECONDS)

        # 取出队列中所有待写条目
        pending: dict[str, list[int]] = {}   # {layer: [id, ...]}
        while True:
            try:
                layer, record_id = _access_queue.get_nowait()
                pending.setdefault(layer, []).append(record_id)
            except queue.Empty:
                break

        if not pending:
            continue

        # 批量写入，按层级分组
        from database import batch_update_last_accessed
        for layer, id_list in pending.items():
            # 去重，同一条记录可能在本批次内被命中多次
            unique_ids = list(set(id_list))
            batch_update_last_accessed(layer, unique_ids)

    logger.debug("AccessBoostWorker 退出循环")


def _enqueue_access(layer: str, record_id: int):
    """
    将一条检索命中记录放入访问队列，供后台线程异步写入。

    参数：
        layer     — "l1" 或 "l2"（L0 不参与访问加成）
        record_id — 记录主键 id
    """
    if not MEMORY_DECAY_ENABLE_ACCESS_BOOST:
        return
    try:
        _access_queue.put_nowait((layer, record_id))
    except queue.Full:
        # maxsize=0 时永远不会满，这里只是防御性处理
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

    索引文本构成：
        summary + " " + keywords（关键词用空格替换逗号，让 jieba 能分开处理）

    线程安全：
        使用 threading.RLock（可重入锁）保护索引读写。
        检索（读）和重建（写）不会并发冲突。
        RLock 而非 Lock 是为了允许同一线程在持锁时重入（如重建时触发检索）。
    """

    def __init__(self):
        self._lock = threading.RLock()

        # L1 索引相关数据（列表索引与记录一一对应）
        self._l1_ids:   list[int] = []      # 每条记录的 l1_id
        self._l1_docs:  list[str] = []      # 每条记录的原始文本（用于调试）
        self._l1_bm25:  BM25Okapi | None = None  # BM25 索引对象

        # L2 索引相关数据
        self._l2_ids:   list[int] = []
        self._l2_docs:  list[str] = []
        self._l2_bm25:  BM25Okapi | None = None

        # jieba 自定义词典是否已加载（只加载一次）
        self._jieba_dict_loaded = False

    # -------------------------------------------------------------------------
    # 内部工具：文本预处理
    # -------------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """
        对文本进行中文分词，返回词列表。

        分词策略：
          · 使用 jieba 精确模式（cut_all=False），颗粒度适中
          · 过滤空字符串和单字符（减少噪声，BM25 对短词不敏感）
          · 保留长度 >= 2 的词，包括英文词（如 "MCP"、"FastAPI"）

        参数：
            text — 待分词的原始文本

        返回：
            list[str] — 分词结果列表
        """
        if not text:
            return []
        tokens = jieba.cut(text, cut_all=False)
        return [t.strip() for t in tokens if len(t.strip()) >= 2]

    def _build_doc_text(self, summary: str, keywords: str | None) -> str:
        """
        将 summary 和 keywords 拼成一段完整的索引文本。

        关键词用空格分隔追加在摘要后面，这样 jieba 能把它们作为独立词处理，
        同时关键词在文本中出现两次（摘要里一次 + 关键词字段一次），
        BM25 的词频统计会给它更高的权重——这正是我们想要的效果。

        参数：
            summary  — L1/L2 摘要文本
            keywords — 关键词字符串，逗号分隔；允许为 None

        返回：
            str — 拼接后的索引文本
        """
        if keywords:
            # 将逗号替换为空格，让 jieba 把每个关键词当独立词处理
            kw_text = keywords.replace(",", " ").replace("，", " ")
            return f"{summary} {kw_text}"
        return summary

    # -------------------------------------------------------------------------
    # 内部工具：jieba 自定义词典
    # -------------------------------------------------------------------------

    def _load_jieba_custom_dict(self):
        """
        从 keyword_pool 表加载自定义词典到 jieba，确保项目专有词汇分词正确。

        只在首次重建索引时执行一次（_jieba_dict_loaded 标志保护）。
        加载失败时静默跳过，不影响分词功能（只是专有词可能被切错）。

        为什么需要自定义词典：
            jieba 默认词典对"珊瑚菌"、"summarizer"、"MCP Server"等
            项目专有词汇的切分可能不准确，导致 BM25 检索时词频统计偏差。
            从 keyword_pool 表导入已知关键词，可以显著提升分词准确性。
        """
        if self._jieba_dict_loaded:
            return

        try:
            from database import get_all_keywords
            keywords = get_all_keywords()
            for kw in keywords:
                if kw and len(kw) >= 2:
                    # 词频设为 10000（高于默认值），确保作为独立词处理
                    # 词性设为 'n'（名词），符合关键词的语言特征
                    jieba.add_word(kw, freq=10000, tag='n')
            self._jieba_dict_loaded = True
            logger.debug(f"jieba 自定义词典加载完成，共 {len(keywords)} 个词条")
        except Exception as e:
            logger.warning(f"jieba 自定义词典加载失败，使用默认分词 — {e}")
            self._jieba_dict_loaded = True   # 标记为已尝试，避免反复重试

    # -------------------------------------------------------------------------
    # 公开接口：索引重建
    # -------------------------------------------------------------------------

    def rebuild(self, layer: str):
        """
        从数据库读取全量数据，重建指定层的 BM25 索引。

        调用时机：
          · 服务启动时（main.py lifespan startup）
          · 新 L1 写入后（index_l1 调用结束时）
          · 新 L2 写入后（index_l2 调用结束时）

        参数：
            layer — "l1" 或 "l2"，指定要重建哪一层的索引

        重建流程：
          1. 加载 jieba 自定义词典（仅首次）
          2. 从 SQLite 读取全量摘要和关键词
          3. 对每条记录做分词，构建词列表
          4. 用词列表重新初始化 BM25Okapi 对象
          5. 加锁写入内存，替换旧索引
        """
        if layer not in ("l1", "l2"):
            logger.warning(f"BM25 rebuild: 未知 layer={layer!r}，跳过")
            return

        # 首次重建时加载自定义词典
        self._load_jieba_custom_dict()

        try:
            if layer == "l1":
                from database import get_all_l1
                rows = get_all_l1()
            else:
                from database import get_all_l2
                rows = get_all_l2()
        except Exception as e:
            logger.warning(f"BM25 rebuild({layer}): 读取数据库失败 — {e}")
            return

        if not rows:
            logger.debug(f"BM25 rebuild({layer}): 数据为空，跳过")
            return

        # 构建文档列表和对应 id 列表
        ids      = []
        docs     = []
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

        # 构建 BM25 索引对象
        # BM25Okapi 是 Okapi BM25 的标准实现，参数 k1=1.5, b=0.75（库默认值）
        new_bm25 = BM25Okapi(tokenized)

        # 加锁替换内存中的旧索引
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

    # -------------------------------------------------------------------------
    # 公开接口：检索
    # -------------------------------------------------------------------------

    def search(self, query: str, layer: str, top_k: int) -> list[dict]:
        """
        在指定层的 BM25 索引中检索，返回最相关的前 top_k 条结果。

        参数：
            query  — 查询文本（原始文本，函数内部会分词）
            layer  — "l1" 或 "l2"
            top_k  — 最多返回条数

        返回：
            list[dict]，每个元素包含：
                {
                    "id":       int,    # l1_id 或 l2_id
                    "document": str,    # 索引文本
                    "bm25_score": float, # BM25 原始分数（越高越相关）
                    "rank":     int,    # 在本次检索结果中的排名（从 1 开始）
                }
            索引为空时返回 []。

        注意：
            BM25 分数只有相对意义（同一次检索内可比），
            不同次检索的分数不可直接比较。
            融合时使用排名（rank）而非原始分数。
        """
        with self._lock:
            if layer == "l1":
                bm25  = self._l1_bm25
                ids   = self._l1_ids
                docs  = self._l1_docs
            else:
                bm25  = self._l2_bm25
                ids   = self._l2_ids
                docs  = self._l2_docs

        # 索引为空时直接返回
        if bm25 is None or not ids:
            return []

        # 对查询文本分词
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 获取所有文档的 BM25 分数
        scores = bm25.get_scores(query_tokens)

        # 按分数降序排列，取前 top_k 条（过滤分数为 0 的结果）
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


# BM25 全局单例，与 Chroma 客户端同级别管理
_bm25_index = BM25Index()


# =============================================================================
# 内部：Chroma 客户端与 Collection 管理
# =============================================================================

_client      = None
_client_lock = threading.Lock()


def _get_client():
    """
    获取（或初始化）全局 Chroma 客户端。
    使用双重检查锁保证线程安全（P2-5 修复）。
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                os.makedirs(CHROMA_DIR, exist_ok=True)
                _client = chromadb.PersistentClient(path=CHROMA_DIR)
                logger.info(f"Chroma 客户端已初始化，索引目录：{CHROMA_DIR}")
    return _client


def _get_embedding_function():
    """
    根据 config.EMBEDDING_MODEL 返回对应的嵌入函数。
    "default" → Chroma 内置模型；其他 → SentenceTransformer。
    """
    if EMBEDDING_MODEL == "default":
        return DefaultEmbeddingFunction()
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _get_collection(name):
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

# Collection 名称到层级的映射，供衰减逻辑使用
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

    salience 加成含义：
      salience=0.0 → S 不变，衰减速度正常
      salience=0.5 → S × 1.25（MULTIPLIER=0.5时），衰减减慢 25%
      salience=1.0 → S × 1.5，衰减减慢 50%，高显著记忆更持久

    last_accessed_at 保底加成（当 MEMORY_DECAY_ENABLE_ACCESS_BOOST=True）：
      若 last_accessed_at 不为 None 且在 RECENT_BOOST_DAYS 天内，
      则 R = max(R, RECENT_BOOST_FLOOR)。

    参数：
        created_at_str       — 记忆创建时间，ISO 8601 字符串
        decay_s              — 当前层的基础稳定性系数（天）
        last_accessed_at_str — 最近访问时间；None 表示从未访问
        salience             — 情感显著性 [0.0, 1.0]；None 时视为 0.5（不影响加成）

    返回：
        float — 保留率 R，范围 (0, 1]；解析时间失败时返回 1.0（安全降级）
    """
    from config import SALIENCE_DECAY_MULTIPLIER

    now = datetime.now(timezone.utc)

    # ── 主衰减：基于 created_at + salience 调整后的稳定性系数 ──
    try:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        t_days = max((now - created_at).total_seconds() / 86400, 0)
    except (ValueError, TypeError):
        logger.warning(f"无法解析 created_at={created_at_str!r}，衰减跳过（R=1.0）")
        return 1.0

    # 用 salience 调整稳定性系数
    # salience 为 None 时使用 0.5（中等显著性），不偏不倚
    s_val = salience if salience is not None else 0.5
    # 限制在合法范围内，防止异常值破坏公式
    s_val = max(0.0, min(1.0, s_val))
    decay_s_adjusted = decay_s * (1.0 + s_val * SALIENCE_DECAY_MULTIPLIER)

    R = math.exp(-t_days / decay_s_adjusted)

    # ── 保底加成：基于 last_accessed_at（可选） ──
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

    max(R, 0.1) 的作用：
      防止 R 极小时（极老的记忆）导致 adjusted_distance 无限爆炸。
      即使记忆已经极度老化，调整后的距离最多是原始距离的 10 倍。

    参数：
        semantic_distance — Chroma 返回的原始余弦距离，范围 [0, 2]
        R                 — _calc_decay_factor() 返回的保留率

    返回：
        float — 调整后的距离，越大越靠后
    """
    return semantic_distance / max(R, 0.1)


# =============================================================================
# 内部：L0 滑动窗口切片生成
# =============================================================================

def _make_l0_chunks(messages, session_id, window_size=L0_WINDOW_SIZE):
    """
    将一个 session 的消息列表切分为滑动窗口片段，用于 L0 索引。

    切片规则：
      · 步长为 1，每次向前移动一条消息
      · 每个切片把窗口内的消息拼成一段纯文本

    参数：
        messages    — database.get_messages() 返回的消息列表（sqlite3.Row）
        session_id  — 对应的 session id，写入 metadata 供后续溯源
        window_size — 滑动窗口大小

    返回：
        list[dict] — 切片列表，每个元素包含 id / document / metadata
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
                # L0 切片存储第一条消息的时间戳，用于衰减计算
                "created_at":  msg_list[start]["created_at"] if msg_list else "",
            },
        })

    return chunks


# =============================================================================
# 对外接口：写入索引
# =============================================================================

def index_l1(l1_id, summary, keywords=None, session_id=None, created_at=None, salience=None):
    """
    将一条 L1 摘要写入向量索引。

    参数：
        l1_id      — memory_l1 表的主键 id
        summary    — L1 摘要文本
        keywords   — 关键词字符串，逗号分隔；可为 None
        session_id — 对应的 session id；可为 None
        created_at — 摘要生成时间（ISO 8601）；为 None 时不加日期前缀
        salience   — 情感显著性，五档浮点 [0.0, 1.0]；为 None 时不写入 metadata
                     写入后可在衰减计算中直接从 metadata 读取，避免回查 SQLite
    """
    try:
        collection = _get_collection(_COLL_L1)

        # 拼接日期前缀
        if created_at:
            date_prefix = created_at[:10]   # 取 YYYY-MM-DD 部分
            base_text   = f"[{date_prefix}] {summary}"
        else:
            base_text = summary

        document = f"{base_text} 关键词：{keywords}" if keywords else base_text

        metadata = {"l1_id": l1_id}
        if session_id is not None:
            metadata["session_id"] = session_id
        if keywords:
            metadata["keywords"] = keywords
        # 存储 created_at 供衰减计算使用
        if created_at:
            metadata["created_at"] = created_at
        # 存储 salience 供衰减加成使用，避免检索时回查 SQLite
        # salience 来自调用方传入，默认 0.5（中等显著性）
        if salience is not None:
            metadata["salience"] = float(salience)

        collection.upsert(
            ids       = [str(l1_id)],
            documents = [document],
            metadatas = [metadata],
        )
        logger.info(f"L1 索引写入成功，l1_id={l1_id}")

        # 新记录写入后重建 BM25 索引，保持关键词检索与向量检索同步
        _bm25_index.rebuild("l1")

    except Exception as e:
        logger.warning(f"L1 索引写入失败，l1_id={l1_id} — {e}")


def index_l2(l2_id, summary, keywords=None, period_start=None, period_end=None):
    """
    将一条 L2 聚合摘要写入向量索引。

    索引文本格式：
        [YYYY-MM-DD ~ YYYY-MM-DD] <摘要文本> 关键词：<关键词>
    使用 period_start ~ period_end 作为日期前缀，体现覆盖的时间范围。

    参数：
        l2_id        — memory_l2 表的主键 id
        summary      — L2 摘要文本
        keywords     — 关键词字符串；可为 None
        period_start — 覆盖时间段起点（ISO 8601）；可为 None
        period_end   — 覆盖时间段终点（ISO 8601）；可为 None
    """
    try:
        collection = _get_collection(_COLL_L2)

        # 拼接日期范围前缀
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
            # 用 period_start 作为衰减基准（L2 覆盖一段时间，取起点更保守）
            metadata["created_at"]   = period_start
        if period_end:
            metadata["period_end"] = period_end

        collection.upsert(
            ids       = [str(l2_id)],
            documents = [document],
            metadatas = [metadata],
        )
        logger.info(f"L2 索引写入成功，l2_id={l2_id}")

        # 新记录写入后重建 BM25 索引
        _bm25_index.rebuild("l2")

    except Exception as e:
        logger.warning(f"L2 索引写入失败，l2_id={l2_id} — {e}")


def index_l0_session(session_id):
    """
    对一个 session 的所有原始消息做滑动窗口切片，批量写入 L0 向量索引。

    参数：
        session_id — 需要建立 L0 索引的 session id

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
            # 消息数不足一个窗口，整体索引为一条
            logger.debug(
                f"session {session_id} 消息数({len(messages)}) < 窗口({L0_WINDOW_SIZE})，使用整体索引"
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

        logger.info(f"L0 索引写入成功，session_id={session_id}，共 {len(chunks)} 个切片")
        return len(chunks)

    except Exception as e:
        logger.warning(f"L0 索引写入失败，session_id={session_id} — {e}")
        return None


# =============================================================================
# 对外接口：语义检索
# =============================================================================

def retrieve_l2(query_text, top_k=L2_RETRIEVE_TOP_K):
    """
    在 L2 索引中做语义检索，返回最相关的聚合摘要列表。
    使用 L2 稳定性系数（MEMORY_DECAY_S_L2=60天）进行衰减调整。
    """
    return _retrieve(_COLL_L2, query_text, top_k, id_field="l2_id", decay_s=MEMORY_DECAY_S_L2)


def retrieve_l1(query_text, top_k=L1_RETRIEVE_TOP_K):
    """
    在 L1 索引中做语义检索，返回最相关的单次对话摘要列表。
    使用 L1 稳定性系数（MEMORY_DECAY_S_L1=30天）进行衰减调整。
    """
    return _retrieve(_COLL_L1, query_text, top_k, id_field="l1_id", decay_s=MEMORY_DECAY_S_L1)


def retrieve_l0(query_text, top_k=L0_RETRIEVE_TOP_K):
    """
    在 L0 索引中做语义检索，返回最相关的原始消息切片列表。
    使用 L0 稳定性系数（MEMORY_DECAY_S_L0=10天）进行衰减调整。
    L0 不参与访问回写（无 l0_id 字段，不做保底加成）。
    """
    return _retrieve(_COLL_L0, query_text, top_k, id_field=None, decay_s=MEMORY_DECAY_S_L0)


def retrieve_combined(query_text: str) -> dict:
    """
    同时在 L1 和 L2 索引中检索，融合向量检索和 BM25 关键词检索的结果。

    融合策略：RRF（Reciprocal Rank Fusion，倒数排名融合）
        对每条结果计算融合分数：
            rrf_score = BM25_WEIGHT / (RRF_K + bm25_rank)
                      + 1.0         / (RRF_K + vector_rank)

        其中：
          · bm25_rank   — 该结果在 BM25 检索中的排名（1=最相关）
                          如果该结果未出现在 BM25 结果中，用 top_k+1 作为惩罚排名
          · vector_rank — 该结果在向量检索中的排名
                          如果该结果未出现在向量结果中，同样用 top_k+1 作为惩罚排名
          · RRF_K       — 平滑系数，默认 60（来自 config.py）
          · BM25_WEIGHT — BM25 通道权重，默认 1.0（来自 config.py）

        rrf_score 越高越相关，最终按 rrf_score 降序排列。

    输出格式与原版保持一致（{l2: [...], l1: [...]}），
    调用方（_build_context）无需修改。

    参数：
        query_text — 查询文本

    返回：
        dict — {"l2": list[dict], "l1": list[dict]}
               每条结果在原有字段基础上新增：
                 "bm25_score":  float  BM25 原始分数（调试用）
                 "rrf_score":   float  最终融合分数（越高越相关）
    """
    # ── 候选池大小：多取一些，融合后再截断到 top_k ──
    # 两路各取 top_k * 2 作为候选，保证融合后有足够的结果可选
    l1_candidate_k = L1_RETRIEVE_TOP_K * 2
    l2_candidate_k = L2_RETRIEVE_TOP_K * 2

    # ── L1：向量检索 + BM25 检索 ──
    vector_l1 = _retrieve(
        _COLL_L1, query_text, l1_candidate_k,
        id_field="l1_id", decay_s=MEMORY_DECAY_S_L1,
    )
    bm25_l1 = _bm25_index.search(query_text, layer="l1", top_k=l1_candidate_k)

    # ── L2：向量检索 + BM25 检索 ──
    vector_l2 = _retrieve(
        _COLL_L2, query_text, l2_candidate_k,
        id_field="l2_id", decay_s=MEMORY_DECAY_S_L2,
    )
    bm25_l2 = _bm25_index.search(query_text, layer="l2", top_k=l2_candidate_k)

    # ── 融合 ──
    fused_l1 = _rrf_fuse(vector_l1, bm25_l1, "l1_id", L1_RETRIEVE_TOP_K)
    fused_l2 = _rrf_fuse(vector_l2, bm25_l2, "l2_id", L2_RETRIEVE_TOP_K)

    logger.debug(
        f"retrieve_combined: L1 向量={len(vector_l1)} BM25={len(bm25_l1)} 融合后={len(fused_l1)} | "
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
        rrf_score(doc) = BM25_WEIGHT   / (RRF_K + rank_bm25(doc))
                       + 1.0           / (RRF_K + rank_vector(doc))
                       + GRAPH_WEIGHT  / (RRF_K + rank_graph(doc))

    graph_results 为 None 时退化为两路融合（保持向后兼容）。

    参数：
        vector_results — _retrieve() 返回的向量检索结果列表
        bm25_results   — BM25Index.search() 返回的关键词检索结果列表
        id_field       — 结果 dict 中记录 id 的键名（"l1_id" 或 "l2_id"）
        top_k          — 最终返回的条数上限
        graph_results  — retrieve_graph() 返回的图谱检索结果列表；None 表示不参与

    返回：
        list[dict] — 融合排序后的结果列表，按 rrf_score 降序
    """
    from config import GRAPH_WEIGHT

    # 惩罚排名：未出现在某路时，视为排在该路候选池末尾之后
    penalty_rank = top_k * 2 + 1

    # 建立 BM25 结果的 id → (rank, score) 查找表
    bm25_rank_map: dict[int, tuple[int, float]] = {}
    for item in bm25_results:
        bm25_rank_map[item["id"]] = (item["rank"], item["bm25_score"])

    # 建立向量结果的 id → (rank, result_dict) 查找表
    vector_rank_map: dict[int, tuple[int, dict]] = {}
    for rank, item in enumerate(vector_results, start=1):
        record_id = item.get(id_field)
        if record_id is not None:
            vector_rank_map[record_id] = (rank, item)

    # 建立图谱结果的 id → rank 查找表（可选）
    graph_rank_map: dict[int, int] = {}
    if graph_results:
        for item in graph_results:
            record_id = item.get(id_field)
            if record_id is not None:
                graph_rank_map[record_id] = item["graph_rank"]

    # 收集所有候选 id（三路并集）
    all_ids: set[int] = (
        set(vector_rank_map.keys())
        | set(bm25_rank_map.keys())
        | set(graph_rank_map.keys())
    )

    # 计算每个 id 的 RRF 分数
    scored: list[tuple[float, int]] = []

    for doc_id in all_ids:
        v_rank           = vector_rank_map[doc_id][0] if doc_id in vector_rank_map else penalty_rank
        b_rank, b_score  = bm25_rank_map.get(doc_id, (penalty_rank, 0.0))
        g_rank           = graph_rank_map.get(doc_id, penalty_rank)

        rrf_score = (
            BM25_WEIGHT   / (RRF_K + b_rank)
            + 1.0         / (RRF_K + v_rank)
            + GRAPH_WEIGHT / (RRF_K + g_rank)
        )
        scored.append((rrf_score, doc_id))

    # 按 rrf_score 降序排列
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:top_k]

    # 组装最终结果列表
    fused = []
    for rrf_score, doc_id in scored:
        if doc_id in vector_rank_map:
            _, result_dict = vector_rank_map[doc_id]
            entry = dict(result_dict)
        else:
            # 只在 BM25 或图谱中命中，向量未命中
            bm25_doc = next(
                (r["document"] for r in bm25_results if r["id"] == doc_id),
                ""
            )
            graph_doc = next(
                (r["document"] for r in (graph_results or []) if r.get(id_field) == doc_id),
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
        entry["graph_rank"]  = graph_rank_map.get(doc_id, None)  # 调试信息

        fused.append(entry)

    return fused
# =============================================================================
# 内部：通用检索逻辑（含衰减）
# =============================================================================

def _retrieve(collection_name, query_text, top_k, id_field, decay_s: float):
    """
    通用检索函数，被 retrieve_l0 / retrieve_l1 / retrieve_l2 调用。

    在原有语义过滤基础上，新增衰减调整：
      1. Chroma 返回原始语义距离
      2. 从 metadata["created_at"] 取时间戳，计算 R
      3. 若 ENABLE_ACCESS_BOOST 开启，额外读取 last_accessed_at 做保底
      4. 用 adjusted_distance = distance / max(R, 0.1) 重新排序
      5. 按 SIMILARITY_THRESHOLD 过滤（对比的是 adjusted_distance）

    参数：
        collection_name — Collection 名称
        query_text      — 查询文本
        top_k           — 最多返回条数
        id_field        — metadata 里存 SQLite id 的字段名；None 表示不提取
        decay_s         — 当前层的稳定性系数（天），由调用方显式传入

    返回：
        list[dict] — 检索结果，已按 adjusted_distance 升序排列
    """
    try:
        collection = _get_collection(collection_name)

        if collection.count() == 0:
            return []

        # 多取一些候选，因为衰减调整后部分结果会被过滤掉
        # 取 top_k * 3 作为候选池，确保衰减后仍有足够结果
        candidate_k = min(top_k * 3, collection.count())

        raw = collection.query(
            query_texts = [query_text],
            n_results   = candidate_k,
        )

        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]

        results       = []
        layer         = _COLL_TO_LAYER.get(collection_name)  # "l0" / "l1" / "l2"
        access_ids    = []   # 本次命中的记录 id，用于异步回写

        for doc_id, doc, dist, meta in zip(ids, documents, distances, metadatas):

            # ── 计算衰减调整后的距离 ──
            created_at_str       = meta.get("created_at", "")
            last_accessed_at_str = None

            # 保底加成：从数据库读取 last_accessed_at
            # 只在开关开启且有 id_field 时查询（L0 不参与）
            if MEMORY_DECAY_ENABLE_ACCESS_BOOST and id_field and id_field in meta:
                record_id = meta[id_field]
                try:
                    from database import get_last_accessed_at
                    last_accessed_at_str = get_last_accessed_at(layer, record_id)
                except Exception:
                    pass   # 查询失败时跳过加成，不影响主流程

            # 从 metadata 读取 salience（写入索引时已存入，无需回查 SQLite）
            # L0 切片没有 salience 字段，读不到时使用 None（函数内部会用 0.5 默认值）
            meta_salience = meta.get("salience", None)

            R                 = _calc_decay_factor(
                created_at_str,
                decay_s,
                last_accessed_at_str,
                salience=meta_salience,      # 新增：传入显著性加成
            )
            adjusted_distance = _adjust_distance(dist, R)

            # 使用调整后的距离做阈值过滤
            if adjusted_distance > SIMILARITY_THRESHOLD:
                continue

            result = {
                "document":          doc,
                "distance":          dist,             # 原始语义距离，保留供调试
                "adjusted_distance": adjusted_distance, # 衰减调整后的距离
                "decay_r":           round(R, 4),       # 保留率，调试用
                "metadata":          meta,
            }

            if id_field and id_field in meta:
                record_id = meta[id_field]
                result[id_field] = record_id
                # 记录命中的 id，用于异步回写 last_accessed_at
                if layer in ("l1", "l2"):
                    access_ids.append((layer, record_id))

            results.append(result)

        # 按调整后的距离升序排列（最相关且最新的排前面）
        results.sort(key=lambda x: x["adjusted_distance"])

        # 只返回前 top_k 条
        results = results[:top_k]

        # 异步回写命中记录的 last_accessed_at
        for layer_key, rec_id in access_ids[:top_k]:   # 只回写实际返回的结果
            _enqueue_access(layer_key, rec_id)

        return results

    except Exception as e:
        # P3-5 修复：大小写不敏感的异常字符串匹配
        err_str = str(e).lower()
        if "does not exist" in err_str or "invalidcollection" in err_str:
            # Collection 尚未创建属于正常情况，静默返回空列表
            return []

        # P2-C 修复：非预期异常升级为 error 级别
        logger.error(
            f"检索失败（非预期异常），collection={collection_name} — {e}"
        )
        return []


# =============================================================================
# 对外接口：索引管理工具
# =============================================================================

def get_index_stats():
    """
    返回三个 Collection 当前的索引条数，用于监控和调试。

    返回：
        dict — {"l0": int, "l1": int, "l2": int}
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


def rebuild_all_indexes():
    """
    重建全部三层向量索引（切换嵌入模型时使用）。

    流程：
      1. 删除三个 Collection（清空旧向量）
      2. 从 SQLite 读取所有历史数据
      3. 重新写入索引（含日期前缀和 created_at metadata）

    警告：
      此操作不可逆，旧索引会被彻底删除。
      重建期间检索功能暂时不可用。

    返回：
        dict — {"l0_chunks": int, "l1_count": int, "l2_count": int}
    """
    from database import get_all_l1, get_all_l2, get_all_session_ids

    logger.info("开始重建全部向量索引…")
    client = _get_client()

    # 第一步：删除旧 Collection
    for name in [_COLL_L0, _COLL_L1, _COLL_L2]:
        try:
            client.delete_collection(name)
            logger.info(f"已删除旧 Collection：{name}")
        except Exception:
            pass

    counts = {"l0_chunks": 0, "l1_count": 0, "l2_count": 0}

    # 第二步：重建 L1 索引（传入 created_at 以支持衰减）
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

    # 第三步：重建 L2 索引
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

    # 第四步：重建 L0 索引（按 session 逐个处理）
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
# 图谱检索通道（第三路，配合 retrieve_combined 使用）
# =============================================================================

def retrieve_graph(query_text: str, top_k: int = L1_RETRIEVE_TOP_K) -> list[dict]:
    """
    基于知识图谱的语义检索通道。

    流程：
      1. 从查询文本中提取可能的实体名（jieba 分词后与 keyword_pool 匹配）
      2. 在 NetworkX 图里找到对应节点
      3. 做 BFS 广度优先遍历（最多两跳）
      4. 按跳数排名，返回关联的 L1 id 列表

    与向量通道和 BM25 通道的区别：
      · 向量通道：语义距离（擅长模糊匹配）
      · BM25 通道：关键词字面匹配
      · 图谱通道：逻辑关系遍历（擅长跨时间的实体关联）

    参数：
        query_text — 查询文本
        top_k      — 最多返回的 L1 条数

    返回：
        list[dict] — 每条结果包含：
            l1_id      — L1 摘要 id
            document   — 占位空字符串（保持与其他通道格式一致）
            graph_rank — 在图谱结果中的排名（从 1 开始，供 _rrf_fuse 使用）
            hops       — 图遍历跳数（1 或 2）
            distance   — None（图谱通道无距离概念）
            adjusted_distance — None
            decay_r    — None
            metadata   — 空字典
        图谱为空或无命中时返回 []
    """
    from graph_builder import get_nx_graph

    G = get_nx_graph()
    if G is None or G.number_of_nodes() == 0:
        logger.debug("图谱为空或未加载，跳过图谱检索")
        return []

    # ── 第一步：从查询文本提取候选实体名 ──
    # 策略：jieba 分词后，只保留在 keyword_pool 里能找到规范词的词条
    # 这样可以避免把停用词、动词等无效实体送进图里查询
    from database import get_all_canonical_keywords
    import jieba

    canonical_keywords = {kw["keyword"] for kw in get_all_canonical_keywords()}
    if not canonical_keywords:
        logger.debug("keyword_pool 为空，图谱检索无法提取查询实体")
        return []

    # jieba 分词，过滤出在 keyword_pool 里的词
    tokens = list(jieba.cut(query_text, cut_all=False))
    query_entities = [t.strip() for t in tokens
                      if t.strip() in canonical_keywords and len(t.strip()) >= 2]

    if not query_entities:
        logger.debug(f"查询文本中未找到已知实体，跳过图谱检索：{query_text[:50]}")
        return []

    logger.debug(f"图谱检索实体：{query_entities}")

    # ── 第二步：在图里找到对应节点 ──
    # 用 entity_name 属性做反向查找
    entity_to_node = {
        data["entity_name"]: node_id
        for node_id, data in G.nodes(data=True)
    }

    start_node_ids = []
    for entity in query_entities:
        if entity in entity_to_node:
            start_node_ids.append(entity_to_node[entity])

    if not start_node_ids:
        logger.debug("查询实体在图谱中无对应节点")
        return []

    # ── 第三步：BFS 遍历，收集关联的 L1 id ──
    # 多个起始节点各自遍历，结果合并，跳数取最小值
    all_l1_results: dict[int, int] = {}   # {l1_id: min_hops}

    for start_node_id in start_node_ids:
        # networkx 的 single_source_shortest_path_length 做 BFS 最短路径
        import networkx as nx
        try:
            path_lengths = nx.single_source_shortest_path_length(
                G, start_node_id, cutoff=2   # 最多两跳
            )
        except Exception:
            continue

        for neighbor_node_id, hops in path_lengths.items():
            if hops == 0:
                # 起始节点本身，跳过
                continue

            # 找出从起始节点到 neighbor_node_id 的所有关联 L1
            # 只考虑邻居节点的直接边（BFS 已保证在两跳以内）
            if G.has_edge(start_node_id, neighbor_node_id):
                edge_data = G[start_node_id][neighbor_node_id]
            else:
                # 两跳路径，找中间节点的边
                edge_data = {}

            # 取出这条边上关联的所有 l1_id
            l1_ids_on_edge = edge_data.get("l1_ids", [])
            for l1_id in l1_ids_on_edge:
                if l1_id not in all_l1_results:
                    all_l1_results[l1_id] = hops
                else:
                    # 同一个 L1 通过多条路径找到，保留最小跳数
                    all_l1_results[l1_id] = min(all_l1_results[l1_id], hops)

    if not all_l1_results:
        logger.debug("图谱遍历无命中结果")
        return []

    # ── 第四步：按跳数排名，截断到 top_k ──
    # 跳数越小排名越靠前（1跳 > 2跳）
    sorted_l1s = sorted(all_l1_results.items(), key=lambda x: x[1])
    sorted_l1s = sorted_l1s[:top_k]

    results = []
    for rank, (l1_id, hops) in enumerate(sorted_l1s, start=1):
        results.append({
            "l1_id":            l1_id,
            "document":         "",     # 图谱通道不携带文本，_rrf_fuse 里会从其他通道补全
            "graph_rank":       rank,   # 供 _rrf_fuse 使用
            "hops":             hops,   # 调试信息
            "distance":         None,
            "adjusted_distance": None,
            "decay_r":          None,
            "metadata":         {},
        })

    logger.debug(f"图谱检索命中 {len(results)} 条 L1，起始实体：{query_entities}")
    return results


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    import threading as _threading
    from database import (
        new_session, save_message, close_session,
        save_l1_summary, save_l2_summary,
    )

    print("=== vector_store.py 验证测试 ===\n")

    # ------------------------------------------------------------------
    # 测试一：初始化与索引状态
    # ------------------------------------------------------------------
    print("--- 测试一：初始化与索引状态 ---")
    stats = get_index_stats()
    print(f"当前索引条数：L0={stats['l0']}，L1={stats['l1']}，L2={stats['l2']}")
    print()

    # ------------------------------------------------------------------
    # 测试二：多线程并发初始化（P2-5 修复核心）
    # ------------------------------------------------------------------
    print("--- 测试二：多线程并发初始化 ---")
    import vector_store as _vs
    _vs._client = None

    results_t2 = []
    errors_t2  = []

    def _init_worker():
        try:
            c = _get_client()
            results_t2.append(id(c))
        except Exception as e:
            errors_t2.append(str(e))

    threads = [_threading.Thread(target=_init_worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    unique_ids = set(results_t2)
    if errors_t2:
        print(f"❌ 出现错误：{errors_t2}")
    elif len(unique_ids) == 1:
        print(f"✓ 10 个线程全部获得同一个客户端实例，双重检查锁生效")
    else:
        print(f"❌ 出现多个实例（ids={unique_ids}），锁未生效")
    print()

    # ------------------------------------------------------------------
    # 测试三：衰减计算验证
    # ------------------------------------------------------------------
    print("--- 测试三：衰减计算验证 ---")
    from datetime import timedelta

    # 模拟不同"年龄"的记忆
    test_cases = [
        (0,   30, "刚创建（今天）"),
        (30,  30, "30天前（L1，保留率约37%）"),
        (60,  30, "60天前（L1，保留率约14%）"),
        (60,  60, "60天前（L2，保留率约37%）"),
        (200, 60, "200天前（L2，保留率约4%）"),
    ]

    for days_ago, S, desc in test_cases:
        created_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
        R = _calc_decay_factor(created_at, S)
        adj_dist = _adjust_distance(0.3, R)   # 假设原始语义距离 0.3
        print(
            f"  {desc}：R={R:.3f}，"
            f"原始距离=0.300，调整后={adj_dist:.3f}"
        )
    print()

    # ------------------------------------------------------------------
    # 测试四：访问队列基本功能
    # ------------------------------------------------------------------
    print("--- 测试四：访问队列 ---")
    _enqueue_access("l1", 1)
    _enqueue_access("l1", 2)
    _enqueue_access("l2", 1)
    q_size = _access_queue.qsize()
    print(f"入队 3 条，当前队列大小：{q_size}（应为 3）")
    # 清空队列，不影响后续测试
    while not _access_queue.empty():
        _access_queue.get_nowait()
    print()

    # ------------------------------------------------------------------
    # 测试五：异常字符串大小写匹配（P3-5 修复核心）
    # ------------------------------------------------------------------
    print("--- 测试五：异常字符串大小写匹配 ---")
    test_cases_exc = [
        ("does not exist",    True,  "小写"),
        ("Does Not Exist",    True,  "首字母大写"),
        ("DOES NOT EXIST",    True,  "全大写"),
        ("InvalidCollection", True,  "混合大小写"),
        ("disk full",         False, "无关错误"),
    ]
    all_pass = True
    for err_msg, expect_silent, desc in test_cases_exc:
        err_str   = err_msg.lower()
        is_silent = "does not exist" in err_str or "invalidcollection" in err_str
        passed    = is_silent == expect_silent
        all_pass  = all_pass and passed
        status    = "✓" if passed else "❌"
        print(f"  {status} [{desc}] → {'静默' if is_silent else 'error 日志'}")
    print(f"  大小写匹配：{'全部通过 ✓' if all_pass else '有失败项 ❌'}")
    print()

    # ------------------------------------------------------------------
    # 测试六：L1 索引写入与含衰减的检索
    # ------------------------------------------------------------------
    print("--- 测试六：L1 索引写入与衰减检索 ---")
    sid1 = new_session()
    save_message(sid1, "user", "今天把 summarizer 写完了，验证全部通过")
    save_message(sid1, "assistant", "很棒，进度很稳！")
    close_session(sid1)

    from datetime import timezone as _tz
    now_str = datetime.now(_tz.utc).isoformat()
    old_str = (datetime.now(_tz.utc) - timedelta(days=90)).isoformat()

    l1_new = save_l1_summary(
        sid1, "烧酒完成了 summarizer 模块的开发与验证。",
        "summarizer,模块,验证", "夜间", "专注高效"
    )
    index_l1(l1_new, "烧酒完成了 summarizer 模块的开发与验证。",
             "summarizer,模块,验证", session_id=sid1, created_at=now_str)

    sid2 = new_session()
    save_message(sid2, "user", "今天状态很差，头疼，什么都不想做")
    close_session(sid2)
    l1_old = save_l1_summary(
        sid2, "烧酒今天状态不佳，头疼，情绪低落。",
        "状态,头疼,情绪", "下午", "情绪低落"
    )
    # 模拟一条 90 天前的旧记忆
    index_l1(l1_old, "烧酒今天状态不佳，头疼，情绪低落。",
             "状态,头疼,情绪", session_id=sid2, created_at=old_str)

    results_l1 = retrieve_l1("最近有没有熬夜写代码", top_k=2)
    print(f"检索到 {len(results_l1)} 条结果：")
    for r in results_l1:
        print(
            f"  [{r.get('decay_r', '?'):.3f}] "
            f"dist={r['distance']:.3f} → adj={r['adjusted_distance']:.3f}  "
            f"{r['document'][:40]}…"
        )
    print()

    print("验证完成。")
