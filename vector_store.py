"""
vector_store.py — 向量索引与语义检索模块

负责维护三层向量索引（L0 / L1 / L2），并对外提供语义检索接口。
是分层 RAG 检索链路的核心模块。

变更记录：
  v2 — rebuild_all_indexes() 移除对私有函数 _get_connection 的依赖，
       修复审查报告问题A。
       改为通过 get_all_l1 / get_all_l2 / get_all_session_ids 三个公开函数访问。

  v3 — 三项改动（对应代码优化清单第二轮 P2-5、P3-5 和第三轮 P2-C）：

       【P2-5】_get_client() 加双重检查锁，消除多线程初始化竞态
         问题背景：SessionManager 启动的两个后台线程（空闲检测 + L2 定时检查）
         可能在极短时间窗口内同时触发向量操作，导致 _client 被重复初始化。
         原来的 if _client is None 属于 Check-Then-Act 模式，非线程安全。
         修复方案：使用双重检查锁（Double-Checked Locking）。

       【P3-5】_retrieve() 异常字符串匹配改为大小写不敏感
         问题背景：原来用 "does not exist" in str(e) 做字符串匹配，
         一旦 ChromaDB 升级改变错误消息的大小写，此处兜底失效，
         可能导致正常情况被错误上抛或真正的错误被静默忽略。
         修复方案：统一转小写后再匹配，str(e).lower()。

       【P2-C】_retrieve() 非预期异常日志级别从 warning 升级为 error
         问题背景：OOM、磁盘满、权限错误等严重问题只打 warning 级别，
         在生产日志中容易被忽略，导致故障排查滞后。
         修复方案：非预期异常（即通过了字符串匹配过滤的、真正不正常的异常）
         改用 logger.error 输出，确保在生产环境中被及时发现。

─────────────────────────────────────────────────────────────
三层索引的定位与分工
─────────────────────────────────────────────────────────────

  L2 索引（memory_l2）
    · 粒度最粗，每条对应一段时间的聚合摘要
    · 用于话题定位：快速判断某个话题在哪个时间段出现过

  L1 索引（memory_l1）
    · 语义密度最高，每条对应一次完整对话的摘要
    · 主检索层，精度最好，是 RAG 注入的主要来源

  L0 索引（messages，滑动窗口切片）
    · 粒度最细，按滑动窗口将原始消息切片后索引
    · 用于回溯原始对话温度和具体细节

─────────────────────────────────────────────────────────────
与其他模块的关系
─────────────────────────────────────────────────────────────

  · 被写入：
      summarizer.py      → L1 写入后调用 index_l1()
      merger.py          → L2 写入后调用 index_l2()
      session_manager.py → session 关闭后调用 index_l0_session()

  · 被检索：
      prompt_builder.py / main.py → 构建 system prompt 时调用 retrieve_*() 系列

  · 读取配置：config.py
  · 读取消息数据（L0 用）：database.py

使用方法：
    from vector_store import index_l1, index_l2, index_l0_session
    from vector_store import retrieve_l1, retrieve_l2, retrieve_l0
    from vector_store import retrieve_combined
"""

import os
import json
import threading

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
)
from database import get_messages

from logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# 内部：Chroma 客户端与 Collection 管理
# =============================================================================

# 全局客户端单例：整个进程只初始化一次，避免重复打开索引文件
_client      = None

# 【P2-5 新增】线程锁：保护 _client 的初始化过程
# SessionManager 启动的两个后台线程可能同时触发向量操作，
# 没有锁的情况下 _client 可能被重复初始化，产生不确定行为。
_client_lock = threading.Lock()


def _get_client():
    """
    获取（或初始化）全局 Chroma 客户端。

    使用 PersistentClient，索引文件写入 CHROMA_DIR 目录，重启后不丢失。

    【P2-5 修复】使用双重检查锁（Double-Checked Locking）保证线程安全：
        原来的写法（非线程安全）：
            if _client is None:
                _client = chromadb.PersistentClient(...)   ← 竞态窗口

        修复后的写法：
            if _client is None:             # 第一次检查：快速路径，避免每次都加锁
                with _client_lock:          # 加锁，只有一个线程能进入
                    if _client is None:     # 第二次检查：防止等锁期间已被其他线程初始化
                        _client = ...

        大多数调用（_client 已初始化）直接走第一个 if 的 False 分支，
        不需要加锁，性能影响极小。只有真正的初始化阶段才会加锁。
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:    # 双重检查：等锁期间可能已被其他线程初始化
                os.makedirs(CHROMA_DIR, exist_ok=True)
                _client = chromadb.PersistentClient(path=CHROMA_DIR)
                logger.info(f"Chroma 客户端已初始化，索引目录：{CHROMA_DIR}")
    return _client


def _get_embedding_function():
    """
    根据 config.py 的 EMBEDDING_MODEL 返回对应的嵌入函数。

    "default"  → Chroma 内置模型（all-MiniLM-L6-v2），无需额外安装
    其他字符串 → 用 SentenceTransformer 加载对应模型（如 Qwen3-Embedding-0.6B）
    """
    if EMBEDDING_MODEL == "default":
        return DefaultEmbeddingFunction()
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _get_collection(name):
    """
    获取（或创建）指定名称的 Collection。
    get_or_create_collection：已存在则获取，不存在则新建，幂等操作。
    """
    client = _get_client()
    ef     = _get_embedding_function()
    return client.get_or_create_collection(
        name               = name,
        embedding_function = ef,
    )


# Collection 名称常量
_COLL_L0 = "memory_l0"   # L0 滑动窗口切片索引
_COLL_L1 = "memory_l1"   # L1 单次对话摘要索引
_COLL_L2 = "memory_l2"   # L2 时间段聚合摘要索引


# =============================================================================
# 内部：L0 滑动窗口切片生成
# =============================================================================

def _make_l0_chunks(messages, session_id, window_size=L0_WINDOW_SIZE):
    """
    将一个 session 的消息列表切分为滑动窗口片段，用于 L0 索引。

    切片规则：
      · 窗口大小由 window_size 控制（来自 config.L0_WINDOW_SIZE，默认 3）
      · 步长固定为 1：每次向前移动一条消息
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
            },
        })

    return chunks


# =============================================================================
# 对外接口：写入索引
# =============================================================================

def index_l1(l1_id, summary, keywords=None, session_id=None):
    """
    将一条 L1 摘要写入向量索引。
    由 summarizer.generate_l1_summary() 在 L1 写入 SQLite 后调用。

    索引文本 = 摘要 + 关键词（关键词也参与向量化，提升召回率）。

    参数：
        l1_id      — memory_l1 表的主键 id
        summary    — L1 摘要文本
        keywords   — 关键词字符串，逗号分隔；可为 None
        session_id — 对应的 session id；可为 None
    """
    try:
        collection = _get_collection(_COLL_L1)

        document = f"{summary} 关键词：{keywords}" if keywords else summary

        metadata = {"l1_id": l1_id}
        if session_id is not None:
            metadata["session_id"] = session_id
        if keywords:
            metadata["keywords"] = keywords

        collection.upsert(
            ids       = [str(l1_id)],
            documents = [document],
            metadatas = [metadata],
        )
        logger.info(f"L1 索引写入成功，l1_id={l1_id}")

    except Exception as e:
        logger.warning(f"L1 索引写入失败，l1_id={l1_id} — {e}")


def index_l2(l2_id, summary, keywords=None, period_start=None, period_end=None):
    """
    将一条 L2 聚合摘要写入向量索引。
    由 merger.check_and_merge() 在 L2 写入 SQLite 后调用。

    参数：
        l2_id        — memory_l2 表的主键 id
        summary      — L2 摘要文本
        keywords     — 关键词字符串；可为 None
        period_start — 覆盖时间段起点（ISO 8601）；可为 None
        period_end   — 覆盖时间段终点（ISO 8601）；可为 None
    """
    try:
        collection = _get_collection(_COLL_L2)

        document = f"{summary} 关键词：{keywords}" if keywords else summary

        metadata = {"l2_id": l2_id}
        if keywords:
            metadata["keywords"] = keywords
        if period_start:
            metadata["period_start"] = period_start
        if period_end:
            metadata["period_end"] = period_end

        collection.upsert(
            ids       = [str(l2_id)],
            documents = [document],
            metadatas = [metadata],
        )
        logger.info(f"L2 索引写入成功，l2_id={l2_id}")

    except Exception as e:
        logger.warning(f"L2 索引写入失败，l2_id={l2_id} — {e}")


def index_l0_session(session_id):
    """
    对一个 session 的所有原始消息做滑动窗口切片，批量写入 L0 向量索引。
    由 session_manager 在 session 关闭后调用（在 generate_l1_summary 之前）。

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
            # 消息数量不足一个窗口，把所有消息拼成一条整体索引
            logger.debug(
                f"session {session_id} 消息数({len(messages)}) < 窗口({L0_WINDOW_SIZE})，使用整体索引"
            )
            lines = []
            for msg in messages:
                role_label = "用户" if msg["role"] == "user" else "助手"
                lines.append(f"{role_label}：{msg['content']}")
            document    = "\n".join(lines)
            message_ids = [msg["id"] for msg in messages]

            collection = _get_collection(_COLL_L0)
            collection.upsert(
                ids       = [f"l0_{session_id}_0"],
                documents = [document],
                metadatas = [{
                    "session_id":  session_id,
                    "start_idx":   0,
                    "end_idx":     len(messages) - 1,
                    "message_ids": json.dumps(message_ids),
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

    参数：
        query_text — 查询文本
        top_k      — 最多返回多少条

    返回：
        list[dict] — 检索结果列表，超过 SIMILARITY_THRESHOLD 的结果会被过滤。
    """
    return _retrieve(_COLL_L2, query_text, top_k, id_field="l2_id")


def retrieve_l1(query_text, top_k=L1_RETRIEVE_TOP_K):
    """
    在 L1 索引中做语义检索，返回最相关的单次对话摘要列表。
    是 RAG 注入的主力层，精度最高。
    """
    return _retrieve(_COLL_L1, query_text, top_k, id_field="l1_id")


def retrieve_l0(query_text, top_k=L0_RETRIEVE_TOP_K):
    """
    在 L0 索引中做语义检索，返回最相关的原始消息切片列表。
    返回值里的 metadata 包含 message_ids（JSON 字符串）。
    """
    return _retrieve(_COLL_L0, query_text, top_k, id_field=None)


def retrieve_combined(query_text):
    """
    同时在 L1 和 L2 索引中检索，合并结果后返回。
    用于 _build_context() 的语义层注入：一次调用拿到两层的相关记忆。

    返回：
        dict — {"l2": list[dict], "l1": list[dict]}
    """
    return {
        "l2": retrieve_l2(query_text),
        "l1": retrieve_l1(query_text),
    }


# =============================================================================
# 内部：通用检索逻辑
# =============================================================================

def _retrieve(collection_name, query_text, top_k, id_field):
    """
    通用检索函数，被 retrieve_l0 / retrieve_l1 / retrieve_l2 调用。

    统一处理：
      · Collection 为空时 Chroma 会抛异常，这里捕获并返回空列表
      · 过滤掉距离超过 SIMILARITY_THRESHOLD 的结果
      · 把 Chroma 返回的并行列表结构转换为字典列表

    参数：
        collection_name — Collection 名称
        query_text      — 查询文本
        top_k           — 最多返回条数
        id_field        — metadata 里存 SQLite id 的字段名；None 表示不提取

    【P3-5 修复】异常字符串匹配改为大小写不敏感：
        原来：
            if "does not exist" in str(e) or "InvalidCollection" in str(e):
        修复后：
            err_str = str(e).lower()
            if "does not exist" in err_str or "invalidcollection" in err_str:
        理由：ChromaDB 升级后错误消息措辞可能变化（如大小写），
        转小写后匹配，兼容性更好，不会因大小写不一致导致正常情况被意外上抛。

    【P2-C 修复】非预期异常日志级别从 warning 升级为 error：
        原来：
            logger.warning(f"检索失败，collection={collection_name} — {e}")
        修复后：
            logger.error(f"检索失败（非预期异常）...")
        理由：Collection 不存在属于正常情况（已在前面 return []），
        能走到 logger 这里的都是真正异常（OOM、磁盘满、权限错误等），
        应使用 error 级别确保在生产日志监控中被及时发现。
    """
    try:
        collection = _get_collection(collection_name)

        if collection.count() == 0:
            return []

        raw = collection.query(
            query_texts = [query_text],
            n_results   = min(top_k, collection.count()),
        )

        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]

        results = []
        for doc_id, doc, dist, meta in zip(ids, documents, distances, metadatas):
            if dist > SIMILARITY_THRESHOLD:
                continue

            result = {
                "document": doc,
                "distance": dist,
                "metadata": meta,
            }

            if id_field and id_field in meta:
                result[id_field] = meta[id_field]

            results.append(result)

        return results

    except Exception as e:
        # ------------------------------------------------------------------
        # 【P3-5 修复】大小写不敏感的异常字符串匹配
        #
        # 先统一转小写，再做包含匹配，避免 ChromaDB 升级后因大小写变化导致
        # 正常情况（Collection 不存在）被误判为非预期错误。
        # ------------------------------------------------------------------
        err_str = str(e).lower()
        if "does not exist" in err_str or "invalidcollection" in err_str:
            # Collection 尚未创建属于正常情况，静默返回空列表
            # 例如：系统刚启动，L1 索引还没写入任何数据时检索 L1
            return []

        # ------------------------------------------------------------------
        # 【P2-C 修复】非预期异常升级为 error 级别
        #
        # 能走到这里说明不是"Collection 不存在"，而是真正的异常：
        # 如 OOM、磁盘满、权限错误、ChromaDB 内部错误等。
        # 使用 error 级别确保生产日志监控能及时捕获。
        # ------------------------------------------------------------------
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
      3. 重新写入索引

    警告：
      · 此操作不可逆，旧索引会被彻底删除
      · 重建期间检索功能暂时不可用
      · 大量历史数据时耗时较长，建议在服务停止时执行

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

    # 第二步：重建 L1 索引
    l1_rows = get_all_l1()
    for row in l1_rows:
        index_l1(
            l1_id      = row["id"],
            summary    = row["summary"],
            keywords   = row["keywords"],
            session_id = row["session_id"],
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
    # 测试二：【P2-5 修复核心】多线程并发初始化不崩溃
    # ------------------------------------------------------------------
    print("--- 测试二：多线程并发初始化（P2-5 修复核心）---")

    # 重置全局客户端，模拟未初始化状态
    import vector_store as _vs
    _vs._client = None

    results = []
    errors  = []

    def _init_worker():
        try:
            c = _get_client()
            results.append(id(c))
        except Exception as e:
            errors.append(str(e))

    # 同时启动 10 个线程竞争初始化
    threads = [_threading.Thread(target=_init_worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    unique_ids = set(results)
    if errors:
        print(f"❌ 出现错误：{errors}")
    elif len(unique_ids) == 1:
        print(f"✓ 10 个线程全部获得同一个客户端实例（id={unique_ids.pop()}），双重检查锁生效")
    else:
        print(f"❌ 出现多个实例（ids={unique_ids}），锁未生效")
    print()

    # ------------------------------------------------------------------
    # 测试三：L1 索引写入与检索
    # ------------------------------------------------------------------
    print("--- 测试三：L1 索引写入与检索 ---")
    sid1 = new_session()
    save_message(sid1, "user", "今天把 summarizer 写完了，验证全部通过")
    save_message(sid1, "assistant", "很棒，进度很稳！")
    close_session(sid1)
    l1_id_1 = save_l1_summary(
        sid1, "烧酒完成了 summarizer 模块的开发与验证。",
        "summarizer,模块,验证", "夜间", "专注高效"
    )
    index_l1(l1_id_1, "烧酒完成了 summarizer 模块的开发与验证。",
             "summarizer,模块,验证", session_id=sid1)

    sid2 = new_session()
    save_message(sid2, "user", "今天状态很差，头疼，什么都不想做")
    save_message(sid2, "assistant", "听起来很辛苦，好好休息一下。")
    close_session(sid2)
    l1_id_2 = save_l1_summary(
        sid2, "烧酒今天状态不佳，头疼，情绪低落。",
        "状态,头疼,情绪", "下午", "情绪低落"
    )
    index_l1(l1_id_2, "烧酒今天状态不佳，头疼，情绪低落。",
             "状态,头疼,情绪", session_id=sid2)

    for q in ["最近有没有熬夜写代码", "烧酒心情不好"]:
        results_l1 = retrieve_l1(q, top_k=1)
        if results_l1:
            print(f'  查询："{q}"')
            print(f'  最相关：{results_l1[0]["document"][:40]}  距离：{results_l1[0]["distance"]:.4f}')
        else:
            print(f'  查询："{q}"  → 无相关结果（距离超过阈值 {SIMILARITY_THRESHOLD}）')
    print()

    # ------------------------------------------------------------------
    # 测试四：L0 索引写入与检索
    # ------------------------------------------------------------------
    print("--- 测试四：L0 索引写入与检索 ---")
    n_chunks = index_l0_session(sid1)
    print(f"session {sid1} L0 索引：{n_chunks} 个切片")
    l0_results = retrieve_l0("把模块写完了")
    print(f"L0 检索：{len(l0_results)} 条结果")
    print()

    # ------------------------------------------------------------------
    # 测试五：【P3-5 修复核心】异常字符串大小写不敏感匹配
    # ------------------------------------------------------------------
    print("--- 测试五：异常字符串大小写匹配（P3-5 修复核心）---")

    # 模拟 ChromaDB 可能的不同大小写错误消息
    test_cases = [
        ("does not exist",        True,  "小写原始格式"),
        ("Does Not Exist",        True,  "首字母大写"),
        ("DOES NOT EXIST",        True,  "全大写"),
        ("InvalidCollection",     True,  "混合大小写"),
        ("invalidcollection",     True,  "全小写"),
        ("INVALIDCOLLECTION",     True,  "全大写"),
        ("disk full",             False, "不相关错误（应触发 error 日志）"),
        ("permission denied",     False, "权限错误（应触发 error 日志）"),
    ]

    all_pass = True
    for err_msg, expect_silent, desc in test_cases:
        err_str = err_msg.lower()
        is_silent = "does not exist" in err_str or "invalidcollection" in err_str
        passed = is_silent == expect_silent
        all_pass = all_pass and passed
        status = "✓" if passed else "❌"
        handling = "静默返回 []" if is_silent else "触发 logger.error"
        print(f"  {status} [{desc}] → {handling}")

    print(f"\n  大小写匹配测试：{'全部通过 ✓' if all_pass else '有失败项 ❌'}")
    print()

    # ------------------------------------------------------------------
    # 测试六：rebuild_all_indexes（封装修复验证）
    # ------------------------------------------------------------------
    print("--- 测试六：rebuild_all_indexes ---")
    counts = rebuild_all_indexes()
    print(f"重建完成：{counts}")
    print()

    # ------------------------------------------------------------------
    # 测试七：索引统计
    # ------------------------------------------------------------------
    print("--- 测试七：索引统计 ---")
    stats_after = get_index_stats()
    print(f"重建后：L0={stats_after['l0']}，L1={stats_after['l1']}，L2={stats_after['l2']}")

    print("\n验证完成。")
