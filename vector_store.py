"""
vector_store.py — 向量索引与语义检索模块

负责维护三层向量索引（L0 / L1 / L2），并对外提供语义检索接口。
是阶段二分层 RAG 检索链路的核心模块。

─────────────────────────────────────────────────────────────
三层索引的定位与分工
─────────────────────────────────────────────────────────────

  L2 索引（memory_l2）
    · 粒度最粗，每条对应一段时间（通常几天到几周）的聚合摘要
    · 用于话题定位：快速判断某个话题在哪个时间段出现过
    · 检索入口：retrieve_l2()

  L1 索引（memory_l1）
    · 语义密度最高，每条对应一次完整对话的摘要
    · 主检索层，精度最好，是 RAG 注入的主要来源
    · 检索入口：retrieve_l1()

  L0 索引（messages，滑动窗口切片）
    · 粒度最细，按"滑动窗口"将原始消息切片后索引
    · 用于回溯原始对话温度和具体细节
    · 检索入口：retrieve_l0()

─────────────────────────────────────────────────────────────
滑动窗口切片说明（L0 专属）
─────────────────────────────────────────────────────────────

  原始消息不适合一条一条地单独建索引——单条消息往往很短、缺乏上下文，
  语义向量的质量很差（例如"好的"这种消息，向量几乎没有意义）。

  解决方案是"滑动窗口"：把 session 里的消息按窗口切片，
  每个切片包含连续的若干条消息，将它们拼成一段文本后再向量化。

  例如 session 里有 10 条消息，窗口大小为 3，步长为 1：
    切片 1：消息 1 + 2 + 3
    切片 2：消息 2 + 3 + 4
    切片 3：消息 3 + 4 + 5
    ...（以此类推）

  检索时找到最相关的切片后，再通过切片里记录的 message_ids，
  去 SQLite 里拿完整的消息内容，还原出带上下文的原始对话片段。

─────────────────────────────────────────────────────────────
与其他模块的关系
─────────────────────────────────────────────────────────────

  · 被写入：
      summarizer.py  → L1 写入后调用 index_l1()
      merger.py      → L2 写入后调用 index_l2()
      session_manager.py → session 关闭后调用 index_l0_session()

  · 被检索：
      prompt_builder.py → 构建 system prompt 时调用 retrieve_*() 系列函数

  · 读取配置：config.py
  · 读取消息数据（L0 用）：database.py

─────────────────────────────────────────────────────────────
嵌入模型切换说明
─────────────────────────────────────────────────────────────

  当前阶段（EMBEDDING_MODEL = "default"）：
    使用 Chroma 内置的 all-MiniLM-L6-v2，无需额外安装，中文效果一般。

  升级到 bge-m3（推荐）：
    1. pip install sentence-transformers
    2. 将 config.py 里 EMBEDDING_MODEL 改为 "BAAI/bge-m3"
    3. 调用 rebuild_all_indexes() 重建全部索引

  升级到 Qwen3-Embedding-0.6B：
    步骤同上，EMBEDDING_MODEL 改为 "Qwen/Qwen3-Embedding-0.6B"

使用方法：
    from vector_store import index_l1, index_l2, index_l0_session
    from vector_store import retrieve_l1, retrieve_l2, retrieve_l0
    from vector_store import retrieve_combined
"""

import os
import json

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


# =============================================================================
# 内部：Chroma 客户端与 Collection 管理
# =============================================================================

# 全局客户端单例：整个进程只初始化一次，避免重复打开索引文件
_client = None


def _get_client():
    """
    获取（或初始化）全局 Chroma 客户端。

    使用 PersistentClient，索引文件写入 CHROMA_DIR 目录，重启后不丢失。
    目录不存在时会自动创建。

    返回：
        chromadb.PersistentClient 实例
    """
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        print(f"[vector_store] Chroma 客户端已初始化，索引目录：{CHROMA_DIR}")
    return _client


def _get_embedding_function():
    """
    根据 config.py 的 EMBEDDING_MODEL 返回对应的嵌入函数。

    "default"  → Chroma 内置模型（all-MiniLM-L6-v2），无需额外安装
    其他字符串 → 用 SentenceTransformer 加载对应模型（需要先 pip install sentence-transformers）

    返回：
        chromadb embedding function 实例
    """
    if EMBEDDING_MODEL == "default":
        return DefaultEmbeddingFunction()

    # 非 default 时，用 SentenceTransformer 加载指定模型
    # 首次调用时会从 HuggingFace 自动下载模型文件（约 300MB~570MB）
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _get_collection(name):
    """
    获取（或创建）指定名称的 Collection。

    Collection 是 Chroma 里的"表"，每层记忆对应一个 Collection。
    get_or_create_collection：已存在则获取，不存在则新建，幂等操作。

    参数：
        name — Collection 名称，固定使用下方三个常量之一

    返回：
        chromadb.Collection 实例
    """
    client = _get_client()
    ef     = _get_embedding_function()
    return client.get_or_create_collection(
        name             = name,
        embedding_function = ef,
    )


# Collection 名称常量，统一定义避免拼写错误
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
      · 每个切片把窗口内的消息拼成一段纯文本（"用户：xxx\n助手：xxx\n..."）

    metadata 里存储的 message_ids 是一个 JSON 字符串（列表），
    检索到切片后可以通过它去 SQLite 里拿完整消息内容。

    参数：
        messages    — database.get_messages() 返回的消息列表（sqlite3.Row）
        session_id  — 对应的 session id，写入 metadata 供后续溯源
        window_size — 滑动窗口大小，默认取 config.L0_WINDOW_SIZE

    返回：
        list[dict] — 切片列表，每个元素包含：
            {
                "id":        切片唯一 ID，格式 "l0_{session_id}_{起始索引}"
                "document":  切片内所有消息拼成的纯文本
                "metadata":  {
                    "session_id":  session id（int）
                    "start_idx":   切片在 session 消息列表中的起始位置
                    "end_idx":     切片的结束位置（含）
                    "message_ids": 切片内所有消息 id 的 JSON 字符串，如 "[1, 2, 3]"
                }
            }

    示例（session 有 4 条消息，window_size=3）：
        切片 0：消息 0、1、2  → id = "l0_5_0"
        切片 1：消息 1、2、3  → id = "l0_5_1"
    """
    chunks = []

    if not messages:
        return chunks

    msg_list = list(messages)   # sqlite3.Row 列表转为普通列表，方便切片

    for start in range(len(msg_list) - window_size + 1):
        end     = start + window_size - 1
        window  = msg_list[start : end + 1]

        # 把窗口内的消息拼成纯文本，格式与 summarizer 的 _format_conversation 一致
        lines = []
        for msg in window:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}：{msg['content']}")
        document = "\n".join(lines)

        # 收集这个窗口内所有消息的 id，序列化为 JSON 字符串存入 metadata
        # Chroma 的 metadata 值只支持 str/int/float/bool，不支持列表，所以要 json.dumps
        message_ids = [msg["id"] for msg in window]

        chunks.append({
            "id":       f"l0_{session_id}_{start}",
            "document": document,
            "metadata": {
                "session_id":  session_id,
                "start_idx":   start,
                "end_idx":     end,
                "message_ids": json.dumps(message_ids),   # 列表 → JSON 字符串
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

    索引的文本内容 = 摘要文本。
    如果有关键词，会把关键词附加在摘要后面一起向量化，
    让关键词也参与语义检索，提高相关记忆的召回率。

    参数：
        l1_id      — memory_l1 表的主键 id（int），用作向量索引的唯一 ID
        summary    — L1 摘要文本
        keywords   — 关键词字符串，逗号分隔，如 "数据库,后端,验证"；可为 None
        session_id — 对应的 session id，写入 metadata 供溯源；可为 None

    返回：
        None（写入失败时打印错误，不抛出异常，不阻断主流程）
    """
    try:
        collection = _get_collection(_COLL_L1)

        # 拼接文档文本：摘要 + 关键词（如果有）
        # 把关键词也纳入向量化，让它们在语义空间里贡献权重
        if keywords:
            document = f"{summary} 关键词：{keywords}"
        else:
            document = summary

        # metadata 存 SQLite 里对应的 id，检索后可以用它去拿完整记录
        metadata = {"l1_id": l1_id}
        if session_id is not None:
            metadata["session_id"] = session_id
        if keywords:
            metadata["keywords"] = keywords

        # upsert：id 已存在则覆盖，不存在则新增
        # 避免重复写入（例如应用重启后重新触发索引时）
        collection.upsert(
            ids       = [str(l1_id)],   # Chroma 要求 id 为字符串
            documents = [document],
            metadatas = [metadata],
        )
        print(f"[vector_store] L1 索引写入成功，l1_id={l1_id}")

    except Exception as e:
        print(f"[vector_store] 警告：L1 索引写入失败，l1_id={l1_id} — {e}")


def index_l2(l2_id, summary, keywords=None, period_start=None, period_end=None):
    """
    将一条 L2 聚合摘要写入向量索引。
    由 merger.check_and_merge() 在 L2 写入 SQLite 后调用。

    参数：
        l2_id        — memory_l2 表的主键 id（int）
        summary      — L2 摘要文本
        keywords     — 关键词字符串；可为 None
        period_start — 覆盖时间段起点（ISO 8601 字符串）；可为 None
        period_end   — 覆盖时间段终点（ISO 8601 字符串）；可为 None

    返回：
        None（写入失败时打印错误，不抛出异常）
    """
    try:
        collection = _get_collection(_COLL_L2)

        if keywords:
            document = f"{summary} 关键词：{keywords}"
        else:
            document = summary

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
        print(f"[vector_store] L2 索引写入成功，l2_id={l2_id}")

    except Exception as e:
        print(f"[vector_store] 警告：L2 索引写入失败，l2_id={l2_id} — {e}")


def index_l0_session(session_id):
    """
    对一个 session 的所有原始消息做滑动窗口切片，批量写入 L0 向量索引。
    由 session_manager 在 session 关闭后调用（在 generate_l1_summary 之前或之后均可）。

    流程：
      1. 从 SQLite 读取该 session 的全部消息
      2. 消息数量少于窗口大小时，跳过（切不出有意义的片段）
      3. 生成滑动窗口切片列表
      4. 批量写入 L0 Collection

    参数：
        session_id — 需要建立 L0 索引的 session id（int）

    返回：
        int  — 成功写入的切片数量
        None — 跳过或失败时返回 None
    """
    try:
        messages = get_messages(session_id)

        if not messages:
            print(f"[vector_store] session {session_id} 无消息，跳过 L0 索引")
            return None

        if len(messages) < L0_WINDOW_SIZE:
            # 消息数量不足一个窗口，无法切片
            # 此时直接把所有消息拼成一条索引，保证这个 session 在 L0 里有记录
            print(f"[vector_store] session {session_id} 消息数({len(messages)})< 窗口大小({L0_WINDOW_SIZE})，使用整体索引")
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

        # 生成切片列表
        chunks = _make_l0_chunks(messages, session_id)

        if not chunks:
            return None

        # 批量写入 Chroma
        collection = _get_collection(_COLL_L0)
        collection.upsert(
            ids       = [c["id"]       for c in chunks],
            documents = [c["document"] for c in chunks],
            metadatas = [c["metadata"] for c in chunks],
        )

        print(f"[vector_store] L0 索引写入成功，session_id={session_id}，共 {len(chunks)} 个切片")
        return len(chunks)

    except Exception as e:
        print(f"[vector_store] 警告：L0 索引写入失败，session_id={session_id} — {e}")
        return None


# =============================================================================
# 对外接口：语义检索
# =============================================================================

def retrieve_l2(query_text, top_k=L2_RETRIEVE_TOP_K):
    """
    在 L2 索引中做语义检索，返回最相关的聚合摘要列表。
    用于话题定位：判断某个话题在哪个时间段出现过。

    参数：
        query_text — 查询文本，通常是当前对话的开场内容
        top_k      — 最多返回多少条，默认取 config.L2_RETRIEVE_TOP_K

    返回：
        list[dict] — 检索结果列表，每个元素包含：
            {
                "l2_id":    int，对应 SQLite memory_l2 表的 id
                "document": str，索引时存入的文本（摘要 + 关键词）
                "distance": float，语义距离（越小越相似）
                "metadata": dict，完整的 metadata 字典
            }
        超过 SIMILARITY_THRESHOLD 的结果会被过滤掉。
        Collection 为空或检索失败时返回空列表。
    """
    return _retrieve(_COLL_L2, query_text, top_k, id_field="l2_id")


def retrieve_l1(query_text, top_k=L1_RETRIEVE_TOP_K):
    """
    在 L1 索引中做语义检索，返回最相关的单次对话摘要列表。
    是 RAG 注入的主力层，精度最高。

    参数与返回值格式同 retrieve_l2，id_field 为 "l1_id"。
    """
    return _retrieve(_COLL_L1, query_text, top_k, id_field="l1_id")


def retrieve_l0(query_text, top_k=L0_RETRIEVE_TOP_K):
    """
    在 L0 索引中做语义检索，返回最相关的原始消息切片列表。
    用于回溯原始对话的具体细节和情感温度。

    返回值里的 metadata 包含 message_ids（JSON 字符串），
    调用方需要 json.loads(result["metadata"]["message_ids"]) 后
    再去 SQLite 里按 id 查完整消息内容。

    参数与返回值格式同 retrieve_l2，id_field 不适用（L0 用 session_id）。
    """
    return _retrieve(_COLL_L0, query_text, top_k, id_field=None)


def retrieve_combined(query_text):
    """
    同时在 L1 和 L2 索引中检索，合并结果后返回。
    用于 prompt_builder 的语义层注入：一次调用拿到两层的相关记忆。

    L2 结果放在前面（话题定位，宏观背景），L1 结果放在后面（具体摘要，细节精度）。
    各层内部按距离升序排列（最相关的在前）。

    参数：
        query_text — 查询文本

    返回：
        dict — {
            "l2": list[dict],   # L2 检索结果（格式同 retrieve_l2）
            "l1": list[dict],   # L1 检索结果（格式同 retrieve_l1）
        }
    """
    l2_results = retrieve_l2(query_text)
    l1_results = retrieve_l1(query_text)

    return {
        "l2": l2_results,
        "l1": l1_results,
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
      · 把 Chroma 返回的并行列表结构转换为更易用的字典列表

    参数：
        collection_name — Collection 名称
        query_text      — 查询文本
        top_k           — 最多返回条数
        id_field        — metadata 里存 SQLite id 的字段名；None 表示不提取

    返回：
        list[dict] — 检索结果列表；失败时返回空列表
    """
    try:
        collection = _get_collection(collection_name)

        # 检查 Collection 是否有内容，空 Collection 直接返回
        if collection.count() == 0:
            return []

        raw = collection.query(
            query_texts = [query_text],
            n_results   = min(top_k, collection.count()),  # 不能超过实际条数
        )

        # Chroma 返回的是并行列表结构，取第一个查询的结果
        # raw["ids"]       = [[id1, id2, ...]]
        # raw["documents"] = [[doc1, doc2, ...]]
        # raw["distances"] = [[dist1, dist2, ...]]
        # raw["metadatas"] = [[meta1, meta2, ...]]
        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]

        results = []
        for i, (doc_id, doc, dist, meta) in enumerate(
            zip(ids, documents, distances, metadatas)
        ):
            # 过滤：距离超过阈值的结果不可信，丢弃
            if dist > SIMILARITY_THRESHOLD:
                continue

            result = {
                "document": doc,
                "distance": dist,
                "metadata": meta,
            }

            # 如果指定了 id_field，把对应的 SQLite id 提到顶层，方便调用方使用
            if id_field and id_field in meta:
                result[id_field] = meta[id_field]

            results.append(result)

        return results

    except Exception as e:
        # Collection 为空时 Chroma 抛 InvalidCollectionException，这里统一捕获
        # 不打印 warning，空 Collection 是正常的冷启动状态
        if "does not exist" in str(e) or "InvalidCollection" in str(e):
            return []
        print(f"[vector_store] 警告：检索失败，collection={collection_name} — {e}")
        return []


# =============================================================================
# 对外接口：索引管理工具
# =============================================================================

def get_index_stats():
    """
    返回三个 Collection 当前的索引条数，用于监控和调试。

    返回：
        dict — {
            "l0": int,   # L0 切片数量
            "l1": int,   # L1 摘要数量
            "l2": int,   # L2 摘要数量
        }
        某个 Collection 不存在时对应值为 0。
    """
    stats = {"l0": 0, "l1": 0, "l2": 0}
    client = _get_client()

    for key, name in [("l0", _COLL_L0), ("l1", _COLL_L1), ("l2", _COLL_L2)]:
        try:
            coll = client.get_collection(name)
            stats[key] = coll.count()
        except Exception:
            # Collection 不存在，保持 0
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
        dict — {
            "l0_chunks": int,   # 重建的 L0 切片数
            "l1_count":  int,   # 重建的 L1 条数
            "l2_count":  int,   # 重建的 L2 条数
        }
    """
    from database import (
        get_unabsorbed_l1, get_recent_l2,
        _get_connection,
    )

    print("[vector_store] 开始重建全部向量索引…")
    client = _get_client()

    # 第一步：删除旧 Collection
    for name in [_COLL_L0, _COLL_L1, _COLL_L2]:
        try:
            client.delete_collection(name)
            print(f"[vector_store] 已删除旧 Collection：{name}")
        except Exception:
            pass   # Collection 不存在时直接跳过

    counts = {"l0_chunks": 0, "l1_count": 0, "l2_count": 0}

    # 第二步：重建 L1 索引
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, summary, keywords, session_id FROM memory_l1")
    l1_rows = cursor.fetchall()

    for row in l1_rows:
        index_l1(
            l1_id      = row["id"],
            summary    = row["summary"],
            keywords   = row["keywords"],
            session_id = row["session_id"],
        )
        counts["l1_count"] += 1

    # 第三步：重建 L2 索引
    cursor.execute("SELECT id, summary, keywords, period_start, period_end FROM memory_l2")
    l2_rows = cursor.fetchall()

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
    cursor.execute("SELECT DISTINCT session_id FROM messages")
    session_ids = [r["session_id"] for r in cursor.fetchall()]
    conn.close()

    for sid in session_ids:
        n = index_l0_session(sid)
        if n:
            counts["l0_chunks"] += n

    print(f"[vector_store] 索引重建完成：L1={counts['l1_count']} 条，"
          f"L2={counts['l2_count']} 条，L0={counts['l0_chunks']} 个切片")
    return counts


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    from database import (
        new_session, save_message, close_session,
        save_l1_summary, save_l2_summary,
    )

    print("=== vector_store.py 验证测试 ===\n")

    # ------------------------------------------------------------------
    # 测试一：初始化与索引状态查询
    # ------------------------------------------------------------------
    print("--- 测试一：初始化 ---")
    stats = get_index_stats()
    print(f"当前索引条数：L0={stats['l0']}，L1={stats['l1']}，L2={stats['l2']}")
    print()

    # ------------------------------------------------------------------
    # 测试二：写入 L1 索引并检索
    # ------------------------------------------------------------------
    print("--- 测试二：L1 索引写入与检索 ---")

    # 写入三条测试 L1
    sid1 = new_session()
    save_message(sid1, "user", "今天把 summarizer 写完了，验证全部通过")
    save_message(sid1, "assistant", "很棒，进度很稳！")
    close_session(sid1)
    l1_id_1 = save_l1_summary(sid1, "烧酒完成了 summarizer 模块的开发与验证。",
                               "summarizer,模块,验证", "夜间", "专注高效")
    index_l1(l1_id_1, "烧酒完成了 summarizer 模块的开发与验证。",
             "summarizer,模块,验证", session_id=sid1)

    sid2 = new_session()
    save_message(sid2, "user", "今天状态很差，头疼，什么都不想做")
    save_message(sid2, "assistant", "听起来很辛苦，好好休息一下。")
    close_session(sid2)
    l1_id_2 = save_l1_summary(sid2, "烧酒今天状态不佳，头疼，情绪低落。",
                               "状态,头疼,情绪", "下午", "情绪低落")
    index_l1(l1_id_2, "烧酒今天状态不佳，头疼，情绪低落。",
             "状态,头疼,情绪", session_id=sid2)

    sid3 = new_session()
    save_message(sid3, "user", "在想记忆系统的树状结构怎么设计")
    save_message(sid3, "assistant", "我们可以从关联键出发，逐层往下穿透。")
    close_session(sid3)
    l1_id_3 = save_l1_summary(sid3, "烧酒讨论了记忆系统的树状关联网络架构。",
                               "记忆系统,架构,设计", "上午", "轻松愉快")
    index_l1(l1_id_3, "烧酒讨论了记忆系统的树状关联网络架构。",
             "记忆系统,架构,设计", session_id=sid3)

    print(f"已写入 3 条 L1 索引（id={l1_id_1}, {l1_id_2}, {l1_id_3}）")

    # 检索测试
    queries = [
        "最近有没有熬夜写代码",
        "烧酒心情不好",
        "系统架构",
    ]
    for q in queries:
        results = retrieve_l1(q, top_k=1)
        if results:
            print(f'  查询："{q}"')
            print(f'  最相关：{results[0]["document"]}')
            print(f'  距离：{results[0]["distance"]:.4f}\n')
        else:
            print(f'  查询："{q}" → 无相关结果（距离超过阈值）\n')

    # ------------------------------------------------------------------
    # 测试三：写入 L0 索引并检索
    # ------------------------------------------------------------------
    print("--- 测试三：L0 索引写入与检索 ---")

    n_chunks = index_l0_session(sid1)
    print(f"session {sid1} 的 L0 索引：{n_chunks} 个切片")

    l0_results = retrieve_l0("把模块写完了")
    if l0_results:
        print(f"L0 检索结果（共 {len(l0_results)} 条）：")
        for r in l0_results:
            msg_ids = json.loads(r["metadata"]["message_ids"])
            print(f'  切片内容：{r["document"][:60]}...')
            print(f'  message_ids：{msg_ids}，距离：{r["distance"]:.4f}')
    else:
        print("L0 无相关结果")
    print()

    # ------------------------------------------------------------------
    # 测试四：combined 检索
    # ------------------------------------------------------------------
    print("--- 测试四：retrieve_combined ---")
    combined = retrieve_combined("最近在做什么项目")
    print(f"L2 结果：{len(combined['l2'])} 条，L1 结果：{len(combined['l1'])} 条")
    print()

    # ------------------------------------------------------------------
    # 测试五：索引统计
    # ------------------------------------------------------------------
    print("--- 测试五：索引统计 ---")
    stats_after = get_index_stats()
    print(f"写入后索引条数：L0={stats_after['l0']}，"
          f"L1={stats_after['l1']}，L2={stats_after['l2']}")

    print("\n验证完成。")
