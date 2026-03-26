"""
test_chroma.py — Chroma 连通性测试脚本
目的：验证 Chroma 安装是否正常，读写链路是否跑通。
使用 Chroma 内置的默认嵌入模型（无需额外安装 sentence-transformers）。

运行方式：
    python test_chroma.py

预期输出：
    [1] 客户端初始化成功
    [2] Collection 创建成功
    [3] 写入 3 条测试文档
    [4] 语义检索结果：
        - 查询："最近有没有熬夜写代码"
          最相关：烧酒今晚完成了 summarizer 模块的验证测试。（距离：x.xx）
        ...
    [5] metadata 过滤检索结果：
        ...
    [6] 清理测试数据完成
    连通性测试全部通过 ✓
"""

import os
import chromadb

# =============================================================================
# 配置：测试用的 Chroma 持久化目录
# 放在项目根目录下的 chroma_db/ 文件夹，和 assistant.db 同级
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


def run_test():
    print("=== Chroma 连通性测试 ===\n")

    # ------------------------------------------------------------------
    # [1] 初始化持久化客户端
    # PersistentClient 会把索引文件写入磁盘，重启后数据不丢失
    # ------------------------------------------------------------------
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    print(f"[1] 客户端初始化成功，索引目录：{CHROMA_DIR}")

    # ------------------------------------------------------------------
    # [2] 创建测试 Collection
    # get_or_create_collection：已存在则获取，不存在则新建
    # 不传 embedding_function 时使用 Chroma 内置默认模型
    # ------------------------------------------------------------------
    collection = client.get_or_create_collection(
        name="test_memory_l1",
        metadata={"description": "连通性测试用，可安全删除"}
    )
    print(f"[2] Collection 创建成功：{collection.name}")

    # ------------------------------------------------------------------
    # [3] 写入测试文档
    # ids      → 唯一标识符，对应 SQLite 里的 L1 id
    # documents → 原始文本，Chroma 会自动生成向量
    # metadatas → 附加字段，用于后续过滤
    # ------------------------------------------------------------------
    test_docs = [
        {
            "id":       "l1_test_001",
            "document": "烧酒今晚完成了 summarizer 模块的验证测试。",
            "metadata": {"time_period": "夜间", "atmosphere": "专注高效", "l1_id": 1}
        },
        {
            "id":       "l1_test_002",
            "document": "烧酒今天情绪有点低落，聊了一些最近的压力和疲惫感。",
            "metadata": {"time_period": "下午", "atmosphere": "情绪低落", "l1_id": 2}
        },
        {
            "id":       "l1_test_003",
            "document": "烧酒讨论了记忆系统的树状关联网络架构设计，思路清晰。",
            "metadata": {"time_period": "上午", "atmosphere": "轻松愉快", "l1_id": 3}
        },
    ]

    collection.add(
        ids       = [d["id"]       for d in test_docs],
        documents = [d["document"] for d in test_docs],
        metadatas = [d["metadata"] for d in test_docs],
    )
    print(f"[3] 写入 {len(test_docs)} 条测试文档")

    # ------------------------------------------------------------------
    # [4] 基础语义检索
    # query_texts → 查询文本，Chroma 会把它转成向量再做相似度搜索
    # n_results   → 返回最相似的 N 条
    # distances   → 返回值里包含距离分数（越小越相似）
    # ------------------------------------------------------------------
    print("\n[4] 语义检索结果：")

    queries = [
        "最近有没有熬夜写代码",
        "心情不好，压力很大",
        "系统架构设计",
    ]

    for q in queries:
        results = collection.query(
            query_texts = [q],
            n_results   = 1,
        )
        top_doc      = results["documents"][0][0]
        top_distance = results["distances"][0][0]
        print(f'  查询："{q}"')
        print(f'  最相关：{top_doc}')
        print(f'  距离分数：{top_distance:.4f}（越小越相似）\n')

    # ------------------------------------------------------------------
    # [5] metadata 过滤检索
    # where 参数支持对 metadata 字段做精确过滤
    # 例如只在"夜间"的记录里做语义检索
    # ------------------------------------------------------------------
    print("[5] metadata 过滤检索（只查 time_period=夜间 的记录）：")

    filtered = collection.query(
        query_texts = ["写代码"],
        n_results   = 1,
        where       = {"time_period": "夜间"},
    )
    if filtered["documents"][0]:
        print(f'  结果：{filtered["documents"][0][0]}')
        print(f'  距离分数：{filtered["distances"][0][0]:.4f}')
    else:
        print("  无结果")

    # ------------------------------------------------------------------
    # [6] 清理测试数据
    # 测试完成后删除这个 Collection，不污染正式数据
    # ------------------------------------------------------------------
    client.delete_collection("test_memory_l1")
    print("\n[6] 清理测试数据完成")

    print("\n连通性测试全部通过 ✓")
    print(f"Chroma 索引目录已创建：{CHROMA_DIR}")
    print("可以开始写 vector_store.py 了。")


if __name__ == "__main__":
    run_test()
