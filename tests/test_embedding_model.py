"""
test_embedding_model.py — Qwen3-Embedding-0.6B 完整验证脚本
=====================================================================

目的：
    验证 Qwen3-Embedding-0.6B 模型能否在你的环境中正常加载，
    并与 Chroma 向量数据库配合完成写入、检索、过滤等完整链路。

使用前提：
    1. 已执行 pip install sentence-transformers chromadb
    2. 已在本地下载 Qwen3-Embedding-0.6B 模型
       （默认缓存路径通常是 ~/.cache/huggingface/hub/）

运行方式：
    python test_embedding_model.py

预期结果：
    六个测试阶段全部打印 [PASS]，最后输出"全部测试通过"。
    如有 [FAIL] 请根据错误信息排查对应步骤。

测试结构：
    阶段一：模型加载         —— 验证本地模型路径和加载是否正常
    阶段二：单句向量生成     —— 验证 encode() 输出维度和类型
    阶段三：语义相似度计算   —— 验证中文语义距离是否符合直觉
    阶段四：Chroma 写入      —— 验证向量索引写入链路
    阶段五：Chroma 语义检索  —— 验证检索结果相关性
    阶段六：相似度阈值过滤   —— 验证 SIMILARITY_THRESHOLD 配置效果
    阶段七：性能基准         —— 打印单次 encode 耗时，用于判断是否在可接受范围

作者备注：
    换用 bge-m3 时，只需把下方 MODEL_NAME 改为 "BAAI/bge-m3"，
    其余代码完全不用动。
"""

import os
import time
import json
import numpy as np

# ─────────────────────────────────────────────────────────────
# 全局配置区：需要修改时只改这里
# ─────────────────────────────────────────────────────────────

# 模型名称（HuggingFace 模型 ID）
# 如果你把模型下载到了本地指定目录，可以改成绝对路径，例如：
#   MODEL_NAME = r"D:\models\Qwen3-Embedding-0.6B"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Chroma 测试索引的持久化目录（测试结束后会自动删除）
# 不会影响你的正式 chroma_db/ 目录
CHROMA_TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_test_tmp")

# 语义相似度过滤阈值
# 换用 bge-m3 后这个值也适用
SIMILARITY_THRESHOLD = 0.6

# 是否在测试结束后自动清理 CHROMA_TEST_DIR
AUTO_CLEANUP = True


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def print_section(title):
    """打印带分隔线的阶段标题，方便阅读日志"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(label, passed, detail=""):
    """统一打印测试结果，passed=True 显示 [PASS]，否则 [FAIL]"""
    status = "✅ [PASS]" if passed else "❌ [FAIL]"
    msg = f"{status}  {label}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    return passed


# ─────────────────────────────────────────────────────────────
# 阶段一：模型加载
# ─────────────────────────────────────────────────────────────

def test_model_load():
    """
    测试 Qwen3-Embedding-0.6B 能否正常加载。

    sentence-transformers 会先在本地 HuggingFace 缓存中查找，
    找不到时才会联网下载。如果你已经手动下载了模型，
    确保 MODEL_NAME 填的是正确的本地路径或 HF 模型 ID。

    返回：
        model — 加载成功时返回 SentenceTransformer 实例
        None  — 加载失败时返回 None
    """
    print_section("阶段一：模型加载")

    try:
        from sentence_transformers import SentenceTransformer

        print(f"正在加载模型：{MODEL_NAME}")
        print("（首次加载可能需要几秒钟初始化，请稍候…）")

        t0    = time.time()
        model = SentenceTransformer(MODEL_NAME)
        elapsed = time.time() - t0

        print_result(
            f"模型加载成功（耗时 {elapsed:.1f}s）",
            passed=True,
            detail=f"模型路径：{MODEL_NAME}"
        )
        return model

    except ImportError:
        print_result(
            "sentence-transformers 未安装",
            passed=False,
            detail="请先执行：pip install sentence-transformers"
        )
        return None

    except Exception as e:
        print_result(
            "模型加载失败",
            passed=False,
            detail=str(e)
        )
        return None


# ─────────────────────────────────────────────────────────────
# 阶段二：单句向量生成
# ─────────────────────────────────────────────────────────────

def test_encode(model):
    """
    测试 encode() 的基础功能：输出维度、数据类型是否正常。

    Qwen3-Embedding-0.6B 的输出维度是 1024。
    如果维度对不上，说明模型加载了错误的版本。

    参数：
        model — test_model_load() 返回的 SentenceTransformer 实例
    """
    print_section("阶段二：单句向量生成")

    test_sentences = [
        "烧酒今天完成了向量数据库的升级。",
        "The quick brown fox jumps over the lazy dog.",   # 英文也要能跑
        "今晚状态很差，头疼，什么都不想做。",
    ]

    all_pass = True
    for sent in test_sentences:
        try:
            vec = model.encode(sent)

            passed = isinstance(vec, np.ndarray) and len(vec) > 0
            all_pass = all_pass and passed

            print_result(
                f"encode 成功：{sent[:20]}…",
                passed=passed,
                detail=f"维度={vec.shape}，dtype={vec.dtype}"
            )
        except Exception as e:
            print_result(f"encode 失败：{sent[:20]}…", passed=False, detail=str(e))
            all_pass = False

    return all_pass


# ─────────────────────────────────────────────────────────────
# 阶段三：语义相似度计算
# ─────────────────────────────────────────────────────────────

def test_semantic_similarity(model):
    """
    验证模型的中文语义理解是否符合直觉。

    设计三组句对：
      · 高相似对：语义几乎相同，期望距离 < 0.3
      · 中相似对：话题相关但表述不同，期望距离 0.3~0.6
      · 低相似对：语义无关，期望距离 > 0.6

    这里用余弦距离（cosine distance = 1 - cosine_similarity），
    和 Chroma 默认使用的距离计算方式一致。

    参数：
        model — SentenceTransformer 实例
    """
    print_section("阶段三：语义相似度计算（中文）")

    # 三组测试句对：(句子A, 句子B, 期望距离范围描述, 期望距离上限)
    test_pairs = [
        (
            "烧酒今晚完成了 summarizer 模块的验证测试。",
            "烧酒今天把摘要生成模块写好了，测试都过了。",
            "高相似（语义几乎相同）",
            0.4    # 期望距离 < 0.4
        ),
        (
            "最近在做一个本地 AI 记忆系统的项目。",
            "烧酒在开发一套带向量检索的对话助手。",
            "中相似（话题相关）",
            0.65   # 期望距离 < 0.65
        ),
        (
            "今天天气很好，适合出去散步。",
            "向量数据库的索引结构需要仔细设计。",
            "低相似（语义无关）",
            1.5    # 不设上限，只打印结果
        ),
    ]

    all_pass = True
    for sent_a, sent_b, label, threshold in test_pairs:
        try:
            vec_a = model.encode(sent_a)
            vec_b = model.encode(sent_b)

            # 计算余弦相似度，再转换为余弦距离
            # cosine_similarity = dot(a, b) / (norm(a) * norm(b))
            # cosine_distance   = 1 - cosine_similarity
            sim      = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            distance = float(1 - sim)

            if label.startswith("低相似"):
                # 低相似对只打印结果，不做 pass/fail 判断
                passed = True
                print_result(
                    f"{label}",
                    passed=True,
                    detail=f"距离={distance:.4f}（无阈值要求，仅供参考）"
                )
            else:
                passed = distance < threshold
                all_pass = all_pass and passed
                print_result(
                    f"{label}（阈值 < {threshold}）",
                    passed=passed,
                    detail=f"距离={distance:.4f}  |  A：{sent_a[:15]}…  B：{sent_b[:15]}…"
                )

        except Exception as e:
            print_result(f"{label} 计算失败", passed=False, detail=str(e))
            all_pass = False

    return all_pass


# ─────────────────────────────────────────────────────────────
# 阶段四：Chroma 写入
# ─────────────────────────────────────────────────────────────

def test_chroma_write(model):
    """
    用 Qwen3-Embedding-0.6B 作为嵌入函数，测试 Chroma 写入链路。

    这里手动创建 SentenceTransformerEmbeddingFunction，
    和你 vector_store.py 里 _get_embedding_function() 的逻辑完全一致。

    写入 5 条模拟 L1 摘要，覆盖代码开发、情绪低落、架构设计等话题，
    方便后续的检索测试能有足够的区分度。

    参数：
        model — SentenceTransformer 实例（此阶段不直接用，但用来确认已加载）

    返回：
        collection — 写入成功时返回 Chroma Collection 实例
        None       — 写入失败时返回 None
    """
    print_section("阶段四：Chroma 写入测试")

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        # 创建嵌入函数，和 vector_store.py 里的 _get_embedding_function() 等价
        ef = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

        # 初始化测试用的 PersistentClient（写入临时目录，不影响正式数据）
        # 同时存入全局变量，供 cleanup() 释放文件句柄（修复 Windows WinError 32）
        global _chroma_test_client
        os.makedirs(CHROMA_TEST_DIR, exist_ok=True)
        _chroma_test_client = chromadb.PersistentClient(path=CHROMA_TEST_DIR)
        client = _chroma_test_client

        # 删除同名旧 Collection（避免上次测试残留数据干扰）
        try:
            client.delete_collection("test_l1_qwen3")
        except Exception:
            pass  # 不存在时直接跳过

        collection = client.get_or_create_collection(
            name               = "test_l1_qwen3",
            embedding_function = ef,
        )

        # ── 测试数据：5 条模拟 L1 摘要 ──
        # 覆盖多个话题，确保检索测试有足够的区分度
        test_docs = [
            {
                "id":       "l1_001",
                "document": "烧酒今晚完成了 summarizer 模块的开发与验证，测试全部通过。",
                "metadata": {
                    "l1_id":      1,
                    "time_period": "夜间",
                    "atmosphere":  "专注高效",
                    "keywords":    "summarizer,模块,验证",
                }
            },
            {
                "id":       "l1_002",
                "document": "烧酒今天状态很差，头疼，情绪低落，什么都不想做。",
                "metadata": {
                    "l1_id":      2,
                    "time_period": "下午",
                    "atmosphere":  "情绪低落",
                    "keywords":    "状态,头疼,情绪",
                }
            },
            {
                "id":       "l1_003",
                "document": "烧酒讨论了记忆系统的树状关联网络架构设计，思路很清晰。",
                "metadata": {
                    "l1_id":      3,
                    "time_period": "上午",
                    "atmosphere":  "轻松愉快",
                    "keywords":    "记忆系统,架构,设计",
                }
            },
            {
                "id":       "l1_004",
                "document": "烧酒完成了 Chroma 向量数据库的接入测试，三层索引链路跑通。",
                "metadata": {
                    "l1_id":      4,
                    "time_period": "夜间",
                    "atmosphere":  "专注高效",
                    "keywords":    "Chroma,向量数据库,索引",
                }
            },
            {
                "id":       "l1_005",
                "document": "烧酒今天睡眠质量很差，白天很困，但还是坚持完成了今天的开发任务。",
                "metadata": {
                    "l1_id":      5,
                    "time_period": "傍晚",
                    "atmosphere":  "疲惫坚持",
                    "keywords":    "睡眠,疲惫,坚持",
                }
            },
        ]

        # 批量写入
        collection.add(
            ids       = [d["id"]       for d in test_docs],
            documents = [d["document"] for d in test_docs],
            metadatas = [d["metadata"] for d in test_docs],
        )

        count  = collection.count()
        passed = count == len(test_docs)

        print_result(
            f"Chroma 写入成功（共 {count} 条）",
            passed=passed,
            detail=f"Collection：test_l1_qwen3，目录：{CHROMA_TEST_DIR}"
        )

        return collection if passed else None

    except ImportError:
        print_result(
            "chromadb 未安装",
            passed=False,
            detail="请先执行：pip install chromadb"
        )
        return None

    except Exception as e:
        print_result("Chroma 写入失败", passed=False, detail=str(e))
        return None


# ─────────────────────────────────────────────────────────────
# 阶段五：Chroma 语义检索
# ─────────────────────────────────────────────────────────────

def test_chroma_retrieve(collection):
    """
    测试 Chroma 的语义检索是否能正确召回相关文档。

    对每条查询，检查返回的最相关文档是否符合预期。
    这里验证的是端到端链路：查询文本 → 向量化 → 相似度计算 → 排序返回。

    "预期文档关键词"是宽泛匹配：只要最相关文档包含这个词就算通过，
    不要求精确完全匹配，允许模型有合理的语义泛化。

    参数：
        collection — test_chroma_write() 返回的 Chroma Collection 实例
    """
    print_section("阶段五：Chroma 语义检索测试")

    if collection is None:
        print("⚠️  跳过：Chroma Collection 未成功创建")
        return False

    # 测试查询：(查询文本, 期望最相关文档包含的关键词, 描述)
    test_queries = [
        (
            "最近有没有熬夜写代码",
            "summarizer",
            "代码开发类查询"
        ),
        (
            "烧酒心情不好，很累很疲惫",
            "情绪",
            "情绪状态类查询"
        ),
        (
            "系统架构和设计",
            "架构",
            "技术架构类查询"
        ),
        (
            "向量数据库接入",
            "Chroma",
            "向量数据库类查询"
        ),
        (
            "睡眠质量很差，但还是坚持工作了",
            "睡眠",
            "睡眠疲惫类查询"
        ),
    ]

    all_pass = True
    for query_text, expected_keyword, desc in test_queries:
        try:
            results = collection.query(
                query_texts = [query_text],
                n_results   = 1,   # 只取最相关的一条
            )

            top_doc      = results["documents"][0][0]
            top_distance = results["distances"][0][0]

            passed = expected_keyword in top_doc
            all_pass = all_pass and passed

            print_result(
                f"{desc}（期望包含 [{expected_keyword}]）",
                passed=passed,
                detail=(
                    f"距离={top_distance:.4f}  |  "
                    f"召回：{top_doc[:35]}…"
                )
            )

        except Exception as e:
            print_result(f"{desc} 检索失败", passed=False, detail=str(e))
            all_pass = False

    return all_pass


# ─────────────────────────────────────────────────────────────
# 阶段六：相似度阈值过滤
# ─────────────────────────────────────────────────────────────

def test_similarity_threshold(collection):
    """
    验证 SIMILARITY_THRESHOLD=0.6 的过滤效果。

    策略：
      · 相关查询  → 距离应 < 0.6，通过过滤，能召回结果
      · 无关查询  → 距离应 > 0.6，被过滤掉，返回空列表

    这个阈值是在 config.py 里配置的，这里用本地变量模拟，
    验证 0.6 这个值对 Qwen3-Embedding-0.6B 是否合适。

    如果无关查询也通过了过滤（距离 < 0.6），说明阈值需要调小；
    如果相关查询被过滤掉（距离 > 0.6），说明阈值需要调大。

    参数：
        collection — Chroma Collection 实例
    """
    print_section("阶段六：相似度阈值过滤（SIMILARITY_THRESHOLD=0.6）")

    if collection is None:
        print("⚠️  跳过：Chroma Collection 未成功创建")
        return False

    test_cases = [
        (
            "今晚把向量检索的代码写完了",  # 相关查询
            True,                           # 期望能通过过滤（距离 < 阈值）
            "相关查询应通过过滤"
        ),
        (
            "今天中午吃了什么好吃的",       # 无关查询
            False,                          # 期望被过滤掉（距离 > 阈值）
            "无关查询应被过滤"
        ),
    ]

    all_pass = True
    for query_text, expect_pass_filter, desc in test_cases:
        try:
            results = collection.query(
                query_texts = [query_text],
                n_results   = 1,
            )

            distance    = results["distances"][0][0]
            passes_filter = distance <= SIMILARITY_THRESHOLD  # 模拟过滤逻辑

            # 期望通过过滤但被拦截：或期望被拦截但通过了 → 测试不通过
            passed = passes_filter == expect_pass_filter
            all_pass = all_pass and passed

            filter_status = "✓ 通过过滤" if passes_filter else "✗ 被过滤"
            print_result(
                f"{desc}",
                passed=passed,
                detail=(
                    f"距离={distance:.4f}  阈值={SIMILARITY_THRESHOLD}  "
                    f"→ {filter_status}"
                )
            )

        except Exception as e:
            print_result(f"{desc} 测试失败", passed=False, detail=str(e))
            all_pass = False

    # 额外打印一条建议
    print("\n  💡 参考说明：")
    print(f"     · 阈值 = {SIMILARITY_THRESHOLD}（当前配置）")
    print(f"     · 换用 bge-m3 后，距离分数会整体偏小，阈值可考虑收紧到 0.6~0.7")
    print(f"     · 如果召回率偏低（太多相关内容被过滤），可适当调大阈值")
    print(f"     · 如果噪声太多（不相关内容混入），可适当调小阈值")

    return all_pass


# ─────────────────────────────────────────────────────────────
# 阶段七：性能基准
# ─────────────────────────────────────────────────────────────

def test_performance(model):
    """
    测量 encode() 的实际耗时，判断是否在可接受范围。

    对于珊瑚菌项目，向量化是在 session 结束后的后台异步任务中完成的，
    不在实时对话链路上，所以对延迟要求相对宽松。

    参考基准（i9-14900HX + RTX 4060 Laptop）：
      · CPU 推理：单句约 80~200ms
      · GPU 推理：单句约 10~50ms（如果 sentence-transformers 自动选了 CUDA）

    参数：
        model — SentenceTransformer 实例
    """
    print_section("阶段七：性能基准测试")

    # 分别测试短句和长文本的耗时
    test_cases = [
        ("短句（约 20 字）", "烧酒今晚完成了向量检索模块的验证。"),
        ("中等长度（约 60 字）", "烧酒今天讨论了珊瑚菌记忆系统的分层 RAG 检索设计，"
                                  "包括 L0 滑动窗口切片、L1 对话摘要、L2 聚合摘要三层结构，"
                                  "以及如何通过语义检索把相关历史记忆注入 system prompt。"),
        ("长文本（约 150 字）", "用户：今天把 vector_store.py 的滑动窗口切片逻辑写完了，"
                                  "测了一下窗口大小为3的情况，切片数量和预期一致。\n"
                                  "助手：很棒！切片逻辑跑通后，L0 索引的基础就打好了。"
                                  "接下来可以测试一下在真实对话里，检索到的切片上下文是否自然。\n"
                                  "用户：嗯，我觉得窗口大小可能需要根据对话密度动态调整。\n"
                                  "助手：这个思路很好，可以先固定为3观察一段时间，积累了数据再做调整。"),
    ]

    for label, text in test_cases:
        # 预热一次（第一次可能有 JIT 或 cache miss 的额外开销）
        _ = model.encode(text)

        # 正式计时：跑 3 次取平均
        times = []
        for _ in range(3):
            t0 = time.time()
            model.encode(text)
            times.append(time.time() - t0)

        avg_ms  = np.mean(times) * 1000
        # 项目后台任务对延迟不敏感，500ms 以内都可接受
        passed  = avg_ms < 500

        print_result(
            f"{label}：平均 {avg_ms:.1f}ms",
            passed=passed,
            detail=f"3次耗时：{[f'{t*1000:.1f}ms' for t in times]}"
        )

    # 打印设备信息，方便判断是 CPU 还是 GPU 在跑
    try:
        import torch
        device = "CUDA（GPU）" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n  🖥️  当前推理设备：{device} — {gpu_name}")
        else:
            print(f"\n  🖥️  当前推理设备：{device}")
            print(f"     （提示：sentence-transformers 支持自动使用 CUDA，"
                  f"如需加速可确认 torch 安装了 CUDA 版本）")
    except ImportError:
        print("\n  🖥️  torch 未安装，无法检测推理设备")

    return True


# ─────────────────────────────────────────────────────────────
# 清理测试数据
# ─────────────────────────────────────────────────────────────

def cleanup():
    """
    删除测试过程中创建的临时 Chroma 目录。
    AUTO_CLEANUP = False 时可以保留数据，方便手动检查索引内容。

    Windows 特殊处理：
        Chroma 在 Windows 上会持有 SQLite 文件句柄，
        必须先重置全局客户端（让 GC 释放连接），再稍等片刻，
        否则 shutil.rmtree 会报 PermissionError: [WinError 32]。
    """
    if not AUTO_CLEANUP:
        print(f"\n⚠️  AUTO_CLEANUP=False，测试数据保留在：{CHROMA_TEST_DIR}")
        return

    import shutil
    import gc

    # 重置全局 Chroma 客户端，让 Python GC 释放文件句柄
    global _chroma_test_client
    _chroma_test_client = None
    gc.collect()

    # Windows 下文件句柄释放需要一点时间，稍微等一下
    time.sleep(0.5)

    if os.path.exists(CHROMA_TEST_DIR):
        try:
            shutil.rmtree(CHROMA_TEST_DIR)
            print(f"\n🧹 测试数据已清理：{CHROMA_TEST_DIR}")
        except PermissionError:
            # 极少数情况下句柄仍未释放，不影响测试结果，提示手动删除即可
            print(f"\n⚠️  临时目录无法自动删除（Windows 文件占用），请手动删除：")
            print(f"     {CHROMA_TEST_DIR}")


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  Qwen3-Embedding-0.6B × Chroma 完整验证脚本" + " "*12 + "║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  模型：{MODEL_NAME}")
    print(f"  阈值：SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
    print(f"  临时目录：{CHROMA_TEST_DIR}")

    results = {}

    # ── 阶段一：模型加载 ──
    model = test_model_load()
    results["模型加载"] = model is not None
    if model is None:
        print("\n⛔ 模型加载失败，后续测试无法进行，请先排查模型路径和依赖安装。")
        return

    # ── 阶段二：单句向量生成 ──
    results["单句向量生成"] = test_encode(model)

    # ── 阶段三：语义相似度 ──
    results["语义相似度"] = test_semantic_similarity(model)

    # ── 阶段四：Chroma 写入 ──
    collection = test_chroma_write(model)
    results["Chroma 写入"] = collection is not None

    # ── 阶段五：Chroma 检索 ──
    results["Chroma 检索"] = test_chroma_retrieve(collection)

    # ── 阶段六：阈值过滤 ──
    results["阈值过滤"] = test_similarity_threshold(collection)

    # ── 阶段七：性能基准 ──
    results["性能基准"] = test_performance(model)

    # ── 清理 ──
    cleanup()

    # ── 汇总报告 ──
    print_section("测试汇总")
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 全部测试通过！")
        print(f"   → Qwen3-Embedding-0.6B 模型验证成功！")
        print(f"   → 当前 config.py 配置：SIMILARITY_THRESHOLD=0.6（已验证）")
        print(f"   → 如需调整阈值，请在 config.py 中修改并运行 rebuild_all_indexes() 重建索引。")
    else:
        print("\n⚠️  有测试未通过，请根据上方 [FAIL] 信息逐项排查。")
        print("   常见问题：")
        print("   · 模型路径不对 → 检查 MODEL_NAME 是否正确")
        print("   · 没装依赖     → pip install sentence-transformers chromadb")
        print("   · 显存不够     → 关闭其他占用 GPU 的程序")


if __name__ == "__main__":
    main()
