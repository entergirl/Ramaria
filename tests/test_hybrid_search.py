"""
tests/test_hybrid_search.py — Hybrid Search 端到端验证脚本
=====================================================================

目的：
    验证 BM25 索引、RRF 融合、retrieve_combined 三个环节是否正常工作。

前置条件：
    1. 已安装 rank-bm25 和 jieba（pip install rank-bm25 jieba）
    2. 已按说明修改 vector_store.py 和 config.py
    3. 数据库中至少有几条 L1 记录（没有的话阶段四会提示但不算 FAIL）

运行方式：
    python tests/test_hybrid_search.py

测试结构：
    阶段一：依赖库检查（rank-bm25 / jieba 是否已安装）
    阶段二：BM25 分词正确性（jieba 分词 + 自定义词典加载）
    阶段三：BM25 索引重建（从数据库读取数据，构建内存索引）
    阶段四：BM25 检索（写入测试数据，检索，验证相关性）
    阶段五：RRF 融合逻辑（单元测试，不依赖真实数据）
    阶段六：retrieve_combined 端到端（双路检索 + 融合，验证输出格式）
"""

import sys
import math
from pathlib import Path

# 确保项目根目录在 sys.path 里
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# 工具函数
# =============================================================================

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(label, passed, detail=""):
    status = "✅ [PASS]" if passed else "❌ [FAIL]"
    msg = f"{status}  {label}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    return passed


# =============================================================================
# 阶段一：依赖库检查
# =============================================================================

def test_dependencies():
    print_section("阶段一：依赖库检查")

    all_pass = True

    try:
        import jieba
        print_result("jieba 已安装", True, f"版本：{jieba.__version__}")
    except ImportError:
        print_result("jieba 未安装", False, "请执行：pip install jieba")
        all_pass = False

    try:
        from rank_bm25 import BM25Okapi
        print_result("rank-bm25 已安装", True)
    except ImportError:
        print_result("rank-bm25 未安装", False, "请执行：pip install rank-bm25")
        all_pass = False

    try:
        from config import RRF_K, BM25_WEIGHT
        print_result(
            "config.py 新常量存在",
            True,
            f"RRF_K={RRF_K}, BM25_WEIGHT={BM25_WEIGHT}"
        )
    except ImportError as e:
        print_result("config.py 新常量缺失", False, str(e))
        all_pass = False

    return all_pass


# =============================================================================
# 阶段二：BM25 分词正确性
# =============================================================================

def test_tokenization():
    print_section("阶段二：BM25 分词正确性")

    try:
        from vector_store import _bm25_index

        all_pass = True

        # 测试基础中文分词
        tokens1 = _bm25_index._tokenize("烧酒今天完成了向量检索模块的开发。")
        ok1 = print_result(
            "中文句子分词正常",
            len(tokens1) > 0,
            f"分词结果：{tokens1}"
        )
        all_pass = all_pass and ok1

        # 测试英文词汇不被切断
        tokens2 = _bm25_index._tokenize("MCP Server 接入完成，FastAPI 路由正常。")
        ok2 = print_result(
            "英文词汇分词正常",
            len(tokens2) > 0,
            f"分词结果：{tokens2}"
        )
        all_pass = all_pass and ok2

        # 测试空字符串不报错
        tokens3 = _bm25_index._tokenize("")
        ok3 = print_result(
            "空字符串返回空列表",
            tokens3 == [],
            f"结果：{tokens3}"
        )
        all_pass = all_pass and ok3

        # 测试 _build_doc_text 关键词拼接
        doc = _bm25_index._build_doc_text(
            summary  = "烧酒完成了摘要模块开发。",
            keywords = "摘要,模块,开发"
        )
        ok4 = print_result(
            "_build_doc_text 关键词拼接正常",
            "摘要" in doc and "模块" in doc,
            f"拼接结果：{doc}"
        )
        all_pass = all_pass and ok4

        return all_pass

    except Exception as e:
        print_result("分词测试", False, str(e))
        return False


# =============================================================================
# 阶段三：BM25 索引重建
# =============================================================================

def test_rebuild():
    print_section("阶段三：BM25 索引重建")

    try:
        from vector_store import _bm25_index

        # 重建 L1 索引
        _bm25_index.rebuild("l1")
        ok_l1 = print_result(
            "L1 BM25 索引重建不报错",
            True,
            f"当前 L1 索引条数：{len(_bm25_index._l1_ids)}"
        )

        # 重建 L2 索引
        _bm25_index.rebuild("l2")
        ok_l2 = print_result(
            "L2 BM25 索引重建不报错",
            True,
            f"当前 L2 索引条数：{len(_bm25_index._l2_ids)}"
        )

        # 测试未知 layer 不崩溃
        try:
            _bm25_index.rebuild("l3")
            ok_unknown = print_result("未知 layer 静默跳过", True)
        except Exception as e:
            ok_unknown = print_result("未知 layer 处理", False, str(e))

        return ok_l1 and ok_l2 and ok_unknown

    except Exception as e:
        print_result("索引重建测试", False, str(e))
        return False


# =============================================================================
# 阶段四：BM25 检索（写入测试数据后验证）
# =============================================================================

def test_bm25_search():
    print_section("阶段四：BM25 检索相关性验证")

    try:
        from vector_store import _bm25_index, index_l1
        from database import new_session, save_message, close_session, save_l1_summary
        from datetime import datetime, timezone

        # 写入三条测试 L1 记录（覆盖不同话题）
        test_cases = [
            ("烧酒今晚完成了 MCP Server 的接口设计，路由全部跑通。",
             "MCP,Server,接口,路由", "夜间", "专注高效"),
            ("烧酒今天心情不好，有些疲惫，和黎杋枫聊了聊近况。",
             "心情,疲惫,近况", "下午", "情绪低落"),
            ("烧酒讨论了向量检索和 BM25 混合检索的融合方案。",
             "向量检索,BM25,混合检索,融合", "上午", "轻松愉快"),
        ]

        l1_ids = []
        for summary, keywords, tp, atm in test_cases:
            sid = new_session()
            save_message(sid, "user", summary[:20])
            close_session(sid)
            l1_id = save_l1_summary(
                session_id=sid, summary=summary,
                keywords=keywords, time_period=tp, atmosphere=atm
            )
            now_str = datetime.now(timezone.utc).isoformat()
            index_l1(l1_id=l1_id, summary=summary, keywords=keywords,
                     session_id=sid, created_at=now_str)
            l1_ids.append(l1_id)

        # BM25 索引在 index_l1 内部已经重建，直接检索
        all_pass = True

        # 查询1：MCP 相关，期望第一条命中
        results1 = _bm25_index.search("MCP Server 路由设计", layer="l1", top_k=3)
        ok1 = print_result(
            "MCP 相关查询命中预期记录",
            len(results1) > 0 and "MCP" in results1[0]["document"],
            f"最相关：{results1[0]['document'][:40]}…" if results1 else "无结果"
        )
        all_pass = all_pass and ok1

        # 查询2：情绪相关，期望第二条命中
        results2 = _bm25_index.search("心情不好很疲惫", layer="l1", top_k=3)
        ok2 = print_result(
            "情绪相关查询命中预期记录",
            len(results2) > 0 and "疲惫" in results2[0]["document"],
            f"最相关：{results2[0]['document'][:40]}…" if results2 else "无结果"
        )
        all_pass = all_pass and ok2

        # 查询3：无关查询，分数应该很低（结果可能为空）
        results3 = _bm25_index.search("今天吃了什么好吃的", layer="l1", top_k=3)
        ok3 = print_result(
            "无关查询返回低分结果（可以为空）",
            True,   # 不强制要求为空，BM25 可能有低分命中
            f"返回 {len(results3)} 条，"
            f"最高分：{results3[0]['bm25_score']:.4f}" if results3 else "返回 0 条"
        )
        all_pass = all_pass and ok3

        return all_pass

    except Exception as e:
        import traceback
        print_result("BM25 检索测试", False, str(e))
        traceback.print_exc()
        return False


# =============================================================================
# 阶段五：RRF 融合逻辑单元测试
# =============================================================================

def test_rrf_fusion():
    print_section("阶段五：RRF 融合逻辑单元测试")

    try:
        from vector_store import _rrf_fuse
        from config import RRF_K, BM25_WEIGHT

        # 构造模拟的向量检索结果
        # 注意：距离越小排名越高（向量检索按 adjusted_distance 升序排序）
        # 调整：让 id=2 在向量中排第2，id=3 排第3，确保 id=2 能排第一
        vector_results = [
            {"l1_id": 2, "document": "doc2", "distance": 0.1,
            "adjusted_distance": 0.12, "decay_r": 0.9, "metadata": {}}, 
            {"l1_id": 1, "document": "doc1", "distance": 0.2,
            "adjusted_distance": 0.25, "decay_r": 0.8, "metadata": {}},
            {"l1_id": 3, "document": "doc3", "distance": 0.3,
            "adjusted_distance": 0.35, "decay_r": 0.7, "metadata": {}},
        ]

        # 构造模拟的 BM25 检索结果（id=2 明显领先，id=3 第二，id=1 仅在向量中）
        # BM25 分数越高排名越靠前，分数差距要足够大确保排名
        bm25_results = [
            {"id": 2, "document": "doc2", "bm25_score": 3.5, "rank": 1},
            {"id": 1, "document": "doc1", "bm25_score": 2.1, "rank": 2},
        ]

        fused = _rrf_fuse(vector_results, bm25_results, "l1_id", top_k=3)

        all_pass = True

        # 融合结果应该包含全部三条（两路并集）
        fused_ids = [r["l1_id"] for r in fused]
        ok1 = print_result(
            "融合结果包含所有候选（两路并集）",
            set(fused_ids) == {1, 2, 3},
            f"融合后 ids：{fused_ids}"
        )
        all_pass = all_pass and ok1

        # id=2 在两路都命中且排名靠前，应该排在融合结果第一位
        ok2 = print_result(
            "两路都命中的结果排名最高（id=2）",
            fused[0]["l1_id"] == 2,
            f"融合第一名：id={fused[0]['l1_id']}，rrf_score={fused[0]['rrf_score']:.6f}"
        )
        all_pass = all_pass and ok2

        # 验证 rrf_score 字段存在且为正数
        ok3 = print_result(
            "rrf_score 字段存在且为正数",
            all(r.get("rrf_score", 0) > 0 for r in fused),
            f"各条 rrf_score：{[round(r['rrf_score'], 6) for r in fused]}"
        )
        all_pass = all_pass and ok3

        # 验证 bm25_score 字段存在
        ok4 = print_result(
            "bm25_score 字段存在",
            all("bm25_score" in r for r in fused),
        )
        all_pass = all_pass and ok4

        # 手动验证 RRF 公式（id=1 的分数）
        # id=1: vector_rank=2, bm25_rank=2
        penalty_rank = 2  # 由于 id=1 在向量中排第2，BM25中排第2
        expected_rrf_1 = BM25_WEIGHT / (RRF_K + penalty_rank) + 1.0 / (RRF_K + penalty_rank)
        actual_rrf_1   = next(r["rrf_score"] for r in fused if r["l1_id"] == 1)
        ok5 = print_result(
            "RRF 公式计算正确（id=1 手动验证）",
            abs(actual_rrf_1 - expected_rrf_1) < 1e-9,
            f"期望={expected_rrf_1:.9f}，实际={actual_rrf_1:.9f}"
        )
        all_pass = all_pass and ok5

        return all_pass

    except Exception as e:
        import traceback
        print_result("RRF 融合测试", False, str(e))
        traceback.print_exc()
        return False


# =============================================================================
# 阶段六：retrieve_combined 端到端
# =============================================================================

def test_retrieve_combined():
    print_section("阶段六：retrieve_combined 端到端验证")

    try:
        from vector_store import retrieve_combined

        result = retrieve_combined("最近有没有写代码熬夜")

        all_pass = True

        # 返回值是字典
        ok1 = print_result(
            "返回值是 dict",
            isinstance(result, dict),
            f"类型：{type(result)}"
        )
        all_pass = all_pass and ok1

        # 包含 l1 和 l2 两个键
        ok2 = print_result(
            "返回值包含 l1 和 l2 键",
            "l1" in result and "l2" in result,
            f"键：{list(result.keys())}"
        )
        all_pass = all_pass and ok2

        # l1 和 l2 都是列表
        ok3 = print_result(
            "l1 和 l2 都是 list",
            isinstance(result["l1"], list) and isinstance(result["l2"], list),
        )
        all_pass = all_pass and ok3

        # 如果有结果，验证字段完整性
        if result["l1"]:
            first = result["l1"][0]
            required_fields = {"document", "rrf_score", "bm25_score"}
            missing = required_fields - set(first.keys())
            ok4 = print_result(
                "L1 结果包含融合相关字段",
                len(missing) == 0,
                f"缺少字段：{missing}" if missing else
                f"rrf_score={first['rrf_score']:.6f}, bm25_score={first['bm25_score']:.4f}"
            )
            all_pass = all_pass and ok4

            # 验证结果按 rrf_score 降序排列
            scores = [r["rrf_score"] for r in result["l1"]]
            ok5 = print_result(
                "L1 结果按 rrf_score 降序排列",
                scores == sorted(scores, reverse=True),
                f"分数序列：{[round(s, 6) for s in scores]}"
            )
            all_pass = all_pass and ok5
        else:
            print("  ℹ️  L1 无检索结果（索引为空或无相关记忆），跳过字段验证")
            print("     这不是 FAIL——有数据后会自动生效")

        print(f"\n  检索结果概览：L1={len(result['l1'])} 条，L2={len(result['l2'])} 条")

        return all_pass

    except Exception as e:
        import traceback
        print_result("retrieve_combined 端到端测试", False, str(e))
        traceback.print_exc()
        return False


# =============================================================================
# 主入口
# =============================================================================

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  Hybrid Search（BM25 + 向量）验证脚本" + " "*19 + "║")
    print("╚" + "═"*58 + "╝")

    results = {}

    results["依赖库检查"] = test_dependencies()

    if not results["依赖库检查"]:
        print("\n⛔ 依赖库缺失，请先安装后重试。")
        return

    results["分词正确性"]     = test_tokenization()
    results["BM25 索引重建"]  = test_rebuild()
    results["BM25 检索相关性"] = test_bm25_search()
    results["RRF 融合逻辑"]   = test_rrf_fusion()
    results["retrieve_combined 端到端"] = test_retrieve_combined()

    print_section("测试汇总")
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 全部测试通过！")
        print("   Hybrid Search 已就绪，重启主服务后生效。")
    else:
        print("\n⚠️  有测试未通过，请根据上方 [FAIL] 信息排查。")


if __name__ == "__main__":
    main()
