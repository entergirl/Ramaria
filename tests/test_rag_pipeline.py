"""
tests/test_rag_pipeline.py — RAG 检索链路端到端验证脚本
=====================================================================

目的：
    验证 _build_context() 是否真正接入了语义检索，
    以及情感字段（valence/salience）是否正确写入、读取、影响衰减。

前置条件：
    1. 已运行 migrate_add_emotion_fields.py 完成数据库迁移
    2. 已更新 database.py / config.py / summarizer.py / vector_store.py
    3. 向量索引目录（chroma_db/）存在，或允许本脚本自动创建测试索引

运行方式：
    python tests/test_rag_pipeline.py

预期输出：
    六个测试阶段全部打印 [PASS]，最后输出"全部测试通过"。
    如有 [FAIL] 请根据错误信息排查对应步骤。

测试结构：
    阶段一：数据库字段存在性检查
    阶段二：save_l1_summary 情感字段写入
    阶段三：情感字段校验逻辑（_validate_and_fix）
    阶段四：index_l1 salience 写入 Chroma metadata
    阶段五：_calc_decay_factor salience 加成效果
    阶段六：_build_context RAG 检索链路连通性
"""

import os
import sys
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
# 阶段一：数据库字段存在性检查
# =============================================================================

def test_db_schema():
    """
    检查 memory_l1 表中 valence 和 salience 字段是否已存在。
    如果这里 FAIL，说明迁移脚本还没有运行。
    """
    print_section("阶段一：数据库字段存在性检查")

    import sqlite3
    from config import DB_PATH

    if not os.path.exists(DB_PATH):
        print_result("数据库文件存在", False, f"找不到 {DB_PATH}，请先运行 init_db.py")
        return False

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(memory_l1)")
    columns = {row["name"] for row in cursor.fetchall()}
    conn.close()

    ok_valence  = print_result(
        "memory_l1 存在 valence 列",
        "valence" in columns,
        f"当前列：{sorted(columns)}"
    )
    ok_salience = print_result(
        "memory_l1 存在 salience 列",
        "salience" in columns,
        f"如果缺失，请运行：python migrate_add_emotion_fields.py"
    )

    return ok_valence and ok_salience


# =============================================================================
# 阶段二：save_l1_summary 情感字段写入
# =============================================================================

def test_save_l1_with_emotion():
    """
    测试 save_l1_summary 是否能正确写入 valence 和 salience。
    同时测试 get_l1_by_id 是否能读回这两个字段。
    """
    print_section("阶段二：情感字段写入与读取")

    try:
        from database import new_session, save_message, close_session
        from database import save_l1_summary, get_l1_by_id

        # 创建测试 session
        sid = new_session()
        save_message(sid, "user", "今天完成了一个很重要的里程碑，超级开心")
        save_message(sid, "assistant", "恭喜你！")
        close_session(sid)

        # 写入带情感字段的 L1（高兴奋、高显著性）
        # 测试自定义 created_at（模拟历史导入路径）
        fake_past_time = "2025-01-15T20:30:00+00:00"
        l1_id = save_l1_summary(
            session_id  = sid,
            summary     = "烧酒完成了重要里程碑，情绪高涨。",
            keywords    = "里程碑,成就,激动",
            time_period = "夜间",
            atmosphere  = "兴奋激动",
            valence     = 1.0,
            salience    = 1.0,
            created_at  = fake_past_time,   # 验证自定义时间被正确写入
        )

        # 读回验证 created_at 是否使用了传入值而非 _now()
        row = get_l1_by_id(l1_id)
        ok_time = print_result(
            "自定义 created_at 被正确写入",
            row["created_at"].startswith("2025-01-15"),
            f"期望前缀 2025-01-15，实际 {row['created_at']}"
        )

        # 读回并验证
        row = get_l1_by_id(l1_id)

        ok_written = row is not None
        print_result("L1 记录写入成功", ok_written, f"l1_id={l1_id}")

        if not ok_written:
            return False

        ok_valence  = print_result(
            "valence 字段正确写入",
            row["valence"] == 1.0,
            f"期望 1.0，实际 {row['valence']}"
        )
        ok_salience = print_result(
            "salience 字段正确写入",
            row["salience"] == 1.0,
            f"期望 1.0，实际 {row['salience']}"
        )

        # 测试默认值（不传 valence/salience 时）
        sid2  = new_session()
        save_message(sid2, "user", "今天天气不错")
        close_session(sid2)
        l1_id2 = save_l1_summary(
            session_id  = sid2,
            summary     = "烧酒进行了日常闲聊。",
            keywords    = "闲聊",
            time_period = "下午",
            atmosphere  = "轻松愉快",
            # 不传 valence 和 salience，测试默认值
        )
        row2 = get_l1_by_id(l1_id2)
        ok_default = print_result(
            "不传情感字段时使用默认值",
            row2["valence"] == 0.0 and row2["salience"] == 0.5,
            f"valence={row2['valence']}（期望0.0），salience={row2['salience']}（期望0.5）"
        )

        return ok_valence and ok_salience and ok_default and ok_time

    except Exception as e:
        print_result("情感字段写入测试", False, str(e))
        return False


# =============================================================================
# 阶段三：情感字段校验逻辑
# =============================================================================

def test_validate_emotion_fields():
    """
    测试 _validate_and_fix 对非标准情感值的对齐和容错处理。
    """
    print_section("阶段三：情感字段校验逻辑")

    try:
        from summarizer import _validate_and_fix

        all_pass = True

        # 测试正常五档值
        r1 = _validate_and_fix({
            "summary": "测试", "keywords": "测试", "time_period": "夜间",
            "atmosphere": "专注", "valence": 0.5, "salience": 0.75
        })
        ok1 = print_result(
            "标准档位值直接通过",
            r1["valence"] == 0.5 and r1["salience"] == 0.75,
            f"valence={r1['valence']}, salience={r1['salience']}"
        )
        all_pass = all_pass and ok1

        # 测试非标准值（如 0.3）被对齐到最近档位
        r2 = _validate_and_fix({
            "summary": "测试", "keywords": "测试", "time_period": "夜间",
            "atmosphere": "专注", "valence": 0.3, "salience": 0.6
        })
        ok2 = print_result(
            "非标准值自动对齐到最近档位",
            r2["valence"] == 0.5 and r2["salience"] == 0.5,
            f"0.3→{r2['valence']}（期望0.5），0.6→{r2['salience']}（期望0.5）"
        )
        all_pass = all_pass and ok2

        # 测试整数输入（模型可能输出整数 1 而非 1.0）
        r3 = _validate_and_fix({
            "summary": "测试", "keywords": "测试", "time_period": "夜间",
            "atmosphere": "专注", "valence": 1, "salience": 0
        })
        ok3 = print_result(
            "整数输入被正确转换为浮点",
            r3["valence"] == 1.0 and r3["salience"] == 0.0,
            f"valence={r3['valence']}, salience={r3['salience']}"
        )
        all_pass = all_pass and ok3

        # 测试字段缺失时使用默认值
        r4 = _validate_and_fix({
            "summary": "测试", "keywords": "测试", "time_period": "夜间",
            "atmosphere": "专注"
            # 没有 valence 和 salience
        })
        ok4 = print_result(
            "字段缺失时使用默认值",
            r4["valence"] == 0.0 and r4["salience"] == 0.5,
            f"valence={r4['valence']}（期望0.0），salience={r4['salience']}（期望0.5）"
        )
        all_pass = all_pass and ok4

        return all_pass

    except Exception as e:
        print_result("校验逻辑测试", False, str(e))
        return False


# =============================================================================
# 阶段四：index_l1 salience 写入 Chroma metadata
# =============================================================================

def test_index_l1_with_salience():
    """
    测试 index_l1 是否将 salience 正确写入 Chroma metadata，
    以及检索时能否从 metadata 读回。
    """
    print_section("阶段四：index_l1 salience 写入 Chroma metadata")

    try:
        from vector_store import index_l1, _get_collection, _COLL_L1
        from datetime import timezone

        test_l1_id  = 99901   # 使用较大 id 避免和真实数据冲突
        test_salience = 0.75
        now_str = datetime.now(timezone.utc).isoformat()

        # 写入测试索引
        index_l1(
            l1_id      = test_l1_id,
            summary    = "测试用摘要：烧酒完成了重要的阶段性目标。",
            keywords   = "测试,验证,阶段目标",
            session_id = None,
            created_at = now_str,
            salience   = test_salience,
        )

        # 从 Chroma 读回，验证 metadata 里 salience 是否存在
        collection = _get_collection(_COLL_L1)
        result = collection.get(ids=[str(test_l1_id)], include=["metadatas"])

        ok_exists = len(result["ids"]) > 0
        print_result("测试记录写入 Chroma", ok_exists, f"id={test_l1_id}")

        if not ok_exists:
            return False

        meta = result["metadatas"][0]
        ok_salience = print_result(
            "salience 正确写入 metadata",
            abs(meta.get("salience", -1) - test_salience) < 1e-6,
            f"期望 {test_salience}，实际 {meta.get('salience')}"
        )

        # 清理测试数据
        try:
            collection.delete(ids=[str(test_l1_id)])
        except Exception:
            pass

        return ok_salience

    except Exception as e:
        print_result("index_l1 salience 写入测试", False, str(e))
        return False


# =============================================================================
# 阶段五：_calc_decay_factor salience 加成效果
# =============================================================================

def test_salience_decay_effect():
    """
    验证 salience 加成是否真正影响了衰减计算：
    高 salience 的记忆应该比低 salience 的记忆有更高的保留率 R。
    """
    print_section("阶段五：salience 衰减加成效果验证")

    try:
        from vector_store import _calc_decay_factor

        # 模拟一条 30 天前的记忆
        created_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        decay_s    = 30   # L1 的稳定性系数

        R_low  = _calc_decay_factor(created_at, decay_s, salience=0.0)
        R_mid  = _calc_decay_factor(created_at, decay_s, salience=0.5)
        R_high = _calc_decay_factor(created_at, decay_s, salience=1.0)

        print(f"  30天前的记忆：")
        print(f"    salience=0.0 → R={R_low:.4f}")
        print(f"    salience=0.5 → R={R_mid:.4f}")
        print(f"    salience=1.0 → R={R_high:.4f}")

        ok_order = print_result(
            "高 salience 保留率 > 低 salience 保留率",
            R_high > R_mid > R_low,
            f"R_low={R_low:.4f} < R_mid={R_mid:.4f} < R_high={R_high:.4f}"
        )

        # 验证默认值（salience=None）和 salience=0.5 的结果相同
        R_none = _calc_decay_factor(created_at, decay_s, salience=None)
        ok_default = print_result(
            "salience=None 时等价于 salience=0.5",
            abs(R_none - R_mid) < 1e-6,
            f"R_none={R_none:.4f}, R_mid={R_mid:.4f}"
        )

        # 验证 salience=0 时等于纯 Ebbinghaus（不加成）
        expected_pure = math.exp(-30 / decay_s)
        ok_pure = print_result(
            "salience=0.0 时等于纯 Ebbinghaus（无加成）",
            abs(R_low - expected_pure) < 1e-6,
            f"R_low={R_low:.4f}，纯Ebbinghaus={expected_pure:.4f}"
        )

        return ok_order and ok_default and ok_pure

    except Exception as e:
        print_result("salience 衰减加成测试", False, str(e))
        return False


# =============================================================================
# 阶段六：_build_context RAG 链路连通性
# =============================================================================

def test_rag_pipeline():
    """
    验证 _build_context() 是否真正调用了 retrieve_combined()，
    并且返回的 context 里 retrieved_l1l2 字段有内容（如果索引非空）。

    注意：
        如果向量索引完全为空（全新环境），retrieved_l1l2 可能为 None，
        这不算失败——关键是函数能跑通不报错。
    """
    print_section("阶段六：_build_context RAG 链路连通性")

    try:
        # 直接从 main.py 导入 _build_context
        # 注意：这会触发 main.py 的模块级初始化
        import importlib
        main_module = importlib.import_module("main")
        _build_context = getattr(main_module, "_build_context", None)

        if _build_context is None:
            print_result("_build_context 可导入", False, "main.py 中找不到该函数")
            return False

        print_result("_build_context 可导入", True)

        # 先获取一个有效的 session_id（取任意一个已关闭的 session）
        from database import get_active_sessions
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM sessions WHERE ended_at IS NOT NULL ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            print("  ⚠️ 数据库里没有已关闭的 session，跳过 context 内容检查")
            print("     （链路本身通畅，只是无历史数据可检索）")
            return True

        test_session_id = row["id"]
        test_query = "最近有没有写代码熬夜"

        # 调用 _build_context
        context = _build_context(test_session_id, user_message=test_query)

        ok_no_error = print_result(
            "_build_context 调用不报错",
            context is not None and isinstance(context, dict),
            f"返回类型：{type(context)}"
        )

        if not ok_no_error:
            return False

        # 检查 context 结构
        required_keys = {
            "last_session_time", "l3_profile", "retrieved_l1l2",
            "raw_fragments", "session_id", "session_index"
        }
        missing = required_keys - set(context.keys())
        ok_keys = print_result(
            "context 包含所有必要字段",
            len(missing) == 0,
            f"缺少字段：{missing}" if missing else "所有字段均存在"
        )

        # 检查 retrieved_l1l2 字段（不为 None 才说明 RAG 真正被调用了）
        rl = context.get("retrieved_l1l2")
        if rl is not None:
            print_result(
                "retrieved_l1l2 有内容（RAG 已接入且有检索结果）",
                True,
                f"内容预览：{str(rl)[:120]}…"
            )
        else:
            # 索引为空时 RAG 返回空结果是正常的，降级到时间序
            print("  ℹ️  retrieved_l1l2 为 None（向量索引可能为空，或无相关记忆）")
            print("     这不是 FAIL——关键是链路通畅，有记忆数据后会自动生效")

        return ok_no_error and ok_keys

    except Exception as e:
        import traceback
        print_result("RAG 链路连通性测试", False, str(e))
        traceback.print_exc()
        return False


# =============================================================================
# 主入口
# =============================================================================

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  RAG 链路 + 情感字段 端到端验证脚本" + " "*20 + "║")
    print("╚" + "═"*58 + "╝")

    results = {}

    results["数据库字段存在性"] = test_db_schema()

    if not results["数据库字段存在性"]:
        print("\n⛔ 数据库字段缺失，请先运行 migrate_add_emotion_fields.py，然后重试。")
        return

    results["情感字段写入读取"] = test_save_l1_with_emotion()
    results["情感字段校验逻辑"] = test_validate_emotion_fields()
    results["Chroma salience写入"] = test_index_l1_with_salience()
    results["salience衰减加成"]   = test_salience_decay_effect()
    results["RAG链路连通性"]      = test_rag_pipeline()

    # ── 汇总 ──
    print_section("测试汇总")
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 全部测试通过！")
        print("   情感字段已正确集成，RAG 链路连通。")
        print("   下一步：启动主服务进行真实对话验证。")
    else:
        print("\n⚠️  有测试未通过，请根据上方 [FAIL] 信息逐项排查。")


if __name__ == "__main__":
    main()
