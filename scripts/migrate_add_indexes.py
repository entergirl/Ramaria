"""
migrate_add_indexes.py — 数据库索引迁移脚本
=====================================================================

作用：
    为高频查询字段添加索引，提升检索性能。
    重点覆盖：对话消息查询、L1 未吸收查询、冲突检测查询、推送消息查询。

幂等设计：
    所有索引使用 CREATE INDEX IF NOT EXISTS，
    已存在时自动跳过，可以安全地反复运行。

运行时机：
    · 已有数据库的用户：运行此脚本一次即可
    · 全新安装：init_db.py 已包含这些索引，无需单独运行

使用方法：
    python migrate_add_indexes.py

索引清单（共 7 个）：
    idx_messages_session_id     messages.session_id
    idx_messages_created_at     messages.created_at
    idx_memory_l1_session_id    memory_l1.session_id
    idx_memory_l1_absorbed      memory_l1.absorbed
    idx_memory_l1_created_at    memory_l1.created_at
    idx_conflict_queue_status   conflict_queue.status
    idx_pending_push_status     pending_push.status
"""

import os
import sqlite3

from ramaria.config import DB_PATH


# =============================================================================
# 索引定义表
# =============================================================================

# 每个元素：(索引名, 建索引的 SQL, 说明)
# 使用 CREATE INDEX IF NOT EXISTS，幂等安全
INDEX_DEFINITIONS = [
    (
        "idx_messages_session_id",
        "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)",
        "messages 表 session_id 索引 — get_messages() / get_last_message_time() 高频查询",
    ),
    (
        "idx_messages_created_at",
        "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
        "messages 表 created_at 索引 — get_last_message_time() 排序，空闲检测每分钟触发",
    ),
    (
        "idx_memory_l1_session_id",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_session_id ON memory_l1(session_id)",
        "memory_l1 表 session_id 索引 — get_l1_by_session() 查询",
    ),
    (
        "idx_memory_l1_absorbed",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_absorbed ON memory_l1(absorbed)",
        "memory_l1 表 absorbed 索引 — get_unabsorbed_l1() 每次 L1 写入后触发",
    ),
    (
        "idx_memory_l1_created_at",
        "CREATE INDEX IF NOT EXISTS idx_memory_l1_created_at ON memory_l1(created_at)",
        "memory_l1 表 created_at 索引 — L2 触发检查的排序字段 / get_latest_l1()",
    ),
    (
        "idx_conflict_queue_status",
        "CREATE INDEX IF NOT EXISTS idx_conflict_queue_status ON conflict_queue(status)",
        "conflict_queue 表 status 索引 — get_pending_conflicts() 每次对话前检查",
    ),
    (
        "idx_pending_push_status",
        "CREATE INDEX IF NOT EXISTS idx_pending_push_status ON pending_push(status)",
        "pending_push 表 status 索引 — get_pending_pushes() WebSocket 建立时查询",
    ),
]


# =============================================================================
# 迁移执行函数
# =============================================================================

def migrate():
    print("=== 数据库索引迁移脚本 ===")
    print(f"数据库路径：{DB_PATH}\n")

    if not os.path.exists(DB_PATH):
        print("❌ 数据库文件不存在，请先运行 init_db.py 初始化数据库。")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # 读取当前已有索引，用于判断是新建还是已存在
    # sqlite_master 表记录了所有数据库对象（表、索引、触发器等）
    # ------------------------------------------------------------------
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    )
    existing_indexes = {row["name"] for row in cursor.fetchall()}

    results = {}

    for index_name, sql, description in INDEX_DEFINITIONS:
        print(f"  处理：{index_name}")
        print(f"        {description}")

        if index_name in existing_indexes:
            # 索引已存在，跳过（IF NOT EXISTS 也会跳过，但这里打印更友好的提示）
            print(f"  ✓  已存在，跳过\n")
            results[index_name] = "已存在"
        else:
            try:
                cursor.execute(sql)
                print(f"  ✅ 新建成功\n")
                results[index_name] = "新建成功"
            except Exception as e:
                print(f"  ❌ 新建失败：{e}\n")
                results[index_name] = f"失败：{e}"

    conn.commit()

    # ------------------------------------------------------------------
    # 验证：重新查询实际存在的索引，与预期对比
    # ------------------------------------------------------------------
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    )
    final_indexes = {row["name"] for row in cursor.fetchall()}
    conn.close()

    # ------------------------------------------------------------------
    # 汇总报告
    # ------------------------------------------------------------------
    print("=" * 56)
    print("  迁移结果汇总")
    print("=" * 56)

    all_ok = True
    for index_name, status in results.items():
        mark = "✅" if status in ("新建成功", "已存在") else "❌"
        in_db = "（已在数据库中确认）" if index_name in final_indexes else "（⚠️ 数据库中未找到）"
        print(f"  {mark} {index_name}: {status} {in_db}")
        if "失败" in status:
            all_ok = False

    print()
    if all_ok:
        print("🎉 迁移完成，所有索引已就绪。")
        print(f"   数据库当前共有 {len(final_indexes)} 个索引（含系统自动创建的主键索引）。")
    else:
        print("⚠️  部分索引创建失败，请检查上方错误信息。")


if __name__ == "__main__":
    migrate()