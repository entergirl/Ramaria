"""
migrate_add_graph_tables.py — 知识图谱数据库迁移脚本
=====================================================================

作用：
    1. 新建 graph_nodes 表（图谱实体节点）
    2. 新建 graph_edges 表（图谱关系边）
    3. keyword_pool 表新增 canonical_id + alias_status 两列
    4. conflict_queue 表新增 conflict_type 列

幂等设计：
    所有操作在执行前检查是否已存在，已存在则跳过，
    可以安全地反复运行。

运行时机：
    在启动 graph_builder.py 之前运行一次。
    建议在修改其他模块之前先跑这个脚本，确认数据库结构就绪。

使用方法：
    python migrate_add_graph_tables.py
"""

import os
import sqlite3
from datetime import datetime, timezone

from ramaria.config import DB_PATH


def migrate():
    print(f"=== 知识图谱数据库迁移脚本 ===")
    print(f"数据库路径：{DB_PATH}\n")

    if not os.path.exists(DB_PATH):
        print("❌ 数据库文件不存在，请先运行 init_db.py 初始化数据库。")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    results = {}

    # =========================================================================
    # 第一部分：新建 graph_nodes 表
    # =========================================================================
    print("【1/4】检查 graph_nodes 表…")
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_nodes'"
    )
    if cursor.fetchone():
        print("  ✓ graph_nodes 表已存在，跳过")
        results["graph_nodes"] = "已存在"
    else:
        cursor.execute("""
            CREATE TABLE graph_nodes (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,

                -- 归一化后的实体名（规范词，全局唯一）
                -- 写入前必须经过 _normalize_entity() 归一化
                entity_name  TEXT NOT NULL UNIQUE,

                -- 实体类型，五选一：
                --   person   人物（烧酒、撒老师等）
                --   project  项目（珊瑚菌、Ramaria）
                --   module   代码模块/文件（vector_store.py、summarizer.py）
                --   concept  抽象概念（BM25、RRF、遗忘曲线）
                --   time     时间节点（2026-03 等时间标签）
                entity_type  TEXT NOT NULL,

                -- 首次出现在哪条 L1 摘要（可溯源）
                source_l1_id INTEGER,

                -- 节点创建时间
                created_at   TEXT NOT NULL,

                -- 被引用次数，类比 keyword_pool.use_count
                -- 每次有新的 graph_edge 指向本节点时 +1
                use_count    INTEGER NOT NULL DEFAULT 1,

                FOREIGN KEY (source_l1_id) REFERENCES memory_l1(id)
            )
        """)
        print("  ✅ graph_nodes 表创建成功")
        results["graph_nodes"] = "新建成功"

    # =========================================================================
    # 第二部分：新建 graph_edges 表
    # =========================================================================
    print("【2/4】检查 graph_edges 表…")
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'"
    )
    if cursor.fetchone():
        print("  ✓ graph_edges 表已存在，跳过")
        results["graph_edges"] = "已存在"
    else:
        cursor.execute("""
            CREATE TABLE graph_edges (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,

                -- 主语节点 id（指向 graph_nodes.id）
                source_node_id  INTEGER NOT NULL,

                -- 宾语节点 id（指向 graph_nodes.id）
                target_node_id  INTEGER NOT NULL,

                -- 关系类型，七类大类之一：
                --   TASK_STATUS   任务状态类（开始/进行中/完成/放弃）
                --   OBSTACLE      遇到障碍类（报错/卡住/待解决）
                --   USES_DEPENDS  使用或依赖类（使用了某工具/库/方案）
                --   BELONGS_TO    归属类（某模块属于某项目）
                --   EMOTION_STATE 情感状态类（情绪、疲惫、开心等）
                --   SOCIAL_EVENT  社交事件类（与某人发生的互动）
                --   TIME_ANCHOR   时间归属类（某事发生于某时间段）
                relation_type   TEXT NOT NULL,

                -- 模型提取的原始细节描述（一句话）
                -- 这是"固定大类 + 模型填细节"方案里的细节部分
                relation_detail TEXT,

                -- 来源 L1 摘要 id（核心可追溯字段）
                -- 通过这个字段可以回查 SQLite 获取完整的原始对话内容
                source_l1_id    INTEGER NOT NULL,

                -- 边的创建时间
                created_at      TEXT NOT NULL,

                FOREIGN KEY (source_node_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (target_node_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (source_l1_id)   REFERENCES memory_l1(id)
            )
        """)
        # 为常用查询路径建立索引，加速图遍历
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_node_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_node_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_l1 ON graph_edges(source_l1_id)"
        )
        print("  ✅ graph_edges 表及索引创建成功")
        results["graph_edges"] = "新建成功"

    # =========================================================================
    # 第三部分：keyword_pool 表新增两列
    # =========================================================================
    print("【3/4】检查 keyword_pool 表新增列…")
    cursor.execute("PRAGMA table_info(keyword_pool)")
    kp_columns = {row["name"] for row in cursor.fetchall()}

    # 新增 canonical_id 列
    if "canonical_id" in kp_columns:
        print("  ✓ keyword_pool.canonical_id 已存在，跳过")
        results["kp_canonical_id"] = "已存在"
    else:
        try:
            cursor.execute(
                "ALTER TABLE keyword_pool ADD COLUMN canonical_id INTEGER"
            )
            # NULL 表示本词就是规范词，没有上级
            # 现有词条全部视为规范词，canonical_id 保持 NULL
            print("  ✅ keyword_pool.canonical_id 列新增成功")
            results["kp_canonical_id"] = "新增成功"
        except Exception as e:
            print(f"  ❌ keyword_pool.canonical_id 新增失败：{e}")
            results["kp_canonical_id"] = f"失败：{e}"

    # 新增 alias_status 列
    if "alias_status" in kp_columns:
        print("  ✓ keyword_pool.alias_status 已存在，跳过")
        results["kp_alias_status"] = "已存在"
    else:
        try:
            cursor.execute(
                "ALTER TABLE keyword_pool ADD COLUMN alias_status TEXT DEFAULT 'confirmed'"
            )
            # 现有词条全部视为已确认状态
            cursor.execute(
                "UPDATE keyword_pool SET alias_status = 'confirmed' WHERE alias_status IS NULL"
            )
            print("  ✅ keyword_pool.alias_status 列新增成功（历史词条已填充 confirmed）")
            results["kp_alias_status"] = "新增成功"
        except Exception as e:
            print(f"  ❌ keyword_pool.alias_status 新增失败：{e}")
            results["kp_alias_status"] = f"失败：{e}"

    # =========================================================================
    # 第四部分：conflict_queue 表新增 conflict_type 列
    # =========================================================================
    print("【4/4】检查 conflict_queue 表新增列…")
    cursor.execute("PRAGMA table_info(conflict_queue)")
    cq_columns = {row["name"] for row in cursor.fetchall()}

    if "conflict_type" in cq_columns:
        print("  ✓ conflict_queue.conflict_type 已存在，跳过")
        results["cq_conflict_type"] = "已存在"
    else:
        try:
            cursor.execute(
                "ALTER TABLE conflict_queue ADD COLUMN conflict_type TEXT DEFAULT 'profile'"
            )
            # 现有冲突记录全部标记为原有的 profile 类型
            cursor.execute(
                "UPDATE conflict_queue SET conflict_type = 'profile' WHERE conflict_type IS NULL"
            )
            print("  ✅ conflict_queue.conflict_type 列新增成功（历史记录已填充 profile）")
            results["cq_conflict_type"] = "新增成功"
        except Exception as e:
            print(f"  ❌ conflict_queue.conflict_type 新增失败：{e}")
            results["cq_conflict_type"] = f"失败：{e}"

    conn.commit()
    conn.close()

    # =========================================================================
    # 验证：重新读取表结构确认迁移结果
    # =========================================================================
    print("\n=== 迁移结果汇总 ===")
    all_ok = True
    for key, status in results.items():
        mark = "✅" if "成功" in status or "已存在" in status else "❌"
        print(f"  {mark} {key}: {status}")
        if "失败" in status:
            all_ok = False

    if all_ok:
        print("\n🎉 迁移完成，可以继续开发 graph_builder.py。")
    else:
        print("\n⚠️  部分迁移失败，请检查上方错误信息。")


if __name__ == "__main__":
    migrate()