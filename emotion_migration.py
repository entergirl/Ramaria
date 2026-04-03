"""
migrate_add_emotion_fields.py — 数据库迁移脚本
=====================================================================

作用：
    给 memory_l1 表新增两个情感元数据字段：
        valence  REAL  — 情绪效价，五档浮点：-1.0 / -0.5 / 0.0 / 0.5 / 1.0
        salience REAL  — 情感显著性，五档浮点：0.0 / 0.25 / 0.5 / 0.75 / 1.0

使用方法：
    python migrate_add_emotion_fields.py

注意事项：
    · 幂等设计：列已存在时自动跳过，重复运行不会报错
    · 历史数据：已有的 L1 记录自动填充默认值
        valence  = 0.0  （中性）
        salience = 0.5  （中等显著性）
    · 运行完成后可以保留此文件作为迁移记录，也可以删除

运行时机：
    在修改 summarizer.py / vector_store.py 之前运行，
    确保数据库结构就绪后再启动主服务。
"""

import sqlite3
import os

# 数据库路径：和 config.py 保持一致，取本文件所在目录下的 assistant.db
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assistant.db")


def migrate():
    print(f"=== 情感字段迁移脚本 ===")
    print(f"数据库路径：{DB_PATH}\n")

    if not os.path.exists(DB_PATH):
        print("❌ 数据库文件不存在，请先运行 init_db.py 初始化数据库。")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ── 检查 memory_l1 表是否存在 ──
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_l1'")
    if not cursor.fetchone():
        print("❌ memory_l1 表不存在，请先运行 init_db.py。")
        conn.close()
        return

    # ── 读取当前列信息 ──
    cursor.execute("PRAGMA table_info(memory_l1)")
    existing_columns = {row["name"] for row in cursor.fetchall()}
    print(f"当前 memory_l1 列：{sorted(existing_columns)}")

    results = {"valence": None, "salience": None}

    # ── 新增 valence 列 ──
    if "valence" in existing_columns:
        print("✓ valence 列已存在，跳过")
        results["valence"] = "已存在"
    else:
        try:
            cursor.execute("ALTER TABLE memory_l1 ADD COLUMN valence REAL")
            # 历史数据填充中性默认值 0.0
            cursor.execute("UPDATE memory_l1 SET valence = 0.0 WHERE valence IS NULL")
            print("✅ valence 列新增成功（历史记录已填充默认值 0.0）")
            results["valence"] = "新增成功"
        except Exception as e:
            print(f"❌ valence 列新增失败：{e}")
            results["valence"] = f"失败：{e}"

    # ── 新增 salience 列 ──
    if "salience" in existing_columns:
        print("✓ salience 列已存在，跳过")
        results["salience"] = "已存在"
    else:
        try:
            cursor.execute("ALTER TABLE memory_l1 ADD COLUMN salience REAL")
            # 历史数据填充中等显著性默认值 0.5
            cursor.execute("UPDATE memory_l1 SET salience = 0.5 WHERE salience IS NULL")
            print("✅ salience 列新增成功（历史记录已填充默认值 0.5）")
            results["salience"] = "新增成功"
        except Exception as e:
            print(f"❌ salience 列新增失败：{e}")
            results["salience"] = f"失败：{e}"

    conn.commit()

    # ── 验证迁移结果 ──
    cursor.execute("PRAGMA table_info(memory_l1)")
    final_columns = {row["name"] for row in cursor.fetchall()}
    print(f"\n迁移后 memory_l1 列：{sorted(final_columns)}")

    # ── 打印当前 L1 记录数，确认历史数据填充情况 ──
    cursor.execute("SELECT COUNT(*) AS cnt FROM memory_l1")
    total = cursor.fetchone()["cnt"]
    if total > 0:
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM memory_l1 WHERE valence IS NOT NULL AND salience IS NOT NULL"
        )
        filled = cursor.fetchone()["cnt"]
        print(f"历史 L1 记录：共 {total} 条，已填充情感字段 {filled} 条")

    conn.close()

    # ── 汇总 ──
    print("\n=== 迁移结果汇总 ===")
    for field, status in results.items():
        print(f"  {field}: {status}")

    all_ok = all("失败" not in str(v) for v in results.values())
    if all_ok:
        print("\n🎉 迁移完成，可以继续更新其他模块。")
    else:
        print("\n⚠️ 部分迁移失败，请检查上方错误信息。")


if __name__ == "__main__":
    migrate()
