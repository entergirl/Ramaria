"""
migrate_add_push.py — 主动推送功能数据库迁移脚本
=====================================================================

作用：
    1. 新建 pending_push 表（暂存用户离线时触发的主动消息）
    2. 向 settings 表写入推送相关默认配置项：
        debounce_seconds    防抖等待时长（秒），默认 3
        push_enabled        主动推送总开关，默认 1（开启）
        push_window_start   推送时间窗口开始（24小时制），默认 8
        push_window_end     推送时间窗口结束（24小时制），默认 24
        push_daily_limit    每日推送条数上限，默认 4

幂等设计：
    所有操作在执行前检查是否已存在，重复运行不会报错、不会覆盖
    用户已修改过的配置值。

运行时机：
    在启动新版本服务之前运行一次。

使用方法：
    python migrate_add_push.py
"""

import sqlite3
import os
from datetime import datetime, timezone
from ramaria.config import DB_PATH


def migrate():
    print("=== 主动推送功能数据库迁移脚本 ===")
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
    # 第一部分：新建 pending_push 表
    # =========================================================================
    print("【1/2】检查 pending_push 表…")
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_push'"
    )
    if cursor.fetchone():
        print("  ✓ pending_push 表已存在，跳过")
        results["pending_push"] = "已存在"
    else:
        cursor.execute("""
            CREATE TABLE pending_push (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,

                -- 消息内容（已生成好的文本，直接推送给用户）
                content     TEXT    NOT NULL,

                -- 消息生成时间（UTC ISO 8601）
                -- 用于上线后按时间顺序推送，还原"错过消息"的真实感
                created_at  TEXT    NOT NULL,

                -- 推送状态
                -- pending  — 用户尚未上线，等待推送
                -- sent     — 已成功推送给用户
                status      TEXT    NOT NULL DEFAULT 'pending',

                -- 推送完成时间（sent 时写入，pending 时为 NULL）
                sent_at     TEXT
            )
        """)
        print("  ✅ pending_push 表创建成功")
        results["pending_push"] = "新建成功"

    # =========================================================================
    # 第二部分：写入默认配置项
    # 使用 INSERT OR IGNORE，不覆盖用户已修改的值
    # =========================================================================
    print("【2/2】写入默认配置项…")

    now = datetime.now(timezone.utc).isoformat()

    # 默认配置项列表：(key, value, 说明)
    default_settings = [
        ("debounce_seconds",  "3",  "防抖等待时长（秒）"),
        ("push_enabled",      "1",  "主动推送总开关：1=开启，0=关闭"),
        ("push_window_start", "8",  "推送时间窗口开始（24小时制整点）"),
        ("push_window_end",   "24", "推送时间窗口结束（24小时制，24表示午夜）"),
        ("push_daily_limit",  "4",  "每日推送条数上限"),
    ]

    for key, value, desc in default_settings:
        cursor.execute(
            "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now)
        )
        # 检查是插入了新行还是已存在
        if cursor.rowcount > 0:
            print(f"  ✅ {key} = {value}（{desc}）")
            results[key] = "新增"
        else:
            # 读出当前值显示
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            current = cursor.fetchone()["value"]
            print(f"  ✓  {key} = {current}（已存在，保留现有值）")
            results[key] = "已存在"

    conn.commit()
    conn.close()

    # =========================================================================
    # 汇总
    # =========================================================================
    print("\n=== 迁移结果汇总 ===")
    all_ok = True
    for key, status in results.items():
        mark = "✅" if status in ("新建成功", "新增", "已存在") else "❌"
        print(f"  {mark} {key}: {status}")
        if "失败" in status:
            all_ok = False

    if all_ok:
        print("\n🎉 迁移完成，可以继续后续步骤。")
    else:
        print("\n⚠️  部分迁移失败，请检查上方错误信息。")


if __name__ == "__main__":
    migrate()
