"""
tests/unit/storage/test_database.py — database.py 单元测试

覆盖范围：
    · Session 创建与关闭
    · 消息写入与读取
    · L1 摘要写入与查询
    · 配置项读写
    · 时间戳修改（测试专用函数）
    · 连接异常后自动释放（P0-2 修复验证）
    · 批量 last_accessed_at 更新（P0-1 修复验证）

运行方式：
    # 在项目根目录
    pytest tests/unit/storage/test_database.py -v
"""

import pytest
from datetime import datetime, timezone, timedelta

from ramaria.storage.database import (
    new_session,
    close_session,
    get_session,
    get_active_sessions,
    save_message,
    get_messages,
    get_messages_as_dicts,
    get_last_message_time,
    update_message_time_for_test,
    save_l1_summary,
    get_l1_by_id,
    get_unabsorbed_l1,
    get_all_l1,
    get_all_l2,
    get_all_session_ids,
    get_setting,
    set_setting,
    batch_update_last_accessed,
    get_last_accessed_at,
    _db_conn,
)


# =============================================================================
# Session 操作
# =============================================================================

class TestSession:

    def test_new_session_returns_int(self):
        sid = new_session()
        assert isinstance(sid, int)
        assert sid > 0

    def test_close_session(self):
        sid = new_session()
        row = get_session(sid)
        assert row["ended_at"] is None

        close_session(sid)
        row = get_session(sid)
        assert row["ended_at"] is not None

    def test_get_session_not_found(self):
        row = get_session(999999)
        assert row is None

    def test_get_active_sessions(self):
        # 确保有至少一个活跃 session
        sid = new_session()
        active = get_active_sessions()
        ids = [s["id"] for s in active]
        assert sid in ids

        # 关闭后不应再出现
        close_session(sid)
        active_after = get_active_sessions()
        ids_after = [s["id"] for s in active_after]
        assert sid not in ids_after


# =============================================================================
# 消息操作
# =============================================================================

class TestMessages:

    def setup_method(self):
        """每个测试方法前创建一个独立的 session"""
        self.sid = new_session()

    def teardown_method(self):
        """测试结束后关闭 session"""
        close_session(self.sid)

    def test_save_and_get_messages(self):
        mid1 = save_message(self.sid, "user", "你好")
        mid2 = save_message(self.sid, "assistant", "你好！")
        assert isinstance(mid1, int)
        assert isinstance(mid2, int)

        msgs = get_messages(self.sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "你好"
        assert msgs[1]["role"] == "assistant"

    def test_get_messages_as_dicts(self):
        save_message(self.sid, "user", "测试")
        dicts = get_messages_as_dicts(self.sid)
        assert len(dicts) == 1
        assert set(dicts[0].keys()) == {"role", "content"}

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            save_message(self.sid, "system", "不合法的角色")

    def test_get_last_message_time(self):
        assert get_last_message_time(self.sid) is None

        save_message(self.sid, "user", "第一条")
        save_message(self.sid, "user", "第二条")
        t = get_last_message_time(self.sid)
        assert t is not None
        assert isinstance(t, str)

    def test_update_message_time_for_test(self):
        save_message(self.sid, "user", "消息")
        fake_time = (
            datetime.now(timezone.utc) - timedelta(minutes=15)
        ).isoformat()
        update_message_time_for_test(self.sid, fake_time)

        t = get_last_message_time(self.sid)
        # 只比较到秒级别，避免时区格式差异
        assert t[:19] == fake_time[:19]


# =============================================================================
# L1 摘要操作
# =============================================================================

class TestL1Summary:

    def setup_method(self):
        self.sid = new_session()
        save_message(self.sid, "user", "测试消息")
        close_session(self.sid)

    def test_save_and_get_l1(self):
        l1_id = save_l1_summary(
            session_id  = self.sid,
            summary     = "烧酒完成了测试。",
            keywords    = "测试,验证",
            time_period = "夜间",
            atmosphere  = "专注高效",
        )
        assert isinstance(l1_id, int)

        row = get_l1_by_id(l1_id)
        assert row is not None
        assert row["summary"] == "烧酒完成了测试。"
        assert row["time_period"] == "夜间"
        # 默认情感字段
        assert row["valence"] == 0.0
        assert row["salience"] == 0.5

    def test_invalid_time_period_becomes_none(self):
        l1_id = save_l1_summary(
            session_id  = self.sid,
            summary     = "测试摘要。",
            keywords    = "测试",
            time_period = "中午",   # 不合法值
            atmosphere  = "正常",
        )
        row = get_l1_by_id(l1_id)
        assert row["time_period"] is None

    def test_get_l1_not_found(self):
        assert get_l1_by_id(999999) is None

    def test_get_unabsorbed_l1(self):
        l1_id = save_l1_summary(
            self.sid, "摘要", "关键词", "夜间", "专注"
        )
        rows = get_unabsorbed_l1()
        ids = [r["id"] for r in rows]
        assert l1_id in ids

    def test_get_unabsorbed_l1_with_limit(self):
        rows = get_unabsorbed_l1(limit=1)
        assert len(rows) <= 1


# =============================================================================
# 配置项操作
# =============================================================================

class TestSettings:

    def test_get_default_setting(self):
        val = get_setting("l1_idle_minutes")
        assert val is not None

    def test_set_and_get_setting(self):
        set_setting("test_key_xyz", "test_value_123")
        val = get_setting("test_key_xyz")
        assert val == "test_value_123"

    def test_get_missing_key_returns_default(self):
        val = get_setting("nonexistent_key_xyz", default="fallback")
        assert val == "fallback"


# =============================================================================
# 修复验证：batch_update_last_accessed 白名单保护
# =============================================================================

class TestBatchUpdateLastAccessed:

    def test_empty_list_returns_zero(self):
        result = batch_update_last_accessed("l1", [])
        assert result == 0

    def test_invalid_layer_returns_zero(self):
        # 非法 layer 应该安全返回 0，不执行 SQL
        result = batch_update_last_accessed("l3", [1, 2])
        assert result == 0

    def test_invalid_layer_get_returns_none(self):
        result = get_last_accessed_at("l3", 1)
        assert result is None


# =============================================================================
# 修复验证：连接异常后自动释放
# =============================================================================

class TestConnectionRelease:

    def test_connection_released_after_exception(self):
        """
        验证 _db_conn 上下文管理器在异常发生时也能释放连接。
        如果连接未释放，后续操作会因文件锁而失败（Windows 环境尤其明显）。
        """
        try:
            with _db_conn() as conn:
                conn.execute("SELECT 1")
                raise RuntimeError("模拟异常")
        except RuntimeError:
            pass

        # 异常后连接应已正确释放，下一条操作应该正常执行
        sid = new_session()
        close_session(sid)
        assert sid > 0