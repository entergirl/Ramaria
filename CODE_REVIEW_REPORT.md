# 珊瑚菌 (Ramaria) 代码审查报告

> 审查时间: 2026-04-22
> 项目版本: v0.6.0

---

## 一、大文件拆分建议 (P2)

| 文件 | 大小 | 行数(估) | 建议拆分方案 |
|------|------|---------|-------------|
| `src/ramaria/storage/vector_store.py` | 51.54 KB | ~1400 | 拆分为 `bm25_index.py` + `retrieval.py` |
| `src/ramaria/storage/database.py` | 50.16 KB | ~1350 | 按功能拆分为 `db_queries_*.py` |
| `src/ramaria/memory/graph_builder.py` | 31.62 KB | ~850 | 拆分为 `graph_builder.py` + `graph_analyzer.py` |
| `scripts/setup_db.py` | 36.71 KB | ~1000 | 迁移逻辑独立模块 |

---

## 二、代码质量问题 (P1-P3)

### 2.1 重复代码模式

#### 2.1.1 重复常量字符串模板

**文件**: `src/ramaria/memory/merger.py` 第124行

```python
[序号] 时间段 | 氛围：xxx
摘要：xxx
关键词：xxx
```

此字符串模板在 `src/ramaria/memory/prompt_builder.py` 中可能有重复定义，需合并。

#### 2.1.2 异常处理可改进

**文件**: `src/ramaria/storage/vector_store.py` 第267-269行

```python
except Exception as e:
    logger.warning(f"jieba 自定义词典加载失败，使用默认分词 — {e}")
```

建议记录异常类型，便于调试。

### 2.2 字符串处理不一致 (P1)

项目中存在多处 `.strip()` / `.lower()` 调用，顺序不统一：
- 有些使用 `t.lower().strip()`
- 有些使用 `t.strip().lower()`

**建议**: 统一为 `t.lower().strip()` 模式

---

## 三、低优先级问题 (P3)

### 3.1 空目录待说明

| 目录 | 用途 | 建议 |
|------|------|------|
| `src/ramaria/backends/` | 预留后端适配目录 | 添加 `README.md` 说明用途 |

### 3.2 安全提示

**`.env.example`** 和前端配置页面中包含示例 API Key:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxx
```

建议添加更明显的注释说明这只是占位符。

---

## 四、测试覆盖率不足 (P2)

### 建议补充的测试

| 模块 | 测试文件 | 说明 |
|------|---------|------|
| API 路由 | 缺失 | 缺少对 `/chat`, `/memory`, `/sessions` 路由的单元测试 |
| Session 管理 | 缺失 | `session_manager.py` 缺少测试 |
| 路由逻辑 | 缺失 | `router.py` 缺少测试 |

---

## 五、长期改进方向 (P3)

- [ ] **类型检查**: 引入 pyright/mypy
- [ ] **代码格式化**: 添加 black/isort
- [ ] **文档完善**: 补充 docstring
- [ ] **dataclass**: 考虑引入替代部分字典

---

## 六、已完成清理记录 (v0.5.0)

以下为本次审查已完成的工作，记录备查：

| 项目 | 状态 | 说明 |
|------|------|------|
| 版本号统一 | ✅ | 仅在 `pyproject.toml` 保留 `0.5.0` |
| 遗留文件删除 | ✅ | 删除根目录 `logger.py`、`constants.py` |
| 关键词配置集中 | ✅ | 移至 `constants.py` |
| 业务参数集中 | ✅ | `MAX_HISTORY_MESSAGES` 移至 `config.py` |
| 消息来源追踪 | ✅ | 添加 `source` 字段支持 |
| 重复函数合并 | ✅ | `_sort_key_l1/l2` → `_calc_sort_score` |
| 云端 API 文本通用化 | ✅ | 用户可见文本已替换为通用表述 |

---

## 七、待处理清单

| 优先级 | 任务 | 关联 |
|--------|------|------|
| P1 | 统一字符串处理模式 | 2.2 |
| P2 | 拆分大型文件 | 一 |
| P2 | 补充测试覆盖 | 四 |
| P3 | 添加空目录说明 | 3.1 |
| P3 | 长期代码质量改进 | 五 |
