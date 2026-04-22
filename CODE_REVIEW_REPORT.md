# 珊瑚菌 (Ramaria) 代码审查报告

> 审查时间: 2026-04-22  
> 项目版本: v0.5.0

---

## 一、中等问题 (建议处理)

### 1.1 大型文件需要拆分

| 文件 | 当前大小 | 问题 | 建议 |
|------|---------|------|------|
| `src/ramaria/storage/vector_store.py` | 51.54 KB | 功能过多 | 拆分为 `bm25_index.py` + `vector_store.py` |
| `src/ramaria/storage/database.py` | 50.16 KB | 功能过多 | 拆分为 `db_queries_*.py` |
| `src/ramaria/memory/graph_builder.py` | 31.62 KB | 功能过多 | 拆分为 `graph_builder.py` + `graph_analyzer.py` |
| `scripts/setup_db.py` | 36.71 KB | 迁移脚本 | 拆分迁移逻辑 |

### 1.2 魔法数字/字符串

**`app/routes/chat.py`**:
```python
_MAX_HISTORY_MESSAGES = 40           # 应从配置读取
_CONFLICT_REPLY_MAX_LEN = 20         # 应从配置读取
```

**`src/ramaria/core/router.py`**:
```python
FORCE_ONLINE_TIMEOUT_MINUTES = 30    # 已在配置中定义，可复用
```

### 1.3 关键词硬编码

**`app/routes/chat.py`** 第85-88行:
```python
_RESOLVE_KEYWORDS = {"更新", "接受", "对", "是的", "没错", "update", "是", "确认", "合并"}
_IGNORE_KEYWORDS  = {"忽略", "不用", "不", "算了", "保持", "ignore", "不是", "分开"}
```

**建议**: 移至 `constants.py` 或配置文件，便于维护。

### 1.4 TODO 标记

| 文件 | 行号 | 内容 |
|------|-----|------|
| `static/js/app.js` | 302 | `TODO: 需要从消息中判断是否为在线消息` |

---

## 二、代码质量问题

### 2.1 重复代码模式

#### 重复的距离计算逻辑

**`app/routes/chat.py`** 第121-130行:
```python
def _sort_key_l2(hit, weight_l2: float):
    dist = hit.get("adjusted_distance") or hit.get("distance") or 1.0
    return dist * weight_l2

def _sort_key_l1(hit, weight_l1: float):
    dist = hit.get("adjusted_distance") or hit.get("distance") or 1.0
    return dist * weight_l1
```

**建议**: 合并为一个通用函数：
```python
def _calc_sort_score(hit: dict, weight: float) -> float:
    dist = hit.get("adjusted_distance") or hit.get("distance") or 1.0
    return dist * weight
```

#### 重复常量字符串模板

**`src/ramaria/memory/merger.py`** 第124行:
```python
[序号] 时间段 | 氛围：xxx
摘要：xxx
关键词：xxx
```
此字符串模板在 `src/ramaria/memory/prompt_builder.py` 中可能有重复定义。

### 2.2 异常处理

**`src/ramaria/storage/vector_store.py`** 第267-269行:
```python
except Exception as e:
    logger.warning(f"jieba 自定义词典加载失败，使用默认分词 — {e}")
```

**建议**: 记录异常类型，便于调试。

### 2.3 字符串处理不一致

项目中存在 21+ 处 `.strip()` / `.lower()` / `.upper()` 调用，但顺序不统一：
- 有些使用 `t.lower().strip()`
- 有些使用 `t.strip().lower()`

**建议**: 统一为一种模式（推荐 `t.lower().strip()`）

### 2.4 缺少类型注解

部分函数缺少返回类型注解：
- `app/routes/chat.py`: `_detect_conflict_action()` → 应返回 `str | None`
- `app/routes/chat.py`: `_format_rag_results()` → 应返回 `str | None`

---

## 三、低优先级问题

### 3.1 空目录

| 目录 | 用途 | 建议 |
|------|------|------|
| `src/ramaria/backends/` | 预留后端适配目录 | 添加 `README.md` 说明 |
| `.codebuddy/rules/tcb/rules/` | AI 规则目录 | 保持现状 |

### 3.2 安全提示

**`.env.example`** 和 **`static/setup.html`** 中包含示例 API Key:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxx
```
建议添加注释说明这只是占位符。

---

## 四、测试覆盖率

### 4.1 现有测试

| 测试文件 | 覆盖模块 |
|---------|---------|
| `tests/test_tools.py` | 工具模块 |
| `tests/test_rag_pipeline.py` | RAG 链路 |
| `tests/test_embedding_model.py` | 嵌入模型 |
| `tests/test_hybrid_search.py` | 混合检索 |
| `tests/test_chroma.py` | ChromaDB |

### 4.2 建议补充的测试

1. **API 路由测试**: 缺少对 `/chat`, `/memory`, `/sessions` 路由的单元测试
2. **会话管理测试**: `session_manager.py` 缺少测试
3. **路由逻辑测试**: `router.py` 缺少测试

---

## 五、清理建议清单

### P1 近期处理

- [x] **移动关键词配置** 到 `constants.py`:
  - [x] `app/routes/chat.py` 中的 `_RESOLVE_KEYWORDS` 和 `_IGNORE_KEYWORDS`
  - [x] `_CONFLICT_REPLY_MAX_LEN` 移至 `constants.py`
  - [x] `_MAX_HISTORY_MESSAGES` 移至 `config.py` → `MAX_HISTORY_MESSAGES`
- [ ] **统一字符串处理** 模式 (统一为 `t.lower().strip()`)
- [ ] **合并** `_sort_key_l1` 和 `_sort_key_l2` 函数
- [ ] **处理** `static/js/app.js` 中的 TODO

### P2 中期优化

- [ ] **拆分** `src/ramaria/storage/vector_store.py`:
  - 新建 `src/ramaria/storage/bm25_index.py`
  - 新建 `src/ramaria/storage/retrieval.py`
- [ ] **拆分** `src/ramaria/storage/database.py`:
  - 新建 `src/ramaria/storage/db_queries.py`
- [ ] **拆分** `src/ramaria/memory/graph_builder.py`:
  - 新建 `src/ramaria/memory/graph_analyzer.py`
- [ ] **添加** API 路由测试
- [ ] **添加** `src/ramaria/backends/README.md`

### P3 长期改进

- [ ] **引入** 类型检查 (pyright/mypy)
- [ ] **添加** 代码格式化 (black/isort)
- [ ] **完善** 文档字符串
- [ ] **考虑** 引入 `dataclass` 替代字典

---

## 六、已完成的清理 (v0.5.0)

- ✅ 版本号统一为 `0.5.0`，仅在 `pyproject.toml` 中保留
- ✅ 删除根目录遗留文件 `logger.py` 和 `constants.py`
- ✅ 删除所有代码中的硬编码版本号

---

## 七、统计汇总

| 类别 | 数量 |
|------|------|
| 中等问题 | 4 |
| 代码质量问题 | 4 |
| 低优先级问题 | 2 |
| 测试缺失 | 3 |
| P1 已完成 | 1 |
| P1 待处理 | 3 |
| P2 待处理 | 5 |
| P3 待处理 | 4 |
