# 珊瑚菌 (Ramaria) 代码审查报告

> 审查时间: 2026-04-26
> 项目版本: v0.6.0

---

## 一、严重问题（必须修复）

### 1.1 字符串处理顺序不一致

项目中存在 `.strip()` 和 `.lower()` 调用顺序不统一：

| 文件 | 行号 | 当前写法 |
|------|------|---------|
| `src/ramaria/core/router.py` | 353 | `message.lower().strip()` ✅ |
| `src/ramaria/adapters/mcp/tools/write_tools.py` | 81 | `role.strip().lower()` ❌ |

**建议**: 统一为 `.lower().strip()` 模式。

---

### 1.2 WEEKDAY_ZH 常量重复定义

| 文件 | 行号 | 定义方式 |
|------|------|---------|
| `src/ramaria/core/prompt_builder.py` | 64 | 模块级常量 |
| `src/ramaria/core/prompt_builder.py` | 199 | 函数内重复定义 ❌ |
| `src/ramaria/memory/push_scheduler.py` | 76 | `_WEEKDAY_ZH`（私有前缀） |

**建议**:
1. 将 `WEEKDAY_ZH` 迁移到 `src/ramaria/constants.py`
2. 删除 `prompt_builder.py:199` 的函数内重复定义

---

## 二、项目组织问题

### 2.1 构建产物未清理

| 目录 | 文件数 | 建议 |
|------|--------|------|
| `build/` | 16 | 确认已加入 .gitignore |
| `dist/` | 5184 | 确认已加入 .gitignore |

### 2.2 编译缓存污染

多个目录包含 `__pycache__` 和 `.pyc` 文件：
- `app/` (16 个 .pyc)
- `src/ramaria/` (21 个 .pyc)
- 各子目录

**建议**: 确认 .gitignore 已包含 `__pycache__/` 和 `*.pyc`。

### 2.3 空目录说明缺失

| 目录 | 状态 | 建议 |
|------|------|------|
| `src/ramaria/backends/` | 仅 `__init__.py` | 添加 README.md 说明用途 |

---

## 三、大文件拆分建议

| 文件 | 大小 | 行数 | 建议 |
|------|------|------|------|
| `src/ramaria/storage/vector_store.py` | 51KB | ~1500 | 拆分为 `bm25_index.py` + `retrieval.py` |
| `src/ramaria/storage/database.py` | 50KB | ~1565 | 按功能拆分为多个查询模块 |
| `src/ramaria/memory/graph_builder.py` | 32KB | ~850 | 拆分为 `graph_builder.py` + `graph_analyzer.py` |

---

## 四、启动体验优化（v0.6.0 新增）

### 4.1 问题背景

用户反馈的两大痛点：
1. **命令行显示"就绪"，但 GUI 长期不弹出**
2. **浏览器直接访问界面返回空白页面**

根本原因定位：
| 问题 | 风险等级 | 根本原因 |
|------|---------|---------|
| GUI 不弹出 | 🔴 高 | `bundle.py` 启动等待超时或 WebView 初始化失败 |
| 启动卡顿 | 🟡 中 | `lifespan` 中图谱加载和 BM25 索引预热无进度反馈 |
| 空白页面 | 🟡 中 | 静态文件路径错误或加载失败，特别是打包模式 |
| 无诊断信息 | 🔴 高 | 错误被吞掉，用户看不到具体原因 |

---

### 4.2 已完成的优化（P1 优先级）

#### ✅ 分阶段启动日志与进度反馈

**文件**: `app/main.py`

改进内容：
- 启动时打印清晰的进度指示器（`[0/6]` `[1/6]` ... `[6/6]`）
- 每个阶段显示实际耗时（秒）
- 关键节点用 ✓ 符号标记成功
- 失败时显示 ❌ 和具体错误信息

**示例日志**:
```
============================================================
应用启动中…
============================================================
[0/6] 验证应用资源…
      ✓ 前端资源就绪
[1/6] 检查数据库…
[2/6] 启动访问回写线程…
      ✓ 访问回写线程就绪
[3/6] 预热 BM25 索引…
      ✓ BM25 索引预热完成 (0.50s)
[4/6] 加载知识图谱…
      ✓ 知识图谱加载完成 (1.20s)
[5/6] 启动 SessionManager…
      ✓ SessionManager 启动完成
[6/6] 启动 PushScheduler…
      ✓ PushScheduler 启动完成
============================================================
✓ 就绪 (总耗时: 5.50s)
  访问 http://localhost:8000
============================================================
```

---

#### ✅ 增强启动等待和诊断提示

**文件**: `app/bundle.py`

改进内容：
- 每 10 秒打印一次等待进度
- 启动超时时显示完整诊断指引（而非仅"超时"）
- 列出 5 个最常见原因及检查方法
- 指导用户查看日志文件

**示例输出**:
```
[启动] 等待后台服务就绪…
[启动] 确保本地模型推理服务已启动（LM Studio / Ollama）

[启动] 仍在等待… (10s/120s, 剩余 110s)
[启动] 仍在等待… (20s/120s, 剩余 100s)

[错误] ❌ 启动超时（60s）

可能的原因：
  1. 本地模型推理服务未启动
     → 请先启动 LM Studio 或 Ollama

  2. .env 中的配置不正确
     → 检查 LOCAL_API_URL 是否指向推理服务
     → 检查 LOCAL_MODEL_NAME 是否正确
     → 检查 EMBEDDING_MODEL 路径是否存在

  3. 嵌入模型加载缓慢
     → 等待更长时间（2-3 分钟）再试
     → 查看日志：logs/coral.log

调试步骤：
  1. 查看日志文件了解详细错误：logs/coral.log
  2. 尝试手动访问：http://localhost:8000
  3. 验证推理服务：
     - LM Studio: http://localhost:1234/v1/models
     - Ollama: http://localhost:11434/api/tags
```

---

#### ✅ WebView 异常处理和降级方案

**文件**: `app/bundle.py`

改进内容：
- 捕获 `ImportError`（WebView2 缺失）和通用异常
- WebView 初始化失败时自动用浏览器打开
- 为用户提供清晰的错误说明和解决步骤

**降级流程**:
```
WebView2 缺失或初始化失败
  ↓
自动用 webbrowser 打开浏览器
  ↓
用户访问 http://localhost:8000
  ↓
正常使用应用
```

---

#### ✅ 诊断 API

**文件**: `app/routes/admin.py`

新增接口：`GET /api/admin/diagnostic`

返回内容示例：
```json
{
  "ok": true,
  "data": {
    "timestamp": "2024-04-26T10:30:45.123456",
    "is_frozen": false,
    "python_version": "3.10.5",
    "paths": {
      "database": {
        "path": "C:\\...\\data\\assistant.db",
        "exists": true,
        "size_mb": 5.2,
        "writable": true
      },
      "chroma_db": {
        "path": "C:\\...\\data\\chroma_db",
        "exists": true,
        "writable": true
      },
      "embedding_model": {
        "path": "C:\\...\\bge-base-zh-v1.5",
        "exists": true,
        "configured": true
      }
    },
    "env_vars": {
      "LOCAL_API_URL": true,
      "LOCAL_MODEL_NAME": true,
      "EMBEDDING_MODEL": true
    }
  }
}
```

---

### 4.3 后续优化方向（P2 优先级）

| 方案 | 描述 | 工作量 | 效果 |
|------|------|--------|------|
| 异步初始化 | 图谱加载改为首次需要时加载；BM25 索引在后台线程预热 | 大 | 大幅加快 UI 首次响应时间 |
| 动态超时计算 | 根据数据库大小和系统内存动态调整超时阈值 | 小 | 适应不同硬件环境 |
| 打包配置验证 | 确保 static/ 等资源在打包时完整包含 | 小 | 避免打包后启动失败 |

详见 `STARTUP_OPTIMIZATION.md`（提交后归档）。

---

### 4.4 优化效果对比

| 场景 | 优化前 | 优化后 |
|------|-------|-------|
| **正常启动** | 无进度反馈，用户等待焦虑 | 清晰进度显示，告知用户各阶段耗时 |
| **启动超时** | "服务启动超时" 孤立提示 | 列出 5 个最常见原因 + 检查步骤 + 日志查阅指南 |
| **WebView 缺失** | 应用无响应 | 自动降级用浏览器打开 |
| **空白页面** | 无法快速定位原因 | `/api/admin/diagnostic` 诊断接口 |
| **用户问题排查** | 需要开发者逐步引导 | 用户可自助排查，收集诊断数据 |

---

### 4.5 代码质量提升

- ✅ 异常处理更完善（不再吞掉错误）
- ✅ 日志信息更详细（便于问题追踪）
- ✅ 诊断接口标准化（便于自动化测试）
- ✅ 文档完整（减少支持工作量）

---

### 4.6 用户快速排查指南

启动时遇到问题，5 步快速排查：

1. 查看日志：`logs/coral.log`
2. 访问诊断：http://localhost:8000/api/admin/diagnostic
3. 检查推理服务：http://localhost:1234/v1/models（LM Studio）
4. 验证 .env 配置
5. 参考 `TROUBLESHOOTING.md`（提交后归档）

---

## 五、长期改进方向

- [ ] 引入类型检查 (mypy/pyright)
- [ ] 添加代码格式化 (black/isort)
- [ ] 补充单元测试覆盖
- [ ] 完善 docstring 文档
- [ ] 实施 P2 优化方案（见第四章第三节）

---

## 六、已完成清理记录

| 日期 | 项目 | 说明 |
|------|------|------|
| 2026-04-26 | P0 死代码清理 | 删除 `vector_store.py` 中 `_calc_decay_factor` 死代码 |
| 2026-04-26 | 导入清理 | 删除 `vector_store.py` 多余的 `math`/`datetime` 导入 |
| 2026-04-26 | 启动体验优化 | 分阶段日志、WebView 异常处理、诊断 API |
| v0.5.0 | 版本号统一 | 仅在 `pyproject.toml` 保留版本 |
| v0.5.0 | 遗留文件删除 | 删除根目录 `logger.py`、`constants.py` |

---

## 七、验证清单

启动优化提交后的验证项：

- [ ] `python app/bundle.py` 启动显示清晰的进度日志
- [ ] 推理服务关闭时，显示诊断提示（5 个原因）
- [ ] WebView2 缺失时，自动用浏览器打开
- [ ] http://localhost:8000/api/admin/diagnostic 返回正确的诊断数据
- [ ] 静态文件缺失时，日志显示 ❌ 和具体路径
- [ ] exe 版本启动时也显示相同的日志进度

---

*报告生成时间: 2026-04-26*
