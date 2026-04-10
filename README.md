# 珊瑚菌 · Ramaria

> 大模型懂一切，唯独不懂你。

---

## 项目简介

现有 AI 助手存在一个根本性缺陷：没有记忆。每次对话都从零开始，无论之前交流过多少，下一次都是陌生人。

珊瑚菌以「记忆」为核心，构建一套本地运行的个人 AI 陪伴系统。它不只记住「怎么说话」，而是记住**你经历过什么、关心什么、思考问题的方式是什么**。

---

## 与现有方案的本质差异

| 维度 | 微调方案 | 本项目方案 |
|------|----------|------------|
| 学习内容 | 语气、风格、措辞习惯 | 经历、习惯、情感纹理、思维方式 |
| 记忆方式 | 固化在模型权重中 | 结构化存储，动态检索注入 |
| 可解释性 | 黑盒，无法追溯 | 树状关联，可回溯至原始对话 |
| 更新机制 | 需重新训练 | 实时写入，持续积累 |
| 隐私控制 | 数据上传至训练方 | 全量本地化，用户完全掌控 |

---

## 系统架构

采用「本地推理 + 云端辅助」混合架构。本地负责日常对话处理、隐私数据存储与记忆管理；云端负责超出本地模型能力范围的深度推理任务。

```
用户浏览器
    ↓↑ WebSocket / HTTP
FastAPI 服务 (app/main.py, port:8000)
    ├── 路由层 (app/routes/)
    │     ├── 对话路由 (chat.py)         — WebSocket 防抖 + HTTP 兼容接口
    │     ├── 路由控制 (router_ctrl.py)  — 本地 / 云端模式切换
    │     ├── Session 管理 (sessions.py)
    │     ├── 导入控制 (import_ctrl.py)  — 历史聊天记录导入
    │     ├── 图谱控制 (graph_ctrl.py)   — 知识图谱构建接口
    │     └── 配置接口 (settings.py)
    ├── 核心模块 (src/ramaria/core/)
    │     ├── LLM 请求封装 (llm_client.py)
    │     ├── 任务路由 (router.py)
    │     ├── System Prompt 构建 (prompt_builder.py)
    │     └── Session 生命周期管理 (session_manager.py)
    ├── 记忆模块 (src/ramaria/memory/)
    │     ├── L1 摘要生成 (summarizer.py)
    │     ├── L2 聚合合并 (merger.py)
    │     ├── L3 画像维护 (profile_manager.py)
    │     ├── 冲突检测 (conflict_checker.py)
    │     ├── 知识图谱构建 (graph_builder.py)
    │     └── 主动推送调度 (push_scheduler.py)
    └── 存储层 (src/ramaria/storage/)
          ├── SQLite (data/assistant.db)    — 结构化存储
          ├── ChromaDB (data/chroma_db/)    — 向量索引
          └── NetworkX                      — 知识图谱内存图
```

| 模块 | 技术选型 | 职责 |
|------|----------|------|
| 本地对话模型 | 任意兼容 OpenAI API 的本地推理服务 | 日常对话、摘要生成、记忆合并推理 |
| 向量嵌入模型 | 本地嵌入模型（文件加载，无需额外服务） | 将文本转换为语义向量坐标 |
| 向量数据库 | ChromaDB（本地持久化） | 存储多层向量索引，支持语义检索 |
| 结构化存储 | SQLite | 存储所有原始消息与结构化记忆数据 |
| 知识图谱 | NetworkX | 实体关系图，支持语义图谱检索 |
| 云端推理 | 任意兼容 Anthropic API 的云端模型 | 复杂推理、长文档分析、深度调试 |
| 前端界面 | 内嵌 Web UI（WebSocket 实时通信） | 跨设备对话入口，支持局域网访问 |

### 任务路由策略

- 日常陪伴、情绪对话、记忆摘要生成 → 本地模型
- 复杂 bug 调试、架构设计、长文档深度分析 → 云端推理
- 发送 `/online` 强制切换至云端；`/local` 切回本地；30 分钟无消息自动切回

### WebSocket 通信层

- 主对话通道采用 WebSocket，替代传统轮询 HTTP
- 服务端防抖机制：连续快速发送的消息合并后统一处理（默认等待 3 秒）
- 主动推送调度器：在配置的时间窗口内主动发送消息；用户离线时暂存，上线后自动补发
- 消息队列保障：离线积压消息在连接建立时按时序推送

---

## 分层记忆体系

记忆体系的核心设计理念是：**不只是简单的逐层压缩，而是树状的、相互关联的有机网络**。不同层级之间通过数据库外键保持双向可追溯性。

```
L3 用户画像（长期稳定特征）
    ↑ 提炼自
L2 时间段聚合摘要（多条 L1 合并）
    ↑ 提炼自
L1 单次对话摘要（每次 session 结束后生成）
    ↑ 提炼自
L0 原始消息（永久保留，不删除，不过滤）
```

### L0 — 原始消息层

- 存储所有原始对话消息，永久保留
- 细枝末节的日常闲聊同样写入，保留生活质感与情感纹理
- 按滑动窗口切片向量化，建立语义索引

### L1 — 单次对话摘要层

- 触发时机：对话空闲超过 10 分钟自动触发
- 结构化字段：摘要文本、关键词标签、时间段（六选一）、对话氛围（四字以内）
- 情感元数据：效价（valence）和显著性（salience）双字段，驱动记忆衰减差异化

### L2 — 时间段聚合摘要层

- 条数触发：未吸收的 L1 累计达到 5 条
- 时间触发：最早一条未吸收的 L1 距今超过 7 天
- L1 被吸收后降权保留，支持溯源与回滚

### L3 — 长期用户画像层

按六大板块分行管理：基础信息、个人状况、兴趣爱好、社交情况、历史事件、近期背景

---

## 核心创新设计

### 关键词词典系统

传统方案每次摘要由模型自由发挥关键词，随时间积累会产生大量同义词，导致检索精度持续下降。

本项目维护 `keyword_pool` 词典表，每次 L1 生成时将历史词条作为候选列表喂给模型，引导模型优先复用已有词条——**让关键词随时间收敛而非发散**。

### 分层混合 RAG 检索

三通道融合检索路径：

1. **向量通道**：对 L1/L2 向量索引做语义相似度检索
2. **BM25 通道**：基于 jieba 分词的关键词精确匹配，弥补语义检索对专有名词的不足
3. **图谱通道**：在 NetworkX 知识图谱上做 BFS 遍历，召回实体关联的历史记录

三路结果通过 **RRF（倒数排名融合）** 算法加权合并，并叠加 **Ebbinghaus 遗忘曲线**衰减权重（salience 越高的记忆衰减越慢），最终注入 System Prompt。

### 知识图谱

每条 L1 摘要在生成后会触发三元组提取（主语 → 关系 → 宾语），写入 `graph_nodes` 和 `graph_edges` 表。关系类型覆盖任务状态、遇到障碍、使用依赖、情感状态等七大类。

实体归一化采用向量相似度三档策略：高置信度自动合并别名、中置信度转交用户确认、低置信度独立为新词。NetworkX 图对象在服务启动时加载至内存，图谱检索无需重复查询数据库。

### L3 画像半自动维护

L1 生成后，系统自动从摘要中提取新信息静默写入画像，发现矛盾时才以关心口吻询问用户：

> 「之前记得你说毕业设计压力很大，但今天好像轻松很多了？是顺利了吗」
>
> 你可以回复「更新」让我记住新的情况，或者回复「忽略」保持现状。

冲突不是错误，而是用户状态变化的信号。

### MCP Server 集成

提供标准 MCP Server 接口，支持通过 Claude Desktop 等 MCP 客户端直接访问珊瑚菌的记忆系统。包含七个工具，覆盖记忆检索、画像读取、摘要触发等核心操作。详细文档见 [`docs/mcp-server.md`](docs/mcp-server.md)（待补充）。

---

## 文件结构

```
ramaria/
├── app/                             # FastAPI 应用层
│   ├── main.py                      # 服务入口，lifespan 管理
│   ├── dependencies.py              # 全局单例（SessionManager、Router、WebSocket 连接池）
│   └── routes/                      # 路由模块
│       ├── chat.py                  # 对话路由（WebSocket + HTTP）
│       ├── router_ctrl.py           # 模式切换接口
│       ├── sessions.py              # Session 查询接口
│       ├── import_ctrl.py           # 聊天记录导入接口
│       ├── graph_ctrl.py            # 知识图谱接口
│       └── settings.py              # 配置读写接口
├── src/ramaria/                     # 核心业务包
│   ├── config.py                    # 全局配置中心
│   ├── core/                        # 核心模块
│   │   ├── llm_client.py            # LLM 请求封装（本地 / 云端统一接口）
│   │   ├── router.py                # 任务路由层
│   │   ├── prompt_builder.py        # System Prompt 构建
│   │   └── session_manager.py       # Session 生命周期管理
│   ├── memory/                      # 记忆模块
│   │   ├── summarizer.py            # L1 摘要生成
│   │   ├── merger.py                # L2 聚合合并
│   │   ├── profile_manager.py       # L3 画像半自动维护
│   │   ├── conflict_checker.py      # 记忆冲突检测
│   │   ├── graph_builder.py         # 知识图谱批处理
│   │   └── push_scheduler.py        # 主动推送调度器
│   ├── storage/                     # 存储层
│   │   ├── database.py              # SQLite 操作封装
│   │   └── vector_store.py          # 向量索引与混合检索
│   ├── adapters/                    # 外部接口适配
│   │   ├── mcp/                     # MCP Server
│   │   └── telegram/                # Telegram Bot 桥接（可选）
│   └── importer/                    # 历史数据导入
│       ├── qq/                      # QQ 聊天记录解析与写入
│       └── batch.py                 # L1 批量生成
├── scripts/                         # 数据库管理脚本
│   ├── setup_db.py                  # 一键初始化（含全部迁移）
│   ├── init_db.py                   # 数据库建表
│   └── migrate_*.py                 # 各版本迁移脚本
├── static/                          # 前端静态资源
│   ├── index.html                   # 主对话界面
│   ├── import.html                  # 聊天记录导入界面
│   ├── css/                         # 样式文件
│   └── js/                          # 前端脚本（state / api / ui / app）
├── config/
│   └── persona.toml.example         # 人设配置模板
├── tests/                           # 测试脚本
├── data/                            # 运行数据（不上传）
│   ├── assistant.db                 # SQLite 数据库
│   └── chroma_db/                   # 向量索引
├── logs/                            # 运行日志（不上传）
├── .env.example                     # 环境变量模板
├── pyproject.toml                   # 项目元信息与依赖
└── README.md
```

---

## 数据库结构

| 表名 | 用途 |
|------|------|
| `sessions` | 对话 session 生命周期管理 |
| `messages` | L0 原始消息流水，永久保留 |
| `memory_l1` | 单次对话摘要（L1 层），含情感元数据 |
| `memory_l2` | 时间段聚合摘要（L2 层） |
| `l2_sources` | L2 溯源关联，记录合并来源 |
| `user_profile` | 长期用户画像（L3 层） |
| `keyword_pool` | 关键词词典，支持复用与频次统计，含别名归一化 |
| `graph_nodes` | 知识图谱实体节点 |
| `graph_edges` | 知识图谱关系边，含三元组溯源 |
| `conflict_queue` | 冲突检测待确认队列 |
| `pending_push` | 主动推送消息暂存（用户离线时缓冲） |
| `settings` | 全局运行配置 |

---

## 快速开始

### 环境要求

- **Python 3.10+**
- **本地模型推理服务**（如 LM Studio、Ollama 等，兼容 OpenAI API 格式即可）
- **本地嵌入模型文件**（需提前下载至本地路径）
- **云端推理 API Key**（可选，不配置不影响核心功能）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/entergirl/Ramaria.git
cd Ramaria

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. 安装依赖
pip install -e .

# 4. 配置环境变量（可选，使用云端推理时需要）
cp .env.example .env
# 编辑 .env，填入 API Key
```

### 配置模型

#### 对话模型（必须）

启动本地模型推理服务，加载你选择的对话模型，确认服务运行在本地某个端口（默认 `http://localhost:1234`）。

在 `src/ramaria/config.py` 中配置：

```python
LOCAL_API_URL    = "http://localhost:1234/v1/chat/completions"  # 推理服务地址
LOCAL_MODEL_NAME = "your-model-name"                            # 模型标识符
```

#### 嵌入模型（必须）

嵌入模型用于将文本转换为语义向量，程序直接加载本地文件，**无需启动额外服务**。

将模型文件夹放置到本地任意路径，然后在 `src/ramaria/config.py` 中配置：

```python
EMBEDDING_MODEL = r"C:\your\path\to\embedding-model"  # 改为你的模型文件路径
```

#### 云端推理 API（可选）

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-api-key"

# macOS / Linux
export ANTHROPIC_API_KEY="your-api-key"
```

也可以直接写入 `.env` 文件。

### 配置人设

```bash
cp config/persona.toml.example config/persona.toml
# 编辑 persona.toml，填写角色设定
```

### 初始化并启动

```bash
# 一键初始化数据库（首次运行，包含全部建表与迁移）
python scripts/setup_db.py

# 启动服务
python app/main.py
```

浏览器访问 `http://localhost:8000` 开始对话。

局域网其他设备访问：`http://[电脑IP]:8000`

---

## 开发阶段规划

### 阶段一（已完成）

核心对话链路验证，L1 / L2 / L3 主线稳定运行，关键词词典系统，记忆冲突检测，分层 RAG 检索基础框架。

### 阶段二（已完成）

任务路由层，本地嵌入模型接入，L3 画像半自动维护，分层 RAG 检索全线接通，情感元数据（valence / salience）与 Ebbinghaus 衰减融合，BM25 + 向量 + RRF 混合检索。

### 阶段三（已完成）

知识图谱（三元组提取、实体归一化、NetworkX 图谱检索），WebSocket 通信层（服务端防抖、主动推送调度器、离线消息队列），历史聊天记录导入（QQ Chat Exporter v5 格式），MCP Server（七工具，支持 Claude Desktop 集成）。

### 阶段四（规划中）

- LoRA 微调：基于历史对话数据蒸馏，对本地模型进行角色对齐微调
- 前端记忆可视化：衰减曲线展示、知识图谱菌丝网络视图
- 多平台接入：Telegram Bot 桥接完善，移动端适配
- 嵌入模型 CUDA 加速

---

## 隐私说明

所有对话数据均存储在本地，不上传至任何服务器。云端推理仅在用户主动切换或系统判断需要深度推理时调用，调用时只传入当前消息内容，不携带任何历史记忆数据。
