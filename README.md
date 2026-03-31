 # 珊瑚菌 · Ramaria

> 目标不是一个更聪明的助手，而是一个**真正认识你**的虚拟伙伴。

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

采用「本地 Agent + 云端 AI」混合架构。本地负责日常对话处理、隐私数据存储与记忆管理；云端负责超出本地模型能力范围的深度推理任务。

```
用户浏览器
    ↓↑ HTTP
FastAPI 服务 (main.py, port:8000)
    ├── 路由判断 (router.py)
    │     ├── 本地 qwen/qwen3.5-9b (LM Studio, localhost:1234)
    │     └── Claude API (复杂任务，用完即回本地)
    ├── LLM 请求封装 (llm_client.py)
    ├── 记忆注入 (prompt_builder.py)
    ├── Session 管理 (session_manager.py)
    └── 数据层
          ├── SQLite (assistant.db) — 结构化存储
          └── ChromaDB (chroma_db/) — 向量索引
```

| 模块 | 技术选型 | 职责 |
|------|----------|------|
| 本地对话模型 | qwen/qwen3.5-9b（LM Studio） | 日常对话、摘要生成、记忆合并推理 |
| 向量嵌入模型 | Qwen3-Embedding-0.6B | 将文本转换为语义向量坐标 |
| 向量数据库 | Chroma（本地持久化） | 存储多层向量索引，支持语义检索 |
| 结构化存储 | SQLite | 存储所有原始消息与结构化记忆数据 |
| 云端推理 | Claude API | 复杂推理、长文档分析、深度代码调试 |
| 调度层 | Python 脚本 | 任务路由、记忆注入、检索协调 |
| 前端界面 | 内嵌 Web UI | 跨设备对话入口，局域网 / Tailscale 访问 |

### 任务路由策略

- 日常陪伴、情绪对话、记忆摘要生成 → 本地 Qwen
- 复杂 bug 调试、架构设计、长文档深度分析 → Claude API
- 发送 `/online` 强制切换至云端；`/local` 切回本地；30分钟无消息自动切回

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

### 分层树状 RAG 检索

三层穿透检索路径：

1. **关键词匹配**：用当前对话内容匹配关键词词典，找到相关语义标签
2. **语义检索**：在 L1/L2 向量索引中做语义检索，定位相关摘要节点
3. **原始回溯**：通过外键穿透至 L0，召回原始对话的具体片段（以最相关消息为中心，前后各取 N 条）

### L3 画像半自动维护

L1 生成后，系统自动从摘要中提取新信息静默写入画像，发现矛盾时才以关心口吻询问用户，不做逐条确认打扰：

> 「之前记得你说毕业设计压力很大，但今天好像轻松很多了？是顺利了吗」
>
> 你可以回复「更新」让我记住新的情况，或者回复「忽略」保持现状。

冲突不是错误，而是用户状态变化的信号。

---

## 文件结构

```
demo/
├── tests/
│   ├── test_chroma.py           # Chroma 向量库功能测试
│   └── test_qwen3_embedding.py  # Qwen3 嵌入模型测试
├── static/                      # 前端静态资源
│   ├── index.html               # 对话页面
│   ├── import.html              # 聊天记录导入页面
│   ├── css/                     # 样式文件
│   └── js/                      # 前端脚本
├── logs/                        # 运行日志（不上传）
├── importer/                    # 外部聊天记录导入模块
│   ├── qq_parser.py             # QQ 聊天记录解析器
│   ├── qq_importer.py           # QQ 导入写入层
│   └── l1_batch.py              # L1 批量生成后台任务
├── mcp_server/                  # MCP 服务器（可选）
│   ├── server.py                # MCP 服务入口
│   └── tools/                   # MCP 工具集
├── main.py                      # FastAPI 服务入口
├── config.py                    # 全局配置中心
├── database.py                  # 数据库操作层
├── init_db.py                   # 数据库初始化脚本
├── llm_client.py                # LLM 请求封装层（本地/Claude 统一接口）
├── logger.py                    # 日志配置模块
├── session_manager.py           # Session 生命周期管理
├── summarizer.py                # L1 摘要生成模块
├── merger.py                    # L2 合并模块
├── prompt_builder.py            # System Prompt 构建模块
├── conflict_checker.py          # 记忆冲突检测模块
├── profile_manager.py           # L3 画像半自动维护模块
├── vector_store.py              # 向量索引与检索模块
├── router.py                    # 任务路由层
├── telegram_bridge.py           # Telegram Bot 桥接服务（可选）
├── persona.toml                 # 人设配置文件
└── README.md
```

---

## 数据库结构

| 表名 | 用途 |
|------|------|
| `sessions` | 对话 session 生命周期管理 |
| `messages` | L0 原始消息流水，永久保留 |
| `memory_l1` | 单次对话摘要（L1 层） |
| `memory_l2` | 时间段聚合摘要（L2 层） |
| `l2_sources` | L2 溯源关联，记录合并来源 |
| `user_profile` | 长期用户画像（L3 层） |
| `keyword_pool` | 关键词词典，支持复用与频次统计 |
| `conflict_queue` | 冲突检测待确认队列 |
| `settings` | 全局运行配置 |

---

## 快速开始

### 环境要求

- **Python 3.10+**
- **LM Studio** — 本地对话模型运行环境，[下载地址](https://lmstudio.ai/)
- **Qwen3-Embedding-0.6B 模型文件** — 向量嵌入模型，从 [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 下载，需本地放置
- **Claude API Key**（可选，用于复杂任务云端推理）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/ramaria.git
cd ramaria

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量（可选，使用 Claude API 时需要）
cp .env.example .env
# 编辑 .env，填入你的 Claude API Key
```

### 配置模型

项目依赖两类模型，它们的运行方式不同，请分别配置：

#### 对话模型 — LM Studio（必须）

对话模型通过 LM Studio 以 API 服务方式运行，负责日常对话和摘要生成。

1. 打开 [LM Studio](https://lmstudio.ai/)
2. 搜索并下载 `qwen/qwen3.5-9b`
3. 加载模型后，点击「Start Server」启动本地 API 服务
4. 确认服务运行在 `http://localhost:1234`

#### 嵌入模型 — 本地文件（必须）

嵌入模型用于将文本转换为语义向量，程序直接加载本地文件，**不需要启动额外服务**。

1. 从 [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 下载模型文件
2. 将模型文件夹放置到本地任意路径（如 `F:\models\Qwen3-Embedding-0.6B`）
3. 在 `config.py` 中将路径指向你的模型位置：

```python
# config.py 中需要修改的参数
LOCAL_API_URL    = "http://localhost:1234/v1/chat/completions"  # LM Studio 地址
LOCAL_MODEL_NAME = "qwen/qwen3.5-9b"                           # LM Studio 加载的模型名
EMBEDDING_MODEL  = r"F:\models\Qwen3-Embedding-0.6B"           # ← 改为你的嵌入模型路径
```

> **提示**：首次启动时程序会自动加载嵌入模型权重（约 1.2GB），请确保路径正确。模型需提前下载到本地，离线环境也可运行。

#### Claude API（可选）

不配置 Claude API 不影响核心功能，仅在需要复杂推理时使用。

```bash
# Mac/Linux
export ANTHROPIC_API_KEY=sk-ant-xxxxxx

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-xxxxxx"
```

或编辑 `.env` 文件，填入 API Key。

### 初始化并启动

```bash
# 初始化数据库（首次运行）
python init_db.py

# 启动服务
python main.py
```

浏览器访问 `http://localhost:8000` 开始对话。

局域网其他设备访问：`http://[电脑IP]:8000`

---

## 开发阶段规划

- **阶段一（已完成）**：核心对话链路跑通，L1/L2/L3 主线稳定运行，关键词词典，冲突检测，分层 RAG 检索
- **阶段二（已完成）**：任务路由层，Qwen3-Embedding 嵌入模型，L3 画像半自动维护，分层 RAG 检索全线接通
- **阶段三（进行中）**：外部聊天记录导入（QQ Chat Exporter），MCP 服务器，Telegram Bot 桥接

---

## 隐私说明

所有对话数据均存储在本地，不上传至任何服务器。Claude API 仅在用户主动切换或系统判断需要深度推理时调用，调用时只传入当前消息，不携带任何历史记忆数据。
