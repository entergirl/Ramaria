# 变更日志

## [0.6.0] - 2026-04-25

### 新增功能

#### EXE 一键分发
- 打包为 PyInstaller 文件夹模式 exe，用户无需安装 Python 或任何依赖，双击即可使用
- `ramaria.spec` 不再打包开发者 `.env`，仅包含 `.env.example` 模板，避免配置泄露
- `app/bundle.py` 作为桌面版主入口，集成 pywebview 窗口 + 系统托盘 + 后台 FastAPI 服务
- 打包模式下 `db_initializer.py` 改为进程内调用 `setup_db.main()`，避免 subprocess 执行 `.py` 文件失败
- 打包模式下自动跳过 venv / 依赖检测（exe 已包含全部运行时）

#### 首次启动配置向导
- 首次运行时自动进入配置向导页面（5 步引导：推理服务 → 嵌入模型 → 人格配置 → 可选配置 → 完成）
- `env_checker.py` 路径常量改为延迟计算 + 缓存，打包模式下从 `ramaria.config` 动态获取正确路径
- `admin.py` 路径常量区分打包/开发模式，静态文件从 `_MEIPASS` 加载，用户数据从 exe 同级目录加载
- 配置向导完成后自动初始化数据库，随后跳转聊天界面

#### 配置热更新
- `write_env()` 写入 `.env` 文件后同步更新 `os.environ`，当前进程立即可读
- `admin_save_config()` 保存配置后调用 `_reload_config_values()` 刷新 `ramaria.config` 模块级常量
- `llm_client.py` 改用 `from ramaria import config as _config` 模块引用，不再绑定 import 时的旧值
- 模型名称、API 地址等配置修改后无需重启服务即可生效

---

### 修复

#### 打包模式启动报错（UnicodeEncodeError）
- 修复 `setup_db.py` 中的 emoji 字符（`✅`/`❌`）在 Windows GBK 控制台下的编码错误
- 开发模式 subprocess 调用添加 `PYTHONIOENCODING=utf-8` 环境变量

#### 首次运行跳过配置向导
- 修复 `.env` 被打包进 exe 导致首次运行时误判配置完整的问题
- 修复 `bundle.py` `_load_dotenv()` 将开发者配置复制到用户目录的问题
- 首次运行时不再创建空 `.env`，让 `can_start_directly()` 自然判定需要引导

#### 模型名称配置不生效
- 修复配置向导保存 `LOCAL_MODEL_NAME` 后仍使用默认值 `qwen/qwen3.5-9b` 的问题
- 根因：`write_env()` 只写文件不更新 `os.environ`，`ramaria.config` 模块变量不刷新，`llm_client.py` 使用 import 绑定旧值

#### 思考模式干扰对话输出
- 修复 gemma 等模型默认启用思考模式后，`<think〉...<／think〉` 标签出现在对话回复中的问题
- 新增 `_remove_think_tags()` 通用函数，在 `_call()` 中统一剥离思考链标签
- `strip_thinking()` 拆分为标签剥离 + JSON 截断两步，摘要类任务继续使用，对话类任务只做标签剥离
- 预编译正则 `_THINK_TAG_RE` 提升匹配性能

#### 路径分隔符不统一
- 向量模型路径提示信息中 `\` 统一替换为 `/`，保持跨平台一致性
- 配置向导输入时实时将 `\` 转换为 `/`
- `.env.example` 路径示例改为正斜杠格式

---

### 改进优化

#### 数据库列级校验
- `check_database()` 新增关键列校验，检测 v0.3.x 之前数据库缺少的新增字段
- 缺失字段时返回具体修复建议，引导运行 `setup_db.py` 迁移

#### 端口检测修复
- 修复 FastAPI 服务启动后 `check_port()` 永远报告端口被占用的问题
- 使用 psutil 检查占用进程是否为自身，本进程占用视为正常

---

## [0.5.0] - 2026-04-19

### 新增功能

#### 感知工具集（Tool Registry）
- 新增工具意图检测与分发中心 `src/ramaria/tools/tool_registry.py`
- 基于语义相似度（嵌入模型）判断用户消息是否需要调用感知工具
- 支持三种感知工具：
  - **硬件状态监控** (`hardware_monitor.py`)：读取 CPU、内存、GPU、电池等系统信息
  - **文件系统扫描** (`fs_scanner.py`)：扫描指定目录，生成目录树文本注入上下文
  - **天气查询** (`weather.py`)：获取当前天气数据，支持自动定位
- 防抖设计：硬件感知每 60 秒最多触发一次
- 路径提取智能识别：支持引号包裹路径（推荐）、Windows/Linux 绝对路径、波浪号路径
- 路径提取失败时自动提示用户使用引号包裹路径

#### Telegram Bot 桥接
- 新增 `src/ramaria/adapters/telegram/bridge.py`
- 支持 Telegram 消息与珊瑚菌双向同步
- 支持私聊和群组消息处理

#### 网页内容获取
- 新增 `web_fetcher.py` 工具，支持获取网页内容并转换为 Markdown

---

### 改进优化

#### 文件系统扫描路径提取
- 优化 `extract_path_from_message` 函数正则表达式
- 优先级1：引号包裹的路径（最可靠，支持中文路径）
- 优先级2：使用 `\S+` 匹配非空白字符，支持中文路径
- 失败时返回友好提示，引导用户使用引号包裹路径

#### 安装流程简化
- Windows 安装脚本统一放在 `win/` 目录，无需复制到根目录
- 简化 README 和快速开始文档中的安装步骤

---

## [0.4.0] - 2026-04-15

### 新增功能

#### 记忆可视化（Memory Viewer）
- 新增独立页面 `/static/memory.html`，可通过主界面侧边栏"🧠 记忆查看"入口跳转
- L1 标签页：分页展示所有单次对话摘要，每条显示实时计算的记忆留存百分比、
  情感显著性、时间段、氛围、是否已被 L2 吸收，支持删除
- L2 标签页：分页展示所有聚合摘要，显示覆盖时间段和记忆留存百分比，支持删除
- L3 标签页：只读展示六大板块用户画像（基础信息、近期状态、兴趣爱好、
  社交情况、历史事件、近期背景）
- 删除采用二次点击确认机制（第一次变红提示，3 秒内再次点击执行），防止误删
- L1 删除时若已被 L2 引用，返回 409 冲突提示，需先删对应 L2
- L2 删除同时清理 `l2_sources` 关联行（不修改来源 L1 的 absorbed 标记）
- 删除操作 Chroma 向量索引与 SQLite 记录双向同步删除

#### 启动前健康检查（Health Check）
- 新增 `scripts/health_check.py`，启动前自动验证五项：
  venv 完整性（pip check）、.env 必填项、嵌入模型路径、
  本地推理服务可达性、数据库完整性（PRAGMA integrity_check + 12 张表存在性）
- 任一检查失败立即退出并输出具体修复步骤，退出码 0 表示全部通过
- `win/start.py` 集成健康检查，作为子进程调用，失败时终止启动
- 可单独运行 `python scripts/health_check.py` 进行手动排障

#### 后端记忆查看接口
- `GET  /api/memory/l1`：L1 分页列表，含实时衰减值 R
- `GET  /api/memory/l2`：L2 分页列表，含实时衰减值 R
- `GET  /api/memory/profile`：L3 画像六板块
- `DELETE /api/memory/l1/{id}`：删除单条 L1
- `DELETE /api/memory/l2/{id}`：删除单条 L2

---

### 性能优化

#### BM25 增量更新
- 新增 `BM25_INCREMENTAL_THRESHOLD`（默认 10）和
  `BM25_REBUILD_INTERVAL`（默认 300 秒）两个配置项
- `BM25Index` 新增 `_pending` 缓冲区，写入时先追加缓冲，
  不再每次触发全量重建
- 缓冲区达到阈值或定时器触发时合并重建；重建期间旧索引继续服务，
  完成后原子替换（`threading.RLock` 保护）
- 新增后台定时重建线程 `BM25TimerRebuilder`，
  由 `main.py` lifespan 管理启停
- 单条写入延迟从约 200ms 降至 <10ms

#### 知识图谱增量更新
- 新增 `_add_edge_to_graph()` 函数，三元组写入数据库后同步增量更新
  NetworkX 内存图，无需全量重载
- 节点已存在时只更新 `use_count`，边已存在时追加 `l1_ids`
- `_nx_graph` 所有读写操作统一用 `_nx_graph_lock`（`threading.RLock`）保护
- `reload_graph()` 保留用于服务启动和手动触发，不再在每次三元组写入后调用

#### 数据库连接复用
- `get_all_l1()` 和 `get_all_l2()` 新增可选 `conn` 参数，
  允许调用方传入外部连接复用，减少 BM25 重建时的重复开关连接开销
- 接口向后兼容，不传参数时行为与原版完全一致

---

### 工程改善

#### logger / constants 包迁移
- 将根目录 `logger.py` 迁移至 `src/ramaria/logger.py`
- 将根目录 `constants.py` 迁移至 `src/ramaria/constants.py`
- 全项目统一使用 `from ramaria.logger import get_logger` 和
  `from ramaria.constants import ...` 导入
- 清理 `pyproject.toml` 中 v0.3.6-hotfix 的补丁配置，
  恢复标准 `where = ["src"]` 简洁写法
- `pip install -e .` 后无需额外操作，`ModuleNotFoundError: No module named 'logger'`
  问题彻底修复

---

## [0.3.6] - 2026-04-12

### 核心特性

#### 分层记忆系统
- L0 原始消息层：完整保留每一轮对话，支持滑动窗口语义索引
- L1 单次摘要层：对话空闲自动生成结构化摘要，包含关键词标签、情感元数据
- L2 聚合摘要层：多条 L1 智能合并为时间段总结，支持溯源回滚
- L3 用户画像层：六大维度长期画像自动维护，冲突检测确保一致性

#### 关键词词典系统
通过复用已有词条引导模型生成收敛的关键词体系，避免同义词发散，确保长期检索精度。

#### 混合检索架构
三通道融合检索：向量通道（ChromaDB）、BM25 通道（jieba 分词）、
图谱通道（NetworkX），三路结果通过 RRF 算法加权合并，
叠加 Ebbinghaus 遗忘曲线衰减权重。

#### 知识图谱
从 L1 摘要自动提取三元组，实体归一化三档策略，NetworkX 图对象内存加载。

#### 通信层
WebSocket 实时双向通信，服务端防抖机制，主动推送调度器，离线消息暂存补发。

#### MCP Server
标准 MCP Server 接口，七个工具，支持 Claude Desktop 集成。

#### 外部集成
Telegram Bot、QQ 聊天记录导入（Chat Exporter v5）、Docker 部署。

---