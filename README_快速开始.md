# 快速开始

> 本章节面向**首次部署**的用户，按顺序完成以下步骤即可在本地运行珊瑚菌。
> 如果中途遇到报错，请优先查阅文末的[常见问题](#常见问题)。

---

## 环境要求

在开始之前，请确认以下软件已经安装并正常运行：

| 软件 | 最低版本 | 用途 | 下载地址 |
|------|---------|------|---------|
| Python | **3.10+** | 运行后端服务 | https://www.python.org/downloads/ |
| LM Studio 或 Ollama | 任意新版 | 提供本地模型推理服务 | https://lmstudio.ai / https://ollama.com |
| 嵌入模型文件 | — | 向量检索，需提前下载到本地 | 见下方说明 |

**Python 安装注意**：安装时务必勾选 **"Add Python to PATH"**，否则命令行找不到 `python` 命令。

**嵌入模型**：项目默认使用 `Qwen/Qwen3-Embedding-0.6B`。
请提前从 HuggingFace 或镜像站下载模型文件夹，记录其本地绝对路径（例如 `F:\models\Qwen3-Embedding-0.6B`），后续配置 `.env` 时会用到。

---

## 第一步：下载项目

从 GitHub 下载发布包并解压，或使用 Git 克隆：

```bash
git clone https://github.com/entergirl/Ramaria.git
cd Ramaria
```

---

## 第二步：安装依赖

### Windows 用户（推荐）

双击运行项目根目录下的：

```
win\install.bat
```

脚本会自动完成：检查 Python 版本 → 创建虚拟环境 → 安装依赖 → 生成 `.env` → 初始化数据库。

> **如果弹出"Windows 已保护你的电脑"的蓝色警告框**：
> 点击"更多信息" → "仍要运行"即可。这是 Windows 对从网络下载的脚本的默认拦截，
> 不代表文件有问题。或者右键 `install.bat` → 属性 → 勾选"解除锁定" → 确定，再运行。

### Linux / macOS 用户

```bash
bash linux/install.sh   # Linux
bash mac/install.sh     # macOS
```

### 手动安装（任意平台）

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -e .
```

---

## 第三步：配置 `.env`

安装完成后，用任意文本编辑器打开项目根目录的 `.env` 文件（由安装脚本从 `.env.example` 自动生成）。

**必填项**（缺少任意一项都会导致服务启动失败）：

```ini
# 本地模型推理服务地址
# LM Studio 默认地址如下，Ollama 默认为 http://localhost:11434/v1/chat/completions
LOCAL_API_URL=http://localhost:1234/v1/chat/completions

# 模型名称，需与推理服务中实际加载的模型一致
# 在 LM Studio 的 "Local Server" 页面可以查看当前加载的模型名称
LOCAL_MODEL_NAME=qwen/qwen3.5-9b

# 嵌入模型的本地绝对路径（文件夹路径，不是 .bin 文件）
# 示例（Windows）：F:\models\Qwen3-Embedding-0.6B
# 示例（Linux）：/home/user/models/Qwen3-Embedding-0.6B
EMBEDDING_MODEL=这里填入你的嵌入模型文件夹路径
```

> **重要**：`EMBEDDING_MODEL` 必须填写，且路径必须指向已下载完整的模型文件夹。
> 留空或路径不存在会导致服务在启动阶段崩溃，报错位置在 `app/main.py` 的 `lifespan` 函数。

---

## 第四步：启动本地模型服务

在启动珊瑚菌之前，请确保本地推理服务已经运行并加载了模型：

- **LM Studio**：打开软件 → 切换到 "Local Server" 标签页 → 选择模型 → 点击 "Start Server"
- **Ollama**：在命令行运行 `ollama serve`，再用 `ollama run <模型名>` 加载模型

可以用以下命令验证推理服务是否正常：

```bash
# 将地址替换为你的实际配置
curl http://localhost:1234/v1/models
```

返回 JSON 说明服务正常。

---

## 第五步：启动珊瑚菌

### Windows 用户

双击运行：

```
win\start.bat
```

### Linux / macOS 用户

```bash
# 先激活虚拟环境
source venv/bin/activate  # Linux
source venv/bin/activate  # macOS

python app/main.py
```

### 手动启动（任意平台）

```bash
# 确保虚拟环境已激活
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / macOS

python app/main.py
```

---

## 第六步：访问界面

服务启动成功后，终端会输出：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

打开浏览器访问：

- **本机**：http://localhost:8000
- **局域网内其他设备**：http://\<本机IP地址\>:8000

---

## 常见问题

### 安装阶段

---

**Q：`install.bat` 双击直接闪退，什么都看不到。**

原因：脚本报错后窗口立即关闭。
解决：右键 `install.bat` → "在终端中打开"（或先打开命令提示符 `cmd`，`cd` 到项目目录，再输入 `win\install.bat` 回车），这样窗口不会自动关闭，可以看到完整错误信息。

---

**Q：安装时提示 `Microsoft Visual C++ 14.0 or greater is required`。**

原因：部分依赖包（如 `chromadb`）需要在本地编译 C 扩展，要求安装 MSVC 编译工具链。
解决：安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾选"使用 C++ 的桌面开发"工作负载，安装后重新运行 `install.bat`。

---

**Q：`pip install` 过程网络超时或下载失败。**

解决：临时使用国内镜像源：

```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

**Q：提示 `ModuleNotFoundError: No module named 'logger'` 或 `No module named 'constants'`。**

原因：`pyproject.toml` 配置问题，`logger.py` 和 `constants.py` 位于项目根目录，旧版配置没有将其纳入安装范围。
解决：确认使用的是最新的 `pyproject.toml`（v0.3.6-hotfix 版本），然后重新执行：

```bash
pip install -e .
```

---

**Q：提示 `RuntimeError: Form data requires "python-multipart" to be installed`。**

原因：`python-multipart` 库未安装，FastAPI 处理文件上传时依赖此库，旧版 `pyproject.toml` 中漏掉了这一项。
解决：手动安装，或更新到最新 `pyproject.toml` 后重新 `pip install -e .`：

```bash
pip install python-multipart
```

---

### 启动阶段

---

**Q：`start.bat` 双击直接闪退。**

原因：最常见的情况是虚拟环境不存在（没有先运行 `install.bat`）。
解决：先运行 `win\install.bat` 完成安装，再运行 `start.bat`。
如果安装过但仍闪退，同样建议在 `cmd` 里手动运行查看完整报错。

---

**Q：启动时报错，traceback 指向 `app/main.py` 的 `lifespan` 函数。**

`lifespan` 函数在服务启动时依次执行多个初始化步骤，任何一步失败都会导致服务无法启动。
需要查看 traceback **最底部**的那一行错误信息来确定具体原因：

| 底部报错关键词 | 原因 | 解决方法 |
|--------------|------|---------|
| `OSError` / `no such file` / `path` | `EMBEDDING_MODEL` 路径不存在或填写有误 | 检查 `.env` 中 `EMBEDDING_MODEL` 的路径，确保文件夹存在且包含完整的模型文件 |
| `ConnectionRefusedError` / `Cannot connect` | 本地推理服务未启动 | 先启动 LM Studio 或 Ollama 并加载模型 |
| `ModuleNotFoundError` | 依赖未安装或虚拟环境未激活 | 激活虚拟环境后重新 `pip install -e .` |
| `sqlite3.OperationalError` | 数据库文件损坏或未初始化 | 运行 `python scripts/setup_db.py` |

---

**Q：服务启动成功，但打开网页报 `502` 或无法连接。**

原因：浏览器访问太快，服务还在初始化中（嵌入模型首次加载需要数秒到数十秒）。
解决：等待终端输出 `INFO: Application startup complete.` 后再访问。

---

**Q：对话有回复，但回复内容是 `（错误：本地模型调用失败，请确认 LM Studio 服务已启动）`。**

原因：嵌入模型加载成功，但聊天模型推理服务不可用。
解决：
1. 确认 LM Studio / Ollama 正在运行
2. 确认 `.env` 中 `LOCAL_API_URL` 和 `LOCAL_MODEL_NAME` 与推理服务中实际加载的模型一致
3. 用 `curl` 或浏览器访问 `LOCAL_API_URL` 替换为 `/v1/models` 验证服务是否响应

---

**Q：Windows 下中文显示乱码（bat 脚本输出全是问号或方块）。**

原因：bat 文件编码不正确。正确的编码是 **UTF-8 with BOM**，而不是无 BOM 的 UTF-8 或 GBK。
解决：这是发行包本身的问题，请更新到最新版脚本文件。如自行修改过 bat 文件，在 VS Code 中用"以编码保存" → 选择"UTF-8 with BOM"重新保存。

---

### 其他

---

**Q：如何更新到新版本？**

```bash
git pull
# 激活虚拟环境后
pip install -e .
python scripts/setup_db.py  # 幂等操作，可安全重复运行，自动补齐新增的表和字段
```

---

**Q：如何完全重置，清空所有对话记录和记忆？**

```bash
# 删除数据库和向量索引（操作不可逆）
# Windows
del data\assistant.db
rmdir /s /q data\chroma_db

# Linux / macOS
rm data/assistant.db
rm -rf data/chroma_db

# 然后重新初始化数据库
python scripts/setup_db.py
```

---

**Q：防火墙或杀毒软件拦截了服务端口（8000）。**

解决：在防火墙设置中为 Python 或端口 8000 添加入站规则放行，或临时关闭防火墙测试。局域网访问时此问题尤为常见。
