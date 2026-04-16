# 快速开始

本章节帮助你从零完成珊瑚菌的安装与启动。按顺序执行每个步骤，遇到报错请直接查阅文末的[常见问题](#常见问题)。

---

## 目录

- [前置准备](#前置准备)
- [第一步：获取项目](#第一步获取项目)
- [第二步：安装依赖](#第二步安装依赖)
- [第三步：配置 .env](#第三步配置-env)
- [第四步：准备嵌入模型](#第四步准备嵌入模型)
- [第五步：启动推理服务](#第五步启动推理服务)
- [第六步：启动珊瑚菌](#第六步启动珊瑚菌)
- [第七步：访问界面](#第七步访问界面)
- [常见问题](#常见问题)
- [更新与重置](#更新与重置)

---

## 前置准备

开始之前，请确认以下软件已安装：

| 软件 | 最低版本 | 说明 | 下载 |
|------|---------|------|------|
| Python | **3.10+** | 运行后端服务 | https://www.python.org/downloads/ |
| LM Studio 或 Ollama | 最新版 | 本地模型推理服务 | https://lmstudio.ai · https://ollama.com |

**Python 安装注意（Windows）**：安装时必须勾选 **"Add Python to PATH"**，否则命令行找不到 `python` 命令，所有脚本都无法运行。

---

## 第一步：获取项目

从 GitHub 下载发布包解压，或使用 Git 克隆：

```bash
git clone https://github.com/entergirl/Ramaria.git
cd Ramaria
```

---

## 第二步：安装依赖

根据你的操作系统选择对应方式。

### Windows

  1. 在根目录下文件夹 `win` 中双击 `check_python.bat` 检测python环境，确保python语句可以正常运行
  2. 将 `install.py` 与 `start.py` 两个文件完整的复制并粘贴到根目录下
  3. 进入项目根目录，右键窗口空白处选择 `在终端中打开`
  4. 输入以下命令运行启动文件

```python
#在项目根目录运行
python install.py
```


### Linux

```bash
bash linux/install.sh
```

### macOS

```bash
bash mac/install.sh
```

---

安装脚本会自动完成以下工作：

1. 检查 Python 版本
2. 创建 `venv` 虚拟环境
3. 安装全部依赖（`pip install -e .`）
4. 从 `.env.example` 生成 `.env`
5. 初始化数据库

**安装耗时约 3~10 分钟**，主要取决于网络速度。网络不稳定时重新运行脚本即可，pip 会自动跳过已安装的包。

---

## 第三步：配置 .env

安装完成后，用任意文本编辑器打开项目根目录的 `.env` 文件。

**三个必填项**，缺少任意一项都会导致启动失败：

```ini
# 本地模型推理服务地址
# LM Studio 默认：http://localhost:1234/v1/chat/completions
# Ollama 默认：   http://localhost:11434/v1/chat/completions
LOCAL_API_URL=http://localhost:1234/v1/chat/completions

# 模型名称，必须与推理服务中实际加载的模型名称完全一致
# 在 LM Studio 的 Local Server 页面可以查看当前加载的模型名称
LOCAL_MODEL_NAME=qwen/qwen3.5-9b   #(LM studio) 
# 或者
LOCAL_MODEL_NAME=qwen3.5:9b        #(Ollama)

# 嵌入模型的本地文件夹路径（第四步会说明如何获取）
# Windows 示例：F:\models\Qwen3-Embedding-0.6B
# Linux/macOS 示例：/home/user/models/Qwen3-Embedding-0.6B
EMBEDDING_MODEL=
```

> **`EMBEDDING_MODEL` 不能留空**，这是目前最常见的启动失败原因，请务必完整填写，详见下一步。

---

## 第四步：准备嵌入模型

嵌入模型用于将文本转换为向量，是记忆检索功能的基础。**必须提前下载到本地**，不能留空或填写不存在的路径。

项目默认使用 `Qwen/Qwen3-Embedding-0.6B`（约 600MB）。

### 下载方法

**方法一：使用脚本下载**

激活虚拟环境后运行：

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

然后运行：

```python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-Embedding-0.6B',
    local_dir='./models/Qwen3-Embedding-0.6B'
)
print('下载完成')
"
```

**国内用户**在运行前先设置镜像站环境变量：

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows（命令提示符）
set HF_ENDPOINT=https://hf-mirror.com
```

**方法二：手动下载 （推荐）**

访问 https://hf-mirror.com/Qwen/Qwen3-Embedding-0.6B ，点击页面上的文件列表，将全部文件下载后放入同一个文件夹（如 `F:\models\Qwen3-Embedding-0.6B`）。

### 填写路径

下载完成后，将模型**文件夹**的完整绝对路径填入 `.env`：

```ini
EMBEDDING_MODEL=F:\models\Qwen3-Embedding-0.6B
```

**验证路径正确的方法**：打开该路径对应的文件夹，里面应能看到 `config.json`、`tokenizer.json` 等文件。如果这些文件存在，路径就是正确的。

---

## 第五步：启动推理服务

启动珊瑚菌之前，本地推理服务必须已经运行并加载了模型。

### LM Studio

1. 打开 LM Studio
2. 左侧切换到 **"Local Server"** 标签页
3. 选择模型（如 Qwen3.5-9B）
4. 点击 **"Start Server"**
5. 确认底部状态栏显示 **"Running"**

### Ollama

```bash
# 首次使用，拉取模型（以 qwen2.5:7b 为例）
ollama pull qwen2.5:7b

# 启动服务（保持此终端不关闭）
ollama serve
```

### 验证推理服务正常

在浏览器中访问（地址换成你的实际配置）：

```
http://localhost:1234/v1/models
```

返回 JSON 数据说明推理服务运行正常。无法访问则说明推理服务尚未启动。

---

## 第六步：启动珊瑚菌

### Windows

 1. 通过启动脚本启动
```
python start.py
```

 2. 跳过配置检测直接启动
```
python app/main.py
```

### Linux

```bash
bash linux/start.sh
```

### macOS

```bash
bash mac/start.sh
```

启动脚本在运行前会自动检查：虚拟环境、`.env` 必填项、嵌入模型路径、数据库，任何一项有问题都会给出具体提示，并在提示处停住等待用户操作。

---

## 第七步：访问界面

服务启动成功后，终端会出现：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**等到出现 `Application startup complete.` 后再打开浏览器**，这是关键。过早访问会得到 502 或连接拒绝错误。

| 场景 | 地址 |
|------|------|
| 本机访问 | http://localhost:8000 |
| 局域网内其他设备 | http://\<本机IP\>:8000 |

---

## 常见问题

### 安装阶段

---

**Q：提示 `python: command not found` 或 `'python' 不是内部命令`**

Windows 安装 Python 时未勾选"Add Python to PATH"。

解决：重新安装 Python，安装界面第一步勾选 **"Add Python to PATH"**，然后重新打开命令提示符再运行脚本。

---

**Q：`install.py` 双击后窗口一闪而过**

Python 没有关联 `.py` 文件，或脚本在极早期就报错了。

解决：打开命令提示符（`cmd`），`cd` 到项目目录，手动运行 `python install.py`，这样窗口不会关闭，可以看到完整报错。

---

**Q：Linux 运行 `install.sh` 提示 venv 相关错误**

部分 Linux 发行版的 Python 未自带 venv 模块。

解决：
```bash
# Ubuntu / Debian
sudo apt install python3-venv python3-pip

# Fedora
sudo dnf install python3-virtualenv
```

---

**Q：macOS 提示 Python 版本过低**

macOS 系统自带的 Python 通常是 3.9 以下。

解决：通过 Homebrew 安装新版本：
```bash
# 先安装 Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Python 3.11
brew install python@3.11
```

---

**Q：安装时提示 `Microsoft Visual C++ 14.0 or greater is required`（Windows）**

部分依赖包需要本地 C++ 编译工具链。

解决：安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾选"使用 C++ 的桌面开发"，安装完成后重新运行 `python install.py`。

---

**Q：安装时提示 `error: externally-managed-environment`（Linux）**

说明脚本使用的是系统 Python 而非 venv 内的 Python。

解决：重新运行安装脚本。如果问题持续，手动执行：
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

**Q：安装时网络超时或下载失败**

解决：使用国内镜像源：
```bash
# 激活虚拟环境后
# Linux / macOS
venv/bin/pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Windows
venv\Scripts\pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

**Q：提示 `ModuleNotFoundError: No module named 'logger'` 或 `No module named 'constants'`**

使用了旧版 `pyproject.toml`，`logger.py` 和 `constants.py` 未被正确安装。

解决：确认使用最新版 `pyproject.toml`（v0.3.6-hotfix），然后重新安装：
```bash
pip install -e .
```

---

**Q：提示 `RuntimeError: Form data requires "python-multipart" to be installed`**

旧版 `pyproject.toml` 遗漏了 `python-multipart` 依赖。

解决：
```bash
pip install python-multipart
```
或更新 `pyproject.toml` 后重新 `pip install -e .`。

---

### 启动阶段

---

**Q：启动脚本提示"未找到虚拟环境"**

未完成安装，或在错误的目录下运行了安装脚本。

解决：确认在**项目根目录**（包含 `pyproject.toml` 的目录）下运行安装脚本：
```bash
python install.py      # Windows
bash linux/install.sh  # Linux
bash mac/install.sh    # macOS
```

---

**Q：启动脚本提示"EMBEDDING_MODEL 路径不存在"**

`.env` 中填写的路径有误，或嵌入模型尚未下载完整。

解决：
1. 确认嵌入模型已完整下载（文件夹内有 `config.json`、`tokenizer.json` 等文件）
2. 确认填写的是**文件夹**的完整绝对路径，而非相对路径或某个具体文件的路径
3. Windows 路径中的反斜杠和正斜杠均可，但不能有多余空格

---

**Q：服务启动时 traceback 指向 `app/main.py` 的 `lifespan` 函数**

查看 traceback **最底部**的错误行来确定具体原因：

| 底部错误关键词 | 原因 | 解决方法 |
|--------------|------|---------|
| `OSError` / `No such file` | `EMBEDDING_MODEL` 路径不存在 | 检查并修正 `.env` 中的路径 |
| `ConnectionRefusedError` | 推理服务未启动 | 先启动 LM Studio 或 Ollama |
| `ModuleNotFoundError` | 依赖未安装或虚拟环境有问题 | 重新运行安装脚本 |
| `sqlite3.OperationalError` | 数据库文件损坏 | 运行 `python scripts/setup_db.py` |
| `FileNotFoundError` / `TOML` 相关 | `persona.toml` 缺失 | 将 `config/persona.toml.example` 复制为 `config/persona.toml` |

---

**Q：终端显示 `Uvicorn running`，但浏览器访问返回 502 或连接拒绝**

访问过早，服务仍在初始化。嵌入模型首次加载需要数秒到数十秒。

解决：等待终端出现 `Application startup complete.` 后再访问。

---

**Q：对话有回复，但内容是 `（错误：本地模型调用失败）`**

嵌入模型加载成功，但聊天推理服务不可用。

解决：
1. 确认 LM Studio / Ollama 正在运行且已加载模型
2. 在浏览器访问 `http://localhost:1234/v1/models`，确认返回 JSON 数据
3. 确认 `LOCAL_MODEL_NAME` 与推理服务中加载的模型名称完全一致（区分大小写）

---

**Q：端口 8000 被占用**

解决：

```bash
# Linux / macOS - 查找并结束占用进程
lsof -i:8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

或在 `.env` 中修改 `SERVER_PORT=8001` 换一个端口。

---

**Q：局域网内其他设备无法访问**

解决：
1. 确认防火墙已放行 8000 端口：
   - **Windows**：控制面板 → Windows Defender 防火墙 → 高级设置 → 入站规则 → 新建规则 → 端口 8000
   - **Linux**：`sudo ufw allow 8000`
   - **macOS**：系统设置 → 隐私与安全性 → 防火墙，添加例外
2. 使用本机**局域网 IP**（如 `192.168.x.x`），不是 `localhost`

---

**Q：所有检查都通过了，但服务仍然启动失败**

启动脚本的检查只覆盖最常见情况。直接在终端运行服务查看完整报错：

```bash
# 激活虚拟环境后
python app/main.py
```

找到 traceback 最底部的错误行，对照上方的报错对照表排查，或将完整报错截图提交 Issue。

---

## 更新与重置

### 更新到新版本

```bash
git pull

# 激活虚拟环境后
pip install -e .

# 幂等更新数据库（自动补齐新增字段，不删除已有数据）
python scripts/setup_db.py
```

### 清空所有对话记录和记忆（重置为初始状态）

> ⚠️ 此操作不可逆，所有数据将永久删除。

```bash
# Windows
del data\assistant.db
rmdir /s /q data\chroma_db

# Linux / macOS
rm data/assistant.db
rm -rf data/chroma_db

# 重新初始化数据库
python scripts/setup_db.py
```
