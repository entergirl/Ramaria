"""
src/ramaria/tools/fs_scanner.py — 文件系统扫描模块（只读）

职责：
    扫描用户指定目录，生成可读的目录树文本，注入当次对话上下文。

对外接口：
    scan_directory(path_str: str) -> str
        扫描指定目录，返回格式化的目录树文本。
        路径不存在、不合法或触碰安全禁区时返回错误说明文本。

安全约束（本版本严格只读）：
    · 禁止扫描系统敏感目录（见 _BLOCKED_PATHS）
    · 不读取任何文件内容，只列出文件名、大小、类型
    · 扫描深度最大 3 层，单次最多展示 200 个条目，避免 Prompt 爆炸
    · 不做任何 rename / move / delete 操作
"""

from __future__ import annotations

import os
from pathlib import Path

from ramaria.logger import get_logger

logger = get_logger(__name__)

# ── 安全黑名单：禁止扫描的系统目录（前缀匹配）────────────────────────────
# Windows 和 Linux/macOS 的常见系统目录均纳入
_BLOCKED_PATHS: list[str] = [
    # Windows
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
    # Linux / macOS
    "/etc",
    "/sys",
    "/proc",
    "/dev",
    "/boot",
    "/usr/lib",
    "/usr/bin",
    "/bin",
    "/sbin",
]

# 扫描深度上限（层数）
_MAX_DEPTH = 3

# 单次展示条目上限（文件 + 目录合计）
_MAX_ENTRIES = 200

# 文件大小单位格式化阈值
_KB = 1024
_MB = 1024 ** 2
_GB = 1024 ** 3


def _fmt_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的大小字符串。"""
    if size_bytes >= _GB:
        return f"{size_bytes / _GB:.1f} GB"
    if size_bytes >= _MB:
        return f"{size_bytes / _MB:.1f} MB"
    if size_bytes >= _KB:
        return f"{size_bytes / _KB:.1f} KB"
    return f"{size_bytes} B"


def _is_blocked(path: Path) -> bool:
    """
    检查路径是否落在安全黑名单范围内。
    使用字符串前缀匹配，同时处理大小写（Windows 不区分大小写）。
    """
    resolved = str(path.resolve())
    resolved_lower = resolved.lower()
    for blocked in _BLOCKED_PATHS:
        if resolved_lower.startswith(blocked.lower()):
            return True
    return False


def _build_tree(
    base: Path,
    current: Path,
    depth: int,
    lines: list[str],
    counter: list[int],   # 用列表包装以便在嵌套函数中修改
) -> None:
    """
    递归构建目录树文本，写入 lines 列表。

    参数：
        base    — 扫描根目录（用于计算相对路径）
        current — 当前递归目录
        depth   — 当前深度（从 0 开始）
        lines   — 输出行列表
        counter — [已处理条目数]，超出 _MAX_ENTRIES 时停止递归
    """
    if depth > _MAX_DEPTH:
        return
    if counter[0] >= _MAX_ENTRIES:
        return

    try:
        entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        lines.append(f"{'  ' * depth}[无权限访问]")
        return

    for entry in entries:
        if counter[0] >= _MAX_ENTRIES:
            lines.append(f"{'  ' * depth}... （条目过多，已截断）")
            return

        indent = "  " * depth
        try:
            if entry.is_symlink():
                # 符号链接：标注但不跟随，防止循环
                lines.append(f"{indent}🔗 {entry.name}  → 符号链接")
                counter[0] += 1

            elif entry.is_dir():
                lines.append(f"{indent}📁 {entry.name}/")
                counter[0] += 1
                # 递归进入子目录
                _build_tree(base, entry, depth + 1, lines, counter)

            elif entry.is_file():
                size = _fmt_size(entry.stat().st_size)
                lines.append(f"{indent}📄 {entry.name}  ({size})")
                counter[0] += 1

        except (PermissionError, OSError):
            lines.append(f"{indent}⚠ {entry.name}  [无法访问]")
            counter[0] += 1


def scan_directory(path_str: str) -> str:
    """
    扫描指定目录，返回格式化的目录树文本。

    参数：
        path_str — 用户提供的路径字符串（绝对路径或相对路径均可）

    返回示例：
        扫描目录：/home/user/projects/ramaria（深度 ≤ 3，最多 200 条）

        📁 src/
          📁 ramaria/
            📄 config.py  (8.2 KB)
            📄 logger.py  (4.1 KB)
        📄 README.md  (12.6 KB)
        📄 pyproject.toml  (3.4 KB)

        共 42 个条目

    返回：
        str — 格式化目录树；错误时返回包含说明的错误文本
    """
    if not path_str or not path_str.strip():
        return "错误：未提供扫描路径。"

    # 路径解析
    try:
        path = Path(path_str.strip()).expanduser().resolve()
    except Exception as e:
        return f"错误：路径解析失败 — {e}"

    # 存在性检查
    if not path.exists():
        return f"错误：路径不存在 — {path}"

    if not path.is_dir():
        # 如果是文件，直接返回文件信息，不做树展开
        try:
            size = _fmt_size(path.stat().st_size)
            return f"📄 {path.name}  ({size})\n路径：{path}"
        except Exception as e:
            return f"错误：无法读取文件信息 — {e}"

    # 安全检查
    if _is_blocked(path):
        return f"错误：该路径在安全限制范围内，无法扫描 — {path}"

    # 构建目录树
    lines: list[str] = []
    counter = [0]   # 用列表包装，供递归内部修改

    _build_tree(path, path, depth=0, lines=lines, counter=counter)

    header = (
        f"扫描目录：{path}"
        f"（深度 ≤ {_MAX_DEPTH}，最多 {_MAX_ENTRIES} 条）\n"
    )
    footer = f"\n共 {counter[0]} 个条目"

    result = header + "\n".join(lines) + footer
    logger.info(f"文件系统扫描完成：{path}，共 {counter[0]} 个条目")
    return result


def extract_path_from_message(message: str) -> str | None:
    """
    从用户消息中提取路径字符串。

    支持格式：
        · 绝对路径（Windows：C:\\... 或 D:/...；Linux/macOS：/home/...）
        · 波浪号路径（~/...）
        · 引号包裹的路径（"..." 或 '...'）

    参数：
        message — 用户消息文本

    返回：
        str  — 提取到的第一个路径字符串
        None — 未找到路径
    """
    import re

    # 优先匹配引号包裹的路径
    quoted = re.search(r'["\']([^"\']+)["\']', message)
    if quoted:
        candidate = quoted.group(1).strip()
        # 粗略判断是否像路径（含斜杠或反斜杠）
        if '/' in candidate or '\\' in candidate or candidate.startswith('~'):
            return candidate

    # Windows 绝对路径：盘符开头
    win_path = re.search(r'[A-Za-z]:[\\\/][^\s，,。.？?！!；;]+', message)
    if win_path:
        return win_path.group(0).rstrip('，,。.？?！!；;')

    # Linux/macOS 绝对路径或波浪号路径
    unix_path = re.search(r'(?:~|\/)[^\s，,。.？?！!；;]*(?:\/[^\s，,。.？?！!；;]*)+', message)
    if unix_path:
        return unix_path.group(0).rstrip('，,。.？?！!；;')

    return None
