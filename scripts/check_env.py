"""
scripts/check_env.py — 环境检查脚本

在启动应用前检查：
1. Python 版本
2. 依赖包是否完整
3. 配置文件是否存在
4. 目录结构是否正确
5. 本地模型服务是否可达

用法: python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Callable


# =============================================================================
# 工具函数
# =============================================================================

def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def check(name: str, condition: bool, fix: str = "") -> None:
    """打印检查结果"""
    symbol = green("✓") if condition else red("✗")
    status = "通过" if condition else "失败"
    print(f"  [{symbol}] {name}: {status}")
    if not condition and fix:
        print(f"      修复: {fix}")


def section(name: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*50}")
    print(f" {name}")
    print("=" * 50)


# =============================================================================
# 检查项
# =============================================================================

def check_python_version() -> bool:
    """检查 Python 版本"""
    version = sys.version_info
    condition = version >= (3, 10)
    fix = "请安装 Python 3.10+: https://www.python.org/downloads/"
    check("Python 版本", condition, fix if not condition else "")
    if condition:
        print(f"      当前版本: {version.major}.{version.minor}.{version.micro}")
    return condition


def check_dependencies() -> bool:
    """检查核心依赖包"""
    print("\n  检查依赖包...")
    all_ok = True

    required = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "chromadb",
        "sentence_transformers",
        "requests",
        "rank_bm25",
        "jieba",
        "networkx",
    ]

    for pkg in required:
        try:
            importlib.import_module(pkg)
            print(f"      {green('✓')} {pkg}")
        except ImportError:
            print(f"      {red('✗')} {pkg} (未安装)")
            all_ok = False

    if not all_ok:
        print(f"\n      修复: pip install -r requirements.txt")

    return all_ok


def check_directories() -> bool:
    """检查目录结构"""
    print("\n  检查目录结构...")
    root = Path(__file__).parent.parent
    all_ok = True

    required_dirs = [
        "app",
        "app/routes",
        "src/ramaria",
        "src/ramaria/core",
        "src/ramaria/memory",
        "src/ramaria/storage",
        "static",
        "static/css",
        "static/js",
        "scripts",
        "config",
        "tests",
    ]

    for dir_path in required_dirs:
        full_path = root / dir_path
        if full_path.exists():
            print(f"      {green('✓')} {dir_path}")
        else:
            print(f"      {red('✗')} {dir_path} (缺失)")
            all_ok = False

    # 创建必要目录
    for dir_name in ["data", "logs"]:
        dir_path = root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"      {yellow('+')} 已创建: {dir_name}")

    return all_ok


def check_config_files() -> bool:
    """检查配置文件"""
    print("\n  检查配置文件...")
    root = Path(__file__).parent.parent
    all_ok = True

    # .env 文件
    env_file = root / ".env"
    if env_file.exists():
        print(f"      {green('✓')} .env")
    else:
        env_example = root / ".env.example"
        if env_example.exists():
            print(f"      {yellow('⚠')} .env (未创建，将使用 .env.example)")
            # 自动复制
            import shutil
            shutil.copy(env_example, env_file)
            print(f"      {green('✓')} 已从 .env.example 创建 .env")
        else:
            print(f"      {red('✗')} .env.example (缺失)")
            all_ok = False

    # persona.toml
    persona = root / "config" / "persona.toml"
    if persona.exists():
        print(f"      {green('✓')} persona.toml")
    else:
        persona_example = root / "config" / "persona.toml.example"
        if persona_example.exists():
            print(f"      {yellow('⚠')} persona.toml (未创建，将使用模板)")
            import shutil
            shutil.copy(persona_example, persona)
            print(f"      {green('✓')} 已从 persona.toml.example 创建")
        else:
            print(f"      {yellow('⚠')} persona.toml (不存在，跳过)")

    return all_ok


def check_database() -> bool:
    """检查数据库"""
    print("\n  检查数据库...")
    root = Path(__file__).parent.parent
    db_path = root / "data" / "assistant.db"

    if db_path.exists():
        print(f"      {green('✓')} 数据库已存在: {db_path}")
        return True
    else:
        print(f"      {yellow('⚠')} 数据库不存在")
        print(f"      提示: 首次运行会自动创建，或运行 python scripts/setup_db.py")
        return True  # 不算错误


def check_local_model() -> bool:
    """检查本地模型服务"""
    print("\n  检查本地模型服务...")
    root = Path(__file__).parent.parent

    # 尝试读取配置
    config_path = root / "src" / "ramaria" / "config.py"
    if not config_path.exists():
        print(f"      {yellow('⚠')} 配置文件不存在，跳过服务检查")
        return True

    try:
        # 简单读取默认配置
        with open(config_path, encoding="utf-8") as f:
            content = f.read()

        # 提取默认 URL
        import re
        match = re.search(r'LOCAL_API_URL\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            api_url = match.group(1)
            print(f"      配置的服务地址: {api_url}")

            # 尝试连接测试
            try:
                import requests
                response = requests.get(
                    api_url.replace("/v1/chat/completions", "/v1/models"),
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"      {green('✓')} 模型服务可达")
                    return True
                else:
                    print(f"      {yellow('⚠')} 模型服务响应异常 (HTTP {response.status_code})")
                    return False
            except requests.exceptions.ConnectionError:
                print(f"      {yellow('⚠')} 模型服务不可达")
                print(f"      提示: 请确保 LM Studio / Ollama 已启动")
                return False
            except Exception as e:
                print(f"      {yellow('⚠')} 检查失败: {e}")
                return False
        else:
            print(f"      {yellow('⚠')} 未找到 API URL 配置")
            return False

    except Exception as e:
        print(f"      {yellow('⚠')} 读取配置失败: {e}")
        return False


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 50)
    print(" 珊瑚菌 Ramaria 环境检查")
    print("=" * 50)

    results = []

    section("Python 环境")
    results.append(check_python_version())

    section("依赖包")
    results.append(check_dependencies())

    section("目录结构")
    results.append(check_directories())

    section("配置文件")
    results.append(check_config_files())

    section("数据库")
    results.append(check_database())

    section("本地模型服务")
    results.append(check_local_model())

    # 总结
    section("检查结果")
    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"\n{green('🎉 所有检查通过！可以启动应用了。')}")
        print("\n  启动方式:")
        print("    Windows: start.bat")
        print("    或:      python app/main.py")
        return 0
    else:
        print(f"\n{yellow('⚠️ 部分检查未通过，请根据上述提示修复。')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
