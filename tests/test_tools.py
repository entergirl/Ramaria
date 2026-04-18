"""
tests/test_tools.py — v0.5.0 感知工具链路端到端验证脚本

目的：
    验证硬件感知、文件扫描、意图检测三个模块是否正常工作。

运行方式：
    python tests/test_tools.py

测试结构：
    阶段一：psutil 依赖检查
    阶段二：硬件状态采集（hardware_monitor）
    阶段三：文件系统扫描（fs_scanner）
    阶段四：路径提取（extract_path_from_message）
    阶段五：意图检测（tool_registry，语义相似度）
    阶段六：resolve_tool_results 端到端
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# 加载 .env 文件到环境变量
_env_file = _PROJECT_ROOT / ".env"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(label, passed, detail=""):
    status = "✅ [PASS]" if passed else "❌ [FAIL]"
    msg = f"{status}  {label}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    return passed


# =============================================================================
# 阶段一：psutil 依赖检查
# =============================================================================

def test_psutil():
    print_section("阶段一：psutil 依赖检查")
    try:
        import psutil
        print_result("psutil 已安装", True, f"版本：{psutil.__version__}")
        return True
    except ImportError:
        print_result("psutil 未安装", False, "请执行：pip install psutil")
        return False


# =============================================================================
# 阶段二：硬件状态采集
# =============================================================================

def test_hardware_monitor():
    print_section("阶段二：硬件状态采集")
    all_pass = True

    try:
        from ramaria.tools.hardware_monitor import get_hardware_stats, is_high_load

        stats = get_hardware_stats()
        ok1 = print_result(
            "get_hardware_stats 返回非空字符串",
            isinstance(stats, str) and len(stats) > 0,
            f"内容预览：{stats[:80]}…" if len(stats) > 80 else stats,
        )
        all_pass = all_pass and ok1

        load = is_high_load()
        ok2 = print_result(
            "is_high_load 返回布尔值",
            isinstance(load, bool),
            f"当前高负载：{load}",
        )
        all_pass = all_pass and ok2

    except Exception as e:
        print_result("硬件状态采集测试", False, str(e))
        all_pass = False

    return all_pass


# =============================================================================
# 阶段三：文件系统扫描
# =============================================================================

def test_fs_scanner():
    print_section("阶段三：文件系统扫描")
    all_pass = True

    try:
        from ramaria.tools.fs_scanner import scan_directory

        # 扫描项目根目录（应该成功）
        result = scan_directory(str(_PROJECT_ROOT))
        ok1 = print_result(
            "扫描项目根目录成功",
            isinstance(result, str) and "共" in result,
            f"内容行数：{len(result.splitlines())}",
        )
        all_pass = all_pass and ok1

        # 扫描不存在的路径（应该返回错误文本而非异常）
        result2 = scan_directory("/this/path/does/not/exist/xyz")
        ok2 = print_result(
            "扫描不存在路径返回错误文本（不抛异常）",
            isinstance(result2, str) and "错误" in result2,
            result2[:60],
        )
        all_pass = all_pass and ok2

        # 扫描系统目录（应该被安全限制拦截）
        import platform
        blocked_path = "C:\\Windows" if platform.system() == "Windows" else "/etc"
        result3 = scan_directory(blocked_path)
        ok3 = print_result(
            "安全限制拦截系统目录",
            isinstance(result3, str) and "错误" in result3,
            result3[:80],
        )
        all_pass = all_pass and ok3

    except Exception as e:
        import traceback
        print_result("文件系统扫描测试", False, str(e))
        traceback.print_exc()
        all_pass = False

    return all_pass


# =============================================================================
# 阶段四：路径提取
# =============================================================================

def test_path_extraction():
    print_section("阶段四：路径提取（extract_path_from_message）")
    all_pass = True

    try:
        from ramaria.tools.fs_scanner import extract_path_from_message

        cases = [
            ("帮我看看 /home/user/projects 目录", "/home/user/projects"),
            ("扫描一下 C:\\Users\\test\\Desktop", "C:\\Users\\test\\Desktop"),
            ('列出 "~/documents/work" 里的文件', "~/documents/work"),
            ("今天天气怎么样", None),   # 无路径，应返回 None
        ]

        for message, expected in cases:
            result = extract_path_from_message(message)
            passed = (result is None and expected is None) or (
                result is not None and expected is not None
                and expected.lower() in result.lower()
            )
            ok = print_result(
                f"提取路径：{message[:30]}…",
                passed,
                f"期望包含：{expected}，实际：{result}",
            )
            all_pass = all_pass and ok

    except Exception as e:
        print_result("路径提取测试", False, str(e))
        all_pass = False

    return all_pass


# =============================================================================
# 阶段五：意图检测
# =============================================================================

def test_intent_detection():
    print_section("阶段五：意图检测（语义相似度）")

    try:
        from ramaria.tools.tool_registry import (
            _build_intent_vectors,
            _should_trigger_hardware,
            _should_trigger_fs_scan,
            _vectors_initialized,
        )

        # 初始化意图向量
        _build_intent_vectors()

        all_pass = True

        # 硬件感知：应该触发的用例
        hw_positive = [
            "现在电脑CPU占用多少",
            "内存够不够用",
            "电脑跑得动吗",
        ]
        for msg in hw_positive:
            result = _should_trigger_hardware(msg)
            ok = print_result(
                f"硬件意图（正例）：{msg}",
                result is True,
                f"触发：{result}",
            )
            all_pass = all_pass and ok

        # 文件扫描：应该触发的用例
        fs_positive = [
            "帮我看看这个目录里有什么文件",
            "扫描一下这个文件夹",
        ]
        for msg in fs_positive:
            result = _should_trigger_fs_scan(msg)
            ok = print_result(
                f"文件扫描意图（正例）：{msg}",
                result is True,
                f"触发：{result}",
            )
            all_pass = all_pass and ok

        # 无关消息：都不应触发
        negative = "今天心情不太好，聊聊天吧"
        hw_neg = _should_trigger_hardware(negative)
        fs_neg = _should_trigger_fs_scan(negative)
        ok = print_result(
            f"无关消息不触发任何工具：{negative}",
            not hw_neg and not fs_neg,
            f"硬件：{hw_neg}，文件扫描：{fs_neg}",
        )
        all_pass = all_pass and ok

        return all_pass

    except Exception as e:
        import traceback
        print_result("意图检测测试", False, str(e))
        traceback.print_exc()
        return False


# =============================================================================
# 阶段六：resolve_tool_results 端到端
# =============================================================================

def test_resolve_tool_results():
    print_section("阶段六：resolve_tool_results 端到端")
    all_pass = True

    try:
        from ramaria.tools.tool_registry import resolve_tool_results

        # 硬件感知触发
        result1 = resolve_tool_results("现在电脑CPU占用多少")
        ok1 = print_result(
            "硬件相关消息触发 hardware 字段",
            isinstance(result1, dict) and "hardware" in result1,
            f"hardware 内容：{str(result1.get('hardware', ''))[:60]}",
        )
        all_pass = all_pass and ok1

        # 文件扫描触发（带路径）
        scan_msg = f"扫描一下 {str(_PROJECT_ROOT)} 目录"
        result2 = resolve_tool_results(scan_msg)
        ok2 = print_result(
            "带路径的扫描消息触发 fs_scan 字段",
            isinstance(result2, dict) and result2.get("fs_scan") is not None,
            f"fs_scan 内容行数：{len(str(result2.get('fs_scan', '')).splitlines())}",
        )
        all_pass = all_pass and ok2

        # 文件扫描意图但无路径（不应触发）
        result3 = resolve_tool_results("帮我扫描一下目录")
        ok3 = print_result(
            "扫描意图但无路径，fs_scan 为 None",
            isinstance(result3, dict) and result3.get("fs_scan") is None,
            f"fs_scan：{result3.get('fs_scan')}",
        )
        all_pass = all_pass and ok3

        # 无关消息（两个字段都应为 None）
        result4 = resolve_tool_results("今天天气不错，你好")
        ok4 = print_result(
            "无关消息两个字段均为 None",
            isinstance(result4, dict)
            and result4.get("hardware") is None
            and result4.get("fs_scan") is None,
            str(result4),
        )
        all_pass = all_pass and ok4

    except Exception as e:
        import traceback
        print_result("resolve_tool_results 端到端测试", False, str(e))
        traceback.print_exc()
        all_pass = False

    return all_pass


# =============================================================================
# 主入口
# =============================================================================

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  v0.5.0 感知工具链路验证脚本" + " "*28 + "║")
    print("╚" + "═"*58 + "╝")

    results = {}

    results["psutil 依赖"]           = test_psutil()
    results["硬件状态采集"]           = test_hardware_monitor()
    results["文件系统扫描"]           = test_fs_scanner()
    results["路径提取"]               = test_path_extraction()
    results["意图检测"]               = test_intent_detection()
    results["resolve_tool_results"]   = test_resolve_tool_results()

    print_section("测试汇总")
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 全部测试通过！感知工具链路已就绪。")
    else:
        print("\n⚠️  有测试未通过，请根据上方 [FAIL] 信息排查。")


if __name__ == "__main__":
    main()
